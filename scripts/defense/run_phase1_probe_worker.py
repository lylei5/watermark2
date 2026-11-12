"""Released GS component baseline and read-only DPM trajectory observer on249."""
import argparse
import datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value
    spec.loader.exec_module(value)
    return value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--stage',choices=['generate','verify'],required=True)
    parser.add_argument('--shard',type=int,choices=range(3),required=True)
    a=parser.parse_args();run=a.run.resolve()
    assert str(run.parent)=='/data0/liyanlei/watermark_research/runs'
    plan=json.loads((run/'protocol.json').read_text())
    for name,digest in plan['source_sha256'].items(): assert sha(run/name)==digest,name
    for path,digest in plan['external_sha256'].items(): assert sha(path)==digest,path
    gpu=plan['gpu_indices'][a.shard]
    info=[x.strip() for x in subprocess.check_output(['nvidia-smi','-i',str(gpu),
        '--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')]
    assert gpu in [7,5,4] and int(info[0])==gpu and int(info[2])<50 and int(info[3])==0,info
    os.environ.update(CUDA_VISIBLE_DEVICES=info[1],HF_HOME=plan['hf_home'],HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0,plan['extra_dependencies'])
    import numpy as np
    import torch
    from PIL import Image
    from codeword_binding import keyed_order
    from sembind_official_audit import load_official
    from semantic_code import SemanticCodeExtractor
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    paths={k:Path(v) for k,v in plan['paths'].items()}
    private=Path(plan['identity_root'])
    assert sha(private/'seeds.json')==plan['private_seed_sha256']
    secrets=json.loads((private/'seeds.json').read_text())
    permutation=keyed_order(bytes.fromhex(secrets['binding_key']),b'sembind/permutation/',16384)
    native_module=module('native_gs',run/'gs_watermark.py')
    module('model',paths['official']/'model.py')
    out=run/f'{a.stage}_{a.shard}';out.mkdir(exist_ok=False)
    for folder in ['images','codes','latents','work']: (out/folder).mkdir()
    os.chdir(out/'work')
    state=dict(status='running',stage=a.stage,pid=os.getpid(),gpu_index=gpu,gpu_uuid=info[1],
        protocol_sha256=sha(run/'protocol.json'),started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(out/'state.json',state)
    started=time.monotonic();records=[]
    try:
        extractor=SemanticCodeExtractor(paths['semantic']/'dinov2-giant',paths['semantic']/'sembind_masker.pth')
        inversion,pipe=load_official(SimpleNamespace(**paths))
        pipe.set_progress_bar_config(disable=True)
        assert pipe.scheduler.config.algorithm_type=='dpmsolver++'
        capture={}
        extractor.model.hash_head.register_forward_hook(lambda m,i,o:capture.update(logits=o.detach().cpu().numpy().reshape(-1)))
        def code(image):
            bits=extractor.code(image).cpu().numpy().astype(np.uint8)
            logits=capture['logits'].copy()
            assert bits.shape==(1024,) and np.array_equal(bits,logits>0)
            return bits,logits
        def artifact(path): return dict(path=str(path.relative_to(run)),sha256=sha(path))
        def save_image(image,name):
            path=out/'images'/(name+'.png');image.save(path)
            return artifact(path)
        def jpeg(image):
            stream=io.BytesIO();image.save(stream,format='JPEG',quality=50)
            stream.seek(0);return Image.open(stream).convert('RGB')
        def generate(prompt,z,callback=None):
            pipe.to('cuda')
            output,final=pipe(prompt,guidance_scale=7.5,num_inference_steps=50,height=512,width=512,
                latents=torch.as_tensor(z,dtype=torch.float32,device='cuda'),callback=callback,callback_steps=1)
            assert not any(output.nsfw_content_detected or [])
            return output.images[0],final.detach().cpu()
        sources=[row for row in plan['sources'] if row['task']%3==a.shard]
        for row in sources:
            task=row['task'];tick=time.monotonic();native=native_module.Gaussian_Shading(1,8,1e-6,1)
            if a.stage=='generate':
                z=np.random.default_rng(row['aux_seed']).standard_normal((1,4,64,64)).astype(np.float32)
                trajectory=[];timesteps=[]
                def observe(index,t,latent):
                    if index+1 in plan['trajectory_steps']:
                        trajectory.append(pipe.scheduler.model_outputs[-1].detach().cpu().clone())
                        timesteps.append(int(t))
                aux,final_latent=generate(row['caption'],z,observe)
                assert len(trajectory)==46 and all(x.shape==(1,4,64,64) and x.isfinite().all() for x in trajectory)
                observer_check=None
                if task==sources[0]['task']:
                    plain,plain_latent=generate(row['caption'],z)
                    assert torch.equal(plain_latent,final_latent) and np.array_equal(np.array(plain),np.array(aux))
                    observer_check=dict(exact_pixel_equality=True,exact_final_latent_equality=True)
                arrays={};arrays['aux_code'],arrays['aux_logits']=code(aux)
                t_codes=[];t_logits=[];t_pixels=[]
                with torch.no_grad():
                    for step,x0 in zip(plan['trajectory_steps'],trajectory):
                        decoded=pipe.numpy_to_pil(pipe.decode_latents(x0.cuda()))[0]
                        b,l=code(decoded);t_codes.append(b);t_logits.append(l)
                        t_pixels.append(save_image(decoded,f'{task:03}_x0_{step:02}'))
                arrays['trajectory_codes']=np.stack(t_codes);arrays['trajectory_logits']=np.stack(t_logits)
                arrays['timesteps']=np.asarray(timesteps)
                xp=out/'latents'/f'{task:03}_trajectory.npz'
                np.savez_compressed(xp,x0=torch.cat(trajectory).numpy(),final=final_latent.numpy())
                same_z=np.random.default_rng(row['independent_seed']).standard_normal((1,4,64,64)).astype(np.float32)
                same,_=generate(row['caption'],same_z)
                arrays['same_code'],arrays['same_logits']=code(same)
                seed=secrets['sources'][str(task)]
                torch.manual_seed(seed['torch_seed']);np.random.seed(seed['numpy_seed'])
                carrier=native.create_watermark_and_return_w().float()
                identity=private/f'{task:03}.npz'
                assert not identity.exists()
                np.savez_compressed(identity,key=native.key.cpu().numpy().astype(np.uint8),
                    message=native.watermark.cpu().numpy().astype(np.uint8),initial=carrier.cpu().numpy())
                identity.chmod(0o600)
                assert native.eval_watermark(carrier)==1.
                mask=np.tile(arrays['aux_code'],16)[permutation].reshape(1,4,64,64)
                masked=carrier*(1-2*torch.tensor(mask,dtype=torch.float32,device='cuda'))
                assert native.eval_watermark(masked*(1-2*torch.tensor(mask,dtype=torch.float32,device='cuda')))==1.
                gs,_=generate(row['caption'],carrier)
                bound,_=generate(row['caption'],masked)
                arrays['gs_code'],arrays['gs_logits']=code(gs)
                arrays['bound_code'],arrays['bound_logits']=code(bound)
                arrays['jpeg_aux_code'],_=code(jpeg(aux));arrays['jpeg_bound_code'],_=code(jpeg(bound))
                cp=out/'codes'/f'{task:03}.npz';np.savez_compressed(cp,**arrays)
                images={k:save_image(im,f'{task:03}_{k}') for k,im in [('aux',aux),('same',same),('gs',gs),('bound',bound)]}
                record=dict(task=task,dataset=row['dataset'],seconds=time.monotonic()-tick,codes=artifact(cp),
                    images=images,trajectory=artifact(xp),trajectory_images=t_pixels,identity_sha256=sha(identity),
                    native_carrier_roundtrip_ba=1.,observer_check=observer_check)
            else:
                gen=run/f'generate_{a.shard}'
                generated=json.loads((gen/'complete.json').read_text())
                original=next(r for r in generated['rows'] if r['task']==task)
                with np.load(private/f'{task:03}.npz') as identity:
                    assert sha(private/f'{task:03}.npz')==original['identity_sha256']
                    native.key=torch.tensor(identity['key'],device='cuda')
                    native.watermark=torch.tensor(identity['message'],device='cuda')
                rows=[]
                with np.load(run/original['codes']['path']) as data: codes={k:data[k] for k in data.files}
                for name in ['gs','bound','aux']:
                    source=original['images'][name];image_path=run/source['path']
                    assert sha(image_path)==source['sha256']
                    image=Image.open(image_path).convert('RGB')
                    torch.manual_seed(202611130000+task)
                    z=inversion.exact_inversion(image,prompt='',guidance_scale=3.,num_inference_steps=50,
                        test_num_inference_steps=50,inv_order=0,decoder_inv=True,pipe=pipe)
                    z=z.detach();assert z.shape==(1,4,64,64) and z.isfinite().all()
                    zp=out/'latents'/f'{task:03}_{name}.npy';np.save(zp,z.cpu().numpy())
                    arms=['gs','bound'] if name=='aux' else [name]
                    for arm in arms:
                        query=z
                        if arm=='bound':
                            q=codes['aux_code' if name=='aux' else 'bound_code']
                            mask=np.tile(q,16)[permutation].reshape(1,4,64,64)
                            query=z*(1-2*torch.tensor(mask,dtype=torch.float32,device='cuda'))
                        ba=native.eval_watermark(query)
                        direct=(((query>0).int()+native.key)%2).cpu().numpy().reshape(1,4,8,8,8,8).sum(axis=(2,4))>32
                        independent=float((direct.reshape(-1)==native.watermark.cpu().numpy().reshape(-1)).mean())
                        assert independent==ba
                        rows.append(dict(task=task,dataset=row['dataset'],image_kind=name,detector=arm,ba=ba,
                            matching_bits=int(round(256*ba)),accepted={str(k):ba>=k/256 for k in plan['working_points']},
                            image=source,latent=artifact(zp)))
                record=dict(task=task,dataset=row['dataset'],seconds=time.monotonic()-tick,decisions=rows)
            records.append(record)
            write(out/'progress.json',dict(rows=records,protocol_sha256=state['protocol_sha256']))
            print(a.stage.upper(),task,round(record['seconds'],2),flush=True)
        state.update(status='complete',rows=records,seconds=time.monotonic()-started,
            completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(out/'complete.json',state);write(out/'state.json',{k:v for k,v in state.items() if k!='rows'})
    except Exception as exc:
        state.update(status='failed',error=repr(exc),completed_tasks=len(records),seconds=time.monotonic()-started)
        write(out/'state.json',state)
        raise


if __name__=='__main__':
    main()
