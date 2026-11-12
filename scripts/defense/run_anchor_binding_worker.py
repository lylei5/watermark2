"""Full-step generation and verification for conditional-anchor defense pilots."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--shard',type=int,choices=range(3),required=True)
    p.add_argument('--stage',choices=['genuine','forgery'],required=True)
    args=p.parse_args();run=args.run.resolve();plan=json.loads((run/'protocol.json').read_text())
    for name,digest in plan['source_sha256'].items():assert sha(run/name)==digest,name
    for name,digest in plan['external_sha256'].items():assert sha(name)==digest,name
    paths={k:Path(v) for k,v in plan['paths'].items()}
    gpu=plan['gpu_indices'][args.shard]
    info=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
    assert gpu in [7,5,4] and int(info[0])==gpu and int(info[2])<50 and int(info[3])==0,info
    os.environ.update(CUDA_VISIBLE_DEVICES=info[1],HF_HOME=plan['hf_home'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                      OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0,str(Path(plan['chain_run'])/'scripts'))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np
    import torch
    from PIL import Image
    carrier=__import__(plan.get('carrier_module','anchor_binding'))
    embed,decode=carrier.embed,carrier.decode
    from sembind_official_audit import load_official
    spec=importlib.util.spec_from_file_location('model',paths['official']/'model.py')
    model_module=importlib.util.module_from_spec(spec);sys.modules['model']=model_module;spec.loader.exec_module(model_module)
    from semantic_code import SemanticCodeExtractor
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=run/f'{args.stage}_{args.shard}';out.mkdir(exist_ok=False)
    for folder in ['images','codes','latents','work']:(out/folder).mkdir()
    os.chdir(out/'work')
    state=dict(status='running',stage=args.stage,pid=os.getpid(),gpu=gpu,protocol_sha256=sha(run/'protocol.json'))
    save(out/'state.json',state)
    try:
        masker=SemanticCodeExtractor(paths['semantic']/'dinov2-giant',paths['semantic']/'sembind_masker.pth')
        inversion,pipe=load_official(SimpleNamespace(**paths))
        spec=importlib.util.spec_from_file_location('gs_distortions',plan['distortion_source'])
        distort=importlib.util.module_from_spec(spec);spec.loader.exec_module(distort)
        rows=[];references=[];captured={}
        if plan.get('capture_semantic_logits',False):
            masker.model.hash_head.register_forward_hook(lambda module,inputs,output: captured.update(logits=output.detach().cpu().numpy().reshape(-1)))
        def code(image,path):
            value=masker.code(image).cpu().numpy().astype(np.uint8)
            extra={'logits':captured['logits']} if plan.get('capture_semantic_logits',False) else {}
            if extra:assert np.array_equal((extra['logits']>0).astype(np.uint8),value)
            np.savez_compressed(path,code=value,**extra)
            if plan.get('carrier_feature')=='logits':
                assert extra
                return dict(code=value,logits=extra['logits'])
            return value
        def semantic_bits(value):
            return value['code'] if isinstance(value,dict) else value
        def invert_score(image,code_value,task,arm,jid,extra):
            torch.manual_seed(202611120000+task)
            z=inversion.exact_inversion(image,prompt='',guidance_scale=3.,num_inference_steps=50,
                test_num_inference_steps=50,inv_order=0,decoder_inv=True,pipe=pipe)
            z=z.detach().cpu().numpy();assert z.shape==(1,4,64,64) and np.isfinite(z).all()
            latent=out/'latents'/(jid+'.npy');np.save(latent,z)
            ba=decode(z,code_value,Path(plan['identity_root']),task,arm)
            result=dict(task=task,arm=arm,id=jid,ba=ba,accepted=ba>181/256,
                latent=str(latent.relative_to(run)),latent_sha256=sha(latent),
                code=str((out/'codes'/(jid+'.npz')).relative_to(run)),code_sha256=sha(out/'codes'/(jid+'.npz')),**extra)
            rows.append(result);save(out/'progress.json',dict(rows=rows))
            print('SCORED',jid,ba,flush=True)
        if args.stage=='genuine':
            for item in [x for x in plan['sources'] if x['task']%3==args.shard]:
                task=item['task'];auxz=np.random.default_rng(item['aux_seed']).standard_normal((1,4,64,64)).astype(np.float32)
                aux,_,_=inversion.generate(prompt=item['caption'],guidance_scale=7.5,num_inference_steps=50,solver_order=1,pipe=pipe,init_latents=torch.from_numpy(auxz).cuda())
                auxpath=out/'images'/f'{task:03}_aux.png';aux.save(auxpath)
                enrollment=code(aux,out/'codes'/f'{task:03}_aux.npz')
                for arm in plan['arms']:
                    initial=embed(auxz,enrollment,Path(plan['identity_root']),task,arm)
                    if plan.get('capture_initial',False):
                        np.save(out/'latents'/f'{task:03}_{arm}_initial.npy',initial)
                    ref,_,_=inversion.generate(prompt=item['caption'],guidance_scale=7.5,num_inference_steps=50,solver_order=1,pipe=pipe,init_latents=torch.from_numpy(initial).cuda())
                    refpath=out/'images'/f'{task:03}_{arm}.png';ref.save(refpath)
                    refcode=code(ref,out/'codes'/f'{task:03}_{arm}.npz')
                    references.append(dict(task=task,arm=arm,image=str(refpath.relative_to(run)),image_sha256=sha(refpath),
                        auxiliary=str(auxpath.relative_to(run)),semantic_mismatch=float((semantic_bits(refcode)!=semantic_bits(enrollment)).mean())))
                    if plan.get('capture_initial',False):
                        ip=out/'latents'/f'{task:03}_{arm}_initial.npy'
                        references[-1].update(initial_latent=str(ip.relative_to(run)),initial_latent_sha256=sha(ip))
                    for c,condition in enumerate(plan['conditions']):
                        jid=f'{task:03}_{arm}_{c:02}';seed=2026111200+task*100+c
                        if condition['name']=='clean':query=ref.copy()
                        else:
                            options=SimpleNamespace(**{x['name']:None for x in plan['conditions'] if x['name']!='clean'})
                            setattr(options,condition['name'],condition['value']);distort.set_random_seed(seed)
                            query=distort.image_distortion(ref.copy(),seed,options).convert('RGB')
                        imagepath=out/'images'/(jid+'.png');query.save(imagepath)
                        q=code(query,out/'codes'/(jid+'.npz'))
                        invert_score(query,q,task,arm,jid,dict(condition=condition['name'],image=str(imagepath.relative_to(run)),
                            image_sha256=sha(imagepath),semantic_mismatch=float((semantic_bits(q)!=semantic_bits(enrollment)).mean())))
            count=sum(x['task']%3==args.shard for x in plan['sources'])*len(plan['arms'])
            assert len(rows)==count*len(plan['conditions']) and len(references)==count
        else:
            attack=json.loads((run/'attack_protocol.json').read_text())
            for job in [j for n,j in enumerate(attack['jobs']) if n%3==args.shard]:
                done=json.loads((run/'attack_outputs'/job['id']/'complete.json').read_text())
                assert done['updates_executed']==151 and len(done['trace'])==151
                assert done['protocol_sha256']==sha(run/'attack_protocol.json')
                assert [c['update'] for c in done['checkpoints']]==list(range(1,152,10))
                inputs=[dict(kind=k,path=run/job[k],sha256=job[k+'_sha256'],update=None) for k in ['reference','cover']]
                inputs += [dict(kind='candidate',path=run/'attack_outputs'/job['id']/c['path'],sha256=c['sha256'],update=c['update']) for c in done['checkpoints']]
                for item in inputs:
                    imagepath=item['path'];assert sha(imagepath)==item['sha256']
                    jid=job['id']+'_'+(item['kind'] if item['update'] is None else f"update_{item['update']:03}")
                    image=Image.open(imagepath).convert('RGB');q=code(image,out/'codes'/(jid+'.npz'))
                    invert_score(image,q,job['task'],job['method'],jid,dict(kind=item['kind'],update=item['update'],
                        image=str(imagepath),image_sha256=sha(imagepath),cover=str(run/job['cover'])))
            assert len(rows)==18*sum(i%3==args.shard for i in range(len(attack['jobs'])))
        del pipe,masker;gc.collect();torch.cuda.empty_cache()
        sys.path.insert(0,plan['upstream'])
        import lpips
        from utils.image_utils import psnr_PIL,ssim_PIL,msssim_PIL,lpips_PIL
        metric=lpips.LPIPS(net='alex').eval().cpu()
        quality=references if args.stage=='genuine' else [r for r in rows if r['kind']=='candidate']
        for row in quality:
            a=Image.open(run/row['image']).convert('RGB')
            b=Image.open(run/row['auxiliary'] if args.stage=='genuine' else row['cover']).convert('RGB')
            with torch.no_grad():row.update(lpips=float(lpips_PIL(a,b,metric,'cpu')),psnr=float(psnr_PIL(a,b)),
                ssim=float(ssim_PIL(a,b)[0]),ms_ssim=float(msssim_PIL(a,b)))
        state.update(status='complete',rows=rows,references=references)
        save(out/'complete.json',state)
    except Exception as exc:state.update(status='failed',error=repr(exc));raise
    finally:save(out/'state.json',state)


if __name__=='__main__':main()
