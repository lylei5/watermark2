"""Resumable generation and verification assets, without reading test decisions."""
import argparse,gc,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from cryptography.hazmat.primitives.ciphers import Cipher,algorithms
from diffusers import StableDiffusionPipeline,DDIMScheduler,DDIMInverseScheduler
from PIL import Image
from codeword_binding import keyed_order,sembind_mask_bits
from research_semantics import ResearchSemantics


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path,data):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2),encoding='utf-8');temp.replace(path)


def load_pipe(model,device):
    pipe=StableDiffusionPipeline.from_pretrained(model,torch_dtype=torch.float16,safety_checker=None,local_files_only=True).to(device)
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    for module in [pipe.unet,pipe.vae,pipe.text_encoder]:module.requires_grad_(False)
    return pipe


@torch.inference_mode()
def invert(pipe,image,steps=50):
    dev=pipe.device
    x=pipe.image_processor.preprocess(image).to(dev,dtype=torch.float16)
    z=(pipe.vae.encode(x).latent_dist.mode()*pipe.vae.config.scaling_factor).float()
    emb=pipe.encode_prompt('',dev,1,False)[0]
    scheduler=DDIMInverseScheduler.from_config(pipe.scheduler.config);scheduler.set_timesteps(steps,device=dev)
    for t in scheduler.timesteps:
        pred=pipe.unet(scheduler.scale_model_input(z,t).half(),t,encoder_hidden_states=emb).sample.float()
        z=scheduler.step(pred,t,z).prev_sample
    return z.cpu().numpy()


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--semantic',required=True);p.add_argument('--coverlock',required=True);p.add_argument('--stage',choices=['prepare','aux','codes','reference'],required=True);args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    out=args.output;out.mkdir(exist_ok=True,parents=True)
    planpath=out/'generation_plan.json'
    if args.stage=='prepare':
        assert not planpath.exists()
        rows=json.loads((args.data/'manifest.json').read_text())['rows']
        chosen=[]
        for split,n in [('development',16),('calibration',32),('confirmation',64)]:
            chosen.extend([dict(r,split_index=i) for i,r in enumerate([r for r in rows if r['split']==split][:n])])
        for i,r in enumerate(chosen):
            r.update(task=i,aux_seed=202611090000+i)
            write(out/'private'/f'{i:03}.json',dict(key=os.urandom(32).hex(),nonce=os.urandom(16).hex(),message=os.urandom(32).hex(),binding_key=os.urandom(32).hex(),amplitude_seed=202611091000+i))
        write(planpath,dict(data_manifest_sha256=sha(args.data/'manifest.json'),rows=chosen,methods=['sembind_scattered','coverlock_g_message']))
        print('PREPARED',len(chosen),flush=True);return
    plan=json.loads(planpath.read_text());start=time.monotonic()
    if args.stage=='codes':
        model=ResearchSemantics(args.semantic,args.coverlock,'cuda')
        for r in plan['rows']:
            path=out/'aux'/f"{r['task']:03}.png";dest=out/'aux_codes'/f"{r['task']:03}.npz"
            if dest.exists():continue
            dest.parent.mkdir(exist_ok=True)
            np.savez_compressed(dest,**model.extract(Image.open(path).convert('RGB')))
            print('AUX_CODES',r['task'],flush=True)
    else:
        pipe=load_pipe(args.model,'cuda')
        for r in plan['rows']:
            i=r['task'];prompt=r['caption']
            if args.stage=='aux':
                dst=out/'aux'/f'{i:03}.png'
                if dst.exists():continue
                dst.parent.mkdir(exist_ok=True)
                rng=np.random.default_rng(r['aux_seed']);z=torch.from_numpy(rng.standard_normal((1,4,64,64)).astype(np.float32))
                image=pipe(prompt,latents=z.cuda().half(),num_inference_steps=50,guidance_scale=7.5,height=512,width=512).images[0]
                image.save(dst);print('AUX',i,flush=True)
            else:
                secret=json.loads((out/'private'/f'{i:03}.json').read_text())
                key,nonce,msg,bkey=[bytes.fromhex(secret[k]) for k in ['key','nonce','message','binding_key']]
                cipher=Cipher(algorithms.ChaCha20(key,nonce),mode=None).encryptor().update(msg*64)
                bits=np.unpackbits(np.frombuffer(cipher,dtype=np.uint8))
                z=np.abs(np.random.default_rng(secret['amplitude_seed']).standard_normal(16384))*(2*bits.astype(np.float32)-1)
                codes=np.load(out/'aux_codes'/f'{i:03}.npz')
                for method in plan['methods']:
                    dst=out/'references'/f'{i:03}_{method}.png'
                    if dst.exists():continue
                    if method=='sembind_scattered':
                        perm=keyed_order(bkey,b'sembind/permutation/',16384)
                        mask=sembind_mask_bits(codes['sembind_code'][None],perm)[0]
                    else:mask=np.tile(codes['coverlock_code'],64)
                    latent=torch.from_numpy((z*(1-2*mask.astype(np.float32))).astype(np.float32).reshape(1,4,64,64)).cuda().half()
                    image=pipe(prompt,latents=latent,num_inference_steps=50,guidance_scale=7.5,height=512,width=512).images[0]
                    dst.parent.mkdir(exist_ok=True);image.save(dst)
                    print('REFERENCE',i,method,flush=True)
    write(out/f'{args.stage}_complete.json',dict(completed=True,seconds=time.monotonic()-start,script_sha256=sha(__file__),plan_sha256=sha(planpath)))


if __name__=='__main__':main()
