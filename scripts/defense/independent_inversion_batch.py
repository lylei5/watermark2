"""Cache full DDIM50 inversions; no secret or detector score is accessed."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from diffusers import DDIMInverseScheduler
from PIL import Image
from independent_generation import load_pipe,write,sha

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--split',default='all');p.add_argument('--batch',type=int,default=4);p.add_argument('--manifest',type=Path);p.add_argument('--output-subdir',default='inversions');args=p.parse_args()
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
records=json.loads((args.manifest or args.root/f'query_manifest_{args.split}.json').read_text())['rows']
pipe=load_pipe(args.model,'cuda');emb=pipe.encode_prompt('',torch.device('cuda'),1,False)[0].detach()
pipe.text_encoder.cpu();torch.cuda.empty_cache()
scheduler=DDIMInverseScheduler.from_config(pipe.scheduler.config);scheduler.set_timesteps(50,device='cuda')
out=args.root/args.output_subdir;out.mkdir(exist_ok=True)
pending=[r for r in records if not (out/f"{r['id']}.npy").exists()];start=time.monotonic()
with torch.inference_mode():
    for offset in range(0,len(pending),args.batch):
        rows=pending[offset:offset+args.batch];latents=[]
        for r in rows:
            path=Path(r['image_path']) if 'image_path' in r else args.root/'queries'/f"{r['id']}.png"
            x=pipe.image_processor.preprocess(Image.open(path).convert('RGB')).cuda().half()
            latents.append((pipe.vae.encode(x).latent_dist.mode()*pipe.vae.config.scaling_factor).float())
        z=torch.cat(latents);context=emb.expand(len(rows),-1,-1)
        for t in scheduler.timesteps:
            pred=pipe.unet(z.half(),t,encoder_hidden_states=context).sample.float()
            z=scheduler.step(pred,t,z).prev_sample
        for r,latent in zip(rows,z):np.save(out/f"{r['id']}.npy",latent.cpu().numpy())
        print('INVERTED',offset+len(rows),len(pending),round(time.monotonic()-start,1),flush=True)
write(args.root/f'{args.output_subdir}_{args.split}_complete.json',dict(completed=True,n=len(records),seconds=time.monotonic()-start,script_sha256=sha(__file__),dtype='fp16 model, fp32 inversion state',steps=50,batch=args.batch))
