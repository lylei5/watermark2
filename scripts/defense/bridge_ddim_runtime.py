"""Replay two recorded development batches before any calibration scoring."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image
from diffusers import DDIMInverseScheduler
from independent_generation import load_pipe


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert not args.output.exists()
    plan=json.loads((args.run/'protocol.json').read_text());torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    pipe=load_pipe(plan['server_model'],'cuda')
    emb=pipe.encode_prompt('',torch.device('cuda'),1,False)[0].detach()
    pipe.text_encoder.cpu();torch.cuda.empty_cache()
    scheduler=DDIMInverseScheduler.from_config(pipe.scheduler.config);scheduler.set_timesteps(50,device='cuda')
    rows=[];start=time.monotonic()
    for labels in [['clean','jpeg50'],['crop90','noise0.01']]:
        latents=[]
        with torch.inference_mode():
            for label in labels:
                ident='000_sembind_scattered_'+label
                x=pipe.image_processor.preprocess(Image.open(args.run/'bridge'/(ident+'.png')).convert('RGB')).cuda().half()
                latents.append((pipe.vae.encode(x).latent_dist.mode()*pipe.vae.config.scaling_factor).float())
            z=torch.cat(latents);context=emb.expand(2,-1,-1)
            for t in scheduler.timesteps:
                prediction=pipe.unet(z.half(),t,encoder_hidden_states=context).sample.float()
                z=scheduler.step(prediction,t,z).prev_sample
        ident='000_sembind_scattered_'+labels[0]
        expected=np.load(args.run/'bridge'/(ident+'.npy'),allow_pickle=False).reshape(-1)
        observed=z[0].cpu().numpy().reshape(-1)
        rows.append(dict(id=ident,batch_members=labels,relative_l2=float(np.linalg.norm(observed-expected)/np.linalg.norm(expected)),
            sign_disagreement=float(np.mean((observed>0)!=(expected>0)))))
    result=dict(rows=rows,steps=50,batch=2,calibration_decisions_read=False,
        relative_l2_gate=plan['bridge_max_relative_l2'],passed=all(r['relative_l2']<=plan['bridge_max_relative_l2'] for r in rows),
        dependencies={k:importlib.metadata.version(k) for k in ['torch','diffusers','transformers','scipy']},seconds=time.monotonic()-start)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
