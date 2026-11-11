"""Full DDIM50 query caches for a frozen, held-out calibration split."""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch
from diffusers import DDIMInverseScheduler

from independent_generation import load_pipe, sha, write
from sembind import SemanticMasker


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--shard',type=int,choices=range(3),required=True)
    p.add_argument('--gpu-index',type=int,choices=[7,5,4])
    args=p.parse_args();run=args.run
    plan=json.loads((run/'protocol.json').read_text())
    evaluation_split=plan.get('evaluation_split','calibration')
    assert evaluation_split in ['calibration','confirmation']
    assert all(r['split']==evaluation_split for r in plan['rows'])
    for name,digest in plan['source_sha256'].items():
        assert sha(Path(__file__).with_name(name))==digest,name
    uuid=os.environ['CUDA_VISIBLE_DEVICES']
    device=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
    assigned=args.gpu_index if args.gpu_index is not None else [7,5,4][args.shard]
    if int(device[0])!=assigned or int(device[1])>50 or int(device[2])!=0:
        raise RuntimeError('Assigned GPU is not idle; no alternative GPU or automatic retry')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=[r for i,r in enumerate(plan['rows']) if i%3==args.shard]
    if args.shard==0: rows=plan['bridge_rows']+rows
    out=run/'cache'/f'shard_{args.shard}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise FileExistsError('Completed shard is immutable')
    for r in rows:assert sha(run/r['image'])==r['image_sha256']
    started=time.monotonic()
    masker=SemanticMasker(args.root/'models/sembind/dinov2-giant',args.root/'models/sembind/sembind_masker.pth')
    for r in rows:
        code=masker.code(Image.open(run/r['image']).convert('RGB')).cpu().numpy().astype(np.uint8)
        np.savez_compressed(out/(r['id']+'.npz'),sembind_code=code)
    del masker;gc.collect();torch.cuda.empty_cache()
    pipe=load_pipe(plan['server_model'],'cuda')
    embedding=pipe.encode_prompt('',torch.device('cuda'),1,False)[0].detach()
    pipe.text_encoder.cpu();torch.cuda.empty_cache()
    scheduler=DDIMInverseScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(50,device='cuda')
    bridge=[]
    # The recorded development inversions used batch two and all 50 steps.
    with torch.inference_mode():
        for offset in range(0,len(rows),2):
            batch=rows[offset:offset+2];latents=[]
            for r in batch:
                image=Image.open(run/r['image']).convert('RGB')
                x=pipe.image_processor.preprocess(image).cuda().half()
                latents.append((pipe.vae.encode(x).latent_dist.mode()*pipe.vae.config.scaling_factor).float())
            z=torch.cat(latents);context=embedding.expand(len(batch),-1,-1)
            for t in scheduler.timesteps:
                pred=pipe.unet(z.half(),t,encoder_hidden_states=context).sample.float()
                z=scheduler.step(pred,t,z).prev_sample
            for r,latent in zip(batch,z.cpu().numpy()):
                assert np.isfinite(latent).all();np.save(out/(r['id']+'.npy'),latent)
                if r['split']=='development_bridge':
                    expected=np.load(run/r['expected_inversion'],allow_pickle=False)
                    expected_code=np.load(run/r['expected_features'],allow_pickle=False)['sembind_code']
                    observed_code=np.load(out/(r['id']+'.npz'),allow_pickle=False)['sembind_code']
                    relative=float(np.linalg.norm(latent.reshape(-1)-expected.reshape(-1))/np.linalg.norm(expected))
                    bridge.append(dict(id=r['id'],relative_l2=relative,code_equal=bool(np.array_equal(expected_code,observed_code))))
            if bridge and offset==0:
                write(out/'bridge.json',dict(rows=bridge))
                if any(not r['code_equal'] or r['relative_l2']>plan['bridge_max_relative_l2'] for r in bridge):
                    raise RuntimeError('Cross-device bridge failed before calibration inversion; do not score decisions')
            print('CACHED_BATCH',offset+len(batch),len(rows),flush=True)
    write(out/'bridge.json',dict(rows=bridge))
    if any(not r['code_equal'] or r['relative_l2']>plan['bridge_max_relative_l2'] for r in bridge):
        raise RuntimeError('Cross-device bridge failed; do not score calibration decisions')
    write(out/'complete.json',dict(completed=True,rows=[dict(r,inversion_sha256=sha(out/(r['id']+'.npy')),
        features_sha256=sha(out/(r['id']+'.npz'))) for r in rows],protocol_sha256=sha(run/'protocol.json'),
        gpu_index=int(device[0]),gpu_uuid=uuid,seconds=time.monotonic()-started,
        steps=50,batch=2,model_dtype='float16',inversion_state_dtype='float32',
        dependencies={k:importlib.metadata.version(k) for k in ['torch','diffusers','transformers']}))


if __name__=='__main__':main()
