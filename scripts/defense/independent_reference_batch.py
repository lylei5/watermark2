"""Batched reference generation with official pretrained content heads."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from cryptography.hazmat.primitives.ciphers import Cipher,algorithms
from codeword_binding import keyed_order,sembind_mask_bits
from independent_generation import load_pipe,write,sha

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--batch',type=int,default=4);args=p.parse_args()
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
plan=json.loads((args.root/'generation_plan.json').read_text())
pipe=load_pipe(args.model,'cuda');start=time.monotonic();pending=[]
for r in plan['rows']:
    methods=plan['methods']+(['gs_control'] if r['split']=='development' and r['split_index']<4 else [])
    for method in methods:
        dest=args.root/'references'/f"{r['task']:03}_{method}.png"
        if not dest.exists():pending.append((r,method,dest))
for offset in range(0,len(pending),args.batch):
    batch=pending[offset:offset+args.batch];latents=[]
    for r,method,dst in batch:
        i=r['task'];s=json.loads((args.root/'private'/f'{i:03}.json').read_text())
        key,nonce,msg,bkey=[bytes.fromhex(s[k]) for k in ['key','nonce','message','binding_key']]
        stream=Cipher(algorithms.ChaCha20(key,nonce),mode=None).encryptor().update(msg*64)
        bits=np.unpackbits(np.frombuffer(stream,dtype=np.uint8))
        z=np.abs(np.random.default_rng(s['amplitude_seed']).standard_normal(16384))*(2*bits.astype(np.float32)-1)
        codes=np.load(args.root/'aux_codes'/f'{i:03}.npz')
        if method=='sembind_scattered':mask=sembind_mask_bits(codes['sembind_code'][None],keyed_order(bkey,b'sembind/permutation/',16384))[0]
        elif method=='coverlock_g_message':mask=np.tile(codes['coverlock_code'],64)
        else:mask=np.zeros(16384,dtype=np.uint8)
        latents.append((z*(1-2*mask.astype(np.float32))).astype(np.float32).reshape(4,64,64))
    latent=torch.from_numpy(np.stack(latents)).cuda().half()
    images=pipe([r['caption'] for r,_,_ in batch],latents=latent,num_inference_steps=50,guidance_scale=7.5,height=512,width=512).images
    for (r,method,dst),image in zip(batch,images):
        dst.parent.mkdir(exist_ok=True);image.save(dst)
        print('REFERENCE',r['task'],method,flush=True)
write(args.root/'reference_complete.json',dict(completed=True,seconds=time.monotonic()-start,n=2*len(plan['rows'])+4,batch=args.batch,script_sha256=sha(__file__)))
