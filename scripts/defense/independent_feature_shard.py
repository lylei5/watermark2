"""CPU-friendly feature cache shard over a frozen query manifest."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from research_semantics import ResearchSemantics
from independent_generation import write,sha

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--semantic',required=True);p.add_argument('--coverlock',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--threads',type=int,default=4);p.add_argument('--device',default='cpu');args=p.parse_args()
torch.set_num_threads(args.threads);torch.backends.cuda.matmul.allow_tf32=False
records=json.loads(args.manifest.read_text())['rows'];args.output.mkdir(exist_ok=True,parents=True)
model=ResearchSemantics(args.semantic,args.coverlock,args.device);start=time.monotonic();count=0
for i,r in enumerate(records):
    if i%args.shards!=args.shard:continue
    dest=args.output/f"{r['id']}.npz"
    if dest.exists():continue
    image=Image.open(args.root/'queries'/f"{r['id']}.png").convert('RGB')
    arrays=model.extract(image)
    tmp=dest.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(dest)
    count+=1
    if count%12==0:print('FEATURE_SHARD',args.shard,count,round(time.monotonic()-start,1),flush=True)
write(args.output/f'shard_{args.shard}_complete.json',dict(shard=args.shard,shards=args.shards,completed=True,computed=count,seconds=time.monotonic()-start,manifest_sha256=sha(args.manifest),script_sha256=sha(__file__)))
