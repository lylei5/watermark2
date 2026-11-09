import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from research_semantics import ResearchSemantics

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--sembind',required=True);p.add_argument('--coverlock',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');args=p.parse_args()
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
args.output.mkdir(exist_ok=True,parents=True)
rows=json.loads((args.data/'manifest.json').read_text())['rows']
model=ResearchSemantics(args.sembind,args.coverlock,args.device)
start=time.monotonic()
for i,row in enumerate(rows):
    dst=args.output/f"{row['image_id']:012}.npz"
    if dst.exists():continue
    path=args.data/row['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
    np.savez_compressed(dst,**model.extract(Image.open(path).convert('RGB')))
    if (i+1)%8==0:print('FEATURES',i+1,round(time.monotonic()-start,1),flush=True)
(args.output/'complete.json').write_text(json.dumps(dict(n=len(rows),device=args.device,seconds=time.monotonic()-start)))
