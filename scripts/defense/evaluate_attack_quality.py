"""Content-preservation metrics independent of verifier acceptance."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
import lpips
from PIL import Image
from skimage.metrics import structural_similarity

p=argparse.ArgumentParser();p.add_argument('--jobs',type=Path,required=True);p.add_argument('--attacks',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');args=p.parse_args()
torch.set_num_threads(4)
model=lpips.LPIPS(net='alex').to(args.device).eval()
jobs=json.loads(args.jobs.read_text())['jobs'];rows=[]
with torch.no_grad():
    for job in jobs:
        a=np.asarray(Image.open(job['cover']).convert('RGB')).astype(np.float32)/255
        for step in [0,30,75,151]:
            path=args.attacks/job['id']/f'update_{step}.png'
            if not path.exists():continue
            b=np.asarray(Image.open(path).convert('RGB')).astype(np.float32)/255
            mse=float(((a-b)**2).mean());psnr=-10*math.log10(max(mse,1e-15))
            ssim=float(structural_similarity(a,b,channel_axis=2,data_range=1.0))
            tensors=[torch.from_numpy(x).permute(2,0,1)[None].to(args.device)*2-1 for x in [a,b]]
            distance=float(model(*tensors).item())
            rows.append(dict(id=job['id'],task=job['task'],method=job['method'],step=step,psnr=psnr,ssim=ssim,lpips_alex=distance))
        print('QUALITY',job['id'],flush=True)
args.output.write_text(json.dumps(dict(rows=rows,lpips_backbone='alex',reference='attack cover',acceptance_not_used=True),indent=2))
