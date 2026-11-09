"""Generate benign-query assets and cache inversions/features by split.

Confirmation numerical decisions are intentionally deferred until calibration
has been frozen. Inversion and feature caches can be built before that freeze.
"""
import argparse,io,json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageFilter
from independent_generation import load_pipe,invert,write,sha
from research_semantics import ResearchSemantics


def variants(image,seed):
    yield 'clean',image
    for q in [50,75]:
        buf=io.BytesIO();image.save(buf,format='JPEG',quality=q);buf.seek(0)
        yield f'jpeg{q}',Image.open(buf).convert('RGB')
    yield 'blur1_resize80',image.filter(ImageFilter.GaussianBlur(1)).resize((409,409),Image.Resampling.BICUBIC).resize((512,512),Image.Resampling.BICUBIC)
    yield 'crop90',image.crop((25,25,486,486)).resize((512,512),Image.Resampling.BICUBIC)
    rng=np.random.default_rng(seed)
    x=np.asarray(image).astype(float)/255+rng.normal(0,.01,(512,512,3))
    yield 'noise0.01',Image.fromarray((x.clip(0,1)*255).round().astype(np.uint8))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--model');p.add_argument('--semantic');p.add_argument('--coverlock');p.add_argument('--stage',choices=['queries','inversions','features'],required=True);p.add_argument('--split',choices=['development','calibration','confirmation','all'],default='all');p.add_argument('--device',default='cuda');args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    root=args.root;plan=json.loads((root/'generation_plan.json').read_text());records=[];start=time.monotonic()
    for r in plan['rows']:
        if args.split!='all' and r['split']!=args.split:continue
        methods=plan['methods']+(['gs_control'] if r['split']=='development' and r['split_index']<4 else [])
        for method in methods:
            base=f"{r['task']:03}_{method}";ref=root/'references'/f'{base}.png'
            if not ref.exists():raise FileNotFoundError(ref)
            for label in ['clean','jpeg50','jpeg75','blur1_resize80','crop90','noise0.01']:
                records.append(dict(task=r['task'],split=r['split'],method=method,label=label,id=base+'_'+label,reference=str(ref)))
    if args.stage=='queries':
        for r in records:
            dest=root/'queries'/f"{r['id']}.png"
            if dest.exists():continue
            dest.parent.mkdir(exist_ok=True)
            im=Image.open(r['reference']).convert('RGB')
            for label,image in variants(im,110900+r['task']):
                if label==r['label']:image.save(dest);break
        write(root/f'query_manifest_{args.split}.json',dict(rows=records,completed=True))
        if args.split=='all':
            for split in ['development','calibration','confirmation']:
                write(root/f'query_manifest_{split}.json',dict(rows=[r for r in records if r['split']==split],completed=True))
        return
    if args.stage=='inversions':model=load_pipe(args.model,args.device)
    else:model=ResearchSemantics(args.semantic,args.coverlock,args.device)
    for i,r in enumerate(records):
        folder='inversions' if args.stage=='inversions' else 'query_features'
        suffix='.npy' if args.stage=='inversions' else '.npz'
        dest=root/folder/(r['id']+suffix)
        if dest.exists():continue
        dest.parent.mkdir(exist_ok=True)
        image=Image.open(root/'queries'/f"{r['id']}.png").convert('RGB')
        if args.stage=='inversions':np.save(dest,invert(model,image))
        else:np.savez_compressed(dest,**model.extract(image))
        if (i+1)%12==0:print(args.stage,i+1,len(records),flush=True)
    write(root/f'{args.stage}_{args.split}_complete.json',dict(completed=True,n=len(records),seconds=time.monotonic()-start,script_sha256=sha(__file__)))


if __name__=='__main__':main()
