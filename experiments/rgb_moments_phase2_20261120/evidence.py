"""Validate score coverage and package the exact frozen comparison images."""
import csv
import gzip
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent/'rgb_moments_phase1_20261120'
sys.path.insert(0,str(PREVIOUS))
from generate import idle_gpu
from prepare import read, hashfile


def main():
    for gpu in (4,5,7):
        uuid = idle_gpu(gpu)
        if uuid:
            break
    else:
        sys.exit('No allowed idle GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    import lpips
    import numpy as np
    import torch
    from PIL import Image
    from quality import tensor

    with gzip.open(HERE/'test_scores.csv.gz','rt') as f:
        rows = list(csv.DictReader(f))
    keys = [(r['split'],r['sample'],r['condition'],r['method']) for r in rows]
    assert len(keys)==len(set(keys))
    for r in rows:
        assert (float(r['score'])<=float(r['threshold']))==bool(int(r['passed']))
    assert sum(r['split']=='test' for r in rows)==40*8*7
    assert sum(r['split']=='public' for r in rows)==(2+16+8)*5
    test = read(HERE/'calibration.json')['test_ids']
    source = read(PREVIOUS/'test_sources.json')
    prior = read(PREVIOUS/'test_quality.json')['rows']
    assert len({s['prompt'] for s in source if s['id'] in test})==20
    net = lpips.LPIPS(net='alex').cuda().eval()
    torch.set_num_threads(2)
    def load_tensor(path):
        with Image.open(path) as im:
            return tensor(im.convert('RGB')).cuda()

    lp = []
    saved = []
    bundle = HERE/'comparison_images.zip'
    with torch.inference_mode(), zipfile.ZipFile(bundle,'w') as archive:
        for i in test:
            s=source[i]
            path=PREVIOUS/'test'/f'{i:03d}'/'joint_gn8.png'
            sib=PREVIOUS/'test'/f'{i^1:03d}'/'joint_gn8.png'
            donor=Path(s['donor'])
            ref=load_tensor(path)
            other=float(net(ref,load_tensor(sib)).item())
            expected=next(r['seed_lpips'] for r in prior if r['id']==(i|1) and r['arm']=='joint_gn8')
            assert abs(other-expected)<.001
            lp.append(dict(id=i,cross_seed_lpips=other,donor_lpips=float(net(ref,load_tensor(donor)).item())))
            for key,p in [('registered',path),('donor',donor)]:
                name=f'phase1/{i:03d}/{key}.png'
                archive.write(p,name)
                saved.append(dict(file=name,sha256=hashfile(p),sample=i,sibling=i^1))
        for p in sorted((HERE/'author').iterdir()):
            archive.write(p,'public/'+p.name)
        archive.writestr('phase1_manifest.json',json.dumps(saved,indent=2))
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None

    cols = []
    for condition in ('all_edits','cross_seed','donor','distribution_output','color_output','replacement_output'):
        sub=[r for r in rows if (r['condition'] in ('compression','blurring','noise','crop50','resize70')
                                if condition=='all_edits' else r['condition']==condition)]
        pairs={}
        for r in sub:
            pairs.setdefault((r['sample'],r['condition']),{})[r['method']]=int(r['passed'])
        cols.append(dict(condition=condition,n=len(pairs),
                         both=sum(v['hist'] and v['local'] for v in pairs.values()),
                         hist_only=sum(v['hist'] and not v['local'] for v in pairs.values()),
                         local_only=sum(v['local'] and not v['hist'] for v in pairs.values()),
                         neither=sum(not v['local'] and not v['hist'] for v in pairs.values())))
    tolerance={}
    for cond in ('distribution_output','distribution_source','color_output','color_source','replacement_output'):
        tolerance[cond]={k:sum(float(r['score'])<=1 for r in rows if r['condition']==cond and r['method']==k)
                         for k in ('mean','moments')}
    files=['calibration.json','pilot_scores.csv.gz','test_scores.csv.gz','pilot_summary.json','test_summary.json']
    out=dict(gpu=gpu,uuid=uuid,internal_lpips=lp,disagreement=cols,original_tolerances=tolerance,
             archive_sha256=hashfile(bundle),archive_images=len(saved)+sum(len(ex['files']) for ex in read(HERE/'author/manifest.json')),
             sha256={f:hashfile(HERE/f) for f in files},heldout_scores=len(rows))
    (HERE/'audit.json').write_text(json.dumps(out,indent=2))
    print('Checked',len(rows),'scores on GPU',gpu,'and saved',out['archive_images'],'images')


if __name__=='__main__':
    main()
