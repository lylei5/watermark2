"""Score frozen enrollments and fixed image pairs on server 249."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import brentq
from scipy.special import rel_entr

HERE = Path(__file__).resolve().parent
PHASE1 = HERE.parent / 'rgb_moments_phase1_20261120'
sys.path.insert(0, str(PHASE1))
from prepare import ROOT, read, hashfile
from edits import official_geometry
from torchvision.transforms import functional
import torch

EDITS = ('compression','blurring','noise','crop50','resize70')
PRIMARY = ('mean','moments','hist','local')
ASSIGNED = ('assigned_mean','assigned_moments')


def feature(image):
    pixels = np.array(image.convert('RGB'))
    x = pixels.astype(np.float64)/255
    flat = x.reshape(-1,3)
    mean, var = flat.mean(0), flat.var(0)
    cdf = np.stack([np.bincount(pixels[:,:,k].ravel()//8, minlength=32).cumsum()/len(flat)
                    for k in range(3)])
    cells = []
    for stripe in np.array_split(x, 4, axis=0):
        for block in np.array_split(stripe, 4, axis=1):
            b = block.reshape(-1,3)
            cells.append(np.r_[b.mean(0), b.var(0)])
    return mean, var, cdf, np.stack(cells)


def distances(ref, query, target=None):
    m, v, cdf, local = query
    dm = np.max(np.abs(m-ref[0]))/.02
    d = dict(mean=dm, moments=max(dm,np.max(np.abs(v-ref[1]))/.01),
             hist=np.abs(cdf[:,:-1]-ref[2][:,:-1]).mean(),
             local=(np.abs(local-ref[3])/np.array([.02]*3+[.01]*3)).mean())
    if target is not None:
        tm = np.max(np.abs(m-target[:3]))/.02
        d.update(assigned_mean=tm, assigned_moments=max(tm,np.max(np.abs(v-target[3:]))/.01))
    return d


def comparisons(part, samples, archive, old):
    from order_recovery_data import waves_module
    waves, geom = waves_module(), official_geometry()
    rows = []
    for s in samples:
        i = s['id']
        folder = PHASE1/archive/f'{i:03d}'
        path = folder/'joint_gn8.png'
        assert hashfile(path)==read(folder/'record.json')['arms']['joint_gn8']['sha256']
        ref_image = Image.open(path).convert('RGB')
        ref = feature(ref_image)
        t = np.array(s['target'])
        x = functional.to_tensor(ref_image)
        imgs = dict(clean=ref_image, crop50=functional.to_pil_image(geom['center_crop'](x,.5)),
                    resize70=functional.to_pil_image(geom['resize'](x,.7)))
        for edit in EDITS[:3]:
            imgs[edit] = waves.apply_distortion([ref_image],edit,strength=.4,distortion_seed=20261120+s['pair'])[0]
        sibling = PHASE1/archive/f'{i^1:03d}'
        assert hashfile(sibling/'joint_gn8.png')==read(sibling/'record.json')['arms']['joint_gn8']['sha256']
        imgs['cross_seed'] = Image.open(sibling/'joint_gn8.png').convert('RGB')
        assert hashfile(Path(s['donor']))==s['donor_sha256']
        imgs['donor'] = Image.open(s['donor']).convert('RGB')
        for condition,img in imgs.items():
            q = ref if condition=='clean' else feature(img)
            if archive=='test' and condition in ('clean',*EDITS):
                assert np.allclose(np.r_[q[0],q[1]],old[i,condition],rtol=0,atol=1e-10)
            if condition=='donor':
                assert np.allclose(np.r_[q[0],q[1]],t,rtol=0,atol=1e-10)
            for method,score in distances(ref,q,t).items():
                rows.append(dict(split=part,domain=s['dataset'],cluster=s['pair'],sample=i,
                                 condition=condition,method=method,score=float(score)))
    return rows


def public_rows():
    folder = HERE/'author'
    rows = []
    for ex in read(folder/'manifest.json'):
        images = {}
        for k,name in ex['files'].items():
            assert hashfile(folder/name)==ex['sha256'][k]
            with Image.open(folder/name) as im:
                images[k] = feature(im)
        for k in ('source','output'):
            if k not in images:
                continue
            for method,score in distances(images['ref'],images[k]).items():
                rows.append(dict(split='public',domain=ex['group'],cluster=ex['item'],sample=ex['item'],
                                 condition=ex['group']+'_'+k,method=method,score=float(score)))
    return rows


def fuse(rows, scales):
    new = []
    groups = {}
    for r in rows:
        groups.setdefault((r['split'],r['condition'],r['sample']),{})[r['method']] = r
    for group in groups.values():
        score = max(group[k]['score']/scales[k] for k in ('hist','local'))
        new.append(dict(group['hist'],method='hist_local',score=score))
    return rows + new


def quantile(rows, k):
    scores = sorted(r['score'] for r in rows if r['method']==k and r['condition'] in EDITS)
    return scores[math.ceil(.95*len(scores))-1]


def bounded_interval(rate,n):
    # Two-sided Chernoff bound for the average of independent [0,1] cluster means.
    def f(p):
        return n*sum(rel_entr([rate,1-rate],[p,1-p]))-math.log(2/.05)
    lo = brentq(f,0,rate) if rate else 0.
    hi = brentq(f,rate,1) if rate<1 else 1.
    return [lo,hi]


def summarize(rows):
    output = []
    for cond in sorted({r['condition'] for r in rows}|({'all_edits'} if any(r['condition'] in EDITS for r in rows) else set())):
        sub = [r for r in rows if (r['condition'] in EDITS if cond=='all_edits' else r['condition']==cond)]
        for method in sorted({r['method'] for r in sub}):
            sel = [r for r in sub if r['method']==method]
            clusters = {}
            for r in sel:
                clusters.setdefault(r['cluster'],[]).append(r['passed'])
            assert len({len(v) for v in clusters.values()})==1
            n = sum(r['passed'] for r in sel)
            rate = n/len(sel)
            lo,hi = bounded_interval(rate,len(clusters))
            output.append(dict(condition=cond,method=method,accepted=n,total=len(sel),rate=rate,
                               clusters=len(clusters),lo95=lo,hi95=hi))
    return output


def save(name,obj):
    (HERE/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8')


def main(stage):
    torch.set_num_threads(1)
    sys.path.insert(0,str(ROOT/'runs/order_capsule_confirmation_20261115'))
    src = read(PHASE1/'test_sources.json')
    pilot = read(PHASE1/'refine_sources.json')
    assert set(s['prompt'] for s in pilot).isdisjoint(s['prompt'] for s in src)
    labels = {}
    for dataset in ('COCO','SDP'):
        prompts = {s['prompt'] for s in src if s['dataset']==dataset}
        ordered = sorted(prompts,key=lambda p:hashlib.sha256(('rgb-phase2/'+p).encode()).digest())
        assert len(ordered)==16
        labels.update({p:'cal' if i<6 else 'test' for i,p in enumerate(ordered)})
    with gzip.open(PHASE1/'test_edits.csv.gz','rt') as f:
        old = {(int(r['id']),r['edit']):[float(r[f'moment{j}']) for j in range(6)]
               for r in csv.DictReader(f) if r['arm']=='joint_gn8'}
    if stage=='pilot':
        subset = [s for s in src if labels[s['prompt']]=='cal']
        cal = comparisons('cal',subset,'test',old)
        scales = {k:quantile(cal,k) for k in PRIMARY+ASSIGNED}
        assert min(scales.values())>0
        all_cal = fuse(cal,scales)
        thresholds = {**scales,'hist_local':quantile(all_cal,'hist_local')}
        save('calibration.json',dict(thresholds=thresholds,calibration_ids=[s['id'] for s in subset],
                                     test_ids=[s['id'] for s in src if labels[s['prompt']]=='test']))
        rows = all_cal+fuse(comparisons('pilot',pilot,'refine',old),scales)
    else:
        frozen = read(HERE/'calibration.json')
        thresholds = frozen['thresholds']
        subset = [s for s in src if labels[s['prompt']]=='test']
        assert [s['id'] for s in subset]==frozen['test_ids']
        rows = fuse(comparisons('test',subset,'test',old)+public_rows(),thresholds)
    for r in rows:
        r['threshold'] = thresholds[r['method']]
        r['passed'] = int(r['score']<=r['threshold'])
    with gzip.open(HERE/(stage+'_scores.csv.gz'),'wt',newline='') as f:
        writer = csv.DictWriter(f,list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save(stage+'_summary.json',{split:summarize([r for r in rows if r['split']==split])
                                 for split in sorted({r['split'] for r in rows})})


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['pilot','test'])
    main(parser.parse_args().stage)
