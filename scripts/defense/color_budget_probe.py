"""Matched-byte image descriptors and split-wise verification on the frozen prompt pairs."""
import csv
import gzip
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from color_identity_probe import LOADER, ROOT, read, save, sha


def color_tables():
    lattice = np.array([(i, j, 63-i-j) for i in range(64) for j in range(64-i)])
    center = np.array([21, 21, 21])
    nearest = np.full(len(lattice), 10**6)
    palette = []
    for _ in range(144):
        palette.append(center)
        nearest = np.minimum(nearest, ((lattice-center)**2).sum(1))
        center = lattice[nearest.argmax()]
    r, g = np.indices((256, 256))
    grid = np.stack((r, g, 255-r-g), axis=2).reshape(-1, 3) * (63/255)
    return {n: cKDTree(palette[:n]).query(grid)[1].astype(np.int32) for n in (48, 96, 144)}


def rank_weights(mass, count):
    end = mass.cumsum()
    start = end-mass
    edges = np.linspace(0, end[-1], count+1)
    overlap = np.maximum(0, np.minimum(end, edges[1:, None])-np.maximum(start, edges[:-1, None]))
    return overlap / np.maximum(1, mass)


def pack(counts):
    mass = np.asarray(counts).ravel()
    exact = mass * (65535/mass.sum())
    ints = np.floor(exact).astype('<u2')
    remainder = 65535-int(ints.sum())
    assert len(ints)==144 and 0 <= remainder <= len(ints)
    ints[np.argsort(-(exact-ints), kind='stable')[:remainder]] += 1
    assert ints.sum()==65535 and ints.nbytes==288
    return ints


def descriptors(im, tables):
    rgb = np.asarray(im.convert('RGB'), dtype=np.int32)
    h, w = rgb.shape[:2]
    rgb = rgb.reshape(-1, 3)
    total = rgb.sum(1, dtype=np.int32)
    rg = (510*rgb[:, :2]+total[:, None]) // (2*np.maximum(1, total[:, None]))
    rg[total==0] = 85
    addr = rg[:, 0]*256 + rg[:, 1]
    y = np.asarray(im.convert('L'), dtype=np.int32).ravel()
    joint = np.bincount(y*48+tables[48][addr], minlength=256*48).reshape(256, 48)
    mass = joint.sum(1)
    thirds = rank_weights(mass, 3)
    tile = (np.arange(h)*4//h)[:, None]*4 + (np.arange(w)*4//w)[None, :]
    spatial = np.bincount(tile.ravel()*256+y, minlength=16*256).reshape(16, 256)
    global96 = np.bincount(tables[96][addr], minlength=96)
    space48 = spatial @ thirds.T
    return dict(global144=pack(np.bincount(tables[144][addr], minlength=144)),
                fixed3x48=pack(np.stack([joint[:85].sum(0), joint[85:170].sum(0), joint[170:].sum(0)])),
                rank3x48=pack(thirds @ joint), hybrid96_48=pack(np.r_[global96, space48.ravel()]),
                space16x9=pack(spatial @ rank_weights(mass, 9).T))


def rootmass(codes):
    return np.sqrt(codes.astype(np.float64)/65535)


def measure(run, plan, split, sources, waves, tables):
    images, clean, owners = [], [], []
    for s in sources:
        if s['split'] != split:
            continue
        dest = run/'images'/f"{s['task']:03d}"
        receipt = read(dest/'receipt.json')
        assert receipt['seeds']==s['seeds'] and receipt['prompt_sha256']==sha(s['prompt'].encode())
        for i, expected in enumerate(receipt['sha256']):
            path = dest/f'{i}.png'
            assert sha(path.read_bytes())==expected
            with Image.open(path) as file:
                im = file.convert('RGB')
            assert im.size==(512,512)
            owners.append((s['dataset'], s['task'], i))
            images.append(im)
            clean.append(descriptors(im, tables))
    assert len({sha(im.tobytes()) for im in images})==len(images)
    gallery = {}
    for m in plan['methods']:
        path = run/f'{split}_{m}.bin'
        path.write_bytes(b''.join(x[m].tobytes() for x in clean))
        assert path.stat().st_size==288*len(images)
        gallery[m] = rootmass(np.fromfile(path, dtype='<u2').reshape(-1,144))
    rows = []
    for q, (im, (dataset, task, seed_idx)) in enumerate(zip(images, owners)):
        for f, family in enumerate(['clean']+plan['families']):
            for j, strength in enumerate(plan['strengths'] if f else [0]):
                seed = 20261119 + 100*task + 2*f + j
                query = waves.apply_distortion([im],family,strength=strength,distortion_seed=seed)[0] if f else im
                desc = descriptors(query, tables) if f else clean[q]
                query_hash = sha(query.tobytes())
                for m in plan['methods']:
                    ds = cdist(rootmass(desc[m])[None], gallery[m])[0]/np.sqrt(2)
                    own, sibling = ds[q], ds[q ^ 1]
                    if not f:
                        assert own==0
                    rows.append(dict(split=split,dataset=dataset,task=task,seed_idx=seed_idx,family=family,
                        strength=strength,edit_seed=seed,query_sha256=query_hash,method=m,own=own,sibling=sibling,
                        pair=float(own<sibling)+.5*(own==sibling),
                        gallery=float(own==ds.min())/(ds==ds.min()).sum()))
    return rows


def quant95(rows):
    pos = sorted(r['own'] for r in rows)
    return float(pos[int(np.ceil(.95*len(pos)))-1])


def strong(r):
    return r['family']=='contrast' and r['strength']==.8


def thresholds(rows, methods):
    out = {}
    for m in methods:
        rr = [r for r in rows if r['method']==m and r['family']!='clean']
        neg = sorted(r['sibling'] for r in rr)
        out[m] = dict(pooled=quant95(rr),contrast=quant95([r for r in rr if strong(r)]),
                      far05=float(np.nextafter(neg[len(neg)//20], -np.inf)))
    return out


METRICS = ['pass95','sibling95','gallery','pair','pass_contrast_cal95','sibling_contrast_cal95',
           'pass_far05','sibling_far05']


def rates(rows, cut):
    return np.mean([[r['own']<=cut['pooled'],r['sibling']<=cut['pooled'],r['gallery'],r['pair'],
                     r['own']<=cut['contrast'],r['sibling']<=cut['contrast'],
                     r['own']<=cut['far05'],r['sibling']<=cut['far05']] for r in rows], axis=0)


def summary(rows, limits, methods):
    rng = np.random.default_rng(20261119)
    groups = sorted({(r['dataset'], r['task']) for r in rows})
    assert len(groups)==60 and groups[29][0]!=groups[30][0]
    ix = np.column_stack([rng.integers(0,30,(5000,30)),rng.integers(30,60,(5000,30))])
    info, samples = {}, {}
    for m in methods:
        rr = [r for r in rows if r['method']==m]
        subsets = dict(pooled=[r for r in rr if r['family']!='clean'],contrast=[r for r in rr if strong(r)],
                       clean=[r for r in rr if r['family']=='clean'])
        info[m], samples[m] = {}, {}
        for label, subset in subsets.items():
            vals = np.stack([rates([r for r in subset if r['task']==task], limits[m]) for _,task in groups])
            boot = vals[ix].mean(1)
            samples[m][label] = boot
            info[m][label] = dict(rates=dict(zip(METRICS, vals.mean(0))),
                ci95=dict(zip(METRICS, np.quantile(boot,[.025,.975],axis=0).T.tolist())),
                counts=dict(queries=len(subset),genuine_pass=sum(r['own']<=limits[m]['pooled'] for r in subset).item(),
                            sibling_pass=sum(r['sibling']<=limits[m]['pooled'] for r in subset).item()))
            if label!='clean':
                oracle = quant95(subset)
                info[m][label]['descriptive_test95'] = dict(cut=oracle,actual_tpr=np.mean([r['own']<=oracle for r in subset]),
                                                        far=np.mean([r['sibling']<=oracle for r in subset]))
        info[m]['conditions'] = {f'{f}/{s}':dict(zip(METRICS,rates([r for r in rr if r['family']==f and
                                   r['strength']==s], limits[m]))) for f,s in sorted({(r['family'],r['strength']) for r in rr})}
    diffs = {}
    for m in methods:
        for base in ('global144','space16x9'):
            diffs[f'{m}-{base}'] = {k:dict(zip(METRICS,np.quantile(samples[m][k]-samples[base][k],
                [.025,.975],axis=0).T.tolist())) for k in ('pooled','contrast')}
    return dict(test=info, paired_difference_ci95=diffs)


def main():
    import os
    import torch
    import scipy
    import PIL
    run = ROOT/'runs/color_budget_20261119'
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_available()
    torch.set_num_threads(1)
    tick = time.perf_counter()
    sys.path.insert(0,str(LOADER))
    from order_recovery_data import waves_module, WAVES
    waves = waves_module()
    tables = color_tables()
    plan = read(run/'color_budget_20261119.json')
    sources = read(run/'sources.json')
    for c in ('black','white'):
        check = descriptors(Image.new('RGB', (64,64), c), tables)
        assert len(check)==5 and np.ptp(check['space16x9'].astype(int))<=1
    cal = measure(run, plan, 'calibration', sources, waves, tables)
    cut = thresholds(cal,plan['methods'])
    save(run/'thresholds.json',cut)
    test = measure(run, plan, 'test', sources, waves, tables)
    assert len(cal)==6800 and len(test)==10200
    with gzip.open(run/'scores.csv.gz','wt',newline='') as f:
        writer = csv.DictWriter(f,list(test[0]))
        writer.writeheader()
        writer.writerows(cal+test)
    save(run/'summary.json',summary(test,cut,plan['methods']))
    files = [*run.glob('*.bin'), *run.glob('*.json'), *run.glob('*.py'), run/'scores.csv.gz',
             WAVES/'distortions/distortions.py', LOADER/'order_recovery_data.py']
    save(run/'complete.json',dict(seconds=time.perf_counter()-tick,score_rows=len(cal)+len(test),
        torch=torch.__version__,pillow=PIL.__version__,numpy=np.__version__,scipy=scipy.__version__,
        sha256={str(p):sha(p.read_bytes()) for p in files}))


if __name__=='__main__':
    main()
