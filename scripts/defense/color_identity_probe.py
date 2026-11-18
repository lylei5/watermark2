"""Paired image statistics on existing same-prompt independent-seed generations."""
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist

ROOT = Path('/data0/liyanlei/watermark_research')
PARENT = ROOT / 'runs/phase1_trajectory_native_gs_20261113'
LOADER = ROOT / 'runs/order_capsule_confirmation_20261115'
METHODS = ['rgb_share', 'rgb_hist', 'chroma', 'absolute_luma_chroma', 'rank_luma_chroma']
SEEDS = ('aux', 'same')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def features(image):
    rgb = np.asarray(image.convert('RGB')).reshape(-1, 3)
    totals = rgb.sum(axis=0, dtype=np.float64)
    marginal = np.concatenate([np.bincount(rgb[:, c] >> 3, minlength=32) for c in range(3)])
    total = rgb.sum(axis=1, dtype=np.int32)
    # Integer arithmetic keeps chromaticity bin boundaries reproducible.
    rg = np.minimum(11, rgb[:, :2].astype(np.int32)*12 // np.maximum(1, total[:, None]))
    rg[total == 0] = 4
    y = np.asarray(image.convert('L')).ravel().astype(np.int32)
    joint = np.bincount(y*144 + rg[:, 0]*12 + rg[:, 1], minlength=256*144).reshape(256, 144)
    mass = joint.sum(1)
    end = mass.cumsum()
    start = end - mass
    edges = np.linspace(0, len(rgb), 4)
    overlap = np.maximum(0, np.minimum(end, edges[1:, None]) - np.maximum(start, edges[:-1, None]))
    ranked = (overlap / np.maximum(mass, 1)) @ joint
    bands = np.stack([joint[:85].sum(0), joint[85:170].sum(0), joint[170:].sum(0)])
    assert np.allclose(ranked.sum(1), len(rgb)/3) and np.allclose(ranked.sum(0), joint.sum(0))
    counts = [totals if totals.sum() else np.ones(3), marginal, joint.sum(0), bands, ranked]
    return {m:np.sqrt(v.ravel()/v.sum()) for m,v in zip(METHODS, counts)}


def sources(plan):
    path = PARENT/'protocol.json'
    assert sha(path.read_bytes()) == plan['source_protocol_sha256']
    raw = read(path)['sources']
    assert len(raw) == len({s['caption'].strip().casefold() for s in raw}) == 100
    rows = {r['task']:r for f in PARENT.glob('generate_*/complete.json') for r in read(f)['rows']}
    grouped = {'calibration':[], 'test':[]}
    for dataset in ('COCO', 'SDP'):
        ordered = sorted((s for s in raw if s['dataset'] == dataset),
                         key=lambda s:sha(('color-id-v1/'+s['caption']).encode()))
        assert len(ordered) == 50
        for j, s in enumerate(ordered):
            assert s['aux_seed'] != s['independent_seed']
            refs = [rows[s['task']]['images'][k] for k in SEEDS]
            grouped['calibration' if j < 20 else 'test'].append(dict(task=s['task'], dataset=dataset,
                prompt_sha256=sha(s['caption'].encode()), images=refs, seeds=[s['aux_seed'],s['independent_seed']]))
    return grouped


def measure(group, plan, waves, split):
    images, owners, original = [], [], []
    for s in group:
        for name, info in zip(SEEDS, s['images']):
            path = PARENT/info['path']
            assert sha(path.read_bytes()) == info['sha256']
            with Image.open(path) as file:
                image = file.convert('RGB')
            assert image.size == (512, 512)
            images.append(image)
            original.append(features(image))
            owners.append((s['dataset'], s['task'], name))
    assert len({sha(im.tobytes()) for im in images}) == len(images)
    gallery = {m:np.stack([x[m] for x in original]) for m in METHODS}
    rows = []
    for q, (im, (dataset, task, name)) in enumerate(zip(images, owners)):
        for f, family in enumerate(['clean']+plan['edit_families']):
            for level, strength in enumerate(plan['strengths'] if f else [0]):
                seed = 20261119 + 100*task + 2*f + level
                query = im if not f else waves.apply_distortion([im],family,strength=strength,distortion_seed=seed)[0]
                stats = original[q] if not f else features(query)
                pixel_hash = sha(query.tobytes())
                for m in METHODS:
                    ds = cdist(stats[m][None], gallery[m], metric='euclidean')[0] / np.sqrt(2)
                    own, other = ds[q], ds[q ^ 1]
                    if not f:
                        assert own == 0
                    minimum = ds.min()
                    rows.append(dict(split=split,dataset=dataset,task=task,seed_image=name,family=family,
                        strength=strength,edit_seed=seed,query_sha256=pixel_hash,method=m,
                        own=own,sibling=other,pair=float(own<other)+.5*(own==other),
                        gallery=float(own==minimum)/np.count_nonzero(ds==minimum)))
    return rows


def thresholds(rows):
    out = {}
    for m in METHODS:
        edited = [r for r in rows if r['method']==m and r['family']!='clean']
        neg = np.sort([r['sibling'] for r in edited])
        pos = np.sort([r['own'] for r in edited])
        out[m] = [float(np.nextafter(neg[len(neg)//20], -np.inf)), float(pos[int(np.ceil(.95*len(pos)))-1])]
    return out


def means(rows, cut):
    a,b = cut
    return np.mean([[r['own']<=a, r['sibling']<=a, r['pair'], r['gallery'],
                     r['own']<=b, r['sibling']<=b] for r in rows], axis=0)


def summarize(rows, cut):
    result, condition, stratified, prompt = {}, {}, {}, {}
    keys = ['true_accept','sibling_accept','pair_top1','gallery_top1','true_at95','sibling_at95']
    for method in METHODS:
        rr = [r for r in rows if r['method']==method]
        edits = [r for r in rr if r['family']!='clean']
        groups = sorted({(r['dataset'],r['task']) for r in edits})
        prompt[method] = np.stack([means([r for r in edits if r['task']==task], cut[method]) for _,task in groups])
        result[method] = dict(zip(keys, prompt[method].mean(0)))
        clean = [r for r in rr if r['family']=='clean']
        result[method]['clean_sibling_accept'] = float(np.mean([r['sibling']<=cut[method][0] for r in clean]))
        condition[method] = {f'{f}/{s}':dict(zip(keys,means([r for r in edits if r['family']==f and r['strength']==s],cut[method])))
            for f,s in sorted({(r['family'],r['strength']) for r in edits})}
        stratified[method] = {d:dict(zip(keys,means([r for r in edits if r['dataset']==d],cut[method]))) for d in ('COCO','SDP')}
    rng = np.random.default_rng(20261119)
    half = len(groups)//2
    ix = np.column_stack([rng.integers(0,half,(5000,half)),rng.integers(half,2*half,(5000,half))])
    intervals, contrasts = {}, {}
    for m in METHODS:
        boot = prompt[m][ix].mean(1)
        intervals[m] = dict(zip(keys,np.quantile(boot,[.025,.975],axis=0).T.tolist()))
        diff = (prompt[m]-prompt['chroma'])[ix].mean(1)
        contrasts[m] = dict(zip(keys,np.quantile(diff,[.025,.975],axis=0).T.tolist()))
    return dict(aggregate=result, by_condition=condition, by_dataset=stratified, ci95=intervals,
                paired_difference_from_global_chroma_ci95=contrasts)


def main(run):
    import torch
    import PIL
    tick = time.perf_counter()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_available()
    torch.set_num_threads(1)
    plan = read(run/'color_identity_20261119.json')
    sys.path.insert(0,str(LOADER))
    from order_recovery_data import waves_module, WAVES
    waves = waves_module()
    catalog = sources(plan)
    save(run/'sources.json', catalog)
    cal = measure(catalog['calibration'], plan, waves,'calibration')
    cut = thresholds(cal)
    save(run/'thresholds.json', cut)
    test = measure(catalog['test'], plan, waves,'test')
    assert len(cal)==40*2*17*5 and len(test)==60*2*17*5
    with (run/'scores.csv').open('x',newline='') as f:
        writer = csv.DictWriter(f,list(test[0]));writer.writeheader();writer.writerows(cal+test)
    save(run/'summary.json',summarize(test,cut))
    files = [*sorted(run.glob('*.json')),run/'scores.csv',Path(__file__),
             WAVES/'distortions/distortions.py',WAVES/'utils/image_utils.py',WAVES/'utils/exp_utils.py',
             LOADER/'order_recovery_data.py',PARENT/'run_phase1_probe_worker.py']
    save(run/'complete.json',dict(seconds=time.perf_counter()-tick,python=platform.python_version(),
         pillow=PIL.__version__,torch=torch.__version__,numpy=np.__version__,device='cpu',rows=len(cal)+len(test),
         sha256={str(f):sha(f.read_bytes()) for f in files}))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
