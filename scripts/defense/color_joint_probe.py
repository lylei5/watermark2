"""Separate color and layout distances at the same 288-byte enrollment budget."""
import csv
import gzip
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist

from color_budget_probe import color_tables, descriptors, rootmass
from color_identity_probe import ROOT, LOADER, read, save, sha

PILOT = ROOT/'runs/color_budget_20261119'
RUN = ROOT/'runs/color_joint_20261119'
METHODS = ('joint','hybrid96_48','space16x9')


def score(x, gallery):
    return cdist(x[None],gallery)[0]/np.sqrt(2)


def blocks(packed):
    hybrid = packed['hybrid96_48']
    color, layout = hybrid[:96], hybrid[96:]
    return dict(color=np.sqrt(color/color.sum()),layout=np.sqrt(layout/layout.sum()),
                hybrid96_48=rootmass(hybrid),space16x9=rootmass(packed['space16x9']))


def joint(c,l,null):
    def lower(v, key):
        return (1+np.searchsorted(null[key],v,side='right'))/(1+len(null[key]))
    return lower(c,'color')*lower(l,'layout')


def collect(path, split, plan, waves, tables, null=None):
    refs, images, owners = [], [], []
    for s in read(path/'sources.json'):
        if s['split']!=split:
            continue
        dest=path/'images'/f"{s['task']:03d}"
        receipt=read(dest/'receipt.json')
        assert s['seeds']==receipt['seeds'] and sha(s['prompt'].encode())==receipt['prompt_sha256']
        for i,h in enumerate(receipt['sha256']):
            file=dest/f'{i}.png'
            assert sha(file.read_bytes())==h
            with Image.open(file) as img:
                im=img.convert('RGB')
            images.append(im)
            owners.append((s['dataset'],s['task'],i))
            refs.append(descriptors(im,tables))
    for m in METHODS[1:]:
        data=b''.join(x[m].tobytes() for x in refs)
        target=RUN/f'{split}_{m}.bin'
        target.write_bytes(data)
        assert target.stat().st_size==288*len(images)
        if split=='calibration':
            assert data==(PILOT/target.name).read_bytes()
        for i,code in enumerate(np.fromfile(target,dtype='<u2').reshape(-1,144)):
            refs[i][m]=code
    decoded=[blocks(x) for x in refs]
    gallery={k:np.stack([x[k] for x in decoded]) for k in decoded[0]}
    rows=[]
    for q,(im,(dataset,task,idx)) in enumerate(zip(images,owners)):
        for f,family in enumerate(['clean']+plan['families']):
            for j,level in enumerate(plan['strengths'] if f else [0]):
                seed=20261119+100*task+2*f+j
                edit=waves.apply_distortion([im],family,strength=level,distortion_seed=seed)[0] if f else im
                b=blocks(descriptors(edit,tables)) if f else decoded[q]
                ds={k:score(b[k],gallery[k]) for k in b}
                if null:
                    ds['joint']=joint(ds['color'],ds['layout'],null)
                row=dict(dataset=dataset,task=task,idx=idx,family=family,level=level,seed=seed,sha256=sha(edit.tobytes()))
                for k,d in ds.items():
                    row[k+'_own']=float(d[q])
                    row[k+'_sibling']=float(d[q^1])
                    row[k+'_gallery']=float(d[q]==d.min())/int((d==d.min()).sum())
                rows.append(row)
    return rows


def cutoff(rows,method):
    return float(np.quantile([r[method+'_own'] for r in rows],.95,method='inverted_cdf'))


def write_rows(path, rows):
    with gzip.open(path,'wt',newline='') as f:
        writer=csv.DictWriter(f,list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    import torch
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_available()
    torch.set_num_threads(1)
    tick=time.monotonic()
    sys.path.insert(0,str(LOADER))
    from order_recovery_data import waves_module
    waves,tables=waves_module(),color_tables()
    plan=read(PILOT/'color_budget_20261119.json')
    cal=collect(PILOT,'calibration',plan,waves,tables)
    edited=[r for r in cal if r['family']!='clean']
    assert len(edited)==1280
    null={k:sorted(r[k+'_sibling'] for r in edited) for k in ('color','layout')}
    for r in cal:
        for key in ('own','sibling'):
            r['joint_'+key]=float(joint(r['color_'+key],r['layout_'+key],null))
    cuts={m:cutoff(edited,m) for m in METHODS}
    old=read(PILOT/'thresholds.json')
    assert all(cuts[m]==old[m]['pooled'] for m in METHODS[1:])
    save(RUN/'calibration.json',dict(thresholds=cuts,null=null))
    write_rows(RUN/'cal_scores.csv.gz',cal)
    rows=collect(RUN,'test',plan,waves,tables,null)
    assert len(rows)==2040
    write_rows(RUN/'test_scores.csv.gz',rows)
    parts=dict(pooled=[r for r in rows if r['family']!='clean'],
               contrast=[r for r in rows if r['family']=='contrast' and r['level']==.8],
               clean=[r for r in rows if r['family']=='clean'])
    keys=('pass','sibling','gallery')
    rng=np.random.default_rng(20261119)
    ix=np.column_stack([rng.integers(0,30,(5000,30)),rng.integers(30,60,(5000,30))])
    result={}
    for group,part in parts.items():
        result[group],boots={},{}
        for m,t in cuts.items():
            values=np.array([[r[m+'_own']<=t,r[m+'_sibling']<=t,r[m+'_gallery']] for r in part])
            task=np.array([r['task'] for r in part])
            boots[m]=np.stack([values[task==i].mean(0) for i in range(60)])[ix].mean(1)
            q=cutoff(part,m)
            result[group][m]=dict(n=len(part),counts=dict(zip(keys,values.sum(0))),
                ci95=dict(zip(keys,np.quantile(boots[m],[.025,.975],axis=0).T.tolist())),
                descriptive_test95=dict(actual=np.mean([r[m+'_own']<=q for r in part]),
                                        far=np.mean([r[m+'_sibling']<=q for r in part])))
        result[group]['joint_minus_hybrid_ci95']=dict(zip(keys,np.quantile(
            boots['joint']-boots['hybrid96_48'],[.025,.975],axis=0).T.tolist()))
    save(RUN/'summary.json',result)
    files=[*RUN.glob('*.json'),*RUN.glob('*.bin'),*RUN.glob('*.gz'),*RUN.glob('*.py')]
    save(RUN/'complete.json',dict(seconds=time.monotonic()-tick,cal_queries=len(cal),test_queries=len(rows),
         sha256={str(p):sha(p.read_bytes()) for p in files}))


if __name__=='__main__':
    main()
