"""Frozen and TPR-matched comparisons of two decisions on identical image bytes."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

from color_identity_probe import LOADER, ROOT, read, save
from color_joint_probe import collect, color_tables, write_rows

RUN=ROOT/'runs/color_veto_20261119'
PARENT=ROOT/'runs/color_joint_20261119'
PILOT=ROOT/'runs/color_budget_20261119'
METHODS=('product','veto')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with gzip.open(path,'rt') as f:
        return list(csv.DictReader(f))


def inputs(rows,null):
    out={}
    for who in ('own','sibling'):
        components=[(1+np.searchsorted(null[k],[float(r[k+'_'+who]) for r in rows],side='right')) /
                    (len(null[k])+1) for k in ('color','layout')]
        product=components[0]*components[1]
        assert np.array_equal(product,np.array([float(r['joint_'+who]) for r in rows]))
        out[who]=(product,np.maximum(*components))
    return out


def scores(data,v):
    return {m:{k:(p if m=='product' else np.where(w<=v,p,np.inf)) for k,(p,w) in data.items()} for m in METHODS}


def at95(values):
    """Return first threshold reaching 95% and the tie fraction for exact ROC interpolation."""
    n=len(values)
    rank=(95*n+99)//100-1
    q=float(np.partition(values,rank)[rank])
    if not math.isfinite(q):
        return None
    below=np.count_nonzero(values<q)
    tied=np.count_nonzero(values==q)
    return q,(95*n-100*below)/(100*tied)


def calibrate():
    cfg=read(PARENT/'calibration.json')
    path=PARENT/'cal_scores.csv.gz'
    cal=[r for r in read_csv(path) if r['family']!='clean']
    assert len(cal)==1280
    data=inputs(cal,cfg['null'])
    v=float(np.quantile(data['own'][1],.98,method='inverted_cdf'))
    result={'veto':v,'methods':{},'input_hashes':{str(p):sha(p) for p in
        (path,PARENT/'calibration.json',RUN/'color_veto_20261119.json')}}
    for m,d in scores(data,v).items():
        point=at95(d['own'])
        assert point is not None
        q=point[0]
        result['methods'][m]=dict(cut=q,accepted=int((d['own']<=q).sum()),wrong=int((d['sibling']<=q).sum()))
    assert result['methods']['product']['cut']==cfg['thresholds']['joint']
    result['calibration_genuine_vetoed']=int((data['own'][1]>v).sum())
    save(RUN/'cutoffs.json',result)


def decisions(array,point):
    q,frac=point
    return (array<q).astype(float)+frac*(array==q)


def rates(d,point,blur,contrast):
    a=decisions(d['own'],point)
    b=decisions(d['sibling'],point)
    return np.array([a.mean(),b.mean(),a[blur].mean(),a[contrast].mean()])


def summarize(rows,cuts,null):
    edited=[r for r in rows if r['family']!='clean']
    assert len(edited)==1920
    edited.sort(key=lambda r:(int(r['task']),int(r['idx']),r['family'],float(r['level'])))
    data=inputs(edited,null)
    ds=scores(data,cuts['veto'])
    assert all(int(r['task'])==i//32 for i,r in enumerate(edited))
    assert [r['dataset'] for r in edited]==['COCO']*960+['SDP']*960
    strong=lambda f: np.array([r['family']==f and float(r['level'])==.8 for r in edited])
    blur,contrast=strong('blurring'),strong('contrast')
    assert blur.sum()==contrast.sum()==120
    keys=['pass','far','strong_blur_pass','strong_contrast_pass']
    methods,dec={},{}
    for m,d in ds.items():
        frozen=(cuts['methods'][m]['cut'],1.)
        q=at95(d['own'])
        methods[m]=dict(frozen=dict(zip(keys,rates(d,frozen,blur,contrast))),
            max_tpr=float(np.isfinite(d['own']).mean()),matched=(dict(
            cutoff=q[0],tie_fraction=q[1],inclusive=dict(zip(keys,rates(d,(q[0],1.),blur,contrast))),
            exact=dict(zip(keys,rates(d,q,blur,contrast)))) if q else None))
        dec[m]={k:arr<=frozen[0] for k,arr in d.items()}
    transitions={}
    for name,k,mask in [('pooled_genuine','own',np.ones(len(edited),dtype=bool)),
                        ('pooled_wrong','sibling',np.ones(len(edited),dtype=bool)),
                        ('strong_blur','own',blur),('strong_contrast','own',contrast)]:
        a,b=dec['product'][k][mask],dec['veto'][k][mask]
        transitions[name]=dict(n=len(a),both=int((a&b).sum()),lost=int((a&~b).sum()),gained=int((~a&b).sum()))
    rng=np.random.default_rng(20261119)
    boots={'frozen':[],'exact':[]}
    for _ in range(5000):
        tasks=np.r_[rng.integers(0,30,30),rng.integers(30,60,30)]
        ix=(32*tasks[:,None]+np.arange(32)).ravel()
        pair={k:[] for k in boots}
        for m in METHODS:
            sample={k:a[ix] for k,a in ds[m].items()}
            for k,point in [('frozen',(cuts['methods'][m]['cut'],1.)),('exact',at95(sample['own']))]:
                if point:
                    pair[k].append(rates(sample,point,blur[ix],contrast[ix]))
        for k,values in pair.items():
            if len(values)==2:
                boots[k].append(values[1]-values[0])
    ci={k:dict(n=len(values),delta_ci95=(dict(zip(keys,np.quantile(values,[.025,.975],axis=0).T.tolist()))
                                         if values else None)) for k,values in boots.items()}
    return dict(n_queries=len(edited),methods=methods,frozen_transitions=transitions,bootstrap=ci)


def evaluate():
    import torch
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_available()
    torch.set_num_threads(1)
    cuts=read(RUN/'cutoffs.json')
    for path,h in cuts['input_hashes'].items():
        assert sha(Path(path))==h
    assert all((RUN/f'generated_{s}.json').exists() for s in range(3))
    sys.path.insert(0,str(LOADER))
    from order_recovery_data import waves_module
    cfg=read(PARENT/'calibration.json')
    rows=collect(RUN,'test',read(PILOT/'color_budget_20261119.json'),waves_module(),color_tables(),
                 cfg['null'],output=RUN,layout_control=False)
    assert len(rows)==2040 and (RUN/'test_hybrid96_48.bin').stat().st_size==120*288
    write_rows(RUN/'test_scores.csv.gz',rows)
    save(RUN/'summary.json',summarize(rows,cuts,cfg['null']))
    files=[*RUN.glob('*.json'),*RUN.glob('*.py'),RUN/'test_hybrid96_48.bin',RUN/'test_scores.csv.gz']
    save(RUN/'complete.json',dict(sha256={p.name:sha(p) for p in files},references=120,
        bytes_per_reference=288,queries=len(rows),parent_receipt_sha256=sha(PARENT/'complete.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['calibrate','evaluate'])
    if parser.parse_args().stage=='calibrate':
        calibrate()
    else:
        evaluate()
