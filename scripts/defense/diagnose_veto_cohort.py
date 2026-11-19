"""Trace sibling acceptances and lost genuine contrast decisions to saved reference bytes."""
import csv
import gzip
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist

from color_identity_probe import ROOT, LOADER, read, save
from color_budget_probe import color_tables, descriptors
from diagnose_color_joint import contact, digest, label, tails

P = ROOT/'runs/color_veto_20261119'
OUT = ROOT/'runs/color_product_cases_20261119'


def table(name, rows):
    with (OUT/name).open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def blocks(code):
    return {k:np.sqrt(x/x.sum()) for k,x in [('color',code[:96]),('layout',code[96:])]}


def run():
    import torch
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_available()
    torch.set_num_threads(1)
    sys.path.insert(0,str(LOADER))
    from order_recovery_data import waves_module
    OUT.mkdir(exist_ok=True)
    for f,h in read(P/'complete.json')['sha256'].items():
        assert digest(P/f)==h
    cuts=read(P/'cutoffs.json')
    for f,h in cuts['input_hashes'].items():
        assert digest(Path(f))==h
    null=read(P.parent/'color_joint_20261119/calibration.json')['null']
    matched=read(P/'summary.json')['methods']
    cut=cuts['methods']['product']['cut']
    mc=matched['product']['matched']['cutoff']
    with gzip.open(P/'test_scores.csv.gz','rt') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==2040
    clean={}
    errors=[]
    lost=[]
    for r in rows:
        task,idx=int(r['task']),int(r['idx'])
        for who in ('own','sibling'):
            assert math.prod(tails(r,who,null))==float(r['joint_'+who])
        if r['family']=='clean':
            clean[task,idx]=r
            continue
        c,l=tails(r,'sibling',null)
        assert (c*l<=cut)==(c*l<=mc)
        if c*l<=cut:
            errors.append(dict(task=task,idx=idx,family=r['family'],level=r['level'],color_tail=c,
                layout_tail=l,attribution=label(c,l,math.sqrt(cut)),color_sufficient=c<=cut,layout_sufficient=l<=cut))
        if r['family']=='contrast' and float(r['level'])==.8:
            c,l=tails(r,'own',null)
            def dropped(old,new):
                return c*l<=old and (c*l>new or max(c,l)>cuts['veto'])
            frozen=dropped(cut,cuts['methods']['veto']['cut'])
            assert frozen==dropped(mc,matched['veto']['matched']['cutoff'])
            if frozen:
                lost.append((task,idx))
    assert len(errors)==49 and len(lost)==len(set(lost))==9
    ids=sorted({e['task'] for e in errors})
    sources=read(P/'sources.json')
    waves,tables=waves_module(),color_tables()
    images,codes={},[]
    for s in sources:
        t=s['task']
        path=P/'images'/f'{t:03d}'
        receipt=read(path/'receipt.json')
        assert receipt['seeds']==s['seeds']
        for i,h in enumerate(receipt['sha256']):
            file=path/f'{i}.png'
            assert digest(file)==h
            with Image.open(file) as img:
                im=img.convert('RGB')
            images[t,i]=im
            codes.append(descriptors(im,tables)['hybrid96_48'])
    assert list(images)==[(t,i) for t in range(60) for i in range(2)]
    assert np.stack(codes).tobytes()==(P/'test_hybrid96_48.bin').read_bytes()
    gallery={k:np.stack([blocks(x)[k] for x in codes]) for k in ('color','layout')}
    prompts=[]
    for s in sources:
        t=s['task']
        rr=[e for e in errors if e['task']==t]
        cc,cl=tails(clean[t,0],'sibling',null)
        assert [cc,cl]==tails(clean[t,1],'sibling',null)
        neutral=[gallery['color'][2*t+i,0]**2 for i in range(2)]
        grids=gallery['layout'][2*t:2*t+2].reshape(2,4,4,3)
        mirrors=np.linalg.norm((grids-grids[:,:,::-1,:]).reshape(2,-1),axis=1)/math.sqrt(2)
        counts=Counter(e['attribution'] for e in rr)
        prompts.append(dict(task=t,dataset=s['dataset'],prompt=s['prompt'],n_errors=len(rr),
            idx0=sum(e['idx']==0 for e in rr),idx1=sum(e['idx']==1 for e in rr),
            color_only=counts['color'],layout_only=counts['layout'],both=counts['both'],
            color_floor=sum(e['color_sufficient'] for e in rr),layout_floor=sum(e['layout_sufficient'] for e in rr),
            clean_color_tail=cc,clean_layout_tail=cl,clean_accepted=cc*cl<=cut,
            neutral0=neutral[0],neutral1=neutral[1],near_gray=min(neutral)>=.9,
            palette_close=cc<=.05,layout_close=cl<=.05,mirror0=mirrors[0],mirror1=mirrors[1],
            worse_mirror=max(mirrors)))
    mirror_edge=float(np.quantile([s['worse_mirror'] for s in prompts],.25,method='inverted_cdf'))
    for s in prompts:
        s['mirror_low_quartile']=s['worse_mirror']<=mirror_edge
    groups={}
    for key in ('near_gray','palette_close','layout_close','mirror_low_quartile'):
        sub=[s for s in prompts if s[key]]
        groups[key]=dict(prompts=len(sub),affected=sum(s['n_errors']>0 for s in sub),
                         errors=sum(s['n_errors'] for s in sub),opportunities=32*len(sub))
    contrast=[]
    thumbs=[]
    checks=0
    for r in rows:
        t,i=int(r['task']),int(r['idx'])
        is_contrast=r['family']=='contrast' and float(r['level'])==.8
        if t not in ids and not is_contrast:
            continue
        im=images[t,i]
        edit=im if r['family']=='clean' else waves.apply_distortion([im],r['family'],
            strength=float(r['level']),distortion_seed=int(r['seed']))[0]
        import hashlib
        assert hashlib.sha256(edit.tobytes()).hexdigest()==r['sha256']
        enc=blocks(descriptors(edit,tables)['hybrid96_48'])
        details={}
        for k,g in gallery.items():
            d=cdist(enc[k][None],g)[0]/math.sqrt(2)
            own=d[2*t+i]
            other=np.delete(d,2*t+i).min()
            assert abs(own-float(r[k+'_own']))<1e-12 and abs(d[2*t+1-i]-float(r[k+'_sibling']))<1e-12
            rank=1+int((d<own).sum())
            ties=int((d==own).sum())
            assert float(rank==1)/ties==float(r[k+'_gallery'])
            details.update({k+'_rank':rank,k+'_ties':ties,k+'_own':own,k+'_nearest_wrong':other,k+'_margin':other-own})
        checks+=1
        if is_contrast:
            c,l=tails(r,'own',null)
            contrast.append(dict(task=t,idx=i,lost=(t,i) in lost,color_tail=c,layout_tail=l,**details))
            if (t,i) in lost:
                thumbs.append((f'{t:02d}/{i}  original | contrast',im,edit))
    contact([(f'{t:02d}  seed 0 | seed 1',images[t,0],images[t,1]) for t in ids],OUT/'pairs.jpg',2,150)
    contact(thumbs,OUT/'contrast.jpg',3,96)
    table('prompts.csv',prompts)
    table('false_accepts.csv',errors)
    table('contrast.csv',contrast)
    sub=[r for r in contrast if r['lost']]
    assert len(contrast)==120 and len(sub)==9 and all(r['color_tail']>cuts['veto']>=r['layout_tail'] for r in sub)
    meta=dict(product_cutoff=cut,matched_product_cutoff=mc,comparison_veto=cuts['veto'],
        groups=groups,mirror_edge=mirror_edge,case_ids=ids,verified_pixels=checks,
        sensitivity={str(e):dict(Counter(label(r['color_tail'],r['layout_tail'],e) for r in errors))
                     for e in (.01,math.sqrt(cut),math.sqrt(mc),.05)},
        contrast_unique_layout_top1=sum(r['layout_rank']==r['layout_ties']==1 for r in sub),
        contrast_unique_layout_top1_all=sum(r['layout_rank']==r['layout_ties']==1 for r in contrast),
        parent_receipt_sha256=digest(P/'complete.json'),script_sha256=digest(Path(__file__)),
        evidence={f:digest(OUT/f) for f in ('prompts.csv','false_accepts.csv','contrast.csv','pairs.jpg','contrast.jpg')})
    save(OUT/'summary.json',meta)


if __name__=='__main__':
    run()
