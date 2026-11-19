"""Replay the confirmation errors and localize them to the two frozen descriptors."""
import csv
import gzip
import hashlib
import json
import math
import os
import sys
from bisect import bisect_right
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.distance import cdist

from color_identity_probe import ROOT, LOADER

PARENT = ROOT/'runs/color_joint_20261119'
CHANNELS = ('color','layout')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(name):
    with gzip.open(PARENT/name,'rt') as stream:
        return list(csv.DictReader(stream))


def write_table(path,rows):
    with path.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tails(row,who,null):
    return [(1+bisect_right(null[k],float(row[k+'_'+who])))/(1+len(null[k])) for k in CHANNELS]


def label(c,l,edge):
    if c<=edge and l<=edge:
        return 'both'
    if c<=edge:
        return 'color'
    if l<=edge:
        return 'layout'
    return 'neither'


def contact(items,path,cols,side):
    rows=math.ceil(len(items)/cols)
    sheet=Image.new('RGB',(cols*side*2,rows*(side+18)),'white')
    pen=ImageDraw.Draw(sheet)
    for i,(tag,a,b) in enumerate(items):
        x,y=(i%cols)*2*side,(i//cols)*(side+18)
        sheet.paste(a.resize((side,side),Image.Resampling.LANCZOS),(x,y))
        sheet.paste(b.resize((side,side),Image.Resampling.LANCZOS),(x+side,y))
        pen.text((x+3,y+side+2),tag,fill='black')
    sheet.save(path,quality=85)


def main(out):
    import torch
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_available()
    torch.set_num_threads(1)
    sys.path[:0]=[str(PARENT),str(LOADER)]
    from order_recovery_data import waves_module, WAVES
    from color_budget_probe import color_tables, descriptors
    plan=json.loads((PARENT/'complete.json').read_bytes())
    for filename,h in plan['sha256'].items():
        assert digest(Path(filename))==h
    waves,tables=waves_module(),color_tables()
    cfg=json.loads((PARENT/'calibration.json').read_bytes())
    null,cut=cfg['null'],cfg['thresholds']['joint']
    balance=math.sqrt(cut)
    sources=json.loads((PARENT/'sources.json').read_bytes())
    rows=load_rows('test_scores.csv.gz')
    cal=[r for r in load_rows('cal_scores.csv.gz') if r['family']!='clean']
    for k in CHANNELS:
        assert null[k]==sorted(float(r[k+'_sibling']) for r in cal)
    for r in rows:
        for who in ('own','sibling'):
            assert math.prod(tails(r,who,null))==float(r['joint_'+who])
    false=[r for r in rows if r['family']!='clean' and float(r['joint_sibling'])<=cut]
    cases={int(r['task']) for r in false}
    assert len(false)==114 and len(cases)==8
    grouped=[]
    for r in false:
        c,l=tails(r,'sibling',null)
        grouped.append(dict(task=int(r['task']),dataset=r['dataset'],idx=int(r['idx']),family=r['family'],
            strength=r['level'],color_tail=c,layout_tail=l,joint=c*l,balanced=label(c,l,balance),
            color_sufficient=c<=cut,layout_sufficient=l<=cut,
            color_distance=r['color_sibling'],layout_distance=r['layout_sibling']))
    write_table(out/'false_accepts.csv',grouped)
    originals,reference={},[]
    for s in sources:
        dest=PARENT/'images'/f"{s['task']:03d}"
        receipt=json.loads((dest/'receipt.json').read_bytes())
        assert s['seeds']==receipt['seeds'] and hashlib.sha256(s['prompt'].encode()).hexdigest()==receipt['prompt_sha256']
        for i,h in enumerate(receipt['sha256']):
            path=dest/f'{i}.png'
            assert digest(path)==h
            with Image.open(path) as file:
                im=file.convert('RGB')
            originals[s['task'],i]=im
            reference.append(descriptors(im,tables)['hybrid96_48'])
    assert np.stack(reference).tobytes()==(PARENT/'test_hybrid96_48.bin').read_bytes()
    def unpack(h):
        return {k:np.sqrt(v/v.sum()) for k,v in zip(CHANNELS,(h[:96],h[96:]))}
    gallery={k:np.stack([unpack(code)[k] for code in reference]) for k in CHANNELS}
    blurred={}
    blur_mass={}
    checks=0
    for r in rows:
        task,idx=int(r['task']),int(r['idx'])
        if task not in cases and r['family'] not in ('clean','blurring'):
            continue
        im=originals[task,idx]
        if r['family']!='clean':
            im=waves.apply_distortion([im],r['family'],strength=float(r['level']),distortion_seed=int(r['seed']))[0]
        assert hashlib.sha256(im.tobytes()).hexdigest()==r['sha256']
        code=unpack(descriptors(im,tables)['hybrid96_48'])
        d={k:cdist(code[k][None],gallery[k])[0]/np.sqrt(2) for k in CHANNELS}
        for k,dist in d.items():
            for who,index in [('own',2*task+idx),('sibling',2*task+1-idx)]:
                assert abs(dist[index]-float(r[k+'_'+who]))<1e-12
            assert float(dist[2*task+idx]==dist.min())/int((dist==dist.min()).sum())==float(r[k+'_gallery'])
        if r['family']=='blurring' and r['level']=='0.8':
            blur_mass[task,idx]={k:float(.5*np.abs(code[k]**2-gallery[k][2*task+idx]**2).sum())
                                 for k in CHANNELS}
        if r['family']=='blurring' and r['level']=='0.8' and float(r['joint_own'])>cut:
            blurred[task,idx]=im
        checks+=1
    assert len(blurred)==45
    clean={(int(r['task']),int(r['idx'])):r for r in rows if r['family']=='clean'}
    group_results=[]
    for task in sorted(cases):
        sub=[r for r in grouped if r['task']==task]
        clean_c,clean_l=tails(clean[task,0],'sibling',null)
        assert [clean_c,clean_l]==tails(clean[task,1],'sibling',null)
        group_results.append(dict(task=task,dataset=sources[task]['dataset'],prompt=sources[task]['prompt'],
            edited_false=len(sub),directions=[sum(r['idx']==i for r in sub) for i in (0,1)],
            clean_tail=[clean_c,clean_l],clean_pass=clean_c*clean_l<=cut,
            clean_neutral_color_bin_mass=[float(gallery['color'][2*task+i,0]**2) for i in (0,1)],
            error_tail_medians=[float(np.median([r[k+'_tail'] for r in sub])) for k in CHANNELS],
            balanced=dict(Counter(r['balanced'] for r in sub)),
            sufficient={k:sum(r[k+'_sufficient'] for r in sub) for k in CHANNELS},
            sensitivity={str(edge):dict(Counter(label(r['color_tail'],r['layout_tail'],edge) for r in sub)) for edge in (.01,.05)}))
    contact([(f'{t:02d} | seed 0, seed 1',originals[t,0],originals[t,1]) for t in sorted(cases)],
            out/'pairs.jpg',2,142)
    contact([(f'{t:02d}/{i} original | blur',originals[t,i],im) for (t,i),im in sorted(blurred.items())],
            out/'blur_cases.jpg',5,85)
    weak={(int(r['task']),int(r['idx'])):r for r in rows if r['family']=='blurring' and r['level']=='0.4'}
    blur=[]
    for r in rows:
        if r['family']!='blurring' or r['level']!='0.8':
            continue
        task,idx=int(r['task']),int(r['idx'])
        w=weak[task,idx]
        c,l=tails(r,'own',null)
        wc,wl=tails(w,'own',null)
        a,b=wc*l<=cut,c*wl<=cut
        restore={(True,True):'either', (True,False):'color_only', (False,True):'layout_only',
                 (False,False):'both_needed' if wc*wl<=cut else 'weak_also_fails'}[a,b]
        blur.append(dict(task=task,dataset=r['dataset'],idx=idx,pass_strong=c*l<=cut,pass_weak=wc*wl<=cut,
            color=float(r['color_own']),layout=float(r['layout_own']),color_weak=float(w['color_own']),
            layout_weak=float(w['layout_own']),pc=c,pl=l,pc_weak=wc,pl_weak=wl,
            restore_color_pass=a,restore_layout_pass=b,restore=restore,
            color_mass_shift=blur_mass[task,idx]['color'],layout_mass_shift=blur_mass[task,idx]['layout'],
            color_top1=float(r['color_gallery']),layout_top1=float(r['layout_gallery']),joint_top1=float(r['joint_gallery']),
            joint_weak=wc*wl,joint_strong=c*l))
    write_table(out/'blur_images.csv',blur)
    strata={}
    for kept,key in ((False,'rejected'),(True,'passed')):
        part=[r for r in blur if r['pass_strong']==kept]
        strata[key]=dict(n=len(part),weak_pass=sum(r['pass_weak'] for r in part),
            color_weaker=sum(r['pc']>r['pl'] for r in part),layout_weaker=sum(r['pc']<r['pl'] for r in part),
            medians={k:float(np.median([r[k] for r in part])) for k in
                ('color','layout','color_weak','layout_weak','pc','pl','pc_weak','pl_weak','color_mass_shift','layout_mass_shift')},
            top1={k:sum(r[k+'_top1'] for r in part) for k in ('color','layout','joint')},
            restored=dict(Counter(r['restore'] for r in part)),
            both_above_balance=sum(min(r['pc'],r['pl'])>balance for r in part))
    assert strata['rejected']['n']==45 and strata['passed']['n']==75
    outputs=dict(groups=group_results,blur=strata,replayed_queries=checks,threshold=cut,balance=balance,
        sensitivity={str(edge):dict(Counter(label(r['color_tail'],r['layout_tail'],edge) for r in grouped))
                     for edge in (.01,balance,.05)},
        parent_receipt_sha256=digest(PARENT/'complete.json'),script_sha256=digest(Path(__file__)),
        waves_sha256=digest(WAVES/'distortions/distortions.py'),
        artifact_sha256={p.name:digest(p) for p in [out/'false_accepts.csv',out/'blur_images.csv',out/'pairs.jpg',out/'blur_cases.jpg']})
    (out/'summary.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    main(Path(sys.argv[1]))
