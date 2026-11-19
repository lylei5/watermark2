"""Reference-conditioned chromatic evidence on a random and an enriched holdout."""
import os
import sys
from pathlib import Path

import numpy as np

from color_identity_probe import ROOT, LOADER, read, save
from color_joint_probe import collect, color_tables, write_rows
from color_veto_probe import at95, decisions, read_csv, sha

RUN=ROOT/'runs/color_richness_20261120'
CAL=ROOT/'runs/color_joint_20261119'
PILOT=ROOT/'runs/color_budget_20261119'
METHODS=('product','weighted')
KEYS=('overall_pass','overall_far','strong_blur','strong_contrast','gray_pass','gray_far',
      'gray_blur','gray_contrast','gray_reference_pass','gray_reference_far','enriched_pass','enriched_far')


def annotate(rows,sources,packed,null):
    refs=np.fromfile(packed,dtype='<u2').reshape(-1,144)
    assert len(refs)==2*len(sources) and np.all(refs.sum(1)==65535)
    neutral=refs[:,0]/refs[:,:96].sum(1)
    lookup={s['task']:j for j,s in enumerate(sources)}
    for r in rows:
        task=lookup[int(r['task'])]
        idx=int(r['idx'])
        r['stratum']=sources[task].get('stratum','calibration')
        for who,i in [('own',idx),('sibling',1-idx)]:
            g=float(neutral[2*task+i])
            w=min(1.,4*(1-g))
            c,l=[(1+np.searchsorted(null[k],float(r[k+'_'+who]),side='right'))/(1+len(null[k]))
                 for k in ('color','layout')]
            assert c*l==float(r['joint_'+who])
            r['neutral_'+who]=g
            r['weighted_'+who]=float(c*l if w==1 else c**w*l**(2-w))
    return rows


def calibrate():
    cfg=read(CAL/'calibration.json')
    files=[CAL/'calibration.json',CAL/'cal_scores.csv.gz',CAL/'calibration_hybrid96_48.bin',
           PILOT/'sources.json',RUN/'color_richness_20261120.json']
    rows=[r for r in read_csv(files[1]) if r['family']!='clean']
    src=[s for s in read(files[3]) if s['split']=='calibration']
    assert len(src)==40 and len(rows)==1280
    annotate(rows,src,files[2],cfg['null'])
    cut={m:at95(np.array([float(r[k+'_own']) for r in rows]))[0] for m,k in zip(METHODS,('joint','weighted'))}
    assert cut['product']==cfg['thresholds']['joint']
    save(RUN/'cutoffs.json',dict(cuts=cut,hashes={str(f):sha(f) for f in files}))


def summarize(rows,cuts):
    rows=sorted((r for r in rows if r['family']!='clean'),key=lambda r:(int(r['task']),int(r['idx']),r['family'],float(r['level'])))
    n=len(rows)
    assert n==3200 and all(int(r['task'])==i//32 for i,r in enumerate(rows))
    random=np.array([r['stratum']=='random' for r in rows])
    own_gray=np.array([r['neutral_own']>=.9 for r in rows])
    sibling_gray=np.array([r['neutral_sibling']>=.9 for r in rows])
    gray=own_gray&sibling_gray
    blur=np.array([r['family']=='blurring' and float(r['level'])==.8 for r in rows])
    contrast=np.array([r['family']=='contrast' and float(r['level'])==.8 for r in rows])
    assert random.sum()==1920 and gray.sum()>0
    masks=[random,random,random&blur,random&contrast,gray,gray,gray&blur,gray&contrast,
           own_gray,sibling_gray,~random,~random]
    is_far=[False,True,False,False,False,True,False,False,False,True,False,True]
    out=dict(counts=dict(zip(KEYS,[int(m.sum()) for m in masks])),gray_pairs_by_stratum={
        st:int(sum(r['stratum']==st and g for r,g in zip(rows,gray))//32) for st in ('random','enriched')},
        methods={},delta_ci95={})
    arrays={m:tuple(np.array([float(r[k+'_'+who]) for r in rows]) for who in ('own','sibling'))
            for m,k in zip(METHODS,('joint','weighted'))}
    def measure(a,b,point,sel):
        ap,bp=decisions(a,point),decisions(b,point)
        return np.array([(bp if far else ap)[s].mean() if s.any() else np.nan for s,far in zip(sel,is_far)])
    for m,(a,b) in arrays.items():
        out['methods'][m]={}
        for name,point in [('frozen',(cuts[m],1.)),('overall95',at95(a[random])),('gray95',at95(a[gray]))]:
            out['methods'][m][name]=dict(cut=float(point[0]),tie_fraction=float(point[1]),
                rates=dict(zip(KEYS,measure(a,b,point,masks))))
    strata={}
    for i in range(n//32):
        r=rows[32*i]
        strata.setdefault((r['stratum'],r['dataset']),[]).append(i)
    rng=np.random.default_rng(20261120)
    diffs={k:[] for k in ('frozen','overall95','gray95')}
    for _ in range(5000):
        tasks=np.concatenate([rng.choice(v,len(v)) for v in strata.values()])
        ix=(tasks[:,None]*32+np.arange(32)).ravel()
        sel=[s[ix] for s in masks]
        tmp={k:[] for k in diffs}
        for m,(a,b) in arrays.items():
            x,y=a[ix],b[ix]
            for k,p in [('frozen',(cuts[m],1.)),('overall95',at95(x[sel[0]])),
                        ('gray95',at95(x[sel[4]]) if sel[4].any() else None)]:
                if p:
                    tmp[k].append(measure(x,y,p,sel))
        for k,v in tmp.items():
            if len(v)==2:
                diffs[k].append(v[1]-v[0])
    for k,v in diffs.items():
        out['delta_ci95'][k]=dict(n=len(v),intervals=dict(zip(KEYS,np.nanquantile(v,[.025,.975],axis=0).T.tolist())))
    return out


def evaluate():
    import torch
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_available()
    torch.set_num_threads(1)
    sys.path.insert(0,str(LOADER))
    from order_recovery_data import waves_module
    cuts=read(RUN/'cutoffs.json')
    for f,h in cuts['hashes'].items():
        assert sha(Path(f))==h
    assert all((RUN/f'generated_{s}.json').is_file() for s in range(3))
    null=read(CAL/'calibration.json')['null']
    rows=collect(RUN,'test',read(PILOT/'color_budget_20261119.json'),waves_module(),color_tables(),
                 null,output=RUN,layout_control=False)
    f=RUN/'test_hybrid96_48.bin'
    assert f.stat().st_size==288*200 and len(rows)==3400
    annotate(rows,read(RUN/'sources.json'),f,null)
    write_rows(RUN/'test_scores.csv.gz',rows)
    save(RUN/'summary.json',summarize(rows,cuts['cuts']))
    files=[*RUN.glob('*.py'),*RUN.glob('*.json'),f,RUN/'test_scores.csv.gz']
    save(RUN/'complete.json',dict(references=200,bytes_each=288,score_rows=len(rows),
                                  sha256={p.name:sha(p) for p in files}))


if __name__=='__main__':
    {'calibrate':calibrate,'evaluate':evaluate}[sys.argv[1]]()
