"""Rebuild the neutral masses, component weights, decision thresholds and paired counts."""
import bisect
import csv
import gzip
import hashlib
import math
from pathlib import Path
import struct
from collections import Counter

from color_identity_probe import ROOT, read, save
from color_veto_probe import sha

RUN=ROOT/'runs/color_richness_20261120'
CAL=ROOT/'runs/color_joint_20261119'


def rates(rows,own,sib,cut,tie):
    prob=lambda x: 1 if x<cut else tie if x==cut else 0
    tp=sum(prob(float(r[own])) for r in rows)
    fp=sum(prob(float(r[sib])) for r in rows)
    return dict(n=len(rows),tp=tp,fp=fp,all_ties_tp=sum(float(r[own])<=cut for r in rows),
                all_ties_fp=sum(float(r[sib])<=cut for r in rows))


def main():
    for f,h in read(RUN/'complete.json')['sha256'].items():
        assert sha(RUN/f)==h
    sources=read(RUN/'sources.json')
    strata=Counter((s['dataset'],s['stratum']) for s in sources)
    assert strata=={(d,k):n for d in ('COCO','SDP') for k,n in [('random',30),('enriched',20)]}
    prior=read(ROOT/'runs/phase1_trajectory_native_gs_20261113/protocol.json')['sources']
    prior+=read(ROOT/'runs/unified_main_sources_20261116/selection.json')['sources']
    for s in ('color_budget_20261119','color_joint_20261119','color_veto_20261119'):
        prior+=read(ROOT/f'runs/{s}/sources.json')
    ids={(s['dataset'],str(s['source_id'])) for s in sources}
    assert len(ids)==100 and ids.isdisjoint((s['dataset'],str(s['source_id'])) for s in prior)
    assert len({v for s in sources for v in s['seeds']})==200
    files=set()
    for shard in range(3):
        gen=read(RUN/f'generated_{shard}.json')
        assert gen['gpu'] in (4,5,7)
        for r in gen['records']:
            t=r['task']
            assert t%3==shard and r['seeds']==sources[t]['seeds']
            assert hashlib.sha256(sources[t]['prompt'].encode()).hexdigest()==r['prompt_sha256']
            for idx,h in enumerate(r['sha256']):
                assert sha(RUN/'images'/f'{t:03d}'/f'{idx}.png')==h
                files.add(h)
    assert len(files)==200
    null=read(CAL/'calibration.json')['null']
    def neutral_values(p):
        codes=list(struct.iter_unpack('<144H',p.read_bytes()))
        assert all(sum(c)==65535 for c in codes)
        return [c[0]/sum(c[:96]) for c in codes]
    def check(r,who,g):
        c,l=[(1+bisect.bisect_right(null[k],float(r[k+'_'+who])))/(len(null[k])+1) for k in ('color','layout')]
        assert c*l==float(r['joint_'+who])
        w=min(4*(1-g),1.)
        return c*l if w==1 else c**w*l**(2-w)
    def read_rows(path):
        with gzip.open(path,'rt') as f:
            return [r for r in csv.DictReader(f) if r['family']!='clean']
    cuts=read(RUN/'cutoffs.json')['cuts']
    cc=neutral_values(CAL/'calibration_hybrid96_48.bin')
    tasks=[s['task'] for s in read(ROOT/'runs/color_budget_20261119/sources.json') if s['split']=='calibration']
    cal=read_rows(CAL/'cal_scores.csv.gz')
    assert len(cal)==1280
    corr=[check(r,'own',cc[2*tasks.index(int(r['task']))+int(r['idx'])]) for r in cal]
    assert math.isclose(sorted(corr)[1215],cuts['weighted'],rel_tol=1e-13)
    assert sorted(float(r['joint_own']) for r in cal)[1215]==cuts['product']
    rows=read_rows(RUN/'test_scores.csv.gz')
    assert len(rows)==len({(r['task'],r['idx'],r['family'],r['level']) for r in rows})==3200
    gs=neutral_values(RUN/'test_hybrid96_48.bin')
    assert len(gs)==200
    for r in rows:
        t,i=int(r['task']),int(r['idx'])
        assert r['stratum']==sources[t]['stratum']
        for who,j in [('own',i),('sibling',1-i)]:
            g=gs[2*t+j]
            assert g==float(r['neutral_'+who]) and math.isclose(check(r,who,g),float(r['weighted_'+who]),rel_tol=1e-13)
    rand=[r for r in rows if r['stratum']=='random']
    gray=[r for r in rows if min(float(r['neutral_own']),float(r['neutral_sibling']))>=.9]
    groups={'overall':rand,'gray':gray,'enriched':[r for r in rows if r['stratum']=='enriched']}
    for k,family in [('strong_blur','blurring'),('strong_contrast','contrast')]:
        groups[k]=[r for r in rand if r['family']==family and r['level']=='0.8']
        groups['gray_'+k]=[r for r in gray if r['family']==family and r['level']=='0.8']
    summary=read(RUN/'summary.json')
    counts={}
    for mode,cohort in [('frozen',None),('overall95',rand),('gray95',gray)]:
        counts[mode]={}
        for method,k in [('product','joint'),('weighted','weighted')]:
            own,sib=k+'_own',k+'_sibling'
            info=summary['methods'][method][mode]
            q,frac=info['cut'],info['tie_fraction']
            if cohort:
                vals=sorted(float(r[own]) for r in cohort)
                assert vals[(95*len(vals)+99)//100-1]==q
                assert math.isclose((bisect.bisect_left(vals,q)+frac*vals.count(q))/len(vals),.95)
            else:
                assert q==cuts[method] and frac==1
            results={g:rates(group,own,sib,q,frac) for g,group in groups.items()}
            for name,g,field in [('overall_far','overall','fp'),('overall_pass','overall','tp'),
                ('gray_far','gray','fp'),('gray_pass','gray','tp'),('strong_blur','strong_blur','tp'),
                ('strong_contrast','strong_contrast','tp'),('gray_blur','gray_strong_blur','tp'),
                ('gray_contrast','gray_strong_contrast','tp')]:
                assert math.isclose(results[g][field]/results[g]['n'],info['rates'][name],abs_tol=1e-12)
            counts[mode][method]=results
    save(RUN/'audit.json',dict(unique_pngs=200,stored_bytes=200*288,unique_edited_queries=len(rows),
         gray_references=sum(g>=.9 for g in gs),gray_pairs=len(gray)//32,
         paired_rates_reconstructed=True,counts=counts,complete_sha256=sha(RUN/'complete.json')))


if __name__=='__main__':
    main()
