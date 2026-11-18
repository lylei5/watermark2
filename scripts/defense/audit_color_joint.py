"""Independently reconstruct confirmation decisions from serialized component distances."""
import bisect
import csv
import gzip
import math
from collections import Counter
from pathlib import Path

from color_identity_probe import ROOT, read, save, sha


def load_csv(path):
    with gzip.open(path,'rt') as f:
        return list(csv.DictReader(f))


def main():
    p=ROOT/'runs/color_joint_20261119'
    pilot=ROOT/'runs/color_budget_20261119'
    for path,expected in read(p/'complete.json')['sha256'].items():
        assert sha(Path(path).read_bytes())==expected
    cohorts=read(p/'sources.json')
    assert Counter(x['dataset'] for x in cohorts)=={'COCO':30,'SDP':30}
    prior=read(pilot/'sources.json')
    assert not ({(s['dataset'],s['source_id']) for s in prior} & {(s['dataset'],s['source_id']) for s in cohorts})
    hashes=set()
    for i,gpu in enumerate([4,5,7]):
        rec=read(p/f'generated_{i}.json')
        assert rec['gpu']==gpu and len(rec['records'])==20
        for r in rec['records']:
            s=cohorts[r['task']]
            assert s['seeds']==r['seeds'] and sha(s['prompt'].encode())==r['prompt_sha256']
            for j,h in enumerate(r['sha256']):
                assert sha((p/'images'/f"{s['task']:03d}"/f'{j}.png').read_bytes())==h
                hashes.add(h)
    assert len(hashes)==120
    cfg=read(p/'calibration.json')
    cal=load_csv(p/'cal_scores.csv.gz')
    rows=load_csv(p/'test_scores.csv.gz')
    assert len(rows)==len({(r['task'],r['idx'],r['family'],r['level']) for r in rows})==2040
    cal_edits=[r for r in cal if r['family']!='clean']
    assert len(cal_edits)==1280
    for key in ('color','layout'):
        assert sorted(float(r[key+'_sibling']) for r in cal_edits)==cfg['null'][key]
    for r in cal+rows:
        for who in ('own','sibling'):
            prod=1.
            for k in ('color','layout'):
                prod *= (1+bisect.bisect_right(cfg['null'][k],float(r[f'{k}_{who}'])))/1281
            assert prod==float(r[f'joint_{who}'])
    results=read(p/'summary.json')
    for m,threshold in cfg['thresholds'].items():
        assert sorted(float(r[m+'_own']) for r in cal_edits)[math.ceil(len(cal_edits)*.95)-1]==threshold
        for group,cohort in [('pooled',[r for r in rows if r['family']!='clean']),
                             ('contrast',[r for r in rows if r['family']=='contrast' and r['level']=='0.8']),
                             ('clean',[r for r in rows if r['family']=='clean'])]:
            d=results[group][m]
            assert d['n']==len(cohort)
            for name,who in [('pass','own'),('sibling','sibling')]:
                assert sum(float(r[m+'_'+who])<=threshold for r in cohort)==d['counts'][name]
            q=sorted(float(r[m+'_own']) for r in cohort)[math.ceil(len(cohort)*.95)-1]
            assert sum(float(r[m+'_sibling'])<=q for r in cohort)/len(cohort)==d['descriptive_test95']['far']
    save(p/'audit.json',dict(images=len(hashes),calibration_queries=len(cal),confirmation_queries=len(rows),
         same_prompt_pairs=60,identity_decisions_recomputed=True,complete_file_hashes_verified=True,
         joint_score_matches_binary_search_product=True))


if __name__=='__main__':
    main()
