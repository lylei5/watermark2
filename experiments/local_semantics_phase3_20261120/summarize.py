"""Paired comparisons, conditional confidence bounds and key sensitivity."""
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'rgb_moments_phase2_20261120'))
from evaluate import bounded_interval, EDITS


def save_csv(path,rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f,list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    expanded = json.loads((HERE/'pilot_decision.json').read_text())['expand']
    files = ['pilot']+(['test'] if expanded else [])
    cases = defaultdict(dict)
    for stage in files:
        with gzip.open(HERE/(stage+'_scores.csv.gz'),'rt') as f:
            for r in csv.DictReader(f):
                identity = (r['split'],r['domain'],int(r['cluster']),int(r['sample']),r['condition'])
                assert r['method'] not in cases[identity]
                score,passed = float(r['score']),int(r['passed'])
                assert passed==(score<=float(r['threshold']))
                cases[identity][r['method']]=(score,passed)
    assert all(set(scores)==set(next(iter(cases.values()))) for scores in cases.values())
    groups = defaultdict(list)
    for (split,domain,cluster,idx,cond),scores in cases.items():
        is_public=split.endswith('_public')
        cohort = split+'_'+domain if is_public else split
        cohorts=[cohort]
        if is_public and domain=='replacement' and split=='test_public':
            cohorts.append('confirm_original4' if idx<100 else 'confirm_additional12')
        if is_public and idx<100:
            cohorts.append('original8_'+domain)
        for group in cohorts:
            groups[group,cond].append((cluster,scores))
            if cond in EDITS:
                groups[group,'all_edits'].append((cluster,scores))
    rows=[]
    for (group,cond),items in sorted(groups.items()):
        for method in items[0][1]:
            clusters=defaultdict(list)
            for cluster,scores in items:
                clusters[cluster].append(scores[method][1])
            assert len({len(v) for v in clusters.values()})==1
            n=sum(map(sum,clusters.values()))
            rate=n/len(items)
            lo,hi=bounded_interval(rate,len(clusters))
            rows.append(dict(cohort=group,condition=cond,method=method,accepted=n,total=len(items),
                             rate=rate,lo95=lo,hi95=hi,clusters=len(clusters)))
    save_csv(HERE/'comparison.csv',rows)
    # Sensitivity only: same empirical SD2.1 legal rate across all methods.
    # Public replacement and transfer pairs do not enter these thresholds.
    if expanded:
        same_rate=[]
        legal=groups['test','all_edits']
        for method in ('A','B','C','D','E','public64','unordered','C_full','spatial_avg','pool'):
            scores=sorted(s[method][0] for _,s in legal)
            threshold=scores[math.ceil(.95*len(scores))-1]
            for cohort,cond in (('test','all_edits'),('test','cross_seed'),('test','donor'),
                                ('test_public_replacement','replacement_output'),('test_public_color','color_output')):
                items=groups[cohort,cond]
                accepted=sum(s[method][0]<=threshold for _,s in items)
                rate=accepted/len(items)
                lo,hi=bounded_interval(rate,len({i for i,_ in items}))
                same_rate.append(dict(cohort=cohort,condition=cond,method=method,threshold=threshold,
                                      accepted=accepted,total=len(items),rate=rate,lo95=lo,hi95=hi))
        save_csv(HERE/'matched95.csv',same_rate)
    subjects=[]
    for k,v in sorted(cases.items()):
        if k[-1]=='replacement_output':
            record=dict(cohort=k[0],item=k[3])
            for m in ['A','B','C','D','E','public64','unordered','spatial_avg','C_full']:
                record[m+'_score'],record[m+'_pass']=v[m]
            subjects.append(record)
    save_csv(HERE/'subject_scores.csv',subjects)
    metric={}
    for (group,cond),items in groups.items():
        if cond not in ('replacement_output','color_output','donor','cross_seed','all_edits'):
            continue
        if group not in ('test_public_replacement','test_public_color','test','pilot_public_replacement'):
            continue
        stats={}
        for a,b in [('A','C'),('B','C'),('unordered','C'),('C_full','C'),('C','E'),('public64','E')]:
            drop=sum(s[a][1] and not s[b][1] for _,s in items)
            gain=sum(not s[a][1] and s[b][1] for _,s in items)
            result=dict(a_accept_b_reject=drop,a_reject_b_accept=gain)
            if cond in ('replacement_output','color_output'):
                result['exact_two_sided_p']=binomtest(min(drop,gain),drop+gain,.5).pvalue if drop+gain else 1.
            stats[a+'__'+b]=result
        stats['C_E_spearman']=float(spearmanr([s['C'][0] for _,s in items],[s['E'][0] for _,s in items]).statistic)
        stats['key_accept_counts']=[sum(s[m][1] for _,s in items) for m in ('E','key2','key3','key4','key5')]
        metric[group+'/'+cond]=stats
    ortho=max(abs(v['C'][0]-v['orthogonal'][0]) for v in cases.values())
    parity=sum(v['C'][1]!=v['orthogonal'][1] for v in cases.values())
    (HERE/'paired.json').write_text(json.dumps(dict(comparisons=metric,orthogonal_max_abs=ortho,
                  orthogonal_decision_differences=parity,pairs=len(cases)),indent=2,allow_nan=False))


if __name__=='__main__':
    main()
