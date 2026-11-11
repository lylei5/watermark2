"""Validate completed frozen confirmation and report source-cluster uncertainty."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run
    assert not (run/'audited_summary.json').exists()
    plan=json.loads((run/'protocol.json').read_text());done=json.loads((run/'complete.json').read_text())
    result=json.loads((run/'scores/results.json').read_text())
    assert done['completed'] and result['completed'] and result['independent_confirmation_scored']
    assert done['protocol_sha256']==result['protocol_sha256']==sha(run/'protocol.json')
    assert done['results_sha256']==sha(run/'scores/results.json')
    for name,digest in plan['source_sha256'].items():assert sha(run/'scripts'/name)==digest,name
    cached=[]
    for shard in range(3):
        folder=run/'cache'/f'shard_{shard}';part=json.loads((folder/'complete.json').read_text())
        assert part['completed'] and part['steps']==50 and part['protocol_sha256']==sha(run/'protocol.json')
        for row in part['rows']:
            for suffix,key in [('.npy','inversion_sha256'),('.npz','features_sha256')]:assert sha(folder/(row['id']+suffix))==row[key]
            if row['split']=='confirmation':cached.append(row['id'])
    assert len(cached)==len(set(cached))==len(plan['rows'])
    assert set(cached)=={r['id'] for r in plan['rows']}
    bridge=json.loads((run/'cache/shard_0/bridge.json').read_text())
    assert all(r['code_equal'] and r['relative_l2']<=plan['bridge_max_relative_l2'] for r in bridge['rows'])
    rows=result['rows'];tasks=plan['identity_tasks'];queries={r['id'] for r in plan['rows']}
    assert len(rows)==len(queries)*len(tasks)
    assert {(r['id'],r['claimed_identity']) for r in rows}=={(q,t) for q in queries for t in tasks}
    assert all(r['default_accepted']==(r['ba']>181/256) for r in rows)
    for key,pvalue in [('all_fit_accepted','all_fit_p'),('split_fit_accepted','split_fit_p')]:
        assert all(r[key]==(r[pvalue]<=plan['nominal_alpha']) for r in rows)
    genuine=[r for r in rows if r['kind']=='genuine'];wrong=[r for r in rows if r['kind']=='wrong_identity']
    assert len(genuine)==len(queries) and len(wrong)==len(queries)*(len(tasks)-1)
    restored=[r for r in genuine if r['all_fit_accepted'] and not r['default_accepted']]
    lost=[r for r in genuine if r['default_accepted'] and not r['all_fit_accepted']]
    rates={key:np.array([np.mean([r[key] for r in genuine if r['task']==t]) for t in tasks])
           for key in ['default_accepted','all_fit_accepted','split_fit_accepted']}
    sample=np.random.default_rng(20261111).integers(0,len(tasks),size=(10000,len(tasks)))
    intervals={key:dict(rate=float(value.mean()),percentile95=np.quantile(value[sample].mean(1),[.025,.975]).tolist())
               for key,value in rates.items()}
    delta=rates['all_fit_accepted']-rates['default_accepted']
    intervals['paired_all_fit_minus_default']=dict(rate=float(delta.mean()),percentile95=np.quantile(delta[sample].mean(1),[.025,.975]).tolist())
    failures=[dict(id=r['id'],task=r['task'],label=r['label'],ba=r['ba'],
        default_accepted=r['default_accepted'],all_fit_accepted=r['all_fit_accepted'],split_fit_accepted=r['split_fit_accepted'],
        all_fit_p=r['all_fit_p'],split_fit_p=r['split_fit_p']) for r in genuine if not r['all_fit_accepted'] or not r['split_fit_accepted']]
    summary=dict(completed=True,confirmation_sources=len(tasks),queries=len(queries),identity_decisions=len(rows),
        summary=result['summary'],genuine_by_condition=result['genuine_by_condition'],
        wrong_identity_query_level=result['wrong_identity_query_level'],supplementary_ba_controls=result['supplementary_ba_controls'],
        restored_genuine=len(restored),lost_genuine=len(lost),restored_sources=len({r['task'] for r in restored}),
        genuine_failures=failures,source_cluster_bootstrap=intervals,
        uncertainty=dict(unit='source; all views retained together',replicates=10000,seed=20261111,
            method='paired source-cluster percentile bootstrap; fixed thresholds',
            limitation='Conditional on frozen calibration and sampled source population; no ultra-low FPR interval'),
        frozen_protocol_sha256=sha(run/'protocol.json'),results_sha256=sha(run/'scores/results.json'),
        all_input_hashes_verified=True,all_identity_pairs_verified=True,thresholds_changed=False,
        confirmation_set_consumed=True,new_attacks_run=plan.get('new_attacks_run',False),novel_or_modified_attacks=False)
    (run/'audited_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(restored=len(restored),lost=len(lost),failures=failures,bootstrap=intervals),indent=2))


if __name__=='__main__':main()
