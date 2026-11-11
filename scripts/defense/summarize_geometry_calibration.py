"""Audit completed calibration and freeze supplementary BA controls on server."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from absolute_calibration import calibrate_absolute
from sembind_official_audit import sha, write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run=args.run
    assert not (args.output/'summary.json').exists()
    complete=json.loads((run/'complete.json').read_text())
    plan=json.loads((run/'protocol.json').read_text())
    result=json.loads((run/'calibration_scores/results.json').read_text())
    assert complete['completed'] and result['completed']
    assert result['protocol_sha256']==sha(run/'protocol.json')
    assert not result['independent_confirmation_scored'] and not result['thresholds_changed']
    assert result['threshold_audit']['applied_alpha']==plan['nominal_alpha']
    for name,digest in plan['source_sha256'].items():
        assert sha(run/'scripts'/name)==digest,name
    bridge=json.loads((run/'cache/shard_0/bridge.json').read_text())
    assert all(r['code_equal'] and r['relative_l2']<=plan['bridge_max_relative_l2'] for r in bridge['rows'])
    cache_rows=[]
    for i in range(3):
        folder=run/'cache'/f'shard_{i}'
        cache=json.loads((folder/'complete.json').read_text())
        assert cache['completed'] and cache['steps']==50 and cache['protocol_sha256']==sha(run/'protocol.json')
        for r in cache['rows']:
            assert sha(folder/(r['id']+'.npy'))==r['inversion_sha256']
            assert sha(folder/(r['id']+'.npz'))==r['features_sha256']
            if r['split']=='calibration':cache_rows.append(r)
    assert len(cache_rows)==192 and {r['id'] for r in cache_rows}=={r['id'] for r in plan['rows']}
    rows=result['rows'];claims=set(range(16,48))
    assert len(rows)==6144 and len({(r['id'],r['claimed_identity']) for r in rows})==6144
    for query in plan['rows']:
        assert {r['claimed_identity'] for r in rows if r['id']==query['id']}==claims
    genuine=[r for r in rows if r['kind']=='genuine']
    wrong=[r for r in rows if r['kind']=='wrong_identity']
    assert len(genuine)==192 and len(wrong)==5952
    rules=['default_accepted','identity_accepted','identity_penalty_accepted','all_fit_accepted','split_fit_accepted']
    for group,subset in [('genuine',genuine),('wrong_identity',wrong)]:
        actual=dict(group=group,n=len(subset),**{k:sum(r[k] for r in subset) for k in rules})
        assert actual==next(x for x in result['summary'] if x['group']==group)
    points=[]
    for target in [.8,.9,.95,1.]:
        point=calibrate_absolute([r['ba'] for r in genuine],np.ones(len(genuine),bool),target)
        threshold=point['threshold'];point['base_required']=False
        point['selection']='Genuine calibration BA only; no prefilter by default GS decision'
        point['wrong_identity_accepted']=sum(r['ba']>=threshold for r in wrong)
        point['wrong_identity_queries']=len({r['id'] for r in wrong if r['ba']>=threshold})
        point['genuine_by_condition']={label:dict(n=sum(r['label']==label for r in genuine),
            accepted=sum(r['label']==label and r['ba']>=threshold for r in genuine)) for label in sorted({r['label'] for r in genuine})}
        points.append(point)
    comparison=dict(queries_with_different_geometry_decisions=sum(r['all_fit_accepted']!=r['split_fit_accepted'] for r in rows),
                    restored_genuine=sum(r['all_fit_accepted'] and not r['default_accepted'] for r in genuine),
                    lost_genuine=sum(r['default_accepted'] and not r['all_fit_accepted'] for r in genuine),
                    restored_sources=len({r['task'] for r in genuine if r['all_fit_accepted'] and not r['default_accepted']}))
    summary=dict(completed=True,calibration_sources=32,calibration_queries=192,identity_decisions=6144,
        summary=result['summary'],genuine_by_condition=result['genuine_by_condition'],
        wrong_identity_query_level=result['wrong_identity_query_level'],comparison=comparison,
        supplementary_ba_controls=points,default_ba_operator='>',default_ba_threshold=181/256,
        detector_changed=False,confirmation_scores_read=False,new_attacks_run=False,
        limitations=['Correlated views and shared identities; counts are not independent Bernoulli trials',
                     'No empirical ultra-low false-positive guarantee',
                     'No added forgery rejection demonstrated beyond calibrated simple controls'],
        provenance=dict(results_sha256=sha(run/'calibration_scores/results.json'),protocol_sha256=sha(run/'protocol.json'),
                        complete_sha256=sha(run/'complete.json'),input_cache_files_verified=True),bridge=bridge)
    write(args.output/'summary.json',summary)
    print(json.dumps(dict(comparison=comparison,controls=points),indent=2))


if __name__=='__main__':main()
