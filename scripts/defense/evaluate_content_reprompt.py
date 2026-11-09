"""Evaluate sealed CVPR Reprompt candidates at genuine-only frozen thresholds."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta, binomtest

from score_content_development import score_query, summarize, sha


def interval(k, n):
    return [float(beta.ppf(.025, k, n-k+1)) if k else 0.,
            float(beta.ppf(.975, k+1, n-k)) if k < n else 1.]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    genuinepath = args.report/'genuine_development.json'
    thresholdpath = args.report/'frozen_development_thresholds.json'
    genuine = json.loads(genuinepath.read_text())['rows']
    frozen = json.loads(thresholdpath.read_text())
    assert frozen['source_sha256'] == sha(genuinepath)
    completed = json.loads((args.root/'reprompt_candidates_20261110'/'complete.json').read_text())
    jobs = args.root/'reprompt_jobs_20261110.json'
    assert completed['completed'] and completed['jobs_sha256'] == sha(jobs)
    image_hashes = {r['id']: r['image_sha256'] for r in completed['rows']}
    records = json.loads(jobs.read_text())['rows']
    if any(r['split'] != 'development' for r in records):
        raise ValueError('Development evaluation only')
    rows = []
    for r in records:
        if sha(args.root/'queries'/(r['id']+'.png')) != image_hashes[r['id']]:
            raise ValueError('Candidate hash mismatch')
        rows.append(score_query(args.root, r))
    comparisons = []
    for threshold in frozen['rows']:
        if not threshold['feasible']:
            continue
        method, field, t = threshold['method'], threshold['score'], threshold['threshold']
        candidates = [r for r in rows if r['method'] == method]
        count = sum(r['accepted'] and r[field] >= t for r in candidates)
        comparisons.append(dict(**threshold, candidate_n=len(candidates),
                                candidate_accepted=count,
                                acceptance_interval_95=interval(count,len(candidates))))
    sem = [r for r in rows if r['method']=='sembind_scattered']
    matched = [r for r in comparisons if r['method']=='sembind_scattered'
               and r['target_absolute_tpr']==.8]
    ba = next(r for r in matched if r['score']=='ba')
    for r in matched:
        if r['calibration_count'] != ba['calibration_count']:
            r['matched_to_ba']=False
            continue
        r['matched_to_ba']=True
        base=np.array([x['accepted'] and x['ba']>=ba['threshold'] for x in sem])
        candidate=np.array([x['accepted'] and x[r['score']]>=r['threshold'] for x in sem])
        gains=int((base & ~candidate).sum()); losses=int((~base & candidate).sum())
        r['paired_ba_accept_candidate_reject']=gains
        r['paired_ba_reject_candidate_accept']=losses
        r['exact_mcnemar_p']=float(binomtest(gains,gains+losses).pvalue) if gains+losses else 1.
    result=dict(scope='development diagnostic, no independent confirmation',
                attack='published CVPR2025 single-candidate Reprompt, fixed benign COCO prompts',
                content_success_annotated=False, candidate_count=len(rows),
                thresholds_sha256=sha(thresholdpath), jobs_sha256=sha(jobs),
                evaluator_sha256=sha(__file__), summary=summarize(rows, rate_name='candidate_acceptance_rate'),
                score_implementation_sha256=sha(Path(__file__).with_name('content_authentication.py')),
                frozen_comparisons=comparisons, rows=rows)
    (args.output or args.report/'reprompt_development.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(summary=result['summary'],comparisons=comparisons),indent=2))


if __name__=='__main__':
    main()
