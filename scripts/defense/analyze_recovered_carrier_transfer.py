"""Describe matched absolute detection and bottlenecks in completed carrier transfer.

Thresholds fitted here on evaluation genuine scores are explicitly descriptive.
They are not substituted for frozen calibration points or deployed as a defense.
"""
import argparse
import json
from pathlib import Path

from recovered_carrier_transfer import ROOT,sha,write,wait


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--audit',default='audited_scores_v3.json');p.add_argument('--parents',type=Path)
    a=p.parse_args();run=a.run.resolve();assert ROOT.is_dir()and run.parent==ROOT/'runs'
    if a.parents:
        for parent in json.loads(a.parents.read_text()):wait(parent)
    assert not(run/'comparison_analysis.json').exists()
    audit=json.loads((run/a.audit).read_text());assert audit['status']=='complete'
    plan=json.loads((run/'protocol.json').read_text());done=json.loads((run/'complete.json').read_text())
    for name,digest in done['artifacts'].items():assert sha(run/name)==digest
    rows=json.loads((run/'rows.json').read_text());controls=json.loads((run/'controls.json').read_text())
    genuine=lambda r:r['kind']in ['clean',*plan['conditions']]
    comparators=['sscd_single','sscd_templates','sscd_int8_templates',
                 'tangent4_margin_fixed_cosine','tangent8_margin_fixed_cosine']
    matched=[];bottlenecks=[];native_by_condition=[]
    for carrier in plan['carriers']:
        cc=[r for r in controls if r['carrier']==carrier]
        for kind in ['all_genuine','clean',*plan['conditions'],'cover','imprint_50','imprint_100','imprint_150']:
            group=[r for r in cc if (genuine(r) if kind=='all_genuine' else r['kind']==kind)]
            native_by_condition.append(dict(carrier=carrier,kind=kind,total=len(group),accepted=sum(r['native_accepted']for r in group)))
        for repetition in range(3):
            for method in plan['methods']:
                rr=[r for r in rows if r['carrier']==carrier and r['method']==method and r['repetition']==repetition]
                gg=[r for r in rr if genuine(r)]
                cells={f'key_{key}_native_{native}':sum(r['key_recovered']==key and r['native_accepted']==native for r in gg)
                       for key in [False,True] for native in [False,True]}
                assert sum(cells.values())==100
                bottlenecks.append(dict(carrier=carrier,method=method,repetition=repetition,total=100,**cells))
                points=[dict(point='maximum_radius127',cutoff=127)]
                points.extend(dict(point=f"frozen_GS_{x['target']}",cutoff=x['cutoff'])for x in plan['operating_points']
                              if x['method']==method and x['repetition']==repetition and x['feasible'])
                for point in points:
                    accept=lambda r:r['joint_accepted']and r['region_distance']<=point['cutoff']
                    count=sum(accept(r)for r in gg)
                    candidate=dict(genuine=count,cover=sum(accept(r)for r in rr if r['kind']=='cover'),
                        **{f'imprint_{k}':sum(accept(r)for r in rr if r['kind']==f'imprint_{k}')for k in [50,100,150]})
                    for baseline in comparators:
                        available=sorted(r[baseline]for r in cc if genuine(r)and r['native_accepted'])
                        assert count<=len(available)
                        threshold=available[count-1]if count else -1.
                        baseline_accept=lambda r:r['native_accepted']and r[baseline]<=threshold
                        observed=dict(genuine=sum(baseline_accept(r)for r in cc if genuine(r)),
                            cover=sum(baseline_accept(r)for r in cc if r['kind']=='cover'),
                            **{f'imprint_{k}':sum(baseline_accept(r)for r in cc if r['kind']==f'imprint_{k}')for k in [50,100,150]})
                        assert observed['genuine']>=count
                        matched.append(dict(carrier=carrier,method=method,repetition=repetition,**point,
                            baseline=baseline,baseline_threshold=threshold,candidate=candidate,baseline_result=observed))
    result=dict(status='complete',scope='Descriptive evaluation-matched absolute genuine comparison; no held-out calibration claim',
                source_sha256=sha(__file__),audit_sha256=sha(run/a.audit),protocol_sha256=sha(run/'protocol.json'),
                comparisons=matched,bottlenecks=bottlenecks,native_by_condition=native_by_condition,
                comparison_count=len(matched),attack_generation=False,method_changed=False)
    write(run/'comparison_analysis.json',result)
    print(json.dumps(dict(status='complete',comparison_count=len(matched),
          strong_control_observed_dominance=sum(x['baseline_result']['genuine']>=x['candidate']['genuine']and
            all(x['baseline_result'][k]<=x['candidate'][k]for k in ['cover','imprint_50','imprint_100','imprint_150'])for x in matched))),flush=True)


if __name__=='__main__':main()
