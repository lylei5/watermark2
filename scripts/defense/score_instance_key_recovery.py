"""Audit actual recovery and compare absolute genuine operating points on249."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT=Path('/data0/liyanlei/watermark_research')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def main():
    assert ROOT.is_dir()
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=ROOT/'runs/instance_coset_recovery_20261113')
    run=parser.parse_args().run.resolve()
    assert run.parent==ROOT/'runs'
    os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='4')
    import numpy as np
    plan=json.loads((run/'protocol.json').read_text())
    names=list(plan['conditions'])
    baseline_names=['sscd_single','sscd_templates','sscd_int8_templates']
    baseline_names += [m+'_cosine' for m in plan['methods'] if m.startswith('tangent')]
    raw_summaries,operating,matched,costs=[],[],[],[]
    methods=plan['methods']
    completions={}
    for repetition in range(3):
        folder=run/f'repetition_{repetition}'
        done=json.loads((folder/'complete.json').read_text())
        assert done['status']=='complete' and done['protocol_sha256']==sha(run/'protocol.json')
        for name,digest in done['artifacts'].items():assert sha(folder/name)==digest
        completions[str(repetition)]=sha(folder/'complete.json')
        rows=json.loads((folder/'rows.json').read_text())
        controls=json.loads((folder/'controls.json').read_text())
        enrollment=json.loads((folder/'enrollment.json').read_text())
        assert len(rows)==4920 and len(controls)==1230
        assert all(r['key_recovered']==(r['region_distance']<=127) for r in rows)
        assert all(not r['key_recovered'] or r['exact_key'] and r['exact_native_material'] for r in rows)
        if repetition==0:control_hash=sha(folder/'controls.json')
        else:assert sha(folder/'controls.json')==control_hash
        lookup={(r['id'],r['kind']):r for r in controls}

        def genuine(row):return row['kind']=='clean' or row['kind'] in names
        def split(part,cohort,subset):return [r for r in part if r['cohort']==cohort and r['split']==subset]
        def quantile(values,target):return sorted(values)[max(0,math.ceil(target*len(values))-1)]
        def summarize(part,accept):
            output=[]
            for cohort,subset in [('spatial_GS','calibration'),('spatial_GS','evaluation'),('PGID_GS','transfer')]:
                selected=split(part,cohort,subset)
                groups=['clean','all_edits','all_genuine']+names+(['same'] if cohort=='spatial_GS' else ['imprint_50','imprint_100','imprint_150'])
                for kind in groups:
                    group=[r for r in selected if (genuine(r) if kind=='all_genuine' else r['kind'] in names if kind=='all_edits' else r['kind']==kind)]
                    assert group
                    accepted=sum(bool(accept(r)) for r in group)
                    joint=(sum(bool(accept(r) and r['native']) for r in group) if all(r['native'] is not None for r in group) else None)
                    output.append(dict(cohort=cohort,split=subset,kind=kind,accepted=accepted,total=len(group),joint_native=joint))
            return output

        for method in methods:
            part=[r for r in rows if r['method']==method]
            cal=[r for r in split(part,'spatial_GS','calibration') if genuine(r)]
            ev=[r for r in split(part,'spatial_GS','evaluation') if genuine(r)]
            assert len(cal)==len(ev)==500
            raw_summaries.append(dict(repetition=repetition,method=method,summary=summarize(part,lambda r:r['key_recovered'])))
            for target in [.90,.95]:
                values=[r['region_distance'] if r['key_recovered'] else float('inf') for r in cal]
                cutoff=quantile(values,target)
                feasible=math.isfinite(cutoff) and cutoff<=127
                operating.append(dict(repetition=repetition,method=method,target=target,feasible=feasible,
                    cutoff=int(cutoff) if feasible else None,
                    summary=summarize(part,lambda r:r['key_recovered'] and r['region_distance']<=cutoff) if feasible else None))
            # Match controls to each candidate's maximum observed absolute evaluation TPR.
            # These are labeled descriptive curves, distinct from deployed calibration points.
            required=sum(r['key_recovered'] for r in ev)
            ev_controls=[lookup[r['id'],r['kind']] for r in ev]
            for baseline in baseline_names:
                cutoff=sorted(r[baseline] for r in ev_controls)[max(0,required-1)] if required else -1.
                matched.append(dict(repetition=repetition,candidate=method,baseline=baseline,
                    scope='Descriptive evaluation-matched operating point; threshold chosen on evaluation genuine scores',
                    candidate_genuine=required,total=500,threshold=cutoff,
                    summary=summarize(controls,lambda r:r[baseline]<=cutoff)))
            timings=np.array([r['decode_seconds'] for r in part])
            helper=[r for r in enrollment if r['method']==method]
            costs.append(dict(repetition=repetition,method=method,
                decode_seconds_mean=float(timings.mean()),decode_seconds_p95=float(np.quantile(timings,.95)),
                mean_attempts=float(np.mean([r['attempts'] for r in part])),
                enrollment_seconds_mean=float(np.mean([r['seconds'] for r in helper])),
                compact_helper_bytes=max(math.ceil(r['compact_helper_bits']/8) for r in helper),
                note='Compact field count excludes identifiers/configuration and native-material wrapper; actual JSON bytes below',
                json_record_bytes_spatial=[min(r['json_record_bytes'] for r in helper if r['id'].startswith('S')),max(r['json_record_bytes'] for r in helper if r['id'].startswith('S'))],
                json_record_bytes_pgid=[min(r['json_record_bytes'] for r in helper if r['id'].startswith('G')),max(r['json_record_bytes'] for r in helper if r['id'].startswith('G'))]))
        if repetition==0:
            for baseline in baseline_names:
                cal=[r for r in split(controls,'spatial_GS','calibration') if genuine(r)]
                for target in [.90,.95]:
                    cutoff=quantile([r[baseline] for r in cal],target)
                    operating.append(dict(repetition=None,method=baseline,target=target,feasible=True,cutoff=cutoff,
                                          summary=summarize(controls,lambda r:r[baseline]<=cutoff)))
    result=dict(status='complete',scope='Development cohort, calibrated points and separately labeled empirical matched curves',
        source_sha256=sha(__file__),protocol_sha256=sha(run/'protocol.json'),completion_sha256=completions,
        actual_key_recovery_decisions=14760,region_agreements=14760,key_bits=128,
        scalar_bch_checks=12,raw=raw_summaries,operating_points=operating,matched_controls=matched,costs=costs,
        baseline_template_bytes=dict(sscd_single=2048,sscd_templates=18432,sscd_int8_templates=4608),
        attack_generated=False,independent_confirmation=False)
    write(run/'scores.json',result)
    compact=[]
    for item in raw_summaries:
        compact.append(dict(repetition=item['repetition'],method=item['method'],
            summary=[r for r in item['summary'] if (r['split']=='evaluation' and r['kind'] in ['clean','all_edits','all_genuine','same']) or
                     (r['cohort']=='PGID_GS' and r['kind'] in ['all_genuine','imprint_150'])]))
    print(json.dumps(compact),flush=True)
    print('BASELINES',json.dumps([x for x in operating if x['repetition'] is None]),flush=True)


if __name__=='__main__':main()
