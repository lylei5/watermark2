"""Source-paired descriptive uncertainty for frozen-GS calibration transfer."""
import argparse
import json
import os
from pathlib import Path

from recovered_carrier_transfer import ROOT,sha,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    run=a.run.resolve();assert ROOT.is_dir()and run.parent==ROOT/'runs'
    assert not(run/'source_uncertainty.json').exists()
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    import numpy as np
    plan=json.loads((run/'protocol.json').read_text());done=json.loads((run/'complete.json').read_text())
    for name,digest in done['artifacts'].items():assert sha(run/name)==digest
    rows=json.loads((run/'rows.json').read_text());controls=json.loads((run/'controls.json').read_text())
    seed=2026111401;draws=np.random.default_rng(seed).integers(0,10,size=(10000,10))
    output=[]
    for carrier in plan['carriers']:
        cc=[r for r in controls if r['carrier']==carrier and r['kind']in ['clean',*plan['conditions']]]
        ids=sorted({r['id']for r in cc});assert len(ids)==10 and len(cc)==100
        lookup={(r['id'],r['kind']):r for r in cc}
        for point in plan['operating_points']:
            if not point['feasible']or point['repetition']is None:continue
            rr=[r for r in rows if r['carrier']==carrier and r['method']==point['method']and
                r['repetition']==point['repetition']and r['kind']in ['clean',*plan['conditions']]]
            assert len(rr)==100
            for baseline in ['sscd_single','sscd_templates','sscd_int8_templates']:
                control=next(x for x in plan['operating_points']if x['method']==baseline and x['target']==point['target'])
                per_source=[]
                for identity in ids:
                    part=[r for r in rr if r['id']==identity];assert len(part)==10
                    delta=0
                    for r in part:
                        b=lookup[r['id'],r['kind']]
                        assert b['pixels_sha256']==r['pixels_sha256']
                        candidate=r['joint_accepted']and r['region_distance']<=point['cutoff']
                        comparison=b['native_accepted']and b[baseline]<=control['cutoff']
                        delta+=int(candidate)-int(comparison)
                    per_source.append(delta/10)
                difference=np.asarray(per_source);boot=difference[draws].mean(axis=1)*100
                output.append(dict(carrier=carrier,method=point['method'],repetition=point['repetition'],target=point['target'],
                    baseline=baseline,delta_percentage_points=float(difference.mean()*100),
                    source_bootstrap_percentile95=list(map(float,np.quantile(boot,[.025,.975]))),
                    per_source_delta=per_source))
    result=dict(status='complete',source_sha256=sha(__file__),protocol_sha256=sha(run/'protocol.json'),
        seed=seed,resamples=10000,source_count_per_carrier=10,genuine_queries_per_source=10,
        scope='Descriptive source-paired intervals conditional on frozen calibration; development cohort, no multiplicity correction or independent confirmation',
        calibration_uncertainty_included=False,rows=output)
    write(run/'source_uncertainty.json',result)
    print(json.dumps([r for r in output if r['baseline']=='sscd_int8_templates'and r['method']=='tangent8_margin_fixed']),flush=True)


if __name__=='__main__':main()
