"""Continuous separation during frozen key decoding; does not alter the experiment."""
import argparse
import math
import os

from shared_channel_experiment import ROOT,read,write,sha
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();assert ROOT.is_dir()and run.parent==ROOT/'runs'
    os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    import numpy as np
    plan=read(run/'protocol.json');done=read(run/'extract_complete.json');assert done['status']=='complete'
    for name,digest in done['artifacts'].items():assert sha(run/name)==digest
    assert not(run/'continuous_separation.json').exists()
    meta=read(run/'feature_rows.json')
    with np.load(run/'descriptors.npz')as archive:features={k:archive[k]for k in archive.files}
    index={(r['id'],r['kind']):i for i,r in enumerate(meta)}
    observations=[];summaries=[]
    for method in features:
        # Global shuffle is recorded as a diagnostic; use the separate
        # within-split control for source-separated mechanism comparisons.
        if method=='permuted_phase':continue
        rows=[]
        for i,r in enumerate(meta):
            if r['kind'].startswith('enroll_'):continue
            j=index[r['id'],'clean']
            value=float(np.clip(1-features[method][j]@features[method][i],0,2))
            split=('fit'if r['task']in plan['fit_sources']else'calibration'if r['task']in plan['calibration_sources']else'evaluation')if r['carrier']=='GS'else'transfer'
            rows.append(dict(id=r['id'],carrier=r['carrier'],task=r['task'],kind=r['kind'],split=split,distance=value))
        cal=sorted(r['distance']for r in rows if r['split']=='calibration'and r['kind']in ['clean',*plan['conditions']]);assert len(cal)==30
        for target in plan['calibration_targets']:
            threshold=cal[math.ceil(target*30)-1]
            for carrier,split in [('GS','evaluation'),('TR','transfer'),('PRC','transfer')]:
                for kind in ['all_genuine','unknown_edits','same_prompt','imprint_150']:
                    part=[r for r in rows if r['carrier']==carrier and r['split']==split and
                          (r['kind']in ['clean',*plan['conditions']]if kind=='all_genuine'else
                           r['kind']in plan['conditions']and r['kind']not in plan['enrollment_families']if kind=='unknown_edits'else r['kind']==kind)]
                    summaries.append(dict(method=method,target=target,threshold=threshold,carrier=carrier,split=split,
                        kind=kind,total=len(part),accepted=sum(r['distance']<=threshold for r in part),
                        median_distance=float(np.median([r['distance']for r in part]))))
        observations.extend(dict(method=method,**r)for r in rows)
    result=dict(status='complete',source_sha256=sha(__file__),protocol_sha256=sha(run/'protocol.json'),
        descriptors_sha256=sha(run/'descriptors.npz'),scope='Content-only continuous separation; all frozen key decoding continues unchanged',
        summaries=summaries,rows=observations)
    write(run/'continuous_separation.json',result)
    print(json.dumps([r for r in summaries if r['target']==.95 and r['kind']in ['all_genuine','same_prompt']]),flush=True)


if __name__=='__main__':main()
