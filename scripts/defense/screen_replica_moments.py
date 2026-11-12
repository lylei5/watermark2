"""Server-only development screen; cached artifacts are not fresh model trials."""
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from replica_moments import statistics


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    shared=args.root/'runs/shared_semantic_evidence_20261112'
    sys.path.insert(0,str(shared))
    from anchor_binding import identity,order
    plan=json.loads((shared/'protocol.json').read_text())
    names=['ba','raw_agreement','content_vote','replica_normalized','replica_lower']
    args.output.mkdir(exist_ok=False)
    freeze=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                scorer_sha256=sha(Path(__file__).with_name('replica_moments.py')),
                runner_sha256=sha(__file__),
                primary='replica_lower', stability_multiplier=2.576,
                threshold_rank=3, threshold_operator='>=',
                calibration='60 full-carrier queries from conditional_quantile_pilot_20261112_v2',
                development='shared_semantic_evidence_20261112; previously inspected outcomes',
                reuse_scope='Cached-model development screen; not fresh execution or independent confirmation',
                attacks='Existing unchanged published CVPR2025 Imprint outputs only')
    (args.output/'protocol.json').write_text(json.dumps(freeze,indent=2)+'\n')
    ids={}
    def get_id(root,t):
        key=str(root),t
        if key not in ids:
            _,c,m,b=identity(root,t);ids[key]=c,m,order(b,16384)
        return ids[key]
    def score(z,q,key):
        c,m,perm=key
        v=statistics(z,q,c,perm)
        bits=(z.reshape(-1)>0).astype(np.uint8)^np.tile(q,16)[perm]^c^np.tile(m,64)
        v['ba']=float(np.mean((bits.reshape(64,256).sum(0)>32)==m))
        # Independent explicit off-diagonal calculation of the second moment.
        r=np.empty(16384);r[perm]=(1-2*((z.reshape(-1)>0).astype(np.uint8)^c).astype(float))
        r=r.reshape(16,1024).T
        b=sum(float(np.mean(r[:,i]*r[:,j])) for i in range(16) for j in range(i+1,16))/120
        assert abs(b-v['second_moment']) < 1e-12
        return v
    calibration=[]
    for x in plan['development']:
        if x['split']!='calibration':continue
        for k in ['latent','code']:assert sha(x[k])==x[k+'_sha256']
        v=score(np.load(x['latent']),np.load(x['code'])['code'],get_id(x['identity_root'],x['task']))
        calibration.append(dict(task=x['task'],condition=x['condition'],scores=v))
    assert len(calibration)==60
    thresholds={k:sorted(r['scores'][k] for r in calibration)[3] for k in names}
    (args.output/'calibration.json').write_text(json.dumps(dict(thresholds=thresholds,rows=calibration),indent=2)+'\n')
    def decide(v):return dict(default_ba=v['ba']>181/256,**{k:v[k]>=thresholds[k] for k in names})
    rows=[];wrong=[]
    for stage in ['genuine','forgery']:
        evidence=json.loads((shared/f'{stage}_verified.json').read_text())
        for x in evidence['rows']:
            for k in ['latent','code','image']:assert sha(shared/x[k])==x[k+'_sha256']
            z=np.load(shared/x['latent']);q=np.load(shared/x['code'])['code']
            v=score(z,q,get_id(plan['identity_root'],x['task']))
            assert v['ba']==x['ba']
            rows.append(dict(id=x['id'],stage=stage,task=x['task'],condition=x.get('condition'),
                             kind=x.get('kind'),update=x.get('update'),scores=v,decisions=decide(v)))
            if stage=='genuine':
                for t in range(12):
                    if t==x['task']:continue
                    w=score(z,q,get_id(plan['identity_root'],t))
                    wrong.append(dict(id=x['id'],identity=t,decisions=decide(w)))
    groups={'genuine':[r for r in rows if r['stage']=='genuine'],
            'clean':[r for r in rows if r['condition']=='clean'],
            'final_imprint':[r for r in rows if r['kind']=='candidate' and r['update']==151],
            'all_checkpoints':[r for r in rows if r['kind']=='candidate'],
            'cover':[r for r in rows if r['kind']=='cover'],'wrong_identity':wrong}
    summary={k:{g:dict(accepted=sum(r['decisions'][k] for r in rr),n=len(rr)) for g,rr in groups.items()} for k in ['default_ba']+names}
    candidate=summary['replica_lower'];control=summary['default_ba']
    advance=(candidate['genuine']['accepted']>=control['genuine']['accepted'] and
             candidate['clean']['accepted']==12 and candidate['final_imprint']['accepted']<control['final_imprint']['accepted'] and
             candidate['wrong_identity']['accepted']==0)
    result=dict(completed=True,execution_host='249',fresh_model_execution=False,independent_confirmation=False,
                protocol_sha256=sha(args.output/'protocol.json'),calibration_sha256=sha(args.output/'calibration.json'),
                summary=summary,rows=rows,wrong=wrong,advance_to_fresh_pilot=advance,
                limitations='Moment cancellation assumes conditionally independent homogeneous sign errors; ratio penalty is not a calibrated confidence bound.')
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(summary=summary,advance_to_fresh_pilot=advance),indent=2))


if __name__=='__main__':main()
