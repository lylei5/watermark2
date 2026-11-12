"""Post-completion oracle decomposition of the clean rotation screen.

Uses saved full-budget inversions only; does not select thresholds, change images,
or expose enrollment information to the deployed verifier.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();run=a.run
    state=json.loads((run/'pipeline_state.json').read_text())
    assert state['status']=='complete'
    plan=json.loads((run/'protocol.json').read_text())
    g=json.loads((run/'genuine_verified.json').read_text())
    assert g['audit_passed'] and len(g['rows'])==24
    sys.path.insert(0,str(run))
    from continuous_geometry_binding import decode,angles,rotate
    from anchor_binding import identity,order
    def bundle(path):
        with np.load(path) as data:return {k:data[k] for k in ['code','logits']}
    rows=[]
    refs={(r['task'],r['arm']):r for r in g['references']}
    for row in g['rows']:
        t=row['task'];arm=row['arm'];ref=refs[t,arm]
        assert sha(run/row['latent'])==row['latent_sha256'] and sha(run/row['code'])==row['code_sha256']
        assert sha(run/ref['initial_latent'])==ref['initial_latent_sha256']
        q=bundle(run/row['code']);h=bundle(run/f'genuine_{t%3}/codes/{t:03}_aux.npz')
        initial=np.load(run/ref['initial_latent']);z=np.load(run/row['latent']);root=Path(plan['identity_root'])
        actual=decode(z,q,root,t,arm);oracle=decode(z,h,root,t,arm);noiseless=decode(initial,q,root,t,arm)
        assert actual==row['ba']
        _,cipher,_,key=identity(root,t)
        if arm=='full':
            target=cipher^np.tile(h['code'],16)[order(key,16384)]
            original=initial.ravel();observed=z.ravel();angular={}
        else:
            theta,perm=angles(h,root,t,arm);query_theta,query_perm=angles(q,root,t,arm)
            assert np.array_equal(perm,query_perm)
            target=cipher
            original=rotate(initial,theta,perm,inverse=True).ravel()
            observed=rotate(z,theta,perm,inverse=True).ravel()
            angular=dict(mean_orientation_cosine=float(np.mean(np.cos(theta-query_theta))),
                         rms_wrapped_angle=float(np.sqrt(np.mean(np.angle(np.exp(1j*(theta-query_theta)))**2))))
        gain=float(original@observed/(original@original))
        residual=observed-gain*original
        sigma=float(np.sqrt(np.mean(residual**2))/abs(gain)) if gain else None
        rows.append(dict(id=row['id'],task=t,arm=arm,actual_ba=actual,actual_accepted=actual>181/256,
            enrollment_oracle_ba=oracle,enrollment_oracle_accepted=oracle>181/256,
            noiseless_query_ba=noiseless,noiseless_query_accepted=noiseless>181/256,
            raw_enrollment_symbol_correctness=float(np.mean((observed>0)==target)),
            reference_distance=ref['continuous_registration_distance'],mean_gain=gain,
            descriptive_effective_sigma=sigma,**angular))
    summary=[]
    for arm in plan['arms']:
        rr=[r for r in rows if r['arm']==arm]
        summary.append(dict(arm=arm,n=len(rr),actual_accepted=sum(r['actual_accepted'] for r in rr),
            enrollment_oracle_accepted=sum(r['enrollment_oracle_accepted'] for r in rr),
            noiseless_query_accepted=sum(r['noiseless_query_accepted'] for r in rr),
            mean_raw_enrollment_correctness=float(np.mean([r['raw_enrollment_symbol_correctness'] for r in rr])),
            mean_reference_distance=float(np.mean([r['reference_distance'] for r in rr]))))
    result=dict(completed=True,rows=rows,summary=summary,script_sha256=sha(__file__),
        protocol_sha256=sha(run/'protocol.json'),manifest_sha256=sha(run/'genuine_verified.json'),
        limitation='Enrollment and initial-noise scores are oracle counterfactuals, unavailable to deployed query-only verification. Gain/sigma are descriptive least-square summaries, not evidence of independent Gaussian noise. No thresholds fitted.')
    (run/'channel_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
