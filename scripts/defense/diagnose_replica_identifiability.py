"""Assess a homogeneous sign-noise model on consumed development artifacts.

No authentication rule or attack is changed. Enrollment quantities are oracle
diagnostics. The mixture independence assumption is not a security guarantee.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fit(hist):
    k=np.arange(17);n=16;logc=gammaln(17)-gammaln(k+1)-gammaln(17-k)
    def fun(theta):
        r,e=theta
        a=logc+k*np.log(r)+(n-k)*np.log1p(-r)+np.log1p(-e)
        b=logc+(n-k)*np.log(r)+k*np.log1p(-r)+np.log(e)
        ll=np.logaddexp(a,b);w=np.exp(a-ll)
        dr=w*(k/r-(n-k)/(1-r))+(1-w)*((n-k)/r-k/(1-r))
        de=-w/(1-e)+(1-w)/e
        return -float(hist@ll),-np.array([hist@dr,hist@de])
    solutions=[minimize(fun,np.array([r,e]),jac=True,method='L-BFGS-B',bounds=[(.500001,.999999),(.000001,.499999)],
                       options=dict(maxiter=300,maxls=60,ftol=1e-10,gtol=1e-6)) for r in [.6,.8,.95] for e in [.05,.35]]
    converged=[s for s in solutions if s.success and np.isfinite(s.fun)]
    assert converged,'No converged mixture fit'
    best=min(converged,key=lambda s:s.fun)
    assert best.fun-min(s.fun for s in solutions)<1e-5,'Unconverged start has materially better objective'
    r,e=map(float,best.x)
    a=logc+k*np.log(r)+(n-k)*np.log1p(-r)+np.log1p(-e)
    b=logc+(n-k)*np.log(r)+k*np.log1p(-r)+np.log(e)
    expected=np.exp(np.logaddexp(a,b));empirical=hist/hist.sum();nz=hist>0
    return dict(estimated_channel_correct=r,estimated_semantic_error=e,
        nll=float(best.fun),histogram_kl=float(np.sum(empirical[nz]*np.log(empirical[nz]/expected[nz]))),
        multistart_objective_span=float(max(s.fun for s in solutions)-best.fun),
        converged_starts=len(converged),
        optimizer_success=bool(best.success))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run=a.root/'runs/shared_semantic_evidence_20261112';plan=json.loads((run/'protocol.json').read_text())
    sys.path.insert(0,str(run));from anchor_binding import identity,order
    records=[];keys={}
    for x in plan['development']:
        if x['split']=='calibration':records.append(dict(x,phase='calibration',kind='genuine',update=None,id=f"cal_{x['task']}_{x['condition']}"))
    for phase in ['genuine','forgery']:
        d=json.loads((run/f'{phase}_verified.json').read_text());assert d['audit_passed']
        for x in d['rows']:
            records.append(dict(x,phase=phase,kind=x.get('kind','genuine'),update=x.get('update'),
                identity_root=plan['identity_root'],latent=str(run/x['latent']),code=str(run/x['code'])))
    rows=[]
    for x in records:
        for key in ['latent','code']:assert sha(x[key])==x[key+'_sha256']
        kid=(x['identity_root'],x['task'])
        if kid not in keys:
            _,c,_,b=identity(kid[0],kid[1]);keys[kid]=(c,order(b,16384))
        cipher,perm=keys[kid];z=np.load(x['latent']).ravel();q=np.load(x['code'])['code']
        if x['phase']=='calibration':hpath=Path(x['code']).parent/f"{x['task']:03}_aux.npz"
        else:hpath=run/f"genuine_{x['task']%3}/codes/{x['task']:03}_aux.npz"
        h=np.load(hpath)['code'];unbound=(z>0).astype(np.uint8)^cipher
        restored=np.empty_like(unbound);restored[perm]=unbound
        aligned=(restored.reshape(16,1024).T==q[:,None]);counts=aligned.sum(1);hist=np.bincount(counts,minlength=17)
        result=fit(hist);actual_error=float((h!=q).mean());channel=float(np.mean(unbound==np.tile(h,16)[perm]))
        rows.append(dict(id=x['id'],phase=x['phase'],kind=x['kind'],update=x['update'],task=x['task'],
            condition=x.get('condition'),oracle_semantic_error=actual_error,oracle_channel_correct=channel,
            histogram=hist.tolist(),**result))
    groups={'calibration':[r for r in rows if r['phase']=='calibration'],
        'genuine':[r for r in rows if r['phase']=='genuine'],
        'final_imprint':[r for r in rows if r['kind']=='candidate' and r['update']==151],
        'cover':[r for r in rows if r['kind']=='cover']}
    summary={}
    for key,rr in groups.items():
        errors=[abs(r['estimated_semantic_error']-r['oracle_semantic_error']) for r in rr]
        summary[key]=dict(n=len(rr),semantic_error_mae=float(np.mean(errors)),semantic_error_max=float(max(errors)),
            channel_correct_mae=float(np.mean([abs(r['estimated_channel_correct']-r['oracle_channel_correct']) for r in rr])),
            mean_histogram_kl=float(np.mean([r['histogram_kl'] for r in rr])))
    result=dict(completed=True,rows=rows,summary=summary,script_sha256=sha(__file__),
        scope='All60 calibration and336 consumed development/forgery records. Cached scoring only. No thresholds selected, no authentication decisions or attacks changed.',
        model='1024 groups of16 signs: mixture (1-epsilon)Binomial(16,r)+epsilon Binomial(16,1-r), six-start MLE; shared homogeneous physical correctness r.',
        limitation='Groups and signs may be dependent and heterogeneous; multistart likelihood fitting is descriptive and does not prove a global optimum. Enrollment oracle is unavailable at inference. Near r=0.5, semantic mismatch is unidentifiable.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
