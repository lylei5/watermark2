"""Experimental content likelihood with a genuine-only inversion-channel model.

The score is a composite likelihood, not a calibrated probability or p-value.
No enrollment code is available at query scoring time.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def channel_features(latent):
    z=np.asarray(latent,dtype=np.float64).reshape(4,64,64)
    rms=max(float(np.sqrt(np.mean(z*z))),1e-6)
    a=np.minimum(np.abs(z)/rms,5.)
    padded=np.pad(a,((0,0),(1,1),(1,1)),mode='reflect')
    local=sum(padded[:,dy:dy+64,dx:dx+64] for dy in range(3) for dx in range(3))/9
    return np.stack([a.ravel(),(a*a).ravel(),local.ravel(),np.full(16384,np.log(rms))],axis=1)


def fit_channel(features,errors,l2=0.001):
    x=np.concatenate(features);y=np.concatenate(errors).astype(np.float64)
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-6)
    x=np.column_stack([np.ones(len(x)),(x-mean)/scale])
    def objective(w):
        t=x@w
        loss=np.mean(np.logaddexp(0,t)-y*t)+l2*np.sum(w[1:]**2)/2
        grad=x.T@(expit(t)-y)/len(y)
        grad[1:]+=l2*w[1:]
        return loss,grad
    result=minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',
                    options=dict(maxiter=200,ftol=1e-12,gtol=1e-8))
    if not result.success:
        raise RuntimeError('Channel fit did not converge: '+str(result.message))
    return dict(mean=mean.tolist(),scale=scale.tolist(),weights=result.x.tolist(),
                l2=l2,iterations=int(result.nit),training_mean_error=float(y.mean()),
                objective=float(result.fun),optimizer_success=True)


def predict_channel(model,features):
    x=(features-np.asarray(model['mean']))/np.asarray(model['scale'])
    p=expit(np.asarray(model['weights'])[0]+x@np.asarray(model['weights'])[1:])
    # A predicted worse-than-chance coordinate supplies no positive evidence.
    return np.clip(p,0.005,0.495)


def score(latent,cipher,permutation,query_code,model,benign_mismatch):
    signs=(np.asarray(latent).reshape(-1)>0).astype(np.uint8)
    mask=np.tile(np.asarray(query_code,dtype=np.uint8),16)[permutation]
    mismatch=signs ^ np.asarray(cipher,dtype=np.uint8) ^ mask
    p=predict_channel(model,channel_features(latent))
    coordinate_llr=(1-2*mismatch.astype(np.float64))*np.log((1-p)/p)
    restored=np.empty(16384);restored[permutation]=coordinate_llr
    group_llr=restored.reshape(16,1024).sum(0)
    d=float(np.clip(benign_mismatch,0.005,0.495))
    llr=np.logaddexp(np.log1p(-d)+group_llr,np.log(d))-np.logaddexp(group_llr,0)+np.log(2.)
    # Ablation: identical semantic hypothesis with one constant channel error.
    e=float(np.clip(model['training_mean_error'],0.005,0.495))
    unweighted=np.empty(16384);unweighted[permutation]=1-2*mismatch.astype(np.float64)
    global_llr=unweighted.reshape(16,1024).sum(0)*np.log((1-e)/e)
    global_score=np.logaddexp(np.log1p(-d)+global_llr,np.log(d))-np.logaddexp(global_llr,0)+np.log(2.)
    # Same learned reliability but no content-group hypothesis (soft-vote control).
    agreement=float((np.sum(restored.reshape(16,1024),0)>0).mean()+0.5*(np.sum(restored.reshape(16,1024),0)==0).mean())
    return dict(conditional_content_llr=float(llr.mean()),global_content_llr=float(global_score.mean()),
                reliability_weighted_content_agreement=agreement,
                predicted_channel_error=float(p.mean()))
