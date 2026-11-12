"""Shared-semantic-bit evidence with calibrated query uncertainty.

Scores are model-based diagnostics, not calibrated p-values. Inference uses
query features, recovered noise and identity secrets, never enrollment codes.
"""
import numpy as np
from scipy.special import erf

NAMES=('ba','latent_soft_ba','coordinate_soft_ba','group_constant','group_query')


def carrier_posterior(z,model):
    a=float(model['gain']);v=float(model['noise_variance'])
    return np.clip(erf(a*np.asarray(z).ravel()/np.sqrt(2*v*(a*a+v))),-1+1e-8,1-1e-8)


def grouped_evidence(carrier,semantic,cipher,permutation):
    # One latent semantic variable per1024-bit group, not per carrier coordinate.
    signed=carrier*(1-2*np.asarray(cipher,dtype=float))
    half_llr=np.bincount(np.asarray(permutation)%1024,weights=np.arctanh(signed),minlength=1024)
    posterior=np.tanh(half_llr)
    return float(np.mean(np.log1p(np.clip(posterior*semantic,-1+1e-8,1-1e-8))))


def scores(z,logits,code,cipher,message,permutation,model):
    z=np.asarray(z).ravel();q=np.asarray(code,dtype=np.uint8);perm=np.asarray(permutation)
    stream=np.asarray(cipher,dtype=np.uint8)^np.tile(message,64)
    mask=np.tile(q,16)[perm];bits=(z>0).astype(np.uint8)^mask^stream
    ba=float(np.mean((bits.reshape(64,256).sum(0)>32)==message))
    carrier=carrier_posterior(z,model)
    semantic=np.tanh(np.asarray(logits,dtype=float)/(2*float(model['temperature'])))
    constant=(2*q.astype(float)-1)*(1-2*float(model['semantic_error']))
    hard_soft=carrier*(1-2*(mask^stream).astype(float))
    uncertain_soft=-carrier*np.tile(semantic,16)[perm]*(1-2*stream.astype(float))
    def recover(values):
        llr=np.arctanh(np.clip(values,-1+1e-8,1-1e-8)).reshape(64,256).sum(0)
        return float(np.mean((llr>0)==message))
    return dict(ba=ba,latent_soft_ba=recover(hard_soft),coordinate_soft_ba=recover(uncertain_soft),
                group_constant=grouped_evidence(carrier,constant,cipher,perm),
                group_query=grouped_evidence(carrier,semantic,cipher,perm))


def decisions(values,model):
    return dict(default_ba=values['ba']>181/256,**{k:values[k]>=model['thresholds'][k] for k in NAMES})
