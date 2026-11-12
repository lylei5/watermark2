"""Independent probability-space audit for shared semantic evidence."""
import numpy as np
from scipy.special import ndtr,expit,log_expit,logsumexp


def grouped_reference(z,logits,code,cipher,permutation,model):
    a=model['gain'];v=model['noise_variance'];index=np.asarray(permutation)%1024
    r=np.clip(2*ndtr(a*np.asarray(z).ravel()/np.sqrt(v*(a*a+v)))-1,-1+1e-8,1-1e-8)
    t=r*np.where(np.asarray(cipher)==0,1.,-1.)
    groups=np.argsort(index,kind='stable').reshape(1024,16)
    assert np.array_equal(index[groups],np.repeat(np.arange(1024)[:,None],16,axis=1))
    first=np.log((1+t[groups])/2).sum(1);second=np.log((1-t[groups])/2).sum(1)
    normalizer=np.logaddexp(first,second)-np.log(2.)
    l=np.asarray(logits)/model['temperature']
    candidate=np.logaddexp(first+log_expit(l),second+log_expit(-l))-normalizer
    e=model['semantic_error'];p=np.where(np.asarray(code)==1,1-e,e)
    constant=np.logaddexp(first+np.log(p),second+np.log1p(-p))-normalizer
    low=np.log(1e-8);high=np.log(2-1e-8)
    return dict(group_query=float(np.clip(candidate,low,high).mean()),group_constant=float(np.clip(constant,low,high).mean()))
