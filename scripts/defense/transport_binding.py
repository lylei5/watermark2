"""Monotone Gaussian-to-orthant coupling for content-bound sign watermarking.

Conditional quantile transport is classical. This tests its causal coupling to
the semantic enrollment preview, with the standard hard sign decoder unchanged.
"""
import numpy as np
from scipy.special import ndtr,ndtri,erfinv
from anchor_binding import identity,order,embed as full_embed,decode as full_decode

ARMS=('full','saturated','transport','shuffled_transport')


def conditional_transport(z,target):
    z=np.asarray(z,dtype=np.float64);s=2*np.asarray(target,dtype=np.float64)-1
    u=s*z;a=np.empty_like(u);low=u<=0
    # Equivalent to Phi^-1((Phi(z)+target)/2), evaluated without cancellation.
    a[low]=np.sqrt(2)*erfinv(ndtr(u[low]))
    a[~low]=-ndtri(.5*ndtr(-u[~low]))
    result=(s*a).astype(np.float32)
    if not np.isfinite(result).all() or not np.array_equal(result>0,np.asarray(target,dtype=bool)):
        raise ValueError('Transport exceeded finite-precision sign range')
    return result


def embed(aux_latent,code,root,task,arm):
    if arm=='full':return full_embed(aux_latent,code,root,task,'full')
    if arm not in ARMS:raise ValueError(arm)
    secret,cipher,_,key=identity(root,task)
    z=np.asarray(aux_latent,dtype=np.float32).ravel()
    target=cipher^np.tile(np.asarray(code,dtype=np.uint8),16)[order(key,16384)]
    if arm=='saturated':result=np.abs(z)*(2*target.astype(np.float32)-1)
    else:
        result=conditional_transport(z,target)
        if arm=='shuffled_transport':
            rng=np.random.default_rng(secret['transport_shuffle_seed'])
            for bit in [0,1]:
                positions=np.flatnonzero(target==bit)
                result[positions]=result[rng.permutation(positions)]
    return result.reshape(1,4,64,64)


def decode(latent,code,root,task,arm):
    if arm not in ARMS:raise ValueError(arm)
    return full_decode(latent,code,root,task,'full')
