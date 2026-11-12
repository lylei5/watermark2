"""Two-moment semantic authentication on an unchanged SemBind carrier.

The ratio cancels a homogeneous independent sign channel in expectation.
Its stability penalty is not a confidence guarantee for diffusion latents.
"""
import numpy as np


def statistics(latent, query_code, cipher, permutation):
    observed = (np.asarray(latent).reshape(-1) > 0).astype(np.uint8)
    unbound = observed ^ np.asarray(cipher, dtype=np.uint8)
    restored = np.empty_like(unbound)
    restored[np.asarray(permutation)] = unbound
    residual = 1.0 - 2.0 * (restored.reshape(16, 1024).T ^ np.asarray(query_code, dtype=np.uint8)[:, None])
    n = residual.shape[1]
    group_a = residual.mean(axis=1)
    group_b = (residual.sum(axis=1)**2 - n) / (n*(n-1))
    a, b = float(group_a.mean()), float(group_b.mean())
    vote = float(np.mean(group_a > 0) + .5*np.mean(group_a == 0))
    if b <= 0:
        ratio, lower, penalty = -1., -1., None
    else:
        ratio = float(a/np.sqrt(b))
        influence = (group_a-a)/np.sqrt(b) - a*(group_b-b)/(2*b**1.5)
        penalty = float(2.576*influence.std(ddof=1)/np.sqrt(len(influence)))
        lower = ratio-penalty
    return dict(raw_agreement=(1+a)/2, content_vote=vote,
                replica_normalized=ratio, replica_lower=lower,
                first_moment=a, second_moment=b, stability_penalty=penalty)
