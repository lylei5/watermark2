"""Content-first verification of an existing SemBind carrier.

The claimed GS identity supplies the encrypted symbol signs. Removing those
signs and undoing SemBind's permutation exposes 16 noisy repetitions of each
content bit. A mixture model separates per-symbol inversion errors from
coherent content disagreement. This is an experimental detector, not a
cryptographic authenticity proof; adaptive feature collisions remain possible.
"""
import numpy as np


def recover_groups(inverted_latent, cipherbits, permutation):
    signs=(np.asarray(inverted_latent).reshape(-1)>0).astype(np.uint8)
    mask=signs ^ np.asarray(cipherbits,dtype=np.uint8)
    restored=np.empty_like(mask)
    restored[np.asarray(permutation)]=mask
    return restored.reshape(16,1024).T


def content_scores(groups,query_code):
    """Known-identity test with a profiled inversion-error nuisance parameter.

    Under the symmetric channel model, a group has mismatch count k~Bin(16,e)
    if content agrees, and Bin(16,1-e) otherwise. The group mixture estimates d
    independently of e. Combinatorial coefficients cancel in likelihood ratios.
    The e grid is fixed, not fitted to attack scores.
    """
    groups=np.asarray(groups,dtype=np.uint8)
    query=np.asarray(query_code,dtype=np.uint8).reshape(-1)
    k=(groups!=query[:,None]).sum(1)
    n=groups.shape[1]
    histogram=np.bincount(k,minlength=n+1)
    counts=np.arange(n+1)
    e_grid=np.linspace(.005,.495,99)
    a=counts[:,None]*np.log(e_grid)+(n-counts[:,None])*np.log1p(-e_grid)
    b=(n-counts[:,None])*np.log(e_grid)+counts[:,None]*np.log1p(-e_grid)
    d_grid=np.linspace(0,.5,101)
    best=(-np.inf,None,None);profile=[]
    for d in d_grid:
        ll=(np.logaddexp(a+np.log(max(1-d,1e-15)),b+np.log(max(d,1e-15)))*histogram[:,None]).sum(0)
        j=ll.argmax()
        profile.append(float(ll[j]))
        if ll[j]>best[0]:best=(float(ll[j]),float(d),float(e_grid[j]))
    # An unidentified noisy channel must not earn confidence from a favorable
    # point estimate. This is an approximate profile support interval, not a
    # coverage guarantee under correlated diffusion inversion errors.
    supported=d_grid[2*(best[0]-np.asarray(profile))<=3.841458820694124]
    lower_bound=1-float(supported.max())
    # Raw and content-first baselines are necessary: the mixture must improve
    # on these at matched genuine retention to justify the extra model.
    strict=(k>n/2).mean()
    ties=(k==n/2).mean()
    return dict(raw_agreement=float(1-k.mean()/n),
                content_agreement=float(1-strict-.5*ties),
                profiled_content_agreement=float(1-best[1]),
                profile_content_lower_bound=lower_bound,
                estimated_channel_error=best[2],profile_log_likelihood=best[0],
                content_tie_fraction=float(ties))
