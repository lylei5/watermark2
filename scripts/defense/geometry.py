"""Geometry defense scores (ICML 2026 published formulas).

Definitions implemented from the final paper text:
- Definition 4.5: G-Cos = <zT, zhatT> / (||zT|| ||zhatT||); SAD = arccos(G-Cos).
- Definition 4.8: local token-centred covariance with 1/M sum and epsilon*I; local
  distance ||log(P^{-1/2} Q P^{-1/2})||_F; 16x16 spatial grid; top-k mean (k=5).

NOT implemented here (still unknown / not derivable from the public paper):
- the deployment thresholds tau_s, tau_l and the final AND decision;
- the exact author token-to-patch mapping, epsilon and matrix-log backend.

The L-SPD implementation below states its one assumption explicitly: a token is a
spatial position inside a grid cell, carrying its 4 latent channels. It is a
paper-formula reimplementation for score/AUC reporting, not the authors' code and
not a claim of numerical equivalence.
"""
import numpy as np


def _flat(x):
    return np.asarray(x, dtype=np.float64).reshape(-1)


def gcos(a, b):
    a = _flat(a)
    b = _flat(b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError('Zero-norm latent')
    return float(np.clip(a @ b / (na * nb), -1.0, 1.0))


def sad(a, b):
    return float(np.arccos(gcos(a, b)))


def _sym(c):
    return (c + c.T) / 2.0


def _cov(tokens, eps):
    tokens = np.asarray(tokens, dtype=np.float64)
    centred = tokens - tokens.mean(axis=0)
    m = tokens.shape[0]
    return _sym(centred.T @ centred / m + eps * np.eye(tokens.shape[1]))


def _eig(c):
    w, v = np.linalg.eigh(_sym(c))
    return np.clip(w, 1e-12, None), v


def _logm(c):
    w, v = _eig(c)
    return (v * np.log(w)) @ v.T


def _inv_sqrt(c):
    w, v = _eig(c)
    return (v * (1.0 / np.sqrt(w))) @ v.T


def lspd(a, b, grid=16, eps=1e-6, k=5):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError('Shape mismatch')
    if a.ndim == 4:
        if a.shape[0] != 1:
            raise ValueError('Expected batch size 1')
        a, b = a[0], b[0]
    if a.ndim != 3:
        raise ValueError('Expected (4, 64, 64) latents')
    c, h, w = a.shape
    if h % grid or w % grid:
        raise ValueError('Grid must divide latent size')
    ch, cw = h // grid, w // grid
    distances = []
    for gh in range(grid):
        for gw in range(grid):
            block_a = a[:, gh * ch:(gh + 1) * ch, gw * cw:(gw + 1) * cw].reshape(c, -1).T
            block_b = b[:, gh * ch:(gh + 1) * ch, gw * cw:(gw + 1) * cw].reshape(c, -1).T
            p = _cov(block_a, eps)
            q = _cov(block_b, eps)
            p_inv_sqrt = _inv_sqrt(p)
            distances.append(float(np.linalg.norm(_logm(p_inv_sqrt @ q @ p_inv_sqrt), 'fro')))
    distances.sort(reverse=True)
    return float(np.mean(distances[:k]))


def auc(positives, negatives):
    """Rank-based AUC (Mann-Whitney), ties counted as half."""
    pos = np.asarray(positives, dtype=np.float64)
    neg = np.asarray(negatives, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        raise ValueError('Empty class')
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind='mergesort')
    ranks = np.empty(allv.size, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1, dtype=np.float64)
    # average ranks for ties
    sorted_vals = allv[order]
    i = 0
    while i < sorted_vals.size:
        j = i
        while j + 1 < sorted_vals.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    pos_ranks = ranks[:pos.size]
    return float((pos_ranks.sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))
