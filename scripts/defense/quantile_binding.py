"""Content-coupled Gaussian bit planes for a defensive embedding pilot.

Quantile modulation is established prior art. This pilot tests sharing the
unmodulated degrees of freedom with the semantic enrollment image.
"""
import numpy as np
from scipy.fft import dctn, idctn
from scipy.special import ndtr, ndtri
from anchor_binding import identity, order, partition
from anchor_binding import embed as anchor_embed, decode as anchor_decode

ARMS = ('quantile_coupled', 'quantile_independent')
MEDIAN = float(ndtri(.75))


def embed(aux_latent, code, root, task, arm):
    if arm not in ARMS:
        return anchor_embed(aux_latent, code, root, task, arm)
    s, cipher, _, key = identity(root, task)
    mask = np.tile(np.asarray(code, dtype=np.uint8), 16)[order(key, 16384)]
    bits = cipher ^ mask
    if arm == 'quantile_coupled':
        base = dctn(np.asarray(aux_latent, dtype=np.float64), axes=(-2, -1), norm='ortho').ravel()
    else:
        base = np.random.default_rng(s['independent_anchor_seed']).standard_normal(16384)
    low, high = partition('anchor_low')
    # The first32 full message copies use magnitude, the next32 use signs.
    coordinate_bits = np.empty_like(bits)
    coordinate_bits[np.concatenate([low, high])] = bits
    # Half-normal CDF; discard its first bit and preserve the residual rank.
    u = 2 * ndtr(np.abs(base[low])) - 1
    residual = np.mod(2 * u, 1)
    target = (residual + coordinate_bits[low]) / 2
    probability = np.clip((target + 1) / 2, .5, np.nextafter(1., 0.))
    z = base.copy()
    z[low] = np.where(base[low] >= 0, 1., -1.) * ndtri(probability)
    z[high] = np.abs(base[high]) * (2 * coordinate_bits[high].astype(float) - 1)
    return idctn(z.reshape(1, 4, 64, 64), axes=(-2, -1), norm='ortho').astype(np.float32)


def decode(latent, code, root, task, arm):
    if arm not in ARMS:
        return anchor_decode(latent, code, root, task, arm)
    _, cipher, message, key = identity(root, task)
    z = dctn(np.asarray(latent, dtype=np.float64), axes=(-2, -1), norm='ortho').ravel()
    low, high = partition('anchor_low')
    observations = (z > 0).astype(np.uint8)
    observations[low] = (np.abs(z[low]) >= MEDIAN).astype(np.uint8)
    observations = observations[np.concatenate([low, high])]
    mask = np.tile(np.asarray(code, dtype=np.uint8), 16)[order(key, 16384)]
    stream = cipher ^ np.tile(message, 64)
    votes = (observations ^ mask ^ stream).reshape(64, 256).sum(0)
    return float(np.mean((votes > 32) == message))
