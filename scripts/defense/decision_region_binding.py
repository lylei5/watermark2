"""Sign-reflection projection into a GS majority-decoding region.

All arms keep the same semantic expansion, cipher and hard decoder.
Minimal56 is L2-optimal only among sign reflections at fixed magnitudes.
"""
import numpy as np
from anchor_binding import identity, order, embed as full_embed, decode as full_decode


ARMS = ('full', 'saturated', 'minimal56', 'random56')


def embed(aux_latent, code, root, task, arm):
    if arm == 'full':
        return full_embed(aux_latent, code, root, task, 'full')
    if arm not in ARMS:
        raise ValueError(arm)
    secret, cipher, _, key = identity(root, task)
    z = np.asarray(aux_latent, dtype=np.float32).reshape(-1).copy()
    target = cipher ^ np.tile(np.asarray(code, dtype=np.uint8), 16)[order(key, 16384)]
    if arm == 'saturated':
        return (np.abs(z)*(2*target.astype(np.float32)-1)).reshape(1,4,64,64)
    wrong = (z > 0) != target
    rng = np.random.default_rng(secret['projection_order_seed'])
    ranks = rng.random(16384)
    for bit in range(256):
        group = np.arange(bit, 16384, 256)
        candidates = group[wrong[group]]
        number = max(0, len(candidates)-8)
        cost = np.abs(z[candidates]) if arm == 'minimal56' else ranks[candidates]
        selected = candidates[np.argsort(cost, kind='stable')[:number]]
        z[selected] *= -1
    return z.reshape(1,4,64,64)


def decode(latent, code, root, task, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return full_decode(latent, code, root, task, 'full')
