"""Experimental content-conditioned rotation of a Gaussian sign code.

Rotations, random Fourier features and conditional Gaussian transport are prior
art. This module tests smooth content-dependent decoding regions, not new attacks.
"""
from functools import lru_cache
import hashlib
import hmac
import numpy as np
from scipy.special import ndtri
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from anchor_binding import identity, order, embed as baseline_embed, decode as baseline_decode
from transport_binding import conditional_transport

ARMS = ('full', 'quantized_geometry', 'continuous_independent', 'continuous_transport')
ALPHA = 2.0


def uniforms(key, domain, count):
    subkey = hmac.new(key, b'continuous-geometry/key/' + domain, hashlib.sha256).digest()
    nonce = hmac.new(key, b'continuous-geometry/nonce/' + domain, hashlib.sha256).digest()[:16]
    stream = Cipher(algorithms.ChaCha20(subkey, nonce), mode=None).encryptor().update(bytes(count * 8))
    integer = np.frombuffer(stream, dtype='<u8') >> 12
    return (integer.astype(np.float64) + .5) / 2.0**52


@lru_cache(maxsize=32)
def geometry(root, task):
    _, _, _, key = identity(root, task)
    frequency = ndtri(uniforms(key, b'frequency', 1024 * 1024)).reshape(1024, 1024)
    phase = 2 * np.pi * uniforms(key, b'phase', 8192)
    permutation = order(key, 16384)
    return frequency, phase, permutation


def feature(bundle):
    value = np.asarray(bundle['logits'], dtype=np.float64).reshape(-1)
    if value.shape != (1024,) or not np.isfinite(value).all() or np.linalg.norm(value) == 0:
        raise ValueError('Expected finite, nonzero1024-dimensional semantic logits')
    return value / np.linalg.norm(value)


def angles(bundle, root, task, arm):
    frequency, phase, permutation = geometry(str(root), task)
    projection = frequency @ feature(bundle)
    if arm == 'quantized_geometry':
        content = (np.pi / 2) * np.where(projection > 0, 1., -1.)
    elif arm in ['continuous_independent', 'continuous_transport']:
        content = ALPHA * projection
    else:
        raise ValueError(arm)
    return phase + np.tile(content, 8), permutation


def rotate(value, theta, permutation, inverse=False):
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    pairs = flat[permutation].reshape(8192, 2)
    c = np.cos(theta)
    s = np.sin(theta) * (-1 if inverse else 1)
    transformed = np.column_stack((c * pairs[:, 0] - s * pairs[:, 1],
                                   s * pairs[:, 0] + c * pairs[:, 1]))
    output = np.empty(16384, dtype=np.float64)
    output[permutation] = transformed.ravel()
    return output.reshape(1, 4, 64, 64)


def embed(aux_latent, bundle, root, task, arm):
    if arm == 'full':
        return baseline_embed(aux_latent, bundle['code'], root, task, 'full')
    if arm not in ARMS:
        raise ValueError(arm)
    secret, cipher, _, _ = identity(root, task)
    theta, permutation = angles(bundle, root, task, arm)
    if arm == 'continuous_independent':
        amplitude = np.abs(np.random.default_rng(secret['amplitude_seed']).standard_normal(16384))
        encoded = amplitude * (2 * cipher.astype(float) - 1)
    else:
        source = rotate(aux_latent, theta, permutation, inverse=True).ravel()
        encoded = conditional_transport(source, cipher)
    return rotate(encoded, theta, permutation).astype(np.float32)


def decode(latent, bundle, root, task, arm):
    if arm == 'full':
        return baseline_decode(latent, bundle['code'], root, task, 'full')
    if arm not in ARMS:
        raise ValueError(arm)
    _, cipher, message, _ = identity(root, task)
    theta, permutation = angles(bundle, root, task, arm)
    coordinates = rotate(latent, theta, permutation, inverse=True).ravel()
    bits = (coordinates > 0).astype(np.uint8) ^ cipher ^ np.tile(message, 64)
    recovered = bits.reshape(64, 256).sum(0) > 32
    return float(np.mean(recovered == message))
