"""SemBind final-paper §3.2 mask expansion; permutation KDF is project specified.

The author release provides the semantic network and diffusion components, but
not this complete binding entry. No carrier values, decoder or threshold are
changed here. The permutation key is private and must be persisted by the owner.
"""
import hashlib
import hmac
import math

import numpy as np


class SemBindMask:
    def __init__(self, key: bytes, sigma: float, shape=(1, 4, 64, 64)):
        if not isinstance(key, bytes) or len(key) < 16:
            raise ValueError('A private key of at least128 bits is required')
        if not math.isfinite(sigma) or not 0 <= sigma <= 1:
            raise ValueError('sigma must be in[0,1]')
        self.shape = tuple(shape)
        self.length = math.prod(self.shape)
        if self.shape != (1, 4, 64, 64):
            raise ValueError('This baseline fixes the SD2.1 latent shape')
        tags = [hmac.new(key, b'sembind/permutation/' + i.to_bytes(4, 'big'), hashlib.sha256).digest()
                for i in range(self.length)]
        self.permutation = np.asarray(sorted(range(self.length), key=tags.__getitem__), dtype=np.int64)
        self.active_length = math.floor(sigma * self.length)

    def bits(self, code):
        values = np.asarray(code)
        if values.shape != (1024,) or not np.isin(values, [0, 1]).all():
            raise ValueError('Expected one1024-bit semantic code')
        expanded = np.tile(values.astype(np.uint8), 16)
        # The paper zeros the suffix before applying the keyed permutation.
        expanded[self.active_length:] = 0
        return expanded[self.permutation].reshape(self.shape)

    def signs(self, code):
        return 1 - 2 * self.bits(code).astype(np.float32)
