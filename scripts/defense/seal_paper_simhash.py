"""SEAL final Algorithm2, including its unreleased secret-salt operations.

This is a paper-specified implementation, separate from the unchanged public
seed42 code. The paper does not fix serialization or a seeded Gaussian RNG:
here SHA256 uses domain-separated fixed-width indices and raw32-byte salt,
and its first8 digest bytes seed a private PyTorch Generator on the device.
Those choices are explicit; author implementation equivalence is not asserted.
"""
import hashlib
import struct

import torch


class SealPaperSimHash:
    def __init__(self, salt: bytes, device='cuda'):
        if not isinstance(salt, bytes) or len(salt) != 32:
            raise ValueError('A private32-byte salt is required; no public default is allowed')
        self._salt = salt
        self.device = torch.device(device)
        self._projections = None
        self._dimension = None

    def _seed(self, domain, payload):
        digest = hashlib.sha256(domain + payload + self._salt).digest()
        return int.from_bytes(digest[:8], 'big')

    def _normal(self, seed, shape):
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        return torch.randn(shape, dtype=torch.float32, device=self.device, generator=generator)

    def _initialize(self, dimension):
        if self._projections is not None:
            if self._dimension != dimension:
                raise ValueError('Embedding dimension changed')
            return
        self._dimension = dimension
        self._projections = torch.stack([
            self._normal(self._seed(b'seal/projection\0', struct.pack('>II', i, j)), (dimension,))
            for i in range(1, 1025) for j in range(1, 8)
        ]).reshape(1024, 7, dimension)

    @torch.no_grad()
    def sample(self, embedding):
        vector = torch.as_tensor(embedding, dtype=torch.float32, device=self.device).flatten()
        if not torch.isfinite(vector).all() or not vector.norm() > 0:
            raise ValueError('Expected a finite, nonzero semantic vector')
        vector = vector / vector.norm()
        self._initialize(vector.numel())
        bits = (torch.sum(self._projections * vector, dim=-1) > 0).to(torch.uint8).cpu().tolist()
        result = torch.empty(1, 4, 64, 64, dtype=torch.float32, device=self.device)
        for offset, values in enumerate(bits):
            i = offset + 1
            seed = self._seed(b'seal/noise\0', struct.pack('>I', i) + bytes(values))
            patch = self._normal(seed, (1, 4, 2, 2))
            y, x = divmod(offset, 32)
            result[:, :, 2*y:2*y+2, 2*x:2*x+2] = patch
        return result
