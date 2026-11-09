"""Codeword-aligned semantic binding for the 64 x 256 GS repetition code.

The semantic sign is constant over a message bit's 64 copies. Consequently,
repetition decoding removes channel noise but preserves semantic mismatch.
A keyed sparse parity map tunes sensitivity before repetition encoding.
This does not prevent collisions in the underlying semantic representation.
"""
import hashlib
import hmac
import numpy as np


def keyed_order(key, domain, size):
    if len(key) < 16:
        raise ValueError('Use at least 128 bits of key material')
    tags = [hmac.new(key, domain + i.to_bytes(4, 'big'), hashlib.sha256).digest()
            for i in range(size)]
    return np.array(sorted(range(size), key=tags.__getitem__), dtype=np.int64)


class CodewordBinding:
    """A research carrier adapter, with the original GS detector unchanged."""
    def __init__(self, key, degree=4, aligned=True, source_bits=1024,
                 message_bits=256, repetitions=64):
        if degree < 1 or source_bits % message_bits:
            raise ValueError('Invalid parity degree or source dimensions')
        self.message_bits, self.repetitions = message_bits, repetitions
        self.source_bits, self.degree, self.aligned = source_bits, degree, aligned
        needed = message_bits * degree
        blocks = [keyed_order(key, b'casb/parity/' + b.to_bytes(4, 'big'), source_bits)
                  for b in range((needed + source_bits - 1)//source_bits)]
        self.groups = np.concatenate(blocks)[:needed].reshape(message_bits, degree)
        self.perm = keyed_order(key, b'casb/scatter/', message_bits * repetitions)

    def semantic_message(self, codes):
        codes = np.asarray(codes, dtype=np.uint8)
        if codes.shape[-1] != self.source_bits or np.any(codes > 1):
            raise ValueError('Expected binary semantic code')
        return np.bitwise_xor.reduce(codes[..., self.groups], axis=-1)

    def mask_bits(self, codes):
        message = self.semantic_message(codes)
        tiled = np.tile(message, (self.repetitions, 1)).reshape(-1) if message.ndim == 1 else np.tile(message, (1, self.repetitions))
        return tiled if self.aligned else tiled[..., self.perm]

    def sign_mask(self, codes):
        bits = self.mask_bits(codes)
        return (1 - 2 * bits.astype(np.float32)).reshape(*bits.shape[:-1], 4, 64, 64)


def sembind_mask_bits(codes, permutation):
    """Paper-based sigma=1 expansion; separate from CASB's changed layout."""
    return np.tile(np.asarray(codes, dtype=np.uint8), (1, 16))[:, permutation]


def bit_accuracy(decrypted_bits, message):
    """Match upstream GS tie-breaking: at exactly 32/64 ones, decode zero."""
    votes = np.asarray(decrypted_bits).reshape(-1, 64, 256).sum(axis=1)
    return ((votes > 32) == np.asarray(message)).mean(axis=-1)
