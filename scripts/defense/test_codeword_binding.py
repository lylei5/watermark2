import unittest
import numpy as np
from codeword_binding import CodewordBinding, bit_accuracy


class BindingTests(unittest.TestCase):
    def test_repetition_decoder_commutes_with_binding_away_from_ties(self):
        rng = np.random.default_rng(17)
        b = CodewordBinding(bytes(range(32)))
        source = rng.integers(0, 2, 1024, dtype=np.uint8)
        candidate = source.copy()
        candidate[rng.choice(1024, 100, replace=False)] ^= 1
        mismatch = b.mask_bits(source) ^ b.mask_bits(candidate)
        msg = rng.integers(0, 2, 256, dtype=np.uint8)
        channel = np.zeros((64, 256), dtype=np.uint8)
        # Correlated and nonuniform channel, but strictly < 32 errors per column.
        channel[:21, :] = 1
        got = bit_accuracy(np.tile(msg, 64) ^ mismatch ^ channel.reshape(-1), msg)[0]
        expected = np.mean(b.semantic_message(source) == b.semantic_message(candidate))
        self.assertEqual(got, expected)

    def test_key_changes_map_and_all_source_bits_used_at_degree_four(self):
        a, b = CodewordBinding(b'a'*32), CodewordBinding(b'b'*32)
        self.assertFalse(np.array_equal(a.groups, b.groups))
        np.testing.assert_array_equal(np.sort(a.groups.ravel()), np.arange(1024))

    def test_involution_shape_and_batch(self):
        b = CodewordBinding(b'k'*32)
        codes = np.zeros((3,1024), dtype=np.uint8)
        signs = b.sign_mask(codes)
        self.assertEqual(signs.shape, (3,4,64,64))
        z = np.random.default_rng(42).normal(size=signs.shape)
        np.testing.assert_array_equal(z*signs*signs,z)

    def test_tie_is_zero(self):
        x = np.zeros((64,256), dtype=np.uint8)
        x[:32] = 1
        self.assertEqual(bit_accuracy(x, np.zeros(256))[0], 1)
        self.assertEqual(bit_accuracy(x, np.ones(256))[0], 0)


if __name__ == '__main__':
    unittest.main()
