import unittest
import numpy as np
from content_authentication import recover_groups,content_scores


class ContentAuthenticationTest(unittest.TestCase):
    def test_permutation_and_cipher_recovery(self):
        rng=np.random.default_rng(13)
        code=rng.integers(0,2,1024,dtype=np.uint8)
        perm=rng.permutation(16384); cipher=rng.integers(0,2,16384,dtype=np.uint8)
        symbols=cipher ^ np.tile(code,16)[perm]
        groups=recover_groups(symbols.astype(float)*2-1,cipher,perm)
        np.testing.assert_array_equal(groups,np.repeat(code[:,None],16,axis=1))
        self.assertEqual(content_scores(groups,code)['profiled_content_agreement'],1)

    def test_identifies_channel_and_content_separately(self):
        rng=np.random.default_rng(42)
        query=rng.integers(0,2,1024,dtype=np.uint8)
        delta=rng.random(1024)<.2
        noise=rng.random((1024,16))<.15
        groups=query[:,None]^delta[:,None]^noise
        score=content_scores(groups,query)
        self.assertLess(abs(score['estimated_channel_error']-.15),.03)
        self.assertLess(abs(score['profiled_content_agreement']-.8),.04)

    def test_unidentified_channel_cannot_claim_content_confidence(self):
        groups=np.tile(np.array([0]*8+[1]*8,dtype=np.uint8),(1024,1))
        score=content_scores(groups,np.zeros(1024,dtype=np.uint8))
        self.assertEqual(score['profile_content_lower_bound'],.5)


if __name__=='__main__':unittest.main()
