"""Check evidence separation and erasure behavior in the defense prototype."""
import unittest
import numpy as np
import torch

from blind_geometry_verifier import geometry_bank,query_observations,select_pose,test_pose,verify


class GeometryEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.poses,cls.grid,cls.support,cls.counts,cls.folds=geometry_bank()

    def test_identity_and_complete_group_partition(self):
        self.assertEqual(len(self.poses),251)
        self.assertTrue(self.support[0].all())
        np.testing.assert_array_equal(np.sort(np.concatenate(self.folds)),np.arange(256))
        self.assertEqual(len(np.intersect1d(*self.folds)),0)
        latent=np.random.default_rng(1).normal(size=(1,4,64,64)).astype(np.float32)
        signs,_=query_observations(latent,self.grid,self.support)
        np.testing.assert_array_equal(signs[0],latent.reshape(-1)>0)

    def test_held_out_bits_cannot_change_fitted_pose(self):
        rng=np.random.default_rng(2)
        counts=rng.binomial(self.counts,.55)
        before=select_pose(counts,self.counts,self.folds[0])
        counts[:,self.folds[1]]=self.counts[:,self.folds[1]]-counts[:,self.folds[1]]
        self.assertEqual(before,select_pose(counts,self.counts,self.folds[0]))

    def test_absence_and_ties_provide_no_evidence(self):
        totals=np.array([[0,8,8,8]])
        correct=np.array([[0,4,8,0]])
        score=test_pose(correct,totals,0,np.arange(4))
        self.assertEqual(score['n_decisive'],2)
        self.assertEqual(score['successes'],1)
        self.assertEqual(score['unknown_or_tied'],2)
        self.assertAlmostEqual(score['p'],.75)
        self.assertEqual(test_pose(np.zeros((1,4)),np.zeros((1,4)),0,np.arange(4))['p'],1.)

    def test_perfect_observation_and_complement(self):
        latent=np.random.default_rng(3).normal(size=(1,4,64,64)).astype(np.float32)
        signs,valid=query_observations(latent,self.grid,self.support)
        expected=latent.reshape(-1)>0
        result=verify(signs,valid,self.counts,expected,self.folds,self.poses)
        self.assertEqual(result['all_fit']['pose_index'],0)
        self.assertTrue(result['all_fit']['accepted'])
        self.assertTrue(result['split_fit']['accepted'])
        opposite=verify(signs,valid,self.counts,~expected,self.folds,self.poses)
        self.assertFalse(opposite['identity_accepted'])


if __name__=='__main__':unittest.main()
