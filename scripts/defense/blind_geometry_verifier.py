"""CPU defense prototype: geometry is a nuisance inferred from the query.

No source image, auxiliary code, true transform, or candidate generator is used.
Binomial scores below are model-based diagnostics, not empirical FPR guarantees.
"""
import numpy as np
from scipy.stats import binom
import torch
import torch.nn.functional as F

ALPHA = float(binom.sf(181, 256, .5))


def geometry_bank():
    """Identity plus 10 square scales and 25 translations each (251 poses)."""
    poses=[dict(scale=1.,left=0.,top=0.)]
    for tick in range(10):
        scale=.75+.025*tick
        for ix in range(5):
            for iy in range(5):
                poses.append(dict(scale=scale,left=(1-scale)*512*ix/4,top=(1-scale)*512*iy/4))
    axis=(torch.arange(64,dtype=torch.float32)+.5)*8
    grids=[];supports=[]
    for pose in poses:
        x=2*(axis-pose['left'])/(pose['scale']*512)-1
        y=2*(axis-pose['top'])/(pose['scale']*512)-1
        yy,xx=torch.meshgrid(y,x,indexing='ij')
        grids.append(torch.stack([xx,yy],-1))
        supports.append((xx>=-1+1/64)&(xx<=1-1/64)&(yy>=-1+1/64)&(yy<=1-1/64))
    grid=torch.stack(grids)
    support=torch.stack(supports)[:,None].expand(-1,4,-1,-1).numpy().reshape(len(poses),16384)
    counts=support.reshape(len(poses),64,256).sum(1)
    order=np.random.default_rng(202611101).permutation(256)
    folds=[np.sort(order[:128]),np.sort(order[128:])]
    return poses,grid,support,counts,folds


def query_observations(latent,grid,support):
    latent=np.asarray(latent,dtype=np.float32).reshape(1,4,64,64)
    if not np.isfinite(latent).all():raise ValueError('Nonfinite inversion')
    values=F.grid_sample(torch.from_numpy(latent).expand(len(grid),-1,-1,-1),grid,
                         mode='bilinear',padding_mode='zeros',align_corners=False)
    return (values.numpy().reshape(len(grid),16384)>0),support


def select_pose(correct_counts,total_counts,groups):
    """Only the designated whole-message groups contribute to alignment."""
    n=total_counts[:,groups].sum(1)
    correlation=(2*correct_counts[:,groups].sum(1)-n)/np.sqrt(np.maximum(n,1))
    return int(np.argmax(correlation))


def test_pose(correct_counts,total_counts,index,groups):
    correct=correct_counts[index,groups];total=total_counts[index,groups]
    decisive=(total>0)&(2*correct!=total)
    successes=int(((2*correct>total)&decisive).sum());n=int(decisive.sum())
    probability=float(binom.sf(successes-1,n,.5)) if n else 1.
    return dict(n_decisive=n,successes=successes,unknown_or_tied=len(groups)-n,p=probability,
                conservative_agreement=successes/len(groups))


def verify(signs,support,total_counts,expected_signs,folds,poses):
    matches=(signs==np.asarray(expected_signs,dtype=bool)[None])&support
    counts=matches.reshape(len(poses),64,256).sum(1)
    all_groups=np.arange(256)
    identity=test_pose(counts,total_counts,0,all_groups)
    all_index=select_pose(counts,total_counts,all_groups)
    joint=test_pose(counts,total_counts,all_index,all_groups)
    joint_corrected=min(1.,len(poses)*joint['p'])
    held=[]
    for train,test in [(folds[0],folds[1]),(folds[1],folds[0])]:
        index=select_pose(counts,total_counts,train)
        held.append(dict(pose_index=index,pose=poses[index],**test_pose(counts,total_counts,index,test)))
    cross_p=min(1.,2*min(x['p'] for x in held))
    return dict(identity=identity,
        all_fit=dict(pose_index=all_index,pose=poses[all_index],**joint,p_adjusted=joint_corrected,
                     accepted=joint_corrected<=ALPHA),
        split_fit=dict(folds=held,p_adjusted=cross_p,accepted=cross_p<=ALPHA),
        identity_accepted=identity['p']<=ALPHA)
