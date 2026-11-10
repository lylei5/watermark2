"""Conditional random-template calibration check; no images or attacks created.

Independent fair template signs model an ideal fresh wrong identity. This checks
statistical behavior at observable levels; it does not measure real-key FPR.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from blind_geometry_verifier import ALPHA,geometry_bank,query_observations,verify
from sembind_official_audit import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise FileExistsError(out)
    torch.set_num_threads(4)
    poses,grid,support,counts,folds=geometry_bank()
    cases=[(task,label) for task in range(16) for label in ['clean','crop90']]
    protocol=dict(scope='Conditional Monte Carlo with ideal random templates; not natural-image or real-key empirical FPR',
        sources=16,queries=32,draws_per_query=64,total_draws=2048,seed=202611102,
        null='Fresh independent fair template sign at each latent coordinate; fixed observed query, independent draws',
        semantic_mask='A fixed query mask xor independent fair cipher signs has the same ideal template distribution',
        levels=[.05,.01,.001,ALPHA],grid_size=len(poses),gpu_used=False,new_attacks_run=False,
        verifier_sha256=sha(Path(__file__).with_name('blind_geometry_verifier.py')),script_sha256=sha(__file__),
        cases=[dict(task=t,label=l,input_sha256=sha(args.root/'inversions'/f'{t:03}_sembind_scattered_{l}.npy')) for t,l in cases])
    (out/'protocol.json').write_bytes((json.dumps(protocol,indent=2)+'\n').encode())
    rng=np.random.default_rng(202611102);rows=[];start=time.monotonic()
    for task,label in cases:
        latent=np.load(args.root/'inversions'/f'{task:03}_sembind_scattered_{label}.npy',allow_pickle=False)
        signs,valid=query_observations(latent,grid,support)
        for draw in range(64):
            template=rng.integers(0,2,16384,dtype=np.uint8)
            result=verify(signs,valid,counts,template,folds,poses)
            rows.append(dict(task=task,label=label,draw=draw,identity_p=result['identity']['p'],
                all_fit_uncorrected_p=result['all_fit']['p'],all_fit_corrected_p=result['all_fit']['p_adjusted'],
                split_fit_corrected_p=result['split_fit']['p_adjusted']))
        if len(rows)%256==0:print('NULL_DRAWS',len(rows),2048,round(time.monotonic()-start,1),flush=True)
    summary=[]
    for level in protocol['levels']:
        summary.append(dict(level=level,n=len(rows),**{field:sum(r[field]<=level for r in rows)
            for field in ['identity_p','all_fit_uncorrected_p','all_fit_corrected_p','split_fit_corrected_p']}))
    (out/'results.json').write_bytes((json.dumps(dict(completed=True,summary=summary,rows=rows,
        seconds=time.monotonic()-start,protocol_sha256=sha(out/'protocol.json')),indent=2)+'\n').encode())
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
