"""Evaluate a frozen CPU-only detector on cached development observations."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from blind_geometry_verifier import ALPHA,geometry_bank,query_observations,verify
from sembind_official_audit import carrier,sha
from codeword_binding import sembind_mask_bits


def write(path,data):
    path.write_bytes((json.dumps(data,indent=2)+'\n').encode())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise FileExistsError('Use a new output for a newly frozen run')
    torch.set_num_threads(4)
    poses,grid,support,counts,folds=geometry_bank()
    files=[args.root/'query_manifest_development.json',args.root/'query_manifest_paper20261110.json',args.root/'query_manifest_reprompt20261110.json']
    rows=[]
    for source in files:
        for r in json.loads(source.read_text())['rows']:
            if r['method']=='sembind_scattered':
                assert r['split']=='development'
                rows.append(dict(id=r['id'],task=r['task'],label=r['label'],kind='candidate' if 'reprompt' in r['label'] else 'genuine'))
    assert len(rows)==176 and len({r['id'] for r in rows})==176
    protocol=dict(scope='Exploratory cached development comparison; no new attacks, no confirmation',
        inputs=[dict(path=str(f),sha256=sha(f)) for f in files],genuine_sources=16,genuine_views=160,existing_reprompt_candidates=16,
        wrong_identity='Each genuine query also tested against (source_task+1)%16; repeated views/identities are correlated',
        query_only_inputs=['query inversion','query semantic code','claimed identity'],
        forbidden_inputs=['source auxiliary code','reference inversion','true geometry','candidate labels in verifier'],
        poses=poses,group_split=[x.tolist() for x in folds],group_unit='All 64 repetitions of a GS message bit stay in one fold',
        alignment='Maximize normalized signed coordinate agreement on designated fitting groups',
        test='Majority over visible repetitions; absent and tied message bits provide no evidence',
        all_fit_correction='251 times single-pose binomial probability, clipped at one',
        split_fit_correction='2 times smaller held-out fold probability, clipped at one',
        nominal_alpha=ALPHA,null_assumption='Independent fair decoded bits conditional on fitting evidence; unproven for actual diffusion images or forgeries',
        empirical_fpr_claim=False,original_baseline='Unchanged GS BA > 181/256 from existing scores',
        environment='CPU only, four threads; no GPU allocation',
        source_sha256={f:sha(Path(__file__).with_name(f)) for f in ['blind_geometry_verifier.py','evaluate_blind_geometry.py']})
    write(out/'protocol.json',protocol)
    # Check implementations and inputs against recorded earlier default scores.
    repo=Path(__file__).resolve().parents[2]
    prior_paths=[repo/'reports/content_verifier_20261110/genuine_full_domain.json',
                 repo/'reports/sembind_official_audit_20261110/paper_distortions.json',
                 repo/'reports/content_verifier_20261110/reprompt_full_domain.json']
    previous={r['id']:r for p in prior_paths for r in json.loads(p.read_text())['rows'] if r['method']=='sembind_scattered'}
    identities={task:carrier(args.root,task) for task in range(16)}
    results=[];started=time.monotonic()
    for i,row in enumerate(rows):
        inv=args.root/'inversions'/(row['id']+'.npy');feat=args.root/'query_features'/(row['id']+'.npz')
        assert sha(inv)==previous[row['id']]['inversion_sha256']
        assert sha(feat)==previous[row['id']]['features_sha256']
        code=np.load(feat,allow_pickle=False)['sembind_code']
        signs,valid=query_observations(np.load(inv,allow_pickle=False),grid,support)
        for claimed in [row['task']]+([(row['task']+1)%16] if row['kind']=='genuine' else []):
            _,cipher,_,perm=identities[claimed]
            expected=cipher^sembind_mask_bits(code[None],perm)[0]
            result=verify(signs,valid,counts,expected,folds,poses)
            kind=row['kind'] if claimed==row['task'] else 'wrong_identity'
            results.append(dict(**{k:v for k,v in row.items() if k!='kind'},kind=kind,claimed_identity=claimed,**result,
                original_ba=previous[row['id']]['ba'] if kind!='wrong_identity' else None,
                original_accepted=previous[row['id']]['accepted'] if kind!='wrong_identity' else None,
                inversion_sha256=sha(inv),features_sha256=sha(feat)))
        if (i+1)%16==0:print('QUERIES',i+1,len(rows),'seconds',round(time.monotonic()-started,1),flush=True)
    summary=[]
    for kind in ['genuine','candidate','wrong_identity']:
        group=[r for r in results if r['kind']==kind]
        summary.append(dict(kind=kind,n=len(group),original_accepted=sum(r['original_accepted'] for r in group) if kind!='wrong_identity' else None,
            identity_accepted=sum(r['identity_accepted'] for r in group),
            all_fit_accepted=sum(r['all_fit']['accepted'] for r in group),split_fit_accepted=sum(r['split_fit']['accepted'] for r in group)))
    strata=[]
    for label in sorted({r['label'] for r in results if r['kind']=='genuine'}):
        group=[r for r in results if r['kind']=='genuine' and r['label']==label]
        strata.append(dict(label=label,n=len(group),original_accepted=sum(r['original_accepted'] for r in group),
            all_fit_accepted=sum(r['all_fit']['accepted'] for r in group),split_fit_accepted=sum(r['split_fit']['accepted'] for r in group),
            all_fit_identity_pose=sum(r['all_fit']['pose_index']==0 for r in group)))
    write(out/'results.json',dict(completed=True,seconds=time.monotonic()-started,summary=summary,strata=strata,rows=results,
        protocol_sha256=sha(out/'protocol.json'),gpu_used=False,confirmation_read=False,new_attacks_run=False))
    print(json.dumps(dict(summary=summary,strata=strata),indent=2))


if __name__=='__main__':main()
