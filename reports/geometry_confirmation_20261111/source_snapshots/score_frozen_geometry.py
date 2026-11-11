"""CPU-only scoring of complete published baselines or held-out calibration."""
import argparse
import json
import math
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import time

import numpy as np
import torch

import blind_geometry_verifier as geometry_verifier
from blind_geometry_verifier import ALPHA,geometry_bank,query_observations,verify
from codeword_binding import sembind_mask_bits,bit_accuracy
from sembind_official_audit import carrier,sha,write


def bind_frozen_alpha(frozen,computed):
    """Keep the protocol value exact; only tolerate numerical library roundoff."""
    if not math.isfinite(frozen) or frozen<=0 or not math.isclose(computed,frozen,rel_tol=1e-12,abs_tol=0.):
        raise ValueError('Runtime binomial tail disagrees with the frozen threshold')
    geometry_verifier.ALPHA=frozen
    return frozen


def validate_full_records(rows):
    """Reject partial or duplicated trajectories before any detector scoring."""
    assert len(rows)==144 and len({r['id'] for r in rows})==144
    groups={(r['task'],r['method']) for r in rows}
    assert len(groups)==8 and {m for _,m in groups}=={'sembind','gs'}
    assert all(len({t for t,m in groups if m==method})==4 for method in ['sembind','gs'])
    for task,method in groups:
        group=[r for r in rows if r['task']==task and r['method']==method]
        assert sum(r['kind']=='reference' for r in group)==1
        assert sum(r['kind']=='cover' for r in group)==1
        assert sorted(r['update'] for r in group if r['kind']=='candidate')==list(range(1,152,10))


def summarize(rows,field):
    result=[]
    for label in sorted({str(r[field]) for r in rows}):
        group=[r for r in rows if str(r[field])==label]
        result.append(dict(group=label,n=len(group),**{key:sum(r[key] for r in group) for key in
            ['default_accepted','identity_accepted','identity_penalty_accepted','all_fit_accepted','split_fit_accepted']}))
    return result


def main():
    global ALPHA
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--mode',choices=['full_baseline','calibration','confirmation'],required=True)
    p.add_argument('--identity-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frozen-protocol',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4)
    plan=json.loads(args.frozen_protocol.read_text())
    for name,digest in plan['source_sha256'].items():assert sha(Path(__file__).with_name(name))==digest,name
    if (args.output/'results.json').exists():raise FileExistsError('Completed scores are immutable')
    rows=[]
    if args.mode=='full_baseline':
        complete=json.loads((args.run/'complete.json').read_text())
        assert complete['completed'] and complete['full_update_jobs']==8 and complete['evaluated_images']==144
        assert complete['updates_per_job']==151 and complete['checkpoint_images']==128
        for i in range(3):
            cache=args.run/'evaluation'/f'shard_{i}'
            rows.extend(dict(r,cache=cache) for r in json.loads((cache/'results.json').read_text())['rows'])
        validate_full_records(rows)
        claims=sorted({r['task'] for r in rows})
    else:
        for i in range(3):
            cache=args.run/'cache'/f'shard_{i}';part=json.loads((cache/'complete.json').read_text())
            assert part['completed'] and part['protocol_sha256']==sha(args.frozen_protocol)
            rows.extend(dict(r,cache=cache,kind='genuine') for r in part['rows'] if r['split']==args.mode)
        expected_claims=list(range(16,48)) if args.mode=='calibration' else plan['identity_tasks']
        assert len(rows)==len(plan['rows']) and {r['task'] for r in rows}==set(expected_claims)
        assert {r['id'] for r in rows}=={r['id'] for r in plan['rows']}
        assert len({r['id'] for r in rows})==len(rows)
        claims=expected_claims
    identities={task:carrier(args.identity_root,task) for task in claims}
    poses,grid,support,counts,folds=geometry_bank();assert len(poses)==251
    computed_alpha=ALPHA
    ALPHA=bind_frozen_alpha(plan['nominal_alpha'],computed_alpha)
    output=[];start=time.monotonic()
    for r in rows:
        cache=r['cache'];inv=cache/(r['id']+'.npy');feat=cache/(r['id']+'.npz')
        assert sha(inv)==r['inversion_sha256'] and sha(feat)==r['features_sha256']
        latent=np.load(inv,allow_pickle=False);code=np.load(feat,allow_pickle=False)['sembind_code']
        signs,valid=query_observations(latent,grid,support)
        for claimed in (claims if args.mode!='full_baseline' else [r['task']]):
            _,cipher,message,perm=identities[claimed]
            mask=sembind_mask_bits(code[None],perm)[0] if r['method']!='gs' else np.zeros(16384,dtype=np.uint8)
            stream=cipher^np.tile(message,64)
            ba=float(bit_accuracy((latent.reshape(-1)>0).astype(np.uint8)^mask^stream,message)[0])
            if args.mode=='full_baseline':assert ba==r['ba']
            score=verify(signs,valid,counts,cipher^mask,folds,poses)
            kind=r['kind'] if claimed==r['task'] else 'wrong_identity'
            item=dict(id=r['id'],task=r['task'],claimed_identity=claimed,method=r['method'],kind=kind,
                label=r.get('label',r['kind']),update=r.get('update'),ba=ba,default_accepted=ba>181/256,
                identity_accepted=score['identity_accepted'],identity_penalty_accepted=min(1,251*score['identity']['p'])<=ALPHA,
                all_fit_accepted=score['all_fit']['accepted'],split_fit_accepted=score['split_fit']['accepted'],
                all_fit_p=score['all_fit']['p_adjusted'],split_fit_p=score['split_fit']['p_adjusted'])
            if kind!='wrong_identity':
                item.update(detail=score,inversion_sha256=sha(inv),features_sha256=sha(feat))
                item.update({k:r[k] for k in ['psnr','ssim','ms_ssim','lpips'] if k in r})
            output.append(item)
    summary=summarize(output,'kind')
    result=dict(completed=True,mode=args.mode,rows=output,summary=summary,
        protocol_sha256=sha(args.frozen_protocol),seconds=time.monotonic()-start,
        unique_queries=len(rows),identity_decisions=len(output),new_attacks_run=False,
        independent_confirmation_scored=args.mode=='confirmation',thresholds_changed=False,
        threshold_audit=dict(runtime_computed_alpha=computed_alpha,applied_alpha=ALPHA,
            frozen_alpha=plan['nominal_alpha'],relative_roundoff_tolerance=1e-12),
        scope='Frozen detector evaluation; correlated views and identities; no empirical ultra-low FPR claim')
    if args.mode!='full_baseline':
        result['genuine_by_condition']=summarize([r for r in output if r['kind']=='genuine'],'label')
        wrong=[r for r in output if r['kind']=='wrong_identity']
        result['wrong_identity_query_level']={k:len({r['id'] for r in wrong if r[k]}) for k in
            ['default_accepted','identity_penalty_accepted','all_fit_accepted','split_fit_accepted']}
        if args.mode=='confirmation':
            controls=[]
            for point in plan['supplementary_ba_controls']:
                assert point['operator']=='>=' and point['feasible']
                threshold=point['threshold']
                genuine=[r for r in output if r['kind']=='genuine']
                controls.append(dict(target_absolute_tpr=point['target_absolute_tpr'],threshold=threshold,operator='>=',
                    calibration_tpr=point['calibration_tpr'],genuine_accepted=sum(r['ba']>=threshold for r in genuine),
                    genuine_n=len(genuine),wrong_identity_accepted=sum(r['ba']>=threshold for r in wrong),
                    wrong_identity_n=len(wrong),wrong_identity_queries=len({r['id'] for r in wrong if r['ba']>=threshold}),
                    genuine_by_condition={label:dict(n=sum(r['label']==label for r in genuine),
                        accepted=sum(r['label']==label and r['ba']>=threshold for r in genuine))
                        for label in sorted({r['label'] for r in genuine})}))
            result['supplementary_ba_controls']=controls
    else:
        result['by_method']={method:dict(
            final=summarize([r for r in output if r['method']==method and (r['kind']!='candidate' or r['update']==151)],'kind'),
            trajectory=summarize([r for r in output if r['method']==method and r['kind']=='candidate'],'update'))
            for method in ['sembind','gs']}
    write(args.output/'results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__':main()
