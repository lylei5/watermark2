"""Audit completed paired-channel recovery and its held-out edit residuals on249."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path('/data0/liyanlei/watermark_research')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();assert ROOT.is_dir()and run.parent==ROOT/'runs'
    assert not(run/'audit.json').exists()
    os.environ.update(CUDA_VISIBLE_DEVICES='',NUMBA_NUM_THREADS='4',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    import numpy as np
    sys.path.insert(0,str(run))
    from instance_coset_recovery import InstanceRecovery
    from channel_conditioned_recovery import ChannelConditionedRecovery,channel_readout,correct
    complete=json.loads((run/'recovery_complete.json').read_text())
    plan=json.loads((run/'protocol.json').read_text());execution=json.loads((run/'recovery_protocol.json').read_text())
    assert complete['status']=='complete'
    for name,digest in complete['artifacts'].items():assert sha(run/name)==digest
    for name,digest in execution['sources'].items():assert sha(run/name)==digest
    rows=json.loads((run/'recovery_rows.json').read_text())
    controls=json.loads((run/'controls.json').read_text());enrollment=json.loads((run/'enrollment.json').read_text())
    assert len(rows)==3960 and len(controls)==220 and len(enrollment)==180
    assert len({(r['id'],r['kind'],r['repetition'],r['method'])for r in rows})==3960
    for r in rows:
        assert r['key_recovered']==(r['region_distance']<=127)
        assert r['joint_accepted']==(r['key_recovered']and r['native_accepted'])
    # Baseline parity is per-query, not just a comparison of aggregate counts.
    old=[]
    for rep in range(3):
        folder=ROOT/f'runs/instance_edit_quotient_20261113/repetition_{rep}'
        prior=json.loads((folder/'complete.json').read_text())
        assert sha(folder/'rows.json')==prior['artifacts']['rows.json']
        old.extend(dict(r,repetition=rep)for r in json.loads((folder/'rows.json').read_text()))
    old_index={(r['id'],r['kind'],r['repetition'],r['method']):r for r in old}
    baseline_parity=0
    for r in rows:
        if r['kind'].startswith('wrong_record_')or r['method']not in ['orbit_margin','tangent8']:continue
        method={'orbit_margin':'orbit_margin_fixed','tangent8':'tangent8_margin_fixed'}[r['method']]
        previous=old_index[r['id'],r['kind'],r['repetition'],method]
        assert previous['key_recovered']==r['key_recovered']
        assert previous['region_distance']==r['region_distance']
        baseline_parity+=1
    base=Path(plan['base']);features=np.load(base/'features.npz')['features']
    meta=json.loads((base/'feature_rows.json').read_text());index={(r['id'],r['kind']):i for i,r in enumerate(meta)}
    channel=Path(plan['channel']);native=json.loads((channel/'rows.json').read_text())
    readouts={}
    for r in native:
        kind=f"imprint_{r['checkpoint']}"if r['kind']=='imprint'else r['kind']
        readouts[f"G{r['task']:03d}",kind]=channel_readout(np.load(channel/r['latent'])['z'])
    private=Path(execution['private']);provider=(private/'provider.bin').read_bytes()
    codec=ChannelConditionedRecovery(InstanceRecovery(np.load(private/'projection_0.npy')))
    clean_key_checks=0;authentication_checks=0;residuals=[]
    for source in plan['sources']:
        identity=source['id'];reference=features[index[identity,'clean']]
        for method in plan['methods']:
            record=json.loads((private/f'{identity}_0_{method}.json').read_text())
            recovered,detail=codec.recover(record,reference,readouts[identity,'clean'],provider)
            assert recovered is not None and detail['residual']==0
            clean_key_checks+=1
            if method=='paired_channel':
                altered=copy.deepcopy(record);altered['normalizer']['l1_limit']+=.1
                assert codec.recover(altered,reference,readouts[identity,'clean'],provider)[1]['reason']=='record_authentication'
                assert codec.recover(record,reference,readouts[identity,'clean'],bytes(32))[1]['reason']=='record_authentication'
                authentication_checks+=2
            for kind in plan['conditions']:
                query=features[index[identity,kind]];q=correct(query,readouts[identity,kind],record['normalizer'])
                f=correct(reference,readouts[identity,'clean'],record['normalizer'])
                residuals.append(dict(id=identity,kind=kind,method=method,
                    split='calibration'if source['task']in plan['calibration_sources']else'evaluation',
                    uncorrected_squared=float(np.sum((query-reference)**2)),corrected_squared=float(np.sum((q-f)**2))))
    aggregate=[]
    for split in ['calibration','evaluation']:
        for method in ['paired_channel','permuted_channel','content_self_fit']:
            for condition in ['all_edits',*plan['conditions']]:
                part=[r for r in residuals if r['split']==split and r['method']==method and(condition=='all_edits'or r['kind']==condition)]
                original=sum(r['uncorrected_squared']for r in part);corrected=sum(r['corrected_squared']for r in part)
                aggregate.append(dict(split=split,method=method,condition=condition,total=len(part),
                    residual_energy_ratio=corrected/original,improved=sum(r['corrected_squared']<r['uncorrected_squared']for r in part)))
    result=dict(status='complete',source_sha256=sha(__file__),protocol_sha256=sha(run/'protocol.json'),
        execution_sha256=sha(run/'recovery_protocol.json'),artifact_hashes=complete['artifacts'],
        actual_recovery_decisions=len(rows),exact_region_agreements=len(rows),baseline_query_parity=baseline_parity,
        additional_clean_recovery_checks=clean_key_checks,record_authentication_checks=authentication_checks,
        native_decoder_calls=complete['native_decoder_calls'],residual_analysis_scope='Post-hoc explanation; no method or threshold changes',
        residual_analysis=aggregate,helper_bytes={m:sorted({r['bytes']for r in enrollment if r['method']==m})for m in plan['methods']})
    (run/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k not in ['residual_analysis','artifact_hashes']}),flush=True)


if __name__=='__main__':main()
