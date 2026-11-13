"""Actual key recovery and native GS checks for paired registration and controls."""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import select
import sys
import time

ROOT=Path('/data0/liyanlei/watermark_research')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--parent-pid',type=int,required=True)
    args=parser.parse_args();run=args.run.resolve();assert ROOT.is_dir()and run.parent==ROOT/'runs'
    state=dict(status='waiting_for_registration_exit',pid=os.getpid(),parent_pid=args.parent_pid,source_sha256=sha(__file__))
    assert not(run/'recovery_state.json').exists();write(run/'recovery_state.json',state)
    try:
        execution=json.loads((run/'recovery_protocol.json').read_text())
        for name,digest in execution['sources'].items():assert sha(run/name)==digest
        libc=ctypes.CDLL(None,use_errno=True);fd=libc.syscall(434,args.parent_pid,0)
        if fd<0:
            if ctypes.get_errno()!=3:raise OSError(ctypes.get_errno(),'pidfd_open')
        else:
            try:
                command=Path(f'/proc/{args.parent_pid}/cmdline').read_bytes()
                assert not command or str(run).encode()in command
                select.select([fd],[],[])
            finally:os.close(fd)
        plan=json.loads((run/'protocol.json').read_text());complete=json.loads((run/'complete.json').read_text())
        assert complete['status']=='complete'and complete['protocol_sha256']==sha(run/'protocol.json')
        assert sha(run/'rows.json')==complete['rows_sha256']
        os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMBA_NUM_THREADS='4')
        import numpy as np
        import torch
        from instance_coset_recovery import InstanceRecovery,canonical
        from channel_conditioned_recovery import ChannelConditionedRecovery,channel_readout
        base=Path(plan['base']);channel=Path(plan['channel'])
        assert sha(base/'features.npz')==plan['features_sha256']and sha(base/'feature_rows.json')==plan['feature_rows_sha256']
        assert sha(channel/'rows.json')==plan['channel_rows_sha256']
        meta=json.loads((base/'feature_rows.json').read_text());features=np.load(base/'features.npz')['features']
        index={(r['id'],r['kind']):i for i,r in enumerate(meta)}
        natives=json.loads((channel/'rows.json').read_text())
        queries={}
        for r in natives:
            path=channel/r['latent'];assert sha(path)==r['latent_sha256'];z=np.load(path)['z']
            kind=f"imprint_{r['checkpoint']}"if r['kind']=='imprint'else r['kind'];identity=f"G{r['task']:03d}"
            assert meta[index[identity,kind]]['pixels_sha256']==r['pixels_sha256']
            queries[identity,kind]=(r,z,channel_readout(z))
        registration={}
        for r in json.loads((run/'rows.json').read_text()):
            path=run/r['latent'];assert sha(path)==r['latent_sha256']
            registration[r['id'],r['kind']]=channel_readout(np.load(path)['z'])
        sys.path.insert(0,str(ROOT/'runs/defense_revalidation_20261111/PGID'))
        from watermarks.gs_watermark import GSWatermark
        from watermarks.keys import KEYS
        from watermarks.nonces import NONCES
        from watermarks.messages import MESSAGES
        private=Path(execution['private']);assert sha(private/'provider.bin')==execution['provider_sha256']
        provider=(private/'provider.bin').read_bytes();codecs=[]
        for i in range(3):
            path=private/f'projection_{i}.npy';assert sha(path)==execution['projection_sha256'][i]
            codecs.append(ChannelConditionedRecovery(InstanceRecovery(np.load(path))))
        rows,controls,enrollment=[],[],[];native_calls=0
        state.update(status='running',started_at=time.time());write(run/'recovery_state.json',state)
        for source in plan['sources']:
            identity,task=source['id'],source['task'];reference=features[index[identity,'clean']]
            split='calibration'if task in plan['calibration_sources']else'evaluation'
            vm=[r for r in meta if r['id']==identity and r['kind'].startswith('enroll_')]
            views=np.stack([features[index[identity,r['kind']]]for r in vm]);families=[r['family']for r in vm]
            h0=queries[identity,'clean'][2];hv=np.stack([registration[identity,r['kind']]for r in vm])
            material=canonical(dict(carrier='PGID_GS',key=KEYS[task].hex(),nonce=NONCES[task].hex(),message=MESSAGES[task][:32].hex()))
            native=GSWatermark(latent_shape=(1,4,64,64),device='cpu',key=KEYS[task],nonce=NONCES[task],message=MESSAGES[task][:32])
            helpers={}
            for rep,codec in enumerate(codecs):
                key=secrets.token_bytes(16)
                for method in plan['methods']:
                    record=codec.enroll(identity,reference,views,families,h0,hv,material,provider,method,key)
                    helpers[rep,method]=(record,key)
                    path=private/f'{identity}_{rep}_{method}.json';write(path,record)
                    enrollment.append(dict(id=identity,repetition=rep,method=method,bytes=len(canonical(record)),sha256=sha(path)))
            templates=np.concatenate([reference[None,:],views]);q8=np.round(templates/np.max(np.abs(templates),axis=1,keepdims=True)*127).astype(np.int8).astype(np.float32)
            q8/=np.linalg.norm(q8,axis=1,keepdims=True)
            cases=[(kind,identity,kind)for i,kind in queries if i==identity]
            cases += [(f'wrong_record_{other:03d}',f'G{other:03d}','clean')for other in range(10)if other!=task]
            assert len(cases)==22
            for kind,query_id,query_kind in cases:
                prior,z,h=queries[query_id,query_kind];feature=features[index[query_id,query_kind]]
                if kind.startswith('wrong_record_'):
                    raw_ba=float(native.get_accuracies(torch.from_numpy(z))['bit_accuracies'][0]);native_calls+=1
                else:raw_ba=prior['ba']
                common=dict(id=identity,task=task,kind=kind,query_id=query_id,split=split,native_ba=raw_ba,native_accepted=raw_ba>181/256)
                control=dict(**common,sscd_single=float(np.clip(1-reference@feature,0,2)),
                    sscd_templates=float(np.clip(1-np.max(templates@feature),0,2)),sscd_int8_templates=float(np.clip(1-np.max(q8@feature),0,2)))
                restored_ba=None
                for(rep,method),(record,key)in helpers.items():
                    codec=codecs[rep];tick=time.monotonic();recovered,detail=codec.recover(record,feature,h,provider);elapsed=time.monotonic()-tick
                    distance,continuous=codec.controls(record,feature,h,reference,h0)
                    assert(recovered is not None)==(distance<=127)
                    if recovered is not None:
                        assert recovered==(key,material)
                        if restored_ba is None:
                            fields=json.loads(recovered[1]);wm=GSWatermark(latent_shape=(1,4,64,64),device='cpu',
                                key=bytes.fromhex(fields['key']),nonce=bytes.fromhex(fields['nonce']),message=bytes.fromhex(fields['message']))
                            restored_ba=float(wm.get_accuracies(torch.from_numpy(z))['bit_accuracies'][0]);native_calls+=1
                            assert restored_ba==raw_ba
                    rows.append(dict(**common,repetition=rep,method=method,key_recovered=recovered is not None,
                        region_distance=distance,decode_seconds=elapsed,joint_accepted=bool(recovered is not None and raw_ba>181/256)))
                    if rep==0:control[method+'_cosine']=continuous
                controls.append(control)
            print('PAIRED_RECOVERY_SOURCE',identity,flush=True)
        assert len(rows)==3960 and len(controls)==220 and len(enrollment)==180
        for name,value in [('recovery_rows.json',rows),('controls.json',controls),('enrollment.json',enrollment)]:write(run/name,value)
        conditions=list(plan['conditions'])
        def genuine(r):return r['kind']=='clean'or r['kind']in conditions
        def summary(part,accept):
            result=[]
            for split in ['calibration','evaluation']:
                for kind in ['all_genuine','all_edits','clean','wrong_record','imprint_50','imprint_100','imprint_150',*conditions]:
                    group=[r for r in part if r['split']==split and(genuine(r)if kind=='all_genuine'else r['kind']in conditions if kind=='all_edits'else r['kind'].startswith('wrong_record_')if kind=='wrong_record'else r['kind']==kind)]
                    assert group
                    result.append(dict(split=split,kind=kind,total=len(group),accepted=sum(bool(accept(r))for r in group),joint=sum(bool(accept(r)and r['native_accepted'])for r in group)))
            return result
        scores=[]
        for rep in range(3):
            for method in plan['methods']:
                part=[r for r in rows if r['method']==method and r['repetition']==rep]
                scores.append(dict(repetition=rep,method=method,target=None,feasible=True,threshold=127,summary=summary(part,lambda r:r['key_recovered'])))
                cal=[r for r in part if r['split']=='calibration'and genuine(r)]
                for target in plan['calibration_targets']:
                    values=sorted(r['region_distance']if r['key_recovered']else float('inf')for r in cal)
                    t=values[math.ceil(target*len(values))-1];feasible=math.isfinite(t)
                    scores.append(dict(repetition=rep,method=method,target=target,feasible=feasible,threshold=t if feasible else None,
                        summary=summary(part,lambda r:r['key_recovered']and r['region_distance']<=t)if feasible else None))
        for method in ['sscd_single','sscd_templates','sscd_int8_templates']+[m+'_cosine'for m in plan['methods']]:
            cal=[r[method]for r in controls if r['split']=='calibration'and genuine(r)]
            for target in plan['calibration_targets']:
                t=sorted(cal)[math.ceil(target*len(cal))-1]
                scores.append(dict(repetition=None,method=method,target=target,feasible=True,threshold=t,summary=summary(controls,lambda r:r[method]<=t)))
        write(run/'scores.json',dict(status='complete',scores=scores,native_decoder_calls=native_calls,recovery_decisions=len(rows),
            protocol_sha256=sha(run/'protocol.json'),execution_sha256=sha(run/'recovery_protocol.json'),independent_confirmation=False))
        state.update(status='complete',recovery_decisions=len(rows),native_decoder_calls=native_calls,
            seconds=time.time()-state['started_at'],artifacts={n:sha(run/n)for n in ['recovery_rows.json','controls.json','enrollment.json','scores.json']})
        write(run/'recovery_complete.json',state)
    except Exception as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:write(run/'recovery_state.json',state)


if __name__=='__main__':main()
