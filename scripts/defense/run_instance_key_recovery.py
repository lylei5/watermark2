"""Execute real128-bit recovery, native-key unwrapping and paired controls on249."""
import argparse
import ctypes
import hashlib
import hmac
import json
import os
from pathlib import Path
import select
import sys
import time

ROOT = Path('/data0/liyanlei/watermark_research')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def execute(run, repetition, parent_pid):
    plan=json.loads((run/'protocol.json').read_text())
    execution=json.loads((run/'execution_protocol.json').read_text())
    assert sha(__file__)==execution['runner_sha256']
    assert sha(run/'instance_coset_recovery.py')==plan['source_sha256']['instance_coset_recovery.py']
    out=run/f'repetition_{repetition}'
    out.mkdir(exist_ok=False)
    state=dict(status='waiting_for_features',pid=os.getpid(),parent_pid=parent_pid,
        protocol_sha256=sha(run/'protocol.json'),execution_protocol_sha256=sha(run/'execution_protocol.json'))
    write(out/'state.json',state)
    try:
        if not (run/'features_complete.json').exists():
            libc=ctypes.CDLL(None,use_errno=True)
            fd=libc.syscall(434,parent_pid,0)
            if fd<0:
                if ctypes.get_errno()!=3:raise OSError(ctypes.get_errno(),'pidfd_open')
            else:
                try:
                    cmd=Path(f'/proc/{parent_pid}/cmdline').read_bytes()
                    assert not cmd or str(run).encode() in cmd
                    select.select([fd],[],[])
                finally:os.close(fd)
        complete=json.loads((run/'features_complete.json').read_text())
        assert complete['status']=='complete' and complete['protocol_sha256']==sha(run/'protocol.json')
        for name,digest in complete['artifacts'].items():assert sha(run/name)==digest
        os.environ.update(CUDA_VISIBLE_DEVICES='',NUMBA_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        import numpy as np
        import galois
        from instance_coset_recovery import InstanceRecovery,canonical
        private=Path(plan['private'])
        assert sha(private/'secrets.json')==plan['secret_sha256']
        secrets=json.loads((private/'secrets.json').read_text())
        provider=bytes.fromhex(secrets['provider'])
        private_out=private/f'repetition_{repetition}'
        private_out.mkdir(mode=0o700,exist_ok=False)
        rng=np.random.default_rng(int(secrets['projection_seeds'][repetition],16))
        projection=rng.standard_normal((512,8192),dtype=np.float32)
        projection/=np.linalg.norm(projection,axis=0,keepdims=True)
        np.save(private_out/'projection.npy',projection)
        codec=InstanceRecovery(projection)
        # Exercise actual decoder at its guaranteed radius before image queries.
        unit=[]
        unit_rng=np.random.default_rng(2026111341)
        for errors in [0,1,63,127]:
            key=unit_rng.bytes(16)
            word=codec.codeword(key)
            observed=word.copy()
            observed[unit_rng.choice(1023,errors,replace=False)]^=1
            start=time.monotonic()
            decoded, count=codec.bch.decode(galois.GF2(observed),errors=True)
            recovered=np.packbits(np.asarray(decoded[5:],dtype=np.uint8)).tobytes()
            assert recovered==key and int(count)==errors
            unit.append(dict(errors=errors,exact_key=True,seconds=time.monotonic()-start))
        write(out/'codec_checks.json',unit)
        data=np.load(run/'features.npz')['features']
        metadata=json.loads((run/'feature_rows.json').read_text())
        assert data.shape==(2110,512)
        index={(r['id'],r['kind']):i for i,r in enumerate(metadata)}
        pgid_root=ROOT/'runs/defense_revalidation_20261111/PGID'
        sys.path.insert(0,str(pgid_root))
        from watermarks.keys import KEYS
        from watermarks.nonces import NONCES
        from watermarks.messages import MESSAGES
        methods=[('clean_margin','fixed'),('orbit_margin','fixed'),('orbit_margin','templates'),('orbit_margin','coset')]
        rows,controls,enrollment=[],[],[]
        state.update(status='running',started_at=time.time())
        write(out/'state.json',state)
        for source in plan['sources']:
            identity=source['id']
            reference=data[index[identity,'clean']]
            view_meta=[r for r in metadata if r['id']==identity and r['kind'].startswith('enroll_')]
            views=np.stack([data[index[identity,r['kind']]] for r in view_meta])
            families=[r['family'] for r in view_meta]
            if source['cohort']=='spatial_GS':
                assert sha(source['identity'])==source['identity_sha256']
                with np.load(source['identity']) as identity_file:
                    material=(b'GS01'+np.packbits(identity_file['key'].reshape(-1)).tobytes()
                              +np.packbits(identity_file['message'].reshape(-1)).tobytes())
                assert len(material)==2084
            else:
                task=source['task']
                material=canonical(dict(carrier='PGID_GS',key=KEYS[task].hex(),nonce=NONCES[task].hex(),message=MESSAGES[task][:32].hex()))
            binding_key=hmac.new(provider,f'binding/{repetition}/{identity}'.encode(),hashlib.sha256).digest()[:16]
            records={}
            for selector,mode in methods:
                method=selector+'_'+mode
                tick=time.monotonic()
                record,key=codec.enroll(identity,reference,views,families,material,provider,selector,mode,binding_key)
                assert key==binding_key
                records[method]=record
                path=private_out/f'{identity}_{method}.json'
                write(path,record)
                enrollment.append(dict(id=identity,method=method,seconds=time.monotonic()-tick,
                    record_sha256=sha(path),json_record_bytes=len(canonical(record)),native_material_bytes=len(material),
                    compact_helper_bits=8192+1023*(1+len(record['metadata']['edit_basis']))+256+256,
                    distinct_offsets=sum(1 for _ in codec.offsets(record['metadata']))))
            templates=np.concatenate([reference[None,:],views])
            quantized=np.round(templates/np.max(np.abs(templates),axis=1,keepdims=True)*127).astype(np.int8).astype(np.float32)
            quantized/=np.linalg.norm(quantized,axis=1,keepdims=True)
            queries=[r for r in metadata if r['id']==identity and not r['kind'].startswith('enroll_')]
            for meta in queries:
                query=data[index[identity,meta['kind']]]
                controls.append(dict(**meta,sscd_single=float(np.clip(1-reference@query,0,2)),
                    sscd_templates=float(np.clip(1-np.max(templates@query),0,2)),
                    sscd_int8_templates=float(np.clip(1-np.max(quantized@query),0,2))))
                for method,record in records.items():
                    tick=time.monotonic()
                    result,detail=codec.recover(record,query,provider)
                    elapsed=time.monotonic()-tick
                    success=result is not None
                    if success:
                        assert result[0]==binding_key and result[1]==material, 'Inexact key or verifier material'
                    # Reference-informed control is evaluated only after actual decoding.
                    distance=codec.recovery_distance(record,query,reference)
                    assert success==(distance<=127), 'Key recovery disagrees with exact bounded-distance region'
                    assert not success or detail['residual']==distance
                    rows.append(dict(**meta,method=method,key_recovered=success,exact_key=success,
                        exact_native_material=success,region_distance=distance,decode_seconds=elapsed,
                        attempts=detail['attempts'],joint_native=(bool(success and meta['native']) if meta['native'] is not None else None)))
            print('KEY_RECOVERY_SOURCE',repetition,identity,flush=True)
        assert len(rows)==4920 and len(controls)==1230 and len(enrollment)==440
        write(out/'rows.json',rows)
        write(out/'controls.json',controls)
        write(out/'enrollment.json',enrollment)
        state.update(status='complete',rows=len(rows),controls=len(controls),enrollments=len(enrollment),
            exact_region_agreements=len(rows),seconds=time.time()-state['started_at'],
            artifacts={n:sha(out/n) for n in ['rows.json','controls.json','enrollment.json','codec_checks.json']},
            private_projection_sha256=sha(private_out/'projection.npy'))
        write(out/'complete.json',state)
        print(json.dumps(state),flush=True)
    except Exception as exc:
        state.update(status='failed',error=repr(exc))
        raise
    finally:write(out/'state.json',state)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--repetition',type=int,choices=[0,1,2],required=True)
    p.add_argument('--parent-pid',type=int,required=True)
    a=p.parse_args()
    run=a.run.resolve()
    assert ROOT.is_dir() and run.parent==ROOT/'runs'
    execute(run,a.repetition,a.parent_pid)


if __name__=='__main__':main()
