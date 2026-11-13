"""Complete spectral binding: actual recovery, authenticated native state and strong controls."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import sys
import time

from shared_channel_experiment import ROOT, BASE, PAIRED, TRANSFER, COVERAGE, NATIVE, read, write, sha


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); a=p.parse_args()
    run=a.run.resolve(); assert ROOT.is_dir() and run.parent == ROOT/'runs'
    plan=read(run/'protocol.json')
    assert sha(__file__) == plan['scripts'][Path(__file__).name]
    state=dict(status='preflight',pid=os.getpid(),protocol_sha256=sha(run/'protocol.json'))
    assert not (run/'recovery_state.json').exists(); write(run/'recovery_state.json',state)
    try:
        done=read(run/'extract_complete.json'); assert done['status']=='complete'
        assert done['protocol_sha256']==plan.get('extraction_protocol_sha256',sha(run/'protocol.json'))
        for name,digest in done['artifacts'].items(): assert sha(run/name)==digest
        for name,digest in plan['scripts'].items(): assert sha(run/name)==digest
        os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMBA_NUM_THREADS='4')
        import numpy as np
        import torch
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from instance_coset_recovery import InstanceRecovery,canonical
        from shared_channel_features import METHODS,describe,unit,mathematical_checks
        from recovered_carrier_transfer import restore_native
        sys.path.insert(0,str(NATIVE/'PGID'))
        from watermarks.gs_watermark import GSWatermark
        from watermarks.keys import KEYS
        from watermarks.nonces import NONCES
        from watermarks.messages import MESSAGES
        torch.set_num_threads(4)
        mathematical=mathematical_checks()
        meta=read(run/'feature_rows.json')
        with np.load(run/'features.npz') as archive:data={k:archive[k] for k in archive.files}
        with np.load(run/'sscd_templates.npz') as archive:template_data={k:archive[k] for k in archive.files}
        index={(r['id'],r['kind']):i for i,r in enumerate(meta)}
        assert len(index)==len(meta)==520 and plan['methods']==METHODS
        fit=[i for i,r in enumerate(meta) if r['carrier']=='GS' and r['task'] in plan['fit_sources'] and
             (r['kind']=='clean' or r['kind'].startswith('enroll_'))]
        assert len(fit)==36
        raw={m:[] for m in METHODS}
        for i,row in enumerate(meta):
            values=describe(data['content'][i],data['noise'][i])
            other=f"G{(row['task']+1)%10:03d}" if row['carrier']=='GS' else f"{row['carrier']}_{(row['task']+1)%10:03d}"
            j=index[other,row['kind']]
            values['permuted_phase']=describe(data['content'][i],data['noise'][j])['shared_phase']
            values['sscd']=values['sscd_centered']=data['sscd'][i]
            for m in METHODS: raw[m].append(values[m])
        raw={m:np.asarray(v,dtype=np.float32) for m,v in raw.items()}
        centers={m:(np.zeros(512,dtype=np.float32) if m=='sscd' else raw[m][fit].mean(axis=0)) for m in METHODS}
        features={m:unit(v-centers[m]).astype(np.float32) for m,v in raw.items()}
        np.savez_compressed(run/'descriptors.npz',**features)
        np.savez_compressed(run/'centers.npz',**centers)
        # Same-pixel SSCD replay against both completed parent cohorts.
        old_meta=read(BASE/'feature_rows.json'); old_features=np.load(BASE/'features.npz')['features']
        old_index={(r['id'],r['kind']):i for i,r in enumerate(old_meta)}
        parity=[]
        for i,r in enumerate(meta):
            if r['carrier']=='GS' and (r['id'],r['kind']) in old_index:
                j=old_index[r['id'],r['kind']]
                assert old_meta[j]['pixels_sha256']==r['pixels_sha256']
                error=float(np.max(np.abs(data['sscd'][i]-old_features[j]))); assert error<1e-5,error; parity.append(error)
        prior_controls={(r['id'],r['kind']):r for r in read(TRANSFER/'controls.json')}
        for i,r in enumerate(meta):
            if (r['id'],r['kind']) in prior_controls:
                previous=prior_controls[r['id'],r['kind']]; j=index[r['id'],'clean']
                assert previous['pixels_sha256']==r['pixels_sha256']
                value=float(np.clip(1-data['sscd'][j]@data['sscd'][i],0,2))
                error=abs(value-previous['sscd_single']); assert error<1e-5,error; parity.append(error)
        private=Path(plan['private']); provider=(private/'provider.bin').read_bytes()
        assert sha(private/'provider.bin')==plan['provider_sha256']
        codecs=[]
        for rep in range(3):
            path=private/f'projection_{rep}.npy'; assert sha(path)==plan['projection_sha256'][rep]
            codecs.append(InstanceRecovery(np.load(path)))
        old_plan=read(TRANSFER/'protocol.json'); old_private=Path(old_plan['private'])
        assert sha(old_private/'provider.bin')==old_plan['provider_sha256']
        old_provider=(old_private/'provider.bin').read_bytes()
        assert sha(old_private/'projection_0.npy')==old_plan['projection_sha256'][0]
        old_codec=InstanceRecovery(np.load(old_private/'projection_0.npy'))
        old_enrollment=read(TRANSFER/'enrollment.json')
        def make_native(carrier,material):
            if carrier!='GS': return restore_native(material,'cpu')
            f=json.loads(material)
            return GSWatermark(latent_shape=(1,4,64,64),device='cpu',key=bytes.fromhex(f['key']),
                               nonce=bytes.fromhex(f['nonce']),message=bytes.fromhex(f['message']))
        threshold=read(COVERAGE/'protocol.json')['thresholds']['TR']['value']
        def detect(wm,carrier,z):
            result=wm.get_accuracies(torch.from_numpy(z.reshape(1,4,64,64)).to('cpu'))
            if carrier=='GS': metric=float(result['bit_accuracies'][0]); accepted=metric>181/256
            elif carrier=='TR': metric=float(result['p_values'][0]); accepted=metric<threshold
            else: metric=bool(result['PRC_detection_results'][0]); accepted=metric
            return metric,bool(accepted)
        def unwrap(payload,blob,identity,carrier):
            fields=json.loads(payload)
            assert fields['blob_sha256']==hashlib.sha256(blob).hexdigest()
            return AESGCM(bytes.fromhex(fields['data_key'])).decrypt(blob[:12],blob[12:],canonical(dict(id=identity,carrier=carrier)))
        rows,controls,enrollment=[],[],[]; native_calls=0; restored_calls=0; auth_checks=0; metric_differences=[]
        state.update(status='running',started_at=time.time()); write(run/'recovery_state.json',state)
        identities=sorted({r['id'] for r in meta})
        for identity in identities:
            ref=index[identity,'clean']; source=meta[ref]; carrier,task=source['carrier'],source['task']
            split=('fit' if task in plan['fit_sources'] else 'calibration' if task in plan['calibration_sources'] else 'evaluation') if carrier=='GS' else 'transfer'
            if carrier=='GS':
                material=canonical(dict(key=KEYS[task].hex(),nonce=NONCES[task].hex(),message=MESSAGES[task][:32].hex()))
            else:
                old_path=old_private/f'{identity}_0_orbit_margin_fixed.json'
                entry=next(x for x in old_enrollment if x['id']==identity and x['repetition']==0 and x['method']=='orbit_margin_fixed')
                assert sha(old_path)==entry['record_sha256']
                recovered,_=old_codec.recover(read(old_path),data['sscd'][ref],old_provider)
                assert recovered is not None
                material=recovered[1]
            native=make_native(carrier,material)
            data_key=secrets.token_bytes(32); nonce=secrets.token_bytes(12)
            blob=nonce+AESGCM(data_key).encrypt(nonce,material,canonical(dict(id=identity,carrier=carrier)))
            (private/f'{identity}_native.bin').write_bytes(blob)
            payload=canonical(dict(data_key=data_key.hex(),blob_sha256=hashlib.sha256(blob).hexdigest()))
            assert unwrap(payload,blob,identity,carrier)==material
            records={}
            for rep,codec in enumerate(codecs):
                key=secrets.token_bytes(16)
                for method in METHODS:
                    record,_=codec.enroll(identity,features[method][ref],np.empty((0,512),dtype=np.float32),[],payload,provider,
                                          selector='clean_margin',mode='fixed',key=key)
                    path=private/f'{identity}_{rep}_{method}.json'; write(path,record)
                    records[rep,method]=(record,key)
                    enrollment.append(dict(id=identity,carrier=carrier,repetition=rep,method=method,
                        helper_bytes=len(canonical(record)),helper_sha256=sha(path),native_blob_bytes=len(blob),native_blob_sha256=sha(private/f'{identity}_native.bin')))
            test=copy.deepcopy(records[0,METHODS[0]][0]);test['metadata']['id']+='x'
            assert codecs[0].recover(test,features[METHODS[0]][ref],provider)[0] is None;auth_checks+=1
            damaged=blob[:-1]+bytes([blob[-1]^1])
            try:
                unwrap(payload,damaged,identity,carrier)
                raise RuntimeError('Unauthenticated native blob accepted')
            except AssertionError:auth_checks+=1
            cases=[(i,r['kind']) for i,r in enumerate(meta) if r['id']==identity and not r['kind'].startswith('enroll_')]
            cases += [(index[other,'clean'],'wrong_record') for other in identities if meta[index[other,'clean']]['carrier']==carrier and other!=identity]
            templates=template_data[identity]
            quantized=np.round(templates/np.max(np.abs(templates),axis=1,keepdims=True)*127).astype(np.int8).astype(np.float32)
            quantized=unit(quantized)
            for i,kind in cases:
                query=meta[i]; metric,accepted=detect(native,carrier,data['noise'][i]);native_calls+=1
                if kind!='wrong_record' and query['native_accepted'] is not None:
                    assert accepted==query['native_accepted'],(identity,kind,metric,query['native_metric'])
                    delta=abs(float(metric)-float(query['native_metric']))
                    if delta: metric_differences.append(dict(id=identity,kind=kind,previous=query['native_metric'],cpu=metric,absolute_difference=delta))
                common=dict(id=identity,carrier=carrier,task=task,kind=kind,query_id=query['id'],split=split,
                    pixels_sha256=query['pixels_sha256'],native_metric=metric,native_accepted=accepted)
                control=dict(**common,sscd_templates=float(np.clip(1-np.max(templates@data['sscd'][i]),0,2)),
                    sscd_int8_templates=float(np.clip(1-np.max(quantized@data['sscd'][i]),0,2)))
                for method in METHODS:control[method]=float(np.clip(1-features[method][ref]@features[method][i],0,2))
                restored_metric=None
                for (rep,method),(record,key) in records.items():
                    codec=codecs[rep];tick=time.monotonic()
                    recovered,detail=codec.recover(record,features[method][i],provider)
                    distance=codec.recovery_distance(record,features[method][i],features[method][ref])
                    assert (recovered is not None)==(distance<=127)
                    if recovered is not None:
                        assert recovered==(key,payload)
                        restored_material=unwrap(recovered[1],blob,identity,carrier); assert restored_material==material
                        if restored_metric is None:
                            restored=make_native(carrier,restored_material)
                            restored_metric,restored_accept=detect(restored,carrier,data['noise'][i]);restored_calls+=1
                            assert (restored_metric,restored_accept)==(metric,accepted)
                    rows.append(dict(**common,method=method,repetition=rep,key_recovered=recovered is not None,
                        region_distance=distance,joint_accepted=bool(recovered is not None and accepted),
                        decode_seconds=time.monotonic()-tick,recovered_detector_equal=True if recovered is not None else None))
                controls.append(control)
            write(run/f'partial_{identity}.json',dict(decisions=len(rows),native_calls=native_calls,restored_calls=restored_calls))
            print('SPECTRAL_SOURCE_COMPLETE',identity,flush=True)
        assert len(controls)==710 and len(rows)==17040 and len(enrollment)==720
        for name,value in [('rows.json',rows),('controls.json',controls),('enrollment.json',enrollment)]:write(run/name,value)
        conditions=list(plan['conditions']);known=plan['enrollment_families'];unknown=[k for k in conditions if k not in known]
        def genuine(r):return r['kind'] in ['clean',*conditions]
        groups=[('GS','fit'),('GS','calibration'),('GS','evaluation'),('TR','transfer'),('PRC','transfer')]
        def summary(part,accept):
            output=[]
            for carrier,split in groups:
                for kind in ['all_genuine','all_edits','known_edits','unknown_edits','clean','cover','same_prompt','wrong_record','imprint_50','imprint_100','imprint_150',*conditions]:
                    gg=[r for r in part if r['carrier']==carrier and r['split']==split and
                        (genuine(r) if kind=='all_genuine' else r['kind'] in conditions if kind=='all_edits' else
                         r['kind'] in known if kind=='known_edits' else r['kind'] in unknown if kind=='unknown_edits' else r['kind']==kind)]
                    if gg:output.append(dict(carrier=carrier,split=split,kind=kind,total=len(gg),
                        accepted=sum(bool(accept(r)) for r in gg),joint=sum(bool(accept(r) and r['native_accepted']) for r in gg)))
            return output
        scores=[]
        for rep in range(3):
            for method in METHODS:
                part=[r for r in rows if r['method']==method and r['repetition']==rep]
                scores.append(dict(method=method,repetition=rep,target=None,feasible=True,threshold=127,
                                   summary=summary(part,lambda r:r['key_recovered'])))
                cal=[r for r in part if r['split']=='calibration' and genuine(r)]; assert len(cal)==30
                for target in plan['calibration_targets']:
                    values=sorted(r['region_distance'] if r['key_recovered'] else float('inf') for r in cal)
                    cutoff=values[math.ceil(target*len(cal))-1];feasible=math.isfinite(cutoff)
                    scores.append(dict(method=method,repetition=rep,target=target,feasible=feasible,threshold=cutoff if feasible else None,
                        summary=summary(part,lambda r:r['key_recovered'] and r['region_distance']<=cutoff) if feasible else None))
        for method in [*METHODS,'sscd_templates','sscd_int8_templates']:
            values=sorted(r[method] for r in controls if r['split']=='calibration' and genuine(r));assert len(values)==30
            for target in plan['calibration_targets']:
                cutoff=values[math.ceil(target*len(values))-1]
                scores.append(dict(method=method,repetition=None,target=target,feasible=True,threshold=cutoff,
                    summary=summary(controls,lambda r:r[method]<=cutoff)))
        write(run/'scores.json',dict(status='complete',scores=scores,protocol_sha256=sha(run/'protocol.json'),
            mathematical_common_filter_checks=mathematical,sscd_replay_count=len(parity),sscd_replay_max_error=max(parity),
            native_metric_differences=metric_differences,native_acceptance_disagreements=0,record_authentication_checks=auth_checks,
            original_native_calls=native_calls,restored_native_calls=restored_calls,actual_recovery_decisions=len(rows),
            success_count=sum(r['key_recovered'] for r in rows),independent_confirmation=False))
        state.update(status='complete',seconds=time.time()-state['started_at'],actual_recovery_decisions=len(rows),
            original_native_calls=native_calls,restored_native_calls=restored_calls,
            artifacts={n:sha(run/n) for n in ['rows.json','controls.json','enrollment.json','scores.json','descriptors.npz','centers.npz']})
        write(run/'complete.json',state)
    except Exception as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:write(run/'recovery_state.json',state)


if __name__=='__main__':main()
