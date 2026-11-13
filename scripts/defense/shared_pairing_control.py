"""Actual recovery with pairing shuffled strictly within fit/calibration/evaluation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import sys

from shared_channel_experiment import ROOT,NATIVE,read,write,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    run=a.run.resolve();assert run.parent==ROOT/'runs'and ROOT.is_dir()
    plan=read(run/'protocol.json');protocol=read(run/'pairing_control_protocol.json')
    assert sha(__file__)==protocol['source_sha256']and sha(run/'protocol.json')==protocol['parent_protocol_sha256']
    assert not(run/'pairing_control_complete.json').exists()
    done=read(run/'complete.json');assert done['status']=='complete'
    for name,digest in done['artifacts'].items():assert sha(run/name)==digest
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMBA_NUM_THREADS='4')
    import numpy as np
    import torch
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from instance_coset_recovery import InstanceRecovery,canonical
    from shared_channel_features import describe,unit
    sys.path.insert(0,str(NATIVE/'PGID'))
    from watermarks.gs_watermark import GSWatermark
    meta=read(run/'feature_rows.json');data=np.load(run/'features.npz')
    index={(r['id'],r['kind']):i for i,r in enumerate(meta)}
    groups=[plan[k]for k in ['fit_sources','calibration_sources','evaluation_sources']]
    pairing={task:group[(j+1)%len(group)]for group in groups for j,task in enumerate(group)}
    assert pairing=={int(k):v for k,v in protocol['pairing'].items()}
    raw={}
    for i,r in enumerate(meta):
        if r['carrier']!='GS':continue
        j=index[f"G{pairing[r['task']]:03d}",r['kind']]
        raw[i]=describe(data['content'][i],data['noise'][j])['shared_phase']
    fit=[i for i in raw if meta[i]['task']in plan['fit_sources']and
         (meta[i]['kind']=='clean'or meta[i]['kind'].startswith('enroll_'))]
    assert len(fit)==36
    center=np.stack([raw[i]for i in fit]).mean(axis=0)
    features={i:unit(v-center).astype(np.float32)for i,v in raw.items()}
    private=Path(plan['private']);out=private/'within_split_pairing';out.mkdir(exist_ok=False)
    provider=(private/'provider.bin').read_bytes();assert sha(private/'provider.bin')==plan['provider_sha256']
    codecs=[InstanceRecovery(np.load(private/f'projection_{rep}.npy'))for rep in range(3)]
    controls=[r for r in read(run/'controls.json')if r['carrier']=='GS']
    rows=[];continuous=[];enrollment=[];calls=0
    for task in range(10):
        identity=f'G{task:03d}';ref=index[identity,'clean']
        original=read(private/f'{identity}_0_sscd.json')
        recovered,_=codecs[0].recover(original,data['sscd'][ref],provider);assert recovered is not None
        payload=recovered[1];blob=(private/f'{identity}_native.bin').read_bytes();fields=json.loads(payload)
        assert hashlib.sha256(blob).hexdigest()==fields['blob_sha256']
        material=AESGCM(bytes.fromhex(fields['data_key'])).decrypt(blob[:12],blob[12:],canonical(dict(id=identity,carrier='GS')))
        records=[]
        for rep,codec in enumerate(codecs):
            key=secrets.token_bytes(16)
            record,_=codec.enroll(identity,features[ref],np.empty((0,512),dtype=np.float32),[],payload,provider,'clean_margin','fixed',key)
            path=out/f'{identity}_{rep}.json';write(path,record)
            enrollment.append(dict(id=identity,repetition=rep,sha256=sha(path)))
            records.append((record,key))
        for original_row in [r for r in controls if r['id']==identity]:
            r={k:v for k,v in original_row.items()if k in ['id','carrier','task','kind','query_id','split','pixels_sha256','native_metric','native_accepted']}
            kind='clean'if r['kind']=='wrong_record'else r['kind'];i=index[r['query_id'],kind]
            continuous.append(dict(**r,distance=float(np.clip(1-features[ref]@features[i],0,2))))
            verified=False
            for rep,(record,key)in enumerate(records):
                codec=codecs[rep];recovered,_=codec.recover(record,features[i],provider)
                distance=codec.recovery_distance(record,features[i],features[ref]);assert(recovered is not None)==(distance<=127)
                if recovered is not None:
                    assert recovered==(key,payload)
                    f=json.loads(recovered[1]);assert hashlib.sha256(blob).hexdigest()==f['blob_sha256']
                    restored=AESGCM(bytes.fromhex(f['data_key'])).decrypt(blob[:12],blob[12:],canonical(dict(id=identity,carrier='GS')))
                    assert restored==material
                    if not verified:
                        values=json.loads(restored);wm=GSWatermark(latent_shape=(1,4,64,64),device='cpu',key=bytes.fromhex(values['key']),
                            nonce=bytes.fromhex(values['nonce']),message=bytes.fromhex(values['message']))
                        metric=float(wm.get_accuracies(torch.from_numpy(data['noise'][i].reshape(1,4,64,64)))['bit_accuracies'][0])
                        assert metric==r['native_metric'];calls+=1;verified=True
                rows.append(dict(**r,repetition=rep,key_recovered=recovered is not None,region_distance=distance,
                                 joint_accepted=bool(recovered is not None and r['native_accepted'])))
    assert len(rows)==690 and len(continuous)==230
    def genuine(r):return r['kind']in ['clean',*plan['conditions']]
    def summary(part,accept):
        answer=[]
        for split in ['fit','calibration','evaluation']:
            for kind in ['all_genuine','unknown_edits','same_prompt','wrong_record','imprint_150']:
                group=[r for r in part if r['split']==split and(genuine(r)if kind=='all_genuine'else
                    r['kind']in plan['conditions']and r['kind']not in plan['enrollment_families']if kind=='unknown_edits'else r['kind']==kind)]
                answer.append(dict(split=split,kind=kind,total=len(group),accepted=sum(bool(accept(r))for r in group),
                                   joint=sum(bool(accept(r)and r['native_accepted'])for r in group)))
        return answer
    scores=[]
    for rep in range(3):
        part=[r for r in rows if r['repetition']==rep]
        for target in [None,*plan['calibration_targets']]:
            values=sorted(r['region_distance']if r['key_recovered']and r['native_accepted']else float('inf')
                for r in part if r['split']=='calibration'and genuine(r));assert len(values)==30
            threshold=127 if target is None else values[math.ceil(target*30)-1]
            scores.append(dict(repetition=rep,target=target,threshold=threshold if math.isfinite(threshold)else None,
                feasible=math.isfinite(threshold),summary=summary(part,lambda r:r['key_recovered']and r['region_distance']<=threshold)if math.isfinite(threshold)else None))
    for target in plan['calibration_targets']:
        values=sorted(r['distance']for r in continuous if r['split']=='calibration'and genuine(r));threshold=values[math.ceil(target*30)-1]
        scores.append(dict(repetition=None,target=target,threshold=threshold,feasible=True,summary=summary(continuous,lambda r:r['distance']<=threshold)))
    write(run/'pairing_control_rows.json',rows);write(run/'pairing_control_continuous.json',continuous)
    write(run/'pairing_control_complete.json',dict(status='complete',source_sha256=sha(__file__),
        protocol_sha256=sha(run/'pairing_control_protocol.json'),pairing=pairing,actual_recoveries=len(rows),
        restored_native_calls=calls,enrollment=enrollment,scores=scores,
        artifacts={n:sha(run/n)for n in ['pairing_control_rows.json','pairing_control_continuous.json']},
        scope='Within-split mechanism control; global-shuffle arm retained separately'))


if __name__=='__main__':main()
