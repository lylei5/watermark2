"""Frozen transported capsule experiment on existing published outputs on249."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import sys
import time
from types import SimpleNamespace

ROOT = Path('/data0/liyanlei/watermark_research')
PARENT = ROOT / 'runs/shared_channel_binding_20261114_v2'
BASIC = ROOT / 'runs/author_reprompt_evaluation_20261114_v2'
PLUS = ROOT / 'runs/reprompt_plus_evaluation_20261115_v2'
CONTROLS = ROOT / 'runs/reprompt_comparison_20261114'
PLUS_CONTROLS = ROOT / 'runs/reprompt_plus_recovery_20261115'
METHODS = [('identity_a01', .1, 'identity'), ('transport_a01', .1, 'transport'), ('transport_a03', .3, 'transport')]


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, value):
    temp = p.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temp.replace(p)


def imports():
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMBA_NUM_THREADS='4')
    sys.path[:0] = [str(ROOT / 'tools/transport_recovery_dependencies'), str(PARENT), str(ROOT / 'runs/defense_revalidation_20261111/PGID')]
    import numpy as np
    import cv2
    import torch
    cv2.setNumThreads(4); torch.set_num_threads(4)
    return np, cv2, torch


def prepare(run):
    assert run.parent == ROOT / 'runs'; run.mkdir(exist_ok=False)
    private = ROOT / 'private_eval' / run.name; private.mkdir(mode=0o700, exist_ok=False)
    (private / 'provider.bin').write_bytes(secrets.token_bytes(32)); os.chmod(private / 'provider.bin', 0o600)
    for name in ['run_transported_capsule.py', 'transported_key_capsule.py']:
        shutil.copy2(Path(__file__).with_name(name), run / name)
    inputs = [PARENT/'protocol.json', PARENT/'feature_rows.json', PARENT/'enrollment.json', CONTROLS/'content_rows.json',
              BASIC/'queries.json', PLUS/'queries.json', PLUS_CONTROLS/'controls.json']
    plan = dict(created_at=time.time(), hypothesis='Forward-transport helper and key basis; infer edit from uncoded pilots, solve key on disjoint coordinates, certify held-out prediction.',
                private=str(private), provider_sha256=sha(private/'provider.bin'), scripts={p.name:sha(p) for p in run.glob('*.py')},
                inputs={str(p):sha(p) for p in inputs}, methods=METHODS, size=64, key_bits=128,
                code='Real-valued128-dimensional code; no BCH/RS or learned feature encoder', sigmas=[0,.5,1,2], filter_support=13,
                calibration_targets=[.9,.95], calibration='GS tasks4,5,6 genuine only; finite coverage required, freeze before non-calibration observations.',
                primary_cohort='GS tasks7,8,9; TR/PRC10sources each; existing development data.',
                constraints='No attack generation or change; published Imprint/basicReprompt/Reprompt+ outputs reused with exact hashes. CPU249 only.',
                unknown_edit_scope='Model-outside JPEG/noise/median/salt-pepper; existing crop/drop are coordinate-preserving masks; no crop-resize claim.',
                security='Authenticated server-secret binding; no public-helper hiding or entropy guarantee.',
                anticipated_pairs=1010, anticipated_key_decodes=3030, independent_confirmation=False, novelty_established=False)
    write(run/'protocol.json', plan)
    print(json.dumps(dict(run=str(run), protocol_sha256=sha(run/'protocol.json'))), flush=True)


def execute(run):
    np, cv2, torch = imports()
    from PIL import Image
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from instance_coset_recovery import InstanceRecovery, canonical
    from recovered_carrier_transfer import restore_native
    from watermarks.gs_watermark import GSWatermark
    import transported_key_capsule as codec
    plan = read(run/'protocol.json'); state = dict(status='preflight', pid=os.getpid(), started_at=time.time(), protocol_sha256=sha(run/'protocol.json'))
    write(run/'state.json', state)
    try:
        for name,digest in plan['scripts'].items(): assert sha(run/name)==digest
        for name,digest in plan['inputs'].items(): assert sha(name)==digest
        parent=read(PARENT/'protocol.json'); private=Path(plan['private']); provider=(private/'provider.bin').read_bytes(); assert sha(private/'provider.bin')==plan['provider_sha256']
        old_private=Path(parent['private']); old_provider=(old_private/'provider.bin').read_bytes(); assert sha(old_private/'provider.bin')==parent['provider_sha256']
        old_codec=InstanceRecovery(np.load(old_private/'projection_0.npy')); features=np.load(PARENT/'features.npz')['sscd']
        meta=read(PARENT/'feature_rows.json'); index={(r['id'],r['kind']):i for i,r in enumerate(meta)}
        old_enroll={(r['id'],r['method'],r['repetition']):r for r in read(PARENT/'enrollment.json')}
        spec=importlib.util.spec_from_file_location('frozen_distortion',parent['distortion_file']);distort=importlib.util.module_from_spec(spec);spec.loader.exec_module(distort)
        assert sha(parent['distortion_file'])==parent['distortion_sha256']
        (run/'work').mkdir(exist_ok=False);os.chdir(run/'work');cache={}
        def pixels(row):
            digest=row['pixels_sha256']
            if digest not in cache:
                assert sha(row['image'])==row['image_sha256']; im=Image.open(row['image'])
                if row.get('resize'): im=im.resize((512,512))
                im=im.convert('RGB')
                if row.get('family'):
                    options=SimpleNamespace(**{k:None for k in parent['conditions']});setattr(options,row['family'],parent['conditions'][row['family']])
                    distort.set_random_seed(row['seed']);im=distort.image_distortion(im.copy(),row['seed'],options).convert('RGB')
                assert hashlib.sha256(im.tobytes()).hexdigest()==digest;cache[digest]=im
            return cache[digest]
        basic={(r['id'],r['proxy']):r for r in read(BASIC/'queries.json')}
        basic_native={};plus_native={}
        for root,destination,carriers in [(BASIC,basic_native,['GS','TR','PRC']),(PLUS,plus_native,['GS','TR'])]:
            assert read(root/'complete.json')['status']=='complete'
            for c in carriers:
                assert sha(root/c/'rows.json')==read(root/c/'complete.json')['rows_sha256']
                for r in read(root/c/'rows.json'):
                    if not r['defended']:destination[r.get('trial',(r['id'],r['proxy']))]=r
        cases=[]
        for i,r in enumerate(read(CONTROLS/'content_rows.json')):
            if r['kind'].startswith('reprompt_'):
                proxy=r['kind'].removeprefix('reprompt_');q=dict(basic[r['id'],proxy]);n=basic_native[r['id'],proxy]
                q.update(latent=n['latent'],latent_sha256=n['latent_sha256'])
            else:q=meta[index[r['query_id'],'clean' if r['kind']=='wrong_record' else r['kind']]]
            assert r['pixels_sha256']==q['pixels_sha256'];cases.append(dict(base=dict(r,case_id='existing_'+str(i)),query=q))
        pq={r['trial']:r for r in read(PLUS/'queries.json')}
        for r in read(PLUS_CONTROLS/'controls.json'):
            q=dict(pq[r['trial']]);n=plus_native[r['trial']];q.update(latent=n['latent'],latent_sha256=n['latent_sha256'])
            cases.append(dict(base=dict(r,case_id=r['trial'],query_id=r['id'],kind='reprompt_plus_'+r['proxy'],task=pq[r['trial']]['task']),query=q))
        assert len(cases)==1010
        write(run/'cases.json',cases)
        refs={identity:meta[index[identity,'clean']] for identity in sorted({r['base']['id'] for r in cases})}
        records={};secrets_by_id={};materials={};payloads={};enrolled=[];auth_checks=0
        for identity,ref in refs.items():
            hp=old_private/f'{identity}_0_sscd.json';assert sha(hp)==old_enroll[identity,'sscd',0]['helper_sha256']
            result,_=old_codec.recover(read(hp),features[index[identity,'clean']],old_provider);assert result is not None
            payload=result[1];fields=json.loads(payload);blob_path=old_private/f'{identity}_native.bin';blob=blob_path.read_bytes();assert sha(blob_path)==fields['blob_sha256']
            materials[identity]=AESGCM(bytes.fromhex(fields['data_key'])).decrypt(blob[:12],blob[12:],canonical(dict(id=identity,carrier=ref['carrier'])))
            payloads[identity]=payload;key=secrets.token_bytes(16);secrets_by_id[identity]=key
            for alpha in [.1,.3]:
                record,_=codec.enroll(identity,pixels(ref),payload,provider,alpha,key=key);records[identity,alpha]=record
                path=private/f'{identity}_{alpha}.json';write(path,record)
                enrolled.append(dict(id=identity,alpha=alpha,helper_bytes=path.stat().st_size,helper_sha256=sha(path),native_blob_bytes=len(blob)))
            record=records[identity,.1];damaged=json.loads(json.dumps(record));damaged['metadata']['id']+='bad'
            assert codec.observe(damaged,pixels(ref),provider)[0] is None
            assert codec.observe(record,pixels(ref),secrets.token_bytes(32))[0] is None;auth_checks+=2
        write(run/'enrollment.json',enrolled)
        genuine=lambda r:r['kind'] in ['clean',*parent['conditions']]
        calibration=[c for c in cases if c['base']['carrier']=='GS' and c['base']['split']=='calibration' and genuine(c['base'])]
        assert len(calibration)==30
        observations={};rows=[]
        def observe_case(case):
            base=case['base'];identity=base['id'];im=pixels(case['query'])
            for name,alpha,mode in METHODS:
                start=time.perf_counter();key,detail=codec.observe(records[identity,alpha],im,provider,mode)
                if key is not None: assert key==secrets_by_id[identity]
                scores=codec.reference_scores(pixels(refs[identity]),im,detail)
                row=dict(base,method=name,key_recovered=key is not None,detail=detail,seconds=time.perf_counter()-start,**scores)
                row['score']=detail['residual'] if key is not None else None
                observations[base['case_id'],name]=(key,row);rows.append(row)
        for case in calibration:observe_case(case)
        thresholds=[]
        for name,_,_ in METHODS:
            part=[r for r in rows if r['method']==name]
            for score_name in ['score','reference_residual','pixel_residual']:
                finite=sorted(r[score_name] for r in part if r[score_name] is not None)
                for target in [.9,.95]:
                    needed=int(np.ceil(target*30));cutoff=finite[needed-1] if len(finite)>=needed else None
                    thresholds.append(dict(method=name,score_name=score_name,target=target,available=len(finite),total=30,threshold=cutoff,feasible=cutoff is not None))
        write(run/'calibration.json',thresholds);state['status']='evaluating';state['calibration_sha256']=sha(run/'calibration.json');write(run/'state.json',state)
        calibration_ids={c['base']['case_id'] for c in calibration}
        for case in cases:
            if case['base']['case_id'] not in calibration_ids:observe_case(case)
        native_calls=0;payload_returns=0;native_cache={};decisions=[]
        tr_threshold=read(ROOT/'runs/pgid_carrier_coverage_20261113/protocol.json')['thresholds']['TR']['value']
        case_lookup={c['base']['case_id']:c for c in cases}
        for row in rows:
            identity=row['id'];name=row['method'];alpha=next(a for n,a,_ in METHODS if n==name);key=observations[row['case_id'],name][0]
            for target in [.9,.95]:
                point=next(r for r in thresholds if r['method']==name and r['score_name']=='score' and r['target']==target)
                payload=codec.unwrap(records[identity,alpha],key,provider,row['score'],point['threshold']) if key is not None else None
                if payload is not None:
                    assert payload==payloads[identity];payload_returns+=1;ck=(identity,row['pixels_sha256'])
                    if ck not in native_cache:
                        material=materials[identity]
                        if row['carrier']=='GS':
                            f=json.loads(material);wm=GSWatermark(latent_shape=(1,4,64,64),device='cpu',key=bytes.fromhex(f['key']),nonce=bytes.fromhex(f['nonce']),message=bytes.fromhex(f['message']))
                        else:wm=restore_native(material,'cpu')
                        q=case_lookup[row['case_id']]['query'];assert sha(q['latent'])==q['latent_sha256'];z=np.load(q['latent'])['z']
                        accuracy=wm.get_accuracies(torch.from_numpy(z.reshape(1,4,64,64)))
                        accepted=bool(float(accuracy['bit_accuracies'][0])>181/256) if row['carrier']=='GS' else bool(float(accuracy['p_values'][0])<tr_threshold) if row['carrier']=='TR' else bool(accuracy['PRC_detection_results'][0])
                        assert accepted==row['native_accepted'];native_cache[ck]=accepted;native_calls+=1
                decisions.append(dict(case_id=row['case_id'],id=identity,method=name,target=target,feasible=point['feasible'],returned=payload is not None,joint=bool(payload is not None and row['native_accepted'])))
        write(run/'rows.json',rows);write(run/'decisions.json',decisions)
        state.update(status='complete',seconds=time.time()-state['started_at'],key_decode_calls=len(rows),exact_key_returns=sum(r['key_recovered'] for r in rows),
                     thresholded_payload_returns=payload_returns,restored_native_calls=native_calls,authentication_rejections=auth_checks,record_query_pairs=len(cases),
                     distinct_pixels=len(cache),artifacts={n:sha(run/n) for n in ['cases.json','enrollment.json','calibration.json','rows.json','decisions.json']})
        write(run/'complete.json',state);print(json.dumps(state),flush=True)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:write(run/'state.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','execute']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    (prepare if a.action=='prepare' else execute)(a.run)
