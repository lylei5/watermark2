"""Execute spatial witness recovery and native-material verification on249."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import select
import shutil
import sys
import time
from types import SimpleNamespace

ROOT=Path('/data0/liyanlei/watermark_research')
PARENT=ROOT/'runs/shared_channel_binding_20261114_v2'
EVAL=ROOT/'runs/author_reprompt_evaluation_20261114_v2'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)


def prepare(run):
    run.mkdir(exist_ok=False)
    private=ROOT/'private_eval'/run.name;private.mkdir(mode=0o700,exist_ok=False)
    (private/'provider.bin').write_bytes(secrets.token_bytes(32))
    for name in ['run_spatial_witness_recovery.py','spatial_witness_recovery.py']:
        shutil.copy2(Path(__file__).with_name(name),run/name)
    parent=read(PARENT/'protocol.json')
    write(run/'protocol.json',dict(created_at=time.time(),scripts={p.name:sha(p) for p in run.glob('*.py')},
        private=str(private),provider_sha256=sha(private/'provider.bin'),parent_protocol_sha256=sha(PARENT/'protocol.json'),
        parent_rows_sha256=sha(PARENT/'feature_rows.json'),parent_controls_sha256=sha(PARENT/'controls.json'),
        evaluation_protocol_sha256=sha(EVAL/'protocol.json'),evaluation_pid=read(EVAL/'launch.json')['pid'],
        methods=['identity','registered','heldout'],payload_bits=128,rs=[255,16,239],
        witness_selection='255right-region symbols,8reference-margin-selected local rank bits per symbol; Gaussian sigma2.',
        registration='Unchanged OpenCV SIFT and RANSAC homography,ratio0.75,threshold3px,2000iterations,8inliers.',
        heldout_scope='Reference anchors whose conservative6*size support ends beforex224; witnesses start atx300. Query descriptors use full image. No statistical independence guarantee.',
        decoding='One RS error-erasure decode; exact key commitment;2errors+erasures<=239; AEAD unwrap of native carrier material.',
        exclusions='Only out-of-view witness symbols erased; no witness-bit fitting, no prompt or attack labels in recovery.',
        calibration='No fitted threshold; prescribed algebraic recovery region. Compare strong controls at same absolute genuine count separately.',
        source_split={k:parent[k] for k in ['fit_sources','calibration_sources','evaluation_sources']},
        scope='Frozen mechanism prototype on existing development data; novelty and security not established.',
        dependency_versions={'opencv-python-headless':'4.10.0.84','reedsolo':'1.7.0'},
        attacks_generated=False,experiment_host='249',device='CPU'))
    print(json.dumps(dict(run=str(run),protocol_sha256=sha(run/'protocol.json'))),flush=True)


def execute(run):
    plan=read(run/'protocol.json')
    state=dict(status='preflight',pid=os.getpid(),protocol_sha256=sha(run/'protocol.json'),started_at=time.time())
    write(run/'state.json',state)
    try:
        for n,h in plan['scripts'].items():assert sha(run/n)==h
        assert sha(PARENT/'protocol.json')==plan['parent_protocol_sha256']
        assert sha(PARENT/'feature_rows.json')==plan['parent_rows_sha256']
        assert sha(PARENT/'controls.json')==plan['parent_controls_sha256']
        os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMBA_NUM_THREADS='4')
        sys.path[:0]=[str(ROOT/'tools/transport_recovery_dependencies'),str(PARENT),str(ROOT/'runs/defense_revalidation_20261111/PGID')]
        import numpy as np
        import torch
        import cv2
        from PIL import Image
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from instance_coset_recovery import InstanceRecovery,canonical
        from recovered_carrier_transfer import restore_native
        from watermarks.gs_watermark import GSWatermark
        import spatial_witness_recovery as codec
        cv2.setNumThreads(4);torch.set_num_threads(4)
        parent=read(PARENT/'protocol.json');private=Path(plan['private']);old_private=Path(parent['private'])
        provider=(private/'provider.bin').read_bytes();assert sha(private/'provider.bin')==plan['provider_sha256']
        old_provider=(old_private/'provider.bin').read_bytes();assert sha(old_private/'provider.bin')==parent['provider_sha256']
        old_codec=InstanceRecovery(np.load(old_private/'projection_0.npy'))
        features=np.load(PARENT/'features.npz')['sscd']
        metadata=read(PARENT/'feature_rows.json');index={(r['id'],r['kind']):i for i,r in enumerate(metadata)}
        spec=importlib.util.spec_from_file_location('frozen_benign_distortions',parent['distortion_file'])
        distort=importlib.util.module_from_spec(spec);spec.loader.exec_module(distort)
        assert sha(parent['distortion_file'])==parent['distortion_sha256']
        (run/'work').mkdir();os.chdir(run/'work')
        cache={}
        def pixels(row):
            digest=row['pixels_sha256']
            if digest not in cache:
                assert sha(row['image'])==row['image_sha256']
                im=Image.open(row['image'])
                if row.get('resize'):im=im.resize((512,512))
                im=im.convert('RGB')
                if row.get('family'):
                    opts=SimpleNamespace(**{k:None for k in parent['conditions']});setattr(opts,row['family'],parent['conditions'][row['family']])
                    distort.set_random_seed(row['seed']);im=distort.image_distortion(im.copy(),row['seed'],opts).convert('RGB')
                assert hashlib.sha256(im.tobytes()).hexdigest()==digest
                cache[digest]=(im,codec.anchors(im))
            return cache[digest]
        state['status']='extracting_public_evidence';write(run/'state.json',state)
        for row in metadata:
            if not row['kind'].startswith('enroll_'):pixels(row)
        # A process-exit event gates new Reprompt/native artifacts; no progress polling.
        state['status']='waiting_for_native_exit';write(run/'state.json',state)
        if not (EVAL/'complete.json').exists():
            libc=ctypes.CDLL(None,use_errno=True);fd=libc.syscall(434,plan['evaluation_pid'],0)
            if fd>=0:
                command=Path(f"/proc/{plan['evaluation_pid']}/cmdline").read_bytes();assert str(EVAL).encode() in command
                select.select([fd],[],[]);os.close(fd)
            else:assert ctypes.get_errno()==3
        assert read(EVAL/'complete.json')['status']=='complete'
        assert sha(EVAL/'protocol.json')==plan['evaluation_protocol_sha256']
        new_queries=read(EVAL/'queries.json');new_native={}
        for c in ['GS','TR','PRC']:
            done=read(EVAL/c/'complete.json');assert sha(EVAL/c/'rows.json')==done['rows_sha256']
            for r in read(EVAL/c/'rows.json'):
                if not r['defended']:new_native[r['id'],r['proxy']]=r
        for row in new_queries:pixels(row)
        old_controls=read(PARENT/'controls.json');enrollment=read(PARENT/'enrollment.json')
        expected={(r['id'],r['repetition'],r['method']):r for r in enrollment}
        threshold=read(ROOT/'runs/pgid_carrier_coverage_20261113/protocol.json')['thresholds']['TR']['value']
        rows=[];enrolled=[];restored_calls=0;auth_checks=0
        state['status']='recovering';write(run/'state.json',state)
        for identity in sorted({r['id'] for r in metadata}):
            j=index[identity,'clean'];source=metadata[j];carrier=source['carrier']
            old_record=old_private/f'{identity}_0_sscd.json';assert sha(old_record)==expected[identity,0,'sscd']['helper_sha256']
            result,_=old_codec.recover(read(old_record),features[j],old_provider);assert result is not None
            payload=result[1];fields=json.loads(payload)
            blob_path=old_private/f'{identity}_native.bin';blob=blob_path.read_bytes();assert sha(blob_path)==fields['blob_sha256']
            material=AESGCM(bytes.fromhex(fields['data_key'])).decrypt(blob[:12],blob[12:],canonical(dict(id=identity,carrier=carrier)))
            record,key=codec.enroll(identity,pixels(source)[0],payload,provider)
            helper=private/f'{identity}.json';write(helper,record)
            enrolled.append(dict(id=identity,helper_sha256=sha(helper),helper_bytes=len(codec.canonical(record)),native_blob_bytes=len(blob),native_blob_sha256=sha(blob_path)))
            damaged=json.loads(json.dumps(record));damaged['metadata']['id']+='x'
            assert codec.recover(damaged,pixels(source)[0],pixels(source)[1],provider,'identity')[0] is None;auth_checks+=1
            assert codec.recover(record,pixels(source)[0],pixels(source)[1],secrets.token_bytes(32),'identity')[0] is None;auth_checks+=1
            cases=[]
            for r in old_controls:
                if r['id']!=identity:continue
                query=metadata[index[r['query_id'],'clean' if r['kind']=='wrong_record' else r['kind']]]
                cases.append((dict(r),query))
            for r in new_queries:
                if r['id']==identity:
                    n=new_native[identity,r['proxy']];assert n['pixels_sha256']==r['pixels_sha256']
                    query=dict(r,latent=n['latent'],latent_sha256=n['latent_sha256'])
                    cases.append((dict(id=identity,query_id=identity,carrier=carrier,task=source['task'],split='transfer' if carrier!='GS' else 'fit' if source['task'] in parent['fit_sources'] else 'calibration' if source['task'] in parent['calibration_sources'] else 'evaluation',
                        kind='reprompt_'+r['proxy'],native_accepted=n['accepted'],native_metric=n['metric'],pixels_sha256=r['pixels_sha256']),query))
            for base,query in cases:
                image,query_anchors=pixels(query);native_cache=None
                for mode in plan['methods']:
                    t=time.perf_counter();result,detail=codec.recover(record,image,query_anchors,provider,mode);elapsed=time.perf_counter()-t
                    success=result is not None
                    if success:
                        assert result==(key,payload)
                        if native_cache is None:
                            recovered_fields=json.loads(result[1]);restored_material=AESGCM(bytes.fromhex(recovered_fields['data_key'])).decrypt(blob[:12],blob[12:],canonical(dict(id=identity,carrier=carrier)))
                            assert restored_material==material
                            if carrier=='GS':
                                f=json.loads(restored_material);native=GSWatermark(latent_shape=(1,4,64,64),device='cpu',key=bytes.fromhex(f['key']),nonce=bytes.fromhex(f['nonce']),message=bytes.fromhex(f['message']))
                            else:native=restore_native(restored_material,'cpu')
                            assert sha(query['latent'])==query['latent_sha256']
                            z=np.load(query['latent'])['z'];accuracy=native.get_accuracies(torch.from_numpy(z.reshape(1,4,64,64)))
                            metric=float(accuracy['bit_accuracies'][0]) if carrier=='GS' else float(accuracy['p_values'][0]) if carrier=='TR' else bool(accuracy['PRC_detection_results'][0])
                            native_cache=bool(metric>181/256 if carrier=='GS' else metric<threshold if carrier=='TR' else metric)
                            assert native_cache==base['native_accepted'],(identity,base['kind'],metric,base['native_metric'])
                            restored_calls+=1
                    rows.append(dict(id=identity,carrier=carrier,task=source['task'],query_id=base['query_id'],kind=base['kind'],split=base['split'],
                        pixels_sha256=query['pixels_sha256'],method=mode,key_recovered=success,native_accepted=base['native_accepted'],
                        joint=bool(success and base['native_accepted']),seconds=elapsed,detail=detail))
            write(run/f'{identity}_complete.json',dict(rows=len(rows),restored_calls=restored_calls))
        assert len(rows)==2310 and len(enrolled)==30
        write(run/'rows.json',rows);write(run/'enrollment.json',enrolled)
        summary=[]
        for carrier in ['GS','TR','PRC']:
            for mode in plan['methods']:
                for kind in ['all_genuine','non_enrollment_edits','same_prompt','wrong_record','imprint_150','reprompt_sd21','reprompt_sd15',*parent['conditions']]:
                    part=[r for r in rows if r['carrier']==carrier and r['method']==mode and r['split'] in ['evaluation','transfer'] and
                        (r['kind'] in ['clean',*parent['conditions']] if kind=='all_genuine' else r['kind'] in parent['conditions'] and r['kind'] not in parent['enrollment_families'] if kind=='non_enrollment_edits' else r['kind']==kind)]
                    if part:summary.append(dict(carrier=carrier,method=mode,kind=kind,total=len(part),recovered=sum(r['key_recovered'] for r in part),joint=sum(r['joint'] for r in part)))
        write(run/'summary.json',dict(status='complete',summary=summary,restored_native_calls=restored_calls,authentication_checks=auth_checks,
            exact_key_payload_checks=sum(r['key_recovered'] for r in rows),unique_pixels=len(cache),scope=plan['scope']))
        state.update(status='complete',seconds=time.time()-state['started_at'],rows=len(rows),artifacts={n:sha(run/n) for n in ['rows.json','enrollment.json','summary.json']})
        write(run/'complete.json',state)
        print(json.dumps(read(run/'summary.json')),flush=True)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:write(run/'state.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','execute']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    assert a.run.resolve().parent==ROOT/'runs'
    globals()[a.mode](a.run)
