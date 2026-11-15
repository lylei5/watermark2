"""Recover frozen instance keys on published multi-candidate Reprompt+ outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path('/data0/liyanlei/watermark_research')
ATTACK=ROOT/'runs/author_reprompt_plus_20261115'
EVAL=ROOT/'runs/reprompt_plus_evaluation_20261115_v2'
WITNESS=ROOT/'runs/observable_evidence_recovery_20261115'
PARENT=ROOT/'runs/shared_channel_binding_20261114_v2'
CONTROL=ROOT/'runs/reprompt_comparison_20261114'
METHODS=['coarse','any_scale','all_scales','observable_scales']
CONTROLS=['sscd','sscd_centered','sscd_templates','sscd_int8_templates','copynce_descriptor','copynce_matcher']


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main(out):
    assert out.resolve().parent==ROOT/'runs';out.mkdir(exist_ok=False)
    shutil.copy2(__file__,out/Path(__file__).name)
    state=dict(status='preflight',pid=os.getpid(),started_at=time.time(),source_sha256=sha(__file__))
    write(out/'state.json',state)
    try:
        for run in [EVAL,WITNESS]:assert read(run/'complete.json')['status']=='complete'
        for name,digest in read(WITNESS/'complete.json')['artifacts'].items():assert sha(WITNESS/name)==digest
        wp=read(WITNESS/'protocol.json');parent=read(PARENT/'protocol.json')
        assert sha(PARENT/'protocol.json')==wp['parent_protocol_sha256']
        for name,digest in read(CONTROL/'complete.json')['artifacts'].items():assert sha(CONTROL/name)==digest
        for name,digest in wp['scripts'].items():assert sha(WITNESS/name)==digest
        for name,digest in read(EVAL/'content_complete.json')['artifacts'].items():assert sha(EVAL/name)==digest
        os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMBA_NUM_THREADS='4')
        sys.path[:0]=[str(ROOT/'tools/transport_recovery_dependencies'),str(WITNESS),str(PARENT),str(ROOT/'runs/defense_revalidation_20261111/PGID')]
        import numpy as np
        import torch
        import cv2
        from PIL import Image
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        import observable_evidence_recovery as codec
        from instance_coset_recovery import canonical
        from recovered_carrier_transfer import restore_native
        from watermarks.gs_watermark import GSWatermark
        cv2.setNumThreads(4);torch.set_num_threads(4)
        refs=read(ATTACK/'protocol.json')['references'];queries=read(EVAL/'queries.json')
        private=Path(wp['private']);provider=(private/'provider.bin').read_bytes();assert sha(private/'provider.bin')==wp['provider_sha256']
        enrollment={r['id']:r for r in read(WITNESS/'enrollment.json')}
        threshold=read(ROOT/'runs/pgid_carrier_coverage_20261113/protocol.json')['thresholds']['TR']['value']
        native={};native_rows=[]
        for carrier in ['GS','TR']:
            done=read(EVAL/carrier/'complete.json');assert sha(EVAL/carrier/'rows.json')==done['rows_sha256']
            for r in read(EVAL/carrier/'rows.json'):
                assert sha(r['latent'])==r['latent_sha256'];native[r['trial'],r['defended']]=r;native_rows.append(r)
        assert len(native)==480
        state['status']='recovering';write(out/'state.json',state)
        rows=[];native_calls=0
        for ref in refs:
            identity=ref['id'];helper_path=private/(identity+'.json');assert sha(helper_path)==enrollment[identity]['helper_sha256']
            helper=read(helper_path);meta=helper['metadata']
            assert sha(ref['image'])==ref['image_sha256']
            image=Image.open(ref['image']).convert('RGB')
            # Auditor reconstructs enrollment codeword from the exact reference.
            # Neither this key nor this reference enters the query recover() API.
            layer=meta['bands'][0];pairs=codec.unpack(layer['pairs']).astype(np.int32)
            observed=np.packbits((codec.measurements(codec.bands(image)[0],pairs)>=0).astype(np.uint8),axis=1).reshape(-1)
            key=bytes(codec.CODEC.decode((observed^codec.unpack(layer['offset'])).tobytes())[0])
            assert hashlib.sha256(b'observable/'+identity.encode()+key).hexdigest()==meta['confirmation']
            wrap=codec.hmac.new(provider,b'observable-wrap/'+identity.encode()+key,hashlib.sha256).digest()
            payload=AESGCM(wrap).decrypt(bytes.fromhex(helper['nonce']),__import__('base64').b64decode(helper['wrapped']),codec.canonical(meta))
            fields=json.loads(payload);blob_path=Path(parent['private'])/(identity+'_native.bin');blob=blob_path.read_bytes()
            assert sha(blob_path)==fields['blob_sha256']
            material=AESGCM(bytes.fromhex(fields['data_key'])).decrypt(blob[:12],blob[12:],canonical(dict(id=identity,carrier=ref['carrier'])))
            for q in [r for r in queries if r['id']==identity]:
                assert sha(q['image'])==q['image_sha256'];im=Image.open(q['image']).convert('RGB')
                assert hashlib.sha256(im.tobytes()).hexdigest()==q['pixels_sha256']
                n=native[q['trial'],False];assert n['pixels_sha256']==q['pixels_sha256'];checked=None
                for method in METHODS:
                    start=time.perf_counter();result,detail=codec.recover(helper,im,None,provider,method);elapsed=time.perf_counter()-start
                    if result is not None:
                        assert result==(key,payload)
                        if checked is None:
                            recovered=json.loads(result[1]);restored=AESGCM(bytes.fromhex(recovered['data_key'])).decrypt(blob[:12],blob[12:],canonical(dict(id=identity,carrier=ref['carrier'])))
                            assert restored==material
                            if ref['carrier']=='GS':
                                f=json.loads(restored);wm=GSWatermark(latent_shape=(1,4,64,64),device='cpu',key=bytes.fromhex(f['key']),nonce=bytes.fromhex(f['nonce']),message=bytes.fromhex(f['message']))
                            else:wm=restore_native(restored,'cpu')
                            z=np.load(n['latent'])['z'];accuracy=wm.get_accuracies(torch.from_numpy(z.reshape(1,4,64,64)))
                            metric=float(accuracy['bit_accuracies'][0]) if ref['carrier']=='GS' else float(accuracy['p_values'][0])
                            checked=bool(metric>181/256 if ref['carrier']=='GS' else metric<threshold)
                            assert checked==n['accepted'];native_calls+=1
                    split='transfer' if ref['carrier']!='GS' else 'fit' if ref['task'] in parent['fit_sources'] else 'calibration' if ref['task'] in parent['calibration_sources'] else 'evaluation'
                    rows.append(dict(id=identity,trial=q['trial'],carrier=ref['carrier'],task=ref['task'],proxy=q['proxy'],split=split,
                        pixels_sha256=q['pixels_sha256'],method=method,key_recovered=result is not None,native_accepted=n['accepted'],
                        joint=bool(result is not None and n['accepted']),seconds=elapsed,detail=detail))
        assert len(rows)==960
        # Continuous controls use the same exact query pixels and frozen genuine calibration.
        content=read(EVAL/'content_rows.json');features=np.load(EVAL/'features.npz')['sscd']
        indices={r['id']:i for i,r in enumerate(refs)};center=np.load(PARENT/'centers.npz')['sscd_centered'];templates=np.load(PARENT/'sscd_templates.npz')
        def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
        controls=[]
        for i,r in enumerate(content):
            j=indices[r['id']];q=features[len(refs)+i];reference=features[j];ts=templates[r['id']]
            int8=unit(np.round(ts/np.max(np.abs(ts),axis=1,keepdims=True)*127).astype(np.int8).astype(np.float32))
            n=native[r['trial'],False];assert n['pixels_sha256']==r['pixels_sha256']
            split='transfer' if r['carrier']!='GS' else 'fit' if r['task'] in parent['fit_sources'] else 'calibration' if r['task'] in parent['calibration_sources'] else 'evaluation'
            controls.append(dict(id=r['id'],trial=r['trial'],carrier=r['carrier'],proxy=r['proxy'],split=split,pixels_sha256=r['pixels_sha256'],native_accepted=n['accepted'],
                sscd=r['sscd_distance'],sscd_centered=float(np.clip(1-unit(reference-center)@unit(q-center),0,2)),
                sscd_templates=float(np.clip(1-np.max(ts@q),0,2)),sscd_int8_templates=float(np.clip(1-np.max(int8@q),0,2)),
                copynce_descriptor=r['descriptor_distance'],copynce_matcher=r['matcher_distance']))
        original_controls=read(CONTROL/'content_rows.json');calibration=read(CONTROL/'calibrated.json');wsummary=read(WITNESS/'summary.json')['summary']
        genuine=lambda r:r['kind'] in ['clean',*parent['conditions']]
        aggregate=[];matched=[];native_summary=[]
        for carrier in ['GS','TR']:
            for proxy in ['sd21','sd15']:
                for defended in [False,True]:
                    part=[r for r in native_rows if r['carrier']==carrier and r['proxy']==proxy and r['defended']==defended];ids=sorted({r['id'] for r in part})
                    native_summary.append(dict(carrier=carrier,proxy=proxy,defended=defended,references=len(ids),candidates=len(part),accepted_candidates=sum(r['accepted'] for r in part),
                        accepted_references=sum(any(r['accepted'] for r in part if r['id']==identity) for identity in ids)))
                cc=[r for r in controls if r['carrier']==carrier and r['proxy']==proxy and r['split'] in ['evaluation','transfer']]
                ids=sorted({r['id'] for r in cc});old=[r for r in original_controls if r['carrier']==carrier and r['split'] in ['evaluation','transfer'] and genuine(r)]
                for control in CONTROLS:
                    for target in [.9,.95]:
                        cal=next(r for r in calibration if r['method']==control and r['target']==target and r['carrier']==carrier and r['kind']=='all_genuine')
                        cutoff=cal['threshold']
                        aggregate.append(dict(carrier=carrier,proxy=proxy,method=control,target=target,threshold=cutoff,genuine_joint=cal['joint'],genuine_total=cal['total'],references=len(ids),
                            accepted_references=sum(any(r['native_accepted'] and r[control]<=cutoff for r in cc if r['id']==identity) for identity in ids)))
                for method in METHODS:
                    ww=[r for r in rows if r['carrier']==carrier and r['proxy']==proxy and r['method']==method and r['split'] in ['evaluation','transfer']]
                    assert {r['trial'] for r in ww}=={r['trial'] for r in cc}
                    gs=next(r for r in wsummary if r['carrier']==carrier and r['method']==method and r['kind']=='all_genuine');count=gs['joint']
                    accepted=sum(any(r['joint'] for r in ww if r['id']==identity) for identity in ids)
                    aggregate.append(dict(carrier=carrier,proxy=proxy,method=method,genuine_joint=count,genuine_total=gs['total'],references=len(ids),accepted_references=accepted))
                    for control in CONTROLS:
                        values=sorted(r[control] for r in old if r['native_accepted']);cutoff=values[count-1] if count else -1
                        attained=sum(r['native_accepted'] and r[control]<=cutoff for r in old)
                        matched.append(dict(carrier=carrier,proxy=proxy,method=method,control=control,candidate_genuine=count,control_genuine=attained,genuine_total=gs['total'],exact_match=count==attained,cutoff=cutoff,references=len(ids),candidate_joint=accepted,
                            control_joint=sum(any(r['native_accepted'] and r[control]<=cutoff for r in cc if r['id']==identity) for identity in ids)))
        write(out/'rows.json',rows);write(out/'controls.json',controls);write(out/'aggregate.json',aggregate);write(out/'matched.json',matched);write(out/'native_summary.json',native_summary)
        state.update(status='complete',seconds=time.time()-state['started_at'],recovery_api_decisions=len(rows),exact_successes=sum(r['key_recovered'] for r in rows),restored_native_calls=native_calls,
            same_candidate_checks=len(controls),matched_comparisons=len(matched),scope='Paired development cohort; any-candidate watermark acceptance, no prompt-success annotation; full paper-scale matrix incomplete.',
            attack_protocol_sha256=sha(ATTACK/'protocol.json'),evaluation_complete_sha256=sha(EVAL/'complete.json'),witness_complete_sha256=sha(WITNESS/'complete.json'),
            artifacts={n:sha(out/n) for n in ['rows.json','controls.json','aggregate.json','matched.json','native_summary.json']})
        write(out/'complete.json',state);print(json.dumps(dict(native=native_summary,recovery_api_decisions=len(rows),exact_successes=state['exact_successes'],native_calls=native_calls)),flush=True)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:write(out/'state.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
