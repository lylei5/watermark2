"""Fresh legacy decoder replay with unchanged enrollment, thresholds and pixels."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path('/data0/liyanlei/watermark_research')
RUN=ROOT/'runs/order_capsule_confirmation_20261115'
OLD=ROOT/'runs/capsule_source_transfer_20261115_v3'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    assert not (RUN/'legacy_protocol.json').exists()
    plan=read(OLD/'protocol.json');private=Path(plan['private'])
    specification=dict(created_at=time.time(),source_sha256=sha(Path(__file__)),old_protocol_sha256=sha(OLD/'protocol.json'),
                       old_complete_sha256=sha(OLD/'complete.json'),new_protocol_sha256=sha(RUN/'protocol.json'),
                       method='spread_order_a03',threshold=plan['thresholds']['spread_order_a03'],
                       codec_sha256=sha(OLD/'transported_key_capsule.py'),data_module_sha256=sha(RUN/'order_recovery_data.py'),
                       scope='Fresh CPU249 execution on the same3800confirmation pixels; original private records/provider retained. No candidate changes, no new attack implementation.')
    write(RUN/'legacy_protocol.json',specification)
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    sys.path[:0]=[str(RUN),str(ROOT/'tools/transport_recovery_dependencies')]
    import order_recovery_data as data
    import cv2
    cv2.setNumThreads(4)
    spec=importlib.util.spec_from_file_location('legacy_capsule',OLD/'transported_key_capsule.py');codec=importlib.util.module_from_spec(spec);spec.loader.exec_module(codec)
    assert specification['codec_sha256']==plan['scripts']['transported_key_capsule.py']
    enrollment={r['id']:r for r in read(OLD/'enrollment.json') if r['method']=='spread_order_a03'}
    provider=(private/'provider.bin').read_bytes();assert sha(private/'provider.bin')==plan['provider_sha256']
    old_rows={(r['task'],r['kind']):r for r in read(OLD/'rows.json') if r['method']=='spread_order_a03'}
    rows=[];start=time.time();original_matches=0
    for identity,reference,_,queries in data.confirmation_sources():
        task=queries[0][0]['task'];old_id=f'transfer_{task:03d}';path=private/f'{old_id}_spread_order_a03.json';assert sha(path)==enrollment[old_id]['helper_sha256'];record=read(path)
        key,_=codec.observe(record,reference,provider,'spread_order');assert key is not None
        payload=codec.canonical(dict(id=old_id,reference_pixels_sha256=data.pixel_sha(reference)))
        for base,image in queries:
            result,detail=codec.observe(record,image,provider,'spread_order')
            if result is not None:assert result==key
            returned=codec.unwrap(record,result,provider,detail['residual'],specification['threshold']) if result is not None else None
            if returned is not None:assert returned==payload
            row=dict(base,pixels_sha256=data.pixel_sha(image),key_recovered=result is not None,returned=returned is not None,detail=detail)
            old=old_rows.get((task,base['kind']))
            if old is not None:
                assert old['pixels_sha256']==row['pixels_sha256'] and old['returned']==row['returned'] and old['key_recovered']==row['key_recovered'];original_matches+=1
            rows.append(row)
        codec._basis_cache.clear()
    assert len(rows)==3800 and original_matches==1300
    write(RUN/'legacy_rows.json',rows)
    complete=dict(status='complete',seconds=time.time()-start,queries=len(rows),enrollment_verification_decodes=100,original_same_pixel_matches=original_matches,
                  exact_keys=sum(r['key_recovered'] for r in rows),payload_returns=sum(r['returned'] for r in rows),rows_sha256=sha(RUN/'legacy_rows.json'),protocol_sha256=sha(RUN/'legacy_protocol.json'))
    write(RUN/'legacy_complete.json',complete);print(json.dumps(complete))


if __name__=='__main__':main()
