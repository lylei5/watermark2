"""Replay every merged native decision with authenticated original material."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path('/data0/liyanlei/watermark_research')
RUN=ROOT/'runs/reprompt_plus_evaluation_20261115_v2'
ATTACK=ROOT/'runs/author_reprompt_plus_20261115'
WITNESS=ROOT/'runs/observable_evidence_recovery_20261115'
PARENT=ROOT/'runs/shared_channel_binding_20261114_v2'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    begin=time.time();assert read(RUN/'complete.json')['status']=='complete'
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
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
    torch.set_num_threads(4);cv2.setNumThreads(4)
    wp=read(WITNESS/'protocol.json');parent=read(PARENT/'protocol.json')
    for name,digest in wp['scripts'].items():assert sha(WITNESS/name)==digest
    private=Path(wp['private']);provider=(private/'provider.bin').read_bytes();assert sha(private/'provider.bin')==wp['provider_sha256']
    enrollment={e['id']:e for e in read(WITNESS/'enrollment.json')};all_rows=[]
    for carrier in ['GS','TR']:
        done=read(RUN/carrier/'complete.json');assert sha(RUN/carrier/'rows.json')==done['rows_sha256']
        all_rows+=read(RUN/carrier/'rows.json')
    assert len(all_rows)==480
    queries={q['trial']:q for q in read(RUN/'queries.json')}
    for q in queries.values():
        assert sha(q['image'])==q['image_sha256'] and sha(q['latent'])==q['latent_sha256']
        assert hashlib.sha256(Image.open(q['image']).convert('RGB').tobytes()).hexdigest()==q['pixels_sha256']
    threshold=read(ROOT/'runs/pgid_carrier_coverage_20261113/protocol.json')['thresholds']['TR']['value']
    checks=[]
    for ref in read(ATTACK/'protocol.json')['references']:
        identity=ref['id'];hp=private/(identity+'.json');assert sha(hp)==enrollment[identity]['helper_sha256']
        helper=read(hp);meta=helper['metadata'];image=Image.open(ref['image']).convert('RGB');layer=meta['bands'][0]
        pairs=codec.unpack(layer['pairs']).astype(np.int32)
        observed=np.packbits((codec.measurements(codec.bands(image)[0],pairs)>=0).astype(np.uint8),axis=1).reshape(-1)
        key=bytes(codec.CODEC.decode((observed^codec.unpack(layer['offset'])).tobytes())[0])
        assert hashlib.sha256(b'observable/'+identity.encode()+key).hexdigest()==meta['confirmation']
        wrapping=codec.hmac.new(provider,b'observable-wrap/'+identity.encode()+key,hashlib.sha256).digest()
        fields=json.loads(AESGCM(wrapping).decrypt(bytes.fromhex(helper['nonce']),__import__('base64').b64decode(helper['wrapped']),codec.canonical(meta)))
        blobpath=Path(parent['private'])/(identity+'_native.bin');assert sha(blobpath)==fields['blob_sha256']
        blob=blobpath.read_bytes();material=AESGCM(bytes.fromhex(fields['data_key'])).decrypt(blob[:12],blob[12:],canonical(dict(id=identity,carrier=ref['carrier'])))
        if ref['carrier']=='GS':
            f=json.loads(material);wm=GSWatermark(latent_shape=(1,4,64,64),device='cpu',key=bytes.fromhex(f['key']),nonce=bytes.fromhex(f['nonce']),message=bytes.fromhex(f['message']))
        else:wm=restore_native(material,'cpu')
        for row in [r for r in all_rows if r['id']==identity]:
            q=queries[row['trial']];assert row['pixels_sha256']==q['pixels_sha256'] and sha(row['latent'])==row['latent_sha256']
            z=np.load(row['latent'])['z'];result=wm.get_accuracies(torch.from_numpy(z.reshape(1,4,64,64)))
            metric=float(result['bit_accuracies'][0]) if ref['carrier']=='GS' else float(result['p_values'][0])
            accepted=bool(metric>181/256 if ref['carrier']=='GS' else metric<threshold)
            assert accepted==row['accepted']
            if ref['carrier']=='GS':assert metric==row['metric']
            checks.append(dict(trial=row['trial'],carrier=ref['carrier'],defended=row['defended'],accepted=accepted,
                original_metric=row['metric'],replayed_metric=metric,absolute_metric_difference=abs(metric-row['metric'])))
    assert len(checks)==480
    out=dict(status='complete',source_sha256=sha(__file__),evaluation_complete_sha256=sha(RUN/'complete.json'),
        new_detector_replays=480,decision_disagreements=0,attack_image_pixel_latent_checks=len(queries),
        material_records=20,seconds=time.time()-begin,scope='CPU replay on existing inversions with authenticated native material; no new inversion or attack.',checks=checks)
    path=RUN/'native_replay_audit.json';assert not path.exists();path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='checks'}))


if __name__=='__main__':main()
