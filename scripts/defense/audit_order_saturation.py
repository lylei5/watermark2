"""Post-confirmation mechanism ablation on the two previously known edit types."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path('/data0/liyanlei/watermark_research')
PARENT=ROOT/'runs/order_capsule_confirmation_20261115'
RUN=ROOT/'runs/order_saturation_ablation_20261115'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def main():
    assert not RUN.exists();RUN.mkdir()
    for name in ['audit_order_saturation.py','order_constrained_capsule.py']:
        shutil.copy2(Path(__file__).with_name(name),RUN/name)
    for name in ['order_recovery_data.py','transported_key_capsule.py']:
        shutil.copy2(PARENT/name,RUN/name)
    parent=read(PARENT/'protocol.json');private=Path(parent['private'])
    plan=dict(created_at=time.time(),parent_complete_sha256=sha(PARENT/'complete.json'),scripts={p.name:sha(p) for p in RUN.glob('*.py')},
              scope='Post-confirmation diagnostic, no new edit families or parameter selection. Reuse exact alpha0.05 records; same two known author brightness/noise conditions. Compare retaining high-saturation order constraints against deleting them.',
              queries=200,actual_decodes=400,enrollment_verification_decodes=100,attacks_generated=False)
    write(RUN/'protocol.json',plan)
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    sys.path[:0]=[str(RUN),str(ROOT/'tools/transport_recovery_dependencies')]
    import order_recovery_data as data
    import order_constrained_capsule as order
    import transported_key_capsule as codec
    from PIL import Image
    data_plan=read(data.DATA/'protocol.json');dist=data.module('known_distortions',data_plan['distortion_file']);assert sha(Path(data_plan['distortion_file']))==data_plan['distortion_sha256']
    previous={(r['task'],r['kind']):r for r in read(PARENT/'rows.json') if r['method']=='order_a005_visible'}
    enrollment={r['id']:r for r in read(PARENT/'enrollment.json') if r['alpha']==.05}
    provider=(private/'provider.bin').read_bytes();assert sha(private/'provider.bin')==parent['provider_sha256']
    (RUN/'work').mkdir();os.chdir(RUN/'work');rows=[];start=time.time()
    for source in data_plan['sources']:
        task=source['task'];identity=f'confirmation_{task:03d}';artifact=source['images']['bound'];path=Path(data_plan['parent'])/artifact['path'];assert sha(path)==artifact['sha256'];reference=Image.open(path).convert('RGB')
        hp=private/f'{identity}_0.05.json';assert sha(hp)==enrollment[identity]['helper_sha256'];record=read(hp)
        truth,_=order.recover(record,reference,provider,6,True);assert truth is not None
        for kind in ['brightness_factor','gaussian_std']:
            i=list(data_plan['conditions']).index(kind);image=data.author_edit(reference,kind,data_plan['seed_base']+task*100+i,data_plan,dist)
            assert data.pixel_sha(image)==previous[task,kind]['pixels_sha256']
            for drop_white in [False,True]:
                result,detail=order.recover(record,image,provider,6,True,drop_white)
                if result is not None:assert result==truth
                if not drop_white:assert (result is not None)==previous[task,kind]['returned']
                rows.append(dict(task=task,kind=kind,drop_white=drop_white,returned=result is not None,detail=detail,pixels_sha256=data.pixel_sha(image)))
        codec._basis_cache.clear()
    write(RUN/'rows.json',rows)
    aggregate=[dict(kind=kind,drop_white=drop_white,total=100,accepted=sum(r['returned'] for r in rows if r['kind']==kind and r['drop_white']==drop_white)) for kind in ['brightness_factor','gaussian_std'] for drop_white in [False,True]]
    write(RUN/'aggregate.json',aggregate)
    complete=dict(status='complete',seconds=time.time()-start,actual_decodes=len(rows),enrollment_verification_decodes=100,original_pixel_decision_matches=200,aggregate=aggregate,
                  artifacts={n:sha(RUN/n) for n in ['rows.json','aggregate.json']})
    write(RUN/'complete.json',complete);print(json.dumps(complete))


if __name__=='__main__':main()
