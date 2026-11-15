"""Frozen partial-order recovery and fresh nine-template controls on249."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time

ROOT=Path('/data0/liyanlei/watermark_research')
METHODS=[dict(name='order_a005_all',alpha=.05,drop_black=False),dict(name='order_a005_visible',alpha=.05,drop_black=True),dict(name='order_a010_visible',alpha=.1,drop_black=True)]


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def prepare(run,stage,development=None):
    assert run.parent==ROOT/'runs';run.mkdir(exist_ok=False)
    private=ROOT/'private_eval'/run.name;private.mkdir(mode=0o700);(private/'provider.bin').write_bytes(secrets.token_bytes(32));os.chmod(private/'provider.bin',0o600)
    names=['run_order_recovery.py','order_recovery_data.py','order_constrained_capsule.py','transported_key_capsule.py']
    for name in names:shutil.copy2(Path(__file__).with_name(name),run/name)
    import order_recovery_data as data
    inputs=[data.PARENT/'cases.json',data.PARENT/'calibration.json',data.SHARED/'protocol.json',data.SHARED/'feature_rows.json',data.DATA/'protocol.json',data.DATA/'sscd/model_manifest.json']
    if stage=='confirmation':
        assert development is not None and read(development/'complete.json')['status']=='complete'
        inputs += [development/'complete.json',development/'calibration.json']
    plan=dict(stage=stage,created_at=time.time(),scripts={n:sha(run/n) for n in names},inputs={str(p):sha(p) for p in inputs},
              methods=METHODS,list_bits=6,private=str(private),provider_sha256=sha(private/'provider.bin'),
              expected_pairs=1010 if stage=='development' else 3800,
              controls=['sscd','sscd_templates','reference_forward','reference_order_all','reference_order_visible'],
              development=str(development) if development is not None else None,
              calibration='GS tasks4,5,6 genuine30;90% and95% thresholds for direct scores. Candidate exact key+AEAD acceptance, no score cutoff.',
              methods_selection='All three prespecified methods retained; confirmation does not select a winner or change parameters.',
              evaluation='Content/key binding only; same pixels for all controls. Historical native acceptance joined separately in development; no new diffusion inversions.',
              unseen_types=['contrast','rotation','resizedcrop'] if stage=='confirmation' else [],
              unseen_scope='Types absent from earlier execution protocols, no transformed output inspected before freeze. Source images previously used; not unseen sources.',
              waves_commit='1477635d306d6f8c77a588e6441bd4ba301a1f7a',
              waves_files={str(data.WAVES/n):sha(data.WAVES/n) for n in ['distortions/distortions.py','utils/image_utils.py','utils/exp_utils.py']},
              author_baselines_only=True,allowed_gpus=[7,5,4],excluded_gpus=[6],key_bits=128,
              payload='Authenticated record identity and reference pixel hash; exact bytes checked.',novelty_established=False)
    write(run/'protocol.json',plan);print(json.dumps(dict(run=str(run),protocol_sha256=sha(run/'protocol.json'))))


def common(run):
    plan=read(run/'protocol.json')
    for name,digest in plan['scripts'].items():assert sha(run/name)==digest,(name,'source changed')
    for name,digest in plan['inputs'].items():assert sha(name)==digest,(name,'input changed')
    for name,digest in plan['waves_files'].items():assert sha(name)==digest,(name,'author source changed')
    sys.path[:0]=[str(run),str(ROOT/'tools/transport_recovery_dependencies')]
    import order_recovery_data as data
    (run/'work').mkdir(exist_ok=True);os.chdir(run/'work')
    return plan,data


def controls(run):
    assert not (run/'controls_complete.json').exists()
    raw=subprocess.check_output(['nvidia-smi','-i','7,5,4','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    devices={int(x[0]):x for x in [line.split(', ') for line in raw.strip().splitlines()]}
    free=[devices[i] for i in [7,5,4] if int(devices[i][2])<100 and int(devices[i][3])==0];assert free,'No allowed idle GPU'
    gpu=free[0];os.environ.update(CUDA_VISIBLE_DEVICES=gpu[1],OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    plan,data=common(run)
    import numpy as np
    import torch
    from torchvision import transforms
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    manifest=read(data.DATA/'sscd/model_manifest.json');assert sha(manifest['path'])==manifest['sha256']
    model=torch.jit.load(manifest['path'],map_location='cuda').eval()
    transform=transforms.Compose([transforms.Resize([320,320]),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    start=time.time();rows=[];templates=[];images=0;cache={}
    with torch.no_grad():
        def encode(image):
            nonlocal images
            digest=data.pixel_sha(image)
            if digest not in cache:
                cache[digest]=torch.nn.functional.normalize(model(transform(image)[None].to('cuda')),dim=1).cpu().numpy()[0];images+=1
            return cache[digest]
        for identity,reference,views,queries in data.sources(plan['stage']):
            vectors=np.stack([encode(im) for im in views]);templates.append(dict(id=identity,pixels=[data.pixel_sha(im) for im in views],bytes=int(vectors.nbytes)))
            for base,image in queries:
                q=encode(image);rows.append(dict(case_id=base['case_id'],id=identity,pixels_sha256=data.pixel_sha(image),sscd=float(np.clip(1-vectors[0]@q,0,2)),sscd_templates=float(np.clip(1-np.max(vectors@q),0,2))))
            cache.clear()
    assert len(rows)==plan['expected_pairs']
    write(run/'controls.json',rows);write(run/'templates.json',templates)
    done=dict(status='complete',seconds=time.time()-start,queries=len(rows),fresh_model_images=images,gpu=int(gpu[0]),gpu_uuid=gpu[1],batch_size=1,model_sha256=manifest['sha256'],artifacts={n:sha(run/n) for n in ['controls.json','templates.json']})
    write(run/'controls_complete.json',done);print(json.dumps(done))


def execute(run):
    assert not (run/'complete.json').exists()
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    plan,data=common(run)
    import numpy as np
    import cv2
    import transported_key_capsule as codec
    import order_constrained_capsule as order
    cv2.setNumThreads(4)
    cdone=read(run/'controls_complete.json');assert cdone['status']=='complete' and sha(run/'controls.json')==cdone['artifacts']['controls.json']
    controls_by_case={r['case_id']:r for r in read(run/'controls.json')};rows=[];enrollment=[];start=time.time()
    state=dict(status='running',pid=os.getpid(),protocol_sha256=sha(run/'protocol.json'));write(run/'state.json',state)
    private=Path(plan['private']);assert sha(private/'provider.bin')==plan['provider_sha256'];provider=(private/'provider.bin').read_bytes()
    try:
        for identity,reference,_,queries in data.sources(plan['stage']):
            x=codec.rgb(reference,'order').reshape(-1,3);payload=codec.canonical(dict(id=identity,reference_pixels_sha256=data.pixel_sha(reference)));key=secrets.token_bytes(16);records={}
            for alpha in sorted({m['alpha'] for m in plan['methods']}):
                record,_=codec.enroll(identity,reference,payload,provider,alpha,key=key,mode='spread_order');records[alpha]=record
                path=private/f'{identity}_{alpha}.json';write(path,record);enrollment.append(dict(id=identity,alpha=alpha,helper_sha256=sha(path),helper_bytes=path.stat().st_size))
            for base,image in queries:
                control=controls_by_case[base['case_id']];assert data.pixel_sha(image)==control['pixels_sha256'];y=codec.rgb(image,'order').reshape(-1,3)
                direct=dict(reference_order_all=order.order_residual(x,y),reference_order_visible=order.order_residual(x,y,True))
                old={}
                for alpha,record in records.items():
                    old_key,detail=codec.observe(record,image,provider,'spread_order')
                    if old_key is not None:assert old_key==key
                    old[alpha]=(old_key,detail)
                direct.update(codec.reference_scores(reference,image,old[.05][1],observation='order'))
                direct['reference_forward']=direct.pop('reference_residual')
                for method in plan['methods']:
                    tick=time.time();out,detail=order.recover(records[method['alpha']],image,provider,6,method['drop_black'])
                    if out is not None:assert out==(key,payload)
                    row=dict({**base,**control},method=method['name'],key_recovered=out is not None,returned=out is not None,detail=detail,seconds=time.time()-tick,
                             rounded_key=bool(detail.get('rounded_key_confirmed',False)),forward_key=old[method['alpha']][0] is not None,forward_detail=old[method['alpha']][1],**direct)
                    rows.append(row)
            codec._basis_cache.clear()
        assert len(rows)==len(plan['methods'])*plan['expected_pairs']
        write(run/'rows.json',rows);write(run/'enrollment.json',enrollment)
        if plan['stage']=='development':
            conditions=read(data.SHARED/'protocol.json')['conditions']
            cal=[r for r in rows if r['method']==plan['methods'][0]['name'] and r['carrier']=='GS' and r['split']=='calibration' and r['kind'] in ['clean',*conditions]];assert len(cal)==30
            calibration=[]
            for method in plan['controls']:
                scores=sorted(r[method] for r in cal if r[method] is not None)
                for target in [.9,.95]:
                    required=int(np.ceil(len(cal)*target));calibration.append(dict(method=method,target=target,total=len(cal),available=len(scores),threshold=scores[required-1] if len(scores)>=required else None))
            write(run/'calibration.json',calibration)
        else:write(run/'calibration.json',read(Path(plan['development'])/'calibration.json'))
        complete=dict(state,status='complete',seconds=time.time()-start,observations=len(rows),paired_queries=plan['expected_pairs'],exact_keys=sum(r['returned'] for r in rows),payload_returns=sum(r['returned'] for r in rows),native_calls=0,
                      artifacts={n:sha(run/n) for n in ['rows.json','enrollment.json','calibration.json']})
        write(run/'complete.json',complete);write(run/'state.json',complete);print(json.dumps(complete))
    except BaseException as exc:
        write(run/'state.json',dict(state,status='failed',error=repr(exc)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','controls','execute']);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--stage',choices=['development','confirmation'],default='development');parser.add_argument('--development',type=Path);args=parser.parse_args()
    if args.action=='prepare':prepare(args.run,args.stage,args.development)
    elif args.action=='controls':controls(args.run)
    else:execute(args.run)
