"""Frozen capsule source transfer with freshly executed SSCD and CopyNCE on249."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path('/data0/liyanlei/watermark_research')
PARENT=ROOT/'runs/transported_capsule_observability_20261115'
DATA=ROOT/'runs/instance_representation_20261113_v2'
COPY=ROOT/'runs/copynce_control_20261114'
METHODS=['spread_order_a03','spread_order_identity_a03']
CONTROLS=['sscd','copynce_descriptor','copynce_matcher']


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)


def prepare(run, reuse=None):
    assert run.parent==ROOT/'runs';run.mkdir(exist_ok=False)
    private=ROOT/'private_eval'/run.name;private.mkdir(mode=0o700,exist_ok=False)
    if reuse is not None:
        previous=read(reuse/'protocol.json');shutil.copy2(Path(previous['private'])/'provider.bin',private/'provider.bin')
    else:(private/'provider.bin').write_bytes(secrets.token_bytes(32))
    os.chmod(private/'provider.bin',0o600)
    shutil.copy2(__file__,run/Path(__file__).name)
    shutil.copy2(PARENT/'transported_key_capsule.py',run/'transported_key_capsule.py')
    shutil.copy2(COPY/'copynce_control.py',run/'copynce_control.py')
    cal=read(PARENT/'calibration.json');controls=read(PARENT/'fixed_controls.json')
    thresholds={m:next(r['threshold'] for r in cal if r['method']==m and r['score_name']=='score' and r['target']==.9) for m in METHODS}
    thresholds.update({m:next(r['threshold'] for r in controls if r['method']==m and r['target']==.9) for m in CONTROLS})
    assert all(v is not None for v in thresholds.values())
    inputs=[PARENT/'complete.json',PARENT/'calibration.json',PARENT/'fixed_controls.json',PARENT/'cases.json',DATA/'protocol.json',DATA/'sscd/complete.json',DATA/'sscd/metadata.json',DATA/'sscd/features.npz',DATA/'sscd/model_manifest.json',COPY/'protocol.json']
    plan=dict(created_at=time.time(),scripts={p.name:sha(p) for p in run.glob('*.py')},inputs={str(p):sha(p) for p in inputs},
        methods=METHODS,controls=CONTROLS,thresholds=thresholds,private=str(private),provider_sha256=sha(private/'provider.bin'),
        source_count=100,query_pairs=1300,planned_decodes=2600,
        scope='Fixed-threshold transfer to100previously generated SemBind-bound reference images, nine author distortions and three related-instance probes. These assets were used in earlier representation studies; not wholly unseen project data.',
        method_selection='Main and identity ablation selected after the prior30source development experiment; no tuning on this transfer cohort.',
        payload='Reference pixel hash and record identity, actually AEAD-unwrapped. This transfer measures content/key recovery only; no new native-watermark or attack decisions.',
        model_execution='Fresh SSCD at the original batch1 and released CopyNCE descriptor/matcher; same exact pixels as capsule, verify cachedSSCD parity.',
        attacks_generated=False,carrier_modified=False,allowed_gpus=[7,5,4],independent_confirmation=False)
    if reuse is not None:
        files=list(Path(previous['private']).glob('*.json'));assert len(files)==200
        plan.update(reused_enrollment={str(p):sha(p) for p in files},reused_run=str(reuse),enrollment_verification_decodes=100)
    write(run/'protocol.json',plan);print(json.dumps(dict(run=str(run),protocol_sha256=sha(run/'protocol.json'))))


def execute(run):
    plan=read(run/'protocol.json');state=dict(status='preflight',pid=os.getpid(),started_at=time.time(),protocol_sha256=sha(run/'protocol.json'));write(run/'state.json',state)
    try:
        for n,h in plan['scripts'].items():assert sha(run/n)==h
        for n,h in plan['inputs'].items():assert sha(Path(n))==h
        devices=subprocess.check_output(['nvidia-smi','-i','7,5,4','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        parsed={int(v[0]):v for v in [line.split(', ') for line in devices.strip().splitlines()]}
        free=[parsed[i] for i in [7,5,4] if int(parsed[i][2])<100 and int(parsed[i][3])==0];assert free,'No allowed idleGPU'
        gpu=free[0];os.environ.update(CUDA_VISIBLE_DEVICES=gpu[1],OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        state.update(gpu=int(gpu[0]),gpu_uuid=gpu[1]);write(run/'state.json',state)
        sys.path.insert(0,str(ROOT/'tools/transport_recovery_dependencies'))
        import numpy as np
        import cv2
        from PIL import Image
        import transported_key_capsule as codec
        import copynce_control as cc
        torch,config,builders=cc.imports()
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
        torch.set_num_threads(4);cv2.setNumThreads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        cp=read(COPY/'protocol.json')
        for n,h in cp['author_files'].items():assert sha(cc.AUTHOR/n)==h
        models={};ts={}
        for kind in cc.WEIGHTS:
            m,cfg,_=cc.load_model(kind,torch,config,builders);models[kind]=m.to('cuda')
            ts[kind]=transforms.Compose([transforms.Resize((cfg.input.img_size,cfg.input.img_size),interpolation=InterpolationMode[cfg.input.augmentation.interpolation.upper()]),transforms.ToTensor(),transforms.Normalize(cfg.input.augmentation.post_process.normalize.mean,cfg.input.augmentation.post_process.normalize.std)])
        data=read(DATA/'protocol.json');sscd_path=Path(data['sscd']['path']);manifest=read(DATA/'sscd/model_manifest.json')
        assert str(sscd_path)==manifest['path'] and sha(sscd_path)==manifest['sha256']
        sscd=torch.jit.load(str(sscd_path),map_location='cuda').eval()
        st=transforms.Compose([transforms.Resize([320,320]),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        metadata=read(DATA/'sscd/metadata.json');old=np.load(DATA/'sscd/features.npz')['features'];index={tuple(k):i for i,k in enumerate(metadata['keys'])}
        assert sha(DATA/'sscd/metadata.json')==read(DATA/'sscd/complete.json')['artifacts']['metadata.json']
        assert sha(DATA/'sscd/features.npz')==read(DATA/'sscd/complete.json')['artifacts']['features.npz']
        old_refs={c['base']['pixels_sha256'] for c in read(PARENT/'cases.json') if c['base']['kind']=='clean'}
        assert sha(Path(data['distortion_file']))==data['distortion_sha256']
        spec=importlib.util.spec_from_file_location('fixed_distortion',data['distortion_file']);dist=importlib.util.module_from_spec(spec);spec.loader.exec_module(dist)
        private=Path(plan['private']);provider=(private/'provider.bin').read_bytes();assert sha(private/'provider.bin')==plan['provider_sha256']
        (run/'work').mkdir(exist_ok=False);os.chdir(run/'work');rows=[];enrollment=[];parity=0.;overlap=0
        state.update(status='evaluating',gpu=int(gpu[0]),gpu_uuid=gpu[1]);write(run/'state.json',state)
        for source in data['sources']:
            task=source['task'];images={}
            for name,artifact in source['images'].items():
                path=Path(data['parent'])/artifact['path'];assert sha(path)==artifact['sha256'];images[name]=Image.open(path).convert('RGB')
            for i,(name,value) in enumerate(data['conditions'].items()):
                args=SimpleNamespace(**{k:None for k in data['conditions']});setattr(args,name,value);seed=data['seed_base']+task*100+i;dist.set_random_seed(seed);images[name]=dist.image_distortion(images['bound'].copy(),seed,args).convert('RGB')
            keys=list(images);hashes={k:hashlib.sha256(images[k].tobytes()).hexdigest() for k in keys}
            for k in keys:assert hashes[k]==metadata['image_hashes'][index[task,k]]
            overlap+=hashes['bound'] in old_refs;assert hashes['bound'] not in old_refs
            identity=f'transfer_{task:03d}';payload=codec.canonical(dict(id=identity,reference_pixels_sha256=hashes['bound']))
            reused={}
            if 'reused_enrollment' in plan:
                for method in METHODS:
                    path=next(Path(n) for n in plan['reused_enrollment'] if Path(n).name==f'{identity}_{method}.json')
                    assert sha(path)==plan['reused_enrollment'][str(path)];reused[method]=read(path)
                key,_=codec.observe(reused[METHODS[0]],images['bound'],provider,'spread_order_identity');assert key is not None
            else:key=secrets.token_bytes(16)
            records={}
            for method in METHODS:
                mode=method.removesuffix('_a03')
                record=reused[method] if reused else codec.enroll(identity,images['bound'],payload,provider,.3,key=key,mode=mode)[0]
                records[method]=record
                hp=private/f'{identity}_{method}.json';write(hp,record);enrollment.append(dict(id=identity,method=method,helper_sha256=sha(hp),helper_bytes=hp.stat().st_size))
            with torch.no_grad():
                sf=np.concatenate([torch.nn.functional.normalize(sscd(st(images[k])[None].to('cuda')),dim=1).cpu().numpy() for k in keys])
                for i,k in enumerate(keys):parity=max(parity,float(np.max(np.abs(sf[i]-old[index[task,k]]))))
                ct=torch.stack([ts['descriptor'](images[k]) for k in keys]);assert all(torch.equal(ct[i],ts['matching'](images[k])) for i,k in enumerate(keys))
                ct=ct.to('cuda');df=torch.nn.functional.normalize(models['descriptor']({'images':ct})['feat'],dim=1).cpu().numpy();j=keys.index('bound')
                match=models['matching']({'queries':ct,'references':ct[j:j+1].expand(len(keys),-1,-1,-1)})['preds'].cpu().numpy().reshape(-1)
            for i,kind in enumerate(keys):
                controls=dict(sscd=float(np.clip(1-sf[j]@sf[i],0,2)),copynce_descriptor=float(np.clip(1-df[j]@df[i],0,2)),copynce_matcher=float(1-match[i]))
                for method in METHODS:
                    start=time.perf_counter();out,detail=codec.observe(records[method],images[kind],provider,method.removesuffix('_a03'))
                    if out is not None:assert out==key
                    score=detail['residual'] if out is not None else None
                    result=codec.unwrap(records[method],out,provider,score,plan['thresholds'][method]) if out is not None else None
                    if result is not None:assert result==payload
                    scores=codec.reference_scores(images['bound'],images[kind],detail,observation='order')
                    rows.append(dict(id=identity,task=task,dataset=source['dataset'],kind=kind,pixels_sha256=hashes[kind],method=method,
                        key_recovered=out is not None,returned=result is not None,score=score,detail=detail,seconds=time.perf_counter()-start,**scores,**controls))
            codec._basis_cache.clear()
        write(run/'rows.json',rows);write(run/'enrollment.json',enrollment)
        write(run/'model_parity.json',dict(rows=len(rows),sscd_max_feature_difference=parity,sscd_batch_size=1,limit=2e-5))
        assert len(rows)==2600 and parity<2e-5,(len(rows),parity)
        genuine=['bound',*data['conditions']];aggregate=[];matched=[]
        for dataset in ['all',*sorted({s['dataset'] for s in data['sources']})]:
            for method in METHODS:
                rr=[r for r in rows if r['method']==method and (dataset=='all' or r['dataset']==dataset)]
                for kind in ['all_genuine','all_edits',*keys]:
                    part=[r for r in rr if r['kind'] in genuine] if kind=='all_genuine' else [r for r in rr if r['kind'] in data['conditions']] if kind=='all_edits' else [r for r in rr if r['kind']==kind]
                    aggregate.append(dict(dataset=dataset,method=method,kind=kind,total=len(part),keys=sum(r['key_recovered'] for r in part),accepted=sum(r['returned'] for r in part)))
                gg=[r for r in rr if r['kind'] in genuine];count=sum(r['returned'] for r in gg)
                for control in CONTROLS+['reference_residual','pixel_residual']:
                    finite=sorted(r[control] for r in gg if r[control] is not None)
                    cutoff=finite[count-1] if count else -1.;attained=sum(r[control] is not None and r[control]<=cutoff for r in gg)
                    for kind in ['same','aux','gs']:
                        part=[r for r in rr if r['kind']==kind];matched.append(dict(dataset=dataset,method=method,control=control,kind=kind,total=len(part),genuine_total=len(gg),candidate_genuine=count,control_genuine=attained,exact_match=count==attained,threshold=cutoff,candidate=sum(r['returned'] for r in part),baseline=sum(r[control] is not None and r[control]<=cutoff for r in part)))
            rr=[r for r in rows if r['method']==METHODS[0] and (dataset=='all' or r['dataset']==dataset)]
            for control in CONTROLS:
                for kind in ['all_genuine','all_edits',*keys]:
                    part=[r for r in rr if r['kind'] in genuine] if kind=='all_genuine' else [r for r in rr if r['kind'] in data['conditions']] if kind=='all_edits' else [r for r in rr if r['kind']==kind]
                    aggregate.append(dict(dataset=dataset,method=control,kind=kind,total=len(part),accepted=sum(r[control]<=plan['thresholds'][control] for r in part)))
        write(run/'aggregate.json',aggregate);write(run/'matched.json',matched)
        state.update(status='complete',seconds=time.time()-state['started_at'],actual_decodes=len(rows),enrollment_verification_decodes=plan.get('enrollment_verification_decodes',0),exact_keys=sum(r['key_recovered'] for r in rows),payload_returns=sum(r['returned'] for r in rows),
            reference_overlap=overlap,paired_pixel_checks=1300,sscd_max_feature_difference=parity,fresh_sscd_images=1300,fresh_copynce_images=1300,fresh_copynce_pairs=1300,
            native_detector_calls=0,attacks_generated=False,artifacts={n:sha(run/n) for n in ['rows.json','enrollment.json','aggregate.json','matched.json','model_parity.json']})
        write(run/'complete.json',state);print(json.dumps(state))
    except BaseException as exc:state.update(status='failed',error=repr(exc));raise
    finally:write(run/'state.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','execute']);p.add_argument('--run',type=Path,required=True);p.add_argument('--reuse-enrollment',type=Path);a=p.parse_args();prepare(a.run,a.reuse_enrollment) if a.action=='prepare' else execute(a.run)
