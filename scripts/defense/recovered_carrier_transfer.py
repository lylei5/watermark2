"""Frozen TR/PRC transfer: recover actual verifier state, then run native detection.

Waits once for the existing full-budget attack/SSCD pipeline via pidfd. Runs new
inversions on all queries, including rejected queries, for fair paired controls.
No attack implementation is included or changed.
"""
import argparse
import ctypes
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import select
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path('/data0/liyanlei/watermark_research')
DEVELOPMENT = ROOT/'runs/instance_edit_quotient_20261113'
PARENT = ROOT/'runs/pgid_carrier_coverage_20261113'
GATE = ROOT/'runs/sscd_pgid_reference_gate_20261113'
GENUINE = ROOT/'runs/pgid_carrier_genuine_20261113'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def pack_native(watermark):
    import torch
    state = {k:(v.detach().cpu() if isinstance(v, torch.Tensor) else v)
             for k,v in vars(watermark).items()}
    state['device'] = 'cpu'
    stream = io.BytesIO()
    torch.save(dict(carrier=watermark.get_wm_type(), state=state), stream)
    return stream.getvalue()


def restore_native(material, device):
    import torch
    from watermarks.tr_watermark import TRWatermark
    from watermarks.prc_watermark import PRCWatermark
    # Only locally enrolled, AEAD-authenticated material reaches this function.
    bundle = torch.load(io.BytesIO(material), map_location=device, weights_only=False)
    cls = {'TR':TRWatermark, 'PRC':PRCWatermark}[bundle['carrier']]
    watermark = object.__new__(cls)
    watermark.__dict__.update(bundle['state'])
    watermark.device = device
    return watermark


def wait(previous):
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(434, previous['pid'], 0)
    if fd < 0:
        if ctypes.get_errno() != 3:
            raise OSError(ctypes.get_errno(), 'pidfd_open')
        return
    try:
        path = Path(f"/proc/{previous['pid']}/cmdline")
        command = path.read_bytes() if path.exists() else b''
        assert not command or hashlib.sha256(command).hexdigest() == previous['cmdline_sha256']
        select.select([fd], [], [])
    finally:
        os.close(fd)


def prepare(run):
    run.mkdir(exist_ok=False)
    private = ROOT/'private_eval'/run.name
    private.mkdir(mode=0o700, exist_ok=False)
    parent = json.loads((PARENT/'protocol.json').read_text())
    dev = json.loads((DEVELOPMENT/'protocol.json').read_text())
    gate = json.loads((GATE/'protocol.json').read_text())
    scores = json.loads((DEVELOPMENT/'scores.json').read_text())
    prior = json.loads((GATE/'state.json').read_text())
    command = Path(f"/proc/{prior['pid']}/cmdline").read_bytes()
    assert str(GATE).encode() in command
    files = ['recovered_carrier_transfer.py', 'instance_coset_recovery.py']
    for name in files:
        origin = Path(__file__) if name == Path(__file__).name else DEVELOPMENT/name
        shutil.copy2(origin, run/name)
    for repetition in range(3):
        shutil.copy2(Path(dev['private'])/f'repetition_{repetition}/projection.npy', private/f'projection_{repetition}.npy')
    secret = private/'provider.bin'
    secret.write_bytes(secrets.token_bytes(32))
    os.chmod(secret, 0o600)
    plan = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        parent_protocol_sha256=sha(PARENT/'protocol.json'), gate_protocol_sha256=sha(GATE/'protocol.json'),
        genuine_protocol_sha256=sha(GENUINE/'protocol.json'), development_scores_sha256=sha(DEVELOPMENT/'scores.json'),
        predecessor=dict(pid=prior['pid'],cmdline_sha256=hashlib.sha256(command).hexdigest()),
        source=parent['source'], model=gate['model'], conditions=gate['conditions'], seed_base=gate['seed_base'],
        distortion_file=gate['distortion_file'], distortion_sha256=gate['distortion_sha256'],
        enrollment_families=dev['enrollment_families'], enrollment_seed_base=2026511300,
        methods=dev['methods'], operating_points=[{k:x[k] for k in ['repetition','method','target','feasible','cutoff']}
            for x in scores['operating_points']],
        private=str(private), provider_sha256=sha(secret),
        projection_sha256=[sha(private/f'projection_{i}.npy') for i in range(3)],
        prc_key_path=json.loads((GENUINE/'protocol.json').read_text())['prc_key_path'],
        prc_key_sha256=json.loads((GENUINE/'protocol.json').read_text())['prc_key_sha256'],
        source_sha256={name:sha(run/name) for name in files}, gpu=7,
        carriers=['TR','PRC'], sources_per_carrier=10, queries_per_carrier=140,
        native_inversions=280, recovery_decisions=3360, attack_generation=False,
        threshold_tuning=False, independent_confirmation=False,
        transfer_scope='Frozen GS-development code and thresholds transferred to native TR/PRC sources',
        native_material='Serialized full author verifier instance state, CPU tensors; authenticated before restore',
        isolation='Function boundary, same process; recovered bytes are the only state supplied to restored detector',
        attack_scope='Existing unchanged CVPR2025 Imprint checkpoints50/100/150 from full151-update runs')
    write(run/'protocol.json', plan)
    print(json.dumps(dict(run=str(run),protocol_sha256=sha(run/'protocol.json'))), flush=True)


def execute(run, plan):
    state = dict(status='waiting_for_attack_and_sscd_exit',pid=os.getpid(),
                 predecessor_pid=plan['predecessor']['pid'],protocol_sha256=sha(run/'protocol.json'))
    assert not (run/'state.json').exists()
    write(run/'state.json', state)
    try:
        wait(plan['predecessor'])
        assert json.loads((GATE/'complete.json').read_text())['status'] == 'complete'
        for path,key in [(PARENT,'parent'),(GATE,'gate'),(GENUINE,'genuine')]:
            assert sha(path/'protocol.json') == plan[key+'_protocol_sha256']
        assert sha(DEVELOPMENT/'scores.json') == plan['development_scores_sha256']
        for name,digest in plan['source_sha256'].items():
            assert sha(run/name) == digest
        for name,digest in plan['source']['files'].items():
            assert sha(Path(plan['source']['path'])/name) == digest
        assert sha(plan['prc_key_path']) == plan['prc_key_sha256']
        assert sha(plan['model']['path']) == plan['model']['sha256']
        assert sha(plan['distortion_file']) == plan['distortion_sha256']
        for carrier in plan['carriers']:
            done = json.loads((PARENT/carrier/'complete.json').read_text())
            assert done['status'] == 'complete' and done['protocol_sha256'] == plan['parent_protocol_sha256']
            assert sha(PARENT/carrier/'attack=forgery/all_results.csv') == done['raw_sha256']
            for name,digest in done['image_sha256'].items():
                assert sha(PARENT/carrier/name) == digest
        fields = subprocess.check_output(['nvidia-smi','-i','7',
            '--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
        assert int(fields[0]) == 7 and int(fields[2]) < 100 and int(fields[3]) == 0, fields
        os.environ.update(CUDA_VISIBLE_DEVICES=fields[1],HF_HOME=str(ROOT/'cache/huggingface'),
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
            OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMBA_NUM_THREADS='4')
        import importlib.util
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from torchvision import transforms
        sys.path.insert(0,str(Path(plan['source']['path'])))
        from pipe import pipe_utils
        from utils import get_watermark_args,configure_prc,set_random_seed
        from watermarks.tr_watermark import parser as tr_parser
        from watermarks.prc_watermark import parser as prc_parser
        from watermarks.watermark_utils import SemanticWatermarks,validate,get_detection_threshold
        from instance_coset_recovery import InstanceRecovery,canonical
        torch.set_num_threads(4)
        private = Path(plan['private'])
        assert sha(private/'provider.bin') == plan['provider_sha256']
        provider_secret = (private/'provider.bin').read_bytes()
        codecs = []
        for rep in range(3):
            assert sha(private/f'projection_{rep}.npy') == plan['projection_sha256'][rep]
            codecs.append(InstanceRecovery(np.load(private/f'projection_{rep}.npy')))
        parent = json.loads((PARENT/'protocol.json').read_text())
        ref = ROOT/'cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/refs/main'
        assert ref.read_text().strip() == parent['revision']
        model = torch.jit.load(plan['model']['path'],map_location='cuda').eval()
        transform = transforms.Compose([transforms.Resize([320,320]),transforms.ToTensor(),
            transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])])
        def encode(image):
            # Match the frozen SSCD extraction without changing native inversion settings.
            flags = (torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,
                     torch.backends.cudnn.benchmark)
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cudnn.benchmark = False
                with torch.no_grad():
                    return F.normalize(model(transform(image).unsqueeze(0).to('cuda')),dim=1).cpu().numpy()[0]
            finally:
                torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark = flags
        spec = importlib.util.spec_from_file_location('transfer_distortions',plan['distortion_file'])
        distort = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(distort)
        (run/'work').mkdir()
        (run/'latents').mkdir()
        os.chdir(run/'work')
        state.update(status='running',gpu=7,gpu_uuid=fields[1],started_at=time.time())
        write(run/'state.json',state)
        rows, controls, enrollments = [], [], []
        for carrier_index,carrier in enumerate(plan['carriers']):
            cfg = (tr_parser if carrier == 'TR' else prc_parser).parse_args([])
            cfg.wm_type, cfg.scheduler_target, cfg.prc_key_path = carrier,parent['target_schedulers'][carrier],plan['prc_key_path']
            set_random_seed(parent['seed'])
            kwargs = get_watermark_args(cfg,'cuda:0')
            provider = pipe_utils.create_diffusion_pipe(pretrained_model_name_or_path=parent['model'],resolution=512,
                device='cuda:0',scheduler_name=cfg.scheduler_target,scheduler_inverse_name=cfg.scheduler_target,
                eager_loading=True,disable_tqdm=True)
            configure_prc(cfg,provider)
            threshold = get_detection_threshold(carrier,parent['model'])
            for wm,idx in SemanticWatermarks[carrier].value.generate_watermark_signals(total_num_signals=10,
                    latent_shape=provider.get_latent_shape(),batch_size=1,target_start_index=0,target_end_index=10,
                    w_seed=parent['seed'],**kwargs):
                identity = f'{carrier}_{idx:03d}'
                clean_path = PARENT/carrier/f'attack=forgery/watermarked_images/image_{idx:03d}.png'
                clean = Image.open(clean_path).convert('RGB')
                reference = encode(clean)
                views, families = [], []
                for family_index,name in enumerate(plan['enrollment_families']):
                    for rep in range(2):
                        seed = plan['enrollment_seed_base']+carrier_index*10000+idx*100+family_index*10+rep
                        options = SimpleNamespace(**{k:None for k in plan['conditions']})
                        setattr(options,name,plan['conditions'][name])
                        distort.set_random_seed(seed)
                        views.append(encode(distort.image_distortion(clean.copy(),seed,options).convert('RGB')))
                        families.append(name)
                views = np.stack(views)
                material = pack_native(wm)
                records = {}
                for repetition,codec in enumerate(codecs):
                    key = secrets.token_bytes(16)
                    for method in plan['methods']:
                        selector,mode = method.rsplit('_',1)
                        helper,_ = codec.enroll(identity,reference,views,families,material,provider_secret,selector,mode,key)
                        records[repetition,method] = (helper,key)
                        record_path = private/f'{identity}_{repetition}_{method}.json'
                        write(record_path,helper)
                        enrollments.append(dict(id=identity,repetition=repetition,method=method,
                            record_sha256=sha(record_path),native_bytes=len(material),record_bytes=len(canonical(helper))))
                item = next(x for x in parent['inputs'] if x['index'] == idx)
                cover = Path(parent['cover_images_dir'])/Path(item['cover']).name
                assert sha(cover) == item['cover_sha256']
                queries = [('clean',clean),('cover',Image.open(cover).resize((512,512)).convert('RGB'))]
                for c,(name,value) in enumerate(plan['conditions'].items()):
                    seed = plan['seed_base']+carrier_index*10000+idx*100+c
                    options = SimpleNamespace(**{k:None for k in plan['conditions']})
                    setattr(options,name,value)
                    distort.set_random_seed(seed)
                    queries.append((name,distort.image_distortion(clean.copy(),seed,options).convert('RGB')))
                for checkpoint in [50,100,150]:
                    path = PARENT/carrier/f'attack=forgery/attacked_images_step_{checkpoint}/image_{idx:03d}.png'
                    queries.append((f'imprint_{checkpoint}',Image.open(path).convert('RGB')))
                templates = np.concatenate([reference[None,:],views])
                quantized = np.round(templates/np.max(np.abs(templates),axis=1,keepdims=True)*127).astype(np.int8).astype(np.float32)
                quantized /= np.linalg.norm(quantized,axis=1,keepdims=True)
                initial = ''.join(format(b,'08b') for b in wm.message) if carrier == 'PRC' else None
                for kind,image in queries:
                    query = encode(image)
                    set_random_seed(parent['seed'])
                    if carrier == 'PRC':
                        z = provider.exact_inversion(images=image,num_inference_steps=50,decode_inv=True)['zT_torch'].detach()
                    else:
                        with torch.no_grad():
                            z = provider.invert_images(image,num_inference_steps=50)['zT_torch']
                    latent_path = run/'latents'/f'{identity}_{kind}.npz'
                    np.savez_compressed(latent_path,z=z.detach().cpu().numpy())
                    result = validate(image,wm,provider,50,initial,provided_zT=z)
                    metric = float(result['p_value']) if carrier == 'TR' else bool(result['PRC_detection_results'])
                    native = metric < threshold if carrier == 'TR' else metric
                    meta = dict(id=identity,carrier=carrier,task=idx,kind=kind,pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                                native_metric=metric,native_accepted=bool(native),latent_sha256=sha(latent_path))
                    control = dict(**meta,sscd_single=float(np.clip(1-reference@query,0,2)),
                        sscd_templates=float(np.clip(1-np.max(templates@query),0,2)),
                        sscd_int8_templates=float(np.clip(1-np.max(quantized@query),0,2)))
                    restored_metric = None
                    for (repetition,method),(helper,key) in records.items():
                        codec = codecs[repetition]
                        tick = time.monotonic()
                        recovered,detail = codec.recover(helper,query,provider_secret)
                        seconds = time.monotonic()-tick
                        distance = codec.recovery_distance(helper,query,reference)
                        assert (recovered is not None) == (distance <= 127)
                        if recovered is not None:
                            assert recovered == (key,material)
                            if restored_metric is None:
                                restored = restore_native(recovered[1],'cuda:0')
                                replay = validate(image,restored,provider,50,initial,provided_zT=z)
                                restored_metric = float(replay['p_value']) if carrier == 'TR' else bool(replay['PRC_detection_results'])
                                assert restored_metric == metric
                        rows.append(dict(**meta,method=method,repetition=repetition,key_recovered=recovered is not None,
                            joint_accepted=bool(recovered is not None and native),region_distance=distance,decode_seconds=seconds,
                            recovered_detector_equal=True if recovered is not None else None))
                        if repetition == 0 and method.startswith('tangent'):
                            trans = helper['metadata']['feature_transform']
                            control[method+'_cosine'] = float(np.clip(1-codec.transform_feature(reference,trans)@
                                codec.transform_feature(query,trans),0,2))
                    controls.append(control)
                print('RECOVERED_CARRIER_SOURCE',identity,flush=True)
            del provider
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        assert len(rows) == 3360 and len(controls) == 280 and len(enrollments) == 240
        write(run/'rows.json',rows)
        write(run/'controls.json',controls)
        write(run/'enrollment.json',enrollments)
        state.update(status='complete',recovery_decisions=len(rows),native_inversions=len(controls),
            seconds=time.time()-state['started_at'],artifacts={n:sha(run/n)for n in ['rows.json','controls.json','enrollment.json']})
        write(run/'complete.json',state)
    except Exception as exc:
        state.update(status='failed',error=repr(exc))
        raise
    finally:
        write(run/'state.json',state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','run'])
    parser.add_argument('--run',type=Path,required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    assert ROOT.is_dir() and run.parent == ROOT/'runs'
    if args.mode == 'prepare':
        prepare(run)
    else:
        plan = json.loads((run/'protocol.json').read_text())
        assert sha(__file__) == plan['source_sha256'][Path(__file__).name]
        execute(run,plan)


if __name__ == '__main__':
    main()
