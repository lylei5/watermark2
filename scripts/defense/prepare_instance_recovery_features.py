"""Fresh final-image and enrollment-view features for actual key recovery on249."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import time
from types import SimpleNamespace

ROOT = Path('/data0/liyanlei/watermark_research')
REPRESENTATIONS = ROOT / 'runs/instance_representation_20261113_v2'
PHASE1 = ROOT / 'runs/phase1_trajectory_native_gs_20261113'
PGID = ROOT / 'runs/defense_revalidation_20261111'
CHANNEL = ROOT / 'runs/carrier_content_channel_20261113'
FAMILIES = ['random_crop_ratio', 'random_drop_ratio', 'gaussian_std', 'sp_prob']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def prepare(run):
    run.mkdir(exist_ok=False)
    private = ROOT / 'private_eval' / run.name
    private.mkdir(mode=0o700, exist_ok=False)
    secret = dict(provider=secrets.token_hex(32), projection_seeds=[secrets.token_hex(32) for _ in range(3)])
    write(private / 'secrets.json', secret)
    os.chmod(private / 'secrets.json', 0o600)
    prior = json.loads((REPRESENTATIONS / 'protocol.json').read_text())
    phase = json.loads((PHASE1 / 'protocol.json').read_text())
    channel = json.loads((CHANNEL / 'protocol.json').read_text())
    sources = []
    for item in prior['sources']:
        task = item['task']
        artifact = item['images']['gs']
        identity = Path(phase['identity_root']) / f'{task:03d}.npz'
        sources.append(dict(id=f'S{task:03d}', cohort='spatial_GS', task=task, dataset=item['dataset'],
            reference=str(PHASE1 / artifact['path']), reference_sha256=artifact['sha256'],
            same=str(PHASE1 / item['images']['same']['path']), same_sha256=item['images']['same']['sha256'],
            identity=str(identity), identity_sha256=sha(identity),
            split='calibration' if task in prior['calibration'] else 'evaluation'))
    for task in range(10):
        items = [x for x in channel['inputs'] if x['task'] == task]
        clean = next(x for x in items if x['kind'] == 'clean')
        sources.append(dict(id=f'G{task:03d}', cohort='PGID_GS', task=task,
            reference=clean['path'], reference_sha256=clean['sha256'],
            attacks=[x for x in items if x['kind'] == 'imprint'], split='transfer'))
    for item in sources:
        assert sha(item['reference']) == item['reference_sha256']
    for name in ['prepare_instance_recovery_features.py', 'instance_coset_recovery.py']:
        shutil.copy2(Path(__file__).parent / name, run / name)
    plan = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), sources=sources,
        model=json.loads((REPRESENTATIONS / 'sscd/model_manifest.json').read_text()),
        distortion_file=prior['distortion_file'], distortion_sha256=prior['distortion_sha256'],
        conditions=prior['conditions'], enrollment_families=FAMILIES, repeats=2,
        enrollment_seed_base=2026211300, query_seed_base=2026311300,
        enrollment='Final nativeGS image plus8 views from4 published stochastic GS transformation families',
        evaluation='Nine official GS transformations with separate seeds; deterministic families absent from enrollment',
        actual_key_bits=128, native_carrier_changed=False,
        code=dict(n=1023, k=133, t=127, designed_distance=255, library='galois'),
        methods=['clean_margin_fixed', 'orbit_margin_fixed', 'orbit_margin_templates', 'orbit_margin_coset'],
        controls=['SSCD single reference', 'SSCD all9 enrollment templates', 'SSCD int8 templates',
                  'Exact Hamming regions with the same projections and edit offsets'],
        projection_repetitions=3, projection_pool=8192, private=str(private), secret_sha256=sha(private / 'secrets.json'),
        key_security='Random128-bit binding key plus provider-secret authenticated wrapping; no content entropy claim',
        attacks='Existing full151-update CVPR2025 Imprint outputs, no attack creation or modification',
        independent_confirmation=False, gpu=7,
        source_sha256={name:sha(run / name) for name in ['prepare_instance_recovery_features.py', 'instance_coset_recovery.py']})
    write(run / 'protocol.json', plan)
    print(json.dumps(dict(run=str(run), protocol_sha256=sha(run / 'protocol.json'))), flush=True)


def execute(run, plan):
    state = dict(status='preflight', pid=os.getpid(), protocol_sha256=sha(run / 'protocol.json'))
    assert not (run / 'feature_state.json').exists()
    write(run / 'feature_state.json', state)
    try:
        for name, digest in plan['source_sha256'].items():
            assert sha(run / name) == digest
        assert sha(plan['model']['path']) == plan['model']['sha256']
        assert sha(plan['distortion_file']) == plan['distortion_sha256']
        fields = subprocess.check_output(['nvidia-smi', '-i', '7', '--query-gpu=index,uuid,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip().split(', ')
        assert int(fields[0]) == 7 and int(fields[2]) < 50 and int(fields[3]) == 0, fields
        os.environ.update(CUDA_VISIBLE_DEVICES=fields[1], OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from torchvision import transforms
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        model = torch.jit.load(plan['model']['path'], map_location='cuda').eval()
        transform = transforms.Compose([transforms.Resize([320, 320]), transforms.ToTensor(),
            transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
        spec = importlib.util.spec_from_file_location('recovery_distortion', plan['distortion_file'])
        distort = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(distort)
        (run / 'feature_work').mkdir()
        os.chdir(run / 'feature_work')
        state.update(status='running', gpu=7, gpu_uuid=fields[1], started_at=time.time())
        write(run / 'feature_state.json', state)
        features, metadata = [], []
        channel_rows = json.loads((CHANNEL / 'rows.json').read_text())
        channel_plan = json.loads((CHANNEL / 'protocol.json').read_text())
        with torch.no_grad():
            for source_index, item in enumerate(plan['sources']):
                assert sha(item['reference']) == item['reference_sha256']
                clean = Image.open(item['reference']).convert('RGB')
                queries = [('clean', None, None, clean)]
                for family_index, name in enumerate(plan['enrollment_families']):
                    for repeat in range(plan['repeats']):
                        options = SimpleNamespace(**{k:None for k in plan['conditions']})
                        setattr(options, name, plan['conditions'][name])
                        seed = plan['enrollment_seed_base']+source_index*100+family_index*10+repeat
                        distort.set_random_seed(seed)
                        image = distort.image_distortion(clean.copy(), seed, options).convert('RGB')
                        queries.append((f'enroll_{family_index}_{repeat}', name, seed, image))
                for condition_index, (name, value) in enumerate(plan['conditions'].items()):
                    options = SimpleNamespace(**{k:None for k in plan['conditions']})
                    setattr(options, name, value)
                    seed = (channel_plan['distortion_seed_base']+item['task']*100+condition_index if item['cohort']=='PGID_GS'
                            else plan['query_seed_base']+source_index*100+condition_index)
                    distort.set_random_seed(seed)
                    image = distort.image_distortion(clean.copy(), seed, options).convert('RGB')
                    queries.append((name, None, seed, image))
                if item['cohort'] == 'spatial_GS':
                    assert sha(item['same']) == item['same_sha256']
                    queries.append(('same', None, None, Image.open(item['same']).convert('RGB')))
                else:
                    for attack in item['attacks']:
                        assert sha(attack['path']) == attack['sha256']
                        queries.append((f"imprint_{attack['checkpoint']}", None, None, Image.open(attack['path']).convert('RGB')))
                for kind, family, seed, image in queries:
                    pixels = hashlib.sha256(image.tobytes()).hexdigest()
                    native = None
                    if item['cohort'] == 'PGID_GS' and not kind.startswith('enroll_'):
                        checkpoint = int(kind.split('_')[1]) if kind.startswith('imprint_') else -1
                        original_kind = 'imprint' if checkpoint != -1 else kind
                        old = next(r for r in channel_rows if r['task']==item['task'] and r['kind']==original_kind and r['checkpoint']==checkpoint)
                        assert pixels == old['pixels_sha256']
                        native = old['native']
                    feature = F.normalize(model(transform(image).unsqueeze(0).to('cuda')), dim=1).cpu().numpy()[0]
                    features.append(feature)
                    metadata.append(dict(id=item['id'], kind=kind, family=family, seed=seed, pixels_sha256=pixels,
                                         native=native, cohort=item['cohort'], split=item['split']))
                print('ENROLLMENT_FEATURES', item['id'], flush=True)
        assert len(features) == 2110
        np.savez_compressed(run / 'features.npz', features=np.stack(features))
        write(run / 'feature_rows.json', metadata)
        state.update(status='complete', rows=len(features), seconds=time.time()-state['started_at'],
            artifacts={name:sha(run / name) for name in ['features.npz','feature_rows.json']})
        write(run / 'features_complete.json', state)
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        write(run / 'feature_state.json', state)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare','run'])
    p.add_argument('--run', type=Path, required=True)
    args=p.parse_args()
    run=args.run.resolve()
    assert ROOT.is_dir() and run.parent==ROOT/'runs'
    if args.mode=='prepare':
        prepare(run)
    else:
        execute(run,json.loads((run/'protocol.json').read_text()))


if __name__=='__main__':
    main()
