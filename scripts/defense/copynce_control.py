"""Released CopyNCE descriptor/matcher controls on frozen watermark query pixels."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path('/data0/liyanlei/watermark_research')
AUTHOR = ROOT / 'third_party/CopyNCE-c2223d18abf3289202666a0d2b9db1c3a7f1b483'
PARENT = ROOT / 'runs/shared_channel_binding_20261114_v2'
WEIGHTS = {
    'descriptor': ('descriptor_vits_224.pth.tar', '829fdf4ae032bac24b4fcc60535cfc3b53026af083b33a0b8c71f1973f2166db'),
    'matching': ('matcher_vits_224.pth.tar', 'dd8dea1c345039bbda303c42457aa3a0389d46fb37ea39506bfb1faddedda1df'),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def imports():
    sys.path[:0] = [str(ROOT / 'tools/copynce_dependencies'), str(AUTHOR)]
    import torch
    import timm
    from omegaconf import OmegaConf
    from core.models import build_copy_descriptor, build_paired_copy_detector
    assert timm.__version__ == '0.9.2'
    torch.set_num_threads(4)
    return torch, OmegaConf, dict(descriptor=build_copy_descriptor, matching=build_paired_copy_detector)


def load_model(kind, torch, config, builders):
    cfg = config.load(AUTHOR / f'core/configs/{kind}_default_config.yaml')
    model = builders[kind](cfg)
    path = ROOT / 'models/copynce' / WEIGHTS[kind][0]
    assert sha(path) == WEIGHTS[kind][1]
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    weights = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
    extra = set(weights) - set(model.state_dict())
    assert extra == ({'infonce_log_scale'} if kind == 'descriptor' else set())
    for key in extra:
        weights.pop(key)
    model.load_state_dict(weights, strict=True)
    return model.eval(), cfg, dict(epoch=checkpoint['epoch'], excluded_training_parameters=sorted(extra), strict_model_parameters=True)


def prepare(run):
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    assert not run.exists()
    torch, config, builders = imports()
    checks = {}
    for kind in WEIGHTS:
        model, cfg, check = load_model(kind, torch, config, builders)
        check['parameter_count'] = sum(p.numel() for p in model.parameters())
        checks[kind] = check
        del model
    parent = read(PARENT / 'protocol.json')
    run.mkdir()
    shutil.copy2(__file__, run / Path(__file__).name)
    protocol = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sha(run / Path(__file__).name), author_commit='c2223d18abf3289202666a0d2b9db1c3a7f1b483',
        author_files={str(p.relative_to(AUTHOR)): sha(p) for p in AUTHOR.rglob('*') if p.is_file() and p.suffix in ['.py', '.yaml']},
        weights={kind: dict(file=str(ROOT / 'models/copynce' / v[0]), sha256=v[1]) for kind, v in WEIGHTS.items()},
        parent_protocol_sha256=sha(PARENT / 'protocol.json'), feature_rows_sha256=sha(PARENT / 'feature_rows.json'),
        distortion_file=parent['distortion_file'], distortion_sha256=parent['distortion_sha256'], conditions=parent['conditions'],
        enrollment_families=parent['enrollment_families'], fit_sources=parent['fit_sources'],
        calibration_sources=parent['calibration_sources'], evaluation_sources=parent['evaluation_sources'], calibration_targets=[.9, .95],
        preprocessing='Author config: PIL RGB, bicubic Resize(224,224), ToTensor, ImageNet mean/std; no crop ensembling or score normalization',
        descriptor='Author CopyDescriptor feat, L2 normalization and cosine distance',
        matching='Author ClassificationCopyDetector evaluation preds; distance is 1-preds',
        adaptation='Image recipe loader and batching for claimed-record watermark verification; released model classes and parameters unchanged',
        scope='Released base ViT-S224 checkpoints, no fine-tuning; not complete paper numerical reproduction',
        batch_size=8, expected_query_observations=440, expected_record_pairs=710,
        native_verification='Join unchanged native observations after parent completion; model worker receives no native keys or helper records',
        allowed_gpus=[7, 5, 4], attack_generation=False, independent_confirmation=False)
    write(run / 'protocol.json', protocol)
    write(run / 'preflight.json', dict(status='complete', models=checks, torch_version=torch.__version__, timm_version='0.9.2',
        environment_difference='Author requirements use Python3.8/PyTorch2.1; runtime uses Python3.10/PyTorch2.5.1'))
    print(json.dumps(dict(run=str(run), protocol_sha256=sha(run / 'protocol.json'))), flush=True)


def extract(run):
    plan = read(run / 'protocol.json')
    assert sha(__file__) == plan['source_sha256'] and not (run / 'extract_state.json').exists()
    state = dict(status='preflight', pid=os.getpid(), protocol_sha256=sha(run / 'protocol.json'))
    write(run / 'extract_state.json', state)
    try:
        for name, digest in plan['author_files'].items():
            assert sha(AUTHOR / name) == digest
        assert sha(plan['distortion_file']) == plan['distortion_sha256']
        assert sha(PARENT / 'feature_rows.json') == plan['feature_rows_sha256']
        devices = subprocess.check_output(['nvidia-smi', '-i', '7,5,4', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        parsed = {int(v[0]): v for v in [line.split(', ') for line in devices.strip().splitlines()]}
        available = [parsed[i] for i in [7, 5, 4] if int(parsed[i][2]) < 100 and int(parsed[i][3]) == 0]
        assert available, 'No allowed idle GPU'
        gpu = available[0]
        os.environ.update(CUDA_VISIBLE_DEVICES=gpu[1], OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        torch, config, builders = imports()
        import numpy as np
        from PIL import Image
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        state.update(status='running', gpu=int(gpu[0]), gpu_uuid=gpu[1], started_at=time.time())
        write(run / 'extract_state.json', state)
        meta = [r for r in read(PARENT / 'feature_rows.json') if not r['kind'].startswith('enroll_')]
        assert len(meta) == 440
        index = {(r['id'], r['kind']): i for i, r in enumerate(meta)}
        spec = importlib.util.spec_from_file_location('official_gs_distortions', plan['distortion_file'])
        distort = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(distort)
        (run / 'work').mkdir()
        os.chdir(run / 'work')
        def pixels(row):
            assert sha(row['image']) == row['image_sha256']
            im = Image.open(row['image'])
            if row['resize']:
                im = im.resize((512, 512))
            im = im.convert('RGB')
            if row['family']:
                options = SimpleNamespace(**{k: None for k in plan['conditions']})
                setattr(options, row['family'], plan['conditions'][row['family']])
                distort.set_random_seed(row['seed'])
                im = distort.image_distortion(im.copy(), row['seed'], options).convert('RGB')
            assert hashlib.sha256(im.tobytes()).hexdigest() == row['pixels_sha256']
            return im
        models, configs = {}, {}
        for kind in WEIGHTS:
            model, cfg, _ = load_model(kind, torch, config, builders)
            models[kind], configs[kind] = model.to('cuda'), cfg
        transforms_by_kind = {}
        for kind, cfg in configs.items():
            transforms_by_kind[kind] = transforms.Compose([transforms.Resize((cfg.input.img_size, cfg.input.img_size),
                interpolation=InterpolationMode[cfg.input.augmentation.interpolation.upper()]), transforms.ToTensor(),
                transforms.Normalize(cfg.input.augmentation.post_process.normalize.mean, cfg.input.augmentation.post_process.normalize.std)])
        tensors = []
        for row in meta:
            im = pixels(row)
            a, b = [transforms_by_kind[k](im) for k in WEIGHTS]
            assert torch.equal(a, b)
            tensors.append(a)
        tensors = torch.stack(tensors)
        batch = plan['batch_size']
        descriptors, pairs, matched = [], [], []
        with torch.no_grad():
            for i in range(0, len(meta), batch):
                out = models['descriptor']({'images': tensors[i:i+batch].to('cuda')})['feat']
                descriptors.append(torch.nn.functional.normalize(out, dim=1).cpu().numpy())
            features = np.concatenate(descriptors)
            assert features.shape == (440, 512)
            for identity in sorted({r['id'] for r in meta}):
                ref = index[identity, 'clean']
                source = meta[ref]
                split = ('fit' if source['task'] in plan['fit_sources'] else 'calibration' if source['task'] in plan['calibration_sources'] else 'evaluation') if source['carrier'] == 'GS' else 'transfer'
                cases = [(i, r['kind']) for i, r in enumerate(meta) if r['id'] == identity]
                cases += [(i, 'wrong_record') for i, r in enumerate(meta) if r['carrier'] == source['carrier'] and r['kind'] == 'clean' and r['id'] != identity]
                for i, kind in cases:
                    pairs.append(dict(id=identity, query_id=meta[i]['id'], carrier=source['carrier'], task=source['task'], split=split,
                        kind=kind, ref_index=ref, query_index=i, pixels_sha256=meta[i]['pixels_sha256'],
                        descriptor_distance=float(np.clip(1-features[ref] @ features[i], 0, 2))))
            assert len(pairs) == 710
            for i in range(0, len(pairs), batch):
                part = pairs[i:i+batch]
                out = models['matching']({'queries': tensors[[r['query_index'] for r in part]].to('cuda'),
                                          'references': tensors[[r['ref_index'] for r in part]].to('cuda')})['preds']
                matched.extend(out.cpu().numpy().reshape(-1).tolist())
        assert len(matched) == 710 and np.isfinite(features).all() and np.isfinite(matched).all()
        for row, score in zip(pairs, matched):
            assert 0 <= score <= 1
            row['matcher_distance'] = 1-score
        write(run / 'raw_rows.json', pairs)
        np.savez_compressed(run / 'features.npz', features=features)
        state.update(status='complete', seconds=time.time()-state['started_at'], encoded_observations=440,
                     unique_pixel_hashes=len({r['pixels_sha256'] for r in meta}), matcher_pairs=710,
                     artifacts={n: sha(run / n) for n in ['raw_rows.json', 'features.npz']})
        write(run / 'extract_complete.json', state)
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        write(run / 'extract_state.json', state)


def summarize(run):
    plan, done = read(run / 'protocol.json'), read(run / 'extract_complete.json')
    assert done['status'] == 'complete' and read(PARENT / 'complete.json')['status'] == 'complete'
    assert not (run / 'summary.json').exists()
    for name, digest in done['artifacts'].items():
        assert sha(run / name) == digest
    controls = {(r['id'], r['query_id'], r['kind']): r for r in read(PARENT / 'controls.json')}
    rows = read(run / 'raw_rows.json')
    assert len(rows) == len(controls) == 710
    for r in rows:
        c = controls[r['id'], r['query_id'], r['kind']]
        assert r['pixels_sha256'] == c['pixels_sha256'] and r['split'] == c['split']
        r['native_accepted'] = c['native_accepted']
    def genuine(r):
        return r['kind'] in ['clean', *plan['conditions']]
    scores = []
    for method in ['descriptor_distance', 'matcher_distance']:
        cal = [r for r in rows if r['split'] == 'calibration' and genuine(r)]
        assert len(cal) == 30 and all(r['native_accepted'] for r in cal)
        values = sorted(r[method] for r in cal)
        for target in plan['calibration_targets']:
            cutoff = values[math.ceil(target * len(cal))-1]
            for carrier, split in [('GS', 'evaluation'), ('TR', 'transfer'), ('PRC', 'transfer')]:
                for kind in ['all_genuine', 'unknown_edits', 'clean', 'cover', 'same_prompt', 'wrong_record', 'imprint_50', 'imprint_100', 'imprint_150', *plan['conditions']]:
                    part = [r for r in rows if r['carrier'] == carrier and r['split'] == split and
                        (genuine(r) if kind == 'all_genuine' else r['kind'] in plan['conditions'] and r['kind'] not in plan['enrollment_families'] if kind == 'unknown_edits' else r['kind'] == kind)]
                    if part:
                        scores.append(dict(method=method, target=target, threshold=cutoff, carrier=carrier, split=split, kind=kind, total=len(part),
                            accepted=sum(r[method] <= cutoff for r in part), joint=sum(r[method] <= cutoff and r['native_accepted'] for r in part)))
    write(run / 'rows.json', rows)
    write(run / 'summary.json', dict(status='complete', protocol_sha256=sha(run / 'protocol.json'), scores=scores,
        matched_pixel_checks=710, parent_controls_sha256=sha(PARENT / 'controls.json'), artifacts={'rows.json': sha(run / 'rows.json')},
        scope=plan['scope'], independent_confirmation=False))
    for r in scores:
        if r['target'] == .95 and r['kind'] in ['all_genuine', 'same_prompt', 'wrong_record', 'imprint_150']:
            print(json.dumps(r), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'extract', 'summarize'])
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    assert ROOT.is_dir() and args.run.resolve().parent == ROOT / 'runs'
    globals()[args.mode](args.run.resolve())
