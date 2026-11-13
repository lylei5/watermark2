"""Frozen, server-only diagnosis of final-instance representation separability.

No training, attack generation, watermark generation, or carrier changes.
The nine image transformations execute the frozen upstream GS implementation.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path('/data0/liyanlei/watermark_research')
PARENT = ROOT / 'runs/phase1_trajectory_native_gs_20261113'
DISTORT = ROOT / 'runs/defense_revalidation_20261111/gs_image_utils.py'
CONDITIONS = dict(random_crop_ratio=.6, random_drop_ratio=.8,
                  brightness_factor=6, jpeg_ratio=25, gaussian_blur_r=4,
                  gaussian_std=.05, median_blur_k=7, sp_prob=.05, resize_ratio=.25)
SSCD_URL = 'https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare(run):
    run.mkdir(exist_ok=False)
    prior = json.loads((PARENT / 'protocol.json').read_text())
    sources = []
    for shard in range(3):
        done = json.loads((PARENT / f'generate_{shard}/complete.json').read_text())
        for row in done['rows']:
            row = {k: row[k] for k in ['task', 'dataset', 'codes', 'images']}
            for artifact in [row['codes'], *row['images'].values()]:
                assert sha(PARENT / artifact['path']) == artifact['sha256']
            sources.append(row)
    sources.sort(key=lambda x: x['task'])
    assert [r['task'] for r in sources] == list(range(100))
    calibration = []
    rng = random.Random(2026111307)
    for dataset in ['COCO', 'SDP']:
        ids = [r['task'] for r in sources if r['dataset'] == dataset]
        assert len(ids) == 50
        rng.shuffle(ids)
        calibration.extend(ids[:25])
    assert sha(DISTORT) == 'aa0dd86b4543bde39983e5b7ec8fd487fba476d7a435f83308891e187ee93fbb'
    shutil.copy2(__file__, run / Path(__file__).name)
    semantic = Path(prior['paths']['semantic'])
    model_files = list((semantic / 'dinov2-giant').glob('*'))
    model_files += [semantic / 'sembind_masker.pth', Path(prior['paths']['official']) / 'model.py']
    plan = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                purpose='Separate representation loss from auxiliary-to-final instance drift',
                parent=str(PARENT), parent_protocol_sha256=sha(PARENT / 'protocol.json'),
                sources=sources, paths=prior['paths'],
                model_sha256={str(p): sha(p) for p in model_files if p.is_file()},
                sscd=dict(url=SSCD_URL, path=str(ROOT / 'models/sscd/sscd_disc_mixup.torchscript.pt'),
                          provenance='Author TorchScript disc_mixup; SHA256 frozen before first forward',
                          preprocessing='Official 320x320 skew, ImageNet normalization'),
                distortion_file=str(DISTORT), distortion_sha256=sha(DISTORT),
                conditions=CONDITIONS, seed_base=2026111300,
                calibration=sorted(calibration), evaluation=sorted(set(range(100)) - set(calibration)),
                threshold='Smallest observed distance retaining at least95% of450 calibration distortions; <=',
                clean_in_calibration=False, anchors=['bound', 'aux'],
                primary_wrong_instance='same: same prompt, independently generated seed',
                extra_wrong_instances_for_bound=['aux', 'gs'],
                methods=['sembind_hamming', 'dinov2_cosine', 'sscd_cosine'],
                batch_size=1, source_count=100, images_per_source=13,
                training=False, attack_jobs=0, independent_confirmation=False,
                cohort='Previously observed development prompts; split by prompt family before these scores',
                interpretation='Content verification diagnostic; no watermark detector decisions or attack-success rate',
                source_sha256=sha(run / Path(__file__).name), gpu=7)
    write(run / 'protocol.json', plan)
    print(json.dumps(dict(run=str(run), protocol_sha256=sha(run / 'protocol.json'))), flush=True)


def extract(run, mode, plan):
    import numpy as np
    from types import SimpleNamespace
    fields = subprocess.check_output(['nvidia-smi', '-i', str(plan['gpu']),
        '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    assert int(fields[0]) == 7 and int(fields[2]) < 100 and int(fields[3]) == 0, fields
    os.environ.update(CUDA_VISIBLE_DEVICES=fields[1], HF_HOME=str(ROOT / 'cache/huggingface'),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    out = run / mode
    out.mkdir(exist_ok=False)
    state = dict(status='loading', pid=os.getpid(), gpu=7, gpu_uuid=fields[1],
                 protocol_sha256=sha(run / 'protocol.json'), started_at=time.time())
    write(out / 'state.json', state)
    try:
        assert sha(DISTORT) == plan['distortion_sha256']
        distortion = load_module('probe_gs_distortion', DISTORT)
        if mode == 'dino':
            from transformers import AutoImageProcessor, AutoModel
            for path, digest in plan['model_sha256'].items():
                assert sha(path) == digest, path
            semantic = Path(plan['paths']['semantic'])
            processor = AutoImageProcessor.from_pretrained(str(semantic / 'dinov2-giant'), local_files_only=True)
            dino = AutoModel.from_pretrained(str(semantic / 'dinov2-giant'), local_files_only=True).to('cuda').eval()
            model_def = load_module('probe_hash_model', Path(plan['paths']['official']) / 'model.py')
            masker = model_def.AttnSupConHashingModel(input_dim=1536, feat_dim=8192, hash_dim=1024)
            masker.load_state_dict(torch.load(semantic / 'sembind_masker.pth', map_location='cpu', weights_only=True), strict=True)
            masker.to('cuda').eval()
        else:
            path = Path(plan['sscd']['path'])
            write(out / 'model_manifest.json', dict(path=str(path), sha256=sha(path), url=SSCD_URL))
            sscd = torch.jit.load(str(path), map_location='cuda').eval()
            transform = transforms.Compose([transforms.Resize([320, 320]), transforms.ToTensor(),
                transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
        state['status'] = 'running'
        write(out / 'state.json', state)
        features, bits, keys, image_hashes, parity = [], [], [], [], []
        work = out / 'work'
        work.mkdir()
        os.chdir(work)
        with torch.no_grad():
            for row in plan['sources']:
                images = {}
                for name, artifact in row['images'].items():
                    path = PARENT / artifact['path']
                    assert sha(path) == artifact['sha256']
                    with Image.open(path) as opened:
                        images[name] = opened.convert('RGB')
                for i, (name, value) in enumerate(plan['conditions'].items()):
                    args = SimpleNamespace(**{k: None for k in plan['conditions']})
                    setattr(args, name, value)
                    seed = plan['seed_base'] + row['task'] * 100 + i
                    distortion.set_random_seed(seed)
                    images[name] = distortion.image_distortion(images['bound'].copy(), seed, args).convert('RGB')
                stored = np.load(PARENT / row['codes']['path']) if mode == 'dino' else None
                for name, image in images.items():
                    keys.append([row['task'], name])
                    image_hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
                    if mode == 'dino':
                        inputs = processor(images=image, return_tensors='pt').to('cuda')
                        feature = F.normalize(dino(**inputs).last_hidden_state[:, 0, :], dim=1)
                        logits = masker.hash_head(masker.proj_head(masker.encoder(feature)))
                        code = ((torch.sign(torch.tanh(12 * logits)) + 1) / 2).floor().to(torch.uint8).cpu().numpy()[0]
                        bits.append(code)
                        if name in row['images']:
                            parity.append(dict(task=row['task'], image=name,
                                differing_bits=int(np.count_nonzero(code != stored[name + '_code'].reshape(-1)))))
                    else:
                        feature = F.normalize(sscd(transform(image).unsqueeze(0).to('cuda')), dim=1)
                    features.append(feature.cpu().numpy()[0])
                if stored is not None:
                    stored.close()
        assert len(keys) == 1300 and len({tuple(x) for x in keys}) == 1300
        payload = dict(features=np.stack(features))
        if bits:
            payload['bits'] = np.stack(bits)
        np.savez_compressed(out / 'features.npz', **payload)
        write(out / 'metadata.json', dict(keys=keys, image_hashes=image_hashes, clean_code_parity=parity))
        (out / 'packages.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
        state.update(status='complete', image_count=len(keys), feature_dim=features[0].size,
                     seconds=time.time() - state['started_at'], artifacts={p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != 'state.json'})
        write(out / 'complete.json', state)
        print(json.dumps({k: v for k, v in state.items() if k != 'artifacts'}), flush=True)
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        write(out / 'state.json', state)


def score(run, plan):
    import numpy as np
    banks = {}
    previous = None
    parity = None
    for mode in ['dino', 'sscd']:
        done = json.loads((run / mode / 'complete.json').read_text())
        assert done['status'] == 'complete'
        for name, digest in done['artifacts'].items():
            assert sha(run / mode / name) == digest
        meta = json.loads((run / mode / 'metadata.json').read_text())
        assert previous is None or meta['keys'] == previous['keys'] and meta['image_hashes'] == previous['image_hashes']
        previous = meta
        arrays = np.load(run / mode / 'features.npz')
        keys = [tuple(x) for x in meta['keys']]
        banks[mode + '_cosine'] = dict(zip(keys, arrays['features']))
        if mode == 'dino':
            banks['sembind_hamming'] = dict(zip(keys, arrays['bits']))
            parity = meta['clean_code_parity']
    banks['dinov2_cosine'] = banks.pop('dino_cosine')
    results, distances = [], []
    for method, bank in banks.items():
        def distance(task, anchor, name):
            a, b = bank[task, anchor], bank[task, name]
            return float(np.count_nonzero(a != b)) if method == 'sembind_hamming' else float(np.clip(1 - np.dot(a, b), 0, 2))
        for anchor in plan['anchors']:
            calibration = [distance(i, anchor, c) for i in plan['calibration'] for c in plan['conditions']]
            threshold = sorted(calibration)[math.ceil(.95 * len(calibration)) - 1]
            entry = dict(method=method, anchor=anchor, threshold=threshold,
                         calibration_accepted=sum(x <= threshold for x in calibration), calibration_total=len(calibration), groups={})
            for split in ['calibration', 'evaluation']:
                for dataset in ['COCO', 'SDP', 'all']:
                    ids = [i for i in plan[split] if dataset == 'all' or plan['sources'][i]['dataset'] == dataset]
                    conditions = ['bound', *plan['conditions'], 'same'] + (['aux', 'gs'] if anchor == 'bound' else [])
                    group = {}
                    all_positive = []
                    negative = []
                    for condition in conditions:
                        values = [distance(i, anchor, condition) for i in ids]
                        accepted = [v <= threshold for v in values]
                        group[condition] = dict(accepted=sum(accepted), total=len(ids),
                            distance_quantiles=np.quantile(values, [0, .05, .5, .95, 1]).tolist())
                        if condition in plan['conditions']:
                            all_positive.extend(accepted)
                        if condition == 'same':
                            negative = accepted
                        if dataset == 'all':
                            distances.extend(dict(method=method, anchor=anchor, split=split, task=i,
                                condition=condition, distance=v, accepted=a) for i, v, a in zip(ids, values, accepted))
                    group['pooled_distortions'] = dict(accepted=sum(all_positive), total=len(all_positive))
                    # Paired source bootstrap: keep all conditions of a prompt together.
                    rng = np.random.default_rng(2026111308)
                    matrix = np.array([[distance(i, anchor, c) <= threshold for c in plan['conditions']] for i in ids])
                    resamples = rng.integers(0, len(ids), (10000, len(ids)))
                    group['pooled_distortions']['source_bootstrap_95'] = np.quantile(matrix[resamples].mean(axis=(1, 2)), [.025, .975]).tolist()
                    group['same']['source_bootstrap_95'] = np.quantile(np.array(negative)[resamples].mean(axis=1), [.025, .975]).tolist()
                    entry['groups'][split + '/' + dataset] = group
            results.append(entry)
    write(run / 'scores.json', dict(protocol_sha256=sha(run / 'protocol.json'), results=results,
        clean_code_parity=dict(images=len(parity), total_differing_bits=sum(x['differing_bits'] for x in parity),
                               max_differing_bits=max(x['differing_bits'] for x in parity)),
        confidence_note='Cluster bootstrap is descriptive; zero observed errors do not establish a zero population rate or1e-6 FPR'))
    write(run / 'distances.json', distances)
    print(json.dumps([dict(method=x['method'], anchor=x['anchor'], threshold=x['threshold'],
        evaluation={k: x['groups']['evaluation/all'][k] for k in ['pooled_distortions', 'same']}) for x in results]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'dino', 'sscd', 'score'])
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    assert ROOT.is_dir() and args.run.resolve().parent == ROOT / 'runs', '249 server workspace required'
    run = args.run.resolve()
    if args.mode == 'prepare':
        prepare(run)
    else:
        plan = json.loads((run / 'protocol.json').read_text())
        assert sha(__file__) == plan['source_sha256']
        if args.mode == 'score':
            score(run, plan)
        else:
            extract(run, args.mode, plan)


if __name__ == '__main__':
    main()
