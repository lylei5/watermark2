"""Server-only measurement of content drift and native GS channel errors.

Uses existing full-budget published Imprint images and unchanged GS transforms.
It does not generate attacks, train an encoder, change a carrier, or recover keys.
"""
import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path('/data0/liyanlei/watermark_research')
PARENT = ROOT / 'runs/defense_revalidation_20261111'
DISTORTIONS = ROOT / 'runs/pgid_published_revalidation_20261112'
PROBE = ROOT / 'runs/instance_representation_20261113_v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare(run):
    run.mkdir(exist_ok=False)
    old = json.loads((DISTORTIONS / 'protocol.json').read_text())
    model = json.loads((PROBE / 'sscd/model_manifest.json').read_text())
    inputs, artifacts = [], {}
    for idx in range(10):
        directory = PARENT / ('pgid0' if idx < 5 else 'pgid5') / 'attack=forgery'
        csv_path = directory / 'all_results.csv'
        artifacts[str(csv_path)] = sha(csv_path)
        for kind, checkpoint in [('clean', -1)] + [('imprint', n) for n in [50, 100, 150]]:
            sub = 'watermarked_images' if kind == 'clean' else f'attacked_images_step_{checkpoint}'
            image = directory / sub / f'image_{idx:03d}.png'
            inputs.append(dict(task=idx, kind=kind, checkpoint=checkpoint, path=str(image), sha256=sha(image)))
    for p in [PARENT / 'protocol.json', DISTORTIONS / 'protocol.json',
              DISTORTIONS / 'shard_0/complete.json', DISTORTIONS / 'shard_1/complete.json']:
        artifacts[str(p)] = sha(p)
    shutil.copy2(__file__, run / Path(__file__).name)
    plan = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        purpose='Test whether native carrier error predicts final-instance content drift',
        scope='Fresh native GS inversion and SSCD on historical published Imprint outputs and reproduced GS transformations',
        sources=10, rows=130, inputs=inputs, artifacts=artifacts,
        parent=str(PARENT), conditions=old['conditions'], distortion_seed_base=old['seed_base'],
        distortion_file=str(PARENT / 'gs_image_utils.py'), distortion_sha256=old['distortion_source_sha256'],
        model=model, model_revision=old['model_revision'], gpu=7,
        carrier='PGID released flat GS, 256 bits, 64 repetitions, public author fixture identities',
        inversion='Native DDIM 50, float32, TF32 disabled', carrier_threshold=181/256,
        frozen_sscd_threshold=0.6199655532836914,
        fit_sources=[0, 1, 2, 3], calibration_sources=[4, 5, 6], evaluation_sources=[7, 8, 9],
        target_distortion_retention=.95, fit_clean=False,
        primary_predictor='Pre-majority coordinate bit error rate against registered carrier identity',
        candidate='Nonnegative-slope linear prediction of SSCD distance plus calibrated upper residual',
        calibration='Smallest threshold covering ceil(.95 * 27) calibration distortions; clean separate',
        controls=['Frozen SSCD gate', 'Recalibrated SSCD gate', 'Native GS alone'],
        analysis=['Source-held-out prediction R2 versus source-held-out mean',
                  'Within-condition centered correlation', 'Per-condition unconditional decisions',
                  'All checkpoint decisions with final checkpoint primary'],
        inference_scope='Development mechanism probe; three held-out sources, no independent confirmation or FPR estimate',
        protocol_constraint='Native identity key is already available to verifier; not a content-key-only extractor',
        new_or_modified_attacks=False, attack_generation=False, encoder_training=False, key_recovery=False,
        source_sha256=sha(run / Path(__file__).name))
    write(run / 'protocol.json', plan)
    print(json.dumps(dict(run=str(run), protocol_sha256=sha(run / 'protocol.json'))), flush=True)


def execute(run, plan):
    state = dict(status='preflight', pid=os.getpid(), protocol_sha256=sha(run / 'protocol.json'))
    assert not (run / 'state.json').exists()
    write(run / 'state.json', state)
    try:
        for path, digest in plan['artifacts'].items():
            assert sha(path) == digest, path
        for item in plan['inputs']:
            assert sha(item['path']) == item['sha256']
        parent_plan = json.loads((PARENT / 'protocol.json').read_text())
        for relative, digest in parent_plan['source_sha256'].items():
            assert sha(PARENT / relative) == digest, relative
        assert sha(plan['distortion_file']) == plan['distortion_sha256']
        assert sha(plan['model']['path']) == plan['model']['sha256']
        ref = ROOT / 'cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/refs/main'
        assert ref.read_text().strip() == plan['model_revision']
        info = subprocess.check_output(['nvidia-smi', '-i', str(plan['gpu']),
            '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
        assert int(info[0]) == 7 and int(info[2]) < 50 and int(info[3]) == 0, info
        os.environ.update(CUDA_VISIBLE_DEVICES=info[1], HF_HOME=str(ROOT / 'cache/huggingface'),
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
            OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        sys.path.insert(0, str(PARENT / 'PGID'))
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from torchvision import transforms
        from pipe import pipe_utils
        from watermarks.gs_watermark import parser as gs_parser
        from watermarks.watermark_utils import SemanticWatermarks
        from utils import get_watermark_args, set_random_seed
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        spec = importlib.util.spec_from_file_location('channel_probe_distortion', plan['distortion_file'])
        distort = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(distort)
        model = torch.jit.load(plan['model']['path'], map_location='cuda').eval()
        transform = transforms.Compose([transforms.Resize([320, 320]), transforms.ToTensor(),
            transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
        work = run / 'work'
        work.mkdir()
        os.chdir(work)
        (run / 'latents').mkdir()
        gs_args = gs_parser.parse_args([])
        gs_args.wm_type = 'GS'
        kwargs = get_watermark_args(gs_args, 'cuda:0')
        assert kwargs['message_width_in_bytes'] == 32 and kwargs['num_replications'] == 64 and kwargs['l'] == 1
        provider = pipe_utils.create_diffusion_pipe(pretrained_model_name_or_path=parent_plan['pgid']['modelid'],
            resolution=512, device='cuda:0', scheduler_name='DDIM', scheduler_inverse_name='DDIM', eager_loading=True)
        state.update(status='running', gpu=7, gpu_uuid=info[1], started_at=time.time())
        write(run / 'state.json', state)
        old_distortions = []
        for shard in range(2):
            old_distortions += json.loads((DISTORTIONS / f'shard_{shard}/complete.json').read_text())['rows']
        rows, feature_rows = [], []
        with torch.no_grad():
            def encode(image):
                return F.normalize(model(transform(image).unsqueeze(0).to('cuda')), dim=1).cpu().numpy()[0]
            for wm, idx in SemanticWatermarks.GS.value.generate_watermark_signals(total_num_signals=10,
                    latent_shape=provider.get_latent_shape(), batch_size=1, target_start_index=0,
                    target_end_index=10, w_seed=1, **kwargs):
                initial = ''.join(format(b, '08b') for b in wm.message)
                truth = np.array(list(initial), dtype=np.int8)
                csv_path = PARENT / ('pgid0' if idx < 5 else 'pgid5') / 'attack=forgery/all_results.csv'
                native_rows = list(csv.DictReader(csv_path.open()))
                native = next(r for r in native_rows if int(r['image_idx']) == idx and r['stage'] == 'watermarked')
                assert initial == native['message_bits_str_initial']
                items = [x for x in plan['inputs'] if x['task'] == idx]
                source = Image.open(items[0]['path']).convert('RGB')
                reference = encode(source)
                cases = []
                for item in items:
                    query = Image.open(item['path']).convert('RGB')
                    stage = 'watermarked' if item['kind'] == 'clean' else 'attacked'
                    old = next(r for r in native_rows if int(r['image_idx']) == idx and r['stage'] == stage
                               and int(r['attack_step']) == item['checkpoint'])
                    cases.append((item['kind'], item['checkpoint'], query, float(old['bit_accuracy']), None))
                for c, condition in enumerate(plan['conditions']):
                    seed = plan['distortion_seed_base'] + idx * 100 + c
                    options = SimpleNamespace(**{x['name']: None for x in plan['conditions']})
                    setattr(options, condition['name'], condition['value'])
                    distort.set_random_seed(seed)
                    query = distort.image_distortion(source.copy(), seed, options).convert('RGB')
                    old = next(r for r in old_distortions if r['index'] == idx and r['condition'] == condition['name'] and not r['defended'])
                    assert hashlib.sha256(query.tobytes()).hexdigest() == old['pixels_sha256']
                    cases.append((condition['name'], -1, query, old['ba'], seed))
                for kind, checkpoint, query, old_ba, seed in cases:
                    set_random_seed(seed if seed is not None else 2026111319 + idx)
                    feature = encode(query)
                    z = provider.invert_images(query, num_inference_steps=50)['zT_torch']
                    recovered = wm.get_accuracies(z)
                    barcode = recovered['barcodes_torch'][0].cpu().numpy()
                    errors = barcode != truth[None, :]
                    assert errors.shape == (64, 256)
                    bits = np.array(list(recovered['message_bits_str_list'][0]), dtype=np.int8)
                    ba = float(np.mean(bits == truth))
                    assert ba == float(recovered['bit_accuracies'][0])
                    majority = barcode.mean(axis=0) > .5
                    disagreement = float((barcode != majority[None, :]).mean())
                    spatial = errors.reshape(4, 64, 64).reshape(4, 4, 16, 4, 16).mean(axis=(2, 4))
                    latent = run / 'latents' / f'{idx:03d}_{kind}_{checkpoint}.npz'
                    np.savez_compressed(latent, z=z.cpu().numpy())
                    rows.append(dict(task=idx, kind=kind, checkpoint=checkpoint, seed=seed,
                        pixels_sha256=hashlib.sha256(query.tobytes()).hexdigest(),
                        content_distance=float(np.clip(1 - reference @ feature, 0, 2)),
                        coordinate_ber=float(errors.mean()), repetition_disagreement=disagreement,
                        spatial_error_std=float(spatial.std()), latent_rms=float(z.square().mean().sqrt()),
                        ba=ba, native=ba > plan['carrier_threshold'], historical_ba=old_ba, ba_replay_equal=ba == old_ba,
                        latent=str(latent.relative_to(run)), latent_sha256=sha(latent)))
                    feature_rows.append(feature)
                write(run / 'progress.json', dict(completed_sources=idx + 1, rows=len(rows)))
                print('SOURCE_COMPLETE', idx, flush=True)
        assert len(rows) == 130
        write(run / 'rows.json', rows)
        np.savez_compressed(run / 'features.npz', features=np.stack(feature_rows))
        state.update(status='complete', rows=len(rows), seconds=time.time()-state['started_at'],
            exact_ba_replays=sum(r['ba_replay_equal'] for r in rows),
            artifacts={p: sha(run / p) for p in ['rows.json', 'features.npz']})
        write(run / 'complete.json', state)
        print(json.dumps(state), flush=True)
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        write(run / 'state.json', state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    assert ROOT.is_dir() and args.run.resolve().parent == ROOT / 'runs'
    run = args.run.resolve()
    if args.mode == 'prepare':
        prepare(run)
    else:
        plan = json.loads((run / 'protocol.json').read_text())
        assert sha(__file__) == plan['source_sha256']
        execute(run, plan)


if __name__ == '__main__':
    main()
