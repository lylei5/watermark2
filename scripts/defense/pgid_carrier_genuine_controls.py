"""Native TR/PRC genuine and cover controls after full Imprint workers exit.

Runs unchanged upstream detector/PGID functions and GS image transformations.
SSCD uses the previously frozen final-reference threshold. No attack generation.
"""
import argparse
import csv
import ctypes
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path('/data0/liyanlei/watermark_research')
PARENT = ROOT / 'runs/pgid_carrier_coverage_20261113'
GATE = ROOT / 'runs/sscd_pgid_reference_gate_20261113'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare(run):
    run.mkdir(exist_ok=False)
    parent = json.loads((PARENT / 'protocol.json').read_text())
    gate = json.loads((GATE / 'protocol.json').read_text())
    predecessors = []
    for job in parent['jobs']:
        state = json.loads((PARENT / job['carrier'] / 'state.json').read_text())
        path = Path(f"/proc/{state['pid']}/cmdline")
        cmdline = path.read_bytes() if path.exists() else b''
        assert not cmdline or str(PARENT).encode() in cmdline
        predecessors.append(dict(carrier=job['carrier'], gpu=job['gpu'], pid=state['pid'],
                                 cmdline_sha256=hashlib.sha256(cmdline).hexdigest()))
    prc_key = Path(parent['source']['path']) / 'watermarks/prc_key.pkl'
    assert prc_key.is_file()
    shutil.copy2(__file__, run / Path(__file__).name)
    plan = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), parent=str(PARENT),
        parent_protocol_sha256=sha(PARENT / 'protocol.json'), gate_protocol_sha256=sha(GATE / 'protocol.json'),
        source=parent['source'], predecessors=predecessors, prc_key_path=str(prc_key), prc_key_sha256=sha(prc_key),
        conditions=gate['conditions'], seed_base=gate['seed_base'], model=gate['model'], threshold=gate['threshold'],
        distortion_file=gate['distortion_file'], distortion_sha256=gate['distortion_sha256'],
        model_revision=parent['revision'], target_schedulers=parent['target_schedulers'],
        sources_per_carrier=10, queries_per_carrier=110, decisions_per_carrier=220,
        genuine_queries_per_carrier=100, cover_queries_per_carrier=10,
        scope='Native carrier, PGID-F and frozen SSCD gate on the same clean/GS-transformed/cover queries',
        clean_failures_retained=True, threshold_tuning=False, attack_generation=False,
        native_functions=['configure_prc', 'configure_PGID', 'validate', 'PGIDPipeline.run_defense'],
        prc_raw_inversion='Author exact_inversion with decode_inv=True and configured DPM',
        prc_defended_inversion='Author PGIDPipeline.run_defense result passed directly to validate',
        cohort='Same10 development sources per carrier as full-budget Imprint coverage',
        source_sha256=sha(run / Path(__file__).name))
    write(run / 'protocol.json', plan)
    print(json.dumps(dict(run=str(run), protocol_sha256=sha(run / 'protocol.json'))), flush=True)


def wait(previous):
    libc = ctypes.CDLL(None, use_errno=True)
    descriptor = libc.syscall(434, previous['pid'], 0)
    if descriptor < 0:
        if ctypes.get_errno() != 3:
            raise OSError(ctypes.get_errno(), 'pidfd_open')
        return
    try:
        path = Path(f"/proc/{previous['pid']}/cmdline")
        cmdline = path.read_bytes() if path.exists() else b''
        assert not cmdline or hashlib.sha256(cmdline).hexdigest() == previous['cmdline_sha256']
        select.select([descriptor], [], [])
    finally:
        os.close(descriptor)


def execute(run, carrier, plan):
    out = run / carrier
    out.mkdir(exist_ok=False)
    previous = next(j for j in plan['predecessors'] if j['carrier'] == carrier)
    state = dict(status='waiting_for_native_exit', pid=os.getpid(), parent_pid=previous['pid'],
                 carrier=carrier, protocol_sha256=sha(run / 'protocol.json'))
    write(out / 'state.json', state)
    try:
        wait(previous)
        done = json.loads((PARENT / carrier / 'complete.json').read_text())
        assert done['status'] == 'complete' and done['protocol_sha256'] == plan['parent_protocol_sha256']
        assert sha(PARENT / 'protocol.json') == plan['parent_protocol_sha256']
        assert sha(GATE / 'protocol.json') == plan['gate_protocol_sha256']
        parent = json.loads((PARENT / 'protocol.json').read_text())
        for name, digest in done['image_sha256'].items():
            assert sha(PARENT / carrier / name) == digest, name
        csv_path = PARENT / carrier / 'attack=forgery/all_results.csv'
        assert sha(csv_path) == done['raw_sha256']
        source = Path(plan['source']['path'])
        for name, digest in plan['source']['files'].items():
            assert sha(source / name) == digest, name
        assert sha(plan['prc_key_path']) == plan['prc_key_sha256']
        assert sha(plan['distortion_file']) == plan['distortion_sha256']
        assert sha(plan['model']['path']) == plan['model']['sha256']
        ref = ROOT / 'cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/refs/main'
        assert ref.read_text().strip() == plan['model_revision']
        gpu = previous['gpu']
        assert gpu in [5, 4]
        fields = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
        assert int(fields[0]) == gpu and int(fields[2]) < 100 and int(fields[3]) == 0, fields
        os.environ.update(CUDA_VISIBLE_DEVICES=fields[1], HF_HOME=str(ROOT / 'cache/huggingface'),
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
            OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        sys.path.insert(0, str(source))
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from torchvision import transforms
        from pipe import pipe_utils
        from pgid_defense import PGIDPipeline
        from watermarks.tr_watermark import parser as tr_parser
        from watermarks.prc_watermark import parser as prc_parser
        from watermarks.watermark_utils import SemanticWatermarks, validate, get_detection_threshold
        from utils import get_watermark_args, set_random_seed, configure_prc, configure_PGID
        torch.set_num_threads(4)
        # Retain the author's backend defaults used by the parent full-budget job.
        cfg = (tr_parser if carrier == 'TR' else prc_parser).parse_args([])
        cfg.wm_type, cfg.attack_type = carrier, 'forgery'
        cfg.scheduler_target = plan['target_schedulers'][carrier]
        cfg.prc_key_path = plan['prc_key_path']
        set_random_seed(parent['seed'])
        kwargs = get_watermark_args(cfg, 'cuda:0')
        provider = pipe_utils.create_diffusion_pipe(pretrained_model_name_or_path=parent['model'], resolution=512,
            device='cuda:0', scheduler_name=cfg.scheduler_target, scheduler_inverse_name=cfg.scheduler_target,
            eager_loading=True, disable_tqdm=True)
        configure_prc(cfg, provider)
        configure_PGID(cfg)
        assert cfg.stopping_timestep == parent['defense']['k'][carrier]
        defense = PGIDPipeline(provider, provider.scheduler.config, num_inference_steps=cfg.num_defense_steps,
            stopping_timestep=cfg.stopping_timestep, num_skip_inversion=cfg.num_skip_inversion,
            guidance_strength=cfg.guidance_strength)
        threshold = get_detection_threshold(carrier, parent['model'])
        if carrier == 'TR':
            assert threshold == parent['thresholds']['TR']['value']
        model = torch.jit.load(plan['model']['path'], map_location='cuda').eval()
        transform = transforms.Compose([transforms.Resize([320, 320]), transforms.ToTensor(),
            transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
        def encode(image):
            with torch.no_grad():
                return F.normalize(model(transform(image).unsqueeze(0).to('cuda')), dim=1).cpu().numpy()[0]
        spec = importlib.util.spec_from_file_location('genuine_distortions', plan['distortion_file'])
        distort = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(distort)
        (out / 'work').mkdir()
        os.chdir(out / 'work')
        state.update(status='running', gpu=gpu, gpu_uuid=fields[1], started_at=time.time())
        write(out / 'state.json', state)
        native = list(csv.DictReader(csv_path.open()))
        rows = []
        for wm, idx in SemanticWatermarks[carrier].value.generate_watermark_signals(total_num_signals=10,
                latent_shape=provider.get_latent_shape(), batch_size=1, target_start_index=0,
                target_end_index=10, w_seed=parent['seed'], **kwargs):
            source_path = PARENT / carrier / f'attack=forgery/watermarked_images/image_{idx:03d}.png'
            clean = Image.open(source_path).convert('RGB')
            reference = encode(clean)
            item = next(x for x in parent['inputs'] if x['index'] == idx)
            cover = Path(parent['cover_images_dir']) / Path(item['cover']).name
            assert sha(cover) == item['cover_sha256']
            queries = [('clean', clean), ('cover', Image.open(cover).resize((512, 512)).convert('RGB'))]
            for c, (name, value) in enumerate(plan['conditions'].items()):
                options = SimpleNamespace(**{k: None for k in plan['conditions']})
                setattr(options, name, value)
                seed = plan['seed_base'] + ['TR', 'PRC'].index(carrier)*10000 + idx*100 + c
                distort.set_random_seed(seed)
                query = distort.image_distortion(clean.copy(), seed, options).convert('RGB')
                queries.append((name, query))
            original = next(x for x in native if int(x['image_idx']) == idx and x['stage'] == 'watermarked')
            initial = original['message_bits_str_initial'] or None
            if carrier == 'PRC':
                assert initial == ''.join(format(b, '08b') for b in wm.message)
            for kind, query in queries:
                distance = float(np.clip(1-reference @ encode(query), 0, 2))
                content_gate = distance <= plan['threshold']
                for defended in [False, True]:
                    set_random_seed(parent['seed'])
                    z = defense.run_defense(query)['zT'] if defended else None
                    result = validate(query, wm, provider, 50, initial, provided_zT=z)
                    metric = float(result['p_value']) if carrier == 'TR' else bool(result['PRC_detection_results'])
                    accepted = metric < threshold if carrier == 'TR' else metric
                    rows.append(dict(task=idx, kind=kind, defended=defended,
                        pixels_sha256=hashlib.sha256(query.tobytes()).hexdigest(),
                        metric=metric, accepted=accepted, content_distance=distance, content_gate=content_gate,
                        combined=bool(accepted and content_gate),
                        clean_native_replay_equal=(accepted == (original['detection_successful'] == 'True')
                                                   if kind == 'clean' and not defended else None)))
            print('CONTROL_SOURCE_COMPLETE', carrier, idx, flush=True)
        assert len(rows) == 220
        summary = []
        for kind in ['clean', 'cover', 'all_distortions', *plan['conditions']]:
            for defended in [False, True]:
                part = [r for r in rows if r['defended'] == defended and
                        (r['kind'] in plan['conditions'] if kind == 'all_distortions' else r['kind'] == kind)]
                summary.append(dict(kind=kind, defended=defended, total=len(part),
                    **{k: sum(r[k] for r in part) for k in ['accepted', 'content_gate', 'combined']}))
        write(out / 'rows.json', rows)
        state.update(status='complete', summary=summary, rows=220, rows_sha256=sha(out / 'rows.json'),
                     native_complete_sha256=sha(PARENT / carrier / 'complete.json'), seconds=time.time()-state['started_at'])
        write(out / 'complete.json', state)
        print(json.dumps(dict(carrier=carrier, summary=summary)), flush=True)
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        write(out / 'state.json', {k:v for k,v in state.items() if k != 'summary'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'TR', 'PRC'])
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    assert ROOT.is_dir() and run.parent == ROOT / 'runs'
    if args.mode == 'prepare':
        prepare(run)
    else:
        plan = json.loads((run / 'protocol.json').read_text())
        assert sha(__file__) == plan['source_sha256']
        execute(run, args.mode, plan)


if __name__ == '__main__':
    main()
