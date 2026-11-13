"""Evaluate a fixed SSCD reference gate after native PGID jobs exit.

This is a simple reference-based defense control, with no training or attacks.
Native carrier decisions come from the fresh, unchanged full-budget PGID run.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import time

ROOT = Path('/data0/liyanlei/watermark_research')
PARENT = ROOT / 'runs/pgid_carrier_coverage_20261113'
PROBE = ROOT / 'runs/instance_representation_20261113_v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def prepare(run):
    run.mkdir(exist_ok=False)
    scores = json.loads((PROBE / 'scores.json').read_text())
    probe_plan = json.loads((PROBE / 'protocol.json').read_text())
    selected = next(x for x in scores['results'] if x['method'] == 'sscd_cosine' and x['anchor'] == 'bound')
    predecessors = []
    for carrier in ['TR', 'PRC']:
        state = json.loads((PARENT / carrier / 'state.json').read_text())
        pid = state['pid']
        path = Path(f'/proc/{pid}/cmdline')
        command = path.read_bytes() if path.exists() else b''
        assert not command or str(PARENT).encode() in command
        predecessors.append(dict(carrier=carrier, pid=pid, cmdline_sha256=hashlib.sha256(command).hexdigest()))
    shutil.copy2(__file__, run / Path(__file__).name)
    plan = dict(parent=str(PARENT), parent_protocol_sha256=sha(PARENT / 'protocol.json'),
        probe_protocol_sha256=sha(PROBE / 'protocol.json'), threshold_source_sha256=sha(PROBE / 'scores.json'),
        threshold=selected['threshold'], operator='1-cosine <= threshold',
        model=json.loads((PROBE / 'sscd/model_manifest.json').read_text()),
        distortion_file=probe_plan['distortion_file'], distortion_sha256=probe_plan['distortion_sha256'],
        conditions=probe_plan['conditions'], seed_base=2026111310,
        predecessors=predecessors, gpu=7, method='Native detector AND SSCD distance to registered final image',
        scope='Fresh author TR/PRC output images; current development transfer of fixed SSCD threshold',
        training=False, attacks_created=0, new_or_modified_attacks=False,
        independent_confirmation=False, selection='SSCD selected after representation development results',
        distortion_scope='Content gate only; no native carrier or PGID distortion re-inversion in this control',
        clean_pgid_scope='Native run validates clean raw carrier only; PGID clean control is pending',
        source_sha256=sha(run / Path(__file__).name))
    write(run / 'protocol.json', plan)
    print(json.dumps(dict(run=str(run), protocol_sha256=sha(run / 'protocol.json'))), flush=True)


def wait_for_exit(previous):
    assert os.uname().machine == 'x86_64'
    libc = ctypes.CDLL(None, use_errno=True)
    descriptor = libc.syscall(434, previous['pid'], 0)
    if descriptor < 0:
        if ctypes.get_errno() != 3:
            raise OSError(ctypes.get_errno(), 'pidfd_open')
        return
    try:
        path = Path(f"/proc/{previous['pid']}/cmdline")
        command = path.read_bytes() if path.exists() else b''
        if command:
            assert hashlib.sha256(command).hexdigest() == previous['cmdline_sha256'], 'PID reused'
        select.select([descriptor], [], [])
    finally:
        os.close(descriptor)


def execute(run, plan):
    state = dict(status='waiting_for_native_exit', pid=os.getpid(), protocol_sha256=sha(run / 'protocol.json'))
    assert not (run / 'state.json').exists()
    write(run / 'state.json', state)
    try:
        for previous in plan['predecessors']:
            wait_for_exit(previous)
        completions = {}
        for carrier in ['TR', 'PRC']:
            out = PARENT / carrier
            completed = json.loads((out / 'complete.json').read_text())
            assert completed['status'] == 'complete'
            assert completed['protocol_sha256'] == plan['parent_protocol_sha256']
            assert sha(out / 'attack=forgery/all_results.csv') == completed['raw_sha256']
            for name, digest in completed['image_sha256'].items():
                assert sha(out / name) == digest
            completions[carrier] = completed
        fields = subprocess.check_output(['nvidia-smi', '-i', '7',
            '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
        assert int(fields[0]) == 7 and int(fields[2]) < 100 and int(fields[3]) == 0, fields
        os.environ.update(CUDA_VISIBLE_DEVICES=fields[1], OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from torchvision import transforms
        from types import SimpleNamespace
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        assert sha(plan['model']['path']) == plan['model']['sha256']
        model = torch.jit.load(plan['model']['path'], map_location='cuda').eval()
        transform = transforms.Compose([transforms.Resize([320, 320]), transforms.ToTensor(),
            transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
        assert sha(plan['distortion_file']) == plan['distortion_sha256']
        spec = importlib.util.spec_from_file_location('gate_distort', plan['distortion_file'])
        distort = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(distort)
        work = run / 'work'
        work.mkdir()
        os.chdir(work)
        state.update(status='running', gpu=7, gpu_uuid=fields[1], started_at=time.time())
        write(run / 'state.json', state)
        rows, feature_rows = [], []
        with torch.no_grad():
            def encode(image):
                return F.normalize(model(transform(image).unsqueeze(0).to('cuda')), dim=1).cpu().numpy()[0]
            for carrier_index, carrier in enumerate(['TR', 'PRC']):
                native = {(r['task'], r['stage'], r['checkpoint']): r['accepted'] for r in completions[carrier]['rows']}
                directory = PARENT / carrier / 'attack=forgery'
                for task in range(10):
                    source_path = directory / f'watermarked_images/image_{task:03d}.png'
                    with Image.open(source_path) as opened:
                        source = opened.convert('RGB')
                    reference = encode(source)
                    cases = [('clean', -1, source, sha(source_path))]
                    for checkpoint in [50, 100, 150]:
                        path = directory / f'attacked_images_step_{checkpoint}/image_{task:03d}.png'
                        with Image.open(path) as opened:
                            cases.append(('imprint', checkpoint, opened.convert('RGB'), sha(path)))
                    for condition_index, (name, value) in enumerate(plan['conditions'].items()):
                        options = SimpleNamespace(**{k: None for k in plan['conditions']})
                        setattr(options, name, value)
                        seed = plan['seed_base'] + carrier_index * 10000 + task * 100 + condition_index
                        distort.set_random_seed(seed)
                        image = distort.image_distortion(source.copy(), seed, options).convert('RGB')
                        cases.append((name, -1, image, hashlib.sha256(image.tobytes()).hexdigest()))
                    for kind, checkpoint, image, digest in cases:
                        feature = encode(image)
                        distance = float(np.clip(1 - np.dot(reference, feature), 0, 2))
                        gate = distance <= plan['threshold']
                        raw = native[task, 'watermarked', -1] if kind == 'clean' else native[task, 'attacked', checkpoint] if kind == 'imprint' else None
                        pgid = native[task, 'defended', checkpoint] if kind == 'imprint' else None
                        rows.append(dict(carrier=carrier, task=task, kind=kind, checkpoint=checkpoint,
                            image_sha256=digest, distance=distance, content_gate=gate, raw_carrier=raw,
                            native_and_gate=(bool(raw and gate) if raw is not None else None), pgid=pgid))
                        feature_rows.append(feature)
        assert len(rows) == 260
        summary = []
        for carrier in ['TR', 'PRC']:
            for kind, checkpoint in [('clean', -1)] + [('imprint', c) for c in [50, 100, 150]] + [(c, -1) for c in plan['conditions']]:
                part = [r for r in rows if r['carrier'] == carrier and r['kind'] == kind and r['checkpoint'] == checkpoint]
                assert len(part) == 10
                summary.append(dict(carrier=carrier, kind=kind, checkpoint=checkpoint, total=10,
                    **{k: (sum(r[k] for r in part) if all(r[k] is not None for r in part) else None)
                       for k in ['content_gate', 'raw_carrier', 'native_and_gate', 'pgid']}))
        write(run / 'rows.json', rows)
        np.savez_compressed(run / 'features.npz', features=np.stack(feature_rows))
        state.update(status='complete', rows=260, summary=summary, seconds=time.time()-state['started_at'],
            native_complete_sha256={c: sha(PARENT / c / 'complete.json') for c in ['TR', 'PRC']},
            artifacts={n: sha(run / n) for n in ['rows.json', 'features.npz']})
        write(run / 'complete.json', state)
        print(json.dumps(summary), flush=True)
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        write(run / 'state.json', {k: v for k, v in state.items() if k != 'summary'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    assert ROOT.is_dir() and run.parent == ROOT / 'runs'
    if args.mode == 'prepare':
        prepare(run)
    else:
        plan = json.loads((run / 'protocol.json').read_text())
        assert sha(__file__) == plan['source_sha256']
        execute(run, plan)


if __name__ == '__main__':
    main()
