"""Freeze six PRC/PRC-S image-chain checks with new private identities on249."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import secrets
import shutil
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--after-run', required=True)
    args = parser.parse_args()
    root = Path('/data0/liyanlei/watermark_research')
    assert root.is_dir() and os.name == 'posix'
    os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='4', NUMBA_NUM_THREADS='4')
    sys.path.insert(0, str(root / 'envs/phase1_prc_extra'))
    import numpy as np
    import torch
    torch.set_num_threads(4)
    run = (root / 'runs' / args.run_name).resolve()
    assert run.parent == root / 'runs'
    run.mkdir(exist_ok=False)
    private = root / 'private_eval' / run.name
    private.mkdir(mode=0o700, exist_ok=False)
    official = root / 'third_party/frozen/SemBind_337c9f9'
    modules = {}
    for name in ['prc', 'pseudogaussians']:
        spec = importlib.util.spec_from_file_location(name, official / 'src' / (name + '.py'))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        modules[name] = m
    encoding, decoding = modules['prc'].KeyGen(16384, message_length=512, false_positive_rate=1e-6, t=3)
    with (private / 'keys.pkl').open('wb') as stream:
        pickle.dump((encoding, decoding), stream)
    (private / 'keys.pkl').chmod(0o600)
    (private / 'binding.json').write_text(json.dumps(dict(key=secrets.token_hex(32))) + '\n')
    (private / 'binding.json').chmod(0o600)
    parent = root / 'runs/phase1_trajectory_native_gs_20261113'
    previous = json.loads((parent / 'protocol.json').read_text())
    sources = []
    for task, original in enumerate([0, 1, 2, 50, 51, 52]):
        item = next(x for x in previous['sources'] if x['task'] == original)
        records = json.loads((parent / f'generate_{original % 3}/complete.json').read_text())['rows']
        generated = next(x for x in records if x['task'] == original)
        message = np.asarray([secrets.randbits(1) for _ in range(512)], dtype=np.uint8)
        codeword = modules['prc'].Encode(encoding, message)
        initial = modules['pseudogaussians'].sample(codeword).float()
        posterior = modules['pseudogaussians'].recover_posteriors(initial.double(), variances=1.5)
        assert modules['prc'].Detect(decoding, posterior)
        assert np.array_equal(modules['prc'].Decode(decoding, posterior), message)
        target = private / f'{task:03}.npz'
        np.savez_compressed(target, initial=initial.numpy(), codeword=codeword.numpy(), message=message)
        target.chmod(0o600)
        sources.append(dict(task=task, source_task=original, dataset=item['dataset'], prompt=item['caption'],
                            auxiliary=str(parent / generated['images']['aux']['path']),
                            auxiliary_sha256=generated['images']['aux']['sha256'], private_carrier_sha256=sha(target)))
        print('PRIVATE_CARRIER_COMPLETE', task, flush=True)
    names = ['run_sembind_prc_native.py', 'run_baseline_after_exit.py', 'prepare_sembind_prc_native.py',
             'sembind_binding.py', 'semantic_code.py', 'sembind_official_audit.py', 'codeword_binding.py']
    for name in names:
        shutil.copy2(Path(__file__).parent / name, run / name)
    wait_run = (root / 'runs' / args.after_run).resolve()
    assert wait_run.parent == root / 'runs'
    pid = json.loads((wait_run / 'pipeline_state.json').read_text())['pid']
    path = Path(f'/proc/{pid}/cmdline')
    command = path.read_bytes() if path.exists() else b''
    assert not command or str(wait_run).encode() in command
    predecessor = dict(pid=pid, cmdline_sha256=hashlib.sha256(command).hexdigest())
    paths = dict(previous['paths'], official=str(official))
    plan = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                scope='6-source normal-image and unwatermarked PRC/PRC-S integration; paper-specified binding, not author full pipeline equivalence',
                python=str(root / 'envs/semantic_forgery_py310/bin/python'),
                paths=paths, private=str(private), private_key_sha256=sha(private / 'keys.pkl'),
                binding_key_sha256=sha(private / 'binding.json'),
                official_sha256={str(p.relative_to(official)): sha(p) for p in official.rglob('*.py')},
                scheduler_sha256=sha(paths['scheduler']), source_commit='337c9f90a150157ac1d410208ae20e6530ca8cdd',
                local_sha256={name: sha(run / name) for name in names},
                worker='run_sembind_prc_native.py', worker_sha256=sha(run / 'run_sembind_prc_native.py'),
                coordinator_sha256=sha(run / 'run_baseline_after_exit.py'),
                sources=sources, parent_protocol_sha256=sha(parent / 'protocol.json'),
                auxiliary_source='Previously generated original normal DPM image, hash checked and re-encoded with the released semantic network',
                sigma=.5, mask='Tile1024bits16times; zero suffix after8192; private HMAC permutation; signs1-2m',
                deterministic_backend=True,
                prc=dict(n=16384, message_length=512, t=3, nominal_fpr=1e-6, posterior_variance=1.5,
                         acceptance='Original Detect OR non-None Decode; report each separately; union FPR not established'),
                generation=dict(steps=50, cfg=7.5, dtype='float32', solver_order=1),
                inversion=dict(steps=50, prompt='', cfg=3, inv_order=0, decoder_inv=True, decoder_steps=100),
                gpu_indices=[7, 5, 4], predecessors=[predecessor] * 3, attack_jobs=0,
                private_carrier_roundtrips=6)
    (run / 'protocol.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(run=str(run), protocol_sha256=sha(run / 'protocol.json'))))


if __name__ == '__main__':
    main()
