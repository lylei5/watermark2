"""Compare carrier-dependent spatial errors with equal-size key-free readouts.

Post-hoc development probe on existing, freshly measured detector latents.
No attacks, generation, carrier modifications or neural training.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path('/data0/liyanlei/watermark_research')
PARENT = ROOT / 'runs/carrier_content_channel_20261113'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare(run):
    run.mkdir(exist_ok=False)
    shutil.copy2(__file__, run / Path(__file__).name)
    plan = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), parent=str(PARENT),
        rows_sha256=sha(PARENT / 'rows.json'), parent_protocol_sha256=sha(PARENT / 'protocol.json'),
        scope='Post-hoc mechanism development; all configurations reported; no independent confirmation',
        readouts=['carrier_error16', 'latent_rms16', 'latent_mean_and_rms32', 'carrier_error_and_rms32'],
        partitions='4 latent channels x2x2 spatial cells,32x32 pixels per cell',
        model='Standardized ridge regression, mean squared loss plus lambda times squared coefficients',
        lambdas=[.1, 1.], fit=[0, 1, 2, 3], calibration=[4, 5, 6], evaluation=[7, 8, 9],
        threshold='Minimum residual threshold retaining26/27 calibration distortions',
        cross_validation='Leave one source out over90 genuine transformations',
        attack_labels_used_for_fit=False, new_or_modified_attacks=False,
        source_sha256=sha(run / Path(__file__).name))
    write(run / 'protocol.json', plan)
    print(json.dumps(dict(run=str(run), protocol_sha256=sha(run / 'protocol.json'))), flush=True)


def execute(run, plan):
    os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    import numpy as np
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
    from cryptography.hazmat.backends import default_backend
    assert sha(PARENT / 'rows.json') == plan['rows_sha256']
    assert sha(PARENT / 'protocol.json') == plan['parent_protocol_sha256']
    rows = json.loads((PARENT / 'rows.json').read_text())
    old = ROOT / 'runs/defense_revalidation_20261111'
    previous_plan = json.loads((old / 'protocol.json').read_text())
    for name in ['keys', 'nonces', 'messages']:
        relative = f'PGID/watermarks/{name}.py'
        assert sha(old / relative) == previous_plan['source_sha256'][relative]
    sys.path.insert(0, str(old / 'PGID'))
    from watermarks.keys import KEYS
    from watermarks.nonces import NONCES
    from watermarks.messages import MESSAGES
    carriers, rms, means = [], [], []
    for row in rows:
        idx = row['task']
        path = PARENT / row['latent']
        assert sha(path) == row['latent_sha256']
        z = np.load(path)['z'].reshape(4, 64, 64)
        encryptor = Cipher(algorithms.ChaCha20(KEYS[idx], NONCES[idx]), mode=None, backend=default_backend()).encryptor()
        encrypted = encryptor.update(MESSAGES[idx][:32]*64)+encryptor.finalize()
        expected = np.unpackbits(np.frombuffer(encrypted, dtype=np.uint8)).reshape(4, 64, 64)
        errors = ((z >= 0) != expected).astype(np.float64)
        assert float(errors.mean()) == row['coordinate_ber']
        def pool(value):
            return value.reshape(4, 2, 32, 2, 32).mean(axis=(2, 4)).reshape(-1)
        carriers.append(pool(errors))
        rms.append(np.sqrt(pool(z.astype(np.float64)**2)))
        means.append(pool(z.astype(np.float64)))
    carriers, rms, means = map(np.stack, [carriers, rms, means])
    readouts = dict(carrier_error16=carriers, latent_rms16=rms,
        latent_mean_and_rms32=np.concatenate([means, rms], axis=1),
        carrier_error_and_rms32=np.concatenate([carriers, rms], axis=1))
    np.savez_compressed(run / 'readouts.npz', **readouts)
    target = np.array([r['content_distance'] for r in rows])
    genuine = np.array([r['kind'] not in ['clean', 'imprint'] for r in rows])
    tasks = np.array([r['task'] for r in rows])

    def predict(features, fit, test, penalty):
        x, y = features[fit], target[fit]
        mu, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-8)
        x = (x-mu)/scale
        coefficients = np.linalg.solve(x.T@x/len(x)+penalty*np.eye(x.shape[1]), x.T@(y-y.mean())/len(x))
        return (features[test]-mu)/scale @ coefficients + y.mean()

    results = []
    for name in plan['readouts']:
        features = readouts[name]
        for penalty in plan['lambdas']:
            predicted = predict(features, genuine & np.isin(tasks, plan['fit']), np.ones(len(rows), dtype=bool), penalty)
            residuals = target-predicted
            calibration = residuals[genuine & np.isin(tasks, plan['calibration'])]
            threshold = float(np.sort(calibration)[math.ceil(.95*len(calibration))-1])
            errors, baseline_errors = [], []
            for task in range(10):
                fit, test = genuine & (tasks != task), genuine & (tasks == task)
                predictions = predict(features, fit, test, penalty)
                errors.extend((target[test]-predictions)**2)
                baseline_errors.extend((target[test]-target[fit].mean())**2)
            evaluation = []
            for kind in ['clean', 'distortions', 'imprint']:
                mask = np.array([r['task'] in plan['evaluation'] and
                    ((r['kind'] not in ['clean', 'imprint']) if kind == 'distortions' else
                     r['kind'] == kind and (kind != 'imprint' or r['checkpoint'] == 150)) for r in rows])
                evaluation.append(dict(kind=kind, total=int(mask.sum()),
                    accepted=sum(bool(r['native'] and residuals[i] <= threshold) for i,r in enumerate(rows) if mask[i])))
            results.append(dict(readout=name, dimensions=features.shape[1], penalty=penalty,
                source_held_out_r2=float(1-np.sum(errors)/np.sum(baseline_errors)),
                residual_threshold=threshold, evaluation=evaluation))
    write(run / 'complete.json', dict(status='complete', protocol_sha256=sha(run / 'protocol.json'),
        readouts_sha256=sha(run / 'readouts.npz'), coordinate_error_parity=130, results=results))
    print(json.dumps(results), flush=True)


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
