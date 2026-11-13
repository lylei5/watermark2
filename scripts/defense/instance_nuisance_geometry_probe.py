"""Measure shared edit directions using an ordinary subspace-removal control.

Fits only genuine residuals. This is a diagnostic baseline, not a new defense.
All CPU computation and private projection access are restricted to server 249.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

ROOT = Path('/data0/liyanlei/watermark_research')
PARENT = ROOT / 'runs/instance_representation_20261113_v2'
BINARY = ROOT / 'runs/sscd_binary_fe_20261113'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def prepare(run):
    run.mkdir(exist_ok=False)
    prior = json.loads((PARENT / 'protocol.json').read_text())
    binary = json.loads((BINARY / 'protocol.json').read_text())
    fit_sources = []
    for dataset in ['COCO', 'SDP']:
        selected = sorted(s['task'] for s in prior['sources'] if s['dataset'] == dataset and s['task'] in prior['calibration'])
        assert len(selected) == 25
        fit_sources += selected[:12]
    calibration = sorted(set(prior['calibration'])-set(fit_sources))
    projections = [str(Path(binary['private']) / f'projection_{i}.npy') for i in range(3)]
    shutil.copy2(__file__, run / Path(__file__).name)
    plan = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        purpose='Determine whether a few shared edit directions explain large instance-code recovery radii',
        parent=str(PARENT), parent_protocol_sha256=sha(PARENT / 'protocol.json'),
        features_sha256=sha(PARENT / 'sscd/features.npz'), metadata_sha256=sha(PARENT / 'sscd/metadata.json'),
        projections=projections, projection_sha256=[sha(p) for p in projections],
        fit=fit_sources, calibration=calibration, evaluation=prior['evaluation'], conditions=prior['conditions'],
        ranks=[0, 4, 16, 64], fit_examples=216, calibration_examples=234, evaluation_examples=450,
        method='Uncentered SVD of genuine query-reference residuals; remove leading right singular vectors and L2 normalize',
        quantizer='Existing private8192 hyperplane pool, select1024 largest reference margins, three repetitions',
        threshold='Independent95% calibration threshold for every configuration, all configurations reported',
        training='Linear nuisance subspace fit on24 sources; no neural training',
        wrong_instance_used_for_fitting=False, attack_generation=False, independent_confirmation=False,
        keys_recovered=False, source_sha256=sha(run / Path(__file__).name))
    write(run / 'protocol.json', plan)
    print(json.dumps(dict(run=str(run), protocol_sha256=sha(run / 'protocol.json'))), flush=True)


def execute(run, plan):
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    import numpy as np
    assert sha(PARENT / 'sscd/features.npz') == plan['features_sha256']
    assert sha(PARENT / 'sscd/metadata.json') == plan['metadata_sha256']
    features = np.load(PARENT / 'sscd/features.npz')['features'].astype(np.float64)
    keys = json.loads((PARENT / 'sscd/metadata.json').read_text())['keys']
    index = {tuple(k): i for i, k in enumerate(keys)}
    assert features.shape == (1300, 512)
    residuals = np.stack([features[index[i, c]]-features[index[i, 'bound']]
                          for i in plan['fit'] for c in plan['conditions']])
    assert residuals.shape == (216, 512)
    _, singular, basis = np.linalg.svd(residuals, full_matrices=False)
    np.savez_compressed(run / 'basis.npz', basis=basis, singular=singular)
    results, all_rows = [], []

    def summarize(distances, rank, metric, repetition=None):
        calibration = [distances[i, c] for i in plan['calibration'] for c in plan['conditions']]
        threshold = sorted(calibration)[math.ceil(.95*len(calibration))-1]
        result = dict(rank=rank, metric=metric, repetition=repetition, threshold=threshold,
                      fit_residual_energy_removed=float(np.sum(singular[:rank]**2)/np.sum(singular**2)), groups={})
        for split in ['calibration', 'evaluation']:
            groups = {}
            for condition in ['bound', *plan['conditions'], 'same']:
                values = [distances[i, condition] for i in plan[split]]
                groups[condition] = dict(accepted=sum(v <= threshold for v in values), total=len(values))
                all_rows.extend(dict(task=i, split=split, condition=condition, rank=rank, metric=metric,
                    repetition=repetition, distance=distances[i, condition], accepted=distances[i, condition] <= threshold)
                    for i in plan[split])
            groups['pooled_distortions'] = dict(accepted=sum(groups[c]['accepted'] for c in plan['conditions']),
                                                total=len(plan[split])*9)
            result['groups'][split] = groups
        results.append(result)

    for rank in plan['ranks']:
        transformed = features - (features @ basis[:rank].T) @ basis[:rank]
        norms = np.linalg.norm(transformed, axis=1, keepdims=True)
        assert float(norms.min()) > 1e-8
        transformed /= norms
        distances = {(i, c): float(np.clip(1-transformed[index[i, 'bound']] @ transformed[index[i, c]], 0, 2))
                     for i in range(100) for c in ['bound', *plan['conditions'], 'same']}
        summarize(distances, rank, 'cosine')
        for repetition, path in enumerate(plan['projections']):
            assert sha(path) == plan['projection_sha256'][repetition]
            matrix = np.load(path)
            projected = transformed.astype(np.float32) @ matrix
            distances = {}
            for task in range(100):
                reference = projected[index[task, 'bound']]
                chosen = np.argsort(-np.abs(reference), kind='stable')[:1024]
                code = reference[chosen] >= 0
                for condition in ['bound', *plan['conditions'], 'same']:
                    distances[task, condition] = int(np.count_nonzero((projected[index[task, condition], chosen] >= 0) != code))
            summarize(distances, rank, 'margin_hamming1024', repetition)
    write(run / 'rows.json', all_rows)
    write(run / 'complete.json', dict(status='complete', protocol_sha256=sha(run / 'protocol.json'),
        basis_sha256=sha(run / 'basis.npz'), rows_sha256=sha(run / 'rows.json'), results=results))
    print(json.dumps([dict(rank=r['rank'], metric=r['metric'], repetition=r['repetition'], threshold=r['threshold'],
        energy_removed=r['fit_residual_energy_removed'], genuine=r['groups']['evaluation']['pooled_distortions'],
        wrong_instance=r['groups']['evaluation']['same']) for r in results]), flush=True)


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
