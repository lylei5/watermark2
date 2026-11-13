"""Server CPU diagnostic: SSCD quantization and bounded-distance code feasibility.

Compares fixed random hyperplanes with enrollment-only high-margin selection.
These are existing quantization ideas, not a new defense or an implemented FE.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil

ROOT = Path('/data0/liyanlei/watermark_research')
PARENT = ROOT / 'runs/instance_representation_20261113_v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(run):
    run.mkdir(exist_ok=False)
    private = ROOT / 'private_eval' / run.name
    private.mkdir(exist_ok=False, mode=0o700)
    seeds = [secrets.token_hex(32) for _ in range(3)]
    write(private / 'projection_seeds.json', seeds)
    os.chmod(private / 'projection_seeds.json', 0o600)
    prior = json.loads((PARENT / 'protocol.json').read_text())
    shutil.copy2(__file__, run / Path(__file__).name)
    plan = dict(parent=str(PARENT), parent_protocol_sha256=sha(PARENT / 'protocol.json'),
        feature_sha256=sha(PARENT / 'sscd/features.npz'), metadata_sha256=sha(PARENT / 'sscd/metadata.json'),
        seed_sha256=sha(private / 'projection_seeds.json'), private=str(private), repetitions=3,
        pool_size=8192, code_lengths=[256, 1024], selectors=['fixed', 'enrollment_margin'],
        enrollment='Select largest absolute normalized projection margins on final clean reference only',
        calibration=prior['calibration'], evaluation=prior['evaluation'], conditions=prior['conditions'],
        threshold='Smallest Hamming radius covering at least95% of450 calibration distortions, separately for every projection/selector/length',
        analysis='Report all12 configurations without choosing a best seed; existing development cohort',
        training=False, fuzzy_extractor_implemented=False, attack_jobs=0,
        code_bound='Binary code-offset unique correction of arbitrary t errors requires d>=2t+1; if2d>n, Plotkin bounds M<=floor(2d/(2d-n))',
        source_sha256=sha(run / Path(__file__).name))
    write(run / 'protocol.json', plan)
    print(json.dumps(dict(run=str(run), protocol_sha256=sha(run / 'protocol.json'))), flush=True)


def execute(run, plan):
    os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', OMP_NUM_THREADS='4')
    import numpy as np
    assert sha(PARENT / 'sscd/features.npz') == plan['feature_sha256']
    assert sha(PARENT / 'sscd/metadata.json') == plan['metadata_sha256']
    seedfile = Path(plan['private']) / 'projection_seeds.json'
    assert sha(seedfile) == plan['seed_sha256']
    features = np.load(PARENT / 'sscd/features.npz')['features']
    keys = json.loads((PARENT / 'sscd/metadata.json').read_text())['keys']
    index = {tuple(k): i for i, k in enumerate(keys)}
    assert features.shape == (1300, 512)
    results, all_rows, projection_hashes = [], [], []
    for repetition, seed in enumerate(json.loads(seedfile.read_text())):
        rng = np.random.default_rng(int(seed, 16))
        matrix = rng.standard_normal((512, plan['pool_size']), dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=0, keepdims=True)
        path = Path(plan['private']) / f'projection_{repetition}.npy'
        assert not path.exists()
        np.save(path, matrix)
        os.chmod(path, 0o600)
        projection_hashes.append(sha(path))
        projected = features @ matrix
        for length in plan['code_lengths']:
            for selector in plan['selectors']:
                distances = {}
                for task in range(100):
                    reference = projected[index[task, 'bound']]
                    chosen = np.arange(length) if selector == 'fixed' else np.argsort(-np.abs(reference), kind='stable')[:length]
                    code = reference[chosen] >= 0
                    for condition in ['bound', *plan['conditions'], 'same']:
                        query = projected[index[task, condition], chosen] >= 0
                        distances[task, condition] = int(np.count_nonzero(query != code))
                    assert distances[task, 'bound'] == 0
                calibration = [distances[i, c] for i in plan['calibration'] for c in plan['conditions']]
                radius = sorted(calibration)[math.ceil(.95 * len(calibration)) - 1]
                assert sum(d <= radius for d in calibration) >= 428
                dmin = 2 * radius + 1
                bound = math.floor(2 * dmin / (2 * dmin - length)) if 2 * dmin > length else None
                result = dict(repetition=repetition, length=length, selector=selector,
                    radius=radius, radius_fraction=radius/length,
                    plotkin_max_codewords=bound,
                    plotkin_max_message_bits=(math.floor(math.log2(bound)) if bound is not None else None),
                    calibration_accepted=sum(d <= radius for d in calibration), calibration_total=len(calibration), groups={})
                for split in ['calibration', 'evaluation']:
                    group = {}
                    for condition in ['bound', *plan['conditions'], 'same']:
                        values = [distances[i, condition] for i in plan[split]]
                        group[condition] = dict(accepted=sum(d <= radius for d in values), total=len(values),
                            quantiles=np.quantile(values, [0, .05, .5, .95, 1]).tolist())
                        all_rows.extend(dict(repetition=repetition, length=length, selector=selector,
                            split=split, task=i, condition=condition, distance=d, accepted=d <= radius)
                            for i, d in zip(plan[split], values))
                    group['pooled_distortions'] = dict(accepted=sum(group[c]['accepted'] for c in plan['conditions']), total=450)
                    result['groups'][split] = group
                results.append(result)
    write(run / 'rows.json', all_rows)
    write(run / 'complete.json', dict(status='complete', protocol_sha256=sha(run / 'protocol.json'),
        results=results, projection_sha256=projection_hashes, rows_sha256=sha(run / 'rows.json'),
        caveat='Code-offset worst-case unique-decoding bound; no conclusion about every FE construction, list decoding, random-error decoding or conditional cryptographic entropy'))
    print(json.dumps([{k:v for k,v in r.items() if k!='groups'} | dict(evaluation={k:r['groups']['evaluation'][k] for k in ['pooled_distortions','same']}) for r in results]), flush=True)


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
