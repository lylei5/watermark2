"""Validate score/threshold interfaces on server249; development data only."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys

ROOT = Path('/data0/liyanlei/watermark_research')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    assert output.parent == ROOT / 'runs'
    output.mkdir(exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', NUMBA_NUM_THREADS='4')
    sys.path.insert(0, str(ROOT / 'envs/phase1_prc_extra'))
    import numpy as np
    import torch
    from empirical_calibration import prc_detect_score, calibrate_upper_tail, exact_binomial_interval
    torch.set_num_threads(4)
    run = ROOT / 'runs/unified_runtime_probe_20261115'
    plan = json.loads((run / 'protocol.json').read_text())
    cfg = plan['methods']['sembind']
    official = Path(cfg['paths']['official'])
    prc_path = official / 'src/prc.py'
    gaussian_path = official / 'src/pseudogaussians.py'
    assert sha(prc_path) == cfg['official_sha256']['src/prc.py']
    assert sha(gaussian_path) == cfg['official_sha256']['src/pseudogaussians.py']
    prc = load('calibration_native_prc', prc_path)
    gaussian = load('calibration_native_gaussian', gaussian_path)
    key_file = Path(cfg['private']) / 'keys.pkl'
    assert sha(key_file) == cfg['private_key_sha256']
    with key_file.open('rb') as stream:
        _, decoding_key = pickle.load(stream)
    cases = []
    for method in ['pgid', 'sembind', 'seal']:
        complete = json.loads((run / method / 'complete.json').read_text())
        for row in complete['measurements']:
            path = run / method / row['artifact']
            assert sha(path) == row['artifact_sha256']
            with np.load(path) as data:
                latent = torch.as_tensor(data['recovered'].astype(np.float64)).flatten()
            cases.append((method + '/' + row['artifact'], gaussian.recover_posteriors(latent, variances=1.5)))
    for item in cfg['sources']:
        path = Path(cfg['private']) / f"{item['task']:03}.npz"
        assert sha(path) == item['private_carrier_sha256']
        with np.load(path) as data:
            latent = torch.as_tensor(data['initial'].astype(np.float64)).flatten()
        cases.append(('native_positive_latent/' + str(item['task']), gaussian.recover_posteriors(latent, variances=1.5)))
    cases.append(('zero_posterior_edge', torch.zeros(16384, dtype=torch.float64)))
    rows = []
    for name, posterior in cases:
        score = prc_detect_score(decoding_key, posterior)
        results = []
        for nominal in [.9, .5, .01, .001, 1e-6]:
            native = bool(prc.Detect(decoding_key, posterior, false_positive_rate=nominal))
            exported = bool(score >= np.sqrt(np.log(1 / nominal)))
            assert native == exported, (name, nominal)
            results.append(dict(nominal_fpr=nominal, native=native, exported=exported))
        rows.append(dict(input=name, score=score if np.isfinite(score) else str(score), parity=results))
    # Score ties and immutable Decode branches are scientific edge cases, not
    # a replacement for an independent natural-image calibration experiment.
    tied = calibrate_upper_tail([1, 2, 2, 2], .25)
    assert tied['accepted'] == 0 and 2 < tied['threshold'] <= 3
    bounded = calibrate_upper_tail([-3, -2, -1, -.5], .25, minimum_threshold=0)
    assert bounded['threshold'] == 0 and bounded['accepted'] == 0
    immutable = calibrate_upper_tail([1, 2, 3, 4], .25, fixed_accept=[True, True, False, False])
    assert immutable['status'] == 'unattainable'
    rng = np.random.default_rng(20261115)
    thresholds_checked = 0
    for n in [10, 101, 1000]:
        values = rng.integers(0, 257, size=n)
        for target in [.1, .01, .001]:
            result = calibrate_upper_tail(values, target)
            assert np.count_nonzero(values >= result['threshold']) <= int(np.floor(n * target))
            thresholds_checked += 1
    receipt = dict(status='complete', scope='CPU development interface validation, not empirical FPR calibration',
                   worker_sha256=sha(__file__), adapter_sha256=sha(Path(__file__).with_name('empirical_calibration.py')),
                   native_source_sha256=sha(prc_path), input_protocol_sha256=sha(run / 'protocol.json'),
                   posterior_cases=len(rows), native_boolean_comparisons=sum(len(row['parity']) for row in rows),
                   mismatches=0, random_tie_cases=thresholds_checked, immutable_branch_test=immutable,
                   illustrative_zero_of_100_95ci=exact_binomial_interval(0, 100),
                   rows=rows)
    (output / 'complete.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
