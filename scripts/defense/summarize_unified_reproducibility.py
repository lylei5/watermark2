"""CPU249 verification of development-only extraction reproducibility evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('/data0/liyanlei/watermark_research')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def compare(first, second):
    delta = np.abs(first.astype(float)-second.astype(float))
    return dict(maximum=float(delta.max()), mean=float(delta.mean()),
                sign_changes=int((np.sign(first) != np.sign(second)).sum()),
                exact=bool(np.array_equal(first, second)),
                allclose=bool(np.allclose(first, second, atol=1e-5, rtol=1e-5)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    assert run.parent == ROOT/'runs'
    assert read(run/'pipeline_state.json')['status'] == 'complete'
    plan = read(run/'protocol.json')
    parity = read(run/'parity.json')
    assert parity['passed'] and parity['protocol_sha256'] == sha(run/'protocol.json')
    assert not (run/'audit.json').exists()
    references = {}
    costs = {}
    for item in plan['plans']:
        folder = Path(item['run'])
        assert sha(folder/'protocol.json') == item['protocol_sha256']
        cfg = read(folder/'protocol.json')
        assert cfg['deterministic_backend']
        assert sha(folder/'unified_runtime_probe.py') == cfg['worker_sha256']
        assert sha(folder/'deterministic_entry.py') == item['entry_sha256']
        for name, digest in cfg['helper_sha256'].items():
            assert sha(folder/name) == digest
        receipt = read(folder/'sembind/complete.json')
        assert receipt['status'] == 'complete'
        rows = receipt['measurements']
        assert len(rows) == 8 and sorted(r['source'] for r in rows) == list(range(8))
        arrays = []
        for row in sorted(rows, key=lambda x:x['source']):
            assert row['unet_calls'] == 50
            assert row['input_sha256'] == cfg['inputs'][row['source']]['sha256']
            path = folder/'sembind'/row['artifact']
            assert sha(path) == row['artifact_sha256']
            with np.load(path, allow_pickle=False) as values:
                assert values['recovered'].shape == (1,4,64,64) and np.isfinite(values['recovered']).all()
                arrays.append({key:values[key].copy() for key in values.files})
        references[item['label']] = arrays
        seconds = [row['seconds'] for row in rows]
        costs[item['label']] = dict(images=8, mean_seconds=float(np.mean(seconds)), summed_image_seconds=sum(seconds),
                                   gpu=item['gpu'], full_unet_calls=sum(row['unet_calls'] for row in rows))
    checks = []
    for row in parity['comparisons']:
        index = row['source']
        first = references['legacy'][index]
        second = references[row['candidate']][index]
        check = compare(first['recovered'], second['recovered'])
        for key in ['maximum', 'mean', 'sign_changes', 'allclose']:
            assert check[key] == row[key]
        semantic = bool(np.array_equal(first['semantic_code'], second['semantic_code']))
        assert semantic == row['semantic_equal'] and semantic and check['allclose']
        checks.append(dict(source=index, candidate=row['candidate'], semantic_equal=semantic, **check))
    assert len(checks) == 16
    old = ROOT/'runs/unified_runtime_probe_20261115'
    failed = ROOT/'runs/unified_null_features_20261116/chunks/development_sembind'
    default_checks = []
    for index in range(8):
        first = np.load(old/'sembind'/f'{index:03}_sembind.npz')
        second = np.load(failed/'sembind'/f'{index:03}_sembind.npz')
        default_checks.append(dict(source=index, semantic_equal=bool(np.array_equal(first['semantic_code'], second['semantic_code'])),
                                   **compare(first['recovered'], second['recovered'])))
    pgid = read(ROOT/'runs/unified_null_features_20261116/chunks/development_pgid/parity.json')
    assert pgid['passed'] and pgid['artifacts'] == 24 and pgid['max_absolute_difference'] == 0
    result = dict(status='complete', completed_at=datetime.now(timezone.utc).isoformat(), protocol_sha256=sha(run/'protocol.json'),
                  parity_sha256=sha(run/'parity.json'), script_sha256=sha(Path(__file__)),
                  strict_artifacts_verified=24, strict_comparisons=checks, default_comparisons=default_checks,
                  strict_costs=costs, pgid_existing_artifacts=24, pgid_existing_maximum_difference=0,
                  gpu_inferences_in_this_audit=0, formal_natural_sources=0, attacks=0,
                  scope='Eight development inputs; same50 DPM steps,100 VAE updates, fixed seeds and unchanged tolerances. Not a detector FPR/TPR result.',
                  preserved_interruption=plan.get('preserved_interruption'))
    (run/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='complete', strict_artifacts=24, comparisons=16,
                         all_exact=all(r['exact'] for r in checks), costs=costs)))


if __name__ == '__main__':
    main()
