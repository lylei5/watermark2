"""Audit completed development-runtime outputs on server249."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    assert run.parent == Path('/data0/liyanlei/watermark_research/runs')
    import numpy as np
    plan = json.loads((run / 'protocol.json').read_text())
    assert sha(run / 'unified_runtime_probe.py') == plan['worker_sha256']
    for name, digest in plan['helper_sha256'].items():
        assert sha(run / name) == digest
    checks, rows, manifests = 0, [], {}
    for method in ['pgid', 'sembind', 'seal']:
        source = Path(plan['sources'][method])
        for name, digest in plan['source_sha256'][method].items():
            assert sha(source / name) == digest
        path = run / method / 'complete.json'
        state = json.loads(path.read_text())
        assert state['status'] == 'complete' and state['protocol_sha256'] == sha(run / 'protocol.json')
        manifests[method] = {'sha256': sha(path), 'gpu': state['gpu'], 'gpu_uuid': state['gpu_uuid'],
                             'loading_seconds': state['loading_seconds'], 'versions': state['versions']}
        for row in state['measurements']:
            assert row['input_sha256'] == plan['inputs'][row['source']]['sha256']
            path = run / method / row['artifact']
            assert sha(path) == row['artifact_sha256']
            with np.load(path) as data:
                assert data['recovered'].shape == (1, 4, 64, 64)
                assert np.isfinite(data['recovered']).all()
                if method == 'sembind':
                    assert data['semantic_code'].shape == (1024,)
                    assert np.isin(data['semantic_code'], [0, 1]).all()
                if method == 'seal':
                    assert data['distances'].shape == (1024,)
                    assert data['semantic_matches'] == (data['distances'] < 2.3).sum()
            checks += 1
            rows.append(row)
    summary = []
    for arm in sorted(set(row['arm'] for row in rows)):
        values = [row for row in rows if row['arm'] == arm]
        assert sorted(row['source'] for row in values) == list(range(8))
        mean = statistics.mean(row['seconds'] for row in values)
        summary.append({'arm': arm, 'n': len(values), 'mean_seconds': mean,
                        'median_seconds': statistics.median(row['seconds'] for row in values),
                        'min_seconds': min(row['seconds'] for row in values),
                        'max_seconds': max(row['seconds'] for row in values),
                        'unet_calls': sorted(set(row['unet_calls'] for row in values), key=str),
                        'max_peak_allocated_gib': max(row['peak_allocated_bytes'] for row in values) / 2**30,
                        'output_bytes': sum(row['artifact_bytes'] for row in values),
                        'projected_40000_gpu_hours': mean * 40000 / 3600})
    result = {'scope': 'eight development inputs; extraction timing only; no calibrated detection result',
              'status': 'complete', 'protocol_sha256': sha(run / 'protocol.json'), 'artifact_checks': checks,
              'source_manifests': manifests, 'summary': summary,
              'projection_limitations': 'Single-image serial execution, same 3090 class, three jobs concurrently. Excludes loading, decoding, final score adapters, failures and attack costs. Projection is not a measured large-run runtime.'}
    (run / 'runtime_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
