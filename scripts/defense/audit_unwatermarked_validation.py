"""Audit frozen natural-negative inputs, complete caches and reported decisions."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    output = run / 'audit.json'
    assert not output.exists()
    plan = json.loads((run / 'protocol.json').read_text())
    done = json.loads((run / 'complete.json').read_text())
    result = json.loads((run / 'scores/results.json').read_text())
    assert done['completed'] and result['completed']
    assert done['protocol_sha256'] == result['protocol_sha256'] == sha(run / 'protocol.json')
    assert done['results_sha256'] == sha(run / 'scores/results.json')
    for name, digest in plan['source_sha256'].items():
        assert sha(run / 'scripts' / name) == digest, name
    for row in plan['rows'] + plan['bridge_rows']:
        assert sha(run / row['image']) == row['image_sha256']
        for key in ['expected_inversion', 'expected_features']:
            if key in row:
                assert sha(run / row[key]) == row[key + '_sha256']
    cached = []
    for shard in range(3):
        folder = run / 'cache' / f'shard_{shard}'
        part = json.loads((folder / 'complete.json').read_text())
        assert part['completed'] and part['steps'] == 50 and part['batch'] == 2
        assert part['protocol_sha256'] == sha(run / 'protocol.json')
        for row in part['rows']:
            for suffix, key in [('.npy', 'inversion_sha256'), ('.npz', 'features_sha256')]:
                assert sha(folder / (row['id'] + suffix)) == row[key]
            if row['split'] == 'unwatermarked':
                cached.append(row['id'])
    bridge = json.loads((run / 'cache/shard_0/bridge.json').read_text())
    assert {r['id'] for r in bridge['rows']} == {r['id'] for r in plan['bridge_rows']}
    assert all(r['code_equal'] and r['relative_l2'] <= plan['bridge_max_relative_l2'] for r in bridge['rows'])
    queries = {r['id'] for r in plan['rows']}
    assert len(cached) == len(set(cached)) == len(queries) == 100 and set(cached) == queries
    rows = result['rows']
    assert len(rows) == 6400
    assert {(r['id'], r['claimed_identity']) for r in rows} == {(q, t) for q in queries for t in plan['identity_tasks']}
    assert all(r['kind'] == 'unwatermarked_natural' and r['default_accepted'] == (r['ba'] > 181/256) for r in rows)
    for flag, probability in [('all_fit_accepted', 'all_fit_p'), ('split_fit_accepted', 'split_fit_p')]:
        assert all(r[flag] == (r[probability] <= plan['nominal_alpha']) for r in rows)
    for flag, summary in result['summary'].items():
        assert summary == dict(accepted=sum(r[flag] for r in rows), n=6400,
                               any_identity_accepted_queries=len({r['id'] for r in rows if r[flag]}), queries=100)
    assert len(result['supplementary_ba_controls']) == len(plan['supplementary_ba_controls'])
    for summary, frozen in zip(result['supplementary_ba_controls'], plan['supplementary_ba_controls']):
        assert summary['threshold'] == frozen['threshold'] and summary['operator'] == frozen['operator'] == '>='
        assert summary['accepted'] == sum(r['ba'] >= frozen['threshold'] for r in rows)
        assert summary['any_identity_accepted_queries'] == len({r['id'] for r in rows if r['ba'] >= frozen['threshold']})
    audit = dict(completed=True, protocol_sha256=sha(run / 'protocol.json'), results_sha256=sha(run / 'scores/results.json'),
                 audit_source_sha256=sha(Path(__file__)), image_and_bridge_hashes_verified=True,
                 frozen_source_and_cache_hashes_verified=True, identity_pairs_verified=6400,
                 full_ddim_steps=50, bridge=bridge, summary=result['summary'],
                 supplementary_ba_controls=result['supplementary_ba_controls'], independent_confirmation=False,
                 limitation='Existing development images and shared identities; no independent-trial or ultra-low FPR claim')
    output.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(dict(completed=True, queries=100, identity_pairs_verified=6400)))


if __name__ == '__main__':
    main()
