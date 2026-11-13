"""Audit completed native baseline artifacts and emit denominator-explicit summaries."""
import argparse
from collections import defaultdict
import datetime
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--method', choices=['seal', 'prc_s', 'tr_s', 'pgid'], required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    assert run.parent == Path('/data0/liyanlei/watermark_research/runs')
    os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='4')
    import numpy as np
    from scipy.stats import beta
    plan = json.loads((run / 'protocol.json').read_text())
    protocol_hash = sha(run / 'protocol.json')
    rows, artifact_checks = [], 0
    source_files = {}
    for shard in range(3):
        file = run / f'native_{shard}/complete.json'
        result = json.loads(file.read_text())
        assert result['status'] == 'complete' and result['protocol_sha256'] == protocol_hash
        rows.extend(result['rows'])
        source_files[str(file.relative_to(run))] = sha(file)
    groups = defaultdict(list)
    public_rows = []
    norm_max_difference = 0.

    def check(path, digest):
        nonlocal artifact_checks
        assert sha(path) == digest, path
        artifact_checks += 1

    if args.method == 'seal':
        assert sorted(x['task'] for x in rows) == sorted(x['task'] for x in plan['sources'])
        for item in rows:
            for artifact in item['images'].values():
                check(run / artifact['path'], artifact['sha256'])
            check(run / item['latents']['path'], item['latents']['sha256'])
            with np.load(run / item['latents']['path']) as data:
                for decision in item['decisions']:
                    assert decision['kind'] in ['genuine', 'null']
                    distances = np.asarray(decision['distances'])
                    assert distances.shape == (1024,) and np.isfinite(distances).all() and (distances >= 0).all()
                    count = int(np.sum(distances < plan['tau']))
                    assert count == decision['matches']
                    assert bool(count >= plan['minimum_matches']) == decision['accepted']
                    z_name, r_name = ('recovered', 'genuine_reference') if decision['kind'] == 'genuine' else ('null_recovered', 'null_reference')
                    # Recreate FP16 subtraction and block geometry independently on CPU.
                    delta = (data[z_name].astype(np.float32) - data[r_name].astype(np.float32)).astype(np.float16)
                    blocks = delta.reshape(1, 4, 32, 2, 32, 2).transpose(2, 4, 0, 1, 3, 5).reshape(1024, 16)
                    norms = np.sqrt(np.sum(blocks.astype(np.float64)**2, axis=1)).astype(np.float16).astype(np.float64)
                    difference = float(np.max(np.abs(norms - distances)))
                    norm_max_difference = max(norm_max_difference, difference)
                    assert np.allclose(norms, distances, atol=.004, rtol=.001)
                    assert int(np.sum(norms < plan['tau'])) == count
                    row = dict(task=item['task'], dataset=item['dataset'], kind=decision['kind'],
                               accepted=decision['accepted'], matching_patches=count,
                               preview_control_equal=item.get('preview_control_equal'))
                    public_rows.append(row)
                    for dataset in [item['dataset'], 'ALL']:
                        groups[(dataset, decision['kind'])].append(row)
            assert sorted(x['kind'] for x in item['decisions']) == ['genuine', 'null']
        assert len(public_rows) == len(plan['sources']) * 2
    elif args.method in ['prc_s', 'tr_s']:
        carrier = 'prc' if args.method == 'prc_s' else 'tr'
        expected = {(x['task'], kind, detector) for x in plan['sources']
                    for kind, detector in [(carrier, carrier), (args.method, args.method), ('null', carrier), ('null', args.method)]}
        assert len(rows) == len(expected) and {(x['task'], x['image_kind'], x['detector']) for x in rows} == expected
        for row in rows:
            check(row['image'], row['image_sha256'])
            check(run / row['latent'], row['latent_sha256'])
            fields = ['semantic_agreement']
            if args.method == 'prc_s':
                assert row['accepted'] == (row['detected'] or row['decodable'])
                assert row['decodable'] == (row['message_ba'] is not None)
                fields.append('codeword_ba')
                if row['message_ba'] is not None:
                    assert 0 <= row['message_ba'] <= 1
            else:
                assert 0 <= row['p_value'] <= 1 and np.isfinite(row['l1_distance'])
                assert row['accepted'] == (row['p_value'] <= 1e-6)
            for field in fields:
                assert 0 <= row[field] <= 1
            public_rows.append(row)
            for dataset in [row['dataset'], 'ALL']:
                groups[(dataset, row['image_kind'], row['detector'])].append(row)
    else:
        indices = [i for shard in plan['shards'] for i in range(shard['start'], shard['start'] + shard['count'])]
        expected = {(i, stage, checkpoint) for i in indices for stage, checkpoint in
                    [('watermarked', -1)] + [(s, c) for s in ['attacked', 'defended'] for c in [50, 100, 150]]}
        assert len(rows) == len(expected) and {(x['task'], x['stage'], x['checkpoint']) for x in rows} == expected
        for shard in range(3):
            out = run / f'native_{shard}'
            result = json.loads((out / 'complete.json').read_text())
            check(out / ('attack=' + plan['attack_type']) / 'all_results.csv', result['raw_sha256'])
            for path, digest in result['image_sha256'].items():
                check(out / path, digest)
        for row in rows:
            assert row['accepted'] == (row['ba'] > 181 / 256)
            public_rows.append(row)
            groups[(row['stage'], row['checkpoint'])].append(row)
    summaries = []
    for key, subset in sorted(groups.items()):
        n = len(subset)
        successes = sum(x['accepted'] for x in subset)
        lower = 0. if successes == 0 else float(beta.ppf(.025, successes, n-successes+1))
        upper = 1. if successes == n else float(beta.ppf(.975, successes+1, n-successes))
        item = dict(group=list(key), n=n, accepted=successes, rate=successes/n, cp95=[lower, upper])
        for field in ['matching_patches', 'ba', 'codeword_ba', 'semantic_agreement', 'p_value', 'l1_distance']:
            if field in subset[0]:
                item['mean_' + field] = float(np.mean([r[field] for r in subset]))
        if args.method == 'prc_s':
            item.update(detected=sum(x['detected'] for x in subset), decodable=sum(x['decodable'] for x in subset),
                        exact_payloads=sum(x['message_ba'] == 1 for x in subset))
        summaries.append(item)
    output = dict(status='complete', method=args.method, protocol_sha256=protocol_hash,
                  source_files=source_files, artifact_checks=artifact_checks, rows=public_rows, summaries=summaries,
                  independent_units='Source images, not repeated checkpoints or detector rows',
                  scope=plan['scope'], empirical_fpr_1e6_established=False,
                  paper_full_benchmark_reproduced=False,
                  seal_cpu_norm_max_difference=norm_max_difference if args.method == 'seal' else None,
                  summary_worker_sha256=sha(__file__), completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path = run / 'audited_summary.json'
    assert not path.exists()
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(dict(run=run.name, summaries=summaries, artifact_checks=artifact_checks)), flush=True)


if __name__ == '__main__':
    main()
