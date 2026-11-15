"""Independent packed-code score and source-aggregation audit on249 CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/data0/liyanlei/watermark_research')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(run):
    import numpy as np
    plan, done = read(run / 'protocol.json'), read(run / 'complete.json')
    assert done['status'] == 'complete' and done['protocol_sha256'] == sha(run / 'protocol.json')
    for name, digest in done['artifacts'].items():
        assert sha(run / name) == digest
    for path, digest in plan['input_sha256'].items():
        assert sha(Path(path)) == digest
    matrix = np.load(run / 'public_projection.npy')
    codes = np.load(run / 'packed_codes.npz')
    base = ROOT / 'runs/instance_representation_20261113_v2'
    shared = ROOT / 'runs/shared_channel_binding_20261114_v2'
    queries = {}
    meta = read(base / 'sscd/metadata.json')
    with np.load(base / 'sscd/features.npz') as cache:
        for vector, (task, kind), pixels in zip(cache['features'], meta['keys'], meta['image_hashes']):
            queries[f'primary_{task:03}', kind, None, pixels] = vector
    with np.load(shared / 'features.npz') as cache:
        for vector, row in zip(cache['sscd'], read(shared / 'feature_rows.json')):
            proxy = 'sd21' if row['kind'].startswith('imprint_') else None
            queries[row['id'], row['kind'], proxy, row['pixels_sha256']] = vector
    for label, folder, references in [('Reprompt', 'author_reprompt_evaluation_20261114_v2', 30),
                                      ('Reprompt+', 'reprompt_plus_evaluation_20261115_v2', 20)]:
        directory = ROOT / 'runs' / folder
        with np.load(directory / 'features.npz') as cache:
            for vector, row in zip(cache['sscd'][references:], read(directory / 'content_rows.json')):
                queries[row['id'], label, row['proxy'], row['pixels_sha256']] = vector
    rows = read(run / 'rows.json')
    errors, hamming_errors = [], []
    for row in rows:
        q = queries[row['id'], row['kind'], row.get('proxy'), row['pixels_sha256']].astype(np.float64)
        q /= np.linalg.norm(q)
        packed = codes[row['id']]
        for length in [128, 256]:
            stored = packed[:length // 8]
            assert len(stored.tobytes()) == length // 8
            signs = 2*np.unpackbits(stored).astype(np.float64)-1
            # Fold the stored sign code back into512 dimensions. This computes
            # the numerator differently and never reads a reference vector.
            numerator = q @ (matrix[:, :length] @ signs)
            denominator = np.sqrt(length * (q @ (matrix[:, :length] @ matrix[:, :length].T) @ q))
            errors.append(abs(float(numerator / denominator)-row['scores'][f'asymmetric{length}']))
            query_bytes = np.packbits((q @ matrix[:, :length]) >= 0)
            mismatch = sum(int(value).bit_count() for value in np.bitwise_xor(stored, query_bytes))
            hamming_errors.append(abs(1-mismatch/length-row['scores'][f'hamming{length}']))
    assert max(errors) < 1e-12 and max(hamming_errors) == 0
    thresholds = read(run / 'thresholds.json')['thresholds']
    for row in rows:
        assert row['accepted'] == {m: score >= thresholds[m] for m, score in row['scores'].items()}
    aggregate_checks = 0
    for cell in read(run / 'results.json')['attacks']:
        subset = [r for r in rows if r['cohort'] == 'attack_cohort' and r['carrier'] == cell['carrier']
                  and r.get('proxy') == cell['proxy'] and
                  (cell['scope'] == 'all_available_sources' or r['split'] in ['evaluation', 'transfer']) and
                  (r['kind'].startswith('imprint_') if cell['attack'] == 'imprint_any_checkpoint' else r['kind'] == cell['attack'])]
        for method, value in cell['results'].items():
            accepted = sorted({r['id'] for r in subset if r['accepted'][method]})
            assert accepted == value['source_ids'] and len(accepted) == value['accepted']
            aggregate_checks += 1
    result = dict(status='complete', cpu_only=True, worker_sha256=sha(Path(__file__)),
                  run_complete_sha256=sha(run / 'complete.json'), audited_query_pairs=len(rows),
                  asymmetric_scores_checked=len(errors), asymmetric_max_absolute_difference=max(errors),
                  hamming_scores_checked=len(hamming_errors), hamming_max_absolute_difference=max(hamming_errors),
                  source_aggregation_checks=aggregate_checks, decision_mismatches=0,
                  input='Packed16/32-byte code, query feature and common public matrix; no reference vector used in compressed-score audit')
    (run / 'independent_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    assert args.run.resolve().parent == ROOT / 'runs'
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    main(args.run.resolve())
