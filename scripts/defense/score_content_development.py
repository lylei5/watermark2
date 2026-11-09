"""Audit genuine development queries without opening confirmation scores.

Only verifier code is executed. Source codes are used for explicitly named
diagnostics; they are never inputs to the proposed detector.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

from codeword_binding import keyed_order, sembind_mask_bits, bit_accuracy
from content_authentication import recover_groups, content_scores


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_query(root, row):
    ident, task = row['id'], row['task']
    invpath = root / 'inversions' / (ident + '.npy')
    featpath = root / 'query_features' / (ident + '.npz')
    latent = np.load(invpath, allow_pickle=False)
    features = np.load(featpath, allow_pickle=False)
    if latent.size != 16384 or not np.isfinite(latent).all():
        raise ValueError(f'Invalid inversion: {ident}')
    secret = json.loads((root / 'private' / f'{task:03}.json').read_text())
    key, nonce, msg, bkey = [bytes.fromhex(secret[k])
                            for k in ['key', 'nonce', 'message', 'binding_key']]
    cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None).encryptor().update(msg * 64)
    cipherbits = np.unpackbits(np.frombuffer(cipher, dtype=np.uint8))
    message = np.unpackbits(np.frombuffer(msg, dtype=np.uint8))
    stream = cipherbits ^ np.tile(message, 64)
    values = {}
    if row['method'] == 'sembind_scattered':
        perm = keyed_order(bkey, b'sembind/permutation/', 16384)
        query = features['sembind_code']
        mask = sembind_mask_bits(query[None], perm)[0]
        groups = recover_groups(latent, cipherbits, perm)
        values.update(content_scores(groups, query))
        source = np.load(root / 'aux_codes' / f'{task:03}.npz')['sembind_code']
        error = groups != source[:, None]
        k = (groups != query[:, None]).sum(1)
        actual_agreement = float((source == query).mean())
        actual_error = float(error.mean())
        # Intraclass correlation of source-relative errors. An independent
        # homogeneous channel predicts zero; this is a diagnostic, not a test.
        p = actual_error
        n = error.shape[1]
        variance_ratio = float(error.sum(1).var() / (n * p * (1-p))) if 0 < p < 1 else None
        values.update(
            diagnostic_actual_content_agreement=actual_agreement,
            diagnostic_actual_channel_error=actual_error,
            diagnostic_error_overdispersion=variance_ratio,
            diagnostic_content_estimation_error=values['profiled_content_agreement']-actual_agreement,
            diagnostic_channel_estimation_error=values['estimated_channel_error']-actual_error,
            diagnostic_support_excludes_true_agreement=bool(values['profile_content_lower_bound'] > actual_agreement),
            group_mismatch_histogram=np.bincount(k, minlength=17).tolist(),
        )
    elif row['method'] == 'coverlock_g_message':
        mask = np.tile(features['coverlock_code'], 64)
        source = np.load(root / 'aux_codes' / f'{task:03}.npz')['coverlock_code']
        values['diagnostic_actual_content_agreement'] = float((source == features['coverlock_code']).mean())
    elif row['method'] == 'gs_control':
        mask = np.zeros(16384, dtype=np.uint8)
    else:
        raise ValueError(row['method'])
    bits = (latent.reshape(-1) > 0).astype(np.uint8) ^ mask ^ stream
    ba = float(bit_accuracy(bits, message)[0])
    return dict(task=task, method=row['method'], label=row['label'], id=ident,
                ba=ba, accepted=ba > 181/256, **values,
                image_sha256=sha(root/'queries'/(ident+'.png')),
                inversion_sha256=sha(invpath), features_sha256=sha(featpath))


def summarize(rows, rate_name='tpr'):
    result = []
    for method in sorted({r['method'] for r in rows}):
        for label in sorted({r['label'] for r in rows}):
            subset = [r for r in rows if r['method']==method and r['label']==label]
            if not subset:
                continue
            item = dict(method=method, label=label, n=len(subset),
                        accepted=sum(r['accepted'] for r in subset),
                        mean_ba=float(np.mean([r['ba'] for r in subset])),
                        min_ba=min(r['ba'] for r in subset))
            item[rate_name] = float(np.mean([r['accepted'] for r in subset]))
            for field in ['diagnostic_actual_content_agreement',
                          'diagnostic_actual_channel_error',
                          'diagnostic_error_overdispersion',
                          'diagnostic_content_estimation_error',
                          'diagnostic_channel_estimation_error',
                          'diagnostic_support_excludes_true_agreement']:
                vals = [r[field] for r in subset if field in r and r[field] is not None]
                if vals:
                    item['mean_'+field] = float(np.mean(vals))
            result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = args.root/'query_manifest_development.json'
    records = json.loads(manifest.read_text())['rows']
    if any(r['split'] != 'development' for r in records):
        raise ValueError('This audit may only read development records')
    rows = [score_query(args.root, r) for r in records]
    result = dict(scope='genuine development audit; no attack or confirmation decisions',
                  pipeline='SD2.1 DDIM50 project adapter',
                  manifest_sha256=sha(manifest), script_sha256=sha(__file__),
                  score_implementation_sha256=sha(Path(__file__).with_name('content_authentication.py')),
                  n=len(rows), source_count=len({r['task'] for r in rows}),
                  summary=summarize(rows), rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
