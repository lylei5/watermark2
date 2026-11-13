"""Use recovered material in unchanged GS detectors on authenticated cached inversions.

Runs only on249. Reuses existing inversions; performs new key recovery and native
bit decoding. The verifier never reads the enrollment identity/key fixture files.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import sys
import time

ROOT = Path('/data0/liyanlei/watermark_research')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    assert ROOT.is_dir() and run.parent == ROOT/'runs'
    os.environ.update(CUDA_VISIBLE_DEVICES='', NUMBA_NUM_THREADS='4',
                      OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    import numpy as np
    import torch
    sys.path.insert(0, str(run))
    from instance_coset_recovery import InstanceRecovery
    started = time.monotonic()
    plan = json.loads((run/'protocol.json').read_text())
    assert sha(run/'instance_coset_recovery.py') == plan['source_sha256']['instance_coset_recovery.py']
    done = json.loads((run/'repetition_0/complete.json').read_text())
    feature_done = json.loads((run/'features_complete.json').read_text())
    for name, digest in feature_done['artifacts'].items():
        assert sha(run/name) == digest
    for name, digest in done['artifacts'].items():
        assert sha(run/'repetition_0'/name) == digest
    private = Path(plan['private'])
    assert sha(private/'secrets.json') == plan['secret_sha256']
    provider = bytes.fromhex(json.loads((private/'secrets.json').read_text())['provider'])
    projection_path = private/'repetition_0/projection.npy'
    assert sha(projection_path) == done['private_projection_sha256']
    codec = InstanceRecovery(np.load(projection_path))
    features = np.load(run/'features.npz')['features']
    metadata = json.loads((run/'feature_rows.json').read_text())
    lookup = {(r['id'], r['kind']): i for i, r in enumerate(metadata)}
    enrollment = json.loads((run/'repetition_0/enrollment.json').read_text())
    expected_records = {(r['id'], r['method']): r['record_sha256'] for r in enrollment}
    old_rows = json.loads((run/'repetition_0/rows.json').read_text())
    expected_recovery = {(r['id'], r['kind'], r['method']): r['key_recovered'] for r in old_rows}

    # A fresh independent random key tests the public API, outside the paired PRF harness.
    ref = features[lookup['S000', 'clean']]
    vm = [r for r in metadata if r['id'] == 'S000' and r['kind'].startswith('enroll_')]
    views = np.stack([features[lookup['S000', r['kind']]] for r in vm])
    payload = secrets.token_bytes(32)
    record, key = codec.enroll('api_check', ref, views, [r['family'] for r in vm],
                               payload, provider, selector='orbit_margin', mode='fixed')
    result, _ = codec.recover(record, ref, provider)
    assert result == (key, payload)
    changed = copy.deepcopy(record)
    changed['metadata']['id'] = 'changed_record'
    assert codec.recover(changed, ref, provider)[0] is None
    assert codec.recover(record, ref, secrets.token_bytes(32))[0] is None

    spatial = ROOT/'runs/phase1_trajectory_native_gs_20261113'
    pgid = ROOT/'runs/defense_revalidation_20261111/PGID'
    channel = ROOT/'runs/carrier_content_channel_20261113'
    module_path = spatial/'gs_watermark.py'
    spec = importlib.util.spec_from_file_location('original_spatial_gs', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.insert(0, str(pgid))
    from watermarks.gs_watermark import GSWatermark
    spatial_decisions = {}
    evidence = {}
    for shard in range(3):
        path = spatial/f'verify_{shard}/complete.json'
        evidence[str(path)] = sha(path)
        for source in json.loads(path.read_text())['rows']:
            decisions = [r for r in source['decisions'] if r['image_kind'] == r['detector'] == 'gs']
            assert len(decisions) == 1
            spatial_decisions[source['task']] = decisions[0]
    evidence[str(channel/'rows.json')] = sha(channel/'rows.json')
    channel_rows = {(r['task'], r['kind'], r['checkpoint']): r
                    for r in json.loads((channel/'rows.json').read_text())}
    cache, rows = {}, []
    for meta in metadata:
        if meta['kind'].startswith('enroll_'):
            continue
        if meta['cohort'] == 'spatial_GS' and meta['kind'] != 'clean':
            continue
        identity, kind = meta['id'], meta['kind']
        task = int(identity[1:])
        if meta['cohort'] == 'spatial_GS':
            prior = spatial_decisions[task]
            latent_path = spatial/prior['latent']['path']
            expected_hash = prior['latent']['sha256']
            assert sha(spatial/prior['image']['path']) == prior['image']['sha256']
        else:
            channel_kind = 'imprint' if kind.startswith('imprint_') else kind
            checkpoint = int(kind.split('_')[1]) if channel_kind == 'imprint' else -1
            prior = channel_rows[task, channel_kind, checkpoint]
            assert prior['pixels_sha256'] == meta['pixels_sha256']
            latent_path = channel/prior['latent']
            expected_hash = prior['latent_sha256']
        assert sha(latent_path) == expected_hash
        for method in plan['methods']:
            record_path = private/'repetition_0'/f'{identity}_{method}.json'
            assert sha(record_path) == expected_records[identity, method]
            helper = json.loads(record_path.read_text())
            recovered, detail = codec.recover(helper, features[lookup[identity, kind]], provider)
            assert (recovered is not None) == expected_recovery[identity, kind, method]
            ba, reused = None, False
            if recovered is not None:
                material = recovered[1]
                token = (identity, kind, hashlib.sha256(material).hexdigest())
                reused = token in cache
                if not reused:
                    if identity.startswith('S'):
                        assert material[:4] == b'GS01' and len(material) == 2084
                        native = module.Gaussian_Shading(1, 8, 1e-6, 1)
                        native.key = torch.from_numpy(np.unpackbits(np.frombuffer(material[4:2052], dtype=np.uint8)).reshape(1,4,64,64))
                        native.watermark = torch.from_numpy(np.unpackbits(np.frombuffer(material[2052:], dtype=np.uint8)).reshape(1,4,8,8))
                        ba = native.eval_watermark(torch.from_numpy(np.load(latent_path)))
                    else:
                        fields = json.loads(material)
                        assert fields['carrier'] == 'PGID_GS'
                        native = GSWatermark(latent_shape=(1,4,64,64), device='cpu',
                            message=bytes.fromhex(fields['message']), key=bytes.fromhex(fields['key']),
                            nonce=bytes.fromhex(fields['nonce']))
                        with np.load(latent_path) as archive:
                            ba = native.get_accuracies(torch.from_numpy(archive['z']))['bit_accuracies'][0]
                    assert ba == prior['ba'], (identity, kind, ba, prior['ba'])
                    cache[token] = ba
                ba = cache[token]
            rows.append(dict(id=identity, kind=kind, method=method, key_recovered=recovered is not None,
                native_ba=ba, joint_accepted=bool(ba is not None and ba > 181/256),
                reused_identical_recovered_material=reused, latent_sha256=expected_hash,
                prior_ba_equal=bool(ba == prior['ba']) if ba is not None else None))
    assert len(rows) == 920
    summaries = []
    for method in plan['methods']:
        for group in ['spatial_clean', 'pgid_genuine', 'pgid_imprint_all', 'pgid_imprint_final']:
            part = [r for r in rows if r['method'] == method and
                    (r['id'].startswith('S') if group == 'spatial_clean' else
                     r['id'].startswith('G') and (not r['kind'].startswith('imprint_') if group == 'pgid_genuine' else
                     r['kind'] == 'imprint_150' if group == 'pgid_imprint_final' else r['kind'].startswith('imprint_')))]
            summaries.append(dict(method=method, group=group, total=len(part),
                key_recovered=sum(r['key_recovered'] for r in part), accepted=sum(r['joint_accepted'] for r in part)))
    result = dict(status='complete', repetition=0, actual_recovery_decisions=len(rows),
        distinct_native_decoder_calls=len(cache), rows=rows, summary=summaries,
        independent_random_key_api=True, record_authentication_checks=2,
        inversion_scope='Existing hash-verified inversions, fresh recovered-material native decoding',
        native_threshold='BA > 181/256', source_sha256=sha(__file__),
        protocol_sha256=sha(run/'protocol.json'), prior_artifacts=evidence,
        detector_sha256={str(module_path):sha(module_path),
                        str(pgid/'watermarks/gs_watermark.py'):sha(pgid/'watermarks/gs_watermark.py')},
        seconds=time.monotonic()-started)
    write(run/'native_key_verification.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','prior_artifacts','detector_sha256']}), flush=True)


if __name__ == '__main__':
    main()
