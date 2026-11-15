"""CPU-only fixed-public-projection SSCD comparison on existing feature caches."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

ROOT = Path('/data0/liyanlei/watermark_research')
BASE = ROOT / 'runs/instance_representation_20261113_v2'
SHARED = ROOT / 'runs/shared_channel_binding_20261114_v2'
REPROMPT = ROOT / 'runs/author_reprompt_evaluation_20261114_v2'
PLUS = ROOT / 'runs/reprompt_plus_evaluation_20261115_v2'
ATTACK = ROOT / 'runs/author_reprompt_control_20261114_v2'
ATTACK_PLUS = ROOT / 'runs/author_reprompt_plus_20261115'
METHODS = ['sscd', 'asymmetric128', 'asymmetric256', 'hamming128', 'hamming256']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare(run):
    import numpy as np
    run.mkdir(exist_ok=False)
    shutil.copy2(__file__, run / Path(__file__).name)
    base, shared = read(BASE / 'protocol.json'), read(SHARED / 'protocol.json')
    files = [BASE / n for n in ['protocol.json', 'sscd/features.npz', 'sscd/metadata.json',
                               'sscd/complete.json', 'sscd/model_manifest.json']]
    files += [SHARED / n for n in ['protocol.json', 'features.npz', 'feature_rows.json', 'extract_complete.json']]
    for directory in [REPROMPT, PLUS]:
        files += [directory / n for n in ['protocol.json', 'features.npz', 'content_rows.json',
                                          'content_complete.json', 'queries.json']]
    files += [ATTACK / 'protocol.json', ATTACK_PLUS / 'protocol.json']
    matrix = np.random.Generator(np.random.PCG64(20261116)).standard_normal((512, 256))
    matrix /= np.linalg.norm(matrix, axis=0, keepdims=True)
    np.save(run / 'public_projection.npy', matrix)
    plan = dict(scope='Existing-feature CPU decision experiment; no new inference, image transforms, attacks or embedding',
        worker_sha256=sha(__file__), input_sha256={str(p): sha(p) for p in files},
        public_seed=20261116, rng='NumPy PCG64 standard_normal float64, shape512x256, column L2 normalization',
        projection_sha256=sha(run / 'public_projection.npy'), lengths=[128, 256],
        prefix='128-bit projection is the first128 columns of the frozen256-column public matrix',
        enrollment='Packed sign(R^T unit(reference)), zero is positive; no reference-specific projection selection',
        asymmetric='dot(2*stored_bits-1, R^T unit(query)) / (sqrt(m)*norm(R^T unit(query)))',
        symmetric='Fraction of matching reference/query sign bits', continuous='Cosine of original512-dimensional SSCD vectors',
        methods=METHODS, target_retention=.95, calibration=base['calibration'], evaluation=base['evaluation'],
        conditions=list(base['conditions']), clean_in_calibration=False,
        threshold='Largest observed similarity threshold retaining at least95% of450 calibration edits; >=, keep ties',
        transfer_threshold='Reuse those450-edit thresholds unchanged for every existing attack and attack-cohort genuine image',
        attack_splits={k: shared[k] for k in ['fit_sources', 'calibration_sources', 'evaluation_sources']},
        attack_reporting='All10 sources per carrier plus original GS evaluation3/TR-PRC transfer10; no source removed after scoring',
        aggregation='Any accepted candidate per source/proxy; Imprint final151-update output and individual saved checkpoints separately',
        primary_pass=dict(length=128, evaluation_same_prompt_max=5, evaluation_same_prompt_n=50,
                          reprompt_source_excess_max=1, check_each_carrier_proxy_and_reported_split=True,
                          report_heldout_legitimate_retention_without_retuning=True),
        attack_budgets=dict(Reprompt=1, RepromptPlus_GS=9, RepromptPlus_TR=3, Imprint_updates=151),
        projection_search=False, per_image_weights=False, centering=False, training=False,
        cpu_only=True, gpu_jobs=0, new_attacks=0, independent_confirmation=False,
        limitations='Previously used development cohorts; fixed projection and frozen calibration only; no authenticated carried-code system')
    write(run / 'protocol.json', plan)
    print(json.dumps(dict(status='prepared', protocol_sha256=sha(run / 'protocol.json'),
                         projection_sha256=plan['projection_sha256'])), flush=True)


def execute(run):
    import numpy as np
    from scipy.stats import beta
    plan = read(run / 'protocol.json')
    assert sha(__file__) == plan['worker_sha256']
    for name, digest in plan['input_sha256'].items():
        assert sha(name) == digest, name
    assert sha(run / 'public_projection.npy') == plan['projection_sha256']
    state = dict(status='running', pid=os.getpid(), protocol_sha256=sha(run / 'protocol.json'),
                 cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], gpu_jobs=0)
    write(run / 'state.json', state)
    started = time.monotonic()
    try:
        def verified(directory, complete, names):
            receipt = read(directory / complete)
            assert receipt['status'] == 'complete'
            for name in names:
                assert sha(directory / name) == receipt['artifacts'][name]
        verified(BASE / 'sscd', 'complete.json', ['features.npz', 'metadata.json', 'model_manifest.json'])
        verified(SHARED, 'extract_complete.json', ['features.npz', 'feature_rows.json'])
        for directory in [REPROMPT, PLUS]:
            verified(directory, 'content_complete.json', ['features.npz', 'content_rows.json'])
        assert read(BASE / 'sscd/model_manifest.json')['sha256'] == read(SHARED / 'protocol.json')['sscd']['sha256']
        matrix = np.load(run / 'public_projection.npy')
        references, observations, enrollments, checks = {}, [], {}, {}
        def unit(vector):
            vector = np.asarray(vector, dtype=np.float64)
            assert vector.shape == (512,) and np.isfinite(vector).all() and np.linalg.norm(vector) > 0
            return vector / np.linalg.norm(vector)
        def enroll(identity, vector, pixels):
            reference = unit(vector)
            references[identity] = dict(vector=reference, pixels=pixels)
            bits = (reference @ matrix) >= 0
            enrollments[identity] = np.packbits(bits)
            assert np.array_equal(np.unpackbits(enrollments[identity]), bits)
        def compressed_score(packed, query, length, symmetric):
            bits = np.unpackbits(packed[:length // 8]).astype(bool)
            projected = unit(query) @ matrix[:, :length]
            assert np.linalg.norm(projected) > 0
            if symmetric:
                return float(np.mean(bits == (projected >= 0)))
            signs = 2 * bits.astype(np.float64) - 1
            return float(signs @ projected / (math.sqrt(length) * np.linalg.norm(projected)))
        def observe(identity, vector, **metadata):
            query = unit(vector)
            scores = dict(sscd=float(references[identity]['vector'] @ query))
            for length in plan['lengths']:
                scores[f'asymmetric{length}'] = compressed_score(enrollments[identity], query, length, False)
                scores[f'hamming{length}'] = compressed_score(enrollments[identity], query, length, True)
            assert all(np.isfinite(list(scores.values())))
            observations.append(dict(id=identity, **metadata, scores=scores))

        meta = read(BASE / 'sscd/metadata.json')
        with np.load(BASE / 'sscd/features.npz') as cache:
            values = cache['features'].copy()
        assert values.shape == (1300, 512)
        index = {tuple(key): i for i, key in enumerate(meta['keys'])}
        for task in range(100):
            identity = f'primary_{task:03}'
            j = index[task, 'bound']
            enroll(identity, values[j], meta['image_hashes'][j])
            for kind in ['bound', *plan['conditions'], 'same']:
                j = index[task, kind]
                observe(identity, values[j], cohort='primary', task=task, kind=kind,
                        split='calibration' if task in plan['calibration'] else 'evaluation',
                        pixels_sha256=meta['image_hashes'][j])
        calibration = [row for row in observations if row['split'] == 'calibration' and row['kind'] in plan['conditions']]
        assert len(calibration) == 450
        thresholds = {method: float(sorted(row['scores'][method] for row in calibration)[
            len(calibration)-math.ceil(plan['target_retention']*len(calibration))]) for method in METHODS}
        # Persist thresholds before loading any attack feature array.
        write(run / 'thresholds.json', dict(protocol_sha256=sha(run / 'protocol.json'), thresholds=thresholds,
              n=450, source_count=50, accepted={m: sum(r['scores'][m] >= thresholds[m] for r in calibration) for m in METHODS}))
        shared_rows = read(SHARED / 'feature_rows.json')
        with np.load(SHARED / 'features.npz') as cache:
            shared_features = cache['sscd'].copy()
        assert len(shared_features) == len(shared_rows)
        for j, row in enumerate(shared_rows):
            if row['kind'] == 'clean':
                enroll(row['id'], shared_features[j], row['pixels_sha256'])
        assert len(references) == 130
        assert not ({r['pixels'] for k, r in references.items() if k.startswith('primary_')} &
                    {r['pixels'] for k, r in references.items() if not k.startswith('primary_')})
        def old_split(row):
            if row['carrier'] != 'GS':
                return 'transfer'
            return next(name.replace('_sources', '') for name, ids in plan['attack_splits'].items() if row['task'] in ids)
        for j, row in enumerate(shared_rows):
            kind = row['kind']
            if kind not in ['clean', 'same_prompt', *plan['conditions']] and not kind.startswith('imprint_'):
                continue
            observe(row['id'], shared_features[j], cohort='attack_cohort', carrier=row['carrier'], task=row['task'],
                    kind=kind, split=old_split(row), proxy='sd21' if kind.startswith('imprint_') else None,
                    pixels_sha256=row['pixels_sha256'])
        max_reference_difference, max_distance_difference = 0., 0.
        for label, directory, attack in [('Reprompt', REPROMPT, ATTACK), ('Reprompt+', PLUS, ATTACK_PLUS)]:
            refs = read(attack / 'protocol.json')['references']
            rows = read(directory / 'content_rows.json')
            queries = read(directory / 'queries.json')
            assert [r['pixels_sha256'] for r in rows] == [r['pixels_sha256'] for r in queries]
            with np.load(directory / 'features.npz') as cache:
                features = cache['sscd'].copy()
            assert features.shape == (len(refs)+len(rows), 512)
            for j, ref in enumerate(refs):
                assert references[ref['id']]['pixels'] == ref['pixels_sha256']
                difference = float(np.max(np.abs(unit(features[j])-references[ref['id']]['vector'])))
                max_reference_difference = max(max_reference_difference, difference)
                assert difference < 1e-5
            for j, row in enumerate(rows):
                observe(row['id'], features[len(refs)+j], cohort='attack_cohort', carrier=row['carrier'], task=row['task'],
                        kind=label, split=old_split(row), proxy=row['proxy'],
                        trial=row.get('trial', row['id']+'_'+row['proxy']), pixels_sha256=row['pixels_sha256'])
                delta = abs(float(np.clip(1-observations[-1]['scores']['sscd'], 0, 2))-row['sscd_distance'])
                max_distance_difference = max(max_distance_difference, delta)
                assert delta < 1e-5
        for row in observations:
            row['accepted'] = {m: row['scores'][m] >= thresholds[m] for m in METHODS}
        def interval(k, n):
            return [0. if k == 0 else float(beta.ppf(.025, k, n-k+1)),
                    1. if k == n else float(beta.ppf(.975, k+1, n-k))]
        def counts(rows, by_source=False):
            identities = sorted({r['id'] for r in rows})
            flags = {m: [any(r['accepted'][m] for r in rows if r['id'] == identity) for identity in identities]
                     if by_source else [r['accepted'][m] for r in rows] for m in METHODS}
            result = {m: dict(accepted=sum(flags[m]), n=len(flags[m]),
                             source_ids=[i for i, yes in zip(identities, flags[m]) if yes] if by_source else None)
                      for m in METHODS}
            for method in METHODS:
                result[method]['minus_sscd'] = result[method]['accepted']-result['sscd']['accepted']
                if by_source:
                    result[method]['ci95'] = interval(result[method]['accepted'], len(identities))
            return result
        primary = {}
        for split in ['calibration', 'evaluation']:
            rows = [r for r in observations if r['cohort'] == 'primary' and r['split'] == split]
            primary[split] = {kind: counts([r for r in rows if r['kind'] == kind], by_source=True)
                              for kind in ['bound', *plan['conditions'], 'same']}
            edited = [r for r in rows if r['kind'] in plan['conditions']]
            primary[split]['all_edits'] = counts(edited)
            # Cluster bootstrap resamples source images, retaining all nine edits.
            identities = sorted({r['id'] for r in edited})
            arrays = np.asarray([[sum(r['accepted'][m] for r in edited if r['id'] == i)/9
                                  for i in identities] for m in METHODS])
            indices = np.random.default_rng(2026111601).integers(0, 50, size=(10000, 50))
            sampled = arrays[:, indices].mean(axis=-1)
            for j, method in enumerate(METHODS):
                primary[split]['all_edits'][method]['source_bootstrap_ci95'] = np.quantile(sampled[j], [.025, .975]).tolist()
                primary[split]['all_edits'][method]['paired_difference_ci95'] = np.quantile(sampled[j]-sampled[0], [.025, .975]).tolist()
        attack_results, attack_genuine = [], []
        attack_rows = [r for r in observations if r['cohort'] == 'attack_cohort']
        for scope in ['all_available_sources', 'original_evaluation_or_transfer']:
            scoped = [r for r in attack_rows if scope == 'all_available_sources' or r['split'] in ['evaluation', 'transfer']]
            for carrier in ['GS', 'TR', 'PRC']:
                genuine = [r for r in scoped if r['carrier'] == carrier and r['kind'] in ['clean', *plan['conditions']]]
                attack_genuine.append(dict(scope=scope, carrier=carrier, sources=len({r['id'] for r in genuine}), results=counts(genuine)))
                for attack_name in ['Reprompt', 'Reprompt+', 'imprint_50', 'imprint_100', 'imprint_150', 'imprint_any_checkpoint']:
                    for proxy in ['sd21', 'sd15']:
                        part = [r for r in scoped if r['carrier'] == carrier and r['proxy'] == proxy and
                                (r['kind'].startswith('imprint_') if attack_name == 'imprint_any_checkpoint' else r['kind'] == attack_name)]
                        if not part:
                            continue
                        per_source = [sum(r['id'] == identity for r in part) for identity in sorted({r['id'] for r in part})]
                        expected = (9 if carrier == 'GS' else 3) if attack_name == 'Reprompt+' else 3 if attack_name == 'imprint_any_checkpoint' else 1
                        assert set(per_source) == {expected}, (attack_name, carrier, per_source)
                        attack_results.append(dict(scope=scope, carrier=carrier, attack=attack_name, proxy=proxy,
                                                   candidates=len(part), candidates_per_source=expected, results=counts(part, True)))
        verdicts = {}
        for length in plan['lengths']:
            method = f'asymmetric{length}'
            same = primary['evaluation']['same'][method]['accepted']
            genuine = primary['evaluation']['all_edits'][method]
            excess = [dict(scope=r['scope'], carrier=r['carrier'], attack=r['attack'], proxy=r['proxy'],
                           excess=r['results'][method]['minus_sscd']) for r in attack_results if r['attack'].startswith('Reprompt')]
            criteria = same <= 5 and all(row['excess'] <= 1 for row in excess)
            verdicts[str(length)] = dict(same_prompt_accepted=same, same_prompt_n=50,
                same_prompt_criterion=same <= 5, reprompt_criterion=all(row['excess'] <= 1 for row in excess),
                reprompt_excess=excess, user_error_criteria_passed=criteria,
                evaluation_genuine_accepted=genuine['accepted'], evaluation_genuine_n=genuine['n'],
                heldout_retention_at_least95=genuine['accepted']/genuine['n'] >= .95,
                full_pass_including_heldout95=criteria and genuine['accepted']/genuine['n'] >= .95)
        checks.update(registered_records=len(enrollments), packed256_bytes=32, packed128_bytes=16,
                      reference_feature_max_difference=max_reference_difference, stored_distance_max_difference=max_distance_difference,
                      calibration_attack_reference_pixel_overlap=0, new_images=0, new_inferences=0, new_attacks=0)
        np.savez_compressed(run / 'packed_codes.npz', **enrollments)
        write(run / 'rows.json', observations)
        write(run / 'results.json', dict(primary=primary, attack_genuine=attack_genuine, attacks=attack_results, verdicts=verdicts))
        state.update(status='complete', seconds=time.monotonic()-started, paired_observations=len(observations),
                     method_decisions=len(observations)*len(METHODS), checks=checks,
                     artifacts={n: sha(run / n) for n in ['rows.json', 'results.json', 'thresholds.json', 'packed_codes.npz', 'public_projection.npy']})
        write(run / 'complete.json', state)
        print(json.dumps(dict(status='complete', verdicts=verdicts, checks=checks, seconds=state['seconds'])), flush=True)
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        write(run / 'state.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    assert args.run.resolve().parent == ROOT / 'runs'
    os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', OMP_NUM_THREADS='4')
    prepare(args.run.resolve()) if args.mode == 'prepare' else execute(args.run.resolve())
