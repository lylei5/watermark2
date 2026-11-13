"""Post-hoc information-loss control on immutable spectral observations."""
import argparse
import math
import os
from pathlib import Path

from shared_channel_experiment import ROOT, read, write, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    assert ROOT.is_dir() and run.parent == ROOT / 'runs'
    os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4')
    import numpy as np
    from shared_channel_features import spectrum, phase, complex_vector, unit

    output = run / 'spectral_information.json'
    assert not output.exists()
    plan = read(run / 'protocol.json')
    done = read(run / 'extract_complete.json')
    assert done['status'] == 'complete'
    for name, digest in done['artifacts'].items():
        assert sha(run / name) == digest
    meta = read(run / 'feature_rows.json')
    with np.load(run / 'features.npz') as archive:
        data = {k: archive[k] for k in archive.files}
    with np.load(run / 'descriptors.npz') as archive:
        pooled = {k: archive[k] for k in ['shared_phase', 'content_phase', 'noise_phase']}
    index = {(r['id'], r['kind']): i for i, r in enumerate(meta)}
    fit = [i for i, r in enumerate(meta) if r['carrier'] == 'GS' and
           r['task'] in plan['fit_sources'] and (r['kind'] == 'clean' or r['kind'].startswith('enroll_'))]
    assert len(fit) == 36
    full = {k: [] for k in pooled}
    for a, b in zip(data['content'], data['noise']):
        pa, pb = phase(spectrum(a)), phase(spectrum(b))
        for method, values in [('shared_phase', pa * np.conj(pb)), ('content_phase', pa), ('noise_phase', pb)]:
            full[method].append(complex_vector(values))
    features = {f'{k}_pooled': v for k, v in pooled.items()}
    for method, values in full.items():
        values = np.asarray(values)
        assert values.shape == (520, 16384)
        features[f'{method}_full'] = unit(values - values[fit].mean(axis=0))
    rows, scores = [], []
    for method, values in features.items():
        current = []
        for i, r in enumerate(meta):
            if r['kind'].startswith('enroll_'):
                continue
            split = ('fit' if r['task'] in plan['fit_sources'] else 'calibration' if r['task'] in plan['calibration_sources'] else 'evaluation') if r['carrier'] == 'GS' else 'transfer'
            ref = index[r['id'], 'clean']
            common = dict(id=r['id'], carrier=r['carrier'], task=r['task'], split=split)
            current.append(dict(**common, query_id=r['id'], kind=r['kind'], distance=float(np.clip(1-values[ref] @ values[i], 0, 2))))
            if r['kind'] == 'clean':
                for j, other in enumerate(meta):
                    if other['kind'] == 'clean' and other['carrier'] == r['carrier'] and other['id'] != r['id']:
                        current.append(dict(**common, query_id=other['id'], kind='wrong_record', distance=float(np.clip(1-values[ref] @ values[j], 0, 2))))
        assert len(current) == 710
        calibration = sorted(r['distance'] for r in current if r['split'] == 'calibration' and r['kind'] in ['clean', *plan['conditions']])
        assert len(calibration) == 30
        for target in plan['calibration_targets']:
            cutoff = calibration[math.ceil(target * 30)-1]
            for carrier, split in [('GS', 'evaluation'), ('TR', 'transfer'), ('PRC', 'transfer')]:
                for kind in ['all_genuine', 'unknown_edits', 'same_prompt', 'wrong_record', 'imprint_150']:
                    part = [r for r in current if r['carrier'] == carrier and r['split'] == split and
                            (r['kind'] in ['clean', *plan['conditions']] if kind == 'all_genuine' else
                             r['kind'] in plan['conditions'] and r['kind'] not in plan['enrollment_families'] if kind == 'unknown_edits' else r['kind'] == kind)]
                    scores.append(dict(method=method, dimensions=values.shape[1], target=target, threshold=cutoff,
                                       carrier=carrier, split=split, kind=kind, total=len(part), accepted=sum(r['distance'] <= cutoff for r in part)))
        rows.extend(dict(method=method, **r) for r in current)
    write(output, dict(status='complete', source_sha256=sha(__file__), protocol_sha256=sha(run / 'protocol.json'),
                       scope='Post-hoc content-only information-loss diagnostic; full vectors use 32 times the dimensions, not a matched-cost method or independent confirmation',
                       source_artifacts=done['artifacts'], scores=scores, rows=rows))
    for row in scores:
        if row['target'] == .95 and row['carrier'] == 'TR' and row['kind'] in ['all_genuine', 'same_prompt', 'wrong_record']:
            print(row, flush=True)


if __name__ == '__main__':
    main()
