"""Same-pixel calibrated and descriptive matched-TPR content comparisons on249."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path('/data0/liyanlei/watermark_research')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    run = a.run.resolve()
    assert ROOT.is_dir() and run.parent == ROOT / 'runs'
    os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4')
    import hashlib
    import numpy as np
    def read(path):
        return json.loads(path.read_text())
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    result, plan = read(run / 'summary.json'), read(run / 'protocol.json')
    assert result['status'] == 'complete'
    assert sha(run / 'rows.json') == result['artifacts']['rows.json']
    parent = ROOT / 'runs/shared_channel_binding_20261114_v2'
    assert sha(parent / 'controls.json') == result['parent_controls_sha256']
    controls = read(parent / 'controls.json')
    rows = {(r['id'], r['query_id'], r['kind']): r for r in read(run / 'rows.json')}
    assert len(rows) == len(controls) == 710
    for c in controls:
        r = rows[c['id'], c['query_id'], c['kind']]
        assert r['pixels_sha256'] == c['pixels_sha256'] and r['native_accepted'] == c['native_accepted']
        c['copynce_descriptor'] = r['descriptor_distance']
        c['copynce_matcher'] = r['matcher_distance']
    def genuine(r):
        return r['kind'] in ['clean', *plan['conditions']]
    methods = ['sscd', 'sscd_centered', 'sscd_templates', 'sscd_int8_templates', 'copynce_descriptor', 'copynce_matcher']
    thresholds, calibrated, matched = {}, [], []
    for method in methods:
        values = sorted(r[method] for r in controls if r['split'] == 'calibration' and genuine(r))
        assert len(values) == 30
        for target, rank in [(.9, 26), (.95, 28)]:
            thresholds[method, target] = values[rank]
    def summarize(part, method, cutoff):
        result = []
        for kind in ['all_genuine', 'unknown_edits', 'same_prompt', 'wrong_record', 'cover', 'imprint_150']:
            group = [r for r in part if (genuine(r) if kind == 'all_genuine' else r['kind'] in plan['conditions'] and
                     r['kind'] not in plan['enrollment_families'] if kind == 'unknown_edits' else r['kind'] == kind)]
            if group:
                result.append(dict(kind=kind, total=len(group), accepted=sum(r[method] <= cutoff for r in group),
                    joint=sum(r[method] <= cutoff and r['native_accepted'] for r in group)))
        return result
    for carrier, split in [('GS', 'evaluation'), ('TR', 'transfer'), ('PRC', 'transfer')]:
        part = [r for r in controls if r['carrier'] == carrier and r['split'] == split]
        for method in methods:
            for target in [.9, .95]:
                cutoff = thresholds[method, target]
                calibrated.append(dict(carrier=carrier, split=split, method=method, target=target, threshold=cutoff,
                    summary=summarize(part, method, cutoff)))
        maximum = sum(r['native_accepted'] for r in part if genuine(r))
        for count in ([27, 28, 29, 30] if carrier == 'GS' else [60, 89, 90, 91, 95, 97, 98, 100] if carrier == 'TR' else [29, 60, 64]):
            if count > maximum:
                continue
            for method in methods:
                values = sorted(r[method] for r in part if genuine(r) and r['native_accepted'])
                cutoff = values[count-1]
                matched.append(dict(carrier=carrier, split=split, method=method, target_joint_count=count, threshold=cutoff,
                    summary=summarize(part, method, cutoff)))
    bootstrap = []
    for carrier, split in [('GS', 'evaluation'), ('TR', 'transfer'), ('PRC', 'transfer')]:
        for target in [.9, .95]:
            for condition in ['all_genuine', 'unknown_edits']:
                part = [r for r in controls if r['carrier'] == carrier and r['split'] == split and genuine(r) and
                        (condition == 'all_genuine' or r['kind'] in plan['conditions'] and r['kind'] not in plan['enrollment_families'])]
                tasks = sorted({r['task'] for r in part})
                per_source = []
                for task in tasks:
                    group = [r for r in part if r['task'] == task]
                    per_source.append(sum(int(r['native_accepted'] and r['copynce_matcher'] <= thresholds['copynce_matcher', target])-
                                          int(r['native_accepted'] and r['sscd'] <= thresholds['sscd', target]) for r in group)/len(group))
                values = np.asarray(per_source)
                rng = np.random.default_rng(2026111407)
                draws = values[rng.integers(0, len(tasks), size=(10000, len(tasks)))].mean(axis=1)
                bootstrap.append(dict(carrier=carrier, target=target, condition=condition, sources=len(tasks), queries=len(part),
                    difference_pp=float(100*values.mean()), interval95_pp=(100*np.quantile(draws, [.025, .975])).tolist()))
    output = run / 'comparison.json'
    assert not output.exists()
    output.write_text(json.dumps(dict(status='complete', source_sha256=sha(Path(__file__)),
        inputs={n: sha(run / n) for n in ['summary.json', 'rows.json', 'protocol.json']},
        calibrated=calibrated, evaluation_matched=matched, source_bootstrap=bootstrap,
        matched_scope='Evaluation-selected thresholds for descriptive equal absolute joint legitimate counts; ties may exceed requested counts',
        bootstrap_scope='Paired source resampling with frozen thresholds on development data; conditional descriptive intervals',
        information_scope='Matcher receives full reference image; descriptor and SSCD use stored vectors; no equal-storage superiority claim',
        independent_confirmation=False), indent=2) + '\n')
    print(json.dumps(dict(calibrated_points=len(calibrated), evaluation_matched_points=len(matched), bootstrap=bootstrap)), flush=True)


if __name__ == '__main__':
    main()
