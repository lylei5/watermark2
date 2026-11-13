"""Genuine-only, source-separated analysis of the frozen channel probe on 249."""
import argparse
import ctypes
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import select

ROOT = Path('/data0/liyanlei/watermark_research')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    run = args.run.resolve()
    assert ROOT.is_dir() and run.parent == ROOT / 'runs'
    plan = json.loads((run / 'protocol.json').read_text())
    receipt = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   pid=os.getpid(), parent_pid=args.wait_pid, source_sha256=sha(__file__),
                   protocol_sha256=sha(run / 'protocol.json'), mechanism='Kernel process-exit event, no polling')
    with (run / 'analysis_freeze.json').open('x') as f:
        json.dump(receipt, f, indent=2)
        f.write('\n')
    if args.wait_pid and not (run / 'complete.json').exists():
        libc = ctypes.CDLL(None, use_errno=True)
        descriptor = libc.syscall(434, args.wait_pid, 0)
        if descriptor < 0:
            if ctypes.get_errno() != 3:
                raise OSError(ctypes.get_errno(), 'pidfd_open')
        else:
            try:
                cmdline_path = Path(f'/proc/{args.wait_pid}/cmdline')
                cmdline = cmdline_path.read_bytes() if cmdline_path.exists() else b''
                assert not cmdline or str(run).encode() in cmdline
                select.select([descriptor], [], [])
            finally:
                os.close(descriptor)
    done = json.loads((run / 'complete.json').read_text())
    assert done['status'] == 'complete'
    assert done['protocol_sha256'] == sha(run / 'protocol.json')
    assert done['artifacts']['rows.json'] == sha(run / 'rows.json')
    rows = json.loads((run / 'rows.json').read_text())
    assert len(rows) == 130 and len({(r['task'], r['kind'], r['checkpoint']) for r in rows}) == 130
    import numpy as np
    from scipy.stats import spearmanr
    names = [c['name'] for c in plan['conditions']]
    genuine = [r for r in rows if r['kind'] in names]
    assert len(genuine) == 90

    def data(part):
        return np.array([r['coordinate_ber'] for r in part]), np.array([r['content_distance'] for r in part])

    def fit(part):
        x, y = data(part)
        denominator = float(np.sum((x-x.mean())**2))
        slope = max(0., float(np.sum((x-x.mean())*(y-y.mean()))) / denominator) if denominator else 0.
        return float(y.mean()-slope*x.mean()), slope

    def quantile(values):
        return float(sorted(values)[math.ceil(plan['target_distortion_retention']*len(values))-1])

    def correlation(x, y):
        return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None

    fitted = [r for r in genuine if r['task'] in plan['fit_sources']]
    calibrated = [r for r in genuine if r['task'] in plan['calibration_sources']]
    assert len(fitted) == 36 and len(calibrated) == 27
    intercept, slope = fit(fitted)
    content_threshold = quantile([r['content_distance'] for r in calibrated])
    residual_threshold = quantile([r['content_distance']-intercept-slope*r['coordinate_ber'] for r in calibrated])
    for r in rows:
        r['split'] = next(name for name in ['fit', 'calibration', 'evaluation'] if r['task'] in plan[name+'_sources'])
        r['predicted_content_distance'] = intercept+slope*r['coordinate_ber']
        r['residual'] = r['content_distance']-r['predicted_content_distance']
        r['content_fixed'] = r['content_distance'] <= plan['frozen_sscd_threshold']
        r['content_calibrated'] = r['content_distance'] <= content_threshold
        r['content_conditioned'] = r['residual'] <= residual_threshold
        for label in ['fixed', 'calibrated', 'conditioned']:
            r['joint_'+label] = r['native'] and r['content_'+label]
    folds, predicted, baseline, actual = [], [], [], []
    for idx in range(10):
        training = [r for r in genuine if r['task'] != idx]
        testing = [r for r in genuine if r['task'] == idx]
        a, b = fit(training)
        x, y = data(testing)
        prediction = a+b*x
        mean = float(np.mean([r['content_distance'] for r in training]))
        folds.append(dict(task=idx, intercept=a, slope=b,
                          mse=float(np.mean((y-prediction)**2)), constant_mse=float(np.mean((y-mean)**2))))
        predicted.extend(prediction.tolist())
        baseline.extend([mean]*len(y))
        actual.extend(y.tolist())
    actual, predicted, baseline = map(np.array, [actual, predicted, baseline])
    cross_validated_r2 = float(1-np.sum((actual-predicted)**2)/np.sum((actual-baseline)**2))
    x, y = data(genuine)
    centered_x, centered_y, conditions = [], [], []
    for name in names:
        part = [r for r in genuine if r['kind'] == name]
        a, b = data(part)
        centered_x.extend((a-a.mean()).tolist())
        centered_y.extend((b-b.mean()).tolist())
        conditions.append(dict(condition=name, n=len(part), mean_ber=float(a.mean()),
            mean_content_distance=float(b.mean()), pearson=correlation(a, b)))
    summary = []
    groups = [('clean', -1), ('all_distortions', -1)] + [(n, -1) for n in names] + [('imprint', n) for n in [50, 100, 150]]
    metrics = ['native', 'content_fixed', 'content_calibrated', 'content_conditioned', 'joint_fixed', 'joint_calibrated', 'joint_conditioned']
    for split in ['fit', 'calibration', 'evaluation', 'all']:
        for kind, checkpoint in groups:
            part = [r for r in rows if (split == 'all' or r['split'] == split) and
                    (r['kind'] in names if kind == 'all_distortions' else r['kind'] == kind and r['checkpoint'] == checkpoint)]
            assert part
            summary.append(dict(split=split, kind=kind, checkpoint=checkpoint, total=len(part),
                                **{k: sum(r[k] for r in part) for k in metrics}))
    result = dict(scope=plan['inference_scope'], protocol_sha256=sha(run / 'protocol.json'),
        rows_sha256=sha(run / 'rows.json'), source_sha256=sha(__file__),
        fit=dict(intercept=intercept, slope=slope), calibration=dict(n=27, target=.95,
        required_count=26, content_threshold=content_threshold, residual_threshold=residual_threshold),
        correlations=dict(pooled_pearson=correlation(x, y), pooled_spearman=float(spearmanr(x, y).statistic),
                          within_condition_pearson=correlation(centered_x, centered_y)),
        source_held_out_r2=cross_validated_r2, folds=folds, conditions=conditions,
        exact_ba_replays=sum(r['ba_replay_equal'] for r in rows),
        native_genuine=sum(r['native'] for r in rows if r['kind'] != 'imprint'), summary=summary)
    for name, value in [('scores.json', result), ('decision_rows.json', rows)]:
        with (run / name).open('x') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n')
    print(json.dumps({k: result[k] for k in ['fit', 'calibration', 'correlations', 'source_held_out_r2', 'exact_ba_replays']}))
    print(json.dumps([r for r in summary if r['split'] == 'evaluation' and r['kind'] in ['clean', 'all_distortions', 'imprint']]))


if __name__ == '__main__':
    main()
