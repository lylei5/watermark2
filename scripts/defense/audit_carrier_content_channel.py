"""Post-hoc mechanism controls and source-bootstrap audit, executed on 249."""
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path('/data0/liyanlei/watermark_research')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert ROOT.is_dir()
    run = ROOT / 'runs/carrier_content_channel_20261113'
    plan = json.loads((run / 'protocol.json').read_text())
    conditions = [c['name'] for c in plan['conditions']]
    done = json.loads((run / 'complete.json').read_text())
    rows = json.loads((run / 'rows.json').read_text())
    assert done['protocol_sha256'] == sha(run / 'protocol.json')
    assert done['artifacts']['rows.json'] == sha(run / 'rows.json')
    assert len(rows) == 130 and all(r['ba_replay_equal'] for r in rows)
    for r in rows:
        assert sha(run / r['latent']) == r['latent_sha256']
    os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='4')
    import numpy as np
    clean = {r['task']: r for r in rows if r['kind'] == 'clean'}
    genuine = [r for r in rows if r['kind'] not in ['clean', 'imprint']]
    controls = {
        'coordinate_ber': (lambda r:r['coordinate_ber'], True),
        'coordinate_ber_minus_clean': (lambda r:r['coordinate_ber']-clean[r['task']]['coordinate_ber'], True),
        'clean_coordinate_ber_only': (lambda r:clean[r['task']]['coordinate_ber'], False),
        'postmajority_bit_error': (lambda r:1-r['ba'], True),
        'repetition_disagreement': (lambda r:r['repetition_disagreement'], True),
        'spatial_error_std': (lambda r:r['spatial_error_std'], False),
        'latent_rms': (lambda r:r['latent_rms'], False),
        'latent_rms_difference_from_clean': (lambda r:abs(r['latent_rms']-clean[r['task']]['latent_rms']), False),
    }

    def fit(part, feature, nonnegative):
        x = np.array([feature(r) for r in part])
        y = np.array([r['content_distance'] for r in part])
        divisor = float(np.sum((x-x.mean())**2))
        b = float(np.sum((x-x.mean())*(y-y.mean())) / divisor) if divisor > 1e-15 else 0.
        if nonnegative:
            b = max(0., b)
        return float(y.mean()-b*x.mean()), b

    def cross_validation(part, feature, nonnegative):
        losses = []
        for task in range(10):
            train, test = [r for r in part if r['task'] != task], [r for r in part if r['task'] == task]
            a, b = fit(train, feature, nonnegative)
            mean = float(np.mean([r['content_distance'] for r in train]))
            losses.append([sum((r['content_distance']-a-b*feature(r))**2 for r in test),
                           sum((r['content_distance']-mean)**2 for r in test)])
        losses = np.array(losses)
        return float(1-losses[:, 0].sum()/losses[:, 1].sum()), losses

    results = []
    primary_losses = None
    for name, (feature, nonnegative) in controls.items():
        a, b = fit([r for r in genuine if r['task'] in plan['fit_sources']], feature, nonnegative)
        cal = [r for r in genuine if r['task'] in plan['calibration_sources']]
        residuals = sorted(r['content_distance']-a-b*feature(r) for r in cal)
        threshold = residuals[math.ceil(.95*len(cal))-1]
        decisions = []
        for kind in ['clean', 'distortions', 'imprint']:
            part = [r for r in rows if r['task'] in plan['evaluation_sources'] and
                    ((r['kind'] in conditions) if kind == 'distortions' else
                     r['kind'] == kind and (kind != 'imprint' or r['checkpoint'] == 150))]
            decisions.append(dict(kind=kind, total=len(part),
                accepted=sum(r['native'] and r['content_distance']-a-b*feature(r) <= threshold for r in part)))
        r2, losses = cross_validation(genuine, feature, nonnegative)
        results.append(dict(predictor=name, intercept=a, slope=b, residual_threshold=threshold,
            source_held_out_r2=r2, evaluation=decisions))
        if name == 'coordinate_ber':
            primary_losses = losses
    rng = np.random.default_rng(2026111321)
    r2_bootstrap, correlations = [], []
    array = np.array([[[r['coordinate_ber'], r['content_distance']] for c in conditions
                      for r in genuine if r['task'] == task and r['kind'] == c] for task in range(10)])
    assert array.shape == (10, 9, 2)
    for _ in range(2000):
        sources = rng.integers(0, 10, size=10)
        losses = primary_losses[sources]
        r2_bootstrap.append(1-losses[:, 0].sum()/losses[:, 1].sum())
        part = array[sources]
        centered = (part-part.mean(axis=0, keepdims=True)).reshape(-1, 2)
        correlations.append(float(np.corrcoef(centered.T)[0, 1]))
    sensitivity = []
    for omitted in conditions:
        part = [r for r in genuine if r['kind'] != omitted]
        r2, _ = cross_validation(part, *controls['coordinate_ber'])
        sensitivity.append(dict(omitted=omitted, source_held_out_r2=r2))
    result = dict(scope='Post-hoc controls after primary scores; development evidence only',
        source_sha256=sha(__file__), protocol_sha256=sha(run / 'protocol.json'),
        rows_sha256=sha(run / 'rows.json'), artifact_latents_verified=130, exact_native_ba_replays=130,
        controls=results, bootstrap=dict(repetitions=2000, unit='source',
            source_held_out_r2_percentile95=np.quantile(r2_bootstrap, [.025, .975]).tolist(),
            within_condition_correlation_percentile95=np.quantile(correlations, [.025, .975]).tolist(),
            note='Held-out predictions fixed before resampling losses; exploratory interval on10 sources'),
        leave_condition_out_sensitivity=sensitivity)
    with (run / 'mechanism_audit.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
