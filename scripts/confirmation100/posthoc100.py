"""Post-hoc analysis for the confirmation100 truncated screen (CPU only, not part of the frozen primary)."""
import json
import math
import os
import statistics
import sys
from pathlib import Path

if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise SystemExit('CPU only')

ROOT = Path('/data0/liyanlei/watermark_research')
sys.path.insert(0, str(ROOT / 'scripts/confirmation100'))
import common100 as m

RUN = Path(sys.argv[1])
PRIVATE = Path(sys.argv[2])
OUT = Path(sys.argv[3])


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def bootstrap_mean(values, n_boot=10000, seed=20261103):
    import numpy as np
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    import numpy as np
    analysis = m.c.read(RUN / 'analysis.json')
    rows = analysis['rows']
    primary_label = analysis['primary']['label']
    dec = {'np': np}
    m.r.definitions(m.BASE / 'cross_generator/verify_message_scores.py', {'decode'}, dec)
    private = m.module('confirmation100_posthoc_private', m.BASE / 'reference_reliability/private_state.py')

    n = len(m.c.read(RUN / 'design.json')['tasks'])
    secrets = {}
    records = {}
    for sid in range(1, n + 1):
        record = m.c.read(RUN / 'prepare' / str(sid) / 'record.json')
        records[sid] = record
        secrets[sid] = private.load(PRIVATE, sid, record['encryption_fingerprints'])

    by = {}
    for row in rows:
        by.setdefault(row['label'], []).append(row)

    out = {'schema': 1, 'run': str(RUN), 'primary_label': primary_label,
           'frozen_primary': analysis['primary'], 'note': 'Post-hoc additions; the frozen primary remains analysis.json.'}

    # exact interval + bootstrap for the primary detection rate and mean BA
    p = analysis['primary']
    k, n = p['gs_passes'], p['n']
    cp = clopper_pearson(k, n)
    bas = [row['bit_accuracy'] for row in by[primary_label]]
    boot = bootstrap_mean(bas)
    out['primary_exact'] = dict(k=k, n=n, rate=k / n, clopper_pearson95=[cp[0], cp[1]],
                                wilson95=p['wilson95'], mean_ba=statistics.mean(bas),
                                mean_ba_bootstrap95=list(boot),
                                median_ba=statistics.median(bas),
                                q1_ba=float(np.percentile(bas, 25)), q3_ba=float(np.percentile(bas, 75)),
                                min_ba=min(bas), max_ba=max(bas))

    # distortion summaries, pooled MSE PSNR
    psnr = [row['psnr_db'] for row in by[primary_label] if row['psnr_db'] is not None]
    mse = [row['mse_rgb01'] for row in by[primary_label]]
    out['distortion'] = dict(psnr_mean=statistics.mean(psnr), psnr_median=statistics.median(psnr),
                             psnr_q1=float(np.percentile(psnr, 25)), psnr_q3=float(np.percentile(psnr, 75)),
                             pooled_mse=statistics.mean(mse),
                             pooled_psnr_db=-10 * math.log10(statistics.mean(mse)),
                             ssim_mean=statistics.mean([row['ssim_grayscale'] for row in by[primary_label]]),
                             ms_ssim_mean=statistics.mean([row['ms_ssim_rgb'] for row in by[primary_label]]))

    # trajectory
    traj = {}
    for label in ['20', '30']:
        rs = by[label]
        traj[label] = dict(n=len(rs), gs_passes=sum(r['gs_detected'] for r in rs),
                           mean_ba=statistics.mean(r['bit_accuracy'] for r in rs),
                           median_psnr=statistics.median(r['psnr_db'] for r in rs if r['psnr_db'] is not None))
    out['trajectory'] = traj

    # null controls
    cover = by['cover']
    out['null_control'] = dict(n=len(cover), mean_ba=statistics.mean(r['bit_accuracy'] for r in cover),
                               gs_passes=sum(r['gs_detected'] for r in cover),
                               min_ba=min(r['bit_accuracy'] for r in cover),
                               max_ba=max(r['bit_accuracy'] for r in cover))
    ref = by['reference']
    out['reference_control'] = dict(n=len(ref), mean_ba=statistics.mean(r['bit_accuracy'] for r in ref),
                                    gs_passes=sum(r['gs_detected'] for r in ref))

    # Spearman BA vs PSNR at primary
    from scipy.stats import spearmanr
    rho, pval = spearmanr(bas, psnr)
    out['ba_psnr_spearman'] = dict(rho=float(rho), p=float(pval), n=len(bas))

    # 100x100 cross-secret matrix on the primary checkpoint
    matrix = np.zeros((n + 1, n + 1), dtype=np.float64)
    for sid in range(1, n + 1):
        inv = np.load(RUN / 'evaluate' / (str(sid) + '_' + primary_label) / 'target_inverse.npy', allow_pickle=False)
        for other in range(1, 101):
            matrix[sid, other] = dec['decode'](inv, secrets[other])
    diag = np.array([matrix[i, i] for i in range(1, n + 1)])
    off = matrix[1:, 1:][~np.eye(n, dtype=bool)]
    out['cross_secret'] = dict(diagonal_mean=float(diag.mean()), offdiagonal_mean=float(off.mean()),
                               offdiagonal_max=float(off.max()), offdiagonal_min=float(off.min()),
                               offdiagonal_ge_threshold=int((off > 0.70703125).sum()),
                               offdiagonal_pairs=int(off.size),
                               diagonal_ge_threshold=int((diag > 0.70703125).sum()))
    out['failures'] = [dict(task_id=r['task_id'], bit_accuracy=r['bit_accuracy'], psnr_db=r['psnr_db'])
                       for r in analysis['per_task'] if not r['gs_detected']]
    m.c.write(OUT, out)
    print(json.dumps({k: out[k] for k in ('primary_exact', 'distortion', 'trajectory', 'null_control',
                                          'reference_control', 'ba_psnr_spearman', 'cross_secret')}, indent=2))
    print('FAILURES', len(out['failures']))


if __name__ == '__main__':
    main()

