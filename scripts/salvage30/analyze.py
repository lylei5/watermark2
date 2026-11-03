"""Aggregate and independently verify a truncated 30-pair run, with post-hoc statistics."""
import math
import sys
import statistics
from pathlib import Path

ROOT = Path('/data0/liyanlei/watermark_research')
sys.path.insert(0, str(ROOT / 'scripts/confirmation100'))
import common100 as m  # noqa: E402

POINTS = m.POINTS
PRIMARY = str(max(POINTS))


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / d, (centre + half) / d


def clopper(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def summarize(rows):
    bas = [r['bit_accuracy'] for r in rows]
    psnrs = [r['psnr_db'] for r in rows if r['psnr_db'] is not None]
    return dict(n=len(rows), gs_passes=sum(bool(r['gs_detected']) for r in rows),
                mean_ba=statistics.mean(bas), median_ba=statistics.median(bas),
                min_ba=min(bas), max_ba=max(bas),
                q1_ba=float(sorted(bas)[len(bas) // 4]), q3_ba=float(sorted(bas)[3 * len(bas) // 4]),
                mean_psnr_db=statistics.mean(psnrs) if psnrs else None,
                mean_ssim=statistics.mean(r['ssim_grayscale'] for r in rows),
                mean_ms_ssim=statistics.mean(r['ms_ssim_rgb'] for r in rows))


def main(run_dir, n_tasks):
    run = Path(run_dir)
    n = int(n_tasks)
    import numpy as np
    from PIL import Image
    private = m.module('salvage_private_cpu', m.BASE / 'reference_reliability/private_state.py')
    decode = {'np': np}
    m.r.definitions(m.BASE / 'cross_generator/verify_message_scores.py', {'decode'}, decode)
    seal = m.c.read(run / 'candidates_sealed_30.json')
    if len(seal['rows']) != len(POINTS) * n:
        raise ValueError('seal differs')
    for sid in range(1, n + 1):
        out = run / ('attack_' + str(sid))
        frozen = m.c.read(out / 'candidates_frozen.json')
        if frozen['job_sha256'] != m.c.sha(run / 'attack_inputs' / str(sid) / 'job.json'):
            raise ValueError('job hash mismatch')
        for point in POINTS:
            if frozen['images'][str(point)] != m.c.png_record(out / ('update_' + str(point) + '.png')):
                raise ValueError('checkpoint changed')
    rows = []
    for shard in range(5):
        part = m.c.read(run / ('evaluate_s30_' + str(shard) + '.json'))
        if not part.get('completed'):
            raise ValueError('incomplete shard')
        rows.extend(part['rows'])
    if len(rows) != (2 + len(POINTS)) * n:
        raise ValueError('wrong denominator ' + str(len(rows)))
    errors = []
    decoded = 0
    for row in rows:
        sid = row['task_id']
        record = m.c.read(run / 'prepare' / str(sid) / 'record.json')
        raw = private.load(run.parent.parent / 'private_eval' / run.name, sid, record['encryption_fingerprints'])
        folder = run / 'evaluate' / row['folder']
        if m.c.sha(folder / 'target_inverse.npy') != row['inverse']['sha256']:
            raise ValueError('inverse changed')
        ba = decode['decode'](np.load(folder / 'target_inverse.npy', allow_pickle=False), raw)
        if ba != row['bit_accuracy']:
            raise ValueError('cpu decode differs')
        decoded += 1
        path = (run / 'prepare' / str(sid) / (row['label'] + '.png')) if row['label'] in ('reference', 'cover') \
            else (run / ('attack_' + str(sid)) / ('update_' + row['label'] + '.png'))
        if m.c.png_record(path) != row['image']:
            raise ValueError('image changed')
        with Image.open(path) as im:
            a = np.asarray(im, dtype=np.float64) / 255
        with Image.open(run / 'prepare' / str(sid) / 'cover.png') as im:
            b = np.asarray(im, dtype=np.float64) / 255
        errors.append(abs(float(np.mean((a - b) ** 2)) - row['mse_rgb01']))
    labels = ['reference', 'cover'] + [str(p) for p in POINTS]
    summary = {lab: summarize([r for r in rows if r['label'] == lab]) for lab in labels}
    prim = summary[PRIMARY]
    wlo, whi = wilson(prim['gs_passes'], prim['n'])
    clo, chi = clopper(prim['gs_passes'], prim['n'])
    per_task = []
    for sid in range(1, n + 1):
        tr = {r['label']: r for r in rows if r['task_id'] == sid}
        p = tr[PRIMARY]
        per_task.append(dict(task_id=sid, bit_accuracy=p['bit_accuracy'], gs_detected=p['gs_detected'],
                             psnr_db=p['psnr_db'], ssim_grayscale=p['ssim_grayscale'],
                             ms_ssim_rgb=p['ms_ssim_rgb'], cover_ba=tr['cover']['bit_accuracy'],
                             reference_ba=tr['reference']['bit_accuracy']))
    bas = [r['bit_accuracy'] for r in rows if r['label'] == PRIMARY]
    rng = np.random.default_rng(20261103)
    arr = np.asarray(bas)
    boot = arr[rng.integers(0, len(arr), size=(10000, len(arr)))].mean(axis=1)
    psnr = [r['psnr_db'] for r in rows if r['label'] == PRIMARY and r['psnr_db'] is not None]
    from scipy.stats import spearmanr
    rho, pval = spearmanr(bas, psnr)
    secrets = {}
    for sid in range(1, n + 1):
        record = m.c.read(run / 'prepare' / str(sid) / 'record.json')
        secrets[sid] = private.load(run.parent.parent / 'private_eval' / run.name, sid, record['encryption_fingerprints'])
    matrix = np.zeros((n + 1, n + 1))
    for sid in range(1, n + 1):
        inv = np.load(run / 'evaluate' / (str(sid) + '_' + PRIMARY) / 'target_inverse.npy', allow_pickle=False)
        for other in range(1, n + 1):
            matrix[sid, other] = decode['decode'](inv, secrets[other])
    diag = np.array([matrix[i, i] for i in range(1, n + 1)])
    off = matrix[1:, 1:][~np.eye(n, dtype=bool)]
    analysis = dict(completed=True, n=n, rows=rows, summary=summary, per_task=per_task,
                    failures=[r for r in per_task if not r['gs_detected']],
                    primary=dict(label=PRIMARY, n=prim['n'], gs_passes=prim['gs_passes'],
                                 detection_rate=prim['gs_passes'] / prim['n'],
                                 wilson95=[wlo, whi], clopper95=[clo, chi],
                                 mean_ba=prim['mean_ba'], median_ba=prim['median_ba'],
                                 mean_psnr_db=prim['mean_psnr_db'], q1_ba=prim['q1_ba'], q3_ba=prim['q3_ba']),
                    independent_cpu_redecodes=decoded, max_mse_abs_error=max(errors),
                    bootstrap_mean_ba95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                    ba_psnr_spearman=dict(rho=float(rho), p=float(pval), n=len(bas)),
                    cross_secret=dict(diagonal_mean=float(diag.mean()), offdiagonal_mean=float(off.mean()),
                                      offdiagonal_max=float(off.max()),
                                      offdiagonal_ge_threshold=int((off > 0.70703125).sum()),
                                      offdiagonal_pairs=int(off.size)),
                    truncated=True, note='Queue stopped at 30 completed pairs by user instruction; 4 images/task evaluated.')
    m.c.write(run / 'analysis_s30.json', analysis)
    print('PRIMARY', n, prim['gs_passes'], round(prim['mean_ba'], 4), [round(x, 3) for x in (clo, chi)])


if __name__ == '__main__':
    main(*sys.argv[1:])
