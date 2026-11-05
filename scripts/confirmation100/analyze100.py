import math
import os
import statistics
from pathlib import Path
from common100 import c, r, module, BASE, POINTS, candidates


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / d, (centre + half) / d


def summarize(rows):
    bas = [row['bit_accuracy'] for row in rows]
    psnrs = [row['psnr_db'] for row in rows if row['psnr_db'] is not None]
    ssims = [row['ssim_grayscale'] for row in rows]
    mss = [row['ms_ssim_rgb'] for row in rows]
    times = [row['inversion_seconds'] for row in rows]
    return dict(n=len(rows), gs_passes=sum(bool(row['gs_detected']) for row in rows),
                mean_ba=statistics.mean(bas), median_ba=statistics.median(bas),
                min_ba=min(bas), max_ba=max(bas),
                mean_psnr_db=statistics.mean(psnrs) if psnrs else None,
                mean_ssim_grayscale=statistics.mean(ssims), mean_ms_ssim_rgb=statistics.mean(mss),
                mean_inversion_seconds=statistics.mean(times))


def analyze(root, private_folder):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU only')
    import numpy as np
    from PIL import Image
    root = Path(root)
    private = module('confirmation100_cpu_private', BASE / 'reference_reliability/private_state.py')
    decoder = {'np': np}
    r.definitions(BASE / 'cross_generator/verify_message_scores.py', {'decode'}, decoder)
    design = c.read(root / 'design.json')
    n = len(design['tasks'])
    if c.read(root / 'candidates_sealed.json')['rows'] != candidates(root, n):
        raise ValueError('Seal differs')
    shards = sorted(root.glob('evaluate_*'))
    if len(shards) != 5:
        raise ValueError('Expected five evaluation shards')
    rows = []
    for folder in shards:
        summary = c.read(folder / 'summary.json')
        if not summary.get('completed'):
            raise ValueError('Incomplete shard')
        rows.extend(summary['rows'])
    if len(rows) != (2 + len(POINTS)) * n:
        raise ValueError('Wrong evaluation denominator')
    errors = []
    decoded = 0
    for row in rows:
        sid = row['task_id']
        record = c.read(root / 'prepare' / str(sid) / 'record.json')
        raw = private.load(private_folder, sid, record['encryption_fingerprints'])
        folder = root / row['folder']
        if c.sha(folder / 'target_inverse.npy') != row['inverse']['sha256']:
            raise ValueError('Inverse changed')
        ba = decoder['decode'](np.load(folder / 'target_inverse.npy', allow_pickle=False), raw)
        if ba != row['bit_accuracy']:
            raise ValueError('CPU message decode differs')
        decoded += 1
        if row['label'] in ('reference', 'cover'):
            path = root / 'prepare' / str(sid) / (row['label'] + '.png')
        else:
            path = root / ('attack_' + str(sid)) / ('update_' + row['label'] + '.png')
        if c.png_record(path) != row['image']:
            raise ValueError('Image changed')
        with Image.open(path) as im:
            a = np.asarray(im, dtype=np.float64) / 255
        with Image.open(root / 'prepare' / str(sid) / 'cover.png') as im:
            b = np.asarray(im, dtype=np.float64) / 255
        mse = float(np.mean((a - b) ** 2))
        errors.append(abs(mse - row['mse_rgb01']))
    if max(errors) > 1e-15:
        raise ValueError('Distortion changed')
    labels = ['reference', 'cover'] + [str(n) for n in POINTS]
    summary = {label: summarize([row for row in rows if row['label'] == label]) for label in labels}
    primary_label = str(max(POINTS))
    primary = summary[primary_label]
    rate, low, high = wilson(primary['gs_passes'], primary['n'])
    per_task = []
    for sid in range(1, n + 1):
        task_rows = {row['label']: row for row in rows if row['task_id'] == sid}
        p = task_rows[primary_label]
        per_task.append(dict(task_id=sid, bit_accuracy=p['bit_accuracy'], gs_detected=p['gs_detected'],
                             psnr_db=p['psnr_db'], ssim_grayscale=p['ssim_grayscale'],
                             ms_ssim_rgb=p['ms_ssim_rgb'], cover_ba=task_rows['cover']['bit_accuracy'],
                             reference_ba=task_rows['reference']['bit_accuracy']))
    failures = [row for row in per_task if not row['gs_detected']]
    core = []
    wall = []
    for sid in range(1, n + 1):
        attack = root / ('attack_' + str(sid))
        core.append(c.read(attack / 'summary.json')['timing']['core_seconds'])
        wall.append(c.read(attack / 'execution.json')['outcome']['process_wall_seconds'])
    result = dict(completed=True, rows=rows, summary=summary, per_task=per_task, failures=failures,
                  primary=dict(label=primary_label, n=primary['n'], gs_passes=primary['gs_passes'],
                               detection_rate=rate, wilson95=[low, high], mean_ba=primary['mean_ba'],
                               median_ba=primary['median_ba'], mean_psnr_db=primary['mean_psnr_db']),
                  independent_cpu_redecodes=decoded, max_mse_abs_error=max(errors),
                  cost=dict(attack_core_seconds_sum=sum(core), attack_core_seconds_mean=statistics.mean(core),
                            attack_core_seconds_median=statistics.median(core),
                            attack_process_wall_seconds_sum=sum(wall),
                            gpu_hours_core=sum(core) / 3600),
                  truncated_schedule_actual_updates=max(POINTS),
                  published_endpoint_actual_updates=151,
                  full_paper_reproduction=False, calibration_only=False,
                  truncation_note='Truncated 40-update schedule (26% of the published 151-update loop), '
                                  'chosen for an 8-hour advisor deadline; the published endpoint is not reproduced by this run',
                  scope='Fixed 100-pair confirmation set with one independent evaluator-only identity per pair; '
                        'descriptive ordinary-GS rate at the truncated endpoint only, not a population attack success rate, '
                        'FPR, or SOTA claim; full geometric/SemBind acceptance and formal human content labels remain unknown',
                  lpips=None, formal_human_labels=None)
    c.write(root / 'analysis.json', result)
    return result

