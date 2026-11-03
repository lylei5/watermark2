"""Evaluate a truncated confirmation100 run (30 pairs) in shards, reusing the frozen code paths."""
import math
import sys
import time
from pathlib import Path

ROOT = Path('/data0/liyanlei/watermark_research')
sys.path.insert(0, str(ROOT / 'scripts/confirmation100'))
import common100 as m  # noqa: E402


def main(run_dir, shard, nshard, n_tasks):
    run = Path(run_dir)
    shard, nshard, n = int(shard), int(nshard), int(n_tasks)
    design = m.c.read(run / 'design.json')
    seal = m.c.read(run / 'candidates_sealed_30.json')
    if len(seal['rows']) != len(m.POINTS) * n:
        raise ValueError('seal row count mismatch')
    for task in design['tasks'][:n]:
        sid = task['id']
        out = run / ('attack_' + str(sid))
        frozen = m.c.read(out / 'candidates_frozen.json')
        if frozen['job_sha256'] != m.c.sha(run / 'attack_inputs' / str(sid) / 'job.json'):
            raise ValueError('job hash mismatch ' + str(sid))
        for point in m.POINTS:
            if frozen['images'][str(point)] != m.c.png_record(out / ('update_' + str(point) + '.png')):
                raise ValueError('checkpoint changed ' + str(sid))
    private = m.module('salvage_private', m.BASE / 'reference_reliability/private_state.py')
    import numpy as np
    import torch
    from PIL import Image
    from skimage.metrics import structural_similarity
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
    provider = m.r.official_provider(design['target'])
    arrays = []
    m.guard(provider.scheduler_inverse, arrays)
    provider.scheduler_inverse.set_timesteps(50, device='cuda:0')
    eval_root = run / 'evaluate'
    eval_root.mkdir(exist_ok=True)
    rows = []
    for task in design['tasks'][:n]:
        sid = task['id']
        if (sid - 1) % nshard != shard:
            continue
        record = m.c.read(run / 'prepare' / str(sid) / 'record.json')
        raw = private.load(run.parent.parent / 'private_eval' / run.name, sid, record['encryption_fingerprints'])
        wm = private.provider(raw, 'cuda:0')
        with Image.open(run / 'prepare' / str(sid) / 'cover.png') as im:
            cover = im.convert('RGB').copy()
        items = [('reference', run / 'prepare' / str(sid) / 'reference.png'),
                 ('cover', run / 'prepare' / str(sid) / 'cover.png')]
        items += [(str(p), run / ('attack_' + str(sid)) / ('update_' + str(p) + '.png')) for p in m.POINTS]
        for label, path in items:
            folder = eval_root / (str(sid) + '_' + label)
            folder.mkdir()
            with Image.open(path) as im:
                image = im.convert('RGB')
            start = time.perf_counter()
            with torch.no_grad():
                z = provider.invert_images(image, num_inference_steps=50)['zT_torch']
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            ba = float(wm.get_accuracies(z)['bit_accuracies'][0])
            a = np.asarray(image, dtype=np.float64) / 255
            b = np.asarray(cover, dtype=np.float64) / 255
            mse = float(np.mean((a - b) ** 2))
            psnr = None if mse == 0 else -10 * math.log10(mse)
            ssim = float(structural_similarity(np.asarray(image.convert('L')), np.asarray(cover.convert('L')), data_range=255))
            ta = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).float()
            tb = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).float()
            official_psnr = float(PeakSignalNoiseRatio(data_range=1.0)(ta, tb))
            ms = float(MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)(ta, tb))
            row = dict(task_id=sid, label=label, image=m.c.png_record(path),
                       inverse=m.r.tensor(folder / 'target_inverse.npy', z), bit_accuracy=ba,
                       gs_detected=ba > .70703125, mse_rgb01=mse, psnr_db=psnr,
                       official_torchmetrics_psnr_db=None if mse == 0 else official_psnr,
                       psnr_infinite=mse == 0, ssim_grayscale=ssim, ms_ssim_rgb=ms, lpips=None,
                       inversion_seconds=seconds)
            m.c.write(folder / 'metrics.json', row)
            rows.append(dict(**row, folder=folder.name))
            print('EVAL', sid, label, flush=True)
    m.c.write(run / ('evaluate_s30_' + str(shard) + '.json'),
              dict(completed=True, shard=shard, rows=rows, n=n))
    print('SHARD_DONE', shard, len(rows))


if __name__ == '__main__':
    main(*sys.argv[1:])
