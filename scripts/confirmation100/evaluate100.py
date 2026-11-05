import sys
import time
import math
from pathlib import Path
from common100 import c, r, module, gate, candidates, guard, POINTS, BASE


def main(runtime, root, private_folder, shard_index, shard_count, output):
    root = Path(root)
    out = Path(output)
    cfg, d = gate(runtime, out, 'evaluator')
    shard_index = int(shard_index)
    shard_count = int(shard_count)
    if not (0 <= shard_index < shard_count):
        raise ValueError('Invalid shard')
    seal = c.read(root / 'candidates_sealed.json')
    if seal['rows'] != candidates(root, len(d['tasks'])):
        raise ValueError('Candidates changed')
    import numpy as np
    import torch
    from PIL import Image
    from skimage.metrics import structural_similarity
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
    private = module('confirmation100_eval_private', BASE / 'reference_reliability/private_state.py')
    provider = r.official_provider(d['target'])
    arrays = []
    guard(provider.scheduler_inverse, arrays)
    provider.scheduler_inverse.set_timesteps(50, device='cuda:0')
    rows = []
    for task in d['tasks']:
        sid = task['id']
        if (sid - 1) % shard_count != shard_index:
            continue
        record = c.read(root / 'prepare' / str(sid) / 'record.json')
        raw = private.load(Path(private_folder), sid, record['encryption_fingerprints'])
        wm = private.provider(raw, 'cuda:0')
        with Image.open(root / 'prepare' / str(sid) / 'cover.png') as im:
            cover = im.convert('RGB').copy()
        items = [('reference', root / 'prepare' / str(sid) / 'reference.png'),
                 ('cover', root / 'prepare' / str(sid) / 'cover.png')]
        items += [(str(n), root / ('attack_' + str(sid)) / ('update_' + str(n) + '.png')) for n in POINTS]
        for label, path in items:
            folder = out / (str(sid) + '_' + label)
            folder.mkdir()
            with Image.open(path) as im:
                image = im.convert('RGB')
            start = time.perf_counter()
            with torch.no_grad():
                z = provider.invert_images(image, num_inference_steps=50)['zT_torch']
            torch.cuda.synchronize()
            inverse_seconds = time.perf_counter() - start
            ba = float(wm.get_accuracies(z)['bit_accuracies'][0])
            a = np.asarray(image, dtype=np.float64) / 255
            b = np.asarray(cover, dtype=np.float64) / 255
            mse = float(np.mean((a - b) ** 2))
            psnr = None if mse == 0 else -10 * math.log10(mse)
            gray1 = np.asarray(image.convert('L'))
            gray2 = np.asarray(cover.convert('L'))
            ssim = float(structural_similarity(gray1, gray2, data_range=255))
            ta = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).float()
            tb = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).float()
            official_psnr = float(PeakSignalNoiseRatio(data_range=1.0)(ta, tb))
            ms = float(MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)(ta, tb))
            row = dict(task_id=sid, label=label, image=c.png_record(path),
                       inverse=r.tensor(folder / 'target_inverse.npy', z), bit_accuracy=ba,
                       gs_detected=ba > .70703125, mse_rgb01=mse, psnr_db=psnr,
                       official_torchmetrics_psnr_db=None if mse == 0 else official_psnr,
                       psnr_infinite=mse == 0, ssim_grayscale=ssim, ms_ssim_rgb=ms, lpips=None,
                       inversion_seconds=inverse_seconds)
            c.write(folder / 'metrics.json', row)
            rows.append(dict(**row, folder=folder.name))
            print('EVALUATED', sid, label, flush=True)
    if seal['rows'] != candidates(root, len(d['tasks'])):
        raise ValueError('Candidate changed')
    c.write(out / 'summary.json', dict(completed=True, shard_index=shard_index, shard_count=shard_count,
                                       rows=rows, candidate_seal_sha256=c.sha(root / 'candidates_sealed.json'),
                                       scheduler_calls=arrays, lpips_status='unknown_no_admitted_backbone'))
    print('COMPLETED_NATIVE_EVAL_SHARD', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])

