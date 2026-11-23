"""Stage 3B (GPU 2): Empirical Natural-Image Null FPR Calibration on 2,000 Real COCO Images
(1,000 calibration + 1,000 independent validation from unified_null_features_20261116_v2/inputs.json).
Uses batched 50-step DDIM inversion (batch_size=8) to measure exact empirical false-positive rates
for Native GS (256-bit), Auth-192b, and Commit-then-Steer Blind Joint Verification.
"""
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import beta

ROOT = Path('/data0/liyanlei/watermark_research')
RUN = ROOT / 'runs/commit_steer_paper_scale_20261123/null_fpr_2000'
GS100 = ROOT / 'runs/commit_steer_paper_scale_20261123/gs100_unified'
NULL_INPUTS = ROOT / 'runs/unified_null_features_20261116_v2/inputs.json'
GS_MOD_PATH = ROOT / 'runs/phase1_trajectory_native_gs_20261113/gs_watermark.py'
PGID_PIPE = ROOT / 'third_party/frozen/PGID_dc4ad96'
EXTRA_DEPS = ROOT / 'envs/phase1_extra'
SSCD_PATH = ROOT / 'models/sscd/sscd_disc_mixup.torchscript.pt'


def read_json(p: Path):
    return json.loads(p.read_bytes())


def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(p)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_idle_gpu(idx: int) -> str:
    assert idx in (0, 1, 2, 3, 4, 5, 7) and idx != 6
    info = subprocess.check_output(
        ['nvidia-smi', '-i', str(idx), '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
        text=True,
    ).strip().split(', ')
    busy = subprocess.check_output(
        ['nvidia-smi', '-i', str(idx), '--query-compute-apps=pid', '--format=csv,noheader'],
        text=True,
    ).strip()
    assert int(info[1]) < 100 and int(info[2]) == 0 and not busy, f'GPU {idx} busy: {info}, {busy}'
    return info[0]


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    return 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))


def main():
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    uuid = check_idle_gpu(gpu)
    RUN.mkdir(parents=True, exist_ok=True)

    os.environ.update(
        CUDA_VISIBLE_DEVICES=uuid,
        HF_HOME=str(ROOT / 'cache/huggingface'),
        HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4',
        TOKENIZERS_PARALLELISM='false',
    )

    import torch
    import torch.nn.functional as F

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(PGID_PIPE))
    sys.path.insert(0, str(EXTRA_DEPS))
    from pipe import pipe_utils

    # Share exact same provider key, auth_192, P64(K), and threshold as GS100!
    proto = read_json(GS100 / 'protocol.json')
    th = proto['threshold']
    priv = np.load(GS100 / 'private_material.npz')
    gs_key_np, auth_192, P64_np = priv['key'], priv['auth_bits'], priv['P64']

    native_gs = load_module('native_gs', GS_MOD_PATH).Gaussian_Shading(1, 8, 1e-6, 1)
    native_gs.key = torch.from_numpy(gs_key_np.astype(np.int64)).cuda()

    provider = pipe_utils.create_diffusion_pipe(
        pretrained_model_name_or_path='Manojb/stable-diffusion-2-1-base',
        resolution=512, device='cuda:0', scheduler_name='DDIM', scheduler_inverse_name='DDIM',
        dtype=torch.float16, eager_loading=True, disable_tqdm=True,
    )
    provider.pipe.set_progress_bar_config(disable=True)
    sscd = torch.jit.load(str(SSCD_PATH), map_location='cuda').eval()
    mean_norm = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    std_norm = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)

    all_inputs = read_json(NULL_INPUTS)
    cal_1000 = [x for x in all_inputs if x['split'] == 'calibration'][:1000]
    val_1000 = [x for x in all_inputs if x['split'] == 'validation'][:1000]
    assert len(cal_1000) == 1000 and len(val_1000) == 1000

    def center_crop_512(p: str) -> Image.Image:
        im = Image.open(p).convert('RGB')
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        return im.resize((512, 512), Image.BICUBIC)

    t0 = time.perf_counter()
    splits_out = {}
    for split_name, items in [('calibration', cal_1000), ('validation', val_1000)]:
        auth_bas, d_blinds, full_random_bas = [], [], []
        bs = 8
        for start in range(0, len(items), bs):
            batch_items = items[start:start + bs]
            imgs = [center_crop_512(x['path']) for x in batch_items]
            with torch.no_grad():
                zTs = provider.invert_images(imgs, num_inference_steps=50)['zT_torch']
                xs = torch.stack([
                    torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
                    for im in imgs
                ]).cuda()
                x_in = (F.interpolate(xs, size=(320, 320), mode='bilinear', align_corners=False) - mean_norm) / std_norm
                embs = F.normalize(sscd(x_in), dim=1).cpu().numpy().astype(np.float64)

            for bi in range(len(imgs)):
                rev_w = native_gs.diffusion_inverse(((zTs[bi:bi+1] > 0).int() + native_gs.key) % 2).reshape(-1).cpu().numpy().astype(np.uint8)
                auth_ba = float((rev_w[:192] == auth_192).mean())
                rec_sgn = 2.0 * rev_w[192:].astype(np.float64) - 1.0
                qp = embs[bi] @ P64_np
                d_b = float(1.0 - np.dot(rec_sgn, qp) / (8.0 * float(np.linalg.norm(qp)) + 1e-12))
                auth_bas.append(auth_ba)
                d_blinds.append(d_b)

            if (start + bs) % 200 == 0:
                print(f'[NullFPR {split_name} | GPU {gpu}] {start+bs}/1000 ({time.perf_counter()-t0:.1f}s) max_auth={max(auth_bas):.4f} min_d={min(d_blinds):.4f}', flush=True)

        ab = np.array(auth_bas)
        db = np.array(d_blinds)
        splits_out[split_name] = dict(
            n=len(ab),
            auth_ba_mean=float(ab.mean()),
            auth_ba_std=float(ab.std()),
            auth_ba_max=float(ab.max()),
            auth_ba_q99=float(np.quantile(ab, 0.99)),
            auth_ba_q999=float(np.quantile(ab, 0.999)),
            d_blind_mean=float(db.mean()),
            d_blind_std=float(db.std()),
            d_blind_min=float(db.min()),
            d_blind_q01=float(np.quantile(db, 0.01)),
            auth_fp_default=int((ab > 136.0 / 192.0).sum()),
            feat_fp_only=int((db <= th).sum()),
            joint_fp_default=int(((ab > 136.0 / 192.0) & (db <= th)).sum()),
        )
        np.savez(RUN / f'{split_name}_scores.npz', auth_ba=ab, d_blind=db)

    # Also compute exact calibrated operating points from calibration split applied to independent validation split!
    cal_ab = np.load(RUN / 'calibration_scores.npz')['auth_ba']
    cal_db = np.load(RUN / 'calibration_scores.npz')['d_blind']
    val_ab = np.load(RUN / 'validation_scores.npz')['auth_ba']
    val_db = np.load(RUN / 'validation_scores.npz')['d_blind']

    th_1pct_auth = float(np.quantile(cal_ab, 0.99, method='higher'))
    th_01pct_auth = float(np.quantile(cal_ab, 0.999, method='higher'))
    val_fp_1pct_auth = int((val_ab > th_1pct_auth).sum())
    val_fp_1pct_joint = int(((val_ab > th_1pct_auth) & (val_db <= th)).sum())
    val_fp_01pct_auth = int((val_ab > th_01pct_auth).sum())
    val_fp_01pct_joint = int(((val_ab > th_01pct_auth) & (val_db <= th)).sum())

    summary = dict(
        elapsed_seconds=time.perf_counter() - t0,
        gpu=gpu, uuid=uuid, sscd_threshold=th,
        splits=splits_out,
        empirical_calibration=dict(
            cal_auth_1pct_cutoff=th_1pct_auth,
            val_auth_only_fp_at_1pct=val_fp_1pct_auth,
            val_joint_fp_at_1pct=val_fp_1pct_joint,
            cal_auth_01pct_cutoff=th_01pct_auth,
            val_auth_only_fp_at_01pct=val_fp_01pct_auth,
            val_joint_fp_at_01pct=val_fp_01pct_joint,
            val_joint_default_fp=splits_out['validation']['joint_fp_default'],
            val_joint_default_cp95_upper=clopper_pearson_upper(splits_out['validation']['joint_fp_default'], 1000),
            pooled_2000_joint_default_fp=splits_out['calibration']['joint_fp_default'] + splits_out['validation']['joint_fp_default'],
            pooled_2000_cp95_upper=clopper_pearson_upper(splits_out['calibration']['joint_fp_default'] + splits_out['validation']['joint_fp_default'], 2000),
        ),
    )
    write_json(RUN / 'summary.json', summary)
    print(f"NULL FPR 2000 DONE in {summary['elapsed_seconds']:.1f}s | val default joint FP={splits_out['validation']['joint_fp_default']}/1000 (pooled {summary['empirical_calibration']['pooled_2000_joint_default_fp']}/2000, max_auth={splits_out['validation']['auth_ba_max']:.4f}, feat_only_fp={splits_out['validation']['feat_fp_only']}/1000) | at cal 1% auth cutoff ({th_1pct_auth:.4f}): val auth-only FP={val_fp_1pct_auth}/1000 -> joint FP={val_fp_1pct_joint}/1000", flush=True)


if __name__ == '__main__':
    main()
