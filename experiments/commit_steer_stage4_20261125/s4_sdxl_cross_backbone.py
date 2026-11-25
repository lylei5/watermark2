"""Stage 4A: Cross-Backbone Validation on SDXL Base 1.0 (1024x1024, Dual Text Encoder, 4x128x128 latent).
Proves Commit-then-Steer scales zero-tuning to modern high-resolution architectures:
  - Native Gaussian Shading (256-bit payload: 192b auth + 64b HMAC-keyed SSCD commitment, 16x16=256x spatial rep)
  - Self-contained 10-step DDIM preview + Mid-step Tweedie guidance steps [3, 21] / 30 (leaving final 8 steps unguided)
  - Zero hyperparameter or threshold retuning (reuses frozen Stage 0 SSCD-64b 95%-TPR threshold & step_size=0.042)

Parts (all executed cleanly on a single idle RTX 3090 in ~50-60 min):
  1. sdxl24        - 24 disjoint unseen official unified prompts (12 COCO + 12 SDP, slices [64:76]):
                     Clean + ALL 9 standard GS distortions (240 1024x1024 DDIM inversions) + 552 cross-prompt FAR pairs.
  2. reprompt_sdxl - Live Full-Budget Reprompt+ Forgery on SDXL (12 guided sources x 3 I2P prompts = 36 candidates):
                     Native GS-SDXL fooled vs Commit-then-Steer blind rejection.
  3. null200_sdxl  - Empirical Natural-Image Null FPR on 200 real COCO validation images at 1024x1024.
  4. quality_viz   - Exact paired LPIPS, CLIP text-image score (unguided vs guided), and visual showcase montages.
"""
import argparse
import hmac
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import beta, norm

ROOT = Path('/data0/liyanlei/watermark_research')
RUN = ROOT / 'runs/commit_steer_stage4_20261125/sdxl_cross_backbone'
STAGE0 = ROOT / 'runs/payload_binding_stage0_20261122'
UNIFIED_SEL = ROOT / 'runs/unified_main_sources_20261116/selection.json'
NULL_INPUTS = ROOT / 'runs/unified_null_features_20261116_v2/inputs.json'
GS_DIST_PATH = ROOT / 'runs/defense_revalidation_20261111/gs_image_utils.py'
EXTRA_DEPS = ROOT / 'envs/phase1_extra'
SSCD_PATH = ROOT / 'models/sscd/sscd_disc_mixup.torchscript.pt'

GS_9_CONDITIONS = {
    'random_crop_ratio': 0.6,
    'random_drop_ratio': 0.8,
    'brightness_factor': 6,
    'jpeg_ratio': 25,
    'gaussian_blur_r': 4,
    'gaussian_std': 0.05,
    'median_blur_k': 7,
    'sp_prob': 0.05,
    'resize_ratio': 0.25,
}


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
    assert idx in (0, 1, 2, 3, 4, 5, 7) and idx != 6, f'Forbidden GPU index {idx}'
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


def make_keyed_projection(key_bytes: bytes, k: int = 64) -> np.ndarray:
    seed = int.from_bytes(hmac.digest(key_bytes, b'commit-steer-s4-sdxl-proj-v1', 'sha256')[:16], 'big')
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((512, k)))
    return q.astype(np.float64)


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    return 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))


def sdxl_gs_encode(w_flat_256: np.ndarray, gs_key_128: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Native Gaussian Shading for SDXL (1, 4, 128, 128): 256-bit message (1,4,8,8) tiled 16x16 (256x rep)."""
    assert w_flat_256.shape == (256,) and gs_key_128.shape == (1, 4, 128, 128)
    w_arr = w_flat_256.reshape(1, 4, 8, 8)
    m_enc = ((np.tile(w_arr, (1, 1, 16, 16)) + gs_key_128) % 2).astype(np.float64)
    u = rng.uniform(1e-6, 1.0 - 1e-6, size=m_enc.shape)
    return norm.ppf((u + m_enc) / 2.0).astype(np.float32)


def sdxl_gs_decode(z_inv_np: np.ndarray, gs_key_128: np.ndarray) -> np.ndarray:
    """Exact majority voting across 16x16 = 256 spatial copies -> 256 recovered bits."""
    assert z_inv_np.shape == (1, 4, 128, 128) and gs_key_128.shape == (1, 4, 128, 128)
    rev_sd = ((z_inv_np > 0).astype(np.int32) + gs_key_128.astype(np.int32)) % 2
    votes = rev_sd.reshape(1, 4, 16, 8, 16, 8).sum(axis=(2, 4))
    return (votes > 128).astype(np.uint8).reshape(-1)


def wiggle_gs_latent_np(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Exact CVPR 2025 gs_provider.wiggle_latents (l=1 binary Gaussian Shading)."""
    y = np.clip((norm.cdf(arr) * 2.0).astype(np.int32), 0, 1)
    u = rng.uniform(1e-6, 1.0 - 1e-6, size=y.shape).astype(np.float64)
    return norm.ppf((u + y) / 2.0).astype(np.float32)


def do_prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / 'protocol.json').exists() and (RUN / 'private_material.npz').exists():
        print('sdxl_cross_backbone already prepared; keeping frozen protocol & private material.', flush=True)
        return

    th = read_json(STAGE0 / 'summary_sscd.json')['thresholds_95']['sscd_64b']
    sel = read_json(UNIFIED_SEL)['sources']
    # Strictly disjoint slice [64:76] (gs100=[0:50], prc16=[50:58], multidim=[60:64])
    coco12 = [s for s in sel if s['dataset'] == 'COCO'][64:76]
    sdp12 = [s for s in sel if s['dataset'] == 'SDP'][64:76]
    sources = coco12 + sdp12
    assert len(sources) == 24

    rng_priv = np.random.default_rng(2026112501)
    gs_key_np = rng_priv.integers(0, 2, size=(1, 4, 128, 128), dtype=np.uint8)
    auth_192 = rng_priv.integers(0, 2, size=(192,), dtype=np.uint8)
    P64_np = make_keyed_projection(np.packbits(gs_key_np.reshape(-1)).tobytes(), 64)

    # Unit self-check: exact SDXL GS encode/decode roundtrip (0 bit errors across 8 random messages)
    for k in range(8):
        test_msg = rng_priv.integers(0, 2, size=(256,), dtype=np.uint8)
        z_test = sdxl_gs_encode(test_msg, gs_key_np, rng_priv)
        rec = sdxl_gs_decode(z_test, gs_key_np)
        assert np.array_equal(rec, test_msg), f'SDXL GS roundtrip failed at trial {k}'

    np.savez(RUN / 'private_material.npz', key=gs_key_np, auth_bits=auth_192, P64=P64_np)
    os.chmod(RUN / 'private_material.npz', 0o600)

    proto = dict(
        run_name='sdxl_cross_backbone',
        backbone='stabilityai/stable-diffusion-xl-base-1.0',
        resolution=1024,
        latent_shape=[1, 4, 128, 128],
        carrier='Native Gaussian Shading 256b (192b auth + 64b SSCD commitment, 16x16=256x spatial repetition)',
        n_prompts=24,
        prompt_slices='COCO[64:76] + SDP[64:76] (zero overlap with Stage 0-3)',
        num_steps=30,
        preview_steps=10,
        step_range=[3, 21],
        step_size=0.042,
        stop_loss=0.08,
        threshold=th,
        threshold_source='Stage 0 frozen 95%-TPR SSCD-64b threshold (zero retuning on SDXL)',
        auth_threshold=136.0 / 192.0,
        gs9_conditions=GS_9_CONDITIONS,
        sources=sources,
    )
    write_json(RUN / 'protocol.json', proto)
    print(f'Prepared sdxl_cross_backbone | 24 disjoint unified prompts | roundtrip verified | th={th:.6f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'all'])
    ap.add_argument('--gpu', type=int, default=1)
    args = ap.parse_args()

    do_prepare()
    if args.mode == 'prepare':
        return

    uuid = check_idle_gpu(args.gpu)
    proto = read_json(RUN / 'protocol.json')
    th = proto['threshold']
    priv = np.load(RUN / 'private_material.npz')
    gs_key_np, auth_192, P64_np = priv['key'], priv['auth_bits'], priv['P64']

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
    from diffusers import StableDiffusionXLPipeline, DDIMScheduler, DDIMInverseScheduler

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(EXTRA_DEPS))
    gs_dist = load_module('gs_dist', GS_DIST_PATH)

    t_global = time.perf_counter()
    print(f'[S4A SDXL | GPU {args.gpu}] Loading SDXL Base 1.0 (bf16)...', flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0',
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.requires_grad_(False); pipe.vae.eval()
    pipe.unet.requires_grad_(False); pipe.unet.eval()
    pipe.vae.enable_slicing()

    fwd_sched = DDIMScheduler.from_config(pipe.scheduler.config)
    inv_sched = DDIMInverseScheduler.from_config(pipe.scheduler.config)

    sscd = torch.jit.load(str(SSCD_PATH), map_location='cuda').eval()
    for p in sscd.parameters():
        p.requires_grad_(False)
    P64_t = torch.from_numpy(P64_np).float().cuda()
    mean_norm = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    std_norm = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)

    with torch.no_grad():
        _, npe_null, _, nppe_null = pipe.encode_prompt(
            prompt='', device='cuda', num_images_per_prompt=1,
            do_classifier_free_guidance=True, negative_prompt=''
        )
        add_time_ids_1 = pipe._get_add_time_ids(
            (1024, 1024), (0, 0), (1024, 1024),
            dtype=npe_null.dtype,
            text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
        ).to('cuda')

    def decode_x(z):
        dec = pipe.vae.decode(z.to(torch.bfloat16) / pipe.vae.config.scaling_factor, return_dict=False)[0]
        return (dec.float() / 2.0 + 0.5).clamp(0.0, 1.0)

    def to_pil(x):
        return Image.fromarray(x.detach()[0].permute(1, 2, 0).mul(255.0).round().byte().cpu().numpy())

    def eval_sscd_batch(pil_list):
        embs = []
        for im in pil_list:
            x = torch.from_numpy(np.asarray(im.convert('RGB').resize((320, 320), Image.BICUBIC), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()
            with torch.no_grad():
                embs.append(F.normalize(sscd((x - mean_norm) / std_norm), dim=1)[0].cpu().numpy().astype(np.float64))
        return np.stack(embs, axis=0)

    def diff_sscd_loss(x, ts_t):
        x320 = F.interpolate(x, size=(320, 320), mode='bilinear', align_corners=False)
        proj = F.normalize(sscd((x320 - mean_norm) / std_norm), dim=1)[0] @ P64_t
        cos_d = 1.0 - F.cosine_similarity(proj.unsqueeze(0), ts_t.unsqueeze(0)).squeeze()
        margin = F.softplus(0.05 - proj * ts_t, beta=20.0).mean()
        return cos_d + 0.5 * margin

    def sdxl_gen(prompt, zT_np, num_steps=30, ts_t=None, step_range=(3, 21), step_size=0.042, stop_loss=0.08):
        fwd_sched.set_timesteps(num_steps, device='cuda')
        with torch.no_grad():
            pe, npe, ppe, nppe = pipe.encode_prompt(
                prompt=prompt, device='cuda', num_images_per_prompt=1,
                do_classifier_free_guidance=True, negative_prompt=''
            )
            te_cat = torch.cat([npe, pe], dim=0)
            added_cond = {
                'text_embeds': torch.cat([nppe, ppe], dim=0),
                'time_ids': torch.cat([add_time_ids_1, add_time_ids_1], dim=0),
            }
        z = torch.from_numpy(zT_np).to(device='cuda', dtype=torch.bfloat16)
        alphas = fwd_sched.alphas_cumprod.to('cuda')
        n_grad_steps = 0
        for idx_t, t in enumerate(fwd_sched.timesteps):
            with torch.no_grad():
                eps = pipe.unet(
                    fwd_sched.scale_model_input(torch.cat([z, z], dim=0), t), t,
                    encoder_hidden_states=te_cat, added_cond_kwargs=added_cond, return_dict=False,
                )[0]
                eu, ec = eps.chunk(2)
                out = fwd_sched.step(eu + 7.5 * (ec - eu), t, z, eta=0.0, return_dict=True)
                zn, px0 = out.prev_sample, out.pred_original_sample
            if ts_t is not None and step_range[0] <= idx_t <= step_range[1]:
                z0l = px0.detach().to(torch.bfloat16).requires_grad_(True)
                loss = diff_sscd_loss(decode_x(z0l), ts_t)
                if float(loss.detach()) > stop_loss:
                    g = torch.autograd.grad(loss, z0l)[0].float()
                    gn = float(g.norm()) + 1e-8
                    pt = int(t) - fwd_sched.config.num_train_timesteps // num_steps
                    ap = alphas[pt] if pt >= 0 else fwd_sched.final_alpha_cumprod.to('cuda')
                    zn = (zn.float() + torch.sqrt(ap.float()) * (-(step_size * float(z0l.detach().float().norm()) / gn) * g)).to(torch.bfloat16)
                    n_grad_steps += 1
            z = zn
        with torch.no_grad():
            im = to_pil(decode_x(z))
        return im, n_grad_steps

    def sdxl_invert_batch(pil_list, num_steps=30, bs=2):
        inv_sched.set_timesteps(num_steps, device='cuda')
        out_list = []
        with torch.no_grad():
            for i in range(0, len(pil_list), bs):
                chunk = pil_list[i:i + bs]
                b = len(chunk)
                xs = torch.stack([
                    torch.from_numpy(np.asarray(im.convert('RGB').resize((1024, 1024), Image.BICUBIC), dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)
                    for im in chunk
                ]).to(device='cuda', dtype=torch.bfloat16)
                z = pipe.vae.encode(xs).latent_dist.mean * pipe.vae.config.scaling_factor
                npe_b = npe_null.expand(b, -1, -1)
                cond_b = {'text_embeds': nppe_null.expand(b, -1), 'time_ids': add_time_ids_1.expand(b, -1)}
                for t in inv_sched.timesteps:
                    eps = pipe.unet(inv_sched.scale_model_input(z, t), t, encoder_hidden_states=npe_b, added_cond_kwargs=cond_b, return_dict=False)[0]
                    z = inv_sched.step(eps, t, z, return_dict=False)[0]
                out_list.append(z.float().cpu().numpy())
        return np.concatenate(out_list, axis=0)

    def blind_verify_from_inv(z_inv_1, q_emb, true_w256=None):
        rev_w = sdxl_gs_decode(z_inv_1, gs_key_np)
        auth_ba = float((rev_w[:192] == auth_192).mean())
        auth_ok = auth_ba > (136.0 / 192.0)
        rec_sgn = 2.0 * rev_w[192:].astype(np.float64) - 1.0
        qp = q_emb @ P64_np
        d_blind = float(1.0 - np.dot(rec_sgn, qp) / (8.0 * float(np.linalg.norm(qp)) + 1e-12))
        feat_ok = d_blind <= th
        res = dict(auth_ba=auth_ba, auth_ok=bool(auth_ok), d_blind=d_blind, feat_ok=bool(feat_ok), joint_blind_ok=bool(auth_ok and feat_ok))
        if true_w256 is not None:
            res['full_ba'] = float((rev_w == true_w256).mean())
            res['payload_ber'] = float((rev_w[192:] != true_w256[192:]).mean())
            true_sgn = 2.0 * true_w256[192:].astype(np.float64) - 1.0
            res['d_oracle'] = float(1.0 - np.dot(true_sgn, qp) / (8.0 * float(np.linalg.norm(qp)) + 1e-12))
        return res

    # =========================================================================
    # PART 1: 24 DISJOINT UNIFIED PROMPTS (Clean + ALL 9 Standard GS Distortions)
    # =========================================================================
    print('=== [S4A Part 1] SDXL 24 Disjoint Unified Prompts (Clean + 9 GS Distortions = 240 inversions) ===', flush=True)
    base_dir = RUN / 'images_unguided'
    guid_dir = RUN / 'images_guided'
    base_dir.mkdir(parents=True, exist_ok=True)
    guid_dir.mkdir(parents=True, exist_ok=True)

    conds = ['clean', *GS_9_CONDITIONS.keys()]
    records, embs_out = [], []
    rng_gen = np.random.default_rng(2026112588)
    t_p1 = time.perf_counter()

    for idx, s in enumerate(proto['sources']):
        prompt = s['prompt']
        # 1. Self-contained 10-step SDXL preview
        t0 = time.perf_counter()
        zT_prev = np.random.default_rng(2026112500 + idx).standard_normal((1, 4, 128, 128)).astype(np.float32)
        prev_im, _ = sdxl_gen(prompt, zT_prev, num_steps=10, ts_t=None)
        torch.cuda.synchronize()
        sec_prev = time.perf_counter() - t0

        prev_emb = eval_sscd_batch([prev_im])[0]
        p_bits = ((prev_emb @ P64_np) >= 0.0).astype(np.uint8)
        true_sgn = 2.0 * p_bits.astype(np.float64) - 1.0
        ts_t = torch.tensor(true_sgn, device='cuda', dtype=torch.float32)

        w_flat = np.concatenate([auth_192, p_bits])
        zT = sdxl_gs_encode(w_flat, gs_key_np, rng_gen)

        t0 = time.perf_counter()
        base_im, _ = sdxl_gen(prompt, zT, num_steps=30, ts_t=None)
        torch.cuda.synchronize()
        sec_base = time.perf_counter() - t0
        base_im.save(base_dir / f'{idx:03d}.png')

        t0 = time.perf_counter()
        guid_im, n_gsteps = sdxl_gen(prompt, zT, num_steps=30, ts_t=ts_t, step_range=(3, 21), step_size=0.042, stop_loss=0.08)
        torch.cuda.synchronize()
        sec_guid = time.perf_counter() - t0
        guid_im.save(guid_dir / f'{idx:03d}.png')

        q_imgs = []
        for ci, cond in enumerate(conds):
            if cond == 'clean':
                q_imgs.append(guid_im)
            else:
                seed = 2026112500 + idx * 100 + ci
                gs_dist.set_random_seed(seed)
                dargs = SimpleNamespace(**{k: None for k in GS_9_CONDITIONS})
                setattr(dargs, cond, GS_9_CONDITIONS[cond])
                q_imgs.append(gs_dist.image_distortion(guid_im.copy(), seed, dargs).convert('RGB'))

        t0 = time.perf_counter()
        z_invs = sdxl_invert_batch(q_imgs, num_steps=30, bs=2)
        torch.cuda.synchronize()
        sec_inv_per_img = (time.perf_counter() - t0) / len(q_imgs)
        q_embs = eval_sscd_batch(q_imgs)
        embs_out.append(dict(idx=idx, target_bits=p_bits.tolist(), clean_emb=q_embs[0].tolist(), z_inv_clean_path=None))

        # Save clean inverted latent for first 12 sources to reuse in Live Reprompt+
        if idx < 12:
            np.save(RUN / f'z_inv_clean_{idx:02d}.npy', z_invs[0:1])

        for ci, cond in enumerate(conds):
            v = blind_verify_from_inv(z_invs[ci:ci+1], q_embs[ci], true_w256=w_flat)
            records.append(dict(
                idx=idx, dataset=s['dataset'], condition=cond,
                sec_prev=sec_prev, sec_base=sec_base, sec_guid=sec_guid, sec_total=sec_prev + sec_guid,
                sec_inv=sec_inv_per_img, n_grad_steps=n_gsteps, **v,
            ))

        cl_r = records[-len(conds)]
        ed_pass = sum(r['joint_blind_ok'] for r in records[-len(conds)+1:])
        print(f'  [SDXL24] {idx+1:2d}/24 ({s["dataset"]}) | prev={sec_prev:.1f}s guid={sec_guid:.1f}s (gsteps={n_gsteps}) | clean_BA={cl_r["full_ba"]:.4f} pBER={cl_r["payload_ber"]:.4f} d={cl_r["d_blind"]:.3f} ok={cl_r["joint_blind_ok"]} | 9-GS={ed_pass}/9', flush=True)
        write_json(RUN / 'sdxl24_records_partial.json', records)

    # Cross-prompt FAR (24 x 23 = 552 pairs)
    sgns = np.array([2.0 * np.array(e['target_bits'], dtype=np.float64) - 1.0 for e in embs_out])
    qps = np.array([np.array(e['clean_emb'], dtype=np.float64) @ P64_np for e in embs_out])
    qps_n = qps / (np.linalg.norm(qps, axis=1, keepdims=True) + 1e-12)
    dist_mat = 1.0 - (sgns @ qps_n.T) / 8.0
    off_diag = dist_mat[~np.eye(24, dtype=bool)]
    cross_far_cnt = int((off_diag <= th).sum())

    # =========================================================================
    # PART 2: LIVE FULL-BUDGET REPROMPT+ FORGERY ON SDXL (12 sources x 3 I2P = 36 cands)
    # =========================================================================
    print('=== [S4A Part 2] Live Reprompt+ Forgery on SDXL (12 sources x 3 I2P = 36 candidates) ===', flush=True)
    rp_proto = read_json(ROOT / 'runs/author_reprompt_plus_20261115/protocol.json')
    gs_refs = [r for r in rp_proto['references'] if r['carrier'] == 'GS']
    all_i2p = []
    for ref in gs_refs:
        for tp in ref['target_prompts']:
            all_i2p.append(tp['prompt'])
    assert len(all_i2p) >= 30

    rp_dir = RUN / 'reprompt_sdxl'
    rp_dir.mkdir(exist_ok=True)
    rng_wig = np.random.default_rng(2026112577)
    rp_rows = []
    for i in range(12):
        z_inv_clean = np.load(RUN / f'z_inv_clean_{i:02d}.npy')
        prompts3 = [all_i2p[(i * 3 + k) % len(all_i2p)] for k in range(3)]
        cands = []
        for pi, ptxt in enumerate(prompts3):
            z_wig = wiggle_gs_latent_np(z_inv_clean, rng_wig)
            cand_im, _ = sdxl_gen(ptxt, z_wig, num_steps=30, ts_t=None)
            cands.append(cand_im)
            if pi == 0 and i < 6:
                cand_im.save(rp_dir / f'src_{i:02d}_forged_p0.png')
        z_cands = sdxl_invert_batch(cands, num_steps=30, bs=3)
        e_cands = eval_sscd_batch(cands)
        for pi in range(3):
            v = blind_verify_from_inv(z_cands[pi:pi+1], e_cands[pi])
            rp_rows.append(dict(idx=i, prompt_idx=pi, target_prompt=prompts3[pi], **v))
        n_gs = sum(r['auth_ok'] for r in rp_rows if r['idx'] == i)
        n_ours = sum(r['joint_blind_ok'] for r in rp_rows if r['idx'] == i)
        print(f'  [SDXL Reprompt+] src {i:02d}: Native_GS_SDXL={n_gs}/3 | CommitSteer_SDXL={n_ours}/3', flush=True)

    # =========================================================================
    # PART 3: EMPIRICAL NATURAL-IMAGE NULL FPR ON SDXL (200 real COCO val images @ 1024x1024)
    # =========================================================================
    print('=== [S4A Part 3] SDXL Empirical Natural-Image Null FPR (200 real COCO validation images) ===', flush=True)
    all_inputs = read_json(NULL_INPUTS)
    val_200 = [x for x in all_inputs if x['split'] == 'validation'][:200]
    assert len(val_200) == 200

    def center_crop_1024(p: str) -> Image.Image:
        im = Image.open(p).convert('RGB')
        w, h = im.size
        s = min(w, h)
        return im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((1024, 1024), Image.BICUBIC)

    t_null = time.perf_counter()
    null_auth, null_d = [], []
    for start in range(0, 200, 4):
        imgs = [center_crop_1024(x['path']) for x in val_200[start:start + 4]]
        z_ns = sdxl_invert_batch(imgs, num_steps=30, bs=2)
        e_ns = eval_sscd_batch(imgs)
        for bi in range(len(imgs)):
            v = blind_verify_from_inv(z_ns[bi:bi+1], e_ns[bi])
            null_auth.append(v['auth_ba'])
            null_d.append(v['d_blind'])
        if (start + 4) % 40 == 0:
            print(f'  [SDXL Null200] {start+4}/200 ({time.perf_counter()-t_null:.1f}s) max_auth={max(null_auth):.4f} min_d={min(null_d):.4f}', flush=True)

    ab_n, db_n = np.array(null_auth), np.array(null_d)
    np.savez(RUN / 'null200_scores.npz', auth_ba=ab_n, d_blind=db_n)
    null_fp_auth = int((ab_n > 136.0 / 192.0).sum())
    null_fp_feat = int((db_n <= th).sum())
    null_fp_joint = int(((ab_n > 136.0 / 192.0) & (db_n <= th)).sum())

    # =========================================================================
    # PART 4: QUALITY METRICS (LPIPS + CLIP) & HIGH-RES VISUAL SHOWCASE
    # =========================================================================
    print('=== [S4A Part 4] Computing exact LPIPS, CLIP & saving visual showcases ===', flush=True)
    del pipe, sscd
    torch.cuda.empty_cache()

    import lpips
    from transformers import CLIPModel, CLIPProcessor
    lpips_fn = lpips.LPIPS(net='alex').cuda().eval()
    clip_asset = read_json(ROOT / 'runs/phase1_backend_audit_20261113_v2/complete.json')['model_files']['clip_vitl14']
    clip_model = CLIPModel.from_pretrained(clip_asset['root'], local_files_only=True).cuda().eval()
    clip_proc = CLIPProcessor.from_pretrained(clip_asset['root'], local_files_only=True)

    def to_m11_512(im):
        im5 = im.convert('RGB').resize((512, 512), Image.BICUBIC)
        return torch.from_numpy(np.asarray(im5, dtype=np.float32).transpose(2, 0, 1)).unsqueeze(0).cuda() / 127.5 - 1.0

    lpips_map, clip_b_map, clip_g_map = {}, {}, {}
    for idx, s in enumerate(proto['sources']):
        im_b = Image.open(base_dir / f'{idx:03d}.png').convert('RGB')
        im_g = Image.open(guid_dir / f'{idx:03d}.png').convert('RGB')
        with torch.no_grad():
            lp = float(lpips_fn(to_m11_512(im_b), to_m11_512(im_g)).item())
            inp = clip_proc(text=[s['prompt']], images=[im_b, im_g], return_tensors='pt', padding=True, truncation=True, max_length=77).to('cuda')
            out = clip_model(**inp)
            ie = F.normalize(out.image_embeds, dim=-1)
            te = F.normalize(out.text_embeds, dim=-1)
            cb, cg = float((ie[0] * te[0]).sum().item()), float((ie[1] * te[0]).sum().item())
        lpips_map[idx], clip_b_map[idx], clip_g_map[idx] = lp, cb, cg

    for r in records:
        r['lpips'] = lpips_map[r['idx']]
        r['clip_base'] = clip_b_map[r['idx']]
        r['clip_guid'] = clip_g_map[r['idx']]

    # Save visual showcase montage (6 samples: 3 COCO [0,1,2] + 3 SDP [12,13,14], Unguided vs Guided)
    picks = [0, 1, 2, 12, 13, 14]
    w, h, pad = 320, 320, 32
    canvas = Image.new('RGB', (w * len(picks), pad + h * 2), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    cl_by_idx = {r['idx']: r for r in records if r['condition'] == 'clean'}
    for ci, pid in enumerate(picks):
        canvas.paste(Image.open(base_dir / f'{pid:03d}.png').convert('RGB').resize((w, h), Image.BICUBIC), (ci * w, pad))
        canvas.paste(Image.open(guid_dir / f'{pid:03d}.png').convert('RGB').resize((w, h), Image.BICUBIC), (ci * w, pad + h))
        r = cl_by_idx[pid]
        draw.text((ci * w + 8, 8), f"SDXL #{pid:02d} {r['dataset']} LP={r['lpips']:.3f} d={r['d_blind']:.2f}", fill=(20, 20, 20))
    canvas.save(RUN / 'sdxl_visual_showcase.jpg', quality=92)

    # Save SDXL Reprompt montage (6 sources x 2 rows: Guided WM Source vs Forged Reprompt)
    rcanvas = Image.new('RGB', (w * 6, pad + h * 2), (250, 250, 250))
    rdraw = ImageDraw.Draw(rcanvas)
    for ci in range(6):
        rcanvas.paste(Image.open(guid_dir / f'{ci:03d}.png').convert('RGB').resize((w, h), Image.BICUBIC), (ci * w, pad))
        rcanvas.paste(Image.open(rp_dir / f'src_{ci:02d}_forged_p0.png').convert('RGB').resize((w, h), Image.BICUBIC), (ci * w, pad + h))
        rr = [x for x in rp_rows if x['idx'] == ci and x['prompt_idx'] == 0][0]
        rdraw.text((ci * w + 8, 8), f"#{ci:02d} GS={rr['auth_ba']:.2f} d={rr['d_blind']:.2f}({'ACC' if rr['joint_blind_ok'] else 'REJ'})", fill=(20, 20, 20))
    rcanvas.save(RUN / 'sdxl_reprompt_montage.jpg', quality=92)

    cl = [r for r in records if r['condition'] == 'clean']
    ed = [r for r in records if r['condition'] != 'clean']
    non_occl = [r for r in ed if r['condition'] not in ('random_crop_ratio', 'random_drop_ratio')]
    by_cond = {}
    for cond in conds:
        sub = [r for r in records if r['condition'] == cond]
        by_cond[cond] = dict(
            n=len(sub),
            auth_pass=sum(r['auth_ok'] for r in sub),
            feat_pass=sum(r['feat_ok'] for r in sub),
            joint_pass=sum(r['joint_blind_ok'] for r in sub),
            full_ba=float(np.mean([r['full_ba'] for r in sub])),
            p_ber=float(np.mean([r['payload_ber'] for r in sub])),
            d_blind=float(np.mean([r['d_blind'] for r in sub])),
        )

    summary = dict(
        backbone='stabilityai/stable-diffusion-xl-base-1.0',
        resolution=1024,
        gpu=args.gpu, uuid=uuid, elapsed_seconds=time.perf_counter() - t_global,
        n_prompts=24, threshold=th,
        sec_prev_mean=float(np.mean([r['sec_prev'] for r in cl])),
        sec_base_mean=float(np.mean([r['sec_base'] for r in cl])),
        sec_guid_mean=float(np.mean([r['sec_guid'] for r in cl])),
        sec_total_mean=float(np.mean([r['sec_total'] for r in cl])),
        sec_inv_mean=float(np.mean([r['sec_inv'] for r in cl])),
        n_grad_steps_mean=float(np.mean([r['n_grad_steps'] for r in cl])),
        lpips_mean=float(np.mean([r['lpips'] for r in cl])),
        lpips_median=float(np.median([r['lpips'] for r in cl])),
        clip_base_mean=float(np.mean([r['clip_base'] for r in cl])),
        clip_guid_mean=float(np.mean([r['clip_guid'] for r in cl])),
        clean_auth_pass=sum(r['auth_ok'] for r in cl),
        clean_joint_pass=sum(r['joint_blind_ok'] for r in cl),
        clean_full_ba=float(np.mean([r['full_ba'] for r in cl])),
        clean_p_ber=float(np.mean([r['payload_ber'] for r in cl])),
        gs7_regular_joint_pass=sum(r['joint_blind_ok'] for r in non_occl),
        gs9_auth_pass=sum(r['auth_ok'] for r in ed),
        gs9_joint_pass=sum(r['joint_blind_ok'] for r in ed),
        gs9_p_ber=float(np.mean([r['payload_ber'] for r in ed])),
        cross_prompt_far_count=cross_far_cnt,
        cross_prompt_far_total=552,
        cross_prompt_far=cross_far_cnt / 552.0,
        by_cond=by_cond,
        live_reprompt_sdxl=dict(
            n_sources=12, n_candidates=len(rp_rows),
            native_gs_cand=sum(r['auth_ok'] for r in rp_rows),
            native_gs_src=sum(any(r['auth_ok'] for r in rp_rows if r['idx'] == i) for i in range(12)),
            commit_steer_cand=sum(r['joint_blind_ok'] for r in rp_rows),
            commit_steer_src=sum(any(r['joint_blind_ok'] for r in rp_rows if r['idx'] == i) for i in range(12)),
            mean_d_blind=float(np.mean([r['d_blind'] for r in rp_rows])),
        ),
        null_fpr_200=dict(
            n=200,
            auth_ba_mean=float(ab_n.mean()), auth_ba_max=float(ab_n.max()),
            d_blind_mean=float(db_n.mean()), d_blind_min=float(db_n.min()),
            auth_fp_default=null_fp_auth,
            feat_fp_only=null_fp_feat,
            joint_fp_default=null_fp_joint,
            joint_cp95_upper=clopper_pearson_upper(null_fp_joint, 200),
        ),
    )
    write_json(RUN / 'summary.json', summary)
    write_json(RUN / 'records.json', records)
    write_json(RUN / 'reprompt_rows.json', rp_rows)
    print(f"[S4A SDXL COMPLETE] {summary['elapsed_seconds']:.1f}s | clean={summary['clean_joint_pass']}/24 | gs9={summary['gs9_joint_pass']}/216 ({100*summary['gs9_joint_pass']/216:.1f}%) | gs7={summary['gs7_regular_joint_pass']}/168 ({100*summary['gs7_regular_joint_pass']/168:.1f}%) | LPIPS={summary['lpips_mean']:.3f} | CLIP={summary['clip_guid_mean']:.4f} vs {summary['clip_base_mean']:.4f} | LiveRP GS={summary['live_reprompt_sdxl']['native_gs_cand']}/36 Ours={summary['live_reprompt_sdxl']['commit_steer_cand']}/36 | Null200={null_fp_joint}/200", flush=True)


if __name__ == '__main__':
    main()
