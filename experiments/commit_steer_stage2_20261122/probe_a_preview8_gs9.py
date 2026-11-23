"""Probe A (GPU 2): Self-contained 8-step fast preview + HMAC-keyed P64(K) + Mid-step Tweedie
on 16 UNSEEN confirmation prompts (color_joint_20261119) evaluated under ALL 9 standard Gaussian Shading
benign distortions (directly comparable to gs_fingerprint_fixedpoint_20261116).
"""
import hashlib
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

ROOT = Path('/data0/liyanlei/watermark_research')
RUN = ROOT / 'runs/commit_steer_stage2_20261122/probe_a_preview8_gs9'
STAGE0 = ROOT / 'runs/payload_binding_stage0_20261122'
CJ_RUN = ROOT / 'runs/color_joint_20261119'
GS_MOD_PATH = ROOT / 'runs/phase1_trajectory_native_gs_20261113/gs_watermark.py'
GS_DIST_PATH = ROOT / 'runs/defense_revalidation_20261111/gs_image_utils.py'
PGID_PIPE = ROOT / 'third_party/frozen/PGID_dc4ad96'
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
    assert idx in (2, 3, 4, 5, 7) and idx != 6
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
    seed = int.from_bytes(hmac.digest(key_bytes, b'commit-steer-sscd64-proj-v1', 'sha256')[:16], 'big')
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((512, k)))
    return q.astype(np.float64)


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
    import lpips
    from transformers import CLIPModel, CLIPProcessor

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sys.path.insert(0, str(PGID_PIPE))
    sys.path.insert(0, str(EXTRA_DEPS))
    from pipe import pipe_utils

    # Frozen threshold strictly from Stage 0's 40-prompt calibration set (ZERO retuning on unseen prompts!)
    th = read_json(STAGE0 / 'summary_sscd.json')['thresholds_95']['sscd_64b']

    # Select 16 UNSEEN confirmation prompts from color_joint_20261119 (8 COCO + 8 SDP)
    cj_sources = read_json(CJ_RUN / 'sources.json')
    selected = [s for s in cj_sources if s['dataset'] == 'COCO'][:8] + [s for s in cj_sources if s['dataset'] == 'SDP'][:8]
    assert len(selected) == 16

    # Provider secret key -> GS OTP key, 192-bit auth tag, and private HMAC-keyed orthogonal P64(K)
    rng_priv = np.random.default_rng(2026112202)
    gs_key_np = rng_priv.integers(0, 2, size=(1, 4, 64, 64), dtype=np.uint8)
    auth_bits_np = rng_priv.integers(0, 2, size=(192,), dtype=np.uint8)
    key_bytes = np.packbits(gs_key_np.reshape(-1)).tobytes()
    P64_np = make_keyed_projection(key_bytes, 64)
    P64_t = torch.from_numpy(P64_np).float().cuda()
    np.savez(RUN / 'private_material.npz', key=gs_key_np, auth_bits=auth_bits_np, P64=P64_np)
    os.chmod(RUN / 'private_material.npz', 0o600)

    native_gs = load_module('native_gs', GS_MOD_PATH).Gaussian_Shading(1, 8, 1e-6, 1)
    native_gs.key = torch.from_numpy(gs_key_np.astype(np.int64)).cuda()
    gs_dist = load_module('gs_dist', GS_DIST_PATH)

    provider = pipe_utils.create_diffusion_pipe(
        pretrained_model_name_or_path='Manojb/stable-diffusion-2-1-base',
        resolution=512, device='cuda:0', scheduler_name='DDIM', scheduler_inverse_name='DDIM',
        dtype=torch.float16, eager_loading=True, disable_tqdm=True,
    )
    pipe = provider.pipe
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.requires_grad_(False); pipe.vae.eval()
    pipe.unet.requires_grad_(False); pipe.unet.eval()

    sscd = torch.jit.load(str(SSCD_PATH), map_location='cuda').eval()
    for p in sscd.parameters(): p.requires_grad_(False)
    lpips_fn = lpips.LPIPS(net='alex').cuda().eval()
    clip_asset = read_json(ROOT / 'runs/phase1_backend_audit_20261113_v2/complete.json')['model_files']['clip_vitl14']
    clip_model = CLIPModel.from_pretrained(clip_asset['root'], local_files_only=True).cuda().eval()
    clip_proc = CLIPProcessor.from_pretrained(clip_asset['root'], local_files_only=True)

    mean_norm = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    std_norm = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)

    def decode_x(z):
        dec = pipe.vae.decode((z.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor), return_dict=False)[0]
        return (dec.float() / 2.0 + 0.5).clamp(0.0, 1.0)

    def to_pil(x):
        return Image.fromarray(x.detach()[0].permute(1, 2, 0).mul(255.0).round().byte().cpu().numpy())

    def eval_sscd_pil(im):
        x = torch.from_numpy(np.asarray(im.convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()
        x_in = (F.interpolate(x, size=(320, 320), mode='bilinear', align_corners=False) - mean_norm) / std_norm
        with torch.no_grad():
            return F.normalize(sscd(x_in), dim=1)[0].cpu().numpy().astype(np.float64)

    def diff_sscd_loss(x, ts_t):
        x_in = (F.interpolate(x, size=(320, 320), mode='bilinear', align_corners=False) - mean_norm) / std_norm
        proj = F.normalize(sscd(x_in), dim=1)[0] @ P64_t
        cos_d = 1.0 - F.cosine_similarity(proj.unsqueeze(0), ts_t.unsqueeze(0)).squeeze()
        margin = F.softplus(0.05 - proj * ts_t, beta=20.0).mean()
        return cos_d + 0.5 * margin

    def ddim_gen(prompt, zT, num_steps=50, ts_t=None, step_range=(5, 35), step_size=0.035, stop_loss=0.10):
        pipe.scheduler = provider.scheduler
        pipe.scheduler.set_timesteps(num_steps, device='cuda')
        with torch.no_grad():
            pos, neg = pipe.encode_prompt(prompt=prompt, device='cuda', num_images_per_prompt=1,
                                          do_classifier_free_guidance=True, negative_prompt='')
            te = torch.cat([neg, pos], dim=0)
        z = zT.clone().to(device='cuda', dtype=torch.float16)
        alphas = pipe.scheduler.alphas_cumprod.to('cuda')
        for idx, t in enumerate(pipe.scheduler.timesteps):
            with torch.no_grad():
                eps = pipe.unet(pipe.scheduler.scale_model_input(torch.cat([z, z], dim=0), t), t,
                                encoder_hidden_states=te, return_dict=False)[0]
                eu, ec = eps.chunk(2)
                out = pipe.scheduler.step(eu + 7.5 * (ec - eu), t, z, eta=0.0, return_dict=True)
                zn, px0 = out.prev_sample, out.pred_original_sample
            if ts_t is not None and step_range[0] <= idx <= step_range[1]:
                z0l = px0.detach().float().requires_grad_(True)
                loss = diff_sscd_loss(decode_x(z0l), ts_t)
                if float(loss.detach()) > stop_loss:
                    g = torch.autograd.grad(loss, z0l)[0]
                    gn = float(g.norm()) + 1e-8
                    pt = t - pipe.scheduler.config.num_train_timesteps // num_steps
                    ap = alphas[pt] if pt >= 0 else pipe.scheduler.final_alpha_cumprod.to('cuda')
                    zn = (zn.float() + torch.sqrt(ap.float()) * (-(step_size * float(z0l.detach().norm()) / gn) * g)).to(torch.float16)
            z = zn
        with torch.no_grad():
            return to_pil(decode_x(z))

    def eval_lpips_clip(base_im, cand_im, prompt):
        def to_m11(im):
            return torch.from_numpy(np.asarray(im, dtype=np.float32).transpose(2, 0, 1)).unsqueeze(0).cuda() / 127.5 - 1.0
        with torch.no_grad():
            lp = float(lpips_fn(to_m11(base_im), to_m11(cand_im)).item())
            inp = clip_proc(text=[prompt], images=[cand_im], return_tensors='pt', padding=True, truncation=True, max_length=77).to('cuda')
            out = clip_model(**inp)
            cs = float((F.normalize(out.image_embeds, dim=-1) * F.normalize(out.text_embeds, dim=-1)).sum().item())
        return lp, cs

    def verify_blind(im, full_w_flat, true_sgn):
        z_inv = provider.invert_images(im, num_inference_steps=50)['zT_torch']
        rev_w = native_gs.diffusion_inverse(((z_inv > 0).int() + native_gs.key) % 2).reshape(-1).cpu().numpy().astype(np.uint8)
        full_ba = float((rev_w == full_w_flat).mean())
        auth_ba = float((rev_w[:192] == full_w_flat[:192]).mean())
        auth_ok = auth_ba > (136.0 / 192.0)
        p_ber = float((rev_w[192:] != full_w_flat[192:]).mean())
        rec_sgn = 2.0 * rev_w[192:].astype(np.float64) - 1.0
        qp = eval_sscd_pil(im) @ P64_np
        denom = 8.0 * float(np.linalg.norm(qp)) + 1e-12
        d_blind = float(1.0 - np.dot(rec_sgn, qp) / denom)
        d_oracle = float(1.0 - np.dot(true_sgn, qp) / denom)
        return dict(
            full_ba=full_ba, auth_ba=auth_ba, auth_ok=bool(auth_ok), payload_ber=p_ber,
            d_blind=d_blind, d_oracle=d_oracle, feat_ok=bool(d_blind <= th),
            joint_blind_ok=bool(auth_ok and d_blind <= th),
        )

    # Warmup
    with torch.no_grad():
        pipe('warmup', generator=torch.Generator(device='cuda').manual_seed(7), num_inference_steps=2)

    t_start = time.perf_counter()
    records = []
    img_root = RUN / 'images'
    img_root.mkdir(exist_ok=True)
    guided_clean_embs, target_sgns, sibling_embs = [], [], []

    for idx, s in enumerate(selected):
        task, prompt = s['task'], s['prompt']
        tdir = img_root / f'{task:03d}'
        tdir.mkdir(exist_ok=True)

        # Step 1: Self-contained 8-step fast DDIM preview from an independent random seed
        t0 = time.perf_counter()
        gen_prev = torch.Generator(device='cuda').manual_seed(2026112250 + task)
        zT_prev = torch.randn((1, 4, 64, 64), generator=gen_prev, device='cuda', dtype=torch.float16)
        prev_im = ddim_gen(prompt, zT_prev, num_steps=8, ts_t=None)
        torch.cuda.synchronize()
        sec_prev = time.perf_counter() - t0
        prev_im.save(tdir / 'preview8.png')

        # Step 2: Extract 64-bit keyed SSCD commitment from 8-step preview
        prev_emb = eval_sscd_pil(prev_im)
        p_bits = ((prev_emb @ P64_np) >= 0.0).astype(np.uint8)
        true_sgn = 2.0 * p_bits.astype(np.float64) - 1.0
        ts_t = torch.tensor(true_sgn, device='cuda', dtype=torch.float32)
        target_sgns.append(true_sgn)

        # Step 3: Pack (192b auth || 64b s*) into native 256-bit GS initial noise zT
        w_flat = np.concatenate([auth_bits_np, p_bits])
        w_arr = w_flat.reshape(1, 4, 8, 8)
        m_enc = ((np.tile(w_arr, (1, 1, 8, 8)) + gs_key_np) % 2).reshape(-1)
        np.random.seed(2026112290 + task)
        zT = native_gs.truncSampling(m_enc)

        # Step 4A: Paired unguided native GS baseline
        t0 = time.perf_counter()
        base_im = ddim_gen(prompt, zT, num_steps=50, ts_t=None)
        torch.cuda.synchronize()
        sec_base = time.perf_counter() - t0
        base_im.save(tdir / 'gs_unguided.png')

        # Step 4B: Commit-then-Steer (mid-step Tweedie guidance, unguided final 14 steps)
        t0 = time.perf_counter()
        guid_im = ddim_gen(prompt, zT, num_steps=50, ts_t=ts_t, step_range=(5, 35), step_size=0.035, stop_loss=0.10)
        torch.cuda.synchronize()
        sec_guid = time.perf_counter() - t0
        guid_im.save(tdir / 'commit_steer_preview8.png')

        guided_clean_embs.append(eval_sscd_pil(guid_im))
        # Independent same-prompt reference from color_joint_20261119 (disjoint seed!)
        sib_im = Image.open(CJ_RUN / 'images' / f'{task:03d}/0.png').convert('RGB')
        sibling_embs.append(eval_sscd_pil(sib_im))

        for arm_name, arm_im, arm_sec in [
            ('gs_unguided', base_im, sec_base),
            ('commit_steer_preview8', guid_im, sec_prev + sec_guid),
        ]:
            lp, cs = eval_lpips_clip(base_im, arm_im, prompt)
            for ci, cond in enumerate(['clean', *GS_9_CONDITIONS.keys()]):
                seed = 2026112200 + task * 100 + ci
                if cond == 'clean':
                    q_im = arm_im
                else:
                    gs_dist.set_random_seed(seed)
                    dargs = SimpleNamespace(**{k: None for k in GS_9_CONDITIONS})
                    setattr(dargs, cond, GS_9_CONDITIONS[cond])
                    q_im = gs_dist.image_distortion(arm_im.copy(), seed, dargs).convert('RGB')
                v = verify_blind(q_im, w_flat, true_sgn)
                records.append(dict(
                    task=task, dataset=s['dataset'], arm=arm_name, condition=cond,
                    preview_seconds=sec_prev, total_gen_seconds=arm_sec, lpips=lp, clip=cs, **v,
                ))
        print(f'[Probe A | GPU {gpu}] {idx+1}/16 task={task:03d} prev={sec_prev:.2f}s guid={sec_guid:.2f}s ({time.perf_counter()-t_start:.1f}s)', flush=True)

    # Same-prompt independent seed FAR & cross-prompt FAR on keyed P64(K)
    sib_far = sum((1.0 - np.dot(target_sgns[i], sibling_embs[i] @ P64_np) / (8.0 * np.linalg.norm(sibling_embs[i] @ P64_np) + 1e-12)) <= th for i in range(16))
    cross_far = sum((1.0 - np.dot(target_sgns[i], guided_clean_embs[j] @ P64_np) / (8.0 * np.linalg.norm(guided_clean_embs[j] @ P64_np) + 1e-12)) <= th for i in range(16) for j in range(16) if i != j)

    # Summarize
    summary = {}
    for arm in ('gs_unguided', 'commit_steer_preview8'):
        sub = [r for r in records if r['arm'] == arm]
        cl = [r for r in sub if r['condition'] == 'clean']
        ed = [r for r in sub if r['condition'] != 'clean']
        by_cond = {}
        for cond in ['clean', *GS_9_CONDITIONS.keys()]:
            cs = [r for r in sub if r['condition'] == cond]
            by_cond[cond] = dict(
                n=len(cs),
                auth_pass=sum(r['auth_ok'] for r in cs),
                feat_pass=sum(r['feat_ok'] for r in cs),
                joint_pass=sum(r['joint_blind_ok'] for r in cs),
                full_ba=float(np.mean([r['full_ba'] for r in cs])),
                p_ber=float(np.mean([r['payload_ber'] for r in cs])),
                d_blind=float(np.mean([r['d_blind'] for r in cs])),
            )
        summary[arm] = dict(
            preview_seconds_mean=float(np.mean([r['preview_seconds'] for r in cl])),
            total_gen_seconds_mean=float(np.mean([r['total_gen_seconds'] for r in cl])),
            lpips_mean=float(np.mean([r['lpips'] for r in cl])),
            clip_mean=float(np.mean([r['clip'] for r in cl])),
            clean_ba=float(np.mean([r['full_ba'] for r in cl])),
            clean_ber=float(np.mean([r['payload_ber'] for r in cl])),
            clean_joint=sum(r['joint_blind_ok'] for r in cl),
            gs9_auth_pass=sum(r['auth_ok'] for r in ed),
            gs9_feat_pass=sum(r['feat_ok'] for r in ed),
            gs9_joint_pass=sum(r['joint_blind_ok'] for r in ed),
            gs9_ber=float(np.mean([r['payload_ber'] for r in ed])),
            by_cond=by_cond,
        )

    # Visual montage (first 4 unseen prompts: 2 COCO + 2 SDP)
    vis = [selected[0]['task'], selected[1]['task'], selected[8]['task'], selected[9]['task']]
    sz, pad = 256, 28
    canvas = Image.new('RGB', (sz * 3, pad + sz * 4), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for ci, cname in enumerate(['preview8 (0.4s)', 'gs_unguided', 'commit_steer_preview8']):
        draw.text((ci * sz + 8, 6), cname, fill=(0, 0, 0))
    for ri, task in enumerate(vis):
        for ci, fname in enumerate(['preview8.png', 'gs_unguided.png', 'commit_steer_preview8.png']):
            im = Image.open(img_root / f'{task:03d}' / fname).convert('RGB').resize((sz, sz), Image.BICUBIC)
            canvas.paste(im, (ci * sz, pad + ri * sz))
    canvas.save(RUN / 'preview8_montage.jpg', quality=92)

    out = dict(
        elapsed_seconds=time.perf_counter() - t_start,
        gpu=gpu, uuid=uuid, threshold=th, n_unseen_prompts=16,
        same_prompt_other_seed_far=f'{sib_far}/16',
        cross_prompt_far=f'{cross_far}/240',
        arms=summary,
    )
    write_json(RUN / 'records.json', records)
    write_json(RUN / 'summary.json', out)
    print(f'PROBE A DONE in {out["elapsed_seconds"]:.1f}s | clean_joint={summary["commit_steer_preview8"]["clean_joint"]}/16 | gs9_joint={summary["commit_steer_preview8"]["gs9_joint_pass"]}/144 | LPIPS={summary["commit_steer_preview8"]["lpips_mean"]:.3f} | sibFAR={sib_far}/16 | crossFAR={cross_far}/240', flush=True)


if __name__ == '__main__':
    main()
