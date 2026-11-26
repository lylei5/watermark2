"""Stage 5D: Live Full-Budget 151-Step CVPR 2025 Imprint Forgery on SDXL Base 1.0 (1024x1024).
Targets the first 4 Commit-then-Steer SDXL guided watermarked images (Stage 4A sdxl_cross_backbone/images_guided/000..003.png)
and imprints their inverted SDXL latent onto 4 unrelated COCO cover images (000_cover..003_cover.png resized to 1024x1024)
via exact 151-step Adam optimization (lr=0.01, checkpoints 50/100/150) using gradient-checkpointed 30-step SDXL DDIM inversion
(bf16 UNet, fp32 latent parameter, ~11.6 GB peak VRAM on RTX 3090).

Evaluates BOTH Native GS-SDXL (192b auth > 136/192) and Commit-then-Steer SDXL (blind 64b SSCD commitment <= 0.6951)
using Stage 4A's exact frozen private_material.npz and threshold.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/data0/liyanlei/watermark_research')
RUN = ROOT / 'runs/commit_steer_stage5_20261126/sdxl_imprint'
SDXL_S4 = ROOT / 'runs/commit_steer_stage4_20261125/sdxl_cross_backbone'
COVER_DIR = ROOT / 'runs/published_imprint_dpm_full_20261110/inputs'
EXTRA_DEPS = ROOT / 'envs/phase1_extra'
SSCD_PATH = ROOT / 'models/sscd/sscd_disc_mixup.torchscript.pt'


def read_json(p: Path):
    return json.loads(p.read_bytes())


def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(p)


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


def sdxl_gs_decode(z_inv_np: np.ndarray, gs_key_128: np.ndarray) -> np.ndarray:
    assert z_inv_np.shape == (1, 4, 128, 128) and gs_key_128.shape == (1, 4, 128, 128)
    rev_sd = ((z_inv_np > 0).astype(np.int32) + gs_key_128.astype(np.int32)) % 2
    votes = rev_sd.reshape(1, 4, 16, 8, 16, 8).sum(axis=(2, 4))
    return (votes > 128).astype(np.uint8).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=5)
    ap.add_argument('--n-cases', type=int, default=4)
    args = ap.parse_args()

    uuid = check_idle_gpu(args.gpu)
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
    from torch.utils.checkpoint import checkpoint
    import lpips
    from diffusers import StableDiffusionXLPipeline, DDIMInverseScheduler

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(EXTRA_DEPS))

    s4_proto = read_json(SDXL_S4 / 'protocol.json')
    th = float(s4_proto['threshold'])
    auth_th = float(s4_proto['auth_threshold'])
    priv = np.load(SDXL_S4 / 'private_material.npz')
    gs_key_np, auth_192, P64_np = priv['key'], priv['auth_bits'], priv['P64']

    write_json(RUN / 'protocol.json', dict(
        run_name='sdxl_imprint',
        reused_stage4a=str(SDXL_S4),
        backbone='stabilityai/stable-diffusion-xl-base-1.0',
        resolution=1024,
        inv_steps=30,
        imprint_steps=151,
        lr=0.01,
        checkpoints=[50, 100, 150],
        n_cases=args.n_cases,
        threshold=th,
        auth_threshold=auth_th,
    ))

    print(f'[S5D SDXL Imprint | GPU {args.gpu}] Loading SDXL Base 1.0 (bf16 + gradient checkpointing)...', flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0',
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.requires_grad_(False); pipe.unet.eval()
    pipe.vae.requires_grad_(False); pipe.vae.eval()
    pipe.vae.enable_slicing()
    pipe.unet.enable_gradient_checkpointing()
    inv_sched = DDIMInverseScheduler.from_config(pipe.scheduler.config)

    sscd = torch.jit.load(str(SSCD_PATH), map_location='cuda').eval()
    for p in sscd.parameters():
        p.requires_grad_(False)
    lpips_fn = lpips.LPIPS(net='alex').cuda().eval()

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
        added_cond = {'text_embeds': nppe_null, 'time_ids': add_time_ids_1}

    def encode_pil_1024(im: Image.Image) -> torch.Tensor:
        x = torch.from_numpy(
            np.asarray(im.convert('RGB').resize((1024, 1024), Image.BICUBIC), dtype=np.float32) / 127.5 - 1.0
        ).permute(2, 0, 1).unsqueeze(0).to(device='cuda', dtype=torch.bfloat16)
        with torch.no_grad():
            return (pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor).float()

    def decode_lat_1024(z: torch.Tensor) -> Image.Image:
        with torch.no_grad():
            dec = pipe.vae.decode(z.to(torch.bfloat16) / pipe.vae.config.scaling_factor, return_dict=False)[0]
            arr = (dec.float()[0].permute(1, 2, 0) / 2.0 + 0.5).clamp(0.0, 1.0).mul(255.0).round().byte().cpu().numpy()
        return Image.fromarray(arr)

    def invert_no_grad(im: Image.Image) -> np.ndarray:
        inv_sched.set_timesteps(30, device='cuda')
        z = encode_pil_1024(im).to(torch.bfloat16)
        with torch.no_grad():
            for t in inv_sched.timesteps:
                eps = pipe.unet(inv_sched.scale_model_input(z, t), t, encoder_hidden_states=npe_null, added_cond_kwargs=added_cond, return_dict=False)[0]
                z = inv_sched.step(eps, t, z, return_dict=False)[0]
        return z.float().cpu().numpy()

    def blind_verify_sdxl(im: Image.Image):
        z_inv = invert_no_grad(im)
        rev_w = sdxl_gs_decode(z_inv, gs_key_np)
        auth_ba = float((rev_w[:192] == auth_192).mean())
        auth_ok = auth_ba > auth_th
        rec_sgn = 2.0 * rev_w[192:].astype(np.float64) - 1.0
        x = torch.from_numpy(np.asarray(im.convert('RGB').resize((320, 320), Image.BICUBIC), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()
        with torch.no_grad():
            emb = F.normalize(sscd((x - mean_norm) / std_norm), dim=1)[0].cpu().numpy().astype(np.float64)
        qp = emb @ P64_np
        d_blind = float(1.0 - np.dot(rec_sgn, qp) / (8.0 * float(np.linalg.norm(qp)) + 1e-12))
        feat_ok = d_blind <= th
        return dict(auth_ba=auth_ba, auth_ok=bool(auth_ok), d_blind=d_blind, feat_ok=bool(feat_ok), joint_accept=bool(auth_ok and feat_ok))

    def lpips_psnr_512(im1: Image.Image, im2: Image.Image):
        a1 = np.asarray(im1.convert('RGB').resize((512, 512), Image.BICUBIC), dtype=np.float32) / 255.0
        a2 = np.asarray(im2.convert('RGB').resize((512, 512), Image.BICUBIC), dtype=np.float32) / 255.0
        mse = float(np.mean((a1 - a2) ** 2))
        psnr = 100.0 if mse < 1e-12 else float(-10.0 * np.log10(mse))
        t1 = torch.from_numpy(a1.transpose(2, 0, 1)).unsqueeze(0).cuda() * 2.0 - 1.0
        t2 = torch.from_numpy(a2.transpose(2, 0, 1)).unsqueeze(0).cuda() * 2.0 - 1.0
        with torch.no_grad():
            lp = float(lpips_fn(t1, t2).item())
        return psnr, lp

    inv_sched.set_timesteps(30, device='cuda')
    t_list = list(inv_sched.timesteps)

    def step_fn(lat, t_idx):
        t = t_list[int(t_idx)]
        eps = pipe.unet(lat.to(torch.bfloat16), t, encoder_hidden_states=npe_null, added_cond_kwargs=added_cond, return_dict=False)[0].float()
        return inv_sched.step(eps, t, lat, return_dict=False)[0]

    imp_dir = RUN / 'imprint_151'
    imp_dir.mkdir(exist_ok=True)
    cases = []
    t_all = time.perf_counter()

    for i in range(args.n_cases):
        wm_im = Image.open(SDXL_S4 / f'images_guided/{i:03d}.png').convert('RGB')
        cover_im = Image.open(COVER_DIR / f'{i:03d}_cover.png').convert('RGB').resize((1024, 1024), Image.BICUBIC)
        zT_target = torch.from_numpy(invert_no_grad(wm_im)).to('cuda', dtype=torch.float32)
        z0_orig = encode_pil_1024(cover_im)

        z0 = torch.nn.Parameter(z0_orig.detach().clone())
        opt = torch.optim.Adam([z0], lr=0.01)
        ckpts = {}
        t0 = time.perf_counter()
        for step in range(151):
            opt.zero_grad()
            lat = z0
            for ti in range(len(t_list)):
                lat = checkpoint(step_fn, lat, torch.tensor(ti), use_reentrant=False)
            loss = F.mse_loss(lat, zT_target)
            loss.backward()
            opt.step()
            if step in (50, 100, 150):
                forged_im = decode_lat_1024(z0.detach())
                forged_im.save(imp_dir / f'idx_{i:03d}_step_{step:03d}.png')
                v = blind_verify_sdxl(forged_im)
                psnr, lp = lpips_psnr_512(cover_im, forged_im)
                ckpts[f'step_{step}'] = dict(loss=float(loss.detach()), psnr_vs_cover=psnr, lpips_vs_cover=lp, **v)
                print(f'  [SDXL Imprint] idx={i:03d} step={step:3d} ({time.perf_counter()-t0:.0f}s) | loss={float(loss.detach()):.3f} PSNR={psnr:.1f} | auth={v["auth_ba"]:.4f}({v["auth_ok"]}) | d_blind={v["d_blind"]:.4f}({v["feat_ok"]}) | joint={v["joint_accept"]}', flush=True)
        cases.append(dict(idx=i, cover=f'{i:03d}_cover.png', seconds=time.perf_counter() - t0, checkpoints=ckpts))
        write_json(RUN / 'summary_partial.json', dict(cases=cases))

    agg = {}
    for ck in ('step_50', 'step_100', 'step_150'):
        agg[ck] = dict(
            n=len(cases),
            native_gs_sdxl_fooled=sum(c['checkpoints'][ck]['auth_ok'] for c in cases),
            commit_steer_sdxl_fooled=sum(c['checkpoints'][ck]['joint_accept'] for c in cases),
            mean_auth_ba=float(np.mean([c['checkpoints'][ck]['auth_ba'] for c in cases])),
            mean_d_blind=float(np.mean([c['checkpoints'][ck]['d_blind'] for c in cases])),
            mean_psnr=float(np.mean([c['checkpoints'][ck]['psnr_vs_cover'] for c in cases])),
        )

    # Visual montage
    sz, pad = 256, 28
    canvas = Image.new('RGB', (sz * 3, pad + sz * len(cases)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for ci, title in enumerate(['SDXL Guided WM (PASS)', 'Unrelated Cover (1024)', 'SDXL Imprint-150 (GS Fooled / Ours REJ)']):
        draw.text((ci * sz + 6, 6), title, fill=(0, 0, 0))
    for ri, c in enumerate(cases):
        ims = [
            Image.open(SDXL_S4 / f'images_guided/{c["idx"]:03d}.png').convert('RGB'),
            Image.open(COVER_DIR / c['cover']).convert('RGB'),
            Image.open(imp_dir / f'idx_{c["idx"]:03d}_step_150.png').convert('RGB'),
        ]
        for ci, im in enumerate(ims):
            canvas.paste(im.resize((sz, sz), Image.BICUBIC), (ci * sz, pad + ri * sz))
    canvas.save(RUN / 'sdxl_imprint_montage.jpg', quality=92)

    out = dict(
        elapsed_seconds=time.perf_counter() - t_all,
        gpu=args.gpu, uuid=uuid, threshold=th,
        imprint_summary=agg,
        imprint_cases=cases,
    )
    write_json(RUN / 'summary.json', out)
    print(f"[S5D SDXL Imprint DONE] {out['elapsed_seconds']:.1f}s | step150: Native_GS_SDXL={agg['step_150']['native_gs_sdxl_fooled']}/{len(cases)} | CommitSteer_SDXL={agg['step_150']['commit_steer_sdxl_fooled']}/{len(cases)} | mean_d={agg['step_150']['mean_d_blind']:.4f}", flush=True)


if __name__ == '__main__':
    main()
