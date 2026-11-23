"""Probe B (GPU 3): LIVE Full-Budget CVPR 2025 Published Forgery Attacks DIRECTLY on our
Stage 1 Commit-then-Steer guided watermarked images (sscd64_mid_tweedie):
  1) Full-budget 151-step Imprint Forgery (4 guided sources -> 4 disjoint COCO covers, ckpts 50/100/150)
  2) Full-budget 9-candidate Reprompt+ across BOTH SD2.1 & SD1.5 proxies (6 sources x 3 prompts x 3 wiggles = 54/proxy = 108 candidates)
"""
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import norm

ROOT = Path('/data0/liyanlei/watermark_research')
STAGE1 = ROOT / 'runs/gs_payload_guided_stage1_20261122'
STAGE0 = ROOT / 'runs/payload_binding_stage0_20261122'
RUN = ROOT / 'runs/commit_steer_stage2_20261122/probe_b_live_attacks'
GS_MOD_PATH = ROOT / 'runs/phase1_trajectory_native_gs_20261113/gs_watermark.py'
PGID_PIPE = ROOT / 'third_party/frozen/PGID_dc4ad96'
SEM_FORGERY = ROOT / 'third_party/semantic-forgery'
EXTRA_DEPS = ROOT / 'envs/phase1_extra'
SSCD_PATH = ROOT / 'models/sscd/sscd_disc_mixup.torchscript.pt'
COVER_DIR = ROOT / 'runs/published_imprint_dpm_full_20261110/inputs'
SD15_PATH = Path('/data0/liyanlei/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14')


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


def wiggle_gs_latent(latents_t, rng):
    """Exact CVPR 2025 gs_provider.wiggle_latents (l=1 binary Gaussian Shading)."""
    arr = latents_t.detach().float().cpu().numpy()
    y = np.clip((norm.cdf(arr) * 2.0).astype(np.int32), 0, 1)
    u = rng.uniform(1e-6, 1.0 - 1e-6, size=y.shape).astype(np.float64)
    new_lat = norm.ppf((u + y) / 2.0)
    return new_lat


def main():
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 3
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
    from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sys.path.insert(0, str(PGID_PIPE))
    sys.path.insert(0, str(EXTRA_DEPS))
    sys.path.insert(0, str(SEM_FORGERY))
    from pipe import pipe_utils
    from utils import imprint_utils

    proto1 = read_json(STAGE1 / 'protocol.json')
    th = proto1['stage0_thresholds']['sscd_64b']
    priv = np.load(STAGE1 / 'private_material.npz')
    gs_key_np, auth_bits_np = priv['key'], priv['auth_bits']
    P64_np = np.load(STAGE0 / 'sscd_projections.npz')['p64'].astype(np.float64)

    native_gs = load_module('native_gs', GS_MOD_PATH).Gaussian_Shading(1, 8, 1e-6, 1)
    native_gs.key = torch.from_numpy(gs_key_np.astype(np.int64)).cuda()

    sscd = torch.jit.load(str(SSCD_PATH), map_location='cuda').eval()
    lpips_fn = lpips.LPIPS(net='alex').cuda().eval()
    mean_norm = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    std_norm = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)

    # Target verifier pipe (SD2.1 fp16 DDIM, exact same as Stage 1)
    verifier = pipe_utils.create_diffusion_pipe(
        pretrained_model_name_or_path='Manojb/stable-diffusion-2-1-base',
        resolution=512, device='cuda:0', scheduler_name='DDIM', scheduler_inverse_name='DDIM',
        dtype=torch.float16, eager_loading=True, disable_tqdm=True,
    )
    verifier.pipe.set_progress_bar_config(disable=True)

    def eval_sscd_pil(im):
        x = torch.from_numpy(np.asarray(im.convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()
        x_in = (F.interpolate(x, size=(320, 320), mode='bilinear', align_corners=False) - mean_norm) / std_norm
        with torch.no_grad():
            return F.normalize(sscd(x_in), dim=1)[0].cpu().numpy().astype(np.float64)

    def blind_verify(im):
        z_inv = verifier.invert_images(im, num_inference_steps=50)['zT_torch']
        rev_w = native_gs.diffusion_inverse(((z_inv > 0).int() + native_gs.key) % 2).reshape(-1).cpu().numpy().astype(np.uint8)
        auth_ba = float((rev_w[:192] == auth_bits_np[:192]).mean())
        auth_ok = auth_ba > (136.0 / 192.0)
        rec_sgn = 2.0 * rev_w[192:].astype(np.float64) - 1.0
        qp = eval_sscd_pil(im) @ P64_np
        d_blind = float(1.0 - np.dot(rec_sgn, qp) / (8.0 * float(np.linalg.norm(qp)) + 1e-12))
        feat_ok = d_blind <= th
        return dict(auth_ba=auth_ba, auth_ok=bool(auth_ok), d_blind=d_blind, feat_ok=bool(feat_ok), joint_accept=bool(auth_ok and feat_ok))

    def lpips_psnr(im1, im2):
        a1 = np.asarray(im1.convert('RGB'), dtype=np.float32) / 255.0
        a2 = np.asarray(im2.convert('RGB'), dtype=np.float32) / 255.0
        mse = float(np.mean((a1 - a2) ** 2))
        psnr = 100.0 if mse < 1e-12 else float(-10.0 * np.log10(mse))
        t1 = torch.from_numpy(a1.transpose(2, 0, 1)).unsqueeze(0).cuda() * 2.0 - 1.0
        t2 = torch.from_numpy(a2.transpose(2, 0, 1)).unsqueeze(0).cuda() * 2.0 - 1.0
        with torch.no_grad():
            lp = float(lpips_fn(t1, t2).item())
        return psnr, lp

    t_all = time.perf_counter()

    # =========================================================================
    # PART 1: LIVE FULL-BUDGET 151-STEP CVPR 2025 IMPRINT FORGERY (4 cases)
    # =========================================================================
    print('=== PART 1: Live 151-step CVPR 2025 Imprint Forgery (4 cases) ===', flush=True)
    pipe_atk, _, inv_sched = imprint_utils.load_attacker_pipe(
        modelid='Manojb/stable-diffusion-2-1-base', scheduler='DDIM', device='cuda'
    )
    pipe_atk.set_progress_bar_config(disable=True)
    diffpipe = imprint_utils.DiffPipe(pipe_atk, scheduler=inv_sched, device=pipe_atk.device)

    imp_dir = RUN / 'imprint_151'
    imp_dir.mkdir(exist_ok=True)
    imprint_results = []
    for i in range(4):
        task = proto1['sources'][i]['task']
        wm_im = Image.open(STAGE1 / 'images' / f'{task:03d}/sscd64_mid_tweedie.png').convert('RGB')
        cover_im = Image.open(COVER_DIR / f'{i:03d}_cover.png').convert('RGB').resize((512, 512), Image.BICUBIC)

        with torch.no_grad():
            zT_target = imprint_utils.invert_image(pipe=pipe_atk, scheduler=inv_sched, image=wm_im, num_inference_steps=50)
            z0_orig = imprint_utils.pixel_to_latent(cover_im, pipe_atk)

        z0 = torch.nn.Parameter(z0_orig.detach().clone())
        opt = torch.optim.Adam([z0], lr=0.01)
        ckpts = {}
        t0 = time.perf_counter()
        for step in range(151):
            opt.zero_grad()
            inv_lat = diffpipe(z0, '', guidance_scale=1.0, num_inference_steps=50)
            loss = F.mse_loss(inv_lat, zT_target.detach())
            loss.backward()
            opt.step()
            if step in (50, 100, 150):
                with torch.no_grad():
                    forged_im = imprint_utils.latent_to_pil(z0, pipe_atk)[0].convert('RGB')
                forged_im.save(imp_dir / f'task_{task:03d}_step_{step:03d}.png')
                v = blind_verify(forged_im)
                psnr, lp = lpips_psnr(cover_im, forged_im)
                ckpts[f'step_{step}'] = dict(loss=float(loss.detach()), psnr_vs_cover=psnr, lpips_vs_cover=lp, **v)
                print(f'  Imprint task={task:03d} step={step:3d} | loss={float(loss.detach()):.4f} | PSNR={psnr:.1f} LPIPS={lp:.3f} | native_GS={v["auth_ba"]:.4f} ({v["auth_ok"]}) | blind_sscd_d={v["d_blind"]:.4f} ({v["feat_ok"]}) | joint={v["joint_accept"]}', flush=True)
        imprint_results.append(dict(task=task, cover=f'{i:03d}_cover.png', seconds=time.perf_counter() - t0, checkpoints=ckpts))

    # Build Imprint visual montage (4 rows x 3 cols: WM source | Cover | Imprint step 150)
    sz, pad = 256, 28
    canvas = Image.new('RGB', (sz * 3, pad + sz * 4), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for ci, cname in enumerate(['Guided WM Source (PASS)', 'Unrelated Cover', 'Imprint Step-150 (GS Fooled / Ours REJ)']):
        draw.text((ci * sz + 6, 6), cname, fill=(0, 0, 0))
    for ri, r in enumerate(imprint_results):
        task = r['task']
        ims = [
            Image.open(STAGE1 / 'images' / f'{task:03d}/sscd64_mid_tweedie.png').convert('RGB'),
            Image.open(COVER_DIR / r['cover']).convert('RGB'),
            Image.open(imp_dir / f'task_{task:03d}_step_150.png').convert('RGB'),
        ]
        for ci, im in enumerate(ims):
            canvas.paste(im.resize((sz, sz), Image.BICUBIC), (ci * sz, pad + ri * sz))
    canvas.save(RUN / 'imprint_live_montage.jpg', quality=92)

    del diffpipe, pipe_atk
    torch.cuda.empty_cache()

    # =========================================================================
    # PART 2: LIVE FULL-BUDGET 9-CANDIDATE REPROMPT+ (SD2.1 + SD1.5 PROXIES)
    # =========================================================================
    print('=== PART 2: Live 9-candidate Reprompt+ (6 sources x 2 proxies x 9 = 108 cands) ===', flush=True)
    rp_proto = read_json(ROOT / 'runs/author_reprompt_plus_20261115/protocol.json')
    gs_refs = [r for r in rp_proto['references'] if r['carrier'] == 'GS'][:6]
    rp_results = {}

    for proxy_name, model_ref in [('sd21', 'Manojb/stable-diffusion-2-1-base'), ('sd15', str(SD15_PATH))]:
        print(f'--- Loading proxy {proxy_name} ---', flush=True)
        proxy_pipe = StableDiffusionPipeline.from_pretrained(model_ref, torch_dtype=torch.float16, local_files_only=True, safety_checker=None).to('cuda')
        fwd_sched = DDIMScheduler.from_config(proxy_pipe.scheduler.config)
        inv_sched_p = DDIMInverseScheduler.from_config(proxy_pipe.scheduler.config)
        proxy_pipe.set_progress_bar_config(disable=True)

        p_rows = []
        rng_wig = np.random.default_rng(2026112203 if proxy_name == 'sd21' else 2026112204)
        for i in range(6):
            task = proto1['sources'][i]['task']
            wm_im = Image.open(STAGE1 / 'images' / f'{task:03d}/sscd64_mid_tweedie.png').convert('RGB')
            # Attacker black-box inversion with proxy model
            proxy_pipe.scheduler = inv_sched_p
            with torch.no_grad():
                img_t = torch.from_numpy(np.asarray(wm_im, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to('cuda', dtype=torch.float16) * 2.0 - 1.0
                z0_p = proxy_pipe.vae.encode(img_t).latent_dist.mean * proxy_pipe.vae.config.scaling_factor
                zT_p = proxy_pipe(prompt=[''], latents=z0_p, num_inference_steps=50, guidance_scale=1.0, output_type='latent', return_dict=False)[0]

            proxy_pipe.scheduler = fwd_sched
            prompts3 = [tp['prompt'] for tp in gs_refs[i]['target_prompts']]
            assert len(prompts3) == 3
            for pi, ptxt in enumerate(prompts3):
                for ri in range(3):
                    lat_np = wiggle_gs_latent(zT_p, rng_wig)
                    lat_t = torch.from_numpy(lat_np).to('cuda', dtype=torch.float16)
                    with torch.no_grad():
                        cand_im = proxy_pipe(prompt=[ptxt], latents=lat_t, num_inference_steps=50, guidance_scale=7.5).images[0].convert('RGB')
                    v = blind_verify(cand_im)
                    p_rows.append(dict(task=task, prompt_idx=pi, resample_idx=ri, **v))
            src_gs = sum(r['auth_ok'] for r in p_rows if r['task'] == task)
            src_ours = sum(r['joint_accept'] for r in p_rows if r['task'] == task)
            print(f'  Reprompt+ {proxy_name} task={task:03d}: native_GS={src_gs}/9 | CommitSteer={src_ours}/9', flush=True)

        tasks_u = sorted({r['task'] for r in p_rows})
        rp_results[proxy_name] = dict(
            n_candidates=len(p_rows), n_sources=len(tasks_u),
            native_gs_cand=sum(r['auth_ok'] for r in p_rows),
            native_gs_src=sum(any(r['auth_ok'] for r in p_rows if r['task'] == t) for t in tasks_u),
            commit_steer_cand=sum(r['joint_accept'] for r in p_rows),
            commit_steer_src=sum(any(r['joint_accept'] for r in p_rows if r['task'] == t) for t in tasks_u),
            mean_d_blind=float(np.mean([r['d_blind'] for r in p_rows])),
            rows=p_rows,
        )
        del proxy_pipe
        torch.cuda.empty_cache()

    imprint_agg = {}
    for ck in ('step_50', 'step_100', 'step_150'):
        imprint_agg[ck] = dict(
            n=len(imprint_results),
            native_gs_fooled=sum(r['checkpoints'][ck]['auth_ok'] for r in imprint_results),
            commit_steer_fooled=sum(r['checkpoints'][ck]['joint_accept'] for r in imprint_results),
            mean_auth_ba=float(np.mean([r['checkpoints'][ck]['auth_ba'] for r in imprint_results])),
            mean_d_blind=float(np.mean([r['checkpoints'][ck]['d_blind'] for r in imprint_results])),
            mean_psnr=float(np.mean([r['checkpoints'][ck]['psnr_vs_cover'] for r in imprint_results])),
        )

    summary = dict(
        elapsed_seconds=time.perf_counter() - t_all,
        gpu=gpu, uuid=uuid, threshold=th,
        imprint_summary=imprint_agg,
        imprint_cases=imprint_results,
        reprompt_plus_summary={k: {kk: vv for kk, vv in v.items() if kk != 'rows'} for k, v in rp_results.items()},
    )
    write_json(RUN / 'summary.json', summary)
    write_json(RUN / 'reprompt_plus_rows.json', rp_results)
    print(f'PROBE B DONE in {summary["elapsed_seconds"]:.1f}s | Imprint150 GS={imprint_agg["step_150"]["native_gs_fooled"]}/4 Ours={imprint_agg["step_150"]["commit_steer_fooled"]}/4 | RP+_sd21 GS={rp_results["sd21"]["native_gs_src"]}/6({rp_results["sd21"]["native_gs_cand"]}/54) Ours={rp_results["sd21"]["commit_steer_src"]}/6({rp_results["sd21"]["commit_steer_cand"]}/54) | RP+_sd15 GS={rp_results["sd15"]["native_gs_src"]}/6({rp_results["sd15"]["native_gs_cand"]}/54) Ours={rp_results["sd15"]["commit_steer_src"]}/6({rp_results["sd15"]["commit_steer_cand"]}/54)', flush=True)


if __name__ == '__main__':
    main()
