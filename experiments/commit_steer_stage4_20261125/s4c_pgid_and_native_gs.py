"""Stage 4C: Head-to-Head Same-Cohort Direct Baseline Evaluation — Native GS-256b & Official PGID-F (PGID_dc4ad96).
Evaluates on the EXACT SAME images, keys, distortion seeds, natural nulls, and attack protocols as Commit-then-Steer:
  1. native100        - Native GS-256b on ALL 100 gs100_unified unguided images x (Clean + 9 GS distortions = 1,000 invs)
                        plus exact single-image verification timing.
  2. pgid30           - Official PGID-F (k=15, s=3, gamma=0.001, fp32) on matched 30 stratified unified prompts
                        (15 COCO [0..14] + 15 SDP [50..64]) x (Clean + 9 GS distortions = 300 PGID-F verifications),
                        with explicit batch-vs-single numerical equivalence verification & single-image timing.
  3. pgid_null200     - Official PGID-F on 200 real COCO validation images (unified_null_features_20261116_v2 val[0:200]).
  4. pgid_live_attacks - Full-budget CVPR 2025 Live 151-step Imprint (4 sources x ckpts 50/100/150) &
                         Live 9-candidate Reprompt+ (6 sources x 3 I2P x 3 wiggles x 2 proxies SD2.1/SD1.5 = 108 cands)
                         evaluated under BOTH Native GS-256b and Official PGID-F!
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.stats import beta, norm

ROOT = Path('/data0/liyanlei/watermark_research')
RUN = ROOT / 'runs/commit_steer_stage4_20261125/head2head_pgid_native_gs'
GS100_S3 = ROOT / 'runs/commit_steer_paper_scale_20261123/gs100_unified'
NULL_INPUTS = ROOT / 'runs/unified_null_features_20261116_v2/inputs.json'
GS_MOD_PATH = ROOT / 'runs/phase1_trajectory_native_gs_20261113/gs_watermark.py'
GS_DIST_PATH = ROOT / 'runs/defense_revalidation_20261111/gs_image_utils.py'
PGID_PIPE = ROOT / 'third_party/frozen/PGID_dc4ad96'
SEM_FORGERY = ROOT / 'third_party/semantic-forgery'
EXTRA_DEPS = ROOT / 'envs/phase1_extra'
COVER_DIR = ROOT / 'runs/published_imprint_dpm_full_20261110/inputs'
SD15_PATH = Path('/data0/liyanlei/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14')

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

# Stratified 30-prompt subset of gs100_unified (15 COCO + 15 SDP), strictly containing the 20-prompt SemBind cohort
MATCHED_30_GIDX = list(range(0, 15)) + list(range(50, 65))


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


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    return 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))


def wiggle_gs_latent(latents_t, rng):
    arr = latents_t.detach().float().cpu().numpy()
    y = np.clip((norm.cdf(arr) * 2.0).astype(np.int32), 0, 1)
    u = rng.uniform(1e-6, 1.0 - 1e-6, size=y.shape).astype(np.float64)
    return norm.ppf((u + y) / 2.0)


def load_w256_map():
    priv = np.load(GS100_S3 / 'private_material.npz')
    auth_192 = priv['auth_bits'].astype(np.uint8)
    embs = []
    for s in (0, 1):
        embs.extend(read_json(GS100_S3 / f'embs_shard{s}.json'))
    w_map = {}
    for e in embs:
        w_map[e['gidx']] = np.concatenate([auth_192, np.array(e['target_bits'], dtype=np.uint8)])
    assert len(w_map) == 100
    return priv['key'], auth_192, w_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['native100', 'pgid30', 'pgid_null200', 'attacks', 'all'])
    ap.add_argument('--gpu', type=int, default=1)
    args = ap.parse_args()

    uuid = check_idle_gpu(args.gpu)
    RUN.mkdir(parents=True, exist_ok=True)
    # Isolated working directory so gs_image_utils tmp_25.jpg never races with other workers
    work_dir = RUN / 'work'
    work_dir.mkdir(exist_ok=True)
    os.chdir(work_dir)

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
    from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sys.path.insert(0, str(PGID_PIPE))
    sys.path.insert(0, str(EXTRA_DEPS))
    from pipe import pipe_utils
    from pgid_defense import PGIDPipeline

    gs_key_np, auth_192, w_map = load_w256_map()
    s3_proto = read_json(GS100_S3 / 'protocol.json')
    sources = s3_proto['sources']
    native_gs = load_module('native_gs', GS_MOD_PATH).Gaussian_Shading(1, 8, 1e-6, 1)
    native_gs.key = torch.from_numpy(gs_key_np.astype(np.int64)).cuda()
    gs_dist = load_module('gs_dist', GS_DIST_PATH)

    # Official PGID uses fp32 SD2.1 DDIM
    provider = pipe_utils.create_diffusion_pipe(
        pretrained_model_name_or_path='Manojb/stable-diffusion-2-1-base',
        resolution=512, device='cuda:0', scheduler_name='DDIM', scheduler_inverse_name='DDIM',
        dtype=torch.float32, eager_loading=True, disable_tqdm=True,
    )
    provider.pipe.set_progress_bar_config(disable=True)
    pgid_f = PGIDPipeline(
        pipe=provider,
        scheduler_config=provider.pipe.scheduler.config,
        num_inference_steps=50,
        stopping_timestep=15,
        num_skip_inversion=3,
        guidance_strength=0.001,
    )

    conds = ['clean', *GS_9_CONDITIONS.keys()]

    def score_zT(zT_1, w256):
        rev_w = native_gs.diffusion_inverse(((zT_1 > 0).int() + native_gs.key) % 2).reshape(-1).cpu().numpy().astype(np.uint8)
        ba_256 = float((rev_w == w256).mean())
        auth_ba = float((rev_w[:192] == auth_192).mean())
        return dict(
            ba_256=ba_256,
            accept_gs_default=bool(ba_256 > 181.0 / 256.0),
            accept_gs_167=bool(ba_256 >= 167.0 / 256.0),
            auth_ba_192=auth_ba,
            accept_auth192=bool(auth_ba > 136.0 / 192.0),
        )

    def make_distorted_list(im, gidx):
        q_imgs = []
        for ci, cond in enumerate(conds):
            if cond == 'clean':
                q_imgs.append(im)
            else:
                seed = 2026112300 + gidx * 100 + ci
                gs_dist.set_random_seed(seed)
                dargs = SimpleNamespace(**{k: None for k in GS_9_CONDITIONS})
                setattr(dargs, cond, GS_9_CONDITIONS[cond])
                q_imgs.append(gs_dist.image_distortion(im.copy(), seed, dargs).convert('RGB'))
        return q_imgs

    def invert_native_batch(pil_list, bs=5):
        out = []
        for i in range(0, len(pil_list), bs):
            out.append(provider.invert_images(pil_list[i:i + bs], num_inference_steps=50)['zT_torch'])
        return torch.cat(out, dim=0)

    def invert_pgid_batch(pil_list, bs=5):
        out = []
        for i in range(0, len(pil_list), bs):
            out.append(pgid_f.run_defense(pil_list[i:i + bs])['zT'])
        return torch.cat(out, dim=0)

    t_global = time.perf_counter()

    # Timing + single-vs-batched numerical equivalence check for PGID-F
    im0 = Image.open(GS100_S3 / 'images_unguided/000.png').convert('RGB')
    im1 = Image.open(GS100_S3 / 'images_unguided/001.png').convert('RGB')
    torch.cuda.synchronize(); t0 = time.perf_counter()
    z_nat_single = provider.invert_images([im0], num_inference_steps=50)['zT_torch']
    torch.cuda.synchronize(); sec_verify_native_single = time.perf_counter() - t0

    torch.cuda.synchronize(); t0 = time.perf_counter()
    z_pgid_single = pgid_f.run_defense([im0])['zT']
    torch.cuda.synchronize(); sec_verify_pgid_single = time.perf_counter() - t0

    z_pgid_bat = pgid_f.run_defense([im0, im1])['zT'][0:1]
    batch_max_diff = float((z_pgid_single - z_pgid_bat).abs().max().item())
    s_single = score_zT(z_pgid_single, w_map[0])
    s_bat = score_zT(z_pgid_bat, w_map[0])
    assert s_single['ba_256'] == s_bat['ba_256'] and batch_max_diff < 1e-3, f'PGID batch mismatch: {batch_max_diff}'
    print(f'[S4C Preflight] single verify: Native_GS={sec_verify_native_single:.2f}s | PGID-F={sec_verify_pgid_single:.2f}s | batch_max_diff={batch_max_diff:.2e} (exact bit match)', flush=True)

    # =========================================================================
    # PART 1: NATIVE GS-256b ON ALL 100 UNIFIED PROMPTS (Clean + 9 GS Distortions)
    # =========================================================================
    if args.mode in ('native100', 'all'):
        print('=== [S4C Part 1] Native GS-256b on ALL 100 Unified Prompts (1,000 DDIM inversions) ===', flush=True)
        t0 = time.perf_counter()
        nat_recs = []
        for gidx in range(100):
            im = Image.open(GS100_S3 / 'images_unguided' / f'{gidx:03d}.png').convert('RGB')
            q_imgs = make_distorted_list(im, gidx)
            zTs = invert_native_batch(q_imgs, bs=5)
            for ci, cond in enumerate(conds):
                sc = score_zT(zTs[ci:ci+1], w_map[gidx])
                nat_recs.append(dict(gidx=gidx, dataset=sources[gidx]['dataset'], condition=cond, **sc))
            if (gidx + 1) % 20 == 0:
                print(f'  [Native100] {gidx+1}/100 done ({time.perf_counter()-t0:.1f}s)', flush=True)

        def summarize_recs(recs, label):
            cl = [r for r in recs if r['condition'] == 'clean']
            ed = [r for r in recs if r['condition'] != 'clean']
            ed7 = [r for r in ed if r['condition'] not in ('random_crop_ratio', 'random_drop_ratio')]
            by_c = {}
            for cond in conds:
                sub = [r for r in recs if r['condition'] == cond]
                by_c[cond] = dict(
                    n=len(sub),
                    pass_gs_default=sum(r['accept_gs_default'] for r in sub),
                    pass_gs_167=sum(r['accept_gs_167'] for r in sub),
                    pass_auth192=sum(r['accept_auth192'] for r in sub),
                    mean_ba_256=float(np.mean([r['ba_256'] for r in sub])),
                )
            return dict(
                label=label, n_prompts=len(cl),
                clean_gs_default=sum(r['accept_gs_default'] for r in cl),
                gs7_regular_default=sum(r['accept_gs_default'] for r in ed7),
                gs9_all_default=sum(r['accept_gs_default'] for r in ed),
                by_cond=by_c,
            )

        nat_30 = [r for r in nat_recs if r['gidx'] in MATCHED_30_GIDX]
        nat_20 = [r for r in nat_recs if r['gidx'] in (list(range(10)) + list(range(50, 60)))]
        nat_sum = dict(
            elapsed_seconds=time.perf_counter() - t0,
            sec_verify_single=sec_verify_native_single,
            all_100=summarize_recs(nat_recs, 'native_gs_100'),
            matched_30=summarize_recs(nat_30, 'native_gs_30'),
            matched_20=summarize_recs(nat_20, 'native_gs_20'),
        )
        write_json(RUN / 'native100_summary.json', nat_sum)
        write_json(RUN / 'native100_records.json', nat_recs)
        print(f"[S4C Native100 DONE] {nat_sum['elapsed_seconds']:.1f}s | clean={nat_sum['all_100']['clean_gs_default']}/100 | gs9={nat_sum['all_100']['gs9_all_default']}/900 | matched30 gs9={nat_sum['matched_30']['gs9_all_default']}/270", flush=True)

    # =========================================================================
    # PART 2: OFFICIAL PGID-F ON MATCHED 30 UNIFIED PROMPTS (Clean + 9 GS = 300 verifs)
    # =========================================================================
    if args.mode in ('pgid30', 'all'):
        print('=== [S4C Part 2] Official PGID-F (k=15, s=3, gamma=0.001) on 30 Matched Prompts (300 verifs) ===', flush=True)
        t0 = time.perf_counter()
        pg_recs = []
        for pi, gidx in enumerate(MATCHED_30_GIDX):
            im = Image.open(GS100_S3 / 'images_unguided' / f'{gidx:03d}.png').convert('RGB')
            q_imgs = make_distorted_list(im, gidx)
            zTs = invert_pgid_batch(q_imgs, bs=5)
            for ci, cond in enumerate(conds):
                sc = score_zT(zTs[ci:ci+1], w_map[gidx])
                pg_recs.append(dict(gidx=gidx, dataset=sources[gidx]['dataset'], condition=cond, **sc))
            if (pi + 1) % 5 == 0:
                print(f'  [PGID30] {pi+1}/30 done ({time.perf_counter()-t0:.1f}s)', flush=True)
                write_json(RUN / 'pgid30_records_partial.json', pg_recs)

        def summarize_pg(recs, label):
            cl = [r for r in recs if r['condition'] == 'clean']
            ed = [r for r in recs if r['condition'] != 'clean']
            ed7 = [r for r in ed if r['condition'] not in ('random_crop_ratio', 'random_drop_ratio')]
            by_c = {}
            for cond in conds:
                sub = [r for r in recs if r['condition'] == cond]
                by_c[cond] = dict(
                    n=len(sub),
                    pass_gs_default=sum(r['accept_gs_default'] for r in sub),
                    pass_gs_167=sum(r['accept_gs_167'] for r in sub),
                    mean_ba_256=float(np.mean([r['ba_256'] for r in sub])),
                )
            return dict(
                label=label, n_prompts=len(cl),
                clean_gs_default=sum(r['accept_gs_default'] for r in cl),
                gs7_regular_default=sum(r['accept_gs_default'] for r in ed7),
                gs9_all_default=sum(r['accept_gs_default'] for r in ed),
                by_cond=by_c,
            )

        pg_20 = [r for r in pg_recs if r['gidx'] in (list(range(10)) + list(range(50, 60)))]
        pg_sum = dict(
            elapsed_seconds=time.perf_counter() - t0,
            sec_verify_single=sec_verify_pgid_single,
            batch_max_diff=batch_max_diff,
            matched_30=summarize_pg(pg_recs, 'pgid_f_30'),
            matched_20=summarize_pg(pg_20, 'pgid_f_20'),
        )
        write_json(RUN / 'pgid30_summary.json', pg_sum)
        write_json(RUN / 'pgid30_records.json', pg_recs)
        print(f"[S4C PGID30 DONE] {pg_sum['elapsed_seconds']:.1f}s | clean={pg_sum['matched_30']['clean_gs_default']}/30 | gs9={pg_sum['matched_30']['gs9_all_default']}/270 ({100*pg_sum['matched_30']['gs9_all_default']/270:.1f}%) | gs7={pg_sum['matched_30']['gs7_regular_default']}/210", flush=True)

    # =========================================================================
    # PART 3: OFFICIAL PGID-F ON 200 REAL COCO NATURAL VALIDATION IMAGES
    # =========================================================================
    if args.mode in ('pgid_null200', 'all'):
        print('=== [S4C Part 3] Official PGID-F Empirical Natural-Image Null FPR (200 real COCO val images) ===', flush=True)
        t0 = time.perf_counter()
        all_inputs = read_json(NULL_INPUTS)
        val_200 = [x for x in all_inputs if x['split'] == 'validation'][:200]
        assert len(val_200) == 200

        def center_crop_512(p: str) -> Image.Image:
            im = Image.open(p).convert('RGB')
            w, h = im.size
            s = min(w, h)
            return im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((512, 512), Image.BICUBIC)

        bas_256, bas_192 = [], []
        for start in range(0, 200, 5):
            imgs = [center_crop_512(x['path']) for x in val_200[start:start + 5]]
            zTs = invert_pgid_batch(imgs, bs=5)
            for bi in range(len(imgs)):
                sc = score_zT(zTs[bi:bi+1], w_map[0])
                bas_256.append(sc['ba_256'])
                bas_192.append(sc['auth_ba_192'])
            if (start + 5) % 50 == 0:
                print(f'  [PGID Null200] {start+5}/200 ({time.perf_counter()-t0:.1f}s) max_ba256={max(bas_256):.4f}', flush=True)

        a256, a192 = np.array(bas_256), np.array(bas_192)
        np.savez(RUN / 'pgid_null200_scores.npz', ba_256=a256, auth_ba_192=a192)
        fp_def = int((a256 > 181.0 / 256.0).sum())
        null_sum = dict(
            elapsed_seconds=time.perf_counter() - t0,
            n=200,
            ba_256_mean=float(a256.mean()), ba_256_std=float(a256.std()), ba_256_max=float(a256.max()),
            fp_gs_default=fp_def,
            cp95_upper=clopper_pearson_upper(fp_def, 200),
            fp_gs_167=int((a256 >= 167.0 / 256.0).sum()),
        )
        write_json(RUN / 'pgid_null200_summary.json', null_sum)
        print(f"[S4C PGID Null200 DONE] {null_sum['elapsed_seconds']:.1f}s | fp_default={fp_def}/200 | max_ba256={null_sum['ba_256_max']:.4f}", flush=True)

    # =========================================================================
    # PART 4: LIVE FULL-BUDGET IMPRINT (151-step) & REPROMPT+ (108 cands) ON NATIVE GS -> PGID-F
    # =========================================================================
    if args.mode in ('attacks', 'all'):
        sys.path.insert(0, str(SEM_FORGERY))
        from utils import imprint_utils

        print('=== [S4C Part 4A] Live 151-step CVPR 2025 Imprint on Native GS (4 cases) -> Native GS vs PGID-F ===', flush=True)
        t_atk = time.perf_counter()
        pipe_atk, _, inv_sched = imprint_utils.load_attacker_pipe(
            modelid='Manojb/stable-diffusion-2-1-base', scheduler='DDIM', device='cuda'
        )
        pipe_atk.set_progress_bar_config(disable=True)
        diffpipe = imprint_utils.DiffPipe(pipe_atk, scheduler=inv_sched, device=pipe_atk.device)

        imp_dir = RUN / 'imprint_151'
        imp_dir.mkdir(exist_ok=True)
        imp_cases = []
        for i in range(4):
            wm_im = Image.open(GS100_S3 / 'images_unguided' / f'{i:03d}.png').convert('RGB')
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
                    forged_im.save(imp_dir / f'gidx_{i:03d}_step_{step:03d}.png')
                    sn = score_zT(provider.invert_images([forged_im], num_inference_steps=50)['zT_torch'], w_map[i])
                    sp = score_zT(pgid_f.run_defense([forged_im])['zT'], w_map[i])
                    ckpts[f'step_{step}'] = dict(loss=float(loss.detach()), native_gs=sn, pgid_f=sp)
                    print(f'  [S4C Imprint] gidx={i:03d} step={step:3d} | Native_GS={sn["ba_256"]:.4f}({sn["accept_gs_default"]}) | PGID-F={sp["ba_256"]:.4f}({sp["accept_gs_default"]})', flush=True)
            imp_cases.append(dict(gidx=i, cover=f'{i:03d}_cover.png', seconds=time.perf_counter() - t0, checkpoints=ckpts))

        del diffpipe, pipe_atk
        torch.cuda.empty_cache()

        print('=== [S4C Part 4B] Live 9-cand Reprompt+ on Native GS (6 sources x 2 proxies x 9 = 108 cands) -> Native GS vs PGID-F ===', flush=True)
        rp_proto = read_json(ROOT / 'runs/author_reprompt_plus_20261115/protocol.json')
        gs_refs = [r for r in rp_proto['references'] if r['carrier'] == 'GS'][:6]
        rp_results = {}
        for proxy_name, model_ref in [('sd21', 'Manojb/stable-diffusion-2-1-base'), ('sd15', str(SD15_PATH))]:
            proxy_pipe = StableDiffusionPipeline.from_pretrained(model_ref, torch_dtype=torch.float16, local_files_only=True, safety_checker=None).to('cuda')
            fwd_sched = DDIMScheduler.from_config(proxy_pipe.scheduler.config)
            inv_sched_p = DDIMInverseScheduler.from_config(proxy_pipe.scheduler.config)
            proxy_pipe.set_progress_bar_config(disable=True)

            p_rows = []
            rng_wig = np.random.default_rng(2026112531 if proxy_name == 'sd21' else 2026112532)
            for i in range(6):
                wm_im = Image.open(GS100_S3 / 'images_unguided' / f'{i:03d}.png').convert('RGB')
                proxy_pipe.scheduler = inv_sched_p
                with torch.no_grad():
                    img_t = torch.from_numpy(np.asarray(wm_im, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to('cuda', dtype=torch.float16) * 2.0 - 1.0
                    z0_p = proxy_pipe.vae.encode(img_t).latent_dist.mean * proxy_pipe.vae.config.scaling_factor
                    zT_p = proxy_pipe(prompt=[''], latents=z0_p, num_inference_steps=50, guidance_scale=1.0, output_type='latent', return_dict=False)[0]

                proxy_pipe.scheduler = fwd_sched
                prompts3 = [tp['prompt'] for tp in gs_refs[i]['target_prompts']]
                cands, meta = [], []
                for pi, ptxt in enumerate(prompts3):
                    for ri in range(3):
                        lat_t = torch.from_numpy(wiggle_gs_latent(zT_p, rng_wig)).to('cuda', dtype=torch.float16)
                        with torch.no_grad():
                            cands.append(proxy_pipe(prompt=[ptxt], latents=lat_t, num_inference_steps=50, guidance_scale=7.5).images[0].convert('RGB'))
                        meta.append((pi, ri))
                # Batch verify all 9 candidates under BOTH Native GS and Official PGID-F
                z_nats = invert_native_batch(cands, bs=5)
                z_pgs = invert_pgid_batch(cands, bs=5)
                for ci, (pi, ri) in enumerate(meta):
                    p_rows.append(dict(
                        gidx=i, prompt_idx=pi, resample_idx=ri,
                        native_gs=score_zT(z_nats[ci:ci+1], w_map[i]),
                        pgid_f=score_zT(z_pgs[ci:ci+1], w_map[i]),
                    ))
                n_nat = sum(r['native_gs']['accept_gs_default'] for r in p_rows if r['gidx'] == i)
                n_pg = sum(r['pgid_f']['accept_gs_default'] for r in p_rows if r['gidx'] == i)
                print(f'  [S4C Reprompt+ {proxy_name}] gidx={i:03d}: Native_GS={n_nat}/9 | PGID-F={n_pg}/9', flush=True)

            rp_results[proxy_name] = dict(
                n_candidates=len(p_rows), n_sources=6,
                native_gs_cand=sum(r['native_gs']['accept_gs_default'] for r in p_rows),
                native_gs_src=sum(any(r['native_gs']['accept_gs_default'] for r in p_rows if r['gidx'] == g) for g in range(6)),
                pgid_f_cand=sum(r['pgid_f']['accept_gs_default'] for r in p_rows),
                pgid_f_src=sum(any(r['pgid_f']['accept_gs_default'] for r in p_rows if r['gidx'] == g) for g in range(6)),
                mean_pgid_ba256=float(np.mean([r['pgid_f']['ba_256'] for r in p_rows])),
                rows=p_rows,
            )
            del proxy_pipe
            torch.cuda.empty_cache()

        imp_agg = {}
        for ck in ('step_50', 'step_100', 'step_150'):
            imp_agg[ck] = dict(
                n=len(imp_cases),
                native_gs_fooled=sum(c['checkpoints'][ck]['native_gs']['accept_gs_default'] for c in imp_cases),
                pgid_f_fooled=sum(c['checkpoints'][ck]['pgid_f']['accept_gs_default'] for c in imp_cases),
                mean_native_ba=float(np.mean([c['checkpoints'][ck]['native_gs']['ba_256'] for c in imp_cases])),
                mean_pgid_ba=float(np.mean([c['checkpoints'][ck]['pgid_f']['ba_256'] for c in imp_cases])),
            )

        atk_sum = dict(
            elapsed_seconds=time.perf_counter() - t_atk,
            gpu=args.gpu, uuid=uuid,
            imprint_summary=imp_agg,
            imprint_cases=imp_cases,
            reprompt_plus_summary={k: {kk: vv for kk, vv in v.items() if kk != 'rows'} for k, v in rp_results.items()},
        )
        write_json(RUN / 'attacks_summary.json', atk_sum)
        write_json(RUN / 'reprompt_plus_rows.json', rp_results)
        print(f"[S4C Attacks DONE] {atk_sum['elapsed_seconds']:.1f}s | Imprint150 Native={imp_agg['step_150']['native_gs_fooled']}/4 PGID-F={imp_agg['step_150']['pgid_f_fooled']}/4 | RP+_sd21 PGID-F={rp_results['sd21']['pgid_f_cand']}/54 ({rp_results['sd21']['pgid_f_src']}/6) | RP+_sd15 PGID-F={rp_results['sd15']['pgid_f_cand']}/54 ({rp_results['sd15']['pgid_f_src']}/6)", flush=True)

    print(f'[S4C COMPLETE] total elapsed={time.perf_counter()-t_global:.1f}s', flush=True)


if __name__ == '__main__':
    main()
