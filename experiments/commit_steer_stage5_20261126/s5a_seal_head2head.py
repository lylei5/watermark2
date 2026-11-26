"""Stage 5A: Complete Same-Protocol Head-to-Head Baseline Evaluation — SEAL (Paper Secret Salt).
Reuses the EXACT audited genuine SEAL images (runs/unified_seal_clean_20261116) and the EXACT
40,000-natural-image calibrated private salt & frozen working points (runs/unified_seal_nulls_20261115):
  - Paper default: >= 12/1024 blocks (val FPR 0/20,000, CP95 < 0.018%)
  - Target 0.1% FPR: >= 4/1024 blocks (val FPR 10/20,000 = 0.050%)
  - Target 1.0% FPR: >= 2/1024 blocks (val FPR 169/20,000 = 0.845%)

Strict process & venv isolation:
  - `attacks_gen` runs in `semantic_forgery_py310` (NO private SEAL salt loaded!): generates
    Live 151-step CVPR 2025 Imprint PNGs (4 cases x 3 ckpts) + Live CVPR 2025 Reprompt PNGs (6 x 3 x 2 = 36 cands).
  - `gs9_30`, `removal20`, `attacks_verify` run in `seal_reproduction_py310` (exact audited SEAL venv).
"""
import argparse
import hashlib
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

ROOT = Path('/data0/liyanlei/watermark_research')
RUN = ROOT / 'runs/commit_steer_stage5_20261126/seal_head2head'
SEAL_CLEAN_RUN = ROOT / 'runs/unified_seal_clean_20261116'
GS100_S3 = ROOT / 'runs/commit_steer_paper_scale_20261123/gs100_unified'
GS_DIST_PATH = ROOT / 'runs/defense_revalidation_20261111/gs_image_utils.py'
WAVES_ROOT = ROOT / 'third_party/WAVES_1477635_source'
SEM_FORGERY = ROOT / 'third_party/semantic-forgery'
EXTRA_DEPS = ROOT / 'envs/phase1_extra'
COVER_DIR = ROOT / 'runs/published_imprint_dpm_full_20261110/inputs'
SD15_PATH = Path('/data0/liyanlei/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14')

MATCHED_30_GIDX = list(range(0, 15)) + list(range(50, 65))
MATCHED_20_GIDX = list(range(0, 10)) + list(range(50, 60))

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

WAVES_8_TYPES = [
    'rotation', 'resizedcrop', 'erasing', 'brightness',
    'contrast', 'blurring', 'noise', 'compression',
]


def read_json(p: Path):
    return json.loads(p.read_bytes())


def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(p)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


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


def gidx_to_seal_idx(gidx: int) -> int:
    return gidx if gidx < 50 else 200 + gidx


def genuine_path(gidx: int) -> Path:
    si = gidx_to_seal_idx(gidx)
    return SEAL_CLEAN_RUN / f'shard_{si % 3}/images/{si:03d}_genuine.png'


def run_attacks_gen(gpu: int):
    """Runs in semantic_forgery_py310 with ZERO access to SEAL private salt."""
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
    from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sys.path.insert(0, str(SEM_FORGERY))
    from utils import imprint_utils

    t_all = time.perf_counter()
    print('=== [S5A attacks_gen Part 1] Live 151-step CVPR 2025 Imprint on SEAL (4 cases, isolated attacker venv) ===', flush=True)
    pipe_atk, _, inv_sched = imprint_utils.load_attacker_pipe(
        modelid='Manojb/stable-diffusion-2-1-base', scheduler='DDIM', device='cuda'
    )
    pipe_atk.set_progress_bar_config(disable=True)
    diffpipe = imprint_utils.DiffPipe(pipe_atk, scheduler=inv_sched, device=pipe_atk.device)

    imp_dir = RUN / 'imprint_151'
    imp_dir.mkdir(exist_ok=True)
    imp_meta = {}
    for i in range(4):
        wm_im = Image.open(genuine_path(i)).convert('RGB')
        cover_im = Image.open(COVER_DIR / f'{i:03d}_cover.png').convert('RGB').resize((512, 512), Image.BICUBIC)
        with torch.no_grad():
            zT_target = imprint_utils.invert_image(pipe=pipe_atk, scheduler=inv_sched, image=wm_im, num_inference_steps=50)
            z0_orig = imprint_utils.pixel_to_latent(cover_im, pipe_atk)
        z0 = torch.nn.Parameter(z0_orig.detach().clone())
        opt = torch.optim.Adam([z0], lr=0.01)
        t0 = time.perf_counter()
        losses = {}
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
                losses[f'step_{step}'] = float(loss.detach())
        imp_meta[str(i)] = dict(seconds=time.perf_counter() - t0, losses=losses)
        print(f'  [SEAL Imprint gen] gidx={i:03d} 151 steps done in {imp_meta[str(i)]["seconds"]:.0f}s (loss150={losses["step_150"]:.4f})', flush=True)
    del diffpipe, pipe_atk
    torch.cuda.empty_cache()

    print('=== [S5A attacks_gen Part 2] Live CVPR 2025 Reprompt on SEAL (6 sources x 3 I2P x 2 proxies = 36 cands) ===', flush=True)
    rp_proto = read_json(ROOT / 'runs/author_reprompt_plus_20261115/protocol.json')
    gs_refs = [r for r in rp_proto['references'] if r['carrier'] == 'GS'][:6]
    rp_dir = RUN / 'reprompt_cands'
    rp_dir.mkdir(exist_ok=True)
    for proxy_name, model_ref in [('sd21', 'Manojb/stable-diffusion-2-1-base'), ('sd15', str(SD15_PATH))]:
        proxy_pipe = StableDiffusionPipeline.from_pretrained(
            model_ref, torch_dtype=torch.float16, local_files_only=True, safety_checker=None
        ).to('cuda')
        fwd_sched = DDIMScheduler.from_config(proxy_pipe.scheduler.config)
        inv_sched_p = DDIMInverseScheduler.from_config(proxy_pipe.scheduler.config)
        proxy_pipe.set_progress_bar_config(disable=True)
        for i in range(6):
            wm_im = Image.open(genuine_path(i)).convert('RGB')
            proxy_pipe.scheduler = inv_sched_p
            with torch.no_grad():
                img_t = torch.from_numpy(np.asarray(wm_im, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to('cuda', dtype=torch.float16) * 2.0 - 1.0
                z0_p = proxy_pipe.vae.encode(img_t).latent_dist.mean * proxy_pipe.vae.config.scaling_factor
                zT_p = proxy_pipe(prompt=[''], latents=z0_p, num_inference_steps=50, guidance_scale=1.0, output_type='latent', return_dict=False)[0]
            proxy_pipe.scheduler = fwd_sched
            prompts3 = [tp['prompt'] for tp in gs_refs[i]['target_prompts']]
            for pi, ptxt in enumerate(prompts3):
                with torch.no_grad():
                    cim = proxy_pipe(prompt=[ptxt], latents=zT_p, num_inference_steps=50, guidance_scale=7.5).images[0].convert('RGB')
                cim.save(rp_dir / f'{proxy_name}_g{i}_p{pi}.png')
        del proxy_pipe
        torch.cuda.empty_cache()
        print(f'  [SEAL Reprompt gen] {proxy_name} 18/18 done', flush=True)

    write_json(RUN / 'attacks_gen_meta.json', dict(
        elapsed_seconds=time.perf_counter() - t_all,
        gpu=gpu, uuid=uuid, imprint=imp_meta,
    ))
    print(f'[S5A attacks_gen DONE] total={time.perf_counter()-t_all:.1f}s', flush=True)


def run_seal_venv_modes(mode: str, gpu: int):
    """Runs in seal_reproduction_py310 (exact audited SEAL environment)."""
    uuid = check_idle_gpu(gpu)
    RUN.mkdir(parents=True, exist_ok=True)
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
        OPENBLAS_NUM_THREADS='4',
        WANDB_MODE='disabled',
        TOKENIZERS_PARALLELISM='false',
    )

    seal_plan = read_json(SEAL_CLEAN_RUN / 'protocol.json')
    cfg = seal_plan['detector']
    models = read_json(SEAL_CLEAN_RUN / 'model_manifest.json')
    fwp = read_json(SEAL_CLEAN_RUN / 'frozen_working_points.json')
    assert sha256_file(Path(cfg['private_salt_file'])) == cfg['private_salt_sha256']

    gs_sources = read_json(GS100_S3 / 'protocol.json')['sources']
    seal_sources_by_idx = {s['index']: s for s in seal_plan['sources']}
    for g in MATCHED_30_GIDX:
        si = gidx_to_seal_idx(g)
        assert seal_sources_by_idx[si]['prompt'] == gs_sources[g]['prompt'], f'Prompt mismatch at gidx={g}, seal_idx={si}'

    import torch
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    from sentence_transformers import SentenceTransformer
    from diffusers import StableDiffusionImg2ImgPipeline

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    official_seal = Path(cfg['source'])
    sys.path.insert(0, str(official_seal))
    sys.path.insert(0, str(SEAL_CLEAN_RUN))
    sys.path.insert(0, str(EXTRA_DEPS))

    from inverse_stable_diffusion import InversableStableDiffusionPipeline
    from utils import generate_caption, transform_img, calculate_patch_l2
    from seal_paper_simhash import SealPaperSimHash

    sys.modules.pop('utils', None)
    sys.path = [p for p in sys.path if p != str(official_seal)]

    gs_dist = load_module('gs_dist', GS_DIST_PATH)

    waves_utils_mod = load_module('waves_img_utils', WAVES_ROOT / 'utils/image_utils.py')
    waves_exp = load_module('waves_exp_utils', WAVES_ROOT / 'utils/exp_utils.py')
    sys.modules['utils'] = SimpleNamespace(
        set_random_seed=waves_exp.set_random_seed,
        to_tensor=waves_utils_mod.to_tensor,
        to_pil=waves_utils_mod.to_pil,
    )
    waves_dist = load_module('waves_distortions', WAVES_ROOT / 'distortions/distortions.py')
    sys.modules.pop('utils', None)

    pipe = InversableStableDiffusionPipeline.from_pretrained(
        models['diffusion_model']['root'], torch_dtype=torch.float16, local_files_only=True
    ).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    assert type(pipe.scheduler).__name__ == cfg['expected_scheduler']
    processor = Blip2Processor.from_pretrained(models['caption_model']['root'], local_files_only=True)
    captioner = Blip2ForConditionalGeneration.from_pretrained(
        models['caption_model']['root'], torch_dtype=torch.float16, local_files_only=True
    ).to('cuda')
    encoder = SentenceTransformer(models['embedding_model']['root'], device='cuda')
    sampler = SealPaperSimHash(bytes.fromhex(read_json(Path(cfg['private_salt_file']))['salt']))

    def seal_models_to(dev: str):
        pipe.to(dev)
        captioner.to(dev)
        encoder.to(dev)
        if dev == 'cpu':
            torch.cuda.empty_cache()

    def seal_verify(im: Image.Image):
        im_rgb = im.convert('RGB')
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        caption = generate_caption(im_rgb, processor, captioner)
        with torch.no_grad():
            emb = encoder.encode(caption, convert_to_tensor=True).to('cuda')
            ref = sampler.sample(emb / torch.norm(emb)).to(dtype=pipe.vae.dtype)
            tensor = transform_img(im_rgb).unsqueeze(0).to('cuda', dtype=pipe.vae.dtype)
            latent = pipe.get_image_latents(tensor, sample=False)
            rec = pipe.forward_diffusion(
                latents=latent,
                text_embeddings=pipe.get_text_embedding(''),
                guidance_scale=cfg['inversion_cfg'],
                num_inference_steps=cfg['inference_steps'],
            )
            dists = np.asarray(calculate_patch_l2(rec, ref, 1024), dtype=np.float64)
        torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        matches = int((dists < 2.3).sum())
        return dict(
            matches=matches,
            accept_12=bool(matches >= 12),
            accept_4=bool(matches >= 4),
            accept_2=bool(matches >= 2),
            caption=caption,
            sec_verify=sec,
        )

    write_json(RUN / 'protocol.json', dict(
        run_name='seal_head2head',
        method='Native SEAL paper-salt (SEAL_92d31b3 + seal_paper_simhash.py + 40k-calibrated frozen working points)',
        reused_clean_run=str(SEAL_CLEAN_RUN),
        calibration_protocol_sha256=fwp['calibration_protocol_sha256'],
        frozen_thresholds=dict(paper_12=12, fpr_01pct=4, fpr_1pct=2, block_l2_cutoff=2.3),
        matched_30_gidx=MATCHED_30_GIDX,
        matched_20_gidx=MATCHED_20_GIDX,
        gs9_conditions=GS_9_CONDITIONS,
        waves8_types=WAVES_8_TYPES,
        waves_relative_strength=0.5,
        regen_conditions=['vae_roundtrip', 'sd21_regen_01', 'sd21_regen_02', 'sd21_regen_03'],
    ))

    # 1. GS9-30
    if mode in ('gs9_30', 'benign_all'):
        print('=== [S5A Part 1] SEAL Matched-30 Clean + 9 Standard GS Distortions (300 verifs) ===', flush=True)
        t0 = time.perf_counter()
        conds = ['clean', *GS_9_CONDITIONS.keys()]
        recs = []
        for pi, gidx in enumerate(MATCHED_30_GIDX):
            si = gidx_to_seal_idx(gidx)
            im = Image.open(genuine_path(gidx)).convert('RGB')
            for ci, cond in enumerate(conds):
                if cond == 'clean':
                    q_im = im
                else:
                    seed = 2026112300 + gidx * 100 + ci
                    gs_dist.set_random_seed(seed)
                    dargs = SimpleNamespace(**{k: None for k in GS_9_CONDITIONS})
                    setattr(dargs, cond, GS_9_CONDITIONS[cond])
                    q_im = gs_dist.image_distortion(im.copy(), seed, dargs).convert('RGB')
                v = seal_verify(q_im)
                recs.append(dict(gidx=gidx, seal_idx=si, dataset=gs_sources[gidx]['dataset'], condition=cond, **v))
            if (pi + 1) % 5 == 0:
                cl_ok = sum(r['accept_12'] for r in recs if r['condition'] == 'clean')
                ed_ok = sum(r['accept_12'] for r in recs if r['condition'] != 'clean')
                print(f'  [SEAL GS9-30] {pi+1}/30 ({time.perf_counter()-t0:.0f}s) | clean@12={cl_ok}/{pi+1} | 9-GS@12={ed_ok}/{(pi+1)*9}', flush=True)

        def summarize_gs9(sub_recs, label):
            cl = [r for r in sub_recs if r['condition'] == 'clean']
            ed = [r for r in sub_recs if r['condition'] != 'clean']
            ed7 = [r for r in ed if r['condition'] not in ('random_crop_ratio', 'random_drop_ratio')]
            by_c = {}
            for cond in conds:
                s = [r for r in sub_recs if r['condition'] == cond]
                by_c[cond] = dict(
                    n=len(s),
                    pass_12=sum(r['accept_12'] for r in s),
                    pass_4=sum(r['accept_4'] for r in s),
                    pass_2=sum(r['accept_2'] for r in s),
                    mean_matches=float(np.mean([r['matches'] for r in s])),
                )
            return dict(
                label=label, n_prompts=len(cl),
                clean_12=sum(r['accept_12'] for r in cl),
                clean_4=sum(r['accept_4'] for r in cl),
                clean_2=sum(r['accept_2'] for r in cl),
                gs7_12=sum(r['accept_12'] for r in ed7),
                gs7_4=sum(r['accept_4'] for r in ed7),
                gs7_2=sum(r['accept_2'] for r in ed7),
                gs9_12=sum(r['accept_12'] for r in ed),
                gs9_4=sum(r['accept_4'] for r in ed),
                gs9_2=sum(r['accept_2'] for r in ed),
                sec_verify_mean=float(np.mean([r['sec_verify'] for r in sub_recs])),
                by_cond=by_c,
            )

        recs_20 = [r for r in recs if r['gidx'] in MATCHED_20_GIDX]
        summ = dict(
            elapsed_seconds=time.perf_counter() - t0,
            gpu=gpu, uuid=uuid,
            matched_30=summarize_gs9(recs, 'seal_gs9_30'),
            matched_20=summarize_gs9(recs_20, 'seal_gs9_20'),
        )
        write_json(RUN / 'gs9_30_summary.json', summ)
        write_json(RUN / 'gs9_30_records.json', recs)
        m30 = summ['matched_30']
        print(f"[S5A GS9-30 DONE] {summ['elapsed_seconds']:.1f}s | clean@12/4/2={m30['clean_12']}/{m30['clean_4']}/{m30['clean_2']} (/30) | gs9@12/4/2={m30['gs9_12']}/{m30['gs9_4']}/{m30['gs9_2']} (/270) | verify={m30['sec_verify_mean']:.2f}s/img", flush=True)

    # 2. Removal20
    if mode in ('removal20', 'benign_all'):
        print('=== [S5A Part 2] SEAL Matched-20 Official WAVES-8 (0.5) + Regeneration Removal (240 verifs) ===', flush=True)
        t0 = time.perf_counter()
        seal_models_to('cpu')
        img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
            models['diffusion_model']['root'], torch_dtype=torch.float16, local_files_only=True, safety_checker=None
        ).to('cuda')
        img2img.set_progress_bar_config(disable=True)

        regen_cache = {}
        for gidx in MATCHED_20_GIDX:
            im = Image.open(genuine_path(gidx)).convert('RGB')
            with torch.no_grad():
                x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to('cuda', dtype=torch.float16)
                z0 = img2img.vae.encode(x).latent_dist.mean
                dec = img2img.vae.decode(z0, return_dict=False)[0]
                regen_cache[(gidx, 'vae_roundtrip')] = Image.fromarray(((dec.float()[0].permute(1, 2, 0) / 2.0 + 0.5).clamp(0, 1).mul(255).round().byte().cpu().numpy()))
            for ri, (cname, rstr) in enumerate([('sd21_regen_01', 0.1), ('sd21_regen_02', 0.2), ('sd21_regen_03', 0.3)]):
                gen = torch.Generator(device='cuda').manual_seed(2026112650 + gidx * 10 + ri)
                with torch.no_grad():
                    regen_cache[(gidx, cname)] = img2img(prompt='', image=im, strength=rstr, num_inference_steps=50, guidance_scale=7.5, generator=gen).images[0].convert('RGB')
        del img2img
        torch.cuda.empty_cache()
        seal_models_to('cuda')

        rem_conds = [*[f'waves_{w}' for w in WAVES_8_TYPES], 'vae_roundtrip', 'sd21_regen_01', 'sd21_regen_02', 'sd21_regen_03']
        rem_recs = []
        for pi, gidx in enumerate(MATCHED_20_GIDX):
            si = gidx_to_seal_idx(gidx)
            im = Image.open(genuine_path(gidx)).convert('RGB')
            for ci, wtype in enumerate(WAVES_8_TYPES):
                abs_s = waves_dist.relative_strength_to_absolute(0.5, wtype)
                q_im = waves_dist.apply_single_distortion(im.copy(), wtype, strength=abs_s, distortion_seed=2026112600 + gidx * 100 + ci).convert('RGB')
                v = seal_verify(q_im)
                rem_recs.append(dict(gidx=gidx, seal_idx=si, dataset=gs_sources[gidx]['dataset'], condition=f'waves_{wtype}', **v))
            for cname in ('vae_roundtrip', 'sd21_regen_01', 'sd21_regen_02', 'sd21_regen_03'):
                v = seal_verify(regen_cache[(gidx, cname)])
                rem_recs.append(dict(gidx=gidx, seal_idx=si, dataset=gs_sources[gidx]['dataset'], condition=cname, **v))
            if (pi + 1) % 5 == 0:
                print(f'  [SEAL Removal20] {pi+1}/20 ({time.perf_counter()-t0:.0f}s)', flush=True)

        by_c = {}
        for cond in rem_conds:
            s = [r for r in rem_recs if r['condition'] == cond]
            by_c[cond] = dict(
                n=len(s),
                pass_12=sum(r['accept_12'] for r in s),
                pass_4=sum(r['accept_4'] for r in s),
                pass_2=sum(r['accept_2'] for r in s),
                mean_matches=float(np.mean([r['matches'] for r in s])),
            )
        w_sub = [r for r in rem_recs if r['condition'].startswith('waves_')]
        r_sub = [r for r in rem_recs if not r['condition'].startswith('waves_')]
        rem_sum = dict(
            elapsed_seconds=time.perf_counter() - t0,
            n_prompts=20,
            waves8_total=len(w_sub),
            waves8_pass_12=sum(r['accept_12'] for r in w_sub),
            waves8_pass_4=sum(r['accept_4'] for r in w_sub),
            waves8_pass_2=sum(r['accept_2'] for r in w_sub),
            regen4_total=len(r_sub),
            regen4_pass_12=sum(r['accept_12'] for r in r_sub),
            regen4_pass_4=sum(r['accept_4'] for r in r_sub),
            regen4_pass_2=sum(r['accept_2'] for r in r_sub),
            by_cond=by_c,
        )
        write_json(RUN / 'removal20_summary.json', rem_sum)
        write_json(RUN / 'removal20_records.json', rem_recs)
        print(f"[S5A Removal20 DONE] {rem_sum['elapsed_seconds']:.1f}s | WAVES8@12/4/2={rem_sum['waves8_pass_12']}/{rem_sum['waves8_pass_4']}/{rem_sum['waves8_pass_2']} (/160) | Regen4@12/4/2={rem_sum['regen4_pass_12']}/{rem_sum['regen4_pass_4']}/{rem_sum['regen4_pass_2']} (/80)", flush=True)

    # 3. Attacks verify (Imprint 12 PNGs + Reprompt 36 PNGs + Official ICCV25 SEAL regeneration.py 18 cands)
    if mode == 'attacks_verify':
        print('=== [S5A Part 3] Verifying Imprint151 (12) + Reprompt (36) + Official SEAL regeneration.py (18) ===', flush=True)
        t_atk = time.perf_counter()
        gen_meta = read_json(RUN / 'attacks_gen_meta.json')
        imp_dir = RUN / 'imprint_151'
        rp_dir = RUN / 'reprompt_cands'

        imp_cases = []
        for i in range(4):
            ckpts = {}
            for step in (50, 100, 150):
                v = seal_verify(Image.open(imp_dir / f'gidx_{i:03d}_step_{step:03d}.png').convert('RGB'))
                ckpts[f'step_{step}'] = dict(loss=gen_meta['imprint'][str(i)]['losses'][f'step_{step}'], **v)
                print(f'  [SEAL Imprint] gidx={i:03d} step={step:3d} | matches={v["matches"]:4d} (@12={v["accept_12"]} @4={v["accept_4"]} @2={v["accept_2"]})', flush=True)
            imp_cases.append(dict(gidx=i, seal_idx=gidx_to_seal_idx(i), cover=f'{i:03d}_cover.png', checkpoints=ckpts))

        rp_results = {}
        for proxy_name in ('sd21', 'sd15'):
            rows = []
            for i in range(6):
                for pi in range(3):
                    v = seal_verify(Image.open(rp_dir / f'{proxy_name}_g{i}_p{pi}.png').convert('RGB'))
                    rows.append(dict(gidx=i, seal_idx=gidx_to_seal_idx(i), prompt_idx=pi, **v))
            rp_results[proxy_name] = dict(
                n_candidates=len(rows), n_sources=6,
                fooled_12_cand=sum(r['accept_12'] for r in rows),
                fooled_4_cand=sum(r['accept_4'] for r in rows),
                fooled_2_cand=sum(r['accept_2'] for r in rows),
                fooled_12_src=sum(any(r['accept_12'] for r in rows if r['gidx'] == g) for g in range(6)),
                fooled_4_src=sum(any(r['accept_4'] for r in rows if r['gidx'] == g) for g in range(6)),
                fooled_2_src=sum(any(r['accept_2'] for r in rows if r['gidx'] == g) for g in range(6)),
                mean_matches=float(np.mean([r['matches'] for r in rows])),
                rows=rows,
            )
            print(f"  [SEAL Reprompt {proxy_name}] @12={rp_results[proxy_name]['fooled_12_cand']}/18 @4={rp_results[proxy_name]['fooled_4_cand']}/18 @2={rp_results[proxy_name]['fooled_2_cand']}/18", flush=True)

        rp_proto = read_json(ROOT / 'runs/author_reprompt_plus_20261115/protocol.json')
        gs_refs = [r for r in rp_proto['references'] if r['carrier'] == 'GS'][:6]
        regen_rows = []
        for i in range(6):
            wm_im = Image.open(genuine_path(i)).convert('RGB')
            with torch.no_grad():
                t_org = transform_img(wm_im).unsqueeze(0).to('cuda', dtype=pipe.vae.dtype)
                lat_org = pipe.get_image_latents(t_org, sample=False)
                recon_org = pipe.forward_diffusion(
                    latents=lat_org, text_embeddings=pipe.get_text_embedding(''),
                    guidance_scale=1, num_inference_steps=50,
                )
            prompts3 = [tp['prompt'] for tp in gs_refs[i]['target_prompts']]
            for pi, ptxt in enumerate(prompts3):
                with torch.no_grad():
                    sec_im = pipe(ptxt, latents=recon_org, num_inference_steps=50, guidance_scale=7.5).images[0].convert('RGB')
                v = seal_verify(sec_im)
                regen_rows.append(dict(gidx=i, seal_idx=gidx_to_seal_idx(i), prompt_idx=pi, **v))
        seal_regen_summary = dict(
            scope='ICCV 2025 SEAL regeneration.py (target-model weight access; distinct from proxy-only Reprompt)',
            n_candidates=len(regen_rows), n_sources=6,
            fooled_12_cand=sum(r['accept_12'] for r in regen_rows),
            fooled_4_cand=sum(r['accept_4'] for r in regen_rows),
            fooled_2_cand=sum(r['accept_2'] for r in regen_rows),
            fooled_12_src=sum(any(r['accept_12'] for r in regen_rows if r['gidx'] == g) for g in range(6)),
            fooled_4_src=sum(any(r['accept_4'] for r in regen_rows if r['gidx'] == g) for g in range(6)),
            fooled_2_src=sum(any(r['accept_2'] for r in regen_rows if r['gidx'] == g) for g in range(6)),
            mean_matches=float(np.mean([r['matches'] for r in regen_rows])),
        )
        print(f"  [SEAL Target Regen.py] @12={seal_regen_summary['fooled_12_cand']}/18 @4={seal_regen_summary['fooled_4_cand']}/18 @2={seal_regen_summary['fooled_2_cand']}/18", flush=True)

        imp_agg = {}
        for ck in ('step_50', 'step_100', 'step_150'):
            imp_agg[ck] = dict(
                n=4,
                fooled_12=sum(c['checkpoints'][ck]['accept_12'] for c in imp_cases),
                fooled_4=sum(c['checkpoints'][ck]['accept_4'] for c in imp_cases),
                fooled_2=sum(c['checkpoints'][ck]['accept_2'] for c in imp_cases),
                mean_matches=float(np.mean([c['checkpoints'][ck]['matches'] for c in imp_cases])),
            )

        atk_sum = dict(
            elapsed_seconds=time.perf_counter() - t_atk + gen_meta['elapsed_seconds'],
            gpu=gpu, uuid=uuid,
            imprint_summary=imp_agg,
            imprint_cases=imp_cases,
            reprompt_summary={k: {kk: vv for kk, vv in v.items() if kk != 'rows'} for k, v in rp_results.items()},
            seal_target_regeneration_summary=seal_regen_summary,
        )
        write_json(RUN / 'attacks_summary.json', atk_sum)
        write_json(RUN / 'attacks_rows.json', dict(reprompt=rp_results, seal_target_regeneration=regen_rows))
        print(f"[S5A Attacks DONE] Imprint150@12/4/2={imp_agg['step_150']['fooled_12']}/{imp_agg['step_150']['fooled_4']}/{imp_agg['step_150']['fooled_2']} (/4)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['gs9_30', 'removal20', 'benign_all', 'attacks_gen', 'attacks_verify'])
    ap.add_argument('--gpu', type=int, default=3)
    args = ap.parse_args()

    if args.mode == 'attacks_gen':
        run_attacks_gen(args.gpu)
    else:
        run_seal_venv_modes(args.mode, args.gpu)


if __name__ == '__main__':
    main()
