"""Stage 5C: Paper-Faithful SemBind GS-S — Same-Prompt Instance-Level Forgery Stress Test
+ Sharded Matched-20 Official WAVES-8 & Regeneration Removal Suite.

Reuses EXACT frozen Stage 4D SemBind GS-S private material, aux_codes, and generated bound/same images
(runs/commit_steer_stage4_20261125/head2head_sembind).

Subcommands:
  instance_forgery  - Unmodified CVPR 2025 151-step Imprint onto frozen independent same-prompt sibling cover
                      (000_bound..003_bound -> 000_same..003_same, checkpoints 50/100/150)
                    + Unmodified CVPR 2025 Same-Prompt Reprompt+ (4 sources x 3 wiggles x 2 proxies = 24 cands
                      using each source's own original prompt), verified by full SemBind DPM50+VAE100.
  removal20         - Matched-20 (10 COCO + 10 SDP) x 12 removal conditions (8 WAVES str=0.5 + VAE + SD2.1 Img2Img
                      0.1/0.2/0.3 = 240 full SemBind DPM50+VAE100 verifications; supports --shard/--num-shards).
  merge_removal     - Merge sharded removal20 records into removal20_records.json & removal20_summary.json.
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
from scipy.stats import norm

ROOT = Path('/data0/liyanlei/watermark_research')
RUN = ROOT / 'runs/commit_steer_stage5_20261126/sembind_instance_and_removal'
SEMBIND_S4 = ROOT / 'runs/commit_steer_stage4_20261125/head2head_sembind'
PHASE1_DIR = ROOT / 'runs/phase1_trajectory_native_gs_20261113'
GS_MOD_PATH = PHASE1_DIR / 'gs_watermark.py'
SEMBIND_OFFICIAL = ROOT / 'third_party/frozen/SemBind_337c9f9'
SEMBIND_SCHED = ROOT / 'runs/sembind_chain_revalidation_20261111/scheduling_dpmsolver_multistep.py'
SD21_SNAP = ROOT / 'cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/snapshots/0094d483a120f3f33dafbd187ea4aa60d10de75c'
SEMBIND_MODELS = ROOT / 'models/sembind'
WAVES_ROOT = ROOT / 'third_party/WAVES_1477635_source'
SEM_FORGERY = ROOT / 'third_party/semantic-forgery'
EXTRA_DEPS = ROOT / 'envs/phase1_extra'
SD15_PATH = Path('/data0/liyanlei/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14')

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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
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


def wiggle_gs_latent(latents_t, rng):
    arr = latents_t.detach().float().cpu().numpy()
    y = np.clip((norm.cdf(arr) * 2.0).astype(np.int32), 0, 1)
    u = rng.uniform(1e-6, 1.0 - 1e-6, size=y.shape).astype(np.float64)
    return norm.ppf((u + y) / 2.0)


def load_waves_distortions():
    waves_img = load_module('waves_image_utils', WAVES_ROOT / 'utils/image_utils.py')
    waves_exp = load_module('waves_exp_utils', WAVES_ROOT / 'utils/exp_utils.py')
    orig_utils = sys.modules.pop('utils', None)
    sys.modules['utils'] = SimpleNamespace(
        set_random_seed=waves_exp.set_random_seed,
        to_tensor=waves_img.to_tensor,
        to_pil=waves_img.to_pil,
    )
    mod = load_module('waves_distortions', WAVES_ROOT / 'distortions/distortions.py')
    if orig_utils is not None:
        sys.modules['utils'] = orig_utils
    else:
        sys.modules.pop('utils', None)
    return mod


def init_sembind_env(uuid: str, work_sub: str):
    work_dir = RUN / work_sub
    work_dir.mkdir(parents=True, exist_ok=True)
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
    sys.path.insert(0, str(EXTRA_DEPS))
    sys.path.insert(0, str(PHASE1_DIR))
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    load_module('model', SEMBIND_OFFICIAL / 'model.py')
    from sembind_official_audit import load_official
    from semantic_code import SemanticCodeExtractor

    paths = SimpleNamespace(
        official=SEMBIND_OFFICIAL,
        scheduler=SEMBIND_SCHED,
        model=SD21_SNAP,
        semantic=SEMBIND_MODELS,
    )
    inversion, pipe = load_official(paths)
    pipe.set_progress_bar_config(disable=True)
    extractor = SemanticCodeExtractor(SEMBIND_MODELS / 'dinov2-giant', SEMBIND_MODELS / 'sembind_masker.pth', device='cpu')

    def extract_code(im: Image.Image) -> np.ndarray:
        extractor.dino.to('cuda'); extractor.model.to('cuda'); extractor.device = 'cuda:0'
        bits = extractor.code(im).cpu().numpy().astype(np.uint8)
        extractor.dino.to('cpu'); extractor.model.to('cpu'); extractor.device = 'cpu'
        torch.cuda.empty_cache()
        return bits

    return torch, F, inversion, pipe, extract_code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['instance_forgery', 'removal20', 'merge_removal'])
    ap.add_argument('--gpu', type=int, default=4)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=2)
    args = ap.parse_args()

    RUN.mkdir(parents=True, exist_ok=True)
    s4_proto = read_json(SEMBIND_S4 / 'protocol.json')

    if args.mode == 'merge_removal':
        recs = []
        for p in sorted(RUN.glob('removal_shard*.json')):
            recs.extend(read_json(p))
        assert len(recs) == 240, f'Expected 240 SemBind removal records, got {len(recs)}'
        rem_conds = [*[f'waves_{w}' for w in WAVES_8_TYPES], 'vae_roundtrip', 'sd21_regen_01', 'sd21_regen_02', 'sd21_regen_03']
        by_c = {}
        for cond in rem_conds:
            sub = [r for r in recs if r['condition'] == cond]
            by_c[cond] = dict(
                n=len(sub),
                pass_gs_default=sum(r['accept_gs_default'] for r in sub),
                pass_gs_167=sum(r['accept_gs_167'] for r in sub),
                mean_ba=float(np.mean([r['ba'] for r in sub])),
                mean_hamming=float(np.mean([r['code_hamming_vs_aux'] for r in sub])),
                mean_sec_verify=float(np.mean([r['sec_verify'] for r in sub])),
            )
        w_sub = [r for r in recs if r['condition'].startswith('waves_')]
        r_sub = [r for r in recs if not r['condition'].startswith('waves_')]
        summ = dict(
            n_prompts=20,
            waves8_pass_default=sum(r['accept_gs_default'] for r in w_sub),
            waves8_pass_167=sum(r['accept_gs_167'] for r in w_sub),
            waves8_total=len(w_sub),
            regen4_pass_default=sum(r['accept_gs_default'] for r in r_sub),
            regen4_pass_167=sum(r['accept_gs_167'] for r in r_sub),
            regen4_total=len(r_sub),
            sec_verify_mean=float(np.mean([r['sec_verify'] for r in recs])),
            by_cond=by_c,
        )
        write_json(RUN / 'removal20_summary.json', summ)
        write_json(RUN / 'removal20_records.json', recs)
        print(f"[S5C SemBind Removal20 MERGED] WAVES8={summ['waves8_pass_default']}/160 (167: {summ['waves8_pass_167']}/160) | Regen4={summ['regen4_pass_default']}/80 (167: {summ['regen4_pass_167']}/80)", flush=True)
        return

    uuid = check_idle_gpu(args.gpu)
    priv = np.load(SEMBIND_S4 / 'private_material.npz')
    gs_key_np, wm_msg_np, permutation = priv['key'], priv['message'], priv['permutation']
    aux_codes = np.load(SEMBIND_S4 / 'aux_codes.npz')

    torch, F, inversion, pipe, extract_code = init_sembind_env(uuid, f'work_{args.mode}_s{args.shard}')
    native_gs = load_module('native_gs', GS_MOD_PATH).Gaussian_Shading(1, 8, 1e-6, 1)
    native_gs.key = torch.from_numpy(gs_key_np.astype(np.int64)).cuda()
    native_gs.watermark = torch.from_numpy(wm_msg_np.astype(np.int64)).cuda()

    def sembind_verify(im: Image.Image, seed_val: int):
        t0 = time.perf_counter()
        q_code = extract_code(im)
        pipe.to('cuda')
        torch.manual_seed(seed_val)
        z = inversion.exact_inversion(
            im, prompt='', guidance_scale=3.0, num_inference_steps=50,
            test_num_inference_steps=50, inv_order=0, decoder_inv=True, pipe=pipe,
        ).detach()
        mask = np.tile(q_code, 16)[permutation].reshape(1, 4, 64, 64)
        unmasked = z * (1.0 - 2.0 * torch.tensor(mask, dtype=torch.float32, device='cuda'))
        ba = float(native_gs.eval_watermark(unmasked))
        sec = time.perf_counter() - t0
        return dict(
            ba=ba,
            accept_gs_default=bool(ba > 181.0 / 256.0),
            accept_gs_167=bool(ba >= 167.0 / 256.0),
            sec_verify=sec,
            q_code=q_code,
        )

    write_json(RUN / 'protocol.json', dict(
        run_name='sembind_instance_and_removal',
        reused_stage4d=str(SEMBIND_S4),
        same_prompt_covers='runs/commit_steer_stage4_20261125/head2head_sembind/images/{000..003}_same.png',
        waves8_types=WAVES_8_TYPES,
        waves_relative_strength=0.5,
        regen_conditions=['vae_roundtrip', 'sd21_regen_01', 'sd21_regen_02', 'sd21_regen_03'],
    ))

    # =========================================================================
    # INSTANCE FORGERY: Same-Prompt 151-step Imprint (4) + Same-Prompt Reprompt+ (24)
    # =========================================================================
    if args.mode == 'instance_forgery':
        sys.path.insert(0, str(SEM_FORGERY))
        from utils import imprint_utils
        import lpips
        from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler

        t_all = time.perf_counter()
        print('=== [S5C Part 1A] SemBind GS-S Same-Prompt 151-step Imprint (000..003_bound -> 000..003_same) ===', flush=True)
        pipe.to('cpu'); torch.cuda.empty_cache()
        pipe_atk, _, inv_sched = imprint_utils.load_attacker_pipe(
            modelid='Manojb/stable-diffusion-2-1-base', scheduler='DPM', device='cuda'
        )
        pipe_atk.set_progress_bar_config(disable=True)
        diffpipe = imprint_utils.DiffPipe(pipe_atk, scheduler=inv_sched, device=pipe_atk.device)

        imp_dir = RUN / 'same_prompt_imprint_151'
        imp_dir.mkdir(exist_ok=True)
        imp_imgs = {}
        imp_losses = {}
        for i in range(4):
            wm_im = Image.open(SEMBIND_S4 / f'images/{i:03d}_bound.png').convert('RGB')
            cover_im = Image.open(SEMBIND_S4 / f'images/{i:03d}_same.png').convert('RGB').resize((512, 512), Image.BICUBIC)
            with torch.no_grad():
                zT_target = imprint_utils.invert_image(pipe=pipe_atk, scheduler=inv_sched, image=wm_im, num_inference_steps=50)
                z0_orig = imprint_utils.pixel_to_latent(cover_im, pipe_atk)
            z0 = torch.nn.Parameter(z0_orig.detach().clone())
            opt = torch.optim.Adam([z0], lr=0.01)
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
                    p_save = imp_dir / f'gidx_{i:03d}_step_{step:03d}.png'
                    forged_im.save(p_save)
                    imp_imgs[(i, step)] = p_save
                    imp_losses[(i, step)] = float(loss.detach())
            print(f'  [SemBind Same-Prompt Imprint opt] gidx={i:03d} done in {time.perf_counter()-t0:.0f}s', flush=True)
        del diffpipe, pipe_atk
        torch.cuda.empty_cache()

        lpips_fn = lpips.LPIPS(net='alex').cuda().eval()
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

        imp_cases = []
        for i in range(4):
            cover_im = Image.open(SEMBIND_S4 / f'images/{i:03d}_same.png').convert('RGB')
            ckpts = {}
            for step in (50, 100, 150):
                forged_im = Image.open(imp_imgs[(i, step)]).convert('RGB')
                psnr, lp = lpips_psnr(cover_im, forged_im)
                v = sembind_verify(forged_im, seed_val=20261126100 + i * 100 + step)
                ham = int((v['q_code'] != aux_codes[str(i)]).sum())
                ckpts[f'step_{step}'] = dict(
                    loss=imp_losses[(i, step)], psnr_vs_cover=psnr, lpips_vs_cover=lp,
                    ba=v['ba'], accept_gs_default=v['accept_gs_default'], accept_gs_167=v['accept_gs_167'],
                    code_hamming=ham,
                )
                print(f'  [SemBind Same-Prompt Imprint] gidx={i:03d} step={step:3d} | ba={v["ba"]:.4f} (>181={v["accept_gs_default"]} >=167={v["accept_gs_167"]}) | H={ham}/1024', flush=True)
            imp_cases.append(dict(gidx=i, prompt=s4_proto['sources'][i]['prompt'], cover=f'{i:03d}_same.png', checkpoints=ckpts))
        del lpips_fn
        torch.cuda.empty_cache()

        # Part 1B: Same-Prompt Reprompt+ (4 sources x 3 wiggles x 2 proxies = 24 cands using own prompt)
        print('=== [S5C Part 1B] SemBind GS-S Same-Prompt Reprompt+ (4 sources x 3 wiggles x 2 proxies = 24 cands) ===', flush=True)
        rp_dir = RUN / 'same_prompt_reprompt_cands'
        rp_dir.mkdir(exist_ok=True)
        rp_results = {}
        for proxy_name, model_ref in [('sd21', 'Manojb/stable-diffusion-2-1-base'), ('sd15', str(SD15_PATH))]:
            pipe.to('cpu'); torch.cuda.empty_cache()
            proxy_pipe = StableDiffusionPipeline.from_pretrained(
                model_ref, torch_dtype=torch.float16, local_files_only=True, safety_checker=None
            ).to('cuda')
            fwd_sched = DDIMScheduler.from_config(proxy_pipe.scheduler.config)
            inv_sched_p = DDIMInverseScheduler.from_config(proxy_pipe.scheduler.config)
            proxy_pipe.set_progress_bar_config(disable=True)

            saved = []
            rng_wig = np.random.default_rng(2026112691 if proxy_name == 'sd21' else 2026112692)
            for i in range(4):
                wm_im = Image.open(SEMBIND_S4 / f'images/{i:03d}_bound.png').convert('RGB')
                proxy_pipe.scheduler = inv_sched_p
                with torch.no_grad():
                    img_t = torch.from_numpy(np.asarray(wm_im, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to('cuda', dtype=torch.float16) * 2.0 - 1.0
                    z0_p = proxy_pipe.vae.encode(img_t).latent_dist.mean * proxy_pipe.vae.config.scaling_factor
                    zT_p = proxy_pipe(prompt=[''], latents=z0_p, num_inference_steps=50, guidance_scale=1.0, output_type='latent', return_dict=False)[0]
                proxy_pipe.scheduler = fwd_sched
                own_prompt = s4_proto['sources'][i]['prompt']
                for ri in range(3):
                    lat_t = torch.from_numpy(wiggle_gs_latent(zT_p, rng_wig)).to('cuda', dtype=torch.float16)
                    with torch.no_grad():
                        cim = proxy_pipe(prompt=[own_prompt], latents=lat_t, num_inference_steps=50, guidance_scale=7.5).images[0].convert('RGB')
                    cp = rp_dir / f'{proxy_name}_g{i}_r{ri}.png'
                    cim.save(cp)
                    saved.append((i, ri, cp))
            del proxy_pipe
            torch.cuda.empty_cache()

            p_rows = []
            for idx_r, (i, ri, cp) in enumerate(saved):
                v = sembind_verify(Image.open(cp).convert('RGB'), seed_val=20261126200 + idx_r)
                ham = int((v['q_code'] != aux_codes[str(i)]).sum())
                p_rows.append(dict(
                    gidx=i, resample_idx=ri, ba=v['ba'],
                    accept_gs_default=v['accept_gs_default'], accept_gs_167=v['accept_gs_167'],
                    code_hamming=ham,
                ))
            rp_results[proxy_name] = dict(
                n_candidates=len(p_rows), n_sources=4,
                sembind_cand_default=sum(r['accept_gs_default'] for r in p_rows),
                sembind_src_default=sum(any(r['accept_gs_default'] for r in p_rows if r['gidx'] == g) for g in range(4)),
                sembind_cand_167=sum(r['accept_gs_167'] for r in p_rows),
                mean_ba=float(np.mean([r['ba'] for r in p_rows])),
                mean_hamming=float(np.mean([r['code_hamming'] for r in p_rows])),
                rows=p_rows,
            )
            print(f"  [SemBind Same-Prompt Reprompt+ {proxy_name}] fooled@181={rp_results[proxy_name]['sembind_cand_default']}/12 (src={rp_results[proxy_name]['sembind_src_default']}/4) | mean_ba={rp_results[proxy_name]['mean_ba']:.4f} | mean_H={rp_results[proxy_name]['mean_hamming']:.1f}", flush=True)

        imp_agg = {}
        for ck in ('step_50', 'step_100', 'step_150'):
            imp_agg[ck] = dict(
                n=4,
                sembind_fooled_default=sum(c['checkpoints'][ck]['accept_gs_default'] for c in imp_cases),
                sembind_fooled_167=sum(c['checkpoints'][ck]['accept_gs_167'] for c in imp_cases),
                mean_ba=float(np.mean([c['checkpoints'][ck]['ba'] for c in imp_cases])),
                mean_hamming=float(np.mean([c['checkpoints'][ck]['code_hamming'] for c in imp_cases])),
            )
        out = dict(
            elapsed_seconds=time.perf_counter() - t_all,
            gpu=args.gpu, uuid=uuid,
            imprint_summary=imp_agg,
            imprint_cases=imp_cases,
            reprompt_plus_summary={k: {kk: vv for kk, vv in v.items() if kk != 'rows'} for k, v in rp_results.items()},
        )
        write_json(RUN / 'instance_forgery_summary.json', out)
        write_json(RUN / 'instance_forgery_reprompt_rows.json', rp_results)
        print(f"[S5C SemBind Instance Forgery DONE] {out['elapsed_seconds']:.1f}s | Imprint150={imp_agg['step_150']['sembind_fooled_default']}/4 (mean_ba={imp_agg['step_150']['mean_ba']:.4f}) | RP+_sd21={rp_results['sd21']['sembind_cand_default']}/12 | RP+_sd15={rp_results['sd15']['sembind_cand_default']}/12", flush=True)

    # =========================================================================
    # REMOVAL20 SHARDED: Matched-20 x (8 WAVES str=0.5 + VAE + SD2.1 Regen 0.1/0.2/0.3)
    # =========================================================================
    if args.mode == 'removal20':
        from diffusers import StableDiffusionImg2ImgPipeline

        waves_dist = load_waves_distortions()
        my_sources = [s for i, s in enumerate(s4_proto['sources']) if i % args.num_shards == args.shard]
        print(f'=== [S5C Removal20 shard {args.shard}/{args.num_shards}] {len(my_sources)} prompts x 12 conds ===', flush=True)
        t0 = time.perf_counter()

        # Pre-generate VAE + SD2.1 Img2Img regen images cleanly before freeing img2img
        pipe.to('cpu'); torch.cuda.empty_cache()
        img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
            'Manojb/stable-diffusion-2-1-base', torch_dtype=torch.float16, local_files_only=True, safety_checker=None
        ).to('cuda')
        img2img.set_progress_bar_config(disable=True)

        regen_cache = {}
        for s in my_sources:
            gidx = s['gidx']
            im = Image.open(SEMBIND_S4 / f'images/{gidx:03d}_bound.png').convert('RGB')
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

        recs = []
        for li, s in enumerate(my_sources):
            gidx = s['gidx']
            bound_im = Image.open(SEMBIND_S4 / f'images/{gidx:03d}_bound.png').convert('RGB')
            aux_c = aux_codes[str(gidx)]
            q_pairs = []
            for ci, wtype in enumerate(WAVES_8_TYPES):
                abs_s = waves_dist.relative_strength_to_absolute(0.5, wtype)
                q_im = waves_dist.apply_single_distortion(bound_im.copy(), wtype, strength=abs_s, distortion_seed=2026112600 + gidx * 100 + ci).convert('RGB')
                q_pairs.append((f'waves_{wtype}', q_im))
            for cname in ('vae_roundtrip', 'sd21_regen_01', 'sd21_regen_02', 'sd21_regen_03'):
                q_pairs.append((cname, regen_cache[(gidx, cname)]))

            for ci, (cond, q_im) in enumerate(q_pairs):
                v = sembind_verify(q_im, seed_val=20261126400 + gidx * 20 + ci)
                ham = int((v['q_code'] != aux_c).sum())
                recs.append(dict(
                    gidx=gidx, dataset=s['dataset'], condition=cond,
                    ba=v['ba'], accept_gs_default=v['accept_gs_default'], accept_gs_167=v['accept_gs_167'],
                    code_hamming_vs_aux=ham, sec_verify=v['sec_verify'],
                ))
            write_json(RUN / f'removal_shard{args.shard}.json', recs)
            ok12 = sum(r['accept_gs_default'] for r in recs[-12:])
            print(f'  [SemBind Removal s{args.shard}] {li+1}/{len(my_sources)} gidx={gidx:03d} | pass={ok12}/12 | elapsed={time.perf_counter()-t0:.0f}s', flush=True)
        print(f'[S5C Removal20 shard {args.shard} DONE] {len(recs)} records in {time.perf_counter()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
