"""Stage 4D: Head-to-Head Same-Cohort Direct Baseline Evaluation — Paper-Faithful SemBind (GS-S).
Uses exact audited SemBind components (third_party/frozen/SemBind_337c9f9, diffusers 0.21.4 DPMSolverMultistepScheduler,
DINOv2-Giant + released sembind_masker.pth, paper §3.2 HMAC-permuted sign binding sigma=1.0, DPM50 + 100-step VAE decoder opt).

Cohort: Exact stratified 20 prompts from gs100_unified (10 COCO [0..9] + 10 SDP [50..59]) for 100% apple-to-apple comparison!
Subcommands:
  prepare         - Freeze protocol, 20 matched unified prompts, private permutation & GS keys (0o600).
  gen20           - Generate aux, unmasked GS, SemBind GS-S bound, and same-prompt sibling for all 20 prompts + LPIPS/CLIP/timing.
  benign20        - Full DPM50 + 100-step VAE opt verification on Clean + all 9 GS distortions (200 verifs; supports --shard/--num-shards).
  merge_benign    - Merge sharded benign20 records & compute summary.
  attacks_null    - Live 151-step Imprint (4 cases x ckpts 50/100/150) + Live 9-cand Reprompt+ (4 sources x 2 proxies x 9 = 72 cands)
                    + Empirical Natural-Image Null FPR (50 real COCO validation images).
"""
import argparse
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
from PIL import Image
from scipy.stats import beta, norm

ROOT = Path('/data0/liyanlei/watermark_research')
RUN = ROOT / 'runs/commit_steer_stage4_20261125/head2head_sembind'
GS100_S3 = ROOT / 'runs/commit_steer_paper_scale_20261123/gs100_unified'
NULL_INPUTS = ROOT / 'runs/unified_null_features_20261116_v2/inputs.json'
PHASE1_DIR = ROOT / 'runs/phase1_trajectory_native_gs_20261113'
GS_MOD_PATH = PHASE1_DIR / 'gs_watermark.py'
GS_DIST_PATH = ROOT / 'runs/defense_revalidation_20261111/gs_image_utils.py'
SEMBIND_OFFICIAL = ROOT / 'third_party/frozen/SemBind_337c9f9'
SEMBIND_SCHED = ROOT / 'runs/sembind_chain_revalidation_20261111/scheduling_dpmsolver_multistep.py'
SD21_SNAP = ROOT / 'cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/snapshots/0094d483a120f3f33dafbd187ea4aa60d10de75c'
SEMBIND_MODELS = ROOT / 'models/sembind'
SEM_FORGERY = ROOT / 'third_party/semantic-forgery'
EXTRA_DEPS = ROOT / 'envs/phase1_extra'
COVER_DIR = ROOT / 'runs/published_imprint_dpm_full_20261110/inputs'
SD15_PATH = Path('/data0/liyanlei/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14')

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


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    return 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))


def keyed_permutation(key: bytes, length: int = 16384) -> np.ndarray:
    tags = [hmac.new(key, b'sembind/permutation/' + i.to_bytes(4, 'big'), hashlib.sha256).digest() for i in range(length)]
    return np.asarray(sorted(range(length), key=tags.__getitem__), dtype=np.int64)


def wiggle_gs_latent(latents_t, rng):
    arr = latents_t.detach().float().cpu().numpy()
    y = np.clip((norm.cdf(arr) * 2.0).astype(np.int32), 0, 1)
    u = rng.uniform(1e-6, 1.0 - 1e-6, size=y.shape).astype(np.float64)
    return norm.ppf((u + y) / 2.0)


def do_prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / 'protocol.json').exists() and (RUN / 'private_material.npz').exists():
        print('head2head_sembind already prepared; keeping frozen protocol & private material.', flush=True)
        return

    s3_proto = read_json(GS100_S3 / 'protocol.json')
    sources = [dict(gidx=g, **s3_proto['sources'][g]) for g in MATCHED_20_GIDX]
    assert len(sources) == 20

    rng_priv = np.random.default_rng(2026112504)
    gs_key_np = rng_priv.integers(0, 2, size=(1, 4, 64, 64), dtype=np.uint8)
    wm_msg_256 = rng_priv.integers(0, 2, size=(1, 4, 8, 8), dtype=np.uint8)
    perm_key = rng_priv.bytes(32)
    permutation = keyed_permutation(perm_key, 16384)

    np.savez(RUN / 'private_material.npz', key=gs_key_np, message=wm_msg_256, permutation=permutation)
    os.chmod(RUN / 'private_material.npz', 0o600)

    proto = dict(
        run_name='head2head_sembind',
        method='Paper-faithful SemBind GS-S (official SemBind_337c9f9 + diffusers 0.21.4 DPM + 100-step VAE opt + paper §3.2 HMAC permutation sigma=1.0)',
        implementation_note='Complete SemBind binding code is not released by authors; uses released DINOv2-Giant + masker + official inversion.py + audited sembind_binding.py per docs/BASELINE_IMPLEMENTATION_DIFFERENCES.md',
        n_prompts=20,
        matched_gidx=MATCHED_20_GIDX,
        sigma=1.0,
        gen_steps=50, gen_cfg=7.5,
        inv_steps=50, inv_cfg=3.0, inv_order=0, decoder_inv=True,
        gs9_conditions=GS_9_CONDITIONS,
        sources=sources,
    )
    write_json(RUN / 'protocol.json', proto)
    print('Prepared head2head_sembind | 20 matched unified prompts (10 COCO + 10 SDP)', flush=True)


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
    ap.add_argument('mode', choices=['prepare', 'gen20', 'benign20', 'merge_benign', 'attacks_null', 'all_single'])
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    args = ap.parse_args()

    do_prepare()
    if args.mode == 'prepare':
        return

    if args.mode == 'merge_benign':
        proto = read_json(RUN / 'protocol.json')
        recs = []
        for p in sorted(RUN.glob('benign_shard*.json')):
            recs.extend(read_json(p))
        assert len(recs) == 200, f'Expected 200 benign records, got {len(recs)}'
        conds = ['clean', *GS_9_CONDITIONS.keys()]
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
                mean_ba=float(np.mean([r['ba'] for r in sub])),
                mean_hamming=float(np.mean([r['code_hamming_vs_aux'] for r in sub])),
                mean_sec_verify=float(np.mean([r['sec_verify'] for r in sub])),
            )
        summary = dict(
            n_prompts=20,
            clean_gs_default=sum(r['accept_gs_default'] for r in cl),
            gs7_regular_default=sum(r['accept_gs_default'] for r in ed7),
            gs9_all_default=sum(r['accept_gs_default'] for r in ed),
            sec_verify_mean=float(np.mean([r['sec_verify'] for r in recs])),
            by_cond=by_c,
        )
        write_json(RUN / 'benign20_summary.json', summary)
        write_json(RUN / 'benign20_records.json', recs)
        print(f"[S4D SemBind Benign20 MERGED] clean={summary['clean_gs_default']}/20 | gs9={summary['gs9_all_default']}/180 ({100*summary['gs9_all_default']/180:.1f}%) | gs7={summary['gs7_regular_default']}/140 ({100*summary['gs7_regular_default']/140:.1f}%) | verify={summary['sec_verify_mean']:.2f}s/img", flush=True)
        return

    uuid = check_idle_gpu(args.gpu)
    proto = read_json(RUN / 'protocol.json')
    priv = np.load(RUN / 'private_material.npz')
    gs_key_np, wm_msg_np, permutation = priv['key'], priv['message'], priv['permutation']

    work_sub = f'work_{args.mode}_s{args.shard}'
    torch, F, inversion, pipe, extract_code = init_sembind_env(uuid, work_sub)
    native_gs = load_module('native_gs', GS_MOD_PATH).Gaussian_Shading(1, 8, 1e-6, 1)
    native_gs.key = torch.from_numpy(gs_key_np.astype(np.int64)).cuda()
    native_gs.watermark = torch.from_numpy(wm_msg_np.astype(np.int64)).cuda()
    gs_dist = load_module('gs_dist', GS_DIST_PATH)

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

    def dpm_generate(prompt: str, z_np: np.ndarray):
        pipe.to('cuda')
        out, _ = pipe(
            prompt, guidance_scale=7.5, num_inference_steps=50, height=512, width=512,
            latents=torch.as_tensor(z_np, dtype=torch.float32, device='cuda'),
        )
        return out.images[0].convert('RGB')

    # =========================================================================
    # GEN20: Generate aux, plain GS, SemBind GS-S bound, same-prompt sibling + LPIPS/CLIP
    # =========================================================================
    if args.mode in ('gen20', 'all_single'):
        print('=== [S4D Gen20] Generating 20 matched unified prompts for SemBind GS-S ===', flush=True)
        img_dir = RUN / 'images'
        img_dir.mkdir(exist_ok=True)
        gen_rows = []
        aux_codes = {}
        t0_all = time.perf_counter()

        for li, s in enumerate(proto['sources']):
            gidx = s['gidx']
            prompt = s['prompt']
            t0 = time.perf_counter()
            z_aux = np.random.default_rng(2026112540 + gidx).standard_normal((1, 4, 64, 64)).astype(np.float32)
            aux_im = dpm_generate(prompt, z_aux)
            aux_code = extract_code(aux_im)
            aux_codes[str(gidx)] = aux_code

            sd = np.tile(wm_msg_np, (1, 1, 8, 8))
            m_enc = ((sd + gs_key_np) % 2).reshape(-1)
            np.random.seed(2026112560 + gidx)
            carrier = native_gs.truncSampling(m_enc).float()
            assert native_gs.eval_watermark(carrier) == 1.0
            mask = np.tile(aux_code, 16)[permutation].reshape(1, 4, 64, 64)
            masked = carrier * (1.0 - 2.0 * torch.tensor(mask, dtype=torch.float32, device='cuda'))

            bound_im = dpm_generate(prompt, masked.cpu().numpy())
            sec_total = time.perf_counter() - t0
            plain_im = dpm_generate(prompt, carrier.cpu().numpy())
            z_same = np.random.default_rng(2026112580 + gidx).standard_normal((1, 4, 64, 64)).astype(np.float32)
            same_im = dpm_generate(prompt, z_same)

            bound_code = extract_code(bound_im)
            same_code = extract_code(same_im)
            aux_im.save(img_dir / f'{gidx:03d}_aux.png')
            plain_im.save(img_dir / f'{gidx:03d}_plain.png')
            bound_im.save(img_dir / f'{gidx:03d}_bound.png')
            same_im.save(img_dir / f'{gidx:03d}_same.png')

            h_bound = int((bound_code != aux_code).sum())
            h_same = int((same_code != aux_code).sum())
            gen_rows.append(dict(
                gidx=gidx, dataset=s['dataset'], sec_gen_total=sec_total,
                hamming_aux_vs_bound=h_bound, hamming_aux_vs_same=h_same,
            ))
            print(f'  [SemBind Gen20] {li+1:2d}/20 gidx={gidx:03d} ({s["dataset"]}) | gen={sec_total:.1f}s | H(aux,bound)={h_bound}/1024 | H(aux,same)={h_same}/1024', flush=True)

        np.savez(RUN / 'aux_codes.npz', **aux_codes)

        # Compute LPIPS & CLIP (plain GS vs SemBind bound)
        import lpips
        from transformers import CLIPModel, CLIPProcessor
        pipe.to('cpu'); torch.cuda.empty_cache()
        lpips_fn = lpips.LPIPS(net='alex').cuda().eval()
        clip_asset = read_json(ROOT / 'runs/phase1_backend_audit_20261113_v2/complete.json')['model_files']['clip_vitl14']
        clip_model = CLIPModel.from_pretrained(clip_asset['root'], local_files_only=True).cuda().eval()
        clip_proc = CLIPProcessor.from_pretrained(clip_asset['root'], local_files_only=True)

        def to_m11(im):
            return torch.from_numpy(np.asarray(im, dtype=np.float32).transpose(2, 0, 1)).unsqueeze(0).cuda() / 127.5 - 1.0

        for r, s in zip(gen_rows, proto['sources']):
            im_p = Image.open(img_dir / f'{r["gidx"]:03d}_plain.png').convert('RGB')
            im_b = Image.open(img_dir / f'{r["gidx"]:03d}_bound.png').convert('RGB')
            with torch.no_grad():
                r['lpips_plain_vs_bound'] = float(lpips_fn(to_m11(im_p), to_m11(im_b)).item())
                inp = clip_proc(text=[s['prompt']], images=[im_p, im_b], return_tensors='pt', padding=True, truncation=True, max_length=77).to('cuda')
                co = clip_model(**inp)
                ie, te = F.normalize(co.image_embeds, dim=-1), F.normalize(co.text_embeds, dim=-1)
                r['clip_plain'] = float((ie[0] * te[0]).sum().item())
                r['clip_bound'] = float((ie[1] * te[0]).sum().item())
        del lpips_fn, clip_model, clip_proc
        torch.cuda.empty_cache()

        gen_sum = dict(
            elapsed_seconds=time.perf_counter() - t0_all,
            n_prompts=20,
            sec_gen_mean=float(np.mean([r['sec_gen_total'] for r in gen_rows])),
            lpips_mean=float(np.mean([r['lpips_plain_vs_bound'] for r in gen_rows])),
            clip_plain_mean=float(np.mean([r['clip_plain'] for r in gen_rows])),
            clip_bound_mean=float(np.mean([r['clip_bound'] for r in gen_rows])),
            mean_hamming_aux_vs_bound=float(np.mean([r['hamming_aux_vs_bound'] for r in gen_rows])),
            mean_hamming_aux_vs_same=float(np.mean([r['hamming_aux_vs_same'] for r in gen_rows])),
            rows=gen_rows,
        )
        write_json(RUN / 'gen20_summary.json', gen_sum)
        print(f"[S4D Gen20 DONE] {gen_sum['elapsed_seconds']:.1f}s | gen={gen_sum['sec_gen_mean']:.2f}s | CLIP={gen_sum['clip_bound_mean']:.4f} vs {gen_sum['clip_plain_mean']:.4f}", flush=True)

    # =========================================================================
    # BENIGN20: Clean + ALL 9 standard GS distortions (200 full SemBind DPM50+VAE100 verifs)
    # =========================================================================
    if args.mode in ('benign20', 'all_single'):
        print(f'=== [S4D Benign20 shard {args.shard}/{args.num_shards}] SemBind GS-S Clean + 9 GS Distortions ===', flush=True)
        aux_codes = np.load(RUN / 'aux_codes.npz')
        conds = ['clean', *GS_9_CONDITIONS.keys()]
        my_sources = [s for i, s in enumerate(proto['sources']) if i % args.num_shards == args.shard]
        recs = []
        t0 = time.perf_counter()
        for li, s in enumerate(my_sources):
            gidx = s['gidx']
            bound_im = Image.open(RUN / 'images' / f'{gidx:03d}_bound.png').convert('RGB')
            aux_c = aux_codes[str(gidx)]
            for ci, cond in enumerate(conds):
                if cond == 'clean':
                    q_im = bound_im
                else:
                    seed = 2026112300 + gidx * 100 + ci
                    gs_dist.set_random_seed(seed)
                    dargs = SimpleNamespace(**{k: None for k in GS_9_CONDITIONS})
                    setattr(dargs, cond, GS_9_CONDITIONS[cond])
                    q_im = gs_dist.image_distortion(bound_im.copy(), seed, dargs).convert('RGB')
                v = sembind_verify(q_im, seed_val=2026112500 + gidx * 20 + ci)
                ham = int((v['q_code'] != aux_c).sum())
                recs.append(dict(
                    gidx=gidx, dataset=s['dataset'], condition=cond,
                    ba=v['ba'], accept_gs_default=v['accept_gs_default'], accept_gs_167=v['accept_gs_167'],
                    code_hamming_vs_aux=ham, sec_verify=v['sec_verify'],
                ))
            write_json(RUN / f'benign_shard{args.shard}.json', recs)
            cl_ok = recs[-len(conds)]['accept_gs_default']
            ed_ok = sum(r['accept_gs_default'] for r in recs[-len(conds)+1:])
            print(f'  [SemBind Benign s{args.shard}] {li+1}/{len(my_sources)} gidx={gidx:03d} | clean={cl_ok} | 9-GS={ed_ok}/9 | elapsed={time.perf_counter()-t0:.0f}s', flush=True)
        print(f'[S4D Benign20 shard {args.shard} DONE] {len(recs)} records in {time.perf_counter()-t0:.1f}s', flush=True)

    # =========================================================================
    # ATTACKS & NULL: Live 151-step Imprint (4) + Live Reprompt+ (4x9x2=72) + COCO Null (50)
    # =========================================================================
    if args.mode in ('attacks_null', 'all_single'):
        sys.path.insert(0, str(SEM_FORGERY))
        from utils import imprint_utils
        from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler

        aux_codes = np.load(RUN / 'aux_codes.npz')
        t_all = time.perf_counter()

        # 1. Empirical Natural-Image Null FPR (50 real COCO val images)
        print('=== [S4D Attacks&Null Part 1] SemBind GS-S Natural-Image Null FPR (50 real COCO val images) ===', flush=True)
        all_inputs = read_json(NULL_INPUTS)
        val_50 = [x for x in all_inputs if x['split'] == 'validation'][:50]

        def center_crop_512(p: str) -> Image.Image:
            im = Image.open(p).convert('RGB')
            w, h = im.size
            s = min(w, h)
            return im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((512, 512), Image.BICUBIC)

        null_bas = []
        for ni, item in enumerate(val_50):
            v = sembind_verify(center_crop_512(item['path']), seed_val=20261125900 + ni)
            null_bas.append(v['ba'])
            if (ni + 1) % 10 == 0:
                print(f'  [SemBind Null50] {ni+1}/50 | max_ba={max(null_bas):.4f}', flush=True)
        ab = np.array(null_bas)
        np.savez(RUN / 'null50_scores.npz', ba=ab)
        null_fp = int((ab > 181.0 / 256.0).sum())
        null_sum = dict(n=50, ba_mean=float(ab.mean()), ba_max=float(ab.max()), fp_gs_default=null_fp, cp95_upper=clopper_pearson_upper(null_fp, 50))

        # 2. Live 151-step Imprint Forgery on 4 SemBind GS-S bound sources (gidx 0..3 -> covers 0..3)
        print('=== [S4D Attacks&Null Part 2] Live 151-step CVPR 2025 Imprint on SemBind GS-S (4 cases) ===', flush=True)
        pipe.to('cpu'); torch.cuda.empty_cache()
        pipe_atk, _, inv_sched = imprint_utils.load_attacker_pipe(
            modelid='Manojb/stable-diffusion-2-1-base', scheduler='DPM', device='cuda'
        )
        pipe_atk.set_progress_bar_config(disable=True)
        diffpipe = imprint_utils.DiffPipe(pipe_atk, scheduler=inv_sched, device=pipe_atk.device)

        imp_dir = RUN / 'imprint_151'
        imp_dir.mkdir(exist_ok=True)
        imp_imgs = {}
        for i in range(4):
            wm_im = Image.open(RUN / 'images' / f'{i:03d}_bound.png').convert('RGB')
            cover_im = Image.open(COVER_DIR / f'{i:03d}_cover.png').convert('RGB').resize((512, 512), Image.BICUBIC)
            with torch.no_grad():
                zT_target = imprint_utils.invert_image(pipe=pipe_atk, scheduler=inv_sched, image=wm_im, num_inference_steps=50)
                z0_orig = imprint_utils.pixel_to_latent(cover_im, pipe_atk)
            z0 = torch.nn.Parameter(z0_orig.detach().clone())
            opt = torch.optim.Adam([z0], lr=0.01)
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
        del diffpipe, pipe_atk
        torch.cuda.empty_cache()

        imp_cases = []
        for i in range(4):
            ckpts = {}
            for step in (50, 100, 150):
                forged_im = Image.open(imp_imgs[(i, step)]).convert('RGB')
                v = sembind_verify(forged_im, seed_val=20261125700 + i * 100 + step)
                ham = int((v['q_code'] != aux_codes[str(i)]).sum())
                ckpts[f'step_{step}'] = dict(ba=v['ba'], accept_gs_default=v['accept_gs_default'], code_hamming=ham)
                print(f'  [SemBind Imprint] gidx={i:03d} step={step:3d} | ba={v["ba"]:.4f} ({v["accept_gs_default"]}) | H={ham}/1024', flush=True)
            imp_cases.append(dict(gidx=i, cover=f'{i:03d}_cover.png', checkpoints=ckpts))

        # 3. Live 9-candidate Reprompt+ on 4 SemBind GS-S sources (4 x 3 I2P x 3 wiggles x 2 proxies = 72 cands)
        print('=== [S4D Attacks&Null Part 3] Live 9-cand Reprompt+ on SemBind GS-S (4 sources x 2 proxies x 9 = 72 cands) ===', flush=True)
        rp_proto = read_json(ROOT / 'runs/author_reprompt_plus_20261115/protocol.json')
        gs_refs = [r for r in rp_proto['references'] if r['carrier'] == 'GS'][:4]
        rp_dir = RUN / 'reprompt_cands'
        rp_dir.mkdir(exist_ok=True)
        rp_results = {}

        for proxy_name, model_ref in [('sd21', 'Manojb/stable-diffusion-2-1-base'), ('sd15', str(SD15_PATH))]:
            pipe.to('cpu'); torch.cuda.empty_cache()
            proxy_pipe = StableDiffusionPipeline.from_pretrained(model_ref, torch_dtype=torch.float16, local_files_only=True, safety_checker=None).to('cuda')
            fwd_sched = DDIMScheduler.from_config(proxy_pipe.scheduler.config)
            inv_sched_p = DDIMInverseScheduler.from_config(proxy_pipe.scheduler.config)
            proxy_pipe.set_progress_bar_config(disable=True)

            saved = []
            rng_wig = np.random.default_rng(2026112541 if proxy_name == 'sd21' else 2026112542)
            for i in range(4):
                wm_im = Image.open(RUN / 'images' / f'{i:03d}_bound.png').convert('RGB')
                proxy_pipe.scheduler = inv_sched_p
                with torch.no_grad():
                    img_t = torch.from_numpy(np.asarray(wm_im, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to('cuda', dtype=torch.float16) * 2.0 - 1.0
                    z0_p = proxy_pipe.vae.encode(img_t).latent_dist.mean * proxy_pipe.vae.config.scaling_factor
                    zT_p = proxy_pipe(prompt=[''], latents=z0_p, num_inference_steps=50, guidance_scale=1.0, output_type='latent', return_dict=False)[0]
                proxy_pipe.scheduler = fwd_sched
                prompts3 = [tp['prompt'] for tp in gs_refs[i]['target_prompts']]
                for pi, ptxt in enumerate(prompts3):
                    for ri in range(3):
                        lat_t = torch.from_numpy(wiggle_gs_latent(zT_p, rng_wig)).to('cuda', dtype=torch.float16)
                        with torch.no_grad():
                            cim = proxy_pipe(prompt=[ptxt], latents=lat_t, num_inference_steps=50, guidance_scale=7.5).images[0].convert('RGB')
                        cp = rp_dir / f'{proxy_name}_g{i}_p{pi}_r{ri}.png'
                        cim.save(cp)
                        saved.append((i, pi, ri, cp))
            del proxy_pipe
            torch.cuda.empty_cache()

            p_rows = []
            for idx_r, (i, pi, ri, cp) in enumerate(saved):
                v = sembind_verify(Image.open(cp).convert('RGB'), seed_val=20261125800 + idx_r)
                ham = int((v['q_code'] != aux_codes[str(i)]).sum())
                p_rows.append(dict(gidx=i, prompt_idx=pi, resample_idx=ri, ba=v['ba'], accept_gs_default=v['accept_gs_default'], code_hamming=ham))
            for i in range(4):
                acc_i = sum(r['accept_gs_default'] for r in p_rows if r['gidx'] == i)
                print(f'  [SemBind Reprompt+ {proxy_name}] gidx={i:03d}: fooled={acc_i}/9', flush=True)
            rp_results[proxy_name] = dict(
                n_candidates=len(p_rows), n_sources=4,
                sembind_cand=sum(r['accept_gs_default'] for r in p_rows),
                sembind_src=sum(any(r['accept_gs_default'] for r in p_rows if r['gidx'] == g) for g in range(4)),
                mean_ba=float(np.mean([r['ba'] for r in p_rows])),
                rows=p_rows,
            )

        imp_agg = {}
        for ck in ('step_50', 'step_100', 'step_150'):
            imp_agg[ck] = dict(
                n=4,
                sembind_fooled=sum(c['checkpoints'][ck]['accept_gs_default'] for c in imp_cases),
                mean_ba=float(np.mean([c['checkpoints'][ck]['ba'] for c in imp_cases])),
            )

        out = dict(
            elapsed_seconds=time.perf_counter() - t_all,
            gpu=args.gpu, uuid=uuid,
            null_fpr_50=null_sum,
            imprint_summary=imp_agg,
            imprint_cases=imp_cases,
            reprompt_plus_summary={k: {kk: vv for kk, vv in v.items() if kk != 'rows'} for k, v in rp_results.items()},
        )
        write_json(RUN / 'attacks_null_summary.json', out)
        write_json(RUN / 'reprompt_plus_rows.json', rp_results)
        print(f"[S4D Attacks&Null DONE] {out['elapsed_seconds']:.1f}s | Null50={null_fp}/50 | Imprint150={imp_agg['step_150']['sembind_fooled']}/4 | RP+_sd21={rp_results['sd21']['sembind_cand']}/36 | RP+_sd15={rp_results['sd15']['sembind_cand']}/36", flush=True)


if __name__ == '__main__':
    main()
