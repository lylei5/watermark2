"""Stage 5B: Official WAVES-8 & Regeneration Removal Suite (Native GS / Commit-then-Steer sscd_64b & hyb)
+ Same-Prompt Instance-Level Forgery Stress Test (Unmodified CVPR 2025 151-step Imprint & Reprompt+
  onto frozen independent same-prompt sibling covers from Stage 4D gen20).

Parts:
  1. removal20           - Exact Matched-20 (COCO[0:10] + SDP[50:60]) x 12 removal conditions:
                           8 official ICML 2024 WAVES (1477635) distortions at strength=0.5 +
                           VAE roundtrip + SD2.1 Img2Img regeneration (strength 0.1, 0.2, 0.3)
                           across Native GS-256b, Commit-then-Steer sscd_64b, and Commit-then-Steer hyb_sscd32_clip32.
  2. same_prompt_sscd64  - Unmodified CVPR 2025 151-step Imprint (gidx 0..3 -> frozen 000_same..003_same) on sscd_64b
                           + Same-Prompt Reprompt+ (4 sources x 3 wiggles x 2 proxies = 24 cands each) on sscd_64b & hyb.
  3. same_prompt_hyb     - Unmodified CVPR 2025 151-step Imprint (gidx 0..3 -> frozen 000_same..003_same) on hyb_sscd32_clip32.
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
RUN = ROOT / 'runs/commit_steer_stage5_20261126/removal_and_instance_cs'
GS100_S3 = ROOT / 'runs/commit_steer_paper_scale_20261123/gs100_unified'
HYB_S4 = ROOT / 'runs/commit_steer_stage4_20261125/multidim100_hyb'
SEMBIND_S4 = ROOT / 'runs/commit_steer_stage4_20261125/head2head_sembind'
GS_MOD_PATH = ROOT / 'runs/phase1_trajectory_native_gs_20261113/gs_watermark.py'
PGID_PIPE = ROOT / 'third_party/frozen/PGID_dc4ad96'
WAVES_ROOT = ROOT / 'third_party/WAVES_1477635_source'
SEM_FORGERY = ROOT / 'third_party/semantic-forgery'
EXTRA_DEPS = ROOT / 'envs/phase1_extra'
SSCD_PATH = ROOT / 'models/sscd/sscd_disc_mixup.torchscript.pt'
SD15_PATH = Path('/data0/liyanlei/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14')

MATCHED_20_GIDX = list(range(0, 10)) + list(range(50, 60))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['removal20', 'same_prompt_sscd64', 'same_prompt_hyb', 'main_lane'])
    ap.add_argument('--gpu', type=int, default=5)
    args = ap.parse_args()

    uuid = check_idle_gpu(args.gpu)
    RUN.mkdir(parents=True, exist_ok=True)
    work_dir = RUN / f'work_{args.mode}'
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
    import lpips
    from transformers import CLIPModel
    from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DDIMScheduler, DDIMInverseScheduler

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sys.path.insert(0, str(PGID_PIPE))
    sys.path.insert(0, str(EXTRA_DEPS))
    from pipe import pipe_utils

    waves_dist = load_waves_distortions()

    # Load frozen protocols & private materials (keys stay strictly in memory/249, never logged)
    s3_proto = read_json(GS100_S3 / 'protocol.json')
    sources = s3_proto['sources']
    th_sscd = float(s3_proto['threshold'])
    auth_th = float(s3_proto['auth_threshold'])
    priv_s3 = np.load(GS100_S3 / 'private_material.npz')
    key_s3, auth_s3, P64_s3 = priv_s3['key'], priv_s3['auth_bits'], priv_s3['P64']

    # Native GS full 256-bit messages for exact native_gs_256 evaluation
    embs_s3 = []
    for sh in (0, 1):
        embs_s3.extend(read_json(GS100_S3 / f'embs_shard{sh}.json'))
    w256_s3 = {e['gidx']: np.concatenate([auth_s3, np.array(e['target_bits'], dtype=np.uint8)]) for e in embs_s3}

    s4b_proto = read_json(HYB_S4 / 'protocol.json')
    th_hyb = float(s4b_proto['threshold'])
    priv_hyb = np.load(HYB_S4 / 'private_material.npz')
    key_hyb, auth_hyb, P_s32, P_c32 = priv_hyb['key'], priv_hyb['auth_bits'], priv_hyb['P_s32'], priv_hyb['P_c32']

    native_gs_mod = load_module('native_gs', GS_MOD_PATH)
    gs_s3 = native_gs_mod.Gaussian_Shading(1, 8, 1e-6, 1)
    gs_s3.key = torch.from_numpy(key_s3.astype(np.int64)).cuda()
    gs_hyb = native_gs_mod.Gaussian_Shading(1, 8, 1e-6, 1)
    gs_hyb.key = torch.from_numpy(key_hyb.astype(np.int64)).cuda()

    provider = pipe_utils.create_diffusion_pipe(
        pretrained_model_name_or_path='Manojb/stable-diffusion-2-1-base',
        resolution=512, device='cuda:0', scheduler_name='DDIM', scheduler_inverse_name='DDIM',
        dtype=torch.float16, eager_loading=True, disable_tqdm=True,
    )
    provider.pipe.set_progress_bar_config(disable=True)

    sscd = torch.jit.load(str(SSCD_PATH), map_location='cuda').eval()
    for p in sscd.parameters():
        p.requires_grad_(False)
    clip_asset = read_json(ROOT / 'runs/phase1_backend_audit_20261113_v2/complete.json')['model_files']['clip_vitl14']
    clip_model = CLIPModel.from_pretrained(clip_asset['root'], local_files_only=True).cuda().eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    lpips_fn = lpips.LPIPS(net='alex').cuda().eval()

    im_mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
    cl_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda').view(1, 3, 1, 1)
    cl_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda').view(1, 3, 1, 1)

    def invert_batch(pil_list, bs=6):
        zs = []
        for i in range(0, len(pil_list), bs):
            zs.append(provider.invert_images(pil_list[i:i + bs], num_inference_steps=50)['zT_torch'])
        return torch.cat(zs, dim=0)

    def extract_feats(pil_list, need_clip=False):
        xs = torch.stack([
            torch.from_numpy(np.asarray(im.convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1)
            for im in pil_list
        ]).cuda()
        with torch.no_grad():
            x320 = F.interpolate(xs, size=(320, 320), mode='bilinear', align_corners=False)
            es = F.normalize(sscd((x320 - im_mean) / im_std), dim=1).cpu().numpy().astype(np.float64)
            ec = None
            if need_clip:
                x224 = F.interpolate(xs, size=(224, 224), mode='bilinear', align_corners=False)
                ec = F.normalize(clip_model.get_image_features(pixel_values=(x224 - cl_mean) / cl_std).float(), dim=1).cpu().numpy().astype(np.float64)
        return es, ec

    def score_native(zT_1, gidx):
        rev_w = gs_s3.diffusion_inverse(((zT_1 > 0).int() + gs_s3.key) % 2).reshape(-1).cpu().numpy().astype(np.uint8)
        ba = float((rev_w == w256_s3[gidx]).mean())
        return dict(ba_256=ba, accept_gs_default=bool(ba > 181.0 / 256.0), accept_gs_167=bool(ba >= 167.0 / 256.0))

    def score_sscd64(zT_1, es_1):
        rev_w = gs_s3.diffusion_inverse(((zT_1 > 0).int() + gs_s3.key) % 2).reshape(-1).cpu().numpy().astype(np.uint8)
        auth_ba = float((rev_w[:192] == auth_s3).mean())
        auth_ok = auth_ba > auth_th
        rec_sgn = 2.0 * rev_w[192:].astype(np.float64) - 1.0
        qp = es_1 @ P64_s3
        d_blind = float(1.0 - np.dot(rec_sgn, qp) / (8.0 * float(np.linalg.norm(qp)) + 1e-12))
        feat_ok = d_blind <= th_sscd
        return dict(auth_ba=auth_ba, auth_ok=bool(auth_ok), d_blind=d_blind, feat_ok=bool(feat_ok), joint_accept=bool(auth_ok and feat_ok))

    def score_hyb(zT_1, es_1, ec_1):
        rev_w = gs_hyb.diffusion_inverse(((zT_1 > 0).int() + gs_hyb.key) % 2).reshape(-1).cpu().numpy().astype(np.uint8)
        auth_ba = float((rev_w[:192] == auth_hyb).mean())
        auth_ok = auth_ba > auth_th
        rec_sgn = 2.0 * rev_w[192:].astype(np.float64) - 1.0
        qps, qpc = es_1 @ P_s32, ec_1 @ P_c32
        ds = float(1.0 - np.dot(rec_sgn[:32], qps) / (np.sqrt(32.0) * float(np.linalg.norm(qps)) + 1e-12))
        dc = float(1.0 - np.dot(rec_sgn[32:], qpc) / (np.sqrt(32.0) * float(np.linalg.norm(qpc)) + 1e-12))
        d_blind = 0.5 * ds + 0.5 * dc
        feat_ok = d_blind <= th_hyb
        return dict(auth_ba=auth_ba, auth_ok=bool(auth_ok), d_sscd32=ds, d_clip32=dc, d_blind=d_blind, feat_ok=bool(feat_ok), joint_accept=bool(auth_ok and feat_ok))

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

    write_json(RUN / 'protocol.json', dict(
        run_name='removal_and_instance_cs',
        matched_20_gidx=MATCHED_20_GIDX,
        waves8_types=WAVES_8_TYPES,
        waves_relative_strength=0.5,
        regen_conditions=['vae_roundtrip', 'sd21_regen_01', 'sd21_regen_02', 'sd21_regen_03'],
        same_prompt_covers='runs/commit_steer_stage4_20261125/head2head_sembind/images/{000..003}_same.png',
        sscd64_threshold=th_sscd,
        hyb_threshold=th_hyb,
        auth_threshold=auth_th,
    ))
    t_global = time.perf_counter()

    # =========================================================================
    # PART 1: MATCHED-20 WAVES-8 (0.5) + REGENERATION REMOVAL (Native GS / sscd_64b / hyb)
    # =========================================================================
    if args.mode in ('removal20', 'main_lane'):
        print('=== [S5B Part 1] Matched-20 WAVES-8 (0.5) + Regeneration Removal (Native GS / sscd64 / hyb) ===', flush=True)
        t0 = time.perf_counter()
        img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
            'Manojb/stable-diffusion-2-1-base', torch_dtype=torch.float16, local_files_only=True, safety_checker=None
        ).to('cuda')
        img2img.set_progress_bar_config(disable=True)

        rem_conds = [*[f'waves_{w}' for w in WAVES_8_TYPES], 'vae_roundtrip', 'sd21_regen_01', 'sd21_regen_02', 'sd21_regen_03']
        arm_specs = [
            ('native_gs', GS100_S3 / 'images_unguided'),
            ('commit_steer_sscd64', GS100_S3 / 'images_guided'),
            ('commit_steer_hyb', HYB_S4 / 'images_guided'),
        ]
        all_recs = {a: [] for a, _ in arm_specs}
        for arm_name, img_dir in arm_specs:
            t_arm = time.perf_counter()
            for pi, gidx in enumerate(MATCHED_20_GIDX):
                im = Image.open(img_dir / f'{gidx:03d}.png').convert('RGB')
                q_imgs = []
                for ci, wtype in enumerate(WAVES_8_TYPES):
                    abs_s = waves_dist.relative_strength_to_absolute(0.5, wtype)
                    q_imgs.append(waves_dist.apply_single_distortion(im.copy(), wtype, strength=abs_s, distortion_seed=2026112600 + gidx * 100 + ci).convert('RGB'))
                with torch.no_grad():
                    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to('cuda', dtype=torch.float16)
                    z0 = img2img.vae.encode(x).latent_dist.mean
                    dec = img2img.vae.decode(z0, return_dict=False)[0]
                    q_imgs.append(Image.fromarray(((dec.float()[0].permute(1, 2, 0) / 2.0 + 0.5).clamp(0, 1).mul(255).round().byte().cpu().numpy())))
                for ri, rstr in enumerate((0.1, 0.2, 0.3)):
                    gen = torch.Generator(device='cuda').manual_seed(2026112650 + gidx * 10 + ri)
                    with torch.no_grad():
                        q_imgs.append(img2img(prompt='', image=im, strength=rstr, num_inference_steps=50, guidance_scale=7.5, generator=gen).images[0].convert('RGB'))

                zTs = invert_batch(q_imgs, bs=6)
                if arm_name == 'native_gs':
                    for ci, cond in enumerate(rem_conds):
                        sc = score_native(zTs[ci:ci+1], gidx)
                        all_recs[arm_name].append(dict(gidx=gidx, dataset=sources[gidx]['dataset'], condition=cond, **sc))
                elif arm_name == 'commit_steer_sscd64':
                    es, _ = extract_feats(q_imgs, need_clip=False)
                    for ci, cond in enumerate(rem_conds):
                        sc = score_sscd64(zTs[ci:ci+1], es[ci])
                        all_recs[arm_name].append(dict(gidx=gidx, dataset=sources[gidx]['dataset'], condition=cond, **sc))
                else:
                    es, ec = extract_feats(q_imgs, need_clip=True)
                    for ci, cond in enumerate(rem_conds):
                        sc = score_hyb(zTs[ci:ci+1], es[ci], ec[ci])
                        all_recs[arm_name].append(dict(gidx=gidx, dataset=sources[gidx]['dataset'], condition=cond, **sc))
            print(f'  [S5B Removal20 {arm_name}] 20/20 done in {time.perf_counter()-t_arm:.1f}s', flush=True)

        del img2img
        torch.cuda.empty_cache()

        rem_summary = {}
        for arm_name, recs in all_recs.items():
            key_ok = 'accept_gs_default' if arm_name == 'native_gs' else 'joint_accept'
            by_c = {}
            for cond in rem_conds:
                sub = [r for r in recs if r['condition'] == cond]
                entry = dict(n=len(sub), pass_count=sum(r[key_ok] for r in sub))
                if arm_name == 'native_gs':
                    entry['pass_gs_167'] = sum(r['accept_gs_167'] for r in sub)
                    entry['mean_ba_256'] = float(np.mean([r['ba_256'] for r in sub]))
                else:
                    entry['auth_pass'] = sum(r['auth_ok'] for r in sub)
                    entry['mean_d_blind'] = float(np.mean([r['d_blind'] for r in sub]))
                by_c[cond] = entry
            w_sub = [r for r in recs if r['condition'].startswith('waves_')]
            r_sub = [r for r in recs if not r['condition'].startswith('waves_')]
            rem_summary[arm_name] = dict(
                n_prompts=20,
                waves8_pass=sum(r[key_ok] for r in w_sub),
                waves8_total=len(w_sub),
                waves8_auth_pass=sum(r['accept_gs_default' if arm_name == 'native_gs' else 'auth_ok'] for r in w_sub),
                regen4_pass=sum(r[key_ok] for r in r_sub),
                regen4_total=len(r_sub),
                regen4_auth_pass=sum(r['accept_gs_default' if arm_name == 'native_gs' else 'auth_ok'] for r in r_sub),
                by_cond=by_c,
            )
            print(f"  [S5B Removal20 SUMMARY {arm_name}] WAVES8={rem_summary[arm_name]['waves8_pass']}/160 (auth={rem_summary[arm_name]['waves8_auth_pass']}/160) | Regen4={rem_summary[arm_name]['regen4_pass']}/80 (auth={rem_summary[arm_name]['regen4_auth_pass']}/80)", flush=True)

        write_json(RUN / 'removal20_summary.json', dict(elapsed_seconds=time.perf_counter() - t0, arms=rem_summary))
        write_json(RUN / 'removal20_records.json', all_recs)

    # =========================================================================
    # PART 2: SAME-PROMPT INSTANCE FORGERY — Commit-then-Steer sscd_64b (Imprint151 + Reprompt+ for sscd64 & hyb)
    # =========================================================================
    def run_same_prompt_imprint_arm(arm_tag: str, wm_dir: Path, score_fn):
        sys.path.insert(0, str(SEM_FORGERY))
        from utils import imprint_utils

        print(f'=== [S5B Same-Prompt 151-step Imprint | {arm_tag}] 4 cases -> frozen 000_same..003_same ===', flush=True)
        provider.pipe.to('cpu')
        torch.cuda.empty_cache()
        pipe_atk, _, inv_sched = imprint_utils.load_attacker_pipe(
            modelid='Manojb/stable-diffusion-2-1-base', scheduler='DDIM', device='cuda'
        )
        pipe_atk.set_progress_bar_config(disable=True)
        diffpipe = imprint_utils.DiffPipe(pipe_atk, scheduler=inv_sched, device=pipe_atk.device)

        imp_dir = RUN / f'same_prompt_imprint_{arm_tag}'
        imp_dir.mkdir(exist_ok=True)
        saved_imgs = {}
        saved_losses = {}
        t_all = time.perf_counter()
        for i in range(4):
            wm_im = Image.open(wm_dir / f'{i:03d}.png').convert('RGB')
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
                    saved_imgs[(i, step)] = p_save
                    saved_losses[(i, step)] = float(loss.detach())
            print(f'  [{arm_tag} Same-Prompt Imprint opt] gidx={i:03d} done in {time.perf_counter()-t0:.0f}s', flush=True)
        del diffpipe, pipe_atk
        torch.cuda.empty_cache()

        provider.pipe.to('cuda')
        cases = []
        for i in range(4):
            cover_im = Image.open(SEMBIND_S4 / f'images/{i:03d}_same.png').convert('RGB')
            ckpts = {}
            for step in (50, 100, 150):
                forged_im = Image.open(saved_imgs[(i, step)]).convert('RGB')
                zT = invert_batch([forged_im], bs=1)
                es, ec = extract_feats([forged_im], need_clip=(arm_tag == 'hyb'))
                sc = score_fn(zT, es[0], ec[0] if ec is not None else None)
                psnr, lp = lpips_psnr(cover_im, forged_im)
                ckpts[f'step_{step}'] = dict(loss=saved_losses[(i, step)], psnr_vs_cover=psnr, lpips_vs_cover=lp, **sc)
                print(f'  [{arm_tag} Same-Prompt Imprint] gidx={i:03d} step={step:3d} | auth={sc["auth_ba"]:.4f}({sc["auth_ok"]}) | d_blind={sc["d_blind"]:.4f}({sc["feat_ok"]}) | joint={sc["joint_accept"]}', flush=True)
            cases.append(dict(gidx=i, prompt=sources[i]['prompt'], cover=f'{i:03d}_same.png', checkpoints=ckpts))

        agg = {}
        for ck in ('step_50', 'step_100', 'step_150'):
            agg[ck] = dict(
                n=4,
                native_auth_fooled=sum(c['checkpoints'][ck]['auth_ok'] for c in cases),
                commit_steer_fooled=sum(c['checkpoints'][ck]['joint_accept'] for c in cases),
                mean_auth_ba=float(np.mean([c['checkpoints'][ck]['auth_ba'] for c in cases])),
                mean_d_blind=float(np.mean([c['checkpoints'][ck]['d_blind'] for c in cases])),
            )
        out = dict(elapsed_seconds=time.perf_counter() - t_all, arm=arm_tag, imprint_summary=agg, imprint_cases=cases)
        write_json(RUN / f'same_prompt_imprint_{arm_tag}_summary.json', out)
        print(f"[{arm_tag} Same-Prompt Imprint DONE] step150: auth_fooled={agg['step_150']['native_auth_fooled']}/4 | joint_fooled={agg['step_150']['commit_steer_fooled']}/4 | mean_d={agg['step_150']['mean_d_blind']:.4f}", flush=True)

    if args.mode in ('same_prompt_sscd64', 'main_lane'):
        run_same_prompt_imprint_arm('sscd64', GS100_S3 / 'images_guided', lambda z, es, ec: score_sscd64(z, es))

        # Same-Prompt Reprompt+ (4 sources x 3 GS wiggles x 2 proxies = 24 cands each) for BOTH sscd64 & hyb
        print('=== [S5B Same-Prompt Reprompt+] 4 sources x 3 wiggles x 2 proxies = 24 cands each (sscd64 & hyb) ===', flush=True)
        t_rp = time.perf_counter()
        rp_out = {'sscd64': {}, 'hyb': {}}
        for proxy_name, model_ref in [('sd21', 'Manojb/stable-diffusion-2-1-base'), ('sd15', str(SD15_PATH))]:
            provider.pipe.to('cpu')
            torch.cuda.empty_cache()
            proxy_pipe = StableDiffusionPipeline.from_pretrained(
                model_ref, torch_dtype=torch.float16, local_files_only=True, safety_checker=None
            ).to('cuda')
            fwd_sched = DDIMScheduler.from_config(proxy_pipe.scheduler.config)
            inv_sched_p = DDIMInverseScheduler.from_config(proxy_pipe.scheduler.config)
            proxy_pipe.set_progress_bar_config(disable=True)

            gen_cache = {'sscd64': [], 'hyb': []}
            for arm_tag, wm_dir, seed_base in [
                ('sscd64', GS100_S3 / 'images_guided', 2026112671 if proxy_name == 'sd21' else 2026112672),
                ('hyb', HYB_S4 / 'images_guided', 2026112681 if proxy_name == 'sd21' else 2026112682),
            ]:
                rng_wig = np.random.default_rng(seed_base)
                for i in range(4):
                    wm_im = Image.open(wm_dir / f'{i:03d}.png').convert('RGB')
                    proxy_pipe.scheduler = inv_sched_p
                    with torch.no_grad():
                        img_t = torch.from_numpy(np.asarray(wm_im, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to('cuda', dtype=torch.float16) * 2.0 - 1.0
                        z0_p = proxy_pipe.vae.encode(img_t).latent_dist.mean * proxy_pipe.vae.config.scaling_factor
                        zT_p = proxy_pipe(prompt=[''], latents=z0_p, num_inference_steps=50, guidance_scale=1.0, output_type='latent', return_dict=False)[0]
                    proxy_pipe.scheduler = fwd_sched
                    own_prompt = sources[i]['prompt']
                    for ri in range(3):
                        lat_t = torch.from_numpy(wiggle_gs_latent(zT_p, rng_wig)).to('cuda', dtype=torch.float16)
                        with torch.no_grad():
                            cim = proxy_pipe(prompt=[own_prompt], latents=lat_t, num_inference_steps=50, guidance_scale=7.5).images[0].convert('RGB')
                        gen_cache[arm_tag].append((i, ri, cim))
            del proxy_pipe
            torch.cuda.empty_cache()
            provider.pipe.to('cuda')

            for arm_tag in ('sscd64', 'hyb'):
                items = gen_cache[arm_tag]
                cims = [x[2] for x in items]
                zTs = invert_batch(cims, bs=6)
                es, ec = extract_feats(cims, need_clip=(arm_tag == 'hyb'))
                rows = []
                for idx_c, (i, ri, _) in enumerate(items):
                    sc = score_sscd64(zTs[idx_c:idx_c+1], es[idx_c]) if arm_tag == 'sscd64' else score_hyb(zTs[idx_c:idx_c+1], es[idx_c], ec[idx_c])
                    rows.append(dict(gidx=i, resample_idx=ri, **sc))
                rp_out[arm_tag][proxy_name] = dict(
                    n_candidates=len(rows), n_sources=4,
                    native_auth_cand=sum(r['auth_ok'] for r in rows),
                    commit_steer_cand=sum(r['joint_accept'] for r in rows),
                    commit_steer_src=sum(any(r['joint_accept'] for r in rows if r['gidx'] == g) for g in range(4)),
                    mean_d_blind=float(np.mean([r['d_blind'] for r in rows])),
                    rows=rows,
                )
                print(f"  [Same-Prompt Reprompt+ {arm_tag} {proxy_name}] auth={rp_out[arm_tag][proxy_name]['native_auth_cand']}/12 | joint={rp_out[arm_tag][proxy_name]['commit_steer_cand']}/12 (src={rp_out[arm_tag][proxy_name]['commit_steer_src']}/4) | mean_d={rp_out[arm_tag][proxy_name]['mean_d_blind']:.4f}", flush=True)

        write_json(RUN / 'same_prompt_reprompt_summary.json', dict(
            elapsed_seconds=time.perf_counter() - t_rp,
            summary={a: {p: {k: v for k, v in d.items() if k != 'rows'} for p, d in pd.items()} for a, pd in rp_out.items()},
        ))
        write_json(RUN / 'same_prompt_reprompt_rows.json', rp_out)

    if args.mode == 'same_prompt_hyb':
        run_same_prompt_imprint_arm('hyb', HYB_S4 / 'images_guided', lambda z, es, ec: score_hyb(z, es, ec))

    print(f'[S5B {args.mode} COMPLETE] total elapsed={time.perf_counter()-t_global:.1f}s', flush=True)


if __name__ == '__main__':
    main()
