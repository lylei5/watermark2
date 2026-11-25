"""Stage 4B: Paper-Scale 100-Prompt Multi-Dimensional Fusion (hyb_sscd32_clip32: 32b SSCD + 32b CLIP)
inside Native 256-bit Gaussian Shading (192b auth + 64b commitment, 64x spatial repetition).

Subcommands:
  prepare   - Freeze protocol, 100 official unified prompts (exact match to Stage 3A gs100_unified),
              private key material (0o600), and Stage 3D frozen 95%-TPR threshold (0.553052...).
  gs100     - 100 unseen unified prompts (50 COCO + 50 SDP): 12-step preview + mid-step Tweedie [5,35],
              Clean + all 9 standard GS distortions (1,000 batched 50-step DDIM inversions), LPIPS, CLIP,
              9,900-pair cross-prompt FAR, and saved unguided/guided PNGs for exact COCO-50 FID.
  null2000  - Empirical Natural-Image Null FPR on exact same 2,000 real COCO images (1,000 cal + 1,000 val).
  attacks   - Full-budget CVPR 2025 Live 151-step Imprint (4 guided sources x ckpts 50/100/150) +
              Live 9-candidate Reprompt+ across BOTH SD2.1 & SD1.5 proxies (6 sources x 9 x 2 = 108 cands).
  all       - Run prepare (if needed) -> gs100 -> null2000 -> attacks on a single idle GPU.
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
RUN = ROOT / 'runs/commit_steer_stage4_20261125/multidim100_hyb'
GS100_S3 = ROOT / 'runs/commit_steer_paper_scale_20261123/gs100_unified'
MULTIDIM_S3 = ROOT / 'runs/commit_steer_paper_scale_20261123/multidim_features'
NULL_INPUTS = ROOT / 'runs/unified_null_features_20261116_v2/inputs.json'
GS_MOD_PATH = ROOT / 'runs/phase1_trajectory_native_gs_20261113/gs_watermark.py'
GS_DIST_PATH = ROOT / 'runs/defense_revalidation_20261111/gs_image_utils.py'
PGID_PIPE = ROOT / 'third_party/frozen/PGID_dc4ad96'
SEM_FORGERY = ROOT / 'third_party/semantic-forgery'
EXTRA_DEPS = ROOT / 'envs/phase1_extra'
SSCD_PATH = ROOT / 'models/sscd/sscd_disc_mixup.torchscript.pt'
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


def make_keyed_projection(key_bytes: bytes, dim_in: int, k: int, tag: bytes) -> np.ndarray:
    seed = int.from_bytes(hmac.digest(key_bytes, tag, 'sha256')[:16], 'big')
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((dim_in, k)))
    return q.astype(np.float64)


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    return 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))


def wiggle_gs_latent(latents_t, rng):
    """Exact CVPR 2025 gs_provider.wiggle_latents (l=1 binary Gaussian Shading)."""
    arr = latents_t.detach().float().cpu().numpy()
    y = np.clip((norm.cdf(arr) * 2.0).astype(np.int32), 0, 1)
    u = rng.uniform(1e-6, 1.0 - 1e-6, size=y.shape).astype(np.float64)
    return norm.ppf((u + y) / 2.0)


def do_prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / 'protocol.json').exists() and (RUN / 'private_material.npz').exists():
        print('multidim100_hyb already prepared; keeping frozen protocol & private material.', flush=True)
        return

    th = read_json(MULTIDIM_S3 / 'summary.json')['stage0_multidim']['hyb_sscd32_clip32']['th']
    s3_proto = read_json(GS100_S3 / 'protocol.json')
    sources = s3_proto['sources']
    assert len(sources) == 100

    rng_priv = np.random.default_rng(2026112502)
    gs_key_np = rng_priv.integers(0, 2, size=(1, 4, 64, 64), dtype=np.uint8)
    auth_192 = rng_priv.integers(0, 2, size=(192,), dtype=np.uint8)
    kbytes = np.packbits(gs_key_np.reshape(-1)).tobytes()
    P_s32_np = make_keyed_projection(kbytes, 512, 32, b'commit-steer-s4-hyb-sscd32-v1')
    P_c32_np = make_keyed_projection(kbytes, 768, 32, b'commit-steer-s4-hyb-clip32-v1')
    np.savez(RUN / 'private_material.npz', key=gs_key_np, auth_bits=auth_192, P_s32=P_s32_np, P_c32=P_c32_np)
    os.chmod(RUN / 'private_material.npz', 0o600)

    proto = dict(
        run_name='multidim100_hyb',
        arm='hyb_sscd32_clip32',
        bit_allocation='192b_auth + 32b_sscd + 32b_clip',
        n_prompts=100,
        preview_steps=12,
        step_range=[5, 35],
        step_size=0.042,
        stop_loss=0.08,
        threshold=th,
        threshold_source='runs/commit_steer_paper_scale_20261123/multidim_features/summary.json (frozen 95% WAVES cal)',
        auth_threshold=136.0 / 192.0,
        gs9_conditions=GS_9_CONDITIONS,
        sources=sources,
    )
    write_json(RUN / 'protocol.json', proto)
    print(f'Prepared multidim100_hyb | 100 unified prompts | frozen th={th:.6f}', flush=True)


def init_env_and_models(uuid: str):
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
    for p in sscd.parameters():
        p.requires_grad_(False)

    clip_asset = read_json(ROOT / 'runs/phase1_backend_audit_20261113_v2/complete.json')['model_files']['clip_vitl14']
    clip_model = CLIPModel.from_pretrained(clip_asset['root'], local_files_only=True).cuda().eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    clip_proc = CLIPProcessor.from_pretrained(clip_asset['root'], local_files_only=True)
    lpips_fn = lpips.LPIPS(net='alex').cuda().eval()

    return torch, F, provider, pipe, sscd, clip_model, clip_proc, lpips_fn


def run_gs100(torch, F, provider, pipe, sscd, clip_model, clip_proc, lpips_fn, gpu: int, uuid: str):
    proto = read_json(RUN / 'protocol.json')
    th = proto['threshold']
    priv = np.load(RUN / 'private_material.npz')
    gs_key_np, auth_192, P_s32_np, P_c32_np = priv['key'], priv['auth_bits'], priv['P_s32'], priv['P_c32']

    native_gs = load_module('native_gs', GS_MOD_PATH).Gaussian_Shading(1, 8, 1e-6, 1)
    native_gs.key = torch.from_numpy(gs_key_np.astype(np.int64)).cuda()
    gs_dist = load_module('gs_dist', GS_DIST_PATH)

    P_s32_t = torch.from_numpy(P_s32_np).float().cuda()
    P_c32_t = torch.from_numpy(P_c32_np).float().cuda()
    im_mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
    cl_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda').view(1, 3, 1, 1)
    cl_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda').view(1, 3, 1, 1)

    def decode_x(z):
        dec = pipe.vae.decode((z.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor), return_dict=False)[0]
        return (dec.float() / 2.0 + 0.5).clamp(0.0, 1.0)

    def to_pil(x):
        return Image.fromarray(x.detach()[0].permute(1, 2, 0).mul(255.0).round().byte().cpu().numpy())

    def eval_feat_batch(pil_list):
        xs = torch.stack([
            torch.from_numpy(np.asarray(im.convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1)
            for im in pil_list
        ]).cuda()
        with torch.no_grad():
            x320 = F.interpolate(xs, size=(320, 320), mode='bilinear', align_corners=False)
            es = F.normalize(sscd((x320 - im_mean) / im_std), dim=1).cpu().numpy().astype(np.float64)
            x224 = F.interpolate(xs, size=(224, 224), mode='bilinear', align_corners=False)
            ec = F.normalize(clip_model.get_image_features(pixel_values=(x224 - cl_mean) / cl_std).float(), dim=1).cpu().numpy().astype(np.float64)
        return es, ec

    def diff_hyb_loss(x, ts_t):
        x320 = F.interpolate(x, size=(320, 320), mode='bilinear', align_corners=False)
        es = F.normalize(sscd((x320 - im_mean) / im_std), dim=1)[0]
        x224 = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        ec = F.normalize(clip_model.get_image_features(pixel_values=(x224 - cl_mean) / cl_std).float(), dim=1)[0]
        ps, pc = es @ P_s32_t, ec @ P_c32_t
        ts_s, ts_c = ts_t[:32], ts_t[32:]
        ls = (1.0 - F.cosine_similarity(ps.unsqueeze(0), ts_s.unsqueeze(0)).squeeze()) + 0.5 * F.softplus(0.05 - ps * ts_s, beta=20.0).mean()
        lc = (1.0 - F.cosine_similarity(pc.unsqueeze(0), ts_c.unsqueeze(0)).squeeze()) + 0.5 * F.softplus(0.05 - pc * ts_c, beta=20.0).mean()
        return 0.5 * ls + 0.5 * lc

    def ddim_gen(prompt, zT, num_steps=50, ts_t=None, step_range=(5, 35), step_size=0.042, stop_loss=0.08):
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
                loss = diff_hyb_loss(decode_x(z0l), ts_t)
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
            inp = clip_proc(text=[prompt], images=[base_im, cand_im], return_tensors='pt', padding=True, truncation=True, max_length=77).to('cuda')
            out = clip_model(**inp)
            ie = F.normalize(out.image_embeds, dim=-1)
            te = F.normalize(out.text_embeds, dim=-1)
            return lp, float((ie[0] * te[0]).sum().item()), float((ie[1] * te[0]).sum().item())

    def invert_batch(pil_list):
        zTs = []
        for i in range(0, len(pil_list), 5):
            zTs.append(provider.invert_images(pil_list[i:i + 5], num_inference_steps=50)['zT_torch'])
        return torch.cat(zTs, dim=0)

    def hyb_dist(sgn64, es, ec):
        qps, qpc = es @ P_s32_np, ec @ P_c32_np
        ds = float(1.0 - np.dot(sgn64[:32], qps) / (np.sqrt(32.0) * float(np.linalg.norm(qps)) + 1e-12))
        dc = float(1.0 - np.dot(sgn64[32:], qpc) / (np.sqrt(32.0) * float(np.linalg.norm(qpc)) + 1e-12))
        return ds, dc, 0.5 * ds + 0.5 * dc

    with torch.no_grad():
        pipe('warmup', generator=torch.Generator(device='cuda').manual_seed(7), num_inference_steps=2)

    base_dir = RUN / 'images_unguided'
    guid_dir = RUN / 'images_guided'
    base_dir.mkdir(parents=True, exist_ok=True)
    guid_dir.mkdir(parents=True, exist_ok=True)

    conds = ['clean', *GS_9_CONDITIONS.keys()]
    records, embs_out = [], []
    t_start = time.perf_counter()

    for gidx, s in enumerate(proto['sources']):
        prompt = s['prompt']
        t0 = time.perf_counter()
        zT_prev = torch.randn((1, 4, 64, 64), generator=torch.Generator(device='cuda').manual_seed(2026112500 + gidx), device='cuda', dtype=torch.float16)
        prev_im = ddim_gen(prompt, zT_prev, num_steps=12, ts_t=None)
        torch.cuda.synchronize()
        sec_prev = time.perf_counter() - t0

        es_p, ec_p = eval_feat_batch([prev_im])
        p_bits = np.concatenate([
            ((es_p[0] @ P_s32_np) >= 0.0).astype(np.uint8),
            ((ec_p[0] @ P_c32_np) >= 0.0).astype(np.uint8),
        ])
        true_sgn = 2.0 * p_bits.astype(np.float64) - 1.0
        ts_t = torch.tensor(true_sgn, device='cuda', dtype=torch.float32)

        w_flat = np.concatenate([auth_192, p_bits])
        m_enc = ((np.tile(w_flat.reshape(1, 4, 8, 8), (1, 1, 8, 8)) + gs_key_np) % 2).reshape(-1)
        np.random.seed(2026112550 + gidx)
        zT = native_gs.truncSampling(m_enc)

        t0 = time.perf_counter()
        base_im = ddim_gen(prompt, zT, num_steps=50, ts_t=None)
        torch.cuda.synchronize()
        sec_base = time.perf_counter() - t0
        base_im.save(base_dir / f'{gidx:03d}.png')

        t0 = time.perf_counter()
        guid_im = ddim_gen(prompt, zT, num_steps=50, ts_t=ts_t, step_range=(5, 35), step_size=0.042, stop_loss=0.08)
        torch.cuda.synchronize()
        sec_guid = time.perf_counter() - t0
        guid_im.save(guid_dir / f'{gidx:03d}.png')

        lp, clip_base, clip_guid = eval_lpips_clip(base_im, guid_im, prompt)

        q_imgs = []
        for ci, cond in enumerate(conds):
            if cond == 'clean':
                q_imgs.append(guid_im)
            else:
                seed = 2026112300 + gidx * 100 + ci
                gs_dist.set_random_seed(seed)
                dargs = SimpleNamespace(**{k: None for k in GS_9_CONDITIONS})
                setattr(dargs, cond, GS_9_CONDITIONS[cond])
                q_imgs.append(gs_dist.image_distortion(guid_im.copy(), seed, dargs).convert('RGB'))

        z_invs = invert_batch(q_imgs)
        q_es, q_ec = eval_feat_batch(q_imgs)
        embs_out.append(dict(gidx=gidx, target_bits=p_bits.tolist(), clean_es=q_es[0].tolist(), clean_ec=q_ec[0].tolist()))

        for ci, cond in enumerate(conds):
            rev_w = native_gs.diffusion_inverse(((z_invs[ci:ci+1] > 0).int() + native_gs.key) % 2).reshape(-1).cpu().numpy().astype(np.uint8)
            full_ba = float((rev_w == w_flat).mean())
            auth_ba = float((rev_w[:192] == auth_192).mean())
            auth_ok = auth_ba > (136.0 / 192.0)
            p_ber = float((rev_w[192:] != p_bits).mean())
            rec_sgn = 2.0 * rev_w[192:].astype(np.float64) - 1.0
            ds_b, dc_b, d_blind = hyb_dist(rec_sgn, q_es[ci], q_ec[ci])
            _, _, d_oracle = hyb_dist(true_sgn, q_es[ci], q_ec[ci])
            feat_ok = d_blind <= th
            records.append(dict(
                gidx=gidx, dataset=s['dataset'], condition=cond,
                sec_prev=sec_prev, sec_base=sec_base, sec_guid=sec_guid, sec_total=sec_prev + sec_guid,
                lpips=lp, clip_base=clip_base, clip_guid=clip_guid,
                full_ba=full_ba, auth_ba=auth_ba, auth_ok=bool(auth_ok), payload_ber=p_ber,
                d_sscd32=ds_b, d_clip32=dc_b, d_blind=d_blind, d_oracle=d_oracle,
                feat_ok=bool(feat_ok), joint_blind_ok=bool(auth_ok and feat_ok),
            ))

        if (gidx + 1) % 10 == 0:
            print(f'[S4B Hyb100 | GPU {gpu}] {gidx+1}/100 done ({time.perf_counter()-t_start:.1f}s)', flush=True)
            write_json(RUN / 'records_partial.json', records)

    # Cross-prompt FAR (9,900 off-diagonal pairs)
    sgns = np.array([2.0 * np.array(e['target_bits'], dtype=np.float64) - 1.0 for e in embs_out])
    qps = np.array([np.array(e['clean_es'], dtype=np.float64) @ P_s32_np for e in embs_out])
    qpc = np.array([np.array(e['clean_ec'], dtype=np.float64) @ P_c32_np for e in embs_out])
    qps_n = qps / (np.linalg.norm(qps, axis=1, keepdims=True) + 1e-12)
    qpc_n = qpc / (np.linalg.norm(qpc, axis=1, keepdims=True) + 1e-12)
    ds_mat = 1.0 - (sgns[:, :32] @ qps_n.T) / np.sqrt(32.0)
    dc_mat = 1.0 - (sgns[:, 32:] @ qpc_n.T) / np.sqrt(32.0)
    dist_mat = 0.5 * ds_mat + 0.5 * dc_mat
    off_diag = dist_mat[~np.eye(100, dtype=bool)]
    cross_far_cnt = int((off_diag <= th).sum())

    cl = [r for r in records if r['condition'] == 'clean']
    ed = [r for r in records if r['condition'] != 'clean']
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

    non_occl = [r for r in ed if r['condition'] not in ('random_crop_ratio', 'random_drop_ratio')]
    summary = dict(
        arm='hyb_sscd32_clip32',
        gpu=gpu, uuid=uuid, elapsed_seconds=time.perf_counter() - t_start,
        n_prompts=100, threshold=th,
        sec_prev_mean=float(np.mean([r['sec_prev'] for r in cl])),
        sec_total_mean=float(np.mean([r['sec_total'] for r in cl])),
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
        cross_prompt_far_total=9900,
        cross_prompt_far=cross_far_cnt / 9900.0,
        by_cond=by_cond,
    )
    write_json(RUN / 'gs100_summary.json', summary)
    write_json(RUN / 'gs100_records.json', records)
    write_json(RUN / 'gs100_embs.json', embs_out)
    print(f"[S4B Hyb100 DONE] {summary['elapsed_seconds']:.1f}s | clean={summary['clean_joint_pass']}/100 | gs9={summary['gs9_joint_pass']}/900 ({100*summary['gs9_joint_pass']/900:.1f}%) | gs7={summary['gs7_regular_joint_pass']}/700 ({100*summary['gs7_regular_joint_pass']/700:.1f}%) | LPIPS={summary['lpips_mean']:.3f} | CLIP={summary['clip_guid_mean']:.4f} vs {summary['clip_base_mean']:.4f} | crossFAR={cross_far_cnt}/9900 ({100*summary['cross_prompt_far']:.2f}%)", flush=True)


def run_null2000(torch, F, provider, sscd, clip_model, gpu: int, uuid: str):
    proto = read_json(RUN / 'protocol.json')
    th = proto['threshold']
    priv = np.load(RUN / 'private_material.npz')
    gs_key_np, auth_192, P_s32_np, P_c32_np = priv['key'], priv['auth_bits'], priv['P_s32'], priv['P_c32']

    native_gs = load_module('native_gs', GS_MOD_PATH).Gaussian_Shading(1, 8, 1e-6, 1)
    native_gs.key = torch.from_numpy(gs_key_np.astype(np.int64)).cuda()

    im_mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
    cl_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda').view(1, 3, 1, 1)
    cl_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda').view(1, 3, 1, 1)

    all_inputs = read_json(NULL_INPUTS)
    cal_1000 = [x for x in all_inputs if x['split'] == 'calibration'][:1000]
    val_1000 = [x for x in all_inputs if x['split'] == 'validation'][:1000]
    assert len(cal_1000) == 1000 and len(val_1000) == 1000

    def center_crop_512(p: str) -> Image.Image:
        im = Image.open(p).convert('RGB')
        w, h = im.size
        s = min(w, h)
        return im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((512, 512), Image.BICUBIC)

    t0 = time.perf_counter()
    splits_out = {}
    null_dir = RUN / 'null_fpr_2000'
    null_dir.mkdir(parents=True, exist_ok=True)

    for split_name, items in [('calibration', cal_1000), ('validation', val_1000)]:
        auth_bas, d_blinds, d_s_list, d_c_list = [], [], [], []
        bs = 8
        for start in range(0, len(items), bs):
            imgs = [center_crop_512(x['path']) for x in items[start:start + bs]]
            with torch.no_grad():
                zTs = provider.invert_images(imgs, num_inference_steps=50)['zT_torch']
                xs = torch.stack([
                    torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
                    for im in imgs
                ]).cuda()
                x320 = F.interpolate(xs, size=(320, 320), mode='bilinear', align_corners=False)
                es_b = F.normalize(sscd((x320 - im_mean) / im_std), dim=1).cpu().numpy().astype(np.float64)
                x224 = F.interpolate(xs, size=(224, 224), mode='bilinear', align_corners=False)
                ec_b = F.normalize(clip_model.get_image_features(pixel_values=(x224 - cl_mean) / cl_std).float(), dim=1).cpu().numpy().astype(np.float64)

            for bi in range(len(imgs)):
                rev_w = native_gs.diffusion_inverse(((zTs[bi:bi+1] > 0).int() + native_gs.key) % 2).reshape(-1).cpu().numpy().astype(np.uint8)
                auth_ba = float((rev_w[:192] == auth_192).mean())
                rec_sgn = 2.0 * rev_w[192:].astype(np.float64) - 1.0
                qps, qpc = es_b[bi] @ P_s32_np, ec_b[bi] @ P_c32_np
                ds = float(1.0 - np.dot(rec_sgn[:32], qps) / (np.sqrt(32.0) * float(np.linalg.norm(qps)) + 1e-12))
                dc = float(1.0 - np.dot(rec_sgn[32:], qpc) / (np.sqrt(32.0) * float(np.linalg.norm(qpc)) + 1e-12))
                auth_bas.append(auth_ba)
                d_s_list.append(ds)
                d_c_list.append(dc)
                d_blinds.append(0.5 * ds + 0.5 * dc)

            if (start + bs) % 200 == 0:
                print(f'[S4B Null2000 {split_name} | GPU {gpu}] {start+bs}/1000 ({time.perf_counter()-t0:.1f}s) max_auth={max(auth_bas):.4f} min_d={min(d_blinds):.4f}', flush=True)

        ab, db = np.array(auth_bas), np.array(d_blinds)
        splits_out[split_name] = dict(
            n=len(ab),
            auth_ba_mean=float(ab.mean()), auth_ba_std=float(ab.std()), auth_ba_max=float(ab.max()),
            d_blind_mean=float(db.mean()), d_blind_std=float(db.std()), d_blind_min=float(db.min()),
            auth_fp_default=int((ab > 136.0 / 192.0).sum()),
            feat_fp_only=int((db <= th).sum()),
            joint_fp_default=int(((ab > 136.0 / 192.0) & (db <= th)).sum()),
        )
        np.savez(null_dir / f'{split_name}_scores.npz', auth_ba=ab, d_blind=db, d_sscd32=np.array(d_s_list), d_clip32=np.array(d_c_list))

    cal_ab = np.load(null_dir / 'calibration_scores.npz')['auth_ba']
    val_ab = np.load(null_dir / 'validation_scores.npz')['auth_ba']
    val_db = np.load(null_dir / 'validation_scores.npz')['d_blind']
    th_1pct = float(np.quantile(cal_ab, 0.99, method='higher'))
    th_01pct = float(np.quantile(cal_ab, 0.999, method='higher'))
    pooled_fp = splits_out['calibration']['joint_fp_default'] + splits_out['validation']['joint_fp_default']

    summary = dict(
        elapsed_seconds=time.perf_counter() - t0,
        gpu=gpu, uuid=uuid, threshold=th,
        splits=splits_out,
        empirical_calibration=dict(
            cal_auth_1pct_cutoff=th_1pct,
            val_auth_only_fp_at_1pct=int((val_ab > th_1pct).sum()),
            val_joint_fp_at_1pct=int(((val_ab > th_1pct) & (val_db <= th)).sum()),
            cal_auth_01pct_cutoff=th_01pct,
            val_auth_only_fp_at_01pct=int((val_ab > th_01pct).sum()),
            val_joint_fp_at_01pct=int(((val_ab > th_01pct) & (val_db <= th)).sum()),
            val_joint_default_fp=splits_out['validation']['joint_fp_default'],
            val_joint_default_cp95_upper=clopper_pearson_upper(splits_out['validation']['joint_fp_default'], 1000),
            pooled_2000_joint_default_fp=pooled_fp,
            pooled_2000_cp95_upper=clopper_pearson_upper(pooled_fp, 2000),
        ),
    )
    write_json(null_dir / 'summary.json', summary)
    print(f"[S4B Null2000 DONE] {summary['elapsed_seconds']:.1f}s | pooled={pooled_fp}/2000 | val@1% auth={summary['empirical_calibration']['val_auth_only_fp_at_1pct']} -> joint={summary['empirical_calibration']['val_joint_fp_at_1pct']}", flush=True)


def run_attacks(torch, F, provider, sscd, clip_model, lpips_fn, gpu: int, uuid: str):
    sys.path.insert(0, str(SEM_FORGERY))
    from utils import imprint_utils
    from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler

    proto = read_json(RUN / 'protocol.json')
    th = proto['threshold']
    priv = np.load(RUN / 'private_material.npz')
    gs_key_np, auth_192, P_s32_np, P_c32_np = priv['key'], priv['auth_bits'], priv['P_s32'], priv['P_c32']

    native_gs = load_module('native_gs', GS_MOD_PATH).Gaussian_Shading(1, 8, 1e-6, 1)
    native_gs.key = torch.from_numpy(gs_key_np.astype(np.int64)).cuda()

    im_mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
    cl_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda').view(1, 3, 1, 1)
    cl_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda').view(1, 3, 1, 1)

    def blind_verify(im):
        z_inv = provider.invert_images(im, num_inference_steps=50)['zT_torch']
        rev_w = native_gs.diffusion_inverse(((z_inv > 0).int() + native_gs.key) % 2).reshape(-1).cpu().numpy().astype(np.uint8)
        auth_ba = float((rev_w[:192] == auth_192).mean())
        auth_ok = auth_ba > (136.0 / 192.0)
        rec_sgn = 2.0 * rev_w[192:].astype(np.float64) - 1.0
        x = torch.from_numpy(np.asarray(im.convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()
        with torch.no_grad():
            x320 = F.interpolate(x, size=(320, 320), mode='bilinear', align_corners=False)
            es = F.normalize(sscd((x320 - im_mean) / im_std), dim=1)[0].cpu().numpy().astype(np.float64)
            x224 = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            ec = F.normalize(clip_model.get_image_features(pixel_values=(x224 - cl_mean) / cl_std).float(), dim=1)[0].cpu().numpy().astype(np.float64)
        qps, qpc = es @ P_s32_np, ec @ P_c32_np
        ds = float(1.0 - np.dot(rec_sgn[:32], qps) / (np.sqrt(32.0) * float(np.linalg.norm(qps)) + 1e-12))
        dc = float(1.0 - np.dot(rec_sgn[32:], qpc) / (np.sqrt(32.0) * float(np.linalg.norm(qpc)) + 1e-12))
        d_blind = 0.5 * ds + 0.5 * dc
        feat_ok = d_blind <= th
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

    atk_dir = RUN / 'live_attacks'
    imp_dir = atk_dir / 'imprint_151'
    imp_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    # 1. Live 151-step Imprint Forgery (4 cases: gidx 0..3 -> covers 0..3)
    print('=== [S4B Attacks Part 1] Live 151-step CVPR 2025 Imprint (4 cases) ===', flush=True)
    pipe_atk, _, inv_sched = imprint_utils.load_attacker_pipe(
        modelid='Manojb/stable-diffusion-2-1-base', scheduler='DDIM', device='cuda'
    )
    pipe_atk.set_progress_bar_config(disable=True)
    diffpipe = imprint_utils.DiffPipe(pipe_atk, scheduler=inv_sched, device=pipe_atk.device)

    imprint_results = []
    for i in range(4):
        wm_im = Image.open(RUN / 'images_guided' / f'{i:03d}.png').convert('RGB')
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
                v = blind_verify(forged_im)
                psnr, lp = lpips_psnr(cover_im, forged_im)
                ckpts[f'step_{step}'] = dict(loss=float(loss.detach()), psnr_vs_cover=psnr, lpips_vs_cover=lp, **v)
                print(f'  Hyb Imprint gidx={i:03d} step={step:3d} | PSNR={psnr:.1f} LPIPS={lp:.3f} | GS={v["auth_ba"]:.4f}({v["auth_ok"]}) | d_hyb={v["d_blind"]:.4f}({v["feat_ok"]}) | joint={v["joint_accept"]}', flush=True)
        imprint_results.append(dict(gidx=i, cover=f'{i:03d}_cover.png', seconds=time.perf_counter() - t0, checkpoints=ckpts))

    sz, pad = 256, 28
    canvas = Image.new('RGB', (sz * 3, pad + sz * 4), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for ci, cname in enumerate(['Hyb Guided WM (PASS)', 'Unrelated Cover', 'Imprint Step-150 (GS Fooled / Hyb REJ)']):
        draw.text((ci * sz + 6, 6), cname, fill=(0, 0, 0))
    for ri, r in enumerate(imprint_results):
        ims = [
            Image.open(RUN / 'images_guided' / f'{r["gidx"]:03d}.png').convert('RGB'),
            Image.open(COVER_DIR / r['cover']).convert('RGB'),
            Image.open(imp_dir / f'gidx_{r["gidx"]:03d}_step_150.png').convert('RGB'),
        ]
        for ci, im in enumerate(ims):
            canvas.paste(im.resize((sz, sz), Image.BICUBIC), (ci * sz, pad + ri * sz))
    canvas.save(atk_dir / 'imprint_live_montage.jpg', quality=92)

    del diffpipe, pipe_atk
    torch.cuda.empty_cache()

    # 2. Live 9-candidate Reprompt+ (SD2.1 + SD1.5 proxies, 6 sources x 9 x 2 = 108 candidates)
    print('=== [S4B Attacks Part 2] Live 9-cand Reprompt+ (6 sources x 2 proxies x 9 = 108 cands) ===', flush=True)
    rp_proto = read_json(ROOT / 'runs/author_reprompt_plus_20261115/protocol.json')
    gs_refs = [r for r in rp_proto['references'] if r['carrier'] == 'GS'][:6]
    rp_results = {}
    for proxy_name, model_ref in [('sd21', 'Manojb/stable-diffusion-2-1-base'), ('sd15', str(SD15_PATH))]:
        proxy_pipe = StableDiffusionPipeline.from_pretrained(model_ref, torch_dtype=torch.float16, local_files_only=True, safety_checker=None).to('cuda')
        fwd_sched = DDIMScheduler.from_config(proxy_pipe.scheduler.config)
        inv_sched_p = DDIMInverseScheduler.from_config(proxy_pipe.scheduler.config)
        proxy_pipe.set_progress_bar_config(disable=True)

        p_rows = []
        rng_wig = np.random.default_rng(2026112511 if proxy_name == 'sd21' else 2026112512)
        for i in range(6):
            wm_im = Image.open(RUN / 'images_guided' / f'{i:03d}.png').convert('RGB')
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
                    lat_t = torch.from_numpy(wiggle_gs_latent(zT_p, rng_wig)).to('cuda', dtype=torch.float16)
                    with torch.no_grad():
                        cand_im = proxy_pipe(prompt=[ptxt], latents=lat_t, num_inference_steps=50, guidance_scale=7.5).images[0].convert('RGB')
                    v = blind_verify(cand_im)
                    p_rows.append(dict(gidx=i, prompt_idx=pi, resample_idx=ri, **v))
            src_gs = sum(r['auth_ok'] for r in p_rows if r['gidx'] == i)
            src_ours = sum(r['joint_accept'] for r in p_rows if r['gidx'] == i)
            print(f'  Hyb Reprompt+ {proxy_name} gidx={i:03d}: native_GS={src_gs}/9 | Hyb_CommitSteer={src_ours}/9', flush=True)

        gidxs_u = sorted({r['gidx'] for r in p_rows})
        rp_results[proxy_name] = dict(
            n_candidates=len(p_rows), n_sources=len(gidxs_u),
            native_gs_cand=sum(r['auth_ok'] for r in p_rows),
            native_gs_src=sum(any(r['auth_ok'] for r in p_rows if r['gidx'] == g) for g in gidxs_u),
            commit_steer_cand=sum(r['joint_accept'] for r in p_rows),
            commit_steer_src=sum(any(r['joint_accept'] for r in p_rows if r['gidx'] == g) for g in gidxs_u),
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
    write_json(atk_dir / 'summary.json', summary)
    write_json(atk_dir / 'reprompt_plus_rows.json', rp_results)
    print(f"[S4B Attacks DONE] {summary['elapsed_seconds']:.1f}s | Imprint150 GS={imprint_agg['step_150']['native_gs_fooled']}/4 Hyb={imprint_agg['step_150']['commit_steer_fooled']}/4 | RP+_sd21 Hyb={rp_results['sd21']['commit_steer_cand']}/54 | RP+_sd15 Hyb={rp_results['sd15']['commit_steer_cand']}/54", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'gs100', 'null2000', 'attacks', 'all'])
    ap.add_argument('--gpu', type=int, default=2)
    args = ap.parse_args()

    if args.mode == 'prepare':
        do_prepare()
        return

    if args.mode == 'all':
        do_prepare()

    uuid = check_idle_gpu(args.gpu)
    torch, F, provider, pipe, sscd, clip_model, clip_proc, lpips_fn = init_env_and_models(uuid)

    if args.mode in ('gs100', 'all'):
        run_gs100(torch, F, provider, pipe, sscd, clip_model, clip_proc, lpips_fn, args.gpu, uuid)
    if args.mode in ('null2000', 'all'):
        run_null2000(torch, F, provider, sscd, clip_model, args.gpu, uuid)
    if args.mode in ('attacks', 'all'):
        run_attacks(torch, F, provider, sscd, clip_model, lpips_fn, args.gpu, uuid)


if __name__ == '__main__':
    main()
