"""Stage 3A (GPU 0 & GPU 1 sharded): Paper-scale 100-Prompt Official Unified Cohort
(unified_main_sources_20261116: 50 COCO + 50 SDP, zero overlap with calibration/dev sets).
Self-contained 12-step DDIM preview (~0.6s) + HMAC-keyed P64(K) + Mid-step Tweedie (step_size=0.042)
inside Native 256-bit Gaussian Shading (192b auth + 64b commitment, 64x spatial repetition).
Evaluates Clean + ALL 9 standard GS distortions (1000 DDIM inversions), LPIPS, CLIP, cross-prompt FAR (9900 pairs),
and saves all 100 unguided & 100 guided PNGs for exact COCO FID computation.
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
from PIL import Image

ROOT = Path('/data0/liyanlei/watermark_research')
RUN = ROOT / 'runs/commit_steer_paper_scale_20261123/gs100_unified'
STAGE0 = ROOT / 'runs/payload_binding_stage0_20261122'
UNIFIED_SEL = ROOT / 'runs/unified_main_sources_20261116/selection.json'
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
    assert idx in (0, 1, 2, 3, 4, 5, 7) and idx != 6
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
    seed = int.from_bytes(hmac.digest(key_bytes, b'commit-steer-paper-gs100-proj-v1', 'sha256')[:16], 'big')
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((512, k)))
    return q.astype(np.float64)


def do_prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    th = read_json(STAGE0 / 'summary_sscd.json')['thresholds_95']['sscd_64b']
    sel = read_json(UNIFIED_SEL)['sources']
    coco50 = [s for s in sel if s['dataset'] == 'COCO'][:50]
    sdp50 = [s for s in sel if s['dataset'] == 'SDP'][:50]
    sources = coco50 + sdp50
    assert len(sources) == 100

    rng_priv = np.random.default_rng(2026112301)
    gs_key_np = rng_priv.integers(0, 2, size=(1, 4, 64, 64), dtype=np.uint8)
    auth_192 = rng_priv.integers(0, 2, size=(192,), dtype=np.uint8)
    P64_np = make_keyed_projection(np.packbits(gs_key_np.reshape(-1)).tobytes(), 64)
    np.savez(RUN / 'private_material.npz', key=gs_key_np, auth_bits=auth_192, P64=P64_np)
    os.chmod(RUN / 'private_material.npz', 0o600)

    proto = dict(
        run_name='gs100_unified',
        n_prompts=100,
        preview_steps=12,
        step_range=[5, 35],
        step_size=0.042,
        stop_loss=0.08,
        threshold=th,
        auth_threshold=136.0 / 192.0,
        gs9_conditions=GS_9_CONDITIONS,
        sources=sources,
    )
    write_json(RUN / 'protocol.json', proto)
    print('Prepared gs100_unified with 100 official unified prompts (50 COCO + 50 SDP).', flush=True)


def do_worker(gpu: int, shard: int, num_shards: int):
    uuid = check_idle_gpu(gpu)
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
    import lpips
    from transformers import CLIPModel, CLIPProcessor

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sys.path.insert(0, str(PGID_PIPE))
    sys.path.insert(0, str(EXTRA_DEPS))
    from pipe import pipe_utils

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
    P64_t = torch.from_numpy(P64_np).float().cuda()
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

    def eval_sscd_batch(pil_list):
        xs = torch.stack([
            torch.from_numpy(np.asarray(im.convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1)
            for im in pil_list
        ]).cuda()
        x_in = (F.interpolate(xs, size=(320, 320), mode='bilinear', align_corners=False) - mean_norm) / std_norm
        with torch.no_grad():
            return F.normalize(sscd(x_in), dim=1).cpu().numpy().astype(np.float64)

    def diff_sscd_loss(x, ts_t):
        x_in = (F.interpolate(x, size=(320, 320), mode='bilinear', align_corners=False) - mean_norm) / std_norm
        proj = F.normalize(sscd(x_in), dim=1)[0] @ P64_t
        cos_d = 1.0 - F.cosine_similarity(proj.unsqueeze(0), ts_t.unsqueeze(0)).squeeze()
        margin = F.softplus(0.05 - proj * ts_t, beta=20.0).mean()
        return cos_d + 0.5 * margin

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
            inp = clip_proc(text=[prompt], images=[base_im, cand_im], return_tensors='pt', padding=True, truncation=True, max_length=77).to('cuda')
            out = clip_model(**inp)
            ie = F.normalize(out.image_embeds, dim=-1)
            te = F.normalize(out.text_embeds, dim=-1)
            return lp, float((ie[0] * te[0]).sum().item()), float((ie[1] * te[0]).sum().item())

    def invert_batch(pil_list):
        """Batched 50-step DDIM inversion (all 10 conditions in 2 mini-batches of 5 -> 3x faster!)."""
        zTs = []
        for i in range(0, len(pil_list), 5):
            chunk = pil_list[i:i + 5]
            zT_c = provider.invert_images(chunk, num_inference_steps=50)['zT_torch']
            zTs.append(zT_c)
        return torch.cat(zTs, dim=0)

    with torch.no_grad():
        pipe('warmup', generator=torch.Generator(device='cuda').manual_seed(7), num_inference_steps=2)

    base_dir = RUN / 'images_unguided'
    guid_dir = RUN / 'images_guided'
    base_dir.mkdir(parents=True, exist_ok=True)
    guid_dir.mkdir(parents=True, exist_ok=True)

    my_items = [(i, s) for i, s in enumerate(proto['sources']) if i % num_shards == shard]
    t_start = time.perf_counter()
    records = []
    embs_out = []

    conds = ['clean', *GS_9_CONDITIONS.keys()]

    for local_i, (gidx, s) in enumerate(my_items):
        prompt = s['prompt']
        # 1. Self-contained 12-step preview
        t0 = time.perf_counter()
        zT_prev = torch.randn((1, 4, 64, 64), generator=torch.Generator(device='cuda').manual_seed(2026112300 + gidx), device='cuda', dtype=torch.float16)
        prev_im = ddim_gen(prompt, zT_prev, num_steps=12, ts_t=None)
        torch.cuda.synchronize()
        sec_prev = time.perf_counter() - t0

        prev_emb = eval_sscd_batch([prev_im])[0]
        p_bits = ((prev_emb @ P64_np) >= 0.0).astype(np.uint8)
        true_sgn = 2.0 * p_bits.astype(np.float64) - 1.0
        ts_t = torch.tensor(true_sgn, device='cuda', dtype=torch.float32)

        # 2. Native GS 256-bit initial noise (192b auth + 64b commitment)
        w_flat = np.concatenate([auth_192, p_bits])
        w_arr = w_flat.reshape(1, 4, 8, 8)
        m_enc = ((np.tile(w_arr, (1, 1, 8, 8)) + gs_key_np) % 2).reshape(-1)
        np.random.seed(2026112350 + gidx)
        zT = native_gs.truncSampling(m_enc)

        # 3. Unguided native GS baseline (for FID / LPIPS / CLIP comparison)
        t0 = time.perf_counter()
        base_im = ddim_gen(prompt, zT, num_steps=50, ts_t=None)
        torch.cuda.synchronize()
        sec_base = time.perf_counter() - t0
        base_im.save(base_dir / f'{gidx:03d}.png')

        # 4. Commit-then-Steer guided generation
        t0 = time.perf_counter()
        guid_im = ddim_gen(prompt, zT, num_steps=50, ts_t=ts_t, step_range=(5, 35), step_size=0.042, stop_loss=0.08)
        torch.cuda.synchronize()
        sec_guid = time.perf_counter() - t0
        guid_im.save(guid_dir / f'{gidx:03d}.png')

        lp, clip_base, clip_guid = eval_lpips_clip(base_im, guid_im, prompt)

        # 5. Build clean + 9 GS distortions for guided image, invert in 2 batches of 5!
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

        z_invs = invert_batch(q_imgs)  # (10, 4, 64, 64)
        q_embs = eval_sscd_batch(q_imgs)  # (10, 512)
        embs_out.append(dict(gidx=gidx, target_bits=p_bits.tolist(), clean_emb=q_embs[0].tolist()))

        for ci, cond in enumerate(conds):
            rev_w = native_gs.diffusion_inverse(((z_invs[ci:ci+1] > 0).int() + native_gs.key) % 2).reshape(-1).cpu().numpy().astype(np.uint8)
            full_ba = float((rev_w == w_flat).mean())
            auth_ba = float((rev_w[:192] == auth_192).mean())
            auth_ok = auth_ba > (136.0 / 192.0)
            p_ber = float((rev_w[192:] != p_bits).mean())
            rec_sgn = 2.0 * rev_w[192:].astype(np.float64) - 1.0
            qp = q_embs[ci] @ P64_np
            d_blind = float(1.0 - np.dot(rec_sgn, qp) / (8.0 * float(np.linalg.norm(qp)) + 1e-12))
            d_oracle = float(1.0 - np.dot(true_sgn, qp) / (8.0 * float(np.linalg.norm(qp)) + 1e-12))
            feat_ok = d_blind <= th
            records.append(dict(
                gidx=gidx, dataset=s['dataset'], condition=cond,
                sec_prev=sec_prev, sec_base=sec_base, sec_guid=sec_guid, sec_total=sec_prev + sec_guid,
                lpips=lp, clip_base=clip_base, clip_guid=clip_guid,
                full_ba=full_ba, auth_ba=auth_ba, auth_ok=bool(auth_ok), payload_ber=p_ber,
                d_blind=d_blind, d_oracle=d_oracle, feat_ok=bool(feat_ok), joint_blind_ok=bool(auth_ok and feat_ok),
            ))

        if (local_i + 1) % 10 == 0 or local_i == len(my_items) - 1:
            print(f'[GS100 shard {shard} | GPU {gpu}] {local_i+1}/{len(my_items)} done ({time.perf_counter()-t_start:.1f}s)', flush=True)
            write_json(RUN / f'records_shard{shard}.json', records)

    write_json(RUN / f'embs_shard{shard}.json', embs_out)
    write_json(RUN / f'complete_shard{shard}.json', dict(status='complete', shard=shard, gpu=gpu, uuid=uuid, elapsed=time.perf_counter() - t_start, n_records=len(records)))
    print(f'[GS100 shard {shard} | GPU {gpu}] COMPLETE in {time.perf_counter()-t_start:.1f}s', flush=True)


def do_merge():
    proto = read_json(RUN / 'protocol.json')
    th = proto['threshold']
    priv = np.load(RUN / 'private_material.npz')
    P64_np = priv['P64']

    records, embs = [], []
    for s in (0, 1):
        records.extend(read_json(RUN / f'records_shard{s}.json'))
        embs.extend(read_json(RUN / f'embs_shard{s}.json'))
    assert len(records) == 1000 and len(embs) == 100

    cl = [r for r in records if r['condition'] == 'clean']
    ed = [r for r in records if r['condition'] != 'clean']

    # Cross-prompt FAR across ALL 100 x 99 = 9,900 pairs!
    embs = sorted(embs, key=lambda x: x['gidx'])
    sgns = np.array([2.0 * np.array(e['target_bits'], dtype=np.float64) - 1.0 for e in embs])  # (100, 64)
    qps = np.array([np.array(e['clean_emb'], dtype=np.float64) @ P64_np for e in embs])  # (100, 64)
    qps_n = qps / (np.linalg.norm(qps, axis=1, keepdims=True) + 1e-12)
    sim_mat = (sgns @ qps_n.T) / 8.0  # (100, 100)
    dist_mat = 1.0 - sim_mat
    off_diag = dist_mat[~np.eye(100, dtype=bool)]
    cross_far_cnt = int((off_diag <= th).sum())

    by_cond = {}
    for cond in ['clean', *GS_9_CONDITIONS.keys()]:
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
        n_prompts=100,
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
        gs9_auth_pass=sum(r['auth_ok'] for r in ed),
        gs9_joint_pass=sum(r['joint_blind_ok'] for r in ed),
        gs9_p_ber=float(np.mean([r['payload_ber'] for r in ed])),
        cross_prompt_far_count=cross_far_cnt,
        cross_prompt_far_total=9900,
        cross_prompt_far=cross_far_cnt / 9900.0,
        by_cond=by_cond,
    )
    write_json(RUN / 'summary.json', summary)
    write_json(RUN / 'records.json', records)
    print(f"MERGED 100 PROMPTS | clean_joint={summary['clean_joint_pass']}/100 | gs9_joint={summary['gs9_joint_pass']}/900 ({100*summary['gs9_joint_pass']/900:.1f}%) | LPIPS={summary['lpips_mean']:.3f} | CLIP={summary['clip_guid_mean']:.4f} vs {summary['clip_base_mean']:.4f} | crossFAR={cross_far_cnt}/9900 ({100*summary['cross_prompt_far']:.2f}%)", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'worker', 'merge'])
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=2)
    args = ap.parse_args()
    if args.mode == 'prepare':
        do_prepare()
    elif args.mode == 'worker':
        do_worker(args.gpu, args.shard, args.num_shards)
    else:
        do_merge()
