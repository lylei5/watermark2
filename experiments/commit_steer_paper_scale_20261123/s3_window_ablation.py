"""Stage 3E (GPU 1): Controlled Steering-Window Ablation on 12 Disjoint Official Unified Prompts
Proves why Mid-Step Manifold-Consistent Steering [5, 35] (leaving 14 unguided final DDIM steps)
dominates Early-Only [0, 20], No-Tail [5, 49], and Late-Only [25, 49] in DDIM invertibility (BA/BER),
perceptual quality (LPIPS/CLIP), and benign-edit survival (Clean + 4 GS edits).
"""
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
RUN = ROOT / 'runs/commit_steer_paper_scale_20261123/window_ablation'
STAGE0 = ROOT / 'runs/payload_binding_stage0_20261122'
UNIFIED_SEL = ROOT / 'runs/unified_main_sources_20261116/selection.json'
GS_MOD_PATH = ROOT / 'runs/phase1_trajectory_native_gs_20261113/gs_watermark.py'
GS_DIST_PATH = ROOT / 'runs/defense_revalidation_20261111/gs_image_utils.py'
PGID_PIPE = ROOT / 'third_party/frozen/PGID_dc4ad96'
EXTRA_DEPS = ROOT / 'envs/phase1_extra'
SSCD_PATH = ROOT / 'models/sscd/sscd_disc_mixup.torchscript.pt'

ALL_GS_KEYS = [
    'random_crop_ratio', 'random_drop_ratio', 'brightness_factor',
    'jpeg_ratio', 'gaussian_blur_r', 'gaussian_std',
    'median_blur_k', 'sp_prob', 'resize_ratio',
]
EDITS_4 = {'jpeg_ratio': 25, 'brightness_factor': 6, 'gaussian_blur_r': 4, 'resize_ratio': 0.25}
ARMS = [
    ('early_0_20', (0, 20)),
    ('mid_5_35', (5, 35)),
    ('notail_5_49', (5, 49)),
    ('late_25_49', (25, 49)),
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


def main():
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 1
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

    th = read_json(STAGE0 / 'summary_sscd.json')['thresholds_95']['sscd_64b']
    rng = np.random.default_rng(2026112377)
    gs_key_np = rng.integers(0, 2, size=(1, 4, 64, 64), dtype=np.uint8)
    auth_192 = rng.integers(0, 2, size=(192,), dtype=np.uint8)
    seed_p = int.from_bytes(hmac.digest(np.packbits(gs_key_np.reshape(-1)).tobytes(), b'window-ablation-v1', 'sha256')[:16], 'big')
    P64_np, _ = np.linalg.qr(np.random.default_rng(seed_p).standard_normal((512, 64)))
    P64_np = P64_np.astype(np.float64)
    P64_t = torch.from_numpy(P64_np).float().cuda()

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

    mean_n = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    std_n = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)

    def decode_x(z):
        return (pipe.vae.decode((z.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor), return_dict=False)[0].float() / 2.0 + 0.5).clamp(0.0, 1.0)

    def to_pil(x):
        return Image.fromarray(x.detach()[0].permute(1, 2, 0).mul(255.0).round().byte().cpu().numpy())

    def eval_sscd_batch(pil_list):
        xs = torch.stack([torch.from_numpy(np.asarray(im.convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1) for im in pil_list]).cuda()
        x_in = (F.interpolate(xs, size=(320, 320), mode='bilinear', align_corners=False) - mean_n) / std_n
        with torch.no_grad():
            return F.normalize(sscd(x_in), dim=1).cpu().numpy().astype(np.float64)

    def diff_sscd_loss(x, ts_t):
        x_in = (F.interpolate(x, size=(320, 320), mode='bilinear', align_corners=False) - mean_n) / std_n
        proj = F.normalize(sscd(x_in), dim=1)[0] @ P64_t
        return (1.0 - F.cosine_similarity(proj.unsqueeze(0), ts_t.unsqueeze(0)).squeeze()) + 0.5 * F.softplus(0.05 - proj * ts_t, beta=20.0).mean()

    def ddim_gen(prompt, zT, num_steps=50, ts_t=None, step_range=(5, 35), step_size=0.042, stop_loss=0.08):
        pipe.scheduler = provider.scheduler
        pipe.scheduler.set_timesteps(num_steps, device='cuda')
        with torch.no_grad():
            pos, neg = pipe.encode_prompt(prompt=prompt, device='cuda', num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt='')
            te = torch.cat([neg, pos], dim=0)
        z = zT.clone().to(device='cuda', dtype=torch.float16)
        alphas = pipe.scheduler.alphas_cumprod.to('cuda')
        for idx, t in enumerate(pipe.scheduler.timesteps):
            with torch.no_grad():
                eps = pipe.unet(pipe.scheduler.scale_model_input(torch.cat([z, z], dim=0), t), t, encoder_hidden_states=te, return_dict=False)[0]
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

    def pil_m11(im):
        return torch.from_numpy(np.asarray(im, dtype=np.float32).transpose(2, 0, 1)).unsqueeze(0).cuda() / 127.5 - 1.0

    sel = read_json(UNIFIED_SEL)['sources']
    sources = [s for s in sel if s['dataset'] == 'COCO'][70:76] + [s for s in sel if s['dataset'] == 'SDP'][70:76]
    assert len(sources) == 12

    t0 = time.perf_counter()
    records = []
    for idx, s in enumerate(sources):
        prompt = s['prompt']
        zT_prev = torch.randn((1, 4, 64, 64), generator=torch.Generator(device='cuda').manual_seed(2026112370 + idx), device='cuda', dtype=torch.float16)
        prev_im = ddim_gen(prompt, zT_prev, num_steps=12)
        p_bits = ((eval_sscd_batch([prev_im])[0] @ P64_np) >= 0.0).astype(np.uint8)
        ts_t = torch.tensor(2.0 * p_bits.astype(np.float64) - 1.0, device='cuda', dtype=torch.float32)

        w_flat = np.concatenate([auth_192, p_bits])
        m_enc = ((np.tile(w_flat.reshape(1, 4, 8, 8), (1, 1, 8, 8)) + gs_key_np) % 2).reshape(-1)
        np.random.seed(2026112375 + idx)
        zT = native_gs.truncSampling(m_enc)
        base_im = ddim_gen(prompt, zT, num_steps=50)

        for arm_name, srange in ARMS:
            guid_im = ddim_gen(prompt, zT, num_steps=50, ts_t=ts_t, step_range=srange, step_size=0.042, stop_loss=0.08)
            with torch.no_grad():
                lp = float(lpips_fn(pil_m11(base_im), pil_m11(guid_im)).item())
                inp = clip_proc(text=[prompt], images=[base_im, guid_im], return_tensors='pt', padding=True, truncation=True, max_length=77).to('cuda')
                co = clip_model(**inp)
                ie, te = F.normalize(co.image_embeds, dim=-1), F.normalize(co.text_embeds, dim=-1)
                cb, cg = float((ie[0] * te[0]).sum().item()), float((ie[1] * te[0]).sum().item())

            q_imgs, conds = [guid_im], ['clean']
            for ci, (cname, cval) in enumerate(EDITS_4.items(), start=1):
                seed = 20261123 + idx * 100 + ci
                gs_dist.set_random_seed(seed)
                dargs = SimpleNamespace(**{k: None for k in ALL_GS_KEYS})
                setattr(dargs, cname, cval)
                q_imgs.append(gs_dist.image_distortion(guid_im.copy(), seed, dargs).convert('RGB'))
                conds.append(cname)

            z_invs = provider.invert_images(q_imgs, num_inference_steps=50)['zT_torch']
            q_embs = eval_sscd_batch(q_imgs)
            for ci, cond in enumerate(conds):
                rev_w = native_gs.diffusion_inverse(((z_invs[ci:ci+1] > 0).int() + native_gs.key) % 2).reshape(-1).cpu().numpy().astype(np.uint8)
                full_ba = float((rev_w == w_flat).mean())
                auth_ok = float((rev_w[:192] == auth_192).mean()) > (136.0 / 192.0)
                p_ber = float((rev_w[192:] != p_bits).mean())
                rec_sgn = 2.0 * rev_w[192:].astype(np.float64) - 1.0
                qp = q_embs[ci] @ P64_np
                d_b = float(1.0 - np.dot(rec_sgn, qp) / (8.0 * float(np.linalg.norm(qp)) + 1e-12))
                records.append(dict(idx=idx, arm=arm_name, condition=cond, lpips=lp, clip_base=cb, clip_guid=cg,
                                    full_ba=full_ba, auth_ok=bool(auth_ok), p_ber=p_ber, d_blind=d_b, joint=bool(auth_ok and d_b <= th)))
        if (idx + 1) % 4 == 0:
            print(f'[WindowAbl | GPU {gpu}] {idx+1}/12 done ({time.perf_counter()-t0:.1f}s)', flush=True)

    summary = {}
    for arm_name, srange in ARMS:
        cl = [r for r in records if r['arm'] == arm_name and r['condition'] == 'clean']
        ed = [r for r in records if r['arm'] == arm_name and r['condition'] != 'clean']
        summary[arm_name] = dict(
            step_range=list(srange),
            lpips=float(np.mean([r['lpips'] for r in cl])),
            clip_guid=float(np.mean([r['clip_guid'] for r in cl])),
            clip_base=float(np.mean([r['clip_base'] for r in cl])),
            clean_full_ba=float(np.mean([r['full_ba'] for r in cl])),
            clean_p_ber=float(np.mean([r['p_ber'] for r in cl])),
            clean_d_blind=float(np.mean([r['d_blind'] for r in cl])),
            clean_joint=f"{sum(r['joint'] for r in cl)}/12",
            edit4_full_ba=float(np.mean([r['full_ba'] for r in ed])),
            edit4_p_ber=float(np.mean([r['p_ber'] for r in ed])),
            edit4_d_blind=float(np.mean([r['d_blind'] for r in ed])),
            edit4_joint=f"{sum(r['joint'] for r in ed)}/48",
        )
        s = summary[arm_name]
        print(f"  {arm_name:12s} | LPIPS={s['lpips']:.3f} CLIP={s['clip_guid']:.4f} | clean BA={s['clean_full_ba']:.4f} BER={s['clean_p_ber']:.4f} d={s['clean_d_blind']:.3f} joint={s['clean_joint']} | edit4 BA={s['edit4_full_ba']:.4f} BER={s['edit4_p_ber']:.4f} joint={s['edit4_joint']}", flush=True)

    write_json(RUN / 'summary.json', dict(elapsed_seconds=time.perf_counter() - t0, gpu=gpu, uuid=uuid, arms=summary))
    write_json(RUN / 'records.json', records)
    print(f"WINDOW ABLATION DONE in {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
