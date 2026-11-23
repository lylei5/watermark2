"""Probe C (GPU 5): Cross-Carrier Proof on Native 512-bit PRC Watermark
(448-bit attribution payload + 64-bit self-contained 8-step preview HMAC-keyed SSCD commitment)
with Mid-Step Jacobian-Free Tweedie Guidance, 8 UNSEEN prompts, benign edits, and live Reprompt defense.
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

sys.path.insert(0, '/data0/liyanlei/watermark_research/envs/phase1_prc_extra')

import numpy as np
from PIL import Image
from scipy.special import erf

ROOT = Path('/data0/liyanlei/watermark_research')
RUN = ROOT / 'runs/commit_steer_stage2_20261122/probe_c_prc512'
STAGE0 = ROOT / 'runs/payload_binding_stage0_20261122'
CJ_RUN = ROOT / 'runs/color_joint_20261119'
PRC_MOD_PATH = ROOT / 'third_party/frozen/SemBind_337c9f9/src/prc.py'
GS_DIST_PATH = ROOT / 'runs/defense_revalidation_20261111/gs_image_utils.py'
PGID_PIPE = ROOT / 'third_party/frozen/PGID_dc4ad96'
EXTRA_DEPS = ROOT / 'envs/phase1_extra'
SSCD_PATH = ROOT / 'models/sscd/sscd_disc_mixup.torchscript.pt'

ALL_GS_KEYS = [
    'random_crop_ratio', 'random_drop_ratio', 'brightness_factor',
    'jpeg_ratio', 'gaussian_blur_r', 'gaussian_std',
    'median_blur_k', 'sp_prob', 'resize_ratio',
]
EDITS = {
    'jpeg_ratio': 25,
    'brightness_factor': 6,
    'gaussian_blur_r': 4,
    'gaussian_std': 0.05,
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


def make_keyed_projection(seed_bytes: bytes, k: int = 64) -> np.ndarray:
    seed = int.from_bytes(hmac.digest(seed_bytes, b'commit-steer-prc512-sscd64-v1', 'sha256')[:16], 'big')
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((512, k)))
    return q.astype(np.float64)


def main():
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 5
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

    sys.path.insert(0, str(ROOT / 'envs/phase1_prc_extra'))
    sys.path.insert(0, str(PGID_PIPE))
    sys.path.insert(0, str(EXTRA_DEPS))
    from pipe import pipe_utils

    prc = load_module('prc_native', PRC_MOD_PATH)
    gs_dist = load_module('gs_dist', GS_DIST_PATH)

    th = read_json(STAGE0 / 'summary_sscd.json')['thresholds_95']['sscd_64b']

    # 8 UNSEEN confirmation prompts (4 COCO + 4 SDP from color_joint_20261119, indices 8..11 so disjoint even from Probe A first 4!)
    cj_sources = read_json(CJ_RUN / 'sources.json')
    selected = [s for s in cj_sources if s['dataset'] == 'COCO'][8:12] + [s for s in cj_sources if s['dataset'] == 'SDP'][8:12]
    assert len(selected) == 8

    # Native 512-bit PRC KeyGen (n = 4*64*64 = 16384, message_length = 512, fpr = 1e-6, t = 3)
    print('Running native PRC KeyGen (n=16384, message_length=512, fpr=1e-6)...', flush=True)
    np.random.seed(2026112205)
    enc_key, dec_key = prc.KeyGen(16384, message_length=512, false_positive_rate=1e-6, t=3)
    rng_priv = np.random.default_rng(2026112205)
    auth_448 = rng_priv.integers(0, 2, size=(448,), dtype=np.uint8)
    P64_np = make_keyed_projection(np.packbits(auth_448).tobytes(), 64)
    P64_t = torch.from_numpy(P64_np).float().cuda()

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

    mean_norm = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    std_norm = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)

    def decode_x(z):
        dec = pipe.vae.decode((z.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor), return_dict=False)[0]
        return (dec.float() / 2.0 + 0.5).clamp(0.0, 1.0)

    def to_pil(x):
        return Image.fromarray(x.detach()[0].permute(1, 2, 0).mul(255.0).round().byte().cpu().numpy())

    def eval_sscd_pil(im):
        x = torch.from_numpy(np.asarray(im.convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()
        x_in = (F.interpolate(x, size=(320, 320), mode='bilinear', align_corners=False) - mean_norm) / std_norm
        with torch.no_grad():
            return F.normalize(sscd(x_in), dim=1)[0].cpu().numpy().astype(np.float64)

    def diff_sscd_loss(x, ts_t):
        x_in = (F.interpolate(x, size=(320, 320), mode='bilinear', align_corners=False) - mean_norm) / std_norm
        proj = F.normalize(sscd(x_in), dim=1)[0] @ P64_t
        cos_d = 1.0 - F.cosine_similarity(proj.unsqueeze(0), ts_t.unsqueeze(0)).squeeze()
        margin = F.softplus(0.05 - proj * ts_t, beta=20.0).mean()
        return cos_d + 0.5 * margin

    def ddim_gen(prompt, zT, num_steps=50, ts_t=None, step_range=(5, 35), step_size=0.035, stop_loss=0.10):
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
            inp = clip_proc(text=[prompt], images=[cand_im], return_tensors='pt', padding=True, truncation=True, max_length=77).to('cuda')
            out = clip_model(**inp)
            cs = float((F.normalize(out.image_embeds, dim=-1) * F.normalize(out.text_embeds, dim=-1)).sum().item())
        return lp, cs

    def prc_posteriors(zT_inv: torch.Tensor) -> torch.Tensor:
        var = 1.5
        denom = float(np.sqrt(2.0 * var * (1.0 + var)))
        z_flat = zT_inv.detach().float().cpu().reshape(-1).numpy()
        return torch.from_numpy(erf(z_flat / denom).astype(np.float64))

    def verify_prc_blind(im, true_msg_512):
        z_inv = provider.invert_images(im, num_inference_steps=50)['zT_torch']
        post = prc_posteriors(z_inv)
        detected = bool(prc.Detect(dec_key, post))
        rec = prc.Decode(dec_key, post)
        decoded_ok = rec is not None
        if decoded_ok:
            rec_bits = np.asarray(rec, dtype=np.uint8)[:512]
            auth_exact = bool(np.array_equal(rec_bits[:448], true_msg_512[:448]))
            p_ber = float((rec_bits[448:512] != true_msg_512[448:512]).mean())
            rec_sgn = 2.0 * rec_bits[448:512].astype(np.float64) - 1.0
            qp = eval_sscd_pil(im) @ P64_np
            d_blind = float(1.0 - np.dot(rec_sgn, qp) / (8.0 * float(np.linalg.norm(qp)) + 1e-12))
            feat_ok = d_blind <= th
        else:
            auth_exact = False
            p_ber = 1.0
            d_blind = 2.0
            feat_ok = False
        return dict(
            prc_detect=detected, prc_decode=decoded_ok, auth_448_exact=auth_exact,
            payload_64_ber=p_ber, d_blind=d_blind, feat_ok=bool(feat_ok),
            joint_blind_ok=bool(detected and decoded_ok and auth_exact and feat_ok),
        )

    with torch.no_grad():
        pipe('warmup', generator=torch.Generator(device='cuda').manual_seed(7), num_inference_steps=2)

    rp_proto = read_json(ROOT / 'runs/author_reprompt_plus_20261115/protocol.json')
    i2p_prompts = [r['target_prompt'] for r in rp_proto['references'] if r['carrier'] == 'GS'][:8]

    t_start = time.perf_counter()
    records = []
    reprompt_rows = []
    img_root = RUN / 'images'
    img_root.mkdir(exist_ok=True)

    for idx, s in enumerate(selected):
        task, prompt = s['task'], s['prompt']
        tdir = img_root / f'{task:03d}'
        tdir.mkdir(exist_ok=True)

        # Self-contained 8-step preview
        t0 = time.perf_counter()
        zT_prev = torch.randn((1, 4, 64, 64), generator=torch.Generator(device='cuda').manual_seed(2026112260 + task), device='cuda', dtype=torch.float16)
        prev_im = ddim_gen(prompt, zT_prev, num_steps=8, ts_t=None)
        torch.cuda.synchronize()
        sec_prev = time.perf_counter() - t0

        p_bits = ((eval_sscd_pil(prev_im) @ P64_np) >= 0.0).astype(np.uint8)
        true_sgn = 2.0 * p_bits.astype(np.float64) - 1.0
        ts_t = torch.tensor(true_sgn, device='cuda', dtype=torch.float32)

        # Native 512-bit PRC Encode: 448 auth/user bits + 64 commitment bits
        msg_512 = np.concatenate([auth_448, p_bits])
        np.random.seed(2026112270 + task)
        cw = prc.Encode(enc_key, msg_512).numpy().astype(np.float64)  # in {-1, +1}^16384
        zT_np = cw * np.abs(np.random.randn(16384))
        zT = torch.from_numpy(zT_np.reshape(1, 4, 64, 64)).to(device='cuda', dtype=torch.float16)

        t0 = time.perf_counter()
        base_im = ddim_gen(prompt, zT, num_steps=50, ts_t=None)
        torch.cuda.synchronize()
        sec_base = time.perf_counter() - t0
        base_im.save(tdir / 'prc_unguided.png')

        t0 = time.perf_counter()
        guid_im = ddim_gen(prompt, zT, num_steps=50, ts_t=ts_t, step_range=(5, 35), step_size=0.035, stop_loss=0.10)
        torch.cuda.synchronize()
        sec_guid = time.perf_counter() - t0
        guid_im.save(tdir / 'prc_commit_steer.png')

        for arm_name, arm_im, arm_sec in [('prc_unguided', base_im, sec_base), ('prc_commit_steer', guid_im, sec_prev + sec_guid)]:
            lp, cs = eval_lpips_clip(base_im, arm_im, prompt)
            for ci, cond in enumerate(['clean', *EDITS.keys()]):
                seed = 2026112200 + task * 100 + ci
                if cond == 'clean':
                    q_im = arm_im
                else:
                    gs_dist.set_random_seed(seed)
                    dargs = SimpleNamespace(**{k: None for k in ALL_GS_KEYS})
                    setattr(dargs, cond, EDITS[cond])
                    q_im = gs_dist.image_distortion(arm_im.copy(), seed, dargs).convert('RGB')
                v = verify_prc_blind(q_im, msg_512)
                records.append(dict(task=task, dataset=s['dataset'], arm=arm_name, condition=cond,
                                    seconds=arm_sec, lpips=lp, clip=cs, **v))

        # Live Reprompt forgery on the guided PRC-512b image (first 4 prompts)
        if idx < 4:
            zT_atk = provider.invert_images(guid_im, num_inference_steps=50)['zT_torch']
            forged_im = provider.generate([i2p_prompts[idx]], num_inference_steps=50, guidance_scale=7.5, latents=zT_atk, return_latents=False)['images_PIL'][0].convert('RGB')
            forged_im.save(tdir / 'prc_reprompt_forged.png')
            vf = verify_prc_blind(forged_im, msg_512)
            reprompt_rows.append(dict(task=task, **vf))

        print(f'[Probe C PRC-512b | GPU {gpu}] {idx+1}/8 task={task:03d} done ({time.perf_counter()-t_start:.1f}s)', flush=True)

    summary = {}
    for arm in ('prc_unguided', 'prc_commit_steer'):
        sub = [r for r in records if r['arm'] == arm]
        cl = [r for r in sub if r['condition'] == 'clean']
        ed = [r for r in sub if r['condition'] != 'clean']
        summary[arm] = dict(
            seconds_mean=float(np.mean([r['seconds'] for r in cl])),
            lpips_mean=float(np.mean([r['lpips'] for r in cl])),
            clip_mean=float(np.mean([r['clip'] for r in cl])),
            clean_prc_detect=sum(r['prc_detect'] for r in cl),
            clean_prc_decode=sum(r['prc_decode'] for r in cl),
            clean_auth_exact=sum(r['auth_448_exact'] for r in cl),
            clean_joint=sum(r['joint_blind_ok'] for r in cl),
            edited_n=len(ed),
            edited_prc_detect=sum(r['prc_detect'] for r in ed),
            edited_prc_decode=sum(r['prc_decode'] for r in ed),
            edited_joint=sum(r['joint_blind_ok'] for r in ed),
            by_cond={c: dict(
                n=len([r for r in sub if r['condition'] == c]),
                detect=sum(r['prc_detect'] for r in sub if r['condition'] == c),
                decode=sum(r['prc_decode'] for r in sub if r['condition'] == c),
                joint=sum(r['joint_blind_ok'] for r in sub if r['condition'] == c),
            ) for c in ['clean', *EDITS.keys()]},
        )

    rp_sum = dict(
        n=len(reprompt_rows),
        native_prc_detect_fooled=sum(r['prc_detect'] for r in reprompt_rows),
        native_prc_decode_fooled=sum(r['prc_decode'] and r['auth_448_exact'] for r in reprompt_rows),
        commit_steer_fooled=sum(r['joint_blind_ok'] for r in reprompt_rows),
        rows=reprompt_rows,
    )
    out = dict(elapsed_seconds=time.perf_counter() - t_start, gpu=gpu, uuid=uuid, threshold=th, arms=summary, live_reprompt=rp_sum)
    write_json(RUN / 'records.json', records)
    write_json(RUN / 'summary.json', out)
    print(f'PROBE C (PRC-512b) DONE in {out["elapsed_seconds"]:.1f}s | clean detect={summary["prc_commit_steer"]["clean_prc_detect"]}/8 decode={summary["prc_commit_steer"]["clean_prc_decode"]}/8 joint={summary["prc_commit_steer"]["clean_joint"]}/8 | LPIPS={summary["prc_commit_steer"]["lpips_mean"]:.3f} | ed_joint={summary["prc_commit_steer"]["edited_joint"]}/32 | live_RP native={rp_sum["native_prc_detect_fooled"]}/4 ours={rp_sum["commit_steer_fooled"]}/4', flush=True)


if __name__ == '__main__':
    main()
