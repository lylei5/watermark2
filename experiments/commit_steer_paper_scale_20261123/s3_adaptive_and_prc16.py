"""Stage 3C (GPU 5):
Part 1: White-Box Adaptive SSCD Collision Stress Test (Public P64 vs Secret HMAC-Keyed P64(K))
        measuring attacker visual distortion cost (PSNR/LPIPS) & post-JPEG25 survival!
Part 2: Expanded 16-Prompt Cross-Carrier Native 512-bit PRC on Official Unified Cohort
        (12-step preview + HMAC-keyed P64(K) + step_size=0.042 + 4 edits + 8 live Reprompt).
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
RUN = ROOT / 'runs/commit_steer_paper_scale_20261123/adaptive_and_prc16'
STAGE0 = ROOT / 'runs/payload_binding_stage0_20261122'
STAGE1 = ROOT / 'runs/gs_payload_guided_stage1_20261122'
UNIFIED_SEL = ROOT / 'runs/unified_main_sources_20261116/selection.json'
COVER_DIR = ROOT / 'runs/published_imprint_dpm_full_20261110/inputs'
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
PRC_EDITS = {'jpeg_ratio': 25, 'brightness_factor': 6, 'gaussian_blur_r': 4, 'gaussian_std': 0.05}


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


def make_keyed_projection(seed_bytes: bytes, k: int = 64) -> np.ndarray:
    seed = int.from_bytes(hmac.digest(seed_bytes, b'commit-steer-paper-prc16-v1', 'sha256')[:16], 'big')
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

    sys.path.insert(0, str(PGID_PIPE))
    sys.path.insert(0, str(EXTRA_DEPS))
    from pipe import pipe_utils

    prc = load_module('prc_native', PRC_MOD_PATH)
    gs_dist = load_module('gs_dist', GS_DIST_PATH)
    th = read_json(STAGE0 / 'summary_sscd.json')['thresholds_95']['sscd_64b']

    sscd = torch.jit.load(str(SSCD_PATH), map_location='cuda').eval()
    for p in sscd.parameters(): p.requires_grad_(False)
    lpips_fn = lpips.LPIPS(net='alex').cuda().eval()
    mean_norm = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    std_norm = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)

    def sscd_emb_t(x):
        x_in = (F.interpolate(x, size=(320, 320), mode='bilinear', align_corners=False) - mean_norm) / std_norm
        return F.normalize(sscd(x_in), dim=1)[0]

    def pil_to_t(im):
        return torch.from_numpy(np.asarray(im.convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()

    def t_to_pil(x):
        return Image.fromarray(x.detach()[0].permute(1, 2, 0).mul(255.0).round().byte().cpu().numpy())

    # =========================================================================
    # PART 1: WHITE-BOX ADAPTIVE SSCD COLLISION STRESS TEST (Public vs Keyed P64)
    # =========================================================================
    print('=== PART 1: White-Box Adaptive PGD Collision Stress Test (Public P64 vs Secret Keyed P64) ===', flush=True)
    t_all = time.perf_counter()
    proto1 = read_json(STAGE1 / 'protocol.json')
    P64_pub_np = np.load(STAGE0 / 'sscd_projections.npz')['p64'].astype(np.float64)
    P64_pub_t = torch.from_numpy(P64_pub_np).float().cuda()
    covers = [Image.open(COVER_DIR / f'{i:03d}_cover.png').convert('RGB').resize((512, 512), Image.BICUBIC) for i in range(4)]

    adaptive_rows = []
    montage_imgs = {}
    for eps_l_inf in (4 / 255.0, 8 / 255.0, 16 / 255.0, 32 / 255.0):
        for mode in ('public_P64_known', 'secret_keyed_P64_unknown'):
            succ_clean, succ_jpg25, lps, psnrs, d_cleans, d_jpgs = 0, 0, [], [], [], []
            for i in range(4):
                task = proto1['sources'][i]['task']
                wm_im = Image.open(STAGE1 / 'images' / f'{task:03d}/sscd64_mid_tweedie.png').convert('RGB')
                with torch.no_grad():
                    e_wm = sscd_emb_t(pil_to_t(wm_im))  # (512,)
                    # Note: for secret_keyed_P64_unknown, use an independent secret projection P64_sec unknown to attacker
                    P_eval_np = P64_pub_np if mode == 'public_P64_known' else make_keyed_projection(f'secret-case-{i}'.encode(), 64)
                    P_eval_t = torch.from_numpy(P_eval_np).float().cuda()
                    sgn_t = torch.where((e_wm @ P_eval_t) >= 0.0, 1.0, -1.0)
                    sgn_np = sgn_t.cpu().numpy().astype(np.float64)

                x0 = pil_to_t(covers[i])
                delta = torch.zeros_like(x0, requires_grad=True)
                opt = torch.optim.Adam([delta], lr=eps_l_inf / 8.0)
                for _ in range(80):
                    x_adv = (x0 + delta.clamp(-eps_l_inf, eps_l_inf)).clamp(0.0, 1.0)
                    e_adv = sscd_emb_t(x_adv)
                    if mode == 'public_P64_known':
                        proj = e_adv @ P_eval_t
                        loss = 1.0 - F.cosine_similarity(proj.unsqueeze(0), sgn_t.unsqueeze(0)).squeeze()
                    else:
                        # Attacker does NOT know secret P64(K), so must collide full 512-dim public SSCD vector!
                        loss = 1.0 - (e_adv * e_wm.detach()).sum()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    with torch.no_grad():
                        delta.clamp_(-eps_l_inf, eps_l_inf)

                x_fin = (x0 + delta.detach().clamp(-eps_l_inf, eps_l_inf)).clamp(0.0, 1.0)
                adv_im = t_to_pil(x_fin)
                if i == 0:
                    montage_imgs[(int(round(eps_l_inf * 255)), mode)] = adv_im.copy()
                # Evaluate clean & post-JPEG25 under verifier's 64-bit projection!
                with torch.no_grad():
                    e_c = sscd_emb_t(pil_to_t(adv_im)).cpu().numpy().astype(np.float64)
                    qp_c = e_c @ P_eval_np
                    d_c = float(1.0 - np.dot(sgn_np, qp_c) / (8.0 * np.linalg.norm(qp_c) + 1e-12))

                    gs_dist.set_random_seed(20261123 + i)
                    dargs = SimpleNamespace(**{k: None for k in ALL_GS_KEYS})
                    dargs.jpeg_ratio = 25
                    adv_jpg = gs_dist.image_distortion(adv_im.copy(), 20261123 + i, dargs).convert('RGB')
                    e_j = sscd_emb_t(pil_to_t(adv_jpg)).cpu().numpy().astype(np.float64)
                    qp_j = e_j @ P_eval_np
                    d_j = float(1.0 - np.dot(sgn_np, qp_j) / (8.0 * np.linalg.norm(qp_j) + 1e-12))

                    mse = float(((x_fin - x0) ** 2).mean().item())
                    psnr = 100.0 if mse < 1e-12 else float(-10.0 * np.log10(mse))
                    lp = float(lpips_fn(x0 * 2.0 - 1.0, x_fin * 2.0 - 1.0).item())

                succ_clean += int(d_c <= th)
                succ_jpg25 += int(d_j <= th)
                lps.append(lp); psnrs.append(psnr); d_cleans.append(d_c); d_jpgs.append(d_j)

            row = dict(eps_255=int(round(eps_l_inf * 255)), mode=mode, n=4,
                       clean_fooled=succ_clean, post_jpeg25_fooled=succ_jpg25,
                       mean_d_clean=float(np.mean(d_cleans)), mean_d_jpg25=float(np.mean(d_jpgs)),
                       mean_psnr=float(np.mean(psnrs)), mean_lpips=float(np.mean(lps)))
            adaptive_rows.append(row)
            print(f"  Adaptive eps={row['eps_255']:2d}/255 | {mode:24s} | clean={succ_clean}/4 (d={row['mean_d_clean']:.3f}) | post_JPEG25={succ_jpg25}/4 (d={row['mean_d_jpg25']:.3f}) | PSNR={row['mean_psnr']:.1f} LPIPS={row['mean_lpips']:.3f}", flush=True)

    # Save visual comparison of adaptive PGD collision on case 0 (Row 0: public_P64_known, Row 1: secret_keyed_P64_unknown)
    canvas = Image.new('RGB', (256 * 5, 256 * 2), (245, 245, 245))
    canvas.paste(covers[0].resize((256, 256), Image.BICUBIC), (0, 0))
    canvas.paste(Image.open(STAGE1 / 'images' / f"{proto1['sources'][0]['task']:03d}/sscd64_mid_tweedie.png").convert('RGB').resize((256, 256), Image.BICUBIC), (0, 256))
    for ci, eps_i in enumerate((4, 8, 16, 32), start=1):
        canvas.paste(montage_imgs[(eps_i, 'public_P64_known')].resize((256, 256), Image.BICUBIC), (ci * 256, 0))
        canvas.paste(montage_imgs[(eps_i, 'secret_keyed_P64_unknown')].resize((256, 256), Image.BICUBIC), (ci * 256, 256))
    canvas.save(RUN / 'adaptive_stress_montage.jpg', quality=92)

    # =========================================================================
    # PART 2: EXPANDED 16-PROMPT CROSS-CARRIER NATIVE 512-BIT PRC (UNIFIED COHORT)
    # =========================================================================
    print('=== PART 2: 16-Prompt Cross-Carrier Native 512-bit PRC (Official Unified Cohort) ===', flush=True)
    sel = read_json(UNIFIED_SEL)['sources']
    prc_sources = [s for s in sel if s['dataset'] == 'COCO'][50:58] + [s for s in sel if s['dataset'] == 'SDP'][50:58]
    assert len(prc_sources) == 16

    np.random.seed(2026112305)
    enc_key, dec_key = prc.KeyGen(16384, message_length=512, false_positive_rate=1e-6, t=3)
    rng_p = np.random.default_rng(2026112305)
    auth_448 = rng_p.integers(0, 2, size=(448,), dtype=np.uint8)
    P64_prc_np = make_keyed_projection(np.packbits(auth_448).tobytes(), 64)
    P64_prc_t = torch.from_numpy(P64_prc_np).float().cuda()

    provider = pipe_utils.create_diffusion_pipe(
        pretrained_model_name_or_path='Manojb/stable-diffusion-2-1-base',
        resolution=512, device='cuda:0', scheduler_name='DDIM', scheduler_inverse_name='DDIM',
        dtype=torch.float16, eager_loading=True, disable_tqdm=True,
    )
    pipe = provider.pipe
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.requires_grad_(False); pipe.vae.eval()
    pipe.unet.requires_grad_(False); pipe.unet.eval()

    clip_asset = read_json(ROOT / 'runs/phase1_backend_audit_20261113_v2/complete.json')['model_files']['clip_vitl14']
    clip_model = CLIPModel.from_pretrained(clip_asset['root'], local_files_only=True).cuda().eval()
    clip_proc = CLIPProcessor.from_pretrained(clip_asset['root'], local_files_only=True)

    def decode_x(z):
        return (pipe.vae.decode((z.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor), return_dict=False)[0].float() / 2.0 + 0.5).clamp(0.0, 1.0)

    def diff_sscd_prc(x, ts_t):
        proj = sscd_emb_t(x) @ P64_prc_t
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
                loss = diff_sscd_prc(decode_x(z0l), ts_t)
                if float(loss.detach()) > stop_loss:
                    g = torch.autograd.grad(loss, z0l)[0]
                    gn = float(g.norm()) + 1e-8
                    pt = t - pipe.scheduler.config.num_train_timesteps // num_steps
                    ap = alphas[pt] if pt >= 0 else pipe.scheduler.final_alpha_cumprod.to('cuda')
                    zn = (zn.float() + torch.sqrt(ap.float()) * (-(step_size * float(z0l.detach().norm()) / gn) * g)).to(torch.float16)
            z = zn
        with torch.no_grad():
            return t_to_pil(decode_x(z))

    def verify_prc(im, true_msg_512):
        z_inv = provider.invert_images(im, num_inference_steps=50)['zT_torch']
        z_flat = z_inv.detach().float().cpu().reshape(-1).numpy()
        post = torch.from_numpy(erf(z_flat / float(np.sqrt(2.0 * 1.5 * 2.5))).astype(np.float64))
        det = bool(prc.Detect(dec_key, post))
        rec = prc.Decode(dec_key, post)
        dec_ok = rec is not None
        if dec_ok:
            rb = np.asarray(rec, dtype=np.uint8)[:512]
            auth_ok = bool(np.array_equal(rb[:448], true_msg_512[:448]))
            rec_sgn = 2.0 * rb[448:512].astype(np.float64) - 1.0
            with torch.no_grad():
                qp = sscd_emb_t(pil_to_t(im)).cpu().numpy().astype(np.float64) @ P64_prc_np
            d_b = float(1.0 - np.dot(rec_sgn, qp) / (8.0 * np.linalg.norm(qp) + 1e-12))
            f_ok = d_b <= th
        else:
            auth_ok, d_b, f_ok = False, 2.0, False
        return dict(detect=det, decode=dec_ok, auth_exact=auth_ok, d_blind=d_b, feat_ok=bool(f_ok), joint=bool(det and dec_ok and auth_ok and f_ok))

    rp_proto = read_json(ROOT / 'runs/author_reprompt_plus_20261115/protocol.json')
    i2p_prompts = [r['target_prompt'] for r in rp_proto['references'] if r['carrier'] == 'GS'][:8]

    prc_recs, prc_rp = [], []
    for idx, s in enumerate(prc_sources):
        prompt = s['prompt']
        t0 = time.perf_counter()
        zT_prev = torch.randn((1, 4, 64, 64), generator=torch.Generator(device='cuda').manual_seed(2026112320 + idx), device='cuda', dtype=torch.float16)
        prev_im = ddim_gen(prompt, zT_prev, num_steps=12, ts_t=None)
        torch.cuda.synchronize()
        sec_prev = time.perf_counter() - t0

        with torch.no_grad():
            pe = sscd_emb_t(pil_to_t(prev_im)).cpu().numpy().astype(np.float64)
        p_bits = ((pe @ P64_prc_np) >= 0.0).astype(np.uint8)
        ts_t = torch.tensor(2.0 * p_bits.astype(np.float64) - 1.0, device='cuda', dtype=torch.float32)

        msg_512 = np.concatenate([auth_448, p_bits])
        np.random.seed(2026112340 + idx)
        cw = prc.Encode(enc_key, msg_512).numpy().astype(np.float64)
        zT = torch.from_numpy((cw * np.abs(np.random.randn(16384))).reshape(1, 4, 64, 64)).to(device='cuda', dtype=torch.float16)

        base_im = ddim_gen(prompt, zT, num_steps=50, ts_t=None)
        t0 = time.perf_counter()
        guid_im = ddim_gen(prompt, zT, num_steps=50, ts_t=ts_t, step_range=(5, 35), step_size=0.042, stop_loss=0.08)
        torch.cuda.synchronize()
        sec_guid = time.perf_counter() - t0

        with torch.no_grad():
            lp = float(lpips_fn(pil_to_t(base_im) * 2.0 - 1.0, pil_to_t(guid_im) * 2.0 - 1.0).item())
            inp = clip_proc(text=[prompt], images=[base_im, guid_im], return_tensors='pt', padding=True, truncation=True, max_length=77).to('cuda')
            co = clip_model(**inp)
            ie, te = F.normalize(co.image_embeds, dim=-1), F.normalize(co.text_embeds, dim=-1)
            cb, cg = float((ie[0] * te[0]).sum().item()), float((ie[1] * te[0]).sum().item())

        for ci, cond in enumerate(['clean', *PRC_EDITS.keys()]):
            if cond == 'clean':
                q_im = guid_im
            else:
                gs_dist.set_random_seed(20261123 + idx * 100 + ci)
                dargs = SimpleNamespace(**{k: None for k in ALL_GS_KEYS})
                setattr(dargs, cond, PRC_EDITS[cond])
                q_im = gs_dist.image_distortion(guid_im.copy(), 20261123 + idx * 100 + ci, dargs).convert('RGB')
            v = verify_prc(q_im, msg_512)
            prc_recs.append(dict(idx=idx, dataset=s['dataset'], condition=cond, sec_total=sec_prev + sec_guid, lpips=lp, clip_base=cb, clip_guid=cg, **v))

        if idx < 8:
            zT_atk = provider.invert_images(guid_im, num_inference_steps=50)['zT_torch']
            forged = provider.generate([i2p_prompts[idx]], num_inference_steps=50, guidance_scale=7.5, latents=zT_atk, return_latents=False)['images_PIL'][0].convert('RGB')
            prc_rp.append(dict(idx=idx, **verify_prc(forged, msg_512)))

        if (idx + 1) % 4 == 0:
            print(f'  [PRC16 | GPU {gpu}] {idx+1}/16 done ({time.perf_counter()-t_all:.1f}s)', flush=True)

    cl = [r for r in prc_recs if r['condition'] == 'clean']
    ed = [r for r in prc_recs if r['condition'] != 'clean']
    prc_sum = dict(
        n_prompts=16,
        sec_total_mean=float(np.mean([r['sec_total'] for r in cl])),
        lpips_mean=float(np.mean([r['lpips'] for r in cl])),
        clip_base_mean=float(np.mean([r['clip_base'] for r in cl])),
        clip_guid_mean=float(np.mean([r['clip_guid'] for r in cl])),
        clean_detect=sum(r['detect'] for r in cl),
        clean_decode=sum(r['decode'] for r in cl),
        clean_joint=sum(r['joint'] for r in cl),
        edited_n=len(ed),
        edited_detect=sum(r['detect'] for r in ed),
        edited_decode=sum(r['decode'] for r in ed),
        edited_joint=sum(r['joint'] for r in ed),
        live_reprompt=dict(
            n=len(prc_rp),
            native_detect_fooled=sum(r['detect'] for r in prc_rp),
            native_decode_fooled=sum(r['decode'] and r['auth_exact'] for r in prc_rp),
            commit_steer_fooled=sum(r['joint'] for r in prc_rp),
        ),
    )

    out = dict(elapsed_seconds=time.perf_counter() - t_all, gpu=gpu, uuid=uuid, adaptive_stress_test=adaptive_rows, prc16_unified=prc_sum)
    write_json(RUN / 'summary.json', out)
    write_json(RUN / 'prc16_records.json', prc_recs)
    print(f"STAGE 3C DONE in {out['elapsed_seconds']:.1f}s | PRC16 clean_joint={prc_sum['clean_joint']}/16 | ed_decode={prc_sum['edited_decode']}/64 ed_joint={prc_sum['edited_joint']}/64 | live_RP native={prc_sum['live_reprompt']['native_detect_fooled']}/8 ours={prc_sum['live_reprompt']['commit_steer_fooled']}/8", flush=True)


if __name__ == '__main__':
    main()
