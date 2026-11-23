"""Stage 3D (GPU 5): Multi-Dimensional Plug-and-Play Feature Binding Benchmark & Live End-to-End Verification.
Directly answers Prof. Luo's multi-dimensional feature question and proves Commit-then-Steer is feature-agnostic!

Part 1 (~45s on GPU 5):
  Exact Stage 0 frozen protocol (40 cal -> frozen 95% WAVES 16-edit threshold -> 60 confirm + 30 Imprint
  + 240 Reprompt+ + 20 MagicBrush + 8 CVPR color transfer + 1 ICCV dist match) across 7 compact 64-bit
  (8-byte) differentiable plug-in features:
    1. sscd_64b       (copy detection)
    2. clip_64b       (high-level text-image semantics)
    3. dinov2_64b     (fine-grained visual geometry/semantics)
    4. hyb_sscd32_clip32 (2-dim fusion: copy-detection + CLIP semantics)
    5. hyb_sscd32_dino32 (2-dim fusion: copy-detection + DINOv2)
    6. hyb_clip32_dino32 (2-dim fusion: CLIP + DINOv2)
    7. hyb_3dim_64b      (3-dim fusion: 24b SSCD + 20b CLIP + 20b DINOv2)

Part 2 (~5m on GPU 5):
  Live End-to-End Commit-then-Steer inside Native 256-bit GS (192b auth + 64b commitment, 12-step preview +
  mid-step Tweedie steps 5..35, step_size=0.042, stop_loss=0.08) on 8 UNSEEN official unified prompts
  (4 COCO + 4 SDP) comparing `clip_64b` and `hyb_sscd32_clip32` (Clean + 4 GS edits + 8 Live Reprompt attacks).
"""
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

ROOT = Path('/data0/liyanlei/watermark_research')
RUN = ROOT / 'runs/commit_steer_paper_scale_20261123/multidim_features'
LOADER = ROOT / 'runs/order_capsule_confirmation_20261115'
UNIFIED_SEL = ROOT / 'runs/unified_main_sources_20261116/selection.json'
GS_MOD_PATH = ROOT / 'runs/phase1_trajectory_native_gs_20261113/gs_watermark.py'
GS_DIST_PATH = ROOT / 'runs/defense_revalidation_20261111/gs_image_utils.py'
PGID_PIPE = ROOT / 'third_party/frozen/PGID_dc4ad96'
EXTRA_DEPS = ROOT / 'envs/phase1_extra'
SSCD_PATH = ROOT / 'models/sscd/sscd_disc_mixup.torchscript.pt'
DINO_PATH = ROOT / 'models/sembind/dinov2-giant'

ALL_GS_KEYS = [
    'random_crop_ratio', 'random_drop_ratio', 'brightness_factor',
    'jpeg_ratio', 'gaussian_blur_r', 'gaussian_std',
    'median_blur_k', 'sp_prob', 'resize_ratio',
]
LIVE_EDITS = {'jpeg_ratio': 25, 'brightness_factor': 6, 'gaussian_blur_r': 4, 'resize_ratio': 0.25}


def read_json(p: Path):
    return json.loads(p.read_bytes())


def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(p)


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


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


def qr_proj(dim_in: int, k: int, tag: bytes) -> np.ndarray:
    seed = int.from_bytes(hmac.digest(b'commit-steer-multidim-v1', tag, 'sha256')[:16], 'big')
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((dim_in, k)))
    return q.astype(np.float64)


def bit_cos_dist(ref_e: np.ndarray, q_e: np.ndarray, P: np.ndarray) -> float:
    k = P.shape[1]
    sgn = np.where((ref_e @ P) >= 0.0, 1.0, -1.0)
    qp = q_e @ P
    return float(1.0 - np.dot(sgn, qp) / (np.sqrt(k) * float(np.linalg.norm(qp)) + 1e-12))


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
    from transformers import CLIPModel, CLIPProcessor, Dinov2Model

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    t0_all = time.perf_counter()

    # Load all 3 vision backbones onto GPU 5 for fast batched Part 1 extraction
    sscd = torch.jit.load(str(SSCD_PATH), map_location='cuda').eval()
    for p in sscd.parameters(): p.requires_grad_(False)

    clip_asset = read_json(ROOT / 'runs/phase1_backend_audit_20261113_v2/complete.json')['model_files']['clip_vitl14']
    clip_model = CLIPModel.from_pretrained(clip_asset['root'], local_files_only=True).cuda().eval()
    for p in clip_model.parameters(): p.requires_grad_(False)
    clip_proc = CLIPProcessor.from_pretrained(clip_asset['root'], local_files_only=True)

    dino_model = Dinov2Model.from_pretrained(str(DINO_PATH), local_files_only=True, torch_dtype=torch.float16).cuda().eval()
    for p in dino_model.parameters(): p.requires_grad_(False)

    im_mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
    cl_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda').view(1, 3, 1, 1)
    cl_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda').view(1, 3, 1, 1)

    def extract_3_batch(pil_list, bs=48):
        s_out, c_out, d_out = [], [], []
        for i in range(0, len(pil_list), bs):
            chunk = pil_list[i:i + bs]
            xs = torch.stack([
                torch.from_numpy(np.asarray(im.convert('RGB').resize((320, 320), Image.BICUBIC), dtype=np.float32) / 255.0).permute(2, 0, 1)
                for im in chunk
            ]).cuda()
            with torch.no_grad():
                xs_im = (xs - im_mean) / im_std
                es = F.normalize(sscd(xs_im), dim=1).cpu().numpy().astype(np.float64)

                xs224 = F.interpolate(xs, size=(224, 224), mode='bilinear', align_corners=False)
                ec = F.normalize(clip_model.get_image_features(pixel_values=(xs224 - cl_mean) / cl_std).float(), dim=1).cpu().numpy().astype(np.float64)
                ed = F.normalize(dino_model(pixel_values=((xs224 - im_mean) / im_std).half()).last_hidden_state[:, 0, :].float(), dim=1).cpu().numpy().astype(np.float64)
            s_out.append(es); c_out.append(ec); d_out.append(ed)
        return np.concatenate(s_out, 0), np.concatenate(c_out, 0), np.concatenate(d_out, 0)

    # Verify exact output dimensions and build orthonormal QR projections
    _ts, _tc, _td = extract_3_batch([Image.new('RGB', (512, 512), (128, 128, 128))])
    D_S, D_C, D_D = _ts.shape[1], _tc.shape[1], _td.shape[1]
    print(f'Feature dims: SSCD={D_S}, CLIP={D_C}, DINOv2={D_D}', flush=True)

    P = dict(
        s64=qr_proj(D_S, 64, b's64'), c64=qr_proj(D_C, 64, b'c64'), d64=qr_proj(D_D, 64, b'd64'),
        s32=qr_proj(D_S, 32, b's32'), c32=qr_proj(D_C, 32, b'c32'), d32=qr_proj(D_D, 32, b'd32'),
        s24=qr_proj(D_S, 24, b's24'), c20=qr_proj(D_C, 20, b'c20'), d20=qr_proj(D_D, 20, b'd20'),
    )
    METHODS = [
        'sscd_64b', 'clip_64b', 'dinov2_64b',
        'hyb_sscd32_clip32', 'hyb_sscd32_dino32', 'hyb_clip32_dino32', 'hyb_3dim_64b',
    ]

    def score_pair(rs, rc, rd, qs, qc, qd) -> dict:
        ds32, dc32, dd32 = bit_cos_dist(rs, qs, P['s32']), bit_cos_dist(rc, qc, P['c32']), bit_cos_dist(rd, qd, P['d32'])
        ds24, dc20, dd20 = bit_cos_dist(rs, qs, P['s24']), bit_cos_dist(rc, qc, P['c20']), bit_cos_dist(rd, qd, P['d20'])
        return dict(
            sscd_64b=bit_cos_dist(rs, qs, P['s64']),
            clip_64b=bit_cos_dist(rc, qc, P['c64']),
            dinov2_64b=bit_cos_dist(rd, qd, P['d64']),
            hyb_sscd32_clip32=0.5 * ds32 + 0.5 * dc32,
            hyb_sscd32_dino32=0.5 * ds32 + 0.5 * dd32,
            hyb_clip32_dino32=0.5 * dc32 + 0.5 * dd32,
            hyb_3dim_64b=(24.0 * ds24 + 20.0 * dc20 + 20.0 * dd20) / 64.0,
        )

    # =========================================================================
    # PART 1: EXACT STAGE 0 FROZEN BENCHMARK FOR ALL 7 MULTI-DIM 64-BIT FEATURES
    # =========================================================================
    print('=== PART 1: Stage 0 Frozen Multi-Dim 64-bit Benchmark (40 cal / 60 confirm / Imprint / Reprompt+ / External) ===', flush=True)
    sys.path.insert(0, str(LOADER))
    from order_recovery_data import waves_module
    waves = waves_module()

    cb_run = ROOT / 'runs/color_budget_20261119'
    cj_run = ROOT / 'runs/color_joint_20261119'
    cb_plan = read_json(cb_run / 'color_budget_20261119.json')
    families, strengths = cb_plan['families'], cb_plan['strengths']
    cal_sources = [s for s in read_json(cb_run / 'sources.json') if s['split'] == 'calibration']
    cj_sources = read_json(cj_run / 'sources.json')

    def run_waves_split(run_dir, sources):
        rows = []
        for s in sources:
            dest = run_dir / 'images' / f"{s['task']:03d}"
            receipt = read_json(dest / 'receipt.json')
            imgs = []
            for i, expected in enumerate(receipt['sha256']):
                p = dest / f'{i}.png'
                assert sha_file(p) == expected
                imgs.append(Image.open(p).convert('RGB'))
            rs, rc, rd = extract_3_batch(imgs)

            q_imgs, q_meta = [], []
            for seed_idx in (0, 1):
                im = imgs[seed_idx]
                for fi, fam in enumerate(['clean'] + families):
                    for ji, st in enumerate(strengths if fi else [0.0]):
                        seed = 20261119 + 100 * s['task'] + 2 * fi + ji
                        q_im = waves.apply_distortion([im], fam, strength=st, distortion_seed=seed)[0] if fi else im
                        q_imgs.append(q_im)
                        q_meta.append((s['dataset'], s['task'], seed_idx, fam, st))
            qs, qc, qd = extract_3_batch(q_imgs)
            for (ds, tk, si, fam, st), qsi, qci, qdi in zip(q_meta, qs, qc, qd):
                rows.append(dict(
                    family=fam, strength=st,
                    own=score_pair(rs[si], rc[si], rd[si], qsi, qci, qdi),
                    sib=score_pair(rs[1 - si], rc[1 - si], rd[1 - si], qsi, qci, qdi),
                ))
        return rows

    cal_rows = run_waves_split(cb_run, cal_sources)
    conf_rows = run_waves_split(cj_run, cj_sources)
    cal_ed = [r for r in cal_rows if r['family'] != 'clean']
    conf_ed = [r for r in conf_rows if r['family'] != 'clean']

    thresholds = {}
    for m in METHODS:
        pos = np.sort([r['own'][m] for r in cal_ed])
        thresholds[m] = float(pos[int(np.ceil(0.95 * len(pos))) - 1])

    # Imprint (30)
    rp_proto = read_json(ROOT / 'runs/author_reprompt_plus_20261115/protocol.json')
    imp_refs, imp_atks = [], []
    for ref in rp_proto['references']:
        if ref['carrier'] == 'GS':
            ref_p = Path(ref['image'])
            imp_refs.append(Image.open(ref_p).convert('RGB'))
            imp_atks.append(Image.open(ref_p.parent.parent / 'attacked_images_step_150' / ref_p.name).convert('RGB'))
    for carrier in ('TR', 'PRC'):
        base = ROOT / f'runs/pgid_carrier_coverage_20261113/{carrier}/attack=forgery'
        for i in range(10):
            imp_refs.append(Image.open(base / 'watermarked_images' / f'image_{i:03d}.png').convert('RGB'))
            imp_atks.append(Image.open(base / 'attacked_images_step_150' / f'image_{i:03d}.png').convert('RGB'))
    irs, irc, ird = extract_3_batch(imp_refs)
    ias, iac, iad = extract_3_batch(imp_atks)
    imp_scores = [score_pair(irs[i], irc[i], ird[i], ias[i], iac[i], iad[i]) for i in range(len(imp_refs))]

    # Reprompt+ (180 GS + 60 TR across SD2.1 & SD1.5)
    rp_dir = ROOT / 'runs/author_reprompt_plus_20261115'
    rp_ref_ids = [r['id'] for r in rp_proto['references']]
    rrs, rrc, rrd = extract_3_batch([Image.open(r['image']).convert('RGB') for r in rp_proto['references']])
    rp_ref_map = {sid: (rrs[i], rrc[i], rrd[i]) for i, sid in enumerate(rp_ref_ids)}
    rp_meta, rp_cands = [], []
    for ref in rp_proto['references']:
        carrier, sid = ref['carrier'], ref['id']
        n_res = 3 if carrier == 'GS' else 1
        for proxy in ('sd21', 'sd15'):
            for pi in range(3):
                for ri in range(n_res):
                    rp_meta.append((carrier, sid, proxy))
                    rp_cands.append(Image.open(rp_dir / proxy / 'images' / f'{sid}_{proxy}_p{pi}_r{ri}.png').convert('RGB'))
    rcs, rcc, rcd = extract_3_batch(rp_cands)
    rp_rows = [dict(carrier=c, id=sid, proxy=px, scores=score_pair(*rp_ref_map[sid], rcs[i], rcc[i], rcd[i]))
               for i, (c, sid, px) in enumerate(rp_meta)]

    # External (MagicBrush 20 + CVPR color transfer 8 + ICCV dist 1)
    p2_dir = ROOT / 'runs/rgb_moments_phase2_20261120/author'
    p2_man = read_json(ROOT / 'runs/rgb_moments_phase2_20261120/public_manifest.json')
    p3_dir = ROOT / 'runs/local_semantics_phase3_20261120/additional'
    p3_man = read_json(ROOT / 'runs/local_semantics_phase3_20261120/additional_manifest.json')
    ext_groups, ext_refs, ext_outs = [], [], []
    for item in p2_man:
        ext_groups.append(item['group'])
        ext_refs.append(Image.open(p2_dir / item['files']['ref']).convert('RGB'))
        ext_outs.append(Image.open(p2_dir / item['files']['output']).convert('RGB'))
    for item in p3_man:
        ext_groups.append('replacement_extra')
        ext_refs.append(Image.open(p3_dir / item['files']['ref']).convert('RGB'))
        ext_outs.append(Image.open(p3_dir / item['files']['output']).convert('RGB'))
    ers, erc, erd = extract_3_batch(ext_refs)
    eos, eoc, eod = extract_3_batch(ext_outs)
    ext_rows = [dict(group=g, scores=score_pair(ers[i], erc[i], erd[i], eos[i], eoc[i], eod[i])) for i, g in enumerate(ext_groups)]

    stage0_table = {}
    for m in METHODS:
        th = thresholds[m]
        tpr = sum(r['own'][m] <= th for r in conf_ed)
        sfar = sum(r['sib'][m] <= th for r in conf_ed)
        blur08 = [r for r in conf_ed if r['family'] == 'blurring' and r['strength'] == 0.8]
        rot08 = [r for r in conf_ed if r['family'] == 'rotation' and r['strength'] == 0.8]
        imp_acc = sum(s[m] <= th for s in imp_scores)
        rp_gs21 = [r for r in rp_rows if r['carrier'] == 'GS' and r['proxy'] == 'sd21']
        rp_gs15 = [r for r in rp_rows if r['carrier'] == 'GS' and r['proxy'] == 'sd15']
        gs21_sids = sorted({r['id'] for r in rp_gs21})
        gs15_sids = sorted({r['id'] for r in rp_gs15})
        cvpr = [r for r in ext_rows if r['group'] == 'color']
        mb = [r for r in ext_rows if r['group'] in ('replacement', 'replacement_extra')]
        iccv = [r for r in ext_rows if r['group'] == 'distribution']
        stage0_table[m] = dict(
            th=th,
            confirm_tpr=f'{tpr}/{len(conf_ed)} ({100*tpr/len(conf_ed):.2f}%)',
            confirm_tpr_num=tpr / len(conf_ed),
            sib_far=f'{sfar}/{len(conf_ed)} ({100*sfar/len(conf_ed):.2f}%)',
            sib_far_num=sfar / len(conf_ed),
            blur08_tpr=f"{sum(r['own'][m] <= th for r in blur08)}/{len(blur08)}",
            rot08_tpr=f"{sum(r['own'][m] <= th for r in rot08)}/{len(rot08)}",
            imprint_30=f'{imp_acc}/30',
            reprompt_gs_sd21_cand=f"{sum(r['scores'][m] <= th for r in rp_gs21)}/{len(rp_gs21)}",
            reprompt_gs_sd21_src=f"{sum(any(r['scores'][m] <= th for r in rp_gs21 if r['id']==sid) for sid in gs21_sids)}/{len(gs21_sids)}",
            reprompt_gs_sd15_cand=f"{sum(r['scores'][m] <= th for r in rp_gs15)}/{len(rp_gs15)}",
            reprompt_gs_sd15_src=f"{sum(any(r['scores'][m] <= th for r in rp_gs15 if r['id']==sid) for sid in gs15_sids)}/{len(gs15_sids)}",
            cvpr_color_8=f"{sum(r['scores'][m] <= th for r in cvpr)}/8",
            iccv_dist_1=f"{sum(r['scores'][m] <= th for r in iccv)}/1",
            magicbrush_20=f"{sum(r['scores'][m] <= th for r in mb)}/20",
        )
        t = stage0_table[m]
        print(f"  {m:18s} | TPR={t['confirm_tpr']:15s} | SibFAR={t['sib_far']:14s} | Blur0.8={t['blur08_tpr']:6s} | Rot0.8={t['rot08_tpr']:7s} | Imp={t['imprint_30']:5s} | RP21={t['reprompt_gs_sd21_cand']:5s}({t['reprompt_gs_sd21_src']}) | RP15={t['reprompt_gs_sd15_cand']:5s}({t['reprompt_gs_sd15_src']}) | CVPR={t['cvpr_color_8']} | MB={t['magicbrush_20']}", flush=True)

    # Free DINOv2 before loading SD2.1 for Part 2
    del dino_model
    torch.cuda.empty_cache()

    # =========================================================================
    # PART 2: LIVE END-TO-END COMMIT-THEN-STEER WITH CLIP-64b & HYBRID (SSCD32+CLIP32)
    # =========================================================================
    print('=== PART 2: Live End-to-End Commit-then-Steer Plug-in Verification (clip_64b vs hyb_sscd32_clip32) ===', flush=True)
    sys.path.insert(0, str(PGID_PIPE))
    sys.path.insert(0, str(EXTRA_DEPS))
    from pipe import pipe_utils

    native_gs = load_module('native_gs', GS_MOD_PATH).Gaussian_Shading(1, 8, 1e-6, 1)
    rng_k = np.random.default_rng(2026112399)
    gs_key_np = rng_k.integers(0, 2, size=(1, 4, 64, 64), dtype=np.uint8)
    auth_192 = rng_k.integers(0, 2, size=(192,), dtype=np.uint8)
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
    lpips_fn = lpips.LPIPS(net='alex').cuda().eval()

    P_c64_t = torch.from_numpy(P['c64']).float().cuda()
    P_s32_t = torch.from_numpy(P['s32']).float().cuda()
    P_c32_t = torch.from_numpy(P['c32']).float().cuda()

    def decode_x(z):
        return (pipe.vae.decode((z.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor), return_dict=False)[0].float() / 2.0 + 0.5).clamp(0.0, 1.0)

    def to_pil(x):
        return Image.fromarray(x.detach()[0].permute(1, 2, 0).mul(255.0).round().byte().cpu().numpy())

    def pil_to_t(im):
        return torch.from_numpy(np.asarray(im.convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()

    def diff_loss(x, ts_t, arm):
        if arm == 'clip_64b':
            x224 = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            ec = F.normalize(clip_model.get_image_features(pixel_values=(x224 - cl_mean) / cl_std).float(), dim=1)[0]
            proj = ec @ P_c64_t
            return (1.0 - F.cosine_similarity(proj.unsqueeze(0), ts_t.unsqueeze(0)).squeeze()) + 0.5 * F.softplus(0.05 - proj * ts_t, beta=20.0).mean()
        else:
            x320 = F.interpolate(x, size=(320, 320), mode='bilinear', align_corners=False)
            es = F.normalize(sscd((x320 - im_mean) / im_std), dim=1)[0]
            x224 = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            ec = F.normalize(clip_model.get_image_features(pixel_values=(x224 - cl_mean) / cl_std).float(), dim=1)[0]
            ps, pc = es @ P_s32_t, ec @ P_c32_t
            ts_s, ts_c = ts_t[:32], ts_t[32:]
            ls = (1.0 - F.cosine_similarity(ps.unsqueeze(0), ts_s.unsqueeze(0)).squeeze()) + 0.5 * F.softplus(0.05 - ps * ts_s, beta=20.0).mean()
            lc = (1.0 - F.cosine_similarity(pc.unsqueeze(0), ts_c.unsqueeze(0)).squeeze()) + 0.5 * F.softplus(0.05 - pc * ts_c, beta=20.0).mean()
            return 0.5 * ls + 0.5 * lc

    def ddim_gen(prompt, zT, num_steps=50, ts_t=None, arm=None, step_range=(5, 35), step_size=0.042, stop_loss=0.08):
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
                loss = diff_loss(decode_x(z0l), ts_t, arm)
                if float(loss.detach()) > stop_loss:
                    g = torch.autograd.grad(loss, z0l)[0]
                    gn = float(g.norm()) + 1e-8
                    pt = t - pipe.scheduler.config.num_train_timesteps // num_steps
                    ap = alphas[pt] if pt >= 0 else pipe.scheduler.final_alpha_cumprod.to('cuda')
                    zn = (zn.float() + torch.sqrt(ap.float()) * (-(step_size * float(z0l.detach().norm()) / gn) * g)).to(torch.float16)
            z = zn
        with torch.no_grad():
            return to_pil(decode_x(z))

    def eval_feat(im):
        x = pil_to_t(im)
        with torch.no_grad():
            x320 = F.interpolate(x, size=(320, 320), mode='bilinear', align_corners=False)
            es = F.normalize(sscd((x320 - im_mean) / im_std), dim=1)[0].cpu().numpy().astype(np.float64)
            x224 = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            ec = F.normalize(clip_model.get_image_features(pixel_values=(x224 - cl_mean) / cl_std).float(), dim=1)[0].cpu().numpy().astype(np.float64)
        return es, ec

    def blind_dist(rec_bits, q_im, arm):
        es, ec = eval_feat(q_im)
        sgn = 2.0 * rec_bits.astype(np.float64) - 1.0
        if arm == 'clip_64b':
            qp = ec @ P['c64']
            return float(1.0 - np.dot(sgn, qp) / (8.0 * float(np.linalg.norm(qp)) + 1e-12))
        else:
            qps, qpc = es @ P['s32'], ec @ P['c32']
            ds = float(1.0 - np.dot(sgn[:32], qps) / (np.sqrt(32.0) * float(np.linalg.norm(qps)) + 1e-12))
            dc = float(1.0 - np.dot(sgn[32:], qpc) / (np.sqrt(32.0) * float(np.linalg.norm(qpc)) + 1e-12))
            return 0.5 * ds + 0.5 * dc

    sel = read_json(UNIFIED_SEL)['sources']
    live_sources = [s for s in sel if s['dataset'] == 'COCO'][60:64] + [s for s in sel if s['dataset'] == 'SDP'][60:64]
    i2p_prompts = [r['target_prompt'] for r in rp_proto['references'] if r['carrier'] == 'GS'][:8]

    live_summary = {}
    for arm in ('clip_64b', 'hyb_sscd32_clip32'):
        th = thresholds[arm]
        recs, rps = [], []
        for idx, s in enumerate(live_sources):
            prompt = s['prompt']
            zT_prev = torch.randn((1, 4, 64, 64), generator=torch.Generator(device='cuda').manual_seed(2026112380 + idx), device='cuda', dtype=torch.float16)
            prev_im = ddim_gen(prompt, zT_prev, num_steps=12)
            es_p, ec_p = eval_feat(prev_im)
            if arm == 'clip_64b':
                p_bits = ((ec_p @ P['c64']) >= 0.0).astype(np.uint8)
            else:
                p_bits = np.concatenate([((es_p @ P['s32']) >= 0.0).astype(np.uint8), ((ec_p @ P['c32']) >= 0.0).astype(np.uint8)])
            ts_t = torch.tensor(2.0 * p_bits.astype(np.float64) - 1.0, device='cuda', dtype=torch.float32)

            w_flat = np.concatenate([auth_192, p_bits])
            m_enc = ((np.tile(w_flat.reshape(1, 4, 8, 8), (1, 1, 8, 8)) + gs_key_np) % 2).reshape(-1)
            np.random.seed(2026112390 + idx)
            zT = native_gs.truncSampling(m_enc)

            base_im = ddim_gen(prompt, zT, num_steps=50)
            guid_im = ddim_gen(prompt, zT, num_steps=50, ts_t=ts_t, arm=arm)

            with torch.no_grad():
                lp = float(lpips_fn(pil_to_t(base_im) * 2.0 - 1.0, pil_to_t(guid_im) * 2.0 - 1.0).item())
                inp = clip_proc(text=[prompt], images=[base_im, guid_im], return_tensors='pt', padding=True, truncation=True, max_length=77).to('cuda')
                co = clip_model(**inp)
                ie, te = F.normalize(co.image_embeds, dim=-1), F.normalize(co.text_embeds, dim=-1)
                cb, cg = float((ie[0] * te[0]).sum().item()), float((ie[1] * te[0]).sum().item())

            q_imgs, conds = [guid_im], ['clean']
            for ci, (cname, cval) in enumerate(LIVE_EDITS.items(), start=1):
                seed = 20261123 + idx * 50 + ci
                gs_dist.set_random_seed(seed)
                dargs = SimpleNamespace(**{k: None for k in ALL_GS_KEYS})
                setattr(dargs, cname, cval)
                q_imgs.append(gs_dist.image_distortion(guid_im.copy(), seed, dargs).convert('RGB'))
                conds.append(cname)

            z_invs = provider.invert_images(q_imgs, num_inference_steps=50)['zT_torch']
            for ci, cond in enumerate(conds):
                rev_w = native_gs.diffusion_inverse(((z_invs[ci:ci+1] > 0).int() + native_gs.key) % 2).reshape(-1).cpu().numpy().astype(np.uint8)
                auth_ok = float((rev_w[:192] == auth_192).mean()) > (136.0 / 192.0)
                d_b = blind_dist(rev_w[192:], q_imgs[ci], arm)
                recs.append(dict(idx=idx, condition=cond, lpips=lp, clip_base=cb, clip_guid=cg, auth_ok=bool(auth_ok), d_blind=d_b, joint=bool(auth_ok and d_b <= th)))

            # Live Reprompt
            forged = provider.generate([i2p_prompts[idx]], num_inference_steps=50, guidance_scale=7.5, latents=z_invs[0:1], return_latents=False)['images_PIL'][0].convert('RGB')
            zf = provider.invert_images(forged, num_inference_steps=50)['zT_torch']
            rf = native_gs.diffusion_inverse(((zf > 0).int() + native_gs.key) % 2).reshape(-1).cpu().numpy().astype(np.uint8)
            f_auth = float((rf[:192] == auth_192).mean()) > (136.0 / 192.0)
            f_d = blind_dist(rf[192:], forged, arm)
            rps.append(dict(idx=idx, auth_ok=bool(f_auth), d_blind=f_d, joint=bool(f_auth and f_d <= th)))

        cl = [r for r in recs if r['condition'] == 'clean']
        ed = [r for r in recs if r['condition'] != 'clean']
        live_summary[arm] = dict(
            th=th, n_prompts=8,
            lpips_mean=float(np.mean([r['lpips'] for r in cl])),
            clip_base_mean=float(np.mean([r['clip_base'] for r in cl])),
            clip_guid_mean=float(np.mean([r['clip_guid'] for r in cl])),
            clean_joint=f"{sum(r['joint'] for r in cl)}/8",
            edited_4gs_joint=f"{sum(r['joint'] for r in ed)}/32",
            live_reprompt_native=f"{sum(r['auth_ok'] for r in rps)}/8",
            live_reprompt_ours=f"{sum(r['joint'] for r in rps)}/8",
        )
        ls = live_summary[arm]
        print(f"  LIVE {arm:18s} | LPIPS={ls['lpips_mean']:.3f} | CLIP={ls['clip_guid_mean']:.4f} vs {ls['clip_base_mean']:.4f} | clean={ls['clean_joint']} | 4-edits={ls['edited_4gs_joint']} | live_RP native={ls['live_reprompt_native']} ours={ls['live_reprompt_ours']}", flush=True)

    out = dict(elapsed_seconds=time.perf_counter() - t0_all, gpu=gpu, uuid=uuid, stage0_multidim=stage0_table, live_end2end=live_summary)
    write_json(RUN / 'summary.json', out)
    print(f"STAGE 3D MULTIDIM DONE in {out['elapsed_seconds']:.1f}s", flush=True)


if __name__ == '__main__':
    main()
