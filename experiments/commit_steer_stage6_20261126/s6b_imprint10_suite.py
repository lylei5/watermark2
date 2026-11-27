"""Stage 6B: SD2.1 Live 151-Step CVPR 2025 Imprint Scale-Up from N=4 to N=10 (30 Checkpoints/Method)
for BOTH Cross-Prompt Unrelated-Cover Forgery AND Same-Prompt Instance-Level Forgery.

All arms reuse exact frozen Stage 3/4/5 private keys, thresholds, watermarked images, and covers:
  - Cross-prompt covers 0..3: runs/published_imprint_dpm_full_20261110/inputs/{000..003}_cover.png (exact reuse of Stage 4B/4C/4D/5A)
  - Cross-prompt covers 4..9: runs/phase1_public_inputs_20261113/rgb512/{004..009}.png (6 audited real MS-COCO covers, unique SHA256)
  - Same-prompt sibling covers 0..9: runs/commit_steer_stage4_20261125/head2head_sembind/images/{000..009}_same.png

Subcommands:
  prepare             - Freeze imprint10/protocol.json with cover SHA256 verification.
  cross_sscd64_p0     - Commit-then-Steer sscd_64b cross-prompt Imprint gidx 0..4 (on gs100_unified/images_guided).
  cross_sscd64_p1     - Commit-then-Steer sscd_64b cross-prompt Imprint gidx 5..9 (on gs100_unified/images_guided).
  cross_hyb           - Commit-then-Steer hyb cross-prompt Imprint gidx 4..9 (merges with Stage 4B 0..3 -> N=10).
  cross_native_pgid   - Native GS-256b & Official PGID-F cross-prompt Imprint gidx 4..9 (merges with Stage 4C 0..3 -> N=10).
  cross_sembind       - Paper-faithful SemBind GS-S cross-prompt Imprint gidx 4..9 (merges with Stage 4D 0..3 -> N=10).
  cross_seal_gen      - SEAL cross-prompt Imprint optimization ONLY gidx 4..9 (isolated env, zero SEAL salt access).
  same_sscd64         - Commit-then-Steer sscd_64b same-prompt Imprint gidx 4..9 (merges with Stage 5B 0..3 -> N=10).
  same_hyb            - Commit-then-Steer hyb same-prompt Imprint gidx 4..9 (merges with Stage 5B 0..3 -> N=10).
  same_sembind        - Paper-faithful SemBind GS-S same-prompt Imprint gidx 4..9 (merges with Stage 5C 0..3 -> N=10).
  merge_all           - Merge all 0..3 + 4..9 into unified N=10 summaries + Fisher exact p-values & Clopper-Pearson CIs.
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
from scipy.stats import beta, fisher_exact

ROOT = Path('/data0/liyanlei/watermark_research')
STAGE6 = ROOT / 'runs/commit_steer_stage6_20261126'
RUN = STAGE6 / 'imprint10'
GS100_S3 = ROOT / 'runs/commit_steer_paper_scale_20261123/gs100_unified'
HYB_S4 = ROOT / 'runs/commit_steer_stage4_20261125/multidim100_hyb'
PGID_S4 = ROOT / 'runs/commit_steer_stage4_20261125/head2head_pgid_native_gs'
SEMBIND_S4 = ROOT / 'runs/commit_steer_stage4_20261125/head2head_sembind'
S5_SEAL = ROOT / 'runs/commit_steer_stage5_20261126/seal_head2head'
S5_CS = ROOT / 'runs/commit_steer_stage5_20261126/removal_and_instance_cs'
S5_SEMBIND = ROOT / 'runs/commit_steer_stage5_20261126/sembind_instance_and_removal'
SEAL_CLEAN_RUN = ROOT / 'runs/unified_seal_clean_20261116'

COVER_0_3_DIR = ROOT / 'runs/published_imprint_dpm_full_20261110/inputs'
COVER_4_9_DIR = ROOT / 'runs/phase1_public_inputs_20261113/rgb512'

PHASE1_DIR = ROOT / 'runs/phase1_trajectory_native_gs_20261113'
GS_MOD_PATH = PHASE1_DIR / 'gs_watermark.py'
PGID_PIPE = ROOT / 'third_party/frozen/PGID_dc4ad96'
SEMBIND_OFFICIAL = ROOT / 'third_party/frozen/SemBind_337c9f9'
SEMBIND_SCHED = ROOT / 'runs/sembind_chain_revalidation_20261111/scheduling_dpmsolver_multistep.py'
SD21_SNAP = ROOT / 'cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/snapshots/0094d483a120f3f33dafbd187ea4aa60d10de75c'
SEMBIND_MODELS = ROOT / 'models/sembind'
SEM_FORGERY = ROOT / 'third_party/semantic-forgery'
EXTRA_DEPS = ROOT / 'envs/phase1_extra'
SSCD_PATH = ROOT / 'models/sscd/sscd_disc_mixup.torchscript.pt'


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


def cross_cover_path(gidx: int) -> Path:
    if gidx < 4:
        return COVER_0_3_DIR / f'{gidx:03d}_cover.png'
    return COVER_4_9_DIR / f'{gidx:03d}.png'


def same_cover_path(gidx: int) -> Path:
    return SEMBIND_S4 / f'images/{gidx:03d}_same.png'


def gidx_to_seal_idx(gidx: int) -> int:
    return gidx if gidx < 50 else 200 + gidx


def seal_genuine_path(gidx: int) -> Path:
    # Exact match with Stage 5A (s5a_seal_head2head.py): shard_{si % 3}/images/{si:03d}_genuine.png
    si = gidx_to_seal_idx(gidx)
    return SEAL_CLEAN_RUN / f'shard_{si % 3}/images/{si:03d}_genuine.png'


def cp95_bounds(k: int, n: int):
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def do_prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    s3_proto = read_json(GS100_S3 / 'protocol.json')
    s4b_proto = read_json(HYB_S4 / 'protocol.json')
    sources_10 = [dict(gidx=i, dataset=s3_proto['sources'][i]['dataset'], prompt=s3_proto['sources'][i]['prompt']) for i in range(10)]

    cross_covers = []
    cross_hashes = set()
    same_covers = []
    for i in range(10):
        cp = cross_cover_path(i)
        assert cp.exists(), f'Missing cross cover {cp}'
        ch = sha256_file(cp)
        assert ch not in cross_hashes, f'Duplicate cross cover hash at gidx={i}'
        cross_hashes.add(ch)
        cross_covers.append(dict(gidx=i, path=str(cp), name=cp.name, sha256=ch))

        sp = same_cover_path(i)
        assert sp.exists(), f'Missing same-prompt cover {sp}'
        same_covers.append(dict(gidx=i, path=str(sp), name=sp.name, sha256=sha256_file(sp)))

        sg = seal_genuine_path(i)
        assert sg.exists(), f'Missing SEAL genuine image {sg}'

    proto = dict(
        run_name='imprint10',
        n_cases=10,
        imprint_steps=151,
        lr=0.01,
        checkpoints=[50, 100, 150],
        sscd64_threshold=float(s3_proto['threshold']),
        hyb_threshold=float(s4b_proto['threshold']),
        auth_threshold=float(s3_proto['auth_threshold']),
        gs_default_threshold=181.0 / 256.0,
        gs_167_threshold=167.0 / 256.0,
        seal_thresholds=dict(paper_12=12, fpr_01pct=4, fpr_1pct=2),
        sources=sources_10,
        cross_covers=cross_covers,
        same_covers=same_covers,
    )
    write_json(RUN / 'protocol.json', proto)
    print(f'[S6B prepare] Verified 10 unique cross covers + 10 same-prompt covers + 10 SEAL genuine images.', flush=True)


def run_cs_imprint(arm_tag: str, threat: str, gindices: list, gpu: int):
    uuid = check_idle_gpu(gpu)
    proto = read_json(RUN / 'protocol.json')
    sources = proto['sources']
    th_sscd = float(proto['sscd64_threshold'])
    th_hyb = float(proto['hyb_threshold'])
    auth_th = float(proto['auth_threshold'])

    case_dir = RUN / 'cases'
    case_dir.mkdir(parents=True, exist_ok=True)
    todo = [g for g in gindices if not (case_dir / f'{threat}_{arm_tag}_gidx{g:03d}.json').exists()]
    if not todo:
        print(f'[S6B {threat}_{arm_tag}] All {len(gindices)} cases already complete.', flush=True)
        return

    work_dir = RUN / f'work_{threat}_{arm_tag}_{gindices[0]}'
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

    import torch
    import torch.nn.functional as F
    import lpips
    from transformers import CLIPModel

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sys.path.insert(0, str(PGID_PIPE))
    sys.path.insert(0, str(EXTRA_DEPS))
    sys.path.insert(0, str(SEM_FORGERY))
    from pipe import pipe_utils
    from utils import imprint_utils

    native_gs_mod = load_module('native_gs', GS_MOD_PATH)
    if arm_tag == 'sscd64':
        priv = np.load(GS100_S3 / 'private_material.npz')
        key_np, auth_np, P64_np = priv['key'], priv['auth_bits'], priv['P64']
        wm_dir = GS100_S3 / 'images_guided'
    else:
        priv = np.load(HYB_S4 / 'private_material.npz')
        key_np, auth_np, P_s32_np, P_c32_np = priv['key'], priv['auth_bits'], priv['P_s32'], priv['P_c32']
        wm_dir = HYB_S4 / 'images_guided'

    gs_obj = native_gs_mod.Gaussian_Shading(1, 8, 1e-6, 1)
    gs_obj.key = torch.from_numpy(key_np.astype(np.int64)).cuda()

    # Step 1: Run 151-step Imprint optimization on all todo cases
    imp_dir = RUN / f'imprint_imgs_{threat}_{arm_tag}'
    imp_dir.mkdir(parents=True, exist_ok=True)
    pipe_atk, _, inv_sched = imprint_utils.load_attacker_pipe(
        modelid='Manojb/stable-diffusion-2-1-base', scheduler='DDIM', device='cuda'
    )
    pipe_atk.set_progress_bar_config(disable=True)
    diffpipe = imprint_utils.DiffPipe(pipe_atk, scheduler=inv_sched, device=pipe_atk.device)

    saved_meta = {}
    for gidx in todo:
        wm_im = Image.open(wm_dir / f'{gidx:03d}.png').convert('RGB')
        cp = cross_cover_path(gidx) if threat == 'cross' else same_cover_path(gidx)
        cover_im = Image.open(cp).convert('RGB').resize((512, 512), Image.BICUBIC)
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
                forged_im.save(imp_dir / f'gidx_{gidx:03d}_step_{step:03d}.png')
                losses[step] = float(loss.detach())
        saved_meta[gidx] = dict(seconds=time.perf_counter() - t0, losses=losses, cover=cp.name)
        print(f'  [{threat}_{arm_tag} opt] gidx={gidx:03d} done in {saved_meta[gidx]["seconds"]:.0f}s (loss150={losses[150]:.4f})', flush=True)

    del diffpipe, pipe_atk
    torch.cuda.empty_cache()

    # Step 2: Verify all todo cases with exact frozen blind verifier
    provider = pipe_utils.create_diffusion_pipe(
        pretrained_model_name_or_path='Manojb/stable-diffusion-2-1-base',
        resolution=512, device='cuda:0', scheduler_name='DDIM', scheduler_inverse_name='DDIM',
        dtype=torch.float16, eager_loading=True, disable_tqdm=True,
    )
    provider.pipe.set_progress_bar_config(disable=True)
    sscd = torch.jit.load(str(SSCD_PATH), map_location='cuda').eval()
    for p in sscd.parameters():
        p.requires_grad_(False)
    clip_model = None
    if arm_tag == 'hyb':
        clip_asset = read_json(ROOT / 'runs/phase1_backend_audit_20261113_v2/complete.json')['model_files']['clip_vitl14']
        clip_model = CLIPModel.from_pretrained(clip_asset['root'], local_files_only=True).cuda().eval()
        for p in clip_model.parameters():
            p.requires_grad_(False)
    lpips_fn = lpips.LPIPS(net='alex').cuda().eval()

    im_mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
    cl_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda').view(1, 3, 1, 1)
    cl_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda').view(1, 3, 1, 1)

    for gidx in todo:
        cp = cross_cover_path(gidx) if threat == 'cross' else same_cover_path(gidx)
        cover_im = Image.open(cp).convert('RGB').resize((512, 512), Image.BICUBIC)
        ckpts = {}
        for step in (50, 100, 150):
            forged_im = Image.open(imp_dir / f'gidx_{gidx:03d}_step_{step:03d}.png').convert('RGB')
            zT = provider.invert_images([forged_im], num_inference_steps=50)['zT_torch']
            xs = torch.from_numpy(np.asarray(forged_im, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()
            with torch.no_grad():
                x320 = F.interpolate(xs, size=(320, 320), mode='bilinear', align_corners=False)
                es = F.normalize(sscd((x320 - im_mean) / im_std), dim=1)[0].cpu().numpy().astype(np.float64)
                rev_w = gs_obj.diffusion_inverse(((zT > 0).int() + gs_obj.key) % 2).reshape(-1).cpu().numpy().astype(np.uint8)
                auth_ba = float((rev_w[:192] == auth_np).mean())
                auth_ok = auth_ba > auth_th
                rec_sgn = 2.0 * rev_w[192:].astype(np.float64) - 1.0
                if arm_tag == 'sscd64':
                    qp = es @ P64_np
                    d_blind = float(1.0 - np.dot(rec_sgn, qp) / (8.0 * float(np.linalg.norm(qp)) + 1e-12))
                    feat_ok = d_blind <= th_sscd
                    sc = dict(auth_ba=auth_ba, auth_ok=bool(auth_ok), d_blind=d_blind, feat_ok=bool(feat_ok), joint_accept=bool(auth_ok and feat_ok))
                else:
                    x224 = F.interpolate(xs, size=(224, 224), mode='bilinear', align_corners=False)
                    ec = F.normalize(clip_model.get_image_features(pixel_values=(x224 - cl_mean) / cl_std).float(), dim=1)[0].cpu().numpy().astype(np.float64)
                    qps, qpc = es @ P_s32_np, ec @ P_c32_np
                    ds = float(1.0 - np.dot(rec_sgn[:32], qps) / (np.sqrt(32.0) * float(np.linalg.norm(qps)) + 1e-12))
                    dc = float(1.0 - np.dot(rec_sgn[32:], qpc) / (np.sqrt(32.0) * float(np.linalg.norm(qpc)) + 1e-12))
                    d_blind = 0.5 * ds + 0.5 * dc
                    feat_ok = d_blind <= th_hyb
                    sc = dict(auth_ba=auth_ba, auth_ok=bool(auth_ok), d_sscd32=ds, d_clip32=dc, d_blind=d_blind, feat_ok=bool(feat_ok), joint_accept=bool(auth_ok and feat_ok))

                a1 = np.asarray(cover_im, dtype=np.float32) / 255.0
                a2 = np.asarray(forged_im, dtype=np.float32) / 255.0
                mse = float(np.mean((a1 - a2) ** 2))
                psnr = 100.0 if mse < 1e-12 else float(-10.0 * np.log10(mse))
                lp = float(lpips_fn(torch.from_numpy(a1.transpose(2, 0, 1)).unsqueeze(0).cuda() * 2.0 - 1.0,
                                    torch.from_numpy(a2.transpose(2, 0, 1)).unsqueeze(0).cuda() * 2.0 - 1.0).item())
            ckpts[f'step_{step}'] = dict(loss=saved_meta[gidx]['losses'][step], psnr_vs_cover=psnr, lpips_vs_cover=lp, **sc)
            print(f'  [{threat}_{arm_tag}] gidx={gidx:03d} step={step:3d} | auth={sc["auth_ba"]:.4f} | d={sc["d_blind"]:.4f} | joint={sc["joint_accept"]}', flush=True)

        case_obj = dict(gidx=gidx, prompt=sources[gidx]['prompt'], cover=saved_meta[gidx]['cover'], seconds=saved_meta[gidx]['seconds'], checkpoints=ckpts)
        write_json(case_dir / f'{threat}_{arm_tag}_gidx{gidx:03d}.json', case_obj)


def run_cross_native_pgid(gindices: list, gpu: int):
    uuid = check_idle_gpu(gpu)
    proto = read_json(RUN / 'protocol.json')
    sources = proto['sources']
    case_dir = RUN / 'cases'
    case_dir.mkdir(parents=True, exist_ok=True)
    todo = [g for g in gindices if not (case_dir / f'cross_native_pgid_gidx{g:03d}.json').exists()]
    if not todo:
        print('[S6B cross_native_pgid] Already complete.', flush=True)
        return

    work_dir = RUN / 'work_cross_native_pgid'
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

    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sys.path.insert(0, str(PGID_PIPE))
    sys.path.insert(0, str(EXTRA_DEPS))
    sys.path.insert(0, str(SEM_FORGERY))
    from pipe import pipe_utils
    from pgid_defense import PGIDPipeline
    from utils import imprint_utils

    priv = np.load(GS100_S3 / 'private_material.npz')
    gs_key_np, auth_192 = priv['key'], priv['auth_bits'].astype(np.uint8)
    embs = []
    for sh in (0, 1):
        embs.extend(read_json(GS100_S3 / f'embs_shard{sh}.json'))
    w_map = {e['gidx']: np.concatenate([auth_192, np.array(e['target_bits'], dtype=np.uint8)]) for e in embs}

    native_gs = load_module('native_gs', GS_MOD_PATH).Gaussian_Shading(1, 8, 1e-6, 1)
    native_gs.key = torch.from_numpy(gs_key_np.astype(np.int64)).cuda()

    imp_dir = RUN / 'imprint_imgs_cross_native_pgid'
    imp_dir.mkdir(parents=True, exist_ok=True)
    pipe_atk, _, inv_sched = imprint_utils.load_attacker_pipe(
        modelid='Manojb/stable-diffusion-2-1-base', scheduler='DDIM', device='cuda'
    )
    pipe_atk.set_progress_bar_config(disable=True)
    diffpipe = imprint_utils.DiffPipe(pipe_atk, scheduler=inv_sched, device=pipe_atk.device)

    saved_meta = {}
    for gidx in todo:
        wm_im = Image.open(GS100_S3 / 'images_unguided' / f'{gidx:03d}.png').convert('RGB')
        cp = cross_cover_path(gidx)
        cover_im = Image.open(cp).convert('RGB').resize((512, 512), Image.BICUBIC)
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
                forged_im.save(imp_dir / f'gidx_{gidx:03d}_step_{step:03d}.png')
                losses[step] = float(loss.detach())
        saved_meta[gidx] = dict(seconds=time.perf_counter() - t0, losses=losses, cover=cp.name)
        print(f'  [cross_native_pgid opt] gidx={gidx:03d} done in {saved_meta[gidx]["seconds"]:.0f}s', flush=True)

    del diffpipe, pipe_atk
    torch.cuda.empty_cache()

    # Exact Stage 4C fp32 Native GS + Official PGID-F verification
    provider = pipe_utils.create_diffusion_pipe(
        pretrained_model_name_or_path='Manojb/stable-diffusion-2-1-base',
        resolution=512, device='cuda:0', scheduler_name='DDIM', scheduler_inverse_name='DDIM',
        dtype=torch.float32, eager_loading=True, disable_tqdm=True,
    )
    provider.pipe.set_progress_bar_config(disable=True)
    pgid_f = PGIDPipeline(
        pipe=provider, scheduler_config=provider.pipe.scheduler.config,
        num_inference_steps=50, stopping_timestep=15, num_skip_inversion=3, guidance_strength=0.001,
    )

    def score_zT(zT_1, w256):
        rev_w = native_gs.diffusion_inverse(((zT_1 > 0).int() + native_gs.key) % 2).reshape(-1).cpu().numpy().astype(np.uint8)
        ba_256 = float((rev_w == w256).mean())
        auth_ba = float((rev_w[:192] == auth_192).mean())
        return dict(
            ba_256=ba_256,
            accept_gs_default=bool(ba_256 > 181.0 / 256.0),
            accept_gs_167=bool(ba_256 >= 167.0 / 256.0),
            auth_ba_192=auth_ba,
            accept_auth192=bool(auth_ba > 136.0 / 192.0),
        )

    for gidx in todo:
        ckpts = {}
        for step in (50, 100, 150):
            forged_im = Image.open(imp_dir / f'gidx_{gidx:03d}_step_{step:03d}.png').convert('RGB')
            z_nat = provider.invert_images([forged_im], num_inference_steps=50)['zT_torch']
            z_pgid = pgid_f.run_defense([forged_im])['zT']
            sn = score_zT(z_nat, w_map[gidx])
            sp = score_zT(z_pgid, w_map[gidx])
            ckpts[f'step_{step}'] = dict(loss=saved_meta[gidx]['losses'][step], native_gs=sn, pgid_f=sp)
            print(f'  [cross_native_pgid] gidx={gidx:03d} step={step:3d} | Native={sn["ba_256"]:.4f}({sn["accept_gs_default"]}) | PGID-F={sp["ba_256"]:.4f}({sp["accept_gs_default"]})', flush=True)
        write_json(case_dir / f'cross_native_pgid_gidx{gidx:03d}.json', dict(
            gidx=gidx, prompt=sources[gidx]['prompt'], cover=saved_meta[gidx]['cover'], seconds=saved_meta[gidx]['seconds'], checkpoints=ckpts,
        ))


def run_sembind_imprint(threat: str, gindices: list, gpu: int):
    uuid = check_idle_gpu(gpu)
    proto = read_json(RUN / 'protocol.json')
    sources = proto['sources']
    case_dir = RUN / 'cases'
    case_dir.mkdir(parents=True, exist_ok=True)
    todo = [g for g in gindices if not (case_dir / f'{threat}_sembind_gidx{g:03d}.json').exists()]
    if not todo:
        print(f'[S6B {threat}_sembind] Already complete.', flush=True)
        return

    work_dir = RUN / f'work_{threat}_sembind'
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
    sys.path.insert(0, str(SEM_FORGERY))
    import torch
    import torch.nn.functional as F
    import lpips
    from utils import imprint_utils

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Step 1: Run 151-step DPM Imprint optimization on todo cases
    imp_dir = RUN / f'imprint_imgs_{threat}_sembind'
    imp_dir.mkdir(parents=True, exist_ok=True)
    pipe_atk, _, inv_sched = imprint_utils.load_attacker_pipe(
        modelid='Manojb/stable-diffusion-2-1-base', scheduler='DPM', device='cuda'
    )
    pipe_atk.set_progress_bar_config(disable=True)
    diffpipe = imprint_utils.DiffPipe(pipe_atk, scheduler=inv_sched, device=pipe_atk.device)

    saved_meta = {}
    for gidx in todo:
        wm_im = Image.open(SEMBIND_S4 / f'images/{gidx:03d}_bound.png').convert('RGB')
        cp = cross_cover_path(gidx) if threat == 'cross' else same_cover_path(gidx)
        cover_im = Image.open(cp).convert('RGB').resize((512, 512), Image.BICUBIC)
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
                forged_im.save(imp_dir / f'gidx_{gidx:03d}_step_{step:03d}.png')
                losses[step] = float(loss.detach())
        saved_meta[gidx] = dict(seconds=time.perf_counter() - t0, losses=losses, cover=cp.name)
        print(f'  [{threat}_sembind opt] gidx={gidx:03d} done in {saved_meta[gidx]["seconds"]:.0f}s', flush=True)

    del diffpipe, pipe_atk
    torch.cuda.empty_cache()

    # Step 2: Exact Paper-Faithful SemBind GS-S verification (DPM50 + VAE100 + HMAC permutation)
    load_module('model', SEMBIND_OFFICIAL / 'model.py')
    from sembind_official_audit import load_official
    from semantic_code import SemanticCodeExtractor

    priv = np.load(SEMBIND_S4 / 'private_material.npz')
    gs_key_np, wm_msg_np, permutation = priv['key'], priv['message'], priv['permutation']
    aux_codes = np.load(SEMBIND_S4 / 'aux_codes.npz')

    paths = SimpleNamespace(official=SEMBIND_OFFICIAL, scheduler=SEMBIND_SCHED, model=SD21_SNAP, semantic=SEMBIND_MODELS)
    inversion, pipe = load_official(paths)
    pipe.set_progress_bar_config(disable=True)
    extractor = SemanticCodeExtractor(SEMBIND_MODELS / 'dinov2-giant', SEMBIND_MODELS / 'sembind_masker.pth', device='cpu')
    native_gs = load_module('native_gs', GS_MOD_PATH).Gaussian_Shading(1, 8, 1e-6, 1)
    native_gs.key = torch.from_numpy(gs_key_np.astype(np.int64)).cuda()
    native_gs.watermark = torch.from_numpy(wm_msg_np.astype(np.int64)).cuda()
    lpips_fn = lpips.LPIPS(net='alex').cuda().eval()

    for gidx in todo:
        cp = cross_cover_path(gidx) if threat == 'cross' else same_cover_path(gidx)
        cover_im = Image.open(cp).convert('RGB').resize((512, 512), Image.BICUBIC)
        aux_c = aux_codes[str(gidx)]
        ckpts = {}
        for step in (50, 100, 150):
            forged_im = Image.open(imp_dir / f'gidx_{gidx:03d}_step_{step:03d}.png').convert('RGB')
            pipe.to('cpu'); torch.cuda.empty_cache()
            extractor.dino.to('cuda'); extractor.model.to('cuda'); extractor.device = 'cuda:0'
            q_code = extractor.code(forged_im).cpu().numpy().astype(np.uint8)
            extractor.dino.to('cpu'); extractor.model.to('cpu'); extractor.device = 'cpu'
            torch.cuda.empty_cache()

            pipe.to('cuda')
            torch.manual_seed(2026112680 + gidx * 1000 + step)
            z = inversion.exact_inversion(
                forged_im, prompt='', guidance_scale=3.0, num_inference_steps=50,
                test_num_inference_steps=50, inv_order=0, decoder_inv=True, pipe=pipe,
            ).detach()
            mask = np.tile(q_code, 16)[permutation].reshape(1, 4, 64, 64)
            unmasked = z * (1.0 - 2.0 * torch.tensor(mask, dtype=torch.float32, device='cuda'))
            ba = float(native_gs.eval_watermark(unmasked))
            ham = int((q_code != aux_c).sum())

            a1 = np.asarray(cover_im, dtype=np.float32) / 255.0
            a2 = np.asarray(forged_im, dtype=np.float32) / 255.0
            mse = float(np.mean((a1 - a2) ** 2))
            psnr = 100.0 if mse < 1e-12 else float(-10.0 * np.log10(mse))
            with torch.no_grad():
                lp = float(lpips_fn(torch.from_numpy(a1.transpose(2, 0, 1)).unsqueeze(0).cuda() * 2.0 - 1.0,
                                    torch.from_numpy(a2.transpose(2, 0, 1)).unsqueeze(0).cuda() * 2.0 - 1.0).item())
            ckpts[f'step_{step}'] = dict(
                loss=saved_meta[gidx]['losses'][step],
                psnr_vs_cover=psnr, lpips_vs_cover=lp,
                ba=ba, accept_gs_default=bool(ba > 181.0 / 256.0), accept_gs_167=bool(ba >= 167.0 / 256.0),
                code_hamming=ham,
            )
            print(f'  [{threat}_sembind] gidx={gidx:03d} step={step:3d} | BA={ba:.4f} (def={ba>181/256}, 167={ba>=167/256}) | H={ham}', flush=True)
        write_json(case_dir / f'{threat}_sembind_gidx{gidx:03d}.json', dict(
            gidx=gidx, prompt=sources[gidx]['prompt'], cover=saved_meta[gidx]['cover'], seconds=saved_meta[gidx]['seconds'], checkpoints=ckpts,
        ))


def run_cross_seal_gen(gindices: list, gpu: int):
    """Strictly isolated SEAL Imprint generation in semantic_forgery_py310 (zero access to SEAL salt)."""
    uuid = check_idle_gpu(gpu)
    meta_path = RUN / 'cross_seal_gen_meta.json'
    meta = read_json(meta_path) if meta_path.exists() else {}
    todo = [g for g in gindices if str(g) not in meta]
    if not todo:
        print('[S6B cross_seal_gen] Already complete.', flush=True)
        return

    work_dir = RUN / 'work_cross_seal_gen'
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

    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sys.path.insert(0, str(SEM_FORGERY))
    from utils import imprint_utils

    imp_dir = RUN / 'imprint_imgs_cross_seal'
    imp_dir.mkdir(parents=True, exist_ok=True)
    pipe_atk, _, inv_sched = imprint_utils.load_attacker_pipe(
        modelid='Manojb/stable-diffusion-2-1-base', scheduler='DDIM', device='cuda'
    )
    pipe_atk.set_progress_bar_config(disable=True)
    diffpipe = imprint_utils.DiffPipe(pipe_atk, scheduler=inv_sched, device=pipe_atk.device)

    for gidx in todo:
        wm_im = Image.open(seal_genuine_path(gidx)).convert('RGB')
        cp = cross_cover_path(gidx)
        cover_im = Image.open(cp).convert('RGB').resize((512, 512), Image.BICUBIC)
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
                forged_im.save(imp_dir / f'gidx_{gidx:03d}_step_{step:03d}.png')
                losses[f'step_{step}'] = float(loss.detach())
        meta[str(gidx)] = dict(gidx=gidx, seal_idx=gidx, cover=cp.name, seconds=time.perf_counter() - t0, losses=losses)
        write_json(meta_path, meta)
        print(f'  [cross_seal_gen] gidx={gidx:03d} done in {meta[str(gidx)]["seconds"]:.0f}s', flush=True)


def do_merge_all():
    proto = read_json(RUN / 'protocol.json')
    case_dir = RUN / 'cases'

    def load_cases(prefix, gindices):
        out = []
        for g in gindices:
            p = case_dir / f'{prefix}_gidx{g:03d}.json'
            assert p.exists(), f'Missing case file {p}'
            out.append(read_json(p))
        return out

    # --- 1. Cross-prompt N=10 ---
    cross_sscd64 = load_cases('cross_sscd64', list(range(10)))

    s4b_imp = read_json(HYB_S4 / 'live_attacks/summary.json')['imprint_cases']
    cross_hyb = s4b_imp + load_cases('cross_hyb', list(range(4, 10)))

    s4c_imp = read_json(PGID_S4 / 'attacks_summary.json')['imprint_cases']
    cross_np = s4c_imp + load_cases('cross_native_pgid', list(range(4, 10)))

    s4d_imp = read_json(SEMBIND_S4 / 'attacks_null_summary.json')['imprint_cases']
    cross_sembind = s4d_imp + load_cases('cross_sembind', list(range(4, 10)))

    s5a_imp = read_json(S5_SEAL / 'attacks_summary.json')['imprint_cases']
    s6_seal_v = read_json(RUN / 'cross_seal_verify_4_9.json')['imprint_cases']
    cross_seal = s5a_imp + s6_seal_v

    for name, arr in [('cross_sscd64', cross_sscd64), ('cross_hyb', cross_hyb), ('cross_np', cross_np), ('cross_sembind', cross_sembind), ('cross_seal', cross_seal)]:
        assert [c['gidx'] for c in arr] == list(range(10)), f'{name} gidx mismatch'

    # --- 2. Same-prompt N=10 ---
    s5b_s = read_json(S5_CS / 'same_prompt_imprint_sscd64_summary.json')['imprint_cases']
    same_sscd64 = s5b_s + load_cases('same_sscd64', list(range(4, 10)))

    s5b_h = read_json(S5_CS / 'same_prompt_imprint_hyb_summary.json')['imprint_cases']
    same_hyb = s5b_h + load_cases('same_hyb', list(range(4, 10)))

    s5c_sb = read_json(S5_SEMBIND / 'instance_forgery_summary.json')['imprint_cases']
    same_sembind = s5c_sb + load_cases('same_sembind', list(range(4, 10)))

    for name, arr in [('same_sscd64', same_sscd64), ('same_hyb', same_hyb), ('same_sembind', same_sembind)]:
        assert [c['gidx'] for c in arr] == list(range(10)), f'{name} gidx mismatch'

    def agg_cs(cases):
        res = {}
        for ck in ('step_50', 'step_100', 'step_150'):
            f_auth = sum(c['checkpoints'][ck]['auth_ok'] for c in cases)
            f_joint = sum(c['checkpoints'][ck]['joint_accept'] for c in cases)
            lo, hi = cp95_bounds(f_joint, 10)
            res[ck] = dict(
                n=10, native_auth_fooled=int(f_auth), commit_steer_fooled=int(f_joint),
                cp95=[lo, hi],
                mean_auth_ba=float(np.mean([c['checkpoints'][ck]['auth_ba'] for c in cases])),
                mean_d_blind=float(np.mean([c['checkpoints'][ck]['d_blind'] for c in cases])),
            )
        return res

    def agg_np(cases):
        res = {}
        for ck in ('step_50', 'step_100', 'step_150'):
            fn = sum(c['checkpoints'][ck]['native_gs']['accept_gs_default'] for c in cases)
            fp = sum(c['checkpoints'][ck]['pgid_f']['accept_gs_default'] for c in cases)
            fp167 = sum(c['checkpoints'][ck]['pgid_f']['accept_gs_167'] for c in cases)
            lo, hi = cp95_bounds(fp, 10)
            res[ck] = dict(
                n=10, native_gs_fooled=int(fn), pgid_f_fooled_default=int(fp), pgid_f_fooled_167=int(fp167),
                pgid_cp95=[lo, hi],
                mean_native_ba=float(np.mean([c['checkpoints'][ck]['native_gs']['ba_256'] for c in cases])),
                mean_pgid_ba=float(np.mean([c['checkpoints'][ck]['pgid_f']['ba_256'] for c in cases])),
            )
        return res

    def agg_sembind(cases):
        res = {}
        for ck in ('step_50', 'step_100', 'step_150'):
            fd = sum(c['checkpoints'][ck]['accept_gs_default'] for c in cases)
            f167 = sum(c['checkpoints'][ck].get('accept_gs_167', c['checkpoints'][ck]['ba'] >= 167.0 / 256.0) for c in cases)
            lo, hi = cp95_bounds(fd, 10)
            res[ck] = dict(
                n=10, fooled_default=int(fd), fooled_167=int(f167),
                cp95=[lo, hi],
                mean_ba=float(np.mean([c['checkpoints'][ck]['ba'] for c in cases])),
                mean_hamming=float(np.mean([c['checkpoints'][ck]['code_hamming'] for c in cases])),
            )
        return res

    def agg_seal(cases):
        res = {}
        for ck in ('step_50', 'step_100', 'step_150'):
            f12 = sum(c['checkpoints'][ck]['accept_12'] for c in cases)
            f4 = sum(c['checkpoints'][ck]['accept_4'] for c in cases)
            f2 = sum(c['checkpoints'][ck]['accept_2'] for c in cases)
            res[ck] = dict(
                n=10, fooled_12=int(f12), fooled_4=int(f4), fooled_2=int(f2),
                cp95_at4=list(cp95_bounds(f4, 10)),
                mean_matches=float(np.mean([c['checkpoints'][ck]['matches'] for c in cases])),
            )
        return res

    cross_summary = dict(
        native_gs_and_pgid_f=agg_np(cross_np),
        seal=agg_seal(cross_seal),
        sembind_gs_s=agg_sembind(cross_sembind),
        commit_steer_sscd64=agg_cs(cross_sscd64),
        commit_steer_hyb=agg_cs(cross_hyb),
    )
    same_summary = dict(
        sembind_gs_s=agg_sembind(same_sembind),
        commit_steer_sscd64=agg_cs(same_sscd64),
        commit_steer_hyb=agg_cs(same_hyb),
    )

    # Exact two-sided Fisher exact tests at step_150 (N=10 vs N=10)
    def fisher_p(a_fooled, b_fooled, n=10):
        _, p = fisher_exact([[a_fooled, n - a_fooled], [b_fooled, n - b_fooled]], alternative='two-sided')
        return float(p)

    stats = dict(
        cross_step150_sscd64_vs_native_gs_p=fisher_p(cross_summary['commit_steer_sscd64']['step_150']['commit_steer_fooled'], cross_summary['native_gs_and_pgid_f']['step_150']['native_gs_fooled']),
        cross_step150_hyb_vs_native_gs_p=fisher_p(cross_summary['commit_steer_hyb']['step_150']['commit_steer_fooled'], cross_summary['native_gs_and_pgid_f']['step_150']['native_gs_fooled']),
        cross_step150_sscd64_vs_pgid_default_p=fisher_p(cross_summary['commit_steer_sscd64']['step_150']['commit_steer_fooled'], cross_summary['native_gs_and_pgid_f']['step_150']['pgid_f_fooled_default']),
        cross_step150_sscd64_vs_pgid_167_p=fisher_p(cross_summary['commit_steer_sscd64']['step_150']['commit_steer_fooled'], cross_summary['native_gs_and_pgid_f']['step_150']['pgid_f_fooled_167']),
        cross_step150_sscd64_vs_seal_at4_p=fisher_p(cross_summary['commit_steer_sscd64']['step_150']['commit_steer_fooled'], cross_summary['seal']['step_150']['fooled_4']),
        same_step150_hyb_vs_sembind_p=fisher_p(same_summary['commit_steer_hyb']['step_150']['commit_steer_fooled'], same_summary['sembind_gs_s']['step_150']['fooled_default']),
        same_step150_sscd64_vs_sembind_p=fisher_p(same_summary['commit_steer_sscd64']['step_150']['commit_steer_fooled'], same_summary['sembind_gs_s']['step_150']['fooled_default']),
    )

    write_json(RUN / 'cross_prompt_n10.json', dict(
        native_gs_and_pgid_f=cross_np, seal=cross_seal, sembind_gs_s=cross_sembind,
        commit_steer_sscd64=cross_sscd64, commit_steer_hyb=cross_hyb,
    ))
    write_json(RUN / 'same_prompt_n10.json', dict(
        sembind_gs_s=same_sembind, commit_steer_sscd64=same_sscd64, commit_steer_hyb=same_hyb,
    ))
    write_json(RUN / 'summary.json', dict(
        n_cases=10,
        cross_prompt=cross_summary,
        same_prompt=same_summary,
        fisher_exact_two_sided_step150=stats,
    ))
    print(f"[S6B MERGE COMPLETE] Cross step150: Native={cross_summary['native_gs_and_pgid_f']['step_150']['native_gs_fooled']}/10 | "
          f"PGID={cross_summary['native_gs_and_pgid_f']['step_150']['pgid_f_fooled_default']}/10 | "
          f"SEAL@12/4/2={cross_summary['seal']['step_150']['fooled_12']}/{cross_summary['seal']['step_150']['fooled_4']}/{cross_summary['seal']['step_150']['fooled_2']} | "
          f"SemBind={cross_summary['sembind_gs_s']['step_150']['fooled_default']}/10 | "
          f"sscd64={cross_summary['commit_steer_sscd64']['step_150']['commit_steer_fooled']}/10 | "
          f"hyb={cross_summary['commit_steer_hyb']['step_150']['commit_steer_fooled']}/10", flush=True)
    print(f"[S6B MERGE COMPLETE] Same-Prompt step150: SemBind={same_summary['sembind_gs_s']['step_150']['fooled_default']}/10 | "
          f"sscd64={same_summary['commit_steer_sscd64']['step_150']['commit_steer_fooled']}/10 (p={stats['same_step150_sscd64_vs_sembind_p']:.4g}) | "
          f"hyb={same_summary['commit_steer_hyb']['step_150']['commit_steer_fooled']}/10 (p={stats['same_step150_hyb_vs_sembind_p']:.4g})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=[
        'prepare', 'cross_sscd64_p0', 'cross_sscd64_p1', 'cross_hyb', 'cross_native_pgid',
        'cross_sembind', 'cross_seal_gen', 'same_sscd64', 'same_hyb', 'same_sembind', 'merge_all',
    ])
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    if args.mode == 'prepare':
        do_prepare()
    elif args.mode == 'cross_sscd64_p0':
        run_cs_imprint('sscd64', 'cross', list(range(0, 5)), args.gpu)
    elif args.mode == 'cross_sscd64_p1':
        run_cs_imprint('sscd64', 'cross', list(range(5, 10)), args.gpu)
    elif args.mode == 'cross_hyb':
        run_cs_imprint('hyb', 'cross', list(range(4, 10)), args.gpu)
    elif args.mode == 'cross_native_pgid':
        run_cross_native_pgid(list(range(4, 10)), args.gpu)
    elif args.mode == 'cross_sembind':
        run_sembind_imprint('cross', list(range(4, 10)), args.gpu)
    elif args.mode == 'cross_seal_gen':
        run_cross_seal_gen(list(range(4, 10)), args.gpu)
    elif args.mode == 'same_sscd64':
        run_cs_imprint('sscd64', 'same', list(range(4, 10)), args.gpu)
    elif args.mode == 'same_hyb':
        run_cs_imprint('hyb', 'same', list(range(4, 10)), args.gpu)
    elif args.mode == 'same_sembind':
        run_sembind_imprint('same', list(range(4, 10)), args.gpu)
    elif args.mode == 'merge_all':
        do_merge_all()


if __name__ == '__main__':
    main()
