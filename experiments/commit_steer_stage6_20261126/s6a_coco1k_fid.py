"""Stage 6A: Standard COCO-1K Quality (FID vs Real-5K & Real-1K, Paired FID, LPIPS, PSNR, CLIP)
+ 1,000-Image Unseen Clean Blind Verification for Commit-then-Steer sscd_64b and hyb_sscd32_clip32.

Subcommands:
  prepare         - Freeze 1,000 non-overlapping MS-COCO val2017 prompts + 5,000 real MS-COCO test2017 reference images.
  run_shard       - Generate paired (Native GS unguided, Commit-then-Steer guided) images for BOTH sscd_64b & hyb_sscd32_clip32,
                    compute LPIPS/PSNR/CLIP, and run batched 50-step DDIM blind verification (resumable every 10 prompts).
  finalize_fid    - Extract Inception-v3 pool3 (verified offline weights) for Real-5K, Real-1K, and all 4 x 1,000 generated sets,
                    compute exact standard FID (verified by both scipy.linalg.sqrtm and exact SVD), aggregate summary & showcase.
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

import numpy as np
from PIL import Image, ImageDraw
from scipy import linalg

ROOT = Path('/data0/liyanlei/watermark_research')
STAGE6 = ROOT / 'runs/commit_steer_stage6_20261126'
RUN = STAGE6 / 'coco1k_fid'
GS100_S3 = ROOT / 'runs/commit_steer_paper_scale_20261123/gs100_unified'
HYB_S4 = ROOT / 'runs/commit_steer_stage4_20261125/multidim100_hyb'
PHASE1_FID = ROOT / 'runs/rgb_moments_phase1_20261120'
GS_MOD_PATH = ROOT / 'runs/phase1_trajectory_native_gs_20261113/gs_watermark.py'
PGID_PIPE = ROOT / 'third_party/frozen/PGID_dc4ad96'
EXTRA_DEPS = ROOT / 'envs/phase1_extra'
SSCD_PATH = ROOT / 'models/sscd/sscd_disc_mixup.torchscript.pt'
COCO_VAL_CAPTIONS = ROOT / 'datasets/confirmation100_relay_v11/metadata/captions_val2017.json'
COCO_TEST_DIR = ROOT / 'datasets/coco_test2017_official/test2017'
UNIFIED_SEL = ROOT / 'runs/unified_main_sources_20261116/selection.json'
FID_WEIGHTS_SHA256 = '6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2'


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


def norm_prompt(s: str) -> str:
    return ' '.join(s.strip().lower().split())


def do_prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / 'protocol.json').exists():
        proto = read_json(RUN / 'protocol.json')
        print(f"[S6A prepare] Reusing frozen protocol: {len(proto['prompts'])} prompts, {len(proto['real_ref_5k'])} real refs", flush=True)
        return

    # Collect all excluded prompts and COCO image_ids from unified_main_sources & gs100_unified
    excluded = set()
    excluded_img_ids = set()
    uni = read_json(UNIFIED_SEL)
    for item in uni.get('sources', []):
        if isinstance(item, dict):
            if 'prompt' in item:
                excluded.add(norm_prompt(item['prompt']))
            if item.get('dataset') == 'COCO' and item.get('image_id') is not None:
                excluded_img_ids.add(int(item['image_id']))
    s3_proto = read_json(GS100_S3 / 'protocol.json')
    for s in s3_proto['sources']:
        excluded.add(norm_prompt(s['prompt']))
        if s.get('dataset') == 'COCO' and s.get('image_id') is not None:
            excluded_img_ids.add(int(s['image_id']))

    # Deterministically sample 1,000 non-overlapping MS-COCO val2017 unique-image prompts
    coco_cap = read_json(COCO_VAL_CAPTIONS)
    by_img = {}
    for a in sorted(coco_cap['annotations'], key=lambda x: (x['image_id'], x['id'])):
        by_img.setdefault(a['image_id'], []).append(a['caption'].strip())

    candidates = []
    seen_norm = set()
    for img_id in sorted(by_img.keys()):
        if int(img_id) in excluded_img_ids:
            continue
        chosen = None
        for cap in by_img[img_id]:
            np_cap = norm_prompt(cap)
            if len(cap) >= 15 and np_cap not in excluded and np_cap not in seen_norm:
                chosen = cap
                seen_norm.add(np_cap)
                break
        if chosen is not None:
            candidates.append(dict(coco_image_id=int(img_id), prompt=chosen))

    assert len(candidates) >= 1000, f'Not enough non-overlapping val2017 captions: {len(candidates)}'
    rng_p = np.random.default_rng(2026112601)
    perm_p = rng_p.permutation(len(candidates))[:1000]
    prompts = []
    for new_idx, ci in enumerate(sorted(perm_p.tolist())):
        prompts.append(dict(idx=new_idx, coco_image_id=candidates[ci]['coco_image_id'], prompt=candidates[ci]['prompt']))

    # Deterministically sample 5,000 real MS-COCO test2017 reference images and verify readable
    all_test = sorted(p.name for p in COCO_TEST_DIR.glob('*.jpg'))
    assert len(all_test) >= 5000, f'Expected >=5000 test2017 images, got {len(all_test)}'
    rng_r = np.random.default_rng(2026112602)
    perm_r = rng_r.permutation(len(all_test))
    real_ref_5k = []
    for ri in perm_r:
        fname = all_test[int(ri)]
        try:
            with Image.open(COCO_TEST_DIR / fname) as im:
                im.verify()
            real_ref_5k.append(fname)
            if len(real_ref_5k) == 5000:
                break
        except Exception:
            continue
    assert len(real_ref_5k) == 5000

    s4b_proto = read_json(HYB_S4 / 'protocol.json')
    proto = dict(
        run_name='coco1k_fid',
        n_prompts=1000,
        n_real_ref_5k=5000,
        n_real_ref_1k=1000,
        prompt_source=str(COCO_VAL_CAPTIONS),
        real_ref_source=str(COCO_TEST_DIR),
        excluded_prompt_count=len(excluded),
        sscd64_protocol_sha256=sha256_file(GS100_S3 / 'protocol.json'),
        hyb_protocol_sha256=sha256_file(HYB_S4 / 'protocol.json'),
        sscd64_threshold=float(s3_proto['threshold']),
        hyb_threshold=float(s4b_proto['threshold']),
        auth_threshold=float(s3_proto['auth_threshold']),
        steering_params=dict(preview_steps=12, step_range=[5, 35], step_size=0.042, stop_loss=0.08, num_steps=50),
        prompts=prompts,
        real_ref_5k=real_ref_5k,
    )
    write_json(RUN / 'protocol.json', proto)
    print(f'[S6A prepare] Froze 1,000 unseen val2017 prompts (0 overlap with {len(excluded)} excluded) + 5,000 verified real test2017 refs.', flush=True)


def do_run_shard(shard: int, num_shards: int, gpu: int):
    uuid = check_idle_gpu(gpu)
    proto = read_json(RUN / 'protocol.json')
    prompts = proto['prompts']
    th_sscd = float(proto['sscd64_threshold'])
    th_hyb = float(proto['hyb_threshold'])
    auth_th = float(proto['auth_threshold'])

    priv_s3 = np.load(GS100_S3 / 'private_material.npz')
    key_s3, auth_s3, P64_np = priv_s3['key'], priv_s3['auth_bits'], priv_s3['P64']

    priv_hyb = np.load(HYB_S4 / 'private_material.npz')
    key_hyb, auth_hyb, P_s32_np, P_c32_np = priv_hyb['key'], priv_hyb['auth_bits'], priv_hyb['P_s32'], priv_hyb['P_c32']

    work_dir = RUN / f'work_s{shard}'
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
    from transformers import CLIPModel, CLIPProcessor

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sys.path.insert(0, str(PGID_PIPE))
    sys.path.insert(0, str(EXTRA_DEPS))
    from pipe import pipe_utils

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

    P64_t = torch.from_numpy(P64_np).float().cuda()
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

    def extract_feats(pil_list, need_clip=True):
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

    def diff_sscd_loss(x, ts_t):
        x_in = (F.interpolate(x, size=(320, 320), mode='bilinear', align_corners=False) - im_mean) / im_std
        proj = F.normalize(sscd(x_in), dim=1)[0] @ P64_t
        cos_d = 1.0 - F.cosine_similarity(proj.unsqueeze(0), ts_t.unsqueeze(0)).squeeze()
        margin = F.softplus(0.05 - proj * ts_t, beta=20.0).mean()
        return cos_d + 0.5 * margin

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

    def ddim_gen(prompt, zT, num_steps=50, ts_t=None, loss_fn=None, step_range=(5, 35), step_size=0.042, stop_loss=0.08):
        pipe.scheduler = provider.scheduler
        pipe.scheduler.set_timesteps(num_steps, device='cuda')
        with torch.no_grad():
            pos, neg = pipe.encode_prompt(
                prompt=prompt, device='cuda', num_images_per_prompt=1,
                do_classifier_free_guidance=True, negative_prompt='',
            )
            te = torch.cat([neg, pos], dim=0)
        z = zT.clone().to(device='cuda', dtype=torch.float16)
        alphas = pipe.scheduler.alphas_cumprod.to('cuda')
        for idx_t, t in enumerate(pipe.scheduler.timesteps):
            with torch.no_grad():
                eps = pipe.unet(
                    pipe.scheduler.scale_model_input(torch.cat([z, z], dim=0), t), t,
                    encoder_hidden_states=te, return_dict=False,
                )[0]
                eu, ec = eps.chunk(2)
                out = pipe.scheduler.step(eu + 7.5 * (ec - eu), t, z, eta=0.0, return_dict=True)
                zn, px0 = out.prev_sample, out.pred_original_sample
            if ts_t is not None and step_range[0] <= idx_t <= step_range[1]:
                z0l = px0.detach().float().requires_grad_(True)
                loss = loss_fn(decode_x(z0l), ts_t)
                if float(loss.detach()) > stop_loss:
                    g = torch.autograd.grad(loss, z0l)[0]
                    gn = float(g.norm()) + 1e-8
                    pt = t - pipe.scheduler.config.num_train_timesteps // num_steps
                    ap = alphas[pt] if pt >= 0 else pipe.scheduler.final_alpha_cumprod.to('cuda')
                    zn = (zn.float() + torch.sqrt(ap.float()) * (-(step_size * float(z0l.detach().norm()) / gn) * g)).to(torch.float16)
            z = zn
        with torch.no_grad():
            return to_pil(decode_x(z))

    def eval_pair_metrics(base_im, guid_im, prompt):
        a1 = np.asarray(base_im, dtype=np.float32) / 255.0
        a2 = np.asarray(guid_im, dtype=np.float32) / 255.0
        mse = float(np.mean((a1 - a2) ** 2))
        psnr = 100.0 if mse < 1e-12 else float(-10.0 * np.log10(mse))
        t1 = torch.from_numpy(a1.transpose(2, 0, 1)).unsqueeze(0).cuda() * 2.0 - 1.0
        t2 = torch.from_numpy(a2.transpose(2, 0, 1)).unsqueeze(0).cuda() * 2.0 - 1.0
        with torch.no_grad():
            lp = float(lpips_fn(t1, t2).item())
            inp = clip_proc(text=[prompt], images=[base_im, guid_im], return_tensors='pt', padding=True, truncation=True, max_length=77).to('cuda')
            out = clip_model(**inp)
            ie = F.normalize(out.image_embeds, dim=-1)
            te = F.normalize(out.text_embeds, dim=-1)
            c_base = float((ie[0] * te[0]).sum().item())
            c_guid = float((ie[1] * te[0]).sum().item())
        return lp, psnr, c_base, c_guid

    def score_sscd64_clean(zT_1, es_1, target_bits):
        rev_w = gs_s3.diffusion_inverse(((zT_1 > 0).int() + gs_s3.key) % 2).reshape(-1).cpu().numpy().astype(np.uint8)
        auth_ba = float((rev_w[:192] == auth_s3).mean())
        pay_ber = float((rev_w[192:] != target_bits).mean())
        auth_ok = auth_ba > auth_th
        rec_sgn = 2.0 * rev_w[192:].astype(np.float64) - 1.0
        qp = es_1 @ P64_np
        d_blind = float(1.0 - np.dot(rec_sgn, qp) / (8.0 * float(np.linalg.norm(qp)) + 1e-12))
        feat_ok = d_blind <= th_sscd
        return dict(auth_ba=auth_ba, pay_ber=pay_ber, auth_ok=bool(auth_ok), d_blind=d_blind, feat_ok=bool(feat_ok), joint_accept=bool(auth_ok and feat_ok))

    def score_hyb_clean(zT_1, es_1, ec_1, target_bits):
        rev_w = gs_hyb.diffusion_inverse(((zT_1 > 0).int() + gs_hyb.key) % 2).reshape(-1).cpu().numpy().astype(np.uint8)
        auth_ba = float((rev_w[:192] == auth_hyb).mean())
        pay_ber = float((rev_w[192:] != target_bits).mean())
        auth_ok = auth_ba > auth_th
        rec_sgn = 2.0 * rev_w[192:].astype(np.float64) - 1.0
        qps, qpc = es_1 @ P_s32_np, ec_1 @ P_c32_np
        ds = float(1.0 - np.dot(rec_sgn[:32], qps) / (np.sqrt(32.0) * float(np.linalg.norm(qps)) + 1e-12))
        dc = float(1.0 - np.dot(rec_sgn[32:], qpc) / (np.sqrt(32.0) * float(np.linalg.norm(qpc)) + 1e-12))
        d_blind = 0.5 * ds + 0.5 * dc
        feat_ok = d_blind <= th_hyb
        return dict(auth_ba=auth_ba, pay_ber=pay_ber, auth_ok=bool(auth_ok), d_sscd32=ds, d_clip32=dc, d_blind=d_blind, feat_ok=bool(feat_ok), joint_accept=bool(auth_ok and feat_ok))

    img_root = RUN / 'images'
    for sub in ('base_sscd', 'guid_sscd', 'base_hyb', 'guid_hyb'):
        (img_root / sub).mkdir(parents=True, exist_ok=True)

    rec_path = RUN / f'shard_{shard}_records.json'
    done_by_idx = {}
    if rec_path.exists():
        for r in read_json(rec_path):
            done_by_idx[r['idx']] = r
        print(f'[S6A shard {shard}] Resuming with {len(done_by_idx)} already completed prompts.', flush=True)

    with torch.no_grad():
        pipe('warmup', generator=torch.Generator(device='cuda').manual_seed(7), num_inference_steps=2)

    my_items = [p for p in prompts if p['idx'] % num_shards == shard]
    pending_verify = []
    t_start = time.perf_counter()

    def flush_verify_buffer():
        if not pending_verify:
            return
        sscd_ims = [x['guid_sscd_im'] for x in pending_verify]
        hyb_ims = [x['guid_hyb_im'] for x in pending_verify]
        zTs_s, zTs_h = [], []
        for i in range(0, len(sscd_ims), 5):
            zTs_s.append(provider.invert_images(sscd_ims[i:i + 5], num_inference_steps=50)['zT_torch'])
            zTs_h.append(provider.invert_images(hyb_ims[i:i + 5], num_inference_steps=50)['zT_torch'])
        zTs_s = torch.cat(zTs_s, dim=0)
        zTs_h = torch.cat(zTs_h, dim=0)
        es_s, _ = extract_feats(sscd_ims, need_clip=False)
        es_h, ec_h = extract_feats(hyb_ims, need_clip=True)
        for bi, item in enumerate(pending_verify):
            sc_s = score_sscd64_clean(zTs_s[bi:bi + 1], es_s[bi], item['bits_s'])
            sc_h = score_hyb_clean(zTs_h[bi:bi + 1], es_h[bi], ec_h[bi], item['bits_h'])
            rec = item['rec']
            rec['sscd64']['clean_verify'] = sc_s
            rec['hyb']['clean_verify'] = sc_h
            done_by_idx[rec['idx']] = rec
        pending_verify.clear()
        ordered = [done_by_idx[k] for k in sorted(done_by_idx.keys())]
        write_json(rec_path, ordered)

    for local_i, p in enumerate(my_items):
        idx = p['idx']
        if idx in done_by_idx:
            continue
        prompt = p['prompt']

        # --- Arm 1: sscd_64b ---
        t0 = time.perf_counter()
        zT_prev_s = torch.randn((1, 4, 64, 64), generator=torch.Generator(device='cuda').manual_seed(2026112600 + idx), device='cuda', dtype=torch.float16)
        prev_s = ddim_gen(prompt, zT_prev_s, num_steps=12, ts_t=None)
        es_p, _ = extract_feats([prev_s], need_clip=False)
        bits_s = ((es_p[0] @ P64_np) >= 0.0).astype(np.uint8)
        ts_s = torch.tensor(2.0 * bits_s.astype(np.float64) - 1.0, device='cuda', dtype=torch.float32)
        w_s = np.concatenate([auth_s3, bits_s]).reshape(1, 4, 8, 8)
        m_s = ((np.tile(w_s, (1, 1, 8, 8)) + key_s3) % 2).reshape(-1)
        np.random.seed(2026112610 + idx)
        zT_s = gs_s3.truncSampling(m_s)
        sec_prev_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        base_s = ddim_gen(prompt, zT_s, num_steps=50, ts_t=None)
        sec_base_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        guid_s = ddim_gen(prompt, zT_s, num_steps=50, ts_t=ts_s, loss_fn=diff_sscd_loss, step_range=(5, 35), step_size=0.042, stop_loss=0.08)
        sec_guid_s = time.perf_counter() - t0
        lp_s, psnr_s, cb_s, cg_s = eval_pair_metrics(base_s, guid_s, prompt)
        base_s.save(img_root / 'base_sscd' / f'{idx:04d}.png')
        guid_s.save(img_root / 'guid_sscd' / f'{idx:04d}.png')

        # --- Arm 2: hyb_sscd32_clip32 ---
        t0 = time.perf_counter()
        zT_prev_h = torch.randn((1, 4, 64, 64), generator=torch.Generator(device='cuda').manual_seed(2026112700 + idx), device='cuda', dtype=torch.float16)
        prev_h = ddim_gen(prompt, zT_prev_h, num_steps=12, ts_t=None)
        es_ph, ec_ph = extract_feats([prev_h], need_clip=True)
        bits_h = np.concatenate([
            ((es_ph[0] @ P_s32_np) >= 0.0).astype(np.uint8),
            ((ec_ph[0] @ P_c32_np) >= 0.0).astype(np.uint8),
        ])
        ts_h = torch.tensor(2.0 * bits_h.astype(np.float64) - 1.0, device='cuda', dtype=torch.float32)
        w_h = np.concatenate([auth_hyb, bits_h]).reshape(1, 4, 8, 8)
        m_h = ((np.tile(w_h, (1, 1, 8, 8)) + key_hyb) % 2).reshape(-1)
        np.random.seed(2026112710 + idx)
        zT_h = gs_hyb.truncSampling(m_h)
        sec_prev_h = time.perf_counter() - t0

        t0 = time.perf_counter()
        base_h = ddim_gen(prompt, zT_h, num_steps=50, ts_t=None)
        sec_base_h = time.perf_counter() - t0
        t0 = time.perf_counter()
        guid_h = ddim_gen(prompt, zT_h, num_steps=50, ts_t=ts_h, loss_fn=diff_hyb_loss, step_range=(5, 35), step_size=0.042, stop_loss=0.08)
        sec_guid_h = time.perf_counter() - t0
        lp_h, psnr_h, cb_h, cg_h = eval_pair_metrics(base_h, guid_h, prompt)
        base_h.save(img_root / 'base_hyb' / f'{idx:04d}.png')
        guid_h.save(img_root / 'guid_hyb' / f'{idx:04d}.png')

        rec = dict(
            idx=idx,
            coco_image_id=p['coco_image_id'],
            sscd64=dict(lpips=lp_s, psnr=psnr_s, clip_base=cb_s, clip_guid=cg_s, sec_prev=sec_prev_s, sec_base=sec_base_s, sec_guid=sec_guid_s),
            hyb=dict(lpips=lp_h, psnr=psnr_h, clip_base=cb_h, clip_guid=cg_h, sec_prev=sec_prev_h, sec_base=sec_base_h, sec_guid=sec_guid_h),
        )
        pending_verify.append(dict(rec=rec, guid_sscd_im=guid_s, guid_hyb_im=guid_h, bits_s=bits_s, bits_h=bits_h))

        if len(pending_verify) >= 10:
            flush_verify_buffer()
            n_done = len(done_by_idx)
            s_ok = sum(r['sscd64']['clean_verify']['joint_accept'] for r in done_by_idx.values())
            h_ok = sum(r['hyb']['clean_verify']['joint_accept'] for r in done_by_idx.values())
            print(
                f'[S6A shard {shard}] {n_done}/{len(my_items)} ({time.perf_counter()-t_start:.0f}s) | '
                f'sscd64 clean={s_ok}/{n_done} lpips={np.mean([r["sscd64"]["lpips"] for r in done_by_idx.values()]):.3f} | '
                f'hyb clean={h_ok}/{n_done} lpips={np.mean([r["hyb"]["lpips"] for r in done_by_idx.values()]):.3f}',
                flush=True,
            )

    flush_verify_buffer()
    assert len(done_by_idx) == len(my_items), f'Shard {shard} incomplete: {len(done_by_idx)} vs {len(my_items)}'
    print(f'[S6A shard {shard} COMPLETE] {len(done_by_idx)} prompts in {time.perf_counter()-t_start:.1f}s', flush=True)


def fid_sqrtm(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    mu1, mu2 = np.mean(a, axis=0), np.mean(b, axis=0)
    sigma1, sigma2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        assert np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3), f'Large imaginary part: {np.max(np.abs(covmean.imag))}'
        covmean = covmean.real
    return float(max(0.0, diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)))


def fid_svd(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    ma, mb = a.mean(0), b.mean(0)
    x, y = (a - ma) / np.sqrt(len(a) - 1), (b - mb) / np.sqrt(len(b) - 1)
    cross = np.linalg.svd(x @ y.T, compute_uv=False).sum()
    return float(max(0.0, ((ma - mb) ** 2).sum() + (x * x).sum() + (y * y).sum() - 2.0 * cross))


def do_finalize_fid(gpu: int):
    proto = read_json(RUN / 'protocol.json')
    all_recs = []
    for p in sorted(RUN.glob('shard_*_records.json')):
        all_recs.extend(read_json(p))
    all_recs.sort(key=lambda r: r['idx'])
    assert len(all_recs) == 1000 and [r['idx'] for r in all_recs] == list(range(1000)), f'Expected 1000 records, got {len(all_recs)}'

    if gpu >= 0:
        uuid = check_idle_gpu(gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid
        dev = 'cuda'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        dev = 'cpu'

    import torch
    import torchvision.transforms.functional as TF
    torch.set_num_threads(4)
    sys.path.insert(0, str(PHASE1_FID / 'assets/lib'))
    from pytorch_fid import inception

    weights = PHASE1_FID / 'assets/fid.pth'
    assert sha256_file(weights) == FID_WEIGHTS_SHA256
    inception.load_state_dict_from_url = lambda *a, **k: torch.load(weights, map_location='cpu', weights_only=True)
    net = inception.InceptionV3().to(dev).eval()

    def extract_pool3(paths, bs=64):
        feats = []
        for i in range(0, len(paths), bs):
            batch = []
            for p in paths[i:i + bs]:
                with Image.open(p) as im:
                    rgb = im.convert('RGB')
                    if rgb.size != (512, 512):
                        # Center-crop square then resize to 512x512 (standard MS-COCO FID preprocessing)
                        w, h = rgb.size
                        s = min(w, h)
                        left, top = (w - s) // 2, (h - s) // 2
                        rgb = rgb.crop((left, top, left + s, top + s)).resize((512, 512), Image.BICUBIC)
                    batch.append(TF.to_tensor(rgb))
            x = torch.stack(batch).to(dev)
            with torch.no_grad():
                f = net(x)[0].squeeze(-1).squeeze(-1).cpu().numpy().astype(np.float32)
            feats.append(f)
        return np.concatenate(feats, axis=0)

    t0 = time.perf_counter()
    real_5k_paths = [COCO_TEST_DIR / fn for fn in proto['real_ref_5k']]
    img_root = RUN / 'images'
    f_real5k = extract_pool3(real_5k_paths, bs=64)
    f_real1k = f_real5k[:1000]
    f_base_s = extract_pool3([img_root / 'base_sscd' / f'{i:04d}.png' for i in range(1000)], bs=64)
    f_guid_s = extract_pool3([img_root / 'guid_sscd' / f'{i:04d}.png' for i in range(1000)], bs=64)
    f_base_h = extract_pool3([img_root / 'base_hyb' / f'{i:04d}.png' for i in range(1000)], bs=64)
    f_guid_h = extract_pool3([img_root / 'guid_hyb' / f'{i:04d}.png' for i in range(1000)], bs=64)

    feat_dir = RUN / 'pool3_cache'
    feat_dir.mkdir(exist_ok=True)
    np.save(feat_dir / 'real_coco_5k.npy', f_real5k)
    np.save(feat_dir / 'base_sscd_1k.npy', f_base_s)
    np.save(feat_dir / 'guid_sscd_1k.npy', f_guid_s)
    np.save(feat_dir / 'base_hyb_1k.npy', f_base_h)
    np.save(feat_dir / 'guid_hyb_1k.npy', f_guid_h)

    def calc_fid_checked(a, b, tag):
        v1 = fid_sqrtm(a, b)
        v2 = fid_svd(a, b)
        diff = abs(v1 - v2)
        assert diff < 1e-2, f'FID mismatch on {tag}: sqrtm={v1} svd={v2} diff={diff}'
        return v1, diff

    fid_base_s_5k, d1 = calc_fid_checked(f_base_s, f_real5k, 'base_sscd_vs_5k')
    fid_guid_s_5k, d2 = calc_fid_checked(f_guid_s, f_real5k, 'guid_sscd_vs_5k')
    fid_base_h_5k, d3 = calc_fid_checked(f_base_h, f_real5k, 'base_hyb_vs_5k')
    fid_guid_h_5k, d4 = calc_fid_checked(f_guid_h, f_real5k, 'guid_hyb_vs_5k')

    fid_base_s_1k, _ = calc_fid_checked(f_base_s, f_real1k, 'base_sscd_vs_1k')
    fid_guid_s_1k, _ = calc_fid_checked(f_guid_s, f_real1k, 'guid_sscd_vs_1k')
    fid_base_h_1k, _ = calc_fid_checked(f_base_h, f_real1k, 'base_hyb_vs_1k')
    fid_guid_h_1k, _ = calc_fid_checked(f_guid_h, f_real1k, 'guid_hyb_vs_1k')

    fid_paired_s, _ = calc_fid_checked(f_guid_s, f_base_s, 'paired_sscd')
    fid_paired_h, _ = calc_fid_checked(f_guid_h, f_base_h, 'paired_hyb')
    fid_base_sanity, _ = calc_fid_checked(f_base_s, f_base_h, 'base_sanity')

    def summarize_arm(arm_key):
        lps = [r[arm_key]['lpips'] for r in all_recs]
        psnrs = [r[arm_key]['psnr'] for r in all_recs]
        cbs = [r[arm_key]['clip_base'] for r in all_recs]
        cgs = [r[arm_key]['clip_guid'] for r in all_recs]
        auth_ok = sum(r[arm_key]['clean_verify']['auth_ok'] for r in all_recs)
        feat_ok = sum(r[arm_key]['clean_verify']['feat_ok'] for r in all_recs)
        joint_ok = sum(r[arm_key]['clean_verify']['joint_accept'] for r in all_recs)
        bers = [r[arm_key]['clean_verify']['pay_ber'] for r in all_recs]
        ds = [r[arm_key]['clean_verify']['d_blind'] for r in all_recs]
        return dict(
            n=1000,
            lpips_mean=float(np.mean(lps)), lpips_median=float(np.median(lps)), lpips_std=float(np.std(lps)),
            psnr_mean=float(np.mean(psnrs)), psnr_median=float(np.median(psnrs)),
            clip_base_mean=float(np.mean(cbs)), clip_guid_mean=float(np.mean(cgs)), clip_delta=float(np.mean(cgs) - np.mean(cbs)),
            clean_auth_pass=int(auth_ok), clean_feat_pass=int(feat_ok), clean_joint_pass=int(joint_ok),
            clean_pay_ber_mean=float(np.mean(bers)), clean_d_blind_mean=float(np.mean(ds)),
            sec_base_mean=float(np.mean([r[arm_key]['sec_base'] for r in all_recs])),
            sec_guid_total_mean=float(np.mean([r[arm_key]['sec_prev'] + r[arm_key]['sec_guid'] for r in all_recs])),
        )

    summary = dict(
        n_prompts=1000,
        n_real_ref_5k=5000,
        n_real_ref_1k=1000,
        fid_elapsed_seconds=time.perf_counter() - t0,
        fid_sqrtm_vs_svd_max_diff=float(max(d1, d2, d3, d4)),
        fid=dict(
            vs_real_coco_5k=dict(
                native_gs_sscd_seed=fid_base_s_5k,
                commit_steer_sscd64=fid_guid_s_5k,
                delta_sscd64=fid_guid_s_5k - fid_base_s_5k,
                native_gs_hyb_seed=fid_base_h_5k,
                commit_steer_hyb=fid_guid_h_5k,
                delta_hyb=fid_guid_h_5k - fid_base_h_5k,
            ),
            vs_real_coco_1k=dict(
                native_gs_sscd_seed=fid_base_s_1k,
                commit_steer_sscd64=fid_guid_s_1k,
                delta_sscd64=fid_guid_s_1k - fid_base_s_1k,
                native_gs_hyb_seed=fid_base_h_1k,
                commit_steer_hyb=fid_guid_h_1k,
                delta_hyb=fid_guid_h_1k - fid_base_h_1k,
            ),
            paired_drift_1k=dict(
                sscd64_vs_paired_unguided=fid_paired_s,
                hyb_vs_paired_unguided=fid_paired_h,
                unguided_seed_sanity_check=fid_base_sanity,
            ),
        ),
        sscd64=summarize_arm('sscd64'),
        hyb_sscd32_clip32=summarize_arm('hyb'),
    )
    write_json(RUN / 'records.json', all_recs)
    write_json(RUN / 'summary.json', summary)

    # Visual showcase montage (8 deterministic unseen COCO-1K samples: Unguided | sscd_64b | hyb)
    pick = [0, 125, 250, 375, 500, 625, 750, 875]
    sz, pad = 256, 26
    canvas = Image.new('RGB', (3 * sz, len(pick) * sz + pad), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for ci, hdr in enumerate(['Native GS Unguided', 'Commit-Steer sscd_64b', 'Commit-Steer hyb (32+32)']):
        draw.text((ci * sz + 6, 6), hdr, fill=(0, 0, 0))
    for ri, idx in enumerate(pick):
        r = all_recs[idx]
        ims = [
            Image.open(img_root / 'base_sscd' / f'{idx:04d}.png').convert('RGB'),
            Image.open(img_root / 'guid_sscd' / f'{idx:04d}.png').convert('RGB'),
            Image.open(img_root / 'guid_hyb' / f'{idx:04d}.png').convert('RGB'),
        ]
        for ci, im in enumerate(ims):
            canvas.paste(im.resize((sz, sz), Image.BICUBIC), (ci * sz, pad + ri * sz))
        draw.rectangle([sz, pad + ri * sz, sz + 140, pad + ri * sz + 16], fill=(0, 0, 0))
        draw.text((sz + 4, pad + ri * sz + 2), f"LP={r['sscd64']['lpips']:.3f} d={r['sscd64']['clean_verify']['d_blind']:.2f}", fill=(255, 255, 255))
        draw.rectangle([2 * sz, pad + ri * sz, 2 * sz + 140, pad + ri * sz + 16], fill=(0, 0, 0))
        draw.text((2 * sz + 4, pad + ri * sz + 2), f"LP={r['hyb']['lpips']:.3f} d={r['hyb']['clean_verify']['d_blind']:.2f}", fill=(255, 255, 255))
    canvas.save(RUN / 'coco1k_showcase.jpg', quality=90)
    f5 = summary['fid']['vs_real_coco_5k']
    print(
        f"[S6A COCO-1K FINALIZED] vs Real-5K FID: "
        f"sscd64 {f5['native_gs_sscd_seed']:.2f}->{f5['commit_steer_sscd64']:.2f} ({f5['delta_sscd64']:+.2f}) | "
        f"hyb {f5['native_gs_hyb_seed']:.2f}->{f5['commit_steer_hyb']:.2f} ({f5['delta_hyb']:+.2f}) | "
        f"Clean TPR: sscd64={summary['sscd64']['clean_joint_pass']}/1000, hyb={summary['hyb_sscd32_clip32']['clean_joint_pass']}/1000",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'run_shard', 'finalize_fid'])
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=3)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    if args.mode == 'prepare':
        do_prepare()
    elif args.mode == 'run_shard':
        do_run_shard(args.shard, args.num_shards, args.gpu)
    elif args.mode == 'finalize_fid':
        do_finalize_fid(args.gpu)


if __name__ == '__main__':
    main()
