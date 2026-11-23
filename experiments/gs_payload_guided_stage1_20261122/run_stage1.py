"""Stage 1: Native 256-bit Gaussian Shading with Forward-Committed Payload (48/64-bit)
and Guided Diffusion Generation (Terminal Latent Gauss-Newton / Adam vs Mid-Step Jacobian-Free Tweedie).
Strictly runs on server 249 on idle authorized GPUs (bound by UUID, never GPU 6).
Supports --prepare, --worker --shard S --num-shards K --gpu G, and --merge.
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
from PIL import Image

ROOT = Path('/data0/liyanlei/watermark_research')
RUN = ROOT / 'runs/gs_payload_guided_stage1_20261122'
STAGE0 = ROOT / 'runs/payload_binding_stage0_20261122'
LOADER = ROOT / 'runs/order_capsule_confirmation_20261115'
GS_MOD_PATH = ROOT / 'runs/phase1_trajectory_native_gs_20261113/gs_watermark.py'
PGID_PIPE = ROOT / 'third_party/frozen/PGID_dc4ad96'
EXTRA_DEPS = ROOT / 'envs/phase1_extra'
SSCD_PATH = ROOT / 'models/sscd/sscd_disc_mixup.torchscript.pt'
SD21_REPO_ID = 'Manojb/stable-diffusion-2-1-base'

WAVES_EDITS = [
    ('clean', 0.0),
    ('compression', 0.8),
    ('contrast', 0.8),
    ('brightness', 0.8),
    ('blurring', 0.8),
    ('noise', 0.8),
    ('resizedcrop', 0.8),
]


def read_json(p: Path):
    return json.loads(p.read_bytes())


def write_json(p: Path, obj):
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


def check_idle_gpu(idx: int, allowed: list[int]) -> str:
    assert idx != 6, 'GPU 6 is strictly excluded'
    assert idx in allowed, f'GPU {idx} not in allowed list {allowed}'
    info = subprocess.check_output(
        ['nvidia-smi', '-i', str(idx), '--query-gpu=uuid,memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'],
        text=True,
    ).strip().split(', ')
    busy = subprocess.check_output(
        ['nvidia-smi', '-i', str(idx), '--query-compute-apps=pid', '--format=csv,noheader'],
        text=True,
    ).strip()
    assert int(info[1]) < 100 and int(info[2]) == 0 and not busy, f'GPU {idx} is occupied: {info}, pids={busy}'
    return info[0]


def make_gs_watermark(auth_bits: np.ndarray, payload_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Pack (auth_bits || payload_bits) into exact 256-bit GS watermark (1,4,8,8)."""
    p_bits = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8)).astype(np.uint8)
    k = len(p_bits)
    assert k in (48, 64)
    a = auth_bits[:256 - k].astype(np.uint8)
    full = np.concatenate([a, p_bits])
    assert len(full) == 256
    return full.reshape(1, 4, 8, 8), p_bits


def do_prepare(n_prompts: int, allowed_gpus: list[int]):
    RUN.mkdir(parents=True, exist_ok=True)
    s0_cal = read_json(STAGE0 / 'calibration.json')
    s0_sscd = read_json(STAGE0 / 'summary_sscd.json')
    thresholds = dict(
        rgb_mean_var_48b=s0_cal['thresholds_95']['rgb_mean_var_48b'],
        spatial_opponent_64b=s0_cal['thresholds_95']['spatial_opponent_64b'],
        sscd_64b=s0_sscd['thresholds_95']['sscd_64b'],
    )
    cb_run = ROOT / 'runs/color_budget_20261119'
    cb_sources = [s for s in read_json(cb_run / 'sources.json') if s['split'] == 'calibration']
    half = n_prompts // 2
    selected = [s for s in cb_sources if s['dataset'] == 'COCO'][:half] + [s for s in cb_sources if s['dataset'] == 'SDP'][:half]
    assert len(selected) == n_prompts

    rng_priv = np.random.default_rng(2026112299)
    gs_key_np = rng_priv.integers(0, 2, size=(1, 4, 64, 64), dtype=np.uint8)
    auth_bits_np = rng_priv.integers(0, 2, size=(256,), dtype=np.uint8)
    np.savez(RUN / 'private_material.npz', key=gs_key_np, auth_bits=auth_bits_np)
    os.chmod(RUN / 'private_material.npz', 0o600)

    protocol = dict(
        run_name=RUN.name,
        allowed_gpus=allowed_gpus,
        excluded_gpus=[6],
        n_prompts=n_prompts,
        sources=selected,
        source_split='color_budget_20261119 calibration split only (zero test/confirm leakage)',
        carrier='Native Gaussian_Shading(1, 8, 1e-6, 1) 256-bit (64x spatial repetition)',
        payload_allocation=dict(rgb48='208 auth + 48 payload', opp64='192 auth + 64 payload', sscd64='192 auth + 64 payload'),
        auth_thresholds=dict(auth208=148 / 208, auth192=136 / 192, full256=181 / 256),
        stage0_thresholds=thresholds,
        waves_edits=WAVES_EDITS,
        script_sha256=sha_file(Path(__file__)),
    )
    write_json(RUN / 'protocol.json', protocol)
    print(f'Prepared {RUN} with {n_prompts} prompts.', flush=True)


def do_worker(gpu: int, shard: int, num_shards: int, allowed_gpus: list[int]):
    proto = read_json(RUN / 'protocol.json')
    uuid = check_idle_gpu(gpu, allowed_gpus)

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
    sys.path.insert(0, str(STAGE0))
    sys.path.insert(0, str(LOADER))
    from pipe import pipe_utils
    import run_stage0 as s0
    from order_recovery_data import waves_module

    waves = waves_module()
    thresholds = proto['stage0_thresholds']
    projs = np.load(STAGE0 / 'sscd_projections.npz')
    P64_np = projs['p64'].astype(np.float64)
    P64_t = torch.from_numpy(P64_np).float().cuda()

    priv = np.load(RUN / 'private_material.npz')
    gs_key_np = priv['key']
    auth_bits_np = priv['auth_bits']

    native_gs = load_module('native_gs', GS_MOD_PATH).Gaussian_Shading(1, 8, 1e-6, 1)
    native_gs.key = torch.from_numpy(gs_key_np.astype(np.int64)).cuda()

    provider = pipe_utils.create_diffusion_pipe(
        pretrained_model_name_or_path=SD21_REPO_ID,
        resolution=512,
        device='cuda:0',
        scheduler_name='DDIM',
        scheduler_inverse_name='DDIM',
        dtype=torch.float16,
        eager_loading=True,
        disable_tqdm=True,
    )
    pipe = provider.pipe
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.requires_grad_(False)
    pipe.vae.eval()
    pipe.unet.requires_grad_(False)
    pipe.unet.eval()

    sscd = torch.jit.load(str(SSCD_PATH), map_location='cuda').eval()
    for p in sscd.parameters():
        p.requires_grad_(False)

    lpips_fn = lpips.LPIPS(net='alex').cuda().eval()
    clip_asset = read_json(ROOT / 'runs/phase1_backend_audit_20261113_v2/complete.json')['model_files']['clip_vitl14']
    clip_model = CLIPModel.from_pretrained(clip_asset['root'], local_files_only=True).cuda().eval()
    clip_proc = CLIPProcessor.from_pretrained(clip_asset['root'], local_files_only=True)

    mean_norm = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    std_norm = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)

    def decode_to_x01(z: torch.Tensor) -> torch.Tensor:
        dec = pipe.vae.decode((z.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor), return_dict=False)[0]
        return (dec.float() / 2.0 + 0.5).clamp(0.0, 1.0)

    def x01_to_pil(x: torch.Tensor) -> Image.Image:
        arr = x.detach()[0].permute(1, 2, 0).mul(255.0).round().byte().cpu().numpy()
        return Image.fromarray(arr)

    def diff_rgb_stats(x: torch.Tensor) -> torch.Tensor:
        v, mu = torch.var_mean(x, dim=(0, 2, 3), unbiased=False)
        return torch.cat([mu, v])

    def diff_opp64_loss(x: torch.Tensor, target_dec: dict) -> torch.Tensor:
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        o1 = r - g
        o2 = 0.5 * (r + g) - b
        y4 = F.adaptive_avg_pool2d(y, (4, 4)).reshape(-1)
        o1_4 = F.adaptive_avg_pool2d(o1, (4, 4)).reshape(-1)
        o2_4 = F.adaptive_avg_pool2d(o2, (4, 4)).reshape(-1)
        y_rel = (y4 - y4.mean()) / (y4.std(unbiased=False) + 0.02)
        o1_rel = (o1_4 - o1_4.mean()) / (o1_4.std(unbiased=False) + 0.01)
        o2_rel = (o2_4 - o2_4.mean()) / (o2_4.std(unbiased=False) + 0.01)

        ty = torch.tensor(target_dec['y_rel'], device='cuda', dtype=torch.float32)
        to1 = torch.tensor(target_dec['o1_sgn'], device='cuda', dtype=torch.float32)
        to2 = torch.tensor(target_dec['o2_sgn'], device='cuda', dtype=torch.float32)
        ly = 1.0 - F.cosine_similarity(y_rel.unsqueeze(0), ty.unsqueeze(0)).squeeze()
        lo1 = 1.0 - F.cosine_similarity(o1_rel.unsqueeze(0), to1.unsqueeze(0)).squeeze()
        lo2 = 1.0 - F.cosine_similarity(o2_rel.unsqueeze(0), to2.unsqueeze(0)).squeeze()
        return 0.6 * ly + 0.2 * lo1 + 0.2 * lo2

    def diff_sscd64_loss(x: torch.Tensor, target_sgn: np.ndarray) -> torch.Tensor:
        x_res = F.interpolate(x, size=(320, 320), mode='bilinear', align_corners=False)
        x_in = (x_res - mean_norm) / std_norm
        emb = F.normalize(sscd(x_in), dim=1)[0]
        proj = emb @ P64_t
        ts = torch.tensor(target_sgn, device='cuda', dtype=torch.float32)
        cos_d = 1.0 - F.cosine_similarity(proj.unsqueeze(0), ts.unsqueeze(0)).squeeze()
        margin_pen = F.softplus(0.05 - proj * ts, beta=20.0).mean()
        return cos_d + 0.5 * margin_pen

    def terminal_gn_rgb(z0: torch.Tensor, target_mu_var: np.ndarray, steps: int = 8, trust: float = 0.35):
        pars = torch.zeros(68, device='cuda', requires_grad=True)
        target = torch.tensor(target_mu_var, device='cuda', dtype=torch.float32)
        weights = torch.tensor([50.0] * 3 + [100.0] * 3, device='cuda', dtype=torch.float32)
        best = None
        for i in range(steps + 1):
            bias = F.interpolate(pars[4:].reshape(1, 4, 4, 4), size=z0.shape[-2:], mode='bilinear', align_corners=False)
            adj = z0.float() * (1.0 + pars[:4].view(1, 4, 1, 1)) + bias
            x = decode_to_x01(adj)
            stats = diff_rgb_stats(x)
            r = (stats - target) * weights
            loss = r.square().sum()
            val = float(loss.detach())
            if best is None or val < best[0]:
                best = (val, x01_to_pil(x))
            err = (stats - target).detach().abs()
            if i == steps or (err[:3].max() < 0.009 and err[3:].max() < 0.0045):
                break
            jac = torch.stack([torch.autograd.grad(r[j], pars, retain_graph=j + 1 < 6)[0] for j in range(6)])
            delta = -jac.T @ torch.linalg.solve(jac @ jac.T + 0.05 * torch.eye(6, device='cuda'), r.detach())
            delta = delta * min(1.0, trust / max(1e-9, float(delta.abs().max())))
            with torch.no_grad():
                pars.add_(delta)
                pars.clamp_(-1.0, 1.0)
                pars[:4].clamp_(-0.5, 0.5)
        return best[1]

    def terminal_adam_z0(z0: torch.Tensor, loss_fn, steps: int = 20, lr: float = 0.03):
        pars = torch.zeros(68, device='cuda', requires_grad=True)
        opt = torch.optim.Adam([pars], lr=lr)
        best = None
        for i in range(steps + 1):
            bias = F.interpolate(pars[4:].reshape(1, 4, 4, 4), size=z0.shape[-2:], mode='bilinear', align_corners=False)
            adj = z0.float() * (1.0 + pars[:4].view(1, 4, 1, 1)) + bias
            x = decode_to_x01(adj)
            loss = loss_fn(x)
            val = float(loss.detach())
            if best is None or val < best[0]:
                best = (val, x01_to_pil(x))
            if i == steps or val < 0.08:
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                pars.clamp_(-1.0, 1.0)
                pars[:4].clamp_(-0.5, 0.5)
        return best[1]

    def ddim_generate(prompt: str, zT: torch.Tensor, guidance_fn=None, step_range=(10, 38), step_size=0.03, stop_loss=0.0):
        pipe.scheduler = provider.scheduler
        pipe.scheduler.set_timesteps(50, device='cuda')
        with torch.no_grad():
            pos_emb, neg_emb = pipe.encode_prompt(
                prompt=prompt, device='cuda', num_images_per_prompt=1,
                do_classifier_free_guidance=True, negative_prompt='',
            )
            text_emb = torch.cat([neg_emb, pos_emb], dim=0)
        z = zT.clone().to(device='cuda', dtype=torch.float16)
        alphas = pipe.scheduler.alphas_cumprod.to('cuda')
        timesteps = pipe.scheduler.timesteps
        for idx, t in enumerate(timesteps):
            with torch.no_grad():
                lat_in = torch.cat([z, z], dim=0)
                lat_in = pipe.scheduler.scale_model_input(lat_in, t)
                eps = pipe.unet(lat_in, t, encoder_hidden_states=text_emb, return_dict=False)[0]
                eps_u, eps_c = eps.chunk(2)
                eps_pred = eps_u + 7.5 * (eps_c - eps_u)
                step_out = pipe.scheduler.step(eps_pred, t, z, eta=0.0, return_dict=True)
                z_next = step_out.prev_sample
                pred_x0 = step_out.pred_original_sample

            if guidance_fn is not None and step_range[0] <= idx <= step_range[1]:
                z0_leaf = pred_x0.detach().float().requires_grad_(True)
                x_hat = decode_to_x01(z0_leaf)
                loss = guidance_fn(x_hat)
                if float(loss.detach()) > stop_loss:
                    grad = torch.autograd.grad(loss, z0_leaf)[0]
                    g_norm = float(grad.norm()) + 1e-8
                    prev_t = t - pipe.scheduler.config.num_train_timesteps // len(timesteps)
                    alpha_prev = alphas[prev_t] if prev_t >= 0 else pipe.scheduler.final_alpha_cumprod.to('cuda')
                    scale = step_size * float(z0_leaf.detach().norm())
                    delta_z0 = -(scale / g_norm) * grad
                    z_next = (z_next.float() + torch.sqrt(alpha_prev.float()) * delta_z0).to(torch.float16)
            z = z_next
        with torch.no_grad():
            img = x01_to_pil(decode_to_x01(z))
        return z, img

    def eval_sscd_pil(im: Image.Image) -> np.ndarray:
        x = torch.from_numpy(np.asarray(im.convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()
        x_res = F.interpolate(x, size=(320, 320), mode='bilinear', align_corners=False)
        x_in = (x_res - mean_norm) / std_norm
        with torch.no_grad():
            return F.normalize(sscd(x_in), dim=1)[0].cpu().numpy().astype(np.float64)

    def eval_lpips_clip(base_im: Image.Image, cand_im: Image.Image, prompt: str):
        def to_m11(im):
            return torch.from_numpy(np.asarray(im, dtype=np.float32).transpose(2, 0, 1)).unsqueeze(0).cuda() / 127.5 - 1.0
        with torch.no_grad():
            lp = float(lpips_fn(to_m11(base_im), to_m11(cand_im)).item())
            inp = clip_proc(text=[prompt], images=[cand_im], return_tensors='pt', padding=True, truncation=True, max_length=77).to('cuda')
            out = clip_model(**inp)
            cs = float((F.normalize(out.image_embeds, dim=-1) * F.normalize(out.text_embeds, dim=-1)).sum().item())
        return lp, cs

    def decode_gs_bits(zT_inv: torch.Tensor) -> np.ndarray:
        rev_m = (zT_inv > 0).int()
        rev_sd = (rev_m + native_gs.key) % 2
        rev_w = native_gs.diffusion_inverse(rev_sd)
        return rev_w.reshape(-1).cpu().numpy().astype(np.uint8)

    def invert_and_verify(im: Image.Image, full_w_bits: np.ndarray, family: str, true_payload_bytes: bytes, true_target_dec):
        z_inv = provider.invert_images(im, num_inference_steps=50)['zT_torch']
        rec_bits = decode_gs_bits(z_inv)
        k = len(true_payload_bytes) * 8
        n_auth = 256 - k
        full_ba = float((rec_bits == full_w_bits).mean())
        auth_ba = float((rec_bits[:n_auth] == full_w_bits[:n_auth]).mean())
        rec_p_bits = rec_bits[n_auth:]
        true_p_bits = full_w_bits[n_auth:]
        payload_ber = float((rec_p_bits != true_p_bits).mean())
        rec_p_bytes = np.packbits(rec_p_bits).tobytes()

        if family == 'rgb48':
            q_mv = s0.unpack_bits(rec_p_bytes, [8] * 6)
            rec_dec = dict(mu=s0.dequantize_u(q_mv[:3], 0.0, 1.0, 8), var=s0.dequantize_u(q_mv[3:], 0.0, 0.25, 8))
            qf = s0.extract_continuous(im)
            d_blind = max(float(np.max(np.abs(rec_dec['mu'] - qf['mu'])) / 0.02),
                          float(np.max(np.abs(rec_dec['var'] - qf['var'])) / 0.01))
            d_oracle = max(float(np.max(np.abs(true_target_dec['mu'] - qf['mu'])) / 0.02),
                           float(np.max(np.abs(true_target_dec['var'] - qf['var'])) / 0.01))
            th = thresholds['rgb_mean_var_48b']
            auth_ok = auth_ba > (148.0 / 208.0)
        elif family == 'opp64':
            q_op = s0.unpack_bits(rec_p_bytes, [2] * 16 + [1] * 32)
            rec_dec = dict(
                y_rel=s0.dequantize_u(q_op[:16], -2.0, 2.0, 2),
                o1_sgn=2.0 * q_op[16:32].astype(np.float64) - 1.0,
                o2_sgn=2.0 * q_op[32:48].astype(np.float64) - 1.0,
            )
            qf = s0.extract_continuous(im)
            d_blind = (0.6 * s0.cos_dist(rec_dec['y_rel'], qf['y_rel'])
                       + 0.2 * s0.cos_dist(rec_dec['o1_sgn'], qf['o1_rel'])
                       + 0.2 * s0.cos_dist(rec_dec['o2_sgn'], qf['o2_rel']))
            d_oracle = (0.6 * s0.cos_dist(true_target_dec['y_rel'], qf['y_rel'])
                        + 0.2 * s0.cos_dist(true_target_dec['o1_sgn'], qf['o1_rel'])
                        + 0.2 * s0.cos_dist(true_target_dec['o2_sgn'], qf['o2_rel']))
            th = thresholds['spatial_opponent_64b']
            auth_ok = auth_ba > (136.0 / 192.0)
        else:
            rec_sgn = 2.0 * rec_p_bits.astype(np.float64) - 1.0
            qe = eval_sscd_pil(im)
            qp = qe @ P64_np
            denom = np.sqrt(64.0) * float(np.linalg.norm(qp)) + 1e-12
            d_blind = float(1.0 - np.dot(rec_sgn, qp) / denom)
            d_oracle = float(1.0 - np.dot(true_target_dec, qp) / denom)
            th = thresholds['sscd_64b']
            auth_ok = auth_ba > (136.0 / 192.0)

        return dict(
            full_ba=full_ba,
            auth_ba=auth_ba,
            auth_ok=bool(auth_ok),
            payload_ber=payload_ber,
            d_blind=d_blind,
            d_oracle=d_oracle,
            feat_blind_ok=bool(d_blind <= th),
            feat_strict_ok=bool(d_oracle <= 1.0) if family == 'rgb48' else bool(d_oracle <= th * 0.5),
            joint_blind_ok=bool(auth_ok and d_blind <= th),
        )

    with torch.no_grad():
        pipe('warmup', generator=torch.Generator(device='cuda').manual_seed(7), num_inference_steps=2)

    cb_run = ROOT / 'runs/color_budget_20261119'
    my_sources = [s for i, s in enumerate(proto['sources']) if i % num_shards == shard]
    t_start = time.perf_counter()
    records = []
    img_root = RUN / 'images'
    img_root.mkdir(parents=True, exist_ok=True)

    for idx, s in enumerate(my_sources):
        task = s['task']
        prompt = s['prompt']
        donor_p = cb_run / 'images' / f'{task:03d}' / '0.png'
        donor_im = Image.open(donor_p).convert('RGB')
        donor_feat = s0.extract_continuous(donor_im)
        donor_enc = s0.encode_payloads(donor_feat)
        donor_dec = s0.decode_payloads(donor_enc)
        donor_sscd_emb = eval_sscd_pil(donor_im)
        donor_sscd_bits = ((donor_sscd_emb @ P64_np) >= 0.0).astype(np.uint8)
        donor_sscd_bytes = np.packbits(donor_sscd_bits).tobytes()
        donor_sscd_sgn = 2.0 * donor_sscd_bits.astype(np.float64) - 1.0

        tdir = img_root / f'{task:03d}'
        tdir.mkdir(exist_ok=True)

        family_specs = [
            ('rgb48', donor_enc['rgb_mean_var_48b'], donor_dec['rgb_mean_var_48b']),
            ('opp64', donor_enc['spatial_opponent_64b'], donor_dec['spatial_opponent_64b']),
            ('sscd64', donor_sscd_bytes, donor_sscd_sgn),
        ]

        for fam, p_bytes, t_dec in family_specs:
            w_arr, _ = make_gs_watermark(auth_bits_np, p_bytes)
            w_flat = w_arr.reshape(-1)
            sd = np.tile(w_arr, (1, 1, 8, 8))
            m_enc = ((sd + gs_key_np) % 2).reshape(-1)
            np.random.seed(2026112200 + task * 10 + {'rgb48': 1, 'opp64': 2, 'sscd64': 3}[fam])
            zT = native_gs.truncSampling(m_enc)

            t0 = time.perf_counter()
            z0_base, base_im = ddim_generate(prompt, zT, guidance_fn=None)
            torch.cuda.synchronize()
            sec_base = time.perf_counter() - t0
            base_im.save(tdir / f'{fam}_unguided.png')

            arms = [(f'{fam}_unguided', base_im, sec_base)]

            if fam == 'rgb48':
                target_mv = np.concatenate([t_dec['mu'], t_dec['var']])
                t0 = time.perf_counter()
                gn_im = terminal_gn_rgb(z0_base, target_mv, steps=8, trust=0.35)
                torch.cuda.synchronize()
                sec_gn = sec_base + (time.perf_counter() - t0)
                gn_im.save(tdir / 'rgb48_terminal_gn.png')
                arms.append(('rgb48_terminal_gn', gn_im, sec_gn))

                tgt_t = torch.tensor(target_mv, device='cuda', dtype=torch.float32)
                wts_t = torch.tensor([1.0] * 3 + [2.0] * 3, device='cuda', dtype=torch.float32)

                def rgb_loss(x):
                    return (((diff_rgb_stats(x) - tgt_t) * wts_t).square().sum())

                t0 = time.perf_counter()
                _, tw_im = ddim_generate(prompt, zT, guidance_fn=rgb_loss, step_range=(10, 38), step_size=0.025, stop_loss=0.0004)
                torch.cuda.synchronize()
                sec_tw = time.perf_counter() - t0
                tw_im.save(tdir / 'rgb48_mid_tweedie.png')
                arms.append(('rgb48_mid_tweedie', tw_im, sec_tw))

            elif fam == 'opp64':
                t0 = time.perf_counter()
                _, tw_im = ddim_generate(prompt, zT, guidance_fn=lambda x: diff_opp64_loss(x, t_dec),
                                         step_range=(5, 35), step_size=0.035, stop_loss=0.06)
                torch.cuda.synchronize()
                sec_tw = time.perf_counter() - t0
                tw_im.save(tdir / 'opp64_mid_tweedie.png')
                arms.append(('opp64_mid_tweedie', tw_im, sec_tw))

                t0 = time.perf_counter()
                term_im = terminal_adam_z0(z0_base, lambda x: diff_opp64_loss(x, t_dec), steps=20, lr=0.04)
                torch.cuda.synchronize()
                sec_term = sec_base + (time.perf_counter() - t0)
                term_im.save(tdir / 'opp64_terminal_z0.png')
                arms.append(('opp64_terminal_z0', term_im, sec_term))

            elif fam == 'sscd64':
                t0 = time.perf_counter()
                _, tw_im = ddim_generate(prompt, zT, guidance_fn=lambda x: diff_sscd64_loss(x, t_dec),
                                         step_range=(5, 35), step_size=0.035, stop_loss=0.10)
                torch.cuda.synchronize()
                sec_tw = time.perf_counter() - t0
                tw_im.save(tdir / 'sscd64_mid_tweedie.png')
                arms.append(('sscd64_mid_tweedie', tw_im, sec_tw))

                t0 = time.perf_counter()
                term_im = terminal_adam_z0(z0_base, lambda x: diff_sscd64_loss(x, t_dec), steps=20, lr=0.03)
                torch.cuda.synchronize()
                sec_term = sec_base + (time.perf_counter() - t0)
                term_im.save(tdir / 'sscd64_terminal_z0.png')
                arms.append(('sscd64_terminal_z0', term_im, sec_term))

            for arm_name, arm_im, arm_sec in arms:
                lp, cs = eval_lpips_clip(base_im, arm_im, prompt)
                for ei, (efam, estr) in enumerate(WAVES_EDITS):
                    eseed = 20261122 + task * 100 + ei
                    q_im = arm_im if efam == 'clean' else waves.apply_distortion([arm_im], efam, strength=estr, distortion_seed=eseed)[0]
                    v = invert_and_verify(q_im, w_flat, fam, p_bytes, t_dec)
                    records.append(dict(
                        task=task, dataset=s['dataset'], family=fam, arm=arm_name,
                        edit=efam, strength=estr, seconds=arm_sec, lpips=lp, clip=cs, **v,
                    ))

        print(f'[shard {shard} | gpu {gpu}] {idx+1}/{len(my_sources)} task={task:03d} done ({time.perf_counter()-t_start:.1f}s)', flush=True)
        write_json(RUN / f'records_shard{shard}.json', records)

    elapsed = time.perf_counter() - t_start
    write_json(RUN / f'complete_shard{shard}.json', dict(
        status='complete', shard=shard, gpu=gpu, uuid=uuid, elapsed_seconds=elapsed, n_records=len(records),
    ))
    print(f'[shard {shard} | gpu {gpu}] COMPLETE in {elapsed:.1f}s', flush=True)


def do_merge(num_shards: int):
    proto = read_json(RUN / 'protocol.json')
    records = []
    shard_meta = []
    for s in range(num_shards):
        shard_meta.append(read_json(RUN / f'complete_shard{s}.json'))
        records.extend(read_json(RUN / f'records_shard{s}.json'))
    expected = proto['n_prompts'] * 9 * len(WAVES_EDITS)
    assert len(records) == expected, f'Expected {expected} records, got {len(records)}'
    write_json(RUN / 'records.json', records)

    arms_order = sorted({r['arm'] for r in records})
    summary = {}
    for arm in arms_order:
        sub = [r for r in records if r['arm'] == arm]
        clean = [r for r in sub if r['edit'] == 'clean']
        edited = [r for r in sub if r['edit'] != 'clean']
        by_edit = {}
        for efam, _ in WAVES_EDITS:
            esub = [r for r in sub if r['edit'] == efam]
            by_edit[efam] = dict(
                n=len(esub),
                full_ba_mean=float(np.mean([r['full_ba'] for r in esub])),
                auth_ba_mean=float(np.mean([r['auth_ba'] for r in esub])),
                auth_pass=sum(r['auth_ok'] for r in esub),
                payload_ber_mean=float(np.mean([r['payload_ber'] for r in esub])),
                d_blind_mean=float(np.mean([r['d_blind'] for r in esub])),
                d_oracle_mean=float(np.mean([r['d_oracle'] for r in esub])),
                feat_blind_pass=sum(r['feat_blind_ok'] for r in esub),
                feat_strict_pass=sum(r['feat_strict_ok'] for r in esub),
                joint_blind_pass=sum(r['joint_blind_ok'] for r in esub),
            )
        summary[arm] = dict(
            n_prompts=len(clean),
            seconds_mean=float(np.mean([r['seconds'] for r in clean])),
            lpips_mean=float(np.mean([r['lpips'] for r in clean])),
            clip_mean=float(np.mean([r['clip'] for r in clean])),
            clean_full_ba=float(np.mean([r['full_ba'] for r in clean])),
            clean_payload_ber=float(np.mean([r['payload_ber'] for r in clean])),
            clean_d_oracle=float(np.mean([r['d_oracle'] for r in clean])),
            clean_joint_pass=sum(r['joint_blind_ok'] for r in clean),
            clean_strict_pass=sum(r['feat_strict_ok'] for r in clean),
            edited_n=len(edited),
            edited_auth_pass=sum(r['auth_ok'] for r in edited),
            edited_payload_ber=float(np.mean([r['payload_ber'] for r in edited])),
            edited_feat_pass=sum(r['feat_blind_ok'] for r in edited),
            edited_joint_pass=sum(r['joint_blind_ok'] for r in edited),
            by_edit=by_edit,
        )

    write_json(RUN / 'summary.json', dict(shards=shard_meta, arms=summary))
    print(f'MERGED {len(records)} records across {num_shards} shards.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'worker', 'merge'])
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=2)
    parser.add_argument('--allowed-gpus', type=str, default='2,3,4,5,7')
    parser.add_argument('--n-prompts', type=int, default=16)
    args = parser.parse_args()
    allowed = [int(x) for x in args.allowed_gpus.split(',') if x.strip()]
    if args.mode == 'prepare':
        do_prepare(args.n_prompts, allowed)
    elif args.mode == 'worker':
        do_worker(args.gpu, args.shard, args.num_shards, allowed)
    else:
        do_merge(args.num_shards)
