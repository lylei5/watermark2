"""Stage 0: Compact (<=64-bit) differentiable payload-binding descriptors evaluated on
WAVES 16-edit holdout/confirmation sets, published Imprint & Reprompt+ forgery attacks,
external color-transfer/object-replacement benchmarks, and Reinhard invariance diagnostics.
CPU-only on server 249.
"""
import csv
import gzip
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path('/data0/liyanlei/watermark_research')
RUN = ROOT / 'runs/payload_binding_stage0_20261122'
LOADER = ROOT / 'runs/order_capsule_confirmation_20261115'

METHODS = [
    'rgb_mean_24b',
    'rgb_mean_var_48b',
    'chroma_ratio_48b',
    'spatial_luma_48b',
    'spatial_luma_chroma_64b',
    'spatial_opponent_64b',
]

BIT_BUDGETS = {
    'rgb_mean_24b': 24,
    'rgb_mean_var_48b': 48,
    'chroma_ratio_48b': 48,
    'spatial_luma_48b': 48,
    'spatial_luma_chroma_64b': 64,
    'spatial_opponent_64b': 64,
}


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    return sha_bytes(p.read_bytes())


def read_json(p: Path):
    return json.loads(p.read_bytes())


def write_json(p: Path, obj):
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(p)


def img_to_float(im: Image.Image) -> np.ndarray:
    arr = np.asarray(im.convert('RGB'), dtype=np.float32) / 255.0
    return arr  # (H, W, 3)


def pool_4x4(ch: np.ndarray) -> np.ndarray:
    """Exact differentiable-equivalent 4x4 block average (works for any H, W)."""
    h, w = ch.shape
    ys = np.linspace(0, h, 5).astype(int)
    xs = np.linspace(0, w, 5).astype(int)
    out = np.empty((4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            out[i, j] = float(ch[ys[i]:ys[i+1], xs[j]:xs[j+1]].mean())
    return out.ravel()


def quantize_u(val: np.ndarray, lo: float, hi: float, bits: int) -> np.ndarray:
    levels = (1 << bits) - 1
    scaled = np.clip((val - lo) / (hi - lo), 0.0, 1.0)
    return np.rint(scaled * levels).astype(np.uint16)


def dequantize_u(q: np.ndarray, lo: float, hi: float, bits: int) -> np.ndarray:
    levels = (1 << bits) - 1
    return lo + (q.astype(np.float64) / levels) * (hi - lo)


def pack_bits(vals, bit_widths) -> bytes:
    bits = []
    for v, bw in zip(vals, bit_widths):
        iv = int(v)
        assert 0 <= iv < (1 << bw)
        for b in range(bw - 1, -1, -1):
            bits.append((iv >> b) & 1)
    assert len(bits) <= 64
    arr = np.packbits(np.array(bits, dtype=np.uint8))
    return arr.tobytes()


def unpack_bits(buf: bytes, bit_widths) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8))
    out = []
    idx = 0
    for bw in bit_widths:
        v = 0
        for _ in range(bw):
            v = (v << 1) | int(bits[idx])
            idx += 1
        out.append(v)
    return np.array(out, dtype=np.uint16)


def extract_continuous(im: Image.Image) -> dict:
    arr = img_to_float(im)  # (H, W, 3) in [0,1]
    r_ch, g_ch, b_ch = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # 1. Global RGB mean & pop variance
    mu = arr.reshape(-1, 3).mean(axis=0).astype(np.float64)
    var = arr.reshape(-1, 3).var(axis=0).astype(np.float64)

    # 2. Global normalized chromaticity stats
    s = r_ch + g_ch + b_ch + 1e-3
    rn = (r_ch / s).astype(np.float64)
    gn = (g_ch / s).astype(np.float64)
    mu_r, mu_g = float(rn.mean()), float(gn.mean())
    std_r, std_g = float(rn.std()), float(gn.std())
    sat = np.sqrt((rn - 1.0 / 3.0) ** 2 + (gn - 1.0 / 3.0) ** 2)
    mu_sat = float(sat.mean())
    std_rg = float((rn - gn).std())
    mean_sum = float(mu.sum() + 1e-3)
    ratio_r, ratio_g = float(mu[0] / mean_sum), float(mu[1] / mean_sum)
    chroma6 = np.array([mu_r, mu_g, std_r, std_g, mu_sat, std_rg], dtype=np.float64)
    chroma3 = np.array([mu_r, mu_g, mu_sat], dtype=np.float64)

    # 3. 4x4 Spatial Relative Luminance (affine invariant)
    y = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    y4 = pool_4x4(y)
    y_std = float(y4.std())
    y_rel = (y4 - float(y4.mean())) / (y_std + 0.02)

    # 4. 4x4 Spatial Opponent Color
    o1 = pool_4x4(r_ch - g_ch)
    o2 = pool_4x4(0.5 * (r_ch + g_ch) - b_ch)
    o1_rel = (o1 - float(o1.mean())) / (float(o1.std()) + 0.01)
    o2_rel = (o2 - float(o2.mean())) / (float(o2.std()) + 0.01)

    return dict(
        mu=mu,
        var=var,
        chroma6=chroma6,
        chroma3=chroma3,
        y_rel=y_rel,
        o1_rel=o1_rel,
        o2_rel=o2_rel,
    )


CHROMA6_BOUNDS = [
    (0.15, 0.55),
    (0.15, 0.55),
    (0.0, 0.20),
    (0.0, 0.20),
    (0.0, 0.45),
    (0.0, 0.30),
]


def encode_payloads(feat: dict) -> dict:
    """Strictly pack each candidate into <=64-bit binary payload bytes."""
    p = {}
    # rgb_mean_24b: 3x8 = 24 bits (3 bytes)
    q_mu = quantize_u(feat['mu'], 0.0, 1.0, 8)
    p['rgb_mean_24b'] = pack_bits(q_mu, [8, 8, 8])
    assert len(p['rgb_mean_24b']) == 3

    # rgb_mean_var_48b: 3x8 + 3x8 = 48 bits (6 bytes)
    q_var = quantize_u(feat['var'], 0.0, 0.25, 8)
    p['rgb_mean_var_48b'] = pack_bits(list(q_mu) + list(q_var), [8] * 6)
    assert len(p['rgb_mean_var_48b']) == 6

    # chroma_ratio_48b: 6x8 = 48 bits (6 bytes)
    q_c6 = [int(quantize_u(np.array([v]), lo, hi, 8)[0]) for v, (lo, hi) in zip(feat['chroma6'], CHROMA6_BOUNDS)]
    p['chroma_ratio_48b'] = pack_bits(q_c6, [8] * 6)
    assert len(p['chroma_ratio_48b']) == 6

    # spatial_luma_48b: 16x3 = 48 bits (6 bytes)
    q_y3 = quantize_u(feat['y_rel'], -2.5, 2.5, 3)
    p['spatial_luma_48b'] = pack_bits(q_y3, [3] * 16)
    assert len(p['spatial_luma_48b']) == 6

    # spatial_luma_chroma_64b: 16x3 (48b) + 6b + 6b + 4b (16b) = 64 bits (8 bytes)
    q_cr = int(quantize_u(np.array([feat['chroma3'][0]]), 0.15, 0.55, 6)[0])
    q_cg = int(quantize_u(np.array([feat['chroma3'][1]]), 0.15, 0.55, 6)[0])
    q_cs = int(quantize_u(np.array([feat['chroma3'][2]]), 0.0, 0.45, 4)[0])
    p['spatial_luma_chroma_64b'] = pack_bits(list(q_y3) + [q_cr, q_cg, q_cs], [3] * 16 + [6, 6, 4])
    assert len(p['spatial_luma_chroma_64b']) == 8

    # spatial_opponent_64b: 16x2 (32b) + 16x1 (16b) + 16x1 (16b) = 64 bits (8 bytes)
    q_y2 = quantize_u(feat['y_rel'], -2.0, 2.0, 2)
    q_o1 = (feat['o1_rel'] >= 0.0).astype(np.uint16)
    q_o2 = (feat['o2_rel'] >= 0.0).astype(np.uint16)
    p['spatial_opponent_64b'] = pack_bits(list(q_y2) + list(q_o1) + list(q_o2), [2] * 16 + [1] * 32)
    assert len(p['spatial_opponent_64b']) == 8
    return p


def decode_payloads(payloads: dict) -> dict:
    d = {}
    q_mu = unpack_bits(payloads['rgb_mean_24b'], [8, 8, 8])
    d['rgb_mean_24b'] = dict(mu=dequantize_u(q_mu, 0.0, 1.0, 8))

    q_mv = unpack_bits(payloads['rgb_mean_var_48b'], [8] * 6)
    d['rgb_mean_var_48b'] = dict(
        mu=dequantize_u(q_mv[:3], 0.0, 1.0, 8),
        var=dequantize_u(q_mv[3:], 0.0, 0.25, 8),
    )

    q_c6 = unpack_bits(payloads['chroma_ratio_48b'], [8] * 6)
    c6 = np.array([dequantize_u(np.array([v]), lo, hi, 8)[0] for v, (lo, hi) in zip(q_c6, CHROMA6_BOUNDS)])
    d['chroma_ratio_48b'] = dict(chroma6=c6)

    q_y3 = unpack_bits(payloads['spatial_luma_48b'], [3] * 16)
    d['spatial_luma_48b'] = dict(y_rel=dequantize_u(q_y3, -2.5, 2.5, 3))

    q_lc = unpack_bits(payloads['spatial_luma_chroma_64b'], [3] * 16 + [6, 6, 4])
    d['spatial_luma_chroma_64b'] = dict(
        y_rel=dequantize_u(q_lc[:16], -2.5, 2.5, 3),
        chroma3=np.array([
            dequantize_u(np.array([q_lc[16]]), 0.15, 0.55, 6)[0],
            dequantize_u(np.array([q_lc[17]]), 0.15, 0.55, 6)[0],
            dequantize_u(np.array([q_lc[18]]), 0.0, 0.45, 4)[0],
        ]),
    )

    q_op = unpack_bits(payloads['spatial_opponent_64b'], [2] * 16 + [1] * 32)
    d['spatial_opponent_64b'] = dict(
        y_rel=dequantize_u(q_op[:16], -2.0, 2.0, 2),
        o1_sgn=2.0 * q_op[16:32].astype(np.float64) - 1.0,
        o2_sgn=2.0 * q_op[32:48].astype(np.float64) - 1.0,
    )
    return d


def cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 and nb < 1e-8:
        return 0.0
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    return float(1.0 - np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def raw_distances(ref_dec: dict, q_feat: dict, chroma6_scale: np.ndarray, chroma3_scale: np.ndarray) -> dict:
    out = {}
    out['rgb_mean_24b'] = float(np.max(np.abs(ref_dec['rgb_mean_24b']['mu'] - q_feat['mu'])) / 0.02)

    dm = float(np.max(np.abs(ref_dec['rgb_mean_var_48b']['mu'] - q_feat['mu'])) / 0.02)
    dv = float(np.max(np.abs(ref_dec['rgb_mean_var_48b']['var'] - q_feat['var'])) / 0.01)
    out['rgb_mean_var_48b'] = max(dm, dv)

    out['chroma_ratio_48b'] = float(np.linalg.norm((ref_dec['chroma_ratio_48b']['chroma6'] - q_feat['chroma6']) / chroma6_scale))

    out['spatial_luma_48b'] = cos_dist(ref_dec['spatial_luma_48b']['y_rel'], q_feat['y_rel'])

    d_lay = cos_dist(ref_dec['spatial_luma_chroma_64b']['y_rel'], q_feat['y_rel'])
    d_chr = float(np.linalg.norm((ref_dec['spatial_luma_chroma_64b']['chroma3'] - q_feat['chroma3']) / chroma3_scale))
    out['spatial_luma_chroma_64b_layout'] = d_lay
    out['spatial_luma_chroma_64b_chroma'] = d_chr

    op = ref_dec['spatial_opponent_64b']
    out['spatial_opponent_64b'] = (
        0.6 * cos_dist(op['y_rel'], q_feat['y_rel'])
        + 0.2 * cos_dist(op['o1_sgn'], q_feat['o1_rel'])
        + 0.2 * cos_dist(op['o2_sgn'], q_feat['o2_rel'])
    )
    return out


def reinhard_align(src_im: Image.Image, ref_im: Image.Image) -> Image.Image:
    """Diagnostic-only: channel-wise RGB mean/std linear alignment to ref_im."""
    src = img_to_float(src_im).astype(np.float64)
    ref = img_to_float(ref_im).astype(np.float64)
    out = np.empty_like(src)
    for c in range(3):
        sm, ss = src[:, :, c].mean(), src[:, :, c].std()
        rm, rs = ref[:, :, c].mean(), ref[:, :, c].std()
        out[:, :, c] = (src[:, :, c] - sm) / (ss + 1e-6) * rs + rm
    u8 = np.rint(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(u8)


def _eval_pair_worker(args):
    """Evaluate one prompt pair (2 seeds x 16 WAVES edits + clean) on CPU."""
    run_dir, s, families, strengths, seed_base = args
    sys.path.insert(0, str(LOADER))
    from order_recovery_data import waves_module
    waves = waves_module()

    dest = Path(run_dir) / 'images' / f"{s['task']:03d}"
    receipt = read_json(dest / 'receipt.json')
    imgs, feats, decs = [], [], []
    for i, expected in enumerate(receipt['sha256']):
        p = dest / f'{i}.png'
        assert sha_file(p) == expected
        with Image.open(p) as fh:
            im = fh.convert('RGB')
        assert im.size == (512, 512)
        f = extract_continuous(im)
        enc = encode_payloads(f)
        dec = decode_payloads(enc)
        imgs.append(im)
        feats.append(f)
        decs.append(dec)

    records = []
    for seed_idx in (0, 1):
        im = imgs[seed_idx]
        for fi, family in enumerate(['clean'] + families):
            for ji, strength in enumerate(strengths if fi else [0.0]):
                seed = seed_base + 100 * s['task'] + 2 * fi + ji
                q_im = waves.apply_distortion([im], family, strength=strength, distortion_seed=seed)[0] if fi else im
                q_feat = feats[seed_idx] if not fi else extract_continuous(q_im)
                records.append(dict(
                    dataset=s['dataset'],
                    task=s['task'],
                    seed_idx=seed_idx,
                    family=family,
                    strength=strength,
                    edit_seed=seed,
                    own_dec=decs[seed_idx],
                    sib_dec=decs[1 - seed_idx],
                    q_feat=q_feat,
                ))
    return dict(task=s['task'], dataset=s['dataset'], ref_feats=feats, records=records)


def score_records(records, chroma6_scale, chroma3_scale, null_layout=None, null_chroma=None):
    rows = []
    for rec in records:
        own_d = raw_distances(rec['own_dec'], rec['q_feat'], chroma6_scale, chroma3_scale)
        sib_d = raw_distances(rec['sib_dec'], rec['q_feat'], chroma6_scale, chroma3_scale)
        if null_layout is not None:
            for d in (own_d, sib_d):
                p_l = (1 + int(np.searchsorted(null_layout, d['spatial_luma_chroma_64b_layout'], side='right'))) / (len(null_layout) + 1)
                p_c = (1 + int(np.searchsorted(null_chroma, d['spatial_luma_chroma_64b_chroma'], side='right'))) / (len(null_chroma) + 1)
                d['spatial_luma_chroma_64b'] = p_l * p_c
        rows.append(dict(
            dataset=rec['dataset'],
            task=rec['task'],
            seed_idx=rec['seed_idx'],
            family=rec['family'],
            strength=rec['strength'],
            own=own_d,
            sib=sib_d,
        ))
    return rows


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES', '') == ''
    t0 = time.perf_counter()

    cb_run = ROOT / 'runs/color_budget_20261119'
    cj_run = ROOT / 'runs/color_joint_20261119'
    cb_plan = read_json(cb_run / 'color_budget_20261119.json')
    cb_sources = read_json(cb_run / 'sources.json')
    cj_sources = read_json(cj_run / 'sources.json')
    families = cb_plan['families']
    strengths = cb_plan['strengths']

    protocol = dict(
        run_name='payload_binding_stage0_20261122',
        scope='CPU oracle screen of <=64-bit differentiable payload-embeddable descriptors across WAVES 16-edit holdout/confirmation, published Imprint & Reprompt+ attacks, external color-transfer/object-replacement benchmarks, and Reinhard invariance diagnostics.',
        methods=METHODS,
        bit_budgets=BIT_BUDGETS,
        calibration_source='color_budget_20261119 calibration split (40 prompts, 1280 edited queries)',
        test_source='color_budget_20261119 test split (60 prompts, 1920 edited queries)',
        confirm_source='color_joint_20261119 confirmation split (60 prompts, 1920 edited queries)',
        target_legitimate_tpr=0.95,
        families=families,
        strengths=strengths,
    )
    write_json(RUN / 'protocol.json', protocol)

    cal_sources = [s for s in cb_sources if s['split'] == 'calibration']
    test_sources = [s for s in cb_sources if s['split'] == 'test']
    assert len(cal_sources) == 40 and len(test_sources) == 60 and len(cj_sources) == 60

    print(f'Extracting features across 40 cal + 60 test + 60 confirm prompts on CPU (12 workers)...', flush=True)
    with ProcessPoolExecutor(max_workers=12) as ex:
        cal_out = list(ex.map(_eval_pair_worker, [(str(cb_run), s, families, strengths, 20261119) for s in cal_sources]))
        test_out = list(ex.map(_eval_pair_worker, [(str(cb_run), s, families, strengths, 20261119) for s in test_sources]))
        conf_out = list(ex.map(_eval_pair_worker, [(str(cj_run), s, families, strengths, 20261119) for s in cj_sources]))

    # Compute frozen calibration scales from the 80 clean calibration references ONLY
    cal_c6 = np.stack([f['chroma6'] for item in cal_out for f in item['ref_feats']])
    cal_c3 = np.stack([f['chroma3'] for item in cal_out for f in item['ref_feats']])
    chroma6_scale = np.maximum(cal_c6.std(axis=0), 1e-4)
    chroma3_scale = np.maximum(cal_c3.std(axis=0), 1e-4)

    cal_recs = [r for item in cal_out for r in item['records']]
    test_recs = [r for item in test_out for r in item['records']]
    conf_recs = [r for item in conf_out for r in item['records']]

    # First pass on calibration to build null empirical CDFs for spatial_luma_chroma_64b
    cal_raw = score_records(cal_recs, chroma6_scale, chroma3_scale)
    cal_edited_raw = [r for r in cal_raw if r['family'] != 'clean']
    assert len(cal_edited_raw) == 1280
    null_layout = np.sort([r['sib']['spatial_luma_chroma_64b_layout'] for r in cal_edited_raw])
    null_chroma = np.sort([r['sib']['spatial_luma_chroma_64b_chroma'] for r in cal_edited_raw])

    cal_rows = score_records(cal_recs, chroma6_scale, chroma3_scale, null_layout, null_chroma)
    test_rows = score_records(test_recs, chroma6_scale, chroma3_scale, null_layout, null_chroma)
    conf_rows = score_records(conf_recs, chroma6_scale, chroma3_scale, null_layout, null_chroma)

    # Freeze 95% legitimate-pass threshold on 1280 edited calibration queries
    cal_edited = [r for r in cal_rows if r['family'] != 'clean']
    thresholds = {}
    for m in METHODS:
        pos = np.sort([r['own'][m] for r in cal_edited])
        thresholds[m] = float(pos[int(np.ceil(0.95 * len(pos))) - 1])

    calibration = dict(
        chroma6_scale=chroma6_scale.tolist(),
        chroma3_scale=chroma3_scale.tolist(),
        null_n=len(null_layout),
        thresholds_95=thresholds,
    )
    write_json(RUN / 'calibration.json', calibration)

    def summarize_waves(rows, name):
        edited = [r for r in rows if r['family'] != 'clean']
        clean = [r for r in rows if r['family'] == 'clean']
        res = {}
        for m in METHODS:
            th = thresholds[m]
            tpr = sum(r['own'][m] <= th for r in edited)
            far = sum(r['sib'][m] <= th for r in edited)
            clean_far = sum(r['sib'][m] <= th for r in clean)

            # Descriptive achieved 95% TPR ROC on this split
            pos = np.sort([r['own'][m] for r in edited])
            roc_th = float(pos[int(np.ceil(0.95 * len(pos))) - 1])
            roc_far = sum(r['sib'][m] <= roc_th for r in edited)

            # Per-edit breakdown (at frozen threshold)
            by_edit = {}
            for fam in families:
                for st in strengths:
                    sub = [r for r in edited if r['family'] == fam and r['strength'] == st]
                    by_edit[f'{fam}_{st}'] = dict(
                        n=len(sub),
                        tpr=sum(r['own'][m] <= th for r in sub),
                        far=sum(r['sib'][m] <= th for r in sub),
                    )
            res[m] = dict(
                bits=BIT_BUDGETS[m],
                n_edited=len(edited),
                tpr_count=tpr,
                tpr=tpr / len(edited),
                sib_far_count=far,
                sib_far=far / len(edited),
                clean_sib_far_count=clean_far,
                clean_sib_far=clean_far / len(clean),
                roc95_sib_far_count=roc_far,
                roc95_sib_far=roc_far / len(edited),
                by_edit=by_edit,
            )
        return res

    test_summary = summarize_waves(test_rows, 'test_60')
    conf_summary = summarize_waves(conf_rows, 'confirm_60')

    # Helper to score a single (ref_image, query_image) pair
    def score_pair(ref_im: Image.Image, q_im: Image.Image) -> dict:
        rf = extract_continuous(ref_im)
        dec = decode_payloads(encode_payloads(rf))
        qf = extract_continuous(q_im)
        d = raw_distances(dec, qf, chroma6_scale, chroma3_scale)
        p_l = (1 + int(np.searchsorted(null_layout, d['spatial_luma_chroma_64b_layout'], side='right'))) / (len(null_layout) + 1)
        p_c = (1 + int(np.searchsorted(null_chroma, d['spatial_luma_chroma_64b_chroma'], side='right'))) / (len(null_chroma) + 1)
        d['spatial_luma_chroma_64b'] = p_l * p_c
        return d

    # === Published Attack 1: Full-budget Imprint (step 150) ===
    print('Evaluating published Imprint attacks (10 GS + 10 TR + 10 PRC)...', flush=True)
    imprint_pairs = []
    # 10 GS sources
    rp_proto = read_json(ROOT / 'runs/author_reprompt_plus_20261115/protocol.json')
    for ref in rp_proto['references']:
        if ref['carrier'] == 'GS':
            ref_p = Path(ref['image'])
            atk_p = ref_p.parent.parent / 'attacked_images_step_150' / ref_p.name
            assert ref_p.exists() and atk_p.exists()
            imprint_pairs.append(('GS', ref['id'], ref_p, atk_p))
    # 10 TR + 10 PRC sources
    for carrier in ('TR', 'PRC'):
        base = ROOT / f'runs/pgid_carrier_coverage_20261113/{carrier}/attack=forgery'
        for i in range(10):
            ref_p = base / 'watermarked_images' / f'image_{i:03d}.png'
            atk_p = base / 'attacked_images_step_150' / f'image_{i:03d}.png'
            assert ref_p.exists() and atk_p.exists()
            imprint_pairs.append((carrier, f'{carrier}_{i:03d}', ref_p, atk_p))

    imprint_rows = []
    for carrier, sid, ref_p, atk_p in imprint_pairs:
        ref_im = Image.open(ref_p).convert('RGB')
        atk_im = Image.open(atk_p).convert('RGB')
        d = score_pair(ref_im, atk_im)
        # Reinhard diagnostic (explicitly non-attack diagnostic)
        atk_reinhard = reinhard_align(atk_im, ref_im)
        d_reinhard = score_pair(ref_im, atk_reinhard)
        imprint_rows.append(dict(carrier=carrier, id=sid, scores=d, reinhard_scores=d_reinhard))

    imprint_summary = {}
    for carrier in ('GS', 'TR', 'PRC', 'ALL'):
        sub = imprint_rows if carrier == 'ALL' else [r for r in imprint_rows if r['carrier'] == carrier]
        imprint_summary[carrier] = {
            m: dict(
                n=len(sub),
                accept=sum(r['scores'][m] <= thresholds[m] for r in sub),
                reinhard_diag_accept=sum(r['reinhard_scores'][m] <= thresholds[m] for r in sub),
            )
            for m in METHODS
        }

    # === Published Attack 2: Full-budget Reprompt+ (180 GS + 60 TR across SD2.1 & SD1.5) ===
    print('Evaluating published Reprompt+ attacks (240 candidates across 20 sources x 2 proxies)...', flush=True)
    rp_dir = ROOT / 'runs/author_reprompt_plus_20261115'
    rp_rows = []
    for ref in rp_proto['references']:
        carrier = ref['carrier']
        sid = ref['id']
        ref_im = Image.open(ref['image']).convert('RGB')
        n_res = 3 if carrier == 'GS' else 1
        for proxy in ('sd21', 'sd15'):
            for pi in range(3):
                for ri in range(n_res):
                    cand_p = rp_dir / proxy / 'images' / f'{sid}_{proxy}_p{pi}_r{ri}.png'
                    assert cand_p.exists(), f'Missing {cand_p}'
                    cand_im = Image.open(cand_p).convert('RGB')
                    d = score_pair(ref_im, cand_im)
                    d_reinhard = score_pair(ref_im, reinhard_align(cand_im, ref_im))
                    rp_rows.append(dict(
                        carrier=carrier, id=sid, proxy=proxy, p=pi, r=ri,
                        scores=d, reinhard_scores=d_reinhard,
                    ))

    reprompt_summary = {}
    for carrier in ('GS', 'TR'):
        for proxy in ('sd21', 'sd15'):
            sub = [r for r in rp_rows if r['carrier'] == carrier and r['proxy'] == proxy]
            sids = sorted({r['id'] for r in sub})
            key = f'{carrier}_{proxy}'
            reprompt_summary[key] = {}
            for m in METHODS:
                th = thresholds[m]
                cand_acc = sum(r['scores'][m] <= th for r in sub)
                src_acc = sum(any(r['scores'][m] <= th for r in sub if r['id'] == sid) for sid in sids)
                cand_rei = sum(r['reinhard_scores'][m] <= th for r in sub)
                src_rei = sum(any(r['reinhard_scores'][m] <= th for r in sub if r['id'] == sid) for sid in sids)
                reprompt_summary[key][m] = dict(
                    n_candidates=len(sub),
                    n_sources=len(sids),
                    cand_accept=cand_acc,
                    source_accept=src_acc,
                    reinhard_diag_cand_accept=cand_rei,
                    reinhard_diag_source_accept=src_rei,
                )

    # === External Benchmarks (Phase 2 + Phase 3 archived images) ===
    print('Evaluating archived MagicBrush (20) + CVPR Color Transfer (8) + ICCV Dist Match (1)...', flush=True)
    p2_dir = ROOT / 'runs/rgb_moments_phase2_20261120/author'
    p2_man = read_json(ROOT / 'runs/rgb_moments_phase2_20261120/public_manifest.json')
    p3_dir = ROOT / 'runs/local_semantics_phase3_20261120/additional'
    p3_man = read_json(ROOT / 'runs/local_semantics_phase3_20261120/additional_manifest.json')

    ext_rows = []
    for item in p2_man:
        ref_im = Image.open(p2_dir / item['files']['ref']).convert('RGB')
        out_im = Image.open(p2_dir / item['files']['output']).convert('RGB')
        ext_rows.append(dict(group=item['group'], item=item['item'], scores=score_pair(ref_im, out_im)))
    for item in p3_man:
        ref_im = Image.open(p3_dir / item['files']['ref']).convert('RGB')
        out_im = Image.open(p3_dir / item['files']['output']).convert('RGB')
        ext_rows.append(dict(group='replacement_extra', item=item['item'], scores=score_pair(ref_im, out_im)))

    ext_summary = {}
    for gname, gfilter in [
        ('iccv_dist_match', lambda r: r['group'] == 'distribution'),
        ('cvpr_color_transfer', lambda r: r['group'] == 'color'),
        ('magicbrush_all20', lambda r: r['group'] in ('replacement', 'replacement_extra')),
        ('magicbrush_p2_8', lambda r: r['group'] == 'replacement'),
        ('magicbrush_p3_extra12', lambda r: r['group'] == 'replacement_extra'),
    ]:
        sub = [r for r in ext_rows if gfilter(r)]
        ext_summary[gname] = {
            m: dict(n=len(sub), accept=sum(r['scores'][m] <= thresholds[m] for r in sub))
            for m in METHODS
        }

    elapsed = time.perf_counter() - t0
    summary = dict(
        elapsed_seconds=elapsed,
        thresholds_95=thresholds,
        test_60=test_summary,
        confirm_60=conf_summary,
        imprint=imprint_summary,
        reprompt_plus=reprompt_summary,
        external=ext_summary,
    )
    write_json(RUN / 'summary.json', summary)

    # Save compact per-sample CSV for auditability
    with gzip.open(RUN / 'waves_scores.csv.gz', 'wt', newline='') as f:
        w = csv.writer(f)
        w.writerow(['split', 'dataset', 'task', 'seed_idx', 'family', 'strength', 'method', 'own', 'sib'])
        for sname, rows in [('cal', cal_rows), ('test', test_rows), ('confirm', conf_rows)]:
            for r in rows:
                for m in METHODS:
                    w.writerow([sname, r['dataset'], r['task'], r['seed_idx'], r['family'], r['strength'], m,
                                f"{r['own'][m]:.8f}", f"{r['sib'][m]:.8f}"])

    write_json(RUN / 'complete.json', dict(
        elapsed_seconds=elapsed,
        script_sha256=sha_file(Path(__file__)),
        counts=dict(cal=len(cal_rows), test=len(test_rows), confirm=len(conf_rows),
                    imprint=len(imprint_rows), reprompt_plus=len(rp_rows), external=len(ext_rows)),
    ))
    print(f'DONE in {elapsed:.1f}s', flush=True)


if __name__ == '__main__':
    main()
