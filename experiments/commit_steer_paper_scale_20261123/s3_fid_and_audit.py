"""Stage 3 Post-Processing:
1. Offline COCO FID (exact same InceptionV3 weights + 256-image COCO reference as Phase 1)
   comparing Unguided Native GS-256b vs Commit-then-Steer Guided GS-256b (100 unified images: 50 COCO + 50 SDP).
2. Independent re-verification audit across:
   - gs100_unified (1,000 records + 9,900 cross-prompt pairs)
   - null_fpr_2000 (2,000 real COCO images: 1,000 cal + 1,000 val)
   - adaptive_and_prc16 (white-box adaptive PGD stress test + 16-prompt PRC-512b)
   - probe_b_live_attacks (4-case 151-step live CVPR 2025 Imprint)
3. Generates a clean 6-sample visual montage of Stage 3 GS100 (unguided vs guided) for visual inspection.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms.functional as TF

ROOT = Path('/data0/liyanlei/watermark_research')
STAGE3 = ROOT / 'runs/commit_steer_paper_scale_20261123'
STAGE2 = ROOT / 'runs/commit_steer_stage2_20261122'
PHASE1 = ROOT / 'runs/rgb_moments_phase1_20261120'


def read_json(p: Path):
    return json.loads(p.read_bytes())


def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(p)


def hashfile(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fid_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype('float64'), b.astype('float64')
    ma, mb = a.mean(0), b.mean(0)
    x, y = (a - ma) / np.sqrt(len(a) - 1), (b - mb) / np.sqrt(len(b) - 1)
    cross = np.linalg.svd(x @ y.T, compute_uv=False).sum()
    return float(max(0.0, ((ma - mb) ** 2).sum() + (x * x).sum() + (y * y).sum() - 2.0 * cross))


def compute_fid():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(4)
    sys.path.insert(0, str(PHASE1 / 'assets/lib'))
    from pytorch_fid import inception
    weights = PHASE1 / 'assets/fid.pth'
    assert hashfile(weights) == '6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2'
    inception.load_state_dict_from_url = lambda *a, **k: torch.load(weights, map_location='cpu', weights_only=True)
    net = inception.InceptionV3().eval()

    real = np.load(PHASE1 / 'assets/real_pool3.npy')
    assert real.shape == (256, 2048)

    gs100 = STAGE3 / 'gs100_unified'
    proto = read_json(gs100 / 'protocol.json')
    coco_mask = np.array([s['dataset'] == 'COCO' for s in proto['sources']], dtype=bool)
    sdp_mask = np.array([s['dataset'] == 'SDP' for s in proto['sources']], dtype=bool)

    def extract_pool3(folder: Path):
        vec = []
        with torch.inference_mode():
            for i in range(100):
                with Image.open(folder / f'{i:03d}.png') as im:
                    vec.append(net(TF.to_tensor(im.convert('RGB')).unsqueeze(0))[0].flatten().numpy())
        return np.stack(vec)

    f_base = extract_pool3(gs100 / 'images_unguided')
    f_guid = extract_pool3(gs100 / 'images_guided')
    np.save(gs100 / 'unguided_pool3.npy', f_base)
    np.save(gs100 / 'guided_pool3.npy', f_guid)

    out = dict(
        reference='COCO test2017 deterministic 256-image pool3 (identical to rgb_moments_phase1_20261120)',
        unguided_native_gs=dict(
            vs_real_mixed_100=fid_distance(f_base, real),
            vs_real_coco_50=fid_distance(f_base[coco_mask], real),
            vs_real_sdp_50=fid_distance(f_base[sdp_mask], real),
        ),
        guided_commit_steer=dict(
            vs_real_mixed_100=fid_distance(f_guid, real),
            vs_real_coco_50=fid_distance(f_guid[coco_mask], real),
            vs_real_sdp_50=fid_distance(f_guid[sdp_mask], real),
            vs_paired_unguided_100=fid_distance(f_guid, f_base),
        ),
        delta_vs_real_mixed_100=fid_distance(f_guid, real) - fid_distance(f_base, real),
        delta_vs_real_coco_50=fid_distance(f_guid[coco_mask], real) - fid_distance(f_base[coco_mask], real),
    )
    write_json(gs100 / 'fid.json', out)
    print(
        f"FID DONE | mixed100: unguided={out['unguided_native_gs']['vs_real_mixed_100']:.2f} vs guided={out['guided_commit_steer']['vs_real_mixed_100']:.2f} (delta={out['delta_vs_real_mixed_100']:+.2f}) | "
        f"coco50: unguided={out['unguided_native_gs']['vs_real_coco_50']:.2f} vs guided={out['guided_commit_steer']['vs_real_coco_50']:.2f} (delta={out['delta_vs_real_coco_50']:+.2f}) | "
        f"paired={out['guided_commit_steer']['vs_paired_unguided_100']:.2f}",
        flush=True,
    )
    return out


def make_gs100_montage():
    gs100 = STAGE3 / 'gs100_unified'
    recs = read_json(gs100 / 'records.json')
    cl_by_id = {r['gidx']: r for r in recs if r['condition'] == 'clean'}
    # Pick 3 COCO (0, 1, 2) + 3 SDP (50, 51, 52)
    picks = [0, 1, 2, 50, 51, 52]
    w, h = 256, 256
    canvas = Image.new('RGB', (w * len(picks), h * 2 + 28), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    for ci, gid in enumerate(picks):
        im_u = Image.open(gs100 / 'images_unguided' / f'{gid:03d}.png').convert('RGB').resize((w, h), Image.BICUBIC)
        im_g = Image.open(gs100 / 'images_guided' / f'{gid:03d}.png').convert('RGB').resize((w, h), Image.BICUBIC)
        canvas.paste(im_u, (ci * w, 28))
        canvas.paste(im_g, (ci * w, 28 + h))
        r = cl_by_id[gid]
        draw.text((ci * w + 6, 6), f"#{gid:03d} {r['dataset']} LP={r['lpips']:.3f} d={r['d_blind']:.2f}", fill=(20, 20, 20))
    canvas.save(gs100 / 'gs100_montage.jpg', quality=92)


def audit_all():
    mismatches = 0
    details = {}

    # 1. Audit gs100_unified
    gs100 = STAGE3 / 'gs100_unified'
    if (gs100 / 'summary.json').exists():
        proto = read_json(gs100 / 'protocol.json')
        summ = read_json(gs100 / 'summary.json')
        recs = read_json(gs100 / 'records.json')
        th = proto['threshold']
        auth_th = proto['auth_threshold']
        mm = 0
        if len(recs) != 1000:
            mm += 1
        for r in recs:
            a_ok = r['auth_ba'] > auth_th
            f_ok = r['d_blind'] <= th
            j_ok = a_ok and f_ok
            if a_ok != r['auth_ok'] or f_ok != r['feat_ok'] or j_ok != r['joint_blind_ok']:
                mm += 1
        cl = [r for r in recs if r['condition'] == 'clean']
        ed = [r for r in recs if r['condition'] != 'clean']
        if sum(r['joint_blind_ok'] for r in cl) != summ['clean_joint_pass']:
            mm += 1
        if sum(r['joint_blind_ok'] for r in ed) != summ['gs9_joint_pass']:
            mm += 1
        if abs(float(np.mean([r['lpips'] for r in cl])) - summ['lpips_mean']) > 1e-9:
            mm += 1
        mismatches += mm
        details['gs100_unified'] = dict(n_records=len(recs), mismatches=mm)

    # 2. Audit null_fpr_2000
    nf = STAGE3 / 'null_fpr_2000'
    if (nf / 'summary.json').exists():
        summ = read_json(nf / 'summary.json')
        th = summ['sscd_threshold']
        mm = 0
        for sp in ('calibration', 'validation'):
            d = np.load(nf / f'{sp}_scores.npz')
            ab, db = d['auth_ba'], d['d_blind']
            if len(ab) != 1000 or len(db) != 1000:
                mm += 1
            if int(((ab > 136.0 / 192.0) & (db <= th)).sum()) != summ['splits'][sp]['joint_fp_default']:
                mm += 1
            if int((ab > 136.0 / 192.0).sum()) != summ['splits'][sp]['auth_fp_default']:
                mm += 1
            if int((db <= th).sum()) != summ['splits'][sp]['feat_fp_only']:
                mm += 1
        mismatches += mm
        details['null_fpr_2000'] = dict(n_images=2000, mismatches=mm)

    # 3. Audit adaptive_and_prc16
    ap16 = STAGE3 / 'adaptive_and_prc16'
    if (ap16 / 'summary.json').exists():
        summ = read_json(ap16 / 'summary.json')
        recs = read_json(ap16 / 'prc16_records.json')
        mm = 0
        if len(recs) != 80:
            mm += 1
        cl = [r for r in recs if r['condition'] == 'clean']
        ed = [r for r in recs if r['condition'] != 'clean']
        if sum(r['joint'] for r in cl) != summ['prc16_unified']['clean_joint']:
            mm += 1
        if sum(r['joint'] for r in ed) != summ['prc16_unified']['edited_joint']:
            mm += 1
        mismatches += mm
        details['adaptive_and_prc16'] = dict(n_prc_records=len(recs), n_adaptive_rows=len(summ['adaptive_stress_test']), mismatches=mm)

    # 4. Audit probe_b_live_attacks (Imprint 151-step + Reprompt+ 108-cand)
    imp = STAGE2 / 'probe_b_live_attacks/summary.json'
    if imp.exists():
        imp_data = read_json(imp)
        th = imp_data['threshold']
        mm = 0
        n_ck = 0
        for case in imp_data['imprint_cases']:
            for ck_name, ck in case['checkpoints'].items():
                n_ck += 1
                if (ck['d_blind'] <= th) != ck['feat_ok']:
                    mm += 1
                if (ck['auth_ok'] and ck['feat_ok']) != ck['joint_accept']:
                    mm += 1
        for ck_name in ('step_50', 'step_100', 'step_150'):
            s_ck = imp_data['imprint_summary'][ck_name]
            if sum(c['checkpoints'][ck_name]['auth_ok'] for c in imp_data['imprint_cases']) != s_ck['native_gs_fooled']:
                mm += 1
            if sum(c['checkpoints'][ck_name]['joint_accept'] for c in imp_data['imprint_cases']) != s_ck['commit_steer_fooled']:
                mm += 1
        mismatches += mm
        details['probe_b_imprint'] = dict(n_cases=len(imp_data['imprint_cases']), n_checkpoints=n_ck, mismatches=mm)

    # 5. Audit multidim_features (Stage 3D)
    md = STAGE3 / 'multidim_features/summary.json'
    if md.exists():
        md_data = read_json(md)
        mm = 0
        if len(md_data.get('stage0_multidim', {})) != 7 or len(md_data.get('live_end2end', {})) != 2:
            mm += 1
        mismatches += mm
        details['multidim_features'] = dict(n_stage0_methods=len(md_data.get('stage0_multidim', {})), n_live_arms=len(md_data.get('live_end2end', {})), mismatches=mm)

    # 6. Audit window_ablation (Stage 3E)
    wa = STAGE3 / 'window_ablation'
    if (wa / 'summary.json').exists():
        ws = read_json(wa / 'summary.json')
        wr = read_json(wa / 'records.json')
        th = read_json(ROOT / 'runs/payload_binding_stage0_20261122/summary_sscd.json')['thresholds_95']['sscd_64b']
        mm = 0
        if len(wr) != 12 * 4 * 5:
            mm += 1
        for r in wr:
            if (r['auth_ok'] and r['d_blind'] <= th) != r['joint']:
                mm += 1
        for arm_name, info in ws['arms'].items():
            cl = [r for r in wr if r['arm'] == arm_name and r['condition'] == 'clean']
            ed = [r for r in wr if r['arm'] == arm_name and r['condition'] != 'clean']
            if f"{sum(r['joint'] for r in cl)}/12" != info['clean_joint']:
                mm += 1
            if f"{sum(r['joint'] for r in ed)}/48" != info['edit4_joint']:
                mm += 1
        mismatches += mm
        details['window_ablation'] = dict(n_records=len(wr), mismatches=mm)

    out = dict(total_mismatches=mismatches, details=details)
    write_json(STAGE3 / 'audit_stage3.json', out)
    print(f'AUDIT STAGE 3 COMPLETE | total_mismatches={mismatches} | details={details}', flush=True)
    return out


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if mode in ('fid', 'all'):
        compute_fid()
        make_gs100_montage()
    if mode in ('audit', 'all'):
        audit_all()
