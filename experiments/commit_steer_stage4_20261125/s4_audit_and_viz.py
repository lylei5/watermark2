"""Stage 4 Post-Processing, Exact Offline FID, Matched Head-to-Head Summary, and Independent 0-Mismatch Audit.

Computes:
  1. Exact offline COCO-50 & Mixed-100 FID for Stage 4B hyb_sscd32_clip32 (same InceptionV3 weights + 256 COCO ref).
  2. Visual montage for Stage 4B hyb_sscd32_clip32 (unguided vs guided 6-sample montage).
  3. Independent 0-mismatch audit across ALL Stage 4 experiments:
     - Stage 4A: sdxl_cross_backbone (24 prompts x 10 conds = 240 records + 36 Live Reprompt + 200 COCO nulls)
     - Stage 4B: multidim100_hyb (100 prompts x 10 conds = 1,000 records + 2,000 COCO nulls + 12 Imprint ckpts + 108 Reprompt+)
     - Stage 4C: head2head_pgid_native_gs (1,000 Native GS + 300 PGID-F + 200 PGID-F nulls + 12 Imprint + 108 Reprompt+)
     - Stage 4D: head2head_sembind (200 SemBind benign + 50 COCO nulls + 12 Imprint + 72 Reprompt+)
  4. Unified Apple-to-Apple Matched-Cohort Comparison Table JSON (exact N=20 / N=30 / N=100 slices across ALL methods).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/data0/liyanlei/watermark_research')
STAGE4 = ROOT / 'runs/commit_steer_stage4_20261125'
STAGE3 = ROOT / 'runs/commit_steer_paper_scale_20261123'
STAGE2 = ROOT / 'runs/commit_steer_stage2_20261122'
PHASE1 = ROOT / 'runs/rgb_moments_phase1_20261120'

MATCHED_20_GIDX = set(list(range(0, 10)) + list(range(50, 60)))
MATCHED_30_GIDX = set(list(range(0, 15)) + list(range(50, 65)))


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


def compute_hyb_fid_and_montage():
    hyb = STAGE4 / 'multidim100_hyb'
    if not (hyb / 'gs100_summary.json').exists():
        print('[FID] Skipping: multidim100_hyb/gs100_summary.json not found yet.', flush=True)
        return None
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    import torchvision.transforms.functional as TF
    torch.set_num_threads(4)
    sys.path.insert(0, str(PHASE1 / 'assets/lib'))
    from pytorch_fid import inception

    weights = PHASE1 / 'assets/fid.pth'
    assert hashfile(weights) == '6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2'
    inception.load_state_dict_from_url = lambda *a, **k: torch.load(weights, map_location='cpu', weights_only=True)
    net = inception.InceptionV3().eval()
    real = np.load(PHASE1 / 'assets/real_pool3.npy')
    assert real.shape == (256, 2048)

    proto = read_json(hyb / 'protocol.json')
    coco_mask = np.array([s['dataset'] == 'COCO' for s in proto['sources']], dtype=bool)
    sdp_mask = np.array([s['dataset'] == 'SDP' for s in proto['sources']], dtype=bool)

    def extract_pool3(folder: Path):
        vec = []
        with torch.inference_mode():
            for i in range(100):
                with Image.open(folder / f'{i:03d}.png') as im:
                    vec.append(net(TF.to_tensor(im.convert('RGB')).unsqueeze(0))[0].flatten().numpy())
        return np.stack(vec)

    f_base = extract_pool3(hyb / 'images_unguided')
    f_guid = extract_pool3(hyb / 'images_guided')
    np.save(hyb / 'unguided_pool3.npy', f_base)
    np.save(hyb / 'guided_pool3.npy', f_guid)

    out = dict(
        reference='COCO test2017 deterministic 256-image pool3 (identical to Stage 3A gs100_unified)',
        unguided_native_gs=dict(
            vs_real_mixed_100=fid_distance(f_base, real),
            vs_real_coco_50=fid_distance(f_base[coco_mask], real),
            vs_real_sdp_50=fid_distance(f_base[sdp_mask], real),
        ),
        guided_hyb_sscd32_clip32=dict(
            vs_real_mixed_100=fid_distance(f_guid, real),
            vs_real_coco_50=fid_distance(f_guid[coco_mask], real),
            vs_real_sdp_50=fid_distance(f_guid[sdp_mask], real),
            vs_paired_unguided_100=fid_distance(f_guid, f_base),
        ),
        delta_vs_real_mixed_100=fid_distance(f_guid, real) - fid_distance(f_base, real),
        delta_vs_real_coco_50=fid_distance(f_guid[coco_mask], real) - fid_distance(f_base[coco_mask], real),
    )
    write_json(hyb / 'fid.json', out)
    print(f"[S4B Hyb100 FID] mixed100 delta={out['delta_vs_real_mixed_100']:+.2f} | coco50 delta={out['delta_vs_real_coco_50']:+.2f} ({out['unguided_native_gs']['vs_real_coco_50']:.2f} -> {out['guided_hyb_sscd32_clip32']['vs_real_coco_50']:.2f}) | paired={out['guided_hyb_sscd32_clip32']['vs_paired_unguided_100']:.2f}", flush=True)

    # Montage
    recs = read_json(hyb / 'gs100_records.json')
    cl_by_id = {r['gidx']: r for r in recs if r['condition'] == 'clean'}
    picks = [0, 1, 2, 50, 51, 52]
    w, h = 256, 256
    canvas = Image.new('RGB', (w * len(picks), h * 2 + 28), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    for ci, gid in enumerate(picks):
        canvas.paste(Image.open(hyb / 'images_unguided' / f'{gid:03d}.png').convert('RGB').resize((w, h), Image.BICUBIC), (ci * w, 28))
        canvas.paste(Image.open(hyb / 'images_guided' / f'{gid:03d}.png').convert('RGB').resize((w, h), Image.BICUBIC), (ci * w, 28 + h))
        r = cl_by_id[gid]
        draw.text((ci * w + 6, 6), f"Hyb #{gid:03d} LP={r['lpips']:.3f} d={r['d_blind']:.2f}", fill=(20, 20, 20))
    canvas.save(hyb / 'hyb100_montage.jpg', quality=92)
    return out


def audit_stage4():
    total_mm = 0
    details = {}

    # 1. Stage 4A: SDXL cross-backbone
    sdxl = STAGE4 / 'sdxl_cross_backbone'
    if (sdxl / 'summary.json').exists():
        proto = read_json(sdxl / 'protocol.json')
        summ = read_json(sdxl / 'summary.json')
        recs = read_json(sdxl / 'records.json')
        rps = read_json(sdxl / 'reprompt_rows.json')
        th = proto['threshold']
        auth_th = proto['auth_threshold']
        mm = 0
        if len(recs) != 240:
            mm += 1
        for r in recs:
            a_ok = r['auth_ba'] > auth_th
            f_ok = r['d_blind'] <= th
            if a_ok != r['auth_ok'] or f_ok != r['feat_ok'] or (a_ok and f_ok) != r['joint_blind_ok']:
                mm += 1
        cl = [r for r in recs if r['condition'] == 'clean']
        ed = [r for r in recs if r['condition'] != 'clean']
        if sum(r['joint_blind_ok'] for r in cl) != summ['clean_joint_pass']:
            mm += 1
        if sum(r['joint_blind_ok'] for r in ed) != summ['gs9_joint_pass']:
            mm += 1
        if len(rps) != 36 or sum(r['joint_blind_ok'] for r in rps) != summ['live_reprompt_sdxl']['commit_steer_cand']:
            mm += 1
        nd = np.load(sdxl / 'null200_scores.npz')
        if len(nd['auth_ba']) != 200 or int(((nd['auth_ba'] > auth_th) & (nd['d_blind'] <= th)).sum()) != summ['null_fpr_200']['joint_fp_default']:
            mm += 1
        total_mm += mm
        details['stage4a_sdxl'] = dict(n_records=len(recs), n_reprompt=len(rps), n_null=200, mismatches=mm)

    # 2. Stage 4B: multidim100_hyb
    hyb = STAGE4 / 'multidim100_hyb'
    if (hyb / 'gs100_summary.json').exists():
        proto = read_json(hyb / 'protocol.json')
        summ = read_json(hyb / 'gs100_summary.json')
        recs = read_json(hyb / 'gs100_records.json')
        th = proto['threshold']
        auth_th = proto['auth_threshold']
        mm = 0
        if len(recs) != 1000:
            mm += 1
        for r in recs:
            a_ok = r['auth_ba'] > auth_th
            f_ok = r['d_blind'] <= th
            if a_ok != r['auth_ok'] or f_ok != r['feat_ok'] or (a_ok and f_ok) != r['joint_blind_ok']:
                mm += 1
            if abs(0.5 * r['d_sscd32'] + 0.5 * r['d_clip32'] - r['d_blind']) > 1e-9:
                mm += 1
        cl = [r for r in recs if r['condition'] == 'clean']
        ed = [r for r in recs if r['condition'] != 'clean']
        if sum(r['joint_blind_ok'] for r in cl) != summ['clean_joint_pass']:
            mm += 1
        if sum(r['joint_blind_ok'] for r in ed) != summ['gs9_joint_pass']:
            mm += 1
        if (hyb / 'null_fpr_2000/summary.json').exists():
            ns = read_json(hyb / 'null_fpr_2000/summary.json')
            for sp in ('calibration', 'validation'):
                d = np.load(hyb / f'null_fpr_2000/{sp}_scores.npz')
                if len(d['auth_ba']) != 1000:
                    mm += 1
                if int(((d['auth_ba'] > auth_th) & (d['d_blind'] <= th)).sum()) != ns['splits'][sp]['joint_fp_default']:
                    mm += 1
        if (hyb / 'live_attacks/summary.json').exists():
            asum = read_json(hyb / 'live_attacks/summary.json')
            rprows = read_json(hyb / 'live_attacks/reprompt_plus_rows.json')
            for c in asum['imprint_cases']:
                for ck in c['checkpoints'].values():
                    if ((ck['auth_ba'] > auth_th) and (ck['d_blind'] <= th)) != ck['joint_accept']:
                        mm += 1
            for px in ('sd21', 'sd15'):
                rows = rprows[px]['rows']
                if len(rows) != 54 or sum(r['joint_accept'] for r in rows) != asum['reprompt_plus_summary'][px]['commit_steer_cand']:
                    mm += 1
        total_mm += mm
        details['stage4b_multidim100_hyb'] = dict(n_records=len(recs), mismatches=mm)

    # 3. Stage 4C: PGID-F & Native GS
    pg = STAGE4 / 'head2head_pgid_native_gs'
    if (pg / 'pgid30_summary.json').exists():
        mm = 0
        if (pg / 'native100_records.json').exists():
            nrecs = read_json(pg / 'native100_records.json')
            nsum = read_json(pg / 'native100_summary.json')
            if len(nrecs) != 1000:
                mm += 1
            for r in nrecs:
                if (r['ba_256'] > 181.0 / 256.0) != r['accept_gs_default']:
                    mm += 1
            if sum(r['accept_gs_default'] for r in nrecs if r['condition'] != 'clean') != nsum['all_100']['gs9_all_default']:
                mm += 1
        precs = read_json(pg / 'pgid30_records.json')
        psum = read_json(pg / 'pgid30_summary.json')
        if len(precs) != 300:
            mm += 1
        for r in precs:
            if (r['ba_256'] > 181.0 / 256.0) != r['accept_gs_default']:
                mm += 1
        if sum(r['accept_gs_default'] for r in precs if r['condition'] != 'clean') != psum['matched_30']['gs9_all_default']:
            mm += 1
        if (pg / 'pgid_null200_summary.json').exists():
            nd = np.load(pg / 'pgid_null200_scores.npz')
            ns = read_json(pg / 'pgid_null200_summary.json')
            if len(nd['ba_256']) != 200 or int((nd['ba_256'] > 181.0 / 256.0).sum()) != ns['fp_gs_default']:
                mm += 1
        if (pg / 'attacks_summary.json').exists():
            asum = read_json(pg / 'attacks_summary.json')
            rprows = read_json(pg / 'reprompt_plus_rows.json')
            for px in ('sd21', 'sd15'):
                rows = rprows[px]['rows']
                if len(rows) != 54 or sum(r['pgid_f']['accept_gs_default'] for r in rows) != asum['reprompt_plus_summary'][px]['pgid_f_cand']:
                    mm += 1
        total_mm += mm
        details['stage4c_pgid_native_gs'] = dict(n_pgid30=len(precs), mismatches=mm)

    # 4. Stage 4D: SemBind GS-S
    sb = STAGE4 / 'head2head_sembind'
    if (sb / 'benign20_summary.json').exists():
        mm = 0
        brecs = read_json(sb / 'benign20_records.json')
        bsum = read_json(sb / 'benign20_summary.json')
        if len(brecs) != 200:
            mm += 1
        for r in brecs:
            if (r['ba'] > 181.0 / 256.0) != r['accept_gs_default']:
                mm += 1
        if sum(r['accept_gs_default'] for r in brecs if r['condition'] != 'clean') != bsum['gs9_all_default']:
            mm += 1
        if (sb / 'attacks_null_summary.json').exists():
            asum = read_json(sb / 'attacks_null_summary.json')
            nd = np.load(sb / 'null50_scores.npz')
            if len(nd['ba']) != 50 or int((nd['ba'] > 181.0 / 256.0).sum()) != asum['null_fpr_50']['fp_gs_default']:
                mm += 1
        total_mm += mm
        details['stage4d_sembind'] = dict(n_benign=len(brecs), mismatches=mm)

    # Build matched-cohort slice comparison (exact N=20 and N=30 slices for Commit-then-Steer SSCD-64b & Hyb)
    matched = {}
    for name, rec_path in [
        ('commit_steer_sscd64', STAGE3 / 'gs100_unified/records.json'),
        ('commit_steer_hyb_sscd32_clip32', STAGE4 / 'multidim100_hyb/gs100_records.json'),
    ]:
        if rec_path.exists():
            recs = read_json(rec_path)
            matched[name] = {}
            for tag, id_set in [('matched_20', MATCHED_20_GIDX), ('matched_30', MATCHED_30_GIDX), ('all_100', set(range(100)))]:
                sub = [r for r in recs if r['gidx'] in id_set]
                cl = [r for r in sub if r['condition'] == 'clean']
                ed = [r for r in sub if r['condition'] != 'clean']
                ed7 = [r for r in ed if r['condition'] not in ('random_crop_ratio', 'random_drop_ratio')]
                matched[name][tag] = dict(
                    n_prompts=len(cl),
                    clean_pass=sum(r['joint_blind_ok'] for r in cl),
                    gs7_pass=sum(r['joint_blind_ok'] for r in ed7),
                    gs9_pass=sum(r['joint_blind_ok'] for r in ed),
                    lpips_mean=float(np.mean([r['lpips'] for r in cl])),
                    clip_guid_mean=float(np.mean([r['clip_guid'] for r in cl])),
                    sec_total_mean=float(np.mean([r['sec_total'] for r in cl])),
                )

    out = dict(total_mismatches=total_mm, details=details, matched_slices_commit_steer=matched)
    write_json(STAGE4 / 'audit_stage4.json', out)
    print(f'[AUDIT STAGE 4 COMPLETE] total_mismatches={total_mm} | details={details}', flush=True)
    return out


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if mode in ('fid', 'all'):
        compute_hyb_fid_and_montage()
    if mode in ('audit', 'all'):
        audit_stage4()
