"""Stage 6 Independent 0-Mismatch Audit & Visual Montages (CPU-only)."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import fisher_exact

ROOT = Path('/data0/liyanlei/watermark_research')
STAGE6 = ROOT / 'runs/commit_steer_stage6_20261126'
SEMBIND_S4 = ROOT / 'runs/commit_steer_stage4_20261125/head2head_sembind'
HYB_S4 = ROOT / 'runs/commit_steer_stage4_20261125/multidim100_hyb'
PGID_S4 = ROOT / 'runs/commit_steer_stage4_20261125/head2head_pgid_native_gs'
S5_SEAL = ROOT / 'runs/commit_steer_stage5_20261126/seal_head2head'
S5_CS = ROOT / 'runs/commit_steer_stage5_20261126/removal_and_instance_cs'
S5_SEMBIND = ROOT / 'runs/commit_steer_stage5_20261126/sembind_instance_and_removal'


def read_json(p: Path):
    return json.loads(p.read_bytes())


def write_json(p: Path, obj):
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(p)


def fid_svd(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    ma, mb = a.mean(0), b.mean(0)
    x, y = (a - ma) / np.sqrt(len(a) - 1), (b - mb) / np.sqrt(len(b) - 1)
    cross = np.linalg.svd(x @ y.T, compute_uv=False).sum()
    return float(max(0.0, ((ma - mb) ** 2).sum() + (x * x).sum() + (y * y).sum() - 2.0 * cross))


def audit_coco1k():
    d = STAGE6 / 'coco1k_fid'
    proto = read_json(d / 'protocol.json')
    recs = read_json(d / 'records.json')
    summ = read_json(d / 'summary.json')
    mm = 0

    if len(recs) != 1000 or [r['idx'] for r in recs] != list(range(1000)):
        mm += 1
    th_s, th_h, auth_th = float(proto['sscd64_threshold']), float(proto['hyb_threshold']), float(proto['auth_threshold'])

    for arm_key, summ_key, th in [('sscd64', 'sscd64', th_s), ('hyb', 'hyb_sscd32_clip32', th_h)]:
        for r in recs:
            cv = r[arm_key]['clean_verify']
            if cv['auth_ok'] != (cv['auth_ba'] > auth_th):
                mm += 1
            if cv['feat_ok'] != (cv['d_blind'] <= th):
                mm += 1
            if cv['joint_accept'] != (cv['auth_ok'] and cv['feat_ok']):
                mm += 1
        s_arm = summ[summ_key]
        if s_arm['clean_joint_pass'] != sum(r[arm_key]['clean_verify']['joint_accept'] for r in recs):
            mm += 1
        if s_arm['clean_auth_pass'] != sum(r[arm_key]['clean_verify']['auth_ok'] for r in recs):
            mm += 1
        if abs(s_arm['lpips_mean'] - float(np.mean([r[arm_key]['lpips'] for r in recs]))) > 1e-6:
            mm += 1
        if abs(s_arm['clip_guid_mean'] - float(np.mean([r[arm_key]['clip_guid'] for r in recs]))) > 1e-6:
            mm += 1

    # Independently recompute all FIDs from cached pool3 arrays
    fc = d / 'pool3_cache'
    f_r5 = np.load(fc / 'real_coco_5k.npy')
    f_r1 = f_r5[:1000]
    f_bs = np.load(fc / 'base_sscd_1k.npy')
    f_gs = np.load(fc / 'guid_sscd_1k.npy')
    f_bh = np.load(fc / 'base_hyb_1k.npy')
    f_gh = np.load(fc / 'guid_hyb_1k.npy')
    assert f_r5.shape == (5000, 2048) and f_gs.shape == (1000, 2048) and f_gh.shape == (1000, 2048)

    checks = [
        (summ['fid']['vs_real_coco_5k']['native_gs_sscd_seed'], fid_svd(f_bs, f_r5)),
        (summ['fid']['vs_real_coco_5k']['commit_steer_sscd64'], fid_svd(f_gs, f_r5)),
        (summ['fid']['vs_real_coco_5k']['native_gs_hyb_seed'], fid_svd(f_bh, f_r5)),
        (summ['fid']['vs_real_coco_5k']['commit_steer_hyb'], fid_svd(f_gh, f_r5)),
        (summ['fid']['vs_real_coco_1k']['commit_steer_sscd64'], fid_svd(f_gs, f_r1)),
        (summ['fid']['vs_real_coco_1k']['commit_steer_hyb'], fid_svd(f_gh, f_r1)),
        (summ['fid']['paired_drift_1k']['sscd64_vs_paired_unguided'], fid_svd(f_gs, f_bs)),
        (summ['fid']['paired_drift_1k']['hyb_vs_paired_unguided'], fid_svd(f_gh, f_bh)),
    ]
    for reported, recomputed in checks:
        if abs(reported - recomputed) > 1e-2:
            mm += 1
    return dict(n_prompts=len(recs), fid_checks=len(checks), mismatches=mm)


def audit_imprint10():
    d = STAGE6 / 'imprint10'
    proto = read_json(d / 'protocol.json')
    cross = read_json(d / 'cross_prompt_n10.json')
    same = read_json(d / 'same_prompt_n10.json')
    summ = read_json(d / 'summary.json')
    mm = 0

    th_s, th_h, auth_th = float(proto['sscd64_threshold']), float(proto['hyb_threshold']), float(proto['auth_threshold'])

    # 1. Verify exact 0..3 provenance against frozen Stage 4/5 files
    if cross['commit_steer_hyb'][:4] != read_json(HYB_S4 / 'live_attacks/summary.json')['imprint_cases']:
        mm += 1
    if cross['native_gs_and_pgid_f'][:4] != read_json(PGID_S4 / 'attacks_summary.json')['imprint_cases']:
        mm += 1
    if cross['sembind_gs_s'][:4] != read_json(SEMBIND_S4 / 'attacks_null_summary.json')['imprint_cases']:
        mm += 1
    if cross['seal'][:4] != read_json(S5_SEAL / 'attacks_summary.json')['imprint_cases']:
        mm += 1
    if same['commit_steer_sscd64'][:4] != read_json(S5_CS / 'same_prompt_imprint_sscd64_summary.json')['imprint_cases']:
        mm += 1
    if same['commit_steer_hyb'][:4] != read_json(S5_CS / 'same_prompt_imprint_hyb_summary.json')['imprint_cases']:
        mm += 1
    if same['sembind_gs_s'][:4] != read_json(S5_SEMBIND / 'instance_forgery_summary.json')['imprint_cases']:
        mm += 1

    # 2. Verify every single decision & aggregate count across all 10 cases x 3 ckpts
    for grp_name, grp_data, grp_summ in [('cross', cross, summ['cross_prompt']), ('same', same, summ['same_prompt'])]:
        for arm_key, th in [('commit_steer_sscd64', th_s), ('commit_steer_hyb', th_h)]:
            cases = grp_data[arm_key]
            if [c['gidx'] for c in cases] != list(range(10)):
                mm += 1
            for ck in ('step_50', 'step_100', 'step_150'):
                c_auth = c_joint = 0
                for c in cases:
                    v = c['checkpoints'][ck]
                    a_ok = v['auth_ba'] > auth_th
                    f_ok = v['d_blind'] <= th
                    if v['auth_ok'] != a_ok or v['feat_ok'] != f_ok or v['joint_accept'] != (a_ok and f_ok):
                        mm += 1
                    c_auth += int(a_ok)
                    c_joint += int(a_ok and f_ok)
                if grp_summ[arm_key][ck]['native_auth_fooled'] != c_auth or grp_summ[arm_key][ck]['commit_steer_fooled'] != c_joint:
                    mm += 1

        sb_cases = grp_data['sembind_gs_s']
        if [c['gidx'] for c in sb_cases] != list(range(10)):
            mm += 1
        for ck in ('step_50', 'step_100', 'step_150'):
            fd = sum(c['checkpoints'][ck]['ba'] > 181.0 / 256.0 for c in sb_cases)
            f167 = sum(c['checkpoints'][ck]['ba'] >= 167.0 / 256.0 for c in sb_cases)
            for c in sb_cases:
                v = c['checkpoints'][ck]
                if v['accept_gs_default'] != (v['ba'] > 181.0 / 256.0):
                    mm += 1
            if grp_summ['sembind_gs_s'][ck]['fooled_default'] != fd or grp_summ['sembind_gs_s'][ck]['fooled_167'] != f167:
                mm += 1

    # Native GS + PGID-F cross
    for ck in ('step_50', 'step_100', 'step_150'):
        fn = fp = fp167 = 0
        for c in cross['native_gs_and_pgid_f']:
            sn, sp = c['checkpoints'][ck]['native_gs'], c['checkpoints'][ck]['pgid_f']
            if sn['accept_gs_default'] != (sn['ba_256'] > 181.0 / 256.0):
                mm += 1
            if sp['accept_gs_default'] != (sp['ba_256'] > 181.0 / 256.0) or sp['accept_gs_167'] != (sp['ba_256'] >= 167.0 / 256.0):
                mm += 1
            fn += int(sn['accept_gs_default'])
            fp += int(sp['accept_gs_default'])
            fp167 += int(sp['accept_gs_167'])
        s_np = summ['cross_prompt']['native_gs_and_pgid_f'][ck]
        if s_np['native_gs_fooled'] != fn or s_np['pgid_f_fooled_default'] != fp or s_np['pgid_f_fooled_167'] != fp167:
            mm += 1

    # SEAL cross
    for ck in ('step_50', 'step_100', 'step_150'):
        f12 = f4 = f2 = 0
        for c in cross['seal']:
            v = c['checkpoints'][ck]
            if v['accept_12'] != (v['matches'] >= 12) or v['accept_4'] != (v['matches'] >= 4) or v['accept_2'] != (v['matches'] >= 2):
                mm += 1
            f12 += int(v['accept_12']); f4 += int(v['accept_4']); f2 += int(v['accept_2'])
        s_sl = summ['cross_prompt']['seal'][ck]
        if s_sl['fooled_12'] != f12 or s_sl['fooled_4'] != f4 or s_sl['fooled_2'] != f2:
            mm += 1

    # Verify Fisher exact p-value for Same-Prompt hyb vs SemBind
    a = summ['same_prompt']['commit_steer_hyb']['step_150']['commit_steer_fooled']
    b = summ['same_prompt']['sembind_gs_s']['step_150']['fooled_default']
    _, p_check = fisher_exact([[a, 10 - a], [b, 10 - b]], alternative='two-sided')
    if abs(summ['fisher_exact_two_sided_step150']['same_step150_hyb_vs_sembind_p'] - float(p_check)) > 1e-12:
        mm += 1

    return dict(n_cross_methods=6, n_same_methods=3, n_cases_per_method=10, mismatches=mm)


def make_same_prompt_10_montage():
    d = STAGE6 / 'imprint10'
    same = read_json(d / 'same_prompt_n10.json')
    sz, pad = 192, 24
    # 10 rows x 4 cols: Source hyb WM | Same-Prompt Sibling Cover | SemBind Imprint-150 | Commit-Steer hyb Imprint-150
    canvas = Image.new('RGB', (4 * sz, 10 * sz + pad), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    headers = ['Source WM (hyb)', 'Same-Prompt Sibling', 'SemBind Imprint-150', 'Commit-Steer hyb Imp-150']
    for ci, h in enumerate(headers):
        draw.text((ci * sz + 4, 5), h, fill=(0, 0, 0))

    for i in range(10):
        src_im = Image.open(HYB_S4 / f'images_guided/{i:03d}.png').convert('RGB')
        cov_im = Image.open(SEMBIND_S4 / f'images/{i:03d}_same.png').convert('RGB')
        sb_p = (S5_SEMBIND / f'same_prompt_imprint_151/gidx_{i:03d}_step_150.png') if i < 4 else (d / f'imprint_imgs_same_sembind/gidx_{i:03d}_step_150.png')
        hyb_p = (S5_CS / f'same_prompt_imprint_hyb/gidx_{i:03d}_step_150.png') if i < 4 else (d / f'imprint_imgs_same_hyb/gidx_{i:03d}_step_150.png')
        sb_im = Image.open(sb_p).convert('RGB')
        hyb_im = Image.open(hyb_p).convert('RGB')

        for ci, im in enumerate([src_im, cov_im, sb_im, hyb_im]):
            canvas.paste(im.resize((sz, sz), Image.BICUBIC), (ci * sz, pad + i * sz))

        v_sb = same['sembind_gs_s'][i]['checkpoints']['step_150']
        v_h = same['commit_steer_hyb'][i]['checkpoints']['step_150']
        tag_sb = 'FOOLED' if v_sb['accept_gs_default'] else 'REJ'
        tag_h = 'FOOLED' if v_h['joint_accept'] else 'REJ'

        draw.rectangle([2 * sz, pad + i * sz, 2 * sz + 170, pad + i * sz + 15], fill=(180, 0, 0) if v_sb['accept_gs_default'] else (0, 130, 0))
        draw.text((2 * sz + 3, pad + i * sz + 2), f"#{i:03d} BA={v_sb['ba']:.2f} [{tag_sb}]", fill=(255, 255, 255))

        draw.rectangle([3 * sz, pad + i * sz, 3 * sz + 170, pad + i * sz + 15], fill=(180, 0, 0) if v_h['joint_accept'] else (0, 130, 0))
        draw.text((3 * sz + 3, pad + i * sz + 2), f"#{i:03d} d={v_h['d_blind']:.2f} [{tag_h}]", fill=(255, 255, 255))

    canvas.save(d / 'same_prompt_imprint10_montage.jpg', quality=90)


def main():
    r_6a = audit_coco1k()
    r_6b = audit_imprint10()
    total_mm = r_6a['mismatches'] + r_6b['mismatches']
    out = dict(total_mismatches=total_mm, details=dict(stage6a_coco1k_fid=r_6a, stage6b_imprint10=r_6b))
    write_json(STAGE6 / 'audit_stage6.json', out)
    make_same_prompt_10_montage()
    print(f'[STAGE 6 AUDIT] total_mismatches={total_mm} | 6A={r_6a} | 6B={r_6b}', flush=True)
    if total_mm != 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
