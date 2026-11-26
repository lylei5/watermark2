"""Stage 5 Independent 0-Mismatch Audit & Same-Prompt Instance Forgery Visual Montage.
Verifies EVERY single record, checkpoint, Reprompt row, and summary aggregate across:
  - Stage 5A: SEAL (gs9_30: 300 records, removal20: 240 records, attacks: 12 Imprint + 36 Reprompt + 18 SEAL regen)
  - Stage 5B: Native GS / Commit-then-Steer sscd_64b / hyb (removal20: 720 records, same-prompt Imprint: 12+12 ckpts, same-prompt Reprompt+: 24+24 rows)
  - Stage 5C: SemBind GS-S (same-prompt Imprint: 12 ckpts, same-prompt Reprompt+: 24 rows, removal20: 240 records)
  - Stage 5D: SDXL 1024x1024 Live 151-step Imprint (12 ckpts)
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path('/data0/liyanlei/watermark_research')
S5 = ROOT / 'runs/commit_steer_stage5_20261126'
S4 = ROOT / 'runs/commit_steer_stage4_20261125'
S3 = ROOT / 'runs/commit_steer_paper_scale_20261123'


def rj(p: Path):
    return json.loads(p.read_bytes())


def wj(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(p)


def make_same_prompt_montage():
    sb_dir = S5 / 'sembind_instance_and_removal'
    cs_dir = S5 / 'removal_and_instance_cs'
    if not ((sb_dir / 'instance_forgery_summary.json').exists() and (cs_dir / 'same_prompt_imprint_sscd64_summary.json').exists()):
        print('[S5 VIZ] Skipping same-prompt montage (summaries not yet ready).', flush=True)
        return
    sb_sum = rj(sb_dir / 'instance_forgery_summary.json')
    cs_sum = rj(cs_dir / 'same_prompt_imprint_sscd64_summary.json')
    sz, pad, hdr = 240, 26, 32
    cols = 4
    canvas = Image.new('RGB', (sz * cols, hdr + (sz + pad) * 4), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    titles = [
        'Source WM (Commit-Steer)',
        'Same-Prompt Sibling Cover',
        'SemBind Imprint-150',
        'Commit-Steer Imprint-150',
    ]
    for ci, t in enumerate(titles):
        draw.text((ci * sz + 6, 8), t, fill=(10, 10, 10))
    for ri in range(4):
        y0 = hdr + ri * (sz + pad)
        ims = [
            Image.open(S3 / f'gs100_unified/images_guided/{ri:03d}.png').convert('RGB'),
            Image.open(S4 / f'head2head_sembind/images/{ri:03d}_same.png').convert('RGB'),
            Image.open(sb_dir / f'same_prompt_imprint_151/gidx_{ri:03d}_step_150.png').convert('RGB'),
            Image.open(cs_dir / f'same_prompt_imprint_sscd64/gidx_{ri:03d}_step_150.png').convert('RGB'),
        ]
        for ci, im in enumerate(ims):
            canvas.paste(im.resize((sz, sz), Image.BICUBIC), (ci * sz, y0))
        sb_c = sb_sum['imprint_cases'][ri]['checkpoints']['step_150']
        cs_c = cs_sum['imprint_cases'][ri]['checkpoints']['step_150']
        sb_tag = 'FOOLED' if sb_c['accept_gs_default'] else 'REJ'
        cs_tag = 'FOOLED' if cs_c['joint_accept'] else 'REJ'
        draw.text((6, y0 + sz + 5), f'#{ri:03d} Same-Prompt Case', fill=(20, 20, 20))
        draw.text((2 * sz + 6, y0 + sz + 5), f'SB BA={sb_c["ba"]:.3f} H={sb_c["code_hamming"]} [{sb_tag}]', fill=(180, 0, 0) if sb_c['accept_gs_default'] else (0, 120, 0))
        draw.text((3 * sz + 6, y0 + sz + 5), f'CS auth={cs_c["auth_ba"]:.2f} d={cs_c["d_blind"]:.2f} [{cs_tag}]', fill=(180, 0, 0) if cs_c['joint_accept'] else (0, 120, 0))
    out_p = S5 / 'same_prompt_instance_forgery_montage.jpg'
    canvas.save(out_p, quality=92)
    print(f'[S5 VIZ] Saved {out_p}', flush=True)


def audit_stage5():
    total_mm = 0
    details = {}

    # 1. Stage 5A: SEAL
    seal = S5 / 'seal_head2head'
    if (seal / 'gs9_30_summary.json').exists():
        mm = 0
        recs = rj(seal / 'gs9_30_records.json')
        summ = rj(seal / 'gs9_30_summary.json')
        if len(recs) != 300:
            mm += 1
        for r in recs:
            if (r['matches'] >= 12) != r['accept_12'] or (r['matches'] >= 4) != r['accept_4'] or (r['matches'] >= 2) != r['accept_2']:
                mm += 1
        if sum(r['accept_12'] for r in recs if r['condition'] != 'clean') != summ['matched_30']['gs9_12']:
            mm += 1
        if sum(r['accept_4'] for r in recs if r['condition'] != 'clean') != summ['matched_30']['gs9_4']:
            mm += 1
        if sum(r['accept_2'] for r in recs if r['condition'] != 'clean') != summ['matched_30']['gs9_2']:
            mm += 1
        if (seal / 'removal20_summary.json').exists():
            rrecs = rj(seal / 'removal20_records.json')
            rsumm = rj(seal / 'removal20_summary.json')
            if len(rrecs) != 240:
                mm += 1
            for r in rrecs:
                if (r['matches'] >= 12) != r['accept_12'] or (r['matches'] >= 4) != r['accept_4'] or (r['matches'] >= 2) != r['accept_2']:
                    mm += 1
            if sum(r['accept_12'] for r in rrecs if r['condition'].startswith('waves_')) != rsumm['waves8_pass_12']:
                mm += 1
            if sum(r['accept_12'] for r in rrecs if not r['condition'].startswith('waves_')) != rsumm['regen4_pass_12']:
                mm += 1
        if (seal / 'attacks_summary.json').exists():
            asum = rj(seal / 'attacks_summary.json')
            arows = rj(seal / 'attacks_rows.json')
            for ck in ('step_50', 'step_100', 'step_150'):
                for c in asum['imprint_cases']:
                    v = c['checkpoints'][ck]
                    if (v['matches'] >= 12) != v['accept_12'] or (v['matches'] >= 4) != v['accept_4'] or (v['matches'] >= 2) != v['accept_2']:
                        mm += 1
                if sum(c['checkpoints'][ck]['accept_12'] for c in asum['imprint_cases']) != asum['imprint_summary'][ck]['fooled_12']:
                    mm += 1
            for px in ('sd21', 'sd15'):
                rows = arows['reprompt'][px]['rows']
                if len(rows) != 18 or sum(r['accept_12'] for r in rows) != asum['reprompt_summary'][px]['fooled_12_cand']:
                    mm += 1
            rrows = arows['seal_target_regeneration']
            if len(rrows) != 18 or sum(r['accept_12'] for r in rrows) != asum['seal_target_regeneration_summary']['fooled_12_cand']:
                mm += 1
        total_mm += mm
        details['stage5a_seal'] = dict(n_gs9=len(recs), mismatches=mm)

    # 2. Stage 5B: Removal & Commit-then-Steer Same-Prompt Forgery
    cs = S5 / 'removal_and_instance_cs'
    if (cs / 'removal20_summary.json').exists():
        mm = 0
        proto = rj(cs / 'protocol.json')
        auth_th = proto['auth_threshold']
        th_s, th_h = proto['sscd64_threshold'], proto['hyb_threshold']
        rrecs = rj(cs / 'removal20_records.json')
        rsumm = rj(cs / 'removal20_summary.json')['arms']
        for arm in ('native_gs', 'commit_steer_sscd64', 'commit_steer_hyb'):
            recs = rrecs[arm]
            if len(recs) != 240:
                mm += 1
            for r in recs:
                if arm == 'native_gs':
                    if (r['ba_256'] > 181.0 / 256.0) != r['accept_gs_default']:
                        mm += 1
                else:
                    th = th_s if arm == 'commit_steer_sscd64' else th_h
                    a = r['auth_ba'] > auth_th
                    f = r['d_blind'] <= th
                    if a != r['auth_ok'] or f != r['feat_ok'] or (a and f) != r['joint_accept']:
                        mm += 1
            k_ok = 'accept_gs_default' if arm == 'native_gs' else 'joint_accept'
            if sum(r[k_ok] for r in recs if r['condition'].startswith('waves_')) != rsumm[arm]['waves8_pass']:
                mm += 1
            if sum(r[k_ok] for r in recs if not r['condition'].startswith('waves_')) != rsumm[arm]['regen4_pass']:
                mm += 1
        for arm_tag, th in [('sscd64', th_s), ('hyb', th_h)]:
            sp = cs / f'same_prompt_imprint_{arm_tag}_summary.json'
            if sp.exists():
                s = rj(sp)
                for ck in ('step_50', 'step_100', 'step_150'):
                    for c in s['imprint_cases']:
                        v = c['checkpoints'][ck]
                        a, f = v['auth_ba'] > auth_th, v['d_blind'] <= th
                        if a != v['auth_ok'] or f != v['feat_ok'] or (a and f) != v['joint_accept']:
                            mm += 1
                    if sum(c['checkpoints'][ck]['joint_accept'] for c in s['imprint_cases']) != s['imprint_summary'][ck]['commit_steer_fooled']:
                        mm += 1
        if (cs / 'same_prompt_reprompt_summary.json').exists():
            rps = rj(cs / 'same_prompt_reprompt_summary.json')['summary']
            rpr = rj(cs / 'same_prompt_reprompt_rows.json')
            for arm_tag, th in [('sscd64', th_s), ('hyb', th_h)]:
                for px in ('sd21', 'sd15'):
                    rows = rpr[arm_tag][px]['rows']
                    if len(rows) != 12:
                        mm += 1
                    for r in rows:
                        a, f = r['auth_ba'] > auth_th, r['d_blind'] <= th
                        if a != r['auth_ok'] or f != r['feat_ok'] or (a and f) != r['joint_accept']:
                            mm += 1
                    if sum(r['joint_accept'] for r in rows) != rps[arm_tag][px]['commit_steer_cand']:
                        mm += 1
        total_mm += mm
        details['stage5b_removal_and_instance_cs'] = dict(mismatches=mm)

    # 3. Stage 5C: SemBind Instance Forgery & Removal
    sb = S5 / 'sembind_instance_and_removal'
    if (sb / 'instance_forgery_summary.json').exists():
        mm = 0
        s = rj(sb / 'instance_forgery_summary.json')
        rpr = rj(sb / 'instance_forgery_reprompt_rows.json')
        for ck in ('step_50', 'step_100', 'step_150'):
            for c in s['imprint_cases']:
                v = c['checkpoints'][ck]
                if (v['ba'] > 181.0 / 256.0) != v['accept_gs_default'] or (v['ba'] >= 167.0 / 256.0) != v['accept_gs_167']:
                    mm += 1
            if sum(c['checkpoints'][ck]['accept_gs_default'] for c in s['imprint_cases']) != s['imprint_summary'][ck]['sembind_fooled_default']:
                mm += 1
        for px in ('sd21', 'sd15'):
            rows = rpr[px]['rows']
            if len(rows) != 12:
                mm += 1
            for r in rows:
                if (r['ba'] > 181.0 / 256.0) != r['accept_gs_default']:
                    mm += 1
            if sum(r['accept_gs_default'] for r in rows) != s['reprompt_plus_summary'][px]['sembind_cand_default']:
                mm += 1
        if (sb / 'removal20_summary.json').exists():
            rrecs = rj(sb / 'removal20_records.json')
            rsumm = rj(sb / 'removal20_summary.json')
            if len(rrecs) != 240:
                mm += 1
            for r in rrecs:
                if (r['ba'] > 181.0 / 256.0) != r['accept_gs_default'] or (r['ba'] >= 167.0 / 256.0) != r['accept_gs_167']:
                    mm += 1
            if sum(r['accept_gs_default'] for r in rrecs if r['condition'].startswith('waves_')) != rsumm['waves8_pass_default']:
                mm += 1
            if sum(r['accept_gs_default'] for r in rrecs if not r['condition'].startswith('waves_')) != rsumm['regen4_pass_default']:
                mm += 1
        total_mm += mm
        details['stage5c_sembind'] = dict(mismatches=mm)

    # 4. Stage 5D: SDXL Imprint
    sdxl = S5 / 'sdxl_imprint'
    if (sdxl / 'summary.json').exists():
        mm = 0
        proto = rj(sdxl / 'protocol.json')
        s = rj(sdxl / 'summary.json')
        for ck in ('step_50', 'step_100', 'step_150'):
            for c in s['imprint_cases']:
                v = c['checkpoints'][ck]
                a, f = v['auth_ba'] > proto['auth_threshold'], v['d_blind'] <= proto['threshold']
                if a != v['auth_ok'] or f != v['feat_ok'] or (a and f) != v['joint_accept']:
                    mm += 1
            if sum(c['checkpoints'][ck]['joint_accept'] for c in s['imprint_cases']) != s['imprint_summary'][ck]['commit_steer_sdxl_fooled']:
                mm += 1
            if sum(c['checkpoints'][ck]['auth_ok'] for c in s['imprint_cases']) != s['imprint_summary'][ck]['native_gs_sdxl_fooled']:
                mm += 1
        total_mm += mm
        details['stage5d_sdxl_imprint'] = dict(n_cases=len(s['imprint_cases']), mismatches=mm)

    out = dict(total_mismatches=total_mm, details=details)
    wj(S5 / 'audit_stage5.json', out)
    print(f'[AUDIT STAGE 5 COMPLETE] total_mismatches={total_mm} | details={details}', flush=True)
    return out


if __name__ == '__main__':
    make_same_prompt_montage()
    audit_stage5()
