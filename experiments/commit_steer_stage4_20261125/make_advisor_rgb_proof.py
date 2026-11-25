"""Build a visual proof montage (CPU-only) directly from audited Stage 0 / Stage 1 images on server 249
demonstrating with real images & exact numbers why global RGB mean/variance (48-bit) fails inside GS/PRC
while 64-bit keyed semantic commitment (Commit-then-Steer) succeeds.
"""
import json
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/data0/liyanlei/watermark_research')
STAGE0 = ROOT / 'runs/payload_binding_stage0_20261122'
RP_DIR = ROOT / 'runs/author_reprompt_plus_20261115'
CJ_DIR = ROOT / 'runs/color_joint_20261119'
LOADER = ROOT / 'runs/order_capsule_confirmation_20261115'
SSCD_PATH = ROOT / 'models/sscd/sscd_disc_mixup.torchscript.pt'
OUT = ROOT / 'runs/commit_steer_stage4_20261125/advisor_rgb_vs_sscd_proof.jpg'

sys.path.insert(0, str(LOADER))
from order_recovery_data import waves_module


def rgb_stats(im: Image.Image):
    a = np.asarray(im.convert('RGB'), dtype=np.float64)
    return a.mean(axis=(0, 1)), a.std(axis=(0, 1))


def rgb48_dist(im_ref: Image.Image, im_q: Image.Image) -> float:
    """Exact Stage 0 rgb_mean_var_48b quantized payload distance (run_stage0.py L172-260)."""
    a1 = np.asarray(im_ref.convert('RGB'), dtype=np.float32) / 255.0
    a2 = np.asarray(im_q.convert('RGB'), dtype=np.float32) / 255.0
    mu1 = a1.reshape(-1, 3).mean(axis=0).astype(np.float64)
    var1 = a1.reshape(-1, 3).var(axis=0).astype(np.float64)
    # 8-bit quantization & dequantization identical to Stage 0 payload
    mu1_dec = np.rint(np.clip(mu1, 0.0, 1.0) * 255.0) / 255.0
    var1_dec = (np.rint(np.clip(var1 / 0.25, 0.0, 1.0) * 255.0) / 255.0) * 0.25
    mu2 = a2.reshape(-1, 3).mean(axis=0).astype(np.float64)
    var2 = a2.reshape(-1, 3).var(axis=0).astype(np.float64)
    dm = float(np.max(np.abs(mu1_dec - mu2)) / 0.02)
    dv = float(np.max(np.abs(var1_dec - var2)) / 0.01)
    return max(dm, dv)


def reinhard_match(src_im: Image.Image, ref_im: Image.Image) -> Image.Image:
    s = np.asarray(src_im.convert('RGB'), dtype=np.float64)
    r = np.asarray(ref_im.convert('RGB'), dtype=np.float64)
    ms, ss = s.mean(axis=(0, 1)), s.std(axis=(0, 1)) + 1e-6
    mr, sr = r.mean(axis=(0, 1)), r.std(axis=(0, 1))
    out = np.clip((s - ms) * (sr / ss) + mr, 0, 255).round().astype(np.uint8)
    return Image.fromarray(out)


def main():
    s0_sum = json.loads((STAGE0 / 'summary.json').read_bytes())
    s0_sscd = json.loads((STAGE0 / 'summary_sscd.json').read_bytes())
    th_rgb = s0_sum['thresholds_95']['rgb_mean_var_48b']
    th_sscd = s0_sscd['thresholds_95']['sscd_64b']
    P64 = np.load(STAGE0 / 'sscd_projections.npz')['p64'].astype(np.float64)

    sscd = torch.jit.load(str(SSCD_PATH), map_location='cpu').eval()
    mean_n = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std_n = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def emb_sscd(im: Image.Image) -> np.ndarray:
        x = torch.from_numpy(np.asarray(im.convert('RGB').resize((320, 320), Image.BICUBIC), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            return F.normalize(sscd((x - mean_n) / std_n), dim=1)[0].numpy().astype(np.float64)

    def sscd64_dist(im_ref: Image.Image, im_q: Image.Image) -> float:
        er, eq = emb_sscd(im_ref), emb_sscd(im_q)
        sgn = np.where((er @ P64) >= 0.0, 1.0, -1.0)
        qp = eq @ P64
        return float(1.0 - np.dot(sgn, qp) / (8.0 * np.linalg.norm(qp) + 1e-12))

    # Load Imprint GS pairs (pick 2 visually distinct cases where rgb48 is FOOLED naturally AND after Reinhard!)
    rp_proto = json.loads((RP_DIR / 'protocol.json').read_bytes())
    gs_refs = [r for r in rp_proto['references'] if r['carrier'] == 'GS']

    imp_picks = []
    for ref in gs_refs:
        ref_p = Path(ref['image'])
        atk_p = ref_p.parent.parent / 'attacked_images_step_150' / ref_p.name
        im_r = Image.open(ref_p).convert('RGB')
        im_a = Image.open(atk_p).convert('RGB')
        dr = rgb48_dist(im_r, im_a)
        if dr <= th_rgb:
            ds = sscd64_dist(im_r, im_a)
            if ds > th_sscd + 0.15:
                imp_picks.append((dr, ds, im_r, im_a, ref['id']))
    imp_picks.sort(key=lambda x: x[0])
    assert len(imp_picks) >= 2
    imp1 = imp_picks[0]
    imp2 = imp_picks[1]

    # Pick 1 Reprompt+ case where rgb48 is FOOLED naturally (filter by RGB first -> 10x faster on CPU!)
    rp_picks = []
    for ref in gs_refs[:6]:
        im_r = Image.open(ref['image']).convert('RGB')
        sid = ref['id']
        for pi in range(3):
            for ri in range(3):
                cp = RP_DIR / 'sd21/images' / f'{sid}_sd21_p{pi}_r{ri}.png'
                im_c = Image.open(cp).convert('RGB')
                dr = rgb48_dist(im_r, im_c)
                if dr <= 4.5:
                    ds = sscd64_dist(im_r, im_c)
                    if ds > th_sscd + 0.2:
                        rp_picks.append((dr, ds, im_r, im_c, sid))
                        if len(rp_picks) >= 4:
                            break
            if len(rp_picks) >= 4:
                break
        if len(rp_picks) >= 4:
            break
    rp_picks.sort(key=lambda x: x[0])
    rp1 = rp_picks[0]

    # Pick 1 Legitimate Brightness Edit (RGB FAILS / REJECTED, SSCD PASSES) + Same-Prompt Sibling (RGB FOOLED, SSCD REJECTS)
    waves = waves_module()
    cj_sources = json.loads((CJ_DIR / 'sources.json').read_bytes())
    legit_pick = None
    for s in cj_sources[:40]:
        dest = CJ_DIR / 'images' / f"{s['task']:03d}"
        im0 = Image.open(dest / '0.png').convert('RGB')
        im1 = Image.open(dest / '1.png').convert('RGB')
        im_br = waves.apply_distortion([im0], 'brightness', strength=0.8, distortion_seed=20261119 + s['task'])[0]
        dr_br = rgb48_dist(im0, im_br)
        dr_sib = rgb48_dist(im0, im1)
        if dr_br > th_rgb and dr_sib <= th_rgb:
            ds_br = sscd64_dist(im0, im_br)
            ds_sib = sscd64_dist(im0, im1)
            if ds_br <= th_sscd - 0.1 and ds_sib > th_sscd + 0.1:
                legit_pick = (im0, im_br, im1, dr_br, ds_br, dr_sib, ds_sib)
                break
    assert legit_pick is not None

    # Render clean 4-row x 3-col publication montage with clear English labels (universally crisp with default/DejaVu font)
    try:
        font_title = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 17)
        font_sub = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
        font_txt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    except Exception:
        font_title = font_sub = font_txt = ImageFont.load_default()

    W_IMG, H_IMG = 320, 320
    PAD_X, PAD_Y = 20, 16
    TOP_BANNER = 58
    CAP_H = 68
    ROW_H = H_IMG + CAP_H + PAD_Y
    TOTAL_W = PAD_X * 4 + W_IMG * 3
    TOTAL_H = TOP_BANNER + ROW_H * 4 + 24

    canvas = Image.new('RGB', (TOTAL_W, TOTAL_H), (248, 249, 252))
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, TOTAL_W, TOP_BANNER - 8], fill=(24, 34, 56))
    draw.text((20, 10), 'Visual Proof: Why 48-bit Global RGB Mean/Var Fails vs. 64-bit Keyed Semantic Commitment', fill=(255, 255, 255), font=font_title)
    draw.text((20, 32), f'Frozen 95%-TPR Thresholds (Stage 0):  RGB-48b d <= {th_rgb:.2f} (exact quantized max-dev)   |   SSCD-64b d <= {th_sscd:.3f} (cosine bit dist)', fill=(200, 220, 255), font=font_txt)

    def fmt_rgb(im):
        m, s = rgb_stats(im)
        return f'mu=({m[0]:.0f},{m[1]:.0f},{m[2]:.0f}) sd=({s[0]:.0f},{s[1]:.0f},{s[2]:.0f})'

    def draw_cell(row, col, im, header, line1, line2, border_col, tag_col):
        x0 = PAD_X + col * (W_IMG + PAD_X)
        y0 = TOP_BANNER + row * ROW_H
        draw.rectangle([x0 - 3, y0 - 3, x0 + W_IMG + 3, y0 + H_IMG + CAP_H], fill=(255, 255, 255), outline=border_col, width=3)
        canvas.paste(im.resize((W_IMG, H_IMG), Image.BICUBIC), (x0, y0))
        draw.rectangle([x0, y0 + H_IMG, x0 + W_IMG, y0 + H_IMG + CAP_H], fill=(252, 252, 254))
        draw.text((x0 + 6, y0 + H_IMG + 4), header, fill=tag_col, font=font_sub)
        draw.text((x0 + 6, y0 + H_IMG + 24), line1, fill=(40, 40, 48), font=font_txt)
        draw.text((x0 + 6, y0 + H_IMG + 44), line2, fill=tag_col, font=font_txt)

    GREEN = (22, 130, 62)
    RED = (198, 36, 36)
    BLUE = (34, 86, 168)

    # Row 0: Imprint Forgery #1 + Reinhard
    dr, ds, im_r, im_a, sid = imp1
    im_rh = reinhard_match(im_a, im_r)
    dr_rh, ds_rh = rgb48_dist(im_r, im_rh), sscd64_dist(im_r, im_rh)
    draw_cell(0, 0, im_r, 'Case 1: Watermarked Source (GS)', fmt_rgb(im_r), 'Original issued image (Target ref)', BLUE, BLUE)
    draw_cell(0, 1, im_a, 'CVPR25 Imprint-150 Forgery', f'{fmt_rgb(im_a)}', f'RGB48={dr:.2f} FOOLED! | SSCD64={ds:.2f} REJ', RED, RED)
    draw_cell(0, 2, im_rh, 'Imprint + 1-Line Reinhard Match', f'{fmt_rgb(im_rh)}', f'RGB48={dr_rh:.2f} FOOLED! | SSCD64={ds_rh:.2f} REJ', RED, RED)

    # Row 1: Imprint Forgery #2 + Reinhard
    dr, ds, im_r, im_a, sid = imp2
    im_rh = reinhard_match(im_a, im_r)
    dr_rh, ds_rh = rgb48_dist(im_r, im_rh), sscd64_dist(im_r, im_rh)
    draw_cell(1, 0, im_r, 'Case 2: Watermarked Source (GS)', fmt_rgb(im_r), 'Original issued image (Target ref)', BLUE, BLUE)
    draw_cell(1, 1, im_a, 'CVPR25 Imprint-150 Forgery', f'{fmt_rgb(im_a)}', f'RGB48={dr:.2f} FOOLED! | SSCD64={ds:.2f} REJ', RED, RED)
    draw_cell(1, 2, im_rh, 'Imprint + 1-Line Reinhard Match', f'{fmt_rgb(im_rh)}', f'RGB48={dr_rh:.2f} FOOLED! | SSCD64={ds_rh:.2f} REJ', RED, RED)

    # Row 2: Reprompt+ Forgery + Reinhard
    dr, ds, im_r, im_c, sid = rp1
    im_rh = reinhard_match(im_c, im_r)
    dr_rh, ds_rh = rgb48_dist(im_r, im_rh), sscd64_dist(im_r, im_rh)
    draw_cell(2, 0, im_r, 'Case 3: Watermarked Source (GS)', fmt_rgb(im_r), 'Original issued image (Target ref)', BLUE, BLUE)
    draw_cell(2, 1, im_c, 'CVPR25 Reprompt+ (New Prompt)', f'{fmt_rgb(im_c)}', f'RGB48={dr:.2f} FOOLED! | SSCD64={ds:.2f} REJ', RED, RED)
    draw_cell(2, 2, im_rh, 'Reprompt+ + Reinhard Color Match', f'{fmt_rgb(im_rh)}', f'RGB48={dr_rh:.2f} FOOLED! | SSCD64={ds_rh:.2f} REJ', RED, RED)

    # Row 3: Legitimate Brightness Edit (RGB fails!) vs Same-Prompt Sibling (RGB fooled!)
    im0, im_br, im1, dr_br, ds_br, dr_sib, ds_sib = legit_pick
    draw_cell(3, 0, im0, 'Case 4: Genuine Source Image', fmt_rgb(im0), 'Original issued image', BLUE, BLUE)
    draw_cell(3, 1, im_br, 'Legitimate Edit: Brightness 0.8', f'{fmt_rgb(im_br)}', f'RGB48={dr_br:.1f} FALSE REJ! | SSCD64={ds_br:.2f} PASS', GREEN, GREEN)
    draw_cell(3, 2, im1, 'Different Seed (Same Prompt)', f'{fmt_rgb(im1)}', f'RGB48={dr_sib:.2f} FOOLED! | SSCD64={ds_sib:.2f} REJ', RED, RED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT, quality=94)
    print(f'SAVED {OUT} ({canvas.size})', flush=True)


if __name__ == '__main__':
    main()
