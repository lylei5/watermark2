"""Stage 0B: Compact differentiable perceptual-hash payloads (SSCD 48/64/96-bit + 512f upper bound)
on the exact same frozen 40-cal / 60-confirm WAVES 16-edit split, 30 Imprint pairs,
240 Reprompt+ candidates, 20 MagicBrush object swaps, 8 CVPR color transfers, and Reinhard diagnostics.
CPU-only on server 249.
"""
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path('/data0/liyanlei/watermark_research')
RUN = ROOT / 'runs/payload_binding_stage0_20261122'
LOADER = ROOT / 'runs/order_capsule_confirmation_20261115'
SSCD_PATH = ROOT / 'models/sscd/sscd_disc_mixup.torchscript.pt'

METHODS = ['sscd_48b', 'sscd_64b', 'sscd_96b', 'sscd_512f']
BIT_BUDGETS = {'sscd_48b': 48, 'sscd_64b': 64, 'sscd_96b': 96, 'sscd_512f': 16384}

TFM = transforms.Compose([
    transforms.Resize([320, 320]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def read_json(p: Path):
    return json.loads(p.read_bytes())


def write_json(p: Path, obj):
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(p)


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def reinhard_align(src_im: Image.Image, ref_im: Image.Image) -> Image.Image:
    src = np.asarray(src_im.convert('RGB'), dtype=np.float64) / 255.0
    ref = np.asarray(ref_im.convert('RGB'), dtype=np.float64) / 255.0
    out = np.empty_like(src)
    for c in range(3):
        sm, ss = src[:, :, c].mean(), src[:, :, c].std()
        rm, rs = ref[:, :, c].mean(), ref[:, :, c].std()
        out[:, :, c] = (src[:, :, c] - sm) / (ss + 1e-6) * rs + rm
    return Image.fromarray(np.rint(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8))


def _worker_waves(args):
    run_dir, sources, families, strengths, seed_base, n_threads = args
    torch.set_num_threads(n_threads)
    model = torch.jit.load(str(SSCD_PATH), map_location='cpu').eval()
    sys.path.insert(0, str(LOADER))
    from order_recovery_data import waves_module
    waves = waves_module()

    def infer_batch(pil_list):
        xs = torch.stack([TFM(im) for im in pil_list])
        with torch.no_grad():
            y = F.normalize(model(xs), dim=1).numpy().astype(np.float32)
        return y

    out_records = []
    for s in sources:
        dest = Path(run_dir) / 'images' / f"{s['task']:03d}"
        receipt = read_json(dest / 'receipt.json')
        imgs = []
        for i, expected in enumerate(receipt['sha256']):
            p = dest / f'{i}.png'
            assert sha_file(p) == expected
            imgs.append(Image.open(p).convert('RGB'))
        ref_embs = infer_batch(imgs)  # (2, 512)

        q_imgs = []
        q_meta = []
        for seed_idx in (0, 1):
            im = imgs[seed_idx]
            for fi, family in enumerate(['clean'] + families):
                for ji, strength in enumerate(strengths if fi else [0.0]):
                    seed = seed_base + 100 * s['task'] + 2 * fi + ji
                    q_im = waves.apply_distortion([im], family, strength=strength, distortion_seed=seed)[0] if fi else im
                    q_imgs.append(q_im)
                    q_meta.append((s['dataset'], s['task'], seed_idx, family, strength))
        # Batch inference in chunks of 34 (one prompt = 34 queries)
        q_embs = infer_batch(q_imgs)  # (34, 512)
        for (dataset, task, seed_idx, family, strength), qe in zip(q_meta, q_embs):
            out_records.append((dataset, task, seed_idx, family, strength, ref_embs[seed_idx], ref_embs[1 - seed_idx], qe))
    return out_records


def infer_pil_list(model, pil_list, batch_size=32):
    out = []
    for i in range(0, len(pil_list), batch_size):
        xs = torch.stack([TFM(im) for im in pil_list[i:i+batch_size]])
        with torch.no_grad():
            y = F.normalize(model(xs), dim=1).numpy().astype(np.float32)
        out.append(y)
    return np.concatenate(out, axis=0)


def make_projections():
    rng = np.random.default_rng(2026112201)
    projs = {}
    for k in (48, 64, 96):
        # Random orthogonal rows (QR) for lower quantization variance
        g = rng.standard_normal((512, k))
        q, _ = np.linalg.qr(g)
        projs[k] = q.astype(np.float64)  # (512, k)
    return projs


def pack_unpack_signs(emb: np.ndarray, proj: np.ndarray) -> np.ndarray:
    k = proj.shape[1]
    bits = ((emb.astype(np.float64) @ proj) >= 0.0).astype(np.uint8)
    packed = np.packbits(bits).tobytes()
    assert len(packed) * 8 == k
    unpacked = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))[:k]
    assert np.array_equal(bits, unpacked)
    return 2.0 * unpacked.astype(np.float64) - 1.0  # in {-1, +1}^k


def score_emb_pair(ref_emb: np.ndarray, q_emb: np.ndarray, projs: dict) -> dict:
    out = {}
    out['sscd_512f'] = float(1.0 - np.dot(ref_emb.astype(np.float64), q_emb.astype(np.float64)))
    for k in (48, 64, 96):
        b = pack_unpack_signs(ref_emb, projs[k])
        qp = q_emb.astype(np.float64) @ projs[k]
        denom = np.sqrt(k) * float(np.linalg.norm(qp)) + 1e-12
        out[f'sscd_{k}b'] = float(1.0 - np.dot(b, qp) / denom)
    return out


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
    cal_sources = [s for s in cb_sources if s['split'] == 'calibration']
    assert len(cal_sources) == 40 and len(cj_sources) == 60

    projs = make_projections()
    np.savez(RUN / 'sscd_projections.npz', p48=projs[48], p64=projs[64], p96=projs[96])

    print('Computing SSCD embeddings on 40 cal + 60 confirm prompts (4 workers x 8 threads)...', flush=True)
    cal_chunks = [cal_sources[i::4] for i in range(4)]
    conf_chunks = [cj_sources[i::4] for i in range(4)]
    with ProcessPoolExecutor(max_workers=4) as ex:
        cal_res = list(ex.map(_worker_waves, [(str(cb_run), ch, families, strengths, 20261119, 8) for ch in cal_chunks]))
        conf_res = list(ex.map(_worker_waves, [(str(cj_run), ch, families, strengths, 20261119, 8) for ch in conf_chunks]))

    cal_recs = [r for part in cal_res for r in part]
    conf_recs = [r for part in conf_res for r in part]
    assert len(cal_recs) == 40 * 34 and len(conf_recs) == 60 * 34

    def score_waves(recs):
        rows = []
        for dataset, task, seed_idx, family, strength, own_e, sib_e, q_e in recs:
            rows.append(dict(
                dataset=dataset, task=task, seed_idx=seed_idx, family=family, strength=strength,
                own=score_emb_pair(own_e, q_e, projs),
                sib=score_emb_pair(sib_e, q_e, projs),
            ))
        return rows

    cal_rows = score_waves(cal_recs)
    conf_rows = score_waves(conf_recs)

    cal_edited = [r for r in cal_rows if r['family'] != 'clean']
    thresholds = {}
    for m in METHODS:
        pos = np.sort([r['own'][m] for r in cal_edited])
        thresholds[m] = float(pos[int(np.ceil(0.95 * len(pos))) - 1])

    # Summarize confirm_60
    conf_edited = [r for r in conf_rows if r['family'] != 'clean']
    conf_clean = [r for r in conf_rows if r['family'] == 'clean']
    conf_summary = {}
    for m in METHODS:
        th = thresholds[m]
        tpr = sum(r['own'][m] <= th for r in conf_edited)
        far = sum(r['sib'][m] <= th for r in conf_edited)
        cfar = sum(r['sib'][m] <= th for r in conf_clean)
        pos = np.sort([r['own'][m] for r in conf_edited])
        rth = float(pos[int(np.ceil(0.95 * len(pos))) - 1])
        rfar = sum(r['sib'][m] <= rth for r in conf_edited)
        by_edit = {}
        for fam in families:
            for st in strengths:
                sub = [r for r in conf_edited if r['family'] == fam and r['strength'] == st]
                by_edit[f'{fam}_{st}'] = dict(
                    n=len(sub),
                    tpr=sum(r['own'][m] <= th for r in sub),
                    far=sum(r['sib'][m] <= th for r in sub),
                )
        conf_summary[m] = dict(
            bits=BIT_BUDGETS[m],
            n_edited=len(conf_edited),
            tpr_count=tpr, tpr=tpr / len(conf_edited),
            sib_far_count=far, sib_far=far / len(conf_edited),
            clean_sib_far_count=cfar, clean_sib_far=cfar / len(conf_clean),
            roc95_sib_far_count=rfar, roc95_sib_far=rfar / len(conf_edited),
            by_edit=by_edit,
        )

    # Evaluate Imprint, Reprompt+, External on main process with 16 threads
    print('Computing SSCD embeddings on Imprint (30), Reprompt+ (240), External (29) + Reinhard diag...', flush=True)
    torch.set_num_threads(16)
    model = torch.jit.load(str(SSCD_PATH), map_location='cpu').eval()

    # Imprint
    rp_proto = read_json(ROOT / 'runs/author_reprompt_plus_20261115/protocol.json')
    imp_meta, imp_refs, imp_atks, imp_reis = [], [], [], []
    for ref in rp_proto['references']:
        if ref['carrier'] == 'GS':
            ref_p = Path(ref['image'])
            atk_p = ref_p.parent.parent / 'attacked_images_step_150' / ref_p.name
            ri, ai = Image.open(ref_p).convert('RGB'), Image.open(atk_p).convert('RGB')
            imp_meta.append(('GS', ref['id']))
            imp_refs.append(ri); imp_atks.append(ai); imp_reis.append(reinhard_align(ai, ri))
    for carrier in ('TR', 'PRC'):
        base = ROOT / f'runs/pgid_carrier_coverage_20261113/{carrier}/attack=forgery'
        for i in range(10):
            ri = Image.open(base / 'watermarked_images' / f'image_{i:03d}.png').convert('RGB')
            ai = Image.open(base / 'attacked_images_step_150' / f'image_{i:03d}.png').convert('RGB')
            imp_meta.append((carrier, f'{carrier}_{i:03d}'))
            imp_refs.append(ri); imp_atks.append(ai); imp_reis.append(reinhard_align(ai, ri))
    imp_ref_e = infer_pil_list(model, imp_refs)
    imp_atk_e = infer_pil_list(model, imp_atks)
    imp_rei_e = infer_pil_list(model, imp_reis)
    imp_rows = []
    for (c, sid), re, ae, rre in zip(imp_meta, imp_ref_e, imp_atk_e, imp_rei_e):
        imp_rows.append(dict(carrier=c, id=sid, scores=score_emb_pair(re, ae, projs), reinhard=score_emb_pair(re, rre, projs)))

    imp_summary = {}
    for c in ('GS', 'TR', 'PRC', 'ALL'):
        sub = imp_rows if c == 'ALL' else [r for r in imp_rows if r['carrier'] == c]
        imp_summary[c] = {
            m: dict(
                n=len(sub),
                accept=sum(r['scores'][m] <= thresholds[m] for r in sub),
                reinhard_diag_accept=sum(r['reinhard'][m] <= thresholds[m] for r in sub),
            )
            for m in METHODS
        }

    # Reprompt+
    rp_dir = ROOT / 'runs/author_reprompt_plus_20261115'
    ref_emb_map = {}
    ref_im_map = {}
    rp_ref_pils, rp_ref_ids = [], []
    for ref in rp_proto['references']:
        im = Image.open(ref['image']).convert('RGB')
        ref_im_map[ref['id']] = im
        rp_ref_ids.append(ref['id']); rp_ref_pils.append(im)
    rp_ref_embs = infer_pil_list(model, rp_ref_pils)
    for sid, e in zip(rp_ref_ids, rp_ref_embs):
        ref_emb_map[sid] = e

    rp_meta, rp_cands, rp_reis = [], [], []
    for ref in rp_proto['references']:
        carrier, sid = ref['carrier'], ref['id']
        n_res = 3 if carrier == 'GS' else 1
        for proxy in ('sd21', 'sd15'):
            for pi in range(3):
                for ri in range(n_res):
                    cim = Image.open(rp_dir / proxy / 'images' / f'{sid}_{proxy}_p{pi}_r{ri}.png').convert('RGB')
                    rp_meta.append((carrier, sid, proxy, pi, ri))
                    rp_cands.append(cim)
                    rp_reis.append(reinhard_align(cim, ref_im_map[sid]))
    rp_cand_e = infer_pil_list(model, rp_cands)
    rp_rei_e = infer_pil_list(model, rp_reis)
    rp_rows = []
    for (c, sid, proxy, pi, ri), ce, rre in zip(rp_meta, rp_cand_e, rp_rei_e):
        rp_rows.append(dict(
            carrier=c, id=sid, proxy=proxy,
            scores=score_emb_pair(ref_emb_map[sid], ce, projs),
            reinhard=score_emb_pair(ref_emb_map[sid], rre, projs),
        ))

    rp_summary = {}
    for c in ('GS', 'TR'):
        for proxy in ('sd21', 'sd15'):
            sub = [r for r in rp_rows if r['carrier'] == c and r['proxy'] == proxy]
            sids = sorted({r['id'] for r in sub})
            k = f'{c}_{proxy}'
            rp_summary[k] = {}
            for m in METHODS:
                th = thresholds[m]
                rp_summary[k][m] = dict(
                    n_candidates=len(sub), n_sources=len(sids),
                    cand_accept=sum(r['scores'][m] <= th for r in sub),
                    source_accept=sum(any(r['scores'][m] <= th for r in sub if r['id'] == sid) for sid in sids),
                    reinhard_diag_cand_accept=sum(r['reinhard'][m] <= th for r in sub),
                    reinhard_diag_source_accept=sum(any(r['reinhard'][m] <= th for r in sub if r['id'] == sid) for sid in sids),
                )

    # External (MagicBrush 20 + Color Transfer 8 + ICCV 1)
    p2_dir = ROOT / 'runs/rgb_moments_phase2_20261120/author'
    p2_man = read_json(ROOT / 'runs/rgb_moments_phase2_20261120/public_manifest.json')
    p3_dir = ROOT / 'runs/local_semantics_phase3_20261120/additional'
    p3_man = read_json(ROOT / 'runs/local_semantics_phase3_20261120/additional_manifest.json')
    ext_meta, ext_refs, ext_outs = [], [], []
    for item in p2_man:
        ext_meta.append((item['group'], item['item']))
        ext_refs.append(Image.open(p2_dir / item['files']['ref']).convert('RGB'))
        ext_outs.append(Image.open(p2_dir / item['files']['output']).convert('RGB'))
    for item in p3_man:
        ext_meta.append(('replacement_extra', item['item']))
        ext_refs.append(Image.open(p3_dir / item['files']['ref']).convert('RGB'))
        ext_outs.append(Image.open(p3_dir / item['files']['output']).convert('RGB'))
    ext_re = infer_pil_list(model, ext_refs)
    ext_oe = infer_pil_list(model, ext_outs)
    ext_rows = [dict(group=g, item=it, scores=score_emb_pair(re, oe, projs)) for (g, it), re, oe in zip(ext_meta, ext_re, ext_oe)]
    ext_summary = {}
    for gname, gfilter in [
        ('iccv_dist_match', lambda r: r['group'] == 'distribution'),
        ('cvpr_color_transfer', lambda r: r['group'] == 'color'),
        ('magicbrush_all20', lambda r: r['group'] in ('replacement', 'replacement_extra')),
    ]:
        sub = [r for r in ext_rows if gfilter(r)]
        ext_summary[gname] = {m: dict(n=len(sub), accept=sum(r['scores'][m] <= thresholds[m] for r in sub)) for m in METHODS}

    elapsed = time.perf_counter() - t0
    out = dict(
        elapsed_seconds=elapsed,
        thresholds_95=thresholds,
        confirm_60=conf_summary,
        imprint=imp_summary,
        reprompt_plus=rp_summary,
        external=ext_summary,
    )
    write_json(RUN / 'summary_sscd.json', out)
    print(f'SSCD Stage 0B DONE in {elapsed:.1f}s', flush=True)


if __name__ == '__main__':
    main()
