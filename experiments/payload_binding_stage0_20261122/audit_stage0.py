import csv, gzip, json
import numpy as np
from pathlib import Path
from PIL import Image
import sys
sys.path.insert(0, '/data0/liyanlei/watermark_research/runs/payload_binding_stage0_20261122')
import run_stage0 as s0

RUN = Path('/data0/liyanlei/watermark_research/runs/payload_binding_stage0_20261122')
cal = json.loads((RUN/'calibration.json').read_text())
summ = json.loads((RUN/'summary.json').read_text())

# 1. Verify bit-packing roundtrip on edge images (black, white, random)
for name, arr in [
    ('black', np.zeros((512, 512, 3), dtype=np.uint8)),
    ('white', np.full((512, 512, 3), 255, dtype=np.uint8)),
    ('rand', np.random.default_rng(42).integers(0, 256, (512, 512, 3), dtype=np.uint8)),
]:
    feat = s0.extract_continuous(Image.fromarray(arr))
    enc = s0.encode_payloads(feat)
    dec = s0.decode_payloads(enc)
    for m, bits in s0.BIT_BUDGETS.items():
        assert len(enc[m]) * 8 == bits, (m, len(enc[m]), bits)
    assert all(np.isfinite(v).all() for v in feat.values())

# 2. Independently recompute thresholds & test/confirm counts from waves_scores.csv.gz
by_split_m = {}
with gzip.open(RUN/'waves_scores.csv.gz', 'rt') as f:
    for r in csv.DictReader(f):
        key = (r['split'], r['method'])
        by_split_m.setdefault(key, []).append((r['family'], float(r['own']), float(r['sib'])))

mismatches = 0
for m in s0.METHODS:
    cal_ed = [own for fam, own, _ in by_split_m[('cal', m)] if fam != 'clean']
    th_re = float(np.sort(cal_ed)[int(np.ceil(0.95 * len(cal_ed))) - 1])
    if abs(th_re - cal['thresholds_95'][m]) > 1e-7:
        mismatches += 1
        print('TH MISMATCH', m, th_re, cal['thresholds_95'][m])
    for sp, skey in [('test', 'test_60'), ('confirm', 'confirm_60')]:
        ed = [(own, sib) for fam, own, sib in by_split_m[(sp, m)] if fam != 'clean']
        tpr_re = sum(own <= cal['thresholds_95'][m] for own, _ in ed)
        far_re = sum(sib <= cal['thresholds_95'][m] for _, sib in ed)
        if tpr_re != summ[skey][m]['tpr_count'] or far_re != summ[skey][m]['sib_far_count']:
            mismatches += 1
            print('COUNT MISMATCH', sp, m, tpr_re, summ[skey][m]['tpr_count'], far_re, summ[skey][m]['sib_far_count'])

audit = dict(status='ok' if mismatches == 0 else 'failed', mismatches=mismatches, verified_methods=s0.METHODS, bit_budgets=s0.BIT_BUDGETS)
(RUN/'audit_stage0a.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit))
