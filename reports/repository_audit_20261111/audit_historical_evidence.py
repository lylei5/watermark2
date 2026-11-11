"""Audit existing research artifacts on server 249; emit no watermark bitstrings."""
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path('/data0/liyanlei/watermark_research')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

out = {'audit_date': '2026-11-11', 'execution_host': '249', 'threshold': 181 / 256,
       'operator': '>', 'pgid': {}}
for name in ['pgid_smoke_20261102T231752Z', 'repro_pgid_n5_20261103']:
    folder = ROOT / 'runs' / name / 'attack=forgery'
    raw = folder / 'all_results.csv'
    summary = folder / 'summary.csv'
    rows = list(csv.DictReader(raw.open()))
    groups = defaultdict(list)
    safe = []
    for row in rows:
        initial, recovered = row['message_bits_str_initial'], row['message_bits_str_recovered']
        assert len(initial) == len(recovered) == 256
        assert set(initial + recovered) <= {'0', '1'}
        ba = sum(a == b for a, b in zip(initial, recovered)) / 256
        accepted = ba > 181 / 256
        assert ba == float(row['bit_accuracy'])
        assert row['detection_successful'] in ('True', 'False')
        assert accepted == (row['detection_successful'] == 'True')
        item = {'image_idx': int(row['image_idx']), 'stage': row['stage'],
                'attack_step': int(row['attack_step']), 'bit_accuracy': ba, 'accepted': accepted}
        safe.append(item)
        groups[(item['stage'], item['attack_step'])].append(item)
    verified = []
    summaries = list(csv.DictReader(summary.open()))
    assert len(summaries) == len(groups)
    for row in summaries:
        group = groups[(row['stage'], int(row['attack_step']))]
        n = len(group)
        assert len({r['image_idx'] for r in group}) == n == int(row['num_images'])
        accepted = sum(r['accepted'] for r in group)
        mean = statistics.mean(r['bit_accuracy'] for r in group)
        assert accepted / n == float(row['detection_rate'])
        assert abs(mean - float(row['mean_bit_accuracy'])) < 1e-12
        verified.append({'stage': row['stage'], 'attack_step': int(row['attack_step']),
                         'n': n, 'accepted': accepted, 'mean_bit_accuracy': mean})
    out['pgid'][name] = {'raw_path': str(raw.relative_to(ROOT)), 'raw_sha256': sha(raw),
        'summary_sha256': sha(summary), 'summary_csv': summary.read_text(),
        'rows': safe, 'verified_summary': verified, 'all_checks_passed': True,
        'distinct_initial_payloads': len({r['message_bits_str_initial'] for r in rows}),
        'key_identity_independence_established': False}
p = ROOT / 'runs/repro_sembind_masker_20261103/masker_equiv.json'
d = json.loads(p.read_text())
rows = d['rows']
assert len(rows) == d['n_images'] == 20
assert max(r['r1_official_vs_ours'] for r in rows) == 0
assert max(r['r1_official_vs_ours_distorted'] for r in rows) == 0
values = [r['r2_same_content_distorted'] for r in rows]
r2 = {'mean': statistics.mean(values), 'median': statistics.median(values), 'min': min(values), 'max': max(values)}
for key in ('mean', 'min', 'max'):
    assert r2[key] == d['r2_distortion_stability'][key]
out['sembind_masker'] = {'raw_path': str(p.relative_to(ROOT)), 'raw_sha256': sha(p),
    'n_images': 20, 'bits': d['bits'], 'r1_original_max': 0, 'r1_distorted_max': 0,
    'r2_recomputed': r2, 'r2_recorded': d['r2_distortion_stability'],
    'median_discrepancy': 'Historical code used the upper middle observation, not the mean of two middle observations.', 'r3_recorded_summary': d['r3_cross_image'],
    'r3_pairwise_distances_independently_recomputed': False,
    'scope': 'R1/R2 recomputed from per-image records; R3 is stored aggregate only, without semantic labels.'}
print(json.dumps(out, indent=2))
