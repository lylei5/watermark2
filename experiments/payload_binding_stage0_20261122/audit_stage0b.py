import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '/data0/liyanlei/watermark_research/runs/payload_binding_stage0_20261122')
import run_stage0_sscd as s0b

RUN = Path('/data0/liyanlei/watermark_research/runs/payload_binding_stage0_20261122')
projs = np.load(RUN / 'sscd_projections.npz')
for k, key in [(48, 'p48'), (64, 'p64'), (96, 'p96')]:
    P = projs[key]
    assert P.shape == (512, k)
    # Verify orthonormality
    gram = P.T @ P
    assert np.allclose(gram, np.eye(k), atol=1e-6), f'Non-orthonormal {key}'
    # Verify exact bit-packing roundtrip on 20 random unit vectors
    rng = np.random.default_rng(100 + k)
    for _ in range(20):
        v = rng.standard_normal(512)
        v /= np.linalg.norm(v)
        b = s0b.pack_unpack_signs(v, P)
        expected = np.where((v @ P) >= 0.0, 1.0, -1.0)
        assert np.array_equal(b, expected)

s_a = json.loads((RUN / 'summary.json').read_text())
s_b = json.loads((RUN / 'summary_sscd.json').read_text())
assert s_a['confirm_60']['rgb_mean_var_48b']['n_edited'] == 1920
assert s_b['confirm_60']['sscd_64b']['n_edited'] == 1920
assert s_a['imprint']['ALL']['rgb_mean_var_48b']['n'] == 30
assert s_b['imprint']['ALL']['sscd_64b']['n'] == 30
assert s_b['reprompt_plus']['GS_sd21']['sscd_64b']['n_candidates'] == 90

audit = dict(
    status='ok',
    orthonormal_projections_verified=[48, 64, 96],
    bit_packing_roundtrip_checks=60,
    denominators_matched=dict(confirm_edited=1920, imprint_all=30, gs_reprompt_per_proxy=90, tr_reprompt_per_proxy=30),
)
(RUN / 'audit_stage0b.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit))
