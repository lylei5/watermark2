"""CPU tests for the geometry-score implementation."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry as G

rng = np.random.default_rng(0)
a = rng.normal(size=(1, 4, 64, 64)).astype(np.float32)
b = rng.normal(size=(1, 4, 64, 64)).astype(np.float32)

assert abs(G.gcos(a, a) - 1.0) < 1e-9
assert abs(G.sad(a, a)) < 1e-6
assert abs(G.gcos(a, b)) < 0.02
assert abs(G.lspd(a, a)) < 1e-9
assert G.lspd(a, b) > 1e-3
assert abs(G.lspd(a, b) - G.lspd(b, a)) < 1e-9
print('PASS gcos/sad/lspd identities')

# positive class separated from negative class
pos = [0.9, 0.8, 0.85]
neg = [0.1, 0.2, 0.3, 0.15]
assert abs(G.auc(pos, neg) - 1.0) < 1e-9
pos_t = [0.5, 0.5]
neg_t = [0.5, 0.5]
assert abs(G.auc(pos_t, neg_t) - 0.5) < 1e-9
print('PASS auc ordering and ties')

# 16x16 grid assumption is explicit and shape-checked
try:
    G.lspd(a, b, grid=15)
    raise AssertionError('grid not divisible must fail')
except ValueError:
    pass
print('PASS grid validation')
print('ALL GEOMETRY TESTS PASSED')

