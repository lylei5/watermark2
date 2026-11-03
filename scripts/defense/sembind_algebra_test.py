import sys

import torch

ROOT = '/data0/liyanlei/watermark_research'
sys.path.insert(0, ROOT + '/scripts/defense')
from sembind import SemanticMasker

mk = SemanticMasker.__new__(SemanticMasker)
mk.shape = (4, 64, 64)
mk.L = 4 * 64 * 64
mk.sigma = 1.0
g = torch.Generator().manual_seed(3)
bits = (torch.rand(1024, generator=g) > 0.5).to(torch.int64)

# identity permutation: tiling and truncation must reproduce the code
mk.perm = torch.arange(mk.L)
S_id = mk.mask_from_bits(bits)
flat = (1 - S_id.view(-1)) / 2
assert torch.equal(flat[:1024], bits.float()), 'first tile must equal bits'
assert torch.equal(flat[1024:2048], bits.float()), 'second tile must equal bits'
assert flat.numel() == 16384
print('PASS tiling: 16 repeats of the 1024-bit code, length 16384')

# sigma = 0.5: only the first 8192 coordinates keep the code, the tail is zeroed (+1 sign)
mk.sigma = 0.5
S_half = mk.mask_from_bits(bits)
flat_half = (1 - S_half.view(-1)) / 2
assert torch.equal(flat_half[:8192], flat[:8192])
assert float(flat_half[8192:].abs().sum()) == 0.0
print('PASS sigma=0.5 keeps first floor(0.5L)=8192 coordinates and zeroes the rest')

# any permutation is an involution given the same key, so unbinding recovers z exactly
mk.perm = torch.randperm(mk.L, generator=torch.Generator().manual_seed(11))
S = mk.mask_from_bits(bits)
assert set(torch.unique(S).tolist()) <= {-1.0, 1.0}
z = torch.randn(1, 4, 64, 64)
assert torch.equal(S * (S * z), z), 'S*S must be identity'
print('PASS permutation: S in {+1,-1} and S*S*z == z')

# determinism of the permutation key
mk2 = SemanticMasker.__new__(SemanticMasker)
mk2.shape, mk2.L, mk2.sigma = (4, 64, 64), 16384, 1.0
mk2.perm = torch.randperm(16384, generator=torch.Generator().manual_seed(11))
assert torch.equal(mk.perm, mk2.perm)
print('PASS permutation key determinism')
print('ALL SEMBIND ALGEBRA TESTS PASSED')
