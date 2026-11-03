import sys
import time

import torch
from PIL import Image

ROOT = '/data0/liyanlei/watermark_research'
sys.path.insert(0, ROOT + '/scripts/defense')
from sembind import SemanticMasker

t = time.time()
mk = SemanticMasker(ROOT + '/models/sembind/dinov2-giant', ROOT + '/models/sembind/sembind_masker.pth', device='cpu')
print('load', round(time.time() - t, 1))
im = Image.open(ROOT + '/datasets/confirmation100_relay_v11/rgb512/000000084477.png').convert('RGB')
t = time.time()
bits = mk.code(im)
print('code', bits.shape, float(bits.float().mean()), round(time.time() - t, 1))
S = mk.mask_from_bits(bits)
print('mask', S.shape, float((S == -1).float().mean()))
# determinism
bits2 = mk.code(im)
print('deterministic', bool(torch.equal(bits, bits2)))
