"""SemBind binding/unbinding (paper arXiv 2601.20310, sections 3.2-3.3).

Implemented from the paper because the authors did not release the binding
pipeline. The released masker (model.py + checkpoint) is used verbatim.

Deviation notes:
- inversion uses this project's pinned DDIM 50-step inversion instead of the
  paper's exact DPM-Solver inversion (inv_order=0); the GS decoder is unchanged.
- the permutation pi is a fixed random permutation shared by embedder/verifier.
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SEMBIND_REPO = '/data0/liyanlei/watermark_research/third_party/SemBind'
if SEMBIND_REPO not in sys.path:
    sys.path.insert(0, SEMBIND_REPO)

from model import AttnSupConHashingModel  # noqa: E402
from transformers import AutoImageProcessor, AutoModel  # noqa: E402


class SemanticMasker:
    def __init__(self, dinov2_dir, ckpt_path, device='cuda:0', s=12.0, sigma=1.0,
                 permutation_seed=20260128, latent_shape=(4, 64, 64)):
        self.device = device
        self.s = s
        self.sigma = sigma
        self.shape = tuple(latent_shape)
        self.processor = AutoImageProcessor.from_pretrained(dinov2_dir, local_files_only=True)
        self.dino = AutoModel.from_pretrained(dinov2_dir, local_files_only=True).to(device).eval()
        self.model = AttnSupConHashingModel(input_dim=1536, feat_dim=8192, hash_dim=1024)
        state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()
        self.L = int(np.prod(self.shape))
        generator = torch.Generator().manual_seed(permutation_seed)
        self.perm = torch.randperm(self.L, generator=generator)

    @torch.no_grad()
    def code(self, image):
        inputs = self.processor(images=image, return_tensors='pt').to(self.device)
        cls = self.dino(**inputs).last_hidden_state[:, 0, :]
        cls = F.normalize(cls, dim=1)
        feat = self.model.proj_head(self.model.encoder(cls))
        logits = self.model.hash_head(feat)
        soft = torch.tanh(self.s * logits)
        bits = ((torch.sign(soft) + 1) / 2).floor().to(torch.int64).view(-1)
        return bits

    def mask_from_bits(self, bits):
        repeats = (self.L + bits.numel() - 1) // bits.numel()
        tiled = bits.repeat(repeats)[:self.L]
        keep = int(self.sigma * self.L)
        ratio = torch.zeros_like(tiled)
        ratio[:keep] = tiled[:keep]
        permuted = ratio[self.perm]
        sign = (1 - 2 * permuted).to(torch.float32)
        return sign.view(1, *self.shape)

    def sign_mask(self, image):
        return self.mask_from_bits(self.code(image))

    def embed(self, latent, image):
        return self.sign_mask(image).to(latent.device, latent.dtype) * latent

    def unbind(self, latent, image):
        return self.sign_mask(image).to(latent.device, latent.dtype) * latent
