"""Released SemBind semantic-code extraction for component experiments.

This module only extracts the 1024-bit code. Carriers own their private-key
binding, and sembind_official_audit.load_official owns diffusion inversion.
It does not implement an end-to-end native SemBind watermark system.
"""
import sys

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


class SemanticCodeExtractor:
    def __init__(self, dinov2_dir, ckpt_path, device='cuda:0', s=12.0):
        # Workers can preload the hash model from their frozen official source.
        if 'model' not in sys.modules:
            sys.path.insert(0, '/data0/liyanlei/watermark_research/third_party/SemBind')
        from model import AttnSupConHashingModel

        self.device = device
        self.s = s
        self.processor = AutoImageProcessor.from_pretrained(dinov2_dir, local_files_only=True)
        self.dino = AutoModel.from_pretrained(dinov2_dir, local_files_only=True).to(device).eval()
        self.model = AttnSupConHashingModel(input_dim=1536, feat_dim=8192, hash_dim=1024)
        state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()

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
