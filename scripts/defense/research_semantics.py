"""Official pretrained SemBind and CoverLock-g heads on shared DINOv2 weights.

Each head retains its own published preprocessing. The HF backbone adaptation
is checked against the official DINO implementation in an equivalence artifact.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel,AutoImageProcessor
from torchvision.transforms import functional as TF


def coverlock_tokens(backbone,pixels):
    """DINO Torch Hub's historical +0.1 positional interpolation convention."""
    emb=backbone.embeddings
    patches=emb.patch_embeddings(pixels)
    tokens=torch.cat([emb.cls_token.expand(pixels.shape[0],-1,-1),patches],dim=1)
    position=emb.position_embeddings.float()
    side=int((position.shape[1]-1)**.5)
    h,w=pixels.shape[-2]//14,pixels.shape[-1]//14
    patch_pos=F.interpolate(position[:,1:].reshape(1,side,side,-1).permute(0,3,1,2),
                            scale_factor=((h+.1)/side,(w+.1)/side),mode='bicubic',align_corners=False,antialias=False)
    patch_pos=patch_pos.permute(0,2,3,1).reshape(1,h*w,-1)
    tokens=tokens+torch.cat([position[:,:1],patch_pos],dim=1).to(tokens.dtype)
    hidden=backbone.encoder(tokens,return_dict=True).last_hidden_state
    return backbone.layernorm(hidden)[:,1:]


class ResearchSemantics:
    def __init__(self,sembind_root,coverlock_repo,device='cpu'):
        root=Path(sembind_root);repo=Path(coverlock_repo)
        sys.path.insert(0,str(root));sys.path.insert(0,str(repo))
        from model import AttnSupConHashingModel
        from coverlock.model import CoverLockHead
        self.device=device
        self.backbone=AutoModel.from_pretrained(root/'dinov2-giant',local_files_only=True).eval().to(device)
        self.processor=AutoImageProcessor.from_pretrained(root/'dinov2-giant',local_files_only=True)
        self.masker=AttnSupConHashingModel(input_dim=1536,feat_dim=8192,hash_dim=1024).eval().to(device)
        self.masker.load_state_dict(torch.load(root/'sembind_masker.pth',weights_only=True,map_location=device))
        state=torch.load(repo/'weights/coverlock_g.pt',weights_only=False,map_location='cpu')
        assert state['config']['backbone']=='dinov2_vitg14' and state['config']['layer']==39
        self.coverlock=CoverLockHead(1536).eval().to(device)
        self.coverlock.load_state_dict(state['state_dict'],strict=True)
        for module in [self.backbone,self.masker,self.coverlock]:module.requires_grad_(False)

    @torch.inference_mode()
    def extract(self,image):
        x=self.processor(images=image,return_tensors='pt').to(self.device)
        cls=F.normalize(self.backbone(**x).last_hidden_state[:,0],dim=1)
        sem=self.masker(cls,stage='hash')[0]
        x=TF.to_tensor(TF.resize(image.convert('RGB'),[224,224],antialias=True))[None].to(self.device)
        x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
        patches=F.normalize(coverlock_tokens(self.backbone,x),dim=-1)
        cov=self.coverlock(patches.transpose(1,2).reshape(1,1536,16,16))[0]
        return dict(cls=cls[0].cpu().numpy(),sembind_logits=sem.cpu().numpy(),
                    sembind_code=(sem>0).cpu().numpy().astype(np.uint8),
                    coverlock_logits=cov.cpu().numpy(),coverlock_code=(cov>=0).cpu().numpy().astype(np.uint8))
