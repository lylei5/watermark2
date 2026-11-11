"""Reproduction check 1 for SemBind (ICML 2026): masker bit-level equivalence.

Question
--------
Our `scripts/defense/sembind.py::SemanticMasker` re-implements the released
hash model's forward path by hand. Does it produce exactly the same 1024-bit
code as the authors' own entry point
`AttnSupConHashingModel.forward(x, stage='binarize', s=12.0)`,
as used in their released notebook `semantimark.ipynb`?

This is the only component of SemBind whose reference implementation was
actually published. The binding/unbinding algebra (S, pi, sigma) was NOT
released, so it cannot be reproduced and is out of scope here.

Two further behavioural checks are reported, because they are what the
released notebook demonstrates:

  R1  bit-level equivalence   : official bits == our bits, Hamming must be 0
  R2  distortion stability    : same content, mild distortion -> small Hamming
  R3  discrimination          : different content -> Hamming near 512/1024

CPU-only mode is supported for small N; GPU is used when available.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

ROOT = Path('/data0/liyanlei/watermark_research')
sys.path.insert(0, str(ROOT / 'scripts/defense'))
sys.path.insert(0, str(ROOT / 'third_party/SemBind'))

SEMBIND_DIR = ROOT / 'models/sembind'
CKPT = SEMBIND_DIR / 'sembind_masker.pth'
DINO = SEMBIND_DIR / 'dinov2-giant'
S = 12.0


def official_bits(model, processor, dino, image, device):
    """Exactly the path used by the released notebook."""
    inp = processor(images=image, return_tensors='pt').to(device)
    with torch.no_grad():
        cls = dino(**inp).last_hidden_state[:, 0, :]
        cls = F.normalize(cls, dim=1)
        bits = model(cls, stage='binarize', s=S)
    return bits.view(-1).to(torch.int64).cpu()


def distort(img):
    """Mild, label-preserving distortions (JPEG-like blur + rescale round trip)."""
    w, h = img.size
    out = img.filter(ImageFilter.GaussianBlur(radius=1.0))
    out = out.resize((int(w * 0.8), int(h * 0.8)), Image.BICUBIC).resize((w, h), Image.BICUBIC)
    return out


def hamming(a, b):
    return int((a != b).sum())


def main(out_path, n_images=20, device=None):
    n_images = int(n_images)
    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    from model import AttnSupConHashingModel
    from transformers import AutoImageProcessor, AutoModel
    from sembind import SemanticMasker

    processor = AutoImageProcessor.from_pretrained(str(DINO), local_files_only=True)
    dino = AutoModel.from_pretrained(str(DINO), local_files_only=True).to(device).eval()
    official = AttnSupConHashingModel(input_dim=1536, feat_dim=8192, hash_dim=1024)
    official.load_state_dict(torch.load(str(CKPT), map_location='cpu', weights_only=True), strict=True)
    official.to(device).eval()

    ours = SemanticMasker(str(DINO), str(CKPT), device=device)

    # Test images: the frozen confirmation100 v1.1 covers present on the host.
    covers = sorted((ROOT / 'datasets/confirmation100_relay_v11/rgb512').glob('*.png'))[:n_images]
    if not covers:
        raise SystemExit('no test images found')

    rows = []
    all_bits = []
    for p in covers:
        with Image.open(p) as im:
            img = im.convert('RGB')
        ob = official_bits(official, processor, dino, img, device)
        mb = ours.code(img).to(torch.int64).cpu()
        d_img = distort(img)
        ob_d = official_bits(official, processor, dino, d_img, device)
        mb_d = ours.code(d_img).to(torch.int64).cpu()
        rows.append(dict(
            image=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
            r1_official_vs_ours=hamming(ob, mb),
            r1_official_vs_ours_distorted=hamming(ob_d, mb_d),
            r2_same_content_distorted=hamming(ob, ob_d),
            ones_fraction=float(ob.float().mean()),
        ))
        all_bits.append(ob)

    # R3: cross-image discrimination on the official path
    cross = []
    for i in range(len(all_bits)):
        for j in range(i + 1, len(all_bits)):
            cross.append(hamming(all_bits[i], all_bits[j]))

    def stats(xs):
        if not xs:
            return None
        xs = sorted(xs)
        n = len(xs)
        return dict(n=n, mean=sum(xs) / n, min=xs[0], max=xs[-1], median=(xs[(n - 1) // 2] + xs[n // 2]) / 2)

    result = dict(
        check='sembind_masker_bit_equivalence',
        device=device,
        s=S,
        n_images=len(rows),
        bits=1024,
        r1_bit_equivalence=dict(
            identical=all(r['r1_official_vs_ours'] == 0 for r in rows),
            max_hamming=max(r['r1_official_vs_ours'] for r in rows),
            distorted_max_hamming=max(r['r1_official_vs_ours_distorted'] for r in rows),
        ),
        r2_distortion_stability=stats([r['r2_same_content_distorted'] for r in rows]),
        r3_cross_image=stats(cross),
        reference_half=512,
        rows=rows,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:])
