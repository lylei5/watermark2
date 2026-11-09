"""Extract real-image semantic codes for a decoder diagnostic, without diffusion.

This is evaluation data, not an attacker input. No watermark keys are accessed.
Supports CPU-only execution on a shared host and CUDA on a personal workstation.
"""
import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from transformers import AutoImageProcessor, AutoModel


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def variants(image):
    yield 'clean', image
    b = io.BytesIO()
    image.save(b, format='JPEG', quality=50)
    b.seek(0)
    yield 'jpeg50', Image.open(b).convert('RGB')
    w, h = image.size
    yield 'blur_resize', image.filter(ImageFilter.GaussianBlur(1)).resize(
        (int(w * .8), int(h * .8)), Image.Resampling.BICUBIC).resize(
            (w, h), Image.Resampling.BICUBIC)
    yield 'crop90', image.crop((int(.05*w), int(.05*h), int(.95*w), int(.95*h))).resize(
        (w, h), Image.Resampling.BICUBIC)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-root', type=Path, required=True)
    p.add_argument('--model-code', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pairs', type=Path)
    p.add_argument('--limit', type=int, default=100)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--half', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(20261109)
    sys.path.insert(0, str(args.model_code))
    from model import AttnSupConHashingModel
    dtype = torch.float16 if args.half else torch.float32
    start = time.monotonic()
    proc = AutoImageProcessor.from_pretrained(args.model_root / 'dinov2-giant', local_files_only=True)
    dino = AutoModel.from_pretrained(args.model_root / 'dinov2-giant', local_files_only=True,
                                   torch_dtype=dtype).to(args.device).eval()
    masker = AttnSupConHashingModel(input_dim=1536, feat_dim=8192, hash_dim=1024)
    masker.load_state_dict(torch.load(args.model_root / 'sembind_masker.pth',
                                     weights_only=True, map_location='cpu'), strict=True)
    masker = masker.to(args.device).eval()
    print('LOADED', round(time.monotonic()-start, 2), flush=True)
    paths = [('natural', x) for x in sorted(args.images.glob('*.png'))[:args.limit]]
    if args.pairs:
        paths += [('historical', x) for x in sorted(args.pairs.glob('*_reference.png'))]
        paths += [('historical', x) for x in sorted(args.pairs.glob('*_forged.png'))]
    records, codes, logits_all, cls_all = [], [], [], []
    for group, path in paths:
        im = Image.open(path).convert('RGB')
        items = variants(im) if group == 'natural' else [('clean', im)]
        for variation, image in items:
            t = time.monotonic()
            with torch.inference_mode():
                x = proc(images=image, return_tensors='pt').to(args.device)
                x['pixel_values'] = x['pixel_values'].to(dtype)
                cls = F.normalize(dino(**x).last_hidden_state[:, 0].float(), dim=1)
                logits = masker(cls, stage='hash')
                code = ((torch.tanh(12*logits).sign()+1)//2).to(torch.uint8)
            records.append(dict(group=group, file=path.name, variant=variation, sha256=sha(path)))
            codes.append(code.cpu().numpy()[0])
            logits_all.append(logits.cpu().numpy()[0])
            cls_all.append(cls.cpu().numpy()[0])
            np.savez_compressed(args.output / 'features.npz', codes=np.array(codes),
                                logits=np.array(logits_all), cls=np.array(cls_all))
            (args.output / 'records.json').write_text(json.dumps(records, indent=2))
            print('DONE',len(records),path.name,variation,round(time.monotonic()-t,2),flush=True)
    meta = dict(completed=True, image_count=len(paths), record_count=len(records),
                dtype=str(dtype), device=args.device, torch=torch.__version__,
                seconds=time.monotonic()-start, script_sha256=sha(__file__),
                masker_sha256=sha(args.model_root/'sembind_masker.pth'),
                dino_sha256=sha(args.model_root/'dinov2-giant/model.safetensors'),
                purpose='real-image feature extraction; not end-to-end defense evaluation')
    (args.output/'manifest.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta),flush=True)


if __name__ == '__main__':
    main()
