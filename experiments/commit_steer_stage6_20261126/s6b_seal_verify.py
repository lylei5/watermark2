"""Stage 6B SEAL Imprint Verification (gidx 4..9) — Runs ONLY in seal_reproduction_py310."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path('/data0/liyanlei/watermark_research')
RUN = ROOT / 'runs/commit_steer_stage6_20261126/imprint10'
SEAL_CLEAN_RUN = ROOT / 'runs/unified_seal_clean_20261116'
EXTRA_DEPS = ROOT / 'envs/phase1_extra'


def read_json(p: Path):
    return json.loads(p.read_bytes())


def write_json(p: Path, obj):
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(p)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def check_idle_gpu(idx: int) -> str:
    assert idx in (0, 1, 2, 3, 4, 5, 7) and idx != 6, f'Forbidden GPU index {idx}'
    info = subprocess.check_output(
        ['nvidia-smi', '-i', str(idx), '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
        text=True,
    ).strip().split(', ')
    busy = subprocess.check_output(
        ['nvidia-smi', '-i', str(idx), '--query-compute-apps=pid', '--format=csv,noheader'],
        text=True,
    ).strip()
    assert int(info[1]) < 100 and int(info[2]) == 0 and not busy, f'GPU {idx} busy: {info}, {busy}'
    return info[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    out_path = RUN / 'cross_seal_verify_4_9.json'
    if out_path.exists():
        print('[S6B SEAL verify] Already complete.', flush=True)
        return

    uuid = check_idle_gpu(args.gpu)
    os.environ.update(
        CUDA_VISIBLE_DEVICES=uuid,
        HF_HOME=str(ROOT / 'cache/huggingface'),
        HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4',
        TOKENIZERS_PARALLELISM='false',
    )

    seal_plan = read_json(SEAL_CLEAN_RUN / 'protocol.json')
    cfg = seal_plan['detector']
    models = read_json(SEAL_CLEAN_RUN / 'model_manifest.json')
    assert sha256_file(Path(cfg['private_salt_file'])) == cfg['private_salt_sha256']

    import torch
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    official_seal = Path(cfg['source'])
    sys.path.insert(0, str(official_seal))
    sys.path.insert(0, str(SEAL_CLEAN_RUN))
    sys.path.insert(0, str(EXTRA_DEPS))
    from inverse_stable_diffusion import InversableStableDiffusionPipeline
    from utils import generate_caption, transform_img, calculate_patch_l2
    from seal_paper_simhash import SealPaperSimHash

    pipe = InversableStableDiffusionPipeline.from_pretrained(
        models['diffusion_model']['root'], torch_dtype=torch.float16, local_files_only=True
    ).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    assert type(pipe.scheduler).__name__ == cfg['expected_scheduler']
    processor = Blip2Processor.from_pretrained(models['caption_model']['root'], local_files_only=True)
    captioner = Blip2ForConditionalGeneration.from_pretrained(
        models['caption_model']['root'], torch_dtype=torch.float16, local_files_only=True
    ).to('cuda')
    encoder = SentenceTransformer(models['embedding_model']['root'], device='cuda')
    sampler = SealPaperSimHash(bytes.fromhex(read_json(Path(cfg['private_salt_file']))['salt']))

    def seal_verify(im: Image.Image):
        im_rgb = im.convert('RGB')
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        caption = generate_caption(im_rgb, processor, captioner)
        with torch.no_grad():
            emb = encoder.encode(caption, convert_to_tensor=True).to('cuda')
            ref = sampler.sample(emb / torch.norm(emb)).to(dtype=pipe.vae.dtype)
            tensor = transform_img(im_rgb).unsqueeze(0).to('cuda', dtype=pipe.vae.dtype)
            latent = pipe.get_image_latents(tensor, sample=False)
            rec = pipe.forward_diffusion(
                latents=latent,
                text_embeddings=pipe.get_text_embedding(''),
                guidance_scale=cfg['inversion_cfg'],
                num_inference_steps=cfg['inference_steps'],
            )
            dists = np.asarray(calculate_patch_l2(rec, ref, 1024), dtype=np.float64)
        torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        matches = int((dists < 2.3).sum())
        return dict(
            matches=matches,
            accept_12=bool(matches >= 12),
            accept_4=bool(matches >= 4),
            accept_2=bool(matches >= 2),
            caption=caption,
            sec_verify=sec,
        )

    meta = read_json(RUN / 'cross_seal_gen_meta.json')
    imp_dir = RUN / 'imprint_imgs_cross_seal'
    t_all = time.perf_counter()
    cases = []
    for gidx in range(4, 10):
        m = meta[str(gidx)]
        ckpts = {}
        for step in (50, 100, 150):
            im = Image.open(imp_dir / f'gidx_{gidx:03d}_step_{step:03d}.png').convert('RGB')
            v = seal_verify(im)
            ckpts[f'step_{step}'] = dict(loss=m['losses'][f'step_{step}'], **v)
            print(f'  [S6B SEAL verify] gidx={gidx:03d} step={step:3d} | matches={v["matches"]} (@12={v["accept_12"]}, @4={v["accept_4"]}, @2={v["accept_2"]})', flush=True)
        cases.append(dict(gidx=gidx, seal_idx=gidx, cover=m['cover'], checkpoints=ckpts))

    write_json(out_path, dict(elapsed_seconds=time.perf_counter() - t_all, gpu=args.gpu, uuid=uuid, imprint_cases=cases))
    print(f'[S6B SEAL verify COMPLETE] 6 cases x 3 ckpts in {time.perf_counter()-t_all:.1f}s', flush=True)


if __name__ == '__main__':
    main()
