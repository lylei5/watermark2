"""SEAL defense smoke test: generate one SEAL-watermarked image, verify it, and
verify unwatermarked controls. Runs the published decision rule (patch L2 < 2.3,
at least 12 matching patches) with k=1024, b=7, seed=42.

Deviations from the official entry, recorded here:
- SD2.1 weights come from the locally cached Manojb mirror (fp32, cast to fp16),
  because stabilityai/stable-diffusion-2-1-base is gated and unreachable offline.
- The caption-embedding model is loaded through transformers with mean pooling
  (the official SentenceTransformer pipeline), because sentence_transformers is
  not installed on the server.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SEAL_DIR = Path('/data0/liyanlei/watermark_research/third_party/SEAL')
sys.path.insert(0, str(SEAL_DIR))

from transformers import AutoModel, AutoTokenizer, Blip2ForConditionalGeneration, Blip2Processor
from diffusers import DDIMScheduler
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from utils import calculate_patch_l2, generate_caption, generate_initial_noise, transform_img

K, B, SEED, TAU, M_MATCH = 1024, 7, 42, 2.3, 12


def embed_text(text, tokenizer, model, device):
    enc = tokenizer([text], return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**enc).last_hidden_state
    mask = enc['attention_mask'].unsqueeze(-1)
    emb = (out * mask).sum(1) / mask.sum(1)
    return emb / emb.norm(dim=1, keepdim=True)


def main(sd21, blip2_dir, embed_dir, out_path, device='cuda:0'):
    torch.manual_seed(0)
    np.random.seed(0)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pipe = InversableStableDiffusionPipeline.from_pretrained(sd21, torch_dtype=torch.float16, local_files_only=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    cap_processor = Blip2Processor.from_pretrained(blip2_dir, local_files_only=True)
    cap_model = Blip2ForConditionalGeneration.from_pretrained(blip2_dir, torch_dtype=torch.float16, local_files_only=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(embed_dir, local_files_only=True)
    embed_model = AutoModel.from_pretrained(embed_dir, local_files_only=True).to(device).eval()

    def caption(image):
        return generate_caption(image, cap_processor, cap_model, device=device)

    def verify(image):
        cap = caption(image)
        emb = embed_text(cap, tokenizer, embed_model, device).to(dtype=pipe.vae.dtype)
        noise = generate_initial_noise(emb, K, B, SEED, device).to(dtype=pipe.vae.dtype)
        tensor = transform_img(image).unsqueeze(0).to(device, dtype=pipe.vae.dtype)
        latents = pipe.get_image_latents(tensor, sample=False)
        recon = pipe.forward_diffusion(latents=latents, text_embeddings=pipe.get_text_embedding(''),
                                       guidance_scale=1, num_inference_steps=50)
        l2 = calculate_patch_l2(recon, noise, K)
        matches = int(sum(1 for v in l2 if v < TAU))
        return dict(caption=cap, patch_l2_mean=float(np.mean(l2)), patch_l2_median=float(np.median(l2)),
                    matches_below_tau=matches, accepted=matches >= M_MATCH)

    prompt = 'a photograph of a red fox standing in a snowy forest'
    start = time.time()
    base_img = pipe(prompt).images[0]
    cap = caption(base_img)
    emb = embed_text(cap, tokenizer, embed_model, device)
    noise = generate_initial_noise(emb, K, B, SEED, device).to(dtype=pipe.vae.dtype)
    wm_img = pipe(prompt, latents=noise).images[0]
    other_img = pipe('a blue bicycle leaning against a brick wall').images[0]

    results = {
        'prompt': prompt,
        'watermarked': verify(wm_img),
        'unwatermarked_same_prompt': verify(base_img),
        'unwatermarked_other_prompt': verify(other_img),
        'params': dict(k=K, b=B, seed=SEED, tau=TAU, m_match=M_MATCH),
        'sd21_snapshot': str(sd21),
        'blip2_dir': str(blip2_dir),
        'embed_dir': str(embed_dir),
        'seconds': time.time() - start,
        'deviations': ['Manojb fp32 SD2.1 snapshot cast to fp16', 'caption embedding via transformers mean pooling'],
    }
    wm_img.save(out.parent / 'seal_watermarked.png')
    base_img.save(out.parent / 'seal_base.png')
    other_img.save(out.parent / 'seal_other.png')
    out.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps({k: results[k] for k in ('watermarked', 'unwatermarked_same_prompt', 'unwatermarked_other_prompt', 'seconds')}, indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:])
