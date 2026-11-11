"""Run the CVPR 2025 single-candidate Reprompt baseline from cached inversions.

Source algorithm: and-mill/semantic-forgery at
ca68950cd7b53b8438a49580f134b2a0db28d778, run_reprompting.py phases 2-3.
This adapter uses SD2.1 DDIM50, RGB PNG inputs and fixed benign COCO prompts.
It is a method-level reproduction, not the original I2P benchmark. It performs
no resampling, optimization, candidate search, semantic alignment or feedback.
The job file supplies only public-image-derived inversion and target prompt;
there are no verifier keys, codes or detector calls in this process.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--jobs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', required=True)
    args = p.parse_args()
    jobs = json.loads(args.jobs.read_text())['rows']
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, safety_checker=None,
        local_files_only=True).to('cuda')
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    rows = []
    start = time.monotonic()
    with torch.inference_mode():
        for r in jobs:
            if sha(r['inversion_path']) != r['inversion_sha256']:
                raise ValueError('Public reference inversion changed')
            z = np.load(r['inversion_path'], allow_pickle=False).reshape(1,4,64,64)
            image = pipe(r['target_prompt'], latents=torch.from_numpy(z).cuda().half(),
                         num_inference_steps=50, guidance_scale=7.5,
                         height=512, width=512).images[0]
            path = args.output/(r['id']+'.png')
            image.save(path)
            rows.append(dict(id=r['id'], image_path=str(path.resolve()),
                             image_sha256=sha(path)))
            print('REPROMPT', len(rows), len(jobs), round(time.monotonic()-start,1), flush=True)
    result = dict(completed=True, rows=rows, seconds=time.monotonic()-start,
                  jobs_sha256=sha(args.jobs), script_sha256=sha(__file__),
                  baseline='CVPR2025 Reprompt without resampling',
                  implementation='DDIM50 fp16 adapter; public reference inversion cached in fp32',
                  candidates_per_reference=1, detector_queries=0,
                  isolation='separate process with public inputs; no OS access sandbox')
    (args.output/'complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
