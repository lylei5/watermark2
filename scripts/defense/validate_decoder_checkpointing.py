"""Numerically check VAE activation recomputation before using it in inversion."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from diffusers import AutoencoderKL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(261110)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae",
        torch_dtype=torch.float32, local_files_only=True).cuda().eval()
    vae.requires_grad_(False)
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in vae.decoder.modules())
    assert all(m.p == 0 for m in vae.decoder.modules() if isinstance(m, torch.nn.Dropout))
    initial = torch.randn(1, 4, 64, 64, device="cuda") * .1
    outputs = []
    for checkpointing in [False, True]:
        vae.decoder.train(checkpointing)
        vae.decoder.gradient_checkpointing = checkpointing
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        z = initial.detach().clone().requires_grad_(True)
        start = time.monotonic()
        y = vae.decode(z / .18215).sample
        loss = y.square().sum()
        gradient, = torch.autograd.grad(loss, z)
        torch.cuda.synchronize()
        outputs.append(dict(output=y.detach().cpu(), gradient=gradient.detach().cpu(),
            seconds=time.monotonic()-start, peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        print("CHECK", checkpointing, outputs[-1]["seconds"], outputs[-1]["peak_allocated_bytes"], flush=True)
        del y, gradient, loss, z
    a, b = outputs
    passed = torch.allclose(a["output"], b["output"], atol=1e-5, rtol=1e-5) and torch.allclose(
        a["gradient"], b["gradient"], atol=1e-5, rtol=1e-5)
    result = dict(passed=passed, dtype="float32", shape=[1,4,64,64],
        max_output_difference=float((a["output"]-b["output"]).abs().max()),
        max_gradient_difference=float((a["gradient"]-b["gradient"]).abs().max()),
        tolerance=dict(atol=1e-5,rtol=1e-5), training_dependent_layers="No BatchNorm; all Dropout p=0",
        modes=[{k:v for k,v in row.items() if k not in ("output","gradient")} for row in outputs],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitation="Single actual VAE full-resolution forward/gradient check; not an all-input proof")
    args.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    if not passed:
        raise RuntimeError("Checkpointed VAE failed equivalence check")


if __name__ == "__main__":
    main()
