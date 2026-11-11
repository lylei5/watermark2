"""Genuine-image audit of released SemBind components; no attacks are run.

Uses the original inv-dpm scheduler (diffusers v0.21.4) with released SemBind
generation/inversion source. Watermark binding is the project's GS adapter.
Stages keep DINO and diffusion models out of GPU memory simultaneously.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from PIL import Image

from codeword_binding import keyed_order, sembind_mask_bits, bit_accuracy

SCHEDULER_KW = dict(beta_end=.012, beta_schedule="scaled_linear", beta_start=.00085,
                    num_train_timesteps=1000, prediction_type="epsilon",
                    steps_offset=1, trained_betas=None, solver_order=1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def legacy_scheduler(path):
    spec = importlib.util.spec_from_file_location("diffusers.schedulers.audit_legacy_dpm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DPMSolverMultistepScheduler


def scheduler_audit(path):
    from diffusers import DPMSolverMultistepScheduler
    result = {}
    for name, cls in [("installed", DPMSolverMultistepScheduler),
                      ("inv_dpm_0_21_4", legacy_scheduler(path))]:
        scheduler = cls(**SCHEDULER_KW)
        scheduler.set_timesteps(50)
        z = torch.ones(1, 4, 64, 64)
        t = scheduler.timesteps[-1] - 20
        try:
            pred = scheduler.convert_model_output(torch.zeros_like(z), t, z)
            result[name] = dict(shape=list(pred.shape), finite=bool(pred.isfinite().all()),
                                scalar_timestep=int(t), same_shape=pred.shape == z.shape,
                                mean=float(pred.mean()))
        except Exception as exc:
            result[name] = dict(error=type(exc).__name__, message=str(exc))
    return result


def load_official(args):
    # Only the unused prompt-dataset loader is stubbed, never numerical code.
    if importlib.util.find_spec("datasets") is None:
        stub = types.ModuleType("datasets")
        stub.__spec__ = importlib.util.spec_from_loader("datasets", loader=None)
        def unused(*a, **kw):
            raise RuntimeError("Dataset loader is outside this fixed-prompt audit")
        stub.load_dataset = unused
        sys.modules["datasets"] = stub
    sys.path.insert(0, str(args.official.resolve()))
    import inversion
    inversion.DPMSolverMultistepScheduler = legacy_scheduler(args.scheduler)
    pipe = inversion.stable_diffusion_pipe(model_id=str(args.model), cache_dir=str(args.model))
    pipe.set_progress_bar_config(disable=False, mininterval=10)
    pipe.enable_attention_slicing("max")
    for model in (pipe.unet, pipe.vae, pipe.text_encoder):
        model.requires_grad_(False)
    original_encode_prompt = pipe.encode_prompt
    def encode_prompt_low_memory(*a, **kw):
        result = original_encode_prompt(*a, **kw)
        pipe.text_encoder.cpu()
        torch.cuda.empty_cache()
        return result
    pipe.encode_prompt = encode_prompt_low_memory
    original_decoder_inv = pipe.decoder_inv
    def decoder_inv_low_memory(x):
        # Same 100-step official latent optimization, with inactive modules off GPU.
        pipe.unet.cpu()
        pipe.text_encoder.cpu()
        decoder = pipe.vae.decoder
        if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in decoder.modules()):
            raise RuntimeError("Checkpointing requires a decoder without BatchNorm")
        if any(m.p != 0 for m in decoder.modules() if isinstance(m, torch.nn.Dropout)):
            raise RuntimeError("Checkpointing requires zero decoder dropout")
        previous_training = decoder.training
        previous_checkpointing = decoder.gradient_checkpointing
        decoder.train(True)
        decoder.gradient_checkpointing = True
        gc.collect()
        torch.cuda.empty_cache()
        try:
            z = original_decoder_inv(x)
            return z.detach()
        finally:
            decoder.train(previous_training)
            decoder.gradient_checkpointing = previous_checkpointing
            gc.collect()
            torch.cuda.empty_cache()
            pipe.unet.cuda()
    pipe.decoder_inv = decoder_inv_low_memory
    return inversion, pipe


def carrier(root, task):
    secret = json.loads((root / "private" / f"{task:03}.json").read_text())
    key, nonce, msg, bkey = [bytes.fromhex(secret[k])
                            for k in ("key", "nonce", "message", "binding_key")]
    cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None).encryptor().update(msg * 64)
    bits = np.unpackbits(np.frombuffer(cipher, dtype=np.uint8))
    message = np.unpackbits(np.frombuffer(msg, dtype=np.uint8))
    z = np.abs(np.random.default_rng(secret["amplitude_seed"]).standard_normal(16384))
    z = (z * (2 * bits.astype(np.float32) - 1)).astype(np.float32)
    return z, bits, message, keyed_order(bkey, b"sembind/permutation/", 16384)


def main():
    p = argparse.ArgumentParser()
    for name in ("root", "output", "official", "scheduler", "model", "semantic", "coverlock"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--stage", choices=["prepare", "aux", "codes", "reference", "invert", "score"], required=True)
    p.add_argument("--features", choices=["aux", "reference"], default="aux")
    p.add_argument("--count", type=int, default=4)
    p.add_argument("--semantic-only", action="store_true", help="Extract only the released SemBind code")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    out = args.output
    if args.stage == "prepare":
        if (out / "protocol.json").exists():
            raise FileExistsError("Protocol is already frozen")
        plan = json.loads((args.root / "generation_plan.json").read_text())
        rows = [r for r in plan["rows"] if r["split"] == "development"][:args.count]
        if len(rows) != args.count:
            raise ValueError("Insufficient development prompts")
        import diffusers, transformers
        protocol = dict(scope="genuine component check, not complete paper reproduction",
            rows=rows, generation=dict(steps=50, cfg=7.5, solver_order=1, dtype="float32"),
            inversion=dict(steps=50, inv_order=0, blind_prompt="", cfg=3.,
                           decoder_inv=[False, True],
                           oracle_prompt="diagnostic only; first task, generation prompt and CFG7.5"),
            adapter="GS sign carrier and keyed semantic permutation reimplementation",
            scheduler="unmodified diffusers v0.21.4 source used by original inv-dpm",
            official_commit="337c9f90a150157ac1d410208ae20e6530ca8cdd",
            source_sha256={str(f.relative_to(args.official)): sha(f)
                           for f in args.official.rglob("*.py")},
            scheduler_sha256=sha(args.scheduler), input_manifest_sha256=sha(args.root / "generation_plan.json"),
            model_files_sha256={str(f.relative_to(args.model)): sha(f)
                                for f in args.model.rglob("*") if f.is_file()},
            dependencies=dict(torch=torch.__version__, diffusers=diffusers.__version__,
                              transformers=transformers.__version__),
            attacks_run=False, confirmation_read=False, semantic_only=args.semantic_only)
        write(out / "protocol.json", protocol)
        write(out / "scheduler_compatibility.json", scheduler_audit(args.scheduler))
        print("PREPARED", len(rows), flush=True)
        return
    protocol = json.loads((out / "protocol.json").read_text())
    rows = protocol["rows"]
    # Refuse mixing output assets after implementation/source or model changes.
    if sha(args.scheduler) != protocol["scheduler_sha256"]:
        raise RuntimeError("Scheduler source changed")
    for rel, digest in protocol["source_sha256"].items():
        if sha(args.official / rel) != digest:
            raise RuntimeError("Official source changed: " + rel)
    start = time.monotonic()
    if args.stage == "codes":
        if args.semantic_only:
            spec = importlib.util.spec_from_file_location("model", args.official / "model.py")
            official_model = importlib.util.module_from_spec(spec)
            sys.modules["model"] = official_model
            spec.loader.exec_module(official_model)
            from sembind import SemanticMasker
            model = SemanticMasker(args.semantic / "dinov2-giant", args.semantic / "sembind_masker.pth")
            def extract(image):
                return dict(sembind_code=model.code(image).cpu().numpy().astype(np.uint8))
        else:
            from research_semantics import ResearchSemantics
            model = ResearchSemantics(str(args.semantic), str(args.coverlock), "cuda")
            extract = model.extract
        for row in rows:
            task = row["task"]
            names = [f"{task:03}_aux"] if args.features == "aux" else [
                f"{task:03}_sembind", f"{task:03}_gs"]
            for name in names:
                dst = out / "codes" / (name + ".npz")
                if dst.exists():
                    continue
                dst.parent.mkdir(exist_ok=True)
                np.savez_compressed(dst, **extract(Image.open(out / "images" / (name + ".png")).convert("RGB")))
                print("CODES", name, flush=True)
    elif args.stage in ("aux", "reference", "invert"):
        inversion, pipe = load_official(args)
        for row in rows:
            task = row["task"]
            if args.stage == "aux":
                dst = out / "images" / f"{task:03}_aux.png"
                if dst.exists():
                    continue
                z = np.random.default_rng(row["aux_seed"]).standard_normal((1, 4, 64, 64)).astype(np.float32)
                image, _, _ = inversion.generate(prompt=row["caption"], guidance_scale=7.5,
                    num_inference_steps=50, solver_order=1, pipe=pipe, init_latents=torch.from_numpy(z).cuda())
                dst.parent.mkdir(exist_ok=True)
                image.save(dst)
                print("AUX", task, flush=True)
            elif args.stage == "reference":
                z, _, _, perm = carrier(args.root, task)
                code = np.load(out / "codes" / f"{task:03}_aux.npz")["sembind_code"]
                for name, mask in [("sembind", sembind_mask_bits(code[None], perm)[0]),
                                   ("gs", np.zeros(16384))]:
                    dst = out / "images" / f"{task:03}_{name}.png"
                    if dst.exists():
                        continue
                    latents = (z * (1 - 2 * mask.astype(np.float32))).reshape(1, 4, 64, 64)
                    image, _, _ = inversion.generate(prompt=row["caption"], guidance_scale=7.5,
                        num_inference_steps=50, solver_order=1, pipe=pipe,
                        init_latents=torch.from_numpy(latents).cuda())
                    image.save(dst)
                    print("REFERENCE", task, name, flush=True)
            else:
                for name in ["sembind", "gs"]:
                    image = Image.open(out / "images" / f"{task:03}_{name}.png").convert("RGB")
                    settings = [("blind_encoder", "", False, 3.), ("blind_decoder", "", True, 3.)]
                    if task == rows[0]["task"]:
                        settings.append(("oracle_decoder", row["caption"], True, 7.5))
                    for setting, prompt, decoder, cfg in settings:
                        dst = out / "inversions" / f"{task:03}_{name}_{setting}.npy"
                        if dst.exists():
                            continue
                        torch.manual_seed(202611100000 + task)
                        tick = time.monotonic()
                        z = inversion.exact_inversion(image, prompt=prompt, guidance_scale=cfg,
                            num_inference_steps=50, test_num_inference_steps=50,
                            inv_order=0, decoder_inv=decoder, pipe=pipe)
                        array = z.detach().cpu().numpy()
                        if array.shape != (1, 4, 64, 64) or not np.isfinite(array).all():
                            raise ValueError("Invalid inversion")
                        dst.parent.mkdir(exist_ok=True)
                        np.save(dst, array)
                        write(dst.with_suffix(".json"), dict(seconds=time.monotonic()-tick,
                            image_sha256=sha(out / "images" / f"{task:03}_{name}.png"),
                            inversion_sha256=sha(dst), setting=setting))
                        print("INVERT", task, name, setting, round(time.monotonic()-tick, 1), flush=True)
    else:
        results = []
        for row in rows:
            task = row["task"]
            _, cipher, message, perm = carrier(args.root, task)
            stream = cipher ^ np.tile(message, 64)
            auxiliary = np.load(out / "codes" / f"{task:03}_aux.npz")["sembind_code"]
            for method in ("sembind", "gs"):
                final = np.load(out / "codes" / f"{task:03}_{method}.npz")["sembind_code"]
                for path in sorted((out / "inversions").glob(f"{task:03}_{method}_*.npy")):
                    signs = (np.load(path).reshape(-1) > 0).astype(np.uint8)
                    metrics = dict(raw_ba_without_semantic=float(bit_accuracy(signs ^ stream, message)[0]))
                    for label, code in (("auxiliary", auxiliary), ("final", final)):
                        mask = sembind_mask_bits(code[None], perm)[0] if method == "sembind" else np.zeros(16384, dtype=np.uint8)
                        bits = signs ^ mask ^ stream
                        ba = float(bit_accuracy(bits, message)[0])
                        metrics[label + "_ba"] = ba
                        metrics[label + "_accepted"] = ba > 181/256
                    source_mask = sembind_mask_bits(auxiliary[None], perm)[0] if method == "sembind" else np.zeros(16384, dtype=np.uint8)
                    results.append(dict(task=task, method=method, setting=path.stem.split("_", 2)[2],
                        semantic_agreement=float((auxiliary == final).mean()),
                        channel_sign_error=float((signs != (cipher ^ source_mask)).mean()), **metrics))
        write(out / "results.json", dict(scope=protocol["scope"], rows=results,
            threshold="BA > 181/256", official_full_reproduction=False,
            protocol_sha256=sha(out / "protocol.json")))
        print(json.dumps(results, indent=2))
    write(out / (args.stage + ("_" + args.features if args.stage == "codes" else "") + "_complete.json"),
          dict(seconds=time.monotonic()-start, script_sha256=sha(__file__),
               protocol_sha256=sha(out / "protocol.json")))
    print("DONE", args.stage, round(time.monotonic()-start, 1), flush=True)


if __name__ == "__main__":
    main()
