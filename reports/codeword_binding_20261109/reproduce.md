# Reproducing the pilots

The experiment bundle contains code, protocols, result JSONs, cached semantic features, paired channel scores and generated images. It excludes pretrained weights. Exact numerical replay across hardware or software versions is not guaranteed. `runtime_versions.json` records the local environment; model and image hashes are preserved in manifests.

Run from the extracted bundle or the repository root. All output paths must be new directories. Inspect GPU occupancy first and bind an available GPU by UUID; never use server GPU index 6. For CPU-only channel evaluation, pass `--device cpu` (random draws can differ from CUDA).

```bash
python scripts/defense/test_codeword_binding.py
python scripts/defense/codeword_channel_pilot.py --features runs/semantic_codes_local --output runs/channel_replay
```

For end-to-end pilots, set `CUDA_VISIBLE_DEVICES` to the chosen UUID and use offline models. `--model` must point to a Diffusers SD2.1 directory; `--semantic` contains `dinov2-giant/`, `sembind_masker.pth`, and the official SemBind `model.py`. We used a CPU-converted fp16 copy of the existing SD2.1 snapshot, revision `0094d483a120f3f33dafbd187ea4aa60d10de75c`. Generation and inversion use DDIM, not SemBind's official exact inversion.

```bash
python scripts/defense/codeword_diffusion_pilot.py --model /path/to/sd21_fp16 --semantic /path/to/sembind --output runs/diffusion_replay --n 5
python scripts/defense/codeword_hard_reprompt_pilot.py --model /path/to/sd21_fp16 --semantic /path/to/sembind --reuse runs/diffusion_replay --output runs/hard_replay --n 5
python scripts/defense/codeword_degree2_pilot.py --model /path/to/sd21_fp16 --semantic /path/to/sembind --reuse runs/diffusion_replay --output runs/degree2_replay --n 5
```

These commands reproduce development pilots with public deterministic key material, not a secret-key security evaluation. The candidate hash is recorded before scoring. A formal evaluation requires an isolated verifier, new keys and data, attack-content labels, full attack budgets and quality measurements. Keep the degree-2 run exploratory: it was motivated by the degree-4 JPEG result.

The plotting scripts in `analysis/` document figure construction. Their original input layout uses `work/` and `outputs/`; adapt those path constants to the extracted `runs/` layout before replaying them.
