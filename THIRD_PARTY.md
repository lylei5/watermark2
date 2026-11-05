# Third-Party Codebases & Baselines

This file documents the third-party watermark defense and attack codebases used in this project, both as Git submodules in `third_party/` and as full offline clones in `/data0/liyanlei/watermark_research/third_party/` on server `172.18.185.249`.

## 1. Git Submodules in Repository (`third_party/`)

| Submodule Path | Upstream Repository | Venue / Paper | Role in This Project |
|---|---|---|---|
| `third_party/semantic-forgery` | `https://github.com/ai-secure/semantic-forgery` (`da7b497`) | ICLR 2026 (`arXiv:2503.05660`) | Provides Gaussian Shading (`GsProvider`), Tree-Ring (`TrProvider`), and black-box `Imprint` / `Reprompt` forgery attackers used to stress-test defenses |
| `third_party/geodistort-wm` | `https://github.com/chenyi99/geodistort-wm` (`21ef69a`) | ICML 2026 (`arXiv:2602.11702`) | Official implementation of **Geometry** (`L-SPD`, `Cond`, `Det` Jacobian distortion detectors) |

## 2. Defense Baselines on Server (`/data0/liyanlei/watermark_research/third_party/`)

| Server Directory | Upstream Source | Venue / Paper | Local Weights & Env on `249` | Wrapper / Reproduction Script |
|---|---|---|---|---|
| `third_party/SemBind` | Official SemBind repo (`models/masker.py`, `sembind_pipeline.py`) | ICML 2026 (`arXiv:2602.07512`) | `models/sembind/dinov2-giant`, `models/sembind/sembind_masker.pth` (`semantic_forgery_py310`) | `scripts/defense/sembind.py`, `scripts/defense/sembind_smoke.py`, `scripts/defense/sembind_pairs.py`, `scripts/defense/repro/sembind_masker_equiv.py` |
| `third_party/PGID` | `https://github.com/jisut/PGID` (`c91071f`) | NeurIPS 2026 | `cache/huggingface` SD1.5/SD2.1 (`pgid_py310`) | `reports/repro_pgid_20261103/report.md` |
| `third_party/SEAL` | Official SEAL repo (`seal_pipe.py`, `perceptual_hash.py`) | ICLR 2026 (`arXiv:2503.12172`) | `models/seal/eva02_base_patch14_448` (`semantic_forgery_py310`) | `scripts/defense/seal_smoke.py`, `scripts/salvage30/seal.py` |
| `third_party/ISTS` | Official ISTS repo (`ists_watermark.py`) | CVPR 2026 Findings | `models/openclip/` (`ists_py311`) | `docs/DEFENSE_BASELINE_INVENTORY.md` |
| `third_party/geodistort-wm` | `https://github.com/chenyi99/geodistort-wm` | ICML 2026 (`arXiv:2602.11702`) | `semantic_forgery_py310` | `scripts/defense/geometry.py`, `scripts/defense/evaluate_geometry.py` |
| `third_party/semantic-forgery` | `https://github.com/ai-secure/semantic-forgery` | ICLR 2026 (`arXiv:2503.05660`) | `semantic_forgery_py310` | `scripts/confirmation100/*`, `scripts/salvage30/eval.py` |
