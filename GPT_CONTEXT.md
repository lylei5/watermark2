# Continuation Context — SemBind+ Semantic Watermark Defense & Robustness

Use this file as the single entry point when continuing research in a new AI session.

## 1. Current Research Goal (CCF-A Defense Track)

Our research objective is **defense-side semantic diffusion watermarking (`SemBind+`)**:
- **Base Paradigm**: Build on **SemBind (ICML 2026)** and **Gaussian Shading (CVPR 2024)** to defend diffusion watermarks against black-box **Imprint forgery**, **Reprompt forgery**, **adaptive feature-alignment forgery**, **regional splicing**, and **composite post-processing**.
- **Core Diagnosis (Already Verified on Server)**:
  1. Unbound initial-noise watermarks (`Gaussian Shading`) fail under 30-step black-box Imprint (`96.0%` forgery success on 50 pairs, `runs/confirmation100_baseline_20261102T204500Z`).
  2. `SemBind` (ICML 2026) reduces mean forged BA (`0.5484` on 28 pairs), but exhibits **four structural vulnerabilities (`V1`–`V4`)**:
     - **V1 (Hash Collision + Majority-Vote Amplification)**: `SemanticMasker` hashes unrelated natural images to as few as `29/1024` Hamming bits apart (`97.2%` agreement across 190 pairs), and $64\times 256$ majority voting amplifies $\sim 60\%$ per-coordinate bit agreement into **2/28 (7.14%) forgery bypasses** (`ba_unbound` up to `0.9844`).
     - **V2 (Public `[CLS]` Bottleneck)**: Because `SemanticMasker` is a deterministic function of the public frozen `DINOv2-giant` `[CLS]` token, an adaptive attacker who adds $\lambda_{\text{sem}}\|\text{CLS}(x_{\text{cand}}) - \text{CLS}(x_{\text{ref}})\|_2^2$ to Imprint forces $S(x_{\text{cand}}) = S(x_{\text{ref}})$ without knowing the private MLP weights or $\pi$.
     - **V3 (Spatial Blindness)**: Discards all 256 spatial patch tokens, failing on regional splicing.
     - **V4 (Hard-Sign Discontinuity & $2\times$ Generation Overhead)**: Flips up to `78/1024` bits under mild blur + $0.8\times$ resize and requires $2\times$ DDIM generation passes.
- **Proposed Solution (`SemBind+`)**:
  - **Module M1**: Keyed Orthogonal Codebook & Sub-block All-or-Nothing Checksum (eliminates V1 collision + voting amplification).
  - **Module M2**: Secret-Keyed Multi-Layer Patch-Attention Binding (eliminates V2 keyless `[CLS]` alignment and V3 spatial blindness).
  - **Module M3**: Single-Pass Mid-Trajectory Lookahead Binding & Confidence-Weighted LLR Verification (eliminates V4 $2\times$ generation cost and hard-sign bit flips).

## 2. Read Order for New Sessions

1. [`README.md`](README.md) — Project overview, empirical summary table, and repository layout.
2. [`docs/SEMBIND_VULNERABILITY_AND_DEFENSE_ANALYSIS.md`](docs/SEMBIND_VULNERABILITY_AND_DEFENSE_ANALYSIS.md) — Complete mathematical and empirical analysis of V1–V4, comparison with PGID/Geometry/SEAL/PAI/ISTS, and SemBind+ M1–M3 formulations.
3. [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) — Stage D1 $\to$ D2 $\to$ D3 experimental execution plan for our CCF-A submission.
4. [`docs/DEFENSE_BASELINE_INVENTORY.md`](docs/DEFENSE_BASELINE_INVENTORY.md) — Detailed inventory of reproduced 2025–2026 defense baselines.
5. [`reports/defense_baselines_20261103/report.md`](reports/defense_baselines_20261103/report.md) — Master empirical report from the 2026-11-03 server runs.
6. [`AGENTS.md`](AGENTS.md) — Server `172.18.185.249` environment, GPU safety, and scientific integrity rules.

## 3. Immediate Next Experimental Step (`Stage D1` $\to$ `Stage D2`)

- Implement the **adaptive `[CLS]`-aligned Imprint attack** (`scripts/defense/sembind_adaptive_cls_attack.py`) on the 28-pair / 50-pair `confirmation100` benchmark to empirically demonstrate **Vulnerability V2** against vanilla `SemBind-GS`.
- Implement **`SemBind+` (`M1 + M2 + M3`)** in `scripts/defense/sembind_plus.py` and evaluate on the 50-pair `confirmation100` benchmark across standard Imprint, adaptive `[CLS]`-aligned Imprint, Reprompt, regional splicing, and composite corruptions.
