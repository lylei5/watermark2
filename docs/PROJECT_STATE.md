# Project State — SemBind+ Semantic Watermark Defense & Robustness

**Updated**: 2026-11-06  
**Research Direction**: **Defense-Side Semantic Diffusion Watermarking (`SemBind+`) for CCF-A Submission**  
**Server Workspace**: `liyanlei@172.18.185.249:/data0/liyanlei/watermark_research`

---

## 1. Strategic Pivot & Current Status

This repository has transitioned from exploratory black-box forgery benchmarking (former stages `01`–`31`) to **defense-side semantic watermark design and robustness enhancement based on SemBind (ICML 2026)**.

Our completed empirical baselines (`2026-11-02` to `2026-11-03`) established both:
1. **Why unbound watermarks fail**: Standard Gaussian Shading (GS) suffers **96.0% (48/50) forgery success** (`mean BA = 0.8588`, `PSNR = 29.01 dB`) under 30-step black-box Imprint (`runs/confirmation100_baseline_20261102T204500Z`).
2. **Why current 2026 semantic/trajectory defenses remain vulnerable**:
   - **SemBind (ICML 2026)** is reproduced bit-for-bit (`Hamming = 0/1024` vs upstream `SemanticMasker`, `reports/repro_sembind_masker_20261103/report.md`), yet exhibits **four concrete vulnerabilities (`V1`–`V4`)**:
     - **V1**: Low minimum pairwise Hamming distance across natural images (`29/1024 = 97.2%` agreement across 190 pairs) combined with $64\times 256$ majority voting amplification allows **2/28 pairs (7.14%)** to bypass `SemBind-GS` (`ba_unbound` up to **0.9844**).
     - **V2**: Deterministic reliance on the public frozen `DINOv2-giant` `[CLS]` token creates a keyless feature-alignment bottleneck ($\text{CLS}(x_{\text{cand}}) \to \text{CLS}(x_{\text{ref}}) \implies S(x_{\text{cand}}) = S(x_{\text{ref}})$).
     - **V3**: Discarding all 256 spatial patch tokens leaves `SemBind` blind to regional splicing/tampering.
     - **V4**: Hard `sign()` binarization flips up to **78/1024 bits** under mild blur + $0.8\times$ resize, partial mask ratio $\sigma < 1$ leaks unbound coordinates, and 2-stage generation costs $2\times$ NFE with cross-trajectory drift.
   - **PGID (NeurIPS 2026)** shows monotonic `ba_forged` degradation (`0.5781 -> 0.6211 -> 0.6641` at 50/100/150 steps, `reports/repro_pgid_20261103/report.md`).
   - **Geometry (ICML 2026)** requires stateful $z_T$ storage and its headline `L-SPD` metric fails (`AUC = 0.480` at $16\times 16$ grid, `reports/repro_geometry_paper_task_20261103/report.md`).

Full technical details and mathematical formulations of our proposed **`SemBind+`** architecture (`M1`: Keyed Orthogonal Codebook & All-or-Nothing Sub-block Checksum; `M2`: Secret-Keyed Multi-Layer Spatial-Semantic Patch Binding; `M3`: Margin-Guided Single-Pass Lookahead & LLR Verification) are documented in:
- [`docs/SEMBIND_VULNERABILITY_AND_DEFENSE_ANALYSIS.md`](SEMBIND_VULNERABILITY_AND_DEFENSE_ANALYSIS.md)
- [`docs/RESEARCH_PLAN.md`](RESEARCH_PLAN.md)
- [`docs/DEFENSE_BASELINE_INVENTORY.md`](DEFENSE_BASELINE_INVENTORY.md)

---

## 2. Repository Layout (Cleaned for Defense Research)

| Path | Role |
|---|---|
| `README.md`, `AGENTS.md`, `GPT_CONTEXT.md` | Project overview, server/GPU operating rules, and agent continuation entry point |
| `docs/SEMBIND_VULNERABILITY_AND_DEFENSE_ANALYSIS.md` | Core technical analysis of SemBind V1–V4 vulnerabilities, 2026 baseline weaknesses, and SemBind+ M1–M3 design |
| `docs/RESEARCH_PLAN.md` | CCF-A paper storyline and 3-stage experimental execution roadmap (`Stage D1`–`D3`) |
| `docs/DEFENSE_BASELINE_INVENTORY.md` | Inventory and empirical comparison of SemBind, PGID, Geometry, SEAL, PAI, ISTS, Sign-Clamping |
| `docs/LITERATURE.md` | Verified literature notes on 2024–2026 diffusion watermark defenses and attacks |
| `docs/EVALUATION_CONTRACT.md`, `docs/*_CONTRACT.md` | Frozen evaluation contracts for GS, SEAL, PAI, and CSI |
| `scripts/defense/` | Defense wrappers, reproduction scripts, and vulnerability diagnostics (`sembind.py`, `sembind_smoke.py`, `sembind_pairs.py`, `sembind_algebra_test.py`, `geometry.py`, `seal_smoke.py`, `repro/sembind_masker_equiv.py`, `repro/sembind_exact_inversion.py`, `run_after_confirmation100.py`) |
| `scripts/confirmation100/` | Frozen 50-pair (`100` cover/reference assets) benchmark harness and identity diagnostic (`common100.py`, `diag_two_pair_identity.py`, `evaluate100.py`, `analyze100.py`, etc.) |
| `scripts/salvage30/` | Post-hoc defense evaluation harness (`eval.py`, `seal.py`, `template_diag.py`, `analyze.py`) |
| `scripts/cover_pilot/`, `scripts/parallel_execution/` | Core SHA-256 contract, runtime verification, and multi-GPU UUID scheduler libraries used by `confirmation100` and `defense` |
| `protocols/` | Frozen `confirmation100` v7 benchmark protocols and `confirmation100_assets_v11` dataset manifests |
| `reports/` | Verified 2026-11-03 defense reproduction reports (`defense_baselines_20261103`, `repro_sembind_masker_20261103`, `repro_pgid_20261103`, `repro_geometry_paper_task_20261103`, `diag_two_pair_identity_20261103`, `confirmation100_assets_v11_completed`, `22_decoder_contract`) |
| `runs/confirmation100_baseline_20261102T204500Z/` | Lightweight sealed summary JSONs (`analysis_s30.json`, `candidates_sealed_30.json`, `design.json`, `evaluate_s30_0..4.json`) for the 50-pair baseline |
| `third_party/` | Git submodules (`geodistort-wm`, `semantic-forgery`) and server-resident defense repos documented in `THIRD_PARTY.md` |

---

## 3. Server Asset Inventory (`172.18.185.249:/data0/liyanlei/watermark_research`)

All heavy models, environments, datasets, third-party defense codebases, and defense run outputs are preserved on `172.18.185.249:/data0/liyanlei/watermark_research`:

- **`models/` (22 GB)**:
  - `models/sembind/dinov2-giant/` (`facebook/dinov2-giant` weights)
  - `models/sembind/sembind_masker.pth` (official pretrained `SemanticMasker` checkpoint)
  - `models/seal/` (`eva02_base_patch14_448` weights)
  - `models/openclip/` (OpenCLIP weights for ISTS/eval)
- **`cache/huggingface/` (18 GB)**:
  - Offline HuggingFace snapshots for `stable-diffusion-v1-5`, `stable-diffusion-2-1-base`, `stable-diffusion-xl-base-1.0`, `clip-vit-large-patch14`, etc.
- **`envs/` (6.2 GB)**:
  - `envs/semantic_forgery_py310` (primary Python 3.10 + PyTorch 2.4.0+cu121 environment for GS, SemBind, SEAL, Geometry)
  - `envs/pgid_py310` (Python 3.10 environment for PGID)
  - `envs/ists_py311` (Python 3.11 environment for ISTS)
- **`datasets/` (68 MB)**:
  - `datasets/confirmation100_relay_v11/rgb512/` (100 verified $512\times 512$ natural cover images with SHA-256 manifest)
- **`third_party/` (426 MB)**:
  - `SemBind`, `PGID`, `SEAL`, `ISTS`, `geodistort-wm`, `semantic-forgery`
- **`runs/` & `private_eval/` (Defense & Baseline Runs Preserved)**:
  - `runs/confirmation100_baseline_20261102T204500Z` + `private_eval/confirmation100_baseline_20261102T204500Z`
  - `runs/sembind_pairs30_20261102T233720Z`
  - `runs/sembind_151step_20261103T002848Z`
  - `runs/sembind_smoke_20261102T224343Z`, `runs/sembind_smoke_20261102T225342Z`
  - `runs/repro_sembind_masker_20261103`
  - `runs/repro_sembind_exact_inv_20261103`
  - `runs/repro_pgid_n5_20261103`, `runs/pgid_smoke_20261102T231752Z`
  - `runs/diag_two_pair_identity_20261103`

---

## 4. Historical Provenance (Archived Attack-Phase Commits)

Prior to the 2026-11-06 defense reorganization, the following commits recorded the earlier black-box attack stages (`01`–`31`) and exploratory branches (retained in Git object history if ever needed for provenance):

- Pre-reorganization `main` tip (`2026-11-03` defense reports + legacy stages): `b2d5d4104d3b82abed1369e9a6a0222e231bdc75`
- Stage 27–31 recovery & `confirmation100` v1.1 asset completion (`research/stage27-recovery-20261019`): `77ca1157279c927d95a2a3eb95d9edba82b270e5`
- Earlier cleanup snapshot (`chore/repo-cleanup-20261030`): `46da8c383d589729c9b2e5e15fbcc04a1b93f8fa`
