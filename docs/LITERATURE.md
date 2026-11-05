# Literature Notes — Semantic Watermark Defense & Robustness

This document records verified literature for our **CCF-A defense-side research** on semantic and trajectory-bound diffusion watermarking. Every entry distinguishes between verified primary-source facts and our empirical/analytical assessments.

---

## 1. Core Semantic & Trajectory-Bound Watermark Defenses (Primary Focus)

### 1.1 SemBind — Binding Watermarks to Image Semantics for Robust Forgery Defense
- **Authors**:iling / SemBind authors (official repository: `third_party/SemBind`)
- **Venue / Status**: ICML 2026 (`arXiv:2602.07512`, Feb 2026)
- **Source Basis**: Verified primary paper + official code (`third_party/SemBind/models/masker.py`) + server reproduction (`reports/repro_sembind_masker_20261103/report.md`, `reports/defense_baselines_20261103/report.md`).
- **Core Mechanism**:
  - Replaces static initial-noise Gaussian Shading (GS) templates with an image-semantics-bound watermark mask.
  - Generation is 2-stage:
    1. Generate an auxiliary clean image $x_{\text{aux}}$ via standard DDIM sampling from initial noise.
    2. Extract the frozen `facebook/dinov2-giant` `[CLS]` token ($1536$-dim), pass it through a trained private `SemanticMasker` MLP ($1536 \to 2048 \to 1024$ with `BatchNorm1d + LeakyReLU` and $3\times$ residual `Linear(1024,1024)` blocks), and take `sign(logits)` to obtain a 1024-bit semantic hash $S(x_{\text{aux}}) \in \{0,1\}^{1024}$.
    3. Expand $S(x_{\text{aux}})$ by $16\times$ repetition, permute by secret $\pi$, XOR with the $64\times$ repeated 256-bit message $m$, and re-sample initial noise $z_T^{\text{sb}}$ via inverse-CDF sign modulation (`SemBind-GS`) or complex-plane phase rotation (`SemBind-TR`), followed by a second DDIM generation pass to produce $x_{\text{wm}}$.
  - Verification extracts $S(x_{\text{test}})$ from the test image itself, unbinds $\hat{y} = \pi^{-1}(\mathbf{1}_{z_T^{\text{inv}} > 0}) \oplus \text{Repeat}_{16}(S(x_{\text{test}}))$, and performs $64\times$ majority voting per message bit.
- **Verified Empirical Findings & Vulnerabilities in This Repo**:
  - **Bit-exact masker equivalence**: Our wrapper matches upstream `SemanticMasker` with `0/1024` bit disagreement across all 20 cover images (`reports/repro_sembind_masker_20261103/report.md`).
  - **V1 — Hash Collision & Voting Amplification**: Over 190 cover pairs, unrelated natural images exhibit minimum Hamming distance **29/1024** (**97.2% bit agreement**). Under 30-step black-box Imprint on 28 pairs, **2/28 pairs (7.14%) bypass SemBind-GS** (`ba_unbound` up to **0.9844**) because $64\times$ majority voting amplifies $\sim 60\%$ per-coordinate bit agreement into $>71.5\%$ message bit accuracy.
  - **V2 — Public Backbone `[CLS]` Bottleneck (Keyless Alignment Bypass)**: Because `SemanticMasker` is a deterministic function of the public frozen `DINOv2-giant` `[CLS]` token, an adaptive attacker who aligns $\text{CLS}(x_{\text{cand}}) \to \text{CLS}(x_{\text{ref}})$ during Imprint forces $S(x_{\text{cand}}) = S(x_{\text{ref}})$ without knowing the private MLP weights or $\pi$.
  - **V3 — Spatial Blindness**: Discards all 256 spatial patch tokens of DINOv2 (`outputs.last_hidden_state[:, 1:, :]`), failing to detect regional splicing or background replacement when `[CLS]` remains dominated by a pasted foreground object.
  - **V4 — Hard-Sign Discontinuity, $\sigma$ Leakage & $2\times$ Generation Overhead**: Mild Gaussian blur + $0.8\times$ resize flips up to **78/1024 bits**; setting $\sigma < 1$ leaves $(1-\sigma)L$ coordinates completely unbound; and 2-stage generation doubles inference cost ($2\times$ NFE) while introducing cross-trajectory drift between $x_{\text{aux}}$ and $x_{\text{wm}}$.

### 1.2 PGID — Prompt-Guided Intermediate Diffusion Watermarking
- **Authors**: Official repository `third_party/PGID` (`jisut/PGID`)
- **Venue / Status**: NeurIPS 2026
- **Source Basis**: Verified official code + server reproduction (`reports/repro_pgid_20261103/report.md`, `reports/defense_baselines_20261103/report.md`).
- **Core Mechanism**:
  - Injects a Gaussian Shading / Tree-Ring watermark at an intermediate denoising step (`target_step = 20`) instead of $t = T$, using a private reference prompt (`"masterpiece, best quality"`) during verification inversion.
- **Verified Empirical Findings**:
  - Using the official empty attacker prompt (`""`), 5-pair Imprint forgery yields monotonic `ba_forged` growth with attack budget: **0.5781** (50 steps) $\to$ **0.6211** (100 steps) $\to$ **0.6641** (150 steps), reaching within 4.3pp of the $\tau = 0.7070$ detection threshold.
  - Relies on obscurity of the private verifier prompt rather than cryptographic or semantic binding to the image content.

### 1.3 Geometry — Detect Any Forgery Ranked by Geometric Distortion in Diffusion Watermarks
- **Authors**: Official repository `third_party/geodistort-wm`
- **Venue / Status**: ICML 2026 (`arXiv:2602.11702`, Feb 2026)
- **Source Basis**: Verified primary paper + official code + server reproduction (`reports/repro_geometry_paper_task_20261103/report.md`, `scripts/defense/geometry.py`).
- **Core Mechanism**:
  - Estimates the local deformation field $T_\theta$ between the original watermarked latent $z_T(I_w)$ and the test inverted latent $z_T(\tilde{I})$ on a $k\times k$ control grid, computes Jacobian $J = I + \nabla d$, and ranks distortion via Log-Euclidean SPD norm (`L-SPD`), condition number (`Cond`), or determinant deviation (`Det`).
- **Verified Empirical Findings**:
  - **Stateful**: Requires storing or looking up the ground-truth initial latent $z_T(I_w)$ for every issued image; cannot run zero-shot on arbitrary incoming images.
  - **Grid Failure**: At the paper's $16\times 16$ control grid (`--num_points 16`), `L-SPD` achieves only **AUC = 0.480** (random chance) on the paper's own distortion task (`reports/repro_geometry_paper_task_20261103/report.md`), while `Cond` achieves `0.958` and `Det` achieves `0.985`. At $32\times 32$ (`num_points=32`), 100 optimization iterations leave $73.4\%$ of control points untouched, causing rank-deficient covariance ($\text{rank} \le 3$).

### 1.4 SEAL — Semantic Entangled Autonomous Locks
- **Authors**: Official repository `third_party/SEAL`
- **Venue / Status**: ICLR 2026 (`arXiv:2503.12172`)
- **Source Basis**: Verified primary paper + official code (`docs/SEAL_EVALUATION_CONTRACT.md`, `scripts/defense/seal_smoke.py`).
- **Core Mechanism**:
  - Splits the $64\times 64\times 4$ latent into $k = 1024$ patches ($2\times 2\times 4 = 16$ dims each), hashes each patch's semantic embedding (`eva02_base_patch14_448` + `PerceptualHashProjection`) to seed a deterministic normal vector $n_i$, and embeds watermark energy via biphasic directional alignment $\langle z_{T,i}, n_i \rangle$.
- **Verified Empirical Findings**:
  - Patch-wise SimHash is non-cryptographic (locality-sensitive): semantically similar covers or feature-aligned candidates share a high fraction of patch seeds.
  - On our 30-step Imprint forgery (`scripts/defense/seal_smoke.py`), SEAL rejects forged candidates (`mean_Sim = 6.97/1024` vs threshold `543/1024`), but clean generation requires biphasic optimization overhead and its patch seeds remain deterministic without a secret key.

### 1.5 PAI — Proof-of-Authenticity of Images viaInitial Noise Binding
- **Authors**: Official repository `third_party/PAI`
- **Venue / Status**: ICLR 2026 (`arXiv:2509.22253`)
- **Source Basis**: Verified primary paper + official code (`docs/PAI_EVALUATION_CONTRACT.md`).
- **Core Mechanism**:
  - Interactive challenge-response protocol where the prover holds $(x_{\text{init}}, \text{prompt}, \text{seed})$ and embeds a frequency-domain response to a verifier challenge $a$, checked via normalized inner product $\rho$ after DDIM inversion.
- **Verified Empirical Findings**:
  - Requires interactive prover participation and persistent storage of exact initial noise $x_{\text{init}}$; cannot serve as a non-interactive public watermark detector.

### 1.6 ISTS — Implicit Diffusion Watermarking via Intermediate Step Selection
- **Authors**: Official repository `third_party/ISTS`
- **Venue / Status**: CVPR 2026 Findings
- **Source Basis**: Verified official code (`third_party/ISTS`).
- **Core Mechanism**:
  - Selects the injection timestep $t \in [1, 32]$ and latent channel $l \in [0, 3]$ via `CLIP-ViT-B/32` image feature clustering (`K-Means` with 1024 centroids) and applies a two-sided bit accuracy test (`BA >= tau` or `BA <= 1 - tau`).
- **Verified Analytical Assessment**:
  - The discrete secret space is tiny ($32 \times 4 = 128$ $(t, l)$ pairs), and two-sided detection only defeats sign-flipping removal (`1 - BA`), not full-latent Imprint forgery which drives all timesteps/channels toward the reference trajectory.

### 1.7 Sign-Clamping Defense for Gaussian Shading
- **Venue / Status**: Preprint (`arXiv:2610.40204`, Oct 2026)
- **Core Mechanism**:
  - Clips per-coordinate log-likelihood / magnitude contributions before decoding or uses sign-only hard decisions so that large-magnitude outlier coordinates created by adversarial optimization cannot dominate soft correlation scores.
- **Relevance to SemBind+**:
  - Shows why soft inner-product detectors are vulnerable to magnitude concentration, and complements our **Sub-block All-or-Nothing Checksum** design where every sub-block must independently pass sign agreement.

### 1.8 CoverLock — Cover-Conditioned Watermark Locking against Imprint Forgery
- **Venue / Status**: Preprint (`arXiv:2609.32241`, Sep 2026)
- **Core Mechanism**:
  - Conditions watermark verification on high-frequency cover-specific residuals that do not survive cross-image latent transfer.
- **Relevance to SemBind+**:
  - Reinforces our **M2 (Keyed Multi-Layer Patch-Attention Binding)** design: binding watermark masks jointly to mid-layer spatial structure and deep semantic tokens prevents coarse `[CLS]`-only impersonation.

---

## 2. Attack & Threat Model Benchmarks Used to Stress-Test Defenses

### 2.1 Black-Box Forgery & Removal on Semantic Watermarks (`semantic-forgery`)
- **Authors**: Official repository `third_party/semantic-forgery`
- **Venue / Status**: ICLR 2026 (`arXiv:2503.05660`)
- **Source Basis**: Verified official code (`third_party/semantic-forgery/imprint_forger.py`, `reprompt_forger.py`, `docs/EVALUATION_CONTRACT.md`).
- **Core Mechanism**:
  - **Imprint Forgery**: Inverts a watermarked reference image $x_{\text{ref}}$ and an unwatermarked cover image $x_{\text{cover}}$ under a surrogate model (e.g., SD2.1 50-step DDIM inversion) and optimizes $x_{\text{cover}} + \delta$ via projected gradient descent (`lr=0.01`, LPIPS penalty) so that $z_T(x_{\text{cover}} + \delta) \to z_T(x_{\text{ref}})$.
  - **Reprompt Forgery**: Re-generates a new image from $z_T(x_{\text{ref}})$ using a new target prompt.
- **Verified Empirical Baseline in This Repo**:
  - Under our frozen 50-pair `confirmation100` benchmark (`runs/confirmation100_baseline_20261102T204500Z`, 30-step SD2.1 surrogate Imprint vs SD1.5 GS target), standard Gaussian Shading fails catastrophically: **96.0% (48/50) forgery success rate**, mean `BA = 0.8588`, mean `PSNR = 29.01 dB`.
  - Under 151-step Imprint (`sembind_smoke.py`), standard GS reaches **`BA = 0.9961`**, proving that unbound initial-noise watermarks are completely broken by black-box Imprint forgery.

### 2.2 Gaussian Shading — Provable Performance-Lossless Image Watermarking for Diffusion Models
- **Authors**: Yang et al.
- **Venue / Status**: CVPR 2024 (`arXiv:2404.04956`)
- **Core Mechanism**:
  - Embeds a $k$-bit watermark message ($k = 256$) by repeating it $f_c \times f_{hw}^2 = 64$ times across the $4\times 64\times 64 = 16384$ latent dimensions, XORing with a secret stream cipher / ChaCha20 key, and sampling initial Gaussian noise conditioned on the resulting sign bits via inverse CDF.
  - Detects watermarks via exact binomial tail test on majority-voted message bit accuracy ($\tau = 0.70703125 = 181/256$ at $\text{FPR} \approx 10^{-8}$).
- **Role in Our Project**:
  - The base carrier for both `SemBind-GS` and our proposed **`SemBind+`**. Our goal is to preserve Gaussian Shading's distribution-preserving generation and stateless verification while eliminating its vulnerability to Imprint forgery and SemBind's V1–V4 failure modes.

### 2.3 Tree-Ring Watermarks — Fingerprints for Diffusion Images that are Invisible and Robust
- **Authors**: Wen et al.
- **Venue / Status**: NeurIPS 2023 (`arXiv:2305.20030`)
- **Core Mechanism**:
  - Embeds concentric Fourier-domain ring patterns into the initial noise latent $z_T$ and detects them via $L_1$ or phase distance after DDIM inversion.
- **Role in Our Project**:
  - Secondary carrier (`SemBind-TR` / `SemBind+-TR`) used to demonstrate that our keyed multi-layer semantic binding generalizes across both bit-level (`GS`) and ring-level (`TR`) diffusion watermarks.
