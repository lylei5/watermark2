# AGENTS.md — Operating Rules for SemBind+ Watermark Defense Research

These instructions apply to the entire repository and the remote server workspace `/data0/liyanlei/watermark_research` on `172.18.185.249`.

## 1. Research Objective & Scope

- **Primary Goal**: **Defense-side semantic diffusion watermarking (`SemBind+`)** targeting a **CCF-A conference submission**.
- **Core Focus**:
  1. Expose and rigorously quantify the structural vulnerabilities (`V1`–`V4`) of **SemBind (ICML 2026)** and contemporary 2025–2026 defenses (**PGID**, **Geometry**, **SEAL**, **PAI**, **ISTS**).
  2. Design, implement, and verify **`SemBind+`** (`M1`: Keyed Orthogonal Codebook & All-or-Nothing Sub-block Checksum; `M2`: Secret-Keyed Multi-Layer Patch-Attention Binding; `M3`: Single-Pass Mid-Trajectory Lookahead & Confidence-Weighted LLR Verification).
  3. Evaluate defense robustness (`FPR` under Imprint, Adaptive `[CLS]`-Aligned Imprint, Reprompt, and Regional Splicing) and distortion resilience (`TPR` under JPEG, blur, resize, crop, noise) alongside image generation quality (`FID`, `CLIP-Score`, `LPIPS`, `PSNR`, `NFE`) on the frozen 50-pair `confirmation100` benchmark.

## 2. Scientific Integrity Rules

1. **No Fabrication**: Never invent experimental numbers, logs, image quality metrics, or citations. Every reported number must trace to a committed JSON/report artifact or server run directory.
2. **Preserve Negative & Diagnostic Results**: If an ablation or baseline variant underperforms, record it honestly in `reports/` and use it to sharpen the analytical diagnosis.
3. **Contract-Verified Evaluation**:
   - Gaussian Shading / SemBind-GS detection threshold is frozen at $\tau = 0.70703125$ (`181/256` bits, $\text{FPR} \approx 10^{-8}$) per [`docs/EVALUATION_CONTRACT.md`](docs/EVALUATION_CONTRACT.md).
   - Never leak private verifier keys ($\pi$, secret projection matrices $R_K$, or stored reference latents) into black-box attacker processes.

## 3. Remote Server & GPU Discipline (`172.18.185.249`)

- **Server Workspace**: `/data0/liyanlei/watermark_research` (SSH alias `249`; on multi-homed campus/VPN Windows hosts, bind SSH to the campus WLAN IP if needed, e.g. `ssh -b 172.26.12.72 249`).
- **Offline Model & Cache Roots**:
  - Set `HF_HOME=/data0/liyanlei/watermark_research/cache/huggingface`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.
  - Reuse local model weights in `/data0/liyanlei/watermark_research/models/` (`sembind/dinov2-giant`, `sembind/sembind_masker.pth`, `seal/`, `openclip/`).
- **Python Environments**:
  - Primary env: `/data0/liyanlei/watermark_research/envs/semantic_forgery_py310/bin/python`
  - PGID env: `/data0/liyanlei/watermark_research/envs/pgid_py310/bin/python`
  - ISTS env: `/data0/liyanlei/watermark_research/envs/ists_py311/bin/python`
- **Shared GPU Safety**:
  - Never touch processes belonging to other users.
  - Never use GPU index `6` (`GPU-518229d6-dc60-c78f-6f59-473ef2ba67c7`, hardware ECC fault).
  - Always check `nvidia-smi` before launching GPU jobs and bind child processes by GPU UUID (`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=<GPU-UUID>`).
  - Never touch unrelated directories under `/data0/liyanlei/` (such as `ceramic_watermark`, `FontGuard_repro`, `genstego_sota_*`, `natural_image_512bit_watermark_poc`, `official_hidden_compare`, `trajectory_orth_20261103`).
