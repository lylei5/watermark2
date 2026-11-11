# Research state: content-bound diffusion watermark defense

Updated 2026-11-11. Goal: a publishable defense with matched genuine detection, image quality, computation budgets and credible false-positive calibration. No superiority or SOTA claim is established.

## Compute and safety

All future research experiments and scoring, including CPU work, run on server 249 only. No local model downloads, experimental execution or GPU fallback. Local code/document editing and file management are allowed. Server root is /data0/liyanlei/watermark_research. Only GPUs 7,5,4, at most three, and only idle devices. Never use GPU6 or terminate other users' jobs. Do not continuously poll; use process-exit events and independent research between meaningful checks.

Only existing published top-conference/top-journal attacks with full documented implementations and schedules are allowed. Never invent or improve attacks. Keep verifier keys outside baseline generation processes. Same-account directory separation is not an OS sandbox. Preserve calibration and confirmation splits.

## Latest completed evidence

Read reports/full_imprint_20261111/report.md and research_direction.md in the same folder. All eight fixed DPM/fp32 Imprint jobs completed 151 updates; all 128 checkpoint images plus 8 references and 8 covers were verified with DPM50 and full 100-step VAE inversion. Four paired sources per method, not a paper-scale reproduction. SemBind carrier/binding is a project adaptation using the released semantic head.

Final update: SemBind references 4/4, covers 0/4, candidates 0/4, mean candidate BA 0.451171875, PSNR 18.41738 dB, LPIPS 0.2782693. GS references 4/4, covers 0/4, candidates 4/4, candidate BA 1, PSNR 18.48838 dB, LPIPS 0.2707488. SemBind rejects all 64 saved candidates; GS accepts 60/64 (all four at every saved checkpoint from update11 onward). Checkpoints repeat the same four sources and are not independent samples.

The frozen geometry verifier, its split variant, identity-only binomial detector, stricter identity-only detector, and default BA make exactly the same decisions on all 144 observations. No added anti-forgery benefit is observed in this complete-baseline experiment. Raw results and trajectory plots are committed in the report folder.

Previous DDIM development evidence: 160 SemBind genuine views, default BA 144/160, query-only geometry 160/160. Existing Reprompt candidates: default 1/16, stricter identity-only and geometry 0/16. Thus the observed gain is crop recovery. Ideal random-template Monte Carlo has 2048 draws; it is not empirical real-key FPR. References: reports/blind_geometry_development_20261110/report.md and method.md. TAG-WM is ICCV2025, not merely a preprint; its current citation has been verified and updated.

## Calibration execution and numerical compatibility

The 32 held-out calibration source identities are tasks16..47, 192 fixed legitimate queries, with 6144 intended real-identity decisions including 5952 wrong identities. Confirmation remains unscored. Thresholds, 251-pose grid and data are frozen.

First continuation failed on exact comparison of two floating-point binomial tails. The runtime computed 5.634806966794081e-12 versus frozen 5.634806966794078e-12 (relative difference 5.7343e-16). score_frozen_geometry.py now validates only library roundoff (relative tolerance1e-12, zero absolute tolerance) and applies the exact protocol value to every rule. Six server tests pass; all 144 original BAs match exactly.

The next execution at runs/geometry_calibration_20261111 completed shard0 caches but failed the development bridge: relative latent L2 0.00177317 and 0.00202889 exceed the fixed0.001 gate, while semantic bits match exactly. A recorded-batch2 replay remains at0.00177830 and0.00203351. No calibration decisions were read. Read reports/geometry_calibration_20261111/runtime_audit.md; preserve both failed attempts.

Current automatic workflow: runs/geometry_calibration_replay_20261111, setup PID36153, last observed stage installing_recorded_runtime. It creates isolated envs/ddim_replay_py310, pins original PyTorch2.1.2+cu121, torchvision0.16.2+cu121 and Diffusers0.30.3 plus recorded dependencies. It first reruns the two development bridge cases; the original0.001 tolerance is unchanged. Only after passing does it run all three calibration shards sequentially on GPU5 (GPUs7/4 were occupied at the last allocation check). It rechecks GPU5 before work, uses complete DDIM50 and recorded batch2, and stops on a failure without retries/polling or threshold changes. Protocol: protocols/geometry_calibration_replay_20261111.json. Current code snapshots match that protocol; the prior20261111 executor source snapshots are in reports/geometry_calibration_20261111/source_snapshots/.

Do not launch duplicate setup/continuation jobs. runtime_setup_status.json and complete.json/failed.json distinguish installation, bridge failure and validation completion. A setup or partial cache is not a completed experiment.

## Runtime assets and local storage

The original DPM server snapshot and local DDIM assets have equal effective fp16 UNet weights but different VAE/text-encoder weights. A separate matched DDIM model is retained on server under runs/geometry_calibration_20261110/model; its 3 weight groups,8 configurations and5 semantic assets were audited. Do not attribute cross-pipeline differences only to schedulers. The current primary server env remains unchanged while the isolated replay env is prepared.

Local model weights and two incomplete downloads were removed after server verification (14.61GB); do not repopulate them. Codes, reports, configs and original experimental data remain locally. Temporary ZIP deletion was blocked by automatic approval and those archives remain; see reports/local_storage_cleanup_20261110/report.md. Private identities are on server under private_eval, not in Git.
