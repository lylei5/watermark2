# Operating rules for semantic watermark defense research

These rules apply to this repository and `/data0/liyanlei/watermark_research` on SSH host `249`.

## Repository workflow

- Maintain `main` as the canonical research branch. Commit and push routine updates to `main`, updating the corresponding code, reports and project-state files together.
- Create an additional branch only when the user explicitly requests one.

## Objective and current state

- Develop a publishable defense with benefits at matched absolute legitimate-image detection rates, documented image quality and computation budgets.
- Read `reports/geometry_confirmation_20261111/report.md` for completed independent confirmation, `reports/geometry_calibration_20261111/completed/report.md` for calibration, and `reports/full_imprint_20261111/report.md` for full-budget forgery results. The active official nine-condition benchmark is documented in `reports/gs_published_conditions_20261111/plan.md`; consult `project_state.json` for execution state.
- Frozen all-fit geometry accepted382/384 genuine queries versus default319/384, both with0/24,192 wrong-identity accepts. Added forgery rejection and SOTA remain unproven. The confirmation set is consumed; future method changes require new independent validation. The nine-condition benchmark shares these64 sources and must be reported separately.

## Evaluation scope

- Use baseline tests, official implementations and stated parameters from SemBind and other established defense papers. Do not devise custom attack methods.
- Work only on authorized offline image-watermark experiments. Do not conduct network-security testing or interact with third-party systems as experimental targets.
- Distinguish a released component, a project adapter and a complete paper reproduction. The CoverLock hash on an auxiliary-image GS pipeline is not its native post-hoc system.
- Keep development, calibration and confirmation data separate. Freeze methods and calibration before viewing confirmation decisions.
- Compare absolute legitimate-image detection rates. Conditional retention among each method's already-accepted samples is not an equal-TPR comparison.
- Preserve the default GS working point `BA > 181/256`. Any supplementary calibrated point must be separately identified and have its false-positive implications reported.

## Scientific integrity

- Version research code, protocols, results, references and reproducibility documentation. Store machine administration records separately from this repository.
- Every result must trace to an artifact and execution record. Prepared assets and short runtime probes are not completed benchmarks.
- Preserve all negative and partial results. Do not omit invalid genuine references from unconditional denominators.
- Keep model weights, private watermark keys and large image assets outside Git; publish manifests and hashes where appropriate.
- Distinguish ideal independent-bit tail probabilities from empirical or multiuser false-positive rates.
- Keep private verifier inputs out of baseline test processes. Record the actual isolation boundary rather than assuming it.

## Server and compute

- Workspace: `/data0/liyanlei/watermark_research` on SSH alias `249`.
- All future research experiments, including CPU scoring and GPU runs, must execute on server 249 only. Do not use the user's PC as a compute fallback or download model weights there. Local code editing, documentation and file management remain allowed.
- Primary Python: `/data0/liyanlei/watermark_research/envs/semantic_forgery_py310/bin/python`.
- Reuse local models and cache. Set `HF_HOME=/data0/liyanlei/watermark_research/cache/huggingface`, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for offline runs.
- Use at most three GPUs concurrently. Current user-selected server GPUs are 7, 5, and 4; use only these, in that order.
- Run the full published implementation and update schedule; do not truncate baselines to save time.
- Check occupancy before launching a GPU job. Use only idle devices, bind by current UUID and always exclude GPU index 6.
- Never terminate another user's process or use their occupied GPU. Under the current device selection, wait if GPUs 7, 5 and 4 are occupied; do not switch devices automatically.
- Do not modify unrelated directories under `/data0/liyanlei/`, including `ceramic_watermark`, `FontGuard_repro`, `genstego_sota_*`, `natural_image_512bit_watermark_poc`, `official_hidden_compare` and `trajectory_orth_20261103`.

- Do not continuously poll experiment progress. The full-budget run has an event-driven completion supervisor; continue independent research while it runs.
