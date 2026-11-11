# Operating rules for semantic watermark defense research

These rules apply to this repository and `/data0/liyanlei/watermark_research` on SSH host `249`.

## Objective and current state

- Develop a publishable defense with benefits at matched absolute legitimate-image detection rates, documented image quality and computation budgets.
- Read `reports/full_imprint_20261111/report.md` for complete baseline findings and `reports/geometry_calibration_20261111/runtime_audit.md` for the active runtime-gated calibration workflow. Also read `reports/blind_geometry_development_20261110/report.md` and `method.md` in that directory for the latest blind geometry and null-calibration evidence; read `reports/crop_transport_diagnostic_20261110/report.md` for the oracle mechanism diagnostic and `reports/sembind_official_audit_20261110/report.md` for completed development results and `project_state.json` for remaining validation.
- The statistical content verifier has no demonstrated advantage over calibrated BA on the completed development Reprompt comparison. Independent confirmation remains unscored.

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
