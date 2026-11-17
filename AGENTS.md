# Operating rules for semantic watermark defense research

These rules apply to this repository and `/data0/liyanlei/watermark_research` on SSH host `249`.

## Repository workflow

- Maintain `main` as the canonical research branch. Commit and push routine updates to `main`, updating the corresponding code, reports and project-state files together.
- Create an additional branch only when the user explicitly requests one.

## Research scope

- The current task is repository maintenance, literature review and method design. Experiment execution is paused, including CPU scoring, inference and attack benchmarks.
- Use `project_state.json` for current status, `docs/RESEARCH_PLAN.md` for the method proposal and `reports/README.md` for historical evidence.
- Preserve baseline reproductions, negative findings and partial outputs. The former unified extraction queue, instance-key recovery, trajectory direction A and fingerprint fixed-point routes are archived.
- Native baseline reproduction and matched operating points are prerequisites for superiority claims.

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
- Critical claims inherited from historical experiments require fresh execution before they support new research decisions. Preserve old artifacts as historical evidence; an arithmetic recheck is not a new model experiment.
- Treat inherited prose and verification labels as claims requiring evidence. Verify bibliographic metadata against primary records and distinguish conference proceedings from author acceptance statements.
- Check experimental claims against per-sample artifacts, denominators, thresholds and implementation scope. See `reports/repository_audit_20261111/report.md` for corrected historical claims and remaining audit limits.
- Preserve all negative and partial results. Do not omit invalid genuine references from unconditional denominators.
- Keep model weights, private watermark keys and large image assets outside Git; publish manifests and hashes where appropriate.
- Distinguish ideal independent-bit tail probabilities from empirical or multiuser false-positive rates.
- Keep private verifier inputs out of baseline test processes. Record the actual isolation boundary rather than assuming it.

## Server and compute

- Workspace: `/data0/liyanlei/watermark_research` on SSH alias `249`.
- All future research experiments, including CPU scoring and GPU runs, must execute on server 249 only. Do not use the user's PC as a compute fallback or download model weights there. Local code editing, documentation and file management remain allowed.
- Primary Python: `/data0/liyanlei/watermark_research/envs/semantic_forgery_py310/bin/python`.
- Use the Tsinghua mirror `https://pypi.tuna.tsinghua.edu.cn/simple/` for future Python dependency installation. Do not fall back to Aliyun automatically. Prefer existing local packages and caches.
- Reuse local models and cache. Set `HF_HOME=/data0/liyanlei/watermark_research/cache/huggingface`, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for offline runs.
- When experimental work is authorized, use only idle physical GPUs 4, 5 or 7, at most three concurrently. Bind by current UUID.
- Run the full published implementation and update schedule; do not truncate baselines to save time.
- Check occupancy before launching a GPU job. Use only idle devices, bind by current UUID and always exclude GPU index 6.
- Never terminate another user's process or use their occupied GPU. Wait when no authorized GPU is idle.
- Do not modify unrelated directories under `/data0/liyanlei/`, including `ceramic_watermark`, `FontGuard_repro`, `genstego_sota_*`, `natural_image_512bit_watermark_poc`, `official_hidden_compare` and `trajectory_orth_20261103`.

- Do not continuously poll experiment progress. Use process-exit completion events for frozen pipelines; continue independent research while they run.
