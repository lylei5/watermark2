# Research handoff: content-bound diffusion watermark defense

Authoritative execution state: [project_state.json](project_state.json). Completed evidence: [reports/README.md](reports/README.md). Updated 2026-11-13. Phase1 contract audit is complete; model experiments are being prepared. Latest report: reports/phase1_contract_20261113/report.md.

## Baseline and implementation

Complete native SemBind reproduction is not established. The current component pipeline uses the released semantic network and DPM generation/inversion, with a project GS carrier and private HMAC binding permutation. Generation and inversion use 50 steps, inverse order 0, decoder inversion enabled and 100 decoder optimization steps. Inverse order 0 disables diffusion fixed-point correction and also matches the paper's default setting; this does not establish equivalence of the full pipeline. BA > 181/256 is not aligned with the paper's reported theoretical FPR 1e-6.

Semantic extraction is in `scripts/defense/semantic_code.py`; it has no watermark embedding, unbinding or public permutation interface. The DPM loader is `scripts/defense/sembind_official_audit.py`. Current worker and carrier entry points are mapped in the README. Frozen server runs retain the exact source and protocol hashes of their executed versions. Use those versions for historical replay.

## Latest completed evidence

[Continuous semantic geometry](reports/continuous_geometry_screen_20261113/report.md), run `continuous_geometry_screen_20261113_v2`, completed at 2026-11-12T16:41:17.775563+00:00. All 24 full-budget clean inversions and 120 wrong-identity comparisons were audited. All arms accept 6/6 clean images. Full / quantized / continuous independent / continuous transport mean BA: 1 / 1 / 0.969401 / 0.968750. Main candidate's normalized-logit binding distance 0.322839 exceeds full 0.140400 and independent 0.312098. Binding and BA gates fail; no attack jobs ran. This is a clean feasibility result, not robustness or forgery evidence.

[Conditional transport](reports/semantic_transport_pilot_20261112/report.md): 240 genuine records, 1200 wrong-identity pairs. Full / saturated / transport / shuffled transport accept 53 / 52 / 55 / 57 of 60. Better preview LPIPS did not provide improved binding or retention over the matched control; no Imprint branch ran.

Other completed negative results, calibration, diagnostics and published baselines are indexed in reports/README.md. No candidate establishes superior defense at matched absolute genuine detection. Confirmation sources already inspected must not be reused as independent validation after a method change.

## Next research requirement

The active task follows docs/PHASE1_PROTOCOL.md: paper-specification baseline, detection working point, and observational trajectory semantics. Carrier variants remain on hold. Audit the full official embedding and verification chain, secret material, preprocessing, inference settings and detection working point; freeze the comparison before new independent experiments. Do not describe released components or the project's GS adapter as a complete native reproduction.

## Execution rules

All model experiments, CPU scoring and GPU work run on 249 only. Root: `/data0/liyanlei/watermark_research`. Python: `envs/semantic_forgery_py310/bin/python`. Only idle GPUs 7, 5 and 4, at most three; never GPU 6. Use UUID binding and offline cached assets. Keep keys and large assets outside Git. Do not continuously poll runs or restart completed jobs. Only unchanged, published top-venue attack baselines and full budgets are allowed. Routine code, report and state updates go together to main.
