# Research state: semantic watermark defense

Updated 2026-11-09. Status: paused at the user's request. Do not start experiments without a new resume instruction.

Read `reports/content_auth_independent_20261109/report.md` and `progress.json` for the latest asset inventory. Read `reports/codeword_binding_20261109/report.md` for completed small-sample and channel results, including robustness regressions. No independent SOTA result is established.

The objective is a publishable defense that improves performance at matched absolute genuine-image detection. Use established defense papers' baseline tests and official implementations with documented parameters. Keep the work to authorized offline image experiments. Do not devise custom attack methods.

Completed new assets: 320 independent COCO val2017 images; 112 auxiliary images; 224 SemBind/CoverLock-GS-adapter references and four plain GS controls; benign query images. Feature extraction and development inversions are partial. Full Imprint151 and final calibration/confirmation have not been completed.

CoverLock-g uses the official checkpoint and preprocessing, with a four-image equivalence check. Its auxiliary-image GS adaptation is not a native post-hoc reproduction. Current SemBind assets use DDIM, not official exact inversion. Resolve these boundaries before comparative paper claims.

The content-first verifier is a tested statistical prototype. Prior work covers message wrapping, semifragile hypothesis tests and semantic payload recovery. A new contribution requires measured advantages over calibrated BA, raw agreement and content-majority controls. Calibration must compare absolute genuine rates, not method-specific conditional retention.

Server root: `/data0/liyanlei/watermark_research`; this round's assets: `runs/content_auth_20261109`. Use only idle GPUs and exclude index 6. Preserve private keys and large model/image assets outside Git. Local and server experiment processes were stopped for this update.
