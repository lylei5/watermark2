# Research state: content-bound diffusion watermark defense

Updated 2026-11-12. Goal: a publishable defense with matched genuine detection, image quality, computation budgets and credible false-positive calibration. No superiority or SOTA claim is established.

## Decision-region embedding experiment

Conditional Gaussian transport started2026-11-12T15:29:43Z on249 after verifying idle7/5/4. Coordinator15104, workers15111/15112/15113. Prelaunch commitda91c9d; protocols/semantic_transport_pilot_20261112.json SHA256df38260a3811a858c10c02c1af4287ec145515108842ca612891d515de95b464. Six new sources, full/saturated/transport/shuffled_transport arms,240 genuine records. Mapping Phi^-1((Phi(z)+target)/2) preserves full64/64 target matches; same-sign permutation control preserves the exact conditional amplitude multiset. Unit checks passed64 identity roundtrips. Report reports/semantic_transport_pilot_20261112/plan.md. All gates required before24 full151-update Imprint jobs and432 verifications. No model result yet. Completion is tracked via process-exit event.

Decision-region experiment completed2026-11-12T15:28:40Z:240 full genuine inversions and1200 wrong identities audited. Full59/60, saturated55/60, minimal5644/60, random5654/60; all clean6/6 and wrong0/300. Reference binding mismatch .063151/.066732/.155273/.065918; LPIPS .729771/.716744/.585440/.683347. Minimal56 failed genuine and binding gates; zero Imprint jobs launched. Oracle accepts60/59/56/59; minimal56 rescues12 with enrollment code, still misses4. Read reports/decision_region_pilot_20261112/report.md. Prelaunch commit7d2b430; server source snapshot immutable. No superiority. New transport protocol was frozen before these outcomes were read.

## Completed semantic experiments

Both pipelines completed on 2026-11-12. Shared semantic evidence finished at 18:13 CST with all 120 genuine and 216 full Imprint verification records audited. Genuine accepts: default BA 107/120, calibrated BA 101/120, latent-soft 95/120, coordinate-soft 100/120, group-constant 87/120, group-query 85/120. Final Imprint accepts are respectively 1/12, 1/12, 1/12, 1/12, 0/12, 0/12; all checkpoints 15/192, 15/192, 15/192, 15/192, 3/192, 4/192. All six rules accept zero of 12 cover observations and zero of 1320 wrong identities. Twelve jobs each completed the unchanged 151-update schedule. No matched-TPR advantage is established. See reports/shared_semantic_evidence_20261112/report.md.

Semantic feedback finished at 19:09 CST with 180 genuine queries, 18 feedback transitions and 900 wrong-identity decisions audited. Full accepts 58/60, fixed-third-round feedback 56/60, generation-budget-matched consensus 57/60. Binding mismatches: 0.105469, 0.093913, 0.119954. All arms zero wrong accepts (300 each); feedback clean 6/6. Three of seven frozen feasibility gates fail. No conditional Imprint jobs were launched. See reports/semantic_feedback_pilot_20261112/report.md. No experiments are queued or running from these pipelines. Future research requires a new protocol and independent sources.

Read reports/completion_audit_20261112/audit.json for the later read-only artifact and arithmetic audit. It does not constitute fresh model execution. Preserve prior launch snapshots as historical records. No SOTA, convergence or distribution-preservation claim is established.

Replica-moment development screen completed on249 using cached full-budget artifacts. Six-source60-query calibration and previously inspected12-source development data. Primary normalized first moment with fixed stability penalty accepts86/120 genuine and0/12 final Imprint versus default107/120 and1/12; wrong identities0/1320. Ratio without penalty accepts8/1320 wrong identities. Screen gate failed, no GPU expansion. This is new CPU scoring, not new model execution or independent confirmation. Read reports/replica_moments_development_20261112/report.md. No threshold retuning on these sources.

## Fresh execution policy and revalidation

The user requires fresh execution before critical inherited experimental claims support further research. Historical artifacts remain comparisons; arithmetic audit is not a fresh model experiment. The authorized first batch is protocols/defense_revalidation_20261111.json, with scope and limitations in reports/defense_revalidation_20261111/plan.md:100-source SemBind component measurements under clean plus9 unchanged published conditions, and all10 supplied PGID examples with151 updates and full defense computation. This reuses development inputs, not an independent confirmation set. Future pip installation uses Tsinghua exclusively by default. Execution state is recorded in project_state.json.

Fresh execution completed: reports/defense_revalidation_20261111/report.md records100 source images ×10 conditions=1000 newly inferred semantic comparisons, maximum official-versus-adapter Hamming0. All4950 cross-image distances are archived. Server audit validates complete combinations, hashes and aggregates. This supports only component equivalence on this development cohort. Both PGID shards completed151 updates per source. Each checkpoint accepted0/10 after defense versus10/10 before. Genuine controls passed10/10 after PGID (meanBA0.926953125); unwatermarked covers0/10. All70 main rows and40 control rows were audited. Initial control startup failed on a nonzero utilization guard after parent exit; unchanged controls were restarted on idle GPUs and completed. Both completed and independently audited: SemBind/GS18/18 clean records accepted; PGID published conditions90/90 standard versus61/90 defended (10 sources). Read both report.md files. The paired and full-baseline chain has now completed; current evidence and next candidate are described in the defense-method section below.

## Literature and historical evidence audit

Read reports/repository_audit_20261111/report.md and docs/LITERATURE.md. Primary records correct the SemBind, Geometry, semantic-forgery and PAI identifiers, PGID expansion and SEAL venue. PAI is confirmed in the ICLR2026 proceedings. Third-party origins and versions are recorded in THIRD_PARTY.md. PGID historical n=1 and completed n=5 CSVs were rechecked on server: all three defended checkpoints accepted0/1 and0/5 respectively. SemBind masker R2 standard median is8, not the historical upper-middle value9; the statistics function is corrected. Geometry aggregates agree with its saved30-task scores, but do not establish author-implementation equivalence. Historical near-code observations do not prove a collision vulnerability. Full reproduction and superiority remain unestablished.

## Compute and safety

All future research experiments and scoring, including CPU work, run on server 249 only. No local model downloads, experimental execution or GPU fallback. Local code/document editing and file management are allowed. Server root is /data0/liyanlei/watermark_research. Only GPUs 7,5,4, at most three, and only idle devices. Never use GPU6 or terminate other users' jobs. Do not continuously poll; use process-exit events and independent research between meaningful checks.

Only existing published top-conference/top-journal attacks with full documented implementations and schedules are allowed. Never invent or improve attacks. Keep verifier keys outside baseline generation processes. Same-account directory separation is not an OS sandbox. Preserve calibration and confirmation splits.

## Latest completed evidence

Read reports/full_imprint_20261111/report.md and research_direction.md in the same folder. All eight fixed DPM/fp32 Imprint jobs completed 151 updates; all 128 checkpoint images plus 8 references and 8 covers were verified with DPM50 and full 100-step VAE inversion. Four paired sources per method, not a paper-scale reproduction. SemBind carrier/binding is a project adaptation using the released semantic head.

Final update: SemBind references 4/4, covers 0/4, candidates 0/4, mean candidate BA 0.451171875, PSNR 18.41738 dB, LPIPS 0.2782693. GS references 4/4, covers 0/4, candidates 4/4, candidate BA 1, PSNR 18.48838 dB, LPIPS 0.2707488. SemBind rejects all 64 saved candidates; GS accepts 60/64 (all four at every saved checkpoint from update11 onward). Checkpoints repeat the same four sources and are not independent samples.

The frozen geometry verifier, its split variant, identity-only binomial detector, stricter identity-only detector, and default BA make exactly the same decisions on all 144 observations. No added anti-forgery benefit is observed in this complete-baseline experiment. Raw results and trajectory plots are committed in the report folder.

Previous DDIM development evidence: 160 SemBind genuine views, default BA 144/160, query-only geometry 160/160. Existing Reprompt candidates: default 1/16, stricter identity-only and geometry 0/16. Thus the observed gain is crop recovery. Ideal random-template Monte Carlo has 2048 draws; it is not empirical real-key FPR. References: reports/blind_geometry_development_20261110/report.md and method.md. TAG-WM is ICCV2025, not merely a preprint; its current citation has been verified and updated.

## Completed calibration and confirmation

Read reports/geometry_calibration_20261111/completed/report.md and reports/geometry_confirmation_20261111/report.md. Calibration tasks16..47:32 sources,192 genuine queries,6144 identity decisions. Default accepted160/192; both frozen geometry rules192/192; all five rules0/5952 wrong identities. Four supplementary BA thresholds were selected from genuine calibration only and frozen before confirmation.

Confirmation tasks48..111:64 sources,384 queries,24576 decisions. Default/identity/strict identity319/384; primary all-fit382/384; secondary split-fit381/384. All five rules0/24192 wrong identities. All-fit restored63 crop queries and lost0 default-accepted queries. Source-cluster bootstrap95% interval for all-fit-minus-default is15.89–16.67 percentage points, conditional on the frozen calibration and sampled source population. Zero correlated wrong-key observations are not an ultra-low FPR guarantee.

BA controls calibrated at80/90/95/100% targets accepted287/344/359/384 genuine queries on confirmation, with0/1778/4962/23950 wrong identities. Actual confirmation TPRs must be reported, not assumed to match calibration targets. Default BA >181/256 remains unchanged.

All-fit failures: task108 JPEG50 andcrop90; split-fit additionally misses itsJPEG75. Crop selected the correct center scale0.9 pose but evidence remained insufficient. Post-hoc clean-query semantic drift is descriptive only, never supplied to the detector; it does not identify a causal explanation. The confirmation set is consumed. New method tuning requires new independent validation.

Confirmation protocol was published in commit1dee4bc before scoring. Original execution sources are preserved in source_snapshots when generic orchestration was later extended to another query count. Audits validate full50-step caches, source/input/result hashes, complete identity combinations and threshold decisions. No new attacks were created or improved.

## Completed published-condition benchmark

Read reports/gs_published_conditions_20261111/plan.md. Official Gaussian Shading CVPR2024 image_distortion at pinned commit09c678fadc7545acf7be12647ddf2a5e66f6a9dc, unchanged function and all9 parameter settings from official scripts/run.sh.64 SemBind adapter clean images ×9 =576 queries;64 identities each =36864 decisions,36288 wrong. It shares sources with the six-condition confirmation, so is not a second independent source cohort.

Protocol published in commit8b7246d before reading the six-condition confirmation scores. No changed verifier or BA thresholds. Official crop zero-pads outside a random retained window; this is different from crop-and-resize. No composition or optimization of published distortions. This reproduces an official distortion component on the project DDIM adapter, not the full Gaussian Shading system.

Server run: runs/gs_published_conditions_20261111 completed. Default554/576, identity-only556/576, identity with251 penalty552/576, all-fit552/576, split-fit529/576. All five0/36288 wrong identities. All-fit restored0 and lost2 default accepts (task084 officialcrop0.6, task097 salt-and-pepper0.05). Identity-penalty and all-fit agree on all36864 decisions. Source-cluster paired gain−0.35 percentage points,95% interval−0.87 to0. Read reports/gs_published_conditions_20261111/report.md. No universal robustness gain is supported.

## Completed natural-negative evaluation

Protocol protocols/unwatermarked_validation_20261111.json was published before launch in3886df4. All100 existing natural RGB512 COCO images completed DDIM50/batch2 and64-identity scoring:6400 decisions. Default, identity-only, identity with251 penalty, all-fit and split-fit all accepted0/6400; each accepted0/100 queries under any identity. Supplementary BA calibration targets80/90/95/100% accepted0/472/1385/6337 pairs and0/100/100/100 queries. Thresholds remain unchanged.

Read reports/unwatermarked_validation_20261111/report.md. Input/bridge/source/cache hashes, all6400 unique pairs, bridge criteria and summary decisions were audited on server. Existing development assets, not independent confirmation. Shared identities and finite sample size preclude an ultra-low FPR claim. Scoring took54.13 seconds excluding inversion. Run root: runs/unwatermarked_validation_20261111; complete.json and audit.json exist. This historical pipeline completed. Current fresh experiments are tracked in the revalidation section and project_state.json.

The outstanding scientific gap is a defense benefit beyond recovery of the studied center crop-resize condition. Official nine-condition results and full-budget Imprint did not establish broader superiority. Further method development must use development data, preserve published-only attack baselines, and obtain new independent validation sources.

## Numerical compatibility evidence

The isolated envs/ddim_replay_py310 matches all11 recorded package versions; imports and pip check pass. Complete DDIM50 development replay passes the unchanged0.001 gate: clean relative L2 0.000547554, crop90 0.000569046; semantic codes are identical. Earlier failed cross-runtime attempts and source snapshots remain documented in reports/geometry_calibration_20261111/runtime_audit.md and runtime_setup_repair/report.md. Fixed dependencies used Tsinghua after Aliyun connection failure.

The frozen nominal threshold is applied exactly. A1e-12 relative-only validation tolerance allows library rounding when recomputing its binomial tail; it does not relax the detector threshold. All144 historical full-budget BA values match exactly under the checked scorer.

## Runtime assets

The original DPM server snapshot and local DDIM assets have equal effective fp16 UNet weights but different VAE/text-encoder weights. A separate matched DDIM model is retained on server under runs/geometry_calibration_20261110/model; its 3 weight groups,8 configurations and5 semantic assets were audited. Do not attribute cross-pipeline differences only to schedulers. The primary server environment remains available alongside the validated isolated replay environment.

Model weights and private verifier assets reside on server 249. Private identities are under private_eval and remain outside Git; the repository records research code, protocols, results and artifact hashes.

## Defense method experiments, 2026-11-12

Fresh paired nine-condition run and full Imprint both completed and independently audited. Paired groups all36/36 accepted (144records); full151-update Imprint final candidates SemBind0/4, GS4/4, all144 records audited. Reports/imprint_fresh_revalidation_20261112 contains full new evidence. The four-source cohort is development, not independent confirmation.

Conditional content likelihood completed genuine and forgery endpoints:30/40 genuine versus calibratedBA39/40, both0/4 final and0/64 checkpoint Imprint accepts. No added rejection; do not expand this candidate or retune frozen thresholds.

Conditional-anchor pilot completed240 full genuine inversions on six new sources. Full53/60, anchor_low53/60, independent_low54/60, anchor_random49/60. Shared low anchor improved binding mismatch and paired LPIPS but failed the original gates; its forgery branch did not execute. Post-hoc enrollment-code oracle gives59/60 for full/anchor_low, and ordinary linear soft control gives57/60 vs56/60. Oracle is diagnostic only; soft control is not GS++ reproduction. Read reports/conditional_anchor_pilot_20261112/report.md and channel_diagnostic.json.

Completed quantile protocol: protocols/conditional_quantile_pilot_20261112.json, new six sources, four arms,240 full inversions. Low DCT coordinates preserve preview signs and residual half-normal quantile ranks while encoding bound bits in magnitude bins; high coordinates use bound signs and preview magnitudes. All16384 coordinates carry bound bits,64 repetitions with exactly32 magnitude and32 sign observations per message bit. Matched independent-coordinate arm isolates coupling; full and32-repetition anchor_low are controls. Unit checks on249 passed32 identities/64 roundtrips with0 raw-bit errors. Completed results below did not establish a generated-image benefit. Read reports/conditional_quantile_pilot_20261112/report.md.

All generation/inversion50 steps, decoder optimization100. Only idle7/5/4, at most3. Child-exit pipeline independently audits BA, semantic mismatch and LPIPS, then gates a12-job unchanged151-update Imprint evaluation and216 full verification records. No polling or new attacks. Prior original scripts/protocols remain frozen in their server run directories.

Novelty boundary: Gaussian Shading already uses equal-probability intervals and inverse-CDF sampling; GS++ soft decoding, SLICE partitions, ShapeMark structure coding and LAW angular carriers are related prior work. Conditional coupling is a hypothesis, not a novelty/superiority claim. IRIS current record is withdrawn and remains related public disclosure. All critical future claims need fresh independent confirmation.

Encoding-layout audit: the initial quantile launch was aborted before any image scores were inspected because per-message channel counts varied. Original protocol and server artifacts are preserved. The current complete-budget run uses balanced32+32 layout in runs/conditional_quantile_pilot_20261112_v2, with the same fixed sources/identities and no partial-output reuse. See implementation_audit.json.

Current completion verified at2026-11-12 13:49 CST: all240 quantile-pilot genuine records independently audited. Full59/60, anchor_low54/60, quantile_coupled48/60, quantile_independent54/60. Coupled binding mismatch0.096517 versus independent0.061686; paired LPIPS improves0.804413->0.643851, but binding and genuine-retention gates fail. All six clean coupled references pass. No Imprint branch executed; GPUs7/5/4 idle. Read reports/conditional_quantile_pilot_20261112/report.md. Do not expand this carrier or interpret preview fidelity as semantic authentication improvement.

## Shared semantic evidence experiment,2026-11-12

Quantile failure decomposition completed on249: magnitude BER0.425372, sign BER0.343854; coupled actual48/60, enrollment oracle55/60, noiseless-channel60/60. Full actual59/60, oracle60/60. Both physical channel and binding errors matter. Diagnostics are not deployment detectors.

Completed shared-evidence protocol: protocols/shared_semantic_evidence_20261112.json. Full carrier unchanged; shared1024 semantic variables each explain16 carrier observations. Candidate score mean log(1+u_k q_k), where u is grouped latent posterior and q calibrated query logit. Controls: default BA, calibratedBA, GS++-inspired latent-soft BA, coordinate-level semantic-soft BA, and grouped constant-semantic-error score. Gaussian soft decoding, temperature scaling, and this project's prior grouped marginalization are existing work, not novelty claims. Test whether query-dependent uncertainty at the shared-bit level adds actual benefit.

Fit6 full-carrier sources from anchor pilot; calibrate6 different full-carrier sources from quantile pilot. Both prior development. Re-extract120 official logits and check exact code equality. Fit global channel gain/variance plus positive temperature and semantic mismatch rate. Freeze thresholds (fourth-smallest/60,>=,95% target) before generating12 new test sources. DefaultBA remains>181/256. New120 full genuine inversions and1320 wrong-identity pairs, then12 unchanged151-update Imprint jobs and216 full verification records regardless of genuine results. No new attacks. Only idleGPUs7/5/4,max3; process-exit orchestration. All336 new inversions full50-step plus100-step decoder optimization. Private keys remainserver-only. Read plan.md and project_state.json for execution state.

Shared-semantic evaluation completed: all120 genuine and216 forgery-verification records independently audited, with12 unchanged151-update Imprint jobs. All120 development code matches passed; model frozen before test generation. Calibration semantic NLL query0.2383675286 versus constant0.2974550374; probability improvement is not authentication improvement. All calibrated scores57/60 versus defaultBA59/60 on calibration. Complete endpoints and interpretation are in the opening section and reports/shared_semantic_evidence_20261112/report.md.
