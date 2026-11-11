# Research state: content-bound diffusion watermark defense

Updated 2026-11-11. Goal: a publishable defense with matched genuine detection, image quality, computation budgets and credible false-positive calibration. No superiority or SOTA claim is established.

## Fresh execution policy and revalidation

The user requires fresh execution before critical inherited experimental claims support further research. Historical artifacts remain comparisons; arithmetic audit is not a fresh model experiment. The authorized first batch is protocols/defense_revalidation_20261111.json, with scope and limitations in reports/defense_revalidation_20261111/plan.md:100-source SemBind component measurements under clean plus9 unchanged published conditions, and all10 supplied PGID examples with151 updates and full defense computation. This reuses development inputs, not an independent confirmation set. Future pip installation uses Tsinghua exclusively by default. Execution state is recorded in project_state.json.

Fresh execution completed: reports/defense_revalidation_20261111/report.md records100 source images ×10 conditions=1000 newly inferred semantic comparisons, maximum official-versus-adapter Hamming0. All4950 cross-image distances are archived. Server audit validates complete combinations, hashes and aggregates. This supports only component equivalence on this development cohort. Both PGID shards completed151 updates per source. Each checkpoint accepted0/10 after defense versus10/10 before. Genuine controls passed10/10 after PGID (meanBA0.926953125); unwatermarked covers0/10. All70 main rows and40 control rows were audited. Initial control startup failed on a nonzero utilization guard after parent exit; unchanged controls were restarted on idle GPUs and completed. Both completed and independently audited: SemBind/GS18/18 clean records accepted; PGID published conditions90/90 standard versus61/90 defended (10 sources). Read both report.md files. Next frozen event-driven chain: protocols/sembind_published_revalidation_20261112.json -> protocols/imprint_fresh_revalidation_20261112.json. 72 genuine queries/144 inversions; audit then unchanged151-update baseline, all16 checkpoints,144 full verification records. GPUs7/5/4 only. See project_state.json for launch state.

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

Fresh paired nine-condition run completed: all144 scores independently audited; both SemBind/GS and both inversion settings accepted36/36. Full Imprint pipeline PID2154 is active onGPUs7/5/4, generation worker PIDs6400/6399/6398. It has full151-update generation, all16 checkpoints, full100-step decoder optimization and50-step verification, then independent CPU audit.

Candidate1: protocols/conditional_content_likelihood_20261112.json. CPU source-disjoint fit/calibration/test completed: conditional content likelihood30/40 versus calibratedBA39/40, defaultBA40/40. All10 candidate rejections are source3. Channel log loss improves0.6310196755->0.6148969568 but authentication does not. Frozen models/thresholds cannot be retuned on these test results. PID41738 waits for2154 exit and scores every fresh SemBind Imprint checkpoint, reference and cover. Read reports/conditional_content_likelihood_20261112/report.md.

Candidate2: protocols/conditional_anchor_pilot_20261112.json. Six new fixed neutral prompts and private identities, four embedding arms: full SemBind project adapter, low-frequency shared DCT anchor, independently resampled low anchor with identical payload, and random-frequency shared anchor. All identity bits in partitioned arms are semantically bound in8192 payload coordinates,32 repetitions; unbound anchor coordinates never contribute identity detection. Full arm has64 repetitions.30 new generated images and240 full genuine-query inversions. After parent2154 exits successfully, three workers on7/5/4 execute; independent CPU audit recomputes BA, semantic mismatch, and LPIPS gate values. Only if all frozen gates pass,12 unchanged published Imprint tasks and216 full verifier records execute automatically. Otherwise archive the negative result. Read reports/conditional_anchor_pilot_20261112/plan.md; project_state.json records coordinator launch.

Novelty constraints: heteroscedastic watermark detection, Gaussian Shading++ soft decoding, and SLICE partitions are prior work. IRIS2608.03539 discusses final-image binding, but currentarXiv v2 is withdrawn (2026-08-26,Substantial revisions). Treatv1 as related public disclosure, not verified effectiveness. Conditional-anchor dependence structure is an experimental hypothesis; no SOTA or cryptographic guarantee is established.
