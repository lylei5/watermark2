# Research state: content-bound diffusion watermark defense

Updated 2026-11-10. Objective: a publishable defense with matched absolute genuine detection, image quality and computation budgets.

Read reports/sembind_official_audit_20261110/report.md and the prior content_verifier report/method. Completed: 296 DDIM genuine development queries and 36 standard CVPR2025 Reprompt candidates; plus a separate DPM/fp32 component audit (8 images, 18 inversion settings). No experiments remain running. Independent calibration and confirmation decisions remain unscored.

SemBind-DDIM accepts 16/16 genuine images in each of nine noncrop conditions, 0/16 center crops, and 1/16 Reprompt candidates. Calibrated BA and content verification both accept 0/16 at matched development genuine count 144/160. All 144 noncrop genuine BAs exceed the single default-accepted candidate BA. No superiority or SOTA result is established.

The content/channel mixture estimates actual rates accurately on noncrop genuine images and Reprompt candidates. At inversion noise near one-half, content is unidentifiable. Content disagreement d spans [0,1], with regression tests for d>0.5. Likelihood support is not a guaranteed confidence interval under correlated errors.

The CoverLock official-head auxiliary-image GS adapter accepts only 10/16 clean references. Its 0/16 Reprompt acceptance cannot establish superiority at unequal genuine utility. It is not native CoverLock. The development comparison uses DDIM. Separate DPM/fp32 normal-image closure passed all 16 blind settings, but GS binding and the environment remain project adaptations; this is not full paper reproduction.

New methods must beat calibrated BA, raw agreement and content majority at equal absolute genuine utility. Priorities are official baseline closure and a defensible mechanism addressing output-content consistency or geometric channel alignment. Message wrapping, regional semantics, soft decoding and statistical authentication have prior work.

Use only published top-venue attack baselines with documented parameters on authorized offline assets. Do not devise or improve attacks. Keep verifier secrets outside baseline processes; distinguish process separation from OS access isolation. Production ceramic-system changes are outside this research repository.

Local assets: sibling independent_research and assets directories. Server: /data0/liyanlei/watermark_research on SSH alias 249. Use only idle GPUs, always exclude index 6, bind by current UUID. Completed GPU work used local RTX 4070 Laptop; server GPU use was zero.

The 80 added distortion queries are complete. Historical 30-pair Imprint CLI budget is unverified; a 2-step probe is not a full benchmark. Latest results are local; SSH to 249 closes at port 22, so new result upload is pending.
