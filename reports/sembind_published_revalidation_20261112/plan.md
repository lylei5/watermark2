# Paired published conditions and full-budget baseline protocol

Frozen protocols: `protocols/sembind_published_revalidation_20261112.json` and `protocols/imprint_fresh_revalidation_20261112.json`.

Stage 1 uses all eight newly generated paired SemBind/GS references, four fixed development identities, and nine unchanged published Gaussian Shading conditions: 72 queries. Each query receives encoder-only and full 100-step decoder optimization followed by 50-step released DPM inversion: 144 inversions. Blank prompt, CFG3, float32, inverse order0, fixed strict BA>181/256. Query-derived semantic unbinding is primary. Enrollment and clean-reference masks are diagnostic only. Save every query image, semantic code, latent, and hash.

An event-driven coordinator waits for child exits, independently recomputes all 144 message scores and sign-error measurements on CPU, and writes per-condition and aggregate analysis. It records whether a rejected query becomes accepted under the private enrollment-code diagnostic; this is a decomposition, not a causal claim or deployable defense.

Only after a successful complete audit, Stage 2 starts the unchanged CVPR2025 Imprint baseline on the eight fresh references and the four previously fixed public development covers. It executes all151 updates and saves all16 original checkpoints per job. A separate verifier runs full100-step decoder optimization and50-step DPM inversion on all128 candidates and16 reference/cover records, including CPU image-quality metrics. All144 saved-latent message scores are independently recomputed before final analysis. No attack modification, detector feedback, threshold fitting, checkpoint selection, or early stopping.

The coordinator uses blocking process-exit waits, with one fixed10-second teardown settling interval before each second-stage process. GPU7/5/4 only, at most three; every worker checks that its assigned GPU is idle. Failures leave a failed record and preserve logs. Dependencies are installed from Tsinghua; model assets stay server-side.

These are four reused development identities with fresh image generation and inference. No independent confirmation, native full SemBind reproduction, ultra-low empirical FPR, or SOTA claim. The next method decision must combine the genuine-query decomposition and baseline outcomes and be tested on new independent sources.

Observed launch: coordinator PID32044; workers32096/32097/32098 reached full inversion onGPU7/5/4. Both protocols and source snapshots were pushed to main at8476bd3 before launch. Stage2 is queued and requires a successful complete Stage1 audit.
