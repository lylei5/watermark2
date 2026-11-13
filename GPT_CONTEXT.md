# Research handoff: three native defense baselines

The active task is to implement and validate SemBind, PGID and SEAL as reliable defense baselines. The user confirmed that no extra SemBind author code is available and approved completing the unreleased parts from the paper, with explicit differences. New defense training is deferred. The single authoritative execution queue is [project_state.json](project_state.json); do not duplicate or restart its jobs.

## Implementation contract

Read docs/THREE_BASELINE_REPRODUCTION.md, docs/SEAL_EVALUATION_CONTRACT.md and docs/BASELINE_IMPLEMENTATION_DIFFERENCES.md. Released components, paper-completed code, image-chain validation, published-attack evaluation and full paper numerical reproduction are separate acceptance levels. No superiority claim is established.

SemBind source337c9f9: released masker and exact DPM components, original GS spatial OTP carrier, private HMAC permutation and paper masking. The100-prompt GS/GS-S run uses DPM50, generationCFG7.5, empty-prompt inversionCFG3, inv_order0, decoder_invTrue and100 decoder optimization steps. Preserve thresholds166/167/182 separately. PRC-S uses original PRC512-bit payload, t3, nominal1e-6, sigma0.5 and Detect OR Decode; this union is not an established1e-6 guarantee. The unreleased full binding entry is paper-completed, not author-byte-equivalent.

PGID source dc4ad96:22 Python files have the same executable AST as prior eb7fb0d. Invoke the released run.py unchanged; preserve151 actual updates and checkpoints51/101/151. PGID-F GS has10 completed full-budget cases; PGID-R has10 completed full-budget removal cases: raw0/10, defended10/10 at all51/101/151-update checkpoints. All four carrier CPU roundtrips passed2/2 each. Their image validations and complete paper matrix are not thereby complete.

SEAL source92d31b3: released code generates using the loaded PNDMScheduler and inverts using its own DDIM algebra. The final supplement specifies k1024,b7,tau2.3,nmatch12. Released seed42 and final-paper secret-salt paths are separate runs. The paper path implements Algorithm2 with explicit SHA256 serialization and private device-local RNG choices; author equivalence is not asserted. BLIP2 and caption-embedding weights match upstream LFS hashes. The cached SD2.1 mirror is pinned, but official stabilityai fp16 snapshot equivalence remains unverified.

## Evidence and pending work

reports/three_baselines_20261113/report.md indexes source, configuration, dependency and CPU checks. OriginalGS verification, releasedSEAL100, paperSEAL100 and PGID-R10 are complete. GS-S accepts100/100; releasedSEAL83/100 and paperSEAL91/100, both null0/100. PRC-S v2 failed one GPU idle guard before model load; v3 completed all6sources with the same private inputs and numeric code: PRC andPRC-S each6/6 accepted with exact512-bit payloads; null0/6 each. All numeric workers are frozen. A process-exit postprocessor validates artifacts and emits audited_summary.json; only the first audit event survived, so later completed runs were audited explicitly. Incomplete runs must be reported as incomplete, never as zero acceptance. Read its output only when an event or a concrete next action warrants it.

The100-prompt trajectory generation,300 full inversions,400 decisions and CPU quality measurements are complete. The step35 P5 gate failed on both datasets. Existing diagnostics and failures are indexed in reports/README.md. Previous project flat-GS adapters do not stand in for the originalGS spatial carrier. PGID's native GS remains its published flat carrier; native results cannot be directly ranked as matched-carrier results.

Next steps: follow real completion events, diagnose any runtime failure without altering frozen outputs, audit decisions and denominators, update reports and project_state.json, and commit/push each completed unit. Continue baseline coverage using only unchanged published attacks. TR-S has its own frozen6-source image protocol using original Tree-Ring3015283, sigma1 and private binding. SEAL completed1400 semantic decisions across7 benign conditions, retaining all clean failures. Results and the distinct exhaustive/TPR@1%FPR entry are documented; no claim that the semantic fixed-threshold table matches Figure6. GS++-S and all PGID image backends are not yet verified. Never promote an untested cell to a completed paper reproduction.

## Execution rules

All model experiments and CPU scoring run on249 only. Root:/data0/liyanlei/watermark_research. Only idle GPUs7,5,4, at most three; neverGPU6. The host uses RTX3090 cards with24GiB each. Keep keys and large assets out of Git. No continuous progress polling; use process-exit events and independent work. Keep main canonical and push completed code/protocol/report/state units. Do not send author emails/issues. No new or improved attacks.
