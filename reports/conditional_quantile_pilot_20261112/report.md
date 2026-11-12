# Content-coupled defense pilot

All240 genuine-query scores independently recomputed. Six new neutral prompts, four embedding arms, ten conditions, full50-step generation/inversion and100-step decoder optimization.

| Arm | Accepted | Mean BA | Binding mismatch | LPIPS to auxiliary |
|---|---:|---:|---:|---:|
| full | 59/60 | 0.937240 | 0.072266 | 0.828131 |
| anchor_low | 54/60 | 0.888802 | 0.046387 | 0.367322 |
| quantile_coupled | 48/60 | 0.850326 | 0.096517 | 0.643851 |
| quantile_independent | 54/60 | 0.869596 | 0.061686 | 0.804413 |

Prespecified feasibility gates: {'binding_vs_independent': False, 'genuine_retention': False, 'clean_retention': True, 'reference_fidelity': True}. Advance: False.
LPIPS is a paired auxiliary-image fidelity diagnostic, not FID or a general quality guarantee. Carrier budgets and matched controls are specified in protocol.json. Shared and independent variants isolate coupling; full-carrier comparisons also change the encoding geometry.

## Research decision

The coupled quantile carrier fails both the binding improvement and genuine-retention gates. Relative to the matched independent carrier, it improves paired preview LPIPS but worsens semantic mismatch and loses six accepted genuine queries. All six clean candidate references pass. The results do not support treating visual similarity to the preview as sufficient evidence of stable semantic binding. The published-forgery branch was not executed; no forgery performance is claimed for this candidate.

Do not expand or retune this frozen carrier. Any subsequent defense hypothesis must distinguish semantic-binding errors from modulation-channel errors on these development artifacts, then use a separately frozen protocol and new sources for validation.

## Failure decomposition on saved full inversions

| Carrier | Actual accepts | Enrollment-code diagnostic | Noiseless-channel diagnostic | Mean carrier BER |
|---|---:|---:|---:|---:|
| full | 59/60 | 60/60 | 60/60 | 0.306273 |
| anchor_low | 54/60 | 58/60 | 60/60 | 0.335185 |
| quantile_coupled | 48/60 | 55/60 | 60/60 | 0.384613 |
| quantile_independent | 54/60 | 56/60 | 60/60 | 0.379144 |

For the coupled carrier, magnitude BER is0.425372 versus sign BER0.343854. Magnitude-only actual/oracle accepts23/29 of60; sign-only51/55 of60. Both channel weakness and semantic drift contribute. These counterfactuals use information unavailable to deployment and do not form detectors or new image experiments. See[failure_decomposition.json](failure_decomposition.json).

The next frozen experiment uses the originalfull carrier and tests[shared semantic evidence](../shared_semantic_evidence_20261112/plan.md), with query uncertainty, matched decoder controls and new test sources.
