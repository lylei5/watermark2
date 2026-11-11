# Geometry selection and authentication evidence

The verifier takes the recovered query latent, its semantic code, and a claimed watermark identity. No reference image, reference inversion, source auxiliary code or true transformation enters the scoring function. A known claimed identity is part of the verification setting; this is not blind message discovery.

For geometry g, let n(g,j) be the number of observed repetitions of message group j, and c(g,j) the number agreeing with the query-conditioned identity template. Both absent groups and exact ties are erased. A decisive group has outcome 1 when 2c(g,j)>n(g,j) and 0 otherwise. With N decisive groups and S successes, define p_g = P[Binomial(N,1/2)>=S], and p_g=1 if N=0.

The fitting score on a set J of whole groups is the sum of signed coordinate agreements divided by the square root of the observed coordinate count: sum_j(2c(g,j)-n(g,j))/sqrt(sum_j n(g,j)). Zero-count candidates receive score zero through a denominator floor of one. The geometry bank and the group split are fixed before scoring. There is no adaptation using a detection decision.

## What the correction proves under an ideal null

Fix the query, semantic code, all resampled signs and observation masks. Suppose the expected template consists of independent fair signs at all latent coordinates, independently of the query. At any fixed geometry, counts for different message groups depend on disjoint random coordinates. Each group's majority is symmetric under complementing its template signs. Conditional on whether groups are decisive, their majority outcomes remain independent fair bits. Therefore p_g is super-uniform at that fixed geometry.

For an arbitrary data-dependent choice g_hat from M fixed geometries, the event M*p_g_hat <= alpha is contained in the union of events p_g <= alpha/M. The union bound makes min(1,M*p_g_hat) a valid p-value under this ideal null. Independence between geometries is unnecessary. M=251 in this prototype; the chosen pose need not minimize p_g for this argument.

For a fixed partition into two sets of whole message groups, fitting on one set uses no template signs from the other. Conditional on fitting evidence, test-group signs remain independent and fair under the same ideal null. Each held-out p-value is thus valid. Multiplying the smaller of the two directional p-values by two controls their union; independence between the two directions is unnecessary. The partition must keep all 64 repetitions of a message bit together.

This argument is a conditional random-template statement. It does not establish conditional independence for a genuine image, a content-correlated forgery, a fixed deployed key, or an image selected using the verification key. It does not replace cryptographic analysis of the actual key generation or empirical natural-image calibration. The Monte Carlo experiment checks the implemented ideal model, not these additional assumptions.

## Distinguishing the questions

- Wrong identity calibration asks about chance agreement under an explicit key/image sampling model.
- Published forgery evaluation asks whether a structured candidate is falsely attributed to an identity. Its latent and semantic content may violate the ideal null.
- Genuine robustness asks whether content-preserving transformations retain adequate decisive evidence.

A low model p-value alone cannot resolve all three. The development results support improved genuine crop recovery; stricter identity-only detection already rejects the existing Reprompt candidates. A defense superiority claim requires independent evidence under all three questions at comparable operating points.

## Reproduction

From the repository root with the authorized sibling assets available:

```powershell
python scripts/defense/test_blind_geometry_verifier.py
python scripts/defense/evaluate_blind_geometry.py --root ..\independent_research --output <fresh-output-directory>
python scripts/defense/check_geometry_null.py --root ..\independent_research --output <fresh-null-output-directory>
```

Both experiments refuse to overwrite an existing frozen protocol. They use CPU tensors and four CPU threads, without allocating a GPU. Preserve source and input bytes when checking the recorded SHA-256 hashes. Timing excludes the upstream generation, inversion and feature extraction.
