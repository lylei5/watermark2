# Content-first verification of semantic-bound diffusion watermarks

This study tests whether the verifier can distinguish inversion-channel noise
from content disagreement using redundancy already present in SemBind. It
retains the SemBind generator and keys; the intervention is the verifier.

## Statistical mechanism

For a claimed GS identity, its encrypted symbol string is known to the
verifier. Removing those signs from the inverted latent and undoing the secret
semantic permutation yields 16 observations of each of the 1024 semantic bits.
Let the query image's bit be the reference for counting disagreements `K_j`.
Under the symmetric independent-channel model,

`P(K_j=k | d,e) = (1-d) Bin(k;16,e) + d Bin(k;16,1-e)`.

Here `d` is content-code disagreement and `e` is per-coordinate channel error.
The original message-majority decision entangles these quantities: scattered
mismatches appear as a single error rate `q=e+d-2ed`. Within-group disagreement
provides another statistic: `r=2e(1-e)`, independent of `d` under the model.
For `e<1/2`, these two moments identify the parameters. The implementation
profiles the grouped likelihood over a fixed `(d,e)` grid instead of relying
on an unstable division near `e=1/2`.

The candidate score uses the least favorable agreement in a profile-likelihood
support interval (`2 delta logL <= 3.841458820694124`), rather than trusting an
optimistic point estimate when content disagreement is not identifiable. The
cutoff follows a one-parameter asymptotic likelihood convention; correlated
inversion errors mean it is not a guaranteed 95% coverage statement.

The decision requires the original GS identity test and a calibrated content
score. Three inexpensive controls are necessary: a stricter calibrated GS BA
threshold, raw symbol agreement, and content-first majority agreement. Extra
likelihood modeling is justified only if it improves the independent tradeoff
over these controls. It is not enough to beat the default GS threshold.

## Falsifiable assumptions and limits

- Diffusion inversion errors may be correlated and heterogeneous. The mixture
  model is an approximation, not a security proof. Record goodness of fit and
  conditional results by estimated channel error.
- Public-feature collisions can defeat every verifier whose code is a
  deterministic function of those features. Secrecy of a downstream head does
  not prove resistance; security claims require established papers' standard
  baseline evaluations.
- Auxiliary-to-final generation drift is a legitimate source of code change.
  It must be included in calibration and reported separately from distortion.
- An additional filter cannot increase genuine acceptance. Compare at matched
  calibration retention and report unconditional, per-distortion test rates.
- Payload recovery and semifragile likelihood tests are established ideas.
  Novelty requires a specific decoder failure, an identifiable statistical
  formulation in the deployed carrier, and measured benefits under published
  baseline tests. No global priority or SOTA claim is established by this design.

## Closest prior work

- [CoverLock](https://arxiv.org/html/2609.32241v2): message wrapping and an
  official pretrained content hash. This study uses its released Giant head,
  with an explicit auxiliary-image GS adaptation boundary.
- [SemBind](https://proceedings.mlr.press/v306/zhang26hj.html): the underlying
  semantic-mask carrier and the principal generation-matched control.
- [Fei, Kwong and Kundur, 2009](https://doi.org/10.1109/TIFS.2009.2015039):
  composite hypothesis testing for semifragile authentication.
- [Cannons and Moulin, 2004](https://doi.org/10.1109/TIP.2004.834660):
  statistical watermark detection with hash side information.
- [Kim et al., 2026 preprint](https://arxiv.org/abs/2609.39623): recoverable
  semantic payloads and channel-aware training.

The experiment protocol, fixed source selection and raw artifacts define the
actual scope. Method development uses only development data. Thresholds use
calibration genuine data. Confirmation scores remain closed until that freeze.
