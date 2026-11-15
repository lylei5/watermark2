"""Monotone detector-score calibration; no attack logic or model inference."""
import math
import numpy as np


def prc_detect_score(decoding_key, posteriors):
    """Export the author's Detect statistic, with threshold sqrt(log(1/fpr)).

    This is an evaluation adapter. Compare its decisions against unchanged
    native Detect before using a new source version or posterior convention.
    """
    _, parity, otp, _, noise_rate, _, _, _, t = decoding_key
    values = (1 - 2 * noise_rate) * (1 - 2 * np.array(otp, dtype=float)) * posteriors.numpy(force=True)
    count = parity.shape[0]
    products = np.prod(values[parity.indices.reshape(count, t)], axis=1)
    plus = np.log((1 + products) / 2)
    minus = np.log((1 - products) / 2)
    log_product = plus + minus
    constant = .5 * np.sum(np.power(plus, 2) + np.power(minus, 2) - .5 * np.power(log_product, 2))
    centered = plus.sum() - .5 * log_product.sum()
    if not np.isfinite(constant) or not np.isfinite(centered) or constant < 0:
        raise ValueError('Native PRC statistic is non-finite or has a negative variance term')
    if constant == 0:
        # Native Detect accepts centered >= 0 at every nominal FPR in (0,1).
        return math.inf if centered >= 0 else -math.inf
    return float(centered / np.sqrt(2 * constant))


def calibrate_upper_tail(scores, target_fpr, minimum_threshold=-math.inf, fixed_accept=None):
    """Least restrictive floating-point threshold with empirical FPR <= target.

    Accept iff fixed_accept OR score >= threshold. Ties are never randomized.
    The caller must keep calibration and validation sources disjoint.
    A fixed branch exceeding the budget is reported as unattainable.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or np.isnan(values).any():
        raise ValueError('Expected a nonempty one-dimensional score array without NaNs')
    if not 0 < target_fpr < 1 or math.isnan(minimum_threshold):
        raise ValueError('Invalid target or minimum threshold')
    fixed = np.zeros(len(values), dtype=bool) if fixed_accept is None else np.asarray(fixed_accept, dtype=bool)
    if fixed.shape != values.shape:
        raise ValueError('Fixed-accept mask does not match score shape')
    budget = math.floor(target_fpr * len(values))
    immutable = fixed | np.isposinf(values)
    if int(immutable.sum()) > budget:
        return dict(status='unattainable', n=len(values), target_fpr=target_fpr,
                    immutable_accepts=int(immutable.sum()), allowed_accepts=budget, threshold=None)
    eligible = values[~fixed]
    # A threshold immediately above a tied score drops the entire tie while
    # retaining larger, as-yet-unseen values. Infinity would reject them too.
    boundaries = np.nextafter(eligible[~np.isposinf(eligible)], math.inf)
    candidates = np.unique(np.r_[minimum_threshold, boundaries, math.inf])
    candidates = candidates[candidates >= minimum_threshold]
    # searchsorted preserves >= at ties without a quadratic threshold scan.
    ordered = np.sort(eligible)
    counts = int(fixed.sum()) + len(ordered) - np.searchsorted(ordered, candidates, side='left')
    choices = np.flatnonzero(counts <= budget)
    if len(choices) == 0:
        raise RuntimeError('No attainable threshold despite a feasible immutable branch')
    index = choices[0]
    threshold = float(candidates[index])
    accepted = fixed | (values >= threshold)
    return dict(status='calibrated', threshold=threshold, comparator='>=', n=len(values),
                accepted=int(accepted.sum()), empirical_fpr=float(accepted.mean()),
                target_fpr=target_fpr, allowed_accepts=budget,
                immutable_accepts=int(immutable.sum()))


def exact_binomial_interval(successes, total, confidence=.95):
    """Clopper-Pearson interval; independent source units must be defined upstream."""
    from scipy.stats import beta
    if total < 1 or successes < 0 or successes > total or not 0 < confidence < 1:
        raise ValueError('Invalid binomial count or confidence')
    tail = (1 - confidence) / 2
    lower = 0. if successes == 0 else float(beta.ppf(tail, successes, total-successes+1))
    upper = 1. if successes == total else float(beta.ppf(1-tail, successes+1, total-successes))
    return [lower, upper]
