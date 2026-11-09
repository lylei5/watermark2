"""Genuine-only threshold calibration with an unconditional denominator."""
import math
import numpy as np


def calibrate_absolute(scores, base_accepted, target):
    scores = np.asarray(scores, dtype=float)
    base = np.asarray(base_accepted, dtype=bool)
    if scores.ndim != 1 or scores.shape != base.shape or not len(scores):
        raise ValueError('Expected equally sized, nonempty score and base vectors')
    if not np.isfinite(scores).all() or not 0 < target <= 1:
        raise ValueError('Finite scores and target in (0,1] are required')
    required = math.ceil(target * len(scores))
    eligible = scores[base]
    common = dict(n=len(scores), base_count=int(base.sum()), required_count=required,
                  target_absolute_tpr=target, operator='>=',
                  base_required=True, base_tpr=float(base.mean()))
    if len(eligible) < required:
        return dict(**common, feasible=False, threshold=None,
                    reason='The base detector accepts fewer than the required unconditional count')
    threshold = float(np.sort(eligible)[-required])
    accepted = base & (scores >= threshold)
    return dict(**common, feasible=True, threshold=threshold,
                calibration_count=int(accepted.sum()), calibration_tpr=float(accepted.mean()),
                tie_overshoot=int(accepted.sum())-required)
