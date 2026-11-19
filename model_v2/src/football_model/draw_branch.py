"""Use a supplied conditional draw signal without relearning the winner ratio.

This transformation fits nothing. The caller is responsible for the signal's
calibration, chronology, class order and suitability for the target fixture.
"""
import math
from numbers import Real

import numpy as np

from .score_reconciliation import reconcile_score_probabilities


def replace_draw_probability(source_hda, draw_probability):
    """Return H/D/A with a new draw mass and the original H:A probability ratio.

All three original probabilities and the supplied draw mass must be strictly
inside (0,1). Only normalization drift <=1e-10 is corrected, never smoothing.
The non-draw sum is computed directly to avoid subtracting a near-one pD.
"""
    raw = np.asarray(source_hda)
    if raw.shape != (3,) or raw.dtype.kind not in 'iuf':
        raise ValueError('Need three numeric ordered home/draw/away probabilities')
    p = raw.astype(float)
    if (not np.isfinite(p).all() or np.any(p <= 0) or np.any(p >= 1)
            or not np.isclose(p.sum(), 1, rtol=0, atol=1e-10)):
        raise ValueError('Source probabilities must be positive, finite and normalized')
    if (isinstance(draw_probability, (bool, np.bool_))
            or not isinstance(draw_probability, Real)
            or not math.isfinite(float(draw_probability))
            or not 0 < draw_probability < 1):
        raise ValueError('Draw probability must be a finite number strictly inside (0,1)')
    p /= p.sum()
    d = float(draw_probability)
    non_draw = p[0] + p[2]
    return ((1-d) * float(p[0]/non_draw), d, (1-d) * float(p[2]/non_draw))


def apply_draw_branch(source_grid, draw_probability):
    """Keep P(score | H/D/A) while assigning the supplied draw probability.

Recompute score expectations and every market from the returned grid. Its
means are not parameters of a new independent-Poisson distribution.
"""
    target = replace_draw_probability(source_grid.home_draw_away, draw_probability)
    return reconcile_score_probabilities(source_grid, target)
