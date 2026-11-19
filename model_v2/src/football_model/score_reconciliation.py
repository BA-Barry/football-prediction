"""Reconcile a supplied 1X2 forecast with an existing score distribution.

Uses the H/D/A region scaling already used by probability.blend_market. It
learns nothing and preserves P(score | outcome); it does not establish that
the supplied class forecast is better or available at the prediction time.
"""
from collections.abc import Sequence

import numpy as np

from .vendor.penaltyblog_grid import FootballProbabilityGrid


def reconcile_score_probabilities(
    grid: FootballProbabilityGrid, target_hda: Sequence[float],
) -> FootballProbabilityGrid:
    """Scale existing home/draw/away regions and recompute both goal means.

    Matrix rows are home goals, columns away goals. ``target_hda`` is ordered
    home/draw/away and must contain three finite, strictly positive normalized
    probabilities. Every source region must already have positive support;
    unsupported scorelines are never fabricated. Normalization drift up to
    1e-10 is accepted, then the target is divided by its sum for roundoff only.
    The original grid is not mutated and stored source means are not reused.
    """
    try:
        source = np.asarray(grid.grid, dtype=float)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError('Need an existing numeric score probability grid') from error
    if (source.ndim != 2 or not source.size or not np.isfinite(source).all()
            or np.any(source < 0)
            or not np.isclose(source.sum(), 1., rtol=0, atol=1e-10)):
        raise ValueError('Source score probabilities must be finite, nonnegative and normalized')
    try:
        target = np.asarray(target_hda, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('Need three numeric home/draw/away probabilities') from error
    if (target.shape != (3,) or not np.isfinite(target).all()
            or np.any(target <= 0) or np.any(target > 1)
            or not np.isclose(target.sum(), 1., rtol=0, atol=1e-10)):
        raise ValueError('Target home/draw/away probabilities must be positive, finite and normalized')
    target = target / target.sum()
    home, away = np.indices(source.shape)
    masks = (home > away, home == away, home < away)
    current = np.asarray([source[mask].sum() for mask in masks], dtype=float)
    if np.any(current <= 0):
        raise ValueError('Cannot reconcile an outcome with zero structural support')
    reconciled = source.copy()
    for mask, desired, existing in zip(masks, target, current, strict=True):
        # Divide first to avoid overflow of desired/existing for tiny regions.
        reconciled[mask] = (source[mask] / existing) * desired
    if (not np.isfinite(reconciled).all() or np.any(reconciled < 0)
            or not np.isclose(reconciled.sum(), 1., rtol=0, atol=1e-10)):
        raise ValueError('Reconciled score probabilities are not finite and normalized')
    return FootballProbabilityGrid(
        reconciled,
        float((reconciled * home).sum()),
        float((reconciled * away).sum()),
        normalize=False,
    )
