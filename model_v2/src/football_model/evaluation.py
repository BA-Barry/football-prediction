"""Probability scoring and chronological partitions, separate from betting returns."""

from datetime import datetime
from typing import Sequence
import numpy as np

from .domain import MatchResult, aware


def chronological_partition(records: Sequence[MatchResult], train_end: datetime, valid_end: datetime):
    aware(train_end)
    aware(valid_end)
    if train_end >= valid_end:
        raise ValueError("Validation must follow training")
    records = sorted(records, key=lambda r: (r.fixture.kickoff, r.fixture.match_id))
    # Results published after a boundary must not enter that boundary's fit.
    train = [r for r in records if r.fixture.kickoff < train_end and r.available_at <= train_end]
    valid = [r for r in records if train_end <= r.fixture.kickoff < valid_end and r.available_at <= valid_end]
    test = [r for r in records if r.fixture.kickoff >= valid_end]
    return train, valid, test


def _validated_arrays(probabilities, outcomes):
    p = np.asarray(probabilities, dtype=float)
    raw = np.asarray(outcomes)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise ValueError("Outcomes must be finite integer labels 0, 1 or 2")
    if np.any(raw != np.floor(raw)) or np.any((raw < 0) | (raw > 2)):
        raise ValueError("Unknown or fractional observed outcome")
    y = raw.astype(int)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) != len(y) or not len(y):
        raise ValueError("Need n x 3 probabilities and n observed outcomes")
    if np.any(p < 0) or not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Probability rows must be non-negative and sum to one")
    return p, y


def probability_scores(probabilities, outcomes):
    """outcomes: 0=home, 1=draw, 2=away; Brier here is sum, not divided by 3."""
    p, y = _validated_arrays(probabilities, outcomes)
    truth = np.eye(3)[y]
    observed_p = p[np.arange(len(y)), y]
    return {
        "n": int(len(y)),
        "log_loss": float(-np.log(np.maximum(observed_p, np.finfo(float).tiny)).mean()),
        "brier_sum": float(np.square(p - truth).sum(axis=1).mean()),
        "rps": float((np.square(np.cumsum(p, axis=1)[:, :2] -
                                 np.cumsum(truth, axis=1)[:, :2]).sum(axis=1) / 2).mean()),
        "accuracy": float((p.argmax(axis=1) == y).mean()),
    }


def calibration_bins(probabilities, outcomes, bins=10):
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    p, y = _validated_arrays(probabilities, outcomes)
    output = []
    for c, name in enumerate(("home", "draw", "away")):
        ids = np.minimum((p[:, c] * bins).astype(int), bins - 1)
        for b in range(bins):
            mask = ids == b
            if mask.any():
                output.append({"outcome": name, "bin": b, "n": int(mask.sum()),
                               "mean_probability": float(p[mask, c].mean()),
                               "observed_frequency": float((y[mask] == c).mean())})
    return output
