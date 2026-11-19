"""Mean-orthogonal diagnostics, not a corrected predictive distribution.

All projections are conditional on the supplied independent-Poisson means.
They remove the event residual's linear projection on the two mean residuals;
they do not remove arbitrary mean error, fit a new model, or estimate a true
variance/covariance. Callers remain responsible for forecast/source chronology.
"""

import math

from scipy.stats import poisson, skellam

from .goal_rate_diagnostics import row_statistics


EVENTS = ("home_zero", "away_zero", "draw", "zero_zero", "one_one", "total_four_plus")
STAT_FIELDS = (
    "home_rate", "away_rate", "home_goals", "away_goals",
    "home_residual", "away_residual", "home_dispersion_residual",
    "away_dispersion_residual", "mean_centered_cross_product",
) + tuple(f"{event}_{suffix}" for event in EVENTS for suffix in (
    "probability", "residual", "home_coefficient", "away_coefficient",
    "mean_orthogonal_residual"))


def _finite(value, label):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Nonfinite shape diagnostic: {label}") from error
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite shape diagnostic: {label}")
    return value


def _probability(value, label):
    value = _finite(value, label)
    if not 0 <= value <= 1:
        raise ValueError(f"Invalid shape diagnostic probability: {label}")
    return value


def _cell_probability(k, l, home, away):
    # Full-support log formula, never an entry of a finite score grid. The caller
    # has validated integer scores via the existing goal-rate diagnostics.
    return _probability(math.exp(
        k * math.log(home) + l * math.log(away) - home - away
        - math.lgamma(k + 1) - math.lgamma(l + 1)), "score cell")


def _cell_projection(k, l, home, away, probability=None):
    p = _cell_probability(k, l, home, away) if probability is None else probability
    # Algebraically c_H=(k-home)*p/home and c_A=(l-away)*p/away.
    # The shifted-pmf identity avoids division by a tiny mean and preserves a
    # representable coefficient when the unshifted cell probability underflows.
    ch = (_cell_probability(k - 1, l, home, away) if k else 0.) - p
    ca = (_cell_probability(k, l - 1, home, away) if l else 0.) - p
    return p, ch, ca


def _projection(event, base, cell=None):
    home, away = base["home_rate"], base["away_rate"]
    if event == "cell":
        p, ch, ca = _cell_projection(*cell, home, away)
    else:
        p = base[f"{event}_probability"]
        if event == "home_zero":
            ch, ca = -p, 0.
        elif event == "away_zero":
            ch, ca = 0., -p
        elif event == "draw":
            ch = _probability(skellam.pmf(-1, home, away), "draw home shift") - p
            ca = _probability(skellam.pmf(1, home, away), "draw away shift") - p
        elif event == "total_four_plus":
            ch = ca = _probability(poisson.pmf(3, home + away), "total four plus coefficient")
        else:
            target = (0, 0) if event == "zero_zero" else (1, 1)
            p, ch, ca = _cell_projection(*target, home, away, probability=p)
    return {"probability": _probability(p, event),
            "home_coefficient": _finite(ch, f"{event} home coefficient"),
            "away_coefficient": _finite(ca, f"{event} away coefficient")}


def event_projection(event, rates, *, score=None):
    """Return P(event), Cov_Pois(I,H)/mu_H and Cov_Pois(I,A)/mu_A.

    The six named EVENTS require score=None. ``event='cell'`` accepts any pair
    of nonnegative integers, with no maximum-score cutoff. Underflow to a finite
    zero is retained; nonfinite/invalid evaluations fail, not clip or fall back.
    """
    if not isinstance(event, str) or event not in (*EVENTS, "cell"):
        raise ValueError("Unknown shape event")
    if event != "cell" and score is not None:
        raise ValueError("Only a cell event accepts a target score")
    base = row_statistics({"actual_score": score if event == "cell" else (0, 0)}, rates)
    cell = (base["home_goals"], base["away_goals"]) if event == "cell" else None
    return _projection(event, base, cell)


def row_shape_statistics(row, rates):
    """Diagnose one full score using untruncated raw means and no new fitting.

    D=((Y-mu)^2-Y)/mu has expectation
    [Var_true(Y)-m_true+(m_true-mu)^2]/mu. If the true conditional outcome is
    Poisson(m_true), mean error alone cannot make that population expectation
    negative. A finite sample, selected coverage or parameter uncertainty still
    cannot establish an alternative family as correct or best.

    For each event save the raw I-p and z=I-p-c_H*(H-mu_H)-c_A*(A-mu_A).
    Under the stated independent-Poisson model, E(z)=Cov(z,H)=Cov(z,A)=0.
    Away from that model this is a local linear projection, not a guarantee
    of no confounding. The unstandardized cross product includes mean errors;
    it is descriptive, not an estimated true covariance or correlation.
    """
    base = row_statistics(row, rates)
    home, away = base["home_rate"], base["away_rate"]
    h, a = base["home_goals"], base["away_goals"]
    rh, ra = base["home_residual"], base["away_residual"]
    try:
        # This is the specified D, computed without first squaring a potentially
        # huge unscaled residual or cancelling the old Pearson statistic's -1.
        dh = (rh / math.sqrt(home)) ** 2 - h / home
        da = (ra / math.sqrt(away)) ** 2 - a / away
        result = {key: base[key] for key in (
            "home_rate", "away_rate", "home_goals", "away_goals", "home_residual", "away_residual")}
        result.update({"home_dispersion_residual": dh, "away_dispersion_residual": da,
                       "mean_centered_cross_product": rh * ra})
        for event in EVENTS:
            projection = _projection(event, base)
            residual = base[f"{event}_residual"]
            ch, ca = projection["home_coefficient"], projection["away_coefficient"]
            result.update({f"{event}_{key}": value for key, value in projection.items()})
            result[f"{event}_residual"] = residual
            result[f"{event}_mean_orthogonal_residual"] = math.fsum((residual, -ch * rh, -ca * ra))
    except (OverflowError, FloatingPointError) as error:
        raise ValueError(f"Nonfinite shape diagnostic: {error}") from error
    for key, value in result.items():
        _finite(value, key)
    return result
