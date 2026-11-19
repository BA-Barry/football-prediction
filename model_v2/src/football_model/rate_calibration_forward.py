"""Strictly earlier window fits of the two existing Poisson rate scalers.

These helpers change no team coefficients or distribution family. Original
time-out forecasts are validated before fitting or application; the supplied
rows alone cannot independently establish their archive provenance. The runner
retains responsibility for replaying the original saved team coefficients.
"""

from collections.abc import Mapping
from copy import deepcopy
from datetime import date

from .cmp_fit import _iso_date, _window_inputs
from .rate_calibration import METHODS, fit_rate_calibrator
from .rate_calibration_experiment import ARMS, research_tracks as _research_tracks


# A research layer supplied as an input must not be silently stacked, including
# its null-parameter case where forecast equality alone cannot identify it.
RESEARCH_FIELDS = (
    "cmp_nu", "cmp_numerics", "dependence_parameter", "dependence_rho", "dependence_k",
    "dependence_numerics", "sarmanov_theta", "calibration_method",
    "calibration_scales", "base_rates",
)


def _original_rows(rows):
    try:
        records = list(rows)
    except TypeError as error:
        raise ValueError("Need nonempty original prediction rows") from error
    if not records:
        raise ValueError("Need nonempty original prediction rows")
    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError("Every original prediction row must be a mapping")
        if any(field in row for field in RESEARCH_FIELDS):
            raise ValueError("Need original predictions, not stacked research layers")
        for field in ("date", "fit_before_date", "training_max_date"):
            _iso_date(row.get(field), field)
    return records


def fit_window(rows, *, training_start, before_date):
    """Fit both existing closed-form scalers on all supplied earlier matches.

    Invalid, duplicate or out-of-window inputs are errors, never dropped. All
    matches have equal weight; no extra time decay, optimization or selection
    occurs. Zero relevant observed-goal totals retain the original fitter's
    explicit no-positive-MLE error instead of adding an epsilon or fallback.
    """
    start = _iso_date(training_start, "training_start")
    cutoff = _iso_date(before_date, "before_date")
    if not start < cutoff:
        raise ValueError("training_start must be strictly earlier than before_date")
    records, _, _ = _window_inputs(_original_rows(rows), start, cutoff,
        "Rate calibration training must satisfy training_start <= date < before_date")
    return {method: {
        **fit_rate_calibrator(records, cutoff, method),
        "training_start": training_start, "training_end_exclusive": before_date,
    } for method in METHODS}


def _parameters(parameters, before_date):
    if not isinstance(parameters, Mapping) or set(parameters) != set(METHODS):
        raise ValueError("Require both declared rate calibration methods")
    for method in METHODS:
        model = parameters[method]
        if not isinstance(model, Mapping):
            raise ValueError("Each rate calibrator must be a mapping")
        if (model.get("method") != method or model.get("before_date") != before_date
                or model.get("training_end_exclusive") != before_date):
            raise ValueError("Rate calibrator method/cutoff differs from declared window")
        start = _iso_date(model.get("training_start"), "training_start")
        first = _iso_date(model.get("training_min_date"), "training_min_date")
        if not start <= first < _iso_date(before_date, "before_date"):
            raise ValueError("Saved training_start differs from fitted training window")
    first, second = (parameters[method] for method in METHODS)
    for field in ("training_start", "training_min_date", "training_max_date",
                  "n_matches", "training_match_ids", "training_goal_sums", "training_rate_sums"):
        if first.get(field) != second.get(field):
            raise ValueError("Both calibrators must use the same training rows and window")


def research_tracks(rows, parameters, *, before_date):
    """Generate three coherent forecasts with explicit per-window metadata.

    The original source identity, score and market information are copied to
    every arm unchanged. Each candidate recomputes the entire independent
    Poisson score distribution from its adjusted means, rather than editing
    one market vector. Full-support score NLL is analytic, not grid-truncated.
    """
    cutoff = _iso_date(before_date, "before_date")
    _parameters(parameters, before_date)
    records, _, _ = _window_inputs(_original_rows(rows), cutoff, date.max,
        "Rate calibration targets must be on/after before_date")
    # Isolate nested markets and saved forecasts as well as top-level fields;
    # neither producing nor later editing an output may mutate source inputs.
    tracks, statistics, exported = _research_tracks(
        deepcopy(records), parameters, before_date=cutoff)
    for arm in ARMS:
        scales = [1., 1.] if arm == "original" else parameters[arm]["parameters"]["scales"]
        for index, row in enumerate(tracks[arm]):
            row.update({
                "base_rates": list(records[index]["rates"]),
                "calibration_before_date": before_date,
                "calibration_training_start": parameters[METHODS[0]]["training_start"],
                "calibration_method": arm,
                "calibration_scales": list(scales),
                "exact_score_nll": statistics[arm][index]["joint_score_nll"],
            })
    return tracks, statistics, exported
