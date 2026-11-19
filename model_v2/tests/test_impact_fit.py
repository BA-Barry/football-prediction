"""Synthetic software checks only; not football out-of-sample validation."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from football_model.impact_fit import StintObservation, fit_player_effects


START = datetime(2025, 1, 1, tzinfo=timezone.utc)
CUTOFF = START + timedelta(days=400)


def _stint(index=0, **changes):
    kickoff = START + timedelta(days=index)
    values = dict(
        match_id=f"match-{index}",
        kickoff=kickoff,
        result_available_at=kickoff + timedelta(hours=3),
        offset_asof=kickoff - timedelta(hours=1),
        reference_asof=START - timedelta(days=1),
        duration_minutes=45.0,
        home_players={"home": 1.0},
        away_players={"away": 1.0},
        home_reference={"home": 1.0},
        away_reference={"away": 1.0},
        base_home_lambda=1.4,
        base_away_lambda=1.0,
        home_target=0.7,
        away_target=0.5,
    )
    values.update(changes)
    return StintObservation(**values)


def _synthetic_rotation(n=320):
    """Exact conditional xG means identify centred effects without RNG noise."""
    rng = np.random.default_rng(718)
    truth = {
        "h-a1": (0.30, 0.00), "h-a2": (-0.30, 0.00),
        "h-d1": (0.00, 0.25), "h-d2": (0.00, -0.25),
        "a-a1": (0.20, 0.00), "a-a2": (-0.20, 0.00),
        "a-d1": (0.00, 0.15), "a-d2": (0.00, -0.15),
    }
    home_reference = {p: 0.5 for p in truth if p.startswith("h-")}
    away_reference = {p: 0.5 for p in truth if p.startswith("a-")}
    rows = []
    for index in range(n):
        home = {f"h-a{rng.integers(1, 3)}": 1.0, f"h-d{rng.integers(1, 3)}": 1.0}
        away = {f"a-a{rng.integers(1, 3)}": 1.0, f"a-d{rng.integers(1, 3)}": 1.0}
        score, man = rng.integers(-1, 2, size=2)
        minutes = float(rng.choice([15, 30, 45, 60]))
        home_attack = sum(truth[p][0] for p in home)
        away_attack = sum(truth[p][0] for p in away)
        home_defence = sum(truth[p][1] for p in home)
        away_defence = sum(truth[p][1] for p in away)
        context = -0.08 * score + 0.30 * man
        rows.append(_stint(
            index,
            duration_minutes=minutes,
            home_players=home, away_players=away,
            home_reference=home_reference, away_reference=away_reference,
            home_score_difference=float(score), home_man_difference=float(man),
            home_target=minutes / 90 * 1.4 * np.exp(home_attack - away_defence + context),
            away_target=minutes / 90 * np.exp(away_attack - home_defence - context),
        ))
    return rows, truth


def test_joint_poisson_recovers_centred_log_rates_and_context():
    rows, truth = _synthetic_rotation()
    fit = fit_player_effects(rows, cutoff=CUTOFF, response_kind="xg", ridge=0.01,
                             context_ridge=0.01)
    for player, expected in truth.items():
        assert fit.effects[player] == pytest.approx(expected, abs=0.005)
    assert fit.context_effects["own_score_difference"] == pytest.approx(-0.08, abs=0.005)
    assert fit.context_effects["own_man_difference"] == pytest.approx(0.30, abs=0.005)
    assert fit.n_matches == len(rows)
    assert fit.n_stints == len(rows)
    assert fit.response_kind == "xg"


def test_positive_defence_reduces_opponent_rate_and_replacement_is_a_gap():
    rows, _ = _synthetic_rotation()
    fit = fit_player_effects(rows, cutoff=CUTOFF, response_kind="xg", ridge=0.01)
    better = fit.effects["h-d1"][1]
    replacement = fit.effects["h-d2"][1]
    assert better > replacement
    # Half of a match moved from the stronger defender to the weaker one.
    opponent_log_change = -0.5 * (replacement - better)
    assert opponent_log_change > 0
    assert np.exp(opponent_log_change) > 1


def test_exposure_is_applied_once_and_reference_lineup_has_zero_effect():
    rows = [_stint(i, duration_minutes=minutes,
                   home_target=1.4 * minutes / 90,
                   away_target=minutes / 90)
            for i, minutes in enumerate([15.0, 30.0, 45.0, 90.0])]
    fit = fit_player_effects(rows, cutoff=CUTOFF, response_kind="xg")
    assert fit.effects == {"away": (0.0, 0.0), "home": (0.0, 0.0)}
    assert fit.context_effects == {"own_score_difference": 0.0, "own_man_difference": 0.0}


def test_larger_ridge_shrinks_the_joint_effects():
    rows, _ = _synthetic_rotation(100)
    weak = fit_player_effects(rows, cutoff=CUTOFF, response_kind="xg", ridge=0.01)
    strong = fit_player_effects(rows, cutoff=CUTOFF, response_kind="xg", ridge=10000)
    weak_norm = np.linalg.norm(list(weak.effects.values()))
    strong_norm = np.linalg.norm(list(strong.effects.values()))
    assert strong_norm < weak_norm / 10


@pytest.mark.parametrize("field", ["result_available_at", "kickoff"])
def test_future_or_cutoff_time_outcomes_are_rejected(field):
    row = _stint(**{field: CUTOFF})
    with pytest.raises(ValueError, match="strictly before cutoff"):
        fit_player_effects([row], cutoff=CUTOFF, response_kind="xg")


@pytest.mark.parametrize("field", ["offset_asof", "reference_asof"])
def test_future_offsets_and_references_are_rejected(field):
    row = _stint(**{field: START + timedelta(minutes=1)})
    with pytest.raises(ValueError, match="must not be later than kickoff"):
        fit_player_effects([row], cutoff=CUTOFF, response_kind="xg")


def test_naive_timestamps_are_rejected():
    row = replace(_stint(), offset_asof=START.replace(tzinfo=None))
    with pytest.raises(ValueError, match="timezone-aware"):
        fit_player_effects([row], cutoff=CUTOFF, response_kind="xg")


def test_fractional_targets_need_explicit_quasi_likelihood():
    with pytest.raises(ValueError, match="fractional targets"):
        fit_player_effects([_stint()], cutoff=CUTOFF)
    integer_row = _stint(home_target=1, away_target=0)
    assert fit_player_effects([integer_row], cutoff=CUTOFF).response_kind == "goals"


@pytest.mark.parametrize("changes, message", [
    ({"duration_minutes": 0}, "positive"),
    ({"base_home_lambda": float("inf")}, "positive"),
    ({"home_target": -1}, "nonnegative"),
    ({"home_players": {"home": 1.2}}, "fraction"),
    ({"away_players": {"home": 1.0}}, "both teams"),
])
def test_invalid_stint_inputs_fail_explicitly(changes, message):
    with pytest.raises(ValueError, match=message):
        fit_player_effects([_stint(**changes)], cutoff=CUTOFF, response_kind="xg")
