"""Ordinary boundary and numerical checks for the independent schedule audit."""

import ast
from datetime import date
import importlib.util
import math
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[1]/"src"/"football_model"/"independent_schedule_audit.py"
_SPEC = importlib.util.spec_from_file_location("standalone_schedule_audit_test", _PATH)
standalone = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(standalone)


def source(mid, day, home="H", away="A", comp="E0", exclusion=None):
    return {"match_id": mid, "date": day, "competition": comp, "home": home, "away": away,
            "actual_score": [1, 0], "goal_fit_exclusion": exclusion}


def test_manual_window_includes_left_endpoint_and_special_but_excludes_today_future_other_leagues():
    rows = [source("old", "2025-01-01"), source("left", "2025-01-02"),
            source("special", "2025-01-10", exclusion="special"), source("today", "2025-01-16"),
            source("future", "2025-01-17"), source("other", "2025-01-10", comp="D1")]
    result = standalone.manual_context(rows, "E0", "H", "A", date(2025, 1, 16))
    assert result["home"]["count"] == result["away"]["count"] == 2
    assert result["home"]["window_match_ids"] == ["left", "special"]
    assert result["home"]["days_since_prior"] == 6
    assert not result["left_window_truncated"]


def test_manual_context_distinguishes_zero_unknown_history_and_left_truncation():
    rows = [source("m1", "2025-01-01"), source("m2", "2025-01-20")]
    old = standalone.manual_context(rows, "E0", "H", "Z", date(2025, 1, 20))
    assert old["home"]["count"] == 0 and old["home"]["has_prior_source_record"]
    assert old["home"]["days_since_prior"] == 19
    assert old["away"]["days_since_prior"] is None
    assert not old["away"]["has_prior_source_record"]
    assert not old["left_window_truncated"]
    assert standalone.manual_context(rows, "E0", "H", "Z", date(2025, 1, 5))["left_window_truncated"]


def test_append_future_and_modify_goals_does_not_change_date_context():
    rows = [source("m1", "2025-01-01"), source("m2", "2025-01-10", "H", "B")]
    expected = standalone.manual_context(rows, "E0", "H", "A", date(2025, 1, 16))
    rows[0]["actual_score"] = [21, 19]
    rows.append(source("future", "2025-01-17"))
    assert standalone.manual_context(rows, "E0", "H", "A", date(2025, 1, 16)) == expected


def test_explicit_goal_rows_exchange_self_and_opponent_counts():
    row = source("m", "2025-01-16")
    context = {"home": {"count": 3}, "away": {"count": 1}}
    h, a = standalone.explicit_features(row, context)
    assert h[standalone.SELF_FEATURE] == a[standalone.OPPONENT_FEATURE] == 3
    assert a[standalone.SELF_FEATURE] == h[standalone.OPPONENT_FEATURE] == 1
    assert h["defense:football-data:E0:A"] == -1
    assert "home:E0" not in a


def test_penalized_gradient_agrees_with_finite_differences_and_unpenalized_intercept():
    coefficients = {"x": .2, "z": -.1}
    examples = [({"x": 1., "z": 2.}, 2, .3), ({"x": 0., "z": -1.}, 0, .7)]
    calculated = standalone.penalized_objective_gradient(.4, coefficients, examples, .01)
    step = 1e-6
    for name in (*coefficients, "(intercept)"):
        values = []
        for direction in (-1, 1):
            beta = coefficients.copy()
            intercept = .4
            if name == "(intercept)":
                intercept += direction*step
            else:
                beta[name] += direction*step
            values.append(standalone.penalized_objective_gradient(intercept, beta, examples, .01)["objective_parameter_dependent"])
        assert calculated["gradient"][name] == pytest.approx((values[1]-values[0])/(2*step), abs=2e-9)
    assert calculated["l2_penalty"] == pytest.approx(.01/2*(.2**2+.1**2))
    rates = [math.exp(.4), math.exp(.5)]
    assert calculated["intercept_gradient"] == pytest.approx(.3*(rates[0]-2)+.7*rates[1])


def test_raw_weight_rescaling_leaves_objective_and_gradient_unchanged():
    examples = [({"x": 1.}, 1, .25), ({"x": 2.}, 3, .75)]
    a = standalone.penalized_objective_gradient(.1, {"x": .2}, examples, .01)
    b = standalone.penalized_objective_gradient(.1, {"x": .2}, [(x, y, w*7) for x, y, w in examples], .01)
    assert a["gradient"] == pytest.approx(b["gradient"])
    assert a["objective_parameter_dependent"] == pytest.approx(b["objective_parameter_dependent"])


def test_training_gradient_uses_each_rows_own_context_and_two_equal_side_weights():
    rows = [source("first", "2025-01-01"), source("second", "2025-01-10")]
    contexts = {r["match_id"]: standalone.manual_context(rows, "E0", "H", "A", date.fromisoformat(r["date"]))
                for r in rows}
    parameters = {"before_date": "2025-01-16", "reference_match_ids": ["first", "second"],
        "half_life_days": 180, "regularization": .01, "intercept": 0.,
        "coefficients": {"attack:football-data:E0:H": 0., "attack:football-data:E0:A": 0.,
            "defense:football-data:E0:H": 0., "defense:football-data:E0:A": 0.,
            "competition:E0": 0., "home:E0": 0., standalone.SELF_FEATURE: .1,
            standalone.OPPONENT_FEATURE: -.1}}
    calculated = standalone.training_gradient(parameters, {r["match_id"]: r for r in rows}, contexts)
    weights = [math.exp(-math.log(2)*days/180) for days in (15, 6)]
    assert calculated["raw_weight_sum_goal_rows"] == pytest.approx(2*sum(weights))
    # Both rates=1; first row count=0, second count=1, home residual=0, away=1.
    assert calculated["gradient"][standalone.SELF_FEATURE] == pytest.approx(weights[1]/(2*sum(weights))+.001)
    assert calculated["gradient"][standalone.OPPONENT_FEATURE] == pytest.approx(weights[1]/(2*sum(weights))-.001)
    assert calculated["intercept_gradient"] == pytest.approx(.5)


def test_export_context_retains_date_order_and_special_exclusion_membership():
    rows = [source("z", "2025-01-02"), source("a", "2025-01-10", exclusion="special")]
    target = source("target", "2025-01-16")
    context = standalone.manual_context(rows, "E0", "H", "A", date(2025, 1, 16))
    exported = standalone.export_context(target, context, {r["match_id"]: r for r in rows})
    assert exported["home"]["previous_14d_ids"] == ["z", "a"]
    assert exported["home"]["special_previous_14d_ids"] == ["a"]
    assert exported["home"]["team"] == "football-data:E0:H"
    assert exported["on_date"] == "2025-01-16"


def test_no_production_package_imports():
    for node in ast.walk(ast.parse(_PATH.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("football_model") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            assert not (node.module or "").startswith("football_model")


def test_independent_full_period_and_csv_replay_catches_numeric_corruption(tmp_path):
    from copy import deepcopy
    import football_model.schedule_experiment as experiment
    from test_schedule_experiment import tracks
    raw_tracks = tracks()
    saved, entries, _ = experiment.analyze_period(raw_tracks, repeats=19)
    rows = entries["original"]
    raw = [{"match_id": row["match_id"], "competition": row["competition"], "date": row["date"],
            "home": row["home"], "away": row["away"], "actual_score": row["actual_score"],
            "goal_fit_exclusion": None} for row in rows]
    raw.append(source("old", "2024-08-01", "A", "B"))
    contexts = [standalone.manual_context(raw, row["competition"], row["home"], row["away"],
                date.fromisoformat(row["date"])) for row in rows]
    forecasts = {arm: [standalone.probability.poisson_forecast(row["rates"]) for row in entries[arm]] for arm in standalone.ARMS}
    stats = {arm: [standalone.independent.statistics(row["actual_score"], row["rates"]) for row in entries[arm]] for arm in standalone.ARMS}
    audit = standalone.independent.Auditor()
    loss = standalone.audit_period(rows, forecasts, stats, contexts, saved, audit)
    assert not audit.errors
    path = tmp_path/"paired.csv"
    experiment.write_csv(path, entries, stats)
    standalone.audit_csv(path, rows, stats, loss, audit)
    assert not audit.errors and sum(audit.checks.values()) > 1000
    broken = deepcopy(saved)
    broken["bootstrap"]["4"]["features"]["schedule_minus_original_one_x_two"]["mean"] += .01
    other = standalone.independent.Auditor()
    standalone.audit_period(rows, forecasts, stats, contexts, broken, other)
    assert any(error["check"].startswith("joint_bootstrap") for error in other.errors)


def test_independent_actual_saved_trial_checks_training_dates_and_gradient():
    from dataclasses import replace
    from football_model.schedule_research import run_trial
    from test_schedule_research import record
    records = [record("h1", "2024-06-01", score=(1, 0)), record("h2", "2024-06-08", score=(0, 2)),
               record("h3", "2024-07-01", score=(3, 1)), record("target", "2024-08-05", score=(1, 1))]
    records = [replace(row, markets={}) for row in records]
    trial = run_trial(records, date(2024, 8, 1), date(2024, 9, 1))
    raw = [{"match_id": r.match_id, "competition": r.competition, "date": r.match_date.isoformat(),
            "home": r.home_label, "away": r.away_label, "actual_score": list(r.full_time),
            "actual_outcome": 0 if r.full_time[0] > r.full_time[1] else 2 if r.full_time[0] < r.full_time[1] else 1,
            "goal_fit_exclusion": None} for r in records]
    contexts = {r["match_id"]: standalone.manual_context(raw, r["competition"], r["home"], r["away"], date.fromisoformat(r["date"])) for r in raw}
    audit = standalone.independent.Auditor()
    _, _, _, _, gradients = standalone.audit_trial(trial, raw, contexts, "schedule", audit)
    assert not audit.errors
    assert len(gradients) == 1 and gradients[0]["n_goal_rows"] == 6
    assert gradients[0]["max_absolute_gradient"] >= 0
