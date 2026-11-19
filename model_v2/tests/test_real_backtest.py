from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

from football_model.football_data import HistoricalMatch
from football_model.real_backtest import de_vig, evaluate, paired_log_loss_interval, walk_forward


def history():
    records = []
    start = date(2024, 1, 1)
    for i in range(24):
        records.append(HistoricalMatch(
            str(i), "E0", "2024/25", start + timedelta(days=i), "15:00",
            "A" if i % 2 else "B", "B" if i % 2 else "A", "A", "B",
            (i % 4, (i + 1) % 3), (0, 0),
            {"avg_preclosing": (2.0, 3.5, 4.0)}, "local", "official"))
    return records


def test_walk_forward_is_date_blocked_and_future_changes_do_not_change_past():
    records = history()
    first = walk_forward(records, start_date=date(2024, 1, 13))
    changed = [replace(r, full_time=(9, 0)) if r.match_date >= date(2024, 1, 20) else r
               for r in records]
    second = walk_forward(changed, start_date=date(2024, 1, 13))
    assert len(first["predictions"]) == 12
    for a, b in zip(first["predictions"][:7], second["predictions"][:7]):
        assert a["probabilities"] == pytest.approx(b["probabilities"])
        assert a["training_max_date"] < a["fit_before_date"] <= a["date"]
        assert sum(a["probabilities"]) == pytest.approx(1)
        assert sum(a["forecast"]["total_goals"].values()) == pytest.approx(1)


def test_unknown_teams_excluded_and_market_comparison_uses_identical_matches():
    records = history()
    records[-1] = replace(records[-1], home="new_team")
    records[-2] = replace(records[-2], markets={})
    result = walk_forward(records, start_date=date(2024, 1, 13))
    assert len(result["skipped"]) == 1
    assert result["skipped"][0]["reason"] == "unseen_team_no_estimated_prior"
    metrics = evaluate(result["predictions"])
    assert metrics["model"]["n"] == 11
    paired = metrics["markets"]["avg_preclosing"]
    assert paired["market"]["n"] == paired["model_on_identical_matches"]["n"] == 10
    assert paired["paired_log_loss"]["n_matches"] == 10


def test_same_day_results_and_awarded_results_are_not_used():
    records = history()
    records[3] = replace(records[3], goal_fit_exclusion="administratively_awarded_result")
    records[-1] = replace(records[-1], goal_fit_exclusion="administratively_awarded_result")
    result = walk_forward(records, start_date=date(2024, 1, 13))
    assert result["fits"][0]["n_matches"] == 11
    assert len(result["predictions"]) == 11
    assert result["skipped"][0]["reason"] == "administratively_awarded_result"


def test_de_vig_and_bootstrap_reproducibility():
    assert de_vig((2, 3.5, 4)).sum() == pytest.approx(1)
    with pytest.raises(ValueError):
        de_vig((1, 3, 4))
    rows = walk_forward(history(), start_date=date(2024, 1, 13))["predictions"]
    a = paired_log_loss_interval(rows, "avg_preclosing", repeats=20)
    b = paired_log_loss_interval(rows, "avg_preclosing", repeats=20)
    assert a == b
    assert np.isfinite(a["interval"]).all()


def test_reject_duplicate_match_ids_and_empty_metrics():
    records = history()
    with pytest.raises(ValueError, match="Duplicate"):
        walk_forward(records + records[:1])
    assert evaluate([])["model"] is None
