from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from football_model.domain import Fixture, MatchResult
from football_model.evaluation import calibration_bins, chronological_partition, probability_scores
from football_model.team_model import TeamGoalModel


def synthetic_history():
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(5)
    records = []
    for i in range(60):
        kickoff = start + timedelta(days=i)
        home, away = ("A", "B") if i % 2 else ("B", "A")
        f = Fixture(str(i), home, away, "SYNTHETIC", kickoff)
        records.append(MatchResult(f, int(rng.poisson(1.6)), int(rng.poisson(1.0)),
                                   kickoff + timedelta(hours=2)))
    return records, start


def test_synthetic_team_fit_and_future_results_do_not_change_past():
    history, start = synthetic_history()
    cutoff = start + timedelta(days=45)
    first = TeamGoalModel().fit(history[:45], cutoff)
    with_future = TeamGoalModel().fit(history, cutoff)
    fixture = Fixture("next", "A", "B", "SYNTHETIC", cutoff + timedelta(hours=5))
    a, b = first.predict(fixture, cutoff), with_future.predict(fixture, cutoff)
    assert a.home > 0 and a.away > 0
    assert (a.home, a.away) == pytest.approx((b.home, b.away))
    assert with_future.n_matches == 45
    with pytest.raises(ValueError, match="Unseen"):
        first.predict(Fixture("new", "C", "A", "SYNTHETIC", fixture.kickoff), cutoff)


def test_chronological_partition_and_scoring():
    history, start = synthetic_history()
    train, valid, test = chronological_partition(history, start + timedelta(days=30), start + timedelta(days=45))
    assert (len(train), len(valid), len(test)) == (30, 15, 15)
    perfect = probability_scores(np.eye(3), [0, 1, 2])
    assert perfect["log_loss"] == 0
    assert perfect["brier_sum"] == 0
    assert perfect["rps"] == 0
    assert sum(r["n"] for r in calibration_bins(np.eye(3), [0, 1, 2])) == 9
    with pytest.raises(ValueError, match="sum"):
        probability_scores([[0.2, 0.2, 0.2]], [0])


@pytest.mark.parametrize("labels", [[0.9], ["0"], [float("nan")], [[0]]])
def test_invalid_outcome_labels_not_silently_coerced(labels):
    with pytest.raises(ValueError):
        probability_scores([[1, 0, 0]], labels)
    with pytest.raises(ValueError):
        calibration_bins([[1, 0, 0]], labels)


def test_late_validation_result_cannot_tune_at_earlier_boundary():
    from dataclasses import replace
    history, start = synthetic_history()
    history[44] = replace(history[44], available_at=start + timedelta(days=46))
    _, valid, _ = chronological_partition(history, start + timedelta(days=30), start + timedelta(days=45))
    assert len(valid) == 14
    with pytest.raises(ValueError, match="boolean"):
        Fixture("bad", "A", "B", "SYNTHETIC", start, neutral="False")
    with pytest.raises(ValueError, match="positive integer"):
        calibration_bins([[1, 0, 0]], [0], bins=0)
