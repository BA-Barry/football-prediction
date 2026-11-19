"""Synthetic checks for a fixed English-tier objective-normalization contrast.

No historical input, network request, actual optimizer, or current forecast is
used here.  The deliberately small fake model exposes fit/order/coverage errors.
"""

from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timedelta
import json
import math

import pytest

from football_model.english_tiers_experiment import METRICS, run_comparison
from football_model.english_tiers_weighting import (
    NEW_ARM,
    run_reference_comparison,
    weight_share,
)
from football_model.football_data import HistoricalMatch


def historical(ident, day, home="Arsenal", away="Chelsea", *,
               competition="E0", season="2025-26", score=(1, 0), exclusion=None):
    return HistoricalMatch(
        ident, competition, season, day, None,
        "england:" + home, "england:" + away, home, away,
        score, None, {}, "synthetic-only.txt",
        "https://example.invalid/synthetic-only", exclusion,
    )


def records():
    return [
        historical("old-epl", date(2024, 9, 1), season="2024-25"),
        historical("old-champ", date(2024, 9, 2), "Hull", "Coventry",
                   competition="E1", season="2024-25"),
        historical("old-champ-second", date(2024, 10, 1), "Coventry", "Hull",
                   competition="E1", season="2024-25"),
        historical("target-promoted-early", date(2025, 8, 9), "Hull", "Arsenal"),
        historical("target-known", date(2025, 8, 15)),
        historical("target-at-next-cutoff", date(2025, 8, 29), "Hull", "Chelsea"),
        historical("target-later", date(2025, 9, 6), "Chelsea", "Hull"),
        historical("current-champ", date(2025, 8, 10), "Coventry", "Hull",
                   competition="E1"),
        historical("excluded-prior", date(2025, 7, 1), competition="E1",
                   exclusion="synthetic abandoned-match exclusion"),
        historical("next-season", date(2026, 8, 8), season="2026-27"),
    ]


class FakeFactory:
    def __init__(self, output_dir, *, accepts_alpha=False, drop_teams=(), fit_error=None):
        self.output_dir = output_dir
        self.accepts_alpha = accepts_alpha
        self.drop_teams = set(drop_teams)
        self.fit_error = fit_error
        self.instances = []
        self.alphas = []
        self.predictions = []

    def __call__(self, *args):
        assert len(args) == int(self.accepts_alpha), "only the new fixed arm receives alpha"
        alpha = args[0] if args else .01
        self.alphas.append(alpha)
        factory = self
        token = f"{self.output_dir.name}-synthetic-{len(self.instances)}"

        class FakeModel:
            def fit(self, history, cutoff):
                if factory.fit_error is not None:
                    raise factory.fit_error
                self.history = list(history)
                self.cutoff = cutoff
                self.alpha = alpha
                assert self.history
                assert all(r.match_date < cutoff for r in self.history)
                assert all(r.goal_fit_exclusion is None for r in self.history)
                assert {r.competition for r in self.history} <= {"E0", "E1"}
                assert len({r.match_id for r in self.history}) == len(self.history)
                self.teams = {t for r in self.history for t in (r.home, r.away)} - factory.drop_teams
                self.competitions = {r.competition for r in self.history}
                return self

            def parameters(self):
                return {
                    "_fake_model_token": token,
                    "regularization": alpha,
                    "half_life_days": 180,
                    "before_date": self.cutoff.isoformat(),
                    "teams": sorted(self.teams),
                    "reference_match_ids": [r.match_id for r in self.history],
                    "not_for": "synthetic test only; never a real prediction",
                }

            def predict(self, home, away, competition, match_date):
                saved = [json.loads(p.read_text(encoding="utf-8"))
                         for p in (factory.output_dir / "parameters").glob("*.json")]
                assert any(p.get("parameters", {}).get("_fake_model_token") == token for p in saved), \
                    "save parameters before the first forecast, not after scoring"
                assert {home, away} <= self.teams
                assert home != away and home.startswith("england:") and away.startswith("england:")
                assert competition == "E0" and competition in self.competitions
                assert match_date >= self.cutoff
                factory.predictions.append((token, home, away, competition, match_date))
                return 1.5 + alpha, .8

        model = FakeModel()
        self.instances.append(model)
        return model


@pytest.fixture
def reference(tmp_path):
    history = records()
    factory = FakeFactory(tmp_path / "old-two-arms")
    value = run_comparison(history, factory.output_dir, model_factory=factory,
                           start=date(2025, 8, 1), stop=date(2025, 9, 12), interval_days=28)
    assert len(factory.instances) == 4
    return history, value


def test_weight_share_uses_decay_and_both_goal_rows_cancel_equally():
    cutoff = date(2025, 8, 1)
    history = [historical("epl", cutoff - timedelta(days=180)),
               historical("champ", cutoff - timedelta(days=360), competition="E1")]
    # E0 has weight .5, E1 .25. Each match contributes two weighted goal rows.
    assert weight_share(history, cutoff) == pytest.approx((2 * .5) / (2 * .5 + 2 * .25))
    assert weight_share(history, cutoff, half_life_days=360) == pytest.approx(
        math.sqrt(.5) / (math.sqrt(.5) + .5))


def test_share_excludes_same_day_future_and_explicitly_ineligible_results():
    cutoff = date(2025, 8, 1)
    history = [historical("epl", cutoff - timedelta(days=1)),
               historical("champ", cutoff - timedelta(days=1), competition="E1"),
               historical("same-day", cutoff),
               historical("future", cutoff + timedelta(days=1)),
               historical("excluded", cutoff - timedelta(days=1), competition="E1",
                          exclusion="synthetic exclusion")]
    assert weight_share(history, cutoff) == pytest.approx(.5)
    assert history[-1].goal_fit_exclusion == "synthetic exclusion"


def test_only_epl_history_is_a_valid_share_one():
    assert weight_share([historical("epl", date(2025, 7, 31))], date(2025, 8, 1)) == 1


@pytest.mark.parametrize("history", [
    [],
    [historical("future", date(2025, 8, 2))],
    [historical("champ-only", date(2025, 7, 31), competition="E1")],
    [historical("excluded", date(2025, 7, 31), exclusion="synthetic excluded")],
])
def test_share_requires_eligible_earlier_epl_data(history):
    with pytest.raises(ValueError):
        weight_share(history, date(2025, 8, 1))


@pytest.mark.parametrize("half_life", [0, -1, float("nan"), float("inf"), True, False])
def test_share_rejects_invalid_half_life(half_life):
    with pytest.raises((ValueError, TypeError)):
        weight_share([historical("epl", date(2025, 7, 31))], date(2025, 8, 1), half_life)


@pytest.mark.parametrize("cutoff", ["2025-08-01", datetime(2025, 8, 1), None])
def test_share_does_not_fabricate_calendar_dates_from_other_types(cutoff):
    with pytest.raises((ValueError, TypeError)):
        weight_share([historical("epl", date(2025, 7, 31))], cutoff)


def test_unknown_eligible_earlier_competition_cannot_change_share_silently():
    with pytest.raises(ValueError):
        weight_share([historical("epl", date(2025, 7, 30)),
                      historical("unknown", date(2025, 7, 31), competition="unknown")],
                     date(2025, 8, 1))


def test_future_unknown_competition_is_not_an_available_training_observation():
    assert weight_share([historical("epl", date(2025, 7, 30)),
                         historical("future-unknown", date(2025, 8, 2), competition="unknown")],
                        date(2025, 8, 1)) == 1


def test_one_fixed_new_fit_per_reference_window_and_original_rows_unchanged(tmp_path, reference):
    history, original = reference
    original_copy, history_copy = deepcopy(original), deepcopy(history)
    factory = FakeFactory(tmp_path / "new-fixed-arm", accepts_alpha=True)
    result = run_reference_comparison(history, original, factory.output_dir, model_factory=factory)
    assert original == original_copy and history == history_copy
    assert result["fit_count"] == len(factory.instances) == 2
    assert len(result["fits"]) == 2
    assert result["parameter_searches"] == 0
    assert len(result["per_fixture_rows"]) == 4
    assert result["target_count"] == 4
    assert all(set(row["arms"]) == {"epl_only", "joint", NEW_ARM}
               for row in result["per_fixture_rows"])
    by_id = {r["match_id"]: r for r in result["per_fixture_rows"]}
    for row in original["per_fixture_rows"]:
        assert {key: value for key, value in by_id[row["match_id"]].items() if key != "arms"} == {
            key: value for key, value in row.items() if key != "arms"}
        for arm in ("epl_only", "joint"):
            assert by_id[row["match_id"]]["arms"][arm] == row["arms"][arm]
    assert {model.cutoff for model in factory.instances} == {date(2025, 8, 1), date(2025, 8, 29)}
    # Parameters-before-predict is asserted inside FakeModel, on every call.
    assert len(factory.predictions) == 4
    for fit, model, alpha in zip(result["fits"], factory.instances, factory.alphas):
        cutoff = date.fromisoformat(fit["cutoff"])
        expected_history = [r for r in history if r.match_date < cutoff and r.goal_fit_exclusion is None]
        q = weight_share(expected_history, cutoff)
        assert fit["arm"] == NEW_ARM
        assert fit["epl_weight_share"] == pytest.approx(q)
        assert fit["alpha"] == pytest.approx(.01 * q)
        assert alpha == pytest.approx(.01 * q)
        assert fit["training_matches"] == len(expected_history) == len(model.history)
        assert fit["training_max_date"] == max(r.match_date for r in expected_history).isoformat()
        assert fit["parameters_saved_at"]
        payload = json.loads((factory.output_dir / "parameters" /
                              f"{NEW_ARM}_{fit['cutoff']}.json").read_text(encoding="utf-8"))
        assert payload["parameters"]["regularization"] == pytest.approx(alpha)
        assert payload["parameters"]["not_for"] == "synthetic test only; never a real prediction"
        assert {r.competition for r in model.history} == {"E0", "E1"}
        assert all(r.home.startswith("england:") and r.away.startswith("england:") for r in model.history)
        assert all(r.match_date < cutoff for r in model.history)


def test_missing_is_explicit_and_common_three_line_metrics_use_only_shared_fixtures(tmp_path, reference):
    history, original = reference
    factory = FakeFactory(tmp_path / "new-missing-arm", accepts_alpha=True,
                          drop_teams=("england:Hull",))
    result = run_reference_comparison(history, original, factory.output_dir, model_factory=factory)
    assert result["arms"]["epl_only"]["coverage"] == 3
    assert result["arms"]["joint"]["coverage"] == 4
    assert result["arms"][NEW_ARM]["coverage"] == 1
    common = result["paired_comparison"]["all_paired"]
    assert common["n"] == 1
    assert {value["n"] for value in common["arms"].values()} == {1}
    assert result["paired_comparison"]["promoted_paired"]["n"] == 0
    assert result["paired_comparison"]["other_paired"]["n"] == 1
    assert result["paired_comparison"]["from_2026_paired"]["n"] == 0
    known = next(r for r in result["per_fixture_rows"] if r["match_id"] == "target-known")
    for metric in METRICS:
        assert common["arms"][NEW_ARM][metric] == pytest.approx(known["arms"][NEW_ARM][metric])
        assert common["reference_minus_epl_only"][metric] == pytest.approx(
            common["arms"][NEW_ARM][metric] - common["arms"]["epl_only"][metric])
        assert common["reference_minus_joint"][metric] == pytest.approx(
            common["arms"][NEW_ARM][metric] - common["arms"]["joint"][metric])
        assert result["paired_comparison"]["promoted_paired"]["reference_minus_joint"][metric] is None
    missing = [r["arms"][NEW_ARM] for r in result["per_fixture_rows"] if r["arms"][NEW_ARM]["status"] == "missing"]
    assert len(missing) == 3
    assert all(r["reason"] == "unseen_team_no_estimated_prior" for r in missing)
    assert all(r["missing_teams"] == ["england:Hull"] for r in missing)
    assert all(r.get("rates") is None and r.get("forecast") is None for r in missing)


def test_preserves_all_probability_and_secondary_metric_outputs(tmp_path, reference):
    history, original = reference
    factory = FakeFactory(tmp_path / "new-secondary-arm", accepts_alpha=True)
    result = run_reference_comparison(history, original, factory.output_dir, model_factory=factory)
    for row in result["per_fixture_rows"]:
        value = row["arms"][NEW_ARM]
        assert set(METRICS) <= value.keys()
        assert all(math.isfinite(value[key]) for key in METRICS)
        assert set(value["forecast"]["one_x_two"]) == {"home", "draw", "away"}
        assert sum(value["forecast"]["one_x_two"].values()) == pytest.approx(1)
        assert set(value["forecast"]["total_goals"]) == {"0", "1", "2", "3", "4+"}
        assert set(value["forecast"]["away_minus_two_jincai"]) == {"win", "draw", "lose"}
    assert result["paired_comparison"]["all_paired"]["n"] == 3
    # Available-case coverage is not silently used as the common scoring count.
    assert result["arms"][NEW_ARM]["metrics"]["n"] == 4


@pytest.mark.parametrize("field,value", [
    ("match_id", "not-a-source-target"),
    ("match_date", "2025-08-10"),
    ("home", "england:NotArsenal"),
    ("away", "england:NotChelsea"),
    ("actual", [0, 12]),
    ("cutoff", "2025-08-02"),
])
def test_conflicting_reference_target_is_rejected_before_new_fit(tmp_path, reference, field, value):
    history, original = reference
    damaged = deepcopy(original)
    damaged["per_fixture_rows"][0][field] = value
    factory = FakeFactory(tmp_path / "new-invalid-target", accepts_alpha=True)
    with pytest.raises((ValueError, TypeError)):
        run_reference_comparison(history, damaged, factory.output_dir, model_factory=factory)
    assert factory.instances == []


@pytest.mark.parametrize("field,value", [("regularization", .02), ("half_life_days", 90)])
def test_different_reference_settings_cannot_be_called_the_fixed_contrast(tmp_path, reference, field, value):
    history, original = reference
    damaged = deepcopy(original)
    damaged["parameters"][field] = value
    factory = FakeFactory(tmp_path / "new-invalid-settings", accepts_alpha=True)
    with pytest.raises(ValueError):
        run_reference_comparison(history, damaged, factory.output_dir, model_factory=factory)
    assert factory.instances == []


def test_duplicate_reference_fixture_is_not_more_evaluation_data(tmp_path, reference):
    history, original = reference
    damaged = deepcopy(original)
    damaged["per_fixture_rows"].append(deepcopy(damaged["per_fixture_rows"][0]))
    factory = FakeFactory(tmp_path / "new-duplicate-target", accepts_alpha=True)
    with pytest.raises(ValueError):
        run_reference_comparison(history, damaged, factory.output_dir, model_factory=factory)
    assert factory.instances == []


def test_duplicate_source_id_is_rejected_before_new_fit(tmp_path, reference):
    history, original = reference
    damaged_history = history + [deepcopy(history[0])]
    factory = FakeFactory(tmp_path / "new-duplicate-source", accepts_alpha=True)
    with pytest.raises(ValueError):
        run_reference_comparison(damaged_history, original, factory.output_dir, model_factory=factory)
    assert factory.instances == []


def test_fit_failure_propagates_without_reusing_an_old_arm_or_searching_alpha(tmp_path, reference):
    history, original = reference
    original_copy = deepcopy(original)
    factory = FakeFactory(tmp_path / "new-failed-arm", accepts_alpha=True,
                          fit_error=RuntimeError("synthetic optimizer failed"))
    with pytest.raises(RuntimeError, match="synthetic optimizer failed"):
        run_reference_comparison(history, original, factory.output_dir, model_factory=factory)
    assert len(factory.instances) == 1
    assert len(factory.alphas) == 1
    assert factory.predictions == []
    assert original == original_copy
