from dataclasses import replace
from datetime import timedelta

import pytest

from football_model.demo import synthetic_inputs
from football_model.domain import OddsSnapshot, ProjectedStint, StintLineupScenario
from football_model.match_context import (
    ContextEvidence, LineupAnnouncement, MatchContext, TeamCompetitionContext, compare_versions,
)
from football_model.probability import forecast_scenarios


def context_inputs():
    fixture, cutoff, rates, home, away, effects, scenarios = synthetic_inputs()
    fixture = replace(fixture, kickoff=cutoff + timedelta(hours=24))
    home_context = TeamCompetitionContext(home.team, (
        ContextEvidence("rest_hours", 72, "synthetic schedule known before cutoff", cutoff),
        ContextEvidence("qualification_state", None, "synthetic unknown", cutoff),
    ), ("No quantified coach-priority effect",))
    away_context = TeamCompetitionContext(away.team)
    early = MatchContext(fixture.match_id, "day_before", home_context, away_context,
                         scenario_weight_source="Synthetic assumptions, not learned probabilities",
                         scenario_weights_as_of=cutoff)
    late_time = cutoff + timedelta(hours=23)
    hm, am = dict(home.minutes), dict(away.minutes)
    hm["H_FW_10"], hm["H_FW_sub"] = 90, 0
    am["A_FW_10"], am["A_FW_sub"] = 90, 0
    hxi, axi = replace(home, minutes=hm), replace(away, minutes=am)
    full = StintLineupScenario("same_XI_full_match_assumption", 0.5, (ProjectedStint(90, hxi, axi),))
    sub = StintLineupScenario("same_XI_then_forward_substitution", 0.5, (
        ProjectedStint(60, hxi, axi), ProjectedStint(30, scenarios[1].home, axi),
    ))
    announcements = tuple(LineupAnnouncement(
        xi.team, tuple(p for p, m in xi.minutes.items() if m == 90), "synthetic official source",
        late_time - timedelta(minutes=10), late_time - timedelta(minutes=9),
        fixture.match_id,
    ) for xi in (hxi, axi))
    late = replace(early, version="lineup_confirmed", announcements=announcements)
    return (fixture, cutoff, rates, home, away, effects, scenarios), early, late_time, late, [full, sub]


def test_context_records_unknowns_without_inventing_a_motivation_adjustment():
    args, early, *_ = context_inputs()
    plain = forecast_scenarios(*args)
    contextual = forecast_scenarios(*args, context=early)
    assert contextual["summary"] == plain["summary"]
    context = contextual["prediction_context"]
    assert context["hours_to_kickoff"] == 24
    assert context["competition_context_applied_to_rates"] is False
    assert context["team_contexts"][0]["evidence"][1]["value"] is None


def test_future_context_is_not_backfilled_into_early_prediction():
    args, early, *_ = context_inputs()
    item = ContextEvidence("coach_statement", "will rotate", "synthetic later interview",
                           args[1] + timedelta(minutes=1))
    future = replace(early, home=replace(early.home, evidence=(item,)))
    with pytest.raises(ValueError, match="Future"):
        forecast_scenarios(*args, context=future)


def test_confirmed_version_requires_both_observed_official_lineups():
    args, early, late_time, late, scenarios = context_inputs()
    with pytest.raises(ValueError, match="both"):
        forecast_scenarios(args[0], late_time, *args[2:6], scenarios,
                           context=replace(late, announcements=late.announcements[:1]))
    with pytest.raises(ValueError, match="unavailable"):
        forecast_scenarios(*args, context=late)
    with pytest.raises(ValueError, match="projected stints"):
        forecast_scenarios(args[0], late_time, *args[2:6], args[6], context=late)


def test_confirmed_XI_does_not_collapse_substitution_uncertainty():
    args, _, late_time, late, scenarios = context_inputs()
    result = forecast_scenarios(args[0], late_time, *args[2:6], scenarios, context=late)
    assert len(result["scenarios"]) == 2
    assert result["prediction_context"]["hours_to_kickoff"] == 1
    assert result["pre_market_scenario_variance_decomposition"]["home"]["between_scenario_probability_variance"] > 0
    assert result["statistical_parameter_intervals"] is None


def test_official_starters_must_match_initial_stint():
    args, _, late_time, late, scenarios = context_inputs()
    official = late.announcements[0]
    wrong = replace(official, starters=tuple("H_FW_sub" if p == "H_FW_10" else p for p in official.starters))
    with pytest.raises(ValueError, match="initial XI"):
        forecast_scenarios(args[0], late_time, *args[2:6], scenarios,
                           context=replace(late, announcements=(wrong, late.announcements[1])))


def test_variance_decomposition_is_not_a_confidence_interval():
    args, early, *_ = context_inputs()
    result = forecast_scenarios(*args, context=early)
    for row in result["pre_market_scenario_variance_decomposition"].values():
        assert row["within_scenario_event_variance"] + row["between_scenario_probability_variance"] == pytest.approx(row["total_event_variance"])
    assert "not a confidence interval" in result["variance_note"]


def test_version_comparison_keeps_market_and_structural_changes_separate():
    args, early, late_time, late, scenarios = context_inputs()
    early_odds = OddsSnapshot(2.1, 3.4, 3.6, args[1], "synthetic", args[0].match_id)
    late_odds = replace(early_odds, home=1.9, away=4.2, observed_at=late_time)
    first = forecast_scenarios(*args, context=early, market=early_odds)
    second = forecast_scenarios(args[0], late_time, *args[2:6], scenarios, context=late, market=late_odds)
    comparison = compare_versions(first, second)
    assert comparison["market_one_x_two_change"]["home"] > 0
    assert "not isolated causal effects" in comparison["note"]
    assert comparison["structural_one_x_two_change"] == comparison["final_one_x_two_change"]
    with pytest.raises(ValueError, match="day-before"):
        compare_versions(replace_dict(first, prediction_context=None), second)


def replace_dict(value, **changes):
    return {**value, **changes}


def test_context_identity_and_publication_time_checked():
    args, early, *_ = context_inputs()
    with pytest.raises(ValueError, match="identities"):
        forecast_scenarios(*args, context=replace(early, fixture_id="other"))
    with pytest.raises(ValueError, match="before publication"):
        ContextEvidence("rest_hours", 72, "synthetic", args[1], args[1] + timedelta(seconds=1))
    with pytest.raises(ValueError, match="11 distinct"):
        LineupAnnouncement(args[3].team, ("same",) * 11, "synthetic", args[1], args[1], args[0].match_id)


def test_previous_fixture_announcement_is_not_a_current_confirmation():
    args, _, late_time, late, scenarios = context_inputs()
    previous = replace(late.announcements[0], fixture_id="previous_match",
                       published_at=args[1] - timedelta(days=7), observed_at=args[1] - timedelta(days=7))
    with pytest.raises(ValueError, match="another fixture"):
        forecast_scenarios(args[0], late_time, *args[2:6], scenarios,
                           context=replace(late, announcements=(previous, late.announcements[1])))


def test_future_scenario_weights_and_changed_forecast_target():
    args, early, late_time, late, scenarios = context_inputs()
    with pytest.raises(ValueError, match="weights were unavailable"):
        forecast_scenarios(*args, context=replace(early, scenario_weights_as_of=late_time))
    first = forecast_scenarios(*args, context=early)
    second = forecast_scenarios(args[0], late_time, *args[2:6], scenarios, context=late)
    changed = {**second, "fixture": {**second["fixture"], "kickoff": (args[0].kickoff + timedelta(hours=1)).isoformat()}}
    with pytest.raises(ValueError, match="target must agree"):
        compare_versions(first, changed)


def test_nearly_normalized_weights_use_consistent_probability_variance():
    args, early, *_ = context_inputs()
    changed = [replace(s, weight=s.weight * 1.000001) for s in args[6]]
    result = forecast_scenarios(*args[:6], changed, context=early)
    for name, row in result["pre_market_scenario_variance_decomposition"].items():
        p = result["structural_summary"]["one_x_two"][name]
        assert row["total_event_variance"] == pytest.approx(p * (1 - p), abs=1e-12)
        assert row["within_scenario_event_variance"] + row["between_scenario_probability_variance"] == pytest.approx(p * (1 - p), abs=1e-12)
