"""Small synthetic tests of diagnostic definitions, not provider-data repairs."""

import pytest

from football_model.wyscout_probe import fd_index, goal_team, minute_hypothesis, substitutions


def event(name="Shot", team=1, markers=(101,)):
    return {"teamId": team, "eventName": name, "tags": [{"id": t} for t in markers]}


def test_primary_goals_do_not_double_count_keeper_and_reverse_own_goals():
    assert goal_team(event(), {1, 2}) == 1
    assert goal_team(event("Save attempt"), {1, 2}) is None
    assert goal_team(event("Touch", markers=(102,)), {1, 2}) == 2
    assert goal_team(event(markers=()), {1, 2}) is None
    with pytest.raises(ValueError, match="team"):
        goal_team(event(team=3), {1, 2})
    with pytest.raises(ValueError, match="two known"):
        goal_team(event("Touch", team=3, markers=(102,)), {1, 2})


def test_45_minute_assignment_depends_on_observed_period_for_diagnostic_only():
    sub = {"playerIn": 7, "playerOut": 8, "minute": 45}
    first_half = minute_hypothesis(sub, {7: [(0, 2699.0)], 8: [(0, 2660.0)]})
    assert first_half["assumed_period"] == "1H"
    assert not first_half["ceil_minute_action_conflict"]
    halftime = minute_hypothesis(sub, {7: [(1, 2.0)], 8: [(0, 2790.0)]})
    assert halftime["assumed_period"] == "2H"
    assert not halftime["ceil_minute_action_conflict"]


def test_46_is_interval_hypothesis_not_exact_60_seconds():
    sub = {"playerIn": 7, "playerOut": 8, "minute": 46}
    row = minute_hypothesis(sub, {7: [(1, 3.0)], 8: [(0, 2800.0)]})
    assert row["assumed_period"] == "2H"
    assert not row["ceil_minute_action_conflict"]
    sub["minute"] = 49
    row = minute_hypothesis(sub, {7: [(1, 3.0)], 8: [(0, 2800.0)]})
    assert row["ceil_minute_action_conflict"]
    assert row["gap_seconds"] == 177.0


def test_actor_order_conflict_cannot_be_fixed_by_moving_one_boundary():
    row = minute_hypothesis({"playerIn": 7, "playerOut": 8, "minute": 82},
                            {7: [(1, 2311.9)], 8: [(1, 2345.9)]})
    assert row["actor_order_conflict"]
    assert row["ceil_minute_action_conflict"]
    # The minute interval is even earlier than the incoming actor: this is
    # gap to the intersection, not just the two actors' 34-second inversion.
    assert row["gap_seconds"] == pytest.approx(125.9)


def test_no_incoming_event_is_unknown_not_zero_minute_entry():
    row = minute_hypothesis({"playerIn": 7, "playerOut": 8, "minute": 90}, {8: [(1, 2660.0)]})
    assert row["no_incoming_action"]
    assert row["first_in_action"] is None


def test_incompatible_period_does_not_emit_json_infinity():
    row = minute_hypothesis({"playerIn": 7, "playerOut": 8, "minute": 20},
                            {8: [(1, 2.0)], 7: [(0, 1200.0)]})
    assert row["incompatible_period"]
    assert row["gap_seconds"] is None


@pytest.mark.parametrize("minute", [True, 20.0, "20", -1])
def test_bad_substitution_minute_is_not_silently_converted(minute):
    with pytest.raises(ValueError, match="Integer"):
        minute_hypothesis({"playerIn": 7, "playerOut": 8, "minute": minute}, {})


@pytest.mark.parametrize("value", [None, "null", 0])
def test_null_substitutions_are_empty_lists(value):
    assert substitutions({"substitutions": value}) == []


def test_unknown_substitutions_structure_refuses():
    with pytest.raises(ValueError, match="Unexpected"):
        substitutions({"substitutions": {"minute": 60}})


def test_independent_reader_ignores_odds_and_rejects_duplicate_match_key(tmp_path):
    path = tmp_path / "fd.csv"
    header = "Date,HomeTeam,AwayTeam,FTHG,FTAG,HTHG,HTAG,HR,AR,B365H\n"
    row = "11/08/2017,Arsenal,Leicester,4,3,2,2,0,0,not-an-odds-input\n"
    path.write_text(header + row, encoding="utf-8")
    result = fd_index(path)
    assert result == {("2017-08-11", "Arsenal", "Leicester"):
                      {"FTHG": 4, "FTAG": 3, "HTHG": 2, "HTAG": 2, "HR": 0, "AR": 0}}
    path.write_text(header + row + row, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        fd_index(path)
