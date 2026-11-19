from copy import deepcopy

import pytest

from football_model.understat_shot_states import reconstruct_shot_states


def event(uid, minute, side="h", result="SavedShot", **extra):
    return {"id": str(uid), "minute": str(minute), "h_a": side, "result": result,
            "match_id": "m", "xG": "0.123", "h_goals": "99", "a_goals": "98", **extra}


def keyed(shots, **kwargs):
    return {row["shot_id"]: row for row in reconstruct_shot_states(shots, **kwargs)["shots"]}


def test_own_goal_credits_opposite_side_and_never_counts_itself():
    shots = {"h": [event(1, 10, result="OwnGoal"), event(2, 11)], "a": []}
    result = reconstruct_shot_states(shots, expected_full_time=(0, 1), match_id="m")
    assert result["goal_accounting_consistent"] is True
    assert result["shots"][0]["unique_score"] == [0, 0]
    assert result["shots"][0]["unique_own_state"] == "drawing"
    assert result["shots"][1]["unique_score"] == [0, 1]
    assert result["shots"][1]["unique_own_state"] == "trailing"
    assert result["shots"][1]["source_xG"] == "0.123"


def test_same_minute_score_set_can_still_have_known_own_state():
    shots = {"h": [event(1, 1, result="Goal"), event(2, 2, result="Goal"),
                    event(3, 5), event(4, 5, result="Goal")], "a": []}
    rows = keyed(shots)
    assert rows["3"]["possible_scores"] == [[2, 0], [3, 0]]
    assert rows["3"]["score_is_unique"] is False
    assert rows["3"]["possible_own_states"] == ["leading"]
    assert rows["3"]["own_state_is_unique"] is True
    assert rows["4"]["unique_score"] == [2, 0]


def test_opposite_same_minute_goals_keep_all_possible_before_states():
    shots = {"h": [event(30, 20), event(2, 20, result="Goal")],
             "a": [event(1, 20, "a", "Goal")]}
    rows = keyed(shots)
    assert rows["30"]["possible_scores"] == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert rows["30"]["possible_own_states"] == ["trailing", "drawing", "leading"]
    assert rows["2"]["possible_scores"] == [[0, 0], [0, 1]]
    changed = {"h": list(reversed(shots["h"])), "a": shots["a"]}
    assert keyed(changed)["30"]["possible_scores"] == rows["30"]["possible_scores"]


def test_halftime_overlap_and_later_shots_stay_unknown_despite_ids_or_array_order():
    shots = {"h": [event(900, 54, result="Goal"), event(1, 47),
                    event(2, 44), event(3, 45), event(4, 105)], "a": []}
    result = reconstruct_shot_states(shots)
    rows = {row["shot_id"]: row for row in result["shots"]}
    assert rows["2"]["unique_score"] == [0, 0]
    for uid in ("900", "1", "3", "4"):
        assert rows[uid]["possible_scores"] is None
        assert rows[uid]["possible_own_states"] is None
    assert result["source_minute_regressions"]


def test_final_score_fields_are_ignored_and_conflicts_are_not_backfilled():
    shots = {"h": [event(1, 1), event(2, 10, result="Goal")], "a": []}
    original = reconstruct_shot_states(shots, expected_full_time=(1, 0))
    polluted = deepcopy(shots)
    polluted["h"][0].update(h_goals="-100", a_goals="1000")
    assert reconstruct_shot_states(polluted, expected_full_time=(1, 0)) == original
    conflict = reconstruct_shot_states(shots, expected_full_time=(2, 0))
    assert conflict["goal_accounting_consistent"] is False
    assert conflict["shots"][0]["possible_scores"] is None
    assert conflict["counted_goal_credit"] == [1, 0]


def test_duplicate_event_or_unparseable_clock_is_reported_as_invalid():
    with pytest.raises(ValueError, match="duplicate"):
        reconstruct_shot_states({"h": [event(1, 5)], "a": [event(1, 6, "a")]})
    with pytest.raises(ValueError, match="minute"):
        reconstruct_shot_states({"h": [event(1, "45+2")], "a": []})
