import pytest

from football_model.archive_shot_context import reconstruct_archive_context


def minute_rows(events=(), first_end=48, second_end=52):
    rows = [{"side": s, "half_id": h, "minute": m, "goals": 0, "red_cards": 0}
            for h, minutes in ((1, range(first_end + 1)), (2, range(46, second_end + 1)))
            for m in minutes for s in ("h", "a")]
    # Event state-split rows supplement, not replace, the zero rows.
    rows += [{"side": s, "half_id": h, "minute": m, "goals": g, "red_cards": r,
              "source_score_diff": 999, "source_red_diff": 999} for s, h, m, g, r in events]
    return rows


def shot(uid="s", minute=10, side="h", result="MissedShots"):
    return {"id": uid, "minute": minute, "side": side, "result": result, "xG": 0.123}


@pytest.mark.parametrize("side,result,credited,score", [
    ("h", "Goal", "h", [1, 0]), ("h", "OwnGoal", "a", [0, 1])])
def test_current_goal_is_excluded_including_own_goal(side, result, credited, score):
    rows = minute_rows([(credited, 1, 11, 1, 0)])
    result = reconstruct_archive_context([shot(side=side, result=result)], rows,
                                         expected_full_time=score, clock_offsets=(1,))
    assert result["shots"][0]["window_unique_joint_state"] == [0, 0, 0, 0]


def test_same_minute_events_are_unordered_and_independent_of_row_order():
    shots = [shot("g1", 10, "h", "Goal"), shot("g2", 10, "a", "Goal"), shot("x", 10)]
    rows = minute_rows([("h", 1, 11, 1, 1), ("a", 1, 11, 1, 1)])
    result = reconstruct_archive_context(shots, rows, expected_full_time=[1, 1], clock_offsets=(1,))
    target = result["shots"][2]
    assert len(target["joint_candidate_bounds"]) == 16
    assert target["window_unique_joint_state"] is None
    replay = reconstruct_archive_context(list(reversed(shots)), list(reversed(rows)),
                                         expected_full_time=[1, 1], clock_offsets=(1,))
    assert replay["shots"][0] == target


def test_halftime_numeric_overlap_preserves_joint_coordinate_provenance():
    # First-half minute 48 precedes EVERY second-half event, including minute 46.
    shots = [shot("g", 47, "h", "Goal"), shot("x", 46, "a")]
    rows = minute_rows([("h", 1, 48, 1, 0), ("a", 2, 46, 0, 1)])
    target = reconstruct_archive_context(shots, rows, expected_full_time=[1, 0],
                                         clock_offsets=(1,))["shots"][1]
    branches = {b["half_id"]: b for b in target["branches"]}
    assert branches[1]["joint_candidate_bounds"] == [[0, 0, 0, 0]]
    assert branches[2]["joint_candidate_bounds"] == [[1, 0, 0, 1]]
    assert target["joint_candidate_bounds"] == [[0, 0, 0, 0], [1, 0, 0, 1]]
    assert target["window_unique_joint_state"] is None


def test_partial_clock_coverage_cannot_be_called_unique():
    target = reconstruct_archive_context([shot(minute=52)], minute_rows(),
                                         expected_full_time=[0, 0])["shots"][0]
    assert target["joint_candidate_bounds"] == [[0, 0, 0, 0]]
    assert target["all_branches_covered"] is False
    assert target["window_unique_score"] is None
    assert "missing_two_sided_minute_coordinate" in target["unknown_reasons"]


def test_goal_matches_only_one_offset_does_not_erase_unknown_alternatives():
    rows = minute_rows([("h", 1, 11, 1, 0)])
    target = reconstruct_archive_context([shot(result="Goal")], rows,
                                         expected_full_time=[1, 0])["shots"][0]
    assert target["joint_candidate_bounds"] == [[0, 0, 0, 0]]
    assert target["all_branches_covered"] is False
    assert target["window_unique_joint_state"] is None
    assert sum("unknown_reason" in b for b in target["branches"]) == 2


def test_red_card_source_difference_is_ignored_and_before_after_retained():
    rows = minute_rows([("h", 1, 11, 0, 1)])
    target = reconstruct_archive_context([shot()], rows, expected_full_time=[0, 0],
                                         clock_offsets=(1,))["shots"][0]
    assert target["joint_candidate_bounds"] == [[0, 0, 0, 0], [0, 0, 1, 0]]
    assert target["window_unique_own_red_card_record_difference"] is None
    after = reconstruct_archive_context([shot(minute=12)], rows, expected_full_time=[0, 0],
                                        clock_offsets=(1,))["shots"][0]
    assert after["window_unique_own_red_card_record_difference"] == 1


def test_goal_conflict_withholds_states_and_preserves_shot_values():
    rows = minute_rows([("h", 1, 11, 1, 0)])
    result = reconstruct_archive_context([shot()], rows, expected_full_time=[0, 0])
    assert "minute_full_time_goal_accounting_conflict" in result["withhold_reasons"]
    assert result["shots"][0]["joint_candidate_bounds"] == []
    assert result["shots"][0]["source_xG"] == 0.123


def test_equal_totals_do_not_hide_clock_window_contradiction():
    rows = minute_rows([("h", 1, 20, 1, 0)])
    result = reconstruct_archive_context([shot(result="Goal")], rows, expected_full_time=[1, 0])
    assert result["goal_window_matching"]["maximum_matched"] == 0
    assert result["withhold_reasons"] == ["goal_landmarks_incompatible_with_declared_clock_window"]


def test_goal_matching_requires_distinct_minute_goal_units():
    shots = [shot("g1", 10, result="Goal"), shot("g2", 10, result="Goal")]
    rows = minute_rows([("h", 1, 11, 1, 0), ("h", 1, 20, 1, 0)])
    result = reconstruct_archive_context(shots, rows, expected_full_time=[2, 0])
    assert result["goal_window_matching"]["maximum_matched"] == 1
    assert result["withhold_reasons"]


def test_invalid_counts_and_duplicate_shots_fail_explicitly():
    with pytest.raises(ValueError, match="duplicate"):
        reconstruct_archive_context([shot(), shot()], minute_rows(), expected_full_time=[0, 0])
    rows = minute_rows()
    rows[0]["goals"] = -1
    with pytest.raises(ValueError, match="goal count"):
        reconstruct_archive_context([shot()], rows, expected_full_time=[0, 0])
