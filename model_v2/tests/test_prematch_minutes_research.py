from copy import deepcopy
from datetime import date
import json

import pytest

from football_model import prematch_minutes_research as research
from football_model.prematch_minutes import TeamMinutesObservation


def path(mid="statsbomb:match:1", day="2016-01-01", home=1, away=2):
    h = [f"statsbomb:player:{i}" for i in range(1, 12)]
    a = [f"statsbomb:player:{i}" for i in range(21, 32)]
    return {"match_id": mid, "match_date": day, "home_team_id": home, "away_team_id": away,
        "diagnostics": {"fixed_window_seconds_by_half": {"1": 2700, "2": 2700}},
        "stints": [{"period": p, "start_second": 0., "end_second": 2700., "duration_seconds": 2700.,
                    "home_players": h[:], "away_players": a[:]} for p in (1, 2)],
        "player_minutes": [{"team_id": t, "player_id": pid, "on_pitch_seconds": 5400.}
                           for t, players in ((home, h), (away, a)) for pid in players]}


def fixture(mid, day, teams=(1, 2)):
    return {"match_id": mid, "match_date": date.fromisoformat(day), "team_ids": list(teams)}


def obs(mid, day, minutes, team=1, source="s"):
    return TeamMinutesObservation(mid, team, date.fromisoformat(day), minutes, source)


def test_extracts_complete_fixed_exposures_without_source_mutation():
    source = path()
    before = deepcopy(source)
    records, fixtures = research.observations_from_paths([source], "s", "statsbomb")
    assert before == source
    assert len(records) == 2 and len(fixtures) == 1
    assert sum(records[0].minutes.values()) == 990
    assert set(records[0].minutes) == set(source["stints"][0]["home_players"])


def test_red_or_temporary_absence_deficit_not_repaired():
    source = path()
    source["stints"][1]["home_players"].remove("statsbomb:player:11")
    source["player_minutes"][10]["on_pitch_seconds"] = 2700.
    records, _ = research.observations_from_paths([source], "s", "statsbomb")
    assert sum(records[0].minutes.values()) == 945
    assert records[0].minutes["statsbomb:player:11"] == 45


@pytest.mark.parametrize("damage", ["gap", "short_half", "full_window", "foreign_player", "derived_mismatch",
                                   "same_players_both_teams", "duplicate_roster", "negative_duration", "reversed_periods"])
def test_unsafe_derived_paths_are_not_silently_repaired(damage):
    source = path()
    if damage == "gap":
        source["stints"][0]["start_second"] = 10
    elif damage == "short_half":
        source["stints"][1].update(end_second=2600, duration_seconds=2600)
    elif damage == "full_window":
        source["diagnostics"]["fixed_window_seconds_by_half"] = {"1": 2800, "2": 2800}
    elif damage == "foreign_player":
        source["stints"][0]["home_players"][0] = "wyscout:player:99"
    elif damage == "derived_mismatch":
        source["player_minutes"][0]["on_pitch_seconds"] = 0
    elif damage == "same_players_both_teams":
        source["stints"][0]["away_players"][0] = "statsbomb:player:1"
    elif damage == "duplicate_roster":
        source["player_minutes"].append(source["player_minutes"][0])
    elif damage == "negative_duration":
        source["stints"][0]["duration_seconds"] = -1
    else:
        source["stints"].reverse()
    with pytest.raises(ValueError):
        research.observations_from_paths([source], "s", "statsbomb")


def test_duplicate_source_ids_and_empty_archive_fail():
    for sources in ([], [path(), path()]):
        with pytest.raises(ValueError):
            research.observations_from_paths(sources, "s", "statsbomb")


def test_common_evaluation_union_penalises_omitted_newcomer_and_retained_departure():
    candidates = {
        "last_match": {"status": "estimated", "predicted_minutes": {"old": 90}, "total_predicted_minutes": 90},
        "recent_mean": {"status": "estimated", "predicted_minutes": {"old": 45, "other": 45}, "total_predicted_minutes": 90},
    }
    before = deepcopy(candidates)
    score = research.score_team(candidates, {"new": 90}, {"old", "other"})
    assert candidates == before
    assert score["evaluation_player_ids"] == ["new", "old", "other"]
    assert score["actual_minutes_not_in_earlier_team_history"] == 90
    for method in candidates:
        metrics = score["metrics"][method]
        assert metrics["common_union_player_mae_minutes"] == 60
        assert metrics["omitted_actual_minute_fraction"] == 1
        assert metrics["total_absolute_error_minutes"] == 180
    assert "new" not in candidates["last_match"]["predicted_minutes"]


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), True, "90", 91])
def test_actual_exposure_is_validated_before_zero_filtering(bad):
    candidates = {m: {"status": "estimated", "predicted_minutes": {"p": 90},
                       "total_predicted_minutes": 90} for m in research.METHODS}
    with pytest.raises(ValueError):
        research.score_team(candidates, {"p": 90, "bad": bad}, {"p"})


def test_prediction_calls_receive_only_prior_dates_and_not_actual_target_identity(monkeypatch):
    observations = []
    for team in (1, 2):
        observations.extend([obs("earlier", "2016-01-01", {f"old{team}": 90}, team),
                             obs("target", "2016-02-01", {f"new{team}": 90}, team),
                             obs("later", "2016-02-02", {f"future{team}": 90}, team)])
    original = research.estimate_minutes
    calls = []
    def witness(history, **kwargs):
        calls.append((list(history), kwargs))
        assert all(r.match_date < kwargs["prediction_date"] for r in history)
        assert all(r.match_id != "target" for r in history)
        return original(history, **kwargs)
    monkeypatch.setattr(research, "estimate_minutes", witness)
    summary, rows, scores, skipped = research.evaluate_track(observations,
        [fixture("target", "2016-02-01")], date(2016, 2, 1))
    assert len(calls) == 4 and len(rows) == 2 and len(scores) == 1 and not skipped
    assert summary["scored_matches"] == 1
    assert summary["earlier_team_unseen_actual_minutes"] == 180
    assert all(not any(p.startswith("new") or p.startswith("future") for p in c["predicted_minutes"])
               for row in rows for c in row["candidates"].values())


def test_changed_target_and_future_minutes_cannot_change_target_candidates():
    records = [obs(mid, day, {name: 90}, team)
        for team in (1, 2)
        for mid, day, name in (("old", "2016-01-01", "old"),
                               ("target", "2016-02-01", "target"), ("future", "2016-03-01", "future"))]
    changed = [r if r.match_id == "old" else obs(r.match_id, r.match_date.isoformat(), {"edited": 85}, r.team_id) for r in records]
    a = research.evaluate_track(records, [fixture("target", "2016-02-01")], date(2016, 2, 1))[1]
    b = research.evaluate_track(changed, [fixture("target", "2016-02-01")], date(2016, 2, 1))[1]
    assert [r["candidates"] for r in a] == [r["candidates"] for r in b]


def test_whole_match_skipped_if_either_team_has_no_earlier_history():
    records = [obs("old", "2016-01-01", {"p": 90}),
               obs("target", "2016-02-01", {"p": 90}),
               obs("target", "2016-02-01", {"q": 90}, 2)]
    summary, rows, scores, skipped = research.evaluate_track(records, [fixture("target", "2016-02-01")], date(2016, 2, 1))
    assert summary["requested_matches"] == summary["skipped_matches"] == 1
    assert summary["scored_matches"] == 0 and not rows and not scores
    assert skipped[0]["teams"][1]["candidates"]["last_match"]["total_predicted_minutes"] is None


def test_foreign_source_cannot_hide_first_team_appearance():
    records = [obs("older", "2016-01-01", {"old": 90}, t)
               for t in (1, 2)]
    records += [obs("foreign", "2016-01-10", {"new": 90}, t, "another-provider")
                for t in (1, 2)]
    records += [obs("target", "2016-02-01", {"new": 90}, t) for t in (1, 2)]
    summary, rows, _, _ = research.evaluate_track(records, [fixture("target", "2016-02-01")], date(2016, 2, 1))
    assert summary["earlier_team_unseen_actual_minutes"] == 180
    assert all(row["score"]["actual_ids_not_in_earlier_team_history"] == ["new"] for row in rows)


def test_latest_five_team_matches_not_player_appearances_and_two_teams_one_match():
    records = []
    for team in (1, 2):
        records.extend(obs(f"h{i}", f"2016-01-{i:02d}", {"old": 90} if i == 1 else {"current": 90}, team)
                       for i in range(1, 7))
        records.append(obs("target", "2016-02-01", {"current": 90}, team))
    summary, rows, scores, _ = research.evaluate_track(records, [fixture("target", "2016-02-01")], date(2016, 2, 1))
    assert summary["scored_team_rows"] == 2 and summary["scored_matches"] == 1
    assert all(r["candidates"]["recent_mean"]["history_match_ids"] == [f"h{i}" for i in range(2, 7)] for r in rows)
    assert all("old" not in r["candidates"]["recent_mean"]["predicted_minutes"] for r in rows)
    assert scores[0]["recent_mean_minus_last_match_mae"] == 0


def test_end_to_end_generates_real_candidates_and_finite_json_without_model_fit(tmp_path):
    matches = [path(day="2016-01-01"), path("statsbomb:match:2", "2016-02-01")]
    tracks = [("s", matches, "statsbomb", date(2016, 2, 1), "fixture.json")]
    output = research.run(tmp_path, tracks=tracks)
    result = json.loads((output / "分钟研究结果.json").read_text(encoding="utf-8"))
    assert result["new_optimizer_fits"] == 0 and result["distinct_matches"] == 1
    assert result["candidate_team_projections"] == 4
    assert result["tracks"]["s"]["summary"]["methods"]["recent_mean"]["common_union_player_mae_minutes"]["mean"] == 0
    assert (output / "分钟外推研究说明.md").exists()
    rows = json.loads((output / "s_逐队候选与误差.json").read_text(encoding="utf-8"))["rows"]
    assert all(r["candidates"]["last_match"]["history_match_ids"] == ["statsbomb:match:1"] for r in rows)


def test_clock_tracks_do_not_multiply_distinct_match_count(tmp_path):
    matches = [path(), path("statsbomb:match:2", "2016-02-01")]
    tracks = [(s, matches, "statsbomb", date(2016, 2, 1), "fixture.json") for s in ("a", "b", "c")]
    output = research.run(tmp_path, tracks=tracks)
    result = json.loads((output / "分钟研究结果.json").read_text(encoding="utf-8"))
    assert result["distinct_matches"] == 1 and result["candidate_team_projections"] == 12


def test_failures_preserve_incomplete_record_not_success(tmp_path):
    source = path()
    source["stints"][0]["duration_seconds"] = 0
    with pytest.raises(ValueError):
        research.run(tmp_path, tracks=[("s", [source], "statsbomb", date(2016, 2, 1), "fixture.json")])
    partial = next((tmp_path / "outputs").iterdir())
    record = json.loads((partial / "运行未完成说明.json").read_text(encoding="utf-8"))
    assert record["status"] == "incomplete" and record["completed_tracks"] == []
    assert not (partial / "分钟研究结果.json").exists()
