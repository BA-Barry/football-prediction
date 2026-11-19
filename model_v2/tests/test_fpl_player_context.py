"""Synthetic saved-snapshot tests; no network/model/LLM performance claims."""

from copy import deepcopy
from datetime import timedelta

import pytest

from test_prospective import F, NOW
from football_model.fpl_player_context import build_context


def inputs():
    fixture = {"match_id": F.match_id, "home": F.home, "away": F.away,
               "competition": F.competition, "kickoff": F.kickoff.isoformat(), "neutral": False}
    players = [{"id": i, "team": i, "first_name": "Synthetic", "second_name": f"Player{i}",
                "web_name": f"Player{i}", "status": "i", "news": "Source injury flag",
                "news_added": (NOW-timedelta(days=3)).isoformat(),
                "minutes": 596 if i == 1 else 0, "starts": 7 if i == 1 else 0,
                "element_type": 3} for i in (1, 2)]
    observed = (NOW-timedelta(hours=2)).isoformat()
    facts = [{"id": f"old:{i}", "source_team_id": i, "team": team, "observed_at": observed,
              "source_players": [{"id": i, "status": "i", "news": "Source injury flag",
                                  "news_added": players[i-1]["news_added"]}]}
             for i, team in ((1, F.home), (2, F.away))]
    case = {"records": [{"fixture": fixture}]}
    news = {"evidence_by_fixture": {F.match_id: facts}, "unknowns_by_fixture": {F.match_id: ["Lineup unknown"]}}
    bootstrap = {"teams": [{"id": 1, "name": F.home}, {"id": 2, "name": F.away}], "elements": players}
    raw_fixtures = [{"id": 100, "team_h": 1, "team_a": 2, "started": False, "finished": False,
                     "finished_provisional": False, "kickoff_time": F.kickoff.isoformat()}]
    metadata = {"sources": {name: {"received_at": (NOW-timedelta(hours=1)).isoformat(), "status": 200,
                  "source_url": f"https://example.test/{name}", "age_seconds": "45"}
                 for name in ("bootstrap", "fixtures")}}
    observations = {"fixture_observations": [{"match_id": F.match_id, "official_fixture_id": 100,
                     "home": F.home, "away": F.away, "home_team_id": 1, "away_team_id": 2,
                     "expected_kickoff": F.kickoff.isoformat(), "verified": True}]}
    return case, news, bootstrap, raw_fixtures, metadata, observations


def test_enriches_real_descriptive_fields_without_fabricating_new_injury_changes():
    args = inputs()
    original = deepcopy(args)
    news, audit = build_context(*args, NOW)
    assert args == original
    assert len(audit["future_fixtures"]) == 1
    assert audit["changed_status_news_metadata"] == []
    rows = news["evidence_by_fixture"][F.match_id]
    assert [r["source_players"][0]["season_minutes"] for r in rows] == [596, 0]
    assert [r["source_players"][0]["season_starts"] for r in rows] == [7, 0]
    assert all(r["observed_at"] == args[4]["sources"]["bootstrap"]["received_at"] for r in rows)
    assert audit["model_fits"] == audit["llm_calls"] == audit["new_http_requests"] == 0
    assert audit["old_forecasts_changed"] is False


def test_changed_raw_news_is_reported_not_converted_to_probability_effect():
    args = inputs()
    args[2]["elements"][0]["news"] = "Source says returned to training"
    news, audit = build_context(*args, NOW)
    assert len(audit["changed_status_news_metadata"]) == 1
    assert audit["changed_status_news_metadata"][0]["player_id"] == 1
    p = news["evidence_by_fixture"][F.match_id][0]["source_players"][0]
    assert p["news"] == "Source says returned to training"
    assert "probabilities" not in p and "effect" not in p


def test_equal_metadata_instants_with_different_text_formats_are_not_new_news():
    args = inputs()
    for p in args[2]["elements"]:
        p["news_added"] = "2026-10-20T20:30:09.724000Z"
    for p in args[1]["evidence_by_fixture"][F.match_id]:
        p["source_players"][0]["news_added"] = "2026-10-21T04:30:09.724+08:00"
    _, audit = build_context(*args, NOW)
    assert audit["changed_status_news_metadata"] == []


def test_available_rows_not_counted_as_injuries_and_status_u_is_separate():
    args = inputs()
    base = args[2]["elements"][0]
    args[2]["elements"] += [{**base, "id": 3, "status": "a"}, {**base, "id": 4, "status": "u"}]
    news, audit = build_context(*args, NOW)
    home = audit["team_summaries"][0]
    assert home["flagged_players"] == 1 and home["unavailable_other_players"] == 1
    assert [p["id"] for p in news["evidence_by_fixture"][F.match_id][0]["source_players"]] == [1, 4]


def test_unknown_counters_stay_missing_not_zero():
    args = inputs()
    args[2]["elements"][0]["minutes"] = None
    del args[2]["elements"][0]["starts"]
    news, _ = build_context(*args, NOW)
    p = news["evidence_by_fixture"][F.match_id][0]["source_players"][0]
    assert "season_minutes" not in p and "season_starts" not in p


def test_old_non_fpl_claims_kept_and_old_fpl_summary_replaced_in_new_copy():
    args = inputs()
    fact = {"id": "club-news", "claim": "Sourced coach statement"}
    args[1]["evidence_by_fixture"][F.match_id].append(fact)
    news, _ = build_context(*args, NOW)
    rows = news["evidence_by_fixture"][F.match_id]
    assert rows[0] == fact
    assert all(not r["id"].startswith("old:") for r in rows)


def test_past_by_clock_is_not_repackaged_as_a_new_prematch_input():
    args = inputs()
    args[0]["records"][0]["fixture"]["kickoff"] = NOW.isoformat()
    news, audit = build_context(*args, NOW)
    assert audit["future_fixtures"] == []
    assert audit["skipped_started_by_clock"] == [F.match_id]
    assert news["evidence_by_fixture"] == {}


@pytest.mark.parametrize("field,value", [("started", True), ("started", None),
    ("finished", True), ("finished_provisional", True), ("team_h", 2),
    ("kickoff_time", (F.kickoff+timedelta(minutes=1)).isoformat())])
def test_conflicting_future_fixture_cannot_receive_a_packet(field, value):
    args = inputs()
    args[3][0][field] = value
    with pytest.raises(ValueError):
        build_context(*args, NOW)


@pytest.mark.parametrize("label", ["bootstrap", "fixtures"])
@pytest.mark.parametrize("change", [{"status": 429}, {"received_at": (NOW+timedelta(seconds=1)).isoformat()}])
def test_failed_or_future_snapshot_not_available(label, change):
    args = inputs()
    args[4]["sources"][label].update(change)
    with pytest.raises(ValueError):
        build_context(*args, NOW)


def test_duplicate_player_ids_are_not_silently_merged():
    args = inputs()
    args[2]["elements"].append(deepcopy(args[2]["elements"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        build_context(*args, NOW)


def test_wrong_source_team_name_rejected_without_fuzzy_matching():
    args = inputs()
    args[2]["teams"][0]["name"] = "Another team"
    with pytest.raises(ValueError, match="name"):
        build_context(*args, NOW)


def test_missing_or_duplicate_old_team_observation_not_silently_invented():
    args = inputs()
    args[1]["evidence_by_fixture"][F.match_id].pop()
    with pytest.raises(ValueError, match="earlier team"):
        build_context(*args, NOW)


def test_older_snapshot_cannot_replace_newer_saved_news():
    args = inputs()
    args[1]["evidence_by_fixture"][F.match_id][0]["observed_at"] = NOW.isoformat()
    with pytest.raises(ValueError, match="newer"):
        build_context(*args, NOW)
