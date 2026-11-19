"""Descriptive player inputs survive projection without leaking other arms."""

from copy import deepcopy
from datetime import timedelta

import pytest

from test_prospective import F, NOW, evidence
from football_model.prospective import build_llm_packet


def player(**changes):
    return {"id": 1, "name": "Synthetic Player", "status": "i", "web_name": "Player",
            "news": "Source injury flag, not a medical diagnosis",
            "news_added": (NOW-timedelta(days=3)).isoformat(),
            "season_minutes": 596, "season_starts": 7, "position": 3, "source_team_id": 1,
            **changes}


def test_preserve_descriptive_player_context_and_strip_probabilities_and_effects():
    source = evidence(source_players=[player(chance_label=0, effect=-.8,
                      probabilities=[.9, .05, .05], decimal_odds=[1.1, 4, 5])],
                      limits=["news_added is API metadata, not injury onset"])
    original = deepcopy(source)
    packet = build_llm_packet(F, NOW, [source])
    row = packet["evidence"][0]
    assert row["source_players"] == [player()]
    assert row["limits"] == source["limits"]
    assert "not injury onset" in packet["instructions"]
    assert "not causal effects" in packet["instructions"]
    assert source == original


def test_empty_news_and_null_metadata_do_not_invent_an_injury_date():
    packet = build_llm_packet(F, NOW, [evidence(source_players=[player(news="", news_added=None)])])
    assert packet["evidence"][0]["source_players"][0]["news_added"] is None
    assert packet["evidence"][0]["source_players"][0]["news"] == ""


@pytest.mark.parametrize("updates", [{"id": True}, {"id": 0}, {"name": ""},
    {"status": None}, {"news": 5}, {"news_added": "2026-10-23"},
    {"news_added": NOW+timedelta(seconds=1)}, {"season_minutes": -1},
    {"season_minutes": 5.5}, {"season_starts": True}, {"position": 0}, {"source_team_id": -2}])
def test_bad_player_data_is_not_transformed_into_facts(updates):
    with pytest.raises(ValueError):
        build_llm_packet(F, NOW, [evidence(source_players=[player(**updates)])])


@pytest.mark.parametrize("players", [None, {}, "players", [player(), player()], [False]])
def test_invalid_player_collection_or_duplicate_ids_rejected(players):
    with pytest.raises(ValueError):
        build_llm_packet(F, NOW, [evidence(source_players=players)])


def test_future_player_evidence_is_ignored_before_payload_validation():
    packet = build_llm_packet(F, NOW, [evidence(observed_at=NOW+timedelta(seconds=1), source_players=None)])
    assert packet["evidence"] == []


@pytest.mark.parametrize("limits", ["unknown", [""], [True]])
def test_limits_are_explicit_text_not_unchecked_payload(limits):
    with pytest.raises(ValueError):
        build_llm_packet(F, NOW, [evidence(source_players=[player()], limits=limits)])
