from collections import defaultdict
import pytest

from build_personnel_semantic_packets_20261028 import usage_evidence, llm_semantic_view, materialize_match


def records():
    ids = ['m' + str(i) for i in range(5)]
    team = 'football-data:E0:Example'
    minutes = {(m, team): {'date': '2024-08-01', 'team_id': team,
                           'minutes': {'understat:player:9': 90}} for m in ids}
    mapping = {100: {'24-25': '7', 'understat': '8'}}
    players = [{'source_player_code': 100, 'current': {'source_player_id': 7}}]
    return ids, team, minutes, mapping, players


def test_complete_zero_and_missing_window_are_distinct():
    ids, team, minutes, mapping, players = records()
    zero = usage_evidence('2024-25', '2024-08-10', team, ids, players, mapping, minutes, {})
    assert zero['players'][100]['window_minutes'] == 0
    assert zero['players'][100]['prior_positive_same_team_minutes_observed'] is False
    minutes.pop((ids[0], team))
    unknown = usage_evidence('2024-25', '2024-08-10', team, ids, players, mapping, minutes, {})
    assert unknown['players'][100]['window_minutes'] is None
    assert unknown['actual_observed_team_minutes'] is None


def test_conflicting_same_season_identity_remains_unknown():
    ids, team, minutes, mapping, players = records()
    players[0]['current']['source_player_id'] = 99
    result = usage_evidence('2024-25', '2024-08-10', team, ids, players, mapping, minutes, {})
    assert result['players'][100]['understat_player_id'] is None
    assert result['players'][100]['window_minutes'] is None


def test_reference_on_forecast_date_is_rejected():
    ids, team, minutes, mapping, players = records()
    minutes[ids[0], team]['date'] = '2024-08-10'
    with pytest.raises(ValueError, match='strictly before'):
        usage_evidence('2024-25', '2024-08-10', team, ids, players, mapping, minutes, {})


def test_earlier_positive_not_lost_when_recent_window_is_zero():
    ids, team, minutes, mapping, players = records()
    history = {team: [{'date': '2023-08-10', 'minutes': {'understat:player:8': 5}},
                      {'date': '2024-08-11', 'minutes': {'understat:player:10': 100}}]}
    result = usage_evidence('2024-25', '2024-08-10', team, ids, players, mapping, minutes, history)
    assert result['players'][100]['window_minute_share'] == 0
    assert result['players'][100]['prior_positive_same_team_minutes_observed'] is True


def test_llm_view_excludes_future_catalogue_counts_and_observation_spans():
    source = {'news_id': 'n1', 'raw_status': 'i', 'raw_news': 'Knee injury',
              'semantics': {'text_categories': ['medical_reference']},
              'last_snapshot_proxy_at': '2026-05-01', 'target_match_count': 100,
              'source_player_observation_count': 1000}
    assert set(llm_semantic_view(source)) == {'news_id', 'raw_status', 'raw_news', 'semantics'}


def test_match_rendering_includes_only_its_cited_news_and_fixture():
    text = {'news_id': 'n1', 'raw_status': 'a', 'raw_news': '', 'semantics': {},
            'target_match_count': 500}
    row = {'match_id': 'm1', 'personnel_context': {'teams': [{'players': [
        {'current': {'news_id': 'n1'}, 'references': [{'observation': None}]}]}]}}
    payload = {'created_at_utc': '2026-10-29', 'purpose': 'research', 'instructions': [],
               'rows': [row, {'match_id': 'later'}],
               'news_semantics': {'n1': text, 'future_text': {'news_id': 'future_text'}}}
    rendered = materialize_match(payload, 'm1')
    assert rendered['record']['match_id'] == 'm1'
    assert set(rendered['news_semantics']) == {'n1'}
    assert 'target_match_count' not in rendered['news_semantics']['n1']


def test_match_rendering_resolves_text_ids_of_prior_roster_only_records():
    older = {'news_id': 'older', 'raw_status': 'u', 'raw_news': 'Loaned to Example', 'semantics': {}}
    row = {'match_id': 'm1', 'personnel_context': {'teams': [{'players': [], 'reference_sources': [
        {'reference_only_players': [{'news_id': 'older'}]}]}]}}
    payload = {'created_at_utc': '2026-10-29', 'purpose': 'research', 'instructions': [],
               'rows': [row], 'news_semantics': {'older': older}}
    assert set(materialize_match(payload, 'm1')['news_semantics']) == {'older'}
