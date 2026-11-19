from dataclasses import replace
from datetime import date
import json
import math

import pytest

from football_model.all_competition_features import AllCompetitionFeatures, ScheduleClub, ScheduleEvent


TEAM_A, TEAM_B = 'football-data:E0:A', 'football-data:E0:B'


def event(identifier, day, *, slug='eng.1', completed=True, reason=None, category='competitive', regular=True):
    return ScheduleEvent(identifier, date(2024, 1, day), slug, category, completed, regular, reason)


def builder(events=None, coverage=None):
    events = events if events is not None else [event('league1', 4), event('cup', 10, slug='eng.fa', regular=False), event('league2', 17)]
    coverage = coverage or [(date(2023, 7, 1), date(2024, 7, 1))]
    return AllCompetitionFeatures([ScheduleClub(TEAM_A, 'E0', '1', coverage, events),
                                  ScheduleClub(TEAM_B, 'E0', '2', coverage, [event('opponent', 16)])])


def forecast(source=None, *, cutoff=date(2024, 1, 20), match_date=date(2024, 1, 25), home=TEAM_A, away=TEAM_B):
    return (source or builder()).for_match(home=home, away=away, competition='E0', cutoff=cutoff, match_date=match_date)


def test_counts_use_all_competitions_at_cutoff_and_nonleague_is_distinct():
    context = forecast()
    features = context['home_features']
    assert features['own_allcomp_formal_count_7d'] == 1.
    assert features['own_allcomp_formal_count_14d'] == 2.
    assert features['own_allcomp_formal_count_28d'] == 3.
    assert features['own_allcomp_nonleague_count_7d'] == 0.
    assert features['own_allcomp_nonleague_count_14d'] == 1.
    assert features['own_allcomp_last_log1p_gap_to_cutoff_days'] == math.log1p(3)
    assert features['own_allcomp_previous3_log1p_mean_interval_days'] == math.log1p(6.5)
    assert features['allcomp_log1p_forecast_lead_days'] == math.log1p(5)
    assert context['metadata']['history']['home']['windows']['14']['selected_event_ids'] == ['cup', 'league2']
    assert json.loads(json.dumps(context, allow_nan=False)) == context


def test_future_and_same_day_events_cannot_enter_any_earlier_count_or_gap():
    earlier = [event('league1', 4), event('cup', 10, slug='eng.fa', regular=False), event('league2', 17)]
    with_future = builder(earlier+[event('cutoffday', 20), event('futurecup', 22, slug='eng.fa', reason='poison'), event('target', 25)])
    assert forecast(with_future) == forecast(builder(earlier))


def test_target_date_moves_only_lead_not_retrospective_schedule_windows():
    source = builder()
    first, second = forecast(source), forecast(source, match_date=date(2024, 2, 10))
    a, b = first['home_features'].copy(), second['home_features'].copy()
    assert a.pop('allcomp_log1p_forecast_lead_days') != b.pop('allcomp_log1p_forecast_lead_days')
    assert a == b


def test_window_left_boundary_inclusive_and_cutoff_exclusive():
    ctx = forecast(builder([event('left', 13), event('too_old', 12), event('same_day', 20)]))
    assert ctx['home_features']['own_allcomp_formal_count_7d'] == 1.
    assert ctx['metadata']['history']['home']['windows']['7']['selected_event_ids'] == ['left']


@pytest.mark.parametrize('reason,completed', [('partial', True), ('resumed', True), (None, False)])
def test_partial_and_uncertain_status_do_not_become_normal_load(reason, completed):
    source = builder([event('known', 10), event('issue', 18, reason=reason, completed=completed)])
    ctx = forecast(source)
    f = ctx['home_features']
    assert f['own_allcomp_formal_count_14d'] is None
    assert f['own_allcomp_nonleague_count_14d'] is None
    assert f['own_allcomp_window_usable_14d'] == 0.
    assert f['own_allcomp_unresolved_count_14d'] == 1.
    assert f['own_allcomp_last_log1p_gap_to_cutoff_days'] is None
    assert ctx['metadata']['history']['home']['windows']['14']['excluded_or_unresolved_events'][0]['event_id'] == 'issue'


def test_completed_event_after_partial_restores_last_gap_but_not_contaminated_window():
    ctx = forecast(builder([event('issue', 10, reason='partial'), event('ordinary', 17)]))
    f = ctx['home_features']
    assert f['own_allcomp_last_log1p_gap_to_cutoff_days'] == math.log1p(3)
    assert f['own_allcomp_formal_count_7d'] == 1.
    assert f['own_allcomp_formal_count_14d'] is None


def test_friendlies_are_excluded_without_fabricating_minutes_or_unknown_formal_load():
    source = builder([event('league', 14), event('friendly', 18, category='friendly_exhibition', slug='club.friendly', regular=False)])
    f = forecast(source)['home_features']
    assert f['own_allcomp_formal_count_7d'] == 1.
    assert f['own_allcomp_nonleague_count_7d'] == 0.
    assert f['own_allcomp_unresolved_count_7d'] == 0.
    assert not any('minute' in key for key in f)


def test_unknown_club_keeps_missing_counts_and_known_identity_flags():
    f = forecast(home='football-data:E0:Unknown')['home_features']
    assert f['own_allcomp_identity_available'] == 0.
    assert f['own_allcomp_formal_count_28d'] is None
    assert f['own_allcomp_last_log1p_gap_to_cutoff_days'] is None
    assert f['opponent_allcomp_formal_count_7d'] == 1.


def test_short_member_scope_does_not_invent_zero_counts_for_unobserved_days():
    source = builder(events=[], coverage=[(date(2024, 1, 15), date(2024, 7, 1))])
    f = forecast(source)['home_features']
    assert f['own_allcomp_covered_days_within_28'] == 5.
    assert f['own_allcomp_formal_count_7d'] is None
    assert f['own_allcomp_unresolved_count_7d'] is None


def test_missing_member_season_does_not_join_a_hundreds_day_rest_gap():
    old = ScheduleEvent('old', date(2022, 5, 1), 'eng.1', 'competitive', True, True)
    coverage = [(date(2021, 7, 1), date(2022, 7, 1)), (date(2023, 7, 1), date(2024, 7, 1))]
    f = forecast(builder([old], coverage))['home_features']
    assert f['own_allcomp_formal_count_28d'] == 0.
    assert f['own_allcomp_last_log1p_gap_to_cutoff_days'] is None
    assert f['own_allcomp_previous3_count'] == 0.


def test_cutoff_outside_member_scope_has_no_measured_zero_counts():
    source = builder(coverage=[(date(2022, 7, 1), date(2023, 7, 1))])
    f = forecast(source)['home_features']
    assert f['own_allcomp_scope_contains_cutoff'] == 0.
    assert f['own_allcomp_formal_count_7d'] is None
    assert f['own_allcomp_previous3_count'] is None


def test_adjacent_member_seasons_form_one_contiguous_scope():
    source = builder(coverage=[(date(2023, 7, 1), date(2024, 1, 1)), (date(2024, 1, 1), date(2024, 7, 1))])
    assert forecast(source) == forecast()


def test_home_away_features_are_exact_own_opponent_mirrors_and_no_is_home_collision():
    ctx = forecast()
    h, a = ctx['home_features'], ctx['away_features']
    for key, value in h.items():
        counterpart = key.replace('own_', 'opponent_', 1) if key.startswith('own_') else key.replace('opponent_', 'own_', 1)
        assert a[counterpart] == value
    assert 'is_home' not in h and all('allcomp_' in k for k in h)


def test_last_three_intervals_missing_until_three_usable_events_exist():
    f = forecast(builder([event('one', 3), event('two', 16)]))['home_features']
    assert f['own_allcomp_previous3_count'] == 2.
    assert f['own_allcomp_previous3_usable'] == 0.
    assert f['own_allcomp_previous3_log1p_mean_interval_days'] is None


def test_irrelevant_completed_scores_are_never_accepted_as_model_inputs(tmp_path):
    season_dir = tmp_path/'2024-25全量'
    season_dir.mkdir()
    events_path = season_dir/'相邻赛季补全_去重历史赛事.json'
    summary_path = season_dir/'相邻赛季补全_全量覆盖与负荷测量.json'
    summary_path.write_text(json.dumps({'known_specials': [{'event_id': 'blocked', 'reason': 'resumed'}]}), encoding='utf-8')
    (tmp_path/'四季覆盖汇总.json').write_text(json.dumps({'seasons': [{'season': '2024-25', 'events_file': str(events_path), 'summary_file': str(summary_path)}]}), encoding='utf-8')
    raw = {'event_id': 'blocked', 'date_utc_text': '2024-10-01T19:00Z', 'competition_slug': 'eng.1',
           'category': 'competitive', 'home_team_id': '1', 'away_team_id': '2', 'completed': True,
           'is_regular_league_fixture': True, 'source_final_score': [999999, -3], 'lineups': ['POISON']}
    clubs = {'1': {'competition': 'E0', 'team_id': '1', 'independent_label': 'A', 'events': [raw]}}
    (season_dir/'相邻赛季补全_逐队历史观察.json').write_text(json.dumps({'clubs': clubs}), encoding='utf-8')
    source = AllCompetitionFeatures.from_directory(tmp_path)
    ctx = forecast(source, cutoff=date(2024, 10, 3), match_date=date(2024, 10, 5))
    assert ctx['home_features']['own_allcomp_formal_count_7d'] is None
    assert source.source_metadata['uses_scores_or_lineups'] is False
    assert source.source_metadata['club_seasons'] == 1
    assert 'source_final_score' not in source._clubs[('E0', TEAM_A)].events[0].__dict__


@pytest.mark.parametrize('bad', [
    {'cutoff': date(2024, 1, 26)}, {'home': TEAM_B}, {'cutoff': '2024-01-20'},
])
def test_invalid_prediction_context_is_rejected(bad):
    with pytest.raises(ValueError):
        forecast(**bad)


def test_duplicate_event_id_is_not_silently_double_counted():
    with pytest.raises(ValueError, match='unique'):
        builder([event('duplicate', 3), event('duplicate', 17)])
