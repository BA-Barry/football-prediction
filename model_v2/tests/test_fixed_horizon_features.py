from dataclasses import replace
from datetime import date, timedelta
import json
import math

import pytest

from football_model.all_competition_features import AllCompetitionFeatures, ScheduleClub, ScheduleEvent
from football_model.early_history_features import EarlyTeamObservation
from football_model.fixed_horizon_features import (
    PreparedPredictionFeatures, derive_timing, OMITTED_LEAGUE_DATE_SUFFIXES,
)


A, B = 'football-data:E0:A', 'football-data:E0:B'
P, F, D = date(2024, 1, 10), date(2024, 1, 24), date(2024, 1, 25)


def observations(*, add_between=False):
    result = []
    for index in range(8):
        day = date(2023, 11, 1)+timedelta(days=index*7)
        for team, opponent, home in ((A, B, index % 2 == 0), (B, A, index % 2 != 0)):
            result.append(EarlyTeamObservation(
                f'h{index}', day, team, opponent, 'E0', home,
                1, 2, 1.5, 2., 1., 1.6, 12., 2., 15., 3., 4., 6.))
    if add_between:
        for team, opponent, home in ((A, B, True), (B, A, False)):
            result.append(EarlyTeamObservation(
                'between', date(2024, 1, 20), team, opponent, 'E0', home,
                5, 0, 4.8, .1, 4.5, .1, 5., 2., 8., 3., 10., 1.))
    return result


def schedule(*, add_between=False, extra_events=()):
    events = [ScheduleEvent('old', date(2024, 1, 5), 'eng.1', 'competitive', True, True)]
    if add_between:
        events.append(ScheduleEvent('new-cup', date(2024, 1, 21), 'eng.fa', 'competitive', True, False))
    events.extend(extra_events)
    coverage = [(date(2023, 7, 1), date(2024, 7, 1))]
    return AllCompetitionFeatures([ScheduleClub(A, 'E0', '1', coverage, events),
                                   ScheduleClub(B, 'E0', '2', coverage, events)])


def features(builder=None, *, refresh=True, **overrides):
    kwargs = dict(home=A, away=B, competition='E0', parameter_cutoff=P,
        forecast_cutoff=F, match_date=D, goal_rates=[1.2, .9], xg_rates=[1.4, 1.1], refresh=refresh)
    kwargs.update(overrides)
    return (builder or PreparedPredictionFeatures(observations(), schedule())).for_match(**kwargs)


def test_derive_timing_never_uses_parameters_after_publication_day():
    one = derive_timing(date(2025, 8, 29), 1, season_start_year=2025)
    three = derive_timing(date(2025, 8, 31), 3, season_start_year=2025)
    assert one['forecast_cutoff'] == three['forecast_cutoff'] == date(2025, 8, 28)
    assert one['parameter_cutoff'] == three['parameter_cutoff'] == date(2025, 8, 1)
    # A parameter fit on the publication day is allowed; its labels are earlier.
    exact = derive_timing(date(2025, 8, 30), 1, season_start_year=2025)
    assert exact['parameter_cutoff'] == exact['forecast_cutoff'] == date(2025, 8, 29)


def test_derive_timing_july_floor_and_default_season_are_explicit():
    july = derive_timing(date(2024, 7, 31), 1, season_start_year=2024)
    assert july['parameter_cutoff'] == date(2024, 7, 4)
    january = derive_timing(date(2025, 1, 3), 3)
    assert january['anchor_date'] == date(2024, 8, 1)
    assert january['parameter_cutoff'] <= january['forecast_cutoff'] < january['match_date']


@pytest.mark.parametrize('lead', [0, -1, True, 1.5])
def test_derive_timing_requires_a_positive_calendar_lead(lead):
    with pytest.raises(ValueError):
        derive_timing(D, lead)


def test_parameter_after_forecast_day_is_rejected_even_if_before_match():
    with pytest.raises(ValueError, match='publication'):
        features(parameter_cutoff=F+timedelta(days=1))
    with pytest.raises(ValueError, match='publication'):
        features(forecast_cutoff=D)


def test_new_earlier_measurements_change_refreshed_inputs_only():
    old = PreparedPredictionFeatures(observations(), schedule())
    updated = PreparedPredictionFeatures(observations(add_between=True), schedule(add_between=True))
    assert features(old, refresh=False) == features(updated, refresh=False)
    before, after = features(old), features(updated)
    assert before['performance_features'][0]['own_goals_for_mean_5'] == 1.
    assert after['performance_features'][0]['own_goals_for_mean_5'] == 1.8
    assert before['allcomp_features'][0]['own_allcomp_formal_count_7d'] == 0.
    assert after['allcomp_features'][0]['own_allcomp_formal_count_7d'] == 1.
    assert after['allcomp_history']['history']['home']['last_completed_event']['event_id'] == 'new-cup'


def test_forecast_day_target_and_future_poison_cannot_enter_snapshots():
    clean = observations(add_between=True)
    poison = []
    for shift in (0, 1, 3):
        for row in clean[:2]:
            poison.append(replace(row, match_id=f'poison-{shift}', match_date=F+timedelta(days=shift),
                                  goals_for=True, xg_for=float('nan'), ppda_def=-100.))
    future_schedule = [ScheduleEvent(f'poison-{shift}', F+timedelta(days=shift), 'eng.fa',
                                    'competitive', False, False, 'future-poison') for shift in (0, 1, 3)]
    old = PreparedPredictionFeatures(clean, schedule(add_between=True))
    changed = PreparedPredictionFeatures(clean+poison, schedule(add_between=True, extra_events=future_schedule))
    assert features(old) == features(changed)
    assert features(old, refresh=False) == features(changed, refresh=False)


def test_matched_performance_and_allcomp_keep_same_features_plus40_only():
    context = features()
    for pf, allcomp in zip(context['performance_features'], context['allcomp_features'], strict=True):
        assert len(pf) == 121 and len(allcomp) == 161
        assert all(allcomp[name] == value for name, value in pf.items())
        extra = set(allcomp)-set(pf)
        assert len(extra) == 40 and all('allcomp_' in name for name in extra)
        assert 'allcomp_log1p_forecast_lead_days' not in extra
        assert not any(name.endswith(suffix) for name in allcomp for suffix in OMITTED_LEAGUE_DATE_SUFFIXES)
    assert json.loads(json.dumps(context, allow_nan=False)) == context


def test_missing_history_retains_known_basic_home_and_schedule_values():
    builder = PreparedPredictionFeatures([], schedule())
    context = features(builder)
    for index, pf in enumerate(context['performance_features']):
        assert len(pf) == 121
        assert pf['is_home'] == float(index == 0)
        assert pf['history_features_available'] == 0.
        assert pf['own_goals_for_mean_5'] is None
        assert pf['log_goal_mean_for'] == math.log((1.2, .9)[index])
        assert context['allcomp_features'][index]['own_allcomp_identity_available'] == 1.
    assert context['feature_history']['history']['home']['unavailable_reason'] == 'insufficient_league_history'


def test_missing_schedule_is_unknown_while_performance_remains_available():
    context = features(PreparedPredictionFeatures(observations(), AllCompetitionFeatures([])))
    pf, rich = context['performance_features'][0], context['allcomp_features'][0]
    assert pf['history_features_available'] == 1.
    assert rich['own_allcomp_identity_available'] == 0.
    assert rich['own_allcomp_formal_count_28d'] is None


def test_input_lead_and_parameter_age_are_distinct_and_common_to_both_sets():
    for refresh, snapshot in ((True, F), (False, P)):
        context = features(refresh=refresh)
        expected_lead = math.log1p((D-snapshot).days)
        for field in ('performance_features', 'allcomp_features'):
            for values in context[field]:
                assert values['forecast_log1p_lead_days'] == expected_lead
                assert values['parameter_log1p_age_at_forecast_days'] == math.log1p(14)
        assert context['feature_before_date'] == snapshot.isoformat()
        assert context['parameter_fit_before_date'] == P.isoformat()
        assert context['forecast_before_date'] == F.isoformat()
        assert context['timing']['publication_lead_days'] == 1


@pytest.mark.parametrize('rates', [[0., 1.], [1., float('nan')], [True, 1.], [1.]])
def test_missing_or_invalid_earlier_rates_do_not_become_unknown_team_zeros(rates):
    with pytest.raises(ValueError, match='earlier positive'):
        features(goal_rates=rates)


def test_target_statistics_are_not_part_of_composer_api():
    with pytest.raises(TypeError):
        features(actual_score=[9, 9])


def same_day_features(builder=None, *, refresh=True, **overrides):
    kwargs = dict(home=A, away=B, competition='E0', parameter_cutoff=P,
        forecast_cutoff=D, match_date=D, goal_rates=[1.2, .9], xg_rates=[1.4, 1.1], refresh=refresh)
    kwargs.update(overrides)
    return (builder or PreparedPredictionFeatures(observations(), schedule())).for_same_day_match(**kwargs)


def test_same_day_target_and_future_observations_cannot_enter_either_snapshot():
    clean = observations(add_between=True)
    poison = [replace(row, match_id=f'target-poison-{shift}', match_date=D+timedelta(days=shift),
                      goals_for=True, xg_for=float('nan'), ppda_def=-100.)
              for shift in (0, 1) for row in clean[:2]]
    future = [ScheduleEvent(f'target-poison-{shift}', D+timedelta(days=shift), 'eng.fa',
                            'competitive', True, False)
              for shift in (0, 1)]
    old = PreparedPredictionFeatures(clean, schedule(add_between=True))
    changed = PreparedPredictionFeatures(clean+poison,
        schedule(add_between=True, extra_events=future))
    for refresh in (True, False):
        before = same_day_features(old, refresh=refresh)
        after = same_day_features(changed, refresh=refresh)
        assert before == after
        for meta in after['feature_history']['history'].values():
            assert meta['latest_used_date'] < D.isoformat()
        for meta in after['allcomp_history']['history'].values():
            assert meta['last_completed_event']['date'] < D.isoformat()


def test_same_day_yesterday_information_enters_refreshed_inputs_only():
    clean = observations(add_between=True)
    yesterday = [replace(row, match_id='yesterday', match_date=D-timedelta(days=1),
                         goals_for=3, goals_against=1)
                 for row in clean[:2]]
    event = ScheduleEvent('yesterday-cup', D-timedelta(days=1), 'eng.fa', 'competitive', True, False)
    old = PreparedPredictionFeatures(clean, schedule(add_between=True))
    updated = PreparedPredictionFeatures(clean+yesterday,
        schedule(add_between=True, extra_events=[event]))
    assert same_day_features(old, refresh=False) == same_day_features(updated, refresh=False)
    before, after = same_day_features(old), same_day_features(updated)
    assert after['performance_features'][0]['own_history_count'] == (
        before['performance_features'][0]['own_history_count']+1)
    assert after['feature_history']['history']['home']['used_match_ids'][-1] == 'yesterday'
    assert after['allcomp_features'][0]['own_allcomp_formal_count_7d'] == (
        before['allcomp_features'][0]['own_allcomp_formal_count_7d']+1)
    assert after['allcomp_history']['history']['home']['last_completed_event']['event_id'] == 'yesterday-cup'


def test_same_day_api_requires_same_day_and_preserves_old_positive_lead_rules():
    with pytest.raises(ValueError, match='Same-day publication'):
        same_day_features(forecast_cutoff=D-timedelta(days=1))
    with pytest.raises(ValueError, match='Same-day publication'):
        same_day_features(parameter_cutoff=D+timedelta(days=1))
    with pytest.raises(ValueError, match='publication'):
        features(forecast_cutoff=D)
    with pytest.raises(ValueError):
        derive_timing(D, 0)


def test_same_day_and_positive_lead_share_formula_with_explicit_timing():
    builder = PreparedPredictionFeatures(observations(add_between=True), schedule(add_between=True))
    current = same_day_features(builder, parameter_cutoff=D)
    assert current['timing']['publication_lead_days'] == 0
    assert current['timing']['input_lead_days'] == 0
    assert current['timing']['parameter_age_at_publication_days'] == 0
    for lead in (1, 3):
        positive = features(builder, parameter_cutoff=D, forecast_cutoff=D,
                            match_date=D+timedelta(days=lead))
        for key in ('performance_features', 'allcomp_features'):
            for now, later in zip(current[key], positive[key], strict=True):
                assert now['forecast_log1p_lead_days'] == 0.
                assert later['forecast_log1p_lead_days'] == math.log1p(lead)
                assert {name: value for name, value in now.items() if name != 'forecast_log1p_lead_days'} == {
                    name: value for name, value in later.items() if name != 'forecast_log1p_lead_days'}
