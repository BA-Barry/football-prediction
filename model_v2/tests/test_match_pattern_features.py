from dataclasses import replace
from datetime import date, timedelta
import json

import pytest

from football_model.early_history_features import EarlyTeamObservation
from football_model.match_pattern_features import FEATURE_NAMES, MatchPatternFeatures


START = date(2023, 1, 1)


def team_rows(team, pairs, *, competition='E0'):
    return [EarlyTeamObservation(
        match_id=f'{team}{i}', match_date=START+timedelta(days=i*7),
        team=team, opponent=f'{team}_opponent_{i}', competition=competition,
        is_home=i % 2 == 0, goals_for=i % 3, goals_against=1,
        xg_for=first, xg_against=second, npxg_for=first, npxg_against=second,
        ppda_att=None, ppda_def=None, ppda_allowed_att=None, ppda_allowed_def=None,
        deep_for=None, deep_against=None) for i, (first, second) in enumerate(pairs)]


def context(rows=None, provider=None, *, cutoff=None, match_date=None, home='A', away='B'):
    if rows is None:
        rows = team_rows('A', [(1., 1.)]*20)+team_rows('B', [(1., 1.)]*20)
    if provider is None:
        provider = {(row.competition, row.match_id): .25 for row in rows}
    cutoff = cutoff or START+timedelta(days=20*7)
    return MatchPatternFeatures(rows, provider).for_match(
        home=home, away=away, competition='E0', cutoff=cutoff,
        match_date=match_date or cutoff+timedelta(days=3))


def test_equal_means_retain_different_match_balance_and_covariance():
    rows = team_rows('A', [(1., 1.)]*10)+team_rows('B', [(2., 0.), (0., 2.)]*5)
    result = context(rows)
    features = result['features']
    assert len(features) == 24 and set(features) == set(FEATURE_NAMES)
    assert features['pattern_own_mean_abs_npxg_difference_15'] == 0.
    assert features['pattern_opponent_mean_abs_npxg_difference_15'] == 2.
    assert features['pattern_own_npxg_for_against_population_covariance_15'] == 0.
    assert features['pattern_opponent_npxg_for_against_population_covariance_15'] == -1.
    assert features['pattern_opponent_npxg_total_population_variance_15'] == 0.
    assert result['metadata']['history']['away']['windows']['15']['count'] == 10
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_target_and_future_pollution_is_never_read():
    rows = team_rows('A', [(1., 2.)]*22)+team_rows('B', [(2., 1.)]*22)
    cutoff = START+timedelta(days=20*7)
    provider = {(row.competition, row.match_id): .25 for row in rows}
    original = context(rows, provider, cutoff=cutoff)
    changed = [replace(row, goals_for=True, goals_against=-3, npxg_for=float('nan'),
                       npxg_against=-99.) if row.match_date >= cutoff else row for row in rows]
    contaminated_provider = {key: (float('inf') if int(key[1][1:]) >= 20 else value)
                             for key, value in provider.items()}
    assert context(changed, contaminated_provider, cutoff=cutoff) == original
    window = original['metadata']['history']['home']['windows']['15']
    assert window['match_ids'] == [f'A{i}' for i in range(5, 20)]
    assert all(day < cutoff.isoformat() for day in window['dates'])


def test_provider_missingness_is_per_window_without_shortening_or_zero_fill():
    rows = team_rows('A', [(1., 2.)]*20)+team_rows('B', [(2., 1.)]*20)
    provider = {(row.competition, row.match_id): .4 for row in rows}
    provider.pop(('E0', 'A10'))
    result = context(rows, provider)
    features = result['features']
    assert features['pattern_own_provider_draw_probability_mean_5'] == .4
    assert features['pattern_own_provider_draw_probability_population_variance_5'] == 0.
    assert features['pattern_own_provider_draw_probability_mean_15'] is None
    assert features['pattern_own_provider_draw_probability_population_variance_15'] is None
    assert features['pattern_own_mean_abs_npxg_difference_15'] == 1.
    assert features['pattern_opponent_provider_draw_probability_mean_15'] == .4
    window = result['metadata']['history']['home']['windows']['15']
    assert window['count'] == 15 and window['provider_available_count'] == 14
    assert window['provider_missing_match_ids'] == ['A10']
    provider[('E0', 'A10')] = None
    assert context(rows, provider) == result


def test_population_statistics_and_draw_fraction_use_selected_window_denominator():
    rows = team_rows('A', [(10., 10.), (.5, .5), (1., 1.), (1.5, 1.5),
                           (2., 2.), (2.5, 2.5)])+team_rows('B', [(1., 1.)]*6)
    provider = {('E0', f'A{i}'): value for i, value in enumerate((.9, 0., .2, .4, .6, .8))}
    result = context(rows, provider)['features']
    assert result['pattern_own_npxg_total_population_variance_5'] == 2.
    assert result['pattern_own_npxg_for_against_population_covariance_5'] == .5
    assert result['pattern_own_actual_draw_fraction_5'] == 2/5
    assert result['pattern_own_provider_draw_probability_mean_5'] == .4
    assert result['pattern_own_provider_draw_probability_population_variance_5'] == pytest.approx(.08)


def test_five_games_are_required_and_window_counts_stay_explicit():
    rows = team_rows('A', [(1., 1.)]*4)+team_rows('B', [(1., 1.)]*5)
    unavailable = context(rows)
    assert unavailable['features'] is None
    assert unavailable['metadata']['history']['home']['unavailable_reason'] == 'insufficient_league_history'
    assert context(home='unknown')['features'] is None
    available = context(team_rows('A', [(1., 1.)]*5)+team_rows('B', [(1., 1.)]*5))
    assert len(available['features']) == 24
    for side in ('home', 'away'):
        assert available['metadata']['history'][side]['windows']['15']['count'] == 5


def test_home_away_roles_and_competition_align_with_source_identities():
    rows = team_rows('A', [(1., 1.)]*5)+team_rows('B', [(3., 1.)]*5)
    rows += [replace(row, competition='OTHER', npxg_for=float('nan')) for row in rows]
    provider = {('E0', f'{team}{i}'): value for team, value in [('A', .2), ('B', .5)] for i in range(5)}
    forward = context(rows, provider)
    reversed_roles = context(rows, provider, home='B', away='A')
    for name, value in forward['features'].items():
        reversed_name = name.replace('pattern_own_', 'pattern_TMP_').replace(
            'pattern_opponent_', 'pattern_own_').replace('pattern_TMP_', 'pattern_opponent_')
        assert reversed_roles['features'][reversed_name] == value
    assert forward['features']['pattern_own_provider_draw_probability_mean_5'] == .2
    assert forward['features']['pattern_opponent_mean_abs_npxg_difference_5'] == 2.
    assert forward['metadata']['history']['away']['windows']['5']['match_ids'] == [f'B{i}' for i in range(5)]


@pytest.mark.parametrize('field,value', [('goals_for', True), ('goals_against', 1.5),
    ('goals_for', -1), ('npxg_for', float('nan')), ('npxg_against', -1.)])
def test_invalid_selected_earlier_measurements_are_rejected(field, value):
    rows = team_rows('A', [(1., 1.)]*5)+team_rows('B', [(1., 1.)]*5)
    rows[0] = replace(rows[0], **{field: value})
    with pytest.raises(ValueError, match='measurement'):
        context(rows)


@pytest.mark.parametrize('value', [-.1, 1.1, float('nan'), True])
def test_invalid_selected_provider_probabilities_are_rejected(value):
    with pytest.raises(ValueError, match='measurement'):
        context(provider={('E0', 'A19'): value})


def test_dates_duplicate_history_and_target_identity_are_checked():
    cutoff = START+timedelta(days=20*7)
    with pytest.raises(ValueError, match='cutoff'):
        context(cutoff=cutoff, match_date=cutoff-timedelta(days=1))
    with pytest.raises(ValueError, match='calendar'):
        context(cutoff=cutoff.isoformat(), match_date=cutoff)
    with pytest.raises(ValueError, match='distinct'):
        context(home='A', away='A')
    rows = team_rows('A', [(1., 1.)]*5)+team_rows('B', [(1., 1.)]*5)
    with pytest.raises(ValueError, match='Duplicate'):
        context(rows+[rows[0]])


def test_cached_metadata_and_features_cannot_be_changed_by_caller():
    rows = team_rows('A', [(1., 1.)]*5)+team_rows('B', [(1., 1.)]*5)
    builder = MatchPatternFeatures(rows, {})
    args = dict(home='A', away='B', competition='E0', cutoff=START+timedelta(days=40),
                match_date=START+timedelta(days=43))
    first = builder.for_match(**args)
    first['features']['pattern_own_mean_abs_npxg_difference_5'] = -1
    first['metadata']['history']['home']['windows']['5']['match_ids'].append('polluted')
    second = builder.for_match(**args)
    assert second['features']['pattern_own_mean_abs_npxg_difference_5'] == 0.
    assert second['metadata']['history']['home']['windows']['5']['match_ids'][-1] == 'A4'
