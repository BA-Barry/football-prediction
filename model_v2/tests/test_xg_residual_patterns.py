from copy import deepcopy
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from football_model.xg_residual_patterns import FEATURE_NAMES, summarize_residual_patterns
from xg_residual_pattern_features_20261027 import ADDED_FEATURE_NAMES, build_residual_pattern_rows


def pair(first, second):
    return {'residual_for': first, 'residual_against': second}


def fixture(mid, played, home='A', away='B', rates=(1., 1.)):
    forecast = (date.fromisoformat(played)-timedelta(days=3)).isoformat()
    return {'match_id': mid, 'date': played, 'forecast_before_date': forecast,
            'fit_before_date': forecast, 'competition': 'E0', 'period': '2024-25',
            'home': home, 'away': away, 'actual_score': [1, 0], 'xg_rates': list(rates),
            'fresh_performance_features': [{f'old_{i}': float(i) for i in range(121)} for _ in (0, 1)]}


def sample():
    rows, records = [], {}
    for i in range(5):
        home, away = ('A', 'B') if i % 2 == 0 else ('B', 'A')
        row = fixture(f'm{i}', f'2024-01-{i+10:02d}', home, away,
                      (1., 2.) if i % 2 == 0 else (2., 1.))
        rows.append(row)
        # A residual for=+0.5, against=-0.25 in both venue orientations.
        xg = (1.5, 1.75) if i % 2 == 0 else (1.75, 1.5)
        records[row['match_id']] = SimpleNamespace(match_id=row['match_id'],
            match_date=date.fromisoformat(row['date']), home=home, away=away,
            competition='E0', goals=(1, 0), xg=xg)
    target = fixture('target', '2024-02-01')
    target['fresh_feature_history'] = {'history': {
        side: {'team': target[side], 'competition': 'E0',
               'window_match_ids': {'5': [row['match_id'] for row in rows]}}
        for side in ('home', 'away')}}
    return rows+[target], records


def test_same_net_mean_can_hide_different_attack_defence_and_total():
    first = summarize_residual_patterns([pair(1., 0.)]*5)
    second = summarize_residual_patterns([pair(2., 1.)]*5)
    assert first['xg_residual_net_mean_5'] == second['xg_residual_net_mean_5'] == 1.
    assert first['xg_residual_total_mean_5'] == 1.
    assert second['xg_residual_total_mean_5'] == 3.
    assert first['xg_residual_against_mean_5'] != second['xg_residual_against_mean_5']


def test_same_means_can_hide_variation_and_covariance_uses_population_denominator():
    stable = summarize_residual_patterns([pair(0., 0.)]*5)
    varying = summarize_residual_patterns([pair(a, -a) for a in (-2., -1., 0., 1., 2.)])
    assert varying['xg_residual_for_mean_5'] == stable['xg_residual_for_mean_5'] == 0.
    assert varying['xg_residual_against_mean_5'] == 0.
    assert varying['xg_residual_net_population_variance_5'] == 8.
    assert varying['xg_residual_total_population_variance_5'] == 0.
    assert varying['xg_residual_for_against_population_covariance_5'] == -2.
    assert varying['xg_residual_mean_abs_net_5'] == 2.4


def test_missing_window_is_entirely_unknown_and_invalid_values_are_not_zeroed():
    assert summarize_residual_patterns([pair(-1., 2.)]*4) == dict.fromkeys(FEATURE_NAMES)
    assert summarize_residual_patterns(None) == dict.fromkeys(FEATURE_NAMES)
    for value in (True, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='finite'):
            summarize_residual_patterns([pair(value, 0.)]*5)
    with pytest.raises(ValueError, match='five-match'):
        summarize_residual_patterns([pair(0., 0.)]*6)


def test_home_away_orientation_and_opponent_exchange_preserve_original_columns():
    rows, records = sample()
    original = deepcopy(rows)
    derived, report = build_residual_pattern_rows(rows, records)
    target = derived[-1]
    home, away = target['residual_pattern_features']
    assert len(home) == len(away) == 137
    assert len(ADDED_FEATURE_NAMES) == 16
    assert home['own_xg_residual_for_mean_5'] == .5
    assert home['own_xg_residual_against_mean_5'] == -.25
    assert home['opponent_xg_residual_for_mean_5'] == -.25
    for key in FEATURE_NAMES:
        assert home[f'own_{key}'] == away[f'opponent_{key}']
        assert home[f'opponent_{key}'] == away[f'own_{key}']
    assert rows == original
    assert all(home[k] == v for k, v in rows[-1]['fresh_performance_features'][0].items())
    assert report['maximum_component_net_difference'] < 1e-12
    assert report['dropped_rows'] == 0


def test_missing_one_reference_does_not_take_an_older_available_fixture():
    rows, records = sample()
    older = fixture('older', '2024-01-01')
    rows.insert(0, older)
    records['older'] = SimpleNamespace(match_id='older', match_date=date(2024, 1, 1),
        home='A', away='B', competition='E0', goals=(1, 0), xg=(4., 0.))
    records.pop('m2')
    derived, _ = build_residual_pattern_rows(rows, records)
    target = derived[-1]
    assert all(target['residual_pattern_features'][0][k] is None for k in ADDED_FEATURE_NAMES)
    history = target['residual_pattern_history']['history']['home']
    assert history['reference_match_ids'] == [f'm{i}' for i in range(5)]
    assert history['available_residual_count'] == 4
    assert 'older' not in [row['match_id'] for row in history['observations']]
    assert history['missing_references'] == [{'match_id': 'm2', 'reason': 'exact_actual_xg_source_missing'}]


def test_target_or_future_reference_is_rejected_and_unused_target_measurements_do_not_enter():
    rows, records = sample()
    initial, _ = build_residual_pattern_rows(rows, records)
    altered = deepcopy(rows)
    altered[-1]['actual_score'] = [90, 90]
    altered[-1]['xg_rates'] = [99., 99.]
    future = fixture('unused_future', '2024-03-01')
    repeated, _ = build_residual_pattern_rows(altered+[future], records)
    assert repeated[-2]['residual_pattern_features'] == initial[-1]['residual_pattern_features']
    for bad in ('target', 'future'):
        polluted = deepcopy(rows)
        if bad == 'future':
            polluted.append(future)
            identifier = 'unused_future'
        else:
            identifier = 'target'
        polluted[5]['fresh_feature_history']['history']['home']['window_match_ids']['5'][0] = identifier
        with pytest.raises(ValueError, match='crosses target cutoff'):
            build_residual_pattern_rows(polluted, records)
