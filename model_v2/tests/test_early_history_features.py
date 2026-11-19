from dataclasses import replace
from datetime import date, timedelta
import json
import math

import pytest

from football_model.early_history_features import EarlyHistoryFeatures, EarlyTeamObservation, build_early_features


START = date(2023, 1, 1)


def observations(n=20):
    rows = []
    for i in range(n):
        for team, opponent, is_home in [('A', 'B', i % 2 == 0), ('B', 'A', i % 2 != 0)]:
            rows.append(EarlyTeamObservation(
                match_id=f'm{i}', match_date=START+timedelta(days=i*7), team=team,
                opponent=opponent, competition='E0', is_home=is_home,
                goals_for=i % 4, goals_against=(i+1) % 4, xg_for=float(i+1), xg_against=2.,
                npxg_for=float(i)+.5, npxg_against=1.5,
                ppda_att=float(i+2), ppda_def=float(i+1),
                ppda_allowed_att=15., ppda_allowed_def=3., deep_for=float(i), deep_against=4.))
    return rows


def feature_context(rows=None, *, cutoff=None, match_date=None, home='A', away='B'):
    cutoff = cutoff or START+timedelta(days=20*7)
    return build_early_features(observations() if rows is None else rows,
        home=home, away=away, competition='E0', cutoff=cutoff,
        match_date=match_date or cutoff+timedelta(days=3))


def test_trailing_windows_and_trends_use_exact_values_and_earlier_ids():
    ctx = feature_context()
    home = ctx['home_features']
    assert home['own_xg_for_mean_5'] == sum(range(16, 21))/5
    assert home['own_xg_for_mean_15'] == sum(range(6, 21))/15
    assert home['own_trend_xg_for_5_minus_15'] == 5
    assert home['own_history_count'] == 20.
    assert home['own_history_count_5'] == 5.
    assert home['own_history_count_15'] == 15.
    assert ctx['metadata']['history']['home']['used_match_ids'] == [f'm{i}' for i in range(5, 20)]
    assert max(ctx['metadata']['history']['home']['window_match_ids']['5']) == 'm19'
    assert json.loads(json.dumps(ctx, allow_nan=False)) == ctx


def test_ppda_ratio_aggregates_components_instead_of_individual_ratios():
    ctx = feature_context()
    expected = sum(i+2 for i in range(15, 20))/sum(i+1 for i in range(15, 20))
    assert ctx['home_features']['own_ppda_ratio_5'] == expected
    individual_mean = sum((i+2)/(i+1) for i in range(15, 20))/5
    assert not math.isclose(expected, individual_mean, rel_tol=1e-9)


def test_same_and_future_day_measurements_cannot_change_past_features():
    rows = observations(22)
    cutoff = START+timedelta(days=20*7)
    original = feature_context(rows, cutoff=cutoff)
    changed = [replace(row, xg_for=float('nan'), goals_for=True, ppda_def=-999.)
               if row.match_date >= cutoff else row for row in rows]
    assert feature_context(changed, cutoff=cutoff) == original
    assert original['metadata']['history']['home']['latest_used_date'] == (cutoff-timedelta(days=7)).isoformat()


def test_other_competition_measurements_do_not_enter_same_league_history():
    rows = observations()
    unrelated = [replace(r, competition='OTHER', xg_for=10000.) for r in rows]
    assert feature_context(rows+unrelated) == feature_context(rows)


def test_unknown_and_insufficient_team_history_are_explicit():
    unknown = feature_context(home='unknown')
    assert unknown['home_features'] is None and unknown['away_features'] is None
    assert unknown['metadata']['history']['home']['unavailable_reason'] == 'insufficient_league_history'
    assert unknown['metadata']['history']['home']['history_count'] == 0
    assert feature_context(observations(4))['home_features'] is None


def test_five_earlier_games_support_fifteen_window_with_actual_count():
    ctx = feature_context(observations(5))
    assert ctx['home_features']['own_history_count_15'] == 5.
    assert ctx['home_features']['own_xg_for_mean_15'] == 3.
    assert ctx['home_features']['own_trend_xg_for_5_minus_15'] == 0.
    # Home side has three previous home games; opponent's away venue likewise.
    assert ctx['home_features']['own_venue_count_15'] == 3.
    assert ctx['home_features']['own_venue_xg_for_mean_15'] == 3.
    # Reversing the forthcoming venue leaves only two matching earlier games.
    reversed_venue = feature_context(observations(5), home='B', away='A')
    assert reversed_venue['home_features']['own_venue_count_15'] == 2.
    assert reversed_venue['home_features']['own_venue_xg_for_mean_15'] is None


def test_zero_denominator_and_missing_style_observation_remain_missing():
    rows = [replace(r, ppda_def=0.) for r in observations()]
    ctx = feature_context(rows)
    assert ctx['home_features']['own_ppda_ratio_5'] is None
    assert ctx['home_features']['own_ppda_def_mean_5'] == 0.
    changed = [replace(r, deep_for=None) if r.match_id == 'm19' else r for r in rows]
    missing = feature_context(changed)
    assert missing['home_features']['own_deep_for_mean_5'] is None
    assert missing['home_features']['own_deep_for_mean_15'] is None
    assert missing['home_features']['own_trend_deep_for_5_minus_15'] is None


def test_home_away_feature_views_are_mirrored_but_venues_are_relevant():
    ctx = feature_context()
    home, away = ctx['home_features'], ctx['away_features']
    assert home['is_home'] == 1. and away['is_home'] == 0.
    assert home['own_xg_for_mean_5'] == away['opponent_xg_for_mean_5']
    assert home['opponent_npxg_against_mean_15'] == away['own_npxg_against_mean_15']
    assert home['own_venue_xg_for_mean_15'] == away['opponent_venue_xg_for_mean_15']


def test_cached_snapshot_keeps_match_gap_specific_and_outputs_isolated():
    builder = EarlyHistoryFeatures(observations())
    cutoff = START+timedelta(days=20*7)
    one = builder.for_match(home='A', away='B', competition='E0', cutoff=cutoff, match_date=cutoff)
    two = builder.for_match(home='A', away='B', competition='E0', cutoff=cutoff, match_date=cutoff+timedelta(days=4))
    assert two['home_features']['own_same_league_gap_to_match_days']-one['home_features']['own_same_league_gap_to_match_days'] == 4
    assert one['home_features']['own_cutoff_staleness_days'] == two['home_features']['own_cutoff_staleness_days'] == 7.
    one['metadata']['history']['home']['used_match_ids'].append('pollution')
    one['home_features']['own_xg_for_mean_5'] = -1.
    three = builder.for_match(home='A', away='B', competition='E0', cutoff=cutoff, match_date=cutoff)
    assert three['home_features']['own_xg_for_mean_5'] > 0
    assert 'pollution' not in three['metadata']['history']['home']['used_match_ids']


@pytest.mark.parametrize('field,value', [('goals_for', 1.5), ('xg_for', True),
    ('ppda_att', float('inf')), ('deep_against', -1.), ('npxg_against', 3.)])
def test_invalid_earlier_measurements_are_rejected(field, value):
    rows = observations()
    rows[-1] = replace(rows[-1], **{field: value})
    with pytest.raises(ValueError):
        feature_context(rows)


def test_duplicate_earlier_team_fixture_rejected_but_future_metrics_not_used():
    rows = observations()
    with pytest.raises(ValueError, match='Duplicate'):
        feature_context(rows+[rows[-1]])


def test_bad_target_cutoff_and_identity_rejected():
    cutoff = START+timedelta(days=20*7)
    with pytest.raises(ValueError, match='cutoff'):
        feature_context(cutoff=cutoff, match_date=cutoff-timedelta(days=1))
    with pytest.raises(ValueError, match='distinct'):
        feature_context(home='A', away='A')
