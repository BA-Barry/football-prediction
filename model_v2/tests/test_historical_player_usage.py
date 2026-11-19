from datetime import date
from copy import deepcopy

import pytest

from football_model.historical_player_usage import extract_usage, usage_features


def observation(day, minutes, mid=None):
    return {'match_id': mid or str(day), 'date': f'2024-01-{day:02}',
            'team_id': 'team', 'minutes': minutes}


def test_cutoff_excludes_same_day_and_future_poison():
    earlier = [observation(1, {'p': 90, 'q': 90})]
    result = usage_features(earlier, team_id='team', before_date=date(2024, 1, 7))
    poisoned = earlier + [observation(7, {'target_only': 900}),
                          observation(8, {'future': -100})]
    assert usage_features(poisoned, team_id='team', before_date=date(2024, 1, 7)) == result


def test_minutes_are_source_counts_and_do_not_repair_to_990():
    payload = {'rosters': {side: {'x': {'player_id': p, 'team_id': t, 'h_a': side,
                                      'time': '90', 'position': 'UNUSED', 'goals': 'POISON'}}
                           for side, p, t in [('h', '1', '10'), ('a', '2', '20')]}}
    fixture = {'match_id': 'fixture', 'date': '2024-01-01', 'home': 'team', 'away': 'other',
               'competition': 'E0', 'season': '2023/24', 'source_url': 'source'}
    rows = extract_usage(payload, fixture)
    assert rows[0]['source_team_minutes'] == 90
    f = usage_features(rows, team_id='team', before_date=date(2024, 1, 2))['features']
    assert f['usage_effective_players_recent5'] == 1
    assert f['usage_reference_weighted_minutes_previous7days'] == 90


def test_usage_shift_compares_disjoint_historical_blocks():
    rows = [observation(day, {'old': 90}) for day in range(1, 6)]
    rows += [observation(day, {'new': 90}) for day in range(6, 11)]
    f = usage_features(rows, team_id='team', before_date=date(2024, 1, 11))['features']
    assert f['usage_comparison_matches'] == 5
    assert f['usage_shift_vs_previous5'] == 1
    assert f['usage_adjacent_overlap_recent5'] == 1
    assert f['usage_matches_previous7days'] == 7


def test_empty_history_is_unknown_and_duplicates_do_not_double_workload():
    result = usage_features([], team_id='team', before_date=date(2024, 1, 10))
    assert result['features']['usage_effective_players_recent5'] is None
    assert result['features']['usage_shift_vs_previous5'] is None
    assert result['history_available'] is False
    row = observation(1, {'p': 90})
    with pytest.raises(ValueError, match='Duplicate'):
        usage_features([row, deepcopy(row)], team_id='team', before_date=date(2024, 1, 10))
