"""Independent identity and time-isolation checks for source non-penalty means."""
from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

from football_model.football_data import parse_row
from football_model.npxg_history import (
    HistoricalNpxgMatch, DateTeamNpxgModel, join_understat_npxg,
)
from football_model.xg_history import DateTeamXgModel


def source():
    row = {'id': '12345', 'isResult': True, 'datetime': '2024-08-17 14:00:00',
           'h': {'id': '88', 'title': 'Manchester City'},
           'a': {'id': '89', 'title': 'Wolverhampton Wanderers'},
           'goals': {'h': '2', 'a': '1'}, 'xG': {'h': '1.875', 'a': '.8'},
           'forecast': {'w': 'not used'}}
    home = {'date': row['datetime'], 'h_a': 'h', 'xG': 1.875, 'xGA': .8,
            'npxG': 1.1, 'npxGA': .8, 'xpts': 999}
    away = {'date': row['datetime'], 'h_a': 'a', 'xG': .8, 'xGA': 1.875,
            'npxG': .8, 'npxGA': 1.1, 'xpts': -999}
    return {'dates': [row], 'teams': {
        '88': {'id': '88', 'title': 'Manchester City', 'history': [home]},
        '89': {'id': '89', 'title': 'Wolverhampton Wanderers', 'history': [away]},
    }, 'players': ['not used'], 'rosters': ['not used']}


def independent():
    return parse_row({'Div': 'E0', 'Date': '17/08/2024', 'HomeTeam': 'Man City',
                      'AwayTeam': 'Wolves', 'FTHG': '2', 'FTAG': '1', 'FTR': 'H'},
                     competition='E0', season='2024/25', source_path='fixture.csv',
                     source_url='https://example.test/fixture.csv')[0]


def observations():
    start = date(2024, 1, 1)
    pairs = [('A', 'B'), ('B', 'C'), ('C', 'A'), ('B', 'A'), ('C', 'B'), ('A', 'C')] * 4
    return [HistoricalNpxgMatch(
        f'np-{i}', start+timedelta(days=i), h, a,
        (1.0+(i % 5)*.4, .8+(i % 4)*.375), (i % 3, (i+1) % 4),
        str(100+i), f'https://understat.com/match/{100+i}',
        npxg=(0.0 if i % 7 == 0 else .25+(i % 5)*.4, .125+(i % 4)*.375))
        for i, (h, a) in enumerate(pairs)]


def test_exact_source_np_join_preserves_original_totals_and_does_not_mutate_payload():
    payload = source()
    before = deepcopy(payload)
    rows, report = join_understat_npxg(payload, [independent()], '2024/25')
    assert payload == before
    assert len(rows) == report['npxg_exact_matches'] == 1
    record = rows[0]
    assert record.xg == (1.875, .8)
    assert record.npxg == (1.1, .8)
    assert record.goals == (2, 1)
    assert record.home == 'football-data:E0:Man City'
    assert record.away == 'football-data:E0:Wolves'


def test_no_same_match_forecast_roster_player_or_other_history_fields_used():
    payload = source()
    expected, _ = join_understat_npxg(payload, [independent()], '2024/25')
    payload['dates'][0]['forecast'] = None
    payload['players'] = {'unexpected': True}
    payload['rosters'] = None
    for team in payload['teams'].values():
        team['npxG'] = 'invalid season aggregate'
        team['history'][0]['xpts'] = object()
        team['history'][0]['scored'] = 999999
    actual, _ = join_understat_npxg(payload, [independent()], '2024/25')
    assert actual == expected


@pytest.mark.parametrize('mutation', [
    lambda p: p['teams']['88'].update(id='99'),
    lambda p: p['teams']['88'].update(title='different club'),
    lambda p: p['teams']['88'].update(history=[]),
    lambda p: p['teams']['88']['history'].append(deepcopy(p['teams']['88']['history'][0])),
    lambda p: p['teams']['88']['history'][0].update(date='2024-08-17 14:01:00'),
    lambda p: p['teams']['88']['history'][0].update(h_a='a'),
    lambda p: p['teams']['88']['history'][0].pop('npxGA'),
    lambda p: p['dates'][0]['h'].update(id=True),
])
def test_missing_ambiguous_or_inexact_side_identity_cannot_join(mutation):
    payload = source()
    mutation(payload)
    with pytest.raises(ValueError):
        join_understat_npxg(payload, [independent()], '2024/25')


@pytest.mark.parametrize('field,value', [
    ('xG', 1.8), ('xGA', .9), ('npxG', 2.0), ('npxGA', .9),
    ('npxG', -.1), ('npxG', float('nan')), ('npxG', float('inf')),
    ('npxG', True), ('npxGA', False), ('npxG', None),
])
def test_invalid_measurement_or_opposing_source_disagreement_rejected(field, value):
    payload = source()
    payload['teams']['88']['history'][0][field] = value
    with pytest.raises(ValueError):
        join_understat_npxg(payload, [independent()], '2024/25')


def test_original_independent_score_join_still_applies():
    payload = source()
    payload['dates'][0]['goals']['h'] = '3'
    with pytest.raises(ValueError, match='score disagrees'):
        join_understat_npxg(payload, [independent()], '2024/25')


def test_np_mean_fit_matches_known_np_quasi_loss_and_preserves_records():
    data = observations()
    before = deepcopy(data)
    cutoff = date(2024, 2, 1)
    reference = DateTeamXgModel().fit([replace(r, xg=r.npxg) for r in data], cutoff)
    model = DateTeamNpxgModel().fit(data, cutoff)
    assert data == before
    assert model.predict('A', 'B', 'E0', cutoff) == reference.predict('A', 'B', 'E0', cutoff)
    metadata = model.parameters()
    assert metadata['response'] == 'provider_match_npxg'
    assert 'not a full-goal rate' in metadata['rate_use']
    assert metadata['reference_match_ids'] == reference.parameters()['reference_match_ids']


def test_same_day_future_np_and_future_teams_cannot_affect_fit():
    data = observations()
    cutoff = date(2024, 1, 21)
    expected = DateTeamNpxgModel().fit(data[:20], cutoff)
    changed = deepcopy(data)
    for record in changed[20:]:
        record.npxg = (float('nan'), True)
        record.home = 'Future club'
    actual = DateTeamNpxgModel().fit(changed, cutoff)
    assert actual.parameters() == expected.parameters()
    assert actual.parameters()['training_max_date'] == '2024-01-20'
    assert actual.parameters()['n_matches'] == 20
    with pytest.raises(ValueError, match='known'):
        actual.predict('Future club', 'A', 'E0', cutoff)


def test_full_goals_and_total_xg_are_not_the_fit_response():
    data = observations()
    cutoff = date(2024, 2, 1)
    expected = DateTeamNpxgModel().fit(data, cutoff)
    changed = deepcopy(data)
    for record in changed:
        record.goals = (999, 888)
        record.xg = (900., 800.)
    actual = DateTeamNpxgModel().fit(changed, cutoff)
    assert actual.parameters() == expected.parameters()


@pytest.mark.parametrize('bad', [(-.1, .2), (float('nan'), .2), (.2, float('inf')),
                                 (True, .2), (.2, False), (.2,), (.2, .3, .4), (90., .2)])
def test_bad_np_fit_measurements_not_repaired_or_zero_filled(bad):
    data = observations()
    data[0].npxg = bad
    with pytest.raises(ValueError):
        DateTeamNpxgModel().fit(data, date(2024, 2, 1))


def test_prediction_still_requires_fit_known_teams_and_nonpast_date():
    model = DateTeamNpxgModel()
    with pytest.raises(ValueError):
        model.predict('A', 'B', 'E0', date(2024, 2, 1))
    model.fit(observations(), date(2024, 2, 1))
    for h, a, comp, played in [('A', 'A', 'E0', date(2024, 2, 1)),
                              ('A', 'B', 'D1', date(2024, 2, 1)),
                              ('A', 'B', 'E0', date(2024, 1, 31))]:
        with pytest.raises(ValueError):
            model.predict(h, a, comp, played)
