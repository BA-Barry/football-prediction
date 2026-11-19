"""Independent tests of exact xG joins and date-only quasi-loss fitting."""

from copy import deepcopy
from datetime import date, datetime, timedelta
import math

import numpy as np
import pytest

from football_model.football_data import parse_row
from football_model.team_model import team_features
from football_model.xg_history import HistoricalXgMatch, DateTeamXgModel, join_understat


def fd_record(home='Man City', away='Wolves', played='17/08/2024', goals=(2, 1)):
    result = 'H' if goals[0] > goals[1] else 'A' if goals[0] < goals[1] else 'D'
    row = {'Div': 'E0', 'Date': played, 'HomeTeam': home, 'AwayTeam': away,
           'FTHG': str(goals[0]), 'FTAG': str(goals[1]), 'FTR': result}
    return parse_row(row, competition='E0', season='2024/25',
                     source_path='fixture.csv', source_url='https://example.test/E0.csv')[0]


def understat_row():
    return {'id': '12345', 'isResult': True, 'datetime': '2024-08-17 14:00:00',
            'h': {'title': 'Manchester City'}, 'a': {'title': 'Wolverhampton Wanderers'},
            'goals': {'h': '2', 'a': '1'}, 'xG': {'h': '1.875', 'a': '0.0'},
            'forecast': {'w': '.999', 'd': '.0005', 'l': '.0005'}}


def observations():
    start = date(2024, 1, 1)
    pairs = [('A', 'B'), ('B', 'C'), ('C', 'A'), ('B', 'A'), ('C', 'B'), ('A', 'C')] * 4
    return [HistoricalXgMatch(
        f'xg-{i}', start + timedelta(days=i), h, a,
        (0.0 if i % 7 == 0 else .25 + (i % 5)*.4, .125 + (i % 4)*.375),
        (i % 3, (i+1) % 4), str(100+i), f'https://understat.com/match/{100+i}')
        for i, (h, a) in enumerate(pairs)]


def test_exact_alias_date_and_score_join_keeps_fractional_zero_xg():
    row = understat_row()
    parsed, report = join_understat([row], [fd_record()], '2024/25')
    assert len(parsed) == 1
    match = parsed[0]
    assert (match.home, match.away) == ('football-data:E0:Man City', 'football-data:E0:Wolves')
    assert match.match_date == date(2024, 8, 17)
    assert match.xg == (1.875, 0.0)
    assert match.goals == (2, 1)
    assert match.understat_id == '12345'
    assert match.source_url == 'https://understat.com/match/12345'
    assert report['exact_matches'] == report['independent_rows'] == 1
    assert report['missing_independent_match_ids'] == []


def test_same_match_forecast_and_season_aggregates_are_not_joined_features():
    row = understat_row()
    expected, _ = join_understat([row], [fd_record()], '2024/25')
    row['forecast'] = {'w': 'not a probability'}
    row['teams'] = {'season_xg': 999999}
    row['players'] = ['irrelevant']
    actual, _ = join_understat([row], [fd_record()], '2024/25')
    assert actual == expected


def test_missing_independent_fixture_stays_explicit_instead_of_zero_xg():
    extra = fd_record('Arsenal', 'Chelsea', '18/08/2024', (1, 1))
    parsed, report = join_understat([understat_row()], [fd_record(), extra], '2024/25')
    assert len(parsed) == 1
    assert report['missing_independent_match_ids'] == [extra.match_id]


@pytest.mark.parametrize('competition,season,played,home,away,raw_home,raw_away', [
    ('E0', '2020/21', '13/09/2020', 'West Brom', 'Leicester', 'West Bromwich Albion', 'Leicester'),
    ('D1', '2020/21', '19/09/2020', 'Ein Frankfurt', 'Bielefeld', 'Eintracht Frankfurt', 'Arminia Bielefeld'),
    ('D1', '2021/22', '14/08/2021', 'Stuttgart', 'Greuther Furth', 'VfB Stuttgart', 'Greuther Fuerth'),
    ('SP1', '2020/21', '13/09/2020', 'Villarreal', 'Huesca', 'Villarreal', 'SD Huesca'),
])
def test_earlier_year_explicit_team_aliases_keep_exact_fixture_identity(
        competition, season, played, home, away, raw_home, raw_away):
    record, _ = parse_row({'Div': competition, 'Date': played, 'HomeTeam': home, 'AwayTeam': away,
        'FTHG': '0', 'FTAG': '3', 'FTR': 'A'}, competition=competition, season=season,
        source_path='historical.csv', source_url='https://example.test/historical.csv')
    row = understat_row()
    row.update(datetime=record.match_date.isoformat()+' 13:00:00', goals={'h':'0','a':'3'})
    row['h']['title'], row['a']['title'] = raw_home, raw_away
    parsed, report = join_understat([row], [record], season, competition)
    assert len(parsed) == 1 and report['exact_matches'] == 1
    assert parsed[0].match_id == record.match_id
    assert (parsed[0].home, parsed[0].away) == (record.home, record.away)


@pytest.mark.parametrize('competition,season,played,home,away,goals,reason', [
    ('D1', '2021/22', '18/03/2022', 'Bochum', "M'gladbach", (0,2), 'administratively_awarded_incomplete_match'),
    ('I1', '2020/21', '19/09/2020', 'Verona', 'Roma', (0,0), 'administratively_awarded_result'),
    ('F1', '2021/22', '27/10/2021', 'Nice', 'Marseille', (1,1), 'neutral_venue_not_represented'),
])
def test_equal_provider_scores_do_not_hide_incomplete_awarded_or_neutral_history(
        competition, season, played, home, away, goals, reason):
    result = 'H' if goals[0]>goals[1] else 'A' if goals[0]<goals[1] else 'D'
    record, _ = parse_row({'Div': competition, 'Date': played, 'HomeTeam': home, 'AwayTeam': away,
        'FTHG': str(goals[0]), 'FTAG': str(goals[1]), 'FTR': result},
        competition=competition, season=season, source_path='historical.csv',
        source_url='https://example.test/historical.csv')
    row = understat_row()
    row.update(datetime=record.match_date.isoformat()+' 20:00:00',
               goals={'h': str(goals[0]), 'a': str(goals[1])})
    row['h']['title'], row['a']['title'] = home, away
    parsed, report = join_understat([row], [record], season, competition)
    assert record.goal_fit_exclusion == reason
    assert record.full_time == goals
    assert parsed == []
    assert report['excluded_special_matches'][0]['reason'] == reason
    assert report['missing_independent_match_ids'] == []


@pytest.mark.parametrize('mutation', [
    lambda r: r['h'].update(title='Manchester City FC'),
    lambda r: r.update(datetime='2024-08-18 14:00:00'),
    lambda r: r.update(datetime='2025-08-17 14:00:00'),
    lambda r: r.update(datetime='17/08/2024'),
    lambda r: r.update(isResult=False),
    lambda r: r.update(isResult=1),
    lambda r: r.update(id=True),
    lambda r: r['goals'].update(h='3'),
    lambda r: r['goals'].update(h='2.1'),
    lambda r: r['goals'].update(h=True),
])
def test_join_refuses_wrong_identity_date_completion_or_score(mutation):
    row = understat_row()
    mutation(row)
    with pytest.raises(ValueError):
        join_understat([row], [fd_record()], '2024/25')


def test_duplicate_understat_id_or_duplicate_fixture_cannot_overwrite():
    first = understat_row()
    with pytest.raises(ValueError, match='duplicate Understat'):
        join_understat([first, deepcopy(first)], [fd_record()], '2024/25')
    second = deepcopy(first)
    second['id'] = '12346'
    with pytest.raises(ValueError, match='unique exact'):
        join_understat([first, second], [fd_record()], '2024/25')
    with pytest.raises(ValueError, match='Duplicate Football-Data'):
        join_understat([first], [fd_record(), fd_record()], '2024/25')


@pytest.mark.parametrize('bad', [-.01, float('nan'), float('inf'), '-inf', True, False])
def test_join_rejects_invalid_xg_measurements_without_coercion(bad):
    row = understat_row()
    row['xG']['h'] = bad
    with pytest.raises(ValueError):
        join_understat([row], [fd_record()], '2024/25')


def test_fit_ignores_cutoff_day_future_xg_and_future_teams():
    cutoff = date(2024, 1, 21)
    data = observations()
    expected = DateTeamXgModel().fit(data[:20], cutoff)
    changed = deepcopy(data)
    for record in changed[20:]:
        record.xg = (999., 888.)
    changed.append(HistoricalXgMatch('new-team', date(2024, 2, 1), 'X', 'Y',
                                     (500., 600.), (0, 0), '999', 'source'))
    actual = DateTeamXgModel().fit(changed, cutoff)
    assert actual.parameters() == expected.parameters()
    assert actual.parameters()['training_max_date'] == '2024-01-20'
    assert actual.parameters()['n_matches'] == 20
    assert actual.predict('A', 'B', 'E0', cutoff) == expected.predict('A', 'B', 'E0', cutoff)
    with pytest.raises(ValueError, match='known'):
        actual.predict('X', 'A', 'E0', cutoff)


def test_fit_uses_xg_not_same_match_goals_or_market_forecasts():
    cutoff = date(2024, 2, 1)
    data = observations()
    expected = DateTeamXgModel().fit(data, cutoff)
    changed = deepcopy(data)
    for record in changed:
        record.goals = (999, 888)
    actual = DateTeamXgModel().fit(changed, cutoff)
    assert actual.parameters() == expected.parameters()


@pytest.mark.parametrize('bad', [(-.1, .2), (float('nan'), .2), (.2, float('inf')),
                                 (True, .2), (.2, False), (.2,), (.2, .3, .4)])
def test_fit_refuses_bad_per_match_xg_values(bad):
    data = observations()
    data[0].xg = bad
    with pytest.raises(ValueError):
        DateTeamXgModel().fit(data, date(2024, 2, 1))


def test_fractional_and_zero_targets_are_supported_and_metadata_are_explicit():
    model = DateTeamXgModel().fit(observations(), date(2024, 2, 1))
    rates = model.predict('A', 'B', 'E0', date(2024, 2, 2))
    assert all(math.isfinite(v) and v > 0 for v in rates)
    metadata = model.parameters()
    assert metadata['response'] == 'provider_match_xg'
    assert 'fractional' in metadata['loss']
    assert 'experimental' in DateTeamXgModel.__doc__
    assert 'not assumed true' in metadata['rate_use']


@pytest.mark.parametrize('home,away,competition,played', [
    ('X', 'B', 'E0', date(2024, 2, 1)),
    ('A', 'A', 'E0', date(2024, 2, 1)),
    ('A', 'B', 'E1', date(2024, 2, 1)),
    ('A', 'B', 'E0', date(2024, 1, 31)),
    ('A', 'B', 'E0', datetime(2024, 2, 1)),
])
def test_unseen_team_competition_or_earlier_date_is_not_a_zero_effect(home, away, competition, played):
    model = DateTeamXgModel().fit(observations(), date(2024, 2, 1))
    with pytest.raises(ValueError):
        model.predict(home, away, competition, played)


def test_fit_needs_earlier_unique_history_and_calendar_cutoff():
    data = observations()
    with pytest.raises(ValueError):
        DateTeamXgModel().fit(data, date(2024, 1, 1))
    with pytest.raises(ValueError):
        DateTeamXgModel().fit(data + [deepcopy(data[0])], date(2024, 2, 1))
    with pytest.raises(ValueError):
        DateTeamXgModel().fit(data, datetime(2024, 2, 1))
    with pytest.raises(ValueError):
        DateTeamXgModel().predict('A', 'B', 'E0', date(2024, 2, 1))


def test_saved_parameters_replay_predictions_and_independent_quasi_loss_gradient():
    records = observations()
    cutoff = date(2024, 2, 1)
    penalty = .01
    model = DateTeamXgModel(penalty, 180).fit(records, cutoff)
    meta = model.parameters()
    coefs = meta['coefficients']
    replay = [math.exp(meta['intercept'] + sum(coefs.get(k, 0.)*v for k, v in f.items()))
              for f in team_features('A', 'B', 'E0')]
    assert model.predict('A', 'B', 'E0', cutoff) == pytest.approx(replay, abs=1e-12)
    keys = sorted(coefs)
    beta = np.array([meta['intercept']] + [coefs[k] for k in keys])
    design, target, weights = [], [], []
    for r in records:
        for features, y in zip(team_features(r.home, r.away, 'E0'), r.xg):
            design.append([1.] + [features.get(k, 0.) for k in keys])
            target.append(y)
            weights.append(math.exp(-math.log(2)*(cutoff-r.match_date).days/180))
    design, target, weights = np.asarray(design), np.asarray(target), np.asarray(weights)
    weights /= weights.sum()
    def objective(b):
        eta = design @ b
        return float(np.sum(weights*(np.exp(eta)-target*eta)) + penalty*np.dot(b[1:], b[1:])/2)
    gradient = design.T @ (weights*(np.exp(design @ beta)-target))
    gradient[1:] += penalty*beta[1:]
    numeric = np.array([(objective(beta+np.eye(len(beta))[i]*1e-5)-objective(beta-np.eye(len(beta))[i]*1e-5))/2e-5
                        for i in range(len(beta))])
    assert gradient == pytest.approx(numeric, abs=2e-10)
    assert np.max(np.abs(gradient)) < 1e-6
    assert meta['gradient_max_abs'] == pytest.approx(float(np.max(np.abs(gradient))), abs=1e-12)


def league_fd_record(competition, home='A', away='B', played='17/08/2024', goals=(2, 1), season='2024/25'):
    result = 'H' if goals[0] > goals[1] else 'A' if goals[0] < goals[1] else 'D'
    row = {'Div': competition, 'Date': played, 'HomeTeam': home, 'AwayTeam': away,
           'FTHG': str(goals[0]), 'FTAG': str(goals[1]), 'FTR': result}
    return parse_row(row, competition=competition, season=season,
                     source_path=f'{competition}.csv', source_url=f'https://example.test/{competition}.csv')[0]


def league_understat_row(home='A', away='B'):
    row = understat_row()
    row['h']['title'], row['a']['title'] = home, away
    return row


@pytest.mark.parametrize('competition', ['D1', 'SP1', 'I1', 'F1'])
def test_join_uses_explicit_league_and_never_same_names_from_another_league(competition):
    requested = league_fd_record(competition)
    same_names_elsewhere = league_fd_record('E0')
    parsed, report = join_understat([league_understat_row()],
                                    [same_names_elsewhere, requested], '2024/25', competition=competition)
    assert len(parsed) == 1
    assert parsed[0].competition == competition
    assert parsed[0].match_id == requested.match_id
    assert parsed[0].home == f'football-data:{competition}:A'
    assert parsed[0].away == f'football-data:{competition}:B'
    assert report['independent_rows'] == 1
    assert report['missing_independent_match_ids'] == []
    with pytest.raises(ValueError):
        join_understat([league_understat_row()], [same_names_elsewhere], '2024/25', competition=competition)


def test_join_does_not_accept_an_unknown_competition():
    with pytest.raises(ValueError):
        join_understat([league_understat_row()], [league_fd_record('E0')],
                        '2024/25', competition='E1')


def test_legacy_xg_records_still_default_to_epl():
    assert observations()[0].competition == 'E0'
    parsed, _ = join_understat([understat_row()], [fd_record()], '2024/25')
    assert parsed[0].competition == 'E0'


@pytest.mark.parametrize('competition', ['D1', 'SP1', 'I1', 'F1'])
def test_xg_model_learns_and_replays_the_training_league_without_epl_features(competition):
    data = observations()
    for r in data:
        r.competition = competition
    cutoff = date(2024, 2, 1)
    model = DateTeamXgModel().fit(data, cutoff)
    params = model.parameters()
    assert params['competition'] == competition
    assert f'competition:{competition}' in params['coefficients']
    assert f'home:{competition}' in params['coefficients']
    assert 'competition:E0' not in params['coefficients']
    assert 'home:E0' not in params['coefficients']
    replay = tuple(math.exp(params['intercept'] + sum(params['coefficients'].get(k, 0.)*v
                         for k, v in features.items()))
                   for features in team_features('A', 'B', competition))
    assert model.predict('A', 'B', competition, cutoff) == pytest.approx(replay, abs=1e-12)
    with pytest.raises(ValueError):
        model.predict('A', 'B', 'E0', cutoff)


def test_model_refuses_to_pool_multiple_training_leagues_or_unknown_league():
    data = observations()
    data[0].competition = 'D1'
    with pytest.raises(ValueError):
        DateTeamXgModel().fit(data, date(2024, 2, 1))
    for r in data:
        r.competition = 'E1'
    with pytest.raises(ValueError):
        DateTeamXgModel().fit(data, date(2024, 2, 1))


def test_future_observations_from_another_league_do_not_change_the_earlier_fit():
    data = observations()
    cutoff = date(2024, 1, 21)
    expected = DateTeamXgModel().fit(data[:20], cutoff)
    for r in data[20:]:
        r.competition = 'D1'
        r.xg = (999., 888.)
    actual = DateTeamXgModel().fit(data, cutoff)
    assert actual.parameters() == expected.parameters()


@pytest.mark.parametrize('competition', ['D1', 'SP1', 'I1', 'F1'])
def test_ordinary_cross_league_games_still_require_exact_date_and_score(competition):
    record = league_fd_record(competition)
    row = league_understat_row()
    row['goals']['h'] = '4'
    with pytest.raises(ValueError):
        join_understat([row], [record], '2024/25', competition=competition)
    row = league_understat_row()
    row['datetime'] = '2024-08-18 14:00:00'
    with pytest.raises(ValueError):
        join_understat([row], [record], '2024/25', competition=competition)


def test_known_administrative_award_is_excluded_with_both_original_scores_preserved():
    record = league_fd_record('D1', 'Union Berlin', 'Bochum', '14/12/2024', (0, 2))
    original = deepcopy(record)
    row = league_understat_row('Union Berlin', 'Bochum')
    row['datetime'] = '2024-12-14 14:30:00'
    row['goals'] = {'h': '1', 'a': '1'}
    parsed, report = join_understat([row], [record], '2024/25', competition='D1')
    assert parsed == []
    assert record == original
    assert report['missing_independent_match_ids'] == []
    assert report['missing_special_match_ids'] == []
    assert len(report['excluded_special_matches']) == 1
    special = report['excluded_special_matches'][0]
    assert special['match_id'] == record.match_id
    assert special['reason'] == 'administratively_awarded_result'
    assert special['understat_goals'] == [1, 1]
    assert special['football_data_goals'] == [0, 2]
    assert special['understat_date'] == special['football_data_date'] == '2024-12-14'
    assert special['source_url'].startswith('https://www.bundesliga.com/')


@pytest.mark.parametrize('competition,home,away,fd_day,source_day', [
    ('I1', 'Udinese', 'Roma', '25/04/2024', '2024-04-14 16:00:00'),
    ('SP1', 'Granada', 'Ath Bilbao', '11/12/2023', '2023-12-10 15:15:00'),
])
def test_only_registered_resumed_fixture_can_be_excluded_with_distinct_source_dates(
        competition, home, away, fd_day, source_day):
    record = league_fd_record(competition, home, away, fd_day, (1, 2), season='2023/24')
    row = league_understat_row(home, away)
    row['datetime'] = source_day
    row['goals'] = {'h': '1', 'a': '2'}
    parsed, report = join_understat([row], [record], '2023/24', competition=competition)
    assert parsed == []
    assert report['missing_independent_match_ids'] == []
    special = report['excluded_special_matches'][0]
    assert special['reason'] == 'resumed_partially_played_match'
    assert special['understat_date'] == source_day[:10]
    assert special['football_data_date'] == record.match_date.isoformat()
    assert special['understat_date'] != special['football_data_date']
    assert 'not_an_ordinary_date_join' in special['identity']


def test_missing_registered_special_is_separate_from_ordinary_missingness():
    special = league_fd_record('D1', 'Union Berlin', 'Bochum', '14/12/2024', (0, 2))
    ordinary = league_fd_record('D1')
    parsed, report = join_understat([], [special, ordinary], '2024/25', competition='D1')
    assert parsed == []
    assert report['missing_special_match_ids'] == [special.match_id]
    assert report['missing_independent_match_ids'] == [ordinary.match_id]
    assert report['excluded_special_matches'] == []


def test_arbitrary_exclusion_flag_cannot_relax_ordinary_score_or_date_checks():
    ordinary = league_fd_record('D1')
    ordinary.goal_fit_exclusion = 'administratively_awarded_result'
    row = league_understat_row()
    row['goals']['h'] = '4'
    with pytest.raises(ValueError, match='score'):
        join_understat([row], [ordinary], '2024/25', competition='D1')
    row = league_understat_row()
    row['datetime'] = '2024-08-18 14:00:00'
    with pytest.raises(ValueError, match='exact'):
        join_understat([row], [ordinary], '2024/25', competition='D1')


def test_known_fixture_without_matching_registered_reason_is_not_silently_excluded():
    record = league_fd_record('D1', 'Union Berlin', 'Bochum', '14/12/2024', (0, 2))
    record.goal_fit_exclusion = None
    row = league_understat_row('Union Berlin', 'Bochum')
    row['datetime'] = '2024-12-14 14:30:00'
    row['goals'] = {'h': '1', 'a': '1'}
    with pytest.raises(ValueError, match='score'):
        join_understat([row], [record], '2024/25', competition='D1')


def test_registered_special_still_rejects_duplicates_and_invalid_xg():
    record = league_fd_record('D1', 'Union Berlin', 'Bochum', '14/12/2024', (0, 2))
    row = league_understat_row('Union Berlin', 'Bochum')
    row['datetime'] = '2024-12-14 14:30:00'
    duplicate = deepcopy(row)
    duplicate['id'] = '12346'
    with pytest.raises(ValueError, match='Duplicate known'):
        join_understat([row, duplicate], [record], '2024/25', competition='D1')
    row['xG']['h'] = True
    with pytest.raises(ValueError, match='Boolean'):
        join_understat([row], [record], '2024/25', competition='D1')
