"""Tiny chronological checks for the persistent conditional-state candidate.

These fixtures exercise information boundaries and state evolution; they do not
select hyperparameters or claim out-of-sample forecasting improvements.
"""

from copy import deepcopy
import json

import numpy as np
import pytest

from football_model.persistent_goal_difference import (
    PersistentGoalDifferenceFilter,
    forecast_targets,
)


PARAMETERS = {
    'attack_step': .06,
    'defence_step': .07,
    'attack_retention_week': .8,
    'defence_retention_week': .9,
}


def static_panel():
    teams = [
        ('X', 'A', .15, -.08), ('X', 'B', -.05, .10),
        ('X', 'C', -.05, -.01), ('X', 'D', -.05, -.01),
        ('Y', 'A', .30, .10), ('Y', 'B', -.30, -.10),
    ]
    return {
        'competitions': {'X': {'mu': 0., 'home': .2},
                         'Y': {'mu': .1, 'home': -.1}},
        'teams': [
            {'competition': competition, 'team': team, 'attack': attack,
             'defence': defence, 'observations': 1,
             'last_observed_date': '2023-06-01'}
            for competition, team, attack, defence in teams
        ],
        'training_min_date': '2023-06-01',
        'training_max_date': '2023-06-01',
        'training_matches': 3,
        'training_ids': ['static-x-ab', 'static-x-cd', 'static-y-ab'],
    }


def match(identity, day, home='A', away='B', hg=1, ag=0,
          competition='X', season='2023/24'):
    return {
        'match_id': identity, 'date': day, 'competition': competition,
        'season': season, 'home': home, 'away': away, 'hg': hg, 'ag': ag,
    }


def target(identity, day, forecast_day, home='A', away='B', **kwargs):
    return {
        **match(identity, day, home, away, **kwargs),
        'F': forecast_day, 'original_hda': [.46, .29, .25],
    }


def model(parameters=None):
    return PersistentGoalDifferenceFilter(static_panel(), parameters or PARAMETERS)


def test_same_day_order_is_invariant_and_all_likelihoods_use_day_start_state():
    # A shared team makes an accidental within-day update observable. The filter
    # accepts a panel of observations, so this also covers repeated identities
    # within one day without relying on a particular competition's scheduling.
    rows = [match('day-b', '2023-07-01', 'A', 'B', 3, 0),
            match('day-c', '2023-07-01', 'A', 'C', 0, 2)]
    forward, reverse = model(), model()
    batch = forward.observe_day(rows)
    assert reverse.observe_day(list(reversed(rows))) == batch
    assert reverse.state_dict() == forward.state_dict()

    separate = [model().observe_day([row])['log_likelihoods'][0] for row in rows]
    assert batch['log_likelihoods'] == pytest.approx(separate, abs=1e-12)
    assert forward.forecast('X', 'A', 'B', '2023-07-02') == reverse.forecast(
        'X', 'A', 'B', '2023-07-02')
    assert forward.state_dict()['observed_matches'] == 5


def test_forecasts_do_not_change_state_membership_or_apply_query_count_decay():
    fitted = model()
    fitted.observe_day([match('played', '2023-07-01', hg=3, ag=0)])
    before = deepcopy(fitted.state_dict())
    expected = fitted.forecast('X', 'A', 'B', '2023-07-15')
    for day in ['2023-07-02', '2023-08-01', '2023-07-15']:
        fitted.forecast('X', 'A', 'B', day)
        unknown = fitted.forecast('X', 'New', 'B', day)
        assert unknown['unknown_teams'] == ['New']
    assert fitted.forecast('X', 'A', 'B', '2023-07-15') == expected
    assert fitted.state_dict() == before


def test_calendar_decay_depends_on_elapsed_days_and_keeps_static_anchor():
    fitted = model()
    fitted.observe_day([match('played', '2023-07-01', hg=3, ag=0)])
    saved = fitted.state_dict()
    teams = {row['team']: row for row in saved['teams'] if row['competition'] == 'X'}
    week = np.array([PARAMETERS['attack_retention_week'],
                     PARAMETERS['defence_retention_week']])

    def expected_rates(weeks):
        home = np.array(teams['A']['anchor']) + week**weeks * teams['A']['deviation']
        away = np.array(teams['B']['anchor']) + week**weeks * teams['B']['deviation']
        return [.2 + home[0] - away[1], away[0] - home[1]]

    for day, weeks in [('2023-07-08', 1), ('2023-07-15', 2)]:
        forecast = fitted.forecast('X', 'A', 'B', day)
        assert forecast['log_rates'] == pytest.approx(expected_rates(weeks), abs=1e-12)
    assert fitted.state_dict() == saved


def test_season_labels_do_not_reset_the_persistent_team_state():
    carried, relabelled = model(), model()
    before_change = match('old-season', '2023-07-01', hg=3, ag=0, season='2022/23')
    after_change = match('new-season', '2023-08-01', hg=0, ag=2, season='2023/24')
    carried.observe_day([before_change])
    carried.observe_day([after_change])
    relabelled.observe_day([{**before_change, 'season': 'same-season'}])
    relabelled.observe_day([{**after_change, 'season': 'same-season'}])
    assert carried.state_dict() == relabelled.state_dict()
    assert carried.forecast('X', 'A', 'B', '2023-08-02') == relabelled.forecast(
        'X', 'A', 'B', '2023-08-02')
    assert next(row for row in carried.state_dict()['teams']
                if row['competition'] == 'X' and row['team'] == 'A')['observations'] == 3


def test_same_team_name_in_another_competition_has_separate_state():
    shared_clock, untouched = model(), model()
    first = match('y-first', '2023-07-01', hg=0, ag=3, competition='Y')
    for fitted in (shared_clock, untouched):
        fitted.observe_day([first])
    shared_clock.observe_day([match('x-only', '2023-07-08', hg=4, ag=0)])
    expected = untouched.forecast('Y', 'A', 'B', '2023-07-09')
    actual = shared_clock.forecast('Y', 'A', 'B', '2023-07-09')
    assert actual['gas_probabilities'] == pytest.approx(expected['gas_probabilities'], abs=1e-12)
    assert actual['log_rates'] == pytest.approx(expected['log_rates'], abs=1e-12)
    before_x = untouched.forecast('X', 'A', 'B', '2023-07-09')['gas_probabilities']
    assert shared_clock.forecast('X', 'A', 'B', '2023-07-09')['gas_probabilities'] != before_x


@pytest.mark.parametrize('home,away,unknown', [
    ('New', 'B', ['New']), ('A', 'New', ['New']),
    ('New', 'Other', ['New', 'Other']),
])
def test_unknown_teams_are_explicitly_unavailable_and_are_never_registered_by_query(home, away, unknown):
    fitted = model()
    before = fitted.state_dict()
    prediction = fitted.forecast('X', home, away, '2023-07-01')
    assert prediction['gas_available'] is False
    assert prediction['unknown_teams'] == unknown
    assert prediction['gas_probabilities'] is None
    assert prediction['log_rates'] is None
    assert prediction['net_goal_distribution'] is None
    assert prediction['unavailable_reason'] == 'no_observed_history'
    assert fitted.state_dict() == before


def test_unknown_team_becomes_observed_only_after_a_completed_observation_day():
    fitted = model()
    assert fitted.forecast('X', 'New', 'B', '2023-07-01')['gas_available'] is False
    fitted.observe_day([match('new-first', '2023-07-01', 'New', 'B', 0, 1)])
    prediction = fitted.forecast('X', 'New', 'B', '2023-07-02')
    assert prediction['gas_available'] is True
    assert prediction['unknown_teams'] == []
    assert sum(prediction['gas_probabilities']) == pytest.approx(1., abs=1e-12)
    row = next(row for row in fitted.state_dict()['teams'] if row['team'] == 'New')
    assert row['observations'] == 1
    assert row['last_observed_date'] == '2023-07-01'
    # This is the explicitly documented algorithmic anchor after first exposure,
    # not a measured pre-debut ability or an estimated uncertainty distribution.
    assert row['anchor'] == [0., 0.]


def test_same_day_results_and_future_results_do_not_enter_an_earlier_target():
    history = [match('earlier', '2023-07-01', hg=3, ag=0),
               match('same-f', '2023-07-04', 'A', 'D', 0, 4),
               match('later', '2023-07-10', 'A', 'D', 5, 0)]
    early = target('early-target', '2023-07-10', '2023-07-04', 'A', 'D')
    late = target('late-target', '2023-07-20', '2023-07-11', 'A', 'D')
    predictions, state = forecast_targets(static_panel(), PARAMETERS, history, [early, late])
    manual = model()
    manual.observe_day([history[0]])
    assert predictions[0]['gas_probabilities'] == manual.forecast(
        'X', 'A', 'D', early['F'])['gas_probabilities']
    assert predictions[0]['observed_matches'] == 4
    assert predictions[0]['last_observed_date'] == '2023-07-01'

    changed = deepcopy(history)
    changed[1].update(hg=6, ag=0)
    changed[2].update(hg=0, ag=5)
    revised, _ = forecast_targets(static_panel(), PARAMETERS, changed, [early, late])
    assert revised[0]['gas_probabilities'] == predictions[0]['gas_probabilities']
    assert revised[1]['gas_probabilities'] != predictions[1]['gas_probabilities']
    assert state['date'] == '2023-07-10'
    assert state['observed_matches'] == 6


def test_target_truth_is_unused_and_future_file_tail_does_not_advance_saved_state():
    history = [match('played', '2023-07-01', hg=2, ag=0),
               match('file-tail', '2023-09-01', hg=0, ag=5)]
    requested = target('to-predict', '2023-07-10', '2023-07-04')
    clean, state = forecast_targets(static_panel(), PARAMETERS, history, [requested])
    poisoned = {**requested, 'hg': float('nan'), 'ag': object(), 'actual_outcome': 'unused'}
    output, other_state = forecast_targets(static_panel(), PARAMETERS, history, [poisoned])
    assert output[0]['gas_probabilities'] == clean[0]['gas_probabilities']
    assert output[0]['deployment_probabilities'] == clean[0]['deployment_probabilities']
    assert other_state == state
    assert state['date'] == '2023-07-01'
    assert state['observed_matches'] == 4


def test_targets_keep_requested_order_and_exact_original_fallback_without_dropping_rows():
    history = [match('first', '2023-07-01', hg=2, ag=0),
               match('second', '2023-07-08', 'C', 'D', 1, 1)]
    requested = [target('late-unknown', '2023-07-20', '2023-07-11', 'New', 'B'),
                 target('early-known', '2023-07-10', '2023-07-04'),
                 target('early-unknown', '2023-07-10', '2023-07-04', 'New', 'Other')]
    requested[0]['original_hda'] = [.123456789, .234567891, .64197532]
    before = deepcopy(requested)
    output, _ = forecast_targets(static_panel(), PARAMETERS, list(reversed(history)), requested)
    assert len(output) == len(requested)
    assert [row['match_id'] for row in output] == [row['match_id'] for row in requested]
    assert requested == before
    for source, row in zip(requested, output):
        if row['gas_available']:
            assert row['deployment_probabilities'] == row['gas_probabilities']
        else:
            assert row['deployment_probabilities'] == source['original_hda']
            assert row['gas_probabilities'] is None
        assert row['deployment_probabilities'] is not source['original_hda']


def test_static_only_targets_are_supported_including_unknown_fallback():
    requested = [target('known', '2023-07-10', '2023-07-04'),
                 target('unknown', '2023-07-10', '2023-07-04', 'New', 'B')]
    output, state = forecast_targets(static_panel(), PARAMETERS, [], requested)
    expected = model().forecast('X', 'A', 'B', '2023-07-04')
    assert output[0]['gas_probabilities'] == expected['gas_probabilities']
    assert output[1]['gas_probabilities'] is None
    assert output[1]['deployment_probabilities'] == requested[1]['original_hda']
    assert state == model().state_dict()


def test_json_saved_state_restores_identical_forecast_and_future_updates():
    original = model()
    original.observe_day([match('first', '2023-07-01', hg=3, ag=0)])
    saved = json.loads(json.dumps(original.state_dict(), allow_nan=False))
    restored = PersistentGoalDifferenceFilter.from_state_dict(saved)
    assert restored.state_dict() == original.state_dict()
    assert restored.forecast('X', 'A', 'B', '2023-07-08') == original.forecast(
        'X', 'A', 'B', '2023-07-08')
    next_day = [match('second', '2023-07-08', hg=0, ag=2)]
    assert restored.observe_day(next_day) == original.observe_day(next_day)
    assert restored.state_dict() == original.state_dict()


def test_same_day_or_backwards_forecasts_and_observations_are_rejected_without_state_changes():
    fitted = model()
    fitted.observe_day([match('played', '2023-07-01')])
    before = fitted.state_dict()
    for day in ['2023-06-30', '2023-07-01']:
        with pytest.raises(ValueError, match='strictly before'):
            fitted.forecast('X', 'A', 'B', day)
        with pytest.raises(ValueError, match='increase strictly'):
            fitted.observe_day([match('invalid', day)])
        assert fitted.state_dict() == before
    with pytest.raises(ValueError, match='single day'):
        fitted.observe_day([match('two-a', '2023-07-02'), match('two-b', '2023-07-03')])
    assert fitted.state_dict() == before
