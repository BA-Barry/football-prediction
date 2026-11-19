"""Date-safe reuse of GAS state information, not another learned predictor.

Tiny examples exercise the new bridge's source and perspective boundaries.
No fit, current results, parameter choice or evaluation season is involved.
"""
from copy import deepcopy

import pytest

from football_model.persistent_goal_difference import (
    PersistentGoalDifferenceFilter,
    forecast_targets,
)
from football_model.persistent_state_features import (
    STATE_FEATURE_NAMES,
    build_persistent_state_features,
)


PARAMETERS = {
    'attack_step': .06, 'defence_step': .07,
    'attack_retention_week': .8, 'defence_retention_week': .9,
}


def static_panel():
    return {
        'competitions': {'X': {'mu': .1, 'home': .2},
                         'Y': {'mu': -.1, 'home': .3}},
        'teams': [
            {'competition': league, 'team': team, 'attack': attack, 'defence': defence,
             'observations': 2, 'last_observed_date': '2023-06-01'}
            for league, team, attack, defence in (
                ('X', 'A', .15, -.08), ('X', 'B', -.05, .10),
                ('X', 'C', -.10, -.02),
                ('Y', 'A', .30, .10), ('Y', 'B', -.30, -.10),
            )
        ],
        'training_min_date': '2023-01-01', 'training_max_date': '2023-06-01',
        'training_matches': 5, 'training_ids': ['initial-1', 'initial-2'],
    }


def dynamics():
    return {'training_max_date': '2023-06-15', 'parameters': dict(PARAMETERS)}


def match(identity, day, home='A', away='B', hg=1, ag=0, league='X'):
    return {'match_id': identity, 'date': day, 'season': '2023/24',
            'competition': league, 'home': home, 'away': away, 'hg': hg, 'ag': ag}


def target(identity, cutoff='2023-07-01', day='2023-07-04', home='A', away='B', league='X'):
    # No target goals or original probabilities are needed by the state bridge.
    return {'match_id': identity, 'date': day, 'F': cutoff,
            'competition': league, 'home': home, 'away': away}


def history():
    return [match('prior', '2023-06-10', hg=3, ag=0),
            match('on-cutoff', '2023-07-01', home='A', away='C', hg=0, ag=4)]


def build(targets, rows=None, fitted=None):
    return build_persistent_state_features(static_panel(), fitted or dynamics(),
                                           history() if rows is None else rows, targets)


def reconstruct_log_rate(features):
    return (features['gas_league_base_log_rate'] + features['gas_home_log_rate_effect']
            + features['gas_own_attack_state'] - features['gas_opponent_defence_state'])


def test_parameters_must_precede_information_day_and_early_rows_are_retained():
    targets = [target('earlier', cutoff='2023-06-14', day='2023-06-20'),
               target('same-day', cutoff='2023-06-15', day='2023-06-20'),
               target('after', cutoff='2023-06-16', day='2023-06-20')]
    result = build(targets)
    assert [r['match_id'] for r in result['rows']] == ['earlier', 'same-day', 'after']
    assert result['metadata']['target_count'] == 3
    assert result['metadata']['counts'] == {'parameters_not_yet_fitted': 2, 'available': 1}
    for row in result['rows'][:2]:
        assert row['available'] is False
        assert row['unavailable_reason'] == 'parameters_not_yet_fitted'
        assert row['state_history_max_date'] is None
        assert row['explanation'] is None
        assert row['parameter_training_max_date'] == '2023-06-15'
        assert all(value is None for side in row['side_features'] for value in side.values())
    assert result['rows'][2]['available'] is True


def test_bridge_uses_strict_prior_dates_and_reproduces_previous_gas_rates():
    targets = [target('first'), target('second', cutoff='2023-07-02', day='2023-07-05')]
    rows = build(targets)['rows']
    assert rows[0]['state_history_max_date'] == '2023-06-10'
    assert rows[1]['state_history_max_date'] == '2023-07-01'
    for row in rows:
        assert row['state_history_max_date'] < row['F']
    original_targets = [{**row, 'hg': 9, 'ag': 0, 'original_hda': [.4, .3, .3]} for row in targets]
    previous, _ = forecast_targets(static_panel(), PARAMETERS, history(), original_targets)
    for row, old in zip(rows, previous, strict=True):
        assert row['explanation']['log_rates'] == pytest.approx(old['log_rates'], abs=1e-12)
        assert [reconstruct_log_rate(s) for s in row['side_features']] == pytest.approx(old['log_rates'], abs=1e-12)
    assert rows[0]['side_features'] != rows[1]['side_features']


def test_home_away_feature_perspectives_swap_teams_and_keep_actual_venue_effect():
    normal, reversed_fixture = build([target('normal'), target('reverse', home='B', away='A')])['rows']
    home, away = normal['side_features']
    reverse_home, reverse_away = reversed_fixture['side_features']
    assert home['gas_own_attack_state'] == reverse_away['gas_own_attack_state']
    assert home['gas_own_defence_state'] == reverse_away['gas_own_defence_state']
    assert away['gas_own_attack_state'] == reverse_home['gas_own_attack_state']
    assert away['gas_own_defence_state'] == reverse_home['gas_own_defence_state']
    assert home['gas_opponent_attack_state'] == away['gas_own_attack_state']
    assert home['gas_opponent_defence_state'] == away['gas_own_defence_state']
    assert away['gas_opponent_attack_state'] == home['gas_own_attack_state']
    assert away['gas_opponent_defence_state'] == home['gas_own_defence_state']
    assert [home['gas_home_log_rate_effect'], away['gas_home_log_rate_effect']] == [.2, 0.]
    assert [reverse_home['gas_home_log_rate_effect'], reverse_away['gas_home_log_rate_effect']] == [.2, 0.]
    assert all(tuple(side) == STATE_FEATURE_NAMES for side in (home,away,reverse_home,reverse_away))


def test_unknown_team_is_missing_not_zero_while_known_side_and_league_remain_available():
    row = build([target('new', home='New')], rows=[])['rows'][0]
    assert row['available'] is False
    assert row['unavailable_reason'] == 'no_observed_history'
    own, opponent = row['side_features']
    assert own['gas_own_attack_state'] is None
    assert own['gas_own_defence_state'] is None
    assert opponent['gas_opponent_attack_state'] is None
    assert opponent['gas_opponent_defence_state'] is None
    assert own['gas_opponent_attack_state'] == -.05
    assert opponent['gas_own_attack_state'] == -.05
    assert own['gas_league_base_log_rate'] == opponent['gas_league_base_log_rate'] == .1
    assert own['gas_home_log_rate_effect'] == .2
    assert opponent['gas_home_log_rate_effect'] == 0.
    assert row['explanation']['home']['state'] is None
    assert row['explanation']['home']['observations'] == 0
    assert row['explanation']['log_rates'] is None


def test_two_unknown_teams_keep_target_and_all_team_states_missing():
    result = build([target('both-new', home='New-A', away='New-B')], rows=[])
    assert len(result['rows']) == 1
    row = result['rows'][0]
    assert row['available'] is False
    assert row['explanation']['unknown_teams'] == ['New-A', 'New-B']
    for side in row['side_features']:
        assert all(side[k] is None for k in STATE_FEATURE_NAMES[:4])
        assert side['gas_league_base_log_rate'] == .1


def test_unknown_turns_known_only_after_first_actual_observation_not_on_its_day():
    observations = [match('new-debut', '2023-07-01', home='New', away='B', hg=2, ag=0)]
    rows = build([target('before', home='New'),
                  target('after', cutoff='2023-07-02', day='2023-07-05', home='New')], rows=observations)['rows']
    assert rows[0]['available'] is False
    assert rows[0]['side_features'][0]['gas_own_attack_state'] is None
    assert rows[1]['available'] is True
    assert rows[1]['side_features'][0]['gas_own_attack_state'] is not None
    assert rows[1]['explanation']['home']['observations'] == 1


def test_explain_state_is_pure_and_does_not_expose_neutral_unknown_coordinate():
    fitted = PersistentGoalDifferenceFilter(static_panel(), PARAMETERS)
    fitted.observe_day([history()[0]])
    before = deepcopy(fitted.state_dict())
    known = fitted.explain_state('X','A','B','2023-07-01')
    for cutoff in ('2023-06-20', '2023-08-01', '2023-07-01'):
        fitted.explain_state('X','A','B',cutoff)
        unknown = fitted.explain_state('X','New','B',cutoff)
        assert unknown['home']['state'] is None
        assert unknown['home']['anchor'] is None
        assert unknown['home']['deviation_at_forecast'] is None
    assert fitted.explain_state('X','A','B','2023-07-01') == known
    assert fitted.state_dict() == before
    assert known['log_rates'] == pytest.approx(fitted.forecast('X','A','B','2023-07-01')['log_rates'], abs=1e-12)
    # Returned explanation arrays/dicts must not let a consumer edit stored state.
    known['home']['state'][0] = 999
    known['home']['anchor'][0] = 999
    known['league']['mu'] = 999
    assert fitted.state_dict() == before


def test_future_results_and_target_goals_do_not_change_earlier_state_features():
    targets = [target('only')]
    observations = history() + [match('future', '2023-08-01', hg=8, ag=0)]
    reference = build(targets, rows=observations)
    altered = deepcopy(observations)
    for row in altered[1:]:
        row['hg'], row['ag'] = 0, 7
    changed_targets = [{**targets[0], 'hg': 0, 'ag': 99, 'original_hda': [.01,.01,.98]}]
    assert build(changed_targets, rows=altered) == reference


def test_target_order_is_preserved_and_forecast_date_alias_is_equivalent():
    later = target('later', cutoff='2023-07-02', day='2023-07-05')
    earlier = target('earlier')
    result = build([later, earlier])
    assert [r['match_id'] for r in result['rows']] == ['later','earlier']
    alias_targets = [{**later, 'forecast_before_date': later['F']},
                     {**earlier, 'forecast_before_date': earlier['F']}]
    for row in alias_targets:
        row.pop('F')
    assert build(alias_targets) == result
    assert build([{**later, 'forecast_before_date': later['F']}, earlier]) == result


def test_same_team_name_in_different_leagues_does_not_share_coordinates():
    rows = build([target('x', league='X'), target('y', league='Y')], rows=[])['rows']
    assert rows[0]['side_features'][0]['gas_own_attack_state'] == .15
    assert rows[1]['side_features'][0]['gas_own_attack_state'] == .30
    assert rows[0]['side_features'][0]['gas_league_base_log_rate'] == .1
    assert rows[1]['side_features'][0]['gas_league_base_log_rate'] == -.1


def test_arguments_are_not_mutated_and_metadata_does_not_expand_numeric_feature_schema():
    static, dynamic, observations, targets = static_panel(), dynamics(), history(), [target('one')]
    before = deepcopy((static,dynamic,observations,targets))
    output = build_persistent_state_features(static,dynamic,observations,targets)
    assert (static,dynamic,observations,targets) == before
    assert output['metadata']['feature_names'] == list(STATE_FEATURE_NAMES)
    for side in output['rows'][0]['side_features']:
        assert set(side) == set(STATE_FEATURE_NAMES)
        assert not {'available','observations','parameter_training_max_date','state_history_max_date'} & set(side)
    assert output['rows'][0]['explanation']['home']['observations'] == 3
    assert output['rows'][0]['explanation']['away']['observations'] == 3


def test_conflicting_cutoffs_and_self_day_targets_are_rejected():
    with pytest.raises(ValueError, match='Conflicting'):
        build([{**target('bad'), 'forecast_before_date': '2023-07-02'}])
    with pytest.raises(ValueError, match='precede'):
        build([target('same-day', cutoff='2023-07-04', day='2023-07-04')])
    with pytest.raises(ValueError, match='Duplicate'):
        build([target('duplicate'), target('duplicate')])


def test_pre_static_or_empty_feature_cohorts_do_not_backfill_states():
    early = target('pre-static', cutoff='2023-05-01', day='2023-05-04')
    result = build([early], rows=[])
    assert result['rows'][0]['unavailable_reason'] == 'parameters_not_yet_fitted'
    assert all(v is None for side in result['rows'][0]['side_features'] for v in side.values())
    empty = build([], rows=[])
    assert empty['rows'] == []
    assert empty['metadata']['target_count'] == 0
    assert empty['metadata']['counts'] == {}
