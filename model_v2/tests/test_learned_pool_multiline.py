"""A learned weight must survive recording/scoring with its temporal provenance."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math

import pytest

from football_model.domain import Fixture
from football_model.learned_pool import LearnedLinearPool
from football_model.multiline import learned_arithmetic_pool, multi_record, score_multi


NOW = datetime(2026, 10, 28, 5, 0, tzinfo=timezone.utc)
FUTURE = Fixture('future', 'A', 'B', 'I1', NOW+timedelta(days=1))


def learner():
    rows = []
    for day in (2, 3):
        kickoff = NOW-timedelta(days=day)
        rows.append({'match_id': f'old-{day}', 'kickoff': kickoff.isoformat(),
            'statistical_generated_at': (kickoff-timedelta(hours=3)).isoformat(),
            'llm_generated_at': (kickoff-timedelta(hours=2)).isoformat(),
            'statistical_probabilities': [.7, .2, .1],
            'llm_probabilities': [.4, .3, .3],
            'result_observed_at': (kickoff+timedelta(hours=3)).isoformat(),
            'actual_outcome': 0, 'llm_track': 'llm_independent'})
    return LearnedLinearPool().fit(rows, NOW)


def tracks():
    shared = {'fixture_id': FUTURE.match_id, 'generated_at': NOW.isoformat()}
    statistical = {**shared, 'id': 'stat', 'model': 'synthetic-stat',
                   'probabilities': [.5, .3, .2]}
    llm = {**shared, 'id': 'llm', 'model': 'synthetic-llm',
           'probabilities': [.4, .3, .3],
           'exposed_to_statistical': False, 'exposed_to_market': False}
    pool, reason = learned_arithmetic_pool(statistical, llm, learner(), FUTURE, NOW)
    assert reason is None
    return {'statistical': statistical, 'llm_independent': llm, 'joint_pool': pool}


def test_fitted_pool_record_and_scorer_preserve_weights_and_training_ids():
    record = multi_record(FUTURE, NOW, tracks=tracks())
    pool = record['tracks']['joint_pool']
    assert pool['weight_source'] == 'earlier_prospective_log_loss_fit'
    assert pool['weights'] == {'statistical': 1., 'llm': 0.}
    assert pool['probabilities'] == [.5, .3, .2]
    assert pool['learned_parameters']['n_matches'] == 2
    assert score_multi([record], {})['coverage']['joint_pool']['available'] == 1


def test_fitted_pool_settled_score_uses_its_actual_three_probabilities():
    record = multi_record(FUTURE, NOW, tracks=tracks())
    result = {'fixture_id': FUTURE.match_id, 'home_goals': 1, 'away_goals': 0,
              'finished_by': (FUTURE.kickoff+timedelta(hours=2)).isoformat(),
              'available_at': (FUTURE.kickoff+timedelta(hours=3)).isoformat(),
              'settlement': 'regulation_including_stoppage'}
    scores = score_multi([record], {FUTURE.match_id: result})
    joint = scores['per_fixture'][0]['scores']['joint_pool']
    assert joint['log_loss'] == pytest.approx(-math.log(.5))
    assert joint['brier'] == pytest.approx(.5**2+.3**2+.2**2)


@pytest.mark.parametrize('tamper', ['fit_after_generation', 'late_result', 'target_in_training',
                                  'duplicate_ids', 'count', 'weight', 'not_identified', 'missing',
                                  'wrong_llm_line', 'late_training_kickoff'])
def test_corrupt_fit_metadata_cannot_be_archived_as_a_learned_pool(tamper):
    supplied = tracks()
    params = supplied['joint_pool']['learned_parameters']
    if tamper == 'fit_after_generation':
        params['fit_before'] = (NOW+timedelta(seconds=1)).isoformat()
    elif tamper == 'late_result':
        params['training_results_max_observed_at'] = NOW.isoformat()
    elif tamper == 'target_in_training':
        params['reference_match_ids'][0] = FUTURE.match_id
    elif tamper == 'duplicate_ids':
        params['reference_match_ids'][0] = params['reference_match_ids'][1]
    elif tamper == 'count':
        params['n_matches'] += 1
    elif tamper == 'weight':
        params['weights'] = {'statistical': .5, 'llm': .5}
    elif tamper == 'not_identified':
        params['identified'] = False
    elif tamper == 'wrong_llm_line':
        params['training_llm_track'] = 'llm_model_informed'
    elif tamper == 'late_training_kickoff':
        params['training_kickoff_max'] = params['training_results_max_observed_at']
    else:
        del supplied['joint_pool']['learned_parameters']
    with pytest.raises(ValueError):
        multi_record(FUTURE, NOW, tracks=supplied)


def test_archived_fit_metadata_is_a_copy():
    supplied = tracks()
    record = multi_record(FUTURE, NOW, tracks=supplied)
    expected = deepcopy(record['tracks']['joint_pool']['learned_parameters'])
    supplied['joint_pool']['learned_parameters']['reference_match_ids'][0] = 'changed'
    assert record['tracks']['joint_pool']['learned_parameters'] == expected
