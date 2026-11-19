"""Check shot-strength chronology and independently recoverable goal effects."""

from copy import deepcopy
from datetime import date, timedelta
import math
from types import SimpleNamespace

import numpy as np
import pytest

from football_model.shot_increment import ShotObservation
from football_model.shot_strength import DateTeamShotModel, ShotRateGoalIncrement
import football_model.shot_strength as shot_strength
from football_model.team_model import team_features


def observations():
    start = date(2024, 1, 1)
    fixtures = [('A', 'B'), ('B', 'C'), ('C', 'A'),
                ('B', 'A'), ('C', 'B'), ('A', 'C')] * 4
    home_counts, away_counts = {'A': 8, 'B': 3, 'C': 5}, {'A': 6, 'B': 2, 'C': 4}
    return [ShotObservation(f's-{i}', start + timedelta(days=i), home, away,
                            (home_counts[home], away_counts[away]))
            for i, (home, away) in enumerate(fixtures)]


def rows(n=30):
    start = date(2024, 1, 1)
    return [{'match_id': f'r-{i}', 'date': (start + timedelta(days=i)).isoformat(),
             'fit_before_date': '2023-12-31',
             'rates': [1.1 + (i % 2) * .2, 1.5],
             'shot_rates': [2. + (i % 5), 3. + ((i * 3) % 7)],
             'actual_score': [i % 4, (i + 2) % 3]}
            for i in range(n)]


def test_shot_team_fit_ignores_cutoff_day_and_future_observations():
    cutoff = date(2024, 1, 21)
    early = DateTeamShotModel().fit(observations()[:20], cutoff)
    changed = observations()
    for obs in changed[20:]:
        obs.shots_on_target = (999, 888)
    changed.append(ShotObservation('future-new-team', date(2024, 2, 1), 'X', 'Y', (500, 600)))
    actual = DateTeamShotModel().fit(changed, cutoff)
    assert actual.parameters() == early.parameters()
    assert actual.parameters()['n_matches'] == 20
    assert actual.parameters()['training_max_date'] == '2024-01-20'
    assert actual.predict('A', 'B', 'E0', cutoff) == early.predict('A', 'B', 'E0', cutoff)
    with pytest.raises(ValueError):
        actual.predict('X', 'A', 'E0', cutoff)


def test_missing_shots_are_excluded_and_do_not_establish_a_zero_strength_team():
    cutoff = date(2024, 2, 1)
    source = observations()
    source.append(ShotObservation('missing', date(2024, 1, 29), 'X', 'A', None))
    model = DateTeamShotModel().fit(source, cutoff)
    explicit_complete = DateTeamShotModel().fit(source[:-1], cutoff)
    assert model.predict('A', 'B', 'E0', cutoff) == explicit_complete.predict('A', 'B', 'E0', cutoff)
    assert model.parameters()['n_matches'] == 24
    assert model.parameters()['missing_shot_match_ids'] == ['missing']
    assert 'missing' not in model.parameters()['reference_match_ids']
    with pytest.raises(ValueError):
        model.predict('X', 'A', 'E0', cutoff)
    with pytest.raises(ValueError):
        DateTeamShotModel().fit([source[-1]], cutoff)


def test_saved_shot_coefficients_independently_replay_attack_defence_and_home():
    cutoff = date(2024, 2, 1)
    model = DateTeamShotModel().fit(observations(), cutoff)
    parameters = model.parameters()
    assert parameters['model_type'] == 'date_level_regularized_poisson_shots_on_target'
    assert parameters['competition'] == 'E0'
    assert parameters['regularization'] == .01
    assert parameters['half_life_days'] == 180
    for home, away in [('A', 'B'), ('B', 'A'), ('C', 'A')]:
        independent = [math.exp(parameters['intercept'] + math.fsum(
            parameters['coefficients'].get(key, 0.) * value for key, value in vector.items()))
            for vector in team_features(home, away, 'E0')]
        actual = model.predict(home, away, 'E0', cutoff)
        assert actual == pytest.approx(independent, abs=1e-12)
        assert all(math.isfinite(x) and x > 0 for x in actual)


@pytest.mark.parametrize('home,away,competition,played', [
    ('A', 'B', 'E1', date(2024, 2, 1)),
    ('A', 'A', 'E0', date(2024, 2, 1)),
    ('A', 'B', 'E0', date(2024, 1, 31)),
])
def test_shot_team_prediction_does_not_silently_cross_leagues_or_dates(home, away, competition, played):
    model = DateTeamShotModel().fit(observations(), date(2024, 2, 1))
    with pytest.raises(ValueError):
        model.predict(home, away, competition, played)


@pytest.mark.parametrize('bad', [(True, 2), (-1, 2), (1.5, 2), (2,), (2, 3, 4)])
def test_bad_shot_counts_are_not_relabelled_as_valid_goal_training_targets(bad):
    source = observations()
    source[0].shots_on_target = bad
    with pytest.raises(ValueError):
        DateTeamShotModel().fit(source, date(2024, 2, 1))


def test_shot_team_duplicates_are_not_counted_as_extra_evidence():
    source = observations()
    with pytest.raises(ValueError):
        DateTeamShotModel().fit(source + [deepcopy(source[0])], date(2024, 2, 1))


def test_increment_scales_log_shot_rates_using_only_earlier_weighted_rows():
    data = rows()
    cutoff = date(2024, 1, 21)
    early = ShotRateGoalIncrement().fit(data[:20], cutoff)
    changed = deepcopy(data)
    for row in changed[20:]:
        row['shot_rates'] = [1000., 900.]
        row['actual_score'] = [30, 40]
    actual = ShotRateGoalIncrement().fit(changed, cutoff)
    assert actual.parameters() == early.parameters()
    p = actual.parameters()
    weights = np.repeat([math.exp(-math.log(2) *
        (cutoff - date.fromisoformat(row['date'])).days / 180) for row in data[:20]], 2)
    weights /= weights.sum()
    raw = np.log(np.array([row['shot_rates'] for row in data[:20]], dtype=float).reshape(-1))
    mean = float(np.sum(weights * raw))
    scale = float(np.sqrt(np.sum(weights * (raw - mean) ** 2)))
    assert p['feature_mean'] == pytest.approx(mean, abs=1e-12)
    assert p['feature_scale'] == pytest.approx(scale, abs=1e-12)
    assert p['n_matches'] == 20
    assert p['training_max_date'] == '2024-01-20'


def test_constant_shot_strength_learns_zero_increment_instead_of_a_new_intercept():
    data = rows()
    for row in data:
        row['shot_rates'] = [4., 4.]
    model = ShotRateGoalIncrement().fit(data, date(2024, 2, 1))
    parameters = model.parameters()
    assert parameters['coefficient'] == pytest.approx(0., abs=0)
    assert parameters['feature_mean'] == pytest.approx(math.log(4), abs=1e-12)
    assert parameters['feature_scale'] == 1.
    assert model.predict([1.3, 2.1], [2., 20.]) == pytest.approx([1.3, 2.1], abs=0)


def test_single_known_goal_effect_is_recovered_and_is_symmetric_across_scoring_sides():
    # log shot-rate features are exactly -1/+1, so their shared mean is 0 and SD 1.
    # With offset 6, beta=log(2) gives exactly the independently specified means 3/12.
    data = [{'match_id': 'known', 'date': '2024-01-01', 'fit_before_date': '2023-12-31',
             'rates': [6., 6.], 'shot_rates': [math.exp(-1), math.exp(1)],
             'actual_score': [3, 12]}]
    later = deepcopy(data[0])
    later.update(match_id='later', date='2024-01-02', actual_score=[100, 200])
    model = ShotRateGoalIncrement(1e-8, 180).fit(data + [later], date(2024, 1, 2))
    p = model.parameters()
    assert p['reference_match_ids'] == ['known']
    assert p['coefficient'] == pytest.approx(math.log(2), abs=1e-7)
    assert model.predict(data[0]['rates'], data[0]['shot_rates']) == pytest.approx([3., 12.], rel=1e-7)
    normal = model.predict([1.2, 1.4], [2., 6.])
    swapped = model.predict([1.4, 1.2], [6., 2.])
    assert swapped == pytest.approx(normal[::-1], abs=0)
    assert model.predict([2., 2.], [8., 8.])[0] > model.predict([2., 2.], [2., 2.])[0]


@pytest.mark.parametrize('field,bad', [
    ('rates', [0., 1.]), ('rates', [1., float('nan')]), ('rates', [[1., 2.]]),
    ('rates', [True, 2.]),
    ('shot_rates', [0., 2.]), ('shot_rates', [2., float('inf')]),
    ('shot_rates', [True, False]), ('shot_rates', [True, 3.]), ('shot_rates', [[2., 3.]]),
    ('actual_score', [True, False]), ('actual_score', [True, 2]), ('actual_score', [1.5, 0.]),
    ('actual_score', [-1, 0]), ('actual_score', [[1, 2]]),
])
def test_increment_validates_each_match_before_combining_the_two_sides(field, bad):
    data = rows()
    data[0][field] = bad
    with pytest.raises(ValueError):
        ShotRateGoalIncrement().fit(data, date(2024, 2, 1))


def test_complementary_bad_match_lengths_cannot_realign_goals_with_another_fixture():
    data = rows()
    data[0]['actual_score'] = [1]
    data[1]['actual_score'] = [0, 1, 2]
    with pytest.raises(ValueError):
        ShotRateGoalIncrement().fit(data, date(2024, 2, 1))


def test_increment_rejects_offsets_from_after_the_original_match_and_duplicate_rows():
    data = rows()
    data[0]['fit_before_date'] = '2024-01-02'
    with pytest.raises(ValueError):
        ShotRateGoalIncrement().fit(data, date(2024, 2, 1))
    data = rows()
    with pytest.raises(ValueError):
        ShotRateGoalIncrement().fit(data + [deepcopy(data[0])], date(2024, 2, 1))


def test_failed_refit_cannot_leave_old_beta_with_a_new_feature_scaler(monkeypatch):
    model = ShotRateGoalIncrement().fit(rows(), date(2024, 2, 1))
    old_parameters = deepcopy(model.parameters())
    old_prediction = model.predict([1.3, 1.7], [2., 7.])
    changed = rows()
    for row in changed:
        row['shot_rates'] = [value * 10 for value in row['shot_rates']]
    monkeypatch.setattr(shot_strength, 'minimize', lambda *args, **kwargs: SimpleNamespace(
        success=False, x=np.array([0.]), message='injected numerical failure', nit=0))
    with pytest.raises(ValueError):
        model.fit(changed, date(2024, 2, 1))
    assert model.parameters() == old_parameters
    assert model.predict([1.3, 1.7], [2., 7.]) == old_prediction


@pytest.mark.parametrize('rates,shots', [([1.], [2., 3.]), ([1., 2.], [0., 3.]),
                                      ([1., 2.], [3., float('nan')]),
                                      ([True, 2.], [2., 3.]), ([1., 2.], [True, 3.])])
def test_prediction_cannot_silently_accept_missing_or_nonpositive_shot_strength(rates, shots):
    model = ShotRateGoalIncrement().fit(rows(), date(2024, 2, 1))
    with pytest.raises(ValueError):
        model.predict(rates, shots)
