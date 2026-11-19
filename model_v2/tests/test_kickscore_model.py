from copy import deepcopy
from datetime import date, timedelta
import json
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from football_model.kickscore_model import (
    KickScoreOutcomeModel, _ordered_probit_probabilities,
)


START = date(2022, 1, 1)
CUTOFF = date(2022, 2, 1)


def history(n=24):
    teams = ('A', 'B', 'C')
    return [{'match_id': f'h-{i:02}', 'date': START+timedelta(days=i),
             'competition': 'E0', 'home': teams[i % 3], 'away': teams[(i+1) % 3],
             'actual_outcome': i % 3} for i in range(n)]


def fixture(home='A', away='B', played=None, competition='E0', identity='f-1'):
    return {'match_id': identity, 'date': played or CUTOFF+timedelta(days=2),
            'competition': competition, 'home': home, 'away': away}


@pytest.mark.parametrize('kind', ['dynamic', 'static'])
def test_real_fit_returns_ordered_probabilities_and_inspectable_upstream_inference(kind):
    model = KickScoreOutcomeModel(kind).fit(history(), CUTOFF)
    assert model.converged is True
    metadata = model.parameters()
    assert metadata['n_matches'] == len(model.upstream_model.observations) == 24
    assert metadata['upstream_package'] == 'kickscore'
    assert metadata['upstream_version'] == '0.2.0'
    assert math.isfinite(metadata['log_likelihood'])
    assert json.loads(json.dumps(metadata, allow_nan=False)) == metadata
    metadata['hyperparameters']['cvar'] = -1
    prediction = model.predict_many([fixture()], CUTOFF)[0]
    probabilities = prediction['probabilities']
    assert all(0 < p < 1 for p in probabilities)
    assert sum(probabilities) == pytest.approx(1., abs=1e-12)
    assert probabilities == pytest.approx(prediction['raw_upstream_probabilities'], abs=1e-12)
    assert prediction['variance_difference'] > 0
    assert prediction['unknown_teams'] == []
    assert model.parameters()['hyperparameters']['cvar'] == 1.


class FutureDateOnly(dict):
    def __getitem__(self, key):
        if key != 'date':
            raise AssertionError(f'An excluded future field was inspected: {key}')
        return super().__getitem__(key)


def test_prefix_selection_does_not_read_any_same_day_or_future_identity_or_outcome():
    data = history()
    cutoff = START+timedelta(days=16)
    earlier = [row for row in data if row['date'] < cutoff]
    poison = [row if row['date'] < cutoff else FutureDateOnly(date=row['date']) for row in data]
    clean = KickScoreOutcomeModel().fit(earlier, cutoff)
    contaminated = KickScoreOutcomeModel().fit(list(reversed(poison)), cutoff)
    assert clean.parameters() == contaminated.parameters()
    assert clean.parameters()['training_max_date'] == (cutoff-timedelta(days=1)).isoformat()
    assert clean.parameters()['n_matches'] == 16
    assert clean.predict_many([fixture()], cutoff) == contaminated.predict_many([fixture()], cutoff)


def test_prediction_ignores_supplied_outcomes_and_never_updates_its_fitted_history():
    model = KickScoreOutcomeModel().fit(history(), CUTOFF)
    clean = model.predict_many([fixture()], CUTOFF)
    poisoned = fixture()
    poisoned.update(actual_outcome=object(), actual_home_goals=float('nan'), actual_away_goals='future')
    before = model.parameters()
    assert model.predict_many([poisoned], CUTOFF) == clean
    assert model.parameters() == before
    assert len(model.upstream_model.observations) == before['n_matches']


def test_home_advantage_stays_on_the_correct_side_of_away_winning_observations():
    data = history(16)
    all_home, all_away = deepcopy(data), deepcopy(data)
    for row in all_home:
        row['actual_outcome'] = 0
    for row in all_away:
        row['actual_outcome'] = 2
    home_model = KickScoreOutcomeModel('static').fit(all_home, CUTOFF)
    away_model = KickScoreOutcomeModel('static').fit(all_away, CUTOFF)
    home = home_model.predict_many([fixture()], CUTOFF)[0]
    away = away_model.predict_many([fixture()], CUTOFF)[0]
    assert home['mean_difference'] > 0 > away['mean_difference']
    assert home['probabilities'][0] > home['probabilities'][2]
    assert away['probabilities'][2] > away['probabilities'][0]
    assert home['probabilities'] == pytest.approx(list(reversed(away['probabilities'])), abs=1e-11)


def test_dynamic_state_reverts_and_variance_grows_during_absence_while_static_does_not():
    data = []
    for i in range(16):
        a_home = i % 2 == 0
        data.append({'match_id': str(i), 'date': START+timedelta(days=i), 'competition': 'E0',
            'home': 'A' if a_home else 'B', 'away': 'B' if a_home else 'A',
            'actual_outcome': 0 if a_home else 2})
    cutoff = START+timedelta(days=16)
    forecasts = [fixture(played=cutoff, identity='near'),
                 fixture(played=cutoff+timedelta(days=180), identity='far')]
    dynamic = KickScoreOutcomeModel('dynamic', {'cvar': .05, 'dvar': 1., 'lscale_days': 10.}).fit(data, cutoff)
    static = KickScoreOutcomeModel('static', {'cvar': .05}).fit(data, cutoff)
    near, far = dynamic.predict_many(forecasts, cutoff)
    assert near['mean_difference'] > far['mean_difference'] > 0
    assert far['variance_difference'] > near['variance_difference']
    static_near, static_far = static.predict_many(forecasts, cutoff)
    assert static_near['mean_difference'] == static_far['mean_difference']
    assert static_near['variance_difference'] == static_far['variance_difference']
    assert static_near['probabilities'] == static_far['probabilities']
    assert near['probabilities'] != near['mean_only_probabilities']


def test_unseen_teams_and_competitions_use_nondegenerate_prior_and_do_not_mutate_training():
    model = KickScoreOutcomeModel(parameters={'cvar': .8, 'dvar': .6, 'havar': 1.}).fit(history(), CUTOFF)
    before = model.parameters()
    unseen = fixture('A', 'B', competition='NEW-LEAGUE')
    p = model.predict_many([unseen], CUTOFF)[0]
    assert p['unknown_teams'] == ['A', 'B']
    assert p['mean_difference'] == 0
    assert p['variance_difference'] == pytest.approx(2*(.8+.6)+1.)
    assert p['probabilities'][0] == p['probabilities'][2]
    assert p['probabilities'][1] < p['mean_only_probabilities'][1]
    assert model.predict_many([unseen], CUTOFF)[0] == p
    assert model.parameters() == before
    assert len(model.upstream_model.observations) == 24


def test_nonconvergence_is_not_silently_refit_or_reported_as_success():
    model = KickScoreOutcomeModel(parameters={'max_iter': 1}).fit(history(), CUTOFF)
    assert model.converged is False
    assert model.parameters()['converged'] is False
    assert model.parameters()['fit_controls']['max_iter'] == 1
    with pytest.raises(ValueError, match='did not converge'):
        model.predict_many([fixture()], CUTOFF)


def test_stable_small_draw_probability_agrees_with_density_integral_and_team_swap():
    mean, variance, margin = 10., 0., .35
    home, draw, away = _ordered_probit_probabilities(mean, variance, margin)
    # Direct density integration remains well resolved where subtracting two
    # positive-tail CDFs rounds the draw probability to zero.
    integrated, _ = quad(norm.pdf, mean-margin, mean+margin, epsabs=1e-40, epsrel=1e-12)
    assert draw == pytest.approx(integrated, rel=1e-12, abs=0.)
    assert home > draw > away > 0
    assert [home, draw, away] == list(reversed(_ordered_probit_probabilities(-mean, variance, margin)))
    with pytest.raises(ValueError, match='zero or invalid'):
        _ordered_probit_probabilities(50., 0., margin)


@pytest.mark.parametrize('outcome', [True, np.bool_(False), -1, 3, .5, '1', None])
def test_invalid_earlier_outcomes_are_not_coerced(outcome):
    data = history()
    data[0]['actual_outcome'] = outcome
    with pytest.raises(ValueError, match='Outcome'):
        KickScoreOutcomeModel().fit(data, CUTOFF)


def test_time_validation_rejects_retroactive_publication_and_older_fixtures():
    model = KickScoreOutcomeModel().fit(history(), CUTOFF)
    with pytest.raises(ValueError, match='at or after the fit cutoff'):
        model.predict_many([fixture()], CUTOFF-timedelta(days=1))
    with pytest.raises(ValueError, match='earlier than its publication'):
        model.predict_many([fixture(played=CUTOFF-timedelta(days=1))], CUTOFF)
    with pytest.raises(ValueError, match='calendar-date fit cutoff'):
        KickScoreOutcomeModel().fit(history(), CUTOFF.isoformat())


def test_earlier_duplicate_matches_and_identity_collisions_are_rejected():
    data = history()
    with pytest.raises(ValueError, match='unique earlier'):
        KickScoreOutcomeModel().fit(data+[deepcopy(data[0])], CUTOFF)
    data[0]['away'] = data[0]['home']
    with pytest.raises(ValueError, match='different home and away'):
        KickScoreOutcomeModel().fit(data, CUTOFF)


def test_real_prefix_advance_matches_fresh_fit_ignores_future_and_rejects_changed_history():
    data = history(30)
    first_cutoff, later_cutoff = START+timedelta(days=16), START+timedelta(days=24)
    settings = {}
    model = KickScoreOutcomeModel(parameters=settings).fit(data, first_cutoff)
    original_upstream = model.upstream_model
    model.predict_many([fixture('New', 'Else', competition='UNSEEN')], first_cutoff)
    poisoned = [row if row['date'] < later_cutoff else FutureDateOnly(date=row['date']) for row in data]
    assert model.advance(poisoned, later_cutoff) is model
    assert model.upstream_model is not original_upstream
    fresh = KickScoreOutcomeModel(parameters=settings).fit(data, later_cutoff)
    assert model.converged and fresh.converged
    advanced_prediction = model.predict_many([fixture()], later_cutoff)[0]
    fresh_prediction = fresh.predict_many([fixture()], later_cutoff)[0]
    assert advanced_prediction['probabilities'] == pytest.approx(fresh_prediction['probabilities'], abs=1e-7)
    assert advanced_prediction['mean_difference'] == pytest.approx(fresh_prediction['mean_difference'], abs=1e-6)
    assert advanced_prediction['variance_difference'] == pytest.approx(fresh_prediction['variance_difference'], abs=1e-6)
    assert model.parameters()['n_upstream_fit_calls'] == 2
    assert model.parameters()['fit_mode'] == 'fresh_advance'
    assert model.parameters()['n_matches'] == len(model.upstream_model.observations) == 24
    assert model.parameters()['n_competitions'] == 1
    assert model.parameters()['n_teams'] == 3
    still_unknown = model.predict_many([fixture('New', 'Else', competition='UNSEEN')], later_cutoff)[0]
    assert still_unknown['unknown_teams'] == ['New', 'Else']
    unchanged = model.predict_many([fixture()], later_cutoff)
    model.advance(data, later_cutoff)
    assert model.parameters()['n_upstream_fit_calls'] == 2
    assert model.predict_many([fixture()], later_cutoff) == unchanged
    rewritten = deepcopy(data)
    rewritten[0]['actual_outcome'] = 2
    with pytest.raises(ValueError, match='change earlier history'):
        model.advance(rewritten, later_cutoff)
    with pytest.raises(ValueError, match='backwards'):
        model.advance(data, first_cutoff)
