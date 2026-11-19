"""Persistent calendar-time, score-driven Skellam football panel.

The first season supplies regularized static anchors. Four shared dynamic
parameters are estimated on the next season, then remain fixed. States persist
across seasons and forecast calls do not mutate them. This is a conditional
observation-driven filter, not Bayesian inference or a complete-score model.
"""

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date
import math

import numpy as np
from scipy.optimize import minimize

from .skellam_core import log_probability_and_lograte_score, net_goal_probabilities


PARAMETER_NAMES = ("attack_step", "defence_step", "attack_retention_week", "defence_retention_week")
DYNAMIC_START = (.04, .04, .97, .97)
DYNAMIC_BOUNDS = ((0., .2), (0., .2), (.5, 1.), (.5, 1.))
STATIC_RIDGE = 4.0


def _date(value):
    result = date.fromisoformat(value)
    if result.isoformat() != value:
        raise ValueError("Require ISO calendar dates")
    return result


def _rows(rows, *, allow_empty=False):
    result = sorted(list(rows), key=lambda r: (r['date'], r['match_id']))
    if (not result and not allow_empty) or len({r['match_id'] for r in result}) != len(result):
        raise ValueError("Need nonempty matches with unique identifiers")
    for r in result:
        _date(r['date'])
        if r['home'] == r['away'] or any(type(r[k]) is not int or r[k] < 0 for k in ('hg', 'ag')):
            raise ValueError("Need different teams and nonnegative integer goals")
    return result


def _parameters(parameters):
    p = {name: float(parameters[name]) for name in PARAMETER_NAMES}
    if any(not math.isfinite(p[n]) or not lo <= p[n] <= hi
           for n, (lo, hi) in zip(PARAMETER_NAMES, DYNAMIC_BOUNDS)):
        raise ValueError("Dynamic parameters outside their stated bounds")
    return p


def fit_static_panel(rows):
    """One penalized net-goal fit on the initial season, no tuning of ridge.

    Within each competition both attack and defence effects are centered.
    The uncentered coordinates receive ridge regularization as well, removing
    their redundant means. Only supplied initial-season identities are used.
    """
    rows = _rows(rows)
    if len({r['season'] for r in rows}) != 1:
        raise ValueError("Static initialization must use one explicit initial season")
    comps = sorted({r['competition'] for r in rows})
    keys = sorted({(r['competition'], r[side]) for r in rows for side in ('home', 'away')})
    ci = {c: i for i, c in enumerate(comps)}
    ti = {k: i for i, k in enumerate(keys)}
    group = np.array([ci[c] for c, _ in keys])
    cidx = np.array([ci[r['competition']] for r in rows])
    hidx = np.array([ti[(r['competition'], r['home'])] for r in rows])
    aidx = np.array([ti[(r['competition'], r['away'])] for r in rows])
    z = np.array([r['hg'] - r['ag'] for r in rows])
    nc, nt = len(comps), len(keys)
    group_counts = np.bincount(group, minlength=nc)

    def centered(raw):
        means = np.bincount(group, weights=raw, minlength=nc) / group_counts
        return raw - means[group]

    initial = np.zeros(2 * nc + 2 * nt)
    for c in range(nc):
        d = z[cidx == c]
        # Moment start only; all fitted quantities maximize the net-goal likelihood.
        variance = max(float(np.var(d)), abs(float(np.mean(d))) + .2)
        lh, la = (variance + np.mean(d)) / 2, (variance - np.mean(d)) / 2
        initial[c] = math.log(la)
        initial[nc + c] = math.log(lh / la)

    def objective(x):
        mu, home = x[:nc], x[nc:2*nc]
        raw_a, raw_d = x[2*nc:2*nc+nt], x[2*nc+nt:]
        attack, defence = centered(raw_a), centered(raw_d)
        logh = mu[cidx] + home[cidx] + attack[hidx] - defence[aidx]
        loga = mu[cidx] + attack[aidx] - defence[hidx]
        lp, gh, ga = log_probability_and_lograte_score(z, logh, loga)
        value = -float(np.sum(lp)) + .5 * STATIC_RIDGE * float(raw_a @ raw_a + raw_d @ raw_d)
        grad_mu = -np.bincount(cidx, weights=gh+ga, minlength=nc)
        grad_home = -np.bincount(cidx, weights=gh, minlength=nc)
        grad_a = -np.bincount(hidx, weights=gh, minlength=nt) - np.bincount(aidx, weights=ga, minlength=nt)
        grad_d = np.bincount(aidx, weights=gh, minlength=nt) + np.bincount(hidx, weights=ga, minlength=nt)
        grad = np.concatenate((grad_mu, grad_home, centered(grad_a)+STATIC_RIDGE*raw_a,
                               centered(grad_d)+STATIC_RIDGE*raw_d))
        return value, grad

    result = minimize(objective, initial, method='L-BFGS-B', jac=True,
                      bounds=[(-2., 2.)]*nc + [(-1.5, 1.5)]*nc + [(-2., 2.)]*(2*nt),
                      options={'maxiter': 250, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 30})
    if not result.success or not math.isfinite(result.fun):
        raise RuntimeError(f"Static panel fit did not converge: {result.message}")
    x = result.x
    attack, defence = centered(x[2*nc:2*nc+nt]), centered(x[2*nc+nt:])
    counts, last = Counter(), {}
    for r in rows:
        for side in ('home', 'away'):
            k = (r['competition'], r[side])
            counts[k] += 1
            last[k] = r['date']
    return {
        'method': 'centered_penalized_static_skellam', 'ridge': STATIC_RIDGE,
        'competitions': {c: {'mu': float(x[i]), 'home': float(x[nc+i])} for c, i in ci.items()},
        'teams': [{'competition': c, 'team': t, 'attack': float(attack[i]), 'defence': float(defence[i]),
                   'observations': counts[(c,t)], 'last_observed_date': last[(c,t)]}
                  for (c,t), i in ti.items()],
        'training_min_date': rows[0]['date'], 'training_max_date': rows[-1]['date'],
        'training_matches': len(rows), 'training_ids': [r['match_id'] for r in rows],
        'optimizer': {'success': bool(result.success), 'message': str(result.message),
                      'iterations': int(result.nit), 'evaluations': int(result.nfev),
                      'penalized_nll': float(result.fun),
                      'unpenalized_nll': float(result.fun - .5*STATIC_RIDGE*(x[2*nc:]@x[2*nc:])),
                      'max_abs_gradient': float(np.max(np.abs(result.jac)))},
    }


class PersistentGoalDifferenceFilter:
    """Immutable forecasts; all observations from a day use its same prior state.

    Deviation from each initial static anchor decays by retention**(days/7).
    A match contributes step*gradient after the day's predictions. Newly observed
    teams have a neutral algorithmic anchor, not an estimated prior distribution;
    before their first observation public forecasts remain unavailable.
    """

    def __init__(self, static, parameters):
        self.parameters = _parameters(parameters)
        self.competitions = deepcopy(static['competitions'])
        self.cursor = _date(static['training_max_date'])
        self.observed_matches = int(static['training_matches'])
        self.training_max_date = static['training_max_date']
        self.teams = {}
        for r in static['teams']:
            self.teams[(r['competition'], r['team'])] = {
                'anchor': np.array([r['attack'], r['defence']], dtype=float),
                'deviation': np.zeros(2), 'observations': int(r['observations']),
                'last_observed_date': r['last_observed_date'],
            }

    def _decay(self, day):
        days = (day - self.cursor).days
        if days < 0:
            raise ValueError("Cannot move state backwards")
        return np.power([self.parameters['attack_retention_week'], self.parameters['defence_retention_week']], days / 7.)

    def _values(self, key, decay):
        team = self.teams.get(key)
        return np.zeros(2) if team is None else team['anchor'] + decay * team['deviation']

    def _rates(self, competition, home, away, decay):
        league = self.competitions[competition]
        ah, dh = self._values((competition, home), decay)
        aa, da = self._values((competition, away), decay)
        return float(league['mu'] + league['home'] + ah - da), float(league['mu'] + aa - dh)

    def observe_day(self, rows):
        rows = _rows(rows)
        if len({r['date'] for r in rows}) != 1:
            raise ValueError("Observe a whole single day together")
        day = _date(rows[0]['date'])
        if day <= self.cursor:
            raise ValueError("Observed days must increase strictly; group all same-day matches")
        decay = self._decay(day)
        rates = np.array([self._rates(r['competition'], r['home'], r['away'], decay) for r in rows])
        z = np.array([r['hg'] - r['ag'] for r in rows])
        lp, gh, ga = log_probability_and_lograte_score(z, rates[:,0], rates[:,1])
        pending = defaultdict(lambda: np.zeros(2))
        for r, h, a in zip(rows, gh, ga):
            pending[(r['competition'], r['home'])] += [h, -a]
            pending[(r['competition'], r['away'])] += [a, -h]
        # Every likelihood above used the day-start state. Decay once, then add.
        for team in self.teams.values():
            team['deviation'] *= decay
        steps = np.array([self.parameters['attack_step'], self.parameters['defence_step']])
        for key, score in pending.items():
            if key not in self.teams:
                self.teams[key] = {'anchor': np.zeros(2), 'deviation': np.zeros(2),
                                   'observations': 0, 'last_observed_date': None}
            self.teams[key]['deviation'] += steps * score
        for r in rows:
            for side in ('home', 'away'):
                team = self.teams[(r['competition'], r[side])]
                team['observations'] += 1
                team['last_observed_date'] = r['date']
        self.observed_matches += len(rows)
        self.cursor = day
        return {'log_likelihoods': np.asarray(lp).tolist()}

    def forecast(self, competition, home, away, forecast_date):
        day = _date(forecast_date)
        if day <= self.cursor:
            raise ValueError("Only results strictly before forecast date are allowed")
        if competition not in self.competitions or home == away:
            raise ValueError("Unknown competition or identical teams")
        unknown = [name for name in (home, away)
                   if (competition, name) not in self.teams or self.teams[(competition,name)]['observations'] == 0]
        common = {'gas_available': not unknown, 'unknown_teams': unknown,
                  'observed_matches': self.observed_matches, 'last_observed_date': self.cursor.isoformat()}
        if unknown:
            return {**common, 'gas_probabilities': None, 'log_rates': None,
                    'net_goal_distribution': None, 'unavailable_reason': 'no_observed_history'}
        rates = self._rates(competition, home, away, self._decay(day))
        distribution = net_goal_probabilities(*rates)
        return {**common, 'gas_probabilities': distribution['hda'], 'log_rates': list(rates),
                'net_goal_distribution': distribution, 'unavailable_reason': None}

    def explain_state(self, competition, home, away, forecast_date):
        """Expose already learned conditional states without creating forecasts.

        Unknown teams have no public numerical state. A neutral coordinate used
        internally to start their first observation is not an estimated ability.
        These are state coordinates, not player effects or posterior samples.
        """
        day = _date(forecast_date)
        if day <= self.cursor:
            raise ValueError("Only results strictly before forecast date are allowed")
        if competition not in self.competitions or home == away:
            raise ValueError("Unknown competition or identical teams")
        decay = self._decay(day)
        sides, unknown = {}, []
        for side, name in (('home', home), ('away', away)):
            team = self.teams.get((competition, name))
            known = team is not None and team['observations'] > 0
            if not known:
                unknown.append(name)
            sides[side] = {
                'team': name, 'available': known,
                'state': (team['anchor'] + decay * team['deviation']).tolist() if known else None,
                'anchor': team['anchor'].tolist() if known else None,
                'deviation_at_forecast': (decay * team['deviation']).tolist() if known else None,
                'observations': team['observations'] if known else 0,
                'last_observed_date': team['last_observed_date'] if known else None,
            }
        return {'forecast_date': forecast_date, 'state_history_max_date': self.cursor.isoformat(),
                'available': not unknown, 'unknown_teams': unknown,
                'league': deepcopy(self.competitions[competition]), **sides,
                'log_rates': list(self._rates(competition, home, away, decay)) if not unknown else None}

    def state_dict(self):
        return {'date': self.cursor.isoformat(), 'observed_matches': self.observed_matches,
                'static_training_max_date': self.training_max_date,
                'parameters': deepcopy(self.parameters), 'competitions': deepcopy(self.competitions),
                'teams': [{'competition': c, 'team': t, 'anchor': s['anchor'].tolist(),
                           'deviation': s['deviation'].tolist(), 'observations': s['observations'],
                           'last_observed_date': s['last_observed_date']}
                          for (c,t), s in sorted(self.teams.items())]}

    @classmethod
    def from_state_dict(cls, saved):
        """Restore saved persistent state without learning from later matches."""
        static = {'competitions': saved['competitions'], 'training_max_date': saved['date'],
                  'training_matches': saved['observed_matches'],
                  'teams': [{**r, 'attack': r['anchor'][0], 'defence': r['anchor'][1]}
                            for r in saved['teams']]}
        result = cls(static, saved['parameters'])
        result.training_max_date = saved['static_training_max_date']
        for r in saved['teams']:
            deviation = np.array(r['deviation'], dtype=float)
            if deviation.shape != (2,) or not np.all(np.isfinite(deviation)):
                raise ValueError("Invalid saved state deviation")
            result.teams[(r['competition'], r['team'])]['deviation'] = deviation
        return result


def fit_dynamics(static, rows):
    """One four-parameter likelihood optimization, with fixed static anchors."""
    rows = _rows(rows)
    if rows[0]['date'] <= static['training_max_date']:
        raise ValueError("Dynamic fitting must follow the static initialization season")
    days = defaultdict(list)
    for r in rows:
        days[r['date']].append(r)
    history, failed = [], []

    def objective(x):
        params = dict(zip(PARAMETER_NAMES, map(float, x)))
        try:
            state = PersistentGoalDifferenceFilter(static, params)
            nll = 0.
            for matches in days.values():
                nll -= math.fsum(state.observe_day(matches)['log_likelihoods'])
            if not math.isfinite(nll):
                raise FloatingPointError('Nonfinite likelihood')
        except (FloatingPointError, OverflowError, ArithmeticError) as error:
            # This is an optimizer failure marker, never a substituted density.
            failed.append({'parameters': params, 'error': str(error)})
            return 1e100
        history.append({'parameters': params, 'nll': nll})
        return nll

    result = minimize(objective, DYNAMIC_START, method='L-BFGS-B', bounds=DYNAMIC_BOUNDS,
                      options={'maxiter': 150, 'maxfun': 1500, 'ftol': 1e-9, 'gtol': 1e-5, 'maxls': 30})
    if not result.success or not math.isfinite(result.fun) or result.fun >= 1e99:
        raise RuntimeError(f"Dynamic panel fit did not converge: {result.message}; failed evaluations={len(failed)}")
    return {'method': 'calendar_week_retention_identity_score_skellam',
            'parameters': dict(zip(PARAMETER_NAMES, map(float, result.x))),
            'initial_parameters': dict(zip(PARAMETER_NAMES, DYNAMIC_START)),
            'bounds': dict(zip(PARAMETER_NAMES, DYNAMIC_BOUNDS)),
            'training_min_date': rows[0]['date'], 'training_max_date': rows[-1]['date'],
            'training_matches': len(rows), 'training_ids': [r['match_id'] for r in rows],
            'optimizer': {'success': bool(result.success), 'message': str(result.message),
                          'iterations': int(result.nit), 'evaluations': int(result.nfev),
                          'nll': float(result.fun), 'start_nll': history[0]['nll'],
                          'gradient': dict(zip(PARAMETER_NAMES, map(float, result.jac))),
                          'max_abs_gradient': float(np.max(np.abs(result.jac)))},
            'evaluations': history, 'failed_evaluations': failed}


def forecast_targets(static, parameters, history_after_static, targets):
    """Replay all targets with results date < F, no deletion of unknown teams."""
    matches = _rows(history_after_static, allow_empty=True)
    if matches and matches[0]['date'] <= static['training_max_date']:
        raise ValueError("History overlaps static initialization")
    targets = list(targets)
    if len({r['match_id'] for r in targets}) != len(targets):
        raise ValueError("Duplicate target identifiers")
    grouped = defaultdict(list)
    for r in matches:
        grouped[r['date']].append(r)
    observed_days = sorted(grouped)
    index, output = 0, [None]*len(targets)
    state = PersistentGoalDifferenceFilter(static, parameters)
    for target_index, r in sorted(enumerate(targets), key=lambda p: (p[1]['F'], p[1]['match_id'])):
        if not static['training_max_date'] < r['F'] < r['date']:
            raise ValueError("Target forecast must follow initialization and precede kickoff day")
        while index < len(observed_days) and observed_days[index] < r['F']:
            state.observe_day(grouped[observed_days[index]])
            index += 1
        forecast = state.forecast(r['competition'], r['home'], r['away'], r['F'])
        original = r['original_hda']
        if len(original) != 3 or any(not math.isfinite(v) or v <= 0 for v in original) or abs(sum(original)-1) > 1e-8:
            raise ValueError("Require the original complete H/D/A probabilities")
        output[target_index] = {**deepcopy(r), **forecast, 'deployment_probabilities':
                               deepcopy(forecast['gas_probabilities'] if forecast['gas_available'] else original)}
    # State is at the last forecast's information prefix, not at the file's end.
    return output, state.state_dict()
