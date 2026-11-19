"""Date-prefix football 1X2 adapter over the unmodified MIT KickScore package.

Upstream: https://github.com/lucasmaystre/kickscore (version 0.2.0).
KickScore owns the Gaussian-process priors and EP state inference.  The only
probability arithmetic here is an equivalent, stable evaluation of its ordered
probit prediction, avoiding cancellation in its public tie-probability method.
Item variances are KickScore's independent-item approximation, not a complete
joint posterior or uncertainty in the shared hyperparameters.
"""
from copy import deepcopy
from datetime import date
from importlib.metadata import version
import json
import math

import kickscore as ks
import numpy as np
from scipy.special import log_ndtr

from kickscore.observation.observation import Observation


_DEFAULTS = {'cvar': 1., 'dvar': 1., 'lscale_days': 180., 'margin': .35,
             'havar': 1., 'max_iter': 100, 'tol': 1e-3, 'lr': 1.}


def _calendar(value, label):
    if type(value) is date:
        return value
    if not isinstance(value, str):
        raise ValueError(f'Need an ISO calendar date: {label}')
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f'Need an ISO calendar date: {label}') from error


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'Need a nonempty {label}')
    return value


def _outcome(value):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or int(value) not in (0, 1, 2)):
        raise ValueError('Outcome must be integer 0=home, 1=draw or 2=away')
    return int(value)


def _ordered_probit_probabilities(mean, variance, margin):
    """Integrate N(mean, variance) through unit-noise ordered probit exactly.

    Symmetry keeps the tie interval out of the positive extreme tail.  Log-CDF
    subtraction with expm1 retains tiny draws; no clipping or renormalization.
    """
    if not all(math.isfinite(x) for x in (mean, variance, margin)):
        raise ValueError('Ordered-probit moments and margin must be finite')
    if variance < 0 or margin <= 0:
        raise ValueError('Ordered probit needs nonnegative variance and positive margin')
    scale = math.sqrt(1. + variance)
    log_home = float(log_ndtr((mean-margin)/scale))
    log_away = float(log_ndtr((-mean-margin)/scale))
    upper = float(log_ndtr((margin-abs(mean))/scale))
    lower = float(log_ndtr((-margin-abs(mean))/scale))
    difference = lower-upper
    if difference >= 0:
        raise ValueError('Tie probability lost numeric resolution; prediction failed')
    log_draw = upper+math.log(-math.expm1(difference))
    probabilities = [math.exp(log_home), math.exp(log_draw), math.exp(log_away)]
    if (any(not math.isfinite(p) or p <= 0 or p > 1 for p in probabilities)
            or not math.isclose(sum(probabilities), 1., rel_tol=0., abs_tol=1e-12)):
        raise ValueError('Ordered-probit probabilities are nonfinite, zero or invalid; prediction failed')
    return probabilities


class KickScoreOutcomeModel:
    """Competition-specific teams, a home item, and shared prior parameters.

    ``fit`` returns self; ``converged`` is the actual upstream convergence flag.
    An unconverged model remains inspectable, but cannot issue predictions.
    Cutoff arguments are calendar ``date`` objects; row dates may also be ISO
    strings.  A fit sees only rows with date strictly before its cutoff.  All
    prediction state is extrapolated to the fixture date, not its publication.
    """

    def __init__(self, kind='dynamic', parameters=None):
        if kind not in ('dynamic', 'static'):
            raise ValueError('KickScore kind must be dynamic or static')
        supplied = {} if parameters is None else dict(parameters)
        if set(supplied)-set(_DEFAULTS):
            raise ValueError(f'Unknown KickScore parameters: {sorted(set(supplied)-set(_DEFAULTS))}')
        settings = {**_DEFAULTS, **supplied}
        for name, value in settings.items():
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not math.isfinite(float(value)) or value <= 0):
                raise ValueError(f'KickScore {name} must be finite and positive')
        if int(settings['max_iter']) != settings['max_iter']:
            raise ValueError('KickScore max_iter must be a positive integer')
        if settings['lr'] > 1:
            raise ValueError('KickScore EP lr must be at most one')
        self.kind = kind
        self.settings = {k: (int(v) if k == 'max_iter' else float(v)) for k, v in settings.items()}
        self.converged = False

    def _team_kernel(self):
        kernel = ks.kernel.Constant(var=self.settings['cvar'])
        if self.kind == 'dynamic':
            kernel += ks.kernel.Exponential(var=self.settings['dvar'], lscale=self.settings['lscale_days'])
        return kernel

    def _ensure_team(self, competition, team):
        key = (competition, team)
        if key not in self._team_items:
            name = json.dumps(['team', competition, team], ensure_ascii=False)
            self.upstream_model.add_item(name, kernel=self._team_kernel())
            self._team_items[key] = name
        return self._team_items[key]

    def _ensure_home(self, competition):
        if competition not in self._home_items:
            name = json.dumps(['home', competition], ensure_ascii=False)
            self.upstream_model.add_item(name, kernel=ks.kernel.Constant(var=self.settings['havar']))
            self._home_items[competition] = name
        return self._home_items[competition]

    def _sides(self, competition, home, away):
        return ([self._ensure_team(competition, home), self._ensure_home(competition)],
                [self._ensure_team(competition, away)])

    @staticmethod
    def _identity(row):
        identity = _text(row['match_id'], 'match identity')
        competition = _text(row['competition'], 'competition')
        home, away = _text(row['home'], 'home team'), _text(row['away'], 'away team')
        if home == away:
            raise ValueError('A fixture needs different home and away teams')
        return identity, competition, home, away

    @classmethod
    def _prepare_history(cls, history_rows, before_date):
        if type(before_date) is not date:
            raise ValueError('Use a calendar-date fit cutoff')
        # Filtering precedes reading identities, teams or outcome labels.
        selected = []
        for row in history_rows:
            played = _calendar(row['date'], 'history date')
            if played < before_date:
                selected.append((played, row))
        prepared = []
        for played, row in selected:
            prepared.append((played, cls._identity(row), _outcome(row['actual_outcome'])))
        if not prepared:
            raise ValueError('Need at least one strictly earlier fixture')
        ids = [identity[0] for _, identity, _ in prepared]
        if len(ids) != len(set(ids)):
            raise ValueError('Need unique earlier fixture identities')
        prepared.sort(key=lambda entry: (entry[0], entry[1][0]))
        return prepared

    def _observe(self, prepared):
        for played, (_, competition, home, away), outcome in prepared:
            home_side, away_side = self._sides(competition, home, away)
            # The home term stays on the home side even when the away team wins.
            winners, losers = (away_side, home_side) if outcome == 2 else (home_side, away_side)
            self.upstream_model.observe(winners=winners, losers=losers,
                t=float(played.toordinal()), tie=outcome == 1)

    def fit(self, history_rows, before_date):
        self.converged = False
        if hasattr(self, '_parameters'):
            del self._parameters
        prepared = self._prepare_history(history_rows, before_date)
        self.upstream_model = ks.TernaryModel(margin=self.settings['margin'], obs_type='probit')
        self._team_items, self._home_items = {}, {}
        self._observe(prepared)
        self._fit_calls = 0
        self._run_fit()
        self._record_fit(prepared, before_date, len(prepared), 'fresh')
        return self

    def _run_fit(self):
        self.converged = False
        self.converged = bool(self.upstream_model.fit(method='ep',
            max_iter=self.settings['max_iter'], tol=self.settings['tol'], lr=self.settings['lr']))
        self._fit_calls += 1

    def _record_fit(self, prepared, before_date, n_added, mode):
        self.before_date = before_date
        self._training_rows = prepared
        # Prediction-only prior items must not become known training coverage.
        self._known_teams = {(competition, team) for _, (_, competition, home, away), _ in prepared
                             for team in (home, away)}
        likelihood, likelihood_error = None, None
        try:
            likelihood = float(self.upstream_model.log_likelihood)
            if not math.isfinite(likelihood):
                raise ValueError('nonfinite upstream approximate log likelihood')
        except (ArithmeticError, ValueError, RuntimeError) as error:
            likelihood, likelihood_error = None, str(error)
        self._parameters = {
            'model_type': 'kickscore_ordered_probit_date_prefix', 'kind': self.kind,
            'upstream_package': 'kickscore', 'upstream_version': version('kickscore'),
            'upstream_repository': 'https://github.com/lucasmaystre/kickscore', 'upstream_license': 'MIT',
            'before_date': before_date.isoformat(), 'converged': self.converged,
            'training_min_date': prepared[0][0].isoformat(), 'training_max_date': prepared[-1][0].isoformat(),
            'n_matches': len(prepared), 'reference_match_ids': [identity[0] for _, identity, _ in prepared],
            'n_teams': len(self._known_teams),
            'n_competitions': len({competition for competition, _ in self._known_teams}),
            'fit_mode': mode, 'n_added_matches': n_added, 'n_upstream_fit_calls': self._fit_calls,
            'hyperparameters': {k: self.settings[k] for k in ('cvar', 'dvar', 'lscale_days', 'margin', 'havar')},
            'fit_controls': {'method': 'ep', **{k: self.settings[k] for k in ('max_iter', 'tol', 'lr')}},
            'log_likelihood': likelihood, 'log_likelihood_error': likelihood_error,
            'likelihood_note': 'upstream EP approximate log marginal likelihood, not held-out outcome score',
            'classes': [0, 1, 2], 'class_names': ['home', 'draw', 'away'],
            'time_units': 'calendar date ordinal in days',
            'history_rule': 'date strictly before fit cutoff; smoothing only within that known prefix',
            'unknown_team_policy': 'independent nondegenerate same-competition GP prior; coverage reported separately',
            'variance_note': 'upstream independent-item approximation; shared hyperparameter uncertainty excluded',
            'probability_policy': 'equivalent stable analytic probit CDF; no clipping or renormalization',
            'output_scope': '1X2 only; no score, total goals, handicap or advancement distribution',
        }

    def advance(self, history_rows, before_date):
        """Check a complete later prefix and fit it afresh when matches changed.

        Earlier data must match the already observed rows.  A changed history
        needs a fresh fit; changing an observation in place is not supported by
        KickScore.  No-new-match dates reuse the identical posterior without an
        invented training call.  Integration checks exposed upstream EP numeric
        failures on some prefixes with both fresh and warm initialization.  No
        reliable warm optimization is assumed here: this implementation uses
        the same original fresh algorithm without changing the model or EP rate.
        """
        if not hasattr(self, '_parameters'):
            raise ValueError('Fit KickScore before advancing its history')
        if type(before_date) is not date or before_date < self.before_date:
            raise ValueError('Advance cutoff cannot go backwards')
        prepared = self._prepare_history(history_rows, before_date)
        n_previous = len(self._training_rows)
        if prepared[:n_previous] != self._training_rows:
            raise ValueError('Advance cannot delete or change earlier history; use a fresh fit')
        added = prepared[n_previous:]
        if any(played < self.before_date for played, _, _ in added):
            raise ValueError('Advance cannot insert older history; use a fresh fit')
        if added:
            previous_calls = self._fit_calls
            rows = [{'date': played, 'match_id': identity, 'competition': competition,
                     'home': home, 'away': away, 'actual_outcome': outcome}
                    for played, (identity, competition, home, away), outcome in prepared]
            self.fit(rows, before_date)
            self._fit_calls += previous_calls
        self._record_fit(prepared, before_date, len(added), 'fresh_advance' if added else 'unchanged_prefix')
        return self

    def parameters(self):
        if not hasattr(self, '_parameters'):
            raise ValueError('Fit KickScore before reading fitted parameters')
        return deepcopy(self._parameters)

    def predict_many(self, fixture_rows, publication_before_date):
        if not hasattr(self, '_parameters'):
            raise ValueError('Fit strictly earlier KickScore history before predicting')
        if not self.converged:
            raise ValueError('Upstream KickScore EP did not converge; prediction refused')
        if type(publication_before_date) is not date or publication_before_date < self.before_date:
            raise ValueError('Publication date must be a calendar date at or after the fit cutoff')
        prepared, seen = [], set()
        for row in fixture_rows:
            played = _calendar(row['date'], 'fixture date')
            identity, competition, home, away = self._identity(row)
            if played < publication_before_date:
                raise ValueError('Cannot predict a fixture earlier than its publication date')
            if identity in seen:
                raise ValueError('Need unique prediction fixture identities')
            seen.add(identity)
            prepared.append((played, identity, competition, home, away))
        results = []
        for played, identity, competition, home, away in prepared:
            unknown = [team for team in (home, away) if (competition, team) not in self._known_teams]
            home_side, away_side = self._sides(competition, home, away)
            elements = (self.upstream_model.process_items(home_side, +1)
                        + self.upstream_model.process_items(away_side, -1))
            mean, variance = (float(x) for x in Observation.f_params(elements, float(played.toordinal())))
            probabilities = _ordered_probit_probabilities(mean, variance, self.settings['margin'])
            mean_only = _ordered_probit_probabilities(mean, 0., self.settings['margin'])
            raw, raw_error = None, None
            try:
                raw = [float(p) for p in self.upstream_model.probabilities(home_side, away_side, float(played.toordinal()))]
                if not all(math.isfinite(p) for p in raw):
                    raise ValueError('upstream public probabilities were nonfinite')
            except (ArithmeticError, ValueError, RuntimeError) as error:
                raw, raw_error = None, str(error)
            results.append({'match_id': identity, 'date': played.isoformat(), 'competition': competition,
                'probabilities': probabilities, 'mean_difference': mean, 'variance_difference': variance,
                'mean_only_probabilities': mean_only, 'unknown_teams': unknown,
                'raw_upstream_probabilities': raw, 'raw_upstream_probability_error': raw_error})
        return results
