"""Fixed logistic combination of dated, saved probability forecasts.

The inputs are genuinely earlier/out-of-sample member probabilities supplied
by the caller. This layer cannot certify their origin from the numbers alone.
It uses only probability contrasts and a few explicit interactions, not the
original 161/236-column designs. Missing personnel residuals are encoded as an
unavailable branch, never as a known zero player effect.
"""
from copy import deepcopy
from datetime import date
import math
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


BASE_ROUTE = 'rich_performance'
OTHER_ROUTES = ('rich_allcomp', 'goal', 'xg', 'gap', 'kickscore', 'rating_joint')
ROUTES = (BASE_ROUTE, *OTHER_ROUTES)
LOG_ODDS = ('home_away', 'draw_away')
INTERACTION_ROUTES = ('xg', 'gap', 'rating_joint')
COMPETITIONS = ('E0', 'D1', 'SP1', 'I1', 'F1')
PERSONNEL = ('availability_control', 'availability_status', 'availability_usage')
FEATURE_NAMES = (
    *(f'base_log_{axis}' for axis in LOG_ODDS),
    *(f'delta_{route}_log_{axis}' for route in OTHER_ROUTES for axis in LOG_ODDS),
    *(f'personnel_status_minus_control_log_{axis}' for axis in LOG_ODDS),
    *(f'personnel_usage_minus_status_log_{axis}' for axis in LOG_ODDS),
    'personnel_probability_available',
    *(f'league_{competition}' for competition in COMPETITIONS[1:]),
    *(f'balance_delta_{route}_log_{axis}'
      for route in INTERACTION_ROUTES for axis in LOG_ODDS),
)
CONFIGURATION = {
    'C': 1.0, 'solver': 'lbfgs', 'max_iter': 2000, 'tol': 1e-8,
}


def _day(value, label):
    if type(value) is date:
        return value
    if not isinstance(value, str):
        raise ValueError(f'Need an ISO calendar date: {label}')
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f'Need an ISO calendar date: {label}') from error


def _probability(value, label):
    try:
        p = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f'Need three ordered H/D/A probabilities: {label}') from error
    if (p.shape != (3,) or not np.isfinite(p).all()
            or np.any(p <= 0) or np.any(p > 1)
            or not np.isclose(p.sum(), 1., rtol=0, atol=1e-10)):
        raise ValueError(f'Need positive finite normalized H/D/A probabilities: {label}')
    return p


def _log_odds(p):
    # Subtract logs rather than dividing, preserving finite tiny probabilities.
    logp = np.log(p)
    return logp[:2] - logp[2]


def _features(row):
    competition = row.get('competition')
    if competition not in COMPETITIONS:
        raise ValueError('Need a supported explicit competition')
    played = _day(row.get('date'), 'fixture date')
    forecast = _day(row.get('forecast_before_date'), 'forecast publication date')
    if forecast >= played:
        raise ValueError('Forecast publication must be strictly before fixture date')
    supplied = row.get('probabilities_by_route')
    if not isinstance(supplied, dict) or set(supplied) != set(ROUTES):
        raise ValueError('Need exactly the seven named probability routes')
    odds = {route: _log_odds(_probability(supplied[route], route)) for route in ROUTES}
    base = odds[BASE_ROUTE]
    deltas = {route: odds[route] - base for route in OTHER_ROUTES}
    values = list(base)
    values.extend(number for route in OTHER_ROUTES for number in deltas[route])
    if any(key not in row for key in PERSONNEL):
        raise ValueError('Personnel routes must be explicit probabilities or None')
    personnel = [row[key] for key in PERSONNEL]
    observed = [p is not None for p in personnel]
    if any(observed) and not all(observed):
        raise ValueError('All three personnel probabilities must be observed together or all unknown')
    if competition != 'E0' and any(observed):
        raise ValueError('Non-EPL personnel routes are unavailable and must be None')
    available = all(observed)
    if available:
        control, status, usage = [_log_odds(_probability(p, label))
                                  for p, label in zip(personnel, PERSONNEL, strict=True)]
        values.extend(status - control)
        values.extend(usage - status)
    else:
        values.extend((0., 0., 0., 0.))
    values.append(float(available))
    values.extend(float(competition == c) for c in COMPETITIONS[1:])
    balance = 1. / (1. + abs(float(base[0])))
    values.extend(number * balance for route in INTERACTION_ROUTES for number in deltas[route])
    result = np.asarray(values, dtype=float)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError('Expected the declared finite probability-contrast design')
    return result, played, forecast


class IntegratedProbabilityStack:
    """StandardScaler plus one fixed regularized three-class logistic model."""

    def fit(self, rows, before_date):
        if type(before_date) is not date:
            raise ValueError('Use an explicit calendar-date fit cutoff')
        rows = list(rows)
        if not rows:
            raise ValueError('Need nonempty earlier training predictions')
        # Reject future rows before reading their labels; they are not silently
        # filtered and cannot influence this fit or its feature scaling.
        if any(_day(row.get('date'), 'training fixture date') >= before_date for row in rows):
            raise ValueError('Every training fixture date must be strictly before fit cutoff')
        identities = [row.get('match_id') for row in rows]
        if (any(not isinstance(identifier, str) or not identifier for identifier in identities)
                or len(set(identities)) != len(identities)):
            raise ValueError('Need unique nonempty training fixture identities')
        rows.sort(key=lambda row: (_day(row['date'], 'training fixture date'), row['match_id']))
        design, outcomes, availability = [], [], []
        for row in rows:
            features, _, _ = _features(row)
            outcome = row.get('actual_outcome')
            if (isinstance(outcome, (bool, np.bool_))
                    or not isinstance(outcome, (int, np.integer))
                    or int(outcome) not in (0, 1, 2)):
                raise ValueError('Training outcome must be integer H=0/D=1/A=2')
            design.append(features)
            outcomes.append(int(outcome))
            availability.append(bool(features[FEATURE_NAMES.index('personnel_probability_available')]))
        if set(outcomes) != {0, 1, 2}:
            raise ValueError('All three earlier training classes are required')
        matrix = np.asarray(design)
        scaler = StandardScaler()
        standardized = scaler.fit_transform(matrix)
        estimator = LogisticRegression(**CONFIGURATION)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            estimator.fit(standardized, np.asarray(outcomes, dtype=int))
        failures = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        if failures:
            raise RuntimeError('Fixed logistic stack did not converge: ' + '; '.join(failures))
        if estimator.classes_.tolist() != [0, 1, 2]:
            raise ValueError('Estimator must preserve ordered H/D/A classes')
        self.before_date = before_date
        self.feature_names = list(FEATURE_NAMES)
        self.scaler, self.estimator = scaler, estimator
        self._parameters = {
            'model_type': 'fixed_scaled_logistic_probability_residual_stack',
            'before_date': before_date.isoformat(),
            'class_order': ['home', 'draw', 'away'], 'classes': [0, 1, 2],
            'route_names': list(ROUTES), 'feature_names': list(FEATURE_NAMES),
            'n_features': len(FEATURE_NAMES), 'n_matches': len(rows),
            'reference_match_ids': [row['match_id'] for row in rows],
            'training_date_min': rows[0]['date'].isoformat() if type(rows[0]['date']) is date else rows[0]['date'],
            'training_date_max': rows[-1]['date'].isoformat() if type(rows[-1]['date']) is date else rows[-1]['date'],
            'training_class_counts': {str(c): outcomes.count(c) for c in (0, 1, 2)},
            'personnel_probability_available_matches': sum(availability),
            'configuration': deepcopy(CONFIGURATION), 'sklearn_version': sklearn.__version__,
            'scaler': {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                       'variance': scaler.var_.tolist(), 'n_samples_seen': int(scaler.n_samples_seen_)},
            'coefficients': estimator.coef_.tolist(), 'intercepts': estimator.intercept_.tolist(),
            'n_iterations': estimator.n_iter_.tolist(), 'convergence_warnings': [],
            'balance_definition': '1 / (1 + abs(base log(P_home / P_away)))',
            'personnel_missing_policy': 'four residual zeros plus personnel_probability_available=0; missing branch, not a known zero effect',
            'normalization': 'input probabilities positive normalized H/D/A; estimator softmax output',
            'training_information_assumption': 'caller-supplied earlier/out-of-sample member forecasts; publication date precedes each fixture',
            'parameter_search': False, 'base_model_refits': 0,
            'output_scope': '1X2 only; score reconciliation is a separate explicit operation',
        }
        return self

    def predict_many(self, rows):
        if not hasattr(self, '_parameters'):
            raise ValueError('Fit the probability stack before predicting')
        rows = list(rows)
        if not rows:
            raise ValueError('Need nonempty prediction rows')
        features = []
        for row in rows:
            values, _, forecast = _features(row)
            if forecast < self.before_date:
                raise ValueError('Prediction publication cannot precede stack fit cutoff')
            features.append(values)
        probabilities = np.asarray(self.estimator.predict_proba(
            self.scaler.transform(np.asarray(features))), dtype=float)
        if (probabilities.shape != (len(rows), 3) or not np.isfinite(probabilities).all()
                or np.any(probabilities <= 0)
                or not np.allclose(probabilities.sum(axis=1), 1., rtol=0, atol=1e-10)):
            raise ValueError('Stack returned invalid positive ordered H/D/A probabilities')
        return probabilities

    def parameters(self):
        if not hasattr(self, '_parameters'):
            raise ValueError('Fit the probability stack before reading parameters')
        return deepcopy(self._parameters)
