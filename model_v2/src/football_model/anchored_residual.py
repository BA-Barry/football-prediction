"""Learn symmetric H/D/A logit residuals around a fixed reference probability.

No free intercept, league terms, reference log odds, centering or scaling are
fitted. Zero residuals therefore preserve the supplied reference distribution.
Chronological provenance of member probabilities remains the caller's duty.
"""
from copy import deepcopy
from datetime import date
import math

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp

from .integrated_stack import BASE_ROUTE, FEATURE_NAMES, _day, _features, _probability


RESIDUAL_FEATURE_NAMES = tuple(name for name in FEATURE_NAMES if (
    name.startswith('delta_') or name.startswith('personnel_status_minus_control_')
    or name.startswith('personnel_usage_minus_status_') or name.startswith('balance_delta_')))
RESIDUAL_INDICES = tuple(FEATURE_NAMES.index(name) for name in RESIDUAL_FEATURE_NAMES)
OPTIMIZER_OPTIONS = {'maxiter': 2000, 'gtol': 1e-8, 'ftol': 1e-12}


def _outcome(row):
    outcome = row.get('actual_outcome')
    if (isinstance(outcome, (bool, np.bool_))
            or not isinstance(outcome, (int, np.integer)) or int(outcome) not in (0, 1, 2)):
        raise ValueError('Training outcome must be integer H=0/D=1/A=2')
    return int(outcome)


def _training_rows(rows, before_date):
    if type(before_date) is not date:
        raise ValueError('Use an explicit calendar-date fit cutoff')
    rows = list(rows)
    if not rows:
        raise ValueError('Need nonempty earlier training predictions')
    if any(_day(row.get('date'), 'training fixture date') >= before_date for row in rows):
        raise ValueError('Every training fixture date must be strictly before fit cutoff')
    identities = [row.get('match_id') for row in rows]
    if (any(not isinstance(identifier, str) or not identifier for identifier in identities)
            or len(set(identities)) != len(identities)):
        raise ValueError('Need unique nonempty training fixture identities')
    return sorted(rows, key=lambda row: (_day(row['date'], 'training fixture date'), row['match_id']))


def _objective_and_gradient(flat_coefficients, design, reference_log_probabilities, outcomes):
    """Mean NLL + ||B||^2/(2n), with both objective terms scaled by n."""
    n = len(outcomes)
    coefficients = np.asarray(flat_coefficients).reshape(design.shape[1], 3)
    logits = reference_log_probabilities + design @ coefficients
    normalizer = logsumexp(logits, axis=1)
    objective = (float(np.sum(normalizer-logits[np.arange(n), outcomes]))
                 + .5*float(np.square(coefficients).sum()))/n
    probabilities = np.exp(logits-normalizer[:, None])
    residual = probabilities.copy()
    residual[np.arange(n), outcomes] -= 1.
    gradient = (design.T @ residual + coefficients)/n
    return objective, gradient.ravel()


def _attenuation(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('Attenuation must be finite and between zero and one')
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError('Attenuation must be finite and between zero and one') from error
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Attenuation must be finite and between zero and one')
    return value


def _apply(reference, residuals, attenuation):
    if attenuation == 0:
        return reference.copy()
    logits = np.log(reference) + attenuation*residuals
    probabilities = np.exp(logits-logsumexp(logits, axis=1)[:, None])
    # The exact zero branch is the reference, without a numerical log/softmax
    # round trip. It matters when the routes genuinely supply no correction.
    zero = np.all(residuals == 0, axis=1)
    probabilities[zero] = reference[zero]
    return probabilities


class AnchoredResidualStack:
    """One fixed C=1 L2 residual fit, anchored on rich_performance."""

    def fit(self, rows, before_date):
        rows = _training_rows(rows, before_date)
        design, reference, outcomes = [], [], []
        for row in rows:
            features, _, _ = _features(row)
            design.append(features[list(RESIDUAL_INDICES)])
            reference.append(_probability(row['probabilities_by_route'][BASE_ROUTE], BASE_ROUTE))
            outcomes.append(_outcome(row))
        if set(outcomes) != {0, 1, 2}:
            raise ValueError('All three earlier training classes are required')
        matrix = np.asarray(design, dtype=float)
        reference_log = np.log(np.asarray(reference, dtype=float))
        outcomes = np.asarray(outcomes, dtype=int)
        initial = np.zeros(3*len(RESIDUAL_FEATURE_NAMES), dtype=float)
        initial_objective, _ = _objective_and_gradient(initial, matrix, reference_log, outcomes)
        result = minimize(_objective_and_gradient, initial,
            args=(matrix, reference_log, outcomes), jac=True, method='L-BFGS-B',
            options=dict(OPTIMIZER_OPTIONS))
        if (not result.success or not np.isfinite(result.fun)
                or not np.isfinite(result.x).all() or not np.isfinite(result.jac).all()):
            raise RuntimeError(f'Fixed anchored residual optimizer did not converge: {result.message}')
        self.before_date = before_date
        self.feature_names = list(RESIDUAL_FEATURE_NAMES)
        self.coefficients = result.x.reshape(len(RESIDUAL_FEATURE_NAMES), 3).copy()
        self._parameters = {
            'model_type': 'anchored_probability_logit_residual_stack',
            'before_date': before_date.isoformat(), 'class_order': ['home', 'draw', 'away'],
            'reference_route': BASE_ROUTE, 'feature_names': list(RESIDUAL_FEATURE_NAMES),
            'n_features': len(RESIDUAL_FEATURE_NAMES), 'n_matches': len(rows),
            'reference_match_ids': [row['match_id'] for row in rows],
            'training_date_min': _day(rows[0]['date'], 'date').isoformat(),
            'training_date_max': _day(rows[-1]['date'], 'date').isoformat(),
            'coefficients_home_draw_away_columns': self.coefficients.tolist(),
            'n_stored_coefficients': int(self.coefficients.size),
            'n_effective_logit_contrasts': 2*len(RESIDUAL_FEATURE_NAMES),
            'class_symmetric_l2': True,
            'common_shift_identification': 'L2 selects minimum-norm coefficients with approximately zero sum across H/D/A; no class is fixed as reference',
            'maximum_absolute_class_coefficient_sum': float(np.max(np.abs(self.coefficients.sum(axis=1)))),
            'intercept': None, 'centering': False, 'scaling': False,
            'objective': '(sum multinomial NLL + ||B||^2/2) / n',
            'C': 1., 'optimizer': 'scipy L-BFGS-B, one zero start, analytic gradient',
            'optimizer_options': dict(OPTIMIZER_OPTIONS),
            'optimization': {'success': bool(result.success), 'message': str(result.message),
                'status': int(result.status), 'iterations': int(result.nit),
                'function_evaluations': int(result.nfev), 'gradient_evaluations': int(result.njev),
                'initial_objective': float(initial_objective), 'final_objective': float(result.fun),
                'maximum_absolute_gradient': float(np.max(np.abs(result.jac)))},
            'class_counts': {str(c): int(np.sum(outcomes == c)) for c in (0,1,2)},
            'parameter_search': False, 'restarts': 0, 'base_model_refits': 0,
            'personnel_missing_policy': 'zero residual inputs identify no available correction; not known zero player ability',
            'input_assumption': 'caller-supplied earlier/out-of-sample member forecasts; no target labels used in prediction',
            'output_scope': '1X2 only; score reconciliation is separate',
        }
        return self

    def _prediction_design(self, rows):
        if not hasattr(self, '_parameters'):
            raise ValueError('Fit the anchored residual stack before predicting')
        rows = list(rows)
        if not rows:
            raise ValueError('Need nonempty prediction rows')
        design, reference = [], []
        for row in rows:
            features, _, forecast = _features(row)
            if forecast < self.before_date:
                raise ValueError('Prediction publication cannot precede residual fit cutoff')
            design.append(features[list(RESIDUAL_INDICES)])
            reference.append(_probability(row['probabilities_by_route'][BASE_ROUTE], BASE_ROUTE))
        return np.asarray(design, dtype=float), np.asarray(reference, dtype=float)

    def residuals_many(self, rows):
        design, _ = self._prediction_design(rows)
        residuals = design @ self.coefficients
        if not np.isfinite(residuals).all():
            raise ValueError('Residual logits must be finite')
        return residuals

    def predict_many(self, rows, attenuation=1.0):
        attenuation = _attenuation(attenuation)
        design, reference = self._prediction_design(rows)
        residuals = design @ self.coefficients
        probabilities = _apply(reference, residuals, attenuation)
        if (not np.isfinite(probabilities).all() or np.any(probabilities <= 0)
                or not np.allclose(probabilities.sum(axis=1), 1., rtol=0, atol=1e-10)):
            raise ValueError('Anchored stack returned invalid positive H/D/A probabilities')
        return probabilities

    def parameters(self):
        if not hasattr(self, '_parameters'):
            raise ValueError('Fit the anchored residual stack before reading parameters')
        return deepcopy(self._parameters)


def fit_attenuation(rows, before_date):
    """Learn one gamma in [0,1] from genuinely earlier OOF residual forecasts.

    Each row supplies ``reference_probabilities``, ``residual_logits``,
    ``actual_outcome``, ``date``, ``forecast_before_date``, ``meta_fit_date`` and
    ``match_id``. The dated member/meta provenance is the caller's obligation.
    Both endpoints are evaluated explicitly, so gamma=0 remains achievable.
    """
    rows = _training_rows(rows, before_date)
    reference, residuals, outcomes = [], [], []
    for row in rows:
        played = _day(row['date'], 'fixture date')
        forecast = _day(row.get('forecast_before_date'), 'forecast date')
        meta_fit = _day(row.get('meta_fit_date'), 'residual meta-fit date')
        if not meta_fit <= forecast < played:
            raise ValueError('Residual meta-fit date must be at or before publication, before fixture')
        reference.append(_probability(row.get('reference_probabilities'), 'reference'))
        residual = np.asarray(row.get('residual_logits'), dtype=float)
        if residual.shape != (3,) or not np.isfinite(residual).all():
            raise ValueError('Need three finite residual logits ordered H/D/A')
        residuals.append(residual)
        outcomes.append(_outcome(row))
    reference, residuals = np.asarray(reference), np.asarray(residuals)
    outcomes = np.asarray(outcomes, dtype=int)
    log_reference = np.log(reference)

    def objective(gamma):
        logits = log_reference + gamma*residuals
        return float(np.mean(logsumexp(logits, axis=1)-logits[np.arange(len(rows)), outcomes]))

    result = minimize_scalar(objective, method='bounded', bounds=(0.,1.),
                             options={'xatol': 1e-10, 'maxiter': 500})
    if not result.success or not math.isfinite(float(result.fun)):
        raise RuntimeError(f'Attenuation optimizer did not converge: {result.message}')
    candidates = [(0., objective(0.)), (1., objective(1.)), (float(result.x), float(result.fun))]
    gamma, loss = min(candidates, key=lambda item: (item[1], item[0]))
    return {
        'attenuation': gamma, 'before_date': before_date.isoformat(),
        'n_matches': len(rows), 'reference_match_ids': [row['match_id'] for row in rows],
        'training_date_min': _day(rows[0]['date'], 'date').isoformat(),
        'training_date_max': _day(rows[-1]['date'], 'date').isoformat(),
        'objective': 'earlier OOF mean 1X2 negative log likelihood', 'selected_mean_nll': loss,
        'candidate_mean_nll': [{'attenuation':g,'mean_nll':v} for g,v in candidates],
        'optimizer': 'bounded scalar [0,1] plus explicit endpoints; lower gamma wins exact ties',
        'optimization': {'success': bool(result.success), 'message': str(result.message),
                         'n_function_evaluations': int(result.nfev),
                         'interior_attenuation': float(result.x)},
        'base_model_refits': 0, 'residual_model_refits': 0,
        'interpretation': 'attenuation fitted only to supplied earlier OOF residuals; gamma=0 exactly retains reference probabilities',
    }
