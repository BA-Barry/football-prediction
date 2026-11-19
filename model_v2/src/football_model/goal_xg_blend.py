"""One bounded shared log-rate weight, learned from earlier real goals only.

Original goal/xG rate provenance must be checked by the caller. This module
neither refits either rate model nor changes the conditional score family.
"""
from datetime import date
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln


def _pair(value, label, *, counts=False):
    raw = np.asarray(value, dtype=object)
    if any(isinstance(v, (bool, np.bool_)) for v in raw.flat):
        raise ValueError(f'Boolean is not a count or rate: {label}')
    values = np.asarray(value)
    if values.shape != (2,) or values.dtype.kind not in 'iuf':
        raise ValueError(f'Need two numeric values: {label}')
    values = values.astype(float)
    if not np.isfinite(values).all():
        raise ValueError(f'Need finite values: {label}')
    if counts:
        if np.any(values < 0) or np.any(values != np.floor(values)):
            raise ValueError(f'Need nonnegative integer counts: {label}')
    elif np.any(values <= 0):
        raise ValueError(f'Need positive rates: {label}')
    return values


def _weight(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError('Need a numeric weight in [0, 1]')
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Need a finite weight in [0, 1]')
    return value


def predict_goal_xg_rates(goal_rates, xg_rates, weight):
    """Geometric interpolation; exact endpoint values, no renormalization."""
    goal, xg, weight = _pair(goal_rates, 'goal_rates'), _pair(xg_rates, 'xg_rates'), _weight(weight)
    if weight == 0:
        return tuple(map(float, goal))
    if weight == 1:
        return tuple(map(float, xg))
    # Interpolating logs avoids overflow from dividing very different rates.
    values = np.exp((1-weight)*np.log(goal) + weight*np.log(xg))
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Invalid blended rates')
    return tuple(map(float, values))


def goal_xg_objective(weight, goal_rates, xg_rates, actual_score):
    """Full per-match goal NLL and first/second weight derivatives.

    Arrays are n by 2; each fixture has equal weight, and both scoring sides
    share the same scalar. No new shrinkage or time decay is introduced here.
    """
    weight = _weight(weight)
    arrays=[]
    for value in (goal_rates,xg_rates,actual_score):
        original=np.asarray(value,dtype=object)
        native=np.asarray(value)
        if (any(isinstance(v,(bool,np.bool_)) for v in original.flat)
                or native.dtype.kind not in 'iuf'):
            raise ValueError('Boolean or nonnumeric values are not count/rate arrays')
        arrays.append(native.astype(float))
    goal,xg,y=arrays
    if (goal.ndim != 2 or goal.shape[1:] != (2,) or not len(goal)
            or xg.shape != goal.shape or y.shape != goal.shape
            or not np.isfinite(goal).all() or not np.isfinite(xg).all() or not np.isfinite(y).all()
            or np.any(goal <= 0) or np.any(xg <= 0) or np.any(y < 0) or np.any(y != np.floor(y))):
        raise ValueError('Need aligned positive n by 2 rate arrays and nonnegative count targets')
    log_goal, log_xg = np.log(goal), np.log(xg)
    difference = log_xg-log_goal
    log_mu = (1-weight)*log_goal+weight*log_xg
    mu = np.exp(log_mu)
    value = float(np.mean(np.sum(mu-y*log_mu+gammaln(y+1), axis=1)))
    gradient = float(np.mean(np.sum((mu-y)*difference, axis=1)))
    curvature = float(np.mean(np.sum(mu*difference*difference, axis=1)))
    if not all(math.isfinite(v) for v in (value,gradient,curvature)):
        raise ValueError('Non-finite goal-xG blend objective')
    return value, gradient, curvature


def fit_goal_xg_weight(rows, before_date):
    """Fit once on strict earlier dates, checking the two closed endpoints.

    The objective is convex on [0, 1]. Endpoint gradient signs identify a
    boundary solution; otherwise a scalar root of the monotone gradient is
    found. This is not a hyperparameter grid and does not use later outcomes.
    """
    if type(before_date) is not date:
        raise ValueError('Need a calendar cutoff')
    history=[]
    for row in rows:
        played=date.fromisoformat(row['date'])
        if played < before_date:
            history.append(row)
    if not history:
        raise ValueError('Need earlier observations to estimate a weight')
    ids=[r['match_id'] for r in history]
    if any(not isinstance(key,str) or not key for key in ids) or len(set(ids)) != len(ids):
        raise ValueError('Need unique nonempty match IDs')
    history.sort(key=lambda r:(r['date'],r['match_id']))
    ids=[r['match_id'] for r in history]
    goal=np.asarray([_pair(r['goal_rates'],'goal_rates') for r in history])
    xg=np.asarray([_pair(r['xg_rates'],'xg_rates') for r in history])
    y=np.asarray([_pair(r['actual_score'],'actual_score',counts=True) for r in history])
    evaluations=0
    def evaluate(weight):
        nonlocal evaluations
        evaluations+=1
        return goal_xg_objective(weight,goal,xg,y)
    left, right = evaluate(0.), evaluate(1.)
    flat=bool(np.array_equal(goal,xg))
    if flat:
        weight, method = 0., 'identical_rates_choose_original'
    elif left[1] >= 0:
        weight, method = 0., 'convex_left_endpoint'
    elif right[1] <= 0:
        weight, method = 1., 'convex_right_endpoint'
    else:
        weight=float(brentq(lambda w:evaluate(w)[1],0.,1.,xtol=1e-14,rtol=1e-14))
        method='brentq_convex_gradient_root'
    value,gradient,curvature=evaluate(weight)
    return {
        'weight':weight,'gradient':gradient,'objective':value,'full_nll':value,
        'curvature':curvature,'endpoint_gradients':[left[1],right[1]],
        'endpoint_full_nll':[left[0],right[0]],'not_identifiable':flat,
        'n':len(history),'before_date':before_date.isoformat(),
        'training_min_date':history[0]['date'],'training_max_date':history[-1]['date'],
        'reference_match_ids':ids,'optimizer_function_evals':evaluations,'method':method,
        'constraint':'single shared weight in [0,1] on goal/xG log rates',
        'objective_definition':'uniform mean per fixture complete-score NLL including log-factorials',
        'new_intercept':False,'new_decay':False,'new_regularization':False,
        'input_requirement':'both rate tracks are out-of-time at identical original information cutoffs',
    }
