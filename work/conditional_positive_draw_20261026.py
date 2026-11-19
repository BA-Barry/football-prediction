"""Two-parameter conditional positive-draw tilt of a saved score distribution.

This independent research transformation learns no new team abilities.  It is
an adaptation of diagonal-inflation ideas, not an unchanged external package.
The final marginal goal means change and are recomputed from the same grid.
"""
from datetime import date
import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln

from football_model.probability import score_grid, summarize
from football_model.vendor.penaltyblog_grid import FootballProbabilityGrid


OPTIMIZER_OPTIONS = {'gtol': 1e-8, 'maxiter': 1000}


def _date(value, name):
    if type(value) is date:
        return value
    if not isinstance(value, str):
        raise ValueError(f'{name} must be an ISO calendar date')
    result = date.fromisoformat(value)
    if result.isoformat() != value:
        raise ValueError(f'{name} must be an ISO calendar date')
    return result


def _pair(value, name, *, counts=False):
    raw = np.asarray(value)
    if raw.shape != (2,) or raw.dtype.kind not in 'iuf':
        raise ValueError(f'{name} must contain two numeric measurements')
    numbers = raw.astype(float)
    if (not np.isfinite(numbers).all() or np.any(numbers < 0 if counts else numbers <= 0)
            or counts and np.any(numbers != np.floor(numbers))):
        raise ValueError(f'{name} contains invalid goal counts or means')
    return numbers


def row_pieces(row):
    """Return earlier-information inputs, base positive-draw mass and outcome."""
    mid = row.get('match_id')
    if not isinstance(mid, str) or not mid:
        raise ValueError('Need a nonempty match_id')
    fit = _date(row['fit_before_date'], 'fit_before_date')
    forecast = _date(row['forecast_before_date'], 'forecast_before_date')
    played = _date(row['date'], 'date')
    if not fit <= forecast < played:
        raise ValueError('Saved forecast must precede its match')
    rich = _pair(row['rich_rates'], 'rich_rates')
    goal = _pair(row['goal_rates'], 'goal_rates')
    actual = _pair(row['actual_score'], 'actual_score', counts=True)
    raw = np.asarray(row['rich_probabilities'])
    if raw.shape != (3,) or raw.dtype.kind not in 'iuf':
        raise ValueError('Need three saved rich probabilities')
    p = raw.astype(float)
    if (not np.isfinite(p).all() or np.any(p <= 0) or np.any(p >= 1)
            or abs(p.sum()-1) > 1e-10):
        raise ValueError('Saved probabilities must already be positive and sum to one')
    p00 = math.exp(-float(rich.sum()))
    positive_draw = float(p[1]-p00)
    if not 0 < positive_draw < 1:
        raise ValueError('Saved positive-score draw probability must lie strictly inside (0,1)')
    gap = float(abs(math.log(float(goal[0]))-math.log(float(goal[1]))))
    base_nll = float(np.sum(rich-actual*np.log(rich)+gammaln(actual+1)))
    return {'match_id':mid, 'date':played, 'forecast_before_date':forecast,
            'fit_before_date':fit, 'rich_rates':rich, 'goal_rates':goal,
            'actual_score':actual, 'gap':gap, 'base_positive_draw_probability':positive_draw,
            'positive_draw_indicator':int(actual[0] == actual[1] and actual[0] >= 1),
            'base_exact_score_nll':base_nll}


def _objective(theta, design, log_s, log_not_s, labels):
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (2,) or not np.isfinite(theta).all():
        raise ValueError('Need two finite tilt coefficients')
    eta = design @ theta
    logz = np.logaddexp(log_not_s, log_s+eta)
    conditional_probability = expit(log_s+eta-log_not_s)
    loss = float(np.mean(logz-eta*labels))
    gradient = design.T @ (conditional_probability-labels)/len(labels)
    if not math.isfinite(loss) or not np.isfinite(gradient).all():
        raise ValueError('Tilt objective or analytic gradient is nonfinite')
    return loss, gradient


def objective_and_gradient(theta, design, positive_draw_probabilities, labels):
    """Mean full-score NLL difference and its analytic gradient; no fitting."""
    x = np.asarray(design, dtype=float)
    s = np.asarray(positive_draw_probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    if (s.ndim != 1 or not len(s) or x.shape != (len(s),2) or y.shape != s.shape
            or not np.isfinite(x).all() or not np.isfinite(s).all()
            or np.any(s <= 0) or np.any(s >= 1) or np.any((y != 0) & (y != 1))):
        raise ValueError('Need finite n-by-two design, open-unit probabilities and binary labels')
    return _objective(theta, x, np.log(s), np.log1p(-s), y)


def fit_adjustment(rows, before_date='2025-08-01'):
    """One zero-initialized BFGS fit on explicitly supplied earlier forecasts."""
    cutoff = _date(before_date, 'before_date')
    records = [row_pieces(r) for r in rows]
    if not records:
        raise ValueError('Need nonempty earlier fit rows')
    records.sort(key=lambda r:(r['date'],r['match_id']))
    ids = [r['match_id'] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate training match_id')
    if any(r['date'] >= cutoff for r in records):
        raise ValueError('Training outcomes must be strictly before the fit cutoff')
    design = np.asarray([[1.,r['gap']] for r in records])
    s = np.asarray([r['base_positive_draw_probability'] for r in records])
    y = np.asarray([r['positive_draw_indicator'] for r in records])
    if not 0 < y.sum() < len(y):
        raise ValueError('Need positive-score draws and other outcomes in the fit rows')
    log_s, log_not_s = np.log(s), np.log1p(-s)
    trace = []

    def callback(theta):
        value, gradient = _objective(theta,design,log_s,log_not_s,y)
        trace.append({'iteration':len(trace)+1,'beta':np.asarray(theta).tolist(),
                      'mean_exact_score_nll_difference':value,
                      'gradient_max_abs':float(np.max(np.abs(gradient)))})

    fit = minimize(_objective, np.zeros(2), args=(design,log_s,log_not_s,y),
                   method='BFGS', jac=True, callback=callback, options=dict(OPTIMIZER_OPTIONS))
    if not fit.success:
        raise RuntimeError(f'Conditional positive-draw BFGS failed: status={fit.status}, '
                           f'message={fit.message}, iterations={fit.nit}, trace={trace}')
    final, gradient = _objective(fit.x,design,log_s,log_not_s,y)
    base_nll = float(np.mean([r['base_exact_score_nll'] for r in records]))
    return {'model_type':'conditional_positive_draw_joint_score_tilt',
            'before_date':cutoff.isoformat(), 'beta':np.asarray(fit.x).tolist(),
            'coefficient_names':['intercept','absolute_log_goal_expert_rate_ratio'],
            'feature_definition':'gap=abs(log(goal_rates_home)-log(goal_rates_away)); eta=beta0+beta1*gap',
            'tilted_event':'home_goals=away_goals>=1',
            'n_matches':len(records), 'positive_draw_n':int(y.sum()),
            'training_min_date':records[0]['date'].isoformat(),
            'training_max_date':records[-1]['date'].isoformat(),
            'training_latest_forecast_date':max(r['forecast_before_date'] for r in records).isoformat(),
            'training_match_ids':ids,
            'base_exact_score_nll':base_nll, 'adjusted_exact_score_nll':base_nll+final,
            'mean_exact_score_nll_difference':final,
            'objective':'equal-match mean full-score NLL; exact difference equals conditional binary event offset likelihood',
            'optimizer':{'method':'BFGS','success':bool(fit.success),'status':int(fit.status),
                         'message':str(fit.message),'iterations':int(fit.nit),'nfev':int(fit.nfev),
                         'njev':int(fit.njev),'initial_beta':[0.,0.],
                         'options':dict(OPTIMIZER_OPTIONS),'gradient':gradient.tolist(),'trace':trace},
            'fits':1,'grid_searches':0,'restarts':0,'regularization':None,
            'new_team_fits':0,'new_team_information':False,'base_goal_rates_preserved_as_inputs':True,
            'final_marginal_means_preserved':False,
            'home_vs_away_probability_ratio_preserved':True,
            'conditional_positive_draw_score_ratios_preserved':True}


def predict_adjusted(row, params):
    """Produce all full-time markets from a single conditionally tilted grid."""
    pieces = row_pieces(row)
    cutoff = _date(params['before_date'], 'adjustment before_date')
    if pieces['forecast_before_date'] < cutoff:
        raise ValueError('Adjustment cannot be available after the forecast date')
    beta = np.asarray(params['beta'], dtype=float)
    if beta.shape != (2,) or not np.isfinite(beta).all():
        raise ValueError('Need two finite saved coefficients')
    eta = float(beta @ np.asarray([1.,pieces['gap']]))
    s = pieces['base_positive_draw_probability']
    logz = float(np.logaddexp(math.log1p(-s), math.log(s)+eta))
    base = score_grid(*pieces['rich_rates'],rho=0)
    h,a = np.indices(base.grid.shape)
    positive_draw_mask = (h == a) & (h >= 1)
    grid_s = float(base.grid[positive_draw_mask].sum())
    if abs(grid_s-s) > 1e-9:
        raise ValueError('Saved draw probability and original score grid do not agree')
    with np.errstate(over='raise',invalid='raise',under='ignore'):
        matrix = base.grid * np.exp(eta*positive_draw_mask-logz)
    if (not np.isfinite(matrix).all() or np.any(matrix < 0)
            or abs(float(matrix.sum())-1)>1e-9):
        raise ValueError('Tilted joint score probabilities are invalid; no fallback')
    if np.any((base.grid > 0) & (matrix == 0)):
        raise ValueError('Tilted positive probabilities underflowed; no clipping or fallback')
    matrix = matrix/float(matrix.sum())
    new_home, new_away = float((matrix*h).sum()), float((matrix*a).sum())
    grid = FootballProbabilityGrid(matrix,new_home,new_away,normalize=False)
    forecast = summarize(grid)
    # Preserve the actual-score likelihood even when its score lies outside
    # the reporting grid.  Normalizer uses the saved negligible-tail full mass.
    nll = pieces['base_exact_score_nll']+logz-eta*pieces['positive_draw_indicator']
    if not math.isfinite(nll) or nll < -1e-12:
        raise ValueError('Adjusted full-support exact-score NLL is invalid')
    return {'forecast':forecast,'probabilities':list(forecast['one_x_two'].values()),
            'base_rates':pieces['rich_rates'].tolist(),
            'adjusted_expected_goals':[new_home,new_away],
            'exact_score_nll':float(nll), 'gap':pieces['gap'], 'd':pieces['gap'],
            'eta':eta,'logZ':logz,'positive_draw_probability':float(expit(math.log(s)-math.log1p(-s)+eta)),
            'base_positive_draw_probability':s,'adjustment_before_date':cutoff.isoformat()}
