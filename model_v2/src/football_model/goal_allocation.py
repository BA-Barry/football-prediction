"""One shared home/away allocation weight with the total goal rate fixed.

The training total h+a belongs only to the conditional Binomial outcome. The
prediction function accepts a predicted total rate, never an observed score.
Rate provenance and common information dates remain the caller's responsibility.
"""
from datetime import date
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, gammaln, log_expit


def _numeric(value, label):
    raw=np.asarray(value,dtype=object)
    native=np.asarray(value)
    if (any(isinstance(v,(bool,np.bool_)) for v in raw.flat)
            or native.dtype.kind not in 'iuf'):
        raise ValueError(f'Need numeric values, not Boolean flags: {label}')
    values=native.astype(float)
    if not np.isfinite(values).all():
        raise ValueError(f'Need finite values: {label}')
    return values


def _pair(value,label,*,counts=False):
    values=_numeric(value,label)
    if values.shape!=(2,):
        raise ValueError(f'Need two numeric values: {label}')
    if counts:
        if np.any(values<0) or np.any(values!=np.floor(values)):
            raise ValueError(f'Need nonnegative integer counts: {label}')
    elif np.any(values<=0):
        raise ValueError(f'Need positive rates: {label}')
    return values


def _weight(value):
    raw=_numeric(value,'weight')
    if raw.shape!=() or not 0<=float(raw)<=1:
        raise ValueError('Need one weight in [0, 1]')
    return float(raw)


def _logits(goal,xg):
    def log_ratio(rates):
        # Direct ratios preserve exactly equal representable proportions after
        # total scaling; log subtraction handles ratios outside float range.
        with np.errstate(over='ignore',under='ignore',divide='ignore'):
            ratio=rates[...,0]/rates[...,1]
            direct=np.log(ratio)
        fallback=np.log(rates[...,0])-np.log(rates[...,1])
        return np.where(np.isfinite(direct),direct,fallback)
    return log_ratio(goal),log_ratio(xg)


def predict_allocation_rates(total_rate,goal_rates,xg_rates,weight):
    """Keep T fixed and interpolate only the conditional scoring allocation.

    Small-tail rates use log-space multiplication, then the larger side is
    obtained from T minus the smaller side. No probability clipping is used.
    A strictly positive rate that cannot be represented as a float is reported
    rather than silently replaced with zero.
    """
    raw=_numeric(total_rate,'predicted total rate')
    if raw.shape!=() or float(raw)<=0:
        raise ValueError('Need one positive predicted total rate')
    total=float(raw)
    goal,xg,w=_pair(goal_rates,'goal_rates'),_pair(xg_rates,'xg_rates'),_weight(weight)
    lg,lx=_logits(goal,xg)
    eta=(1-w)*lg+w*lx
    small=math.exp(math.log(total)+float(log_expit(-abs(eta))))
    large=total-small
    if not (math.isfinite(small) and math.isfinite(large) and small>0 and large>0):
        raise ValueError('Positive allocation rates are outside floating-point representability')
    return (large,small) if eta>=0 else (small,large)


def allocation_objective(weight,goal_rates,xg_rates,actual_scores):
    """Mean conditional Binomial NLL, gradient and Hessian per fixture.

    h,a and their sum n are training outcomes only. All goal/xG means are
    already out-of-time predictions. The allocation parameter affects neither
    the supplied predicted total rate nor its count distribution.
    """
    w=_weight(weight)
    goal,xg,y=(_numeric(value,label) for value,label in (
        (goal_rates,'goal_rates'),(xg_rates,'xg_rates'),(actual_scores,'actual_scores')))
    if (goal.ndim!=2 or goal.shape[1:]!=(2,) or not len(goal)
            or xg.shape!=goal.shape or y.shape!=goal.shape
            or np.any(goal<=0) or np.any(xg<=0)
            or np.any(y<0) or np.any(y!=np.floor(y))):
        raise ValueError('Need positive n by 2 rates and aligned nonnegative integer scores')
    lg,lx=_logits(goal,xg)
    delta=lx-lg
    eta=(1-w)*lg+w*lx
    home,away=y[:,0],y[:,1]
    n=home+away
    if not np.isfinite(n).all():
        raise ValueError('Total goal count is outside floating-point representability')
    log_choose=gammaln(n+1)-gammaln(home+1)-gammaln(away+1)
    losses=-home*log_expit(eta)-away*log_expit(-eta)-log_choose
    q,other=expit(eta),expit(-eta)
    # Algebraically n*q-home, but avoids cancellation in an extreme tail.
    gradient=np.mean(delta*(away*q-home*other))
    hessian=np.mean(n*q*other*delta*delta)
    values=tuple(map(float,(np.mean(losses),gradient,hessian)))
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Non-finite conditional allocation objective')
    return values


def fit_allocation_weight(rows,before_date):
    """Estimate one bounded convex weight using strict earlier dates only."""
    if type(before_date) is not date:
        raise ValueError('Need a calendar cutoff')
    history=[]
    for row in rows:
        if date.fromisoformat(row['date'])<before_date:
            history.append(row)
    if not history:
        raise ValueError('Need earlier allocation observations')
    ids=[r['match_id'] for r in history]
    if any(not isinstance(key,str) or not key for key in ids) or len(set(ids))!=len(ids):
        raise ValueError('Need unique nonempty training match IDs')
    history.sort(key=lambda r:(r['date'],r['match_id']))
    ids=[r['match_id'] for r in history]
    goal=np.asarray([_pair(r['goal_rates'],'goal_rates') for r in history])
    xg=np.asarray([_pair(r['xg_rates'],'xg_rates') for r in history])
    scores=np.asarray([_pair(r['actual_score'],'actual_score',counts=True) for r in history])
    lg,lx=_logits(goal,xg)
    totals=np.sum(scores,axis=1)
    informative=(totals>0)&(lg!=lx)
    flat=not bool(np.any(informative))
    evaluations=0
    def evaluate(w):
        nonlocal evaluations
        evaluations+=1
        return allocation_objective(w,goal,xg,scores)
    left,right=evaluate(0.),evaluate(1.)
    if flat:
        weight,method=0.,'no_allocation_information_choose_goal'
    elif left[1]>=0:
        weight,method=0.,'convex_left_endpoint'
    elif right[1]<=0:
        weight,method=1.,'convex_right_endpoint'
    else:
        weight=float(brentq(lambda w:evaluate(w)[1],0.,1.,xtol=1e-14,rtol=1e-14))
        method='brentq_convex_gradient_root'
    value,gradient,hessian=evaluate(weight)
    return {
        'weight':weight,'objective':value,'conditional_nll':value,'gradient':gradient,
        'hessian':hessian,'curvature':hessian,'endpoint_objectives':[left[0],right[0]],
        'endpoint_gradients':[left[1],right[1]],'endpoint_hessians':[left[2],right[2]],
        'not_identifiable':flat,'n':len(history),'n_informative_matches':int(np.count_nonzero(informative)),
        'n_scoring_matches':int(np.count_nonzero(totals)), 'training_total_goals':int(np.sum(totals)),
        'before_date':before_date.isoformat(),'training_min_date':history[0]['date'],
        'training_max_date':history[-1]['date'],'reference_match_ids':ids,
        'optimizer_function_evals':evaluations,'method':method,
        'constraint':'one weight in [0,1], shared across leagues and scoring sides',
        'objective_definition':'uniform mean per fixture conditional Binomial NLL including log-choose',
        'new_intercept':False,'new_decay':False,'new_regularization':False,
        'prediction_total':'supplied statistical total only; observed total is not a prediction input',
    }
