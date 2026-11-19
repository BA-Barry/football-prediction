"""Independent algebra, temporal and numerical checks for goal allocation."""
from copy import deepcopy
from datetime import date,datetime
import math

import numpy as np
import pytest
from scipy.stats import binom

from football_model.goal_allocation import (
    allocation_objective,fit_allocation_weight,predict_allocation_rates,
)


def rows():
    # q_goal=.2, q_xg=.8; equal h/a gives the known optimum w=.5.
    return [{'match_id':f'm-{i}','date':f'2024-01-{i+1:02d}',
             'goal_rates':[1.,4.],'xg_rates':[4.,1.],'actual_score':[2,2]}
            for i in range(8)]


def test_known_interior_weight_and_full_conditional_nll():
    result=fit_allocation_weight(rows(),date(2024,2,1))
    assert result['weight']==pytest.approx(.5,abs=1e-12)
    assert abs(result['gradient'])<1e-11
    assert result['conditional_nll']==pytest.approx(-binom.logpmf(2,4,.5),abs=1e-12)
    assert result['hessian']>0
    assert result['n']==result['n_informative_matches']==8


def test_endpoint_solutions_not_forced_to_mix():
    data=rows()
    for r in data:r['actual_score']=[0,5]
    left=fit_allocation_weight(data,date(2024,2,1))
    assert left['weight']==0 and left['method']=='convex_left_endpoint'
    for r in data:r['actual_score']=[5,0]
    right=fit_allocation_weight(data,date(2024,2,1))
    assert right['weight']==1 and right['method']=='convex_right_endpoint'


def test_same_share_even_with_different_totals_is_unidentifiable():
    data=rows()
    for r in data:r['xg_rates']=[2.,8.]
    result=fit_allocation_weight(data,date(2024,2,1))
    assert result['not_identifiable'] and result['weight']==0
    assert result['n_informative_matches']==0


def test_zero_zero_scores_have_no_allocation_information():
    data=rows()
    for r in data:r['actual_score']=[0,0]
    result=fit_allocation_weight(data,date(2024,2,1))
    assert result['weight']==0 and result['not_identifiable']
    assert result['objective']==result['gradient']==result['hessian']==0
    assert result['n_scoring_matches']==result['training_total_goals']==0


def test_only_scoreless_matches_with_different_shares_is_also_unidentifiable():
    data=rows()
    for r in data:r['xg_rates']=[2.,8.]
    data[0]['xg_rates']=[4.,1.]
    data[0]['actual_score']=[0,0]
    result=fit_allocation_weight(data,date(2024,2,1))
    assert result['not_identifiable'] and result['n_informative_matches']==0
    assert result['n_scoring_matches']==7 and result['n']==8


def test_independent_binomial_nll_and_finite_difference_derivatives():
    goal=np.array([[.7,1.2],[2.,.8],[1.3,.5]])
    xg=np.array([[1.1,.9],[1.4,1.7],[.8,1.1]])
    y=np.array([[0,2],[3,1],[1,1]])
    w=.31
    def independent(t):
        total=0.
        for g,x,score in zip(goal,xg,y):
            odds=(g[0]/g[1])**(1-t)*(x[0]/x[1])**t
            q=odds/(1+odds)
            h,a=score
            total-=math.log(math.comb(h+a,h))+h*math.log(q)+a*math.log1p(-q)
        return total/len(y)
    value,gradient,hessian=allocation_objective(w,goal,xg,y)
    eps=1e-5
    assert value==pytest.approx(independent(w),abs=1e-12)
    assert gradient==pytest.approx((independent(w+eps)-independent(w-eps))/(2*eps),abs=1e-9)
    assert hessian==pytest.approx((independent(w+eps)-2*independent(w)+independent(w-eps))/eps**2,abs=1e-5)


def test_future_and_cutoff_day_do_not_change_weight_or_metadata():
    data=rows()
    expected=fit_allocation_weight(data,date(2024,2,1))
    data+= [dict(match_id='cutoff',date='2024-02-01',goal_rates=None,xg_rates=None,actual_score=None),
            dict(match_id='future',date='2025-01-01',goal_rates=[999,1],xg_rates=[1,999],actual_score=[99999,0])]
    assert fit_allocation_weight(data,date(2024,2,1))==expected


def test_predictions_preserve_total_and_share_under_total_scaling():
    first=predict_allocation_rates(3.1,[.7,1.8],[1.6,.9],.42)
    second=predict_allocation_rates(31.,[.7,1.8],[1.6,.9],.42)
    assert sum(first)==pytest.approx(3.1,abs=1e-15)
    assert sum(second)==pytest.approx(31.,abs=1e-14)
    assert np.asarray(second)==pytest.approx(np.asarray(first)*10,rel=1e-14)


def test_endpoints_recover_each_share_while_respecting_supplied_total():
    assert predict_allocation_rates(10,[1,4],[4,1],0)==pytest.approx((2,8),abs=1e-14)
    assert predict_allocation_rates(10,[1,4],[4,1],1)==pytest.approx((8,2),abs=1e-14)


def test_swapping_home_away_swaps_rates_and_preserves_fitted_weight():
    data=rows()
    for i,r in enumerate(data):r['actual_score']=[i%4,(i+2)%3]
    swapped=deepcopy(data)
    for r in swapped:
        for key in ('goal_rates','xg_rates','actual_score'):r[key]=r[key][::-1]
    first=fit_allocation_weight(data,date(2024,2,1))
    second=fit_allocation_weight(swapped,date(2024,2,1))
    assert first['weight']==pytest.approx(second['weight'],abs=1e-13)
    assert predict_allocation_rates(2.7,[1.2,.9],[.7,1.5],.4)[::-1]==pytest.approx(
        predict_allocation_rates(2.7,[.9,1.2],[1.5,.7],.4),abs=1e-14)


def test_extreme_logits_are_stable_and_representable_small_rate_is_kept():
    value,gradient,hessian=allocation_objective(.5,[[1e-200,1e200]],[[1e200,1e-200]],[[2,2]])
    assert value==pytest.approx(-binom.logpmf(2,4,.5),abs=1e-12)
    assert gradient==pytest.approx(0,abs=1e-12)
    assert math.isfinite(hessian) and hessian>0
    # q_small~1e-400 cannot itself be a float, but T*q_small~1e-100 can.
    home,away=predict_allocation_rates(1e300,[1e200,1e-200],[1e200,1e-200],.5)
    assert home==1e300 and away==pytest.approx(1e-100,rel=1e-12)
    with pytest.raises(ValueError,match='representability'):
        predict_allocation_rates(1.,[1e200,1e-200],[1e200,1e-200],.5)


@pytest.mark.parametrize('field,bad',[
    ('goal_rates',[True,1]),('xg_rates',[1,False]),('actual_score',[1,True]),
    ('goal_rates',[0,1]),('xg_rates',[-1,1]),('actual_score',[-1,1]),
    ('goal_rates',[np.nan,1]),('xg_rates',[1,np.inf]),('actual_score',[1,np.nan]),
    ('goal_rates',[1,2,3]),('xg_rates',[[1,2]]),('actual_score',[1]),
    ('goal_rates',None),('xg_rates',['1','2']),('actual_score',[1,.5]),
])
def test_invalid_training_fields_are_rejected(field,bad):
    data=rows()
    data[0][field]=bad
    with pytest.raises(ValueError):fit_allocation_weight(data,date(2024,2,1))


@pytest.mark.parametrize('total',[True,False,0,-1,np.nan,np.inf,None,[2]])
def test_invalid_prediction_totals_rejected(total):
    with pytest.raises(ValueError):predict_allocation_rates(total,[1,2],[2,1],.5)


@pytest.mark.parametrize('weight',[True,-.01,1.01,np.nan,np.inf,None,[.5]])
def test_invalid_prediction_weights_rejected(weight):
    with pytest.raises(ValueError):predict_allocation_rates(3,[1,2],[2,1],weight)


@pytest.mark.parametrize('field',range(3))
def test_public_objective_rejects_mixed_boolean_arrays(field):
    arguments=[[[1,2]],[[2,1]],[[1,2]]]
    arguments[field][0][0]=True
    with pytest.raises(ValueError,match='Boolean'):
        allocation_objective(.5,*arguments)


def test_duplicate_invalid_ids_empty_or_datetime_cutoff_rejected():
    data=rows()
    with pytest.raises(ValueError,match='unique'):fit_allocation_weight(data+[data[0]],date(2024,2,1))
    data[1]['date']=data[0]['date']
    data[1]['match_id']=None
    with pytest.raises(ValueError,match='unique'):fit_allocation_weight(data,date(2024,2,1))
    with pytest.raises(ValueError,match='earlier'):fit_allocation_weight([],date(2024,2,1))
    with pytest.raises(ValueError,match='calendar'):fit_allocation_weight(rows(),datetime(2024,2,1))


def test_prediction_signature_cannot_accept_actual_total_or_score():
    with pytest.raises(TypeError):
        predict_allocation_rates(3,[1,2],[2,1],.5,actual_score=[9,9])


def test_conditional_loss_plus_fixed_total_loss_is_full_score_loss():
    predicted_total=2.7
    goal,xg,w=[1.2,.9],[.7,1.5],.4
    observed=[1,2]
    rates=predict_allocation_rates(predicted_total,goal,xg,w)
    full=sum(rate-count*math.log(rate)+math.lgamma(count+1)
             for rate,count in zip(rates,observed))
    n=sum(observed)
    total_nll=predicted_total-n*math.log(predicted_total)+math.lgamma(n+1)
    conditional_nll=allocation_objective(w,[goal],[xg],[observed])[0]
    assert total_nll+conditional_nll==pytest.approx(full,abs=1e-13)
