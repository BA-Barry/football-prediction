"""Independent numerical and temporal checks of the one-weight blend."""
from copy import deepcopy
from datetime import date,datetime
import math

import numpy as np
import pytest

from football_model.goal_xg_blend import (
    predict_goal_xg_rates,fit_goal_xg_weight,goal_xg_objective,
)


def rows():
    return [{'match_id':f'm-{i}','date':f'2024-01-{i+1:02d}',
             'goal_rates':[1.,1.],'xg_rates':[4.,4.],'actual_score':[2,2]}
            for i in range(8)]


def test_known_interior_weight_half_is_recovered():
    result=fit_goal_xg_weight(rows(),date(2024,2,1))
    assert result['weight']==pytest.approx(.5,abs=1e-12)
    assert abs(result['gradient'])<1e-11
    assert result['method']=='brentq_convex_gradient_root'
    expected=2*(2-2*math.log(2)+math.lgamma(3))
    assert result['full_nll']==pytest.approx(expected,abs=1e-12)
    assert result['n']==8


def test_both_boundary_solutions_are_available_not_forced_blending():
    low=rows()
    for r in low:r['actual_score']=[0,0]
    left=fit_goal_xg_weight(low,date(2024,2,1))
    assert left['weight']==0 and left['method']=='convex_left_endpoint'
    high=rows()
    for r in high:r['actual_score']=[8,8]
    right=fit_goal_xg_weight(high,date(2024,2,1))
    assert right['weight']==1 and right['method']=='convex_right_endpoint'


def test_identical_rates_choose_original_and_report_unidentifiable():
    data=rows()
    for r in data:r['xg_rates']=list(r['goal_rates'])
    result=fit_goal_xg_weight(data,date(2024,2,1))
    assert result['weight']==0 and result['not_identifiable']
    assert result['gradient']==result['curvature']==0


def test_nll_gradient_and_curvature_match_independent_finite_difference():
    a=np.array([[.8,1.7],[2.,.4],[1.3,1.1]])
    b=np.array([[1.6,.9],[2.7,.7],[.6,1.5]])
    y=np.array([[1,0],[3,1],[0,2]])
    w=.37
    def nll(weight):
        total=0.
        for aa,bb,yy in zip(a,b,y):
            for xg,xxg,goal in zip(aa,bb,yy):
                rate=xg**(1-weight)*xxg**weight
                total+=rate-goal*math.log(rate)+math.lgamma(goal+1)
        return total/len(y)
    value,gradient,curvature=goal_xg_objective(w,a,b,y)
    eps=1e-5
    assert value==pytest.approx(nll(w),abs=1e-12)
    assert gradient==pytest.approx((nll(w+eps)-nll(w-eps))/(2*eps),abs=1e-9)
    assert curvature==pytest.approx((nll(w+eps)-2*nll(w)+nll(w-eps))/eps**2,abs=1e-5)
    assert curvature>0


def test_future_and_cutoff_day_outcomes_do_not_change_fit():
    original=rows()
    expected=fit_goal_xg_weight(original,date(2024,2,1))
    modified=deepcopy(original)
    modified.extend([{'match_id':'same','date':'2024-02-01','goal_rates':[999,999],
                     'xg_rates':[1,1],'actual_score':[10000,10000]},
                     {'match_id':'future','date':'2025-01-01','goal_rates':None,
                      'xg_rates':None,'actual_score':None}])
    assert fit_goal_xg_weight(modified,date(2024,2,1))==expected
    assert expected['training_max_date']=='2024-01-08'


def test_swapping_sides_does_not_change_weight_and_swaps_prediction():
    data=rows()
    for i,r in enumerate(data):
        r['goal_rates']=[.9+i*.1,1.8-i*.1]
        r['xg_rates']=[1.7-i*.1,.8+i*.1]
        r['actual_score']=[i%3,(i+1)%3]
    swapped=deepcopy(data)
    for r in swapped:
        for k in ('goal_rates','xg_rates','actual_score'):r[k]=r[k][::-1]
    first=fit_goal_xg_weight(data,date(2024,2,1))
    second=fit_goal_xg_weight(swapped,date(2024,2,1))
    assert first['weight']==pytest.approx(second['weight'],abs=1e-13)
    assert predict_goal_xg_rates([.9,1.8],[1.7,.8],.3)[::-1]==pytest.approx(
        predict_goal_xg_rates([1.8,.9],[.8,1.7],.3),abs=1e-14)


def test_prediction_preserves_exact_endpoints_and_geometric_midpoint():
    a=[.123456789123456,2.34567891234567]
    b=[4.56789123456789,.76543219876543]
    assert predict_goal_xg_rates(a,b,0)==tuple(a)
    assert predict_goal_xg_rates(a,b,1)==tuple(b)
    assert predict_goal_xg_rates(a,b,.5)==pytest.approx(tuple(math.sqrt(x*y) for x,y in zip(a,b)),abs=1e-14)


@pytest.mark.parametrize('key,bad',[
    ('goal_rates',[0,1]),('xg_rates',[-1,2]),('actual_score',[-1,2]),
    ('goal_rates',[True,1]),('xg_rates',[1,False]),('actual_score',[1,True]),
    ('goal_rates',[np.nan,1]),('xg_rates',[1,np.inf]),('actual_score',[1,np.nan]),
    ('goal_rates',[1,2,3]),('xg_rates',[[1,2]]),('actual_score',[1]),
    ('goal_rates',None),('xg_rates',['1','2']),('actual_score',[1,.5]),
])
def test_invalid_per_match_measurements_cannot_be_used(key,bad):
    data=rows()
    data[0][key]=bad
    with pytest.raises(ValueError):fit_goal_xg_weight(data,date(2024,2,1))


@pytest.mark.parametrize('weight',[True,False,-.1,1.1,float('nan'),float('inf'),'0.5',None])
def test_invalid_prediction_weight_rejected(weight):
    with pytest.raises(ValueError):predict_goal_xg_rates([1,1],[2,2],weight)


def test_duplicate_missing_and_empty_training_are_rejected():
    data=rows()
    with pytest.raises(ValueError,match='unique'):fit_goal_xg_weight(data+[data[0]],date(2024,2,1))
    data[0]['match_id']=''
    with pytest.raises(ValueError,match='unique'):fit_goal_xg_weight(data,date(2024,2,1))
    with pytest.raises(ValueError,match='earlier'):fit_goal_xg_weight([],date(2024,2,1))
    with pytest.raises(ValueError,match='calendar'):fit_goal_xg_weight(rows(),datetime(2024,2,1))


def test_extreme_positive_rate_interpolation_avoids_ratio_overflow():
    result=predict_goal_xg_rates([1e-200,1e200],[1e200,1e-200],.5)
    assert result==pytest.approx((1,1),abs=1e-12)


@pytest.mark.parametrize('field',range(3))
def test_public_objective_also_rejects_mixed_boolean_arrays(field):
    arguments=[[[1.,2.]],[[2.,3.]],[[1,2]]]
    arguments[field][0][0]=True
    with pytest.raises(ValueError,match='Boolean'):
        goal_xg_objective(.5,*arguments)


def test_invalid_id_same_day_cannot_fail_by_sorting_mixed_python_types():
    data=rows()
    data[1]['date']=data[0]['date']
    data[1]['match_id']=None
    with pytest.raises(ValueError,match='unique'):
        fit_goal_xg_weight(data,date(2024,2,1))
