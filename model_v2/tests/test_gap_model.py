from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from football_model.gap_model import GAPState, STATISTICS, _sequential_mae, optimize_gap_updates
import football_model.gap_model as gap_model


def parameters(rate=.5, ph=.7, pa=.4):
    return {s:{'lambda':rate,'phi_home':ph,'phi_away':pa} for s in STATISTICS}


def game(h='A',a='B',score=(4,2),day='2022-08-05',season='2022/23'):
    return {'match_id':f'{season}:{day}:{h}|{a}','competition':'E0','season':season,
            'home':h,'away':a,'date':day,'actual_score':list(score),
            'counts':{s:list(score) for s in STATISTICS if s!='goals'}}


def test_eight_role_updates_share_one_pre_match_residual():
    model=GAPState(parameters())
    row=game()
    model.predict(row)
    model.states['E0']['A'][:]=[2,3,4,5]
    model.states['E0']['B'][:]=[6,7,8,9]
    before=model.update(row)
    assert before['expected_statistics']['goals']==[5.5,5.5]
    # Residuals are -1.5 and -3.5 for every statistic; no later update can
    # recompute one of those residuals from already-modified attack/defence.
    assert model.states['E0']['A'][0]==pytest.approx([1.475,1.775,3.775,4.475])
    assert model.states['E0']['B'][0]==pytest.approx([4.95,6.55,7.3,8.7])
    assert len(model.states['E0'])==2
    assert model.team_observations['E0']=={'A':1,'B':1}


def test_away_defensive_rating_is_not_written_to_attack():
    model=GAPState(parameters())
    model.update(game(score=(4,0)))
    assert model.states['E0']['B'][0]==pytest.approx([0,1.2,0,.8])
    assert model.states['E0']['A'][0]==pytest.approx([1.4,0,.6,0])


def test_predict_ignores_same_match_counts_and_outcomes():
    model=GAPState(parameters())
    model.update(game())
    target=game(day='2022-08-12')
    p=model.predict(target)
    future=deepcopy(target)
    future['actual_score']=None
    future['counts']={'corners':None}
    assert model.predict(future)==p


def test_new_season_initializes_from_departed_group_without_new_results():
    rosters={'E0':{'2022/23':['A','B'],'2023/24':['A','C']}}
    model=GAPState(parameters(),rosters)
    model.update(game())
    old_b=model.states['E0']['B'].copy()
    old_a=model.states['E0']['A'].copy()
    p=model.predict(game(h='A',a='C',day='2023-08-05',season='2023/24'))
    assert model.states['E0']['C']==pytest.approx(old_b)
    assert model.states['E0']['A']==pytest.approx(old_a)
    assert p['team_observations']=={'home':1,'away':0}
    assert p['initialization']['away']['donor_team_ids']==['B']
    assert p['initialization']['away']['method']=='previous_season_departed_group_mean'


def test_unknown_observation_refused_without_zero_filling_or_state_mutation():
    model=GAPState(parameters())
    bad=game()
    bad['counts']['corners']=None
    with pytest.raises(ValueError,match='no imputation'):
        model.update(bad)
    assert model.states=={}
    bad['counts']=None
    with pytest.raises(ValueError,match='no imputation'):
        model.update(bad)
    assert model.states=={}
    with pytest.raises(ValueError,match='nonnegative integer'):
        model.update(game(score=(True,2)))


def test_old_date_update_is_refused_and_clone_is_independent():
    model=GAPState(parameters())
    model.update(game(day='2022-08-12'))
    cloned=model.clone()
    cloned.update(game(day='2022-08-19'))
    assert model.team_observations['E0']['A']==1
    assert cloned.team_observations['E0']['A']==2
    with pytest.raises(ValueError,match='calendar-date order'):
        model.update(game(day='2022-08-05'))


def test_fast_learning_objective_matches_chronological_core():
    rows=[game(),game(h='B',a='C',score=(1,3),day='2022-08-12'),
          game(h='A',a='C',score=(0,2),day='2022-08-19')]
    model=GAPState(parameters())
    errors=[]
    for row in rows:
        expected=model.update(row)['expected_statistics']['goals']
        errors.extend(abs(y-p) for y,p in zip(row['actual_score'],expected))
    fast=_sequential_mae(np.array([0,1,0]),np.array([1,2,2]),
                         np.array([[4.,2.],[1.,3.],[0.,2.]]),3,.5,.7,.4)
    assert fast==pytest.approx(np.mean(errors),abs=1e-14)


def test_parameter_training_refuses_later_season():
    with pytest.raises(ValueError,match='only.*2022/23'):
        optimize_gap_updates([game(day='2023-08-05',season='2023/24')])


def test_optimizer_records_all_calls_and_does_not_hide_nonconvergence(monkeypatch):
    def stopped_optimizer(objective,start,method):
        assert method=='Nelder-Mead'
        objective(start)
        value=objective(start)
        return SimpleNamespace(x=start,fun=value,nfev=2,nit=1,success=False,
                               message='test numerical stopping')
    monkeypatch.setattr(gap_model,'minimize',stopped_optimizer)
    result=optimize_gap_updates([game()])
    assert result['all_converged'] is False
    for report in result['optimization'].values():
        assert report['success'] is False
        assert report['objective_evaluations']==3
        assert report['optimizer_objective_evaluations']==2
        assert report['initial_mae_evaluations']==1
