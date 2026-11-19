"""Numerical allocation and time-isolation checks, not forecasting efficacy."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math

import joblib
import numpy as np
import pytest

from football_model.learned_pool import LearnedLinearPool, _objective


START = datetime(2026, 10, 1, tzinfo=timezone.utc)
BEFORE = START+timedelta(days=10)


def row(i=0, *, stat=(.8,.1,.1), llm=(.2,.4,.4), actual=0):
    kickoff = START+timedelta(days=i+1)
    return {'match_id': f'match-{i}', 'kickoff': kickoff.isoformat(),
        'statistical_generated_at': (kickoff-timedelta(hours=8)).isoformat(),
        'llm_generated_at': (kickoff-timedelta(hours=7)).isoformat(),
        'result_observed_at': (kickoff+timedelta(hours=3)).isoformat(),
        'statistical_probabilities': list(stat), 'llm_probabilities': list(llm),
        'actual_outcome': actual, 'statistical_family':'synthetic-stat', 'llm_family':'unknown'}


def fitted_interior():
    # Half home and half draw; target probabilities are .2+.6w and .4-.3w.
    # Their product peaks at w=.5, where both routes contribute to the optimum.
    return LearnedLinearPool().fit([row(0), row(1, actual=1)], BEFORE)


def target(model, **changes):
    kwargs={'match_id':'future-match', 'kickoff':BEFORE+timedelta(days=2),
            'as_of':BEFORE+timedelta(hours=1)} | changes
    return model.predict([.6,.25,.15], [.3,.35,.35], **kwargs)


def test_known_interior_optimum_and_complete_parameter_provenance():
    model=fitted_interior()
    p=model.parameters()
    assert p['weight']==pytest.approx(.5, abs=1e-12)
    assert p['weights']==pytest.approx({'statistical':.5,'llm':.5}, abs=1e-12)
    assert abs(p['gradient_at_solution'])<1e-12
    assert p['identified'] and p['identified_weight_interval']==pytest.approx([.5,.5])
    assert p['method']=='brentq_convex_gradient_root'
    assert p['training_log_loss']==pytest.approx(-math.log(.5*.25)/2, abs=1e-12)
    assert p['training_log_loss']<min(p['endpoint_log_loss'].values())
    assert p['n_matches']==2 and p['reference_match_ids']==['match-0','match-1']
    assert p['weight_source']=='earlier_prospective_log_loss_fit'
    assert p['fit_before']==BEFORE.isoformat()
    assert p['statistical_family']=={'synthetic-stat':2}
    assert p['llm_family']=={'unknown':2}
    assert p['training_results_max_observed_at']==row(1)['result_observed_at']
    assert p['training_kickoff_max']==row(1)['kickoff']
    assert json.loads(json.dumps(p, allow_nan=False))==p


@pytest.mark.parametrize(('actual','expected'), [(0,1.), (1,0.)])
def test_closed_endpoints_are_retained(actual,expected):
    model=LearnedLinearPool().fit([row(actual=actual)], BEFORE)
    assert model.weight==expected
    assert target(model)==([.6,.25,.15] if expected else [.3,.35,.35])


def test_swapping_route_labels_complements_the_allocation():
    rows=[row(0),row(1,actual=1),row(2,actual=0)]
    a=LearnedLinearPool().fit(rows,BEFORE)
    swapped=deepcopy(rows)
    for r in swapped:
        r['statistical_probabilities'],r['llm_probabilities']=r['llm_probabilities'],r['statistical_probabilities']
    b=LearnedLinearPool().fit(swapped,BEFORE)
    assert a.weight+b.weight==pytest.approx(1,abs=1e-12)


@pytest.mark.parametrize('different_unobserved', [False, True])
def test_flat_empirical_loss_is_unidentified_not_fabricated_half_weight(different_unobserved):
    r=row(stat=(.5,.25,.25),llm=(.5,.3,.2) if different_unobserved else (.5,.25,.25))
    model=LearnedLinearPool().fit([r],BEFORE)
    p=model.parameters()
    assert not p['identified'] and p['weight'] is None and p['weights'] is None
    assert p['identified_weight_interval']==[0.,1.]
    assert p['gradient_at_solution']==p['curvature_at_solution']==0
    with pytest.raises(ValueError,match='does not identify'):
        target(model)


def test_zero_endpoint_mass_uses_finite_interior_without_clipping():
    rows=[row(0,stat=(1.,0.,0.),llm=(0.,1.,0.)),
          row(1,stat=(1.,0.,0.),llm=(0.,1.,0.),actual=1)]
    model=LearnedLinearPool().fit(rows,BEFORE)
    p=model.parameters()
    assert model.weight==pytest.approx(.5,abs=1e-12)
    assert p['endpoint_log_loss']=={'llm':None,'statistical':None}
    assert p['endpoint_log_loss_is_finite']=={'llm':False,'statistical':False}
    assert p['training_log_loss']==pytest.approx(math.log(2),abs=1e-12)
    json.dumps(p,allow_nan=False)


def test_both_routes_zero_realised_class_is_reported_not_smoothed():
    with pytest.raises(ValueError,match='zero to a realised outcome'):
        LearnedLinearPool().fit([row(stat=(.5,.5,0),llm=(.4,.6,0),actual=2)],BEFORE)


def test_objective_derivatives_against_independent_finite_differences():
    a=np.array([.7,.15,.3,.6]);b=np.array([.2,.4,.5,.35]);w=.37
    value,g,h=_objective(w,a,b)
    nll=lambda x:sum(-math.log(x*aa+(1-x)*bb) for aa,bb in zip(a,b))/len(a)
    eps=1e-5
    assert value==pytest.approx(nll(w),abs=1e-14)
    assert g==pytest.approx((nll(w+eps)-nll(w-eps))/(2*eps),abs=1e-9)
    assert h==pytest.approx((nll(w+eps)-2*nll(w)+nll(w-eps))/eps**2,abs=1e-5)
    assert h>0


def test_unavailable_future_and_cutoff_results_are_not_read_or_used():
    data=[row(),row(1,actual=1)]
    expected=LearnedLinearPool().fit(data,BEFORE).parameters()
    poison={'result_observed_at':BEFORE.isoformat(),'match_id':None,
            'actual_outcome':object(),'statistical_probabilities':None,'kickoff':'not read'}
    future=poison|{'result_observed_at':(BEFORE+timedelta(days=1)).isoformat()}
    pending={'result_observed_at':None,'actual_outcome':object()}
    assert LearnedLinearPool().fit(data+[poison,future,pending],BEFORE).parameters()==expected


def test_delayed_result_observation_excludes_already_played_fixture():
    early=row()
    delayed=row(1,actual=1)
    delayed['result_observed_at']=(BEFORE+timedelta(hours=1)).isoformat()
    p=LearnedLinearPool().fit([early,delayed],BEFORE).parameters()
    assert p['n_matches']==1 and p['reference_match_ids']==['match-0']
    assert p['weight']==1


def test_duplicate_match_versions_are_rejected_even_with_different_probabilities():
    r=row();other=row(1,actual=1);other['match_id']=r['match_id']
    with pytest.raises(ValueError,match='Duplicate match_id'):
        LearnedLinearPool().fit([r,other],BEFORE)


@pytest.mark.parametrize('key', ['statistical_generated_at','llm_generated_at'])
@pytest.mark.parametrize('offset', [0,1])
def test_late_source_prediction_cannot_enter_weight_fit(key,offset):
    r=row();r[key]=(datetime.fromisoformat(r['kickoff'])+timedelta(seconds=offset)).isoformat()
    with pytest.raises(ValueError,match='forecasts must precede'):
        LearnedLinearPool().fit([r],BEFORE)


@pytest.mark.parametrize('offset', [0,-1])
def test_result_cannot_be_observed_at_or_before_kickoff(offset):
    r=row();r['result_observed_at']=(datetime.fromisoformat(r['kickoff'])+timedelta(seconds=offset)).isoformat()
    with pytest.raises(ValueError,match='observed result must follow'):
        LearnedLinearPool().fit([r],BEFORE)


@pytest.mark.parametrize('value', [[.5,.2,.2], [True,0,0], [math.nan,.5,.5], [1.1,-.1,0], [.5,.5], 'bad'])
def test_malformed_probabilities_are_not_normalised_or_coerced(value):
    r=row();r['statistical_probabilities']=value
    with pytest.raises(ValueError):LearnedLinearPool().fit([r],BEFORE)


@pytest.mark.parametrize('value', [True,3,-1,.5,'H'])
def test_invalid_labels_are_rejected(value):
    r=row(actual=value)
    with pytest.raises(ValueError,match='actual_outcome'):
        LearnedLinearPool().fit([r],BEFORE)


def test_string_outcomes_and_unknown_identity_are_recorded_without_guessing():
    r=row(actual='home');r.pop('llm_family');r.pop('statistical_family')
    p=LearnedLinearPool().fit([r],BEFORE).parameters()
    assert p['statistical_family']==p['llm_family']=={'unknown':1}
    assert p['training_llm_track']=='unknown'
    assert p['training_llm_track_counts']=={'unknown':1}


@pytest.mark.parametrize('track', ['llm_independent','llm_model_informed','unknown'])
def test_declared_llm_route_is_recorded_without_changing_loss_or_weight(track):
    data=[row(),row(1,actual=1)]
    for r in data:r['llm_track']=track
    p=LearnedLinearPool().fit(data,BEFORE).parameters()
    assert p['training_llm_track']==track
    assert p['training_llm_track_counts']=={track:2}
    assert p['weight']==pytest.approx(.5,abs=1e-12)


def test_mixed_llm_routes_are_identified_as_mixed_research_not_one_exposure_line():
    data=[row()|{'llm_track':'llm_independent'},
          row(1,actual=1)|{'llm_track':'llm_model_informed'}]
    p=LearnedLinearPool().fit(data,BEFORE).parameters()
    assert p['training_llm_track']=='mixed'
    assert p['training_llm_track_counts']=={'llm_independent':1,'llm_model_informed':1}


@pytest.mark.parametrize('track', ['independent','llm_market',None,False])
def test_invalid_llm_route_declarations_are_rejected(track):
    with pytest.raises(ValueError,match='llm_track'):
        LearnedLinearPool().fit([row()|{'llm_track':track}],BEFORE)


def test_prediction_temporal_isolation_and_training_fixture_rejection():
    model=fitted_interior()
    with pytest.raises(ValueError,match='follow weight fitting'):
        target(model,as_of=BEFORE-timedelta(microseconds=1))
    with pytest.raises(ValueError,match='precede kickoff'):
        target(model,as_of=BEFORE+timedelta(days=2))
    with pytest.raises(ValueError,match='own training fixture'):
        target(model,match_id='match-0')
    assert target(model,as_of=BEFORE)==pytest.approx([.45,.30,.25],abs=1e-12)


def test_save_replay_parameter_copy_and_probability_sum(tmp_path):
    model=fitted_interior(); expected=target(model)
    assert sum(expected)==pytest.approx(1,abs=1e-15)
    assert all(0<=p<=1 for p in expected)
    copied=model.parameters();copied['weights']['statistical']=0;copied['reference_match_ids'].clear()
    assert model.parameters()['weights']['statistical']==pytest.approx(.5)
    assert model.parameters()['reference_match_ids']
    path=tmp_path/'weight.joblib';joblib.dump(model,path)
    replay=joblib.load(path)
    assert replay.parameters()==model.parameters()
    assert target(replay)==expected


@pytest.mark.parametrize('method', ['predict','parameters'])
def test_unfitted_use_is_rejected(method):
    model=LearnedLinearPool()
    with pytest.raises(ValueError,match='Fit|fitted'):
        target(model) if method=='predict' else model.parameters()


def test_no_earlier_settled_pairs_and_naive_times_are_not_valid_fit_inputs():
    with pytest.raises(ValueError,match='earlier settled'):
        LearnedLinearPool().fit([{'result_observed_at':None}],BEFORE)
    with pytest.raises(ValueError,match='timezone-aware'):
        LearnedLinearPool().fit([row()],BEFORE.replace(tzinfo=None))
