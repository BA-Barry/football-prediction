from copy import deepcopy

import pytest

from football_model.evidence_reconciliation import reconcile_evidence


def inputs():
    fixture = {'match_id':'test:a:b','home':'A','away':'B','competition':'test',
        'kickoff':'2026-11-01T15:00:00+00:00','neutral':False}
    record = {'fixture':fixture,'as_of':'2026-10-24T12:00:00+00:00','tracks':{}}
    for name,p in [('statistical',[.45,.25,.30]),('llm_independent',[.40,.35,.25])]:
        record['tracks'][name] = {'id':name,'fixture_id':fixture['match_id'],
            'generated_at':'2026-10-24T11:00:00+00:00','probabilities':p,'model':name}
    record['tracks']['llm_independent'].update(exposed_to_statistical=False,exposed_to_market=False)
    packet = {'fixture':fixture,'as_of':'2026-10-24T10:00:00+00:00',
        'evidence':[], 'unknowns':['current personnel effect unknown'],
        'descriptive_context':[{'id':'past','observed_at':'2026-10-24T09:00:00+00:00',
            'text':'saved older results; not current player ability'}]}
    review = {'fixture_id':fixture['match_id'],'generated_at':'2026-10-25T12:00:00+00:00',
        'model':'test reviewer','exposed_to_statistical':True,'exposed_to_market':False,
        'issues':[{'id':'old','kind':'historical_transfer','summary':'Prior squad continuity not measured',
            'evidence_ids':['past'],'next_question':'Which players contributed to the earlier observations?'}]}
    return record,packet,review


def test_common_rank_does_not_create_or_raise_a_probability():
    record,packet,review = inputs()
    original=deepcopy(record)
    result=reconcile_evidence(record,packet,review,as_of='2026-10-25T13:00:00+00:00')
    assert record == original
    assert result['comparison']['same_unique_top'] is True
    assert result['comparison']['common_top_outcomes'] == ['home']
    assert result['combined_probabilities'] is None
    assert result['current_personnel_continuity'] is None
    assert result['tracks']['statistical']['probabilities'] == [.45,.25,.30]


def test_tied_top_is_preserved_and_missing_line_does_not_become_agreement():
    record,packet,review=inputs()
    record['tracks']['llm_independent']['probabilities']=[.4,.4,.2]
    r=reconcile_evidence(record,packet,review,as_of='2026-10-25T13:00:00+00:00')
    assert r['comparison']['llm_top_outcomes'] == ['home','draw']
    assert r['comparison']['same_unique_top'] is False
    record['tracks']['llm_independent']=None
    assert reconcile_evidence(record,packet,review,as_of='2026-10-25T13:00:00+00:00')['comparison'] is None


def test_interpretations_cannot_cite_later_evidence_or_silently_adjust_numbers():
    record,packet,review=inputs()
    packet['descriptive_context'][0]['observed_at']='2026-10-26T09:00:00+00:00'
    with pytest.raises(ValueError,match='available packet evidence'):
        reconcile_evidence(record,packet,review,as_of='2026-10-25T13:00:00+00:00')
    record,packet,review=inputs()
    review['probability_adjustment']=[.1,0,-.1]
    with pytest.raises(ValueError,match='not a new numerical forecast'):
        reconcile_evidence(record,packet,review,as_of='2026-10-25T13:00:00+00:00')


def test_model_exposure_and_before_kickoff_time_are_explicit():
    record,packet,review=inputs()
    review['exposed_to_statistical']=False
    with pytest.raises(ValueError,match='disclose model exposure'):
        reconcile_evidence(record,packet,review,as_of='2026-10-25T13:00:00+00:00')
    review['exposed_to_statistical']=True
    with pytest.raises(ValueError,match='precede kickoff'):
        reconcile_evidence(record,packet,review,as_of='2026-11-01T15:00:00+00:00')


def test_later_analysis_does_not_make_a_prediction_available_to_an_earlier_review():
    record,packet,review=inputs()
    record['tracks']['statistical']['generated_at']='2026-10-25T12:30:00+00:00'
    with pytest.raises(ValueError,match='not available at as_of'):
        reconcile_evidence(record,packet,review,as_of='2026-10-25T13:00:00+00:00')
    record,packet,review=inputs()
    packet['descriptive_context'][0]['observed_at']='2026-10-24T10:30:00+00:00'
    with pytest.raises(ValueError,match='available packet evidence'):
        reconcile_evidence(record,packet,review,as_of='2026-10-25T13:00:00+00:00')
