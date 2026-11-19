from copy import deepcopy
from datetime import date, timedelta

import pytest

from football_model.historical_relevance import personnel_usage_overlap


def history(n=20):
    return [{'match_id': f'match:{i}', 'match_date': (date(2024,1,1)+timedelta(days=i)).isoformat(),
             'season':'2023/24', 'home':'team:one','away':'team:two',
             'home_starters':[f'provider:player:{j}' for j in range(11)],
             'away_starters':[f'provider:player:{j+20}' for j in range(11)]}
            for i in range(n)]


def test_disjoint_windows_and_identical_distribution():
    result=personnel_usage_overlap(history(),'team:one','2024-02-01')
    assert result['overlap']==pytest.approx(1)
    assert result['recent']['n_matches']==5
    assert result['preceding_reference']['n_matches']==15
    assert not set(result['recent']['match_ids']) & set(result['preceding_reference']['match_ids'])
    assert result['reference_start_share_seen_in_recent']==pytest.approx(1)


def test_fully_changed_distribution_has_actual_zero():
    rows=history()
    for r in rows[-5:]:r['home_starters']=[f'provider:player:{j+100}' for j in range(11)]
    assert personnel_usage_overlap(rows,'team:one','2024-02-01')['overlap']==0


def test_incomplete_history_is_unknown_not_zero():
    result=personnel_usage_overlap(history(19),'team:one','2024-02-01')
    assert result['overlap'] is None
    assert result['preceding_reference']['n_matches']==14
    assert result['recent']['n_matches']==5
    assert result['reference_start_share_seen_in_recent'] is None


def test_same_day_or_future_target_xi_cannot_affect_measurement():
    rows=history()
    expected=personnel_usage_overlap(rows,'team:one','2024-02-01')
    # Deliberately malformed future XI must not even be inspected.
    extra=deepcopy(rows[-1]);extra.update(match_id='target:match',match_date='2024-02-01',home_starters=['unavailable'])
    rows.append(extra)
    assert personnel_usage_overlap(rows,'team:one','2024-02-01')==expected
    extra['match_date']='2024-03-01'
    assert personnel_usage_overlap(rows,'team:one','2024-02-01')==expected


def test_player_ids_require_provider_namespace_and_cannot_silently_mix():
    rows=history();rows[0]['home_starters'][0]='123'
    with pytest.raises(ValueError,match='namespaced'):personnel_usage_overlap(rows,'team:one','2024-02-01')
    rows=history();rows[0]['home_starters'][0]='other:player:999'
    with pytest.raises(ValueError,match='provider'):personnel_usage_overlap(rows,'team:one','2024-02-01')
