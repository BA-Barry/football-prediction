from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from build_current_adjusted_context_20261028 import arithmetic_checks, team_context


class FixedPastModel:
    def __init__(self, known=True):
        self.teams={'A','B'} if known else {'A'}

    def predict(self, home, away, competition, match_date):
        return (1., .8)

    def parameters(self):
        return {'training_max_date':'2026-10-01'}


def history():
    return [SimpleNamespace(match_id=f'I1:2026/27:{i}',match_date=date(2026,10,11)+timedelta(days=i),
        home='A',away='B',understat_id=str(i),xg=(2.,.5),goals=(1,0),source_url='source') for i in range(5)]


@pytest.mark.parametrize('team,attack,defense,net', [('A',1.,-.3,1.3),('B',-.3,1.,-1.3)])
def test_each_side_uses_its_own_attack_and_defense_sign(team,attack,defense,net):
    rows=history()
    out=team_context(team,rows,{r.match_date:FixedPastModel() for r in rows},rows)
    assert out['mean_attack_xg_residual']==pytest.approx(attack)
    assert out['mean_defense_xga_residual']==pytest.approx(defense)
    assert out['mean_net_xg_residual']==pytest.approx(net)
    assert arithmetic_checks({team:out})['team_observations_checked']==5


def test_unknown_earlier_team_is_not_zero_residual_or_replaced_window():
    rows=history()
    models={r.match_date:FixedPastModel(known=i!=0) for i,r in enumerate(rows)}
    out=team_context('A',rows,models,rows)
    assert out['reference_count']==5
    assert out['available_residual_count']==4
    assert out['mean_actual_xg_for']==2.
    assert out['mean_earlier_expected_xg_for'] is None
    assert out['mean_net_xg_residual'] is None
    assert out['observations'][0]['earlier_expected_xg_for'] is None
    assert out['observations'][0]['missing_reason']=='unknown_team_in_earlier_fit:B'


def test_short_window_retains_missing_mean_and_original_observations():
    rows=history()[:4]
    out=team_context('A',rows,{r.match_date:FixedPastModel() for r in rows},rows)
    assert len(out['observations'])==4
    assert out['mean_net_xg_residual'] is None


def test_actual_arithmetic_review_rejects_unsafe_training_date():
    rows=history()
    out=team_context('A',rows,{r.match_date:FixedPastModel() for r in rows},rows)
    out['observations'][0]['past_training_max_date']=out['observations'][0]['date']
    with pytest.raises(AssertionError):
        arithmetic_checks({'A':out})


def test_actual_arithmetic_review_rejects_inverted_defense_residual():
    rows=history()
    out=team_context('A',rows,{r.match_date:FixedPastModel() for r in rows},rows)
    out['observations'][0]['defense_xga_residual'] *= -1
    with pytest.raises(AssertionError):
        arithmetic_checks({'A':out})
