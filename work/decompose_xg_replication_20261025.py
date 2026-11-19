"""Exact Poisson-total/binomial-allocation identity, no additional fitting."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from football_model.forecast_diagnostics import moving_block_means

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
NEW = ROOT/'outputs/进球xG更早年份复现_20261024T180238377843Z'
PREVIOUS = ROOT/'outputs/进球xG共同训练记录_20261024T174247963857Z/共同训练记录融合结果.json'


def losses(row):
    h,a = row['actual_score']
    lh,la = row['rates']
    n,mu = h+a,lh+la
    p = lh/mu
    total = mu-n*math.log(mu)+math.lgamma(n+1)
    allocation = -(math.lgamma(n+1)-math.lgamma(h+1)-math.lgamma(a+1)
                   +h*math.log(p)+a*math.log1p(-p))
    exact = math.fsum(l-y*math.log(l)+math.lgamma(y+1) for l,y in ((lh,h),(la,a)))
    if not math.isclose(total+allocation,exact,rel_tol=0,abs_tol=1e-11):
        raise ValueError('Exact goal probability factorization failed')
    return total,allocation,exact


def diagnostic(goal,blend):
    goal = sorted(goal,key=lambda r:(r['date'],r['match_id']))
    blend = sorted(blend,key=lambda r:(r['date'],r['match_id']))
    if [r['match_id'] for r in goal] != [r['match_id'] for r in blend]:
        raise ValueError('Paired prediction identities differ')
    before,after = np.asarray([losses(r) for r in goal]),np.asarray([losses(r) for r in blend])
    delta = after-before
    order = ['exact_total_goals','home_away_allocation_given_actual_total','exact_score']
    if not np.allclose(delta[:,0]+delta[:,1],delta[:,2],rtol=0,atol=2e-11):
        raise ValueError('Difference decomposition failed')
    return {'n':len(goal),'column_order':order,'goal_losses':dict(zip(order,before.mean(axis=0).tolist())),
            'blend_losses':dict(zip(order,after.mean(axis=0).tolist())),
            'mean_loss_differences':dict(zip(order,delta.mean(axis=0).tolist())),
            'block_intervals':{str(b):moving_block_means([r['date'] for r in goal],delta,block_weeks=b)
                               for b in (2,4,8)},
            'max_factorization_error':float(np.max(np.abs((before[:,0]+before[:,1])-before[:,2]))),
            'match_ids':[r['match_id'] for r in goal]}


def main():
    new = json.loads((NEW/'更早年份复现结果.json').read_text(encoding='utf-8'))
    old = json.loads(PREVIOUS.read_text(encoding='utf-8'))
    groups = {
        '2023-24':{'goal':[r for r in new['predictions']['goal'] if r['period']=='2023-24'],
                   'blend':[r for r in new['predictions']['blend'] if r['period']=='2023-24']},
        '2025-26':{'goal':[r for r in old['common_goal_predictions'] if r['period']=='2025-26'],
                   'blend':[r for r in old['common_blend_predictions'] if r['period']=='2025-26']},
    }
    result = {'created_at':datetime.now(timezone.utc).isoformat(),
              'identity':'independent Poisson home/away score equals Poisson exact total times Binomial home allocation conditional on that total',
              'purpose':'locate saved predictive improvement, no features, fitting, coefficient choice or current score review',
              'new_fits':0,
              'limitations':['actual total used only to score the conditional decomposition after a historical match; never an input to a prematch prediction',
                             'conditional block intervals on already saved predictions, no full model/selection uncertainty',
                             'decomposition is arithmetic attribution, not causal attribution to xG quality or a proposed new model'],
              'source_results':[str(NEW/'更早年份复现结果.json'),str(PREVIOUS)],
              'held_seasons':{p:diagnostic(v['goal'],v['blend']) for p,v in groups.items()},
              'by_league':{p:{c:diagnostic([r for r in v['goal'] if r['competition']==c],
                                         [r for r in v['blend'] if r['competition']==c])
                              for c in ('E0','D1','SP1','I1','F1')} for p,v in groups.items()}}
    (NEW/'比分增益精确分解.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({p:{'deltas':v['mean_loss_differences'],'intervals':{b:e['interval_95'] for b,e in v['block_intervals'].items()}}
                      for p,v in result['held_seasons'].items()},ensure_ascii=False))


if __name__ == '__main__':
    main()
