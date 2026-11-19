"""Ticket arithmetic and predictions from the existing saved EPL fit.

No result collection, refit, parameter selection or probability calibration.
"""
from datetime import date, datetime, timezone
from itertools import product
import json
import math
from pathlib import Path

from football_model.date_model import DateTeamGoalModel
from football_model.probability import score_grid, summarize

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OUT = ROOT / 'outputs/用户六串一分析_20261025'
OUT.mkdir(parents=True, exist_ok=True)
SAVED = ROOT / 'outputs/批量球队投影_20261024T150805722459Z/批量球队投影与来源.json'

TICKET = [
    dict(id='7013', name='水晶宫—纽卡斯尔', prices=[2.56, 3.2, 2.35], selected=[0, 1],
         kickoff='2026-10-25T14:00:00+00:00', llm=[.34, .27, .39]),
    dict(id='7014', name='曼联—伯恩茅斯', prices=[2.08, 3.43, 2.8], selected=[1, 2],
         kickoff='2026-10-25T14:00:00+00:00', llm=[.40, .28, .32]),
    dict(id='7022', name='弗赖堡—拜仁', prices=[6.15, 5.4, 1.28], selected=[2],
         kickoff='2026-10-25T16:30:00+00:00', llm=[.14, .23, .63]),
    dict(id='7024', name='皇家社会—莱万特', prices=[1.37, 4.5, 5.65], selected=[0],
         kickoff='2026-10-25T17:30:00+00:00', llm=[.66, .22, .12]),
    dict(id='7026', name='乌迪内斯—AC米兰', prices=[4.6, 3.75, 1.55], selected=[2],
         kickoff='2026-10-25T19:45:00+00:00', llm=[.15, .23, .62]),
    dict(id='7029', name='洛杉矶FC—洛城银河', prices=[1.43, 4.35, 5.0], selected=[0],
         kickoff='2026-10-26T01:00:00+00:00', llm=[.59, .24, .17]),
]

def main():
    now = datetime.now(timezone.utc)
    saved = json.loads(SAVED.read_text(encoding='utf-8'))
    params = saved['model_parameters']
    model = DateTeamGoalModel(params['regularization'], params['half_life_days'])
    model.coefficients = params['coefficients']
    model.intercept = params['intercept']
    model.before_date = date.fromisoformat(params['before_date'])
    model.teams = set(params['teams'])
    model.competitions = set(params['competitions'])
    epl = []
    for ticket, home, away in ((TICKET[0], 'Crystal Palace', 'Newcastle'),
                                (TICKET[1], 'Man United', 'Bournemouth')):
        if now >= datetime.fromisoformat(ticket['kickoff']):
            raise ValueError('Cannot create a new pre-match forecast after kickoff')
        rates = model.predict('football-data:E0:' + home, 'football-data:E0:' + away,
                              'E0', date(2026, 10, 25))
        grid = score_grid(*rates, rho=0)
        forecast = summarize(grid)
        p = list(forecast['one_x_two'].values())
        assert len(p) == 3 and min(p) > 0 and math.isclose(sum(p), 1, abs_tol=1e-12)
        epl.append(dict(id=ticket['id'], name=ticket['name'], generated_at=now.isoformat(),
                        model_fit_generated_at=saved['generated_at'], probabilities=p,
                        selected_probability=sum(p[k] for k in ticket['selected']),
                        rates=list(rates), forecast=forecast,
                        class_order=['home', 'draw', 'away'],
                        training_matches=saved['training_matches'],
                        current_season_matches=saved['current_season_matches'],
                        source_model=str(SAVED),
                        basis='existing saved .01/180 EPL fit replay; no new fit'))
    for r in TICKET:
        inv = [1/o for o in r['prices']]
        r['inverse_odds_sum'] = sum(inv)
        r['market_proportional_devig'] = [p/sum(inv) for p in inv]
        r['market_selected_probability'] = sum(r['market_proportional_devig'][i] for i in r['selected'])
        r['llm_selected_probability'] = sum(r['llm'][i] for i in r['selected'])
        r['llm_fair_single_odds'] = [1/p for p in r['llm']]
        r['break_even_single_probabilities'] = inv
    bets = []
    for combo in product(*(r['selected'] for r in TICKET)):
        odds = math.prod(r['prices'][i] for r, i in zip(TICKET, combo))
        q = math.prod(r['market_proportional_devig'][i] for r, i in zip(TICKET, combo))
        bets.append(dict(outcomes=list(combo), product_odds=odds,
                         gross_payout_at_two_yuan=2*odds, market_probability_independence=q))
    result = dict(generated_at=now.isoformat(), screenshot_clock='21:01; date/year and quote update time not independently visible',
                  class_order=['home', 'draw', 'away'], fixtures=TICKET,
                  market_all_selected_independence=math.prod(r['market_selected_probability'] for r in TICKET),
                  llm_all_selected_independence=math.prod(r['llm_selected_probability'] for r in TICKET),
                  component_bets=bets, assumed_cost_two_yuan_per_bet=2*len(bets),
                  market_expected_gross_payout_ratio=sum(b['product_odds']*b['market_probability_independence'] for b in bets)/len(bets),
                  new_epl_fits=0, parameter_searches=0, target_results_read=False,
                  quote_source='user image, not verified executable live odds; opening/current movement unavailable',
                  llm_status='same-platform subjective research judgement from sourced prematch context; context already includes screenshot odds; not calibrated or fully blind',
                  fusion_status='no market fusion; statistical/LLM arithmetic reference may be displayed separately and is untrained',
                  limits=['One quote proportional de-vig is an approximation, not true probability.',
                          'Cross-match independence is an approximation, not a guarantee.',
                          'No complete six-leg statistical model probability: MLS is unsupported.',
                          'Saved EPL fit is dated 2026-10-24 and has not incorporated every later result.',
                          'Payout ignores any official rounding, tax or undisclosed multiplier.'])
    (OUT/'英超统计线路.json').write_text(json.dumps(epl,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'票面与概率计算.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(epl=epl, market_all_selected=result['market_all_selected_independence'],
                         llm_all_selected=result['llm_all_selected_independence'],
                         expected_gross_payout_ratio=result['market_expected_gross_payout_ratio']),ensure_ascii=False))

if __name__ == '__main__':
    main()
