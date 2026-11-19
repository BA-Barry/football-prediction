"""Approved fixed-total allocation contrast, learned earlier and held later."""
from datetime import date, datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.stats import poisson

from football_model.goal_allocation import (fit_allocation_weight, predict_allocation_rates,
                                            allocation_objective)
from football_model.forecast_diagnostics import moving_block_means
from football_model.evaluation import probability_scores
from football_model.probability import score_grid, summarize
from run_shot_increment_20261025 import score_details, write
from decompose_xg_replication_20261025 import losses as partition_losses

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
EARLY = ROOT/'outputs/进球xG更早年份复现_20261024T180238377843Z/更早年份复现结果.json'
LATER = ROOT/'outputs/进球xG共同训练记录_20261024T174247963857Z/共同训练记录融合结果.json'
MARKET = ROOT/'outputs/进球xG历史市场对照_20261024T181414691003Z/历史市场对照结果.json'
XG = [ROOT/'outputs/独立xG跨联赛复现_20261024T172741074372Z/跨联赛xG复现结果.json',
      ROOT/'outputs/独立历史xG前推_20261024T171031805341Z/独立xG前推结果.json']
COMPS = ('E0','D1','SP1','I1','F1')
CONTRASTS = (('allocation','full_blend'),('allocation','goal'),('allocation','xg'),('full_blend','goal'))


def comparison(tracks):
    ids = [r['match_id'] for r in tracks['full_blend']]
    if any([r['match_id'] for r in rows]!=ids for rows in tracks.values()):
        raise ValueError('Allocation contrast needs identical ordered targets')
    scores,losses = {},{}
    for arm,rows in tracks.items():
        scores[arm],losses[arm] = score_details(rows)
        partition = np.asarray([partition_losses(r) for r in rows])
        losses[arm]['exact_integer_total_goals'] = partition[:,0]
        losses[arm]['conditional_home_allocation'] = partition[:,1]
        scores[arm]['exact_integer_total_goals'] = {'n':len(rows),'log_loss':float(partition[:,0].mean())}
        scores[arm]['conditional_home_allocation'] = {'n':len(rows),'log_loss':float(partition[:,1].mean())}
    order,values = [],[]
    for arm,ref in CONTRASTS:
        for domain in losses[arm]:
            order.append(f'{arm}_minus_{ref}:{domain}')
            values.append(losses[arm][domain]-losses[ref][domain])
    matrix = np.column_stack(values)
    change = losses['allocation']['exact_score']-losses['full_blend']['exact_score']
    allocation_change = losses['allocation']['conditional_home_allocation']-losses['full_blend']['conditional_home_allocation']
    if not np.allclose(change,allocation_change,rtol=0,atol=1e-10):
        raise ValueError('With total fixed the full-score change must equal the allocation change')
    return {'n':len(ids),'scores':scores,'difference_order':order,
            'mean_loss_differences':dict(zip(order,matrix.mean(axis=0).tolist())),
            'block_intervals':{str(b):moving_block_means([r['date'] for r in tracks['full_blend']],matrix,block_weeks=b)
                               for b in (2,4,8)},'match_ids':ids,
            'max_full_score_change_minus_allocation_change':float(np.max(np.abs(change-allocation_change)))}


def market_comparison(rows, observations):
    source = {r['match_id']:r for r in observations}
    eligible = [r for r in rows if 'avg_preclosing' in source[r['match_id']]['markets']]
    y = np.asarray([r['actual_outcome'] for r in eligible])
    new = np.asarray([r['probabilities'] for r in eligible])
    market = np.asarray([source[r['match_id']]['markets']['avg_preclosing']['probabilities'] for r in eligible])
    delta = -np.log(new[np.arange(len(y)),y])+np.log(market[np.arange(len(y)),y])
    return {'n':len(eligible),'allocation_scores':probability_scores(new,y),'market_scores':probability_scores(market,y),
            'allocation_minus_market_LL':float(delta.mean()),
            'block_intervals':{str(b):moving_block_means([r['date'] for r in eligible],delta[:,None],block_weeks=b)
                               for b in (2,4,8)},
            'missing_match_ids':[r['match_id'] for r in rows if r not in eligible]}


def main():
    now = datetime.now(timezone.utc)
    target = ROOT/'outputs'/now.strftime('固定总球主客分配研究_%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True,exist_ok=False)
    plan = {'created_at':now.isoformat(),'user_approved':'explicit reply: first study home/away allocation',
            'structure':'save T=sum(original full blend rates); q=expit((1-wQ)logit(qGoal)+wQ logit(qXg)); rates=(Tq,T(1-q))',
            'target_total':'predicted saved rate, not actual total goals; actual h+a belongs only to the training Binomial target',
            'weight':'one global bounded allocation coefficient per chronological replication; uniform per-fixture conditional Binomial NLL including log-choose',
            'chronological_splits':[{'train':'2022-23','held':'2023-24'},{'train':'2024-25','held':'2025-26'}],
            'flat_case':'record fit non-identifiability and apply original full-blend weight, preserving the previous allocation',
            'budget':{'new_team_fits':0,'new_scalar_allocation_fits':2,'parameter_grids':0},
            'primary':'held complete-score NLL allocation minus saved full blend, reported separately for both replications',
            'secondary':['1X2 LL/Brier/Top1','total distribution invariance','away-minus-two LL','pure goal/xG reference','archived avg_preclosing comparison'],
            'uncertainty':'paired noncircular calendar blocks2/4/8weeks2000 repeats,conditional descriptive intervals',
            'validation':['original old weight must replay old allocation','earlier optimized conditional loss no worse than feasible old weight',
                          'total rates invariant','analytic Poisson totals agree within negligible grid tail error',
                          'full-score difference equals conditional allocation difference'],
            'source_results':list(map(str,[EARLY,LATER,MARKET,*XG])),
            'limitations':['historical research and method selection already informed by outcomes, not blind',
                           'conditional intervals omit full parameter and research-selection uncertainty',
                           'historical xG publication/revision and quote times unknown',
                           'fixed totals deliberately isolate allocation; no promise of 1X2 improvement'],
            'new_live_results_reviewed':False,'adoption':'research only, all original forecasts and markets/LLMs retained'}
    write(target/'分析前计划.json',plan)
    first = json.loads(EARLY.read_text(encoding='utf-8'))
    second = json.loads(LATER.read_text(encoding='utf-8'))
    markets = json.loads(MARKET.read_text(encoding='utf-8'))
    xg_later = [r for p in XG for r in json.loads(p.read_text(encoding='utf-8'))['new_predictions']]
    splits = {
        '2022-23_to_2023-24':{'train':'2022-23','held':'2023-24','old_w':first['parameters']['weight'],
            'tracks':{'goal':first['predictions']['goal'],'xg':first['predictions']['xg'],'full_blend':first['predictions']['blend']},'expected_early':1790},
        '2024-25_to_2025-26':{'train':'2024-25','held':'2025-26','old_w':second['parameters']['weight'],
            'tracks':{'goal':second['common_goal_predictions'],'xg':xg_later,'full_blend':second['common_blend_predictions']},'expected_early':1734},
    }
    output = {}
    for name,split in splits.items():
        maps = {a:{r['match_id']:r for r in rows} for a,rows in split['tracks'].items()}
        ids = sorted(maps['full_blend'],key=lambda k:(maps['full_blend'][k]['date'],k))
        if any(set(m)!=set(ids) or len(m)!=len(split['tracks'][a]) for a,m in maps.items()):
            raise ValueError('Original split target identities changed')
        inputs,totals = [],{}
        max_replay = 0.
        for identifier in ids:
            g,x,b = (maps[a][identifier] for a in ('goal','xg','full_blend'))
            for field in ('date','period','competition','home','away','actual_score','actual_outcome','fit_before_date'):
                if g[field]!=x[field] or g[field]!=b[field]:
                    raise ValueError('Goal/xG/full mixture context differs')
            if not b['training_max_date'] < b['fit_before_date'] <= b['date']:
                raise ValueError('Original information dates invalid')
            total = sum(b['rates'])
            totals[identifier] = total
            replay = predict_allocation_rates(total,g['rates'],x['rates'],split['old_w'])
            max_replay = max(max_replay,float(np.max(np.abs(np.asarray(replay)-b['rates']))))
            if not np.allclose(replay,b['rates'],rtol=0,atol=1e-10):
                raise ValueError('Original allocation is not recovered at the original full weight')
            inputs.append({'match_id':identifier,'date':b['date'],'goal_rates':g['rates'],
                           'xg_rates':x['rates'],'actual_score':b['actual_score']})
        cutoff = date(int(split['held'][:4]),8,1)
        early = [r for r in inputs if r['date']<cutoff.isoformat()]
        parameters = fit_allocation_weight(early,cutoff)
        if parameters['n']!=split['expected_early']:
            raise ValueError('Allocation training target cohort changed')
        applied = split['old_w'] if parameters['not_identifiable'] else parameters['weight']
        old_loss = allocation_objective(split['old_w'],[r['goal_rates'] for r in early],
            [r['xg_rates'] for r in early],[r['actual_score'] for r in early])[0]
        if parameters['objective']>old_loss+1e-10:
            raise ValueError('Early optimized loss is worse than the feasible old allocation')
        predictions = []
        max_total,max_total_grid,max_analytic = 0.,0.,0.
        for identifier in ids:
            g,x,b = (maps[a][identifier] for a in ('goal','xg','full_blend'))
            total = totals[identifier]
            rates = predict_allocation_rates(total,g['rates'],x['rates'],applied)
            forecast = summarize(score_grid(*rates,rho=0))
            probs = np.asarray([forecast['total_goals'][k] for k in ('0','1','2','3','4+')])
            previous = np.asarray([b['forecast']['total_goals'][k] for k in ('0','1','2','3','4+')])
            analytic = np.r_[poisson.pmf(np.arange(4),total),poisson.sf(3,total)]
            max_total = max(max_total,abs(sum(rates)-total))
            max_total_grid = max(max_total_grid,float(np.max(np.abs(probs-previous))))
            max_analytic = max(max_analytic,float(np.max(np.abs(probs-analytic))))
            if abs(sum(rates)-total)>1e-12 or not np.allclose(probs,analytic,rtol=0,atol=1e-10):
                raise ValueError('Predicted total changed or grid total approximation differs materially')
            context = {k:v for k,v in b.items() if k!='shared_xg_weight'}
            predictions.append({**context,'original_full_blend_rates':b['rates'],'fixed_total_rate':total,
                'goal_rates':g['rates'],'xg_rates':x['rates'],'full_blend_xg_weight':split['old_w'],
                'allocation_xg_weight':applied,'rates':list(rates),'forecast':forecast,
                'probabilities':list(forecast['one_x_two'].values()),
                'model_type':'fixed_total_conditional_allocation',
                'allocation_fit_before_date':parameters['before_date'],
                'allocation_training_matches':parameters['n'],
                'prediction_role':('allocation_estimation_apparent_fit_not_out_of_time' if b['period']==split['train']
                                   else 'held_earlier_allocation_weight')})
        tracks = {**{a:[m[k] for k in ids] for a,m in maps.items()},'allocation':predictions}
        seasons,leagues = {},{}
        for period in (split['train'],split['held']):
            selected = {a:[r for r in rows if r['period']==period] for a,rows in tracks.items()}
            seasons[period] = comparison(selected)
            seasons[period]['role'] = 'allocation estimation apparent fit' if period==split['train'] else 'held allocation weight retrospective evaluation'
            leagues[period] = {c:comparison({a:[r for r in rows if r['competition']==c] for a,rows in selected.items()}) for c in COMPS}
        held_rows = [r for r in predictions if r['period']==split['held']]
        if len(held_rows)!=1733:
            raise ValueError('Held allocation coverage changed')
        output[name] = {'parameters':parameters,'applied_allocation_weight':applied,'old_full_blend_weight':split['old_w'],
            'old_weight_early_conditional_nll':old_loss,'weight_training_rows':early,
            'predictions':predictions,'seasons':seasons,'by_league':leagues,
            'market_reference':market_comparison(held_rows,markets['observations'][split['held']]),
            'invariance':{'max_old_weight_rate_replay_error':max_replay,'max_total_rate_error':max_total,
                          'max_total_group_probability_change':max_total_grid,'max_new_total_grid_vs_analytic':max_analytic}}
        print(f'Completed {name}: old allocation {split["old_w"]:.6f}, fitted allocation {applied:.6f}',flush=True)
    result = {'plan':plan,'finished_at':datetime.now(timezone.utc).isoformat(),'splits':output,
              'new_team_fits':0,'new_scalar_allocation_fits':2,'parameter_grids':0,'adopted':False}
    write(target/'固定总球主客分配结果.json',result)
    print(json.dumps({'directory':str(target),'splits':{name:{'weight':v['applied_allocation_weight'],
        'held_n':v['seasons'][name[-7:]]['n'],
        'deltas':{k:x for k,x in v['seasons'][name[-7:]]['mean_loss_differences'].items() if k.startswith('allocation_minus_full_blend:')},
        'market_gap':v['market_reference']['allocation_minus_market_LL'],'invariance':v['invariance']}
        for name,v in output.items()}},ensure_ascii=False))


if __name__ == '__main__':
    main()
