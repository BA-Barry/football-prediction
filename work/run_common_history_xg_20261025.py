"""Measure whether the goal/xG improvement survives identical fit histories."""
from datetime import date,datetime,timezone
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import numpy as np

from football_model.date_model import DateTeamGoalModel
from football_model.native_experiment import load_seasons,season_directories
from football_model.team_model import team_features
from football_model.goal_xg_blend import fit_goal_xg_weight,predict_goal_xg_rates
from football_model.forecast_diagnostics import moving_block_means
from football_model.probability import score_grid,summarize
from run_shot_increment_20261025 import score_details,write

ROOT=REPOSITORY_ROOT
CROSS=ROOT/'outputs/独立xG跨联赛复现_20261024T172741074372Z/跨联赛xG复现结果.json'
EPL=ROOT/'outputs/独立历史xG前推_20261024T171031805341Z/独立xG前推结果.json'
BLEND=ROOT/'outputs/进球xG单权重融合_20261024T173226297740Z/进球xG单权重融合结果.json'
CORE=ROOT/'outputs/核心参数训练_20261023T115640092598Z'
COMPS=('E0','D1','SP1','I1','F1')
CONTRASTS=(('common_blend','common_goal'),('fixed_old_weight','common_goal'),
           ('common_goal','original_goal'),('fixed_old_weight','old_blend'),
           ('common_blend','fixed_old_weight'),('common_blend','old_blend'),
           ('common_blend','xg'))


def comparison(tracks):
    ordered=[r['match_id'] for r in tracks['common_goal']]
    if any([r['match_id'] for r in rows]!=ordered for rows in tracks.values()):
        raise ValueError('All arms must retain identical ordered target fixtures')
    scores,losses={},{}
    for arm,rows in tracks.items():
        scores[arm],losses[arm]=score_details(rows)
    columns,values=[],[]
    for arm,reference in CONTRASTS:
        for domain in losses[arm]:
            columns.append(f'{arm}_minus_{reference}:{domain}')
            values.append(losses[arm][domain]-losses[reference][domain])
    matrix=np.column_stack(values)
    return {'n':len(ordered),'scores':scores,'difference_order':columns,
            'mean_loss_differences':dict(zip(columns,matrix.mean(axis=0).tolist())),
            'block_intervals':{str(b):moving_block_means([r['date'] for r in tracks['common_goal']],matrix,block_weeks=b)
                               for b in (2,4,8)},'match_ids':ordered}


def prediction(row,rates):
    forecast=summarize(score_grid(*rates,rho=0))
    return {**row,'rates':list(rates),'forecast':forecast,
            'probabilities':list(forecast['one_x_two'].values())}


def replay(parameters,row):
    return tuple(math.exp(parameters['intercept']+math.fsum(parameters['coefficients'].get(k,0)*v
        for k,v in vector.items())) for vector in team_features(row['home'],row['away'],row['competition']))


def main():
    now=datetime.now(timezone.utc)
    target=ROOT/'outputs'/now.strftime('进球xG共同训练记录_%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True,exist_ok=False)
    plan={'created_at':now.isoformat(),'purpose':'isolate the 15 ordinary source-date conflict exclusions from the xG increment',
          'specific_failure':'previous goal and xG fits did not use identical records; loss gain could partly come from preprocessing',
          'measurement_needed':'IDs/types and ordinary arithmetic tests check correctness but cannot quantify the counterfactual predictions after removing training records',
          'fit_history':'goal fit IDs exactly equal each saved xG fit IDs, same cutoff/weights/preset; only response differs',
          'preset':{'regularization':.01,'half_life_days':180,'update_days':28,'rho':0},
          'reuse':'identical original fit history reuses its original readable coefficients; xG fits all reused',
          'budget':{'new_goal_fits':69,'reused_goal_fits':41,'reused_xg_fits':110,
                    'new_single_weight_fits':1,'parameter_grids':0},
          'weight':'one global log-rate interpolation weight, fit only on earlier 2024/25 1734 fixtures; held throughout later 2025/26 1733',
          'fixed_old_weight':'also apply previous .6132130768996271 without fitting, to separate preprocessing and weight re-estimation',
          'primary':'later common-history blend minus common-history goal complete-score NLL',
          'secondary':['1X2 LL/Brier/Top1','total-goals LL','away-minus-two three-way LL','comparison to old mixture and pure xG'],
          'uncertainty':'paired noncircular calendar blocks 2/4/8 weeks,2000 repeats, conditional descriptive intervals',
          'unchanged':['all prior historic outputs','all current prematch probabilities','market and LLM lines'],
          'new_live_results_reviewed':False,
          'limitations':['already-viewed outcomes, exploratory not new blind validation',
                         'intervals condition on fitted weight; no full model-selection or parameter bootstrap',
                         'historical xG publication/revisions unknown, calendar-date availability assumption'],
          'adoption':'research only, no later-season choice of weight or league-specific exceptions'}
    write(target/'分析前计划.json',plan)
    cross=json.loads(CROSS.read_text(encoding='utf-8'))
    epl=json.loads(EPL.read_text(encoding='utf-8'))
    old_blend=json.loads(BLEND.read_text(encoding='utf-8'))
    xg_parameters=dict(cross['new_xg_parameters'])
    xg_parameters.update({f'E0:{k}':v for k,v in epl['xg_parameters'].items()})
    originals={r['match_id']:r for r in cross['original_predictions']+epl['original_predictions']}
    xg=sorted(cross['new_predictions']+epl['new_predictions'],key=lambda r:(r['date'],r['match_id']))
    previous={r['match_id']:r for r in old_blend['blend_predictions']}
    records,_=load_seasons(season_directories(ROOT))
    sources={r.match_id:r for r in records}
    old_parameters={}
    for period,arm in (('2024-25','regularized_poisson_h180_a0p01'),('2025-26','original')):
        for comp in COMPS:
            trial=json.loads((CORE/period/arm/f'{comp}.json').read_text(encoding='utf-8'))
            for parameters in trial['fits']:
                old_parameters[f'{comp}:{parameters["before_date"]}']=parameters
    fitted,new_count,reused={},0,0
    for key,xp in xg_parameters.items():
        cutoff=date.fromisoformat(xp['before_date'])
        comp=xp['competition']
        ids=xp['reference_match_ids']
        history=[sources[k] for k in ids]
        if (len(set(ids))!=len(ids) or len(history)!=xp['n_matches']
                or any(r.competition!=comp or not r.match_date<cutoff or r.goal_fit_exclusion is not None for r in history)
                or ids!=[r.match_id for r in sorted(history,key=lambda r:(r.match_date,r.match_id))]):
            raise ValueError('Common training identity or chronology conflict')
        original=old_parameters[key]
        if original['reference_match_ids']==ids:
            parameters=original
            action='reused_original'
            reused+=1
        else:
            parameters=DateTeamGoalModel(.01,180).fit(history,cutoff).parameters()
            action='new_common_history_fit'
            new_count+=1
        if (parameters['reference_match_ids']!=ids or parameters['n_matches']!=xp['n_matches']
                or parameters['training_min_date']!=xp['training_min_date']
                or parameters['training_max_date']!=xp['training_max_date']
                or parameters['regularization']!=xp['regularization']
                or parameters['half_life_days']!=xp['half_life_days']):
            raise ValueError('The two response tracks must use exactly the same weighted design history')
        fitted[key]={'action':action,'parameters':parameters,
                     'removed_from_original_match_ids':sorted(set(original['reference_match_ids'])-set(ids))}
    if (new_count,reused)!=(69,41):
        raise ValueError('Fit budget differs from premeasured source differences')
    write(target/'共同训练进球参数.json',fitted)
    common_goal=[]
    for row in xg:
        original=originals[row['match_id']]
        entry=fitted[f'{row["competition"]}:{row["fit_before_date"]}']
        rates=replay(entry['parameters'],row)
        if entry['action']=='reused_original':
            if not np.allclose(rates,original['rates'],rtol=0,atol=1e-10):
                raise ValueError('Reused rates must replay original coefficients')
            rates=original['rates']
        parameters=entry['parameters']
        context={**original,'original_rates':original['rates'],
                 'original_training_matches':original['training_matches'],
                 'original_training_max_date':original['training_max_date'],
                 'training_matches':parameters['n_matches'],
                 'training_max_date':parameters['training_max_date']}
        common_goal.append(prediction(context,rates))
    common_map={r['match_id']:r for r in common_goal}
    xg_map={r['match_id']:r for r in xg}
    weight_rows=[{'match_id':r['match_id'],'date':r['date'],'goal_rates':common_map[r['match_id']]['rates'],
                  'xg_rates':r['rates'],'actual_score':r['actual_score']} for r in xg]
    parameters=fit_goal_xg_weight(weight_rows,date(2025,8,1))
    if parameters['n']!=1734 or len(weight_rows)!=3467:
        raise ValueError('Target cohort changed during history matching')
    old_weight=old_blend['parameters']['weight']
    new_weight=parameters['weight']
    fixed,new=[],[]
    for row in xg:
        goal=common_map[row['match_id']]
        context={**goal,'common_goal_rates':goal['rates'],'xg_rates':row['rates']}
        fixed.append(prediction({**context,'shared_xg_weight':old_weight},
                                predict_goal_xg_rates(goal['rates'],row['rates'],old_weight)))
        new.append(prediction({**context,'shared_xg_weight':new_weight},
                              predict_goal_xg_rates(goal['rates'],row['rates'],new_weight)))
    seasons,leagues={},{}
    for period in ('2024-25','2025-26'):
        indices=[i for i,r in enumerate(xg) if r['period']==period]
        tracks={'original_goal':[originals[xg[i]['match_id']] for i in indices],
                'common_goal':[common_goal[i] for i in indices],'xg':[xg[i] for i in indices],
                'old_blend':[previous[xg[i]['match_id']] for i in indices],
                'fixed_old_weight':[fixed[i] for i in indices],'common_blend':[new[i] for i in indices]}
        seasons[period]=comparison(tracks)
        leagues[period]={comp:comparison({arm:[r for r in rows if r['competition']==comp]
                                          for arm,rows in tracks.items()}) for comp in COMPS}
    result={'plan':plan,'finished_at':datetime.now(timezone.utc).isoformat(),
            'new_goal_fits':new_count,'reused_goal_fits':reused,'reused_xg_fits':110,
            'new_scalar_weight_fits':1,'parameter_grids':0,'parameters':parameters,'old_weight':old_weight,
            'common_goal_parameters':fitted,'seasons':seasons,'by_league':leagues,
            'common_goal_predictions':common_goal,'common_blend_predictions':new,
            'fixed_old_weight_predictions':fixed,'weight_training_rows':[r for r in weight_rows if r['date']<'2025-08-01'],
            'source_results':[str(CROSS),str(EPL),str(BLEND)],'adopted':False}
    write(target/'共同训练记录融合结果.json',result)
    later=seasons['2025-26']
    report=['# 进球与xG：共同训练记录的实际对照', '',
            '本轮只优化模型，未查询或复核当前比赛赛果。针对上轮15条日期冲突导致两线训练记录不同的问题，本轮让每个截止的进球训练ID、日期、主客身份和衰减权重与xG逐条相同，仅训练目标不同。', '',
            f'实际新增进球拟合{new_count}次，原训练记录相同者复用{reused}次，110次xG拟合全部复用。原.01／180及28天截止不变，参数网格0个。', '',
            f'较早1734场重估一次共享xG权重：旧{old_weight:.9f}，共同训练记录后{new_weight:.9f}；较晚1733场固定检验。另将旧权重原样作用于共同训练进球率，不再拟合，用于区分资料处理与权重重估。', '',
            '| 较晚赛季线路 | 比分NLL | 胜平负LL | 总球LL | 最高概率命中 |', '|---|---:|---:|---:|---:|']
    labels={'original_goal':'原进球','common_goal':'共同训练进球','xg':'独立xG','old_blend':'上轮融合',
            'fixed_old_weight':'共同训练＋旧权重','common_blend':'共同训练＋早季重估权重'}
    for arm,scores in later['scores'].items():
        report.append(f'| {labels[arm]} | {scores["exact_score"]["log_loss"]:.6f} | {scores["one_x_two"]["log_loss"]:.6f} | {scores["total_goals_0_1_2_3_4plus"]["log_loss"]:.6f} | {round(scores["one_x_two"]["accuracy"]*later["n"])}/{later["n"]} |')
    report+=['', '主比较是共同训练融合对共同训练进球模型，不能只相对旧模型来声称隔离资料影响。较早季参与估计权重，分数属于表观拟合；目标队列仍为较早1734、较晚1733，未因重训改变支持。', '',
             '| 后季差值 | 比分NLL | 胜平负LL |', '|---|---:|---:|']
    for arm,reference in CONTRASTS:
        name=f'{arm}_minus_{reference}'
        report.append(f'| {labels[arm]} − {labels[reference]} | {later["mean_loss_differences"][name+":exact_score"]:+.6f} | {later["mean_loss_differences"][name+":one_x_two"]:+.6f} |')
    report+=['', '| 主比较块长 | 比分NLL差95%条件区间 | 胜平负LL差95%条件区间 |', '|---|---|---|']
    for block,entry in later['block_intervals'].items():
        def ci(key):
            lo,hi=entry['interval_95'][later['difference_order'].index('common_blend_minus_common_goal:'+key)]
            return f'[{lo:+.6f}, {hi:+.6f}]'
        report.append(f'| {block}周 | {ci("exact_score")} | {ci("one_x_two")} |')
    report+=['', '## 解释范围', '',
             '这个实际对照移除了训练ID和权重不同这一具体混杂因素，并不消除历史xG的发布时间／修订未知、两季赛果已被查看、参数与模型选择不确定性未纳入条件区间等限制。未把一次历史改善当作未来命中率保证。', '',
             '保留固定旧权重与早季重估权重两条输出，不依据后季结果选一条再改报告。逐联赛及全部次指标完整保留，不加联赛专属权重，不改变原预测或LLM／市场线路。', '',
             '没有新增hash、冻结contract或发布门禁；新增拟合用于测量明确的数据处理混杂，普通ID、类型和算术测试本身不能替代这项实测。所有产物与临时文件在项目目录。', '']
    (target/'共同训练记录融合说明.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps({'directory':str(target),'new_goal_fits':new_count,'reused_goal_fits':reused,
        'old_weight':old_weight,'new_weight':new_weight,'later_n':later['n'],
        'scores':later['scores'],'differences':later['mean_loss_differences'],
        'main_intervals':{b:{k:e['interval_95'][later['difference_order'].index('common_blend_minus_common_goal:'+k)]
            for k in ('exact_score','one_x_two')} for b,e in later['block_intervals'].items()}},ensure_ascii=False))


if __name__=='__main__':
    main()
