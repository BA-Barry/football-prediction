"""One approved global goal/xG weight, estimated earlier and held later."""
from datetime import date,datetime,timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import numpy as np

from football_model.goal_xg_blend import fit_goal_xg_weight,predict_goal_xg_rates
from football_model.forecast_diagnostics import moving_block_means
from football_model.probability import score_grid,summarize
from run_shot_increment_20261025 import score_details,write

ROOT=REPOSITORY_ROOT
CROSS=ROOT/'outputs/独立xG跨联赛复现_20261024T172741074372Z/跨联赛xG复现结果.json'
EPL=ROOT/'outputs/独立历史xG前推_20261024T171031805341Z/独立xG前推结果.json'


def paired_summary(tracks):
    if len({tuple(r['match_id'] for r in rows) for rows in tracks.values()})!=1:
        raise ValueError('Identical ordered fixtures required')
    scores,losses={},{}
    for arm,rows in tracks.items():
        scores[arm],losses[arm]=score_details(rows)
    columns,values=[],[]
    for arm,reference in (('blend','original'),('blend','xg'),('xg','original')):
        for domain in losses[arm]:
            columns.append(f'{arm}_minus_{reference}:{domain}')
            values.append(losses[arm][domain]-losses[reference][domain])
    matrix=np.column_stack(values)
    first=next(iter(tracks.values()))
    return {'n':len(first),'scores':scores,'difference_order':columns,
            'mean_loss_differences':dict(zip(columns,matrix.mean(axis=0).tolist())),
            'block_intervals':{str(b):moving_block_means([r['date'] for r in first],matrix,block_weeks=b)
                               for b in (2,4,8)},
            'match_ids':[r['match_id'] for r in first]}


def main():
    now=datetime.now(timezone.utc)
    target=ROOT/'outputs'/now.strftime('进球xG单权重融合_%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True,exist_ok=False)
    plan={'created_at':now.isoformat(),'user_approved':'2026-10-25 single global historical goal/xG rate blend',
          'structure':'log mu_blend=(1-w)*log mu_goal+w*log mu_xg; one w in [0,1] for all leagues and both scoring sides',
          'estimation':'mean full-time complete-score goal NLL over 2024/25 out-of-time paired predictions',
          'before_date':'2025-08-01','evaluation':'2025/26, same earlier-selected w throughout',
          'training_weighting':'uniform per fixture, no second half-life or new regularization',
          'solver':'convex scalar objective; endpoint derivative tests or bracketed derivative root, no grid',
          'unchanged':['all original team coefficients','all xG coefficients','Poisson score structure rho=0',
                       'all prospective forecasts','market/LLM tracks and weights'],
          'budget':{'new_scalar_weight_fits':1,'new_team_fits':0,'parameter_grids':0},
          'primary':'later same-fixture complete-score NLL versus original goal model',
          'secondary':['same-fixture 1X2 LL/Brier/Top1','total-goals LL','away-minus-two three-way LL','comparison to pure xG'],
          'uncertainty':'paired noncircular calendar blocks 2/4/8 weeks,2000 repeats, conditional descriptive intervals',
          'limitations':['earlier and later outcomes already viewed; exploratory, not new blind validation',
                         'weight-estimation uncertainty not included in conditional intervals',
                         '15 ordinary source-date conflicts removed from xG history, present in original goal history',
                         'historical xG availability/revisions unknown, date-level convention only',
                         'same source provider across leagues, not independent provider replication'],
          'new_live_results_reviewed':False,'adoption':'research only, no automatic replacement or winner by later season'}
    write(target/'分析前计划.json',plan)
    cross=json.loads(CROSS.read_text(encoding='utf-8'))
    epl=json.loads(EPL.read_text(encoding='utf-8'))
    originals={r['match_id']:r for r in cross['original_predictions']+epl['original_predictions']}
    xg=sorted(cross['new_predictions']+epl['new_predictions'],key=lambda r:(r['date'],r['match_id']))
    inputs=[]
    for row in xg:
        old=originals[row['match_id']]
        for field in ('date','competition','home','away','actual_score','fit_before_date'):
            if row[field]!=old[field]:
                raise ValueError(f'Goal/xG row identity conflict: {field}')
        if not row['fit_before_date']<=row['date']:
            raise ValueError('Both rate sources must be from original earlier cutoff')
        inputs.append({'match_id':row['match_id'],'date':row['date'],
                       'goal_rates':old['rates'],'xg_rates':row['rates'],'actual_score':row['actual_score']})
    if len(inputs)!=3467:
        raise ValueError('Expected full common historical cohort')
    parameters=fit_goal_xg_weight(inputs,date(2025,8,1))
    if parameters['n']!=1734 or not parameters['training_max_date']<'2025-08-01':
        raise ValueError('Single weight trained on wrong period or coverage')
    weight=parameters['weight']
    blend=[]
    for row in xg:
        original=originals[row['match_id']]
        rates=predict_goal_xg_rates(original['rates'],row['rates'],weight)
        forecast=summarize(score_grid(*rates,rho=0))
        blend.append({**row,'goal_rates':original['rates'],'xg_rates':row['rates'],'rates':list(rates),
                      'forecast':forecast,'probabilities':list(forecast['one_x_two'].values()),
                      'shared_xg_weight':weight})
    xg_map={r['match_id']:r for r in xg}
    seasons,leagues={},{}
    for period in ('2024-25','2025-26'):
        selected=[r for r in blend if r['period']==period]
        tracks={'original':[originals[r['match_id']] for r in selected],
                'xg':[xg_map[r['match_id']] for r in selected],'blend':selected}
        seasons[period]=paired_summary(tracks)
        seasons[period]['role']='weight estimation / apparent fit' if period=='2024-25' else 'held earlier weight / exploratory later evaluation'
        leagues[period]={}
        for competition in ('E0','D1','SP1','I1','F1'):
            leagues[period][competition]=paired_summary({arm:[r for r in rows if r['competition']==competition]
                                                        for arm,rows in tracks.items()})
    result={'plan':plan,'finished_at':datetime.now(timezone.utc).isoformat(),
            'parameters':parameters,'new_scalar_fits':1,'new_team_fits':0,'parameter_grids':0,
            'source_results':[str(CROSS),str(EPL)],'seasons':seasons,'by_league':leagues,
            'blend_predictions':blend,'weight_training_rows':[r for r in inputs if r['date']<'2025-08-01'],
            'adopted':False}
    write(target/'进球xG单权重融合结果.json',result)
    report=['# 真实进球＋xG：单共享权重研究', '',
            '用户同意后实际估计一个五联赛、主客共享权重。融合只作用于预期进球率的对数，不另改比分分布、不新增截距。原模型、独立xG、市场与LLM线路保留。赛果复核继续暂停。', '',
            f'较早2024/25共同1734场学习到xG权重 **{weight:.6f}**，原进球权重{1-weight:.6f}。较晚2025/26共同1733场全程保持该值，未使用后季选择权重。', '',
            '公式：μ融合=μ进球^(1−w)×μxG^w。它是预期进球率的几何融合，不是两个比分概率分布各抽一部分。仍由同一独立Poisson比分网格导出胜平负、总球与让球。', '',
            '只估计1个标量，球队重训0次、参数网格0个。目标是每场完整比分NLL平均值，采用端点导数或凸目标导数求根；不强制两线都占权重。', '',
            '| 阶段／指标 | 原进球 | 独立xG | 单权重融合 | 融合减原 |', '|---|---:|---:|---:|---:|']
    for period,item in seasons.items():
        for domain in ('exact_score','one_x_two','total_goals_0_1_2_3_4plus','away_minus_two_three_way'):
            report.append(f'| {period}／{domain} LL | {item["scores"]["original"][domain]["log_loss"]:.6f} | {item["scores"]["xg"][domain]["log_loss"]:.6f} | {item["scores"]["blend"][domain]["log_loss"]:.6f} | {item["mean_loss_differences"]["blend_minus_original:"+domain]:+.6f} |')
        report.append('| '+period+'／最高概率命中 | '+' | '.join(f'{item["scores"][a]["one_x_two"]["accuracy"]:.2%}' for a in ('original','xg','blend'))+' | — |')
    report+=['', '较早季用于估计权重，分数属于表观拟合；后季只检验固定权重。差值负值较好，完整比分为事前主指标。', '',
             '| 后季块长 | 融合减原比分LL区间 | 融合减原胜平负LL区间 |', '|---|---|---|']
    later=seasons['2025-26']
    for block,entry in later['block_intervals'].items():
        def show(key):
            lo,hi=entry['interval_95'][later['difference_order'].index('blend_minus_original:'+key)]
            return f'[{lo:+.6f}, {hi:+.6f}]'
        report.append(f'| {block}周 | {show("exact_score")} | {show("one_x_two")} |')
    report+=['', '| 后季联赛 | 场数 | 融合减原比分LL | 融合减原胜平负LL |', '|---|---:|---:|---:|']
    for comp,item in leagues['2025-26'].items():
        report.append(f'| {comp} | {item["n"]} | {item["mean_loss_differences"]["blend_minus_original:exact_score"]:+.6f} | {item["mean_loss_differences"]["blend_minus_original:one_x_two"]:+.6f} |')
    report+=['', '## 采用边界', '',
             '单权重只由较早季估计，不能看到后季后重选；分联赛评分只作描述，不临时给德甲、法甲等单独权重。两季赛果原本已看过，所以这是探索研究，不能包装成新盲测。移动块区间条件于该已估权重及已存预测，没有包含权重重估不确定性。', '',
             '两条输入共用相同原截止日期，但xG源排除15条普通日期冲突、原进球训练仍可能含它们，训练观察量并非完全相同；不能把全部差异纯因果解释成xG特征价值。源xG历史发布时点／修订也未知。', '',
             '本轮没有自动替换正式模型或更改已有20场预测。只有后续更晚资料提供可迁移证据，才讨论正式采用；概率损失改善与投注价值、收益仍不同。', '']
    (target/'进球xG单权重融合说明.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps({'directory':str(target),'weight':weight,'training_n':parameters['n'],
        'gradient':parameters['gradient'],'method':parameters['method'],
        'seasons':{p:{'n':i['n'],'scores':i['scores'],'diffs':i['mean_loss_differences'],
                      '4weekCI':i['block_intervals']['4']['interval_95']} for p,i in seasons.items()}},ensure_ascii=False))


if __name__=='__main__':
    main()
