"""Independent historical xG line, scored against real goals; no blending."""
from datetime import date,datetime,timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import numpy as np

from football_model.native_experiment import season_directories,load_seasons
from football_model.forecast_diagnostics import moving_block_means
from football_model.probability import score_grid,summarize
from football_model.xg_history import join_understat,DateTeamXgModel
from run_shot_strength_20261025 import load_originals
from run_shot_increment_20261025 import score_details,write

ROOT=REPOSITORY_ROOT
SOURCE=ROOT/'data/understat/history_20261024T170717354684Z'


def main():
    now=datetime.now(timezone.utc)
    target=ROOT/'outputs'/now.strftime('独立历史xG前推_%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True,exist_ok=False)
    plan={
        'created_at':now.isoformat(),'scope':'previously accepted goals-primary, xG separate comparison',
        'source':'Understat individual historical EPL matches 2022/23..2025/26',
        'source_directory':str(SOURCE), 'new_live_results_reviewed':False,
        'model':'team attack/defense and home Poisson log-link fitted with nonnegative fractional xG quasi-loss',
        'hypothesis':'predicted xG mean used directly as future goal Poisson rate; evaluated rather than assumed correct',
        'preset':{'regularization':.01,'half_life_days':180,'update_days':28,'rho':0},
        'budget':{'new_xg_fits':22,'new_goal_fits':0,'fusion_coefficients':0,'parameter_searches':0},
        'availability':'strictly earlier calendar date xG assumed available at each original goal-model cutoff',
        'comparison':'same original EPL fixtures for 2024/25 and 2025/26, no goal-to-xG fit feedback',
        'primary':'full-time actual complete-score NLL',
        'secondary':['actual 1X2 LL/Brier/top1','actual total 0/1/2/3/4+ LL','away-minus-two three-way LL'],
        'uncertainty':'paired calendar noncircular blocks 2/4/8 weeks, 2000 repeats; descriptive conditional intervals',
        'not_inputs':['same-match xG','Understat forecast','final season aggregates','future xG','market probabilities'],
        'limitations':['historical xG currently retrieved may have provider revisions; original publication times unknown',
                       'already-viewed score seasons; not a fresh blind or prospective test',
                       'xG source is a model output, neither true scoring strength nor an integer goal count',
                       'one shared preset reused, not selected as best for fractional xG',
                       'no Championship/cups/player news or unknown-team zero imputation',
                       'conditional independent goal Poisson; no parameter uncertainty integration'],
        'adoption':'independent research only; preserve primary goal model and all prospective predictions',
    }
    write(target/'分析前计划.json',plan)
    records,reports=load_seasons(season_directories(ROOT))
    epl=[r for r in records if r.competition=='E0']
    matches={r.match_id:r for r in epl}
    xg_records,joins=[],{}
    for year in (2022,2023,2024,2025):
        season=f'{year}/{str(year+1)[-2:]}'
        raw=json.loads((SOURCE/f'EPL_{year}.json').read_text(encoding='utf-8'))
        parsed,report=join_understat(raw['dates'],epl,season)
        joins[season]=report
        xg_records.extend(parsed)
        if report['missing_independent_match_ids'] or report['exact_matches']!=380:
            write(target/'逐季数据联结.json',joins)
            raise ValueError('Resolve source fixture missingness before comparing xG and goals')
    write(target/'逐季数据联结.json',joins)
    write(target/'逐场xG与独立比分.json',[{
        'match_id':r.match_id,'date':r.match_date.isoformat(),'home':r.home,'away':r.away,
        'xg':list(r.xg),'independently_verified_goals':list(r.goals),
        'understat_id':r.understat_id,'source_url':r.source_url,
    } for r in xg_records])
    originals,paths=load_originals(matches)
    models,new,excluded={},[],[]
    for row in originals:
        cutoff=row['fit_before_date']
        if cutoff not in models:
            models[cutoff]=DateTeamXgModel().fit(xg_records,date.fromisoformat(cutoff))
        model=models[cutoff]
        if {row['home'],row['away']}-model.teams:
            excluded.append({'match_id':row['match_id'],'date':row['date'],'reason':'unseen_xg_team'})
            continue
        rates=model.predict(row['home'],row['away'],'E0',date.fromisoformat(row['date']))
        forecast=summarize(score_grid(*rates,rho=0))
        new.append({**row,'original_rates':row['rates'],'rates':list(rates),'forecast':forecast,
                    'probabilities':list(forecast['one_x_two'].values())})
    if len(models)!=22:
        raise ValueError('Unexpected number of fixed xG fits')
    original_map={r['match_id']:r for r in originals}
    periods={}
    for period in ('2024-25','2025-26'):
        current=[r for r in new if r['period']==period]
        old=[original_map[r['match_id']] for r in current]
        old_scores,old_losses=score_details(old)
        new_scores,new_losses=score_details(current)
        order=list(new_losses)
        differences=np.column_stack([new_losses[k]-old_losses[k] for k in order])
        intervals={str(b):moving_block_means([r['date'] for r in old],differences,block_weeks=b)
                   for b in (2,4,8)}
        windows=[]
        for cutoff in sorted({r['fit_before_date'] for r in old}):
            idx=[i for i,r in enumerate(old) if r['fit_before_date']==cutoff]
            windows.append({'cutoff':cutoff,'n':len(idx),
                            'mean_loss_differences':dict(zip(order,differences[idx].mean(axis=0).tolist()))})
        periods[period]={'original_full_n':sum(r['period']==period for r in originals),
                         'common_n':len(current),'old_scores':old_scores,'xg_scores':new_scores,
                         'difference_order':order,
                         'mean_loss_differences':dict(zip(order,differences.mean(axis=0).tolist())),
                         'block_intervals':intervals,'windows':windows}
    result={'plan':plan,'finished_at':datetime.now(timezone.utc).isoformat(),
            'source_matches':len(xg_records),'join_reports':joins,'original_paths':paths,
            'new_xg_fits':len(models),'new_goal_fits':0,'parameter_searches':0,'fusion_coefficients':0,
            'exclusions':excluded,'seasons':periods,
            'xg_parameters':{k:m.parameters() for k,m in models.items()},
            'new_predictions':new,'original_predictions':originals,'adopted':False}
    write(target/'独立xG前推结果.json',result)
    report=['# 历史xG独立对照：以真实进球检验', '',
            '本轮只推进模型研究，没有复核任何当前比赛赛果。按既有“进球为主、xG单独对照”安排，接通四季Understat逐场xG，不融合、不增加校准系数、不改变原预测。', '',
            '四季各380场，共1520场；日期、显式球队名映射、完整比分与Football-Data逐场精确一致。Understat同场forecast和赛季汇总未输入。源响应和实际接收时刻完整保存，但接收时刻不是原始历史发布时刻。', '',
            '模型用较早xG估计球队攻击、防守与主场系数。xG是非负小数，拟合使用Poisson准损失，不将它称为整数xG计数似然；把其预测均值用于下一场进球率，是本次待检验假设。最后评分目标始终为真实完整比分。', '',
            f'实际xG拟合{len(models)}次，进球模型重训0次、参数搜索0次、融合系数0个。固定.01／180与原28天截止，训练最多只用严格更早日期的xG。', '',
            '| 赛季／指标 | 原进球模型 | 独立xG线 | xG减原 |', '|---|---:|---:|---:|']
    for period,item in periods.items():
        for domain in ('exact_score','one_x_two','total_goals_0_1_2_3_4plus','away_minus_two_three_way'):
            report.append(f'| {period}／{domain} LL | {item["old_scores"][domain]["log_loss"]:.6f} | {item["xg_scores"][domain]["log_loss"]:.6f} | {item["mean_loss_differences"][domain]:+.6f} |')
        report.append(f'| {period}／1X2最高概率命中 | {item["old_scores"]["one_x_two"]["accuracy"]:.2%} | {item["xg_scores"]["one_x_two"]["accuracy"]:.2%} | — |')
    report+=['', '每季原378场，报告各自共同覆盖；不足部分不填零。LL越低越好。两季均完整呈现，不把较好赛季挑作采用依据。', '',
             '| 赛季／块长 | 比分LL差95%描述区间 | 胜平负LL差95%描述区间 |', '|---|---|---|']
    for period,item in periods.items():
        for block,entry in item['block_intervals'].items():
            def bounds(key):
                lo,hi=entry['interval_95'][item['difference_order'].index(key)]
                return f'[{lo:+.6f}, {hi:+.6f}]'
            report.append(f'| {period}／{block}周 | {bounds("exact_score")} | {bounds("one_x_two")} |')
    report+=['', '## 解释边界', '',
             '历史xG是数据商模型输出，今天下载的值可能经历修订；没有当年的发布快照。这里只用日期级更早比赛可用的假设，不能声称还原精确T−24h资料。两个赛季的赛果此前已看过，属于探索而非新盲测。移动块区间条件于已保存预测，未重演所有拟合或选择不确定性。', '',
             '固定参数沿用是为了有界研究，不代表它对xG最优。没有英冠、杯赛和球员信息，不把未知队或缺阵影响填零。xG包含射门场景质量信息，但也有数据商偏差；不能把xG直接当真值或保证增准。', '',
             '本轮没有学融合权重或按后季调参；保持原进球主线与所有赛前记录。若后续要把xG加入主模型，需要另行讨论结构，并用更晚、未开场数据检验。', '',
             '参考：[penaltyblog Understat读取方式](https://penaltyblog.readthedocs.io/en/latest/scrapers/understat.html)；数据来自各季Understat公开getLeagueData响应。底层数据权利不随代码许可转移，本项目保留个人研究用途。', '']
    (target/'独立xG前推研究说明.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps({'directory':str(target),'joins':{s:r['exact_matches'] for s,r in joins.items()},
                      'fits':len(models),'exclusions':excluded,'seasons':{
                          p:{'n':r['common_n'],'differences':r['mean_loss_differences'],
                             'old_top1':r['old_scores']['one_x_two']['accuracy'],
                             'xg_top1':r['xg_scores']['one_x_two']['accuracy'],
                             'intervals_4weeks':r['block_intervals']['4']['interval_95']}
                          for p,r in periods.items()}},ensure_ascii=False))


if __name__=='__main__':
    main()
