"""Withhold recent observed matches; measure sensitivity, never rescore targets.

The published random-tree model and its preprocessing stay fixed. Two existing
goal/xG expert fits per scenario isolate the additional expert-input pathway.
This is not a causal estimate, a hyperparameter search, or an accuracy test.
"""
from collections import defaultdict
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import statistics

import joblib

from football_model.batch_projection import _history
from football_model.date_model import DateTeamGoalModel
from football_model.fixed_horizon_features import PreparedPredictionFeatures
from football_model.xg_history import DateTeamXgModel
from current_rich_model_helpers_20261026 import load_current_rich_sources
from refresh_current_all_competition_20261026 import load_current_schedule
from run_current_rich_forecasts_20261026 import ROOT, INPUT, CHINESE, current_independent, prob, read, write

PUBLISHED=ROOT/'outputs/当前丰富信息预测_20261026T030805654404Z'


def compare(p, q):
    old=max(range(3),key=p.__getitem__)
    new=max(range(3),key=q.__getitem__)
    ordered=sorted(p,reverse=True)
    return {'max_abs_change_pp':100*max(abs(a-b) for a,b in zip(p,q)),
            'changes_pp':[100*(b-a) for a,b in zip(p,q)],
            'original_top_class':old,'withheld_top_class':new,'top_class_changed':old!=new,
            'original_top_margin_pp':100*(ordered[0]-ordered[1]),
            'original_top_probability_change_pp':100*(q[old]-p[old])}


def summary(rows, key):
    values=[r[key]['max_abs_change_pp'] for r in rows]
    return {'matches':len(rows),'mean_per_match_max_abs_change_pp':statistics.mean(values),
            'median_per_match_max_abs_change_pp':statistics.median(values),
            'max_abs_change_pp':max(values),
            'top_class_changed_n':sum(r[key]['top_class_changed'] for r in rows)}


def main():
    published=read(PUBLISHED/'完整预测与来源.json')
    F=date.fromisoformat(published['as_of'][:10])
    cutoff=datetime.fromisoformat(published['as_of'])
    experts=joblib.load(PUBLISHED/'当前基专家.joblib')
    tree=joblib.load(PUBLISHED/'丰富信息候选模型.joblib')
    schedule,schedule_report=load_current_schedule()
    supplement=Path(schedule_report['directory'])/'epl_oct24_25_source_observations.json'
    current,_,current_report=current_independent(F,supplement)
    historical,_=_history(ROOT,cutoff)
    goal_rows=historical+current
    rich=load_current_rich_sources(INPUT/'understat/EPL_2026.json',current,before_date=F)
    xg_rows=[r for r in rich['np_records'].values() if r.competition=='E0']
    targets=published['records']
    teams={f'football-data:E0:{r[k]}' for r in targets for k in ('home_label','away_label')}
    by_team=defaultdict(list)
    for r in goal_rows:
        if r.match_date<F:
            for t in (r.home,r.away):
                if t in teams: by_team[t].append(r)
    for rows in by_team.values(): rows.sort(key=lambda r:(r.match_date,r.match_id))
    # Check inference against an already published record before interventions.
    dates=[datetime.fromisoformat(r['fixture']['kickoff']).date() for r in targets]
    original_features=[r['current_feature_details']['performance_features'] for r in targets]
    reproduced=tree.predict_many(original_features,dates)
    if any(max(abs(a-b) for a,b in zip(means,r['rich_candidate']['rates']))>1e-12
           for means,r in zip(reproduced,targets)):
        raise ValueError('Published model and published feature pairs differ')
    before_parameters=tree.parameters()
    scenarios=[]
    for k in (1,2):
        selected={t:[r.match_id for r in rows[-k:]] for t,rows in by_team.items()}
        removed={mid for ids in selected.values() for mid in ids}
        team_reports=[]
        for t,rows in sorted(by_team.items()):
            weights=[math.exp(-math.log(2)*(F-r.match_date).days/180) for r in rows]
            chosen=set(selected[t])
            team_reports.append({'team':t,'total_league_history_n':len(rows),
                'selected_last_k_ids':selected[t],
                'actually_removed_ids':[r.match_id for r in rows if r.match_id in removed],
                'actual_removed_n':sum(r.match_id in removed for r in rows),
                'selected_last_k_share_of_team_observation_weight':
                    sum(w for w,r in zip(weights,rows) if r.match_id in chosen)/sum(weights),
                'interpretation':'Share of weighted team match observations, not coefficient influence or forecast probability weight.'})
        kept_goal=[r for r in goal_rows if r.match_id not in removed]
        kept_xg=[r for r in xg_rows if r.match_id not in removed]
        new_goal=DateTeamGoalModel(.01,180).fit(kept_goal,F)
        new_xg=DateTeamXgModel(.01,180).fit(kept_xg,F)
        kept_observations=[o for o in rich['observations'] if o.match_id not in removed]
        builder=PreparedPredictionFeatures(kept_observations,schedule)
        feature_only=[]
        all_paths=[]
        row_details=[]
        for original,D in zip(targets,dates):
            h,a=(f'football-data:E0:{original[k]}' for k in ('home_label','away_label'))
            frozen_goal=experts['goal'].predict(h,a,'E0',D)
            frozen_xg=experts['xg'].predict(h,a,'E0',D)
            adjusted_goal=new_goal.predict(h,a,'E0',D)
            adjusted_xg=new_xg.predict(h,a,'E0',D)
            kwargs=dict(home=h,away=a,competition='E0',parameter_cutoff=F,
                forecast_cutoff=F,match_date=D,refresh=True)
            A=builder.for_match(**kwargs,goal_rates=frozen_goal,xg_rates=frozen_xg)
            B=builder.for_match(**kwargs,goal_rates=adjusted_goal,xg_rates=adjusted_xg)
            feature_only.append(A['performance_features'])
            all_paths.append(B['performance_features'])
            newly_missing=[]
            for side,(old_values,new_values) in enumerate(zip(
                    original['current_feature_details']['performance_features'],B['performance_features'])):
                newly_missing.extend({'side':side,'feature':name} for name,value in new_values.items()
                                     if value is None and old_values[name] is not None)
            simple=prob(adjusted_goal)['probabilities']
            row_details.append({'fixture':original['fixture'],'home_label':original['home_label'],
                'away_label':original['away_label'],'original_rich':original['rich_candidate']['probabilities'],
                'original_simple':original['refreshed_simple']['probabilities'],
                'withheld_simple':simple,
                'simple_change':compare(original['refreshed_simple']['probabilities'],simple),
                'newly_missing_features':newly_missing,'withheld_history':B['feature_history']})
        predictions_A=tree.predict_many(feature_only,dates)
        predictions_B=tree.predict_many(all_paths,dates)
        for r,meansA,meansB in zip(row_details,predictions_A,predictions_B):
            r['withheld_rich_features_only']=prob(meansA)['probabilities']
            r['withheld_rich_all_input_paths']=prob(meansB)['probabilities']
            r['feature_only_change']=compare(r['original_rich'],r['withheld_rich_features_only'])
            r['all_input_paths_change']=compare(r['original_rich'],r['withheld_rich_all_input_paths'])
        scenarios.append({'k':k,'whole_fixture_ids_removed':sorted(removed),
            'unique_removed_fixtures_n':len(removed),'team_removal_reports':team_reports,
            'new_fits':2,'goal_training_n':new_goal.n_matches,'xg_training_n':new_xg.parameters()['n_matches'],
            'summary':{key:summary(row_details,key) for key in
                ('simple_change','feature_only_change','all_input_paths_change')},'records':row_details})
    if tree.parameters()!=before_parameters:
        raise ValueError('Fixed published learner was changed')
    if read(PUBLISHED/'完整预测与来源.json')!=published:
        raise ValueError('Published forecast was changed')
    output=ROOT/'outputs'/f'近两场敏感性测量_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}'
    output.mkdir(parents=True)
    result={'created_at':datetime.now(timezone.utc).isoformat(),'published_source':str(PUBLISHED),
        'information_cutoff_held_at':published['as_of'],'class_order':['home','draw','away'],
        'scenarios':scenarios,'new_base_expert_fits':4,'new_tree_fits':0,'parameter_searches':0,
        'prior_failed_reporting_attempt_base_fits':2,'total_base_fits_including_reporting_retry':6,
        'published_predictions_unchanged':True,'tree_and_preprocessing_fixed':True,
        'no_target_outcomes_scored':True,
        'limits':['Withholding removes both result and performance information, replaces rolling-window members with older observations, and changes counts; it is not an isolated score or causal form effect.',
            'Latest k selection uses each team then whole-fixture union. Actual removed counts are reported and can exceed k.',
            'Both experts keep the original fitting date and fixed hyperparameters. The tree and its fitted imputer are unchanged.',
            'The all-input-path scenario does not retrain the meta-model and does not include LLM sensitivity.',
            'Probability sensitivity does not establish accuracy benefit or harm. No justified universal percentage-point threshold is asserted.',
            'No future target result, market movement, injury or lineup is used. No published forecast is replaced.']}
    write(output/'敏感性完整测量.json',result)
    lines=['# 最近1—2场对当前预测的敏感性测量','',
        '固定10月26日已发布的随机树及其预处理，使用相同10场未来目标；未复核赛果，也未改原预测。',
        '', 'A：仅从近期描述特征隐藏最新比赛，保留原进球／xG基专家。B：同时从基专家的拟合样本移除这些整场比赛，重新拟合固定设置的两个基专家，随机树不变。',
        '', '| 隐藏信息 | 简单模型平均／最大变化 | 候选A平均／最大变化 | 候选B平均／最大变化 | 候选B最高单项改变 |',
        '|---|---:|---:|---:|---:|']
    for s in scenarios:
        def pair(key):
            v=s['summary'][key]
            return f"{v['mean_per_match_max_abs_change_pp']:.2f}／{v['max_abs_change_pp']:.2f}"
        lines.append(f"| 每队最近{s['k']}场 | {pair('simple_change')} | {pair('feature_only_change')} | {pair('all_input_paths_change')} | {s['summary']['all_input_paths_change']['top_class_changed_n']}/10 |")
    lines+=['','变化单位为百分点。每场先取主胜／平局／客胜三项中最大的绝对变化，再跨10场求平均或最大；不是准确率变化。',
        '', '| 比赛 | 原候选主／平／客 | 隐藏最近2场并重估基专家后 | 最大变化（百分点） |',
        '|---|---:|---:|---:|']
    for r in scenarios[1]['records']:
        def pct(p): return '／'.join(f'{100*x:.1f}' for x in p)
        name=CHINESE[r['home_label']]+'—'+CHINESE[r['away_label']]
        lines.append(f"| {name} | {pct(r['original_rich'])} | {pct(r['withheld_rich_all_input_paths'])} | {r['all_input_paths_change']['max_abs_change_pp']:.2f} |")
    lines+=['','## 权重机制及解释边界','',
        '180天半衰期意味着一周新旧之差只有约2.73%的相对样本权重；这是拟合样本权重，不是预测概率权重。',
        '', '近5场算术均值中两场占2/5，近15场、5减15趋势和主客子集也包含这些比赛。它们不是独立证据，不能把各份额相加；随机树没有可直接读出的“最近两场最终占比”。',
        '', '移除某些比赛会让更早比赛补入窗口，也可能使主客子样本不足3场而转为缺失。完整JSON记录每队实际删除数量及新增缺失。当前测量因此回答“对这些信息有多敏感”，不能回答“若比赛换成另一个比分，球队真实实力会改变多少”。',
        '', '是否应该有这种敏感度，要看预先固定规则后在较晚比赛上的概率评分和可靠性；不以概率变化越小越好，也不凭一两场输赢调整权重。本次未测LLM的提示与采样敏感性。',
        '', '成功测量使用4次固定基专家拟合；首次运行在读取xG样本数字段时出错，额外执行了2次后修正报告字段。合计6次基专家拟合、0次随机树训练、0轮调参；原预测、参数与目标结果均未改。']
    (output/'最近两场影响说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(output),'summary':[{'k':s['k'],'removed':s['unique_removed_fixtures_n'],
        'summary':s['summary'],'new_missing_n':sum(len(r['newly_missing_features']) for r in s['records']),
        'removed_per_team':sorted(set(r['actual_removed_n'] for r in s['team_removal_reports'])),
        'observation_weight_share_min_max':[min(r['selected_last_k_share_of_team_observation_weight'] for r in s['team_removal_reports']),max(r['selected_last_k_share_of_team_observation_weight'] for r in s['team_removal_reports'])]}
        for s in scenarios]},ensure_ascii=False))


if __name__=='__main__': main()
