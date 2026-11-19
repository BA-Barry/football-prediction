"""Describe saved historical forecasts by pre-match recent-form conflict.

No fitting, forecast regeneration, parameter selection, or current settlement.
Groups use only earlier source matches and their already saved earlier means.
"""
from collections import Counter
from datetime import date, datetime, timezone
import gc
import gzip
import json
import math
from pathlib import Path

import numpy as np

from football_model.forecast_diagnostics import moving_block_means

ROOT=Path('G:/Codex/football-prediction-2026-10-22')
SOURCE=ROOT/'outputs/固定发布日期信息刷新_20261025T071806561562Z'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def sign(x): return 1 if x>0 else -1 if x<0 else 0


def slim_features(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        payload=json.load(f)['rows']
    selected={}
    for r in payload:
        if r['match_id'] in selected: raise ValueError('Duplicate source feature ID')
        selected[r['match_id']]={k:r[k] for k in ('match_id','date','period','competition','home','away',
            'fit_before_date','forecast_before_date','actual_score','actual_outcome','goal_rates')}
        selected[r['match_id']]['sides']=[{k:side[k] for k in (
            'own_history_count_5','own_history_count_15','own_npxg_for_mean_5','own_npxg_against_mean_5',
            'own_npxg_for_mean_15','own_npxg_against_mean_15')} for side in r['fresh_performance_features']]
        selected[r['match_id']]['history']=r['fresh_feature_history']['history']
    del payload
    gc.collect()
    return selected


def classify(r,index):
    L=r['goal_rates'][0]-r['goal_rates'][1]
    residuals=[]
    references=[]
    season_crossing=False
    ages=[]
    for side,team in zip(('home','away'),(r['home'],r['away'])):
        ids=r['history'][side]['window_match_ids']['5']
        if len(ids)!=5:
            return {'group':'unknown','reason':'fewer_than_five_observations','long_direction':L}
        if any(mid not in index for mid in ids):
            return {'group':'unknown','reason':'earlier_match_forecast_missing','long_direction':L,
                    'missing_reference_ids':[mid for mid in ids if mid not in index]}
        local=[]
        for mid in ids:
            past=index[mid]
            if not past['fit_before_date']<=past['forecast_before_date']<past['date']<r['forecast_before_date']:
                raise ValueError('Recent residual must already be observable before target publication')
            if past['competition']!=r['competition'] or team not in (past['home'],past['away']):
                raise ValueError('History reference identity differs')
            direction=1 if team==past['home'] else -1
            local.append(direction*((past['actual_score'][0]-past['actual_score'][1])-
                                    (past['goal_rates'][0]-past['goal_rates'][1])))
            season_crossing|=past['period']!=r['period']
            references.append(mid)
        residuals.append(math.fsum(local)/5)
        last=max(index[mid]['date'] for mid in ids)
        ages.append((date.fromisoformat(r['forecast_before_date'])-date.fromisoformat(last)).days)
    R=residuals[0]-residuals[1]
    group='conflict' if sign(L)*sign(R)<0 else 'aligned' if sign(L)*sign(R)>0 else 'zero_direction'
    quality=None
    quality_group='unavailable_full_15'
    if all(s['own_history_count_15']==15 and all(s[k] is not None for k in (
            'own_npxg_for_mean_5','own_npxg_against_mean_5',
            'own_npxg_for_mean_15','own_npxg_against_mean_15')) for s in r['sides']):
        acceleration=[]
        for s in r['sides']:
            short=s['own_npxg_for_mean_5']-s['own_npxg_against_mean_5']
            medium=s['own_npxg_for_mean_15']-s['own_npxg_against_mean_15']
            prior10=(15*medium-5*short)/10
            acceleration.append(short-prior10)
        quality=acceleration[0]-acceleration[1]
        quality_group=('quality_confirms_result_swing' if sign(quality)*sign(R)>0 else
                       'quality_opposes_result_swing' if sign(quality)*sign(R)<0 else 'zero_quality_or_result')
    return {'group':group,'long_direction':L,'recent_home_residual_mean':residuals[0],
            'recent_away_residual_mean':residuals[1],'recent_relative_residual':R,
            'recent_supported_class':0 if R>0 else 2 if R<0 else None,
            'quality_disjoint_5_vs_prior10_relative_acceleration':quality,'quality_group':quality_group,
            'referenced_earlier_fixture_ids':references,
            'recent_five_crosses_season':season_crossing,'max_latest_observation_age_days':max(ages)}


def score(p,y):
    if len(p)!=3 or any(not math.isfinite(x) or not 0<x<1 for x in p) or abs(sum(p)-1)>1e-10:
        raise ValueError('Invalid saved class probability')
    return -math.log(p[y]),sum((x-int(i==y))**2 for i,x in enumerate(p))


def summarize(rows,intervals=False):
    if not rows: return {'n':0}
    ll_g=np.asarray([r['goal_ll'] for r in rows]);ll_r=np.asarray([r['rich_ll'] for r in rows])
    br_g=np.asarray([r['goal_brier'] for r in rows]);br_r=np.asarray([r['rich_brier'] for r in rows])
    result={'n':len(rows),'goal_log_loss':float(ll_g.mean()),'rich_log_loss':float(ll_r.mean()),
        'delta_log_loss_rich_minus_goal':float((ll_r-ll_g).mean()),
        'goal_brier':float(br_g.mean()),'rich_brier':float(br_r.mean()),
        'delta_brier_rich_minus_goal':float((br_r-br_g).mean()),
        'goal_top1_n':sum(r['goal_top']==r['actual_outcome'] for r in rows),
        'rich_top1_n':sum(r['rich_top']==r['actual_outcome'] for r in rows),
        'top_class_switch_n':sum(r['goal_top']!=r['rich_top'] for r in rows),
        'earlier_season_in_recent_five_n':sum(r['signal'].get('recent_five_crosses_season',False) for r in rows),
        'competitions':dict(Counter(r['competition'] for r in rows))}
    valid=[r for r in rows if r['signal'].get('recent_supported_class') is not None]
    if valid:
        observed=np.asarray([int(r['actual_outcome']==r['signal']['recent_supported_class']) for r in valid])
        p0=np.asarray([r['goal_probabilities'][r['signal']['recent_supported_class']] for r in valid])
        p1=np.asarray([r['rich_probabilities'][r['signal']['recent_supported_class']] for r in valid])
        result['recent_supported_single_win_event']={
            'n':len(valid),'actual_frequency':float(observed.mean()),
            'goal_mean_probability':float(p0.mean()),'rich_mean_probability':float(p1.mean()),
            'rich_minus_goal_mean_probability_pp':float((p1-p0).mean()*100),
            'goal_overprediction_pp':float((p0-observed).mean()*100),
            'rich_overprediction_pp':float((p1-observed).mean()*100),
            'rich_moves_toward_recent_event_n':int((p1>p0).sum()),
            'rich_moves_away_from_recent_event_n':int((p1<p0).sum()),
            'top_switch_to_recent_event_n':sum(r['goal_top']!=r['rich_top'] and
                r['rich_top']==r['signal']['recent_supported_class'] for r in valid)}
    if intervals:
        matrix=np.column_stack((ll_r-ll_g,br_r-br_g))
        result['conditional_4week_block_loss_interval']=moving_block_means(
            [r['date'] for r in rows],matrix,block_weeks=4)
    return result


def analyze(lead,out):
    index=slim_features(SOURCE/f'features_lead{lead}.json.gz')
    tracks=read(SOURCE/f'lead_{lead}/逐场完整预测.json')
    goal={r['match_id']:r for r in tracks['goal']}
    rich={r['match_id']:r for r in tracks['fresh_performance']}
    if len(goal)!=len(tracks['goal']) or set(goal)!=set(rich):
        raise ValueError('Matched saved tracks required')
    results=[]
    for mid,g in goal.items():
        r=rich[mid];f=index[mid]
        for key in ('date','period','competition','home','away','actual_outcome','actual_score'):
            if g[key]!=r[key] or g[key]!=f[key]: raise ValueError('Historical identity conflict')
        y=g['actual_outcome']
        if y!=(0 if g['actual_score'][0]>g['actual_score'][1] else 2 if g['actual_score'][0]<g['actual_score'][1] else 1):
            raise ValueError('Outcome axis mismatch')
        p0,p1=g['probabilities'],r['probabilities']
        ll0,b0=score(p0,y);ll1,b1=score(p1,y)
        results.append({k:g[k] for k in ('match_id','date','period','competition','home','away','actual_outcome')} |
            {'goal_probabilities':p0,'rich_probabilities':p1,'goal_ll':ll0,'rich_ll':ll1,
             'goal_brier':b0,'rich_brier':b1,'goal_top':max(range(3),key=p0.__getitem__),
             'rich_top':max(range(3),key=p1.__getitem__),'signal':classify(f,index)})
    seasons={}
    for season in ('2024-25','2025-26'):
        rows=[r for r in results if r['period']==season]
        groups={name:[r for r in rows if r['signal']['group']==name] for name in
                ('conflict','aligned','zero_direction','unknown')}
        seasons[season]={'overall':summarize(rows),'groups':{k:summarize(v,intervals=k in ('conflict','aligned'))
            for k,v in groups.items()},
            'unknown_reasons':dict(Counter(r['signal'].get('reason') for r in groups['unknown'])),
            'conflict_quality_auxiliary':{q:summarize([r for r in groups['conflict'] if r['signal']['quality_group']==q])
                for q in ('quality_confirms_result_swing','quality_opposes_result_swing','zero_quality_or_result','unavailable_full_15')}}
    write(out/f'提前{lead}日_逐场分组与误差.json',{'lead':lead,'records':results})
    del tracks,index,results
    gc.collect()
    return seasons


def main():
    now=datetime.now(timezone.utc)
    out=ROOT/'outputs'/f'近期长期冲突诊断_{now.strftime("%Y%m%dT%H%M%S%fZ")}'
    out.mkdir(parents=True)
    plan={'created_at':now.isoformat(),'source':str(SOURCE),'primary_lead_days':3,'check_lead_days':1,
        'periods':['2024-25','2025-26'],'candidate':'fresh_performance_121_existing_saved_predictions',
        'reference':'goal_existing_saved_predictions','grouping':
            'L=target goal home mean-away mean. Each side residual=mean of last5(actual net goals-earlier saved own-fixture goal mean net), oriented to that team. R=home residual-away residual. conflict when L*R<0;aligned when >0;zero separately.',
        'missing':'Require all ten team-match references and their legal earlier forecasts;unknown separate,never fill0.',
        'primary_metrics':['paired rich-minus-goal LogLoss','paired rich-minus-goal Brier'],
        'event_diagnostic':'Recent R supports one single win event (home or away), never double chance. Report probability change and probability minus observed event frequency.',
        'auxiliary':'Only within conflict, compare sign of R with relative NPxG acceleration, latest5 versus disjoint previous10. Require each side full15, otherwise unknown.',
        'intervals':'Reuse conditional 4-calendar-week moving block means, 2000 draws;descriptive,not a new untouched test.',
        'stop':'Finish these saved-data groups;no threshold search,extra grouping/model,fit or current settlement.',
        'new_fits':0,'new_predictions':0,'parameter_searches':0,'current_results_reviewed':False,
        'limits':['Previously viewed historical seasons;not new blind validation.',
                  'Recent residual controls opposition and venue only as far as the saved goal model did;not causal strength or pure outcome luck.',
                  'Rich-versus-goal differs in more than recent form, including xG and nonlinear mapping.',
                  'Last5 can span summer/personnel changes;past-season references retained and counted.',
                  'NPxG auxiliary trend is descriptive and not adjusted for past opponents.',
                  'Lead1 and lead3 overlap matches and are not independent replications.',
                  'These groups do not measure LLM sensitivity or confirm current5-7day deployment performance.']}
    write(out/'分组方法与范围.json',plan)
    horizons={}
    for lead in (3,1):
        horizons[str(lead)]=analyze(lead,out)
        write(out/'完整分组汇总.json',{'method':plan,'horizons':horizons})
        print(json.dumps({'lead':lead,'seasons':{s:{g:{k:v for k,v in m.items() if k in
            ('n','delta_log_loss_rich_minus_goal','delta_brier_rich_minus_goal','recent_supported_single_win_event')}
            for g,m in results['groups'].items()} for s,results in horizons[str(lead)].items()}},ensure_ascii=False),flush=True)
    lines=['# 近期信号与较长期模型冲突：已有历史预测诊断','',
        '本轮直接使用已保存的逐场预测；0次拟合、0次新预测、0轮调参，没有复核当前比赛赛果。历史年份已被查看，属于解释性分组，不是新盲测。',
        '', '近期信号：对双方各最近5场，用真实净胜球减去该场当时较早模型预期净胜球，统一球队方向后取均值。两队差值R与目标较长期进球模型主客预期差L相反时，归为冲突。缺任何更早预测时保留未知。',
        '', '### 主要结果：提前3日','',
        '| 赛季 | 分组 | 场数 | 简单模型LL | 丰富候选LL | LL差（负为候选更好） | Brier差 |',
        '|---|---|---:|---:|---:|---:|---:|']
    labels={'conflict':'冲突','aligned':'同向','unknown':'不可诊断','zero_direction':'零方向'}
    for season,results in horizons['3'].items():
        for group in ('conflict','aligned','unknown','zero_direction'):
            m=results['groups'][group]
            if not m['n']: continue
            lines.append(f"| {season} | {labels[group]} | {m['n']} | {m['goal_log_loss']:.5f} | {m['rich_log_loss']:.5f} | {m['delta_log_loss_rich_minus_goal']:+.5f} | {m['delta_brier_rich_minus_goal']:+.5f} |")
    lines+=['','### 冲突组：模型是否向近期方向抬高概率','',
        '| 提前量／赛季 | 场数 | 近期支持单胜概率变化（百分点） | 候选该事件均值 | 实际发生频率 | 候选高估（百分点） |',
        '|---|---:|---:|---:|---:|---:|']
    for lead in ('3','1'):
        for season,results in horizons[lead].items():
            m=results['groups']['conflict'];v=m['recent_supported_single_win_event']
            lines.append(f"| {lead}日／{season} | {m['n']} | {v['rich_minus_goal_mean_probability_pp']:+.2f} | {100*v['rich_mean_probability']:.2f}% | {100*v['actual_frequency']:.2f}% | {v['rich_overprediction_pp']:+.2f} |")
    lines+=['','### 同定义提前1日检查','',
        '| 赛季 | 分组 | 场数 | LL差 | Brier差 |','|---|---|---:|---:|---:|']
    for season,results in horizons['1'].items():
        for group in ('conflict','aligned'):
            m=results['groups'][group]
            lines.append(f"| {season} | {labels[group]} | {m['n']} | {m['delta_log_loss_rich_minus_goal']:+.5f} | {m['delta_brier_rich_minus_goal']:+.5f} |")
    lines+=['','### 冲突组中的机会质量旁证','',
        '比较最近5场与此前不重叠10场的NPxG净值变化，不把嵌套窗口当独立证据。它没有单独校正此前对手，只作旁证。',
        '', '| 提前量／赛季 | 机会质量与比分残差信号 | 场数 | LL差 | Brier差 |','|---|---|---:|---:|---:|']
    qlabels={'quality_confirms_result_swing':'同向','quality_opposes_result_swing':'相反',
             'zero_quality_or_result':'零方向','unavailable_full_15':'不足完整15场'}
    for lead in ('3','1'):
        for season,results in horizons[lead].items():
            for q,m in results['conflict_quality_auxiliary'].items():
                if m['n']:
                    lines.append(f"| {lead}日／{season} | {qlabels[q]} | {m['n']} | {m['delta_log_loss_rich_minus_goal']:+.5f} | {m['delta_brier_rich_minus_goal']:+.5f} |")
    lines+=['','### 解释范围','',
        '模型向近期方向修正、该单胜事件平均高估、整体概率评分变差，是过度修正的风险信号；需看两季方向和区间，不能只看其中一个数字。',
        '', '“丰富候选减简单模型”同时包含xG、非线性映射和其他描述信息，不能将分组差异全部归因于最近1—2场。本文近期信号使用最近5场，也不等于前一轮隐藏最近两场的实验。',
        '', '完整JSON保留4周时间分块条件区间、全部未知、跨季历史数量及每场引用。两个提前量包含大量相同比赛，不能相加当成独立样本。LLM没有历史同口径概率记录，本轮不对其下结论。',
        '', '本轮到这里停止扩展分组或参数；原赛前概率、当前比赛和原历史预测均不改动。']
    (out/'近期与长期冲突分析.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(out),'new_fits':0,'new_predictions':0,'parameter_searches':0},ensure_ascii=False))


if __name__=='__main__': main()
