"""Publish a separate, current research comparison for the existing future ten.

Reuses fixed estimators. Does not settle, replace, or tune old predictions.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
import csv
import json
import math
import os

import joblib

from football_model.batch_projection import _history
from football_model.date_model import DateTeamGoalModel
from football_model.football_data import HistoricalMatch
from football_model.probability import score_grid, summarize
from football_model.xg_history import DateTeamXgModel
from football_model.fixed_horizon_features import PreparedPredictionFeatures
from football_model.prospective import _time

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
INPUT = ROOT/'data/current_rich_inputs/run_20261026T025507813598Z'
OLD = ROOT/'data/prematch-intelligence/20261024_batch/run_20261024T150900467071Z/batch_case.json'
FEATURES = ROOT/'outputs/固定发布日期信息刷新_20261025T071806561562Z/features_lead3.json.gz'
FPL_NAMES = {'Coventry City': 'Coventry', 'Hull City': 'Hull', 'Ipswich Town': 'Ipswich',
             'Man Utd': 'Man United', 'Spurs': 'Tottenham'}
CHINESE = {'Chelsea':'切尔西','Man United':'曼联','Bournemouth':'伯恩茅斯','Leeds':'利兹联',
           'Brentford':'布伦特福德',"Nott'm Forest":'诺丁汉森林','Coventry':'考文垂',
           'Sunderland':'桑德兰','Hull':'赫尔城','Ipswich':'伊普斯维奇','Man City':'曼城',
           'Brighton':'布莱顿','Tottenham':'热刺','Crystal Palace':'水晶宫',
           'Aston Villa':'阿斯顿维拉','Fulham':'富勒姆','Liverpool':'利物浦',
           'Arsenal':'阿森纳','Newcastle':'纽卡斯尔','Everton':'埃弗顿'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    def convert(x):
        if isinstance(x, Path): return str(x)
        if isinstance(x, (date,datetime)): return x.isoformat()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,default=convert)+'\n', encoding='utf-8')


def current_independent(before_date, supplement_path=None):
    """Use official completed scores; cross-check every overlapping FD row."""
    bootstrap=read(INPUT/'official-fpl/bootstrap.json')
    names={r['id']:FPL_NAMES.get(r['name'],r['name']) for r in bootstrap['teams']}
    fixtures=read(INPUT/'official-fpl/fixtures.json')
    rows=[]
    official={}
    for r in fixtures:
        if not r['finished'] or not r.get('kickoff_time'):
            continue
        d=datetime.fromisoformat(r['kickoff_time'].replace('Z','+00:00')).date()
        if d>=before_date:
            continue
        h,a=names[r['team_h']],names[r['team_a']]
        score=(r['team_h_score'],r['team_a_score'])
        if any(type(x) is not int or x<0 for x in score):
            raise ValueError('Official completed fixture without valid score')
        key=(d,h,a)
        if key in official:
            raise ValueError('Duplicate official fixture')
        official[key]=score
        rows.append(HistoricalMatch(
            match_id=f'football-data:E0:2026/27:{h}|{a}',competition='E0',season='2026/27',
            match_date=d,source_time_text=r['kickoff_time'],
            home=f'football-data:E0:{h}',away=f'football-data:E0:{a}',home_label=h,away_label=a,
            full_time=score,half_time=None,markets={},source_path=str(INPUT/'official-fpl/fixtures.json'),
            source_url='https://fantasy.premierleague.com/api/fixtures/',goal_fit_exclusion=None))
    fpl_count=len(rows)
    supplement_count=0
    if supplement_path is not None:
        supplement=read(supplement_path)
        espn_names={'AFC Bournemouth':'Bournemouth','Brighton & Hove Albion':'Brighton',
            'Coventry City':'Coventry','Hull City':'Hull','Ipswich Town':'Ipswich',
            'Leeds United':'Leeds','Manchester City':'Man City','Manchester United':'Man United',
            'Newcastle United':'Newcastle','Nottingham Forest':"Nott'm Forest",'Tottenham Hotspur':'Tottenham'}
        for r in supplement['records']:
            d=date.fromisoformat(r['match_date'])
            if d>=before_date or not r['completed'] or r['status_name'] not in ('STATUS_FULL_TIME','STATUS_FINAL'):
                raise ValueError('Supplement needs earlier completed regulation-time league fixtures')
            h,a=espn_names.get(r['home_name'],r['home_name']),espn_names.get(r['away_name'],r['away_name'])
            raw_score=(r['home_score'],r['away_score'])
            if any(not (type(x) is int or isinstance(x,str) and x.isdecimal()) for x in raw_score):
                raise ValueError('Supplement score requires nonnegative integer text or value')
            score=tuple(int(x) for x in raw_score)
            if any(type(x) is not int or x<0 for x in score):
                raise ValueError('Supplement score missing')
            key=(d,h,a)
            if key in official:
                if official[key]!=score: raise ValueError('Conflicting completed scores')
                continue
            official[key]=score
            rows.append(HistoricalMatch(
                match_id=f'football-data:E0:2026/27:{h}|{a}',competition='E0',season='2026/27',
                match_date=d,source_time_text=r['date_utc_text'],
                home=f'football-data:E0:{h}',away=f'football-data:E0:{a}',home_label=h,away_label=a,
                full_time=score,half_time=None,markets={},source_path=str(supplement_path),
                source_url=r.get('source_url') or next(u for u in supplement['source_urls'] if d.strftime('%Y%m%d') in u),goal_fit_exclusion=None))
            supplement_count+=1
    checked=0
    with (INPUT/'football-data/E0.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if not r.get('FTHG') or not r.get('FTAG'):
                continue
            d=datetime.strptime(r['Date'],'%d/%m/%Y').date()
            if d>=before_date:
                continue
            key=(d,r['HomeTeam'],r['AwayTeam'])
            score=(int(r['FTHG']),int(r['FTAG']))
            if official.get(key)!=score:
                raise ValueError(f'Football-Data and official score mismatch: {key}')
            checked+=1
    return rows, names, {'completed_matches':len(rows),'fpl_completed_matches':fpl_count,
                        'espn_completed_supplement':supplement_count,'fd_overlapping_scores_checked':checked,
                        'verified_supplement_to_fd':len(rows)-checked,
                        'training_max_date':max(r.match_date for r in rows).isoformat(),
                        'source':'https://fantasy.premierleague.com/api/fixtures/'}


def prob(rates):
    forecast=summarize(score_grid(*rates,rho=0))
    p=[forecast['one_x_two'][k] for k in ('home','draw','away')]
    if any(not math.isfinite(x) or x<0 or x>1 for x in p) or abs(sum(p)-1)>1e-10:
        raise ValueError('Invalid class probabilities')
    return {'rates':list(rates),'probabilities':p,'forecast':forecast}


def validate_future_targets(records, at_time):
    """Check the existing pre-kickoff rule at start and at generation completion."""
    cutoff = _time(at_time, 'prediction time')
    if not records:
        raise ValueError('Need the explicit nonempty future target batch')
    kickoffs = [_time(row['fixture']['kickoff'], 'target kickoff') for row in records]
    if any(kickoff <= cutoff for kickoff in kickoffs):
        raise ValueError('Prediction is no longer pre-match')
    return [(kickoff.date()-cutoff.date()).days for kickoff in kickoffs]


def archived_target_confirmation(previous, confirmations, as_of):
    """Retain actual snapshot age, without pretending an archive was refetched."""
    records = previous['records']
    observations = confirmations['fixture_observations']
    if (confirmations['confirmed_unstarted_count'] != len(records)
            or len(observations) != len(records)):
        raise ValueError('Future fixture verification incomplete')
    for old, observed in zip(records, observations, strict=True):
        if (old['fixture'] != observed['original_fixture']
                or observed['confirmed_unstarted'] is not True):
            raise ValueError('Target identity or status mismatch')
    observed_at = _time(confirmations['checked_at'], 'fixture confirmation observed_at')
    started_at = _time(as_of, 'prediction start')
    if observed_at > started_at:
        raise ValueError('Fixture confirmation was observed after prediction start')
    leads = validate_future_targets(records, started_at)
    return {'observed_at': observed_at.isoformat(),
            'age_seconds_at_start': (started_at-observed_at).total_seconds(),
            'live_status_refetched': False,
            'publication_lead_days_at_start': leads}


def current_feature_snapshot(builder, **kwargs):
    """Reuse the existing same-day path; it still excludes all same-day results."""
    method = (builder.for_same_day_match
              if kwargs['forecast_cutoff'] == kwargs['match_date'] else builder.for_match)
    return method(**kwargs)


def main():
    from current_rich_model_helpers_20261026 import load_current_rich_sources, train_current_candidate
    from refresh_current_all_competition_20261026 import load_current_schedule

    as_of=datetime.now(timezone.utc)
    F=as_of.date()
    previous=read(OLD)
    confirmations=read(INPUT/'official-fpl/confirmed_future_fixtures.json')
    confirmation_info=archived_target_confirmation(previous,confirmations,as_of)
    lead_min=min(confirmation_info['publication_lead_days_at_start'])
    lead_max=max(confirmation_info['publication_lead_days_at_start'])
    schedule,schedule_report=load_current_schedule()
    supplement_path=Path(schedule_report['directory'])/'epl_oct24_25_source_observations.json'
    current,names,current_report=current_independent(F,supplement_path)
    historical,history_sources=_history(ROOT,as_of)
    goal_records=historical+current
    # A shared current-date fit makes the simple and rich inputs comparable.
    goal=DateTeamGoalModel(.01,180).fit(goal_records,F)
    rich_sources=load_current_rich_sources(INPUT/'understat/EPL_2026.json',current,before_date=F)
    xg_records=[r for r in rich_sources['np_records'].values() if r.competition=='E0']
    xg=DateTeamXgModel(.01,180).fit(xg_records,F)
    builder=PreparedPredictionFeatures(rich_sources['observations'],schedule)
    # Both are already researched fixed feature sets. Current schedule coverage
    # is unresolved, so the performance-only line is the usable candidate; the
    # allcomp variant is retained solely to show the missing-input sensitivity.
    candidate,candidate_metadata=train_current_candidate(FEATURES,before_date=F,
        feature_key='fresh_performance_features')
    allcomp_candidate,allcomp_metadata=train_current_candidate(FEATURES,before_date=F)
    output=ROOT/'outputs'/f'当前丰富信息预测_{as_of.strftime("%Y%m%dT%H%M%S%fZ")}'
    output.mkdir(parents=True,exist_ok=False)
    compiled=[]
    factual=[]
    feature_pairs=[]
    allcomp_pairs=[]
    dates=[]
    for old,confirmed in zip(previous['records'],confirmations['fixture_observations'],strict=True):
        fixture=old['fixture']
        if fixture!=confirmed['original_fixture'] or not confirmed['confirmed_unstarted']:
            raise ValueError('Target identity or status mismatch')
        kickoff=_time(fixture['kickoff'],'target kickoff')
        if kickoff<=as_of:
            raise ValueError('Prediction is no longer pre-match')
        h,a=names[confirmed['home_team_id']],names[confirmed['away_team_id']]
        home,away=f'football-data:E0:{h}',f'football-data:E0:{a}'
        D=kickoff.date()
        means=goal.predict(home,away,'E0',D)
        xg_means=xg.predict(home,away,'E0',D)
        features=current_feature_snapshot(builder,home=home,away=away,competition='E0',
            parameter_cutoff=F,forecast_cutoff=F,match_date=D,goal_rates=means,xg_rates=xg_means,refresh=True)
        feature_pairs.append(features['performance_features'])
        allcomp_pairs.append(features['allcomp_features'])
        dates.append(D)
        compiled.append({'fixture':fixture,'home_label':h,'away_label':a,
            'original_20261024':old['tracks']['statistical'],
            'refreshed_simple':prob(means),'current_feature_details':features,
            'xg_expert_means':list(xg_means),'actual_outcome':None})
        factual.append({'fixture':fixture,'home_label':h,'away_label':a,
            'as_of':as_of.isoformat(),'history_features':[
                {k:v for k,v in side.items() if not k.startswith('log_goal_') and not k.startswith('log_xg_')}
                for side in features['allcomp_features']],
            'allcomp_history':features['allcomp_history'],
            'feature_history':features['feature_history'],
            'limits':['No verified current injury bulletin, tactical news or expected starting XI is included.',
                      'These are descriptive earlier-match observations, not demonstrated causal player impacts.']})
    predictions=candidate.predict_many(feature_pairs,dates)
    allcomp_predictions=allcomp_candidate.predict_many(allcomp_pairs,dates)
    for row,means,allcomp_means in zip(compiled,predictions,allcomp_predictions):
        row['rich_candidate']=prob(means)
        row['allcomp_missing_input_diagnostic']=prob(allcomp_means)
        row['candidate_minus_same_day_simple_pp']=[100*(x-y) for x,y in zip(
            row['rich_candidate']['probabilities'],row['refreshed_simple']['probabilities'])]
        row['direction_agreement']=max(range(3),key=row['rich_candidate']['probabilities'].__getitem__)==max(
            range(3),key=row['refreshed_simple']['probabilities'].__getitem__)
    joblib.dump(candidate,output/'丰富信息候选模型.joblib')
    joblib.dump(allcomp_candidate,output/'全赛事缺失输入诊断模型.joblib')
    joblib.dump({'goal':goal,'xg':xg},output/'当前基专家.joblib')
    restored=joblib.load(output/'丰富信息候选模型.joblib')
    again=restored.predict_many(feature_pairs,dates)
    if any(any(abs(a-b)>1e-12 for a,b in zip(x,y)) for x,y in zip(predictions,again)):
        raise ValueError('Saved model does not reproduce predictions')
    if read(OLD)!=previous:
        raise ValueError('Original forecast changed during the run')
    limits=[
        'The rich candidate reuses the existing fixed ExtraTrees settings. This run performs no parameter search.',
        f'Existing historical meta-training examples use a three-day publication lead. Current targets have {lead_min}-{lead_max} calendar-day leads; transfer to other leads is not established by that training.',
        'This is the first live candidate initialization on the current date, with parameter date equal to feature date; it does not claim to reproduce the historical 28-day release schedule.',
        'The candidate meta-training maximum is reported separately from the fresh base experts and feature histories. No current-season meta-training examples were silently fabricated.',
        'Historical source revisions and date-only feature availability remain limitations; download times are real observations, not first-publication times.',
        'Past-only schedule features do not include intervening future cup or European games before the target kickoff.',
        'Current all-competition count windows could not be independently confirmed (0/20 usable). The main candidate uses the existing 121 performance features; the 161-feature version is only a missing-input diagnostic, not a claimed full-workload forecast.',
        'No new LLM, executable current market quote, player-strength effect, injury effect or fused probability is implied by this statistical output.',
        'No target outcomes are used or settled. Forecast changes alone do not establish improved accuracy.']
    generated_at=datetime.now(timezone.utc)
    validate_future_targets(compiled,generated_at)
    result={'as_of':as_of.isoformat(),'generated_at':generated_at.isoformat(),
            'target_confirmation':confirmation_info,
            'target_count':len(compiled),'class_order':['home','draw','away'],
            'new_fits':{'simple_and_base_goal':1,'base_xg':1,'rich_candidate':1,'allcomp_missing_input_diagnostic':1},'parameter_searches':0,
            'current_score_coverage':current_report,'historical_goal_sources':history_sources,
            'rich_source_report':rich_sources['reports'],'schedule_report':schedule_report,
            'simple_parameters':goal.parameters(),'xg_parameters':xg.parameters(),
            'candidate_parameters':candidate.parameters(),'candidate_training':candidate_metadata,
            'allcomp_diagnostic_training':allcomp_metadata,
            'old_source_unchanged':True,'score_distribution':'independent_poisson_rho0_shared',
            'records':compiled,'limits':limits}
    write(output/'完整预测与来源.json',result)
    write(output/'LLM独立输入.json',{'as_of':as_of.isoformat(),'class_order':['home','draw','away'],
          'records':factual,'limits':limits[:6]})
    write(output/'逐场当前特征.json',{'as_of':as_of.isoformat(),'records':[
        {'fixture':r['fixture'],'features':r['current_feature_details']} for r in compiled]})
    write(output/'运行核验.json',{'probability_axes_and_sums':True,'future_targets':True,
          'future_targets_checked_at':generated_at.isoformat(),
          'archived_target_confirmation':confirmation_info,
          'same_current_cutoff':True,'saved_model_reproduces':True,'old_source_unchanged':True,
          'target_outcomes_used':False,'source_current_scores':current_report})
    lines=['# 下一轮10场：旧记录、同日刷新与丰富信息候选','',
           f'生成时间：{generated_at.isoformat()}；本次信息计算截止：{as_of.isoformat()}。三项依次为主胜／平局／客胜，单位%。',
           f"赛程状态档案观察时间：{confirmation_info['observed_at']}；本次没有重新联网确认状态。",
           '','| 比赛 | 原10/24统计 | 今日简单模型 | 今日丰富候选 |',
           '|---|---:|---:|---:|']
    for r in compiled:
        def pct(p): return ' / '.join(f'{100*x:.1f}' for x in p)
        title=CHINESE[r['home_label']]+'—'+CHINESE[r['away_label']]
        lines.append(f"| {title} | {pct(r['original_20261024']['probabilities'])} | {pct(r['refreshed_simple']['probabilities'])} | {pct(r['rich_candidate']['probabilities'])} |")
    lines+=['','本次新增4次固定设置拟合，没有调参。原始记录未更改，没有复核或结算本批目标赛果。',
            '',f"FPL与ESPN已完成比分共{len(current)}场，与Football-Data重叠{current_report['fd_overlapping_scores_checked']}场逐一同日、同队、同比分核对；80场再与Understat逐一连接。",
            '', '主要候选使用真实进球、xG/NPxG、PPDA、deep、近期与主客场。全赛事窗口尚未确认，121项表现版作为主要候选；161项版本只保存为缺失输入诊断。球员影响和伤停没有手工换算为概率。',
            '', f'候选仍属前瞻研究：历史训练以提前3天为主，本次提前{lead_min}—{lead_max}天；跨提前量的表现尚未由该训练证明。全赛事计数不等于疲劳量，也不包含此刻之后到开赛前的赛程。',
            '', '没有把旧LLM或旧赔率冒充本次更新；没有新增未经验证的平均融合概率。','', '## 使用限制','']
    lines += [f'- {x}' for x in limits]
    (output/'十场完整胜平负对照.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(output),'current_scores':current_report,
        'candidate_training':{k:candidate_metadata[k] for k in ('kind','n_matches','training_min_date','training_max_date','source_feature_key')},'predictions':[{'home':r['home_label'],'away':r['away_label'],
        'simple':r['refreshed_simple']['probabilities'],'rich':r['rich_candidate']['probabilities']} for r in compiled]},ensure_ascii=False))


if __name__=='__main__':
    main()
