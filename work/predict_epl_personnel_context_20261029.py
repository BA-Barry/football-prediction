"""Current ten EPL fixtures: two fixed fits and inference from the saved 121 tree.

Earlier completed league observations are training inputs, never a settlement of
an old prediction. Old batch reads project fixture identity fields only.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import date, datetime, timezone
import csv
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import joblib
import numpy as np
from scipy.stats import poisson

from football_model.all_competition_features import AllCompetitionFeatures
from football_model.batch_projection import _history
from football_model.date_model import DateTeamGoalModel
from football_model.early_history_features import MEASUREMENTS
from football_model.fixed_horizon_features import PreparedPredictionFeatures
from football_model.football_data import HistoricalMatch
from football_model.probability import score_grid, summarize
from football_model.xg_history import DateTeamXgModel
from current_rich_model_helpers_20261026 import load_current_rich_sources

ROOT = REPOSITORY_ROOT
INPUT = ROOT/'data/epl_personnel_live_20261029'
SOURCE = INPUT/'stat_sources'
OUT = ROOT/'outputs/英超人员协作前瞻_20261029'
SAVED_TREE = ROOT/'outputs/当前丰富信息预测_20261026T030805654404Z/丰富信息候选模型.joblib'
FIT_DATE = date(2026, 10, 29)
ALIASES = {'Manchester United':'Man United','Man Utd':'Man United','AFC Bournemouth':'Bournemouth',
           'Leeds United':'Leeds','Nottingham Forest':"Nott'm Forest",'Coventry City':'Coventry',
           'Hull City':'Hull','Ipswich Town':'Ipswich','Manchester City':'Man City',
           'Brighton & Hove Albion':'Brighton','Tottenham Hotspur':'Tottenham','Spurs':'Tottenham',
           'Newcastle United':'Newcastle'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        allow_nan=False)+'\n', encoding='utf-8')


def instant(text):
    result = datetime.fromisoformat(text.replace('Z','+00:00'))
    if result.tzinfo is None:
        raise ValueError('Need an aware source time')
    return result.astimezone(timezone.utc)


def canonical(name):
    return ALIASES.get(name,name)


def distribution(rates):
    grid = score_grid(*rates, rho=0)
    matrix, summary = grid.grid, summarize(grid)
    p = [summary['one_x_two'][k] for k in ('home','draw','away')]
    if not np.isfinite(matrix).all() or np.any(matrix<0) or abs(math.fsum(p)-1)>1e-12:
        raise ValueError('Invalid current probabilities')
    totals = grid.total_goals_distribution()
    top = sorted(({'score':f'{h}-{a}','home_goals':h,'away_goals':a,
                   'probability':float(matrix[h,a])}
                  for h in range(matrix.shape[0]) for a in range(matrix.shape[1])),
                 key=lambda r:(-r['probability'],r['home_goals'],r['away_goals']))[:10]
    return {'rates':list(rates),'expected_goals':dict(zip(('home','away'),map(float,rates))),
        'expected_total_goals':float(sum(rates)), 'class_order':['home','draw','away'],
        'probabilities':p,'top_scores':top,'over_2_5':summary['over_2_5'],
        'both_score':summary['both_score'],
        'total_goals_0_to_7_plus':{**{str(k):float(totals[k]) for k in range(7)},'7+':float(totals[7:].sum())},
        'score_grid':{'orientation':'rows=home_goals; columns=away_goals','rho':0,
            'home_goal_axis':list(range(matrix.shape[0])),'away_goal_axis':list(range(matrix.shape[1])),
            'probabilities':matrix.tolist(),'sum':float(matrix.sum()),
            'omitted_probability_before_normalization':float(1-poisson.cdf(matrix.shape[0]-1,rates[0])
                *poisson.cdf(matrix.shape[1]-1,rates[1]))}}


def current_independent():
    teams = read(INPUT/'raw/bootstrap-static.json')['teams']
    names = {row['id']:canonical(row['name']) for row in teams}
    rows, lookup = [], {}
    for source in read(INPUT/'raw/fixtures.json'):
        # Project scores only after strictly earlier completed-fixture filtering.
        if not source.get('finished') or not source.get('kickoff_time'):
            continue
        played = instant(source['kickoff_time']).date()
        if played >= FIT_DATE:
            continue
        h,a = names[source['team_h']],names[source['team_a']]
        score = source['team_h_score'],source['team_a_score']
        if any(type(v) is not int or v<0 for v in score):
            raise ValueError('Earlier official completed fixture has invalid score')
        key = played,h,a
        if key in lookup:
            raise ValueError('Duplicate earlier official fixture')
        lookup[key]=score
        rows.append(HistoricalMatch(match_id=f'football-data:E0:2026/27:{h}|{a}',
            competition='E0',season='2026/27',match_date=played,source_time_text=source['kickoff_time'],
            home='football-data:E0:'+h,away='football-data:E0:'+a,home_label=h,away_label=a,
            full_time=score,half_time=None,markets={},source_path=str(INPUT/'raw/fixtures.json'),
            source_url='https://fantasy.premierleague.com/api/fixtures/',goal_fit_exclusion=None))
    checked,missing = 0,[]
    with (SOURCE/'E0.csv').open(encoding='utf-8-sig',newline='') as handle:
        for raw in csv.DictReader(handle):
            if not raw.get('FTHG') or not raw.get('FTAG'):
                continue
            played=datetime.strptime(raw['Date'],'%d/%m/%Y').date()
            if played>=FIT_DATE:
                continue
            key=played,canonical(raw['HomeTeam']),canonical(raw['AwayTeam'])
            if key not in lookup:
                missing.append({'date':played.isoformat(),'home':key[1],'away':key[2]})
            elif lookup[key]!=(int(raw['FTHG']),int(raw['FTAG'])):
                raise ValueError('Earlier independent source values disagree')
            else:
                checked+=1
    if missing:
        raise ValueError('Earlier FD fixtures missing from official source: '+json.dumps(missing))
    return rows,{'completed_n':len(rows),'overlapping_fd_n':checked,
        'official_supplement_beyond_fd_n':len(rows)-checked,
        'date_min':min(r.match_date for r in rows).isoformat(),
        'date_max':max(r.match_date for r in rows).isoformat(),
        'completed_match_ids':[r.match_id for r in rows],
        'value_crosscheck':'exact independent date/team/score agreement; no old prediction scoring'}


def targets(started):
    verified=read(INPUT/'已核实赛程.json')
    current=read(INPUT/'未来未开场赛程.json')
    bank={(canonical(r['home_source_team_name']),canonical(r['away_source_team_name']),
           instant(r['kickoff_time'])):r for r in current['fixtures']}
    result=[]
    for row in verified['fixtures']:
        h,a=canonical(row['home']),canonical(row['away'])
        key=h,a,instant(row['kickoff_utc'])
        observed=bank.get(key)
        if observed is None or observed['started'] is not False or observed['finished'] is not False:
            raise ValueError('Current official target identities/status do not align')
        if key[2]<=started:
            raise ValueError('Target is no longer future')
        result.append({'fixture_id':row['fixture_id'],'source_fixture_id':observed['source_fixture_id'],
            'home':row['home'],'away':row['away'],'home_model_label':h,'away_model_label':a,
            'kickoff':key[2].isoformat(),'sourceURLs':row['sourceURLs'],
            'observed_at':current['observed_at_utc'],'status':'scheduled_unstarted',
            'home_source_team_id':observed['home_source_team_id'],
            'away_source_team_id':observed['away_source_team_id']})
    return result


def old_fixture_projection():
    paths=[ROOT/'data/prematch-intelligence/20261024_batch/run_20261024T150900467071Z/batch_case.json',
           ROOT/'outputs/当前丰富信息预测_20261026T030805654404Z/完整预测与来源.json']
    result=[]
    for path in paths:
        # Nothing from an old prediction, LLM, market or outcome is returned.
        fixtures=[r['fixture'] for r in read(path)['records']]
        for fixture in fixtures:
            result.append({'source_file':str(path),'fixture':{
                k:v for k,v in fixture.items() if k in ('match_id','home','away','kickoff','competition')}})
    return result


def revision(fixture,previous):
    found=[]
    for old in previous:
        r=old['fixture']
        if (canonical(r['home']),canonical(r['away']),instant(r['kickoff']))==(
                fixture['home_model_label'],fixture['away_model_label'],instant(fixture['kickoff'])):
            found.append(old)
    return {'same_fixture_revision':bool(found),'new_unique_fixture':not bool(found),
            'matched_previous_fixture_references':found,
            'matching_rule':'canonical home/away and exact aware UTC kickoff; fixture metadata only'}


def history_scope(rows):
    return {'n':len(rows),'date_min':min(r.match_date for r in rows).isoformat(),
            'date_max':max(r.match_date for r in rows).isoformat(),
            'season_counts':dict(Counter(r.match_id.split(':')[2] for r in rows)),
            'match_ids':[r.match_id for r in rows]}


def side_facts(observations,team,is_home):
    rows=sorted((r for r in observations if r.competition=='E0' and r.team==team
                 and r.match_date<FIT_DATE),key=lambda r:(r.match_date,r.match_id))
    def window(limit):
        selected=rows[-limit:]
        means={}
        for key in MEASUREMENTS:
            values=[getattr(r,key) for r in selected]
            means[key]=None if not values or any(v is None for v in values) else math.fsum(values)/len(values)
        for key,att,defense in [('ppda_ratio','ppda_att','ppda_def'),
                                ('ppda_allowed_ratio','ppda_allowed_att','ppda_allowed_def')]:
            denominator=[getattr(r,defense) for r in selected]
            numerator=[getattr(r,att) for r in selected]
            known=selected and all(v is not None for v in denominator+numerator)
            means[key]=math.fsum(numerator)/math.fsum(denominator) if known and math.fsum(denominator)>0 else None
        return {'n':len(selected),'requested_n':limit,'measurements':means,
                'history':[{'match_id':r.match_id,'date':r.match_date.isoformat(),
                    'opponent':r.opponent,'is_home':r.is_home} for r in selected]}
    return {'team':team,'history_count':len(rows),'identity_status':'source_joined' if rows else 'unknown',
            'relevant_venue':'home' if is_home else 'away','window_5':window(5),'window_15':window(15),
            'measurement_scope':'observed league form; descriptive, not opponent-adjusted; missing stays null'}


def main():
    started=datetime.now(timezone.utc)
    if started.date()!=FIT_DATE:
        raise ValueError('Run must use its actual UTC date')
    OUT.mkdir(parents=True,exist_ok=True)
    fixtures=targets(started)
    current,coverage=current_independent()
    historical,history_sources=_history(ROOT,started)
    goal_rows=[replace(r,markets={}) for r in historical+current]
    sources=load_current_rich_sources(SOURCE/'EPL_2026.json',current,before_date=FIT_DATE)
    xg_rows=[r for r in sources['np_records'].values() if r.competition=='E0' and r.match_date<FIT_DATE]
    target_ids={f"football-data:E0:2026/27:{r['home_model_label']}|{r['away_model_label']}" for r in fixtures}
    if target_ids & {r.match_id for r in goal_rows+xg_rows}:
        raise ValueError('A target identity entered training')
    print(json.dumps({'stage':'sources_joined','current_n':len(current),'current_xg_n':len(sources['current_np_records']),
        'goal_input_n':len(goal_rows),'xg_input_n':len(xg_rows),'target_n':len(fixtures)}),flush=True)
    goal=DateTeamGoalModel(.01,180).fit(goal_rows,FIT_DATE)
    print(json.dumps({'stage':'goal_fit_complete','n':goal.n_matches,'maxdate':goal.training_max_date.isoformat()}),flush=True)
    xg=DateTeamXgModel(.01,180).fit(xg_rows,FIT_DATE)
    print(json.dumps({'stage':'xg_fit_complete','n':xg.parameters()['n_matches'],
        'maxdate':xg.parameters()['training_max_date']}),flush=True)
    joblib.dump({'goal':goal,'xg':xg},OUT/'本次英超统计模型.joblib')
    tree=joblib.load(SAVED_TREE)
    tree_parameters=tree.parameters()
    if len(tree.feature_names)!=121:
        raise ValueError('Saved current candidate is not the existing 121 performance tree')
    builder=PreparedPredictionFeatures(sources['observations'],AllCompetitionFeatures([]))
    old=old_fixture_projection()
    records,facts,pairs,dates=[],[],[],[]
    for fixture in fixtures:
        home,away=('football-data:E0:'+fixture[k+'_model_label'] for k in ('home','away'))
        d=instant(fixture['kickoff']).date()
        rates=goal.predict(home,away,'E0',d)
        xg_rates=xg.predict(home,away,'E0',d)
        feature=builder.for_match(home=home,away=away,competition='E0',parameter_cutoff=FIT_DATE,
            forecast_cutoff=FIT_DATE,match_date=d,goal_rates=rates,xg_rates=xg_rates,refresh=True)
        if any(len(side)!=121 for side in feature['performance_features']):
            raise ValueError('Current performance inputs differ from the saved candidate feature set')
        pairs.append(feature['performance_features']); dates.append(d)
        records.append({'fixture':fixture,'home_model_identity':home,'away_model_identity':away,
            'revision':revision(fixture,old),'goal_simple':distribution(rates),
            'independent_xg':distribution(xg_rates),'rich_feature_details':feature})
        facts.append({'fixture':fixture,'home_model_identity':home,'away_model_identity':away,
            'feature_before_date':FIT_DATE.isoformat(),'sides':{
                'home':side_facts(sources['observations'],home,True),
                'away':side_facts(sources['observations'],away,False)}})
    rich_rates=tree.predict_many(pairs,dates)
    for record,rates in zip(records,rich_rates,strict=True):
        record['rich_121_research']=distribution(rates)
    restored=joblib.load(OUT/'本次英超统计模型.joblib')
    for r,d in zip(records,dates,strict=True):
        for key,model in [('goal_simple','goal'),('independent_xg','xg')]:
            check=restored[model].predict(r['home_model_identity'],r['away_model_identity'],'E0',d)
            if any(abs(a-b)>1e-12 for a,b in zip(check,r[key]['rates'],strict=True)):
                raise ValueError('Saved model does not reproduce its forecast')
    generated=datetime.now(timezone.utc)
    if any(instant(r['kickoff'])<=generated for r in fixtures):
        raise ValueError('A target started before publication')
    metadata={'new_fits':{'goal_simple':1,'independent_xg':1,'rich_121':0},'parameter_searches':0,
        'historical_backtests_run':0,'fit_before_date':FIT_DATE.isoformat(),
        'goal_input':history_scope(goal_rows),'goal_parameters':goal.parameters(),
        'xg_input':history_scope(xg_rows),'xg_parameters':xg.parameters(),
        'goal_fit_exclusions':[{'match_id':r.match_id,'reason':r.goal_fit_exclusion} for r in goal_rows if r.goal_fit_exclusion],
        'rich_121':{'saved_model_file':str(SAVED_TREE),'new_fit':False,'parameters':tree_parameters,
            'historical_training_lead_days':3,'current_leads_days':sorted(set((d-FIT_DATE).days for d in dates)),
            'current_base_parameter_age_days':0,'lead_transfer_validated_by_this_run':False,
            'missing_handling':'unchanged saved training median and missing indicators; raw current unknowns retain None'},
        'target_exclusion':{'excluded_target_ids':sorted(target_ids),'overlap_with_either_base_training':[]}}
    limits=[
        'Earlier completed league observations are training inputs; no old prediction is scored, settled or changed.',
        'All base training and descriptive windows use date < 2026-10-29. Source receipt times are real observations, not reconstructed first publication times.',
        'The xG quasi-loss predicted means are used as full-goal rates and share the independent Poisson rho=0 law; this remains a research assumption.',
        'The saved 121 performance candidate was trained at lead 3; this run reuses it at lead 2 to 4 with current base parameters, without retraining or validating lead transfer.',
        'No hand-written promoted-team discount, player impact, injury multiplier, market, LLM judgment or model averaging enters these three separate statistical lines.',
        'All-competition history is omitted from the 121 input; empty all-competition diagnostics must not be interpreted as zero workload.',
        'Null performance measurements remain unknown in the factual packet; the saved tree retains its existing training preprocessing.']
    write(OUT/'统计预测.json',{'as_of':started.isoformat(),'generated_at':generated.isoformat(),
        'target_count':len(records),'new_unique_fixture_count':sum(r['revision']['new_unique_fixture'] for r in records),
        'same_fixture_revision_count':sum(r['revision']['same_fixture_revision'] for r in records),
        'class_order':['home','draw','away'],'model_file':str(OUT/'本次英超统计模型.joblib'),
        'source_files':{'official_fixture_projection':str(INPUT/'未来未开场赛程.json'),
            'confirmed_schedule':str(INPUT/'已核实赛程.json'),'fd':str(SOURCE/'E0.csv'),
            'understat':str(SOURCE/'EPL_2026.json'),'official_receipts':str(INPUT/'获取记录.json'),
            'fd_receipt':str(SOURCE/'E0_receipt.json'),'understat_receipt':str(SOURCE/'EPL_2026_receipt.json')},
        'current_coverage':coverage,'historical_goal_sources':history_sources,
        'rich_source_report':sources['reports'],'model_metadata':metadata,'records':records,
        'checks':{'saved_base_models_reproduce':True,'targets_future_at_save':True,'target_labels_used':False,
            'market_inputs_used':False,'llm_inputs_used':False,'current_missing_zero_filled':False},'limits':limits})
    write(OUT/'给LLM的表现事实包.json',{'as_of':started.isoformat(),'generated_at':generated.isoformat(),
        'feature_before_date':FIT_DATE.isoformat(),'records':facts,
        'limits':['Descriptive earlier observed league matches only, no odds, predictions, fitted rates, old LLM or target labels.',
            'Each 5/15 window retains its original actual count, full IDs and dates; no older substitution for missing measurements.',
            'Historical opponent strength and all-competition workload are not estimated by these raw means.',
            'Source publication/revision timing is not reconstructed; current source receipts are recorded separately.']})
    write(OUT/'统计运行证据.json',{'started_at':started.isoformat(),'finished_at':generated.isoformat(),
        'source_receipts':[read(SOURCE/'E0_receipt.json'),read(SOURCE/'EPL_2026_receipt.json')],
        'new_fits':metadata['new_fits'],'new_unique_fixture_count':sum(r['revision']['new_unique_fixture'] for r in records),
        'model_file':str(OUT/'本次英超统计模型.joblib'),'saved_base_models_reproduce':True,
        'training_date_max':{'goal':goal.parameters()['training_max_date'],'xg':xg.parameters()['training_max_date']},
        'all_windows_strictly_earlier':all(q['date']<FIT_DATE.isoformat() for r in facts for side in r['sides'].values()
            for window in ('window_5','window_15') for q in side[window]['history']),
        'target_training_overlap':[],'old_prediction_values_or_labels_read_for_analysis':False})
    print(json.dumps({'stage':'complete','output':str(OUT),'target_n':len(records),
        'goal_n':goal.n_matches,'xg_n':xg.parameters()['n_matches'],
        'same_fixture_revision_n':sum(r['revision']['same_fixture_revision'] for r in records),
        'new_unique_fixture_n':sum(r['revision']['new_unique_fixture'] for r in records),
        'rich_feature_count':len(tree.feature_names)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
