"""One GAP -> ordered-logit historical pipeline, without rescue tuning.

Only the already connected four historical seasons are read. Target outcomes
are used for later historical scoring, never for their own pre-match state.
"""
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time

import joblib

from football_model.gap_model import GAPState, STATISTICS, optimize_gap_updates
from football_model.gap_outcome import GAPOutcomeModel

ROOT=Path('G:/Codex/football-prediction-2026-10-22')
OUT=ROOT/'outputs/GAP胜平负研究_20261025'
SOURCE=ROOT/'outputs/多项进攻统计接入_20261025/历史多项统计.json'
OLD=ROOT/'outputs/直接胜平负分类研究_20261025T091329913808Z/lead_3/逐场胜平负概率.json'
CONTEXT=('match_id','date','period','competition','home','away','actual_score','actual_outcome',
         'fit_before_date','forecast_before_date','horizon_days')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def snapshots(rows,parameters):
    rosters=defaultdict(lambda:defaultdict(set))
    for row in rows:
        rosters[row['competition']][row['season']].update((row['home'],row['away']))
    roster_data={c:{s:sorted(teams) for s,teams in seasons.items()} for c,seasons in rosters.items()}
    state=GAPState(parameters,season_rosters=roster_data)
    queries=[]
    for r in rows:
        if r['season']=='2022/23':
            continue
        f=(date.fromisoformat(r['date'])-timedelta(days=3)).isoformat()
        queries.append((f,r))
    queries.sort(key=lambda pair:(pair[0],pair[1]['match_id']))
    idx=0; output=[]; last_ingested=None
    for f,row in queries:
        while idx<len(rows) and rows[idx]['date']<f:
            state.update(rows[idx]);last_ingested=rows[idx]['date'];idx+=1
        # Query does not receive same-match counts, goals or actual result.
        identity={k:row[k] for k in ('match_id','date','season','competition','home','away')}
        predicted=state.predict(identity)
        if last_ingested is not None and not last_ingested<f:
            raise ValueError('A same-date or later match entered the state')
        output.append({k:row[k] for k in ('match_id','date','competition','home','away','actual_score','actual_outcome')}
            |{'period':row['season'].replace('/','-'),'season':row['season'],
              'forecast_before_date':f,'feature_before_date':f,'horizon_days':3,
              'gap_differences':predicted['differences'],
              'expected_statistics':predicted['expected_statistics'],
              'team_observations':predicted['team_observations'],
              'initialization':predicted['initialization'],
              'state_history_last_date':last_ingested,'state_history_n':idx})
    output.sort(key=lambda r:(r['date'],r['match_id']))
    return output,roster_data

def dependency_receipt():
    packages={}
    folder=OUT/'upstream';folder.mkdir(exist_ok=True)
    for name in ('statsmodels','patsy','formulaic','interface-meta','wrapt'):
        dist=importlib.metadata.distribution(name)
        licenses=[]
        for item in dist.files or []:
            if 'license' in str(item).lower() and '.dist-info' in str(item):
                path=Path(dist.locate_file(item))
                if path.is_file():
                    dest=folder/(name+'_'+path.name)
                    shutil.copyfile(path,dest);licenses.append(str(dest))
        packages[name]={'version':dist.version,'install_path':str(dist.locate_file('')),
                        'license_copies':licenses}
    write(OUT/'复用组件.json',{'packages':packages,
       'GAP_structure_source':'https://journals.sagepub.com/doi/10.3233/JSA-200462',
       'probability_layer_source':'https://www.statsmodels.org/stable/generated/statsmodels.miscmodels.ordinal_model.OrderedModel.html',
       'reuse':'statsmodels native OrderedModel fitting and prediction; local GAP state adapter from published equations',
       'not_copied':['klynch999 gist','pratikkk97 R project'],
       'dependency_install_scope':'new package files and pip cache/temp on G; pre-existing numerical libraries may be read from bundled C runtime'})

def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'models').mkdir(exist_ok=True)
    rows=sorted(read(SOURCE)['rows'],key=lambda r:(r['date'],r['match_id']))
    if len({r['match_id'] for r in rows})!=len(rows):
        raise ValueError('Historical match identities must be unique')
    early=[r for r in rows if r['season']=='2022/23']
    params_path=OUT/'统计更新参数.json'
    plan={'created_at':datetime.now(timezone.utc).isoformat(),
        'instruction_context':'Continue optimization after the previously presented concrete GAP proposal; independent research route, no live promotion.',
        'horizon_days':3,'statistics':list(STATISTICS),'update_parameter_count':12,
        'parameter_fit_scope':'2022/23 only, one shared three-parameter update per statistic across five leagues; league/team states separate',
        'mapping':'four standardized expected home-minus-away statistic differences; native unweighted OrderedModel logit ML, four slopes/two cutpoints',
        'mapping_initial_history':'2023/24 date-level forward features; labels strictly before each parameter cutoff',
        'target_seasons':['2024-25','2025-26'],'publication_dates':'same original lead3 dates and parameter cutoffs',
        'roster_convention':'season club membership assumed known from schedule; no future season results initialize state',
        'state_time_rule':'ingest completed records dated strictly before each F=D-3; label date strictly before P<=F',
        'adaptations':['ML not original least-squares mapping','all 2022 statistic errors included, no original home-six-game burn-in','equal mapping fixture weights, not old 180-day classifier weighting'],
        'stop':'Complete one route; no parameter grids, restarts, secondary BA model, or post-hoc selection by league',
        'limitations':['Previously inspected historical seasons, not a new blind test','First publication/revision times unknown','New-team initialization is a point heuristic, not learned cross-tier uncertainty'],
        'target_results_2026_27_read':False,'target_lineups_read':False,'existing_forecasts_changed':False}
    write(OUT/'本轮安排.json',plan);dependency_receipt()
    if params_path.exists():
        learned=read(params_path)
        print(json.dumps({'stage':'resume_saved_update_parameters','parameter_learning_matches':learned['n_matches']}),flush=True)
    else:
        learned=optimize_gap_updates(early)
        write(params_path,learned)
        print(json.dumps({'stage':'update_parameters_learned',**learned},ensure_ascii=False),flush=True)
    if any(not v['success'] for v in learned['optimization'].values()):
        raise ValueError('Update-parameter learning did not converge; preserve failure, do not rescue tune')
    features_path=OUT/'赛前统计状态.json'
    if features_path.exists():
        feature_rows=read(features_path)['rows']
    else:
        feature_rows,rosters=snapshots(rows,learned['parameters'])
        write(features_path,{'rows':feature_rows,'season_rosters':rosters,
            'convention':'season membership assumed known from published schedule; no outcomes used for membership prior'})
    by_id={r['match_id']:r for r in feature_rows}
    saved=read(OLD);targets={};excluded=[]
    for track in ('goal','regression_allcomp','logistic_allcomp','clean_blend'):
        for r in saved[track]:
            if r['match_id'] not in by_id:
                excluded.append({'match_id':r['match_id'],'track':track,'reason':'absent from previously accepted canonical history'})
                continue
            actual=by_id[r['match_id']]
            if any(actual[k]!=r[k] for k in ('date','competition','home','away','actual_score','actual_outcome','forecast_before_date','period')):
                raise ValueError('Old reference and GAP exact fixture identity/timing disagree')
            if r['match_id'] in targets and targets[r['match_id']]['fit_before_date']!=r['fit_before_date']:
                raise ValueError('Reference parameter dates disagree')
            targets[r['match_id']]={**actual,'fit_before_date':r['fit_before_date']}
    write(OUT/'覆盖安排.json',{'source_matches':len(rows),'update_learning_matches':len(early),
        'feature_rows':len(feature_rows),'feature_rows_by_period':dict(Counter(r['period'] for r in feature_rows)),
        'target_rows':len(targets),'target_rows_by_period':dict(Counter(r['period'] for r in targets.values())),
        'excluded_reference_rows':excluded})
    predictions=[];windows=[]
    for cutoff in sorted({r['fit_before_date'] for r in targets.values()}):
        t0=time.perf_counter()
        history=[r for r in feature_rows if r['date']<cutoff]
        held=sorted((r for r in targets.values() if r['fit_before_date']==cutoff),key=lambda r:(r['date'],r['match_id']))
        fitted=GAPOutcomeModel().fit(history,date.fromisoformat(cutoff))
        path=OUT/'models'/f'ordered_logit_{cutoff}.joblib'
        joblib.dump(fitted,path,compress=3)
        # Predict from the saved object, making the reusable artifact concrete.
        model=joblib.load(path)
        p=model.predict_many(held)
        for r,values in zip(held,p,strict=True):
            predictions.append({k:r[k] for k in CONTEXT}|{
                'feature_before_date':r['feature_before_date'],'probabilities':values.tolist(),
                'training_max_date':history[-1]['date'],'training_matches':len(history),
                'gap_differences':r['gap_differences'],'expected_statistics':r['expected_statistics'],
                'state_history_last_date':r['state_history_last_date'],
                'state_history_n':r['state_history_n'],'team_observations':r['team_observations'],
                'initialization':r['initialization'],'output_scope':'one_x_two_only'})
        window={'cutoff':cutoff,'mapping_training_matches':len(history),'target_matches':len(held),
                'seconds':time.perf_counter()-t0,'parameters':model.parameters(),'model_path':str(path)}
        windows.append(window)
        write(OUT/'逐场胜平负概率.json',{'rows':sorted(predictions,key=lambda r:(r['date'],r['match_id']))})
        write(OUT/'各次训练记录.json',{'windows':windows})
        print(json.dumps({'stage':'ordered_mapping','fit':len(windows),'cutoff':cutoff,
                         'training':len(history),'targets':len(held),'seconds':window['seconds']},ensure_ascii=False),flush=True)
    write(OUT/'执行结果.json',{'completed_at':datetime.now(timezone.utc).isoformat(),
        'historical_matches':len(rows),'update_optimizer_runs':4,
        'update_objective_evaluations':sum(x['objective_evaluations'] for x in learned['optimization'].values()),
        'mapping_fits':len(windows),'probability_rows':len(predictions),'hyperparameter_grids':0,
        'parameters_changed_after_viewing_later_scores':False,'existing_forecasts_changed':False,
        'seconds':time.perf_counter()-start,'source':str(SOURCE),'reference':str(OLD)})

if __name__=='__main__':
    try:
        main()
    except Exception as error:
        OUT.mkdir(parents=True,exist_ok=True)
        write(OUT/'失败.json',{'at':datetime.now(timezone.utc).isoformat(),'type':type(error).__name__,
            'error':str(error),'solver_restarts':0,'rescued_parameters':False})
        raise
