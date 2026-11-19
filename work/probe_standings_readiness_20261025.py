"""Bounded descriptive points reconstruction; no model fit or official ranks.

Run once through run_bounded_experiment.py with --max-seconds 60. This pilot
does not certify points deductions, future fixture notices or qualifications.
"""
import csv
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import time

ROOT=Path('G:/Codex/football-prediction-2026-10-22')
OUT=ROOT/'outputs/赛前新信息可行性_20261025'
SOURCE_DIRS={
    '2023/24': ROOT/'data/football-data/2023-24/20261022T113804786593Z',
    '2024/25': ROOT/'outputs/Football-Data_2024-25_个人使用_20261022T111235Z',
}


def load(season, folder):
    candidates=list(folder.rglob('E0.csv'))
    if len(candidates)!=1:
        raise ValueError(f'Pilot requires exactly one existing E0 file for {season}, found {len(candidates)}')
    path=candidates[0]
    rows=[]
    with path.open(encoding='utf-8-sig',newline='') as handle:
        for raw in csv.DictReader(handle):
            if not any(raw.values()):
                continue
            text=raw['Date']
            day=datetime.strptime(text,'%d/%m/%Y' if len(text)==10 else '%d/%m/%y').date()
            rows.append({'date':day,'home':raw['HomeTeam'],'away':raw['AwayTeam'],
                         'h':int(raw['FTHG']),'a':int(raw['FTAG'])})
    if len(rows)!=380:
        raise ValueError(f'Preselected season scope expected 380 rows, got {len(rows)}')
    return sorted(rows,key=lambda r:(r['date'],r['home'],r['away'])),str(path)


def points_before(rows, cutoff):
    # Read outcomes only after chronological filtering, without target results.
    states={}
    selected=[r for r in rows if r['date']<cutoff]
    for r in selected:
        for team in (r['home'],r['away']):
            states.setdefault(team,{'played':0,'earned_points':0})
            states[team]['played']+=1
        hp,ap=(3,0) if r['h']>r['a'] else (0,3) if r['h']<r['a'] else (1,1)
        states[r['home']]['earned_points']+=hp
        states[r['away']]['earned_points']+=ap
    return states,selected[-1]['date'].isoformat() if selected else None,len(selected)


def main():
    started=time.perf_counter()
    begun=datetime.now(timezone.utc).isoformat()
    datasets={s:load(s,p) for s,p in SOURCE_DIRS.items()}
    records=[]
    for season,(history,path) in datasets.items():
        for target in history:
            for lead in (1,3):
                cutoff=target['date']-timedelta(days=lead)
                table,last,n=points_before(history,cutoff)
                records.append({'season':season,'date':target['date'].isoformat(),
                    'home':target['home'],'away':target['away'],'horizon_days':lead,
                    'forecast_before_date':cutoff.isoformat(),'n_prior_source_matches':n,
                    'training_max_date':last,
                    'home_raw_points':table.get(target['home'],{}).get('earned_points',0),
                    'away_raw_points':table.get(target['away'],{}).get('earned_points',0),
                    'home_prior_played':table.get(target['home'],{}).get('played',0),
                    'away_prior_played':table.get(target['away'],{}).get('played',0),
                    'official_points_certified':False,'qualification_status':'unknown',
                    'reason':'no complete as-of administrative adjustments / tie rules / qualification ledger'})
    # Concrete integration check: same-date and later poisoned outcomes unread.
    history=datasets['2023/24'][0]
    f=date(2024,3,18)
    prior,_,_=points_before(history,f)
    poisoned=[r if r['date']<f else {'date':r['date'],'bad_future_outcome':object()} for r in history]
    check,_,_=points_before(poisoned,f)
    assert check==prior
    # One official historical administrative event, not a complete standings table.
    case=[]
    for cutoff in (date(2024,3,18),date(2024,3,19)):
        state,_,_=points_before(history,cutoff)
        adjustment=-4 if date(2024,3,18)<cutoff else 0
        raw=state['Nott\'m Forest']['earned_points']
        case.append({'forecast_before_date':cutoff.isoformat(),'raw_match_points':raw,
            'known_single_event_adjustment':adjustment,'single_event_adjusted_points':raw+adjustment,
            'announcement_day':'2024-03-18','effective_day':'2024-03-18',
            'same_day_policy':'announcement day excluded when clock precision unknown',
            'source_url':'https://www.premierleague.com/en/news/3936397/nottingham-forest-deducted-four-points-by-independent-commission',
            'complete_table_certified':False})
    assert case[0]['raw_match_points']==25 and case[1]['single_event_adjusted_points']==21
    result={'status':'completed','started_at':begun,'completed_at':datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds':time.perf_counter()-started,'execution_budget_seconds':60,
        'stop_reason':'two_preselected_seasons_and_two_leads_completed; no_additional_season_or_model',
        'new_model_fits':0,'seasons':list(datasets),'source_rows':sum(len(v[0]) for v in datasets.values()),
        'snapshot_rows':len(records),'sources':{s:v[1] for s,v in datasets.items()},
        'official_qualification_flags_verified':0,'administrative_adjustment_history_complete':False,
        'source_publication_and_revision_times_known':False,'future_schedule_announcement_history_known':False,
        'ready_to_fit_as_motivation':False,'current_2026_27_results_read':False,
        'temporal_checks':{'future_poison_did_not_change_prior_state':True,
            'strict_prior_dates':all(r['training_max_date'] is None or r['training_max_date']<r['forecast_before_date'] for r in records)},
        'administrative_points_example':case,
        'interpretation':'raw historical match-earned points are reproducible; official points / qualification / motivation are not certified',
        'next_action':'stop this feasibility pilot; do not invent motivation coefficients'}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'积分重建测量.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'积分重建逐场快照.json').write_text(json.dumps({'rows':records},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
