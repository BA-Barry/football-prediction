"""Prepare finite, isolated factual packets for an LLM recent-input probe.

Two packets are identical repeats; one withholds complete recent fixtures.
No estimated model probabilities, markets, source forecasts or xpts are copied.
"""
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

ROOT=Path('G:/Codex/football-prediction-2026-10-22')
CURRENT=ROOT/'data/current_rich_inputs/run_20261026T025507813598Z/understat/EPL_2026.json'
PRIOR=ROOT/'data/understat/history_20261024T170717354684Z/EPL_2025.json'
FIXTURES=ROOT/'data/current_rich_inputs/run_20261026T025507813598Z/official-fpl/confirmed_future_fixtures.json'
AS_OF='2026-10-26T03:08:05.654404+00:00'
ALIASES={'Manchester City':'Man City','Manchester United':'Man United','Newcastle United':'Newcastle',
         'Nottingham Forest':"Nott'm Forest"}


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def observations(path,season):
    payload=read(path)
    histories={(str(tid),r['date'],r['h_a']):r for tid,t in payload['teams'].items() for r in t['history']}
    rows=[]
    for m in payload['dates']:
        if m['isResult'] is not True or m['datetime'][:10]>=AS_OF[:10]: continue
        for side,other in (('h','a'),('a','h')):
            h=histories[(str(m[side]['id']),m['datetime'],side)]
            gf,ga=int(m['goals'][side]),int(m['goals'][other])
            if gf!=h['scored'] or ga!=h['missed']: raise ValueError('Raw score inconsistency')
            if abs(float(m['xG'][side])-h['xG'])>1e-5: raise ValueError('Raw xG inconsistency')
            team=ALIASES.get(m[side]['title'],m[side]['title'])
            opponent=ALIASES.get(m[other]['title'],m[other]['title'])
            rows.append({'source_match_id':str(m['id']),'date':m['datetime'][:10],'season':season,
                'team':team,'opponent':opponent,'venue':'home' if side=='h' else 'away',
                'goals_for':gf,'goals_against':ga,'xg_for':float(h['xG']),'xg_against':float(h['xGA']),
                'npxg_for':float(h['npxG']),'npxg_against':float(h['npxGA']),
                'ppda_att':float(h['ppda']['att']),'ppda_def':float(h['ppda']['def']),
                'deep_for':float(h['deep']),'deep_against':float(h['deep_allowed'])})
    return rows


def summary(rows):
    n=len(rows)
    if not n: return {'n':0,'available':False}
    output={'n':n,'available':True,'first_date':rows[0]['date'],'last_date':rows[-1]['date'],
        'wins':sum(r['goals_for']>r['goals_against'] for r in rows),
        'draws':sum(r['goals_for']==r['goals_against'] for r in rows),
        'losses':sum(r['goals_for']<r['goals_against'] for r in rows),
        'goals_for':sum(r['goals_for'] for r in rows),'goals_against':sum(r['goals_against'] for r in rows)}
    for key in ('xg_for','xg_against','npxg_for','npxg_against','deep_for','deep_against'):
        output[key+'_per_match']=round(math.fsum(r[key] for r in rows)/n,4)
    denominator=math.fsum(r['ppda_def'] for r in rows)
    output['ppda_sum_att_over_sum_def']=round(math.fsum(r['ppda_att'] for r in rows)/denominator,4) if denominator else None
    return output


def packet(rows,fixtures):
    by_team=defaultdict(list)
    for r in rows: by_team[r['team']].append(r)
    teams={}
    for team in sorted(by_team):
        if team not in fixtures['teams']: continue
        local=sorted(by_team[team],key=lambda r:(r['date'],r['source_match_id']))
        current=[r for r in local if r['season']=='2026/27']
        previous=[r for r in local if r['season']=='2025/26']
        teams[team]={'current_season_available':summary(current),'previous_season_available':summary(previous),
            'last5_available':summary(local[-5:]),'last15_available':summary(local[-15:]),
            'current_home_available':summary([r for r in current if r['venue']=='home']),
            'current_away_available':summary([r for r in current if r['venue']=='away']),
            'last5_observation_rows':[{k:r[k] for k in ('source_match_id','date','opponent','venue',
                'goals_for','goals_against','xg_for','xg_against','npxg_for','npxg_against')} for r in local[-5:]]}
    return {'as_of':AS_OF,'class_order':['home','draw','away'],'fixtures':fixtures['fixtures'],
        'teams':teams,'scope':[
            'Only supplied completed domestic-league observations before as_of are available. Available subsets need not exhaust played games; missing is not zero.',
            'All aggregates are calculated from the same available subset. Last5 overlaps current and last15, so they are not independent evidence.',
            'Current venue splits are small. Raw form does not adjust past opponent strength. Cross-season personnel/tactics continuity is unverified.',
            'No injury bulletin, lineups, player strength effects, market, expert probabilities or intervening future cup schedule is supplied.',
            'PPDA and deep are descriptive source metrics, not causal strength estimates. Previous-season Premier League data missing for newly promoted teams is retained as unavailable.']}


def main():
    raw=read(FIXTURES)['fixture_observations']
    fpl=read(FIXTURES)['teams']
    names={r['id']:r['name'] for r in fpl} if isinstance(fpl,list) else None
    # Resolve from the independently checked IDs via the saved official catalog.
    bootstrap=read(FIXTURES.with_name('bootstrap.json'))
    aliases={'Coventry City':'Coventry','Hull City':'Hull','Ipswich Town':'Ipswich','Man Utd':'Man United','Spurs':'Tottenham'}
    canonical={r['id']:aliases.get(r['name'],r['name']) for r in bootstrap['teams']}
    fixtures={'fixtures':[r['original_fixture']|{'home_label':canonical[r['home_team_id']],
        'away_label':canonical[r['away_team_id']]} for r in raw], 'teams':set(canonical.values())}
    if not all(r['confirmed_unstarted'] for r in raw): raise ValueError('Future targets changed')
    rows=observations(PRIOR,'2025/26')+observations(CURRENT,'2026/27')
    by_team=defaultdict(list)
    for r in rows:
        if r['team'] in fixtures['teams']: by_team[r['team']].append(r)
    removed=set()
    selected={}
    for t,values in by_team.items():
        ordered=sorted(values,key=lambda r:(r['date'],r['source_match_id']))
        selected[t]=[r['source_match_id'] for r in ordered[-2:]]
        removed.update(selected[t])
    full=packet(rows,fixtures)
    partial=packet([r for r in rows if r['source_match_id'] not in removed],fixtures)
    if set(full['teams'])!=fixtures['teams'] or set(partial['teams'])!=fixtures['teams']:
        raise ValueError('Team identity coverage incomplete')
    out=ROOT/'outputs'/f'LLM近期信息敏感性_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}'
    (out/'inputs').mkdir(parents=True)
    (out/'responses').mkdir()
    write(out/'inputs/packet_r1.json',full)
    write(out/'inputs/packet_r2.json',full)
    write(out/'inputs/packet_r3.json',partial)
    if read(out/'inputs/packet_r1.json')!=read(out/'inputs/packet_r2.json'):
        raise ValueError('Repeated factual packet differs')
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'information_cutoff':AS_OF,
        'mapping':{'r1':'full_available','r2':'same_full_available_repeat','r3':'withhold_recent_two_whole_fixtures'},
        'data_sources':[str(PRIOR),str(CURRENT),str(FIXTURES)],
        'selected_last_two_ids_by_team':selected,'whole_fixture_ids_removed':sorted(removed),
        'removed_unique_fixture_n':len(removed),'per_team_actual_removed':{
            t:sum(r['source_match_id'] in removed for r in values) for t,values in by_team.items()},
        'initial_counts':{'targets':len(raw),'teams':len(by_team)},
        'llm_design':'Three isolated same-parent-configuration agents, same factual task. No chosen model overrides; actual model ID and sampling settings unavailable.',
        'metrics':['full r1-vs-r2 per-match max three-class pp gap (repeat variation)',
                   'r3-vs-each-full same gap;no probability averaging or combined forecast',
                   'top-class changes and repeated-control direction agreement'],
        'stopping':'Exactly three responses;no resampling to choose a better answer,no tuning,training,current settlement or live forecast replacement.',
        'limits':['One response per condition plus one full repeat cannot estimate a distribution or significance.',
                  'Removing observations removes goals and performance together, changes sample counts and refills windows from earlier games.',
                  'The repeat and intervention effects are mixed with prompt interpretation and stochastic variation;cannot identify a causal recency weight.',
                  'Numeric fact packet differs from the original live LLM prompt;findings do not retroactively replace its record.',
                  'Published model/LLM probabilities and source forecast/xpts fields are excluded from all packets.',
                  'Teams remain named;latent model prior knowledge cannot be independently ruled out,although external knowledge is prohibited.',
                  'No outcome scoring or accuracy claim;information sensitivity can be useful or harmful.']}
    write(out/'方法与停止点.json',plan)
    print(json.dumps({'output':str(out),'removed_fixture_n':len(removed),'actual_removed_counts':sorted(set(plan['per_team_actual_removed'].values())),'packet_teams':len(full['teams'])},ensure_ascii=False))


if __name__=='__main__': main()
