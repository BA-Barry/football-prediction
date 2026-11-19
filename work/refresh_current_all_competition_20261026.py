"""Read current EPL clubs' past all-competition dates; no fit or forecast edits."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
import argparse
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import requests

from football_model.all_competition_features import AllCompetitionFeatures, ScheduleClub, ScheduleEvent


ROOT = REPOSITORY_ROOT
MEMBERS = ROOT/'data/espn/personnel_20261026/eng.1_2026_teams.json'
OFFICIAL_TEAMS = ROOT/'data/current_rich_inputs/run_20261026T025507813598Z/official-fpl/bootstrap.json'
START, CUTOFF = date(2026,7,1), date(2026,10,26)
ALIASES = {'AFC Bournemouth':'Bournemouth', 'Brighton & Hove Albion':'Brighton',
    'Coventry City':'Coventry','Hull City':'Hull',
    'Ipswich Town':'Ipswich', 'Leeds United':'Leeds', 'Manchester City':'Man City',
    'Manchester United':'Man United', 'Newcastle United':'Newcastle',
    'Nottingham Forest':"Nott'm Forest", 'Tottenham Hotspur':'Tottenham'}
FPL_ALIASES = {**ALIASES, 'Coventry City':'Coventry City','Hull City':'Hull City',
               'Ipswich Town':'Ipswich Town', 'Manchester United':'Man Utd', 'Tottenham Hotspur':'Spurs'}
FRIENDLIES = {'club.friendly', 'friendly.emirates_cup', 'esp.joan_gamper', 'jpn.world_challenge'}
COMPETITIVE = {'eng.1','eng.fa','eng.league_cup','eng.charity','uefa.champions',
    'uefa.europa','uefa.europa.conf','uefa.europa.conf_qual','uefa.champions_qual',
    'uefa.europa_qual','uefa.super_cup','fifa.cwc','fifa.intercontinental_cup'}
# Independently request competitions that commonly involve EPL clubs in this window.
CHECK_SLUGS = ['eng.1','eng.league_cup','eng.fa','eng.charity','uefa.champions',
               'uefa.europa','uefa.europa.conf','uefa.super_cup']


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, payload):
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def fetch(item):
    label,url,target = item
    existing=target/'receipts'/f'{label}.json'
    if existing.exists():
        raw_json=target/'raw'/f'{label}.json'
        return label,read(raw_json) if raw_json.exists() else None,read(existing)
    receipt={'source_url':url,'request_started_at':stamp(),'label':label}
    payload=None
    try:
        response=requests.get(url,headers={'User-Agent':'FootballResearchPersonal/0.31'},timeout=25,allow_redirects=False)
        raw=target/'raw'/f'{label}.bin'
        raw.write_bytes(response.content)
        receipt.update(received_at=stamp(),status=response.status_code,response_url=response.url,
                       headers=dict(response.headers),bytes=len(response.content),raw_file=str(raw))
        if response.status_code==200:
            payload=response.json()
            write(target/'raw'/f'{label}.json',payload)
        else:
            receipt['error']='HTTP failure; no automatic retry or alternate-host workaround'
    except (requests.RequestException,ValueError) as error:
        receipt.update(received_at=stamp(),error=str(error))
    write(target/'receipts'/f'{label}.json',receipt)
    return label,payload,receipt


def parse_event(event, default_slug=None):
    when=datetime.fromisoformat(event['date'].replace('Z','+00:00')).date()
    if not START <= when < CUTOFF:
        return None
    contests=event.get('competitions',[])
    if len(contests)!=1:
        raise ValueError('Expected one competition per soccer event')
    contest=contests[0]
    sides={r['homeAway']:r for r in contest['competitors']}
    if set(sides)!={'home','away'} or len(contest['competitors'])!=2:
        raise ValueError('Unresolved home-away event identity')
    slug=event.get('league',{}).get('slug') or default_slug
    status=contest.get('status',event.get('status',{})).get('type',{})
    category='competitive' if slug in COMPETITIVE else 'friendly_exhibition' if slug in FRIENDLIES else 'unclassified'
    completed=status.get('completed') is True
    scores={}
    for side in ('home','away'):
        score=sides[side].get('score')
        scores[side]=score.get('value') if isinstance(score,dict) else score
    return {'event_id':str(event['id']),'date_utc_text':event['date'],'match_date':when.isoformat(),
            'competition_slug':slug,'category':category,'completed':completed,
            'source_completed_flag':status.get('completed'),'status_name':status.get('name'),
            'source_final_score':scores,'home_score':scores['home'],'away_score':scores['away'],
            'home_team_id':str(sides['home']['team']['id']),'away_team_id':str(sides['away']['team']['id']),
            'home_name':sides['home']['team']['displayName'],'away_name':sides['away']['team']['displayName'],
            'is_regular_league_fixture':slug=='eng.1' and event.get('seasonType',{}).get('name')!='Relegation Playoff',
            'exclusion_reason': 'unclassified_competition' if category=='unclassified' else
                'unresolved_source_completion_status' if category=='competitive' and not completed else None}


def load_current_schedule_clubs(path):
    """Restore saved date-only clubs, keeping any uncertainty exclusions."""
    payload=read(path)
    clubs=[]
    for row in payload['clubs']:
        clubs.append(ScheduleClub(row['team'],row['competition'],row['source_team_id'],
            [(date.fromisoformat(a),date.fromisoformat(b)) for a,b in row['coverage']],
            [ScheduleEvent(e['event_id'],date.fromisoformat(e['match_date']),e['competition_slug'],
                e['category'],e['completed'],e['is_regular_league_fixture'],e.get('exclusion_reason')) for e in row['events']]))
    return clubs


def extend_all_competition_features(historical,path):
    """Return a new schedule feature object; leave the historical object untouched."""
    by_identity={(c.competition,c.team):c for c in historical._clubs.values()}
    for new in load_current_schedule_clubs(path):
        key=(new.competition,new.team)
        old=by_identity.get(key)
        if old is None:
            by_identity[key]=new
        else:
            if old.source_team_id!=new.source_team_id:
                raise ValueError('Current and historical source team IDs differ')
            events={e.event_id:e for e in old.events}
            for event in new.events:
                if event.event_id in events and events[event.event_id]!=event:
                    raise ValueError('A repeated event changes schedule feature fields')
                events[event.event_id]=event
            by_identity[key]=ScheduleClub(old.team,old.competition,old.source_team_id,
                                          old.coverage+new.coverage,list(events.values()))
    return AllCompetitionFeatures(list(by_identity.values()),source_metadata={
        **historical.source_metadata,'current_extension':str(path),
        'current_scope':'requested July1-Oct25 current season, not certified all-competition completeness'})


def load_current_schedule(path=None):
    """Load the latest completed current intake and extend the original history."""
    if path is None:
        reports=sorted((ROOT/'data/current_all_competition').glob('run_*/coverage_report.json'))
        if not reports:
            raise FileNotFoundError('No completed current all-competition intake')
        report=read(reports[-1])
        path=Path(report['schedule_clubs_file'])
    else:
        path=Path(path)
        report=read(path.with_name('coverage_report.json'))
    historical=AllCompetitionFeatures.from_directory(ROOT/'outputs/全赛事赛程可用性_20261025')
    return extend_all_competition_features(historical,path),report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-directory',type=Path,help='Reprocess saved responses without repeating requests')
    args=parser.parse_args()
    target=(args.reuse_directory or ROOT/'data/current_all_competition'/datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S%fZ')).resolve()
    (target/'raw').mkdir(parents=True,exist_ok=bool(args.reuse_directory))
    (target/'receipts').mkdir(exist_ok=bool(args.reuse_directory))
    members=[r['team'] for sport in read(MEMBERS)['sports'] for league in sport['leagues'] for r in league['teams']]
    official={r['name'] for r in read(OFFICIAL_TEAMS)['teams']}
    if len(members)!=20 or len({r['id'] for r in members})!=20 or {FPL_ALIASES.get(r['displayName'],r['displayName']) for r in members}!=official:
        raise ValueError('Current EPL identity catalog does not equal official 20-team set')
    source_results={}
    tasks=[(f'team_{r["id"]}',f'https://site.api.espn.com/apis/site/v2/sports/soccer/all/teams/{r["id"]}/schedule?season=2026&dates=20260701-20261025',target) for r in members]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for label,payload,receipt in pool.map(fetch,tasks):
            source_results[label]=(payload,receipt)
    # League-specific scoreboards can reveal a cup missed by the club all endpoint.
    checks=[(f'competition_{slug}',f'https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard?dates=20260701-20261025&limit=1000',target) for slug in CHECK_SLUGS]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for label,payload,receipt in pool.map(fetch,checks):
            source_results[label]=(payload,receipt)
    daily_sources=[]
    for day in ('20261023','20261024','20261025'):
        label,payload,receipt=fetch((f'epl_daily_{day}',
            f'https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard?dates={day}',target))
        daily_sources.append((payload,receipt))
    member_ids={str(r['id']) for r in members}
    extra_events={i:[] for i in member_ids}
    check_status={}
    for slug in CHECK_SLUGS:
        payload,receipt=source_results[f'competition_{slug}']
        check_status[slug]={'http_status':receipt.get('status'),'raw_event_count':len(payload.get('events',[])) if payload else None,
                           'source_success':payload is not None,'relevant_event_count':0}
        if payload:
            for event in payload.get('events',[]):
                row=parse_event(event,slug)
                if row:
                    relevant=member_ids & {row['home_team_id'],row['away_team_id']}
                    if relevant:
                        check_status[slug]['relevant_event_count']+=1
                        for tid in relevant:
                            extra_events[tid].append(row)
    clubs=[]
    for member in members:
        tid=str(member['id'])
        payload,receipt=source_results[f'team_{tid}']
        events={}
        issues=[]
        if payload and str(payload.get('team',{}).get('id'))==tid:
            for event in payload.get('events',[]):
                row=parse_event(event)
                if row:
                    if tid not in (row['home_team_id'],row['away_team_id']):
                        raise ValueError('Requested club absent from returned event')
                    events[row['event_id']]=row
                    if row['category']=='unclassified':
                        issues.append(f'{row["event_id"]}:unclassified_competition')
        else:
            issues.append('club_schedule_missing_or_wrong_identity')
        original_ids=set(events)
        for event in extra_events[tid]:
            prior=events.get(event['event_id'])
            if prior:
                for key in ('match_date','competition_slug','completed','home_team_id','away_team_id'):
                    if prior[key]!=event[key]:
                        prior['exclusion_reason']='source_schedule_scoreboard_conflict'
                        issues.append(f'{event["event_id"]}:{key}_conflict')
            else:
                events[event['event_id']]=event
        failed_checks=[slug for slug,row in check_status.items() if not row['source_success']]
        # Failed sources make the requested scope unusable; keep actual records
        # separately, without manufacturing games or interpreting absence as zero.
        coverage=[] if issues or failed_checks else [[START.isoformat(),CUTOFF.isoformat()]]
        label=ALIASES.get(member['displayName'],member['displayName'])
        values=sorted(events.values(),key=lambda r:(r['match_date'],r['event_id']))
        clubs.append({'team':f'football-data:E0:{label}','competition':'E0','source_team_id':tid,
            'source_name':member['displayName'],'coverage':coverage,
            'requested_coverage':[[START.isoformat(),CUTOFF.isoformat()]],
            'events':values,'coverage_issues':issues,'failed_competition_sources':failed_checks,
            'club_endpoint_event_count':len(original_ids),
            'scoreboard_supplemented_ids':sorted(set(events)-original_ids),
            'competition_counts':dict(Counter(r['competition_slug'] for r in values)),
            'formal_completed_count':sum(r['category']=='competitive' and r['completed'] for r in values),
            'unresolved_formal_count':sum(bool(r['category']=='competitive' and (not r['completed'] or r.get('exclusion_reason'))) for r in values)})
    path=target/'schedule_clubs.json'
    write(path,{'created_at':stamp(),'calendar_scope':[START.isoformat(),CUTOFF.isoformat()],
        'clubs':clubs,'competition_source_checks':check_status,
        'coverage_scope':'requested calendar scope with independent common competition scoreboards, not certified exhaustive all-competition coverage',
        'limitations':['Cup absence is not independently certified non-participation.',
          'Schedule counts are not measured minutes or causal fatigue.',
          'Final schedule publication/revision times unknown; only dates strictly before Oct26 are included.',
          'No scores, lineups, future events or training targets derived.'],
        'forecasts_changed':0,'model_fits':0,'results_settled':0})
    restored=AllCompetitionFeatures(load_current_schedule_clubs(path),source_metadata={'current_extension':str(path)})
    feature_checks={c.team:restored._snapshot(c.competition,c.team,CUTOFF)[0] for c in restored._clubs.values()}
    write(target/'current_feature_availability.json',feature_checks)
    summary={'directory':str(target),'schedule_clubs_file':str(path),'club_count':len(clubs),
        'unique_formal_completed_events':len({e['event_id'] for c in clubs for e in c['events'] if e['category']=='competitive' and e['completed']}),
        'clubs_with_usable_28d_count':sum(v['window_usable_28d']==1 for v in feature_checks.values()),
        'clubs':[{'name':c['source_name'],'formal_completed':c['formal_completed_count'],
                  'competitions':c['competition_counts'],'unresolved_formal':c['unresolved_formal_count'],
                  'scoreboard_supplemented_count':len(c['scoreboard_supplemented_ids'])} for c in clubs],
        'competition_checks':check_status,'model_fits':0,'forecasts_changed':0}
    write(target/'coverage_report.json',summary)
    league_recent=[]
    for league_payload,league_receipt in daily_sources:
        if league_payload:
            for event in league_payload.get('events',[]):
                row=parse_event(event,'eng.1')
                if row and row['match_date'] in ('2026-10-23','2026-10-24','2026-10-25'):
                    league_recent.append(row)
    write(target/'epl_oct24_25_source_observations.json',{'source_urls':[r['source_url'] for _,r in daily_sources],
        'received_at':[r.get('received_at') for _,r in daily_sources],
        'actual_calendar_scope':['2026-10-23','2026-10-26'],'records':league_recent,
        'record_count':len(league_recent),'source_completed_true_count':sum(r['completed'] for r in league_recent),
        'purpose':'Independent earlier input observations only; does not settle or rewrite any forecast.'})
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    main()
