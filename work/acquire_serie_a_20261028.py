"""Archive actual current sources for the five not-yet-started Oct28 I1 games."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import requests

ROOT=Path('G:/Codex/football-prediction-2026-10-22')
OUT=ROOT/'data/serie_a_today_20261028'
ESPN='https://site.api.espn.com/apis/site/v2/sports/soccer/ita.1/scoreboard?dates=20261028&limit=1000'
OFFICIAL='https://www.legaseriea.it/serie-a/news/quando-si-gioca-anticipi-e-posticipi-fino-alla-12a-giornata'
PAIRS=(('AC Milan','Bologna'),('Parma','Udinese'),('Venezia','Internazionale'),('Genoa','Juventus'),('Monza','Napoli'))
ALIASES={'AC Milan':'Milan','Internazionale':'Inter'}

def now():return datetime.now(timezone.utc).isoformat()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def fetch(label,url,json_source=True,headers=None):
    started=now()
    session=requests.Session()
    # The configured proxy produced a TLS EOF on Understat; direct HTTPS with
    # normal certificate verification succeeds. This does not bypass access rules.
    if 'understat.com/' in url:session.trust_env=False
    r=session.get(url,timeout=40,headers=headers or {'User-Agent':'FootballResearchPersonal/0.50'})
    received=now();raw=OUT/'raw'/f'{label}_response.bin';raw.write_bytes(r.content)
    receipt={'source_url':url,'request_started_at':started,'received_at':received,'status':r.status_code,
        'response_url':r.url,'headers':dict(r.headers),'raw_path':str(raw),'bytes':len(r.content)}
    write(OUT/'receipts'/f'{label}.json',receipt);r.raise_for_status()
    if json_source:
        d=r.json();write(OUT/'raw'/f'{label}.json',d)
    else:d=r.content
    return d,receipt
def decimal(v):
    n=float(v)
    if n==0:raise ValueError('American odds cannot be zero')
    return round(1+n/100 if n>0 else 1+100/abs(n),12)
def main():
    (OUT/'raw').mkdir(parents=True,exist_ok=True);(OUT/'receipts').mkdir(exist_ok=True)
    specs=[('espn_ita1_20261028',ESPN,True,None),('I1','https://www.football-data.co.uk/mmz4281/2627/I1.csv',False,None),
        ('Serie_A_2026','https://understat.com/getLeagueData/Serie_A/2026',True,{'User-Agent':'FootballResearchPersonal/0.50','Accept':'application/json','X-Requested-With':'XMLHttpRequest','Referer':'https://understat.com/league/Serie_A/2026'}),
        ('official_schedule',OFFICIAL,False,None)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        values=list(pool.map(lambda s:fetch(*s),specs))
    raw,receipt=values[0];(OUT/'raw/I1.csv').write_bytes(values[1][0])
    html=values[3][0].decode('utf-8',errors='replace')
    # Verify the actual official bulletin contains the five pairings and hours;
    # UTC conversion uses Italy's winter offset on the stated target date.
    import re
    plain=re.sub('<[^>]+>',' ',html)
    plain=' '.join(plain.split())
    markers=['Milan-Bologna','Parma-Udinese','Venezia-Inter','Genoa-Juventus','Monza-Napoli']
    if not all(m in plain for m in markers) or '18.30' not in plain or '20.45' not in plain:
        raise ValueError('Official bulletin does not independently confirm the requested fixtures')
    official={'url':OFFICIAL,'observed_at':values[3][1]['received_at'],'pairings_present':markers,
        'local_timezone':'Europe/Rome','local_hours':['18:30','20:45'],
        'source_text_excerpt':plain[max(0,plain.find('Tre sfide mercoledì')-100):plain.find('Tre sfide mercoledì')+350]}
    write(OUT/'official_schedule_verification.json',official)
    bank={};markets=[];futures=[]
    for event in raw['events']:
        c=event['competitions'][0];sides={q['homeAway']:q['team'] for q in c['competitors']}
        pair=(sides['home']['displayName'],sides['away']['displayName'])
        bank[pair]=(event,c,sides)
    if set(bank)!=set(PAIRS):raise ValueError('Actual source fixtures differ from requested five')
    observed=receipt['received_at'];instant=datetime.now(timezone.utc)
    for pair in PAIRS:
        e,c,sides=bank[pair];status=c.get('status',e.get('status'))['type'];kickoff=datetime.fromisoformat(e['date'].replace('Z','+00:00'))
        if status['name']!='STATUS_SCHEDULED' or status['completed'] is not False or kickoff<=instant:
            raise ValueError('A target is no longer unstarted')
        row={'match_id':'espn:ita.1:'+str(e['id']),'home':pair[0],'away':pair[1],'competition':'Serie A',
            'kickoff':kickoff.isoformat(),'neutral':False,'home_model_label':ALIASES.get(pair[0],pair[0]),
            'away_model_label':ALIASES.get(pair[1],pair[1]),'observed_at':observed,'source_url':ESPN,
            'status':'scheduled','source_event_id':str(e['id']),'source_status':status,
            'home_team_id':str(sides['home']['id']),'away_team_id':str(sides['away']['id'])}
        futures.append(row)
        for odds in c.get('odds',[]):
            ml=odds.get('moneyline',{})
            try:
                close=[ml[k]['close']['odds'] for k in ('home','draw','away')]
                opened=[ml[k]['open']['odds'] for k in ('home','draw','away')]
            except (KeyError,TypeError):continue
            markets.append({'home':pair[0],'away':pair[1],'event_id':str(e['id']),
                'provider':odds.get('provider',{}).get('name'),'provider_metadata':odds.get('provider'),
                'source_url':ESPN,'observed_at':observed,'source_timestamp':None,
                'decimal_odds':[decimal(v) for v in close],'opening_odds':[decimal(v) for v in opened],
                'original_odds':{'format':'American','order':['home','draw','away'],'open':opened,'close':close},
                'source_status':status,'kickoff_utc':kickoff.isoformat(),
                'limitation':'Source fields open/close from scheduled-event snapshot. close is the returned latest available quote, not a certified final closing price. Upstream per-quote/open timestamps unknown; single provider, no cross-provider splice.'})
    write(OUT/'confirmed_fixtures.json',{'observed_at':observed,'fixtures':futures,'official_url':OFFICIAL,'official_verification':official})
    write(OUT/'market_observations.json',{'observed_at':observed,'records':markets})
    # A second real request confirms all identities, scheduled status and kickoff
    # after the model data intake has completed.
    second,second_receipt=fetch('espn_reconfirmation_20261028',ESPN)
    second_bank={str(e['id']):e for e in second['events']}
    for row in futures:
        e=second_bank[row['source_event_id']];c=e['competitions'][0]
        status=c.get('status',e.get('status'))['type']
        if status['name']!='STATUS_SCHEDULED' or status['completed'] is not False or datetime.fromisoformat(e['date'].replace('Z','+00:00')).isoformat()!=row['kickoff']:
            raise ValueError('Second real fixture status/kickoff differs')
    write(OUT/'fixture_reconfirmation.json',{'observed_at':second_receipt['received_at'],'source_url':ESPN,
        'confirmed_event_ids':[r['source_event_id'] for r in futures],'all_unstarted':True})
    # Independent score supplementation for a stale Football-Data download.
    # This dated intake is prior to the five target fixtures, never their labels.
    fetch('espn_ita1_20261027_completed',
          'https://site.api.espn.com/apis/site/v2/sports/soccer/ita.1/scoreboard?dates=20261027&limit=1000')
    print(json.dumps({'directory':str(OUT),'fixtures':[(r['home'],r['away'],r['kickoff']) for r in futures],
        'market_records':len(markets),'second_status_observed_at':second_receipt['received_at']},ensure_ascii=False))
if __name__=='__main__':main()
