"""Fetch a small deterministic historical roster sample below the project root."""
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import requests

from football_model.football_data import load_directory
from football_model.xg_history import join_understat

ROOT = REPOSITORY_ROOT
AUDIT = ROOT/'outputs/跨联赛xG源核对_20261024T172403681186Z/逐季源差异.json'
SLUGS = {'E0':'EPL','D1':'Bundesliga','SP1':'La_Liga','I1':'Serie_A','F1':'Ligue_1'}


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    started = datetime.now(timezone.utc)
    target = ROOT/'data/understat'/started.strftime('roster_pilot_%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True,exist_ok=False)
    audit = json.loads(AUDIT.read_text(encoding='utf-8'))
    fd_dirs = {'2023/24':ROOT/'data/football-data/2023-24/20261022T113804786593Z',
               '2024/25':ROOT/'outputs/Football-Data_2024-25_个人使用_20261022T111235Z'}
    fd = {}
    for season,directory in fd_dirs.items():
        records,report = load_directory(directory,season=season)
        if report['rejected_rows'] or report['duplicate_keys']:
            raise ValueError('Resolve malformed historical source before requesting a roster')
        fd[season] = records
    selection = []
    for comp,slug in SLUGS.items():
        seasons = {}
        for year in (2023,2024):
            season = f'{year}/{str(year+1)[-2:]}'
            stamp = '20261024T170717354684Z' if comp=='E0' else '20261024T172015108538Z'
            source = ROOT/f'data/understat/history_{stamp}/{slug}_{year}.json'
            raw = json.loads(source.read_text(encoding='utf-8'))['dates']
            differences = audit.get(f'{comp}:{season}',{}).get('differences',[])
            bad_source = {str(r['understat_id']) for r in differences}
            bad_fd = {r['football_data_match_id'] for r in differences}
            eligible = [r for r in fd[season] if r.competition==comp and r.match_id not in bad_fd]
            parsed,report = join_understat([r for r in raw if str(r['id']) not in bad_source],eligible,season,comp)
            if report['missing_independent_match_ids']:
                raise ValueError('Unexplained exact-join missingness')
            seasons[season] = sorted(parsed,key=lambda r:(r.match_date,r.match_id))
        team_sets = [{t for r in rows for t in (r.home,r.away)} for rows in seasons.values()]
        anchor = sorted(set.intersection(*team_sets))[0]
        for season,rows in seasons.items():
            selected = [r for r in rows if anchor in (r.home,r.away)][:2]
            if len(selected)!=2:
                raise ValueError('Need two deterministic fixtures for the shared anchor team')
            for r in selected:
                selection.append({'match_id':r.match_id,'understat_id':r.understat_id,'competition':comp,
                    'season':season,'date':r.match_date.isoformat(),'home':r.home,'away':r.away,
                    'goals':list(r.goals),'xg':list(r.xg),'anchor_team':anchor,
                    'source_url':r.source_url})
    write(target/'试样选择.json',{'created_at':started.isoformat(),
        'rule':'per league, alphabetically first canonical team appearing in both 2023/24 and 2024/25; its first two eligible date-ordered matches each season',
        'score_used_for_selection':False,'fixture_identity':'existing exact FD/Understat date+team+score join',
        'purpose':'roster/identity/explicit position feasibility only; no model fit or current result review',
        'expected_requests':20,'selected':selection})
    responses = []
    for item in selection:
        uid = item['understat_id']
        url = f'https://understat.com/getMatchData/{uid}'
        request_at = datetime.now(timezone.utc)
        metadata = {'understat_id':uid,'url':url,'requested_at':request_at.isoformat()}
        try:
            response = requests.get(url,headers={'Accept':'application/json',
                'X-Requested-With':'XMLHttpRequest','Referer':f'https://understat.com/match/{uid}'},timeout=25)
            metadata.update({'received_at':datetime.now(timezone.utc).isoformat(),'status_code':response.status_code,
                'final_url':response.url,'headers':dict(response.headers),'raw_file':f'{uid}_response.bin',
                'bytes':len(response.content)})
            (target/f'{uid}_response.bin').write_bytes(response.content)
            if response.status_code==200:
                data = response.json()
                write(target/f'{uid}.json',data)
                metadata['keys'] = sorted(data) if isinstance(data,dict) else None
                roster = data.get('rosters',{}) if isinstance(data,dict) else {}
                metadata['roster_counts'] = {side:len(roster.get(side,{})) for side in ('h','a')}
                metadata['roster_fields'] = sorted({key for side in ('h','a') for r in roster.get(side,{}).values() for key in r})
            else:
                metadata['error'] = 'HTTP refusal or failure; no alternate source or automatic retry'
        except (requests.RequestException,ValueError,TypeError) as error:
            metadata['error'] = str(error)
        responses.append(metadata)
        write(target/'接入记录.json',{'created_at':started.isoformat(),'purpose':'personal noncommercial historical roster pilot',
            'current_results_reviewed':False,'original_lineup_publication_time_known':False,
            'rights':'Underlying data rights separate from open-source reader code; no commercial reuse asserted',
            'reader_references':['https://soccerdata.readthedocs.io/en/latest/datasources/Understat.html',
                                 'https://github.com/probberechts/soccerdata',
                                 'https://github.com/martineastwood/penaltyblog'],
            'responses':responses})
        if metadata.get('error'):
            break
        print(f'Historical roster {uid}: '+str(metadata['roster_counts']),flush=True)
    print(json.dumps({'directory':str(target),'planned':len(selection),'responses':len(responses),
                      'http_200':sum(r.get('status_code')==200 and not r.get('error') for r in responses),
                      'last_error':responses[-1].get('error') if responses else None,
                      'fields':responses[0].get('roster_fields') if responses else None},ensure_ascii=False))


if __name__ == '__main__':
    main()
