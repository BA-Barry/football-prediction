"""Retrieve completed historical seasons only; never inspect live fixtures."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import requests

ROOT = REPOSITORY_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--leagues', nargs='+', choices=['EPL','Bundesliga','La_Liga','Serie_A','Ligue_1'], default=['EPL'])
    parser.add_argument('--start-years', nargs='+', type=int, choices=range(2014,2026),
                        default=[2022,2023,2024,2025])
    args = parser.parse_args()
    leagues = list(dict.fromkeys(args.leagues))
    years = list(dict.fromkeys(args.start_years))
    now = datetime.now(timezone.utc)
    target = ROOT / 'data/understat' / now.strftime('history_%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True, exist_ok=False)
    records = []
    for league, season in [(league, year) for league in leagues for year in years]:
        url = f'https://understat.com/getLeagueData/{league}/{season}'
        start = datetime.now(timezone.utc)
        try:
            response = requests.get(url, headers={'Accept': 'application/json',
                'X-Requested-With': 'XMLHttpRequest',
                'Referer': f'https://understat.com/league/{league}/{season}'}, timeout=25)
            received = datetime.now(timezone.utc)
            name = f'{league}_{season}_response.bin'
            (target/name).write_bytes(response.content)
            item = {'url': url, 'league': league, 'season_start_year':season,
                    'requested_at': start.isoformat(), 'received_at': received.isoformat(),
                    'status_code': response.status_code, 'final_url': response.url,
                    'headers': dict(response.headers), 'raw_file': name,
                    'bytes': len(response.content)}
            if response.status_code == 200:
                try:
                    data = response.json()
                    item['top_level_keys'] = sorted(data) if isinstance(data, dict) else None
                    dates = data.get('dates', []) if isinstance(data, dict) else []
                    item['match_rows'] = len(dates)
                    item['complete_result_rows'] = sum(r.get('isResult') is True for r in dates)
                    (target/f'{league}_{season}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
                except (ValueError, TypeError) as error:
                    item['parse_error'] = str(error)
            records.append(item)
            if response.status_code in (401, 403, 429):
                break
        except requests.RequestException as error:
            records.append({'url':url,'requested_at':start.isoformat(),'error':str(error)})
            break
    metadata = {'created_at': now.isoformat(), 'purpose': 'historical xG model research, personal use',
                'seasons_requested': years, 'leagues_requested':leagues,
                'current_2026_season_requested':False,
                'source':'https://understat.com',
                'reference_reader':'https://github.com/martineastwood/penaltyblog/blob/master/penaltyblog/scrapers/understat.py',
                'reader_approach': 'public getLeagueData endpoint and explicit requested/referer headers; raw evidence retained',
                'publication_time_warning':'download receipt is not original historical xG availability; retrospective date-level assumption only',
                'data_rights':'Underlying Understat data rights are separate from library code licence; no commercial reuse asserted',
                'responses':records}
    (target/'intake_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'directory':str(target),'responses':[{k:r[k] for k in ('url','status_code','bytes','match_rows','complete_result_rows','parse_error','error') if k in r} for r in records]},ensure_ascii=False))


if __name__=='__main__':
    main()
