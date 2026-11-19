"""Inspect public ESPN team/stat coverage; save files below the project root."""
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import requests

ROOT = REPOSITORY_ROOT
OUT = ROOT/'data/espn/personnel_20261026'


def read_source(label, url):
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = {'url': url, 'requested_at': datetime.now(timezone.utc).isoformat()}
    payload = None
    try:
        response = requests.get(url, timeout=25)
        (OUT/f'{label}_response.bin').write_bytes(response.content)
        receipt.update(status_code=response.status_code, received_at=datetime.now(timezone.utc).isoformat(),
                       final_url=response.url, bytes=len(response.content))
        if response.status_code == 200:
            payload = response.json()
            (OUT/f'{label}.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        else:
            receipt['error'] = 'HTTP failure; no automatic retry or alternate host'
    except (requests.RequestException, ValueError) as error:
        receipt['error'] = str(error)
    (OUT/f'{label}_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload, receipt


if __name__ == '__main__':
    found = []
    for league, year in [('eng.2', 2025), ('eng.1', 2026)]:
        label = f'{league}_{year}_teams'
        payload, receipt = read_source(label,
            f'https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/teams?season={year}&limit=100')
        if payload:
            teams = [item['team'] for sport in payload.get('sports', []) for comp in sport.get('leagues', [])
                     for item in comp.get('teams', [])]
            relevant = [t for t in teams if any(x in t.get('displayName','') for x in ('Coventry','Hull','Ipswich'))]
            found.extend({'league': league, 'requested_year': year, **t} for t in relevant)
            print(json.dumps({'label': label, 'status': receipt.get('status_code'), 'teams': len(teams),
                'relevant': [{k:t.get(k) for k in ('id','displayName','slug')} for t in relevant]}, ensure_ascii=False))
        else:
            print(json.dumps(receipt, ensure_ascii=False))
            break
    (OUT/'selected_teams.json').write_text(json.dumps(found, ensure_ascii=False, indent=2), encoding='utf-8')
