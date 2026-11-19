"""Archive two actual current EPL statistical sources, retaining request evidence."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import requests

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OUT = ROOT/'data/epl_personnel_live_20261029/stat_sources'


def now():
    return datetime.now(timezone.utc).isoformat()


def fetch(spec):
    name, url, is_json = spec
    session = requests.Session()
    if 'understat.com/' in url:
        session.trust_env = False
    headers = {'User-Agent': 'FootballResearchPersonal/0.50'}
    if is_json:
        headers.update({'Accept': 'application/json', 'X-Requested-With': 'XMLHttpRequest',
                        'Referer': 'https://understat.com/league/EPL/2026'})
    receipt = {'source_url': url, 'request_started_at': now(), 'certificate_verification': True,
               'proxy_environment_used': session.trust_env}
    try:
        response = session.get(url, headers=headers, timeout=40)
        receipt.update({'received_at': now(), 'status': response.status_code,
                        'response_url': response.url, 'headers': dict(response.headers),
                        'bytes': len(response.content)})
        raw_path = OUT/(name + ('_response.bin' if is_json else '.csv'))
        raw_path.write_bytes(response.content)
        receipt['raw_path'] = str(raw_path)
        response.raise_for_status()
        if is_json:
            (OUT/(name+'.json')).write_text(json.dumps(response.json(), ensure_ascii=False,
                indent=2, allow_nan=False)+'\n', encoding='utf-8')
    except Exception as exc:
        receipt.update({'failed_at': now(), 'error': f'{type(exc).__name__}: {exc}'})
    (OUT/(name+'_receipt.json')).write_text(json.dumps(receipt, ensure_ascii=False,
        indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return {k: receipt.get(k) for k in ('source_url','request_started_at','received_at','status','bytes','error')}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    specs = [('E0', 'https://www.football-data.co.uk/mmz4281/2627/E0.csv', False),
             ('EPL_2026', 'https://understat.com/getLeagueData/EPL/2026', True)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(fetch, specs))
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
