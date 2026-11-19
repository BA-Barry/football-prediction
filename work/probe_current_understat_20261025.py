"""Read and archive the current public EPL directory, without reporting scores."""
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import requests

ROOT = REPOSITORY_ROOT
target = ROOT/'data/understat/current_directory_EPL_2026_20261025'
target.mkdir(parents=True, exist_ok=True)
url = 'https://understat.com/getLeagueData/EPL/2026'
receipt = {'url': url, 'requested_at': datetime.now(timezone.utc).isoformat()}
try:
    response = requests.get(url, headers={'Accept': 'application/json',
        'X-Requested-With': 'XMLHttpRequest', 'Referer': 'https://understat.com/league/EPL/2026'},
        timeout=25, allow_redirects=False)
    (target/'response.bin').write_bytes(response.content)
    receipt.update(status_code=response.status_code, final_url=response.url,
        received_at=datetime.now(timezone.utc).isoformat(), bytes=len(response.content))
    if response.status_code == 200:
        payload = response.json()
        (target/'EPL_2026.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        rows = payload.get('dates', [])
        # This is a source-coverage summary, not a match-result report.
        historical = [r for r in rows if r.get('isResult') is True and str(r.get('datetime', ''))[:10] < '2026-10-25']
        receipt.update(top_level_keys=sorted(payload), total_rows=len(rows),
            completed_before_today=len(historical), historical_date_min=min((r['datetime'] for r in historical),default=None),
            historical_date_max=max((r['datetime'] for r in historical),default=None))
except (requests.RequestException, ValueError) as exc:
    receipt['error'] = str(exc)
(target/'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'directory': str(target), **receipt}, ensure_ascii=False))
