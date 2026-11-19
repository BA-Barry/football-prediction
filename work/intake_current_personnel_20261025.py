"""Retrieve previous-season and earlier current-season public roster records."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
from pathlib import Path
import requests

from football_model.xg_history import COMPETITION_TEAM_NAMES

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
TARGET = ROOT/'data/understat/roster_EPL_2025_2026_20261025'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def select():
    paths = {
        2025: ROOT/'data/understat/history_20261024T170717354684Z/EPL_2025.json',
        2026: ROOT/'data/understat/current_directory_EPL_2026_20261025/EPL_2026.json',
    }
    fixtures = []
    aliases = COMPETITION_TEAM_NAMES['E0']
    for year, path in paths.items():
        season = f'{year}/{str(year + 1)[-2:]}'
        for row in read(path)['dates']:
            day = row['datetime'][:10]
            if row.get('isResult') is not True or day >= '2026-10-25':
                continue
            home, away = (aliases.get(row[s]['title'], row[s]['title']) for s in ('h', 'a'))
            uid = str(row['id'])
            fixtures.append({'match_id': f'understat:match:{uid}', 'understat_id': uid,
                'date': day, 'competition': 'E0', 'season': season,
                'home': f'football-data:E0:{home}', 'away': f'football-data:E0:{away}',
                'home_source_team_id': str(row['h']['id']), 'away_source_team_id': str(row['a']['id']),
                'source_url': f'https://understat.com/match/{uid}', 'league_directory': str(path),
                'identity_basis': 'source_league_directory_pending_independent_FD_join'})
    fixtures.sort(key=lambda r: (r['date'], r['understat_id']))
    if len({f['understat_id'] for f in fixtures}) != len(fixtures):
        raise ValueError('Duplicate source match IDs')
    return fixtures


def fetch(fixture):
    uid = fixture['understat_id']
    receipt = {'understat_id': uid, 'url': f'https://understat.com/getMatchData/{uid}', 'requested_at': stamp()}
    try:
        response = requests.get(receipt['url'], headers={'Accept': 'application/json',
            'X-Requested-With': 'XMLHttpRequest', 'Referer': fixture['source_url']},
            timeout=25, allow_redirects=False)
        (TARGET/'raw'/f'{uid}_response.bin').write_bytes(response.content)
        receipt.update(received_at=stamp(), status_code=response.status_code,
                       final_url=response.url, bytes=len(response.content))
        if response.status_code != 200:
            receipt['error'] = 'HTTP failure; no automatic retry or alternate route'
        else:
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError('Expected a JSON object')
            write(TARGET/'raw'/f'{uid}.json', payload)
    except (requests.RequestException, ValueError) as error:
        receipt['error'] = str(error)
    write(TARGET/'receipts'/f'{uid}.json', receipt)
    return receipt


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    for name in ('raw', 'receipts'):
        (TARGET/name).mkdir(exist_ok=True)
    selected = select()
    write(TARGET/'比赛选择.json', {'created_at': stamp(), 'cutoff_exclusive': '2026-10-25',
        'rule': 'all completed source matches in 2025/26 and earlier than today in 2026/27',
        'score_used_for_selection': False, 'selected': selected})
    pending, reused, fetched, failures = [], 0, 0, []
    started = stamp()
    for fixture in selected:
        uid = fixture['understat_id']
        receipt = TARGET/'receipts'/f'{uid}.json'
        if receipt.exists() and (TARGET/'raw'/f'{uid}.json').exists() and (TARGET/'raw'/f'{uid}_response.bin').exists():
            record = read(receipt)
            if record.get('status_code') == 200 and not record.get('error'):
                reused += 1
                continue
            if record.get('error'):
                raise RuntimeError('Inspect saved failed request before requesting it again')
        pending.append(fixture)

    def progress(status):
        value = {'started_at': started, 'updated_at': stamp(), 'status': status,
            'expected': len(selected), 'reused': reused, 'newly_fetched': fetched,
            'completed': reused + fetched, 'failures': failures, 'concurrency': 2,
            'cutoff_exclusive': '2026-10-25', 'purpose': 'personal noncommercial personnel continuity analysis',
            'model_fit_count': 0, 'results_evaluation_performed': False,
            'original_publication_time_known': False}
        write(TARGET/'进度.json', value)
        print(json.dumps({k: value[k] for k in ('status', 'expected', 'completed', 'failures')}, ensure_ascii=False), flush=True)

    progress('running')
    todo = iter(pending)
    with ThreadPoolExecutor(max_workers=2) as executor:
        running = {executor.submit(fetch, f) for f in [next(todo, None), next(todo, None)] if f is not None}
        while running:
            done, running = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                receipt = future.result()
                if receipt.get('error'):
                    failures.append(receipt)
                else:
                    fetched += 1
                if not failures:
                    fixture = next(todo, None)
                    if fixture is not None:
                        running.add(executor.submit(fetch, fixture))
            if (reused + fetched) % 25 == 0 or failures:
                progress('stopping_on_failure' if failures else 'running')
    progress('stopped_on_failure' if failures else 'completed')


if __name__ == '__main__':
    main()
