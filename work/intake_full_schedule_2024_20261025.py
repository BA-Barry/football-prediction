"""Read historical2024/25 ESPN core memberships then all-competition schedules.

Two concurrent requests at most; cached JSON+receipts retained on G. This is
an availability import, not a claim of certified all-competition completeness.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import re
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = REPOSITORY_ROOT
YEAR = 2024
OUT = ROOT/'outputs/全赛事赛程可用性_20261025/2024-25全量'
SLUGS = {'E0': 'eng.1', 'D1': 'ger.1', 'SP1': 'esp.1', 'I1': 'ita.1', 'F1': 'fra.1'}
COUNTS = {'E0': 20, 'D1': 18, 'SP1': 20, 'I1': 20, 'F1': 18}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def fetch(key, url):
    path, receipt = OUT/'raw'/f'{key}.json', OUT/'receipts'/f'{key}.json'
    if path.exists():
        payload = json.loads(path.read_text(encoding='utf-8'))
        if not receipt.exists():
            write(receipt, {'source_url': url, 'action': 'reused_saved_historical_probe',
                'raw_file': str(path), 'checked_at': datetime.now(timezone.utc).isoformat()})
        return payload, False
    errors = []
    for attempt in range(3):
        try:
            time.sleep(.35)
            with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0 (historical personal football research)'}), timeout=30) as response:
                raw, status = response.read(), response.status
            payload = json.loads(raw)
            path.write_bytes(raw)
            write(receipt, {'source_url': url, 'action': 'historical_request', 'status': status,
                'retrieved_at': datetime.now(timezone.utc).isoformat(), 'raw_file': str(path),
                'bytes': len(raw), 'attempts': attempt+1, 'previous_errors': errors})
            return payload, True
        except (HTTPError, OSError, ValueError) as error:
            errors.append(str(error))
            code = error.code if isinstance(error, HTTPError) else None
            if attempt == 2 or (code is not None and code < 500 and code != 429):
                write(receipt, {'source_url': url, 'action': 'failed_historical_request',
                    'checked_at': datetime.now(timezone.utc).isoformat(), 'errors': errors})
                raise
            time.sleep((8, 20)[attempt] if code == 429 else (2, 5)[attempt])


def catalog(item):
    comp, slug = item
    url = f'https://sports.core.api.espn.com/v2/sports/soccer/leagues/{slug}/seasons/{YEAR}/teams?limit=1000'
    payload, downloaded = fetch(f'core_catalog_{comp}', url)
    if payload['pageCount'] != 1 or payload['count'] != COUNTS[comp] or len(payload['items']) != COUNTS[comp]:
        raise ValueError(f'Unresolved historical membership coverage for {comp}')
    teams = []
    for item in payload['items']:
        match = re.fullmatch(r'http[s]?://sports\.core\.api\.espn\.com/v2/sports/soccer/leagues/'+re.escape(slug)+f'/seasons/{YEAR}'+r'/teams/([0-9]+)\?lang=en&region=us', item['$ref'])
        if not match:
            raise ValueError('Historical membership reference identity differs')
        teams.append({'competition': comp, 'league_slug': slug, 'team_id': match.group(1),
                      'historical_membership_url': item['$ref']})
    if len({t['team_id'] for t in teams}) != len(teams):
        raise ValueError('Repeated historical team membership')
    return comp, teams, downloaded


def schedule(team):
    identifier = team['team_id']
    url = f'https://site.api.espn.com/apis/site/v2/sports/soccer/all/teams/{identifier}/schedule?season={YEAR}&dates={YEAR}0701-{YEAR+1}0630'
    try:
        payload, downloaded = fetch(f'team_{identifier}', url)
        if str(payload['team']['id']) != identifier:
            raise ValueError('Returned club ID differs from historical membership')
        events = payload.get('events', [])
        dates = [e['date'] for e in events]
        in_window, outside = [], []
        for value in dates:
            date = datetime.fromisoformat(value.replace('Z', '+00:00'))
            if datetime(YEAR,7,1,tzinfo=timezone.utc) <= date < datetime(YEAR+1,7,1,tzinfo=timezone.utc):
                in_window.append(value)
            else:
                outside.append(value)
        if len({e['id'] for e in events}) != len(events) or not events or not in_window:
            raise ValueError('Empty or duplicate event schedule')
        return {**team, 'name': payload['team']['displayName'], 'events': len(events),
                'in_window_events': len(in_window), 'excluded_outside_window_dates': outside,
                'earliest': min(in_window), 'latest': max(in_window), 'downloaded': downloaded,
                'file': str(OUT/'raw'/f'team_{identifier}.json'), 'source_url': url, 'error': None}
    except Exception as error:
        return {**team, 'source_url': url, 'error': str(error)}


def main():
    global YEAR, OUT, COUNTS
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, choices=(2022, 2023, 2024, 2025), default=2024)
    YEAR = parser.parse_args().year
    OUT = ROOT/f'outputs/全赛事赛程可用性_20261025/{YEAR}-{str(YEAR+1)[-2:]}全量'
    COUNTS = {'E0': 20, 'D1': 18, 'SP1': 20, 'I1': 20, 'F1': 20 if YEAR == 2022 else 18}
    expected = sum(COUNTS.values())
    (OUT/'raw').mkdir(parents=True, exist_ok=True)
    (OUT/'receipts').mkdir(parents=True, exist_ok=True)
    plan = {'scope': f'Five leagues historical {YEAR}/{str(YEAR+1)[-2:]} membership and July{YEAR}-June{YEAR+1} recorded events',
            'started_at': datetime.now(timezone.utc).isoformat(), 'max_concurrency': 2,
            'max_attempts': 3, 'catalog': f'historical core API seasons/{YEAR}/teams; site-level season query rejected as current membership',
            'teams_expected': expected, 'new_live_results_requested': False, 'lineups_read': False,
            'raw_top_level_current_team_stats': 'ignored; event dates independently restricted to historical window'}
    write(OUT/'全量接入计划.json', plan)
    teams, catalogs = [], {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for comp, local, downloaded in pool.map(catalog, SLUGS.items()):
            teams.extend(local)
            catalogs[comp] = {'teams': len(local), 'downloaded': downloaded}
    if len(teams) != expected or len({t['team_id'] for t in teams}) != expected:
        raise ValueError('Historical club catalog coverage unresolved')
    write(OUT/'历史参赛身份目录.json', {'catalogs': catalogs, 'clubs': teams})
    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(schedule, teams):
            results.append(result)
            progress = {'status': 'running', 'updated_at': datetime.now(timezone.utc).isoformat(),
                        'target_clubs': len(teams), 'completed_clubs': len(results),
                        'failed_clubs': sum(r['error'] is not None for r in results), 'results': results}
            write(OUT/'接入进度.json', progress)
            print(json.dumps({'completed': len(results), 'club': result.get('name', result['team_id']),
                              'events': result.get('events'), 'error': result['error']}, ensure_ascii=False), flush=True)
    progress['status'] = 'completed' if not progress['failed_clubs'] else 'completed_with_missing'
    write(OUT/'接入进度.json', progress)
    print(json.dumps({'directory': str(OUT), 'completed': len(results), 'failed': progress['failed_clubs']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
