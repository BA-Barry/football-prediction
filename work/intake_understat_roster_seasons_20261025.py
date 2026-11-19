"""Retrieve two complete historical EPL roster seasons, with resumable receipts."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import shutil
import time

import requests

from football_model.football_data import load_directory
from football_model.xg_history import join_understat

ROOT = REPOSITORY_ROOT
PILOT = ROOT/'data/understat/roster_pilot_20261024T183946498125Z'
DEFAULT = ROOT/'data/understat/roster_seasons_EPL_2023_2024_20261025'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def select():
    result = []
    paths = {2023: ROOT/'data/football-data/2023-24/20261022T113804786593Z',
             2024: ROOT/'outputs/Football-Data_2024-25_个人使用_20261022T111235Z'}
    for year, directory in paths.items():
        season = f'{year}/{str(year + 1)[-2:]}'
        independent, report = load_directory(directory, season=season)
        if report['rejected_rows'] or report['duplicate_keys']:
            raise ValueError('Independent source contains malformed or duplicate records')
        source = ROOT/f'data/understat/history_20261024T170717354684Z/EPL_{year}.json'
        joined, report = join_understat(read(source)['dates'], independent, season, 'E0')
        if report['missing_independent_match_ids'] or len(joined) != 380:
            raise ValueError('Full season requires all 380 independently joined fixtures')
        for row in sorted(joined, key=lambda r: (r.match_date, r.match_id)):
            result.append({'match_id': row.match_id, 'understat_id': row.understat_id,
                'competition': 'E0', 'season': season, 'date': row.match_date.isoformat(),
                'home': row.home, 'away': row.away, 'source_url': row.source_url})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    args = parser.parse_args()
    target = args.directory.resolve()
    if not target.is_relative_to(REPOSITORY_ROOT):
        raise ValueError('Archive storage must remain below the project root')
    target.mkdir(parents=True, exist_ok=True)
    (target/'raw').mkdir(exist_ok=True)
    (target/'receipts').mkdir(exist_ok=True)
    selected = select()
    selection_path = target/'比赛选择.json'
    if selection_path.exists():
        if read(selection_path)['selected'] != selected:
            raise ValueError('Existing archive selects different fixtures; inspect before reuse')
    else:
        write(selection_path, {'created_at': stamp(), 'rule': 'all independently exact-joined EPL matches in 2023/24 and 2024/25',
            'expected_matches': len(selected), 'score_used_for_selection': False,
            'current_results_requested': False, 'selected': selected})
    pilot_receipts = {r['understat_id']: r for r in read(PILOT/'接入记录.json')['responses']}
    reused, fetched, existing = 0, 0, 0
    failed = None
    started = stamp()

    def progress(status):
        state = {'started_at': started, 'updated_at': stamp(), 'process_id': os.getpid(),
            'status': status, 'expected_matches': len(selected), 'already_present': existing,
            'reused_from_pilot': reused, 'newly_fetched': fetched, 'completed': existing + reused + fetched,
            'failure': failed, 'historical_lineup_publication_time_known': False,
            'model_fit_count': 0, 'current_results_requested': False,
            'purpose': 'personal noncommercial historical roster analysis',
            'rights': 'Underlying data rights are separate from reader code licensing; no commercial use asserted.'}
        temp = target/'进度.tmp'
        write(temp, state)
        temp.replace(target/'进度.json')

    progress('running')
    with requests.Session() as session:
        session.headers.update({'Accept': 'application/json', 'X-Requested-With': 'XMLHttpRequest'})
        for fixture in selected:
            uid = fixture['understat_id']
            raw_path = target/'raw'/f'{uid}_response.bin'
            parsed_path = target/'raw'/f'{uid}.json'
            record_path = target/'receipts'/f'{uid}.json'
            if record_path.exists() and raw_path.exists() and parsed_path.exists():
                record = read(record_path)
                if record.get('status_code') == 200 and not record.get('error'):
                    parsed = read(parsed_path)
                    if not isinstance(parsed, dict):
                        raise ValueError('Saved response no longer decodes to an object')
                    existing += 1
                    continue
            pilot = pilot_receipts.get(uid)
            if pilot and pilot.get('status_code') == 200 and not pilot.get('error'):
                parsed = read(PILOT/f'{uid}.json')
                shutil.copyfile(PILOT/f'{uid}_response.bin', raw_path)
                write(parsed_path, parsed)
                write(record_path, {**pilot, 'reused_from': str(PILOT), 'copied_at': stamp()})
                reused += 1
            else:
                record = {'understat_id': uid, 'url': f'https://understat.com/getMatchData/{uid}', 'requested_at': stamp()}
                try:
                    response = session.get(record['url'], headers={'Referer': fixture['source_url']},
                                           timeout=25, allow_redirects=False)
                    raw_path.write_bytes(response.content)
                    record.update(received_at=stamp(), status_code=response.status_code, final_url=response.url,
                        bytes=len(response.content), content_type=response.headers.get('Content-Type'))
                    if response.status_code != 200:
                        record['error'] = 'HTTP failure; stop without automatic retry or alternate route'
                    else:
                        parsed = response.json()
                        if not isinstance(parsed, dict):
                            raise ValueError('Response JSON must be an object')
                        write(parsed_path, parsed)
                        fetched += 1
                except (requests.RequestException, ValueError) as error:
                    record['error'] = str(error)
                write(record_path, record)
                if record.get('error'):
                    failed = record
                    progress('stopped_on_failure')
                    print(json.dumps({'directory': str(target), 'failure': failed}, ensure_ascii=False), flush=True)
                    return
                time.sleep(0.1)
            completed = existing + reused + fetched
            if completed % 25 == 0:
                progress('running')
                print(json.dumps({'completed': completed, 'expected': len(selected), 'new_requests': fetched,
                                  'last_understat_id': uid}, ensure_ascii=False), flush=True)
    progress('completed')
    print(json.dumps({'directory': str(target), 'completed': existing + reused + fetched,
                      'new_requests': fetched, 'reused_from_pilot': reused, 'already_present': existing}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
