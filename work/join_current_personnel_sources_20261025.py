"""Join historical EPL identities; no prediction, training or result scoring."""
import csv
from datetime import date, datetime, timezone
import json
from pathlib import Path

import requests

from football_model.football_data import _date, parse_row
from football_model.xg_history import COMPETITION_TEAM_NAMES, join_understat

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
CURRENT = ROOT/'data/football-data/2026-27/personnel_20261025'
OUTPUT = ROOT/'outputs/当前人员历史代表性_20261025/独立比赛连接.json'
CUTOFF = date(2026, 10, 25)
URL = 'https://www.football-data.co.uk/mmz4281/2627/E0.csv'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def acquire():
    CURRENT.mkdir(parents=True, exist_ok=True)
    receipt_path, source_path = CURRENT/'receipt.json', CURRENT/'E0.csv'
    requested_url, redirects = URL, []
    if receipt_path.exists():
        previous = read(receipt_path)
        if previous.get('status_code') == 200 and not previous.get('error') and source_path.exists():
            return source_path, previous, True
        if previous.get('status_code') == 302 and previous.get('headers', {}).get('Location'):
            write(CURRENT/'receipt_initial_302.json', previous)
            redirects.append(previous)
            requested_url = previous['headers']['Location']
        else:
            raise RuntimeError('Prior unsuccessful receipt exists; no automatic retry or alternate source')
    receipt = {'url': requested_url, 'requested_at': stamp(),
               'prior_redirect_responses': redirects,
               'purpose': 'personal noncommercial historical personnel identity join'}
    try:
        response = requests.get(requested_url, headers={'User-Agent': 'Codex-personal-file-download/1.0'},
                                timeout=30)
        (CURRENT/'response.bin').write_bytes(response.content)
        receipt.update(status_code=response.status_code, received_at=stamp(), final_url=response.url,
                       bytes=len(response.content), headers=dict(response.headers))
        if response.status_code != 200:
            raise RuntimeError(f'HTTP {response.status_code}; stop without retry or alternate route')
        first = response.content.splitlines()[0] if response.content else b''
        if not first.startswith((b'Div,', b'\xef\xbb\xbfDiv,')):
            raise ValueError('Official response lacks the expected CSV header')
        source_path.write_bytes(response.content)
    except (requests.RequestException, ValueError, RuntimeError) as exc:
        receipt['error'] = str(exc)
        write(receipt_path, receipt)
        raise RuntimeError(str(exc)) from exc
    write(receipt_path, receipt)
    return source_path, receipt, False


def independent_rows(path, season):
    source_url = f'https://www.football-data.co.uk/mmz4281/{season[2:4]+season[-2:]}/E0.csv'
    records, rejected, issues, ignored = [], [], [], 0
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        for line, row in enumerate(reader, start=2):
            if not any(value and str(value).strip() for value in row.values()):
                continue
            try:
                # Exclude today/future before parsing any outcome or market fields.
                if _date((row.get('Date') or '').strip()) >= CUTOFF:
                    ignored += 1
                    continue
                record, row_issues = parse_row(row, competition='E0', season=season,
                                             source_path=str(path), source_url=source_url)
                records.append(record)
                issues.extend(row_issues)
            except (ValueError, TypeError) as exc:
                rejected.append({'line': line, 'error': str(exc)})
    ids = [r.match_id for r in records]
    duplicates = sorted({mid for mid in ids if ids.count(mid) > 1})
    return records, {'source_path': str(path), 'parsed_rows': len(records), 'rejected_rows': rejected,
        'duplicate_match_ids': duplicates, 'optional_field_issues_count': len(issues),
        'same_day_or_future_ignored_before_outcome_parse': ignored}


def main():
    report = {'created_at': stamp(), 'cutoff_exclusive': CUTOFF.isoformat(), 'summary': {},
        'matches': [], 'join_reports': {}, 'independent_sources': {}, 'errors': [],
        'unmatched_understat_rows': [],
        'model_fit_count': 0, 'prediction_scoring_count': 0,
        'availability_note': 'Downloaded now; does not establish original historical publication times.',
        'identity_rule': 'Exact provider-calendar date, explicit team aliases, and full-time score equality; no relaxed or fuzzy join.'}
    configurations = [
        ('2025/26', ROOT/'data/football-data/2025-26/20261022T113804786593Z/E0.csv',
         ROOT/'data/understat/history_20261024T170717354684Z/EPL_2025.json'),
        ('2026/27', CURRENT/'E0.csv', ROOT/'data/understat/current_directory_EPL_2026_20261025/EPL_2026.json')]
    season_counts = {}
    for season, fd_path, understat_path in configurations:
        if season == '2026/27':
            # Preserve the completed older season even if the new download fails.
            write(OUTPUT, report)
            try:
                fd_path, receipt, reused = acquire()
                report['current_download_receipt'] = receipt
                report['current_download_reused'] = reused
            except RuntimeError as exc:
                report['errors'].append({'stage': 'current_FD_download', 'error': str(exc)})
                continue
        records, source_report = independent_rows(fd_path, season)
        report['independent_sources'][season] = source_report
        if source_report['rejected_rows'] or source_report['duplicate_match_ids']:
            report['errors'].append({'season': season, 'stage': 'FD_parse', 'error': 'Rejected or duplicate identities; join not attempted'})
            continue
        payload = read(understat_path)
        rows = [row for row in payload['dates'] if row.get('isResult') is True
                and datetime.strptime(row['datetime'], '%Y-%m-%d %H:%M:%S').date() < CUTOFF]
        independent_keys = {(row.match_date, row.home_label, row.away_label) for row in records}
        aliases = COMPETITION_TEAM_NAMES['E0']
        joinable = []
        for row in rows:
            played = datetime.strptime(row['datetime'], '%Y-%m-%d %H:%M:%S').date()
            home = aliases.get(row['h']['title'], row['h']['title'])
            away = aliases.get(row['a']['title'], row['a']['title'])
            if (played, home, away) not in independent_keys:
                report['unmatched_understat_rows'].append({'season': season, 'understat_id': str(row['id']),
                    'date': played.isoformat(), 'home': f'football-data:E0:{home}',
                    'away': f'football-data:E0:{away}', 'verified': False,
                    'reason': 'No independently stored Football-Data fixture on this exact date and home/away pair'})
            else:
                joinable.append(row)
        try:
            joined, join_report = join_understat(joinable, records, season, 'E0')
        except (ValueError, KeyError, TypeError) as exc:
            report['errors'].append({'season': season, 'stage': 'exact_join', 'error': str(exc)})
            continue
        join_report['understat_source_path'] = str(understat_path)
        join_report['understat_completed_before_cutoff'] = len(rows)
        join_report['understat_without_independent_date_team_match'] = len(rows)-len(joinable)
        report['join_reports'][season] = join_report
        season_counts[season] = len(joined)
        report['matches'].extend({'understat_id': row.understat_id, 'verified': True,
            'football_data_match_id': row.match_id, 'match_id': row.match_id,
            'date': row.match_date.isoformat(), 'home': row.home, 'away': row.away,
            'competition': 'E0', 'season': season, 'source_url': row.source_url,
            'identity_basis': 'exact_FootballData_Understat_date_teams_full_score_join'} for row in joined)
    status = ('completed_with_errors' if report['errors'] else
              'completed_with_source_gaps' if report['unmatched_understat_rows'] else 'completed')
    report['summary'] = {'status': status,
        'verified_matches': len(report['matches']), 'verified_by_season': season_counts,
        'unmatched_understat_rows': len(report['unmatched_understat_rows']),
        'missing_independent_match_ids_by_season': {season: value['missing_independent_match_ids']
                                                   for season, value in report['join_reports'].items()},
        'error_count': len(report['errors'])}
    write(OUTPUT, report)
    print(json.dumps({'output': str(OUTPUT), 'summary': report['summary'], 'errors': report['errors']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
