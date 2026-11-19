"""Archive current five-league inputs and check the already selected future EPL ten.

Personal, non-commercial research intake only. This script does not fit models,
produce forecasts, or settle/change any existing forecast or result record.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import io
import json
from pathlib import Path

import requests


ROOT = Path('G:/Codex/football-prediction-2026-10-22')
BATCH = ROOT / 'data/prematch-intelligence/20261024_batch/run_20261024T150900467071Z/batch_case.json'
SEASON = 2026
LEAGUES = [('EPL', 'E0'), ('Bundesliga', 'D1'), ('La_Liga', 'SP1'), ('Serie_A', 'I1'), ('Ligue_1', 'F1')]
ALIASES = {
    'AFC Bournemouth': 'Bournemouth', 'Brighton & Hove Albion': 'Brighton',
    'Leeds United': 'Leeds', 'Manchester United': 'Man Utd',
    'Manchester City': 'Man City', 'Newcastle United': 'Newcastle',
    'Nottingham Forest': "Nott'm Forest", 'Tottenham Hotspur': 'Spurs',
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value: object) -> None:
    with path.open('x', encoding='utf-8') as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write('\n')


def fetch(target: Path, label: str, url: str, headers: dict | None = None,
          follow_redirects: bool = False) -> tuple[dict, bytes]:
    receipt = {'source_url': url, 'request_started_at': now(), 'label': label}
    body = b''
    try:
        response = requests.get(url, headers=headers or {'User-Agent': 'FootballResearchPersonal/0.31'},
                                timeout=30, allow_redirects=follow_redirects)
        body = response.content
        receipt.update(received_at=now(), status=response.status_code,
                       response_url=response.url, headers=dict(response.headers),
                       response_date=response.headers.get('Date'), age_seconds=response.headers.get('Age'),
                       cache_control=response.headers.get('Cache-Control'), etag=response.headers.get('ETag'),
                       last_modified=response.headers.get('Last-Modified'), bytes=len(body),
                       raw_path=str(target / f'{label}_response.bin'),
                       redirects=[{'url': previous.url, 'status': previous.status_code,
                                   'headers': dict(previous.headers)} for previous in response.history])
        (target / f'{label}_response.bin').write_bytes(body)
    except requests.RequestException as error:
        receipt.update(received_at=now(), status=None, error=str(error), bytes=0)
    return receipt, body


def understat(target: Path) -> list[dict]:
    records = []
    for league, _ in LEAGUES:
        label = f'{league}_{SEASON}'
        url = f'https://understat.com/getLeagueData/{league}/{SEASON}'
        receipt, body = fetch(target, label, url, {
            'Accept': 'application/json', 'X-Requested-With': 'XMLHttpRequest',
            'Referer': f'https://understat.com/league/{league}/{SEASON}'})
        receipt.update(league=league, season_start_year=SEASON)
        if receipt.get('status') == 200:
            try:
                data = json.loads(body)
                save_json(target / f'{label}.json', data)
                dates = data.get('dates', [])
                completed = [r for r in dates if r.get('isResult') is True]
                teams = list(data.get('teams', {}).values())
                coverage = [{
                    'id': team.get('id'), 'title': team.get('title'),
                    'history_rows': len(team.get('history', [])),
                    'history_date_max': max((r.get('date', '') for r in team.get('history', [])), default=None),
                    'history_fields': sorted({k for row in team.get('history', []) for k in row}),
                } for team in teams]
                receipt.update(parsed_path=str(target / f'{label}.json'), top_level_keys=sorted(data),
                               match_rows=len(dates), completed_rows=len(completed),
                               unplayed_rows=sum(r.get('isResult') is False for r in dates),
                               unknown_status_rows=sum(r.get('isResult') not in (True, False) for r in dates),
                               completed_date_min=min((r.get('datetime', '') for r in completed), default=None),
                               completed_date_max=max((r.get('datetime', '') for r in completed), default=None),
                               team_count=len(teams), team_history_rows=sum(r['history_rows'] for r in coverage),
                               team_coverage=coverage)
            except (ValueError, TypeError, AttributeError) as error:
                receipt['parse_error'] = str(error)
        records.append(receipt)
        if receipt.get('status') in (401, 403, 429):
            break
    save_json(target / 'intake_metadata.json', {'created_at': now(), 'responses': records,
        'publication_time_warning': 'Download receipt is not original historical xG availability. Understat date strings are retained as supplied.',
        'data_rights': 'Personal non-commercial research; no commercial or unrestricted data licence is asserted.',
        'source_forecast_policy': 'Source forecast fields are archived but are not outputs of our model.'})
    return records


def football_data(target: Path) -> list[dict]:
    records = []
    for league, code in LEAGUES:
        receipt, body = fetch(target, code, f'https://www.football-data.co.uk/mmz4281/2627/{code}.csv',
                              follow_redirects=True)
        receipt.update(league=league, division=code, season='2026-27')
        if receipt.get('status') == 200:
            try:
                text = body.decode('utf-8-sig')
                rows = list(csv.DictReader(io.StringIO(text)))
                if not rows or not {'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'} <= set(rows[0]):
                    raise ValueError('Response does not contain expected match CSV fields')
                (target / f'{code}.csv').write_bytes(body)
                complete = []
                for row in rows:
                    try:
                        hg, ag = int(row['FTHG']), int(row['FTAG'])
                        if hg >= 0 and ag >= 0:
                            complete.append(row)
                    except (ValueError, TypeError):
                        pass
                dates = []
                for row in complete:
                    for fmt in ('%d/%m/%Y', '%d/%m/%y'):
                        try:
                            dates.append(datetime.strptime(row['Date'], fmt).date().isoformat())
                            break
                        except ValueError:
                            continue
                receipt.update(parsed_path=str(target / f'{code}.csv'), match_rows=len(rows),
                               completed_rows=len(complete), other_rows=len(rows)-len(complete),
                               completed_date_min=min(dates, default=None), completed_date_max=max(dates, default=None),
                               columns=list(rows[0]),
                               team_count=len({r[k] for r in rows for k in ('HomeTeam', 'AwayTeam')}))
            except (ValueError, UnicodeDecodeError) as error:
                receipt['parse_error'] = str(error)
        records.append(receipt)
        if receipt.get('status') in (401, 403, 429):
            break
    save_json(target / 'intake_metadata.json', {'created_at': now(), 'responses': records,
        'data_rights': 'Personal non-commercial research; underlying Football-Data terms apply.',
        'match_time_warning': 'CSV match dates are retained; no ingestion timestamp is treated as historical first availability.'})
    return records


def official_fixtures(target: Path) -> dict:
    records = []
    payloads = {}
    for label, url in [('fixtures', 'https://fantasy.premierleague.com/api/fixtures/'),
                       ('bootstrap', 'https://fantasy.premierleague.com/api/bootstrap-static/')]:
        receipt, body = fetch(target, label, url)
        if receipt.get('status') == 200:
            try:
                payloads[label] = json.loads(body)
                save_json(target / f'{label}.json', payloads[label])
                receipt['parsed_path'] = str(target / f'{label}.json')
            except (ValueError, TypeError) as error:
                receipt['parse_error'] = str(error)
        records.append(receipt)
        if receipt.get('status') in (401, 403, 429):
            break
    report = {'checked_at': now(), 'original_batch': str(BATCH), 'sources': records,
              'fixture_observations': [], 'forecasts_changed': 0, 'results_settled': 0}
    if {'fixtures', 'bootstrap'} <= set(payloads):
        team_ids = {t['name']: t['id'] for t in payloads['bootstrap']['teams']}
        report['teams'] = payloads['bootstrap']['teams']
        batch = json.loads(BATCH.read_text(encoding='utf-8-sig'))
        for record in batch['records']:
            old = record['fixture']
            home_id = team_ids.get(ALIASES.get(old['home'], old['home']))
            away_id = team_ids.get(ALIASES.get(old['away'], old['away']))
            old_time = datetime.fromisoformat(old['kickoff'].replace('Z', '+00:00'))
            exact = [r for r in payloads['fixtures'] if r.get('team_h') == home_id and r.get('team_a') == away_id
                     and r.get('kickoff_time') and datetime.fromisoformat(r['kickoff_time'].replace('Z', '+00:00')) == old_time]
            same_pair = [r for r in payloads['fixtures'] if r.get('team_h') == home_id and r.get('team_a') == away_id]
            observed = exact[0] if len(exact) == 1 else (same_pair[0] if len(same_pair) == 1 else None)
            item = {'original_fixture': old, 'home_team_id': home_id, 'away_team_id': away_id,
                    'exact_identity_count': len(exact), 'pair_count': len(same_pair),
                    'identity_and_kickoff_match': len(exact) == 1, 'confirmed_unstarted': False}
            if observed:
                item['source_observation'] = {k: observed.get(k) for k in (
                    'id', 'event', 'team_h', 'team_a', 'kickoff_time', 'started', 'finished',
                    'finished_provisional', 'minutes', 'team_h_score', 'team_a_score')}
                when = observed.get('kickoff_time')
                future = bool(when) and datetime.fromisoformat(when.replace('Z', '+00:00')) > datetime.now(timezone.utc)
                item['confirmed_unstarted'] = bool(len(exact) == 1 and future and observed.get('started') is False
                    and observed.get('finished') is False and observed.get('finished_provisional') is False)
            report['fixture_observations'].append(item)
    report['confirmed_unstarted_count'] = sum(r['confirmed_unstarted'] for r in report['fixture_observations'])
    save_json(target / 'confirmed_future_fixtures.json', report)
    return report


def main() -> None:
    created = datetime.now(timezone.utc)
    target = ROOT / 'data/current_rich_inputs' / created.strftime('run_%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True, exist_ok=False)
    paths = {k: target / k for k in ('understat', 'football-data', 'official-fpl')}
    for directory in paths.values():
        directory.mkdir()
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {k: pool.submit(function, paths[k]) for k, function in (
            ('understat', understat), ('football-data', football_data), ('official-fpl', official_fixtures))}
        results = {}
        for label, job in jobs.items():
            try:
                results[label] = job.result()
            except Exception as error:
                results[label] = {'unexpected_error': f'{type(error).__name__}: {error}'}
    compact = {}
    for provider in ('understat', 'football-data'):
        if isinstance(results.get(provider), list):
            compact[provider] = [{k: r[k] for k in (
                'league', 'status', 'match_rows', 'completed_rows', 'unplayed_rows', 'team_count',
                'completed_date_max', 'parsed_path', 'error', 'parse_error') if k in r} for r in results[provider]]
        else:
            compact[provider] = results.get(provider)
    compact['official-fpl'] = {k: results['official-fpl'].get(k) for k in (
        'confirmed_unstarted_count', 'unexpected_error') if k in results['official-fpl']}
    report = {'created_at': created.isoformat(), 'completed_at': now(), 'directory': str(target),
              'purpose': 'Current five-league rich feature input and original future EPL ten fixture verification',
              'coverage': compact, 'forecast_changes': 0, 'model_fits': 0, 'results_settled': 0,
              'received_at_is_not_first_publication': True,
              'cache_warning': 'Fresh receipt alone does not establish upstream freshness; all response and cache headers retained.',
              'paths': {k: str(p) for k, p in paths.items()}}
    save_json(target / 'coverage_report.json', report)
    lines = ['# 当前数据覆盖报告', '', f'取得时间：{report["completed_at"]}', '',
             '用途：为下一批尚未开赛英超比赛提供当前资料。未训练模型，未改写旧预测，未结算旧赛果。', '',
             '| 来源 | 联赛 | HTTP | 完赛行 | 赛程行 | 已完赛日期最大值 |', '|---|---|---:|---:|---:|---|']
    for provider in ('understat', 'football-data'):
        if isinstance(compact[provider], list):
            for item in compact[provider]:
                lines.append(f'| {provider} | {item.get("league")} | {item.get("status")} | {item.get("completed_rows", "")} | {item.get("match_rows", "")} | {item.get("completed_date_max", "")} |')
    lines.extend(['', f'官方 FPL 核实仍未开赛：{compact["official-fpl"].get("confirmed_unstarted_count", "未能核实")}/10 场。', '',
                  '保留原始响应、响应头和取得时间。取得时间不代表上游首次公开时间；原始 Understat 时间字符串不另猜时区。',
                  'Underlying data source terms apply; personal non-commercial research only.', ''])
    (target / 'coverage_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
