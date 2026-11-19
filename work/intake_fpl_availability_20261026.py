"""Dated FPL archive observations at existing historical forecast cutoffs.

Repository commit times are retrospective availability proxies, not proof of
first public medical announcements. This intake never uses a target result.
"""
from bisect import bisect_left
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import json
import lzma
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import re
import time

import requests


ROOT = REPOSITORY_ROOT
DATA = ROOT/'data/fpl_availability_archive'
SOURCE = ROOT/'outputs/固定发布日期信息刷新_20261025T071806561562Z/features_lead3.json.gz'
OUT = ROOT/'outputs/FPL历史可用性接入_20261026'
ALIASES = {'Man Utd': 'Man United', 'Spurs': 'Tottenham', 'Nott\'m Forest': "Nott'm Forest",
           'Sheffield Utd': 'Sheffield United'}
STATUSES = ('a', 'i', 'd', 's', 'u')
MEASUREMENTS = ('roster_count', 'injury_flag_count', 'doubtful_flag_count',
                'suspension_flag_count', 'other_unavailable_flag_count',
                'unknown_status_count', 'snapshot_age_hours')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def utc(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(timezone.utc)


def archive_index():
    records, current = [], None
    for line in (DATA/'added-file-history.txt').read_text(encoding='utf-8-sig').splitlines():
        if line.startswith('COMMIT\t'):
            _, commit, timestamp = line.split('\t')
            current = {'commit': commit, 'repository_recorded_at': timestamp}
        elif line.startswith('A\tcache/'):
            path = line.split('\t', 1)[1]
            match = re.fullmatch(r'cache/(\d{4})/(\d{1,2})/(\d{1,2})/(\d{2})(\d{2})\.json\.xz', path)
            if not match:
                continue
            year, month, day, hour, minute = map(int, match.groups())
            named_at = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
            rec = current | {'path': path, 'filename_observed_at': named_at.isoformat()}
            # Both the dated filename and the archived commit must be earlier.
            rec['availability_proxy_at'] = max(named_at, utc(current['repository_recorded_at'])).isoformat()
            records.append(rec)
    records.sort(key=lambda r: (r['availability_proxy_at'], r['path']))
    return records


def choose(index, timestamps, forecast_before_date):
    cutoff = utc(forecast_before_date+'T00:00:00Z')
    pos = bisect_left(timestamps, cutoff)-1
    return None if pos < 0 else index[pos]


def download(record, earlier=None):
    relative = Path(record['path']).relative_to('cache')
    local = DATA/'raw'/relative
    local.parent.mkdir(parents=True, exist_ok=True)
    url = 'https://raw.githubusercontent.com/Randdalf/fplcache/'+record['commit']+'/'+record['path']
    received = (earlier or {}).get('received_at', datetime.now(timezone.utc).isoformat())
    status, error = 'reused', None
    if not local.exists():
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=(15, 60))
                response.raise_for_status()
                # Decode and check the expected wrapper before retaining it.
                payload = json.loads(lzma.decompress(response.content))
                if not isinstance(payload.get('elements'), list) or not isinstance(payload.get('teams'), list):
                    raise ValueError('Expected bootstrap-static player and team lists')
                local.write_bytes(response.content)
                status = 'downloaded'
                break
            except (requests.RequestException, ValueError, lzma.LZMAError) as exc:
                error = str(exc)
                if attempt < 2:
                    time.sleep(attempt+1)
        else:
            return record | {'source_url': url, 'received_at': received, 'status': 'failed', 'error': error}
    payload = json.loads(lzma.decompress(local.read_bytes()))
    team_rows = {}
    field_missing = Counter()
    by_team = {}
    for player in payload['elements']:
        by_team.setdefault(player.get('team'), []).append(player)
        for field in ('id', 'code', 'team', 'status', 'news', 'news_added'):
            if field not in player:
                field_missing[field] += 1
    for team in payload['teams']:
        name = ALIASES.get(team['name'], team['name'])
        key = 'football-data:E0:'+name
        players = by_team.get(team['id'], [])
        counts = Counter(player.get('status') for player in players)
        status_complete = bool(players) and all(player.get('status') is not None for player in players)
        team_rows[key] = {
            'source_team_id': team['id'], 'source_team_code': team.get('code'), 'source_name': team['name'],
            'empty_roster': not bool(players), 'status_fields_complete': status_complete,
            'measurements': {
                'roster_count': len(players) if players else None,
                'injury_flag_count': counts['i'] if status_complete else None,
                'doubtful_flag_count': counts['d'] if status_complete else None,
                'suspension_flag_count': counts['s'] if status_complete else None,
                'other_unavailable_flag_count': counts['u'] if status_complete else None,
                'unknown_status_count': sum(v for k, v in counts.items() if k not in STATUSES) if players else None},
            'raw_status_counts': {str(k): v for k, v in counts.items()},
            'flagged_players': [{k: player.get(k) for k in (
                'id', 'code', 'first_name', 'second_name', 'web_name', 'element_type',
                'status', 'news', 'news_added', 'chance_of_playing_this_round',
                'chance_of_playing_next_round', 'minutes', 'starts')}
                for player in players if player.get('status') != 'a']}
    return record | {'source_url': url, 'received_at': received, 'status': status, 'local_path': str(local),
                     'compressed_bytes': local.stat().st_size, 'players': len(payload['elements']),
                     'missing_fields': dict(field_missing), 'teams': team_rows}


def attach(row, observation):
    cutoff = utc(row['forecast_before_date']+'T00:00:00Z')
    known = observation is not None and observation.get('status') != 'failed'
    age = None if not known else (cutoff-utc(observation['filename_observed_at'])).total_seconds()/3600
    own = []
    for team in (row['home'], row['away']):
        found = None if not known else observation['teams'].get(team)
        values = {k: None for k in MEASUREMENTS}
        if found and not found['empty_roster']:
            values.update(found['measurements'], snapshot_age_hours=age)
        own.append(values)
    side_features = []
    for side in range(2):
        extra = {'fpl_own_'+k: v for k, v in own[side].items()}
        extra.update({'fpl_opponent_'+k: v for k, v in own[1-side].items()})
        side_features.append(dict(row['fresh_performance_features'][side]) | extra)
    context = {k: row[k] for k in ('match_id', 'date', 'period', 'competition', 'home', 'away',
               'fit_before_date', 'forecast_before_date', 'horizon_days', 'actual_score', 'actual_outcome')}
    return context | {'side_features': side_features,
        'reference_features': row['fresh_performance_features'],
        'goal_rates': row['goal_rates'], 'xg_rates': row['xg_rates'],
        'archive_observation': None if observation is None else {
            k: observation.get(k) for k in ('path', 'commit', 'filename_observed_at',
                'repository_recorded_at', 'availability_proxy_at', 'status', 'source_url')},
        'snapshot_age_hours': age,
        'team_available': [value['roster_count'] is not None for value in own]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(SOURCE, 'rt', encoding='utf-8') as handle:
        rows = [r for r in json.load(handle)['rows'] if r['competition'] == 'E0']
    index = archive_index()
    timestamps = [utc(r['availability_proxy_at']) for r in index]
    chosen = {day: choose(index, timestamps, day) for day in sorted({r['forecast_before_date'] for r in rows})}
    wanted = {rec['path']: rec for rec in chosen.values() if rec is not None}
    write(OUT/'接入规则.json', {
        'created_at': datetime.now(timezone.utc).isoformat(), 'feature_source': str(SOURCE),
        'repository': 'https://github.com/Randdalf/fplcache', 'horizon_days': 3,
        'availability_rule': 'latest filename-observed time AND first-added commit timestamp strictly before forecast date 00:00 UTC',
        'time_interpretation': 'Git repository observation proxy reconstructed retrospectively; commit time is not proof of first public announcement or push time',
        'feature_block': list(MEASUREMENTS), 'added_columns_per_side': 2*len(MEASUREMENTS),
        'not_features': ['news text', 'news_added', 'chance fields', 'minutes', 'starts', 'confirmed lineup'],
        'unknown_rule': 'retain all targets; missing archive/team/field is not healthy or zero flags',
        'scope': 'England only; archived FPL lists do not cover other leagues',
        'target_use': 'intake and feature construction do not inspect goals/outcome',
        'current_results_checked': False, 'current_forecasts_replaced': False,
        'wanted_snapshots': len(wanted), 'rows': len(rows)})
    observations = {}
    earlier_path = DATA/'selected_snapshot_manifest.json'
    earlier = {} if not earlier_path.exists() else {
        r['path']: r for r in json.loads(earlier_path.read_text(encoding='utf-8'))['observations']}
    print(f'INTAKE {len(rows)} EPL rows, {len(wanted)} distinct snapshots', flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(download, rec, earlier.get(path)): path for path, rec in wanted.items()}
        for n, future in enumerate(as_completed(futures), 1):
            rec = future.result()
            observations[rec['path']] = rec
            if n % 40 == 0 or n == len(futures):
                print(f'SNAPSHOTS {n}/{len(futures)}; failures '+str(sum(r['status']=='failed' for r in observations.values())), flush=True)
    write(DATA/'selected_snapshot_manifest.json', {'observations': [observations[k] for k in sorted(observations)]})
    prepared = [attach(row, None if chosen[row['forecast_before_date']] is None
                       else observations[chosen[row['forecast_before_date']]['path']]) for row in rows]
    with gzip.open(OUT/'逐场赛前状态特征.json.gz', 'wt', encoding='utf-8') as handle:
        json.dump({'rows': prepared}, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    report = {'rows': len(prepared), 'snapshot_n': len(wanted),
        'download_status': dict(Counter(r['status'] for r in observations.values())),
        'compressed_bytes': sum(r.get('compressed_bytes', 0) for r in observations.values()),
        'archive_commits': len(index), 'by_period': {}}
    extra_statuses = Counter()
    extra_snapshots = 0
    for observation in observations.values():
        found_extra = False
        for team in observation.get('teams', {}).values():
            for status, count in team['raw_status_counts'].items():
                if status not in STATUSES:
                    extra_statuses[status] += count
                    found_extra = True
        extra_snapshots += found_extra
    report['unexplained_source_status_counts'] = dict(extra_statuses)
    report['snapshots_with_unexplained_status'] = extra_snapshots
    report['unexplained_status_note'] = 'unknown_status_count includes other source labels such as n, not necessarily missing fields; not classified as injured or healthy'
    for period in sorted({r['period'] for r in rows}):
        group = [r for r in prepared if r['period'] == period]
        ages = sorted(r['snapshot_age_hours'] for r in group if r['snapshot_age_hours'] is not None)
        report['by_period'][period] = {
            'n': len(group), 'both_teams_observed': sum(all(r['team_available']) for r in group),
            'one_team_observed': sum(sum(r['team_available']) == 1 for r in group),
            'neither_observed': sum(not any(r['team_available']) for r in group),
            'age_hours_min': min(ages) if ages else None, 'age_hours_max': max(ages) if ages else None,
            'age_hours_median': ages[len(ages)//2] if ages else None}
    # Mutation check specifically verifies target information is not a feature.
    changed = dict(rows[-1], actual_score=[19, 19], actual_outcome=1)
    observed = observations[chosen[changed['forecast_before_date']]['path']]
    original = prepared[-1]
    report['target_mutation_features_unchanged'] = attach(changed, observed)['side_features'] == original['side_features']
    for rec in observations.values():
        if rec.get('status') == 'failed':
            continue
        if utc(rec['filename_observed_at']) > utc(rec['repository_recorded_at']):
            report.setdefault('filename_after_commit_paths', []).append(rec['path'])
    write(OUT/'历史覆盖实测.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
