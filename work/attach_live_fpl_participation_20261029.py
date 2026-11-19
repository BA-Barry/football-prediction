"""Observe limited current EPL personnel and their prior five same-team games.

No target lineup, target minutes, match score, or fitted personnel effect is used.
History rows are joined through an actual fixture side and opponent rather than
assuming every past appearance belongs to the player's current source team.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = REPOSITORY_ROOT
OUT = ROOT / 'data/epl_personnel_live_20261029'
OLD_DIR = ROOT / 'data/current_rich_inputs/run_20261026T025507813598Z'
WANTED_TEAM_CODES = (8, 1, 91, 2, 94, 17)
OTHER_CATEGORIES = {'nonmedical_absence_reference', 'eligibility_reference',
                    'selection_restriction_reference', 'suspension_reference'}


def now():
    return datetime.now(timezone.utc).isoformat()


def utc(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(timezone.utc)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def scope():
    bootstrap = read(OUT / 'raw/bootstrap-static.json')
    personnel = read(OUT / '当前球员新闻.json')
    old = read(OLD_DIR / 'official-fpl/bootstrap.json')
    old_report = read(OLD_DIR / 'coverage_report.json')
    teams = {t['id']: t for t in bootstrap['teams']}
    old_teams = {t['id']: t for t in old['teams']}
    old_players = {(old_teams[p['team']]['code'], p['code']): p for p in old['elements']}
    selected, excluded = [], []
    for player in personnel['players']:
        if player['source_team_code'] not in WANTED_TEAM_CODES:
            continue
        old_player = old_players.get((player['source_team_code'], player['source_player_code']))
        previous = old_player.get('status') if old_player else None
        current = player['source_status']
        reason = ('current_source_i_d_s' if current in ('i', 'd', 's') else
                  'current_a_after_earlier_i_d_s' if current == 'a' and previous in ('i', 'd', 's') else
                  'other_nonmedical_wording' if OTHER_CATEGORIES.intersection(player['semantics']['text_categories']) else None)
        if not reason:
            if current != 'a':
                excluded.append({'source_player_code': player['source_player_code'],
                                 'source_team_code': player['source_team_code'],
                                 'source_status': current,
                                 'text_categories': player['semantics']['text_categories'],
                                 'reason': 'Not current i/d/s, a source-a return candidate, or specified nonmedical wording; transfer-only u is not an injury.'})
            continue
        selected.append(dict(player, selection_reason=reason, earlier_snapshot_observation={
            'path': str(OLD_DIR / 'official-fpl/bootstrap.json'),
            'run_started_at_utc': old_report['created_at'], 'run_completed_at_utc': old_report['completed_at'],
            'same_team_code_and_player_code': old_player is not None,
            'earlier_status': previous, 'earlier_news': None if old_player is None else old_player.get('news'),
            'earlier_news_added': None if old_player is None else old_player.get('news_added'),
            'current_status': current,
            'interpretation': 'A change between two source snapshots; not an observed medical onset, recovery, or actual return.'}))
    before_limit = len(selected)
    if before_limit > 40:
        selected = [p for p in selected if p['selection_reason'] != 'other_nonmedical_wording']
    summary = {'eligible_before_endpoint_limit': before_limit, 'selected_endpoint_count': len(selected),
               'other_nonmedical_excluded_because_over_40': before_limit > 40,
               'by_selection_reason': dict(Counter(p['selection_reason'] for p in selected)),
               'by_team': dict(Counter(p['source_team_name'] for p in selected)),
               'selected': [{'id': p['source_player_id'], 'code': p['source_player_code'],
                             'name': p['web_name'], 'team': p['source_team_name'],
                             'source_status': p['source_status'], 'reason': p['selection_reason']} for p in selected],
               'other_source_flags_excluded': excluded}
    return selected, summary, teams


def fetch(player, out=None):
    destination = OUT if out is None else Path(out)
    player_id = player['source_player_id']
    url = 'https://fantasy.premierleague.com/api/element-summary/' + str(player_id) + '/'
    rec = {'source_player_id': player_id, 'source_player_code': player['source_player_code'],
           'source_url': url, 'requested_at_utc': now(), 'attempt': 1}
    body = None
    try:
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0 FPL research observation',
                                    'Accept': 'application/json', 'Cache-Control': 'no-cache'})
        with urlopen(req, timeout=45) as response:
            body = response.read()
            rec.update(http_status=response.status, final_url=response.url,
                       response_headers={k: v for k, v in response.headers.items()
                                         if k.lower() in ('date', 'last-modified', 'age', 'content-type', 'cache-control')})
        rec['received_at_utc'] = now()
        payload = json.loads(body)
        if not isinstance(payload, dict) or not isinstance(payload.get('history'), list):
            raise ValueError('Expected current-season history list')
        target = destination / 'raw/element-summary' / (str(player_id) + '.json')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        rec.update(status='downloaded', raw_file=str(target), bytes=len(body))
        return payload, rec
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        rec.update(received_at_utc=now(), status='failed', error=str(exc))
        if isinstance(exc, HTTPError):
            rec['http_status'] = exc.code
            body = exc.read()
        if body is not None:
            target = destination / 'raw/element-summary' / (str(player_id) + '.response')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            rec.update(raw_error_file=str(target), bytes=len(body))
        return None, rec


def fixture_side(fixture, team_id, teams):
    home = fixture['team_h'] == team_id
    opponent_id = fixture['team_a'] if home else fixture['team_h']
    return {'source_fixture_id': fixture['id'], 'kickoff_time': fixture['kickoff_time'],
            'event': fixture.get('event'), 'was_home': home,
            'source_team_id': team_id, 'source_team_code': teams[team_id]['code'],
            'source_team_name': teams[team_id]['name'],
            'opponent_source_team_id': opponent_id, 'opponent_source_team_code': teams[opponent_id]['code'],
            'opponent_source_team_name': teams[opponent_id]['name'],
            'finished': fixture.get('finished')}


def player_record(player, payload, request, team_fixtures, fixture_by_id, teams):
    # Only the current-season history, not history_past or upcoming fixtures.
    by_fixture = {}
    mapped = []
    rejected = []
    for row in ([] if payload is None else payload['history']):
        fixture = fixture_by_id.get(row.get('fixture'))
        if fixture is None or row.get('was_home') not in (True, False):
            rejected.append({'source_fixture_id': row.get('fixture'), 'reason': 'Unknown fixture or missing side'})
            continue
        actual_team_id = fixture['team_h'] if row['was_home'] is True else fixture['team_a']
        actual_opponent_id = fixture['team_a'] if row['was_home'] is True else fixture['team_h']
        if row.get('element') != player['source_player_id'] or row.get('opponent_team') != actual_opponent_id:
            rejected.append({'source_fixture_id': row.get('fixture'), 'reason': 'Element or opponent does not match the fixture side'})
            continue
        if teams[actual_team_id]['code'] != player['source_team_code']:
            rejected.append({'source_fixture_id': row.get('fixture'), 'reason': 'Historical row belongs to another team',
                             'actual_source_team_code': teams[actual_team_id]['code']})
            continue
        evidence = {'history_element': row.get('element'), 'history_fixture': row.get('fixture'),
                    'history_was_home': row.get('was_home'), 'history_opponent_team': row.get('opponent_team'),
                    'fixture_home_source_team_id': fixture['team_h'], 'fixture_away_source_team_id': fixture['team_a'],
                    'derived_actual_source_team_id': actual_team_id,
                    'derived_actual_source_team_code': teams[actual_team_id]['code'],
                    'current_source_team_code': player['source_team_code'],
                    'same_current_team_confirmed_by_fixture_mapping': True}
        item = {'minutes': row.get('minutes'), 'starts': row.get('starts'),
                'history_kickoff_time': row.get('kickoff_time'), 'mapping_evidence': evidence}
        by_fixture.setdefault(row['fixture'], []).append(item)
        mapped.append({'source_fixture_id': row['fixture'], **item})
    recent = []
    for fixture in team_fixtures:
        matches = by_fixture.get(fixture['source_fixture_id'], [])
        if len(matches) == 1:
            recent.append(dict(fixture, history_record_state='present', **matches[0]))
        else:
            recent.append(dict(fixture, history_record_state='missing' if not matches else 'ambiguous_duplicate',
                               minutes=None, starts=None, mapping_evidence=None))
    def number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
    complete = len(recent) == 5 and all(r['history_record_state'] == 'present' and
                                       number(r['minutes']) and number(r['starts']) for r in recent)
    return dict(player, element_summary_request=request, previous_five_same_team_league_games=recent,
                previous_five_summary={'expected_fixture_count': 5, 'known_history_rows': sum(r['history_record_state'] == 'present' for r in recent),
                                       'complete_five_rows_and_fields': complete,
                                       'minutes_sum': sum(r['minutes'] for r in recent) if complete else None,
                                       'starts_sum': sum(r['starts'] for r in recent) if complete else None,
                                       'missing_is_zero': False},
                same_team_history_rows_seen=len(mapped), rejected_history_mapping_rows=rejected)


def main():
    selected, selection, teams = scope()
    if '--count-only' in sys.argv:
        print(json.dumps(selection, ensure_ascii=False, indent=2))
        return
    batch_started = now()
    cutoff = utc(batch_started)
    fixtures = read(OUT / 'raw/fixtures.json')
    # We read fixture dates, completion flags, IDs, and team identities only.
    past = [f for f in fixtures if f.get('finished') is True and f.get('kickoff_time')
            and utc(f['kickoff_time']) < cutoff]
    fixture_by_id = {f['id']: f for f in past}
    recent_by_team = {}
    for team_id, team in teams.items():
        if team['code'] not in WANTED_TEAM_CODES:
            continue
        rows = [f for f in past if team_id in (f['team_h'], f['team_a'])]
        rows.sort(key=lambda f: (f['kickoff_time'], f['id']), reverse=True)
        recent_by_team[team_id] = [fixture_side(f, team_id, teams) for f in reversed(rows[:5])]
    observed = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {pool.submit(fetch, player): player for player in selected}
        for job in as_completed(jobs):
            player = jobs[job]
            observed[player['source_player_id']] = job.result()
    players = [player_record(p, *observed[p['source_player_id']], recent_by_team[p['source_team_id']], fixture_by_id, teams)
               for p in selected]
    completed = now()
    value = {'batch_started_at_utc': batch_started, 'observation_completed_at_utc': completed,
             'past_fixture_cutoff_at_utc': batch_started,
             'bootstrap_observation_at_utc': read(OUT / '获取记录.json')['observation_completed_at_utc'],
             'source_urls': ['https://fantasy.premierleague.com/api/bootstrap-static/',
                             'https://fantasy.premierleague.com/api/fixtures/'] +
                            [p['element_summary_request']['source_url'] for p in players],
             'selection': selection,
             'last_five_fixtures_by_current_team': [{'source_team_id': tid, 'source_team_code': teams[tid]['code'],
                                                    'source_team_name': teams[tid]['name'], 'fixtures': rows}
                                                   for tid, rows in recent_by_team.items()],
             'players': players,
             'limits': ['Source flags and snapshot flag changes do not confirm health, onset, recovery or future availability.',
                        'Previous five games are identified from actually finished league fixtures before the actual cutoff.',
                        'No missing history row or field is imputed as zero; sums require all five matching rows with minutes and starts.',
                        'History maps through fixture, was_home and opponent to the actual team code; other-team rows are excluded.',
                        'These observations support contextual reading alongside recent form; no effect or strength adjustment is fitted.',
                        'No target lineup, target minutes, source chance calibration or completed-match score is projected.']}
    write(OUT / '较早参与记录.json', value)
    print(json.dumps({'selected_endpoints': len(players), 'http_status_counts': dict(Counter(p['element_summary_request'].get('http_status') for p in players)),
                      'download_status_counts': dict(Counter(p['element_summary_request']['status'] for p in players)),
                      'complete_five_players': sum(p['previous_five_summary']['complete_five_rows_and_fields'] for p in players),
                      'observation_completed_at_utc': completed,
                      'players': [{'name': p['web_name'], 'team': p['source_team_name'], 'reason': p['selection_reason'],
                                   **p['previous_five_summary']} for p in players]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
