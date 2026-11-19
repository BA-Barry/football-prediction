"""Repair current all-competition intake with day-level provider requests.

The fixed feature window ends at the original Oct26 data cutoff. All new
retrieval/generation times remain genuine; no old forecasts are rewritten.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from collections import Counter
from pathlib import Path
import argparse
import json

from refresh_current_all_competition_20261026 import (
    ROOT, MEMBERS, OFFICIAL_TEAMS, COMPETITIVE, FRIENDLIES, ALIASES, FPL_ALIASES,
    fetch, read, write,
)

CUTOFF = date(2026, 10, 26)
START = CUTOFF-timedelta(days=28)
DIRECTORY = ROOT/'data/current_all_competition/daily_20261027'


def members():
    rows = [r['team'] for sport in read(MEMBERS)['sports'] for league in sport['leagues'] for r in league['teams']]
    official = {r['name'] for r in read(OFFICIAL_TEAMS)['teams']}
    if len(rows) != 20 or len({r['id'] for r in rows}) != 20 or {FPL_ALIASES.get(r['displayName'], r['displayName']) for r in rows} != official:
        raise ValueError('Source and official club identities differ')
    return rows


def collect(target=DIRECTORY):
    target.mkdir(parents=True, exist_ok=True)
    (target/'raw').mkdir(exist_ok=True)
    (target/'receipts').mkdir(exist_ok=True)
    tasks = []
    for step in range((CUTOFF-START).days):
        day = START+timedelta(days=step)
        for slug in sorted(COMPETITIVE):
            label = f'{slug}_{day:%Y%m%d}'
            url = f'https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard?dates={day:%Y%m%d}&limit=1000'
            tasks.append((label, url, target))
    for club in members():
        tid = str(club['id'])
        tasks.append((f'team_{tid}', f'https://site.api.espn.com/apis/site/v2/sports/soccer/all/teams/{tid}/schedule?season=2026&dates=20260701-20261025', target))
    tasks.extend([('official_fpl_fixtures', 'https://fantasy.premierleague.com/api/fixtures/', target),
                  ('official_fpl_teams', 'https://fantasy.premierleague.com/api/bootstrap-static/', target)])
    counts = {'requests_or_existing_receipts': len(tasks), 'success': 0, 'failed': 0}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for index, (_, payload, receipt) in enumerate(pool.map(fetch, tasks), 1):
            counts['success' if payload is not None and receipt.get('status') == 200 else 'failed'] += 1
            if index % 50 == 0:
                print(json.dumps({'processed': index, **counts}), flush=True)
    write(target/'collection_progress.json', {**counts, 'finished_at': datetime.now(timezone.utc).isoformat(),
        'window': [START.isoformat(), CUTOFF.isoformat()], 'daily_competitions': sorted(COMPETITIVE),
        'publication_note': 'New retrieval, not evidence of historical publication availability.'})
    print(json.dumps(counts), flush=True)


def parse_event(event, default_slug=None):
    instant = datetime.fromisoformat(event['date'].replace('Z', '+00:00'))
    if instant.tzinfo is None:
        raise ValueError('Source event timestamp needs a time zone')
    when = instant.astimezone(timezone.utc).date()
    if not START <= when < CUTOFF:
        return None
    contests = event.get('competitions', [])
    if len(contests) != 1:
        raise ValueError('Soccer event needs exactly one contest')
    contest = contests[0]
    sides = {r['homeAway']: r for r in contest['competitors']}
    if len(contest['competitors']) != 2 or set(sides) != {'home', 'away'}:
        raise ValueError('Event needs two explicit home/away teams')
    slug = event.get('league', {}).get('slug') or default_slug
    status = contest.get('status', event.get('status', {})).get('type', {})
    category = 'competitive' if slug in COMPETITIVE else 'friendly_exhibition' if slug in FRIENDLIES else 'unclassified'
    completed = status.get('completed') is True
    return {'event_id': str(event['id']), 'match_date': when.isoformat(),
            'date_utc_text': event['date'], 'competition_slug': slug, 'category': category,
            'completed': completed, 'source_completed_flag': status.get('completed'),
            'status_name': status.get('name'),
            'home_team_id': str(sides['home']['team']['id']),
            'away_team_id': str(sides['away']['team']['id']),
            'home_name': sides['home']['team']['displayName'],
            'away_name': sides['away']['team']['displayName'],
            'is_regular_league_fixture': slug == 'eng.1' and event.get('seasonType', {}).get('name') != 'Relegation Playoff',
            'exclusion_reason': 'unclassified_competition' if category == 'unclassified' else
                'unresolved_source_completion_status' if category == 'competitive' and not completed else None}


def build(target=DIRECTORY):
    from football_model.allcomp_daily_intake import validate_daily_payload, merge_parsed_events, coverage_intervals
    from football_model.all_competition_features import AllCompetitionFeatures
    from refresh_current_all_competition_20261026 import load_current_schedule_clubs
    catalog = members()
    member_ids = {str(m['id']) for m in catalog}
    successful_days = set()
    daily_failures, daily_rows, checks = [], [], []
    for step in range((CUTOFF-START).days):
        day = START+timedelta(days=step)
        day_ok = True
        for slug in sorted(COMPETITIVE):
            label = f'{slug}_{day:%Y%m%d}'
            try:
                receipt = read(target/'receipts'/f'{label}.json')
                if receipt.get('status') != 200:
                    raise ValueError(f"HTTP {receipt.get('status')}")
                payload = read(target/'raw'/f'{label}.json')
                events = validate_daily_payload(payload, slug, day)
                for event in events:
                    parsed = parse_event(event, slug)
                    if parsed and member_ids & {parsed['home_team_id'], parsed['away_team_id']}:
                        daily_rows.append(parsed)
                checks.append({'competition': slug, 'day': day.isoformat(), 'events': len(events),
                               'received_at': receipt['received_at'], 'source_url': receipt['source_url']})
            except (ValueError, KeyError, TypeError, OSError) as error:
                daily_failures.append({'competition': slug, 'day': day.isoformat(), 'error': str(error)})
                day_ok = False
        if day_ok:
            successful_days.add(day)
    global_conflicts = merge_parsed_events(daily_rows)['conflicts']
    clubs = []
    for member in catalog:
        tid = str(member['id'])
        club_rows, issues = [], []
        try:
            receipt = read(target/'receipts'/f'team_{tid}.json')
            payload = read(target/'raw'/f'team_{tid}.json')
            if receipt.get('status') != 200 or str(payload.get('team', {}).get('id')) != tid or not isinstance(payload.get('events'), list):
                raise ValueError('Club response missing, malformed or wrong identity')
            for event in payload['events']:
                parsed = parse_event(event)
                if parsed:
                    if tid not in {parsed['home_team_id'], parsed['away_team_id']}:
                        raise ValueError('Requested club absent from event')
                    club_rows.append(parsed)
        except (ValueError, KeyError, TypeError, OSError) as error:
            issues.append({'scope': 'whole_window', 'error': str(error)})
        board_rows = [r for r in daily_rows if tid in {r['home_team_id'], r['away_team_id']}]
        merged = merge_parsed_events(club_rows+board_rows)
        blocked_dates = {date.fromisoformat(d) for d in merged['conflict_dates']}
        for conflict in global_conflicts:
            if tid in conflict['team_ids']:
                blocked_dates.update(date.fromisoformat(d) for d in conflict['match_dates'])
                issues.append({'event_id': conflict['event_id'], 'error': 'daily_competition_event_conflict'})
        for r in merged['events']:
            if r['category'] == 'unclassified' or r['exclusion_reason']:
                blocked_dates.add(date.fromisoformat(r['match_date']))
                issues.append({'event_id': r['event_id'], 'day': r['match_date'], 'error': r['exclusion_reason']})
        usable_days = set() if any(i.get('scope') == 'whole_window' for i in issues) else successful_days-blocked_dates
        intervals = coverage_intervals(usable_days)
        board_ids, club_ids = {r['event_id'] for r in board_rows}, {r['event_id'] for r in club_rows}
        canonical = ALIASES.get(member['displayName'], member['displayName'])
        clubs.append({'team': f'football-data:E0:{canonical}', 'competition': 'E0', 'source_team_id': tid,
            'source_name': member['displayName'], 'coverage': [[a.isoformat(), b.isoformat()] for a, b in intervals],
            'requested_coverage': [[START.isoformat(), CUTOFF.isoformat()]], 'events': merged['events'],
            'coverage_issues': issues, 'conflicts': merged['conflicts'],
            'club_only_event_ids': sorted(club_ids-board_ids), 'scoreboard_only_event_ids': sorted(board_ids-club_ids),
            'blocked_dates': sorted(d.isoformat() for d in blocked_dates),
            'formal_completed_count': sum(r['category'] == 'competitive' and r['completed'] and not r['exclusion_reason'] for r in merged['events']),
            'competition_counts': dict(Counter(r['competition_slug'] for r in merged['events']))})
    observed_times = [r['received_at'] for r in checks]
    schedule_path = target/'schedule_clubs.json'
    write(schedule_path, {'created_at': datetime.now(timezone.utc).isoformat(),
        'calendar_scope': [START.isoformat(), CUTOFF.isoformat()], 'clubs': clubs,
        'coverage_scope': 'club schedule reconciled to 13 daily competition scoreboards in requested 28-day window; not proof of provider exhaustive completeness',
        'daily_failures': daily_failures, 'daily_checks': checks,
        'known_future_fixtures_used_as_completed': False, 'source_first_publication_unknown': True})
    builder = AllCompetitionFeatures(load_current_schedule_clubs(schedule_path))
    features = {c.team: builder._snapshot('E0', c.team, CUTOFF)[0] for c in builder._clubs.values()}
    # Verify league membership and dates against today's official source, without settling forecasts.
    official = read(target/'raw/official_fpl_fixtures.json')
    official_teams = read(target/'raw/official_fpl_teams.json')
    fpl_names = {r['id']: r['name'] for r in official_teams['teams']}
    fpl_to_espn = {FPL_ALIASES.get(m['displayName'], m['displayName']): str(m['id']) for m in catalog}
    observed_league = {(r['match_date'], r['home_team_id'], r['away_team_id']) for r in daily_rows if r['competition_slug'] == 'eng.1' and r['completed']}
    official_league = set()
    for r in official:
        if r.get('kickoff_time') and r.get('finished') is True:
            day = datetime.fromisoformat(r['kickoff_time'].replace('Z', '+00:00')).date()
            if START <= day < CUTOFF:
                official_league.add((day.isoformat(), fpl_to_espn[fpl_names[r['team_h']]], fpl_to_espn[fpl_names[r['team_a']]]))
    missing_official = sorted(official_league-observed_league)
    extra_provider = sorted(observed_league-official_league)
    report = {'directory': str(target), 'schedule_clubs_file': str(schedule_path),
        'data_cutoff': CUTOFF.isoformat(), 'first_daily_received_at': min(observed_times) if observed_times else None,
        'last_daily_received_at': max(observed_times) if observed_times else None,
        'club_count': len(clubs), 'daily_requests': 13*(CUTOFF-START).days,
        'successful_daily_responses': len(checks), 'daily_failures': daily_failures,
        'clubs_with_usable_28d_count': sum(f['window_usable_28d'] == 1 for f in features.values()),
        'unique_formal_completed_events': len({r['event_id'] for c in clubs for r in c['events'] if r['category'] == 'competitive' and r['completed'] and not r['exclusion_reason']}),
        'event_conflicts': sum(len(c['conflicts']) for c in clubs),
        'club_only_ids': sorted({i for c in clubs for i in c['club_only_event_ids']}),
        'scoreboard_only_ids': sorted({i for c in clubs for i in c['scoreboard_only_event_ids']}),
        'official_league_dates_checked': len(official_league),
        'missing_official_league_fixtures': missing_official, 'extra_provider_league_fixtures': extra_provider,
        'model_fits': 0, 'forecasts_changed': 0,
        'interpretation': 'Observed past event counts, not measured fatigue or actual player minutes; planned future cup games not inputs.'}
    write(target/'current_feature_availability.json', features)
    write(target/'coverage_report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('fetch', 'build'))
    parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args()
    if args.action == 'fetch':
        collect(args.directory)
    else:
        build(args.directory)


if __name__ == '__main__':
    main()
