"""Refresh the date-level team input from a saved official FPL observation.

Existing data are cross-checked, not relabelled as newly available in the past.
Only strictly earlier UTC-calendar results enter the existing date-level fit.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from shutil import copyfile

from football_model.current_projection import ALIASES, load_current_context

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OLD = ROOT / 'data/prospective_context/20261023T175004Z_completed'
FPL_NAMES = {
    'Arsenal': 'Arsenal FC', 'Aston Villa': 'Aston Villa FC',
    'Bournemouth': 'AFC Bournemouth', 'Brentford': 'Brentford FC',
    'Brighton': 'Brighton & Hove Albion FC', 'Chelsea': 'Chelsea FC',
    'Coventry City': 'Coventry City FC', 'Crystal Palace': 'Crystal Palace FC',
    'Everton': 'Everton FC', 'Fulham': 'Fulham FC', 'Hull City': 'Hull City AFC',
    'Ipswich Town': 'Ipswich Town FC', 'Leeds': 'Leeds United FC',
    'Liverpool': 'Liverpool FC', 'Man City': 'Manchester City FC',
    'Man Utd': 'Manchester United FC', 'Newcastle': 'Newcastle United FC',
    "Nott'm Forest": 'Nottingham Forest FC', 'Sunderland': 'Sunderland AFC',
    'Spurs': 'Tottenham Hotspur FC',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, payload):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--confirmed-case', type=Path)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    if snapshot.drive.upper() != 'G:':
        raise ValueError('Keep project observations on G')
    now = datetime.now(timezone.utc)
    metadata = read(snapshot / 'source_metadata.json')
    source = metadata['sources']['fixtures']
    received = datetime.fromisoformat(source['received_at'])
    if source['status'] != 200 or received > now:
        raise ValueError('Successful, already-received source required')
    teams = {r['id']: FPL_NAMES[r['name']] for r in read(snapshot / 'bootstrap_raw.json')['teams']}
    raw = read(snapshot / 'fixtures_raw.json')
    current, inspection, supplements = load_current_context(OLD, now)
    old = {(r.match_date.isoformat(), r.home_label, r.away_label): r.full_time for r in current}
    fresh = {}
    same_day_excluded = []
    for row in raw:
        if row.get('finished') is not True:
            continue
        day = datetime.fromisoformat(row['kickoff_time'].replace('Z', '+00:00')).date()
        if day >= now.date():
            same_day_excluded.append(row['id'])
            continue
        score = [row['team_h_score'], row['team_a_score']]
        if any(type(x) is not int or x < 0 for x in score):
            raise ValueError('Invalid official completed score')
        home, away = teams[row['team_h']], teams[row['team_a']]
        key = (day.isoformat(), ALIASES[home], ALIASES[away])
        if key in fresh:
            raise ValueError('Duplicate source game')
        fresh[key] = {'match_date': day.isoformat(), 'home_label': home,
                      'away_label': away, 'full_time': score,
                      'source_url': source['source_url'], 'observed_at': source['received_at'],
                      'official_fixture_id': row['id'], 'source_status': 'finished_true'}
    disagreement = [key for key, score in old.items()
                    if key not in fresh or tuple(fresh[key]['full_time']) != score]
    if disagreement:
        raise ValueError(f'Existing results disagree with fresh source: {disagreement}')
    confirmed_added = []
    if args.confirmed_case is not None:
        confirmed_path = args.confirmed_case.resolve()
        if confirmed_path.drive.upper() != 'G:':
            raise ValueError('Keep confirmed observations on G')
        confirmed = read(confirmed_path)
        source_fixtures = {r['match_id']: r for r in read(ROOT / 'data/prematch-intelligence/20261024_batch/fixtures.json')['fixtures']}
        for ident, result in confirmed['results'].items():
            fixture = source_fixtures[ident]
            day = datetime.fromisoformat(fixture['kickoff'].replace('Z', '+00:00')).date()
            if day >= now.date():
                continue
            if (result['completion_basis'] != 'official_completed_match_report'
                    or result['source_kind'] not in {'official_home_club_match_report', 'official_away_club_match_report'}
                    or datetime.fromisoformat(result['available_at']) > now):
                raise ValueError('Require already-observed official completed report')
            home, away = fixture['home_source_label'], fixture['away_source_label']
            key = (day.isoformat(), ALIASES[home], ALIASES[away])
            score = [result['home_goals'], result['away_goals']]
            if any(type(x) is not int or x < 0 for x in score):
                raise ValueError('Invalid confirmed score')
            if key in fresh and fresh[key]['full_time'] != score:
                raise ValueError('Official sources disagree on score')
            if key not in fresh:
                fresh[key] = {'match_date': day.isoformat(), 'home_label': home,
                              'away_label': away, 'full_time': score,
                              'source_url': result['source_url'], 'observed_at': result['available_at'],
                              'source_status': result['source_kind'], 'confirmed_case': str(confirmed_path),
                              'provider_status_preserved': result.get('corroborating_provider')}
                confirmed_added.append(ident)
    additions = [value for key, value in fresh.items() if key not in old]
    target = ROOT / 'data/prospective_context' / now.strftime('%Y%m%dT%H%M%S%fZ_official_refresh')
    target.mkdir(parents=True, exist_ok=False)
    copyfile(OLD / 'en.1.json', target / 'en.1.json')
    write(target / 'inspection.json', inspection)
    supplements['records'] += additions
    supplements['description'] = 'Original observed supplements plus newly received official FPL completed results, all strictly earlier UTC dates.'
    supplements['refresh_snapshot'] = str(snapshot)
    supplements['refresh_recorded_at'] = now.isoformat()
    write(target / 'supplements.json', supplements)
    checked, _, _ = load_current_context(target, now)
    audit = {'recorded_at': now.isoformat(), 'previous_context': str(OLD),
             'snapshot': str(snapshot), 'old_current_matches': len(current),
             'crosschecked_matches': len(old), 'score_disagreements': [],
             'new_results': additions, 'current_matches': len(checked),
             'club_report_additions': confirmed_added,
             'same_day_completed_excluded': same_day_excluded,
             'rule': 'Only earlier UTC calendar dates; no same-day match-time precision manufactured.',
             'old_predictions_modified': False, 'parameter_searches': 0}
    write(target / '更新与交叉核对.json', audit)
    print(json.dumps({'context_directory': str(target), 'current_matches': len(checked),
                      'added_results': len(additions)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
