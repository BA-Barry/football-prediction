"""Measure historical five-league memberships and recorded cross-competition load."""
from collections import Counter
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

from football_model.native_experiment import load_seasons, season_directories
from football_model.xg_history import COMPETITION_TEAM_NAMES
from measure_cross_comp_schedule_20261025 import EXTRA_ALIASES

ROOT = REPOSITORY_ROOT
YEAR = 2024
OUT = ROOT/'outputs/全赛事赛程可用性_20261025/2024-25全量'
FD = ROOT/'outputs/Football-Data_2024-25_个人使用_20261022T111235Z'
SLUGS = {'E0': 'eng.1', 'D1': 'ger.1', 'SP1': 'esp.1', 'I1': 'ita.1', 'F1': 'fra.1'}
ALIASES = {**EXTRA_ALIASES, 'Ipswich Town': 'Ipswich', 'Leicester City': 'Leicester',
    'Atlético Madrid': 'Ath Madrid', 'Alavés': 'Alaves', 'Leganés': 'Leganes',
    'AS Roma': 'Roma', 'Hellas Verona': 'Verona', 'Internazionale': 'Inter',
    'Paris Saint-Germain': 'Paris SG', 'Stade Rennais': 'Rennes', 'AJ Auxerre': 'Auxerre',
    'AS Monaco': 'Monaco', 'Saint-Étienne': 'St Etienne', 'Le Havre AC': 'Le Havre',
    'Stade de Reims': 'Reims'}
ALIASES.update({'Cádiz': 'Cadiz', 'Almería': 'Almeria', 'AC Ajaccio': 'Ajaccio', 'Hamburg SV': 'Hamburg'})
ALIASES['Leeds United'] = 'Leeds'
FRIENDLY_SLUGS = {'club.friendly', 'friendly.emirates_cup', 'esp.joan_gamper', 'jpn.world_challenge'}
COMPETITIVE_SLUGS = set(SLUGS.values()) | {'eng.fa', 'eng.league_cup', 'eng.charity',
    'ger.dfb_pokal', 'ger.super_cup', 'ger.playoff.relegation', 'esp.copa_del_rey',
    'esp.super_cup', 'ita.coppa_italia', 'ita.super_cup', 'fra.coupe_de_france',
    'fra.super_cup', 'fra.1.promotion.relegation', 'uefa.champions', 'uefa.europa',
    'uefa.europa.conf', 'uefa.europa.conf_qual', 'uefa.champions_qual', 'uefa.super_cup',
    'fifa.intercontinental_cup', 'fifa.cwc', 'uefa.europa_qual'}
START, END = datetime(2024,7,1,tzinfo=timezone.utc), datetime(2025,7,1,tzinfo=timezone.utc)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def when(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def normalize(name, comp):
    return ALIASES.get(name, COMPETITION_TEAM_NAMES[comp].get(name, name))


def event_signature(event):
    contest = event['competitions'][0]
    sides = {t['homeAway']: t for t in contest['competitors']}
    scores = []
    for side in ('home', 'away'):
        score = sides[side].get('score')
        scores.append(score.get('value') if isinstance(score, dict) else score)
    return {'date': event['date'], 'slug': event['league']['slug'],
            'teams': [str(sides[s]['team']['id']) for s in ('home', 'away')],
            'scores': scores, 'completed': contest.get('status', {}).get('type', {}).get('completed')}


def main():
    global YEAR, OUT, FD, START, END
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, choices=(2022,2023,2024,2025), default=2024)
    parser.add_argument('--union-neighbors', action='store_true')
    args = parser.parse_args()
    YEAR = args.year
    prefix = '相邻赛季补全_' if args.union_neighbors else ''
    season = f'{YEAR}/{str(YEAR+1)[-2:]}'
    OUT = ROOT/f'outputs/全赛事赛程可用性_20261025/{YEAR}-{str(YEAR+1)[-2:]}全量'
    FD = season_directories(ROOT)[season]
    START, END = datetime(YEAR,7,1,tzinfo=timezone.utc), datetime(YEAR+1,7,1,tzinfo=timezone.utc)
    records, reports = load_seasons({season: FD})
    fixtures = {(r.competition, r.match_date.isoformat(), r.home_label, r.away_label): r for r in records}
    expected_teams = {comp: {name for r in records if r.competition == comp for name in (r.home_label, r.away_label)} for comp in SLUGS}
    members = read(OUT/'历史参赛身份目录.json')['clubs']
    clubs, unique, duplicate_conflicts, outside, issues = {}, {}, [], [], []
    supplemental = []
    for daily in sorted((OUT/'raw').glob('daily_*.json')):
        body = read(daily)
        for event in body.get('events', []):
            supplemental.append({**event, 'league': body['leagues'][0], '_supplement_source': str(daily)})
    for member in members:
        tid, comp = member['team_id'], member['competition']
        payload = read(OUT/'raw'/f'team_{tid}.json')
        name = payload['team']['displayName']
        if str(payload['team']['id']) != tid:
            raise ValueError('Catalog and response team ID disagree')
        label = normalize(name, comp)
        source_payloads = [(YEAR, str(OUT/'raw'/f'team_{tid}.json'), payload)]
        if args.union_neighbors and YEAR < 2025:
            neighbor = ROOT/f'outputs/全赛事赛程可用性_20261025/{YEAR+1}-{str(YEAR+2)[-2:]}全量/raw/team_{tid}.json'
            if neighbor.exists():
                source_payloads.append((YEAR+1, str(neighbor), read(neighbor)))
        selected = {}
        for source_year, source_path, body in source_payloads:
            for event in body['events']:
                if event['id'] not in selected:
                    selected[event['id']] = {'event': event, 'source_years': [source_year], 'source_files': [source_path]}
                else:
                    previous = selected[event['id']]['event']
                    if event_signature(previous) != event_signature(event):
                        duplicate_conflicts.append({'event_id': event['id'], 'reason': 'adjacent_source_payload_difference',
                            'first': event_signature(previous), 'second': event_signature(event)})
                    selected[event['id']]['source_years'].append(source_year)
                    selected[event['id']]['source_files'].append(source_path)
        for event in supplemental:
            if any(str(s['team']['id']) == tid for s in event['competitions'][0]['competitors']):
                if event['id'] not in selected:
                    selected[event['id']] = {'event': event, 'source_years': [YEAR], 'source_files': [event['_supplement_source']]}
        local = []
        for selection in selected.values():
            event = selection['event']
            played = when(event['date'])
            if not START <= played < END:
                outside.append({'team_id': tid, 'event_id': event['id'], 'date': event['date'],
                    'league': event['league']['slug'], 'reason': 'outside_declared_calendar_window'})
                continue
            c = event['competitions'][0]
            sides = {t['homeAway']: t for t in c['competitors']}
            if set(sides) != {'home', 'away'} or len(c['competitors']) != 2:
                raise ValueError('Need explicit unique homeAway')
            if sum(str(s['team']['id']) == tid for s in sides.values()) != 1:
                raise ValueError('Requested club absent from event')
            slug = event['league']['slug']
            category = 'competitive' if slug in COMPETITIVE_SLUGS else 'friendly_exhibition' if slug in FRIENDLY_SLUGS or event['id'] in {'660069', '684033'} else 'unclassified'
            if category == 'unclassified':
                issues.append({'event_id': event['id'], 'reason': 'unclassified_competition', 'slug': slug})
            status = c.get('status', {})
            score = []
            for side in ('home', 'away'):
                raw = sides[side].get('score')
                score.append(raw.get('value') if isinstance(raw, dict) else raw)
            row = {'event_id': str(event['id']), 'date_utc_text': event['date'],
                'competition_slug': slug, 'category': category,
                'home_team_id': str(sides['home']['team']['id']), 'away_team_id': str(sides['away']['team']['id']),
                'home_name': sides['home']['team']['displayName'], 'away_name': sides['away']['team']['displayName'],
                'source_final_score': score, 'completed': status.get('type', {}).get('completed') is True,
                'status_name': status.get('type', {}).get('name'), 'time_valid': event.get('timeValid'),
                'source_clock': status.get('clock'), 'source_period': status.get('period'),
                'source_season_type': event.get('seasonType'),
                'is_regular_league_fixture': slug in SLUGS.values() and event.get('seasonType', {}).get('name') != 'Relegation Playoff' and event['id'] != '660069',
                'source_years': selection['source_years'], 'source_files': selection['source_files'],
                'neutral_site_untrusted': c.get('neutralSite'), 'venue': c.get('venue'),
                'regular_time_score': None, 'extra_time_played': None, 'duration_minutes': None}
            if row['event_id'] in unique:
                original = unique[row['event_id']]
                fields = ('date_utc_text', 'competition_slug', 'home_team_id', 'away_team_id', 'source_final_score', 'completed', 'status_name')
                differences = {f: [original[f], row[f]] for f in fields if original[f] != row[f]}
                if differences:
                    duplicate_conflicts.append({'event_id': row['event_id'], 'differences': differences})
                original['source_files'] = sorted(set(original['source_files']+row['source_files']))
                original['source_years'] = sorted(set(original['source_years']+row['source_years']))
            else:
                unique[row['event_id']] = row
            local.append(row)
        clubs[tid] = {**member, 'source_name': name, 'independent_label': label,
            'raw_event_count': sum(len(body['events']) for _,_,body in source_payloads), 'in_window_count': len(local),
            'competition_counts': dict(Counter(r['competition_slug'] for r in local)),
            'formal_completed_count': sum(r['category'] == 'competitive' and r['completed'] for r in local),
            'events': sorted(local, key=lambda r: (r['date_utc_text'], r['event_id']))}
    membership = {}
    for comp in SLUGS:
        labels = [c['independent_label'] for c in clubs.values() if c['competition'] == comp]
        membership[comp] = {'source_count': len(labels), 'expected_count': len(expected_teams[comp]),
            'missing_independent_labels': sorted(expected_teams[comp]-set(labels)),
            'unexpected_source_labels': sorted(set(labels)-expected_teams[comp]),
            'duplicate_normalized_labels': len(labels) != len(set(labels))}
    league_checks, known_special, matched_ids, incomplete_status = [], [], set(), []
    for row in unique.values():
        if not row['is_regular_league_fixture']:
            continue
        comp = next(k for k,v in SLUGS.items() if v == row['competition_slug'])
        key = (comp, when(row['date_utc_text']).date().isoformat(), normalize(row['home_name'], comp), normalize(row['away_name'], comp))
        match = fixtures.get(key)
        if match is None:
            issues.append({'event_id': row['event_id'], 'reason': 'no_exact_FD_fixture', 'key': key})
            continue
        matched_ids.add(match.match_id)
        equal = tuple(map(float, row['source_final_score'])) == tuple(match.full_time)
        if not row['completed']:
            incomplete_status.append({'event_id': row['event_id'], 'match_id': match.match_id,
                'date': match.match_date.isoformat(), 'status_name': row['status_name'],
                'completed': False, 'source_clock': row['source_clock'],
                'source_period': row['source_period'], 'source_score': row['source_final_score'],
                'FD_full_time': list(match.full_time), 'score_equal': equal,
                'target_use': 'not verified as completed 90-minute match; do not infer full-time target from equal scores',
                'source_files': row['source_files']})
        if match.goal_fit_exclusion is not None:
            known_special.append({'event_id': row['event_id'], 'match_id': match.match_id,
                'reason': match.goal_fit_exclusion, 'score_equal': equal, 'source_score': row['source_final_score'],
                'FD_full_time': list(match.full_time), 'target_use': 'preserve existing goal-fit exclusion'})
        elif not equal:
            issues.append({'event_id': row['event_id'], 'reason': 'FD_full_score_disagreement',
                'source_score': row['source_final_score'], 'FD_full_time': list(match.full_time)})
        league_checks.append({'event_id': row['event_id'], 'match_id': match.match_id,
            'competition': comp, 'date': match.match_date.isoformat(), 'goal_fit_exclusion': match.goal_fit_exclusion,
            'exact_date_home_away': True, 'score_equal': equal, 'completed': row['completed'],
            'completed_score_verified': row['completed'] and equal,
            'ordinary_goal_target_verified': row['completed'] and equal and match.goal_fit_exclusion is None})
    missing_fd = [r.match_id for r in records if r.competition in SLUGS and r.match_id not in matched_ids]
    load_rows = []
    for club in clubs.values():
        local = club['events']
        for row in local:
            if row['competition_slug'] != SLUGS[club['competition']] or not row['is_regular_league_fixture'] or not row['completed']:
                continue
            target = when(row['date_utc_text'])
            earlier = [r for r in local if r['completed'] and r['category'] == 'competitive' and when(r['date_utc_text']) < target]
            league_earlier = [r for r in earlier if r['competition_slug'] == SLUGS[club['competition']] and r['is_regular_league_fixture']]
            last_all, last_league = earlier[-1] if earlier else None, league_earlier[-1] if league_earlier else None
            gap_all = (target-when(last_all['date_utc_text'])).total_seconds()/3600 if last_all else None
            gap_league = (target-when(last_league['date_utc_text'])).total_seconds()/3600 if last_league else None
            load_rows.append({'target_event_id': row['event_id'], 'team_id': club['team_id'],
                'team': club['independent_label'], 'competition': club['competition'], 'date': row['date_utc_text'],
                'previous_all_event_id': last_all['event_id'] if last_all else None,
                'previous_league_event_id': last_league['event_id'] if last_league else None,
                'previous_competition': last_all['competition_slug'] if last_all else None,
                'all_competition_kickoff_gap_hours': gap_all, 'league_only_kickoff_gap_hours': gap_league,
                'previous_differs': last_all['event_id'] != last_league['event_id'] if last_all and last_league else None,
                'prior_formal_events_7d': sum(0 < (target-when(r['date_utc_text'])).total_seconds() <= 7*86400 for r in earlier),
                'prior_formal_events_14d': sum(0 < (target-when(r['date_utc_text'])).total_seconds() <= 14*86400 for r in earlier),
                'window_start_censoring_possible': (target-START).total_seconds() <= 14*86400,
                'actual_minutes': None, 'next_match_pressure': None})
    comparison = [r for r in load_rows if r['previous_differs'] is not None]
    summary = {'measured_at': datetime.now(timezone.utc).isoformat(), 'scope': f'{season} five-league historical members, recorded events July{YEAR}-June{YEAR+1}',
        'adjacent_source_union': args.union_neighbors, 'daily_supplement_events': len(supplemental),
        'clubs': len(clubs), 'raw_club_event_rows': sum(c['raw_event_count'] for c in clubs.values()),
        'in_window_club_event_rows': sum(c['in_window_count'] for c in clubs.values()), 'unique_events': len(unique),
        'unique_competitive_completed_events': sum(r['category'] == 'competitive' and r['completed'] for r in unique.values()),
        'unique_friendly_exhibition_events': sum(r['category'] == 'friendly_exhibition' for r in unique.values()),
        'unique_competition_counts': dict(Counter(r['competition_slug'] for r in unique.values())),
        'membership': membership, 'source_duplicate_conflicts': duplicate_conflicts,
        'league_exact_fixture_checks': len(league_checks), 'league_expected_FD_matches': len(records),
        'league_completed_score_verified': sum(r['completed_score_verified'] for r in league_checks),
        'league_ordinary_goal_target_verified': sum(r['ordinary_goal_target_verified'] for r in league_checks),
        'incomplete_source_status_conflicts': incomplete_status,
        'missing_FD_matches': missing_fd, 'known_specials': known_special,
        'ordinary_score_disagreements': sum(not r['score_equal'] and r['goal_fit_exclusion'] is None for r in league_checks),
        'outside_window_event_rows': outside, 'issues': issues,
        'load_comparison': {'team_match_observations': len(load_rows), 'with_previous_league_in_window': len(comparison),
            'previous_formal_match_is_nonleague': sum(r['previous_differs'] for r in comparison),
            'all_kickoff_gap_le_72h': sum(r['all_competition_kickoff_gap_hours'] <= 72 for r in comparison),
            'league_only_kickoff_gap_le_72h': sum(r['league_only_kickoff_gap_hours'] <= 72 for r in comparison)},
        'earliest_event': min(r['date_utc_text'] for r in unique.values()), 'latest_event': max(r['date_utc_text'] for r in unique.values()),
        'limitations': ['League completeness independently measured; nonleague completeness not certified',
            f'Historical schedule endpoints can include calendar{YEAR} friendlies beforeJuly and omit calendar{YEAR+1} friendlies',
            'Calendar-year tournaments require separate coverage: June2025 Club World Cup not present in season2024 response',
            'No original announcement/rescheduling reconstruction; no future pressure feature',
            'Start-to-start gaps are not final-whistle rest; no verified120-minute duration or90-minute cup score',
            'Current top-level season/team records ignored; only historical restricted event rows used'],
        'models_trained': False, 'current_results_requested': False, 'lineups_read': False}
    write(OUT/(prefix+'全量覆盖与负荷测量.json'), summary)
    write(OUT/(prefix+'去重历史赛事.json'), {'events': sorted(unique.values(), key=lambda r:(r['date_utc_text'], r['event_id'])), 'duplicate_conflicts': duplicate_conflicts})
    write(OUT/(prefix+'逐队历史观察.json'), {'clubs': clubs, 'league_checks': league_checks})
    write(OUT/(prefix+'联赛逐队负荷观察.json'), {'scope': summary['scope'], 'rows': load_rows, 'features_are_not_yet_used_by_models': True})
    print(json.dumps({k: summary[k] for k in ['clubs','raw_club_event_rows','in_window_club_event_rows','unique_events','unique_competitive_completed_events','unique_competition_counts','membership','league_exact_fixture_checks','league_expected_FD_matches','missing_FD_matches','known_specials','ordinary_score_disagreements','load_comparison','earliest_event','latest_event','issues']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
