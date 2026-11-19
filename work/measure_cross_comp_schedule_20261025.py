"""Measure two historical ESPN all-competition schedules; no network or model."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

from football_model.native_experiment import load_seasons
from football_model.xg_history import COMPETITION_TEAM_NAMES

ROOT = REPOSITORY_ROOT
OUT = ROOT/'outputs/全赛事赛程可用性_20261025'
FD = ROOT/'data/football-data/2023-24/20261022T113804786593Z'
EXTRA_ALIASES = {
    'AFC Bournemouth': 'Bournemouth', 'Brighton & Hove Albion': 'Brighton',
    'Luton Town': 'Luton', 'Sheffield United': 'Sheffield United',
    'Tottenham Hotspur': 'Tottenham', 'West Ham United': 'West Ham',
    '1. FC Heidenheim 1846': 'Heidenheim', '1. FC Union Berlin': 'Union Berlin',
    'Bayern Munich': 'Bayern Munich', 'Borussia Mönchengladbach': "M'gladbach",
    'FC Augsburg': 'Augsburg', 'SC Freiburg': 'Freiburg', 'SV Darmstadt 98': 'Darmstadt',
    'TSG Hoffenheim': 'Hoffenheim', 'VfL Bochum': 'Bochum', 'VfL Wolfsburg': 'Wolfsburg',
}
CLUBS = (
    ('arsenal_2023_all.json', '359', 'Arsenal', 'eng.1', 'E0',
     {'eng.1', 'eng.fa', 'eng.league_cup', 'eng.charity', 'uefa.champions'}),
    ('leverkusen_2023_all.json', '131', 'Bayer Leverkusen', 'ger.1', 'D1',
     {'ger.1', 'ger.dfb_pokal', 'uefa.europa'}),
)


def main():
    records, _ = load_seasons({'2023/24': FD})
    fixtures = {(r.competition, r.match_date.isoformat(), r.home_label, r.away_label): r for r in records}
    clubs, issues = {}, []
    for filename, team_id, team_name, league_slug, comp, competitive_slugs in CLUBS:
        payload = json.loads((OUT/filename).read_text(encoding='utf-8'))
        if str(payload['team']['id']) != team_id or payload['team']['displayName'] != team_name:
            raise ValueError('Source identity disagrees with observed provider ID')
        observations, event_ids, league_checks = [], set(), []
        for event in sorted(payload['events'], key=lambda e: (e['date'], e['id'])):
            when = datetime.fromisoformat(event['date'].replace('Z', '+00:00'))
            if not datetime(2023,7,1,tzinfo=timezone.utc) <= when < datetime(2024,7,1,tzinfo=timezone.utc):
                raise ValueError('Returned event falls outside historical requested window')
            if event['id'] in event_ids:
                raise ValueError('Duplicate historical event ID')
            event_ids.add(event['id'])
            if len(event['competitions']) != 1:
                raise ValueError('Need one explicit event competition')
            contest = event['competitions'][0]
            sides = {side['homeAway']: side for side in contest['competitors']}
            if set(sides) != {'home', 'away'}:
                raise ValueError('Need unique named home and away competitors')
            if sum(str(side['team']['id']) == team_id for side in sides.values()) != 1:
                raise ValueError('Requested team not uniquely present in event')
            slug = event['league']['slug']
            row = {
                'event_id': event['id'], 'date_utc_text': event['date'], 'competition_slug': slug,
                'competition_name': event['league']['name'], 'event_season': event['season'],
                'source_home': sides['home']['team']['displayName'],
                'source_away': sides['away']['team']['displayName'],
                'home_team_id': str(sides['home']['team']['id']),
                'away_team_id': str(sides['away']['team']['id']),
                'home_away_source': 'explicit competitor.homeAway; not array index',
                'status': contest.get('status'), 'time_valid': event.get('timeValid'),
                'neutral_site_untrusted': contest.get('neutralSite'), 'venue': contest.get('venue'),
                'category': 'competitive' if slug in competitive_slugs else 'friendly_or_exhibition',
                'duration_minutes': None,
                'duration_note': 'period=5/shootout status and final clock do not establish whether extra time was played',
            }
            observations.append(row)
            if slug == league_slug:
                def label(side):
                    name = side['team']['displayName']
                    return EXTRA_ALIASES.get(name, COMPETITION_TEAM_NAMES[comp].get(name, name))
                key = (comp, when.date().isoformat(), label(sides['home']), label(sides['away']))
                match = fixtures.get(key)
                if match is None:
                    issues.append({'event': event['id'], 'reason': 'no_exact_independent_league_fixture', 'key': key})
                    continue
                # League-only exact-score check, not a cup90-minute interpretation.
                score = tuple(int(sides[side]['score']['value']) for side in ('home', 'away'))
                if score != match.full_time:
                    issues.append({'event': event['id'], 'reason': 'independent_score_disagreement'})
                league_checks.append({'event_id': event['id'], 'match_id': match.match_id,
                                      'exact_date_home_away_score': score == match.full_time})
        counts = Counter(r['competition_slug'] for r in observations)
        gap_comparisons = []
        for row in observations:
            if row['competition_slug'] != league_slug:
                continue
            earlier = [r for r in observations if r['date_utc_text'] < row['date_utc_text'] and r['category'] == 'competitive']
            earlier_league = [r for r in earlier if r['competition_slug'] == league_slug]
            if not earlier or not earlier_league:
                continue
            last_all, last_league = earlier[-1], earlier_league[-1]
            played = datetime.fromisoformat(row['date_utc_text'].replace('Z', '+00:00'))
            def hours(previous):
                return (played-datetime.fromisoformat(previous['date_utc_text'].replace('Z', '+00:00'))).total_seconds()/3600
            gap_comparisons.append({'league_event_id': row['event_id'],
                'previous_competitive_event_id': last_all['event_id'], 'previous_competition': last_all['competition_slug'],
                'previous_league_event_id': last_league['event_id'],
                'kickoff_to_kickoff_all_competition_hours': hours(last_all),
                'kickoff_to_kickoff_league_only_hours': hours(last_league),
                'prior_match_differs': last_all['event_id'] != last_league['event_id'],
                'interpretation': 'descriptive historical start-to-start interval, not final-whistle rest or forecast publication reconstruction'})
        club = {
            'provider_team_id': team_id, 'observed_name': team_name, 'file': filename,
            'source_url': f'https://site.api.espn.com/apis/site/v2/sports/soccer/all/teams/{team_id}/schedule?season=2023&dates=20230701-20240630',
            'events': len(observations), 'competition_counts': dict(counts),
            'competitive_matches': sum(r['category'] == 'competitive' for r in observations),
            'friendly_exhibition_matches': sum(r['category'] != 'competitive' for r in observations),
            'league_exact_checks': league_checks,
            'league_prior_event_comparison': gap_comparisons,
            'league_cases_with_nonleague_previous_match': sum(r['prior_match_differs'] for r in gap_comparisons),
            'returned_top_level_season_not_used': payload.get('season'),
            'requested_season': payload.get('requestedSeason'),
            'source_payload_current_fields_ignored': ['season', 'team.recordSummary', 'team.seasonSummary', 'team.standingSummary', 'team.groups'],
            'observations': observations,
        }
        clubs[team_name] = club
    result = {
        'measured_at': datetime.now(timezone.utc).isoformat(), 'historical_scope': '2023/24 only, two observed clubs',
        'clubs': clubs, 'issues': issues,
        'assertion_scope': 'Two-club historical availability measurement, not a certified complete five-league all-season import',
        'network_requests_in_measurement': 0, 'prior_direct_historical_requests': 2,
        'lineups_read': False, 'models_trained': False, 'current_match_results_requested': False,
        'limitations': ['Public website backend endpoint has no checked supported API/SLA or commercial data licence',
            'Completed historical event kickoff UTC text is available; original schedule publication and revisions unknown',
            'Historical next-event pressure cannot use final schedule as an announcement-time reconstruction',
            'Source neutralSite false even for finals at neutral venues; do not trust by itself',
            'No regular-time/extra-time goal split or duration established by schedule endpoint alone',
            'All national cup, qualifier, supercup, postponement and abandoned-match coverage beyond these clubs unmeasured'],
    }
    (OUT/'两队全赛事实测.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'clubs': {n: {'events': c['events'], 'competitive': c['competitive_matches'],
        'friendly_exhibition': c['friendly_exhibition_matches'], 'competitions': c['competition_counts'],
        'exact_league_checks': len(c['league_exact_checks'])} for n,c in clubs.items()}, 'issues': issues}, ensure_ascii=False))


if __name__ == '__main__':
    main()
