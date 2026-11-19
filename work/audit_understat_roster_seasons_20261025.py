"""Measure saved EPL roster identity coverage; perform no requests or model fits."""
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import argparse
import html
import json
from pathlib import Path
import re
import unicodedata

from football_model.understat_rosters import (
    STARTING_POSITIONS, UnderstatRosterSourceError, extract_starting_xi,
)
from football_model.xg_history import COMPETITION_TEAM_NAMES

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
SOURCE = ROOT/'data/understat/roster_seasons_EPL_2023_2024_20261025'
OUT = ROOT/'outputs/英超跨季首发覆盖_20261025'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def normalized_name(value):
    return unicodedata.normalize('NFC', html.unescape(value)).strip()


def canonical_team(value, competition):
    value = normalized_name(value)
    return f'football-data:{competition}:' + COMPETITION_TEAM_NAMES[competition].get(value, value)


def new_pool():
    return {'appearances': defaultdict(set), 'starters': set(), 'teams': set(), 'matches': 0}


def add_history(pool, observation):
    pool['matches'] += 1
    for side in ('home', 'away'):
        team = observation[side]
        pool['teams'].add(team)
        for player in observation[side+'_appeared']:
            pool['appearances'][player].add(team)
        pool['starters'].update(observation[side+'_starters'])


def side_coverage(players, team, pool, starters_only):
    known = pool['starters'] if starters_only else pool['appearances']
    unknown = [p for p in players if p not in known]
    first = [p for p in players if p not in pool['appearances']]
    elsewhere = [p for p in players if p in pool['appearances'] and team not in pool['appearances'][p]]
    prior_sub_only = [p for p in players if p in pool['appearances'] and p not in pool['starters']]
    return {'team': team, 'team_unknown': team not in pool['teams'], 'starter_seats': len(players),
        'unknown_player_ids': unknown, 'unknown_count': len(unknown),
        'first_observed_player_ids': first, 'first_observed_count': len(first),
        'known_elsewhere_not_previously_observed_at_target_team_ids': elsewhere,
        'known_elsewhere_count': len(elsewhere),
        'previously_appeared_but_never_started_ids': prior_sub_only,
        'previously_appeared_but_never_started_count': len(prior_sub_only)}


def measure_reference(observations, targets, mode, starters_only):
    """A target may use only the declared fixed season or dates strictly before its cutoff."""
    earlier = sorted(observations, key=lambda r: (r['match_date'], r['match_id']))
    indexed = {r['match_id']: r for r in observations}
    pool, pointer, rows, counts = new_pool(), 0, [], Counter()
    unknown_ids, first_ids, elsewhere_ids = set(), set(), set()
    if mode == 'complete_2023':
        for row in earlier:
            if row['season'] == '2023/24' and row['match_date'] < '2024-08-01':
                add_history(pool, row)
    for fixture in sorted(targets, key=lambda r: (r['date'], r['match_id'])):
        played = date.fromisoformat(fixture['date'])
        cutoff = date(2024, 8, 1) if mode == 'complete_2023' else played
        if mode == '28_day':
            anchor = date(2024, 8, 1)
            cutoff = anchor + timedelta(days=((played-anchor).days//28)*28)
        if mode != 'complete_2023':
            while pointer < len(earlier) and earlier[pointer]['match_date'] < cutoff.isoformat():
                add_history(pool, earlier[pointer])
                pointer += 1
        item = {'match_id': fixture['match_id'], 'date': fixture['date'], 'cutoff': cutoff.isoformat(),
                'reference_valid_matches': pool['matches'], 'reference_appearance_players': len(pool['appearances']),
                'reference_starting_players': len(pool['starters'])}
        target = indexed.get(fixture['match_id'])
        if target is None:
            item['status'] = 'target_unusable'; counts['target_unusable_matches'] += 1
        else:
            item['status'] = 'measured'; counts['measured_matches'] += 1
            for side in ('home', 'away'):
                result = side_coverage(target[side+'_starters'], target[side], pool, starters_only)
                item[side] = result
                counts['measured_team_sides'] += 1
                counts['unknown_team_sides'] += result['team_unknown']
                for field in ('starter_seats', 'unknown_count', 'first_observed_count', 'known_elsewhere_count',
                              'previously_appeared_but_never_started_count'):
                    counts[field] += result[field]
                unknown_ids.update(result['unknown_player_ids'])
                first_ids.update(result['first_observed_player_ids'])
                elsewhere_ids.update(result['known_elsewhere_not_previously_observed_at_target_team_ids'])
            all_players_known = item['home']['unknown_count'] + item['away']['unknown_count'] == 0
            both_teams_known = not item['home']['team_unknown'] and not item['away']['team_unknown']
            counts['matches_all_22_starters_known'] += all_players_known
            counts['matches_with_at_least_one_unknown_starter'] += not all_players_known
            counts['matches_both_teams_known'] += both_teams_known
            counts['matches_all_22_starters_and_both_teams_known'] += all_players_known and both_teams_known
        rows.append(item)
    seats = counts['starter_seats']
    summary = {'target_matches': len(targets), 'nominal_target_starter_seats': 22*len(targets),
        **dict(counts), 'unknown_rate_on_measured_seats': counts['unknown_count']/seats if seats else None,
        'distinct_unknown_player_ids': len(unknown_ids), 'distinct_first_observed_player_ids': len(first_ids),
        'distinct_known_elsewhere_player_ids': len(elsewhere_ids),
        'unmeasured_nominal_seats': 22*len(targets)-seats,
        'reference_uses': 'previously started IDs' if starters_only else 'previously appeared IDs',
        'history_rule': mode,
        'same_date_matches_excluded': mode != 'complete_2023',
        'classification_note': 'Team unknown and player-history categories are separate; known elsewhere is observed association, not verified transfer.',
        'effect_for_unknown_players': 'not estimated; never defaulted to zero'}
    return {'summary': summary, 'matches': rows}


def self_test():
    def obs(mid, day, home, starters, appeared):
        return {'match_id': mid, 'match_date': day, 'season': '2024/25', 'home': home, 'away': 'other',
                'home_starters': starters, 'home_appeared': appeared,
                'away_starters': ['opponent'], 'away_appeared': ['opponent']}
    history = [obs('a', '2024-08-02', 'old', ['p1'], ['p1', 'p2']),
               obs('b', '2024-08-03', 'new', ['p2'], ['p2']),
               obs('c', '2024-08-03', 'third', ['p2'], ['p2']),
               obs('d', '2024-08-29', 'new', ['p2'], ['p2'])]
    targets = [{'match_id': r['match_id'], 'date': r['match_date']} for r in history[1:]]
    daily = measure_reference(history, targets, 'daily', True)['matches']
    assert daily[0]['home']['unknown_player_ids'] == ['p2']
    assert daily[1]['home']['unknown_player_ids'] == ['p2']  # same-date b cannot leak into c
    assert daily[0]['home']['first_observed_count'] == 0
    assert daily[0]['home']['known_elsewhere_count'] == 1
    assert daily[0]['home']['previously_appeared_but_never_started_count'] == 1
    assert daily[0]['home']['team_unknown']
    assert daily[2]['home']['unknown_count'] == 0
    blocks = measure_reference(history, targets, '28_day', True)['matches']
    assert blocks[0]['cutoff'] == '2024-08-01' and blocks[0]['reference_valid_matches'] == 0
    assert blocks[2]['cutoff'] == '2024-08-29' and blocks[2]['reference_valid_matches'] == 3
    assert normalized_name('A&#039;B') == normalized_name("A'B")
    print('Chronology/identity function checks passed: same-date exclusion, prior substitute, new team, 28-day anchor, HTML names.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    output_directory = args.output.resolve()
    if output_directory.drive.casefold() != 'g:':
        raise ValueError('All audit outputs must remain on G:')
    progress = read(SOURCE/'进度.json')
    if progress.get('status') != 'completed':
        raise RuntimeError('Saved downloader is not completed; this script never downloads or retries.')
    selection = read(SOURCE/'比赛选择.json')
    planned = selection['selected']
    statuses, observations, name_rows = [], [], defaultdict(lambda: defaultdict(list))
    team_source_ids, coverage = defaultdict(set), defaultdict(Counter)
    totals, errors, reused, minute_values, shot_name_variants = Counter(), [], [], [], []
    for fixture in planned:
        uid = fixture['understat_id']; season = fixture['season']; comp = fixture['competition']
        state = {**fixture, 'download_status': 'missing', 'lineup_status': 'unusable',
                 'identity_status': 'not_checked', 'identity_errors': [], 'unknown_positions': []}
        statuses.append(state)
        for side in ('home', 'away'):
            coverage[(season, fixture[side])]['planned_matches'] += 1
        receipt_path, raw_path = SOURCE/'receipts'/f'{uid}.json', SOURCE/'raw'/f'{uid}.json'
        try:
            receipt = read(receipt_path)
        except (OSError, ValueError) as error:
            state['error'] = f'receipt: {error}'; continue
        state['receipt'] = {k: receipt[k] for k in ('status_code', 'requested_at', 'received_at', 'copied_at',
                               'reused_from', 'final_url', 'error') if k in receipt}
        if receipt.get('status_code') != 200 or receipt.get('error'):
            state['download_status'] = 'download_failed'; continue
        try:
            raw = read(raw_path)
        except (OSError, ValueError) as error:
            state['error'] = f'payload: {error}'; continue
        state['download_status'] = 'downloaded'
        totals['downloaded_matches'] += 1
        if receipt.get('reused_from'):
            reused.append({'understat_id': uid, **state['receipt']})
        try:
            binary = json.loads((SOURCE/'raw'/f'{uid}_response.bin').read_bytes())
            state['response_body_equals_json'] = binary == raw
            totals['response_body_json_equal'] += binary == raw
        except (OSError, ValueError, UnicodeError) as error:
            state['response_body_equals_json'] = False; state['response_body_error'] = str(error)
        if not state['response_body_equals_json']:
            state['identity_errors'].append('response_body_and_json_differ_or_missing')
        try:
            parsed = extract_starting_xi(raw, fixture)
        except UnderstatRosterSourceError as error:
            state['parser_error'] = str(error)
            if isinstance(raw, dict) and isinstance(raw.get('rosters'), dict):
                state['unknown_positions'] = [
                    {'side': side, 'roster_id': key, 'position': row.get('position')}
                    for side, roster in raw['rosters'].items() if isinstance(roster, dict)
                    for key, row in roster.items() if isinstance(row, dict)
                    and (not isinstance(row.get('position'), str)
                         or row.get('position') not in STARTING_POSITIONS | {'Sub'})]
            state['lineup_status'] = 'unknown_position' if state['unknown_positions'] else 'unusable'
            continue
        state['lineup_status'] = 'parsed_before_identity_check'
        rosters = raw['rosters']
        row_by_player = {str(p['player_id']): p for rows in rosters.values() for p in rows.values()}
        evidence, shot_count = [], 0
        for side in ('h', 'a'):
            shots = raw.get('shots', {}).get(side) if isinstance(raw.get('shots'), dict) else None
            if not isinstance(shots, list):
                state['identity_errors'].append(f'{side}:missing_shot_list'); continue
            for shot in shots:
                shot_count += 1
                try:
                    player = row_by_player.get(str(shot['player_id']))
                    checks = {'match_id': str(shot['match_id']) == uid,
                        'date': shot['date'][:10] == fixture['date'],
                        'home': canonical_team(shot['h_team'], comp) == fixture['home'],
                        'away': canonical_team(shot['a_team'], comp) == fixture['away'],
                        'player_side': player is not None and player['h_a'] == side == shot['h_a']}
                    if player is not None and normalized_name(player['player']) != normalized_name(shot['player']):
                        shot_name_variants.append({'understat_id': uid, 'shot_id': shot.get('id'),
                            'player_id': shot['player_id'], 'roster_name': player['player'], 'shot_name': shot['player'],
                            'handling': 'identity follows matching player_id; name text difference is diagnostic only'})
                    for key, passed in checks.items():
                        if not passed: state['identity_errors'].append(f"shot:{shot.get('id')}:{key}")
                    if all(checks.values()): evidence.append(str(shot['id']))
                    totals['own_goal_identity_shots_checked'] += shot.get('result') == 'OwnGoal'
                except (KeyError, TypeError, ValueError) as error:
                    state['identity_errors'].append(f'shot:malformed:{error}')
        state['shots_checked'] = shot_count; totals['shots_checked'] += shot_count
        if not shot_count:
            state['identity_status'] = 'insufficient_evidence'
            state['lineup_status'] = 'identity_unresolved'
            continue
        for side, name in (('h', 'home'), ('a', 'away')):
            template = raw.get('tmpl', {}).get(name, '') if isinstance(raw.get('tmpl'), dict) else ''
            found = re.search(r'<h3>\s*<a\b[^>]*>([^<]*)</a>', template)
            if found and canonical_team(found.group(1), comp) != fixture[name]:
                state['identity_errors'].append(f'tmpl:{side}:team')
        if state['identity_errors']:
            state['identity_status'] = 'conflict'; state['lineup_status'] = 'identity_anomaly'
            errors.append({'understat_id': uid, 'errors': state['identity_errors']}); continue
        state['identity_status'] = 'verified_by_all_available_shots'
        state['lineup_status'] = 'parsed'; totals['parsed_matches'] += 1
        observation = {**parsed, 'season': season,
            'identity_basis': 'independent_FD_xG_join_plus_all_shot_ids_dates_teams_players',
            'receipt': state['receipt']}
        for side, name in (('h', 'home'), ('a', 'away')):
            rows = rosters[side]
            observation[name+'_appeared'] = [f"understat:player:{p['player_id']}" for p in rows.values()]
            source_ids = {str(p['team_id']) for p in rows.values()}
            team_source_ids[fixture[name]].update(source_ids)
            observation[name+'_source_team_id'] = sorted(source_ids)
            coverage[(season, fixture[name])]['parsed_matches'] += 1
            coverage[(season, fixture[name])]['starter_seats'] += 11
            for row in rows.values():
                pid = f"understat:player:{row['player_id']}"
                name_rows[pid][row['player']].append({'match_id': fixture['match_id'], 'season': season,
                                                     'team': fixture[name]})
                try: minute_values.append(int(row['time']))
                except (KeyError, TypeError, ValueError): totals['invalid_minute_rows'] += 1
        observations.append(observation)
    targets = [r for r in planned if r['season'] == '2024/25']
    references = {}
    for mode in ('complete_2023', 'daily', '28_day'):
        for starter_only in (False, True):
            label = mode + ('_ever_started' if starter_only else '_ever_appeared')
            references[label] = measure_reference(observations, targets, mode, starter_only)
    variants = []
    for player, raw_names in sorted(name_rows.items()):
        decoded = sorted({html.unescape(n) for n in raw_names})
        normalized = sorted({normalized_name(n) for n in raw_names})
        if len(raw_names) > 1 or any(html.unescape(n) != n for n in raw_names):
            variants.append({'player_id': player, 'raw_names': list(raw_names), 'html_decoded_names': decoded,
                'normalized_names': normalized, 'classification': 'encoding_only' if len(normalized) == 1 else 'name_variant_requires_review_not_automatic_ID_conflict',
                'occurrences': dict(raw_names)})
    state_counts = Counter(r['lineup_status'] for r in statuses)
    downloaded_counts = Counter(r['download_status'] for r in statuses)
    for fixture, state in zip(planned, statuses):
        for side in ('home', 'away'):
            coverage[(fixture['season'], fixture[side])]['downloaded_matches'] += state['download_status'] == 'downloaded'
            coverage[(fixture['season'], fixture[side])]['unusable_matches'] += state['lineup_status'] != 'parsed'
    summary = {'measured_at': datetime.now(timezone.utc).isoformat(), 'source': str(SOURCE),
        'selection_rule': selection['rule'], 'planned_matches': len(planned), 'target_2024_25_matches': len(targets),
        'lineup_status_counts': dict(state_counts), 'download_status_counts': dict(downloaded_counts),
        'counts': dict(totals), 'all_failures_and_unknowns_retained': True,
        'source_name_variants': variants, 'shot_roster_name_variants': shot_name_variants,
        'canonical_team_source_ids': {k: sorted(v) for k,v in sorted(team_source_ids.items())},
        'canonical_team_id_changes': {k: sorted(v) for k,v in team_source_ids.items() if len(v)>1},
        'team_season_coverage': [{'season': s, 'team': t, **dict(c)} for (s,t),c in sorted(coverage.items())],
        'unknown_starter_coverage': {k:v['summary'] for k,v in references.items()},
        'pilot_reused_receipts': reused, 'identity_errors': errors,
        'minute_measurement_only': {'rows': len(minute_values), 'min': min(minute_values) if minute_values else None,
            'max': max(minute_values) if minute_values else None, 'zero_count': minute_values.count(0),
            'used_to_infer_starters': False, 'stints_reconstructed': False},
        'historical_first_lineup_publication_time_known': False, 'current_results_reviewed': False,
        'model_fit_count': 0, 'same_match_performance_used_as_features': False,
        'reader_reference': 'https://github.com/probberechts/soccerdata', 'soccerdata_installed': False,
        'rights': 'Underlying Understat data rights are separate from open-source reader licensing.',
        'cutoff_reference': 'work/run_earlier_xg_replication_20261025.py:108-117; fixed August 1 + floor(days/28)*28, history date < cutoff',
        'limits': ['Complete 760-match identity cohort; unknown-rate target denominator is all 380 matches, distinct from existing 378 model predictions.',
            'Rates use only measured starter seats; unmeasured nominal seats are shown separately, never assigned known/unknown labels.',
            'Daily and 28-day histories use valid roster identities, not a refit or reproduction of goal-model training eligibility.',
            'Same-date history is excluded; fixed 2023 pools are retrospective completed-season references available before 2024/25.',
            'Name variants do not alone establish an ID conflict. Known elsewhere does not establish a verified transfer transaction.',
            'First observed means not previously observed in this earlier valid EPL cohort, not career debut or first appearance on all of Understat.',
            'Team unknown is relative to these two-season identity pools; it is not the existing goal model team coverage with its longer history.',
            'Absent participants do not identify injuries or unused substitutes; historical lineup publication time remains unknown.']}
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = [('逐场解析状态.json', statuses), ('首发身份观察.json', {'observations': observations, 'references': references}),
               ('覆盖与身份测量.json', summary)]
    for name, value in outputs:
        with (output_directory/name).open('x', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'output': str(output_directory), 'planned': len(planned), 'download_status': dict(downloaded_counts),
        'lineup_status': dict(state_counts), 'identity_errors': len(errors),
        'unknown_rates': {k:v['summary'] for k,v in references.items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
