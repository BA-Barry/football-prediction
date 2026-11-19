"""Recompute descriptive measurements of the saved 20-match roster pilot."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import re

from football_model.xg_history import COMPETITION_TEAM_NAMES

ROOT = REPOSITORY_ROOT
SOURCE = ROOT/'data/understat/roster_pilot_20261024T183946498125Z'
OUT = ROOT/'outputs/历史首发数据接通_20261025'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def text(value):
    return html.unescape(value).strip()


def main():
    selection = read(SOURCE/'试样选择.json')
    receipt = read(SOURCE/'接入记录.json')
    names, ids, team_ids = defaultdict(set), Counter(), defaultdict(set)
    player_occurrences = Counter()
    anchors = defaultdict(lambda: defaultdict(set))
    counts, positions, orders = Counter(), Counter(), defaultdict(set)
    per_match, errors, substitutions, own_goals, zero_minutes = [], [], [], [], []
    for fixture in selection['selected']:
        uid, comp = fixture['understat_id'], fixture['competition']
        raw = read(SOURCE/f'{uid}.json')
        rows = {r['player_id']: r for side in ('h', 'a') for r in raw['rosters'][side].values()}
        fixture_rows = []
        for side, canonical in [('h', fixture['home']), ('a', fixture['away'])]:
            roster = raw['rosters'][side]
            counts['team_sides'] += 1
            starters = sum(r['position'] != 'Sub' for r in roster.values())
            counts['exactly_eleven_sides'] += starters == 11
            counts['starter_seats'] += starters
            fixture_rows.append({'side': side, 'team': canonical, 'roster_rows': len(roster),
                                 'starter_seats': starters})
            if starters != 11:
                errors.append([uid, side, 'not_eleven'])
            template = raw.get('tmpl', {}).get('home' if side == 'h' else 'away', '')
            template_ids = re.findall(r'<label\b[^>]*\bdata-id="([0-9]+)"', template)
            counts['template_player_rows'] += len(template_ids)
            if set(template_ids) != {r['player_id'] for r in roster.values()}:
                errors.append([uid, side, 'template_roster_ids'])
            for key, row in roster.items():
                counts['roster_rows'] += 1
                counts['sub_rows'] += row['position'] == 'Sub'
                ids[row['id']] += 1
                names[row['player_id']].add(text(row['player']))
                player_occurrences[row['player_id']] += 1
                team_ids[canonical].add(row['team_id'])
                positions[row['position']] += 1
                orders[row['position']].add(row['positionOrder'])
                counts['roster_id_equals_player_id'] += row['id'] == row['player_id']
                if key != row['id'] or row['h_a'] != side:
                    errors.append([uid, key, 'row_key_or_side'])
                if int(row['time']) == 0:
                    zero_minutes.append([uid, side, row['player_id'], row['position']])
                if canonical == fixture['anchor_team']:
                    anchors[canonical][fixture['season']].add(row['player_id'])
                for field, reverse in [('roster_in', 'roster_out'), ('roster_out', 'roster_in')]:
                    linked = row[field]
                    if linked == '0':
                        continue
                    counts['nonzero_substitution_links'] += 1
                    other = roster.get(linked)
                    if other is None or other[reverse] != row['id']:
                        errors.append([uid, row['id'], field, 'nonreciprocal_or_other_side'])
                    if field == 'roster_in' and other is not None:
                        substitutions.append({'understat_id': uid, 'side': side,
                            'out_player_id': row['player_id'], 'in_player_id': other['player_id'],
                            'out_position': row['position'], 'in_position': other['position'],
                            'out_minutes': int(row['time']), 'in_minutes': int(other['time'])})
        for side in ('h', 'a'):
            for shot in raw['shots'][side]:
                counts['shots'] += 1
                player = rows.get(shot['player_id'])
                if (player is None or player['h_a'] != side or shot['h_a'] != side
                        or text(player['player']) != text(shot['player'])):
                    errors.append([uid, shot['id'], 'shot_player_identity_or_side'])
                if str(shot['match_id']) != uid or shot['date'][:10] != fixture['date']:
                    errors.append([uid, shot['id'], 'shot_match_identity_or_date'])
                for field, team in [('h_team', 'home'), ('a_team', 'away')]:
                    original = text(shot[field])
                    name = COMPETITION_TEAM_NAMES[comp].get(original, original)
                    if f'football-data:{comp}:{name}' != fixture[team]:
                        errors.append([uid, shot['id'], 'shot_fixture_team', field, original])
                assist = text(shot.get('player_assisted') or '')
                if assist:
                    counts['nonempty_assisted_names'] += 1
                    candidates = [r for r in rows.values() if text(r['player']) == assist and r['h_a'] == side]
                    if len(candidates) != 1:
                        errors.append([uid, shot['id'], 'assisted_name_not_unique_same_side'])
                if shot['result'] == 'OwnGoal':
                    own_goals.append({'understat_id': uid, 'shot_id': shot['id'],
                                      'player_id': shot['player_id'], 'identity_side': side})
        per_match.append({'match_id': fixture['match_id'], 'understat_id': uid,
                          'date': fixture['date'], 'season': fixture['season'], 'sides': fixture_rows})
    repeated = {p: sorted(v) for p, v in names.items() if player_occurrences[p] > 1}
    continuity = []
    for team, seasons in sorted(anchors.items()):
        keys = sorted(seasons)
        common = set.intersection(*(seasons[k] for k in keys))
        continuity.append({'team': team, 'unique_players_by_season': {k: len(seasons[k]) for k in keys},
                           'common_player_ids': sorted(common, key=int), 'common_count': len(common)})
    counts.update(matches=len(per_match), unique_roster_ids=len(ids), unique_player_ids=len(names),
        repeated_player_ids=len(repeated), player_name_conflicts=sum(len(v) != 1 for v in names.values()),
        canonical_team_id_conflicts=sum(len(v) != 1 for v in team_ids.values()),
        cross_season_anchor_common_ids=sum(x['common_count'] for x in continuity),
        substitution_pairs=len(substitutions), zero_minute_rows=len(zero_minutes),
        sub_replaced_by_sub=sum(x['out_position'] == 'Sub' and x['in_position'] == 'Sub' for x in substitutions),
        direct_90_plus_1_pairs=sum(x['out_minutes'] == 90 and x['in_minutes'] == 1 for x in substitutions),
        own_goal_shots=len(own_goals), http_200=sum(x.get('status_code') == 200 for x in receipt['responses']),
        measurement_errors=len(errors))
    result = {'measured_at': datetime.now(timezone.utc).isoformat(), 'source_directory': str(SOURCE),
        'selection_rule': selection['rule'], 'score_used_for_selection': False, 'counts': dict(counts),
        'positions': dict(positions), 'position_order_values': {k: sorted(v, key=int) for k, v in orders.items()},
        'cross_season_continuity': continuity, 'repeated_player_ids_and_names': repeated,
        'substitution_pairs': substitutions, 'own_goal_identity_sides': own_goals,
        'zero_minute_rows': zero_minutes, 'errors': errors,
        'historical_first_lineup_publication_known': False, 'model_fit_count': 0,
        'current_results_reviewed': False,
        'reader_reference': 'https://github.com/probberechts/soccerdata',
        'soccerdata_installed': False,
        'rights': 'Open-source reader licensing does not establish rights to the underlying data.',
        'limits': ['20 selected early-season matches are not a population coverage estimate.',
                   'Non-Sub labels support a starting-XI-known study, not a T-24h lineup forecast.',
                   'Missing participants do not establish injury or unused bench membership.',
                   'Integer minutes and substitution links do not establish exact on-field times.',
                   'Shot date, club names and match_id are checked for identity only; not forecast features.']}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, value in [('试样测量结果.json', result), ('逐场测量.json', per_match)]:
        with (OUT/name).open('x', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'output': str(OUT), 'counts': dict(counts), 'errors': errors}, ensure_ascii=False))


if __name__ == '__main__':
    main()
