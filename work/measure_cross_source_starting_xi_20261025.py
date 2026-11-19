"""Measure two existing historical XI labels; do not merge provider identities."""
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import re
import unicodedata

from football_model.impect_season import TEAM_NAMES

ROOT = REPOSITORY_ROOT
PILOT = ROOT/'data/understat/roster_pilot_20261024T183946498125Z'
IMPECT = ROOT/'data/impect/season_20261024T055349773201Z/raw'
PLAYERS = ROOT/'data/impect/pilot_20261024T053746962994Z/players_743.json'
OUT = ROOT/'outputs/历史首发数据接通_20261025'
# Explicit same-match name evidence only, never a cross-provider ID mapping.
ALIASES = {
    "Nathan N'Goumou Minpol": 'Nathan Ngoumou',
    'Kim Min-Jae': 'Min-jae Kim',
    'Dayotchanculle Upamecano': 'Dayot Upamecano',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def name_key(value):
    value = unicodedata.normalize('NFKD', html.unescape(value))
    return re.sub(r'[^a-z0-9]', '', ''.join(c for c in value if not unicodedata.combining(c)).casefold())


def main():
    selection = {x['understat_id']: x for x in read(PILOT/'试样选择.json')['selected']}
    players = {int(x['id']): x['commonname'] for x in read(PLAYERS)}
    matches = []
    for uid, iid in [('23069', 122843), ('23082', 122855)]:
        selected = selection[uid]
        raw = read(PILOT/f'{uid}.json')
        lineup_path = IMPECT/f'lineups_{iid}.json'
        lineup = read(lineup_path)
        if lineup['id'] != iid or lineup['dateTime'][:10] != selected['date']:
            raise ValueError('Independent historical lineup date or identity mismatch')
        sides = []
        for side, key, external in [('h', 'squadHome', 'home'), ('a', 'squadAway', 'away')]:
            squad = lineup[key]
            if selected[external] != 'football-data:D1:' + TEAM_NAMES[squad['id']]:
                raise ValueError('Independent historical home/away team mismatch')
            first = [x for x in raw['rosters'][side].values() if x['position'] != 'Sub']
            second = squad['startingPositions']
            if len(first) != 11 or len(second) != 11:
                raise ValueError('Both sources must explicitly support eleven opening players')
            remaining = {x['playerId']: x for x in second}
            compared = []
            for row in first:
                name = html.unescape(row['player'])
                exact = [i for i in remaining if players[i] == name]
                equivalent = [i for i in remaining if name_key(players[i]) == name_key(name)]
                manual = [i for i in remaining if ALIASES.get(name) == players[i]]
                candidates, basis = ((exact, 'exact_name') if exact else
                    (equivalent, 'format_or_diacritic') if equivalent else
                    (manual, 'manual_name_alias') if manual else ([], 'unresolved'))
                if len(candidates) > 1:
                    raise ValueError('Name evidence has multiple same-team candidates')
                other_id = candidates[0] if candidates else None
                other = remaining.pop(other_id) if other_id is not None else None
                compared.append({'understat_player_id': row['player_id'], 'understat_name': name,
                    'impect_player_id': other_id, 'impect_name': players[other_id] if other else None,
                    'name_evidence': basis, 'understat_position_label': row['position'],
                    'impect_position_label': other['position'] if other else None,
                    'impect_position_side': other.get('positionSide') if other else None})
            sides.append({'side': side, 'team': selected[external], 'impect_match_squad_id': squad['id'],
                'matched': sum(r['name_evidence'] != 'unresolved' for r in compared),
                'unresolved_understat': sum(r['name_evidence'] == 'unresolved' for r in compared),
                'unmatched_impect': list(remaining), 'measurements': compared})
        matches.append({'match_id': selected['match_id'], 'date': selected['date'],
            'understat_match_id': uid, 'impect_match_id': iid,
            'understat_raw': str(PILOT/f'{uid}.json'), 'impect_raw': str(lineup_path), 'sides': sides})
    rows = [r for m in matches for s in m['sides'] for r in s['measurements']]
    result = {'measured_at': datetime.now(timezone.utc).isoformat(), 'matches': matches,
        'summary': {k: sum(r['name_evidence'] == k for r in rows)
                    for k in ['exact_name', 'format_or_diacritic', 'manual_name_alias', 'unresolved']},
        'manual_aliases': ALIASES, 'impect_player_names_source': str(PLAYERS),
        'persistent_cross_provider_id_map_created': False, 'training_performed': False,
        'current_results_reviewed': False, 'historical_lineup_publication_time_known': False,
        'limits': ['Two matches only; three manual name aliases are same-match evidence, not external identity proof.',
                   'Position labels differ across sources; no tactical role equivalence asserted.',
                   'IMPECT currentSquadId is not used for historical team attribution.',
                   'The roster block alone has no date or club-name identity; this measurement uses its existing exact fixture join.']}
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT/'跨源首发测量.json'
    with target.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'output': str(target), 'summary': result['summary']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
