"""Measure historical chance components without reading starting lineups."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
ARCHIVE = ROOT/'data/understat/roster_seasons_EPL_2023_2024_20261025'
OUT = ROOT/'outputs/早期预测研究_20261025'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    selected = read(ARCHIVE/'比赛选择.json')['selected']
    aggregates = {}
    for year in (2023, 2024):
        entries = read(ROOT/f'data/understat/history_20261024T170717354684Z/EPL_{year}.json')['dates']
        aggregates.update({str(r['id']): r for r in entries})
    rows, season_totals, errors = [], defaultdict(Counter), []
    situations = Counter()
    for fixture in selected:
        uid, season = fixture['understat_id'], fixture['season']
        # Only historical shot observations are accessed. No roster/player fields.
        shots = read(ARCHIVE/f'raw/{uid}.json')['shots']
        penalty_in_match = False
        for side, external in [('h', 'home'), ('a', 'away')]:
            by_type = defaultdict(lambda: {'events': 0, 'shot_attempts_excluding_own_goals': 0, 'xg': 0.0})
            own_goals = 0
            for shot in shots[side]:
                category = shot['situation']
                value = float(shot['xG'])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f'Invalid historical xG in {uid}')
                if str(shot['match_id']) != uid or shot['date'][:10] != fixture['date']:
                    raise ValueError(f'Shot identity/date conflict in {uid}')
                is_own = shot['result'] == 'OwnGoal'
                own_goals += is_own
                if is_own and value != 0:
                    errors.append({'understat_id': uid, 'side': side, 'reason': 'nonzero_own_goal_xg', 'xg': value})
                by_type[category]['events'] += 1
                by_type[category]['shot_attempts_excluding_own_goals'] += not is_own
                by_type[category]['xg'] += value
                situations[category] += 1
            total = sum(v['xg'] for v in by_type.values())
            penalty = by_type.get('Penalty', {'events': 0, 'xg': 0.0})
            npxg = total - penalty['xg']
            np_attempts = sum(v['shot_attempts_excluding_own_goals'] for k,v in by_type.items() if k != 'Penalty')
            aggregate = float(aggregates[uid]['xG'][side])
            delta = total - aggregate
            penalty_in_match |= bool(penalty['events'])
            rows.append({'match_id': fixture['match_id'], 'understat_id': uid, 'season': season,
                'match_date': fixture['date'], 'side': side, 'team': fixture[external],
                'raw_shot_sum_xg': total, 'league_index_total_xg': aggregate,
                'raw_shot_sum_minus_source_xg': delta, 'historical_penalty_attempts': penalty['events'],
                'raw_penalty_shot_sum_xg': penalty['xg'], 'raw_shot_sum_excluding_penalty_shots_xg': npxg,
                'historical_non_penalty_shot_attempts': np_attempts,
                'raw_shot_sum_non_penalty_xg_per_attempt': npxg/np_attempts if np_attempts else None,
                'own_goal_events_excluded_from_attempt_counts': own_goals,
                'by_situation': dict(by_type)})
            totals = season_totals[season]
            totals['team_observations'] += 1
            totals['shot_events'] += sum(v['events'] for v in by_type.values())
            totals['penalty_attempts'] += penalty['events']
            totals['penalty_xg'] += penalty['xg']
            totals['raw_total_shot_xg'] += total
            totals['raw_non_penalty_shot_xg'] += npxg
            totals['own_goal_events'] += own_goals
        season_totals[season]['matches'] += 1
        season_totals[season]['matches_with_penalty_attempt'] += penalty_in_match
    summary = {'measured_at': datetime.now(timezone.utc).isoformat(), 'matches': len(selected),
        'team_observations': len(rows), 'observed_situations': dict(situations),
        'max_abs_shot_sum_vs_league_index_xg_difference': max(abs(r['raw_shot_sum_minus_source_xg']) for r in rows),
        'errors': errors, 'lineup_fields_read': False, 'new_requests': 0, 'new_model_fits': 0,
        'current_results_reviewed': False, 'seasons': {},
        'limits': ['Descriptive historical observations, not evidence of improved predictive accuracy.',
            'Any future feature must use strictly earlier matches; current-match components are never a prematch input.',
            'Removing penalties does not remove set pieces: OpenPlay/FromCorner/SetPiece/DirectFreekick remain distinct.',
            'OwnGoal events are not counted as normal shot attempts; their raw xG is audited.',
            'Non-penalty xG cannot directly replace an all-goals mean: future penalties and own goals still need a stated treatment.',
            'Historical source first-publication times and revisions remain unknown.']}
    for season, totals in season_totals.items():
        summary['seasons'][season] = {**dict(totals),
            'penalty_share_of_raw_shot_xg': totals['penalty_xg']/totals['raw_total_shot_xg'],
            'fraction_matches_with_penalty_attempt': totals['matches_with_penalty_attempt']/totals['matches']}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, value in [('历史机会结构测量.json', summary), ('逐场机会结构.json', rows)]:
        with (OUT/name).open('x', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
