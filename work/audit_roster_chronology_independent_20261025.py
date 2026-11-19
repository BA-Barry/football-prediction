"""Independently reconstruct the coverage pools, without the measurement helper."""
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
OUT = ROOT/'outputs/英超跨季首发覆盖_20261025'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    source = read(OUT/'首发身份观察.json')
    observations = source['observations']
    indexed = {r['match_id']: r for r in observations}
    targets = {r['match_id'] for r in observations if r['season'] == '2024/25'}
    covered = {r['match_id'] for r in read(OUT/'现有球队预测覆盖.json')['per_match']
               if r['saved_team_prediction_available']}
    errors, result, comparisons = [], {}, 0
    cache = {}
    for label, measurement in source['references'].items():
        rows = measurement['matches']
        if len(rows) != len(targets) or {r['match_id'] for r in rows} != targets:
            errors.append([label, 'cohort_mismatch'])
        fixed = label.startswith('complete_2023')
        uses_starters = label.endswith('ever_started')
        unknown_seats = all_known = all_teams = both_known = common_goal_known = 0
        for row in rows:
            played = date.fromisoformat(row['date'])
            anchor = date(2024, 8, 1)
            cutoff = anchor if fixed else (anchor + timedelta(days=((played-anchor).days//28)*28)
                if label.startswith('28_day') else played)
            if row['cutoff'] != cutoff.isoformat():
                errors.append([label, row['match_id'], 'cutoff'])
            key = (fixed, cutoff.isoformat(), uses_starters)
            if key not in cache:
                earlier = [r for r in observations if r['match_date'] < cutoff.isoformat()
                           and (not fixed or r['season'] == '2023/24')]
                teams = {r[s] for r in earlier for s in ('home', 'away')}
                suffix = '_starters' if uses_starters else '_appeared'
                players = {p for r in earlier for s in ('home', 'away') for p in r[s+suffix]}
                cache[key] = (teams, players)
            teams, players = cache[key]
            target = indexed[row['match_id']]
            unknown = {s: set(target[s+'_starters']) - players for s in ('home', 'away')}
            for side in ('home', 'away'):
                actual = row[side]
                if unknown[side] != set(actual['unknown_player_ids']):
                    errors.append([label, row['match_id'], side, 'unknown_ids'])
                if len(unknown[side]) != actual['unknown_count']:
                    errors.append([label, row['match_id'], side, 'unknown_count'])
                if actual['team_unknown'] != (target[side] not in teams):
                    errors.append([label, row['match_id'], side, 'team_history'])
                comparisons += 3
            known = not any(unknown.values())
            teams_known = all(target[s] in teams for s in ('home', 'away'))
            unknown_seats += sum(map(len, unknown.values()))
            all_known += known
            all_teams += teams_known
            both_known += known and teams_known
            common_goal_known += known and row['match_id'] in covered
        result[label] = {'full_target_matches': len(targets), 'unknown_starter_seats': unknown_seats,
            'all_22_starters_known': all_known, 'both_identity_history_teams_known': all_teams,
            'all_starters_and_identity_history_teams_known': both_known,
            'known_starters_and_saved_team_prediction': common_goal_known,
            'saved_team_prediction_available': len(covered)}
        expected = measurement['summary']
        for a, b in [('unknown_starter_seats', 'unknown_count'),
                     ('all_22_starters_known', 'matches_all_22_starters_known'),
                     ('both_identity_history_teams_known', 'matches_both_teams_known'),
                     ('all_starters_and_identity_history_teams_known', 'matches_all_22_starters_and_both_teams_known')]:
            if result[label][a] != expected[b]:
                errors.append([label, a, 'summary_mismatch'])
    record = {'checked_at': datetime.now(timezone.utc).isoformat(),
        'method': 'Reconstruct sets directly from all saved observations with date < cutoff; do not call original coverage functions.',
        'comparisons': comparisons, 'errors': errors, 'results': result,
        'same_date_excluded': True, 'new_requests': 0, 'new_model_fits': 0,
        'current_results_reviewed': False,
        'baseline_warning': 'Saved 2024/25 blend weight was estimated using 2024/25 outcomes. Its coverage may be compared, but it is not a fully forward-fitted offset baseline for this target season.'}
    with (OUT/'时间隔离独立核对.json').open('x', encoding='utf-8') as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'comparisons': comparisons, 'errors': errors, 'results': result}, ensure_ascii=False))


if __name__ == '__main__':
    main()
