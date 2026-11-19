"""Prepare past player-usage model inputs from existing, joined historical data."""
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
from statistics import median

from football_model.historical_player_usage import extract_usage, usage_features

ROOT = REPOSITORY_ROOT
SOURCE = ROOT/'data/understat/roster_seasons_EPL_2023_2024_20261025'
OUT = ROOT/'outputs/历史球员使用输入_20261025'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    fixtures = read(SOURCE/'比赛选择.json')['selected']
    # Reuse the completed fixture identity work; do not redo response/shot audits.
    checked = {r['match_id']: r for r in read(ROOT/'outputs/英超跨季首发覆盖_20261025/逐场解析状态.json')}
    observations, histories = [], defaultdict(list)
    for fixture in fixtures:
        if checked[fixture['match_id']]['identity_status'] != 'verified_by_all_available_shots':
            raise ValueError('Existing fixture identity join is unresolved')
        rows = extract_usage(read(SOURCE/'raw'/f"{fixture['understat_id']}.json"), fixture)
        observations.extend(rows)
        for row in rows:
            histories[row['team_id']].append(row)
    snapshots = []
    for horizon in (1, 3):
        for fixture in fixtures:
            cutoff = date.fromisoformat(fixture['date'])-timedelta(days=horizon)
            sides = {side: usage_features(histories[fixture[side]], team_id=fixture[side], before_date=cutoff)
                     for side in ('home', 'away')}
            snapshots.append({**fixture, 'horizon_days': horizon, 'forecast_before_date': cutoff.isoformat(),
                'match_features': {f'{side}_{name}': value for side in sides
                                   for name, value in sides[side]['features'].items()},
                'team_usage': sides})
    summary = {'created_at': datetime.now(timezone.utc).isoformat(), 'historical_fixtures': len(fixtures),
        'historical_date_min': min(r['date'] for r in fixtures), 'historical_date_max': max(r['date'] for r in fixtures),
        'appearance_rows': sum(len(r['minutes']) for r in observations),
        'unique_players': len({p for r in observations for p in r['minutes']}),
        'team_observations': len(observations), 'forecast_rows': len(snapshots),
        'feature_count_per_fixture': len(snapshots[0]['match_features']),
        'current_results_requested': False, 'target_xi_read': False, 'model_fits': 0,
        'publication_assumption': 'strict previous calendar dates; original publication time not established',
        'league_coverage': 'EPL 2023/24 and 2024/25 only',
        'source_team_minutes_min': min(r['source_team_minutes'] for r in observations),
        'source_team_minutes_max': max(r['source_team_minutes'] for r in observations),
        'source_team_minutes_not_990': sum(r['source_team_minutes'] != 990 for r in observations),
        'horizons': {}}
    for horizon in (1, 3):
        selected = [r for r in snapshots if r['horizon_days'] == horizon]
        teams = [t for r in selected for t in r['team_usage'].values()]
        summary['horizons'][str(horizon)] = {
            'fixtures': len(selected), 'team_sides': len(teams),
            'team_sides_without_history': sum(not t['history_available'] for t in teams),
            'missing_values_by_feature': dict(Counter(name for t in teams for name, value in t['features'].items() if value is None)),
            'chronology_violations': sum(t['latest_history_date'] is not None and t['latest_history_date'] >= t['before_date'] for t in teams),
            'median_effective_players_recent5': median(t['features']['usage_effective_players_recent5'] for t in teams if t['history_available'])}
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT/'历史出场分钟.json', {'observations': observations})
    write(OUT/'赛前使用特征.json', {'rows': snapshots})
    write(OUT/'接入结果.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
