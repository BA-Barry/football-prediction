"""Measure available earlier rows for a proposed conditional head; do not fit."""
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

from football_model.persistent_rich_integration import plan_windows


ROOT = REPOSITORY_ROOT
OUT = ROOT/'outputs/胜平负机制复用收束_20261028'
SOURCE = ROOT/'outputs/持续状态与丰富信息接入_20261028/联合输入_待训练.json.gz'
DRAW = ROOT/'outputs/历史比赛模式与平局分支_20261027/逐场概率与纠错.json'
STATE_FIELDS = (
    'gas_own_attack_state', 'gas_own_defence_state',
    'gas_opponent_attack_state', 'gas_opponent_defence_state',
    'gas_league_base_log_rate', 'gas_home_log_rate_effect',
)


def main():
    with gzip.open(SOURCE, 'rt', encoding='utf-8') as stream:
        rows = json.load(stream)['rows']
    draws = json.loads(DRAW.read_text(encoding='utf-8'))['records']
    draw_bank = {row['match_id']: row for row in draws}
    windows = plan_windows(rows)
    by_id = {row['match_id']: row for row in rows}
    targets = [row for row in rows if row['period'] in ('2024-25', '2025-26')]
    if len(draw_bank) != 3452 or set(draw_bank) != {row['match_id'] for row in targets}:
        raise ValueError('Reuse the entire existing draw-signal cohort')
    for row in targets:
        d = draw_bank[row['match_id']]
        for key in ('date', 'period', 'home', 'away', 'actual_score', 'actual_outcome',
                    'fit_before_date', 'forecast_before_date'):
            if row[key] != d[key]:
                raise ValueError('Draw component and proposed direction targets differ')
        if not 0 < d['classifier_draw_probability'] < 1:
            raise ValueError('The saved draw component must remain a raw positive probability')
    measured = []
    for window in windows:
        history = [by_id[key] for key in window['training_ids']]
        direction_history = [row for row in history if row['actual_outcome'] != 1]
        counts = Counter(row['actual_outcome'] for row in direction_history)
        if set(counts) != {0, 2}:
            raise ValueError('Both earlier home and away outcomes must exist')
        # Only the earlier conditional-head training set can activate a new
        # state column. All original 121 fields are retained, even if missing.
        active = [key for key in STATE_FIELDS if any(
            row['side_features'][0][key] is not None for row in direction_history)]
        inactive = [key for key in STATE_FIELDS if key not in active]
        names = sorted(set(history[0]['side_features'][0])-set(inactive))
        if len(names) != 121+len(active):
            raise ValueError('Unexpected original or state feature schema')
        measured.append({
            'period': window['period'], 'cutoff': window['cutoff'],
            'all_earlier_matches': len(history),
            'direction_earlier_non_draw_matches': len(direction_history),
            'earlier_draw_matches_not_in_direction_fit': len(history)-len(direction_history),
            'earlier_home_wins': counts[0], 'earlier_away_wins': counts[2],
            'direction_training_max_date': max(row['date'] for row in direction_history),
            'target_matches_all_outcomes': len(window['target_ids']),
            'active_state_features': active, 'inactive_state_features': inactive,
            'model_feature_count': len(names),
        })
    result = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'proposal_prepared_not_fitted_major_objective_change_for_discussion',
        'input_source': str(SOURCE), 'existing_draw_source': str(DRAW),
        'prepared_rows': len(rows), 'full_targets': len(targets),
        'window_count': len(measured), 'planned_direction_fits': len(measured),
        'new_fits_so_far': 0, 'new_probabilities_so_far': 0,
        'planned_unchanged_draw_models_refits': 0,
        'windows': measured,
        'rule': 'Training targets date<P; prepared information date<F<match date. Direction labels use only earlier actual H/A. Predict both heads for every target without examining target outcome.',
        'scope_note': 'All original121 plus available six state fields from home perspective, which already contains opponent roles. State-column availability assessed on direction training rows only. Unknown values stay missing, never known zero.',
        'not_authorization': 'Preparation is routine; changing the fitted response from full goals to conditional H/A is a core objective change awaiting user discussion.',
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'条件方向研究准备量.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'rows': len(rows), 'targets': len(targets), 'planned_fits': len(measured),
        'first_window': measured[0], 'last_window': measured[-1],
        'new_fits': 0, 'new_probabilities': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
