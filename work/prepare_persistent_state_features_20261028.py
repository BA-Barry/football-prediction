"""Prepare an information-level reuse package; no new fits or probabilities."""
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from football_model.persistent_state_features import build_persistent_state_features, STATE_FEATURE_NAMES

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
SOURCE = ROOT / 'outputs/持续状态净胜球研究_20261028'
ORIGINAL = ROOT / 'outputs/固定发布日期信息刷新_20261025T071806561562Z/features_lead3.json.gz'
OUT = ROOT / 'outputs/持续状态与丰富信息接入_20261028'
CONTEXT = ('match_id', 'date', 'period', 'competition', 'home', 'away', 'fit_before_date',
           'forecast_before_date', 'horizon_days', 'actual_score', 'actual_outcome')


def read(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding='utf-8'))


def zipped(path, value):
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original = read(ORIGINAL)['rows']
    original_ids = [r['match_id'] for r in original]
    if len(set(original_ids)) != len(original_ids):
        raise ValueError('Original rich-input identities are not unique')
    inputs = read(SOURCE / '输入资料.json.gz')
    static = read(SOURCE / '静态初始化.json')
    dynamics = read(SOURCE / '动态参数拟合.json')
    matches = [r for r in inputs['matches'] if r['date'] > static['training_max_date']]
    targets = [{**{key: r[key] for key in ('match_id','date','competition','home','away')},
                'F': r['forecast_before_date']} for r in original]
    bank = build_persistent_state_features(static, dynamics, matches, targets)
    if [r['match_id'] for r in bank['rows']] != original_ids:
        raise ValueError('State extraction changed the original cohort or order')
    saved = {r['match_id']: r for r in read(SOURCE / '逐场概率与状态.json.gz')['records']}
    max_error, checked, retained_fields, joined = 0., 0, 0, []
    for old, extra in zip(original, bank['rows'], strict=True):
        sides = []
        for old_side, new_side in zip(old['fresh_performance_features'], extra['side_features'], strict=True):
            if len(old_side) != 121 or set(old_side) & set(STATE_FEATURE_NAMES):
                raise ValueError('Expected unchanged 121 rich columns with no state-name collision')
            side = {**old_side, **new_side}
            if len(side) != 127 or any(side[k] != value for k, value in old_side.items()):
                raise ValueError('State join changed an original rich measurement')
            sides.append(side)
            retained_fields += len(old_side)
        if old['match_id'] in saved:
            previous = saved[old['match_id']]
            if not extra['available'] or extra['F'] != previous['F']:
                raise ValueError('Saved GAS prediction and extracted state coverage disagree')
            max_error = max(max_error, *(abs(a-b) for a,b in zip(
                extra['explanation']['log_rates'], previous['log_rates'], strict=True)))
            checked += 1
        joined.append({**{key: old[key] for key in CONTEXT},
                       'feature_before_date': old['fresh_feature_before_date'],
                       'side_features': sides,
                       'persistent_state_available': extra['available'],
                       'persistent_state_unavailable_reason': extra['unavailable_reason'],
                       'state_history_max_date': extra['state_history_max_date'],
                       'state_parameter_training_max_date': extra['parameter_training_max_date']})
    if checked != len(saved) or max_error > 1e-12:
        raise ValueError('Extracted states do not reconstruct the already saved GAS rates')
    coverage = {**bank['metadata'], 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_rich_features': str(ORIGINAL), 'source_trusted_history': str(SOURCE / '输入资料.json.gz'),
        'source_static_fit': str(SOURCE / '静态初始化.json'),
        'source_dynamic_fit': str(SOURCE / '动态参数拟合.json'),
        'original_columns_per_side': 121, 'added_columns_per_side': len(STATE_FEATURE_NAMES),
        'joined_columns_per_side': 127, 'retained_original_measurements_verified': retained_fields,
        'state_counts_by_period': {period: dict(Counter(
            r['persistent_state_unavailable_reason'] or 'available' for r in joined if r['period'] == period))
            for period in sorted({r['period'] for r in joined})},
        'saved_probability_rows_rate_checked': checked, 'max_saved_log_rate_reconstruction_error': max_error,
        'new_model_fits': 0, 'new_probabilities_generated': 0, 'current_results_queries': 0,
        'parameter_time_interpretation': 'historical data-end eligibility, not a claim that the model actually existed in 2024; actual fit was performed in 2026',
        'all_missing_training_snapshot': 'initial 2024-08-01 training has no available state values; reuse the original fitted model until earlier training rows contain observations of new columns',
        'interpretation_limit': 'state availability changes with period; any later fit is a full candidate comparison, not isolated causal attribution to six numerical state values',
        'training_status': 'prepared_only_pending_user_structural_choice'}
    zipped(OUT / '状态输入与来源.json.gz', bank)
    zipped(OUT / '联合输入_待训练.json.gz', {'rows': joined})
    (OUT / '覆盖与时序说明.json').write_text(json.dumps(coverage, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(coverage, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
