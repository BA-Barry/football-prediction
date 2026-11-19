"""Read-only coefficient and input-range audit of saved rich-GLM failures."""
from datetime import date, timedelta
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import joblib
import numpy as np

ROOT = REPOSITORY_ROOT
OUT = ROOT/'outputs/多信息多算法前推_20261025T060704977450Z'
CUTOFF = '2025-08-29'


def main():
    data = json.loads((OUT/'历史特征样本.json').read_text(encoding='utf-8'))
    rows = data['rows']
    by_id = {r['match_id']: r for r in rows}
    rich = joblib.load(OUT/f'models/glm_rich_{CUTOFF}.joblib')
    basic = joblib.load(OUT/f'models/glm_basic_{CUTOFF}.joblib')
    train = sorted((r for r in rows if r['date'] < CUTOFF), key=lambda r: (r['date'], r['match_id']))
    if [r['match_id'] for r in train] != rich._parameters['reference_match_ids']:
        raise ValueError('Training rows disagree with fitted model')
    raw_train = rich._matrix([s for r in train for s in r['rich_features']], rich.feature_names)
    transformed_train = rich.imputer.transform(raw_train)
    standardized_train = rich.scaler.transform(transformed_train)
    coefficients = rich.estimator.coef_
    names = list(rich.imputer.get_feature_names_out(rich.feature_names))
    stats = {}
    for i, name in enumerate(names):
        source = raw_train[:, i] if i < len(rich.feature_names) else transformed_train[:, i]
        valid = source[np.isfinite(source)]
        stats[name] = {
            'n_measured': len(valid), 'n_missing': int((~np.isfinite(source)).sum()),
            'raw_min': float(valid.min()) if len(valid) else None,
            'raw_max': float(valid.max()) if len(valid) else None,
            'raw_quantiles_50_95_99': np.quantile(valid, [.5, .95, .99]).tolist() if len(valid) else None,
            'weighted_scaler_mean': float(rich.scaler.mean_[i]),
            'weighted_scaler_scale': float(rich.scaler.scale_[i]),
            'weighted_scaler_variance': float(rich.scaler.var_[i]),
            'standardized_training_min': float(standardized_train[:, i].min()),
            'standardized_training_max': float(standardized_train[:, i].max()),
            'coefficient': float(coefficients[i]),
        }
    tests = [r for r in rows if r['fit_before_date'] == CUTOFF and r['period'] == '2025-26']
    focus = [r for r in tests if 'Cremonese' in r['match_id'] or 'FC Koln' in r['match_id']]
    # Same-cutoff same-league examples are descriptive controls, not refits or
    # matched causal counterfactuals. Choose by fixture identifier only.
    controls = []
    for comp in ('I1', 'D1'):
        candidates = sorted((r for r in tests if r['competition'] == comp and r not in focus),
                            key=lambda r: r['match_id'])
        if candidates:
            controls.append(candidates[0])
    output_cases = []
    for row in focus+controls:
        raw = rich._matrix(row['rich_features'], rich.feature_names)
        imputed = rich.imputer.transform(raw)
        z = rich.scaler.transform(imputed)
        contribution = z*coefficients
        rates = rich.predict(row['rich_features'], date.fromisoformat(row['date']))
        base_rates = basic.predict(row['basic_features'], date.fromisoformat(row['date']))
        sides = []
        for side in (0, 1):
            top_indices = np.argsort(np.abs(contribution[side]))[-12:][::-1]
            top = []
            for i in top_indices:
                name = names[i]
                raw_value = raw[side, i] if i < len(rich.feature_names) else imputed[side, i]
                measured = np.isfinite(raw_value)
                observed_range = stats[name]
                outside = bool(measured and (raw_value < observed_range['raw_min'] or raw_value > observed_range['raw_max']))
                top.append({'feature': name,
                    'raw_value': float(raw_value) if measured else None,
                    'imputed_value': float(imputed[side, i]), 'standardized_value': float(z[side, i]),
                    'coefficient': float(coefficients[i]), 'log_mean_contribution': float(contribution[side, i]),
                    'outside_earlier_training_raw_range': outside, 'training_summary': observed_range})
            feature_values = row['rich_features'][side]
            meta = row['feature_history']['history']['home' if side == 0 else 'away']
            previous = feature_values['own_previous_league_interval_days']
            latest = meta['latest_used_date']
            prior_date = ((date.fromisoformat(latest)-timedelta(days=int(previous))).isoformat()
                          if latest is not None and previous is not None else None)
            ids = meta['used_match_ids']
            predecessor = ids[-2] if len(ids) >= 2 else None
            if predecessor in by_id and by_id[predecessor]['date'] != prior_date:
                raise ValueError('Previous league interval disagrees with known earlier fixture date')
            eta = float(rich.estimator.intercept_+contribution[side].sum())
            if not math.isclose(math.exp(eta), rates[side], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Coefficient decomposition fails rate replay')
            interval_terms = {name: float(contribution[side, names.index(name)])
                              for name in ('own_previous_league_interval_days', 'opponent_previous_league_interval_days')}
            sides.append({'team': row['home'] if side == 0 else row['away'],
                'is_home': side == 0, 'rate': rates[side], 'basic_rate': base_rates[side],
                'actual_goals': row['actual_score'][side], 'log_rate': eta,
                'intercept': float(rich.estimator.intercept_),
                'sum_log_contributions': float(contribution[side].sum()),
                'interval_terms': interval_terms,
                'interval_terms_sum': sum(interval_terms.values()),
                'rate_multiplier_from_own_interval_term': math.exp(interval_terms['own_previous_league_interval_days']),
                'n_missing_raw_features': int((~np.isfinite(raw[side])).sum()),
                'n_active_missing_indicators': int(imputed[side, len(rich.feature_names):].sum()),
                'history_features_available': feature_values['history_features_available'],
                'observed_same_league_latest_date': latest,
                'observed_previous_same_league_date': prior_date,
                'observed_previous_fixture_id': predecessor,
                'previous_same_league_interval_days': previous,
                'cutoff_staleness_days': feature_values['own_cutoff_staleness_days'],
                'same_league_gap_to_target_days': feature_values['own_same_league_gap_to_match_days'],
                'history_count': feature_values['own_history_count'],
                'top_absolute_log_contributions': top})
        output_cases.append({'match_id': row['match_id'], 'date': row['date'],
            'competition': row['competition'], 'fit_before_date': row['fit_before_date'],
            'role': 'failure_mechanism_focus' if row in focus else 'same_cutoff_league_descriptive_control',
            'actual_score': row['actual_score'], 'sides': sides})
    # Summary across all late matches sharing the exact fitted model only.
    all_rates = rich.predict_many([r['rich_features'] for r in tests], [date.fromisoformat(r['date']) for r in tests])
    raw_test = rich._matrix([s for r in tests for s in r['rich_features']], rich.feature_names)
    interval_i = rich.feature_names.index('own_previous_league_interval_days')
    interval_values = raw_test[:, interval_i]
    interval_valid = interval_values[np.isfinite(interval_values)]
    summary = {
        'scope': 'saved retrospective model/feature diagnostics only; no retraining, clipping, network or current-result review',
        'model': str(OUT/f'models/glm_rich_{CUTOFF}.joblib'), 'cutoff': CUTOFF,
        'earlier_training_matches': len(train), 'training_min_date': train[0]['date'], 'training_max_date': train[-1]['date'],
        'original_features': len(rich.feature_names), 'expanded_features': len(names),
        'scaler_all_finite_and_positive': bool(np.isfinite(rich.scaler.mean_).all() and np.isfinite(rich.scaler.var_).all()
            and np.isfinite(rich.scaler.scale_).all() and (rich.scaler.scale_ > 0).all()),
        'negative_scaler_variances': int((rich.scaler.var_ < 0).sum()),
        'negative_variance_features': [
            {'feature': names[i], 'variance': float(rich.scaler.var_[i]),
             'scale': float(rich.scaler.scale_[i]), 'coefficient': float(coefficients[i]),
             'training_imputed_min': float(transformed_train[:, i].min()),
             'training_imputed_max': float(transformed_train[:, i].max())}
            for i in np.where(rich.scaler.var_ < 0)[0]],
        'nonfinite_coefficients': int((~np.isfinite(coefficients)).sum()),
        'target_matches_this_cutoff': len(tests),
        'target_scoring_sides_rate_above_10': int((np.asarray(all_rates) > 10).sum()),
        'target_scoring_sides_rate_above_5': int((np.asarray(all_rates) > 5).sum()),
        'own_previous_league_interval_training_summary': stats['own_previous_league_interval_days'],
        'own_previous_league_interval_target_max': float(interval_valid.max()),
        'outside_training_raw_interval_scoring_sides': int((interval_valid > stats['own_previous_league_interval_days']['raw_max']).sum()),
        'outside_training_raw_interval_cases': [
            {'match_id': row['match_id'], 'date': row['date'],
             'team': row['home'] if side == 0 else row['away'],
             'own_previous_league_interval_days': row['rich_features'][side]['own_previous_league_interval_days'],
             'latest_observed_same_league_date': row['feature_history']['history']['home' if side == 0 else 'away']['latest_used_date']}
            for row in tests for side in (0, 1)
            if row['rich_features'][side]['own_previous_league_interval_days'] is not None
            and row['rich_features'][side]['own_previous_league_interval_days'] > stats['own_previous_league_interval_days']['raw_max']],
        'cases': output_cases,
        'interpretation': [
            'The dominant exact log-mean terms in these extreme predictions are own/opponent previous observed same-league intervals.',
            'Large intervals between observed league games are not all-competition rest, fitness, absence from all football, or identified promotion effects.',
            'The latest observed league game is recent, but its predecessor can be years earlier in this league-only source history.',
            'Coefficient decomposition proves the fitted numeric mechanism, not a causal sporting explanation or a validated remedy.',
            'All preprocessing values, coefficients and positive scales are finite. Two constant count features retain tiny negative rounding variances (~-3.31e-43), scale1 and negligible coefficients (~-9.33e-18); the extreme interval feature has positive variance53.2675, so those constant-feature artifacts do not explain its large contribution.',
            'No rates are clipped and no failure subgroup is removed from the study scores.'
        ]}
    path = OUT/'丰富线性外推失误诊断.json'
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'path': str(path), 'training_n': len(train),
        'interval_training': summary['own_previous_league_interval_training_summary'],
        'target_interval_max': summary['own_previous_league_interval_target_max'],
        'cases': [{'match_id': c['match_id'], 'rates': [s['rate'] for s in c['sides']],
            'own_intervals': [s['previous_same_league_interval_days'] for s in c['sides']],
            'previous_dates': [s['observed_previous_same_league_date'] for s in c['sides']],
            'top': [{'name': s['top_absolute_log_contributions'][0]['feature'],
                     'term': s['top_absolute_log_contributions'][0]['log_mean_contribution'],
                     'raw': s['top_absolute_log_contributions'][0]['raw_value'],
                     'outside': s['top_absolute_log_contributions'][0]['outside_earlier_training_raw_range']}
                    for s in c['sides']]} for c in output_cases]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
