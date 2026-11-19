"""One fixed historical extension of the approved conditional draw branch.

Reuse all 121 archived early features, add 24 trailing match-pattern features,
and refit the same preset 22 times. No parameter search or current predictions.
"""
from collections import Counter
from datetime import date, datetime, timezone
import gzip
import json
import math
from pathlib import Path
import time

import joblib
import numpy as np

from football_model.early_outcome_learners import EarlyOutcomeLearner
from football_model.match_pattern_features import MatchPatternFeatures, FEATURE_NAMES
from football_model.draw_branch import apply_draw_branch
from football_model.probability import score_grid, summarize
from run_npxg_comparison_20261025 import ROOT, COVERAGE, SLUGS, load_source, read
from run_broad_early_models_20261025 import observations
from run_direct_outcomes_20261025 import load_rows
from run_draw_branch_integration_20261027 import (
    MATCH_FIELDS, RICH, CLASSIFIER, indexed, parameters_check, evaluate, write)

SOURCE = RICH.parent
PREVIOUS = ROOT / 'outputs/条件平局分支整合_20261027'
OUT = ROOT / 'outputs/历史比赛模式与平局分支_20261027'
COMPS = ('E0', 'D1', 'SP1', 'I1', 'F1')


def load_provider_draws(records):
    """Exact trusted record -> raw fixture ID; this is NOT target prematch odds.

    Forecast values describe already completed source matches. They enter a
    target's inputs only after MatchPatternFeatures selects dates before F.
    Four-decimal supplier rounding is accepted, never renormalized.
    """
    raw_by_id = {}
    for entry in read(COVERAGE)['by_season']:
        comp = SLUGS[entry['league']]
        for raw in read(entry['source'])['dates']:
            key = str(raw['id'])
            if key in raw_by_id:
                raise ValueError('Duplicate raw provider ID')
            raw_by_id[key] = (comp, raw)
    probabilities, report = {}, {'matched': 0, 'missing': 0, 'by_competition': {}}
    for record in records.values():
        comp, raw = raw_by_id[record.understat_id]
        if (comp != record.competition or raw['isResult'] is not True
                or datetime.fromisoformat(raw['datetime']).date() != record.match_date
                or tuple(int(raw['goals'][side]) for side in ('h', 'a')) != record.goals):
            raise ValueError('Provider pattern source differs from its independently joined fixture')
        key = (record.competition, record.match_id)
        if key in probabilities:
            raise ValueError('Duplicate trusted pattern fixture')
        forecast = raw.get('forecast')
        if forecast is None or any(forecast.get(k) is None for k in ('w', 'd', 'l')):
            value = None
        else:
            values = []
            for label in ('w', 'd', 'l'):
                raw_value = forecast[label]
                if isinstance(raw_value, bool):
                    raise ValueError('Boolean supplier probability')
                value = float(raw_value)
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('Nonfinite or out-of-range supplier probability')
                values.append(value)
            if abs(math.fsum(values)-1) > .0005:
                raise ValueError('Malformed supplier W/D/L probabilities')
            value = values[1]
        probabilities[key] = value
        report['matched'] += 1
        report['missing'] += int(value is None)
        local = report['by_competition'].setdefault(comp, {'matched': 0, 'missing': 0})
        local['matched'] += 1
        local['missing'] += int(value is None)
    report.update(
        join='trusted competition/match_id -> exact independently joined understat_id',
        meaning='historical completed-match supplier chance distribution; not market or target prematch forecast',
        xpts_used=False, rounding_sum_tolerance=.0005,
        availability='downloaded retrospective archive; original publication/revisions/provider training unknown')
    return probabilities, report


def prepare():
    source_records, independent, joins, _, _ = load_source()
    providers, coverage = load_provider_draws(source_records)
    builder = MatchPatternFeatures(observations(source_records), providers)
    rows = load_rows(SOURCE, 3)
    missing, window_counts, provider_missing_windows = [], Counter(), 0
    for row in rows:
        if len(row['performance_features']) != 121:
            raise ValueError('Do not change or re-create the archived original 121 inputs')
        fd = independent[row['match_id']]
        if (row['date'] != fd.match_date.isoformat() or row['actual_score'] != list(fd.full_time)):
            raise ValueError('Saved target differs from independently joined full-time score')
        pattern = builder.for_match(home=row['home'], away=row['away'], competition=row['competition'],
            cutoff=date.fromisoformat(row['forecast_before_date']), match_date=date.fromisoformat(row['date']))
        features = pattern['features']
        if features is None:
            missing.append(row['match_id'])
            features = {key: None for key in FEATURE_NAMES}
        if set(features) != set(FEATURE_NAMES) or set(features) & set(row['performance_features']):
            raise ValueError('Historical pattern feature names collide or differ')
        row['match_features'] = {**row['performance_features'], **features}
        row['pattern_history'] = pattern['metadata']
        for team in pattern['metadata']['history'].values():
            for window, meta in team['windows'].items():
                window_counts[f"{window}:{meta['count']}"] += 1
                provider_missing_windows += int(meta['provider_missing_count'] > 0)
                if any(d >= row['forecast_before_date'] for d in meta['dates']) or row['match_id'] in meta['match_ids']:
                    raise ValueError('Target/current/future fixture entered a historical pattern')
    report = {'source_rows': len(rows), 'pattern_unavailable_match_ids': missing,
        'history_window_actual_counts': dict(window_counts), 'provider_missing_windows': provider_missing_windows,
        'provider_source': coverage, 'source_join_reports': joins,
        'feature_counts': {'archived_unchanged': 121, 'added': len(FEATURE_NAMES), 'total': 121+len(FEATURE_NAMES)}}
    with gzip.open(OUT / '新增历史模式输入与来源.json.gz', 'wt', encoding='utf-8') as handle:
        json.dump({'rows': [{k: row[k] for k in ('match_id', 'date', 'fit_before_date', 'forecast_before_date',
             'pattern_history')} | {'pattern_features': {key: row['match_features'][key] for key in FEATURE_NAMES}}
             for row in rows]}, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    write(OUT / '真实来源覆盖.json', report)
    return rows


def fit_and_predict(rows):
    rich_rows = indexed(read(RICH / '逐场完整预测.json')['fresh_performance'])
    old_classifier_parameters = read(CLASSIFIER / '全部分类模型参数.json')
    rich_parameters = read(RICH / '全部模型参数.json')
    original_branch = indexed(read(PREVIOUS / '逐场整合与纠错.json')['records'])
    if set(rich_rows) != set(original_branch):
        raise ValueError('Original branch and rich targets differ')
    target_rows = [row for row in rows if row['period'] in ('2024-25', '2025-26')]
    if set(indexed(target_rows)) != set(rich_rows):
        raise ValueError('Keep complete old target cohorts rather than intersecting available rows')
    parameters, windows, predictions, matrices = {}, [], [], []
    max_errors = Counter()
    for period in ('2024-25', '2025-26'):
        targets = [row for row in target_rows if row['period'] == period]
        for cutoff in sorted({row['fit_before_date'] for row in targets}):
            start = time.perf_counter()
            training = [row for row in rows if row['date'] < cutoff]
            held = [row for row in targets if row['fit_before_date'] == cutoff]
            old_key = 'extra_trees_performance:' + cutoff
            model = EarlyOutcomeLearner('extra_trees').fit(training, date.fromisoformat(cutoff))
            p = model.parameters()
            old = old_classifier_parameters[old_key]
            for field in ('reference_match_ids', 'before_date', 'training_min_date', 'training_max_date',
                          'n_matches', 'classes', 'half_life_days', 'configuration', 'weighting'):
                if p[field] != old[field]:
                    raise ValueError(f'New classifier changed original preset/training scope: {field}')
            if len(p['feature_names']) != 145:
                raise ValueError('Expected exactly 121 archived + 24 historical pattern inputs')
            key = 'extra_trees_pattern:' + cutoff
            parameters[key] = p | {'horizon_days': 3, 'pipeline_feature_set': 'archived121_plus_pattern24'}
            joblib.dump(model, OUT / 'models' / f'extra_trees_pattern_{cutoff}.joblib', compress=3)
            probabilities = model.predict_many([row['match_features'] for row in held],
                [date.fromisoformat(row['forecast_before_date']) for row in held])
            for row, classifier in zip(held, probabilities, strict=True):
                base, previous = rich_rows[row['match_id']], original_branch[row['match_id']]
                if any(row[k] != base[k] or base[k] != previous[k] for k in MATCH_FIELDS):
                    raise ValueError('Context/result/forecast dates changed')
                parameters_check(parameters, key, base)
                parameters_check(rich_parameters, 'fresh_performance:'+cutoff, base)
                source = score_grid(*base['rates'], rho=0)
                if not np.allclose(source.home_draw_away, base['probabilities'], rtol=0, atol=1e-12):
                    raise ValueError('Archived rich probabilities differ from original grid')
                combined = apply_draw_branch(source, classifier[1])
                q = np.asarray(combined.home_draw_away)
                p0 = np.asarray(source.home_draw_away)
                h, a = row['actual_score']
                actual = row['actual_outcome']
                if max(h, a) >= min(source.grid.shape):
                    raise ValueError('Observed score outside the source score grid')
                original_nll = -math.log(float(source.grid[h, a]))
                new_nll = -math.log(float(combined.grid[h, a]))
                class_diff = math.log(float(p0[actual]/q[actual]))
                max_errors['exact_class_NLL_difference'] = max(max_errors['exact_class_NLL_difference'],
                    abs(new_nll-original_nll-class_diff))
                old_top, new_top = int(p0.argmax()), int(q.argmax())
                if old_top != new_top and old_top != 1 and new_top != 1:
                    raise ValueError('Draw branch directly reversed H/A direction')
                predictions.append({k: base[k] for k in MATCH_FIELDS} | {
                    'source_original_rates': base['rates'], 'original_probabilities': p0.tolist(),
                    'previous_branch_probabilities': previous['probabilities'],
                    'previous_branch_forecast': previous['forecast'],
                    'previous_branch_exact_score_nll': previous['exact_score_nll'],
                    'classifier_probabilities_reference_only': list(classifier),
                    'classifier_draw_probability': float(classifier[1]), 'probabilities': q.tolist(),
                    'original_forecast': summarize(source), 'forecast': summarize(combined),
                    'original_exact_score_nll': original_nll, 'exact_score_nll': new_nll,
                    'source_classifier_model': key, 'score_law': 'original grid scaled by H/D/A regions',
                    'pattern_history_available': all(row['match_features'][key] is not None
                        for key in FEATURE_NAMES if 'provider_' not in key)})
                matrices.append({'match_id': row['match_id'], 'row_axis': 'home_goals',
                    'column_axis': 'away_goals', 'probabilities': combined.grid.tolist(),
                    'expected_goals': summarize(combined)['expected_goals']})
            windows.append({'period': period, 'cutoff': cutoff, 'training_n': len(training), 'target_n': len(held),
                'seconds': time.perf_counter()-start, 'finished_fits': len(parameters)})
            write(OUT / '训练进度.json', {'windows': windows, 'fits': len(parameters)})
            print(json.dumps(windows[-1], ensure_ascii=False), flush=True)
    if len(parameters) != 22 or set(indexed(predictions)) != set(rich_rows) or max(max_errors.values()) > 1e-10:
        raise ValueError('Incomplete fixed run or inconsistent score-grid integration')
    predictions.sort(key=lambda row: (row['date'], row['match_id']))
    write(OUT / '全部分类模型参数.json', parameters)
    write(OUT / '逐场概率与纠错.json', {'generated_at': datetime.now(timezone.utc).isoformat(), 'records': predictions})
    with gzip.open(OUT / '逐场统一比分网格.json.gz', 'wt', encoding='utf-8') as handle:
        json.dump({'records': matrices}, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    write(OUT / '实际接入检查.json', {'matched_n': len(predictions), 'omitted_targets': 0,
        'classifier_fits': len(parameters), 'original_preset_and_train_ids_identical': True,
        'feature_count': 145, 'maximum_numerical_errors': dict(max_errors),
        'new_parameter_grids': 0, 'current_forecasts_written': 0, 'current_results_queried': False})
    return predictions


def compare(rows, intervals):
    against_previous = [{**row, 'original_probabilities': row['previous_branch_probabilities'],
        'original_forecast': row['previous_branch_forecast'],
        'original_exact_score_nll': row['previous_branch_exact_score_nll']} for row in rows]
    return {'pattern_minus_original_rich': evaluate(rows, intervals),
            'pattern_minus_previous_draw_branch': evaluate(against_previous, intervals)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'models').mkdir(exist_ok=True)
    write(OUT / '本轮固定安排.json', {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'routine input extension within the previously approved draw-branch architecture',
        'unchanged': 'original 121 features, classifier preset/seed/weights, training IDs, rich grids, complete cohorts',
        'new_information': 'past per-match npxG balance/variability and actual draw vs supplier chance description',
        'added_features': list(FEATURE_NAMES), 'total_features': 145,
        'source_meaning': 'supplier forecast.d describes completed past games; not target prematch model/odds',
        'no_duplicate_xpts': True, 'windows': [5, 15], 'minimum_history': 5,
        'missing': 'retain target; None plus earlier-training imputer, never an observed zero or older replacement',
        'time_rule': 'P<=F<D; train labels date<P; each feature only source dates<F for that row',
        'horizon_days': 3, 'seasons': ['2024-25', '2025-26'], 'new_classifier_fits': 22,
        'new_parameter_grids': 0, 'new_calibration_or_threshold': False,
        'draw_formula': 'qD=d; qH=(1-d)*pH/(pH+pA); qA=(1-d)*pA/(pH+pA)',
        'score_rule': 'all outcomes and moments read from final scaled grid; exact-score NLL from grid cell',
        'historical_status': 'previously viewed retrospective chronological comparison, not untouched validation',
        'availability_limit': 'source original publication/revisions/provider model training unknown; date-level assumption',
        'measurement_limit': 'descriptive across-match npxG moments mixing opponents/venues, not causal draw/score correlation',
        'stop': 'one full two-season fixed run; retain findings without subset/threshold/weight rescue',
        'market_LLM_or_lineup_used': False, 'current_forecasts_changed': False, 'current_results_queried': False})
    rows = fit_and_predict(prepare())
    result = {'finished_at': datetime.now(timezone.utc).isoformat(),
        'by_period': {period: compare([r for r in rows if r['period'] == period], True)
            for period in ('2024-25', '2025-26')},
        'by_league_and_period': {period: {comp: compare([r for r in rows
             if r['period'] == period and r['competition'] == comp], False) for comp in COMPS}
             for period in ('2024-25', '2025-26')},
        'classifier_fits': 22, 'current_predictions_replaced': False}
    write(OUT / '两季完整结果.json', result)
    for period, comparisons in result['by_period'].items():
        for name, values in comparisons.items():
            print(json.dumps({'period': period, 'comparison': name, 'correct': values['correct_counts'],
                'log_loss_difference': values['mean_loss_differences']['one_x_two'],
                'draw': {key: v['draw_binary'] for key, v in values['scores'].items()}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
