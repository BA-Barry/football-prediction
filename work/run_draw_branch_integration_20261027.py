"""One authorized fixed reuse of an existing classifier's draw component.

No upstream refits, new weights, probability calibration, thresholds, or current
predictions. The two historical seasons were previously examined in research.
"""
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import gzip
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import numpy as np
from sklearn.metrics import roc_auc_score

from football_model.draw_branch import apply_draw_branch, replace_draw_probability
from football_model.evaluation import probability_scores
from football_model.forecast_diagnostics import moving_block_means
from football_model.probability import score_grid, summarize
from football_model.refresh_experiment import scores_for

ROOT = REPOSITORY_ROOT
OUT = ROOT / 'outputs/条件平局分支整合_20261027'
RICH = ROOT / 'outputs/固定发布日期信息刷新_20261025T071806561562Z/lead_3'
CLASSIFIER = ROOT / 'outputs/直接胜平负分类研究_20261025T091329913808Z/lead_3'
MATCH_FIELDS = ('match_id', 'date', 'period', 'competition', 'home', 'away',
                'actual_score', 'actual_outcome', 'fit_before_date',
                'forecast_before_date', 'horizon_days', 'feature_before_date')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def indexed(rows):
    result = {r['match_id']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate source fixture')
    return result


def parameters_check(parameters, key, row):
    p = parameters[key]
    if (p['before_date'] != row['fit_before_date']
            or p['training_max_date'] >= row['fit_before_date']
            or row['match_id'] in p['reference_match_ids']):
        raise ValueError('A saved component violates its own earlier-training scope')


def replay():
    raw_rich = read(RICH / '逐场完整预测.json')['fresh_performance']
    cls_all = read(CLASSIFIER / '逐场胜平负概率.json')
    classifiers = indexed(cls_all['extra_trees_performance'])
    old_reference = indexed(cls_all['regression_performance'])
    if set(indexed(raw_rich)) != set(classifiers) or set(old_reference) != set(classifiers):
        raise ValueError('Keep complete matched cohorts; do not intersect away unavailable rows')
    rich_params = read(RICH / '全部模型参数.json')
    cls_params = read(CLASSIFIER / '全部分类模型参数.json')
    records, grid_rows = [], []
    max_errors = Counter()
    for base in sorted(raw_rich, key=lambda r: (r['date'], r['match_id'])):
        classifier = classifiers[base['match_id']]
        reference = old_reference[base['match_id']]
        if (any(base[k] != classifier[k] for k in MATCH_FIELDS)
                or any(base[k] != reference[k] for k in MATCH_FIELDS if k != 'feature_before_date')):
            raise ValueError('Fixture, result or publication timing differs between the sources')
        if (base['horizon_days'] != 3
                or date.fromisoformat(base['date'])-timedelta(days=3) != date.fromisoformat(base['forecast_before_date'])
                or not base['fit_before_date'] <= base['forecast_before_date'] < base['date']):
            raise ValueError('Only genuine equal-lead earlier-publication inputs are allowed')
        key = 'extra_trees_performance:' + base['fit_before_date']
        if cls_params[key]['classes'] != [0, 1, 2]:
            raise ValueError('Classifier does not expose ordered H/D/A classes')
        parameters_check(cls_params, key, base)
        parameters_check(rich_params, 'fresh_performance:' + base['fit_before_date'], base)
        p = np.asarray(base['probabilities'], dtype=float)
        c = np.asarray(classifier['probabilities'], dtype=float)
        if (c.shape != (3,) or not np.isfinite(c).all() or (c <= 0).any()
                or not np.isclose(c.sum(), 1, rtol=0, atol=1e-12)
                or not np.allclose(p, reference['probabilities'], rtol=0, atol=1e-12)):
            raise ValueError('Malformed classifier probability or changed rich reference')
        source = score_grid(*base['rates'], rho=0)
        source_p = np.asarray(source.home_draw_away)
        if not np.allclose(p, source_p, rtol=0, atol=1e-12):
            raise ValueError('Archived rich probabilities do not replay their source grid')
        d = float(c[1])
        combined = apply_draw_branch(source, d)
        q = np.asarray(combined.home_draw_away)
        if not np.allclose(q, replace_draw_probability(p, d), rtol=0, atol=1e-12):
            raise ValueError('New score grid differs from the approved draw-branch formula')
        h, a = base['actual_score']
        actual = 0 if h > a else 2 if h < a else 1
        if actual != base['actual_outcome'] or max(h, a) >= min(source.grid.shape):
            raise ValueError('Actual score/outcome/support mismatch')
        source_forecast, forecast = summarize(source), summarize(combined)
        original_top, top = int(p.argmax()), int(q.argmax())
        if original_top != top and original_top != 1 and top != 1:
            raise ValueError('Draw branch directly flipped the H/A winner')
        base_nll = -math.log(float(source.grid[h, a]))
        full_nll = -math.log(float(combined.grid[h, a]))
        class_difference = math.log(float(p[actual]/q[actual]))
        draw_difference = math.log(float(p[1]/d)) if actual == 1 else math.log(float((p[0]+p[2])/(1-d)))
        score_difference = full_nll-base_nll
        max_errors['score_class_NLL_difference'] = max(max_errors['score_class_NLL_difference'], abs(score_difference-class_difference))
        max_errors['score_draw_binary_NLL_difference'] = max(max_errors['score_draw_binary_NLL_difference'], abs(score_difference-draw_difference))
        max_errors['conditional_winner_ratio'] = max(max_errors['conditional_winner_ratio'], abs(float(q[0]/q[2]-p[0]/p[2])))
        if max(max_errors.values()) > 1e-10:
            raise ValueError('Loss decomposition or conditional-winner invariant failed')
        records.append({k: base[k] for k in MATCH_FIELDS} | {
            'source_original_rates': base['rates'], 'original_probabilities': p.tolist(),
            'classifier_draw_probability': d, 'probabilities': q.tolist(),
            'original_forecast': source_forecast, 'forecast': forecast,
            'original_exact_score_nll': base_nll, 'exact_score_nll': full_nll,
            'original_top1': original_top, 'top1': top,
            'top1_changed': original_top != top,
            'corrected': original_top != actual and top == actual,
            'damaged': original_top == actual and top != actual,
            'draw_binary_nll_difference': draw_difference,
            'score_law': 'region-scaled original grid, not independent Poisson with the new expectations',
            'source_classifier_model': key,
            'source_rich_model': 'fresh_performance:' + base['fit_before_date']})
        grid_rows.append({'match_id': base['match_id'], 'row_axis': 'home_goals', 'column_axis': 'away_goals',
                          'probabilities': combined.grid.tolist(),
                          'expected_goals': forecast['expected_goals'],
                          'source_original_rates': base['rates'], 'draw_probability': d})
    with gzip.open(OUT / '逐场统一比分网格.json.gz', 'wt', encoding='utf-8') as f:
        json.dump({'records': grid_rows}, f, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    write(OUT / '逐场整合与纠错.json', {'replayed_at': datetime.now(timezone.utc).isoformat(),
          'historical_input_cutoff_not_actual_publication_instant': True, 'records': records})
    write(OUT / '接入检查.json', {'matched_n': len(records), 'omitted_targets': 0,
          'training_target_overlap': 0, 'chronology_violations': 0, 'direct_H_A_top1_flips': 0,
          'maximum_numerical_errors': dict(max_errors), 'current_forecasts_written': 0,
          'score_NLL_from_final_grid': True, 'new_weights_or_thresholds': False,
          'score_grid_file': str(OUT / '逐场统一比分网格.json.gz')})
    return records


def draw_summary(probabilities, labels):
    d, y = probabilities[:, 1], (labels == 1).astype(int)
    return {'n': len(y), 'mean_probability': float(d.mean()), 'observed_frequency': float(y.mean()),
            'bias_pp': float(100*(d.mean()-y.mean())),
            'auc': float(roc_auc_score(y, d)), 'binary_brier': float(np.mean((d-y)**2)),
            'binary_log_loss': float(-np.mean(y*np.log(d)+(1-y)*np.log1p(-d))),
            'top1_selected': int((probabilities.argmax(axis=1) == 1).sum()),
            'top1_correct': int(((probabilities.argmax(axis=1) == 1) & (labels == 1)).sum())}


def evaluate(rows, intervals=True):
    labels = np.asarray([r['actual_outcome'] for r in rows])
    arrays = {name: np.asarray([r[key] for r in rows]) for name, key in (
        ('original', 'original_probabilities'), ('draw_branch', 'probabilities'))}
    scores, losses = {}, {}
    for name, probabilities in arrays.items():
        prepared = [{**r, 'probabilities': r['original_probabilities'] if name == 'original' else r['probabilities'],
                     'forecast': r['original_forecast'] if name == 'original' else r['forecast']} for r in rows]
        scores[name] = scores_for(prepared)
        exact_key = 'original_exact_score_nll' if name == 'original' else 'exact_score_nll'
        scores[name]['exact_score'] = {'n': len(rows), 'log_loss': float(np.mean([r[exact_key] for r in rows]))}
        scores[name]['draw_binary'] = draw_summary(probabilities, labels)
        # Every market below is read from the region-scaled grid summary.
        y_total = np.asarray([min(sum(r['actual_score']), 4) for r in rows])
        margins = np.asarray([r['actual_score'][1]-r['actual_score'][0]-2 for r in rows])
        y_handicap = np.where(margins > 0, 0, np.where(margins == 0, 1, 2))
        total_p = np.asarray([[r['forecast']['total_goals'][k] for k in ('0', '1', '2', '3', '4+')] for r in prepared])
        handicap_p = np.asarray([[r['forecast']['away_minus_two_jincai'][k] for k in ('win', 'draw', 'lose')] for r in prepared])
        losses[name] = {
            'one_x_two': -np.log(probabilities[np.arange(len(rows)), labels]),
            'exact_score': np.asarray([r[exact_key] for r in rows]),
            'total_goals': -np.log(total_p[np.arange(len(rows)), y_total]),
            'away_minus_two_three_way': -np.log(handicap_p[np.arange(len(rows)), y_handicap])}
    loss_order = list(losses['original'])
    diff = np.column_stack([losses['draw_branch'][k]-losses['original'][k] for k in loss_order])
    original_top = arrays['original'].argmax(axis=1)
    new_top = arrays['draw_branch'].argmax(axis=1)
    corrected = (original_top != labels) & (new_top == labels)
    damaged = (original_top == labels) & (new_top != labels)
    changed = original_top != new_top
    if int(corrected.sum()-damaged.sum()) != int((new_top == labels).sum()-(original_top == labels).sum()):
        raise ValueError('Switch counts fail to reconcile')
    return {'n': len(rows), 'scores': scores,
            'correct_counts': {name: int((p.argmax(axis=1) == labels).sum()) for name, p in arrays.items()},
            'changes': {'changed': int(changed.sum()), 'corrected': int(corrected.sum()), 'damaged': int(damaged.sum()),
                        'both_wrong_after_change': int((changed & (original_top != labels) & (new_top != labels)).sum()),
                        'switched_to_draw': int(((new_top == 1) & (original_top != 1)).sum()),
                        'switched_away_from_draw': int(((new_top != 1) & (original_top == 1)).sum())},
            'difference_order': loss_order, 'mean_loss_differences': dict(zip(loss_order, diff.mean(axis=0).tolist())),
            'four_week_conditional_block_interval': moving_block_means([r['date'] for r in rows], diff, block_weeks=4) if intervals else None,
            'class_loss_contribution': {label: {'n': int((labels == i).sum()),
                'conditional_mean_difference': float(diff[labels == i, 0].mean()),
                'contribution_to_overall_difference': float(np.where(labels == i, diff[:, 0], 0).mean())}
                for i, label in enumerate(('home', 'draw', 'away'))}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / '本次固定整合说明.json', {
        'recorded_at': datetime.now(timezone.utc).isoformat(),
        'authorization': 'Human explicitly agreed to reuse the draw branch in the existing design.',
        'source_rich': str(RICH), 'source_classifier': str(CLASSIFIER), 'horizon_days': 3,
        'formula': 'qD=d; qH=(1-d)*pH/(pH+pA); qA=(1-d)*pA/(pH+pA)',
        'draw_signal': 'raw extra_trees_performance class1 only; no classifier H/A use',
        'score_rule': 'keep source P(score|H/D/A); recompute all final moments/markets',
        'source_seasons': ['2024-25', '2025-26'],
        'historical_status': 'previously examined seasons; retrospective chronological member forecasts, not a fresh untouched test',
        'new_upstream_fits': 0, 'new_fusion_fits': 0, 'parameter_grids': 0,
        'new_calibrations': 0, 'LLM_or_market_probabilities_used': False,
        'current_results_queried': False, 'current_forecasts_replaced': False,
        'stop': 'one full two-season replay; no output-dependent weights, thresholds or re-tuning'})
    rows = replay()
    result = {'finished_at': datetime.now(timezone.utc).isoformat(),
        'by_period': {period: evaluate([r for r in rows if r['period'] == period]) for period in ('2024-25', '2025-26')},
        'by_league_and_period': {period: {comp: evaluate([r for r in rows if r['period'] == period and r['competition'] == comp], False)
                                for comp in ('E0', 'D1', 'SP1', 'I1', 'F1')} for period in ('2024-25', '2025-26')},
        'new_upstream_fits': 0, 'new_fusion_fits': 0, 'parameter_grids': 0,
        'current_forecasts_replaced': False, 'is_same_day_forecast': False,
        'LLM_historical_comparison_available': False}
    write(OUT / '两季完整结果.json', result)
    for period, r in result['by_period'].items():
        print(json.dumps({'period': period, 'n': r['n'], 'correct': r['correct_counts'], 'changes': r['changes'],
                          'HDA_LL': {k: v['one_x_two']['log_loss'] for k,v in r['scores'].items()},
                          'difference': r['mean_loss_differences']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
