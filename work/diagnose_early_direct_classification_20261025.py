"""Read-only replay and mechanism diagnosis of the earlier 2024/25 comparison.

No estimator is fitted here. The later comparison is not inspected or selected.
"""
from collections import Counter, defaultdict
from datetime import date
import gzip
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import joblib
import numpy as np


ROOT = REPOSITORY_ROOT
RUN = ROOT / 'outputs/直接胜平负分类研究_20261025T091329913808Z'
PLAN = RUN / '分析前计划.json'
OUT = RUN / '较早分类机制诊断'
NAMES = ('home', 'draw', 'away')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def quantiles(values):
    a = np.asarray(values, float)
    return {str(q): float(np.quantile(a, q)) for q in (0, .1, .25, .5, .75, .9, .95, .99, 1)}


def losses(p, y):
    return -np.log(p[np.arange(len(y)), y])


def scores(p, y):
    truth = np.eye(3)[y]
    return {'n': len(y), 'log_loss': float(losses(p, y).mean()),
        'brier_sum': float(np.square(p - truth).sum(axis=1).mean()),
        'rps': float(np.square(np.cumsum(p, axis=1)[:, :2] - np.cumsum(truth, axis=1)[:, :2]).sum(axis=1).mean()/2),
        'top1_accuracy': float((p.argmax(axis=1) == y).mean()),
        'mean_probabilities': p.mean(axis=0).tolist(),
        'observed_frequencies': (np.bincount(y, minlength=3)/len(y)).tolist()}


def calibration(p, y):
    result = {'class_bins': {}, 'top_confidence_bins': []}
    for c, name in enumerate(NAMES):
        bins = []
        weighted_gap = 0.
        for k in range(10):
            select = (p[:, c] >= k/10) & (p[:, c] < (k+1)/10 if k < 9 else p[:, c] <= 1)
            n = int(select.sum())
            if n:
                pred = float(p[select, c].mean())
                obs = float((y[select] == c).mean())
                bins.append({'lower': k/10, 'upper': (k+1)/10, 'n': n,
                    'mean_probability': pred, 'observed_frequency': obs, 'gap': pred-obs})
                weighted_gap += n*abs(pred-obs)/len(y)
        result['class_bins'][name] = {'bins': bins, 'ece_10': weighted_gap}
    top = p.max(axis=1)
    right = p.argmax(axis=1) == y
    for lo, hi in ((0, .4), (.4, .5), (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.00000001)):
        select = (top >= lo) & (top < hi)
        if select.any():
            result['top_confidence_bins'].append({'lower': lo, 'upper': min(hi, 1.),
                'n': int(select.sum()), 'mean_confidence': float(top[select].mean()),
                'top1_accuracy': float(right[select].mean())})
    result['true_class_probability_quantiles'] = quantiles(p[np.arange(len(y)), y])
    return result


def feature_detail(model, expanded, zrow, imputed, raw, train_min, train_max, index, coefficient=None):
    name = expanded[index]
    base = model.feature_names.index(name) if name in model.feature_names else None
    return {'feature': name, 'standardized_value': float(zrow[index]),
        'imputed_value': float(imputed[index]), 'training_mean_weighted': float(model.scaler.mean_[index]),
        'training_scale_weighted': float(model.scaler.scale_[index]),
        'training_min': float(train_min[index]), 'training_max': float(train_max[index]),
        'outside_training_raw_range': bool(imputed[index] < train_min[index] or imputed[index] > train_max[index]),
        'raw_value': None if base is None or not math.isfinite(raw[base]) else float(raw[base]),
        **({} if coefficient is None else {'coefficient_difference': float(coefficient),
            'linear_margin_contribution': float(coefficient*zrow[index])})}


def main():
    plan = read(PLAN)
    source = Path(plan['source_study'])
    with gzip.open(source/'features_lead1.json.gz', 'rt', encoding='utf-8') as handle:
        # Reading the existing archive is not target selection. Drop the later
        # period before inspecting any of its labels or measurements.
        prepared = [r for r in json.load(handle)['rows'] if r['period'] in ('2022-23', '2023-24', '2024-25')]
    prepared.sort(key=lambda r: (r['date'], r['match_id']))
    targets = [r for r in prepared if r['period'] == '2024-25']
    if len(targets) != 1733:
        raise ValueError('Earlier comparison support unexpectedly changed')
    refs = read(source/'lead_1/逐场完整预测.json')
    regression = {r['match_id']: r for r in refs['fresh_allcomp'] if r['period'] == '2024-25'}
    del refs
    records, windows, feature_agg = [], [], defaultdict(lambda: {'n': 0, 'outside_range_n': 0, 'max_abs_z': 0., 'max_abs_margin_contribution': 0.})
    max_z_cases, max_contribution_cases = [], []
    for cutoff in sorted({r['fit_before_date'] for r in targets}):
        held = [r for r in targets if r['fit_before_date'] == cutoff]
        history = [r for r in prepared if r['date'] < cutoff]
        modelpath = RUN/f'lead_1/models/logistic_allcomp_{cutoff}.joblib'
        model = joblib.load(modelpath)
        parameters = model.parameters()
        if parameters['reference_match_ids'] != [r['match_id'] for r in history]:
            raise ValueError('Saved training identities differ from independently dated source')
        names = model.feature_names
        expanded = list(model.imputer.get_feature_names_out(names))
        trainraw = model._matrix([r['fresh_allcomp_features'][0] for r in history], names)
        raw = model._matrix([r['fresh_allcomp_features'][0] for r in held], names)
        trainimp = model.imputer.transform(trainraw)
        imp = model.imputer.transform(raw)
        train_z = model.scaler.transform(trainimp)
        z = model.scaler.transform(imp)
        p = np.asarray(model.predict_many([r['fresh_allcomp_features'][0] for r in held],
            [date.fromisoformat(r['forecast_before_date']) for r in held]))
        train_p = model.estimator.predict_proba(train_z)
        pfmodel = joblib.load(RUN/f'lead_1/models/logistic_performance_{cutoff}.joblib')
        pf_p = np.asarray(pfmodel.predict_many([r['fresh_performance_features'][0] for r in held],
            [date.fromisoformat(r['forecast_before_date']) for r in held]))
        treemodel = joblib.load(RUN/f'lead_1/models/extra_trees_allcomp_{cutoff}.joblib')
        tree_p = np.asarray(treemodel.predict_many([r['fresh_allcomp_features'][0] for r in held],
            [date.fromisoformat(r['forecast_before_date']) for r in held]))
        logits = z@model.estimator.coef_.T+model.estimator.intercept_
        e = np.exp(logits-logits.max(axis=1, keepdims=True))
        replay = e/e.sum(axis=1, keepdims=True)
        if not np.allclose(replay, p, rtol=1e-12, atol=1e-12):
            raise ValueError('Saved multinomial softmax cannot be replayed')
        train_y = np.asarray([r['actual_outcome'] for r in history])
        y = np.asarray([r['actual_outcome'] for r in held])
        raw_w = np.exp(-math.log(2)*np.asarray([(date.fromisoformat(cutoff)-date.fromisoformat(r['date'])).days for r in history])/180)
        w = raw_w*len(history)/raw_w.sum()
        if abs(float(w.sum())-parameters['weighting']['sum']) > 1e-7:
            raise ValueError('Replayed normalized earlier weights disagree')
        train_min, train_max = trainimp.min(axis=0), trainimp.max(axis=0)
        varying = np.ptp(trainimp, axis=0) > 1e-12
        singular = np.linalg.svd(train_z[:, varying]*np.sqrt(w)[:, None], compute_uv=False)
        rank_tol = float(singular[0]*max(len(history), int(varying.sum()))*np.finfo(float).eps)
        design_rank = int((singular > rank_tol).sum())
        reference_p = np.asarray([regression[r['match_id']]['probabilities'] for r in held])
        held_loss = losses(p, y)
        windows.append({'cutoff': cutoff, 'held_n': len(held), 'training_n': len(history),
            'effective_sample_size': float(raw_w.sum()**2/np.square(raw_w).sum()),
            'effective_sample_fraction': float(raw_w.sum()**2/np.square(raw_w).sum()/len(history)),
            'raw_weight_sum': float(raw_w.sum()), 'normalizing_factor': float(w[0]/raw_w[0]),
            'normalized_weight_sum': float(w.sum()), 'raw_weight_range': [float(raw_w.min()), float(raw_w.max())],
            'normalized_weight_range': [float(w.min()), float(w.max())],
            'weighted_class_frequencies': [float(w[train_y == c].sum()/w.sum()) for c in range(3)],
            'weighted_training_log_loss': float(np.average(losses(train_p, train_y), weights=w)),
            'unweighted_training_log_loss': float(losses(train_p, train_y).mean()),
            'held_log_loss': float(held_loss.mean()),
            'held_regression_log_loss': float(losses(reference_p, y).mean()),
            'held_top1_accuracy': float((p.argmax(axis=1) == y).mean()),
            'effective_l2_coefficient_in_weighted_mean_objective': 1/float(w.sum()),
            'n_expanded_features': z.shape[1],
            'n_nonconstant_expanded_features': int(varying.sum()),
            'weighted_standardized_design_numerical_rank': design_rank,
            'rank_note': 'Descriptive numerical rank at ordinary float tolerance; no feature selection or refitting.',
            'coefficient_l2_norm': float(np.linalg.norm(model.estimator.coef_)),
            'intercepts': model.estimator.intercept_.tolist(), 'iterations': model.estimator.n_iter_.tolist(),
            'max_abs_standardized_input': float(np.max(np.abs(z))),
            'probability_sum_max_absolute_error': float(np.max(np.abs(p.sum(axis=1)-1))),
            'softmax_max_absolute_replay_error': float(np.max(np.abs(replay-p)))})
        for j, row in enumerate(held):
            actual = int(y[j]); predicted = int(p[j].argmax())
            # If top1 is correct, use the strongest competing class. The signed
            # margin is always competitor-vs-observed-class, not an attribution
            # to an invented binary target.
            competitor = max((c for c in range(3) if c != actual), key=lambda c: logits[j, c])
            beta = model.estimator.coef_[competitor]-model.estimator.coef_[actual]
            contributions = z[j]*beta
            zi = int(np.abs(z[j]).argmax()); ci = int(np.abs(contributions).argmax())
            top_indices = np.argsort(contributions)[-8:][::-1]
            supporting_indices = np.argsort(contributions)[:5]
            ctx = {k: row[k] for k in ('match_id', 'date', 'competition', 'home', 'away', 'fit_before_date', 'forecast_before_date', 'actual_score', 'actual_outcome')}
            item = {**ctx, 'logistic_probabilities': p[j].tolist(), 'performance_logistic_probabilities': pf_p[j].tolist(),
                'extra_trees_probabilities': tree_p[j].tolist(), 'regression_probabilities': reference_p[j].tolist(),
                'logistic_loss': float(held_loss[j]), 'regression_loss': float(-math.log(reference_p[j, actual])),
                'loss_difference': float(held_loss[j]+math.log(reference_p[j, actual])),
                'top1_class': NAMES[predicted], 'strongest_competing_class': NAMES[competitor],
                'competitor_minus_actual_logit_margin': float(logits[j, competitor]-logits[j, actual]),
                'intercept_margin': float(model.estimator.intercept_[competitor]-model.estimator.intercept_[actual]),
                'margin_sum_of_feature_contributions': float(contributions.sum()),
                'margin_contribution_groups': {
                    'allcomp_measurements_and_their_missing_indicators': float(sum(contributions[i] for i, n in enumerate(expanded) if 'allcomp_' in n)),
                    'initial_goal_xg_rate_inputs': float(sum(contributions[i] for i, n in enumerate(expanded) if n.startswith(('log_goal_', 'log_xg_')))),
                    'all_other_features': float(sum(contributions[i] for i, n in enumerate(expanded) if 'allcomp_' not in n and not n.startswith(('log_goal_', 'log_xg_'))))},
                'source_coverage_characteristics': {
                    'any_previous3_count_below_3': any(row['fresh_allcomp_features'][0].get(side+'_allcomp_previous3_count') is not None and row['fresh_allcomp_features'][0][side+'_allcomp_previous3_count'] < 3 for side in ('own', 'opponent')),
                    'any_unresolved_count_positive': any(row['fresh_allcomp_features'][0].get(side+'_allcomp_unresolved_count_'+window+'d', 0) > 0 for side in ('own', 'opponent') for window in ('7', '14', '28')),
                    'any_window_not_usable': any(row['fresh_allcomp_features'][0].get(side+'_allcomp_window_usable_'+window+'d') == 0 for side in ('own', 'opponent') for window in ('7', '14', '28'))},
                'largest_abs_standardized_input': feature_detail(model, expanded, z[j], imp[j], raw[j], train_min, train_max, zi),
                'largest_abs_margin_contribution': feature_detail(model, expanded, z[j], imp[j], raw[j], train_min, train_max, ci, beta[ci]),
                'largest_positive_margin_contributions': [feature_detail(model, expanded, z[j], imp[j], raw[j], train_min, train_max, int(i), beta[i]) for i in top_indices if contributions[i] > 0],
                'largest_negative_margin_contributions': [feature_detail(model, expanded, z[j], imp[j], raw[j], train_min, train_max, int(i), beta[i]) for i in supporting_indices if contributions[i] < 0]}
            records.append(item)
            max_z_cases.append(item)
            max_contribution_cases.append(item)
            for i, name in enumerate(expanded):
                f = feature_agg[name]
                f['n'] += 1
                f['outside_range_n'] += int(imp[j, i] < train_min[i] or imp[j, i] > train_max[i])
                f['max_abs_z'] = max(f['max_abs_z'], float(abs(z[j, i])))
                f['max_abs_margin_contribution'] = max(f['max_abs_margin_contribution'], float(abs(contributions[i])))
        print(json.dumps({'diagnosed_earlier_cutoff': cutoff, 'n': len(held), 'weighted_train_ll': windows[-1]['weighted_training_log_loss'],
            'held_ll': windows[-1]['held_log_loss'], 'effective_n': windows[-1]['effective_sample_size']}, ensure_ascii=False), flush=True)
    records.sort(key=lambda r: (r['date'], r['match_id']))
    y = np.asarray([r['actual_outcome'] for r in records])
    p = np.asarray([r['logistic_probabilities'] for r in records])
    r = np.asarray([r['regression_probabilities'] for r in records])
    pf = np.asarray([r['performance_logistic_probabilities'] for r in records])
    tree = np.asarray([r['extra_trees_probabilities'] for r in records])
    delta = losses(p, y)-losses(r, y)
    order = np.argsort(delta)[::-1]
    upper_tails = []
    for k in sorted({1, 5, 10, 20, 50, math.ceil(len(y)*.01), math.ceil(len(y)*.05), math.ceil(len(y)*.1)}):
        rest = order[k:]
        upper_tails.append({'top_k_largest_excess_losses': k,
            'sum_loss_difference_in_top_k': float(delta[order[:k]].sum()),
            'fraction_of_full_signed_net_excess': float(delta[order[:k]].sum()/delta.sum()),
            'remaining_n': len(rest), 'remaining_mean_difference': float(delta[rest].mean()),
            'remaining_logistic_log_loss': float(losses(p, y)[rest].mean()),
            'remaining_regression_log_loss': float(losses(r, y)[rest].mean())})
    monthly = {}
    for month in sorted({row['date'][:7] for row in records}):
        select = np.asarray([row['date'].startswith(month) for row in records])
        monthly[month] = {'n': int(select.sum()), 'logistic_log_loss': float(losses(p, y)[select].mean()),
            'regression_log_loss': float(losses(r, y)[select].mean()),
            'mean_difference': float(delta[select].mean()), 'logistic_worse_fraction': float((delta[select] > 0).mean()),
            'mean_top_confidence': float(p[select].max(axis=1).mean()),
            'top1_accuracy': float((p[select].argmax(axis=1) == y[select]).mean())}
    grouped = {}
    for label, select in {
            'true_probability_below_0.01': p[np.arange(len(y)), y] < .01,
            'true_probability_below_0.05': p[np.arange(len(y)), y] < .05,
            'top_confidence_at_least_0.8': p.max(axis=1) >= .8,
            'top_confidence_at_least_0.9': p.max(axis=1) >= .9,
            'max_abs_standardized_input_at_least_5': np.asarray([row['largest_abs_standardized_input']['standardized_value'] for row in records]).__abs__() >= 5,
            'max_abs_standardized_input_below_5': np.asarray([row['largest_abs_standardized_input']['standardized_value'] for row in records]).__abs__() < 5,
            'any_previous3_count_below_3': np.asarray([row['source_coverage_characteristics']['any_previous3_count_below_3'] for row in records]),
            'any_unresolved_count_positive': np.asarray([row['source_coverage_characteristics']['any_unresolved_count_positive'] for row in records]),
            'any_window_not_usable': np.asarray([row['source_coverage_characteristics']['any_window_not_usable'] for row in records]),
            'all_windows_usable': np.asarray([not row['source_coverage_characteristics']['any_window_not_usable'] for row in records]),
        }.items():
        grouped[label] = {'n': int(select.sum()),
            'mean_loss_difference': float(delta[select].mean()) if select.any() else None,
            'sum_loss_difference': float(delta[select].sum()),
            'top1_accuracy': float((p[select].argmax(axis=1) == y[select]).mean()) if select.any() else None}
    summary = {'scope': {'period': '2024-25', 'horizon_days': 1, 'classification_run': str(RUN),
            'source_study': str(source), 'original_preanalysis_plan': str(PLAN),
            'source_features': str(source/'features_lead1.json.gz'),
            'regression_probabilities': str(source/'lead_1/逐场完整预测.json'),
            'saved_classifier_models': str(RUN/'lead_1/models'),
            'method': 'read-only replay; no fitting, clipping, smoothing, calibration fitting, or parameter selection',
            'later_period_analyzed': False, 'current_results_reviewed': False, 'lineups_used': False,
            'post_hoc_diagnosis': True, 'causal_failure_claim': False},
        'scores': {'logistic_allcomp': scores(p, y), 'regression_allcomp': scores(r, y),
            'logistic_performance': scores(pf, y), 'extra_trees_allcomp': scores(tree, y)},
        'loss_difference': {'mean': float(delta.mean()), 'sum': float(delta.sum()),
            'positive_n': int((delta > 0).sum()), 'negative_n': int((delta < 0).sum()),
            'positive_fraction': float((delta > 0).mean()), 'quantiles': quantiles(delta),
            'positive_excess_sum': float(delta[delta > 0].sum()),
            'negative_excess_sum': float(delta[delta < 0].sum()),
            'tail_share_interpretation': 'Share is of signed net excess; it can exceed 100% when improvements elsewhere offset tail failures.'},
        'upper_tail_concentration': upper_tails, 'by_month': monthly, 'diagnostic_groups': grouped,
        'calibration': {'logistic_allcomp': calibration(p, y), 'regression_allcomp': calibration(r, y),
            'extra_trees_allcomp': calibration(tree, y)},
        'training_windows': windows,
        'normalization_explanation': {'sample_ess': '(sum raw weight)^2 / sum(raw weight^2); changes in n and ESS are not fitted extra parameters',
            'objective': 'weighted mean multinomial log loss + ||W||^2/(2*C*sum weights); sum weights=n, C=1',
            'effective_l2': '1/n, so normalization to mean weight 1 does not hold effective L2 fixed as n grows',
            'probability_normalizer': 'native learned multinomial softmax; replay checked without clipping or extra renormalization',
            'calibration': 'descriptive held-out bins only; no calibration fit or adjustment',
            'train_test_comparison': 'In-sample weighted losses and unweighted later-window losses are descriptive and have different sampling weights; their gap is not an unbiased estimate of overfit.'},
        'feature_diagnostics': {'removed_raw_gap_features_present': [name for name in feature_agg if any(name.endswith(s) for s in ('cutoff_staleness_days', 'previous_league_interval_days', 'same_league_gap_to_match_days'))],
            'features_by_max_abs_standardized_value': sorted([{'feature': name, **v} for name, v in feature_agg.items()], key=lambda f: f['max_abs_z'], reverse=True),
            'features_by_max_abs_competitor_margin_contribution': sorted([{'feature': name, **v} for name, v in feature_agg.items()], key=lambda f: f['max_abs_margin_contribution'], reverse=True),
            'attribution_caveat': 'Correlated standardized inputs share coefficients. These exact linear contributions explain a saved prediction algebraically, not the causal effect of a football fact.'},
        'largest_excess_loss_cases': [records[int(i)] for i in order[:20]],
        'largest_standardized_input_cases': sorted(max_z_cases, key=lambda row: abs(row['largest_abs_standardized_input']['standardized_value']), reverse=True)[:10],
        'largest_margin_contribution_cases': sorted(max_contribution_cases, key=lambda row: abs(row['largest_abs_margin_contribution']['linear_margin_contribution']), reverse=True)[:10]}
    earlier_comparison = read(RUN/'lead_1/较早完整比较.json')
    summary['by_actual_class'] = {name: {
        'n': earlier_comparison['scores']['logistic_allcomp']['by_actual_class'][name]['n'],
        'logistic_log_loss': earlier_comparison['scores']['logistic_allcomp']['by_actual_class'][name]['log_loss'],
        'regression_log_loss': earlier_comparison['scores']['regression_allcomp']['by_actual_class'][name]['log_loss'],
        'difference': earlier_comparison['scores']['logistic_allcomp']['by_actual_class'][name]['log_loss'] - earlier_comparison['scores']['regression_allcomp']['by_actual_class'][name]['log_loss']}
        for name in NAMES}
    summary['interpretation'] = {
        'scope_limit': 'Only this fixed logistic configuration and this earlier 2024/25 horizon-1 comparison; not a verdict on direct classification as a family.',
        'measured_failure_pattern': 'Recurring upper-tail probability errors across all ten months. Not only one or two pathological fixtures, and not uniformly worse fixture by fixture.',
        'probability_mechanism': 'Native multinomial softmax is reproduced exactly; broad top-confidence bins overstate their observed success rates. A valid softmax is not a calibration guarantee.',
        'rare_coverage_mechanism': 'Some rare coverage/missingness states are magnified by small training weighted standard deviations and materially affect saved logit margins. This is an algebraic explanation in individual cases, not proven causal generalization failure.',
        'rare_coverage_not_total_explanation': 'The 1624 fixtures with every |z|<5 still lose 0.04609 log loss on average; 1723 fixtures with all schedule windows marked usable still lose 0.04996.',
        'not_old_gap_incident': 'Removed raw gap fields are absent. The old 812-day raw-gap extrapolation incident is not reproduced; the largest |z| states here lie within their training raw ranges.',
        'correlation_caveat': '161 inputs become 287 columns after missingness indicators; 277 vary, but weighted numerical rank is 128 in each cutoff. Individual coefficients can cancel, so grouped margins are also reported; rank or large z alone does not prove causation.',
        'weighting_caveat': 'ESS is only a weight-dispersion equivalent, not a guarantee of independent fixtures. Accumulating nominal n reduces effective mean-objective L2 under fixed C even while weighted ESS remains much smaller.',
        'action_taken': 'Documentation and read-only diagnosis only. No changes to C, feature selection, probabilities, calibration, running modules, or later-period model choice.'}
    OUT.mkdir(exist_ok=True)
    write(OUT/'较早分类误差机制诊断.json', summary)
    # A compact factual first pass; explanatory interpretation is written after
    # inspecting these measurements, rather than predeclaring a mechanism.
    print(json.dumps({'output': str(OUT), 'scores': summary['scores'], 'loss_difference': summary['loss_difference'],
        'upper_tails': upper_tails, 'by_month': monthly,
        'diagnostic_groups': grouped,
        'top_features_z': summary['feature_diagnostics']['features_by_max_abs_standardized_value'][:10],
        'top_features_margin': summary['feature_diagnostics']['features_by_max_abs_competitor_margin_contribution'][:10]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
