"""Direct one-fixture 1X2 learning on the same fixed-publication information."""
import argparse
from datetime import date, datetime, timezone
import gc
import gzip
import json
from pathlib import Path
import time

import joblib
import numpy as np

from football_model.early_outcome_learners import EarlyOutcomeLearner
from football_model.evaluation import probability_scores, calibration_bins
from football_model.forecast_diagnostics import moving_block_means
from run_npxg_comparison_20261025 import ROOT, read
from run_shot_increment_20261025 import write


ARMS = {'logistic_performance': ('logistic', 'performance'), 'logistic_allcomp': ('logistic', 'allcomp'),
        'extra_trees_performance': ('extra_trees', 'performance'), 'extra_trees_allcomp': ('extra_trees', 'allcomp')}
COMPS = ('E0', 'D1', 'SP1', 'I1', 'F1')
CONTEXT_KEYS = ('match_id', 'date', 'period', 'competition', 'home', 'away', 'actual_score', 'actual_outcome',
                'fit_before_date', 'forecast_before_date', 'horizon_days')


def load_rows(source, lead):
    with gzip.open(source/f'features_lead{lead}.json.gz', 'rt', encoding='utf-8') as handle:
        prepared = json.load(handle)['rows']
    rows = []
    for row in prepared:
        if not row['fit_before_date'] <= row['forecast_before_date'] < row['date']:
            raise ValueError('Direct classification must retain the same legal publication dates')
        h, a = row['actual_score']
        if row['actual_outcome'] != (0 if h > a else 2 if h < a else 1):
            raise ValueError('Direct label must agree with the independently checked full-time score')
        fresh = row['fresh_timing']
        if (fresh['forecast_before_date'] != row['forecast_before_date']
                or fresh['feature_before_date'] != row['forecast_before_date']
                or fresh['parameter_before_date'] != row['fit_before_date']):
            raise ValueError('Use the earlier prepared fresh snapshots, not hindsight features')
        context = {k: row[k] for k in CONTEXT_KEYS}
        for name in ('performance', 'allcomp'):
            # Home perspective already carries own AND opponent measurements.
            # The away perspective is the corresponding role reversal, not a
            # second independent outcome observation.
            context[name+'_features'] = row['fresh_'+name+'_features'][0]
        context['feature_before_date'] = row['fresh_feature_before_date']
        rows.append(context)
    rows.sort(key=lambda r: (r['date'], r['match_id']))
    if len({r['match_id'] for r in rows}) != len(rows):
        raise ValueError('A match contributes exactly one classification observation')
    return rows


def score_track(rows):
    p = np.asarray([r['probabilities'] for r in rows], dtype=float)
    y = np.asarray([r['actual_outcome'] for r in rows], dtype=int)
    if (not np.isfinite(p).all() or np.any(p <= 0) or np.any(p >= 1)
            or not np.allclose(p.sum(axis=1), 1, atol=1e-12, rtol=0)):
        raise ValueError('Evaluate native positive normalized probabilities without clipping or smoothing')
    truth = np.eye(3)[y]
    losses = {'log_loss': -np.log(p[np.arange(len(y)), y]),
        'brier_sum': np.square(p-truth).sum(axis=1),
        'rps': np.square(np.cumsum(p, axis=1)[:, :2]-np.cumsum(truth, axis=1)[:, :2]).sum(axis=1)/2,
        'misclassification': (p.argmax(axis=1) != y).astype(float)}
    return probability_scores(p, y) | {'mean_probabilities': p.mean(axis=0).tolist(),
        'observed_frequencies': (np.bincount(y, minlength=3)/len(y)).tolist(),
        'class_probability_bins': calibration_bins(p, y, bins=10),
        'by_actual_class': {name: {'n': int((y == c).sum()),
            'log_loss': float(losses['log_loss'][y == c].mean())} for c,name in enumerate(('home', 'draw', 'away'))}}, losses


def compare(tracks, intervals=False):
    ids = [r['match_id'] for r in tracks['regression_allcomp']]
    if any([r['match_id'] for r in values] != ids for values in tracks.values()):
        raise ValueError('Every direct-vs-regression comparison must retain identical targets')
    scores, losses = {}, {}
    for arm, values in tracks.items():
        scores[arm], losses[arm] = score_track(values)
    contrasts = [('extra_trees_allcomp', 'regression_allcomp'), ('extra_trees_performance', 'regression_performance'),
        ('logistic_allcomp', 'regression_allcomp'), ('logistic_performance', 'regression_performance'),
        ('extra_trees_allcomp', 'extra_trees_performance'), ('logistic_allcomp', 'logistic_performance'),
        ('extra_trees_allcomp', 'logistic_allcomp')]
    if 'clean_blend' in tracks:
        contrasts += [('extra_trees_allcomp', 'clean_blend'), ('logistic_allcomp', 'clean_blend')]
    columns, values = [], []
    for arm, reference in contrasts:
        for metric in losses[arm]:
            columns.append(f'{arm}_minus_{reference}:{metric}')
            values.append(losses[arm][metric]-losses[reference][metric])
    matrix = np.column_stack(values)
    return {'n': len(ids), 'scores': scores, 'difference_order': columns,
        'mean_loss_differences': dict(zip(columns, matrix.mean(axis=0).tolist())),
        'block_intervals': {str(w): moving_block_means([r['date'] for r in tracks['regression_allcomp']], matrix,
            block_weeks=w) for w in (2, 4, 8)} if intervals else {}, 'match_ids': ids}


def train_horizon(out, source, lead, rows):
    folder = out/f'lead_{lead}'
    (folder/'models').mkdir(parents=True)
    tracks = {a: [] for a in ARMS}
    regression = read(source/f'lead_{lead}'/'逐场完整预测.json')
    for new, old in (('regression_performance', 'fresh_performance'), ('regression_allcomp', 'fresh_allcomp'),
                     ('goal', 'goal'), ('xg_raw', 'xg_raw'), ('clean_blend', 'clean_blend')):
        tracks[new] = [{k: row[k] for k in (*CONTEXT_KEYS, 'probabilities')} for row in regression[old]]
    del regression
    parameters, windows = {}, []
    for period in ('2024-25', '2025-26'):
        targets = [r for r in rows if r['period'] == period]
        for cutoff in sorted({r['fit_before_date'] for r in targets}):
            started = time.perf_counter()
            history = [r for r in rows if r['date'] < cutoff]
            held = [r for r in targets if r['fit_before_date'] == cutoff]
            for arm, (kind, features) in ARMS.items():
                model = EarlyOutcomeLearner(kind).fit(
                    [{**r, 'match_features': r[features+'_features']} for r in history], date.fromisoformat(cutoff))
                key = f'{arm}:{cutoff}'
                parameters[key] = model.parameters() | {'pipeline_feature_set': features, 'horizon_days': lead}
                joblib.dump(model, folder/'models'/f'{arm}_{cutoff}.joblib', compress=3)
                predicted = model.predict_many([r[features+'_features'] for r in held],
                    [date.fromisoformat(r['forecast_before_date']) for r in held])
                for row, probabilities in zip(held, predicted, strict=True):
                    tracks[arm].append({k: row[k] for k in (*CONTEXT_KEYS, 'feature_before_date')}
                        | {'probabilities': list(probabilities), 'output_scope': 'one_x_two_only'})
            windows.append({'period': period, 'cutoff': cutoff, 'training_n': len(history), 'target_n': len(held),
                            'seconds': time.perf_counter()-started, 'finished_fits': len(parameters)})
            write(folder/'训练进度.json', {'windows': windows, 'fits': len(parameters)})
            print(json.dumps({'lead': lead, **windows[-1]}, ensure_ascii=False), flush=True)
        for arm in tracks:
            tracks[arm].sort(key=lambda r: (r['date'], r['match_id']))
        local = {a: [r for r in values if r['period'] == period] for a,values in tracks.items()
                 if a != 'clean_blend' or period == '2025-26'}
        if period == '2024-25':
            validation = compare(local)
            write(folder/'较早完整比较.json', validation)
            print(json.dumps({'lead': lead, 'validation_log_loss': {a:v['log_loss'] for a,v in validation['scores'].items()},
                              'selection': 'none; all four classifiers and both leads retained'}, ensure_ascii=False), flush=True)
        else:
            test = compare(local, intervals=True)
            by_league = {c: compare({a: [r for r in values if r['competition'] == c]
                                    for a,values in local.items()}) for c in COMPS}
    write(folder/'全部分类模型参数.json', parameters)
    write(folder/'逐场胜平负概率.json', tracks)
    summary = {'horizon_days': lead, 'new_classifier_fits': len(parameters), 'validation': validation,
        'test': test, 'by_league': by_league, 'windows': windows, 'adopted': False}
    write(folder/'直接分类结果汇总.json', summary)
    print(json.dumps({'lead': lead, 'fits': len(parameters), 'test_log_loss': {
        a:v['log_loss'] for a,v in test['scores'].items()}, 'test_differences': test['mean_loss_differences']}, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    now = datetime.now(timezone.utc)
    out = ROOT/'outputs'/now.strftime('直接胜平负分类研究_%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True)
    plan = {'created_at': now.isoformat(), 'source_study': str(source), 'user_approved': 'independent direct1X2 research',
        'horizons': [1, 3], 'arms': ARMS,
        'sample_unit': 'exactly one fixture row; home-perspective has both own and opponent complete feature roles',
        'features': 'same prepared fresh performance121/allcomp161 inputs; no outcomes, same-match stats or market inputs',
        'classifier_config': {'logistic': 'multinomial C1 lbfgs max1000 tol1e-8',
            'extra_trees': '120trees depth6 leaf30 log_loss .8features no bootstrap seed20261025'},
        'class_order': ['home', 'draw', 'away'], 'class_weight': 'none; do not force class frequencies',
        'weighting': '180day decay normalized to sum nfixtures; fixedC1 with l2 strength1/(C*n) in weighted-mean loss, not fixed effective regularization across sample counts',
        'tree_comparability': '30fixtures leaf vs60scoring sides in regression; different response/impurity and representation, not one causal switch',
        'chronology': 'sameP<=F<D; traininglabels date<P, feature histories each own strictlybeforeF; no random row splits',
        'train_initial': ['2022-23', '2023-24'], 'earlier_comparison': '2024-25', 'later_retrospective_comparison': '2025-26',
        'selection': 'none; all classifiers, inputs, leads reported; no later winner or weights selected',
        'primary': 'eachlead extra_trees_allcomp minus sameinput regression_allcomp 1X2 logloss',
        'secondary': ['logistic and performance controls', 'Brier/RPS/Top1', 'class calibration', 'earlierlearned blend', 'fiveleagues'],
        'budget': '4fixedclassifiers x2leads x22pooledcutoffs=176fits,zero hyperparametergrids',
        'zero_probability': 'explicit failure, no clipping, Laplace smoothing, renormalization or silent infinite-loss removal',
        'uncertainty': '2/4/8calendarweek pairedconditional intervals2000repeats; historicalresearch choices not included',
        'output_scope': '1X2 probabilities only; no invented score/total/handicap from3class probabilities',
        'sources': {'logistic_docs': 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html',
            'tree_docs': 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html'},
        'limits': ['alreadyviewed historicalseasons,not untouchedvalidation', 'datelevel availability,not exact24/72hours',
            'unknownsourcepublication/revisions', 'current source exceptions preserved; unknown perfection not claimed'],
        'lineups_used': False, 'current_results_reviewed': False, 'new_dependencies': False,
        'neural_networks_added': False, 'adopted': False}
    write(out/'分析前计划.json', plan)
    write(out/'覆盖与不可预测.json', read(source/'覆盖与不可预测.json'))
    print(json.dumps({'directory': str(out), 'source': str(source)}, ensure_ascii=False), flush=True)
    summaries = {}
    for lead in (1, 3):
        rows = load_rows(source, lead)
        summaries[str(lead)] = train_horizon(out, source, lead, rows)
        del rows
        gc.collect()
    write(out/'直接胜平负完整汇总.json', {'plan': plan, 'finished_at': datetime.now(timezone.utc).isoformat(),
        'horizons': summaries, 'new_classifier_fits': sum(s['new_classifier_fits'] for s in summaries.values()), 'adopted': False})
    print(json.dumps({'directory': str(out), 'fits': sum(s['new_classifier_fits'] for s in summaries.values()),
                      'adopted': False}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
