"""Zero-fit complementary-error and apparent/held performance descriptions.

Training probabilities are intentionally obtained directly from saved
estimators. They are apparent diagnostics, never pre-match predictions.
"""
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

import joblib
import numpy as np

from football_model.evaluation import probability_scores


ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OUT = ROOT / 'outputs/多路评级联合接入_20261026'


def metrics(probabilities, outcomes):
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    return probability_scores(p, y) | {
        'correct': int((p.argmax(axis=1) == y).sum()),
        'mean_top_probability': float(p.max(axis=1).mean()),
        'maximum_raw_row_sum_deviation': float(np.abs(p.sum(axis=1)-1).max()),
    }


def error_overlap(old, new):
    if [r['match_id'] for r in old] != [r['match_id'] for r in new]:
        raise ValueError('Compare identical complete ordered fixtures')
    if any(a['actual_outcome'] != b['actual_outcome'] for a,b in zip(old,new,strict=True)):
        raise ValueError('Same fixture outcomes must agree')
    p = np.asarray([r['probabilities'] for r in old], dtype=float)
    q = np.asarray([r['probabilities'] for r in new], dtype=float)
    y = np.asarray([r['actual_outcome'] for r in old], dtype=int)
    old_top, new_top = p.argmax(axis=1), q.argmax(axis=1)
    a, b = old_top == y, new_top == y
    return {
        'n': len(y), 'old_correct': int(a.sum()), 'new_correct': int(b.sum()),
        'both_correct': int((a & b).sum()), 'new_only_correct': int((~a & b).sum()),
        'old_only_correct': int((a & ~b).sum()), 'both_wrong': int((~a & ~b).sum()),
        'top1_disagreement': int((old_top != new_top).sum()),
        'both_wrong_top1_disagreement': int((~a & ~b & (old_top != new_top)).sum()),
        'interpretation': 'Descriptive overlap only; outcomes cannot identify the right member before kickoff. No oracle selection is reported as achievable accuracy.',
    }


def main():
    predictions_path = OUT / '逐场完整胜平负预测.json'
    predictions = json.loads(predictions_path.read_text(encoding='utf-8'))
    with gzip.open(OUT/'联合赛前输入.json.gz', 'rt', encoding='utf-8') as handle:
        rows = json.load(handle)['rows']
    by_id = {r['match_id']:r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('Prepared fixture IDs must be unique')
    overlap = {}
    for season in ('2024-25','2025-26'):
        overlap[season] = {}
        new = [r for r in predictions['rich_joint'] if r['period'] == season]
        for reference in ('saved_performance','rating_core'):
            old = [r for r in predictions[reference] if r['period'] == season]
            overlap[season]['rich_joint_vs_'+reference] = error_overlap(old,new)
    apparent = []
    for cutoff in ('2024-08-01','2026-05-08'):
        for arm in ('rating_core','rich_joint'):
            model_path = OUT/'models'/f'{arm}_{cutoff}.joblib'
            model = joblib.load(model_path)
            parameters = model.parameters()
            ids = parameters['reference_match_ids']
            train = [by_id[identifier] for identifier in ids]
            if (len(ids) != parameters['n_matches']
                    or any(r['date'] >= cutoff for r in train)
                    or parameters['feature_key'] != arm+'_features'):
                raise ValueError('Saved estimator training design must match its prepared history')
            x = model._matrix([r[parameters['feature_key']] for r in train], model.feature_names)
            # Deliberately not the dated predict API: these are fitted training
            # rows and have earlier publication dates. No fit call occurs.
            train_p = np.asarray(model.estimator.predict_proba(x), dtype=float)
            train_y = [r['actual_outcome'] for r in train]
            train_metrics = metrics(train_p, train_y)
            held = [r for r in predictions[arm] if r['fit_before_date'] == cutoff]
            held_metrics = metrics([r['probabilities'] for r in held],
                                   [r['actual_outcome'] for r in held])
            apparent.append({
                'arm': arm, 'parameter_cutoff': cutoff, 'model_path': str(model_path),
                'training_date_min': parameters['training_min_date'],
                'training_date_max': parameters['training_max_date'],
                'training': train_metrics,
                'same_parameter_date_held_predictions': held_metrics,
                'held_date_min': min(r['date'] for r in held),
                'held_date_max': max(r['date'] for r in held),
                'held_minus_training': {key: held_metrics[key]-train_metrics[key]
                    for key in ('log_loss','accuracy','mean_top_probability')},
                'training_probability_origin': 'saved estimator.predict_proba on the saved reference training features via _matrix; apparent fit only, not legal pre-match predictions',
            })
    result = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'zero-fit descriptive reuse of saved candidate predictions and four saved estimators',
        'source_predictions': str(predictions_path), 'new_fits': 0,
        'current_results_requested': False, 'current_predictions_changed': False,
        'by_season_error_overlap': overlap,
        'training_vs_same_parameter_held': apparent,
        'limitations': [
            'Training metrics are optimistic apparent performance; these are not prospective forecasts.',
            'Training and held cohorts differ in dates, support and difficulty; a gap describes generalization but cannot identify its cause.',
            'These four examples do not isolate any variable, establish all-window behavior, or invalidate XGBoost, rating representations or joint information.',
            'Error complementarity does not establish that the correct member can be selected before a match; no hindsight oracle accuracy is a realizable forecast.',
            'Both historical seasons have already been repeatedly examined, so held chronology is not an untouched research test.',
        ],
    }
    (OUT/'互补性与训练内外描述.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'overlap':overlap,'training_vs_held':[
        {'arm':r['arm'],'cutoff':r['parameter_cutoff'],
         'training':r['training'],'held':r['same_parameter_date_held_predictions']}
        for r in apparent]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
