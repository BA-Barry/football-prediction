"""One fixed direct-1X2 pipeline with earlier player usage; no new tuning."""
from datetime import date, datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import time

import joblib

from football_model.player_usage_outcome import PlayerUsageOutcomeLearner as UsageOutcomeLearner
from run_direct_outcomes_20261025 import load_rows, score_track

ROOT = REPOSITORY_ROOT
OUT = ROOT/'outputs/历史球员使用输入_20261025'
SOURCE = ROOT/'outputs/固定发布日期信息刷新_20261025T071806561562Z'
OLD = ROOT/'outputs/直接胜平负分类研究_20261025T091329913808Z'
LEAD = 3


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    started = time.perf_counter()
    folder = OUT/'胜平负接入_继续求解'
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'models').mkdir(exist_ok=True)
    snapshots = {r['match_id']: r for r in read(OUT/'赛前使用特征.json')['rows'] if r['horizon_days'] == LEAD}
    new_names = sorted(next(iter(snapshots.values()))['match_features'])
    prepared = load_rows(SOURCE, LEAD)
    rows, joined = [], []
    for row in prepared:
        usage = snapshots.get(row['match_id'])
        extra = {key: None for key in new_names}
        if usage is not None:
            if any(usage[k] != row[k] for k in ('home', 'away', 'date', 'forecast_before_date')):
                raise ValueError('Player usage must share the old exact fixture and information cutoff')
            extra = usage['match_features']
            joined.append(row['match_id'])
        original = row['allcomp_features']
        if set(original) & set(extra):
            raise ValueError('New feature names collide with the original pipeline')
        rows.append({**row, 'match_features': {**original, **extra}})
    targets = [r for r in rows if r['period'] == '2024-25' and r['competition'] == 'E0']
    old_tracks = read(OLD/f'lead_{LEAD}'/'逐场胜平负概率.json')
    old_index = {r['match_id']: r for r in old_tracks['logistic_allcomp']}
    cutoffs = sorted({r['fit_before_date'] for r in targets})
    plan = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'question': 'Does the existing fixed 1X2 classifier improve after adding earlier player usage?',
        'algorithm': 'same logistic C1 objective and tolerance; no class weighting or statistical tuning',
        'numerical_repair': 'first 2024-08-01 fit reached old 1000-iteration limit before producing predictions; allow 4000 iterations once, retaining C1 and tolerance. Stop on another solver failure.',
        'initial_failed_attempt': str(OUT/'胜平负接入/失败.json'),
        'horizon_days': LEAD, 'evaluation_period': '2024-25', 'evaluation_competition': 'E0',
        'fit_cutoffs': cutoffs, 'planned_fits': len(cutoffs), 'target_matches': len(targets),
        'training_scope': 'same earlier five-league fixtures as saved classifier; usage observed only for EPL 2023/24 and 2024/25',
        'missing_usage': 'None plus existing training-only imputation and missing indicators; never zero exposure',
        'coverage_confounding': 'usage availability also identifies league/era; this tests the whole added-input pipeline, not a causal effect of continuity',
        'old_probability_source': str(OLD/f'lead_{LEAD}'/'逐场胜平负概率.json'),
        'old_features_source': str(SOURCE/f'features_lead{LEAD}.json.gz'),
        'new_features_source': str(OUT/'赛前使用特征.json'),
        'new_feature_names': new_names, 'joined_historical_feature_rows': len(joined),
        'historical_season_previously_examined': True, 'independent_confirmation_claimed': False,
        'current_results_requested': False, 'target_xi_read': False, 'hyperparameter_searches': 0,
        'stop_decision': 'complete this fixed integration comparison; no rescue tuning or automatic promotion',
    }
    write(folder/'本次接入设置.json', plan)
    predictions, windows, failures = [], [], []
    for cutoff in cutoffs:
        t0 = time.perf_counter()
        history = [r for r in rows if r['date'] < cutoff]
        held = [r for r in targets if r['fit_before_date'] == cutoff]
        try:
            model = UsageOutcomeLearner('logistic').fit(history, date.fromisoformat(cutoff))
            probabilities = model.predict_many([r['match_features'] for r in held],
                [date.fromisoformat(r['forecast_before_date']) for r in held])
        except ValueError as error:
            failures.append({'cutoff': cutoff, 'error': str(error), 'target_match_ids': [r['match_id'] for r in held]})
            write(folder/'失败.json', failures)
            raise
        joblib.dump(model, folder/'models'/f'logistic_usage_{cutoff}.joblib', compress=3)
        for row, p in zip(held, probabilities, strict=True):
            old = old_index[row['match_id']]
            if any(old[k] != row[k] for k in ('date', 'fit_before_date', 'forecast_before_date', 'actual_outcome')):
                raise ValueError('Old and augmented pipelines must share publication, parameter date and label')
            predictions.append({k: row[k] for k in ('match_id', 'date', 'home', 'away', 'competition',
                'period', 'actual_outcome', 'fit_before_date', 'forecast_before_date', 'horizon_days')}
                | {'probabilities': list(p), 'old_probabilities': old['probabilities'],
                   'usage_history_available': row['match_id'] in snapshots})
        windows.append({'cutoff': cutoff, 'training_matches': len(history), 'target_matches': len(held),
            'seconds': time.perf_counter()-t0, 'parameters': model.parameters()})
        write(folder/'逐场概率.json', {'rows': predictions})
        write(folder/'各次训练记录.json', {'windows': windows})
        write(folder/'执行进度.json', {'completed_fits': len(windows), 'planned_fits': len(cutoffs),
            'completed_predictions': len(predictions), 'failures': failures})
        print(json.dumps({'cutoff': cutoff, 'fit': len(windows), 'target_matches': len(held),
                          'seconds': windows[-1]['seconds']}, ensure_ascii=False), flush=True)
    predictions.sort(key=lambda r: (r['date'], r['match_id']))
    old_rows = [{**r, 'probabilities': r['old_probabilities']} for r in predictions]
    scores, losses = score_track(predictions)
    old_scores, old_losses = score_track(old_rows)
    differences = {name: float((losses[name]-old_losses[name]).mean()) for name in losses}
    result = {'n_unique_fixtures': len(predictions), 'new_fits': len(windows), 'failed_fits': len(failures),
        'initial_attempt_failed_fits': 1,
        'horizon_days': LEAD, 'augmented': scores, 'saved_original': old_scores,
        'paired_mean_loss_change_augmented_minus_old': differences,
        'by_actual_outcome': {key: {'augmented': scores['by_actual_class'][key],
                                  'old': old_scores['by_actual_class'][key]} for key in ('home', 'draw', 'away')},
        'seconds_including_input_and_writes': time.perf_counter()-started,
        'adopted': False, 'calibration_claimed': False,
        'interpretation': 'Previously examined historical season; first added-input integration, not new blind or cross-season confirmation. No causal player-impact claim.'}
    write(folder/'逐场概率.json', {'rows': predictions})
    write(folder/'各次训练记录.json', {'windows': windows})
    write(folder/'接入结果.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
