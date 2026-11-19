"""Independent saved-feature chronology and analytical broad-model audit."""
import argparse
from bisect import bisect_left
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import joblib
import numpy as np
from scipy.stats import poisson, skellam

from football_model.native_experiment import load_seasons

ROOT = REPOSITORY_ROOT
OUT = ROOT/'outputs/多信息多算法前推_20261025T060704977450Z'
ARMS = ('glm_basic', 'glm_rich', 'hgb_basic', 'hgb_rich', 'extra_trees_rich')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def metadata_audit():
    payload = read(OUT/'历史特征样本.json')
    rows = payload['rows']
    data = read(OUT/'数据联结.json')
    independent, _ = load_seasons({k: Path(v) for k, v in data['independent_directories'].items()})
    source = {r.match_id: r for r in independent}
    excluded = {mid for e in data['joins'].values() for mid in e['excluded_independent_ids']}
    excluded |= {r['match_id'] for e in data['joins'].values() for r in e['excluded_special_matches']}
    usable = [r for r in independent if r.match_id not in excluded]
    assert len(usable) == data['source_matches'] == 10711
    index = {}
    for r in usable:
        for team in (r.home, r.away):
            index.setdefault((r.competition, team), []).append(r)
    for games in index.values():
        games.sort(key=lambda r: (r.match_date, r.match_id))
    date_index = {k: [r.match_date.isoformat() for r in v] for k, v in index.items()}
    early = read(ROOT/'outputs/进球xG更早年份复现_20261024T180238377843Z/更早年份复现结果.json')
    recent = read(ROOT/'outputs/进球xG共同训练记录_20261024T174247963857Z/共同训练记录融合结果.json')
    cross, epl = (read(p) for p in recent['source_results'][:2])
    goal = {r['match_id']: r for r in early['predictions']['goal']+recent['common_goal_predictions']}
    xg = {r['match_id']: r for r in early['predictions']['xg']+cross['new_predictions']+epl['new_predictions']}
    assert len(rows) == len(goal) == len(xg) == 6990
    assert {r['match_id'] for r in rows} == set(goal) == set(xg)
    missing = Counter()
    used_reference_checks = 0
    for row in rows:
        mid = row['match_id']
        fd, g, x = source[mid], goal[mid], xg[mid]
        for field in ('date', 'home', 'away', 'competition', 'fit_before_date', 'period', 'actual_score'):
            assert row[field] == g[field] == x[field], (mid, field)
        assert row['date'] == fd.match_date.isoformat()
        assert row['actual_score'] == list(fd.full_time)
        h, a = fd.full_time
        assert row['actual_outcome'] == (0 if h > a else 2 if h < a else 1)
        assert row['goal_rates'] == g['rates'] and row['xg_rates'] == x['rates']
        context = row['feature_history']
        assert context['cutoff'] == row['fit_before_date'] <= row['date']
        assert context['match_date'] == row['date']
        counts = []
        for side, team in (('home', row['home']), ('away', row['away'])):
            meta = context['history'][side]
            key = (row['competition'], team)
            stop = bisect_left(date_index.get(key, []), row['fit_before_date'])
            history = index.get(key, [])[:stop]
            assert meta['team'] == team and meta['competition'] == row['competition']
            assert meta['history_count'] == len(history)
            assert meta['used_match_ids'] == [r.match_id for r in history[-15:]]
            for window in (5, 15):
                assert meta['window_match_ids'][str(window)] == [r.match_id for r in history[-window:]]
            assert all(r.match_date.isoformat() < row['fit_before_date'] for r in history[-15:])
            if history:
                assert meta['latest_used_date'] == history[-1].match_date.isoformat()
            counts.append(len(history))
            used_reference_checks += min(len(history), 15)
        available = min(counts) >= 5
        if not available:
            missing[row['period']] += 1
        for i in (0, 1):
            basic, rich = row['basic_features'][i], row['rich_features'][i]
            assert basic['is_home'] == rich['is_home'] == float(i == 0)
            assert all(rich[k] == v for k, v in basic.items())
            assert rich['history_features_available'] == float(available)
            assert basic['log_goal_mean_for'] == math.log(g['rates'][i])
            assert basic['log_goal_mean_against'] == math.log(g['rates'][1-i])
            assert basic['log_xg_mean_for'] == math.log(x['rates'][i])
            assert basic['log_xg_mean_against'] == math.log(x['rates'][1-i])
            assert all(v is not None and math.isfinite(v) for v in basic.values())
            assert all(v is None or math.isfinite(v) for v in rich.values())
            assert not any('blend' in k for k in rich)
    summary = {'measured_at': datetime.now(timezone.utc).isoformat(), 'feature_rows': len(rows),
        'source_matches': len(usable), 'source_exclusions': len(excluded),
        'historical_feature_reference_rows_checked': used_reference_checks,
        'missing_histories_retained': sum(missing.values()), 'missing_by_period': dict(missing),
        'basic_feature_counts': sorted({len(r['basic_features'][0]) for r in rows}),
        'rich_feature_counts': sorted({len(r['rich_features'][0]) for r in rows}),
        'home_flag_preserved_when_missing': True,
        'new_fits': 0, 'network_requests': 0, 'current_results_reviewed': False, 'errors': 0}
    write(OUT/'独立历史特征核对.json', summary)
    return rows, summary


def analytic(rows):
    mu = np.asarray([r['rates'] for r in rows])
    y = np.asarray([r['actual_score'] for r in rows])
    assert mu.shape == y.shape == (len(rows), 2)
    assert np.isfinite(mu).all() and (mu > 0).all()
    ix = np.arange(len(rows))
    # Use the symmetric lower tail rather than 1-CDF, avoiding cancellation
    # for the intentionally retained extreme GLM predictions.
    one = np.column_stack([skellam.cdf(-1, mu[:, 1], mu[:, 0]), skellam.pmf(0, mu[:, 0], mu[:, 1]), skellam.cdf(-1, mu[:, 0], mu[:, 1])])
    total = np.column_stack([poisson.pmf(i, mu.sum(axis=1)) for i in range(4)]+[poisson.sf(3, mu.sum(axis=1))])
    hc = np.column_stack([skellam.cdf(-3, mu[:, 0], mu[:, 1]), skellam.pmf(2, mu[:, 1], mu[:, 0]), skellam.cdf(1, mu[:, 1], mu[:, 0])])
    saved = np.asarray([r['probabilities'] for r in rows])
    ts = np.asarray([[r['forecast']['total_goals'][k] for k in ('0', '1', '2', '3', '4+')] for r in rows])
    hs = np.asarray([[r['forecast']['away_minus_two_jincai'][k] for k in ('win', 'draw', 'lose')] for r in rows])
    probability_error = max(np.max(np.abs(one-saved)), np.max(np.abs(total-ts)), np.max(np.abs(hc-hs)))
    assert probability_error < 1e-10
    outcome = np.where(y[:, 0] > y[:, 1], 0, np.where(y[:, 0] < y[:, 1], 2, 1))
    assert [r['actual_outcome'] for r in rows] == outcome.tolist()
    margin = y[:, 1]-y[:, 0]-2
    hc_outcome = np.where(margin > 0, 0, np.where(margin == 0, 1, 2))
    losses = {'exact_score': -poisson.logpmf(y, mu).sum(axis=1),
        'one_x_two': -np.log(one[ix, outcome]),
        'total_goals_0_1_2_3_4plus': -np.log(total[ix, np.minimum(y.sum(axis=1), 4)]),
        'away_minus_two_three_way': -np.log(hc[ix, hc_outcome])}
    # Equivalent numerical formulas can break exact home/away ties in opposite
    # directions. Verify non-tied predictions analytically and the published
    # argmax convention using its actual saved probabilities.
    sorted_one = np.sort(one, axis=1)
    tied = sorted_one[:, -1]-sorted_one[:, -2] <= 1e-10
    assert np.all((one.argmax(axis=1) == saved.argmax(axis=1)) | tied)
    return losses, float(probability_error), float((saved.argmax(axis=1) == outcome).mean()), int(tied.sum())


def final_audit(rows, feature_summary):
    summary = read(OUT/'多方案结果汇总.json')
    tracks = read(OUT/'逐场预测.json')
    parameters = read(OUT/'所有模型参数.json')
    original = {r['match_id']: r for r in rows}
    parameter_references = 0
    for key, p in parameters.items():
        arm, cutoff = key.split(':')
        expected = [r['match_id'] for r in rows if r['date'] < cutoff]
        assert p['reference_match_ids'] == expected
        assert p['n_matches'] == len(expected) and p['n_scoring_sides'] == 2*len(expected)
        assert p['training_max_date'] < cutoff == p['before_date']
        assert p['configuration']['random_state'] == 20261025 if p['kind'] != 'glm' else True
        if p['kind'] == 'hgb':
            assert p['configuration']['early_stopping'] is False
        parameter_references += len(expected)
    results = {}
    for period, name in (('2024-25', 'validation'), ('2025-26', 'test')):
        scores, arm_losses = {}, {}
        for arm, all_rows in tracks.items():
            local = [r for r in all_rows if r['period'] == period]
            if arm == 'old_blend' and period == '2024-25':
                assert not local
                continue
            assert [r['match_id'] for r in local] == summary[name]['match_ids']
            for r in local:
                old = original[r['match_id']]
                for field in ('date', 'home', 'away', 'competition', 'fit_before_date', 'actual_score', 'actual_outcome'):
                    assert r[field] == old[field]
            losses, error, accuracy, ties = analytic(local)
            arm_losses[arm] = losses
            for domain, values in losses.items():
                assert abs(float(values.mean())-summary[name]['scores'][arm][domain]['log_loss']) < 1e-10
            assert abs(accuracy-summary[name]['scores'][arm]['one_x_two']['accuracy']) < 1e-14
            scores[arm] = {'analytic_mean_losses': {k: float(v.mean()) for k, v in losses.items()},
                          'accuracy': accuracy, 'maximum_probability_difference': error,
                          'numerically_tied_argmax_fixtures': ties}
        for contrast, saved in summary[name]['mean_loss_differences'].items():
            arms, domain = contrast.split(':')
            arm, reference = arms.split('_minus_')
            assert abs(float((arm_losses[arm][domain]-arm_losses[reference][domain]).mean())-saved) < 1e-10
        results[name] = scores
    selected = min(ARMS, key=lambda a: (results['validation'][a]['analytic_mean_losses']['exact_score'], a))
    assert summary['selected'] == read(OUT/'较早选择.json')['selected'] == selected
    assert (OUT/'较早选择.json').stat().st_mtime_ns < min(
        p.stat().st_mtime_ns for p in (OUT/'models').glob('*_2025-*.joblib')
        if p.stem.rsplit('_', 1)[-1] >= '2025-08-01')
    replays = []
    cutoffs = sorted({p['before_date'] for p in parameters.values()})
    for cutoff in (cutoffs[0], cutoffs[-1]):
        batch = [r for r in rows if r['fit_before_date'] == cutoff][:3]
        for arm in ARMS:
            model = joblib.load(OUT/'models'/f'{arm}_{cutoff}.joblib')
            features = model.feature_set
            predicted = model.predict_many([r[f'{features}_features'] for r in batch],
                [datetime.fromisoformat(r['date']).date() for r in batch])
            saved = {r['match_id']: r for r in tracks[arm]}
            expected = [saved[r['match_id']]['rates'] for r in batch]
            error = float(np.max(np.abs(np.asarray(predicted)-expected)))
            assert error < 1e-10
            replays.append({'arm': arm, 'cutoff': cutoff, 'n': len(batch), 'maximum_rate_error': error})
    audit = {'measured_at': datetime.now(timezone.utc).isoformat(),
        'metadata_audit': feature_summary, 'selected_recomputed_from_validation_only': selected,
        'model_parameter_count': len(parameters), 'training_reference_rows_checked': parameter_references,
        'analytic_scores': results, 'first_last_model_replays': replays,
        'new_fits': 0, 'network_requests': 0, 'current_results_reviewed': False, 'errors': 0}
    write(OUT/'独立模型与评分核对.json', audit)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    samples, evidence = metadata_audit()
    result = final_audit(samples, evidence) if args.full else evidence
    print(json.dumps(result, ensure_ascii=False))
