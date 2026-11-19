"""Independent arithmetic and chronology audit of saved q research; never fits."""
from collections import Counter
from datetime import date, datetime, timezone
import gzip
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import warnings

import joblib
import numpy as np

ROOT = REPOSITORY_ROOT
OUT = ROOT / 'outputs/条件主客方向研究_20261028'
RICH = ROOT / 'outputs/持续状态与丰富信息接入_20261028'
DRAW = ROOT / 'outputs/历史比赛模式与平局分支_20261027'
STATE = ['gas_own_attack_state', 'gas_own_defence_state',
         'gas_opponent_attack_state', 'gas_opponent_defence_state',
         'gas_league_base_log_rate', 'gas_home_log_rate_effect']
TRACKS = ['original', 'existing_draw', 'state_rich', 'conditional']
LABELS = ['H', 'D', 'A']
ERRORS = Counter()


def read(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    return json.loads(path.read_text(encoding='utf-8'))


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, name, tolerance=1e-12):
    difference = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    ERRORS[name] = max(ERRORS[name], difference)
    check(difference <= tolerance, f'{name}: {difference}')


def indexed(records):
    result = {r['match_id']: r for r in records}
    check(len(result) == len(records), 'Duplicate match IDs')
    return result


def losses(records, route):
    p = np.array([r['tracks'][route]['probabilities'] for r in records])
    y = np.array([r['actual_outcome'] for r in records])
    check(np.isfinite(p).all() and (p > 0).all(), 'Invalid HDA mass')
    close(p.sum(axis=1), 1., 'HDA_sum')
    chosen = p.argmax(axis=1)
    actual = p[np.arange(len(y)), y]
    non_draw = y != 1
    binary = -np.log(np.where(non_draw, p[:, 0] + p[:, 2], p[:, 1]))
    directional = np.zeros(len(y))
    directional[non_draw] = -np.log(actual[non_draw] / (p[non_draw, 0] + p[non_draw, 2]))
    full = -np.log(actual)
    close(binary + directional, full, 'loss_decomposition')
    totals = np.array([[r['tracks'][route]['forecast']['total_goals'][k]
                        for k in ['0', '1', '2', '3', '4+']] for r in records])
    actual_total = np.array([min(sum(r['actual_score']), 4) for r in records])
    truth = np.eye(3)[y]
    result = {
        'accuracy': (chosen == y).astype(float),
        'one_x_two': full,
        'brier': ((p - truth) ** 2).sum(axis=1),
        'draw_binary': binary,
        'conditional_direction_weighted': directional,
        'total_goals': -np.log(totals[np.arange(len(y)), actual_total]),
        'exact_score': np.array([r['tracks'][route]['exact_score_nll'] for r in records]),
        'rps': ((np.cumsum(p, axis=1)[:, :2] - np.cumsum(truth, axis=1)[:, :2]) ** 2).sum(axis=1) / 2,
    }
    return p, y, chosen, result


def main():
    inputs = read(RICH / '联合输入_待训练.json.gz')['rows']
    source = indexed(inputs)
    old = indexed(read(RICH / '接入逐场概率.json.gz')['records'])
    draws = indexed(read(DRAW / '逐场概率与纠错.json')['records'])
    draw_parameters = read(DRAW / '全部分类模型参数.json')
    parameters = read(OUT / '模型参数.json')
    records = read(OUT / '逐场概率与条件方向.json.gz')['records']
    targets = indexed(records)
    grids = indexed(read(OUT / '逐场统一比分网格.json.gz')['records'])
    reported = read(OUT / '完整结果.json')
    check(len(source) == 6967 and len(targets) == 3452, 'Unexpected cohort')
    check(set(targets) == set(old) == set(draws) == set(grids), 'Target coverage mismatch')
    check(len(parameters) == 22, 'Expected 22 parameter windows')
    check(Counter(r['period'] for r in records) == {'2024-25': 1726, '2025-26': 1726}, 'Season coverage mismatch')
    check(not reported['current_forecasts_replaced'] and reported['parameter_grids'] == 0, 'Research scope changed')
    for r in inputs:
        check(r['fit_before_date'] <= r['forecast_before_date'] < r['date'], 'P <= F < D violated')
        check(r['feature_before_date'] == r['forecast_before_date'], 'Feature publication mismatch')
        h, a = r['actual_score']
        check(r['actual_outcome'] == (0 if h > a else 1 if h == a else 2), 'Outcome/score mismatch')
        available = any(r['side_features'][0][name] is not None for name in STATE)
        if available:
            check(r['state_parameter_training_max_date'] < r['forecast_before_date'], 'State parameter chronology')
            check(r['state_history_max_date'] < r['forecast_before_date'], 'State observation chronology')

    windows = []
    replay_count = 0
    for cutoff, wrapper in parameters.items():
        p = wrapper['model']
        earlier = sorted([r for r in inputs if r['date'] < cutoff], key=lambda r: (r['date'], r['match_id']))
        conditional = [r for r in earlier if r['actual_outcome'] != 1]
        excluded = [r for r in earlier if r['actual_outcome'] == 1]
        ids = [r['match_id'] for r in conditional]
        all_ids = [r['match_id'] for r in earlier]
        check(p['reference_match_ids'] == ids, 'Conditional training ID mismatch')
        check(p['earlier_all_match_ids'] == all_ids, 'Earlier all ID mismatch')
        check(p['excluded_draw_match_ids'] == [r['match_id'] for r in excluded], 'Excluded draws mismatch')
        check(p['n_matches'] == len(conditional) and p['n_earlier_all_matches'] == len(earlier), 'Training count mismatch')
        check(p['classes'] == [0, 2] and p['before_date'] == cutoff and p['half_life_days'] == 180, 'Model target/configuration mismatch')
        check(p['training_min_date'] == conditional[0]['date'] and p['training_max_date'] == conditional[-1]['date'], 'Training date limits mismatch')
        active = [name for name in STATE if any(r['side_features'][0][name] is not None for r in conditional)]
        names = sorted(set(inputs[0]['side_features'][0]) - (set(STATE) - set(active)))
        check(wrapper['active_state_features'] == active and p['feature_names'] == names, 'Feature selection differs from conditional history')
        matrix = np.array([[np.nan if r['side_features'][0][name] is None else r['side_features'][0][name]
                            for name in names] for r in conditional])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            medians = np.nanmedian(matrix, axis=0)
        medians[np.isnan(medians)] = 0.
        close(p['imputation']['statistics'], medians, 'conditional_training_medians')
        check(p['imputation']['indicator_feature_indices'] == np.where(np.isnan(matrix).any(axis=0))[0].tolist(), 'Missing indicators differ from conditional training')
        draw_p = draw_parameters[f'extra_trees_pattern:{cutoff}']
        check(draw_p['reference_match_ids'] == all_ids, 'Reused d training IDs mismatch')
        check(p['configuration'] == draw_p['configuration'], 'Conditional tree differs from existing fixed preset')
        held = sorted([r for r in records if r['fit_before_date'] == cutoff], key=lambda r: (r['date'], r['match_id']))
        check(held and all(r['match_id'] not in set(all_ids) and cutoff <= r['forecast_before_date'] < r['date'] for r in held), 'Held targets overlap training or chronology')
        model = joblib.load(wrapper['model_file'])
        check(model.parameters() == p, 'Persisted model metadata differs from JSON')
        features = [{name: source[r['match_id']]['side_features'][0][name] for name in names} for r in held]
        q = model.predict_many(features, [date.fromisoformat(r['forecast_before_date']) for r in held])
        close(q, [r['conditional_home_probability'] for r in held], 'saved_q_replay')
        replay_count += len(held)
        windows.append({'cutoff': cutoff, 'earlier_all': len(earlier), 'conditional_non_draw': len(conditional),
                        'excluded_draws': len(excluded), 'training_max_date': p['training_max_date'],
                        'n_features': len(names), 'target_n': len(held)})

    context = ['date', 'period', 'competition', 'home', 'away', 'actual_score', 'actual_outcome',
               'fit_before_date', 'forecast_before_date', 'feature_before_date', 'horizon_days']
    for r in records:
        fid = r['match_id']
        s = source[fid]
        for name in context:
            check(r[name] == s[name] == old[fid][name] == draws[fid][name], f'Context mismatch {name}')
        check(r['tracks']['original'] == old[fid]['tracks']['original'], 'Original probabilities altered')
        check(r['tracks']['state_rich'] == old[fid]['tracks']['state_rich'], 'State reference altered')
        check(r['tracks']['existing_draw']['probabilities'] == draws[fid]['probabilities'], 'Draw reference altered')
        q, d = r['conditional_home_probability'], r['fixed_draw_probability']
        check(math.isfinite(q) and 0 < q < 1, 'Conditional q outside native probability range')
        check(d == draws[fid]['classifier_draw_probability'], 'd is not saved historical draw probability')
        hda = np.array([(1 - d) * q, d, (1 - d) * (1 - q)])
        track = r['tracks']['conditional']
        close(track['probabilities'], hda, 'q_d_composition')
        item = grids[fid]
        check(item['axis'] == ['home_goals', 'away_goals'], 'Grid axis mismatch')
        close(item['source_rates'], r['tracks']['state_rich']['rates'], 'source_rates')
        grid = np.array(item['probabilities'])
        check(np.isfinite(grid).all() and (grid >= 0).all(), 'Invalid score grid')
        close(grid.sum(), 1., 'grid_sum')
        hg, ag = np.indices(grid.shape)
        masks = [hg > ag, hg == ag, hg < ag]
        grid_hda = np.array([grid[m].sum() for m in masks])
        close(grid_hda, hda, 'grid_HDA')
        forecast = track['forecast']
        close([forecast['one_x_two'][k] for k in ['home', 'draw', 'away']], hda, 'forecast_HDA')
        total = hg + ag
        total_mass = [grid[total == k].sum() for k in range(4)] + [grid[total >= 4].sum()]
        close([forecast['total_goals'][k] for k in ['0', '1', '2', '3', '4+']], total_mass, 'grid_totals')
        close([forecast['expected_goals'][k] for k in ['home', 'away']], [(grid * hg).sum(), (grid * ag).sum()], 'grid_means')
        close(forecast['over_2_5'], grid[total >= 3].sum(), 'grid_over_2_5')
        close(forecast['both_score'], grid[(hg > 0) & (ag > 0)].sum(), 'grid_both_score')
        handicap = [grid[ag - hg >= 3].sum(), grid[ag - hg == 2].sum(), grid[ag - hg <= 1].sum()]
        close([forecast['away_minus_two_jincai'][k] for k in ['win', 'draw', 'lose']], handicap, 'grid_handicap')
        h, a = r['actual_score']
        close(track['exact_score_nll'], -math.log(grid[h, a]), 'grid_exact_score_loss')
        # Independent Poisson formula verifies that only regional H/D/A masses
        # changed, while conditional score shapes retain the saved v0.50 rates.
        lam_h, lam_a = item['source_rates']
        home = np.array([math.exp(-lam_h) * lam_h ** k / math.factorial(k) for k in range(grid.shape[0])])
        away = np.array([math.exp(-lam_a) * lam_a ** k / math.factorial(k) for k in range(grid.shape[1])])
        base = np.outer(home, away)
        base /= base.sum()
        for m, mass in zip(masks, hda):
            close(grid[m], base[m] * (mass / base[m].sum()), 'conditional_score_shape')

    periods = {}
    for period in ['2024-25', '2025-26']:
        cohort = [r for r in records if r['period'] == period]
        all_losses = {route: losses(cohort, route) for route in TRACKS}
        y = all_losses['conditional'][1]
        target_top = all_losses['conditional'][2]
        report = reported['by_period'][period]
        for route, (p, _, chosen, loss) in all_losses.items():
            saved = report['tracks'][route]
            check(saved['correct'] == int((chosen == y).sum()), 'Correct count mismatch')
            close(saved['accuracy'], (chosen == y).mean(), 'accuracy')
            for name, values in loss.items():
                if name != 'accuracy':
                    close(saved['mean_losses'][name], values.mean(), 'reported_losses')
        comparisons = {}
        newloss = all_losses['conditional'][3]
        for route in ['original', 'existing_draw', 'state_rich']:
            p, _, old_top, old_loss = all_losses[route]
            changes = target_top != old_top
            corrected = int(((old_top != y) & (target_top == y)).sum())
            damaged = int(((old_top == y) & (target_top != y)).sum())
            wrong_changed = int((changes & (old_top != y) & (target_top != y)).sum())
            saved = report['comparisons'][route]
            check((int(changes.sum()), corrected, damaged, wrong_changed) ==
                  (saved['changed'], saved['corrected'], saved['damaged'], saved['both_wrong_changed']), 'Change ledger mismatch')
            for name in saved['difference_order']:
                close(saved['new_minus_reference'][name], (newloss[name] - old_loss[name]).mean(), 'reported_deltas')
            comparisons[route] = {'changed': int(changes.sum()), 'corrected': corrected, 'damaged': damaged,
                                  'both_wrong_changed': wrong_changed,
                                  'transitions': dict(Counter(f'{LABELS[a]}->{LABELS[b]}' for a, b in zip(old_top[changes], target_top[changes])))}
        old_p = all_losses['existing_draw'][0]
        new_p = all_losses['conditional'][0]
        close(new_p[:, 1], old_p[:, 1], 'fixed_d')
        close(newloss['draw_binary'], all_losses['existing_draw'][3]['draw_binary'], 'fixed_d_draw_loss')
        close(newloss['one_x_two'] - all_losses['existing_draw'][3]['one_x_two'],
              newloss['conditional_direction_weighted'] - all_losses['existing_draw'][3]['conditional_direction_weighted'], 'direction_only_HDA_loss_change')
        old_q = old_p[:, 0] / (old_p[:, 0] + old_p[:, 2])
        new_q = np.array([r['conditional_home_probability'] for r in cohort])
        check(np.array_equal(new_q >= .5, new_p[:, 0] >= new_p[:, 2]), 'H/A order differs from q threshold')
        periods[period] = {
            'n': len(cohort), 'correct_by_route': {name: report['tracks'][name]['correct'] for name in TRACKS},
            'new_top_counts': dict(zip(LABELS, np.bincount(target_top, minlength=3).tolist())),
            'new_correct_by_actual': report['tracks']['conditional']['correct_by_actual_class'],
            'new_opposite_HA': report['tracks']['conditional']['opposite_home_away_top1'],
            'old_opposite_HA': report['tracks']['existing_draw']['opposite_home_away_top1'],
            'q': {'old_mean': float(old_q.mean()), 'new_mean': float(new_q.mean()),
                  'old_std': float(old_q.std()), 'new_std': float(new_q.std()),
                  'new_range': [float(new_q.min()), float(new_q.max())],
                  'mean_absolute_change': float(np.abs(new_q - old_q).mean()),
                  'correlation': float(np.corrcoef(old_q, new_q)[0, 1]),
                  'HA_order_flips': int(((old_q >= .5) != (new_q >= .5)).sum())},
            'comparisons': comparisons,
            'new_minus_existing_draw': report['comparisons']['existing_draw']['new_minus_reference'],
        }

    result = {'audited_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
              'input_n': len(inputs), 'target_n': len(records), 'windows_n': len(windows),
              'saved_model_inference_replayed_n': replay_count, 'real_history_fits': 0,
              'parameter_grid_runs': 0, 'windows': windows,
              'max_absolute_numerical_discrepancies': dict(ERRORS), 'by_period': periods,
              'scope': 'Independent saved-output chronology, arithmetic and model inference audit; no fitting or adoption',
              'interpretation': [
                  'q uses only earlier non-draw training rows; targets retain all actual draws and non-draws.',
                  'Against existing_draw, the draw probability is exactly fixed and HDA log-loss changes come only from conditional H/A direction.',
                  'New score grids retain state_rich conditional score shapes; exact-score and total-goal differences against existing_draw also contain this source-shape difference.',
                  'Fixed tree settings operate on fewer non-draw training observations; the task and effective leaf sample scale both change.',
                  'Historical forward windows prevent target-in-training here; reused development seasons are not a new blind or prospective accuracy estimate.',
              ]}
    (OUT / '独立核对.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'windows': len(windows), 'targets': len(records),
                      'replayed': replay_count, 'maximum_discrepancy': max(ERRORS.values()),
                      'by_period': periods}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
