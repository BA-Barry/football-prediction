"""Reuse raw persistent states in the unchanged earlier-information goal learner.

The default inspect mode loads one existing model and checks its genuine
no-new-information window. Run mode performs the separately discussed complete
fixed integration. Neither mode downloads data or changes current forecasts.
"""
import argparse
from collections import Counter
import csv
from datetime import date, datetime, timezone
import gzip
import json
import math
from pathlib import Path
import time
import traceback

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from football_model.early_goal_learners import EarlyGoalLearner
from football_model.forecast_diagnostics import moving_block_means
from football_model.persistent_rich_integration import plan_windows, window_rows
from football_model.probability import score_grid, summarize


ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OUT = ROOT / 'outputs/持续状态与丰富信息接入_20261028'
INPUT = OUT / '联合输入_待训练.json.gz'
SOURCE = ROOT / 'outputs/固定发布日期信息刷新_20261025T071806561562Z/lead_3'
PERIODS = ('2024-25', '2025-26')
COMPETITIONS = ('E0', 'D1', 'SP1', 'I1', 'F1')
TRACKS = ('original', 'state_rich')
LABELS = ('主胜', '平局', '客胜')
CONTEXT = ('match_id', 'date', 'period', 'competition', 'home', 'away', 'fit_before_date',
           'forecast_before_date', 'horizon_days', 'actual_score', 'actual_outcome')
PARAMETER_FIELDS = ('reference_match_ids', 'before_date', 'training_min_date', 'training_max_date',
                    'n_matches', 'n_scoring_sides', 'configuration', 'half_life_days')


def read(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def zipped(path, value):
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def load_inputs():
    rows = read(INPUT)['rows']
    source_rows = read(SOURCE / '逐场完整预测.json')['fresh_performance']
    originals = {r['match_id']: r for r in source_rows}
    if len(originals) != len(source_rows) or len(originals) != 3452:
        raise ValueError('Original saved prediction cohort is not 3452 unique fixtures')
    targets = [r for r in rows if r['period'] in PERIODS]
    if len(targets) != 3452 or {r['match_id'] for r in targets} != set(originals):
        raise ValueError('Keep both complete original cohorts without deleting unavailable states')
    for r in targets:
        old = originals[r['match_id']]
        if any(r[k] != old[k] for k in CONTEXT):
            raise ValueError('Prepared target changed original identity, score or timing')
    return rows, originals, read(SOURCE / '全部模型参数.json')


def verify_window(window, original_parameters):
    expected = original_parameters['fresh_performance:' + window['cutoff']]
    if expected['reference_match_ids'] != window['training_ids']:
        raise ValueError('Window changed the original earlier training fixture IDs/order')
    if expected['before_date'] != window['cutoff']:
        raise ValueError('Window changed the original parameter date')
    return expected


def verify_model(model, window, expected):
    parameters = model.parameters()
    for key in PARAMETER_FIELDS:
        if parameters[key] != expected[key]:
            raise ValueError(f'Original fit scope or configuration differs: {key}')
    if parameters['feature_names'] != window['model_feature_names']:
        raise ValueError('Fitted columns differ from the training-available feature plan')
    return parameters


def fixture_grid(rates, actual_score):
    grid = score_grid(*rates, rho=0)
    h, a = actual_score
    if h >= grid.grid.shape[0] or a >= grid.grid.shape[1]:
        raise ValueError('Actual score lies outside the declared supported grid')
    actual_mass = float(grid.grid[h, a])
    if actual_mass <= 0:
        raise FloatingPointError('Full-score mass is zero; do not silently clip loss')
    probabilities = list(map(float, grid.home_draw_away))
    forecast = summarize(grid)
    if not np.allclose(probabilities, list(forecast['one_x_two'].values()), rtol=0, atol=1e-12):
        raise ValueError('Forecast H/D/A and source grid differ')
    return grid, {'rates': list(map(float, rates)), 'probabilities': probabilities,
                  'forecast': forecast, 'exact_score_nll': -math.log(actual_mass)}


def inspect_first_window(rows, originals, original_parameters, windows):
    """Measure only the genuinely all-missing first window using a saved model."""
    first = windows[0]
    if not first['reuse_original'] or first['active_state_features']:
        raise ValueError('The first window unexpectedly has learned state information')
    expected = verify_window(first, original_parameters)
    history, held = window_rows(rows, first)
    path = SOURCE / 'models' / f"fresh_performance_{first['cutoff']}.joblib"
    model = joblib.load(path)
    parameters = verify_model(model, first, expected)
    rates = model.predict_many([r['side_features'] for r in held],
                               [date.fromisoformat(r['date']) for r in held])
    errors = {'goal_rates': 0., 'hda': 0., 'grid': 0., 'total_goals': 0.}
    for row, predicted in zip(held, rates, strict=True):
        saved = originals[row['match_id']]
        grid, output = fixture_grid(predicted, row['actual_score'])
        old_grid, _ = fixture_grid(saved['rates'], row['actual_score'])
        errors['goal_rates'] = max(errors['goal_rates'], float(np.max(np.abs(np.asarray(predicted)-saved['rates']))))
        errors['hda'] = max(errors['hda'], float(np.max(np.abs(np.asarray(output['probabilities'])-saved['probabilities']))))
        errors['grid'] = max(errors['grid'], float(np.max(np.abs(grid.grid-old_grid.grid))))
        errors['total_goals'] = max(errors['total_goals'], max(abs(output['forecast']['total_goals'][key]-value)
            for key, value in saved['forecast']['total_goals'].items()))
    if max(errors.values()) > 1e-12:
        raise ValueError('First no-new-information window failed unchanged-model reproduction')
    result = {'measured_at_utc': datetime.now(timezone.utc).isoformat(),
        'mode': 'inspect', 'cutoff': first['cutoff'], 'training_matches': len(history),
        'target_matches': len(held), 'active_new_state_columns': first['active_state_features'],
        'inactive_new_state_columns': first['inactive_state_features'],
        'model_feature_count': len(parameters['feature_names']), 'saved_model': str(path),
        'max_absolute_reproduction_errors': errors, 'new_model_fits': 0,
        'new_prediction_versions': 0, 'all_other_model_windows_replayed': False,
        'interpretation': 'actual existing-model replay after pruning wholly unknown new columns; no evidence of predictive improvement',
        'current_results_queries': 0, 'current_forecasts_replaced': False}
    write(OUT / '首窗复用实测.json', result)
    return result


def loss_arrays(rows, track):
    p = np.asarray([r['tracks'][track]['probabilities'] for r in rows], dtype=float)
    y = np.asarray([r['actual_outcome'] for r in rows], dtype=int)
    if (p.shape != (len(rows), 3) or not np.isfinite(p).all() or np.any(p <= 0)
            or not np.allclose(p.sum(axis=1), 1., rtol=0, atol=1e-10)):
        raise FloatingPointError('Need unchanged finite positive H/D/A probabilities; no loss clipping')
    selected = p.argmax(axis=1)
    is_draw = y == 1
    nondraw_mass = p[:, 0] + p[:, 2]
    binary_draw_loss = -np.log(np.where(is_draw, p[:, 1], nondraw_mass))
    actual_mass = p[np.arange(len(y)), y]
    hda_loss = -np.log(actual_mass)
    direction_contribution = np.zeros(len(rows))
    direction_contribution[~is_draw] = -np.log(actual_mass[~is_draw] / nondraw_mass[~is_draw])
    if not np.allclose(binary_draw_loss + direction_contribution, hda_loss, rtol=0, atol=2e-14):
        raise FloatingPointError('Exact H/D/A loss decomposition failed')
    totals = np.asarray([[r['tracks'][track]['forecast']['total_goals'][key]
                          for key in ('0','1','2','3','4+')] for r in rows], dtype=float)
    total_y = np.asarray([min(sum(r['actual_score']), 4) for r in rows], dtype=int)
    total_mass = totals[np.arange(len(rows)), total_y]
    if np.any(total_mass <= 0) or not np.isfinite(totals).all():
        raise FloatingPointError('Need positive total-goal mass; no loss clipping')
    return p, y, selected, {
        'accuracy': (selected == y).astype(float),
        'one_x_two': hda_loss,
        'brier': np.square(p-np.eye(3)[y]).sum(axis=1),
        'draw_binary': binary_draw_loss,
        'conditional_direction_weighted': direction_contribution,
        'total_goals': -np.log(total_mass),
        'exact_score': np.asarray([r['tracks'][track]['exact_score_nll'] for r in rows]),
    }


def metrics(rows, track):
    p, y, selected, losses = loss_arrays(rows, track)
    draw = y == 1
    chosen_draw = selected == 1
    conditional = np.where(p[:,0] >= p[:,2], 0, 2)
    confusion = np.zeros((3,3), dtype=int)
    np.add.at(confusion, (y, selected), 1)
    return {'n': len(rows), 'correct': int((selected == y).sum()),
        'accuracy': float(losses['accuracy'].mean()),
        'mean_losses': {key: float(value.mean()) for key,value in losses.items() if key != 'accuracy'},
        'mean_probabilities': p.mean(axis=0).tolist(),
        'actual_frequencies': (np.bincount(y, minlength=3)/len(y)).tolist(),
        'draw_auc': float(roc_auc_score(draw, p[:,1])) if len(set(draw)) > 1 else None,
        'draw_selected': int(chosen_draw.sum()), 'draw_selected_correct': int((chosen_draw & draw).sum()),
        'conditional_home_away_n': int((~draw).sum()),
        'conditional_home_away_correct': int(((conditional == y) & ~draw).sum()),
        'conditional_home_away_accuracy': float((conditional[~draw] == y[~draw]).mean()) if (~draw).any() else None,
        'conditional_scope': 'diagnostic on actual non-draws; these fixtures cannot be selected before kickoff',
        'opposite_home_away_top1': int(((y != 1) & (selected != 1) & (selected != y)).sum()),
        'non_draw_actual_predicted_draw': int(((y != 1) & chosen_draw).sum()),
        'confusion_actual_rows_predicted_columns': confusion.tolist(),
        'class_order': ['home','draw','away'], 'loss_clipping_applied': False,
        'correct_by_actual_class': [int(((selected == y) & (y == i)).sum()) for i in range(3)]}


def evaluate(rows, *, intervals):
    _, y, old_choice, old_loss = loss_arrays(rows, 'original')
    _, _, new_choice, new_loss = loss_arrays(rows, 'state_rich')
    columns = list(old_loss)
    differences = np.column_stack([new_loss[key]-old_loss[key] for key in columns])
    old_correct, new_correct = old_choice == y, new_choice == y
    change = old_choice != new_choice
    return {'n': len(rows), 'tracks': {name: metrics(rows, name) for name in TRACKS},
        'difference_columns': columns,
        'new_minus_original': dict(zip(columns, differences.mean(axis=0).tolist())),
        'four_week_conditional_intervals': moving_block_means(
            [r['date'] for r in rows], differences, block_weeks=4) if intervals else None,
        'changes': {'changed': int(change.sum()),
                    'corrected': int((~old_correct & new_correct).sum()),
                    'damaged': int((old_correct & ~new_correct).sum()),
                    'both_wrong_changed': int((change & ~old_correct & ~new_correct).sum()),
                    'both_correct': int((old_correct & new_correct).sum()),
                    'transitions': dict(Counter(f'{LABELS[a]}->{LABELS[b]}' for a,b in zip(old_choice,new_choice))),
                    'by_actual_class': {label: {'corrected': int((~old_correct & new_correct & (y == i)).sum()),
                        'damaged': int((old_correct & ~new_correct & (y == i)).sum())}
                        for i,label in enumerate(LABELS)}}}


def export_selection_changes(records):
    """Keep each changed decision visible with separate H/D/A probabilities."""
    path = OUT / '接入全部改向比赛.csv'
    fields = ['match_id', 'date', 'period', 'competition', 'home', 'away',
              'fit_before_date', 'forecast_before_date', 'hg', 'ag', 'actual_result',
              'original_home_probability', 'original_draw_probability', 'original_away_probability',
              'new_home_probability', 'new_draw_probability', 'new_away_probability',
              'old_top1', 'new_top1', 'change_type', 'state_input_available',
              'active_state_feature_count', 'model_reuse_kind']
    counts = Counter()
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            counts[row['change_type']] += 1
            if row['change_type'] == 'unchanged':
                continue
            record = {key: row[key] for key in ('match_id', 'date', 'period', 'competition',
                'home', 'away', 'fit_before_date', 'forecast_before_date', 'change_type',
                'state_input_available', 'model_reuse_kind')}
            record.update({'hg': row['actual_score'][0], 'ag': row['actual_score'][1],
                'actual_result': LABELS[row['actual_outcome']], 'old_top1': LABELS[row['old_top1']],
                'new_top1': LABELS[row['new_top1']],
                'active_state_feature_count': len(row['active_state_features'])})
            for prefix, track in (('original', 'original'), ('new', 'state_rich')):
                for outcome, probability in zip(('home', 'draw', 'away'),
                                                row['tracks'][track]['probabilities'], strict=True):
                    record[f'{prefix}_{outcome}_probability'] = probability
            writer.writerow(record)
    return {'path': str(path), 'changed_records': len(records)-counts['unchanged'],
            'all_record_change_types': dict(counts),
            'probability_order': ['home', 'draw', 'away'],
            'change_type_meanings': {'corrected': 'new correct, original wrong',
                'damaged': 'original correct, new wrong',
                'both_wrong_changed': 'different selections and both wrong',
                'unchanged': 'same selection, whether correct or wrong'}}


def render_report(result):
    table = []
    for period, group in result['by_period'].items():
        for track, label in (('original','原121项模型'),('state_rich','丰富输入＋持续状态')):
            value = group['tracks'][track]
            table.append(f"|{period}|{label}|{value['correct']}/{value['n']}|{value['accuracy']:.2%}|"
                f"{value['mean_losses']['one_x_two']:.6f}|{value['mean_losses']['draw_binary']:.6f}|"
                f"{value['mean_losses']['conditional_direction_weighted']:.6f}|{value['draw_selected_correct']}/{value['draw_selected']}|")
    positive_accuracy = all(v['new_minus_original']['accuracy'] > 0 for v in result['by_period'].values())
    positive_ll = all(v['new_minus_original']['one_x_two'] < 0 for v in result['by_period'].values())
    conclusion = ('两季单选与概率损失点值均改善，仍需真正赛前记录检验。' if positive_accuracy and positive_ll else
                  '未形成两季单选与概率质量一致改善的证据，不替换现有预测。')
    lines = ['# 丰富信息与持续状态：一次固定接入结果', '', f'2026-10-28。{conclusion}', '',
        '原121项输入保留，新加六项已学到的球队攻防状态与联赛率项；每行的参数资料及观察严格早于F。'
        '前3,515个训练历史行新增项保持None，后3,452个目标行可用。'
        '没有用最终状态回填历史，没有平均两条概率，也没有另加平局分支、市场或LLM权重。', '',
        '原进球目标及原独立Poisson比分结构保留。本候选没有直接优化胜平负损失，'
        '也不能自动解决较高总球条件下很少首选平局的结构限制。', '',
        '|赛季|线路|正确/场数|命中率|HDA LL↓|平/非平LL↓|条件方向加权LL↓|选平中/选|',
        '|---|---|---:|---:|---:|---:|---:|---:|', *table, '',
        'HDA LL精确拆为平/非平二分类损失与实际非平时的方向损失。后者在表中按全部目标加权，'
        '不能将实际非平子集成绩当作赛前可执行准确率。完整比分及总球评分保存在完整JSON。', '']
    for period, group in result['by_period'].items():
        change = group['changes']
        lines.extend([f'## {period}', '',
            f"首选变化{change['changed']}场，纠正{change['corrected']}场、改坏{change['damaged']}场，"
            f"改向后双方均错{change['both_wrong_changed']}场。各类概率和每场变化均保留。", ''])
    lines.extend(['## 时间、缺失与解释', '',
        '2024-08-01首个窗口的较早训练行没有可用GAS数值，直接复用原模型和原121项输入。'
        '其余窗口仅当较早训练行已出现新数值时才加入相应列，避免全空列改变随机抽列后被误称新信息。'
        '具体每窗训练ID、日期与活跃列记录在窗口计划及模型参数。', '',
        '参数资料截止2024-06-02是历史回测的前缀条件，不表示此模型当时真实已经发布。'
        '全局GAS实际于2026年拟合，数据商历史修订与当时发布时间限制保留。', '',
        '列可用性与时期共变，原训练期中位数及缺失指示可能表达时期差异；'
        '新增列还改变树分裂机会，因此这是完整候选比较，不将所有变化因果归给六项数值。'
        '训练重要性不能证明时间外贡献。', '',
        f"全部{result['selection_change_csv']['changed_records']}场改向已导出为接入全部改向比赛.csv，"
        '原／新主胜、平局、客胜概率分别列示，并列实际结果、原／新首选及纠正、改坏或双方均错类型。'
        '完整逐场JSON还保留未改向记录的明确类型。', '',
        '后两季已反复用于研发，本结果不是新盲测或当前准确率；四周条件区间不包括完整重训及选择不确定性。'
        '一次固定接入到此停止，不按结果改列、窗口、种子、权重或树参数。', '',
        f"实际新增原设置进球树拟合{result['new_fits']}次；复用原模型{result['original_model_reuses']}窗；"
        f"复用本轮已经保存的模型{result['prepared_model_reuses']}窗。GAS全局参数重训0次、参数网格0次。"
        '当前预测未替换、新赛果查询0，全部新增文件在G盘。', ''])
    lines += ['## 本轮留下的具体认识', '',
        '两季合计命中仍为1,824场；早季增加8场不能掩盖晚季减少8场。H/D/A损失点值均下降，'
        '但两季四周条件区间均跨0，不能称稳定提准。总球损失点值两季均稍差，不能宣布全面改善。', '']
    for period, group in result['by_period'].items():
        ll_index = group['difference_columns'].index('one_x_two')
        ll_interval = group['four_week_conditional_intervals']['interval_95'][ll_index]
        lines.append(f"- {period}：LL差{group['new_minus_original']['one_x_two']:+.8f}，"
                     f"四周条件区间[{ll_interval[0]:+.8f}, {ll_interval[1]:+.8f}]；"
                     f"总球损失差{group['new_minus_original']['total_goals']:+.8f}。")
    usage_path = OUT / '状态列实际使用检查.json'
    if usage_path.exists():
        usage = read(usage_path)
        lines += ['', '对训练结果的只读检查还发现：']
        for item in usage['windows']:
            if item.get('actual_tree_split_nodes_raw_state') == 0:
                indicator_nodes = item['actual_tree_split_nodes_state_missing']
                lines.append(f"- {item['cutoff']}：新增状态数值在树中引用0次，新增缺失指示引用{indicator_nodes}次；"
                    f"{item['target_count']}个目标中首选改变{item['changed_top1']}场、"
                    f"纠正{item['corrected']}、改坏{item['damaged']}。")
        lines += ['', '首个新增拟合没有引用新增字段，仍改变了答案，说明随机分裂机会的变化是实际存在的，'
            '不能把全部结果归因于状态信息。后期确有数值引用；其训练重要性仍不是时间外增量或因果贡献。']
        distribution = usage.get('current_score_distribution_limit')
        if distribution:
            lines += ['', f"当前{distribution['rows_with_draw_bound_below_one_third']}/{distribution['n']}场在其总进球率T下，"
                '独立Poisson平局概率上限exp(-T)I0(T)已低于1/3，任何主客率分配都不能让平局排第一。'
                f"本次总率最小{distribution['new_total_goal_mean_min']:.6f}、最大"
                f"{distribution['new_total_goal_mean_max']:.6f}；所有目标最终仍没有平局首选。"
                '这是当前输出结构的约束，不是普遍预测准确率上限，也不是把总率人为压低的理由。']
        lines += ['', '后续若要处理这一限制，需要讨论训练目标或比分结构；本轮不再增加状态列、换随机种子或调整参数救分。', '']
    (OUT / '接入结果与所学.md').write_text('\n'.join(lines), encoding='utf-8')


def run_integration(rows, originals, original_parameters, windows):
    """One fixed complete research run; called only in explicit run mode."""
    (OUT / 'models').mkdir(exist_ok=True)
    records, grids, model_parameters, timings = [], [], {}, []
    new_fits, original_reuses, candidate_reuses = 0, 0, 0
    for window in windows:
        started = time.perf_counter()
        expected = verify_window(window, original_parameters)
        history, held = window_rows(rows, window)
        cutoff = window['cutoff']
        if window['reuse_original']:
            path = SOURCE / 'models' / f'fresh_performance_{cutoff}.joblib'
            model = joblib.load(path)
            original_reuses += 1
            reuse_kind = 'original_no_available_state_training_rows'
        else:
            path = OUT / 'models' / f'persistent_rich_{cutoff}.joblib'
            if path.exists():
                model = joblib.load(path)
                candidate_reuses += 1
                reuse_kind = 'completed_candidate_model'
            else:
                model = EarlyGoalLearner('extra_trees', 'rich').fit(history, date.fromisoformat(cutoff))
                verify_model(model, window, expected)
                joblib.dump(model, path, compress=3)
                new_fits += 1
                reuse_kind = 'new_fit'
        parameters = verify_model(model, window, expected)
        model_parameters[cutoff] = {'model': parameters, 'active_state_features': window['active_state_features'],
            'inactive_state_features': window['inactive_state_features'], 'reuse_kind': reuse_kind,
            'model_path': str(path)}
        rates = model.predict_many([r['side_features'] for r in held],
                                   [date.fromisoformat(r['date']) for r in held])
        for row, rate in zip(held, rates, strict=True):
            old = originals[row['match_id']]
            if row['match_id'] in parameters['reference_match_ids']:
                raise ValueError('Target entered its own fit')
            if not parameters['training_max_date'] < cutoff <= row['forecast_before_date'] < row['date']:
                raise ValueError('Training and target chronology differ from original dates')
            old_grid, original_output = fixture_grid(old['rates'], row['actual_score'])
            new_grid, new_output = fixture_grid(rate, row['actual_score'])
            if not np.allclose(original_output['probabilities'], old['probabilities'], rtol=0, atol=1e-12):
                raise ValueError('Original saved H/D/A does not reproduce from its saved rates')
            if window['reuse_original'] and not np.allclose(new_grid.grid, old_grid.grid, rtol=0, atol=1e-12):
                raise ValueError('An all-unknown state window changed original probabilities')
            old_top1 = int(np.argmax(original_output['probabilities']))
            new_top1 = int(np.argmax(new_output['probabilities']))
            change_type = ('unchanged' if old_top1 == new_top1 else
                           'corrected' if new_top1 == row['actual_outcome'] else
                           'damaged' if old_top1 == row['actual_outcome'] else 'both_wrong_changed')
            records.append({**{key: row[key] for key in CONTEXT},
                'feature_before_date': row['feature_before_date'],
                'state_input_available': row['persistent_state_available'],
                'state_history_max_date': row['state_history_max_date'],
                'state_parameter_training_max_date': row['state_parameter_training_max_date'],
                'active_state_features': window['active_state_features'], 'model_reuse_kind': reuse_kind,
                'old_top1': old_top1, 'new_top1': new_top1, 'change_type': change_type,
                'tracks': {'original': original_output, 'state_rich': new_output}})
            grids.append({'match_id': row['match_id'], 'axis': ['home_goals','away_goals'],
                          'probabilities': new_grid.grid.tolist(),
                          'rates': new_output['rates'], 'rho': 0})
        timings.append({'period': window['period'], 'cutoff': cutoff, 'training_n': len(history),
            'target_n': len(held), 'new_feature_count': len(window['active_state_features']),
            'reuse_kind': reuse_kind, 'seconds': time.perf_counter()-started})
        write(OUT / '接入训练进度.json', {'windows': timings, 'new_fits': new_fits,
            'original_model_reuses': original_reuses, 'prepared_model_reuses': candidate_reuses})
        print(json.dumps({'stage': 'window_complete', **timings[-1]}, ensure_ascii=False), flush=True)
    if len(records) != 3452 or len({r['match_id'] for r in records}) != 3452:
        raise ValueError('Complete integration did not retain 3452 unique targets')
    records.sort(key=lambda r: (r['date'], r['match_id']))
    write(OUT / '接入模型参数.json', model_parameters)
    zipped(OUT / '接入逐场概率.json.gz', {'records': records})
    zipped(OUT / '接入比分网格.json.gz', {'records': grids})
    result = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'new_fits': new_fits, 'original_model_reuses': original_reuses,
        'prepared_model_reuses': candidate_reuses, 'windows': timings,
        'full_cohort': evaluate(records, intervals=False),
        'by_period': {period: evaluate([r for r in records if r['period'] == period], intervals=True)
                      for period in PERIODS},
        'by_league': {period: {league: evaluate([r for r in records if r['period'] == period
                                               and r['competition'] == league], intervals=False)
                              for league in COMPETITIONS} for period in PERIODS},
        'adopted_for_current_forecasts': False, 'current_results_queries': 0,
        'parameter_grids': 0, 'upstream_gas_fits': 0,
        'selection_change_csv': export_selection_changes(records),
        'interpretation': 'complete candidate in previously reviewed historical cohorts, not a new blind test or causal attribution'}
    write(OUT / '接入完整结果.json', result)
    render_report(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('inspect','run'), default='inspect')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stage, started = 'load', time.perf_counter()
    try:
        rows, originals, original_parameters = load_inputs()
        windows = plan_windows(rows)
        for window in windows:
            verify_window(window, original_parameters)
        if len(windows) != 22:
            raise ValueError('Retain all 22 original historical parameter windows')
        write(OUT / '窗口接入计划.json', {'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
            'input_rows': len(rows), 'target_rows': len(originals), 'windows': windows,
            'execution_mode': args.mode, 'learner_target': 'real full-time goals',
            'probability_distribution': 'unchanged independent Poisson from conditional goal means',
            'parameter_grids': 0, 'new_draw_branch': False, 'current_predictions_replaced': False})
        if args.mode == 'inspect':
            stage = 'first_window_existing_model_measurement'
            result = inspect_first_window(rows, originals, original_parameters, windows)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        else:
            stage = 'fixed_full_integration'
            result = run_integration(rows, originals, original_parameters, windows)
            print(json.dumps({'mode': 'run', 'new_fits': result['new_fits'],
                'periods': {key: {name: value['correct'] for name,value in group['tracks'].items()}
                            for key,group in result['by_period'].items()}}, ensure_ascii=False), flush=True)
        print(json.dumps({'stage': 'complete', 'mode': args.mode,
                          'seconds': time.perf_counter()-started}, ensure_ascii=False), flush=True)
    except Exception as error:
        write(OUT / '接入执行失败记录.json', {'failed_at_utc': datetime.now(timezone.utc).isoformat(),
            'mode': args.mode, 'stage': stage, 'error_type': type(error).__name__, 'error': str(error),
            'traceback': traceback.format_exc(), 'action': 'report concrete failure without rescue parameter variants'})
        raise


if __name__ == '__main__':
    main()
