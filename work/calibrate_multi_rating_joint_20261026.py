"""Earlier-season, two-scalar temperature calibration of saved joint forecasts.

No classifier is trained or loaded. The method is recorded separately before
reading scores. Run only after the root candidate experiment has completed.
"""
from datetime import date, datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import numpy as np

from football_model.forecast_diagnostics import moving_block_means
from football_model.prematch_calibration import fit_calibrator, apply_calibrator
from run_multi_rating_joint_20261026 import CONTEXT, COMPS, score, comparison


ROOT = REPOSITORY_ROOT
OUT = ROOT/'outputs/多路评级联合接入_20261026'
FIT_BEFORE = date(2025, 8, 1)
CALIBRATION_PERIOD = '2024-25'
EVALUATION_PERIOD = '2025-26'
CANDIDATES = ('rating_core', 'rich_joint')
RAW_ROUTES = ('saved_performance', 'saved_allcomp', *CANDIDATES)
ADDED_PAIRS = tuple((arm+'_calibrated', other)
                    for arm in CANDIDATES
                    for other in (arm, 'saved_performance', 'saved_allcomp'))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')


def ordered(rows):
    return sorted(rows, key=lambda row: (row['date'], row['match_id']))


def calibration_rows(rows):
    selected = ordered(row for row in rows if row['period'] == CALIBRATION_PERIOD)
    if not selected:
        raise ValueError('Need the complete earlier calibration season')
    for row in selected:
        fit = date.fromisoformat(row['fit_before_date'])
        publication = date.fromisoformat(row['forecast_before_date'])
        fixture = date.fromisoformat(row['date'])
        if not fit <= publication < fixture < FIT_BEFORE:
            raise ValueError('Calibration forecasts, model dates and labels must all predate the fixed date')
    return selected


def summarize(tracks, *, intervals):
    raw_tracks = {name: tracks[name] for name in RAW_ROUTES}
    # Reuse the original pairing/context checks, original route scores and
    # original raw comparisons, rather than changing their evaluation rules.
    result = comparison(raw_tracks, intervals=intervals)
    reference = tracks['saved_performance']
    losses = {name: score(rows)[1] for name, rows in raw_tracks.items()}
    ids = [row['match_id'] for row in reference]
    for arm in CANDIDATES:
        name = arm+'_calibrated'
        values = tracks[name]
        if ids != [row['match_id'] for row in values]:
            raise ValueError('Calibrated forecasts must retain every identical later-season fixture')
        if any(any(a[key] != b[key] for key in CONTEXT)
               for a, b in zip(reference, values, strict=True)):
            raise ValueError('Calibrated forecasts must retain the exact original contexts')
        result['scores'][name], losses[name] = score(values)
    columns, deltas = [], []
    for calibrated, other in ADDED_PAIRS:
        for metric in ('log_loss', 'brier_sum', 'error'):
            columns.append(f'{calibrated}_minus_{other}:{metric}')
            deltas.append(losses[calibrated][metric] - losses[other][metric])
    matrix = np.column_stack(deltas)
    result['calibration_difference_order'] = columns
    result['calibration_mean_differences'] = dict(zip(columns, matrix.mean(axis=0).tolist()))
    if intervals:
        result['calibration_conditional_four_week_intervals'] = moving_block_means(
            [row['date'] for row in reference], matrix, block_weeks=4)
    return result


def report(result):
    names = {'saved_performance': '原丰富表现', 'saved_allcomp': '原表现＋全赛事',
             'rating_core': '评级核心＋XGBoost（原始）',
             'rich_joint': '完整丰富联合＋XGBoost（原始）',
             'rating_core_calibrated': '评级核心＋共享温度',
             'rich_joint_calibrated': '完整丰富联合＋共享温度'}
    lines = ['# 多路评级联合方案：较早赛季温度校准', '',
             '按先行记录的方法，各用2024/25样本外预测学习一个五联赛共享温度，'
             '2025-08-01固定，只评价2025/26。未重训分类器，没有联赛专属温度、'
             '类别偏置、方法挑选或参数网格。', '',
             '| 候选 | 较早学习场数 | beta | 温度T | 后季最高概率方向变化数 |',
             '|---|---:|---:|---:|---:|']
    for arm in CANDIDATES:
        parameters = result['calibrators'][arm]
        beta, temperature = (parameters['parameters'][key] for key in ('beta', 'temperature'))
        t = '∞（均匀极限）' if temperature is None else f'{temperature:.6f}'
        lines.append(f"| {names[arm]} | {parameters['n_matches']} | {beta:.6f} | {t} | "
                     f"{result['argmax_changes'][arm]} |")
    lines += ['', '| 后季方案 | 场数 | 单选命中 | LogLoss↓ | Brier↓ | RPS↓ | 平局召回 | ECE↓ |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, scores in result['overall']['scores'].items():
        recall = scores['class_recall']['draw']
        recall_text = '无平局样本' if recall is None else f'{recall:.2%}'
        lines.append(f"| {names[arm]} | {scores['n']} | {scores['correct']}/{scores['n']}"
                     f"（{scores['accuracy']:.2%}） | {scores['log_loss']:.6f} | "
                     f"{scores['brier_sum']:.6f} | {scores['rps']:.6f} | {recall_text} | "
                     f"{scores['class_mean_ece_10_equal_width']:.6f} |")
    lines += ['', '## 解释与收束', '',
              '- beta>0时温度只改变概率锐度，保持主／平／客排序；概率质量改善不能称为单选命中改善。'
              'beta=0是现有实现允许的均匀极限，若发生会明确列出方向变化。',
              '- 温度在较早季学习的损失仅是拟合诊断，不与后季效果混为一谈。'
              '该学习集合已由较早训练的分类器生成预测；没有用后季赛果拟合校准。',
              '- 两个候选及两个已有参照都在完全相同的后季比赛上比较；逐联赛结果只作描述，'
              '没有按联赛重新学参数或筛选容易场次。',
              '- 四周条件区间只描述已有预测差异，不包含完整重训不确定性。'
              '这两个历史赛季已多轮用于开发，属于时间前推研究，不是未接触的前瞻确认。',
              '- 本轮只学习2个标量，到此停止；原始概率保留，当前比赛预测与LLM／联合线未替换。'
              '此操作没有查询当前新赛果，也没有新增比分、总球或让球分布。', '',
              '方法见`校准方法.json`；两项拟合与全部概率／分类／校准指标见`校准完整结果.json`；'
              '后季逐场原始及校准概率见`校准逐场胜平负预测.json`。']
    (OUT/'校准结果.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    method = read(OUT/'校准方法.json')
    if (method['candidates'] != list(CANDIDATES)
            or method['fixed_before_date'] != FIT_BEFORE.isoformat()
            or method['calibration_period'] != CALIBRATION_PERIOD
            or method['evaluation_period'] != EVALUATION_PERIOD):
        raise ValueError('Use the previously recorded calibration scope')
    # The complete source is read only when explicitly run after training.
    original = read(OUT/'逐场完整胜平负预测.json')
    raw = {name: ordered(row for row in original[name]
                         if row['period'] == EVALUATION_PERIOD) for name in RAW_ROUTES}
    calibrators, changes = {}, {}
    source_ids = None
    for arm in CANDIDATES:
        earlier = calibration_rows(original[arm])
        ids = [row['match_id'] for row in earlier]
        if source_ids is not None and ids != source_ids:
            raise ValueError('Both temperatures must use the same complete earlier cohort')
        source_ids = ids
        fitted = fit_calibrator(earlier, FIT_BEFORE, method='temperature')
        calibrators[arm] = fitted
        targets = raw[arm]
        for row in targets:
            publication = date.fromisoformat(row['forecast_before_date'])
            fixture = date.fromisoformat(row['date'])
            if not FIT_BEFORE <= publication < fixture:
                raise ValueError('Apply the frozen calibrator only on or after its fixed date')
        probabilities = np.asarray([row['probabilities'] for row in targets], dtype=float)
        calibrated = apply_calibrator(probabilities, fitted)
        changed = int(np.count_nonzero(probabilities.argmax(axis=1) != calibrated.argmax(axis=1)))
        changes[arm] = changed
        if fitted['parameters']['beta'] > 0 and changed:
            raise ValueError('Positive temperature must preserve highest-probability outcome classes')
        raw[arm+'_calibrated'] = [dict(row, probabilities=p.tolist(),
                                           calibration_before_date=FIT_BEFORE.isoformat(),
                                           calibration_method='temperature',
                                           raw_probability_route=arm)
                                  for row, p in zip(targets, calibrated, strict=True)]
    results = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'method': method,
        'classifier_refits': 0, 'scalar_fits': len(calibrators),
        'calibrators': calibrators, 'argmax_changes': changes,
        'overall': summarize(raw, intervals=True),
        'by_league': {league: summarize(
            {arm: [row for row in rows if row['competition'] == league]
             for arm, rows in raw.items()}, intervals=False) for league in COMPS},
        'adopted': False, 'current_forecasts_replaced': False,
        'current_results_checked': False,
    }
    write(OUT/'校准逐场胜平负预测.json', raw)
    write(OUT/'校准完整结果.json', results)
    report(results)
    print(json.dumps({'scalar_fits': len(calibrators), 'classifier_refits': 0,
                      'calibration_parameters': {arm: fit['parameters'] for arm, fit in calibrators.items()},
                      'later_scores': {arm: {key: scores[key] for key in ('n','correct','accuracy','log_loss','brier_sum')}
                                       for arm, scores in results['overall']['scores'].items()},
                      'directory': str(OUT)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
