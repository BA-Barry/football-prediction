"""One fixed-preset, last-five opponent-adjusted xG feature comparison."""
import argparse
from datetime import date, datetime, timezone
import gc
import gzip
import json
from pathlib import Path
import time

import joblib
import numpy as np

from football_model.early_goal_learners import EarlyGoalLearner
from football_model.forecast_diagnostics import moving_block_means
from opponent_adjusted_xg_features_20261026 import build_adjusted_rows, EXTRA_FEATURE_NAMES
from run_broad_early_models_20261025 import predict_row
from run_fixed_horizon_refresh_20261025 import read_features, CONTEXT_KEYS
from run_npxg_comparison_20261025 import load_source, ROOT
from run_shot_increment_20261025 import score_details, write


SOURCE = ROOT/'outputs/固定发布日期信息刷新_20261025T071806561562Z'
PERIODS = ('2024-25', '2025-26')
COMPS = ('E0', 'D1', 'SP1', 'I1', 'F1')


def dump_prepared(path, rows):
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        json.dump({'rows': rows}, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def load_prepared(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)['rows']


def compare(reference, adjusted, *, intervals=False):
    ids = [r['match_id'] for r in reference]
    if ids != [r['match_id'] for r in adjusted] or len(set(ids)) != len(ids):
        raise ValueError('Use identical unique targets in identical order')
    for old, new in zip(reference, adjusted, strict=True):
        if any(old[k] != new[k] for k in CONTEXT_KEYS):
            raise ValueError('Reference and new forecast contexts differ')
    old_scores, old_losses = score_details(reference)
    new_scores, new_losses = score_details(adjusted)
    p = np.asarray([r['probabilities'] for r in reference])
    q = np.asarray([r['probabilities'] for r in adjusted])
    y = np.asarray([r['actual_outcome'] for r in reference])
    truth = np.eye(3)[y]
    brier_delta = (np.square(q-truth)-np.square(p-truth)).sum(axis=1)
    top1_delta = (q.argmax(axis=1) == y).astype(float)-(p.argmax(axis=1) == y).astype(float)
    columns = list(old_losses)+['one_x_two_brier', 'top1_accuracy']
    matrix = np.column_stack([new_losses[k]-old_losses[k] for k in old_losses]
                             + [brier_delta, top1_delta])
    return {'n': len(ids), 'original': old_scores, 'adjusted': new_scores,
            'difference_order': columns,
            'mean_differences_adjusted_minus_original': dict(zip(columns, matrix.mean(axis=0).tolist())),
            'top1_changed_n': int((p.argmax(axis=1) != q.argmax(axis=1)).sum()),
            'conditional_4week_intervals': moving_block_means(
                [r['date'] for r in reference], matrix, block_weeks=4) if intervals else None,
            'match_ids': ids}


def plan(now):
    return {
        'created_at': now.isoformat(), 'source': str(SOURCE), 'horizon_days': 3,
        'scope': 'independent research arm; all five leagues and both previously viewed historical seasons',
        'feature_change': {'original': 121, 'new': 123, 'new_names': list(EXTRA_FEATURE_NAMES),
            'definition': 'For each team, mean of exactly its existing last-five (actual xG net minus '
                          'that old match earlier-cached xG expert expected net), oriented to the team. '
                          'Both own and opponent team means are added symmetrically.'},
        'target': 'real full-time goals; unchanged independent Poisson score grid, rho=0',
        'learner': {'kind': 'extra_trees', 'feature_set': 'rich', 'half_life_days': 180,
                    'configuration': EarlyGoalLearner('extra_trees', 'rich')._estimator().get_params()},
        'cutoffs': 'original Aug1-anchored 28-day model dates; P<=F=D-3<D; train only date<P',
        'reference': 'reuse original fresh_performance saved forecasts and their unchanged 22 parameter dates',
        'comparison_periods': list(PERIODS), 'expected_targets_per_period': 1726,
        'primary': '1X2 LogLoss, adjusted minus original; both seasons separately',
        'secondary': ['1X2 Brier and top1', 'score and total-goal NLL',
                      'away-minus-two NLL', 'all five leagues and missing-feature coverage'],
        'intervals': 'paired conditional noncircular 4-calendar-week blocks, 2000 repeats; '
                     'not a fresh untouched validation or a single-match probability interval',
        'missing': 'retain every target and None. Existing training-only median imputer plus missing '
                   'indicators; no zero residual or replacement with older references',
        'stopping_point': '22 new fixed-preset fits, 0 base-expert refits, 0 grids, 0 extra leads; '
                          'report all declared seasons/leagues, then stop this arm without searching variants',
        'limitations': ['Old expected xG is itself estimated, not an objective true opponent adjustment',
                        'Only league histories and date-level availability, no source publication/revision timestamps',
                        'Historical opponents/venues enter through the earlier expert, not through a newly fitted causal model',
                        'Season 2022/23 has more unknown references because earlier predictions are not cached',
                        'Two added columns also change possible randomized tree splits; no causal feature-effect claim',
                        'Current targets are 5-7 days ahead and new-season replay is not prepared; do not replace current forecasts'],
        'prospective_forecasts_replaced': False, 'current_results_checked': False,
    }


def report(out, result):
    lines = ['# 对手校正后的近期 xG：固定设置完整对照', '',
        '在原121项信息上只加两项：自己、对手最近5场的实际净xG相对各场当时旧预测的平均偏差。'
        '预测目标仍是真实全场进球；比分概率与胜平负使用原来的转换方式。', '',
        '## 两季完整结果', '',
        '| 赛季 | 场数 | 原胜平负LL | 新LL | LL差 | Brier差 | 原单选命中率 | 新单选命中率 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for period, item in result['by_period'].items():
        old, new = item['original']['one_x_two'], item['adjusted']['one_x_two']
        delta = item['mean_differences_adjusted_minus_original']
        lines.append(f"| {period} | {item['n']} | {old['log_loss']:.6f} | {new['log_loss']:.6f} | "
                     f"{delta['one_x_two']:+.6f} | {delta['one_x_two_brier']:+.6f} | "
                     f"{old['accuracy']:.2%} | {new['accuracy']:.2%} |")
    lines += ['', 'LL和Brier差为新减原，负数更好。命中率是每场只选最高概率的一个胜平负结果，未把不败合并计作命中。', '',
        '| 赛季 | LL差的4周条件区间 | Brier差区间 | 比分NLL差 | 总球NLL差 |',
        '|---|---|---|---:|---:|']
    for period, item in result['by_period'].items():
        cis = dict(zip(item['difference_order'], item['conditional_4week_intervals']['interval_95']))
        delta = item['mean_differences_adjusted_minus_original']
        lines.append(f"| {period} | [{cis['one_x_two'][0]:+.6f}, {cis['one_x_two'][1]:+.6f}] | "
                     f"[{cis['one_x_two_brier'][0]:+.6f}, {cis['one_x_two_brier'][1]:+.6f}] | "
                     f"{delta['exact_score']:+.6f} | {delta['total_goals_0_1_2_3_4plus']:+.6f} |")
    lines += ['', '这些区间仅描述既有预测、既定模型下的时间分块波动，不包含完整重新训练和所有模型不确定性；'
              '两季历史赛果此前已用于研究，不能称为全新盲测。', '',
        '## 五联赛结果全部保留', '',
        '| 赛季／联赛 | 场数 | 胜平负LL差 | Brier差 | 单选命中率变化 |', '|---|---:|---:|---:|---:|']
    for period, leagues in result['by_league'].items():
        for comp, item in leagues.items():
            d = item['mean_differences_adjusted_minus_original']
            lines.append(f"| {period}／{comp} | {item['n']} | {d['one_x_two']:+.6f} | "
                         f"{d['one_x_two_brier']:+.6f} | {d['top1_accuracy']*100:+.3f}个百分点 |")
    lines += ['', '## 数据覆盖与执行边界', '']
    for period, counts in result['coverage']['by_period'].items():
        lines.append(f"- {period}：{counts['rows']}场；双方特征完整{counts.get('both_sides_available', 0)}场，"
                     f"仅一方完整{counts.get('one_side_available', 0)}场，双方均未知{counts.get('neither_side_available', 0)}场。")
    lines += ['', f"新增模型训练{result['new_fits_this_execution']}次，复用本研究已完成训练"
              f"{result['reused_completed_fits']}次；没有重训基专家、没有参数网格或额外提前量。",
        '缺失值保留到原有训练期内插补流程，目标比赛没有因新特征缺失而剔除。'
        '当时的预期净xG已利用球队、对手与主客场，但自身也有估计误差；该偏差不是球员能力或球队士气的直接测量。', '',
        '## 本轮决定', '', result['decision']['explanation'], '',
        '本轮22个既定日期完成后停止该路线，不继续改变窗口、树深或联赛权重寻找更好数字。'
        '当前比赛原预测、LLM结果与其他研究线均未替换，也没有复核当前新赛果。', '',
        '## 下一步建议', '',
        '先复用已保存的两季原模型预测，分别检查主胜、平局、客胜的概率偏差和单选错误，'
        '区分均势比赛与明显强弱比赛，并确认偏差是否跨季重复。此步不新增训练或选择参数。'
        '只有找到稳定且可修正的偏差后，再提出对应的核心改动；不因本次某个联赛子组较好而单独上线。']
    (out/'结果与使用决定.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    out = args.resume.resolve() if args.resume else ROOT/'outputs'/now.strftime('对手校正近期xG研究_%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=bool(args.resume))
    (out/'models').mkdir(exist_ok=True)
    method_path = out/'分析前计划.json'
    if not method_path.exists():
        write(method_path, plan(now))
    print(json.dumps({'output': str(out), 'phase': 'prepare'}, ensure_ascii=False), flush=True)
    started = time.perf_counter()
    prepared_path = out/'prepared_lead3.json.gz'
    if prepared_path.exists():
        rows = load_prepared(prepared_path)
        coverage = json.loads((out/'特征来源与覆盖.json').read_text(encoding='utf-8'))
    else:
        original = read_features(SOURCE/'features_lead3.json.gz')
        parsed, independent, joins, source_reports, directories = load_source()
        derived, coverage = build_adjusted_rows(original, parsed)
        # Save only the fields used by this comparison rather than duplicating
        # all three original large feature families in another archive.
        rows = [{**{k: r[k] for k in CONTEXT_KEYS},
                 'side_features': r['adjusted_performance_features'],
                 'adjusted_xg_residual_history': r['adjusted_xg_residual_history'],
                 'feature_before_date': r['fresh_feature_before_date']}
                for r in derived]
        write(out/'特征来源与覆盖.json', coverage)
        write(out/'复用数据源记录.json', {'joins': joins, 'source_reports': source_reports,
                                      'directories': directories})
        dump_prepared(prepared_path, rows)
        del original, parsed, independent, derived
        gc.collect()
    print(json.dumps({'phase': 'prepared', 'n': len(rows), 'coverage': coverage['by_period'],
                      'seconds': time.perf_counter()-started}, ensure_ascii=False), flush=True)
    rows.sort(key=lambda r: (r['date'], r['match_id']))
    if len({r['match_id'] for r in rows}) != len(rows):
        raise ValueError('Prepared rows are not unique')
    for row in rows:
        if not row['fit_before_date'] <= row['forecast_before_date'] < row['date']:
            raise ValueError('Prepared feature timing differs')
    source_tracks = json.loads((SOURCE/'lead_3/逐场完整预测.json').read_text(encoding='utf-8'))
    reference = sorted(source_tracks['fresh_performance'], key=lambda r: (r['date'], r['match_id']))
    target_ids = {r['match_id'] for r in rows if r['period'] in PERIODS}
    if target_ids != {r['match_id'] for r in reference}:
        raise ValueError('Original full-cohort targets differ')
    del source_tracks
    tracks, windows, parameters = [], [], {}
    new_fits, reused_fits = 0, 0
    serialisation_check = None
    for period in PERIODS:
        targets = [r for r in rows if r['period'] == period]
        for cutoff in sorted({r['fit_before_date'] for r in targets}):
            tick = time.perf_counter()
            held = [r for r in targets if r['fit_before_date'] == cutoff]
            history = [r for r in rows if r['date'] < cutoff]
            path = out/'models'/f'adjusted_{cutoff}.joblib'
            if path.exists():
                model = joblib.load(path)
                metadata = model.parameters()
                if (metadata['reference_match_ids'] != [r['match_id'] for r in history]
                        or metadata['before_date'] != cutoff
                        or metadata['feature_names'] != sorted(history[0]['side_features'][0])
                        or metadata['configuration'] != EarlyGoalLearner('extra_trees', 'rich')._estimator().get_params()
                        or metadata['half_life_days'] != 180):
                    raise ValueError('Saved study model differs from current inputs or declared settings')
                reused_fits += 1
            else:
                model = EarlyGoalLearner('extra_trees', 'rich').fit(history, date.fromisoformat(cutoff))
                joblib.dump(model, path, compress=3)
                new_fits += 1
            parameters[cutoff] = model.parameters()
            predicted = model.predict_many([r['side_features'] for r in held],
                                           [date.fromisoformat(r['date']) for r in held])
            if serialisation_check is None:
                replay = joblib.load(path).predict_many([held[0]['side_features']],
                                                       [date.fromisoformat(held[0]['date'])])[0]
                if not np.allclose(replay, predicted[0], atol=1e-12, rtol=0):
                    raise ValueError('Model serialization changed prediction')
                serialisation_check = {'cutoff': cutoff, 'match_id': held[0]['match_id'],
                                       'original': list(predicted[0]), 'reloaded': list(replay)}
            for row, rates in zip(held, predicted, strict=True):
                context = {k: row[k] for k in CONTEXT_KEYS}
                context['feature_before_date'] = row['feature_before_date']
                tracks.append(predict_row(context, rates))
            windows.append({'period': period, 'cutoff': cutoff, 'training_n': len(history),
                            'target_n': len(held), 'seconds': time.perf_counter()-tick})
            write(out/'训练进度.json', {'windows': windows, 'new_fits_this_execution': new_fits,
                                      'reused_completed_fits': reused_fits, 'total_model_dates': len(windows)})
            print(json.dumps({'phase': 'trained', **windows[-1], 'fits_complete': len(windows)},
                             ensure_ascii=False), flush=True)
    tracks.sort(key=lambda r: (r['date'], r['match_id']))
    if len(windows) != 22:
        raise ValueError('Declared 22-cutoff comparison was not completed')
    by_period, by_league = {}, {}
    for period in PERIODS:
        old = [r for r in reference if r['period'] == period]
        new = [r for r in tracks if r['period'] == period]
        if len(old) != 1726 or len(new) != 1726:
            raise ValueError('Expected full 1726-match historical period')
        by_period[period] = compare(old, new, intervals=True)
        by_league[period] = {comp: compare([r for r in old if r['competition'] == comp],
                                         [r for r in new if r['competition'] == comp]) for comp in COMPS}
    deltas = [by_period[p]['mean_differences_adjusted_minus_original']['one_x_two'] for p in PERIODS]
    if any(delta >= 0 for delta in deltas):
        explanation = '两季胜平负概率质量没有同向改善，未确认稳定增量。保留研究文件，不将这两项信息加入当前正式预测。'
    else:
        explanation = '两季胜平负平均概率损失都下降，但这仍是已查看历史数据上的研究结果；'
        explanation += '需结合区间、联赛差异和当前赛季合法历史预期是否齐备再决定使用。当前预测暂不替换。'
    decision = {'adopted': False, 'explanation': explanation,
                'stop_this_comparison': True, 'new_fits_planned_after_result': 0}
    result = {'method': json.loads(method_path.read_text(encoding='utf-8')), 'coverage': coverage,
              'by_period': by_period, 'by_league': by_league, 'windows': windows,
              'new_fits_this_execution': new_fits, 'reused_completed_fits': reused_fits,
              'serialization_check': serialisation_check, 'decision': decision,
              'seconds': time.perf_counter()-started}
    write(out/'全部模型参数.json', parameters)
    write(out/'逐场完整预测.json', {'original': reference, 'adjusted': tracks})
    write(out/'完整研究结果.json', result)
    report(out, result)
    for period, value in by_period.items():
        print(json.dumps({'period': period, 'n': value['n'],
            'original': value['original']['one_x_two'], 'adjusted': value['adjusted']['one_x_two'],
            'deltas': value['mean_differences_adjusted_minus_original'],
            'interval_order': value['difference_order'],
            'intervals': value['conditional_4week_intervals']['interval_95']}, ensure_ascii=False), flush=True)
    print(json.dumps({'output': str(out), 'new_fits': new_fits, 'reuse': reused_fits,
                      'decision': decision, 'seconds': result['seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
