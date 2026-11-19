"""One joint residual-pattern package, fixed goal learner and saved draw branch.

No grids, upstream-expert/classifier refits, current outcomes, or new forecasts.
All evaluated historical seasons have previously been examined in research.
"""
from collections import Counter
from datetime import date, datetime, timezone
import csv
import gc
import gzip
import json
import math
from pathlib import Path
import time

import joblib
import numpy as np

from football_model.early_goal_learners import EarlyGoalLearner
from football_model.draw_branch import apply_draw_branch
from football_model.forecast_diagnostics import moving_block_means
from football_model.probability import score_grid, summarize
from football_model.refresh_experiment import scores_for
from run_fixed_horizon_refresh_20261025 import read_features, CONTEXT_KEYS
from run_npxg_comparison_20261025 import load_source, ROOT
from run_draw_branch_integration_20261027 import draw_summary, MATCH_FIELDS
from xg_residual_pattern_features_20261027 import build_residual_pattern_rows, ADDED_FEATURE_NAMES


SOURCE = ROOT / 'outputs/固定发布日期信息刷新_20261025T071806561562Z'
DRAW = ROOT / 'outputs/历史比赛模式与平局分支_20261027'
OUT = ROOT / 'outputs/对手校正攻防模式整合_20261027'
PERIODS = ('2024-25', '2025-26')
COMPS = ('E0', 'D1', 'SP1', 'I1', 'F1')
TRACKS = ('original', 'residual_only', 'existing_draw', 'combined')
PAIRS = (('residual_only', 'original'), ('combined', 'original'), ('combined', 'existing_draw'))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def zipped(path, value):
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def plan():
    return {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'one joint representation package, all five leagues/two previously viewed seasons',
        'sources': {'features': str(SOURCE), 'saved_draw': str(DRAW)},
        'feature_counts': {'unchanged_original': 121, 'added': 16, 'total': 137},
        'added_names': list(ADDED_FEATURE_NAMES),
        'measurement': 'actual total xG minus that old fixture earlier-cached total-xG expert mean',
        'summaries': 'for/against/net/total means, mean abs net, net/total population variance, for-against population covariance',
        'redundancy': 'Net/total means and covariance have exact algebraic relations; multiple views, not independent evidence.',
        'window': 'exact original last-five league IDs, one missing reference leaves all eight summaries None; no backfill',
        'timing': 'past fit<=past forecast<past match<target F; target P<=F=D-3; train date<P',
        'learner': EarlyGoalLearner('extra_trees', 'rich')._estimator().get_params(),
        'half_life_days': 180,
        'fits': '22 fixed goal-learner fits; 0 base-expert fits; 0 classifier fits; 0 grids or calibration fits',
        'combined': 'apply saved v0.47 class-D to new goal grid, retain new conditional H/A and within-result score ratios',
        'score': 'full-score NLL from each actual final grid cell, not independent Poisson at post-scaling means',
        'diagnostic_tracks': list(TRACKS), 'declared_contrasts_new_minus_reference': list(PAIRS),
        'intervals': 'existing paired four-calendar-week conditional block intervals, not complete model uncertainty',
        'coverage': 'all 1726 targets in each season; no intersection/drop by new availability',
        'limitations': [
            'Historical seasons already used repeatedly for research; not a new blind test or current accuracy.',
            'Old expected total xG is an estimate; residuals do not identify causal ability or tactics.',
            'No reliable goal/man-state adjustment is asserted; no target lineup is used.',
            'Additional algebraic views change random tree split opportunity with unchanged max_features.',
            'Pure goal stage retains independent Poisson limits; final stage retains the existing d signal.',
            'Exact source publication, revisions and provider internal training remain unknown.',
        ],
        'stop': 'one complete package and full two-season report; no result-dependent variants or subgroup weights',
        'current_outcomes_checked': False, 'current_forecasts_replaced': False,
        'LLM_or_market_probabilities_used': False,
    }


def prepare():
    original = read_features(SOURCE / 'features_lead3.json.gz')
    parsed, _, joins, _, directories = load_source()
    derived, coverage = build_residual_pattern_rows(original, parsed)
    rows = [{**{k: r[k] for k in CONTEXT_KEYS},
             'feature_before_date': r['fresh_feature_before_date'],
             'side_features': r['residual_pattern_features'],
             'residual_pattern_history': r['residual_pattern_history']} for r in derived]
    rows.sort(key=lambda r: (r['date'], r['match_id']))
    zipped(OUT / '联合输入与历史来源.json.gz', {'rows': rows})
    write(OUT / '特征覆盖.json', coverage)
    write(OUT / '复用真实数据来源.json', {'joins': joins, 'directories': directories})
    del original, parsed, derived
    gc.collect()
    return rows, coverage


def fit_predict(rows):
    originals = {r['match_id']: r for r in read(SOURCE / 'lead_3/逐场完整预测.json')['fresh_performance']}
    old_params = read(SOURCE / 'lead_3/全部模型参数.json')
    draws = {r['match_id']: r for r in read(DRAW / '逐场概率与纠错.json')['records']}
    targets = [r for r in rows if r['period'] in PERIODS]
    if len(originals) != 3452 or {r['match_id'] for r in targets} != set(originals) or set(draws) != set(originals):
        raise ValueError('Keep both complete original historical cohorts')
    result, grids, parameters, windows = [], [], {}, []
    new_fits, reused = 0, 0
    for period in PERIODS:
        local = [r for r in targets if r['period'] == period]
        if len(local) != 1726:
            raise ValueError('Historical season coverage changed')
        for cutoff in sorted({r['fit_before_date'] for r in local}):
            tick = time.perf_counter()
            history = [r for r in rows if r['date'] < cutoff]
            held = [r for r in local if r['fit_before_date'] == cutoff]
            path = OUT / 'models' / f'residual_patterns_{cutoff}.joblib'
            if path.exists():
                model = joblib.load(path)
                reused += 1
            else:
                model = EarlyGoalLearner('extra_trees', 'rich').fit(history, date.fromisoformat(cutoff))
                joblib.dump(model, path, compress=3)
                new_fits += 1
            p = model.parameters()
            old = old_params['fresh_performance:' + cutoff]
            for field in ('reference_match_ids', 'before_date', 'training_min_date', 'training_max_date',
                          'n_matches', 'n_scoring_sides', 'configuration', 'half_life_days'):
                if p[field] != old[field]:
                    raise ValueError(f'Original learner setting or training scope differs: {field}')
            if p['feature_names'] != sorted(history[0]['side_features'][0]) or len(p['feature_names']) != 137:
                raise ValueError('Prepared and fitted feature names differ')
            parameters[cutoff] = p
            means = model.predict_many([r['side_features'] for r in held],
                                       [date.fromisoformat(r['date']) for r in held])
            if not windows:
                restored = joblib.load(path).predict_many([held[0]['side_features']],
                                                          [date.fromisoformat(held[0]['date'])])[0]
                if not np.allclose(restored, means[0], rtol=0, atol=1e-12):
                    raise ValueError('Stored model changed its prediction')
            for row, rates in zip(held, means, strict=True):
                mid = row['match_id']
                base, draw = originals[mid], draws[mid]
                if any(base[k] != row[k] or base[k] != draw[k] for k in MATCH_FIELDS):
                    raise ValueError('Historical components have different fixture/timing contexts')
                if mid in p['reference_match_ids'] or not p['training_max_date'] < cutoff <= row['forecast_before_date'] < row['date']:
                    raise ValueError('Target entered its own training or timing changed')
                d = draw['classifier_draw_probability']
                original_grid, residual_grid = score_grid(*base['rates'], rho=0), score_grid(*rates, rho=0)
                all_grids = {'original': original_grid, 'residual_only': residual_grid,
                             'existing_draw': apply_draw_branch(original_grid, d),
                             'combined': apply_draw_branch(residual_grid, d)}
                context = {k: base[k] for k in MATCH_FIELDS}
                record = context | {'input_goal_means': list(rates), 'original_goal_means': base['rates'],
                    'unchanged_draw_probability': d, 'tracks': {}}
                h, a = base['actual_score']
                for name, grid in all_grids.items():
                    forecast = summarize(grid)
                    prob = list(grid.home_draw_away)
                    if not np.allclose(prob, list(forecast['one_x_two'].values()), atol=1e-12, rtol=0):
                        raise ValueError('Grid and market probabilities differ')
                    record['tracks'][name] = {'probabilities': prob, 'forecast': forecast,
                        'exact_score_nll': -math.log(float(grid.grid[h, a]))}
                if not np.allclose(record['tracks']['original']['probabilities'], base['probabilities'], atol=1e-12, rtol=0):
                    raise ValueError('Original model no longer replays')
                if not np.allclose(record['tracks']['existing_draw']['probabilities'], draw['probabilities'], atol=1e-12, rtol=0):
                    raise ValueError('Existing draw version no longer replays')
                final_p = record['tracks']['combined']['probabilities']
                residual_p = record['tracks']['residual_only']['probabilities']
                if abs(final_p[1]-d) > 1e-12 or not math.isclose(final_p[0]/final_p[2], residual_p[0]/residual_p[2], rel_tol=1e-12):
                    raise ValueError('Combined output did not retain declared responsibilities')
                grids.append({'match_id': mid, 'axis': ['home_goals', 'away_goals'],
                    'probabilities': all_grids['combined'].grid.tolist(),
                    'input_goal_means': list(rates), 'expected_goals': record['tracks']['combined']['forecast']['expected_goals']})
                result.append(record)
            windows.append({'period': period, 'cutoff': cutoff, 'training_n': len(history),
                            'target_n': len(held), 'seconds': time.perf_counter()-tick})
            write(OUT / '训练进度.json', {'windows': windows, 'new_fits': new_fits, 'reused': reused})
            print(json.dumps({'phase': 'fit', **windows[-1]}, ensure_ascii=False), flush=True)
    if len(windows) != 22:
        raise ValueError('Incomplete declared fit schedule')
    write(OUT / '全部模型参数.json', parameters)
    result.sort(key=lambda r: (r['date'], r['match_id']))
    write(OUT / '逐场概率与分量.json', {'records': result})
    zipped(OUT / '最终统一比分网格.json.gz', {'records': grids})
    return result, windows, new_fits, reused


def evaluate(rows, *, intervals):
    y = np.asarray([r['actual_outcome'] for r in rows])
    truth = np.eye(3)[y]
    y_total = np.asarray([min(sum(r['actual_score']), 4) for r in rows])
    margin = np.asarray([r['actual_score'][1]-r['actual_score'][0]-2 for r in rows])
    y_handicap = np.where(margin > 0, 0, np.where(margin == 0, 1, 2))
    scores, losses, tops = {}, {}, {}
    for name in TRACKS:
        prepared = [{**{k: r[k] for k in MATCH_FIELDS}, **r['tracks'][name]} for r in rows]
        probs = np.asarray([r['probabilities'] for r in prepared])
        tops[name] = probs.argmax(axis=1)
        scores[name] = scores_for(prepared)
        scores[name]['correct_n'] = int((tops[name] == y).sum())
        scores[name]['draw'] = draw_summary(probs, y)
        total = np.asarray([[r['forecast']['total_goals'][k] for k in ('0', '1', '2', '3', '4+')] for r in prepared])
        handicap = np.asarray([[r['forecast']['away_minus_two_jincai'][k] for k in ('win', 'draw', 'lose')] for r in prepared])
        losses[name] = {'one_x_two': -np.log(probs[np.arange(len(y)), y]),
            'exact_score': np.asarray([r['exact_score_nll'] for r in prepared]),
            'total_goals': -np.log(total[np.arange(len(y)), y_total]),
            'away_minus_two': -np.log(handicap[np.arange(len(y)), y_handicap]),
            'brier': np.square(probs-truth).sum(axis=1)}
        scores[name]['exact_score'] = float(losses[name]['exact_score'].mean())
    contrasts = {}
    for candidate, reference in PAIRS:
        matrix = np.column_stack([losses[candidate][k]-losses[reference][k] for k in losses[candidate]])
        corrected = (tops[reference] != y) & (tops[candidate] == y)
        damaged = (tops[reference] == y) & (tops[candidate] != y)
        contrasts[candidate + '_minus_' + reference] = {
            'loss_order': list(losses[candidate]),
            'mean_loss_differences': dict(zip(losses[candidate], matrix.mean(axis=0).tolist())),
            'four_week_conditional_intervals': moving_block_means([r['date'] for r in rows], matrix, block_weeks=4) if intervals else None,
            'changed_n': int((tops[candidate] != tops[reference]).sum()),
            'corrected_n': int(corrected.sum()), 'damaged_n': int(damaged.sum()),
            'corrected_actual_draw': int((corrected & (y == 1)).sum()),
            'corrected_actual_non_draw': int((corrected & (y != 1)).sum()),
            'damaged_actual_draw': int((damaged & (y == 1)).sum()),
            'damaged_actual_non_draw': int((damaged & (y != 1)).sum()),
            'class_LL_contribution': {label: float(np.where(y == i, matrix[:, 0], 0).mean())
                                      for i, label in enumerate(('home', 'draw', 'away'))},
        }
    return {'n': len(rows), 'scores': scores, 'contrasts': contrasts}


def report(result):
    labels = {'original': '原丰富模型', 'residual_only': '攻防模式进球层',
              'existing_draw': '已有平局整合', 'combined': '本轮联合输出'}
    lines = ['# 对手校正攻防模式：一次固定整合结果', '',
        '原121项输入保留；同一原5场窗口内加入攻防残差及净差、总量和波动的16种团队视角表示，合计137列。'
        '总xG对总xG，不混NPxG或实际进球；正against残差表示允许机会多于当时预期。'
        '同一现有进球树完成22个日期拟合，随后复用已有v0.47平局概率，统一到新比分网格。'
        '这里的联合仅指统计进球层与既有平局分支，不包含LLM或市场概率。', '',
        '| 赛季 | 输出阶段 | 正确/场数 | 胜平负LL | 比分NLL | 平局首选：中/选 |',
        '|---|---|---:|---:|---:|---:|']
    for period, value in result['by_period'].items():
        for name, item in value['scores'].items():
            d = item['draw']
            lines.append(f"| {period} | {labels[name]} | {item['correct_n']}/{value['n']} | "
                f"{item['one_x_two']['log_loss']:.6f} | {item['exact_score']:.6f} | {d['top1_correct']}/{d['top1_selected']} |")
    lines += ['', '各阶段用于定位分工；没有按后季成绩挑选其中一条作为正式预测。两季已反复用于研究，成绩不是全新盲测或当前准确率。', '',
        '| 赛季／联合输出相对 | 纠正 | 改坏 | 纠正实际平局 | 纠正实际非平 | LL差 | 4周条件区间 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for period, value in result['by_period'].items():
        for ref in ('original', 'existing_draw'):
            c = value['contrasts']['combined_minus_' + ref]
            interval = c['four_week_conditional_intervals']['interval_95'][0]
            lines.append(f"| {period}／{labels[ref]} | {c['corrected_n']} | {c['damaged_n']} | "
                f"{c['corrected_actual_draw']} | {c['corrected_actual_non_draw']} | "
                f"{c['mean_loss_differences']['one_x_two']:+.6f} | [{interval[0]:+.6f}, {interval[1]:+.6f}] |")
    lines += ['', '区间只描述保存预测下的时间分块波动，不包含完整重训不确定性或反复研发的选择偏差。所有五联赛结果保存在完整JSON，未按联赛成绩挑选上线。', '',
        '## 结果判断与留下的资产', '',
        '相对原丰富模型，两季胜平负LL分别下降约0.001147、0.001728；相对已有平局整合，'
        '本轮新增表示带来的下降仅约0.000332、0.000259，所有相应4周条件区间均包含0。'
        '两季概率评分点值同向，但不足以确认稳定预测增益。单选早季净增3场、晚季净减7场，不能说准确率提高。', '',
        '晚季相对原模型纠正17场、改坏24场；相对已有平局整合纠正13场、改坏17场。'
        '晚季总进球分类损失还上升约0.001158（相对已有平局整合约0.000787），不能宣称全面改善。'
        '概率损失与单选排名衡量的对象不同：真实结果概率提高但仍未排第一，也会改善概率评分。', '',
        '保留可复用的攻防残差序列、缺失语义、统计汇总组件、22个模型以及所有改首选记录；'
        '本轮结果只约束“这套137列表达＋当前进球树＋固定d”的具体接入，不等于攻防条件信息没有价值。'
        '同一信息包不再删列、换窗口或调树来追后季分数，当前预测不采用此研究版本。', '',
        '## 本次整合保留的认识', '',
        '攻防分量可以区分净值相同却产生于不同攻防结构的历史，波动可以区分均值相同而稳定性不同的历史。'
        '但净／总均值及协方差存在代数冗余；这是不同投影表达，不是16条独立新信息。'
        '固定max_features时这些列也改变随机树的分裂机会，因此整体变化不能全归因于新的球队信息。', '',
        '纯进球层仍用独立Poisson；总均值大于约1.760581时，任何主客均值分配都无法使平局成为最高概率类。'
        '联合输出直接保留已有平局d，因此相对已有平局整合，其平局AUC及二元平局损失必然相同；'
        '变化来自非平条件下的主客比例及区域内部比分。不能声称本轮学出了新的平局信号。', '',
        '最终比分NLL从最终网格对应比分格子读取，总球和让球也由同一网格导出。'
        '网格缩放后的期望进球不是重新构造独立Poisson的参数。', '',
        '每个旧场保留fit≤F<比赛日<目标F；只使用原5场参考，缺一场整组未知。'
        '中位数与缺失指示只在较早训练集拟合。对手预期本身是估计值，未声称消除了战术、红牌或比分状态影响。', '',
        '## 执行与采用状态', '',
        f"实际新增进球树拟合{result['new_fits']}次，复用本轮已完成模型{result['reused_fits']}次；"
        '上游专家、平局分类器重训0次，参数搜索和新校准0次。所有目标保留，当前赛果查询0、当前比赛预测未替换。', '',
        '资料覆盖：2024/25双方完整1,597场、单方完整113场、双方未知16场；'
        '2025/26对应1,623、95、8场。未知目标保留给训练期插补，未删场或以0残差代替。'
        '新增6项针对性测试通过；独立只读核查未发现阻断接入的时序、侧别或评分错误。', '',
        '本轮以一次完整联合接入为止。不因结果好坏重新删列、换窗口、换参数或给联赛另配权重。'
        '保留组件与纠错明细，是否采用根据两季完整结果与局限说明，不以某一指标或少数比赛裁定整个方向。', '',
        f'[完整结果]({OUT.as_posix()}/两季完整结果.json) · '
        f'[逐场概率]({OUT.as_posix()}/逐场概率与分量.json) · '
        f'[联合输入及历史来源]({OUT.as_posix()}/联合输入与历史来源.json.gz)', '',
        f'[全部改首选比赛]({OUT.as_posix()}/全部改首选比赛.csv) · '
        f'[方向纠错与损失分解]({OUT.as_posix()}/方向纠错与分解说明.md)', '']
    (OUT / '本轮整合结果与所学.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'models').mkdir(exist_ok=True)
    plan_path = OUT / '本轮方法与范围.json'
    if not plan_path.exists():
        write(plan_path, plan())
    print(json.dumps({'phase': 'prepare', 'output': str(OUT)}, ensure_ascii=False), flush=True)
    started = time.perf_counter()
    rows, coverage = prepare()
    print(json.dumps({'phase': 'prepared', 'n': len(rows), 'coverage': coverage['by_period']}, ensure_ascii=False), flush=True)
    records, windows, fits, reused = fit_predict(rows)
    result = {'method': read(plan_path), 'coverage': coverage,
        'by_period': {period: evaluate([r for r in records if r['period'] == period], intervals=True) for period in PERIODS},
        'by_league': {period: {comp: evaluate([r for r in records if r['period'] == period and r['competition'] == comp], intervals=False)
                              for comp in COMPS} for period in PERIODS},
        'windows': windows, 'new_fits': fits, 'reused_fits': reused,
        'seconds': time.perf_counter()-started, 'adopted_for_current_forecasts': False}
    write(OUT / '两季完整结果.json', result)
    report(result)
    for period, value in result['by_period'].items():
        print(json.dumps({'period': period, 'correct': {k: v['correct_n'] for k, v in value['scores'].items()},
            'LL': {k: v['one_x_two']['log_loss'] for k, v in value['scores'].items()},
            'contrasts': value['contrasts']}, ensure_ascii=False), flush=True)
    print(json.dumps({'phase': 'complete', 'new_fits': fits, 'reused_fits': reused,
                      'seconds': result['seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
