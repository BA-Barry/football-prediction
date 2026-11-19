"""One fixed EPL research arm: archived statuses intersect earlier usage.

This reuses the completed EPL-only control, fits no base expert, and does not
interpret a minute share as ability, missing strength or expected minutes.
"""
from datetime import date, datetime, timezone
import gc
import gzip
import json
from pathlib import Path
import time

import joblib
import numpy as np

from football_model.early_goal_learners import EarlyGoalLearner
from run_adjusted_xg_form_20261026 import compare
from run_broad_early_models_20261025 import predict_row
from run_fixed_horizon_refresh_20261025 import CONTEXT_KEYS
from intake_fpl_availability_20261026 import ROOT, write


PERIODS = ('2024-25', '2025-26')
INTAKE = ROOT/'outputs/FPL可用性变化与参与接入_20261026'
INPUT = INTAKE/'逐场变化参与特征.json.gz'
REFERENCE = ROOT/'outputs/FPL赛前可用性固定对照_20261026T090728310203Z/逐场完整预测.json'
REFERENCE_INPUT = ROOT/'outputs/FPL历史可用性接入_20261026/逐场赛前状态特征.json.gz'
SOURCE_COVERAGE = INTAKE/'接入实测.json'


def load_rows():
    with gzip.open(INPUT, 'rt', encoding='utf-8') as handle:
        rows = json.load(handle)['rows']
    rows.sort(key=lambda r: (r['date'], r['match_id']))
    return rows


def feature_coverage(rows, extra_names):
    result = {}
    for period in sorted({r['period'] for r in rows}):
        cohort = [r for r in rows if r['period'] == period]
        missing = {name: sum(side.get(name) is None for row in cohort for side in row['side_features'])
                   for name in extra_names}
        result[period] = {'n_matches': len(cohort), 'scoring_sides': 2*len(cohort),
            'both_sides_all_added_columns_observed': sum(
                all(side.get(name) is not None for side in row['side_features'] for name in extra_names)
                for row in cohort), 'missing_scoring_sides_by_added_column': missing}
    return result


def ci_for(comparison, metric):
    return dict(zip(comparison['difference_order'],
                    comparison['conditional_4week_intervals']['interval_95']))[metric]


def decision_for(result):
    values = list(result['by_period'].values())
    deltas = [value['mean_differences_adjusted_minus_original']['one_x_two'] for value in values]
    if any(delta >= 0 for delta in deltas):
        finding = '两季胜平负概率损失未同向改善，未确认稳定增量。'
    elif any(ci_for(value, 'one_x_two')[1] >= 0 for value in values):
        finding = '两季平均胜平负概率损失均下降，但至少一季的4周条件区间未完整落在改善方向，尚不能确认可靠增益。'
    else:
        finding = '两季平均胜平负概率损失均下降，条件分块区间也支持同方向；仍属于已查看历史上的研究结果，不是新的盲测或当前比赛验证。'
    pooled_late = result['pooled_reference_by_period']['2025-26']['mean_differences_adjusted_minus_original']['one_x_two']
    finding += ('较晚季仍未超过原五联赛模型。' if pooled_late >= 0 else '较晚季相对原五联赛模型的概率评分也有改善。')
    return {'adopted': False, 'description': finding,
        'reason_current_not_replaced': 'independent research arm, earlier-information history only; current 5-7 day targets not rescored by this experiment',
        'stop_this_comparison': True, 'planned_additional_fits_this_arm': 0,
        'scope': 'descriptive status/earlier-usage interaction, not individual contribution, injury severity or causal replacement effects'}


def report(out, result):
    lines = ['# 可用性变化与此前参与：固定设置历史对照', '',
        '本轮将赛前FPL标记连接到同源已核对球员编号和更早实际参与记录。'
        '参与份额只描述过去如何使用球员，不是能力、预计分钟或因果伤停损失；'
        '没有等待目标首发，也没有把目标比赛实际分钟、结果或换人用于特征。', '',
        '每队11项测量，自己与对手合计新增22列。前三项为当前i/d/s标记对应的较早参与份额，'
        '另三项为相对原模型最近5场较早赛前状态的同权重差；其余为状态可解释覆盖和未知/近期零参与计数。'
        '另单列当前源a、过去5个状态可解释且曾有i/d/s、完整最近5场分钟为零的人数；该项只描述标记变化，不确认康复。'
        '这不是给球队边际进球率直接乘一个球员系数。', '',
        '## 两季完整同场对照', '',
        '| 赛季 | 场数 | 原英超121列LL | 新143列LL | LL差 | 原单选命中 | 新单选命中 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for period, value in result['by_period'].items():
        old, new = value['original']['one_x_two'], value['adjusted']['one_x_two']
        delta = value['mean_differences_adjusted_minus_original']
        lines.append(f"| {period} | {value['n']} | {old['log_loss']:.6f} | {new['log_loss']:.6f} | {delta['one_x_two']:+.6f} | {old['accuracy']:.2%} | {new['accuracy']:.2%} |")
    lines += ['', 'LL差为新减旧，负数更好；命中只取胜、平、负中概率最高的一项。两季各377场全部保留，'
        '不以状态齐全或球员已知为由删场。原英超模型直接复用上一轮保存的预测，没有重新训练。', '',
        '| 赛季 | LL差4周条件区间 | Brier差 | 比分NLL差 | 总球NLL差 | 相对原五联赛LL差 |',
        '|---|---|---:|---:|---:|---:|']
    for period, value in result['by_period'].items():
        delta = value['mean_differences_adjusted_minus_original']
        ci = ci_for(value, 'one_x_two')
        pooled_delta = result['pooled_reference_by_period'][period]['mean_differences_adjusted_minus_original']['one_x_two']
        lines.append(f"| {period} | [{ci[0]:+.6f}, {ci[1]:+.6f}] | {delta['one_x_two_brier']:+.6f} | {delta['exact_score']:+.6f} | {delta['total_goals_0_1_2_3_4plus']:+.6f} | {pooled_delta:+.6f} |")
    lines += ['', '区间是既有逐场预测下的时间分块波动，不包含完整模型重新训练、数据修订或全部参数不确定性。'
        '两季历史此前均被研发查看，本轮是严格前推的探索研究，不是全新盲测。', '',
        '## 特征可观测性', '']
    source_coverage = result['source_availability_coverage']
    lines += ['**2022/23没有本轮所需的Understat实际参与窗口；该季这些参与测量全部未知，不能称四季完整球员分钟库。**'
        '2023/24起资料才逐渐齐备。原球队121列历史仍保留，新增参与缺失不会删除训练比赛或目标比赛。', '',
        '| 赛季 | 球队侧数 | 精确匹配原最近5场分钟窗口完整 | 对应历史状态有正覆盖 |',
        '|---|---:|---:|---:|']
    for period, coverage in source_coverage['by_period'].items():
        lines.append(f"| {period} | {coverage['team_sides']} | {coverage['complete_exact_minute_window']} | {coverage['sides_with_positive_paired_coverage']} |")
    lines += ['', '“完整”只指这项研究所需的原5场比赛能匹配已存来源分钟；不是秒级时钟、完整全赛事履历或个人效应已被估计。'
        '球队侧不是独立比赛，正覆盖也不等于全部球员状态都已知。', '']
    for period, coverage in result['feature_coverage'].items():
        lines.append(f"- {period}：{coverage['n_matches']}场；双方全部新增列均可观测{coverage['both_sides_all_added_columns_observed']}场。各列缺失侧数完整记录于结果JSON。")
    lines += ['', '未知和覆盖不足保留None，进入原有仅训练期拟合的插补与缺失标记流程，'
        '没有把没有较早同队记录的球员当已知零贡献。较早球队分钟是Understat整数分钟计数，'
        '不是秒级暴露；份额使用来源实际合计，不强补990。', '',
        '当前i/d/s是源标签，不证明实际缺阵或严重程度；较早a也不证明健康。'
        '相对历史状态变化只描述被观察标签的差异，不证明伤情首次出现、恢复时刻，'
        '也不证明原球队表现已经完整吸收其影响。FPL百分比、news_added、赛后目标首发和累计FPL分钟均未进入。', '',
        '## 决定与停止节点', '', result['decision']['description'], '',
        '本轮完成22个原日期的固定ExtraTrees模型训练，直接复用英超121列和原五联赛预测；'
        '零参数网格、零基专家重训、零额外提前量。完成后停止，不按目标结果删变量、改窗口或搜索权重。'
        '个人贡献、替补补偿与未知阵容情景尚未估计，不能将本轮输入称为球员实力模型。', '',
        '当前比赛、LLM与联合线预测均未替换，未复核当前新赛果。所有数据、模型与报告均在G盘。', '',
        '来源：[FPL历史快照](https://github.com/Randdalf/fplcache)、'
        '[事实球员编号对照](https://github.com/ChrisMusson/FPL-ID-Map/blob/main/Master.csv)，'
        '以及已保存、按各截点严格截断的Understat球队参与记录。']
    (out/'结果与决定.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    started = time.perf_counter()
    now = datetime.now(timezone.utc)
    rows = load_rows()
    with gzip.open(REFERENCE_INPUT, 'rt', encoding='utf-8') as handle:
        reference_rows = {r['match_id']: r for r in json.load(handle)['rows']}
    old = json.loads(REFERENCE.read_text(encoding='utf-8'))
    control, pooled = old['epl_control'], old['original_pooled']
    for values in (control, pooled):
        values.sort(key=lambda r: (r['date'], r['match_id']))
    targets = [r for r in rows if r['period'] in PERIODS]
    target_ids = [r['match_id'] for r in targets]
    if len(rows) != 1502 or len(set(r['match_id'] for r in rows)) != len(rows):
        raise ValueError('Expected the existing unique 1502-row EPL history')
    if target_ids != [r['match_id'] for r in control] or target_ids != [r['match_id'] for r in pooled]:
        raise ValueError('Prepared, EPL control and pooled targets must have identical complete support')
    extra_names = sorted(set(rows[0]['side_features'][0])-set(rows[0]['reference_features'][0]))
    if len(extra_names) != 22:
        raise ValueError('Expected exactly the declared twenty-two descriptive input columns')
    for row in rows:
        old_row = reference_rows.get(row['match_id'])
        if old_row is None or any(old_row[k] != row[k] for k in CONTEXT_KEYS) or old_row['reference_features'] != row['reference_features']:
            raise ValueError('Reusing saved control requires identical historical 121-column inputs and contexts')
        if row['competition'] != 'E0' or row['horizon_days'] != 3 or not row['fit_before_date'] <= row['forecast_before_date'] < row['date']:
            raise ValueError('Existing EPL publication/fit timing must be retained')
        for full, reference in zip(row['side_features'], row['reference_features'], strict=True):
            if len(full) != 143 or len(reference) != 121 or any(full.get(k) != v for k, v in reference.items()):
                raise ValueError('Original 121 input measurements must be unchanged')
    out = ROOT/'outputs'/now.strftime('FPL可用性变化参与固定对照_%Y%m%dT%H%M%S%fZ')
    (out/'models').mkdir(parents=True)
    method = {
        'created_at': now.isoformat(), 'scope': 'EPL only, already-viewed historical seasons, not untouched validation',
        'input': str(INPUT), 'reference_predictions': str(REFERENCE), 'reference_inputs': str(REFERENCE_INPUT),
        'source_coverage_report': str(SOURCE_COVERAGE),
        'new_information': 'current source statuses intersect strictly earlier same-team usage and paired historical source statuses',
        'added_feature_names': extra_names, 'original_features': 121, 'new_features': 143,
        'pre_fit_definition_completion': {
            'extra_measurement': 'now_a_after_old_flag_zero_window_minutes_count',
            'reason': 'A long-flagged player now labelled a can have zero minutes in the entire last-five reference window; a weighted status change alone would hide that observed change. Preserve a separate count, not a zero-strength assumption or proof of recovery.',
            'timing': 'added before any fit or new evaluation result; one existing arm and same 22 model dates'},
        'control': 'reuse previously completed same-preset EPL-only 121-column predictions on identical 754 target IDs',
        'pooled_reference': 'reuse five-league model EPL predictions on the same 754 IDs, report separately',
        'target': 'real full-time goals, unchanged independent Poisson joint score conversion, rho=0',
        'configuration': EarlyGoalLearner('extra_trees', 'rich')._estimator().get_params(),
        'half_life_days': 180, 'cutoffs': '22 original Aug1-anchored 28-day dates; each row F=D-3, only match date<P enters fits',
        'new_fits_planned': 22, 'base_expert_refits': 0, 'new_control_fits': 0, 'new_parameter_grids': 0,
        'missing': 'all targets retained, None unknown, existing training-only median imputation plus missing indicators',
        'primary': '1X2 LL new minus matched EPL control, each season separately',
        'secondary': ['1X2 Brier', 'single highest-probability outcome accuracy', 'score NLL', 'total-goal NLL', 'pooled reference difference'],
        'interval': 'paired conditional 4-calendar-week moving blocks, no complete refitting uncertainty',
        'not_inputs': ['target XI', 'target minutes or substitutions', 'player ability weights', 'chance percent', 'FPL cumulative minutes', 'news text inference'],
        'stopping_point': 'all 22 existing cutoffs and complete target seasons, then stop without window/variable/weight searches',
        'interpretation': 'usage and source labels, not injury severity, recovery, expected minutes, causal player impact or proof of full absorption by team history',
        'limitations': ['archive Git timestamps are retrospective availability proxies, not first public announcement proof',
            '2022/23 has no Understat appearance windows for this study; its participation inputs remain unknown, not zero',
            'old appearance publication times unknown; calendar-date earlier availability assumed',
            'community cross-provider IDs are identity reconciliation, not official accuracy guarantees',
            'adding features changes randomized tree split opportunities, not a causal ablation',
            'current targets are 5-7 days ahead whereas this historical study is 3 days ahead'],
        'current_forecasts_replaced': False, 'current_results_checked': False}
    write(out/'分析前方法.json', method)
    predictions, windows, parameters = [], [], {}
    serialization = None
    for period in PERIODS:
        cohort = [r for r in targets if r['period'] == period]
        for cutoff in sorted({r['fit_before_date'] for r in cohort}):
            tick = time.perf_counter()
            held = [r for r in cohort if r['fit_before_date'] == cutoff]
            history = [r for r in rows if r['date'] < cutoff]
            model = EarlyGoalLearner('extra_trees', 'rich').fit(history, date.fromisoformat(cutoff))
            path = out/'models'/f'availability_exposure_{cutoff}.joblib'
            joblib.dump(model, path, compress=3)
            parameters[cutoff] = model.parameters()
            rates = model.predict_many([r['side_features'] for r in held], [date.fromisoformat(r['date']) for r in held])
            if serialization is None:
                replay = joblib.load(path).predict_many([held[0]['side_features']], [date.fromisoformat(held[0]['date'])])[0]
                if not np.allclose(replay, rates[0], rtol=0, atol=1e-12):
                    raise ValueError('Saved/restored estimator changed prediction')
                serialization = {'match_id': held[0]['match_id'], 'original': list(rates[0]), 'restored': list(replay)}
            for row, means in zip(held, rates, strict=True):
                context = {k: row[k] for k in CONTEXT_KEYS}
                predictions.append(predict_row(context, means))
            windows.append({'period': period, 'cutoff': cutoff, 'training_n': len(history),
                'target_n': len(held), 'seconds': time.perf_counter()-tick})
            write(out/'训练进度.json', {'completed_fits': len(windows), 'windows': windows})
            print(json.dumps(windows[-1] | {'completed_fits': len(windows)}, ensure_ascii=False), flush=True)
            del model
            gc.collect()
    predictions.sort(key=lambda r: (r['date'], r['match_id']))
    result = {'method': method, 'new_fits': len(windows), 'base_expert_refits': 0, 'new_control_fits': 0,
        'parameter_grids': 0, 'windows': windows, 'serialization_check': serialization,
        'source_availability_coverage': json.loads(SOURCE_COVERAGE.read_text(encoding='utf-8')),
        'feature_coverage': feature_coverage(rows, extra_names), 'by_period': {}, 'pooled_reference_by_period': {}}
    for period in PERIODS:
        original = [r for r in control if r['period'] == period]
        new = [r for r in predictions if r['period'] == period]
        reference = [r for r in pooled if r['period'] == period]
        if len(original) != 377 or len(new) != 377:
            raise ValueError('Expected complete 377-match target season')
        result['by_period'][period] = compare(original, new, intervals=True)
        result['pooled_reference_by_period'][period] = compare(reference, new)
    result['decision'] = decision_for(result)
    result['seconds'] = time.perf_counter()-started
    write(out/'全部训练参数.json', parameters)
    write(out/'逐场完整预测.json', {'epl_control': control, 'availability_exposure': predictions, 'original_pooled': pooled})
    write(out/'完整研究结果.json', result)
    report(out, result)
    for period, value in result['by_period'].items():
        print(json.dumps({'period': period, 'n': value['n'], 'control': value['original']['one_x_two'],
            'new': value['adjusted']['one_x_two'], 'deltas': value['mean_differences_adjusted_minus_original'],
            'll_interval': ci_for(value, 'one_x_two')}, ensure_ascii=False), flush=True)
    print(json.dumps({'output': str(out), 'new_fits': len(windows), 'decision': result['decision'], 'seconds': result['seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
