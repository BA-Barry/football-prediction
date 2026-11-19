"""Finite-route error complementarity and an exact score-law limitation.

No model fitting, probability edits, threshold selection or current results.
All target labels are already archived retrospective 2025/26 results.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import numpy as np
from scipy.optimize import brentq
from scipy.special import i0e

from football_model.decision_diagnostics import independent_poisson_draw_limit

ROOT = REPOSITORY_ROOT
SOURCE = ROOT / 'outputs/同日赛前模型一致化_20261027'
OUT = ROOT / 'outputs/同日误差互补与结构定位_20261027'
ROUTES = ('same_day_rich', 'same_day_goal', 'same_day_xg')
ZH = {'same_day_rich': '同日丰富', 'same_day_goal': '纯进球', 'same_day_xg': '独立xG'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tracks = read(SOURCE / '历史逐场预测.json')
    ids = [r['match_id'] for r in tracks[ROUTES[0]]]
    if len(set(ids)) != len(ids) or any([r['match_id'] for r in tracks[k]] != ids for k in ROUTES):
        raise ValueError('Comparison needs exactly aligned unique fixtures')
    labels = np.asarray([r['actual_outcome'] for r in tracks[ROUTES[0]]], dtype=int)
    for key in ROUTES:
        if [r['actual_outcome'] for r in tracks[key]] != labels.tolist():
            raise ValueError('Routes disagree on the saved target outcome')
    probs = np.asarray([[r['probabilities'] for r in tracks[k]] for k in ROUTES])
    if (not np.isfinite(probs).all() or (probs < 0).any()
            or not np.allclose(probs.sum(axis=2), 1, rtol=0, atol=1e-12)):
        raise ValueError('Need already normalized three-outcome probabilities')
    sorted_probs = np.sort(probs, axis=2)
    if np.any(sorted_probs[:, :, 2] - sorted_probs[:, :, 1] < 1e-12):
        raise ValueError('Exact/near top ties need explicit treatment, not arbitrary argmax')
    top = probs.argmax(axis=2)
    correct = top == labels
    any_right, all_right = correct.any(axis=0), correct.all(axis=0)
    same = np.all(top == top[0], axis=0)
    confusion = np.zeros((3, 3), dtype=int)
    np.add.at(confusion, (labels, top[0]), 1)
    critical = float(brentq(lambda total: float(i0e(total)) - 1/3, .01, 10))
    route_summary, rows = {}, []
    diagnostics = {key: [independent_poisson_draw_limit(*r['rates']) for r in tracks[key]] for key in ROUTES}
    for j, key in enumerate(ROUTES):
        forbidden = np.asarray([d['draw_unique_top1_impossible_at_fixed_total'] for d in diagnostics[key]])
        actual_draw = np.asarray([d['draw_probability_under_supplied_means'] for d in diagnostics[key]])
        if not np.allclose(actual_draw, probs[j, :, 1], rtol=0, atol=1e-10):
            raise ValueError('The supplied forecast is not the claimed independent-Poisson law')
        route_summary[key] = {
            'correct': int(correct[j].sum()), 'accuracy': float(correct[j].mean()),
            'draw_top1_n': int((top[j] == 1).sum()),
            'total_mean_min': min(d['total_goal_mean'] for d in diagnostics[key]),
            'total_mean_max': max(d['total_goal_mean'] for d in diagnostics[key]),
            'draw_top1_impossible_at_fixed_total_n': int(forbidden.sum()),
            'actual_draws_in_impossible_group': int((forbidden & (labels == 1)).sum())}
    switches = {}
    for j, key in enumerate(ROUTES[1:], start=1):
        rescued = correct[0] & ~correct[j]
        damaged = ~correct[0] & correct[j]
        different = top[0] != top[j]
        switches[key] = {
            'changed_direction_n': int(different.sum()), 'corrected_reference_errors': int(rescued.sum()),
            'lost_reference_correct': int(damaged.sum()),
            'both_wrong_despite_different_top1': int((different & ~correct[0] & ~correct[j]).sum()),
            'net_correct_difference': int(correct[0].sum() - correct[j].sum())}
        if int(rescued.sum()-damaged.sum()) != switches[key]['net_correct_difference']:
            raise ValueError('Switch attribution does not reconcile to overall correct counts')
    for i, row in enumerate(tracks[ROUTES[0]]):
        rows.append({k: row[k] for k in ('match_id', 'date', 'competition', 'home', 'away', 'actual_score', 'actual_outcome')} | {
            'probabilities': {k: probs[j, i].tolist() for j, k in enumerate(ROUTES)},
            'top1': {k: int(top[j, i]) for j, k in enumerate(ROUTES)},
            'correct': {k: bool(correct[j, i]) for j, k in enumerate(ROUTES)},
            'same_top1': bool(same[i]),
            'error_partition': 'all_correct' if all_right[i] else 'all_wrong' if not any_right[i] else 'complementary',
            'rich_structural_diagnostic': diagnostics['same_day_rich'][i]})
    n = len(ids)
    shared_wrong = ~any_right
    summary = {
        'generated_at': datetime.now(timezone.utc).isoformat(), 'n': n,
        'class_order': ['home', 'draw', 'away'], 'source': str(SOURCE / '历史逐场预测.json'),
        'all_correct_n': int(all_right.sum()), 'all_wrong_n': int(shared_wrong.sum()),
        'complementary_n': int((any_right & ~all_right).sum()),
        'all_wrong_actual_draws': int((shared_wrong & (labels == 1)).sum()),
        'all_wrong_actual_decisive': int((shared_wrong & (labels != 1)).sum()),
        'same_top1_n': int(same.sum()), 'same_top1_correct': int((same & correct[0]).sum()),
        'same_top1_accuracy': float(correct[0, same].mean()),
        'same_top1_mean_rich_max_probability': float(probs[0, same].max(axis=1).mean()),
        'different_top1_mean_rich_max_probability': float(probs[0, ~same].max(axis=1).mean()),
        'finite_saved_top1_oracle': {'correct_n': int(any_right.sum()), 'accuracy': float(any_right.mean()),
            'requires_target_outcome_to_choose_route': True,
            'is_implementable_forecast_or_general_model_limit': False,
            'does_not_bound_probability_pooling_or_new_decision_rules': True},
        'rich_confusion_actual_rows_predicted_columns': confusion.tolist(),
        'route_summary': route_summary, 'rich_switch_attribution': switches,
        'critical_total_mean_for_balanced_draw_equal_one_third': critical,
        'critical_equation': 'exp(-T)*I0(T)=1/3; independent Poisson only',
        'what_it_proves': 'At each fixed total above the boundary, reallocating the two independent Poisson means cannot make draw the most likely class.',
        'what_it_does_not_prove': ['draws are identifiable from current information',
            'raising all draw probabilities improves forecasts', 'every model is bounded by the finite saved-route oracle',
            'the observed common errors are caused by missing information', 'neural networks would solve this issue'],
        'new_fits': 0, 'new_probability_predictions': 0, 'new_current_results_queries': 0,
        'historical_seasons_previously_viewed': True}
    if summary['all_correct_n'] + summary['all_wrong_n'] + summary['complementary_n'] != n:
        raise ValueError('Error partition is not exhaustive')
    write(OUT / '互补与结构汇总.json', summary)
    write(OUT / '逐场错误与结构说明.json', {'records': rows})
    live = read(SOURCE / '意甲同日新版本.json')
    annotations = []
    for r in live['records']:
        annotations.append({'fixture_id': r['fixture']['match_id'], 'home': r['fixture']['home'],
            'away': r['fixture']['away'], 'prediction_generated_at': live['generated_at'],
            'probabilities_unchanged': r['model']['probabilities'],
            'structure': independent_poisson_draw_limit(*r['model']['rates']),
            'actual_outcome': None})
    write(OUT / '今晚预测的结构说明.json', {'assembled_at': summary['generated_at'],
          'original_prediction_times_retained': True, 'records': annotations, 'new_forecasts': 0})
    lines = ['# 同日模型：错误互补与比分结构定位', '',
        '本轮只分析已保存的2025/26同日预测，没有训练、调整概率或查询当前赛果。目的在于区分路线可互补的错误与共享限制，避免再按单个模型分数反复淘汰。', '',
        '## 三条路线各自学到了什么', '',
        f'在同一批{n}场上，三线共同正确{summary["all_correct_n"]}场，共同错误{summary["all_wrong_n"]}场，存在正确与错误互补{summary["complementary_n"]}场。共同错误中，实际平局{summary["all_wrong_actual_draws"]}场、主客方向反选{summary["all_wrong_actual_decisive"]}场。共享错误不能单独证明缺的是信息，也可能来自共有表示、目标或分布假设。', '',
        '| 丰富线相对参考 | 首选不同 | 纠正参考错误 | 改坏参考正确 | 净增加正确 |', '|---|---:|---:|---:|---:|']
    for key, s in switches.items():
        lines.append(f'| {ZH[key]} | {s["changed_direction_n"]} | {s["corrected_reference_errors"]} | {s["lost_reference_correct"]} | {s["net_correct_difference"]:+d} |')
    lines += ['', f'三线同首选{summary["same_top1_n"]}场，实际正确{summary["same_top1_correct"]}场（{100*summary["same_top1_accuracy"]:.2f}%）。它们共用大量历史信息及比分假设；同向组平均最高概率{100*summary["same_top1_mean_rich_max_probability"]:.2f}%，分歧组{100*summary["different_top1_mean_rich_max_probability"]:.2f}%，比赛难度不同，不能把同向解释为独立增益。这是三条统计路线，不是统计与LLM的联合历史检验。', '',
        f'如果事后知道赛果，每场只从这三条已保存的最高概率选项中挑一个，最多覆盖{int(any_right.sum())}/{n}（{100*any_right.mean():.2f}%）。这是有限候选答案的事后覆盖上限，无法赛前执行；既不限制新模型，也不限制新决策或概率融合。因此只在三条既有单选间切换，不足以把这批比赛整体命中推到70%。', '',
        '## 一个可以直接证明的结构限制', '',
        '现有三条统计线均用两队独立Poisson分布生成比分。令主客进球均值为h、a，总均值T=h+a，精确平局概率为exp(-T)×I0(2√(ha))，I0为第一类零阶修正Bessel函数。固定T时，2√(ha)≤T，故平局概率最多为exp(-T)×I0(T)，在h=a=T/2时取得。', '',
        f'方程exp(-T)×I0(T)=1/3的根约为**{critical:.6f}**。若T大于此值，平局概率连1/3都达不到；主胜与客胜至少一个大于1/3，所以无论怎样重新分配主客均值，平局都不能成为最高概率结果。这不是“没有给平局概率”，也不是证明所有平局本应预测成功。', '',
        '| 路线 | 总进球均值范围 | 固定总球下平局无法首选的场数 | 实际平局位于其中 |', '|---|---:|---:|---:|']
    for key, s in route_summary.items():
        lines.append(f'| {ZH[key]} | {s["total_mean_min"]:.3f}—{s["total_mean_max"]:.3f} | {s["draw_top1_impossible_at_fixed_total_n"]} | {s["actual_draws_in_impossible_group"]} |')
    lines += ['', f'![独立泊松的平局首选边界](../../outputs/{OUT.name}/平局首选的结构边界.png)', '',
        '丰富模型440场实际平局都处于这个结构条件中。只改主客进球分配仍无效；下调总球率虽可能越过边界，但会同时改变总球与比分概率，不能仅为抬平而调整，效果需单独评价。依赖比分或重新校准的模型不受这条独立Poisson公式直接限制；摆脱限制只是必要能力，不代表具备识别平局的有效信息。', '',
        '## 对既有工作的具体含义', '',
        '单温度不会改变H/D/A的排序；既有条件正平局研究只使用截距与强弱差两个输入，虽可改变比分形状，但保存结果没有改变首选。它们不是无价值资产：前者可校准概率，后者证明比分结构可以在原模型外调整；但不能把同一个修正再次包装为平局辨别提升。', '',
        '本轮确认下一步需要讨论的是“哪些赛前条件支持不同的胜平负／联合比分形状”，而不是继续微调近期权重或在三个单选之间选择。现有121项表现、时间外预测与score_reconciliation可复用；LLM需单列，当前没有同口径的1732场历史LLM结果。神经网络并非由本诊断自然推导出的必需品。', '',
        '本轮到一次完整错误分解与结构测量为止。保留全部互补和失败记录，不学习新阈值、不修改概率、不据同向自动提高置信度，今晚三场的原生成时间与结果留空状态均保留。新加入decision_diagnostics只输出数学诊断，不参与胜平负计算。', '',
        f'[逐场错误](../../outputs/{OUT.name}/逐场错误与结构说明.json)；[今晚结构说明](../../outputs/{OUT.name}/今晚预测的结构说明.json)；[下一步具体整合方案（待讨论）](../../outputs/{OUT.name}/下一步核心整合方案_待讨论.md)。全部新增资料在项目输出目录。']
    (OUT / '本轮定位与下一步.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('n', 'all_correct_n', 'all_wrong_n', 'complementary_n', 'finite_saved_top1_oracle',
                     'critical_total_mean_for_balanced_draw_equal_one_third')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
