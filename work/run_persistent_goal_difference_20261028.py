"""One fixed persistent Skellam/GAS study using saved full-time goal records.

This runner does not download data, query results, tune variants, or replace
current predictions. Previously examined seasons remain development evidence.
"""
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
from sklearn.metrics import roc_auc_score

from football_model.forecast_diagnostics import moving_block_means
from football_model.persistent_goal_difference import fit_static_panel, fit_dynamics, forecast_targets


ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OUT = ROOT / 'outputs/持续状态净胜球研究_20261028'
INPUT = OUT / '输入资料.json.gz'
SEASONS = ('2024/25', '2025/26')
COMPETITIONS = ('E0', 'D1', 'SP1', 'I1', 'F1')
NAMES = ('主胜', '平局', '客胜')


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def zipped(path, value):
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)


def read_input():
    with gzip.open(INPUT, 'rt', encoding='utf-8') as stream:
        data = json.load(stream)
    if data['metadata']['target_unknown_rows']:
        raise ValueError('Do not silently omit or repair unknown original targets')
    if len(data['targets']) != 3452 or len({r['match_id'] for r in data['targets']}) != 3452:
        raise ValueError('Keep the complete existing 3452-target cohort')
    return data


def execution_plan(data):
    return {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'input_file': str(INPUT), 'trusted_history_matches': len(data['matches']),
        'targets': len(data['targets']), 'scope': 'all five leagues and both complete saved three-day target cohorts',
        'static': {'season': '2022/23', 'fits': 1, 'ridge': 4.,
                   'method': 'centered attacks/defences plus league mu/home; penalized Skellam net-goal likelihood'},
        'dynamics': {'season': '2023/24', 'fits': 1, 'parameter_count': 4,
            'names': ['attack_step', 'defence_step', 'attack_retention_week', 'defence_retention_week'],
            'initial': [.04, .04, .97, .97], 'bounds': [[0., .2], [0., .2], [.5, 1.], [.5, 1.]],
            'method': 'one L-BFGS-B start; subsequent parameter values fixed',
            'interpretation': 'conditional score-driven point estimates, no posterior uncertainty'},
        'state': 'deviations decay toward static anchors by calendar days, persist across seasons; observe complete days together',
        'chronology': 'replay static-post history; observe match date strictly before each target F=D-3; no target or same-day result enters itself',
        'unknown_teams': 'before first observed match, GAS publicly unavailable and deployment copies original HDA; first internal observation uses neutral algorithm coordinates, not an estimated prior',
        'comparison': 'full deployment vs original on all 3452; GAS vs original separately on identical available targets',
        'metrics': ['full-cohort Top1', '1X2 log loss', 'three-class sum Brier', 'draw AUC/mean/selection',
                    'opposite H/A directions', 'conditional H/A diagnostic on actual non-draws', 'corrected/harmed decomposition'],
        'intervals': 'existing four-calendar-week paired conditional blocks, 2000 repeats; not full fitting/selection uncertainty',
        'comparison_limit': 'new candidate sees only goals/net goals, rich original also sees xG and richer history; whole-candidate comparison, not causal isolation of persistent state',
        'stop': 'one static fit, one four-parameter optimization, one complete replay and report; numerical failure recorded without rescue variants',
        'parameter_grid': 0, 'new_calibration': 0, 'current_results_queries': 0,
        'network_requests': 0, 'current_predictions_replaced': False,
        'historical_scope': 'previously examined development seasons, not fresh blind testing',
        'output_limit': 'net-goal/HDA distribution only; no full joint scores or total-goal forecast reconstructed'}


def arrays(rows, field):
    p = np.asarray([r[field] for r in rows], dtype=float)
    if p.shape != (len(rows), 3) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError(f'Invalid H/D/A values: {field}')
    if len(rows) and not np.allclose(p.sum(axis=1), 1., rtol=0, atol=1e-10):
        raise ValueError(f'H/D/A totals differ from one: {field}')
    y = np.asarray([0 if r['hg'] > r['ag'] else 2 if r['hg'] < r['ag'] else 1 for r in rows])
    return p, y


def metrics(rows, field):
    if not rows:
        return {'n': 0, 'unavailable_reason': 'no_rows', 'top1_correct': 0,
                'top1_accuracy': None, 'log_loss': None, 'brier_sum': None}
    p, y = arrays(rows, field)
    selected = np.argmax(p, axis=1)
    correct = selected == y
    # The common numerical floor is only used for loss evaluation. Saved
    # probabilities are never renormalized, clipped, or written back.
    nll = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-15))
    brier = np.sum((p-np.eye(3)[y])**2, axis=1)
    is_draw = y == 1
    choose_draw = selected == 1
    non_draw = ~is_draw
    conditional = np.where(p[:, 0] >= p[:, 2], 0, 2)
    confusion = np.zeros((3, 3), dtype=int)
    np.add.at(confusion, (y, selected), 1)
    opposite = non_draw & (selected != 1) & (selected != y)
    conditional_wrong = non_draw & (conditional != y)
    return {'n': len(rows), 'top1_correct': int(correct.sum()), 'top1_accuracy': float(correct.mean()),
        'log_loss': float(nll.mean()), 'brier_sum': float(brier.mean()),
        'loss_probability_floor': 1e-15,
        'loss_floor_applied_rows': int((p[np.arange(len(y)), y] < 1e-15).sum()),
        'draw_auc': float(roc_auc_score(is_draw, p[:, 1])) if len(set(is_draw)) == 2 else None,
        'mean_draw_probability': float(p[:, 1].mean()), 'actual_draw_count': int(is_draw.sum()),
        'actual_draw_rate': float(is_draw.mean()), 'selected_draw_count': int(choose_draw.sum()),
        'selected_draw_correct': int((choose_draw & is_draw).sum()),
        'selected_draw_precision': float((choose_draw & is_draw).sum()/choose_draw.sum()) if choose_draw.any() else None,
        'draw_recall': float((choose_draw & is_draw).sum()/is_draw.sum()) if is_draw.any() else None,
        'actual_non_draw_count': int(non_draw.sum()),
        'opposite_home_away_top1_count': int(opposite.sum()),
        'non_draw_actual_predicted_draw_count': int((non_draw & choose_draw).sum()),
        'conditional_home_away_correct': int((non_draw & (conditional == y)).sum()),
        'conditional_home_away_accuracy': float((conditional[non_draw] == y[non_draw]).mean()) if non_draw.any() else None,
        'conditional_home_away_opposite_count': int(conditional_wrong.sum()),
        'conditional_home_away_scope': 'diagnostic on actual non-draw matches, always compare pH against pA; not an executable pre-match selection of actual non-draws',
        'class_order': ['home', 'draw', 'away'], 'confusion_actual_rows_predicted_columns': confusion.tolist(),
        'selected_by_class': np.bincount(selected, minlength=3).tolist(),
        'correct_by_actual_class': [int((correct & (y == i)).sum()) for i in range(3)]}


def paired_decomposition(rows, new_field='deployment_probabilities'):
    if not rows:
        return {'n': 0}
    old, y = arrays(rows, 'original_hda')
    new, yn = arrays(rows, new_field)
    if not np.array_equal(y, yn):
        raise ValueError('Paired target labels differ')
    a, b = np.argmax(old, axis=1), np.argmax(new, axis=1)
    ac, bc = a == y, b == y
    original_opposite = (y != 1) & (a != 1) & (a != y)
    return {'n': len(rows), 'top1_changed': int((a != b).sum()),
        'both_correct': int((ac & bc).sum()), 'only_new_correct': int((~ac & bc).sum()),
        'only_original_correct': int((ac & ~bc).sum()), 'both_wrong': int((~ac & ~bc).sum()),
        'both_wrong_changed': int((~ac & ~bc & (a != b)).sum()),
        'net_correct_change': int(bc.sum()-ac.sum()),
        'original_opposite_home_away_changed_to_draw': int((original_opposite & (b == 1)).sum()),
        'by_actual_class': [{'class': name, 'actual_count': int((y == i).sum()),
            'original_correct': int((ac & (y == i)).sum()), 'new_correct': int((bc & (y == i)).sum()),
            'corrected': int((~ac & bc & (y == i)).sum()), 'harmed': int((ac & ~bc & (y == i)).sum())}
            for i, name in enumerate(NAMES)],
        'prediction_transitions': dict(Counter(f'{NAMES[x]}->{NAMES[z]}' for x, z in zip(a, b))),
        'interpretation': 'saved-prediction descriptive decomposition, not causal attribution or post-hoc branch selection'}


def paired_intervals(rows):
    p0, y = arrays(rows, 'original_hda')
    p1, _ = arrays(rows, 'deployment_probabilities')
    correct0 = (p0.argmax(axis=1) == y).astype(float)
    correct1 = (p1.argmax(axis=1) == y).astype(float)
    ll0 = -np.log(np.maximum(p0[np.arange(len(y)), y], 1e-15))
    ll1 = -np.log(np.maximum(p1[np.arange(len(y)), y], 1e-15))
    bs0 = np.sum((p0-np.eye(3)[y])**2, axis=1)
    bs1 = np.sum((p1-np.eye(3)[y])**2, axis=1)
    result = moving_block_means([r['date'] for r in rows],
        np.column_stack((correct1-correct0, ll1-ll0, bs1-bs0)), block_weeks=4)
    return {'columns': ['accuracy_new_minus_original', 'log_loss_new_minus_original', 'brier_sum_new_minus_original'],
            'scope': 'conditional on saved forecasts and learned parameters; repeated-development history; no independent confirmation',
            **result}


def summarize_rows(rows, *, intervals=True):
    available = [r for r in rows if r['gas_available']]
    unavailable = [r for r in rows if not r['gas_available']]
    result = {'n': len(rows), 'gas_available_n': len(available), 'fallback_n': len(unavailable),
        'original': metrics(rows, 'original_hda'), 'deployment': metrics(rows, 'deployment_probabilities'),
        'deployment_vs_original': paired_decomposition(rows),
        'gas_available_same_cohort': {'n': len(available), 'original': metrics(available, 'original_hda'),
            'gas': metrics(available, 'gas_probabilities'),
            'gas_vs_original': paired_decomposition(available, 'gas_probabilities')},
        'unknown_team_occurrences': dict(Counter(t for r in unavailable for t in r['unknown_teams']))}
    if intervals and rows:
        result['paired_four_week_intervals'] = paired_intervals(rows)
    return result


def analyze_predictions(predictions, static, dynamics, elapsed=None):
    summary = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'full_cohort': summarize_rows(predictions, intervals=False),
        'by_season': {season: summarize_rows([r for r in predictions if r['season'] == season]) for season in SEASONS},
        'by_competition': {c: summarize_rows([r for r in predictions if r['competition'] == c], intervals=False) for c in COMPETITIONS},
        'static_optimizer': static['optimizer'], 'dynamic_optimizer': dynamics['optimizer'],
        'parameters': dynamics['parameters'], 'elapsed_seconds': elapsed,
        'notes': ['All targets retain the original three-calendar-day information horizon.',
                  'Fallback is original H/D/A, not an estimated GAS prediction for an unobserved team.',
                  'No posterior distribution, player or lineup uncertainty, market, LLM or contextual shots were used.',
                  'New goal-only candidate vs richer original compares complete candidates, not a causal persistence ablation.',
                  'Previously reviewed historical seasons are development evidence, not new blind testing.',
                  'Net-goal model does not identify full joint scores or total-goal distributions.']}
    diagnostic_file = OUT / '最终点数值诊断.json'
    if diagnostic_file.exists():
        summary['final_point_numerical_diagnostic'] = json.loads(diagnostic_file.read_text(encoding='utf-8'))
    summary['selection_change_csv'] = export_selection_changes(predictions)
    write(OUT / '完整结果与分解.json', summary)
    render_report(summary)
    return summary


def export_selection_changes(predictions):
    """Export every changed top choice; probabilities remain three separate values."""
    path = OUT / '逐场改向与纠正改坏.csv'
    fields = ['match_id', 'date', 'F', 'competition', 'season', 'home', 'away', 'hg', 'ag',
              'actual', 'original_home_probability', 'original_draw_probability', 'original_away_probability',
              'candidate_home_probability', 'candidate_draw_probability', 'candidate_away_probability',
              'original_choice', 'candidate_choice', 'change_type', 'gas_available', 'unknown_teams']
    counts = Counter()
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in predictions:
            old, new = row['original_hda'], row['deployment_probabilities']
            original_choice, candidate_choice = int(np.argmax(old)), int(np.argmax(new))
            if original_choice == candidate_choice:
                continue
            actual = 0 if row['hg'] > row['ag'] else 2 if row['hg'] < row['ag'] else 1
            change_type = ('纠正' if candidate_choice == actual else
                           '改坏' if original_choice == actual else '双方均错但改向')
            counts[change_type] += 1
            record = {k: row[k] for k in ('match_id', 'date', 'F', 'competition', 'season',
                                          'home', 'away', 'hg', 'ag', 'gas_available')}
            record.update({'actual': NAMES[actual], 'original_choice': NAMES[original_choice],
                'candidate_choice': NAMES[candidate_choice], 'change_type': change_type,
                'unknown_teams': '|'.join(row['unknown_teams'])})
            for prefix, values in (('original', old), ('candidate', new)):
                for outcome, probability in zip(('home', 'draw', 'away'), values, strict=True):
                    record[f'{prefix}_{outcome}_probability'] = probability
            writer.writerow(record)
    return {'path': str(path), 'rows': sum(counts.values()), 'change_counts': dict(counts),
            'scope': 'all changed top choices from complete cohort; unchanged choices remain in the full probability file',
            'probabilities': 'home/draw/away independently copied without rounding or merging'}


def render_report(summary):
    lines = []
    for season, row in summary['by_season'].items():
        for track, label in (('original', '原丰富模型'), ('deployment', '持续状态候选＋必要回退')):
            m = row[track]
            lines.append(f"|{season}|{label}|{m['top1_correct']}/{m['n']}|{m['top1_accuracy']:.2%}|{m['log_loss']:.6f}|{m['brier_sum']:.6f}|{m['draw_auc']:.4f}|{m['selected_draw_count']}/{m['selected_draw_correct']}|")
    detail = []
    for season, row in summary['by_season'].items():
        original, new = row['original'], row['deployment']
        d = row['deployment_vs_original']
        ci = row['paired_four_week_intervals']['interval_95']
        detail.append(f"### {season}\n\nGAS可用{row['gas_available_n']}/{row['n']}，原线回退{row['fallback_n']}场。纠正{d['only_new_correct']}场、改坏{d['only_original_correct']}场，首选改变{d['top1_changed']}场。\n\n"
            f"平均平局概率{original['mean_draw_probability']:.2%}→{new['mean_draw_probability']:.2%}，实际平局{new['actual_draw_count']}/{new['n']}（{new['actual_draw_rate']:.2%}）。表中的“选平／选对”分别是首选平局场数与其中正确数。\n\n"
            f"主客方向反选（首选为主或客且选反）{original['opposite_home_away_top1_count']}→{new['opposite_home_away_top1_count']}；实际非平却选平{original['non_draw_actual_predicted_draw_count']}→{new['non_draw_actual_predicted_draw_count']}。实际非平比赛中，仅比较主／客相对概率的条件诊断为{original['conditional_home_away_correct']}/{original['actual_non_draw_count']}→{new['conditional_home_away_correct']}/{new['actual_non_draw_count']}，不能在赛前据此筛出实际非平比赛。\n\n"
            f"四周条件区间：命中率差[{ci[0][0]*100:+.2f},{ci[0][1]*100:+.2f}]个百分点；LL差[{ci[1][0]:+.6f},{ci[1][1]:+.6f}]。区间只描述已存预测及参数，不覆盖训练与研发选择不确定性。\n")
    seasons = list(summary['by_season'].values())
    acc = [r['deployment']['top1_correct']-r['original']['top1_correct'] for r in seasons]
    ll = [r['deployment']['log_loss']-r['original']['log_loss'] for r in seasons]
    if all(v < 0 for v in acc) and all(v > 0 for v in ll):
        conclusion = '两季单选命中率点值均下降，胜平负LL也均变差；这次固定候选没有带来整体提准，不据此替换当前预测。'
    elif all(v > 0 for v in acc) and all(v < 0 for v in ll):
        conclusion = '两季单选数量和概率损失点值均改善，值得作为完整候选继续前瞻观察；这些反复开发的历史结果仍不能确认当前或未来准确率。'
    elif all(v < 0 for v in ll):
        conclusion = '两季概率损失点值改善，但单选没有跨季一致增加，不能宣布胜平负单选提准。'
    else:
        conclusion = '未得到跨季一致的单选与概率质量改善；这次固定候选尚未形成整体提准证据，不据此替换当前预测。'
    full = summary['full_cohort']
    old_full, new_full = full['original'], full['deployment']
    changed = full['deployment_vs_original']
    class_deltas = [new_full['correct_by_actual_class'][i]-old_full['correct_by_actual_class'][i] for i in range(3)]
    coverage = (f"本次GAS实际覆盖全部{full['gas_available_n']:,}场，原线回退{full['fallback_n']}场；全队列候选就是纯GAS，两个比较口径本次完全一致。"
                if not full['fallback_n'] else
                f"本次GAS覆盖{full['gas_available_n']:,}/{full['n']:,}场，另{full['fallback_n']}场逐值回退原H/D/A。")
    report = f'''# 持续状态净胜球模型：实际运行与判断

2026-10-28。{conclusion}

## 本轮完整方案

一次2022/23静态初始化，估计各联赛净胜球潜在率截距、主场效应及居中球队攻防锚点，固定ridge=4；一次2023/24四参数拟合，之后参数固定。模型允许球队偏差按实际日历天衰减并跨季延续，每个观察日的比赛统一使用日初状态，再按净胜球似然得分更新。新队首次观察前公开未知、部署逐值回退原H/D/A；首次内部观察使用中性算法坐标，并非经验估计的新队先验。

这是有持续状态的条件点估计，尚无后验不确定性。新线仅使用真实比分／净胜球，原丰富模型还含xG等信息，因此下表比较完整候选与原线，不能单独归因于持续状态机制。净胜球概率能给出胜平负，不能唯一恢复完整比分、总进球或半全场。

静态优化{summary['static_optimizer']['iterations']}次迭代、{summary['static_optimizer']['evaluations']}次目标评估；动态优化{summary['dynamic_optimizer']['iterations']}次迭代、{summary['dynamic_optimizer']['evaluations']}次目标评估。四参数见动态拟合JSON和完整汇总。一次拟合调用不等于一次目标函数评估；没有网格、重启、事后权重或新校准。

本次两项每周保留率均估计为1.0的允许上界，表示早期训练样本偏好无日历衰减、持续累积更新，不能声称已经学会休赛期回归。动态优化按ftol停止；上界处未约束原始梯度可能较大，须结合约束方向下的投影梯度解释。最终点的单次局部数值诊断另存，未改参数或重新拟合。Skellam两项潜在率按净胜球似然估计，不直接等同于真实主客进球均值。

## 完整历史队列

|赛季|线路|正确场数|命中率|胜平负LL↓|Brier求和↓|平局AUC|选平／选对|
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

两个原目标季各1,726场，合计3,452场全部保留。{coverage}接口仍保留未知队伍回退处理，不能把接口能力写成本次实际发生过回退。细分结果和各联赛分解保存在完整JSON。

对数损失仅在评估时共享1e-15数值下限，本次原线与候选均触发0场，下限没有改变实际分数；存储概率从未据此截断或回写。

{chr(10).join(detail)}

## 这次具体学到什么

1. 平局的总体频率和逐场识别必须分开。候选平均平局概率{new_full['mean_draw_probability']:.4%}接近实际{new_full['actual_draw_rate']:.4%}，但平局AUC为{new_full['draw_auc']:.4f}，低于原线{old_full['draw_auc']:.4f}。总体比例更接近，没有转化成更好的逐场排序。

2. 选平带来了实际命中，同时损失了别处。候选首选平局{new_full['selected_draw_count']}场、命中{new_full['selected_draw_correct']}场；正确主胜数{class_deltas[0]:+d}、正确平局数{class_deltas[1]:+d}、正确客胜数{class_deltas[2]:+d}，合计{changed['net_correct_change']:+d}场。不能仅用新增平局命中证明整条线路改善，也不能把26场原主客反选改成平局视为方向修复。

3. 新线包含原线没有的正确预测，但事后互补不等于赛前可选对。{changed['top1_changed']}场首选改向中，纠正{changed['only_new_correct']}场、改坏{changed['only_original_correct']}场、双方均错但改向{changed['both_wrong_changed']}场；若只展示独有正确的{changed['only_new_correct']}场，会隐去更大的{changed['only_original_correct']}场损失。逐场改向CSV保留目标、实际结果、原／新主胜平局客胜各自概率和变化类型，不能据此事后挑路线或当作已证明的联合增益。

上述分解定位到当前候选的取舍；它既未隔离持续状态机制本身，也未证明把哪个参数换掉就能恢复提准。本轮保留可复用的持续状态、时间递推和同日观察组件，停止救分调参。

## 时序、解释与停止点

所有输入均为既有资料，没有下载或查询新赛果。每场只观察日期严格早于原发布日期F=D减3天的比赛；静态和四个动态参数的学习先于两个目标季。目标季早先结束且日期早于后续F的结果可更新后续状态，并未进入预测自身。最终状态只走到最后目标的资料前缀，未读到输入文件末尾。

初始参数与动态参数训练分数属于拟合内目标，不作为验证准确率。2024/25、2025/26已经反复用于研发；本报告是完整前推开发结果，不是全新盲测。平均平局比例接近实际频率不能代替逐场识别；把主客反选改成平局也不自动代表方向学对，因此条件主／客方向诊断单列。损益分解属于观察，不是某一算法参数的因果贡献。

本轮到一次固定结构的完整拟合、重放与报告停止，不根据后季换参数边界、种子、衰减、联赛权重或做救分变体。保留持续状态、同日批处理、未知回退和净胜球输出等组件；一次接法的结果不用于否定整个状态建模方向。是否进入真正前瞻应结合完整结果讨论，当前原预测、LLM和市场线路保持原记录。

所有新文件、缓存与临时目录均在G盘。主要文件为`执行前说明.json`、`静态初始化.json`、`动态参数拟合.json`、`逐场概率与状态.json.gz`、`逐场改向与纠正改坏.csv`、`最终过滤状态.json`与`完整结果与分解.json`。
'''
    (OUT / '实际结果与优化判断.md').write_text(report, encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    stage = 'load_input'
    new_fit_calls = {'static': 0, 'dynamic': 0}
    reused_fit_files = {'static': False, 'dynamic': False}
    try:
        data = read_input()
        write(OUT / '执行前说明.json', execution_plan(data))
        early = [r for r in data['matches'] if r['season'] == '2022/23']
        dynamic_rows = [r for r in data['matches'] if r['season'] == '2023/24']
        later = [r for r in data['matches'] if r['season'] != '2022/23']
        earliest_target_F = min(r['F'] for r in data['targets'])
        if not max(r['date'] for r in early) < min(r['date'] for r in dynamic_rows):
            raise ValueError('Static and global dynamic parameter training periods overlap')
        if not max(r['date'] for r in dynamic_rows) < earliest_target_F:
            raise ValueError('Global parameter fitting used results at or after the first target F')
        stage = 'static_initialization'
        print(json.dumps({'stage': stage, 'matches': len(early)}, ensure_ascii=False), flush=True)
        static_file = OUT / '静态初始化.json'
        if static_file.exists():
            static = json.loads(static_file.read_text(encoding='utf-8'))
            reused_fit_files['static'] = True
        else:
            new_fit_calls['static'] += 1
            static = fit_static_panel(early)
            write(static_file, static)
        expected_static_ids = [r['match_id'] for r in sorted(early, key=lambda r: (r['date'], r['match_id']))]
        if static['training_ids'] != expected_static_ids or static['training_max_date'] != max(r['date'] for r in early):
            raise ValueError('Saved static fit has different initial-season identities or dates')
        print(json.dumps({'stage': 'static_complete', 'optimizer': static['optimizer']}, ensure_ascii=False), flush=True)
        stage = 'four_parameter_dynamic_fit'
        print(json.dumps({'stage': stage, 'matches': len(dynamic_rows)}, ensure_ascii=False), flush=True)
        dynamic_file = OUT / '动态参数拟合.json'
        if dynamic_file.exists():
            dynamics = json.loads(dynamic_file.read_text(encoding='utf-8'))
            reused_fit_files['dynamic'] = True
        else:
            new_fit_calls['dynamic'] += 1
            dynamics = fit_dynamics(static, dynamic_rows)
            write(dynamic_file, dynamics)
        expected_dynamic_ids = [r['match_id'] for r in sorted(dynamic_rows, key=lambda r: (r['date'], r['match_id']))]
        if (dynamics['training_ids'] != expected_dynamic_ids
                or dynamics['training_max_date'] != max(r['date'] for r in dynamic_rows)
                or not dynamics['training_max_date'] < earliest_target_F):
            raise ValueError('Saved global dynamic fit differs from the earlier training cohort or precedes no target')
        print(json.dumps({'stage': 'dynamics_complete', 'parameters': dynamics['parameters'],
                          'optimizer': dynamics['optimizer']}, ensure_ascii=False), flush=True)
        stage = 'strict_date_target_replay'
        predictions, state = forecast_targets(static, dynamics['parameters'], later, data['targets'])
        if len(predictions) != len(data['targets']):
            raise ValueError('Prediction count differs from full target cohort')
        for original, prediction in zip(data['targets'], predictions, strict=True):
            if any(prediction[k] != original[k] for k in ('match_id', 'date', 'F', 'original_hda', 'hg', 'ag')):
                raise ValueError('Core replay changed target identity, timing, label or original HDA')
            if not prediction['last_observed_date'] < prediction['F']:
                raise ValueError('Replay observed information at or after target F')
            if not prediction['gas_available'] and prediction['deployment_probabilities'] != original['original_hda']:
                raise ValueError('Unknown team fallback changed the original HDA')
        zipped(OUT / '逐场概率与状态.json.gz', {'records': predictions})
        write(OUT / '最终过滤状态.json', state)
        stage = 'score_and_report'
        summary = analyze_predictions(predictions, static, dynamics, time.perf_counter()-began)
        write(OUT / '执行记录.json', {'completed_at_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'completed', 'study_total_static_fit_calls': 1, 'study_total_dynamic_fit_calls': 1,
            'new_fit_calls_this_runner': new_fit_calls, 'reused_fit_files': reused_fit_files,
            'target_rows': len(predictions), 'parameter_grids': 0, 'current_results_queries': 0,
            'elapsed_seconds': time.perf_counter()-began})
        print(json.dumps({'stage': 'complete', 'by_season': {s: {
            'n': r['n'], 'gas_available': r['gas_available_n'], 'fallback': r['fallback_n'],
            'original_correct': r['original']['top1_correct'], 'new_correct': r['deployment']['top1_correct'],
            'original_LL': r['original']['log_loss'], 'new_LL': r['deployment']['log_loss']}
            for s, r in summary['by_season'].items()}}, ensure_ascii=False), flush=True)
    except Exception as error:
        write(OUT / '执行失败说明.json', {'failed_at_utc': datetime.now(timezone.utc).isoformat(),
            'stage': stage, 'error_type': type(error).__name__, 'error': str(error),
            'traceback': traceback.format_exc(), 'elapsed_seconds': time.perf_counter()-began,
            'action': 'stop and report; no parameter rescue or variant rerun'})
        raise


if __name__ == '__main__':
    main()
