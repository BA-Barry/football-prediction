"""Explain archived errors and rate changes; no fitting, new predictions or network.

The two-factor decomposition is a numerical identity of Poisson probabilities,
not a causal attribution to source features and not a calibrated intervention.
"""
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

import numpy as np

from football_model.probability import score_grid

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
NAMES = ('主胜', '平局', '客胜')
OUTCOMES = ('home', 'draw', 'away')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def describe(probabilities, outcome=None):
    p = list(probabilities)
    top = [i for i, v in enumerate(p) if v == max(p)]
    ordered = sorted(p, reverse=True)
    result = {'probabilities': p, 'top_outcomes': top,
              'top_two_gap_pp': 100*(ordered[0]-ordered[1])}
    if outcome is not None:
        result.update(actual_probability=p[outcome], actual_rank=1+sum(v > p[outcome] for v in p),
            top1_credit=1/len(top) if outcome in top else 0.,
            log_loss=None if p[outcome] == 0 else -math.log(p[outcome]),
            log_loss_infinite=p[outcome] == 0,
            brier_sum=sum((v-int(i == outcome))**2 for i, v in enumerate(p)))
    return result


def decompose(simple_rates, rich_rates):
    total0, total1 = sum(simple_rates), sum(rich_rates)
    share0, share1 = simple_rates[0]/total0, rich_rates[0]/total1
    def p(total, share):
        return np.asarray(score_grid(total*share, total*(1-share)).home_draw_away)
    p00, p10, p01, p11 = p(total0, share0), p(total1, share0), p(total0, share1), p(total1, share1)
    total_effect = .5*((p10-p00)+(p11-p01))
    share_effect = .5*((p01-p00)+(p11-p10))
    error = float(np.max(np.abs(total_effect+share_effect-(p11-p00))))
    if error > 1e-12:
        raise AssertionError('Two-factor numerical contributions must sum to the actual change')
    return {'simple_total_goals': total0, 'rich_total_goals': total1,
            'simple_home_share': share0, 'rich_home_share': share1,
            'simple_probabilities': p00.tolist(), 'rich_probabilities': p11.tolist(),
            'total_rate_contribution_pp': (100*total_effect).tolist(),
            'home_share_contribution_pp': (100*share_effect).tolist(),
            'actual_change_pp': (100*(p11-p00)).tolist(), 'sum_error': error}


def run():
    source = ROOT/'outputs/前瞻评价接续_20261027/分组前瞻记录与评价.json'
    data = read(source)
    original = data['cohort_records']['original10']
    scores = {r['fixture_id']: r for r in data['cohort_scores']['original10']['per_fixture']}
    explained = []
    for row in original:
        fid = row['fixture']['match_id']
        truth = OUTCOMES.index(scores[fid]['outcome'])
        tracks = {arm: describe(row['tracks'][arm]['probabilities'], truth)
                  for arm in ('statistical', 'llm_independent')}
        stat, llm = tracks['statistical'], tracks['llm_independent']
        # Reconcile the newly presented numbers with the established scorer.
        for arm, description in tracks.items():
            archived = scores[fid]['scores'][arm]
            if abs(description['log_loss']-archived['log_loss']) > 1e-12 or abs(description['brier_sum']-archived['brier']) > 1e-12:
                raise AssertionError('Description and established scorer disagree')
        same = len(stat['top_outcomes']) == len(llm['top_outcomes']) == 1 and stat['top_outcomes'] == llm['top_outcomes']
        explained.append({'fixture': row['fixture'], 'outcome': truth, 'tracks': tracks,
                          'same_unique_top': same,
                          'llm_minus_stat_log_loss': llm['log_loss']-stat['log_loss']})
    groups = {}
    for label, same in (('same_unique_top', True), ('different_or_tied_top', False)):
        group = [r for r in explained if r['same_unique_top'] is same]
        groups[label] = {'n': len(group), 'fixture_ids': [r['fixture']['match_id'] for r in group],
            'statistical_hits': sum(r['tracks']['statistical']['top1_credit'] for r in group),
            'llm_hits': sum(r['tracks']['llm_independent']['top1_credit'] for r in group),
            'both_wrong': sum(all(t['top1_credit'] == 0 for t in r['tracks'].values()) for r in group)}
    calibration = {'observed_counts': [sum(r['outcome'] == c for r in explained) for c in range(3)]}
    for arm in ('statistical', 'llm_independent'):
        calibration[arm] = {'expected_counts': [sum(r['tracks'][arm]['probabilities'][c] for r in explained) for c in range(3)],
            'top1_draw_count': sum(r['tracks'][arm]['top_outcomes'] == [1] for r in explained)}
    informed_copies = sum(r['tracks']['llm_model_informed'] is not None and
        r['tracks']['llm_model_informed']['probabilities'] == r['tracks']['statistical']['probabilities'] for r in original)

    current_folder = ROOT/'outputs/当前丰富信息预测_20261026T030805654404Z'
    current = read(current_folder/'多线逐场对照.json')
    current_rows = []
    for row in current['records']:
        parts = decompose(row['current_simple']['rates'], row['current_rich_candidate']['rates'])
        for kind, key in (('simple', 'current_simple'), ('rich', 'current_rich_candidate')):
            if not np.allclose(parts[kind+'_probabilities'], row[key]['probabilities'], atol=1e-12, rtol=0):
                raise AssertionError('Decomposition must reproduce both archived forecasts')
        current_rows.append({'fixture': row['fixture'], 'home_label': row['home_label'], 'away_label': row['away_label'],
            'statistical': describe(row['current_rich_candidate']['probabilities']),
            'llm': describe(row['current_llm']['probabilities']), 'rate_decomposition': parts,
            'actual_outcome': None})
    summary = {'mean_change_pp': np.mean([r['rate_decomposition']['actual_change_pp'] for r in current_rows], axis=0).tolist(),
        'mean_total_contribution_pp': np.mean([r['rate_decomposition']['total_rate_contribution_pp'] for r in current_rows], axis=0).tolist(),
        'mean_share_contribution_pp': np.mean([r['rate_decomposition']['home_share_contribution_pp'] for r in current_rows], axis=0).tolist(),
        'draw_probability_increases': sum(r['rate_decomposition']['actual_change_pp'][1] > 0 for r in current_rows),
        'same_model_llm_unique_top': sum(r['statistical']['top_outcomes'] == r['llm']['top_outcomes'] and len(r['statistical']['top_outcomes']) == 1 for r in current_rows),
        'max_decomposition_error': max(r['rate_decomposition']['sum_error'] for r in current_rows)}
    result = {'recorded_at': datetime.now(timezone.utc).isoformat(), 'source': str(source),
        'class_order': OUTCOMES, 'archived_simple10': explained, 'agreement_groups': groups,
        'marginal_counts': calibration, 'model_informed_exact_copies': informed_copies,
        'archived_future10': current_rows, 'future_change_summary': summary,
        'new_fits': 0, 'new_forecasts': 0, 'new_results_looked_up': 0,
        'interpretation': 'Descriptive diagnostic only. No calibration, confidence threshold, component weight or outcome-based selector learned.',
        'decomposition_note': 'Average of both orders of changing total rate and home allocation; exact numerical decomposition, not causal feature importance.'}
    output = ROOT/'outputs/前瞻误差与概率变化_20261027'
    output.mkdir(parents=True, exist_ok=True)
    (output/'逐场误差与进球率变化.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    pct = lambda p: '／'.join(f'{100*x:.1f}' for x in p)
    pick = lambda v: '、'.join(NAMES[i] for i in v['top_outcomes'])
    lines = ['# 已存前瞻误差与下一轮概率变化', '',
        '本轮直接解释既有预测，不重复旧历史分组，不重训、调参、查询新赛果或生成新的比赛预测。原10场属于简单模型；当前121丰富候选没有已结算前瞻成绩。', '',
        '## 原10场：保留全部比赛的逐场误差', '',
        '主／平／客三项单位%；“实际概率”为赛前给后来真实事件的概率，损失越低越好。首选与实际事件概率分别展示。', '',
        '| 比赛 | 实际类别 | 统计主／平／客 | LLM主／平／客 | 统计／LLM首选 | 实际概率：统计／LLM | LLM−统计损失 |',
        '|---|---|---:|---:|---|---:|---:|']
    for r in explained:
        s, l = r['tracks']['statistical'], r['tracks']['llm_independent']
        lines.append(f"| {r['fixture']['home']}—{r['fixture']['away']} | {NAMES[r['outcome']]} | {pct(s['probabilities'])} | {pct(l['probabilities'])} | {pick(s)}／{pick(l)} | {s['actual_probability']:.1%}／{l['actual_probability']:.1%} | {r['llm_minus_stat_log_loss']:+.4f} |")
    agree, disagree = groups['same_unique_top'], groups['different_or_tied_top']
    lines += ['', f"同首选{agree['n']}场，两线各中{agree['statistical_hits']:g}场，共同错{agree['both_wrong']}场；不同首选{disagree['n']}场，统计中{disagree['statistical_hits']:g}场、LLM中{disagree['llm_hits']:g}场，双方都错{disagree['both_wrong']}场。不能据此建立‘同向就加仓’或‘有分歧就选某线’规则。", '',
        '## 这批误差能支持什么', '',
        f"实际主／平／客为{calibration['observed_counts']}场；统计期望场数为{'／'.join(f'{x:.2f}' for x in calibration['statistical']['expected_counts'])}，LLM为{'／'.join(f'{x:.2f}' for x in calibration['llm_independent']['expected_counts'])}。两线均没有将平局列为唯一首选，但没有给平局零概率。10场的频数差仅描述这批结果，不是稳定校准结论。", '',
        f'模型知情LLM中有{informed_copies}/10条三项概率与统计完全相同，不能当成额外独立预测。普通LLM也可能与统计共享信息盲点；同向和反向都需要真实评价。', '',
        '近期结果、伤停、战术或红牌是否导致错误，不能由此表的概率差直接识别。旧原轮结果不归给121，历史上均势平局辨别薄弱的结论也不能由这10场单独确认。', '',
        '## 下一轮丰富候选究竟改了什么', '',
        '将简单模型与丰富候选的主客预期进球写成“总球率×主客分配”。先改总球再改分配、先改分配再改总球，各算一次并平均；两部分相加精确还原实际三项概率变化。这只解释比分公式，不识别是哪条xG／PPDA等输入造成变化，也不证明变化更准。', '',
        '| 比赛 | 总预期进球：简单→丰富 | 主队份额：简单→丰富 | 主胜变化 | 其中总球／分配 | 平局变化 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in current_rows:
        d = r['rate_decomposition']
        lines.append(f"| {r['fixture']['home']}—{r['fixture']['away']} | {d['simple_total_goals']:.2f}→{d['rich_total_goals']:.2f} | {d['simple_home_share']:.1%}→{d['rich_home_share']:.1%} | {d['actual_change_pp'][0]:+.2f}pp | {d['total_rate_contribution_pp'][0]:+.2f}／{d['home_share_contribution_pp'][0]:+.2f}pp | {d['actual_change_pp'][1]:+.2f}pp |")
    lines += ['', f"平均主／平／客变化为{'／'.join(f'{x:+.2f}' for x in summary['mean_change_pp'])}个百分点；10场中{summary['draw_probability_increases']}场平局概率上升。丰富候选与LLM有{summary['same_model_llm_unique_top']}/10场同首选，这只是当前判断一致性，未构成准确率。", '',
        f"平均主胜变化中，总球贡献{summary['mean_total_contribution_pp'][0]:+.2f}个百分点，主客分配贡献{summary['mean_share_contribution_pp'][0]:+.2f}个百分点；平均平局变化中，总球贡献{summary['mean_total_contribution_pp'][1]:+.2f}个百分点，主客分配贡献{summary['mean_share_contribution_pp'][1]:+.2f}个百分点。该批主胜方向变化主要来自分配调整，平局下降主要来自总球上调；没有赛果时不能判为改进或退步。", '',
        '## 下一步针对的缺口', '',
        '1. 继续保留简单线、丰富线、LLM原始三项概率；赛果到来后，先判断损失变化来自实际事件概率改善还是仅改了首选，不使用赛后赢家选择器。',
        '2. 按已保存的总球／主客分配拆解观察误差，帮助判断下一项研究应针对进球强度还是比分结构；这种拆解本身不是训练，也没有给新权重。',
        '3. 均势平局、弱方单胜偏高及0:0／1:1／更高平局的旧诊断仍保留；条件正平局修正、温度和全局DC不重复运行。提高平均平局质量与识别具体平局比赛是两件事。',
        '4. 真正缺少的仍有球队变化、可靠全赛事负荷和球员可用性等输入的可学习表示；现有记录不能证明只靠再叠算法会解决。重大结构调整先讨论，日常资料错误修复正常推进。', '',
        '本轮停止在逐场误差、两线分歧及既有概率的数值分解；无新增训练、预测、阈值选择或赛果查询。所有原始预测未改，全部新文件位于G盘。', '',
        '[前瞻评价总表](G:/Codex/football-prediction-2026-10-22/outputs/前瞻评价接续_20261027/前瞻评价接续与优化安排.md)；[历史121问题定位](G:/Codex/football-prediction-2026-10-22/outputs/当前121模型胜平负定位_20261026T051442339672Z/结论与下一步选择.md)。']
    (output/'误差定位与下一步优化.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'agreement': groups, 'future_change': summary, 'fits': 0}, ensure_ascii=False))


if __name__ == '__main__':
    run()
