"""Describe saved common-history results without refitting or choosing subgroups."""
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
RUN = ROOT / 'outputs/进球xG共同训练记录_20261024T174247963857Z'


def exact_loss(row):
    return math.fsum(mu-y*math.log(mu)+math.lgamma(y+1)
                     for mu, y in zip(row['rates'], row['actual_score']))


def main():
    result = json.loads((RUN/'共同训练记录融合结果.json').read_text(encoding='utf-8'))
    late = result['seasons']['2025-26']
    old = {r['match_id']: r for r in result['common_goal_predictions']}
    months = defaultdict(list)
    for row in result['common_blend_predictions']:
        if row['period'] != '2025-26':
            continue
        reference = old[row['match_id']]
        y = row['actual_outcome']
        months[row['date'][:7]].append((exact_loss(row)-exact_loss(reference),
                                      -math.log(row['probabilities'][y])
                                      +math.log(reference['probabilities'][y])))
    monthly = {m: {'n': len(v), 'exact_score_delta': math.fsum(z[0] for z in v)/len(v),
                  'one_x_two_delta': math.fsum(z[1] for z in v)/len(v)}
               for m, v in sorted(months.items())}
    diagnostic = {'created_at': datetime.now(timezone.utc).isoformat(),
                  'purpose': 'descriptive complete calendar months, no subgroup selection or new fitting',
                  'source': str(RUN/'共同训练记录融合结果.json'),
                  'new_fits': 0, 'monthly': monthly,
                  'limitations': 'same already-viewed later season, not additional independent validation'}
    (RUN/'逐月描述对照.json').write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2,
                                                allow_nan=False), encoding='utf-8')
    scores = late['scores']
    goal, blend = scores['common_goal'], scores['common_blend']
    w = result['parameters']['weight']
    report = [
        '# 本轮结论与下一步优化', '',
        '## 结论', '',
        '**统一进球线与xG线的训练比赛后，融合改善仍然存在；目前继续保留为研究候选。**', '',
        '上轮两条线对15场日期冲突记录的处理不同，可能混入资料处理带来的影响。此次让每个截止的训练比赛、日期、主客场和时间权重相同，再比较只用进球与加入xG的预测。实际新增69次进球拟合、复用41次，110次xG拟合全部复用；没有参数网格搜索。', '',
        f'权重只用较早2024/25赛季的1,734场估计，再固定检验较晚2025/26赛季的1,733场。xG权重从{result["old_weight"]:.2%}变为{w:.2%}；它是对数预期进球率的融合系数，**不是xG的可信度，也不是命中率**。主、客与五个联赛共享同一个系数。', '',
        '## 同场比较', '',
        '| 较晚赛季指标 | 相同训练记录的纯进球线 | 进球＋xG |',
        '|---|---:|---:|',
    ]
    for label, key in [('比分概率损失', 'exact_score'), ('胜平负概率损失', 'one_x_two'),
                       ('总进球分组概率损失', 'total_goals_0_1_2_3_4plus'),
                       ('客队让两球三项概率损失', 'away_minus_two_three_way')]:
        report.append(f'| {label}，越低越好 | {goal[key]["log_loss"]:.6f} | {blend[key]["log_loss"]:.6f} |')
    report += [
        f'| 胜平负最高概率命中 | {round(goal["one_x_two"]["accuracy"]*late["n"])}/{late["n"]}（{goal["one_x_two"]["accuracy"]:.2%}） | {round(blend["one_x_two"]["accuracy"]*late["n"])}/{late["n"]}（{blend["one_x_two"]["accuracy"]:.2%}） |', '',
        '概率分配改善，最高概率命中数却略降。当前证据支持的是较小的概率质量增量，不能把它描述成已提高猜中胜负的比例，也不能直接推断盈利。客队让两球的历史分组评分也不能代替任何具体比赛净胜三球的评估。', '',
        '比分主比较的2／4／8周移动块95%条件区间均低于零，胜平负也是如此。以4周为例：', '',
    ]
    for label, domain in [('比分概率损失', 'exact_score'), ('胜平负概率损失', 'one_x_two')]:
        idx = late['difference_order'].index('common_blend_minus_common_goal:'+domain)
        lo, hi = late['block_intervals']['4']['interval_95'][idx]
        delta = late['mean_loss_differences']['common_blend_minus_common_goal:'+domain]
        report.append(f'- {label}差（融合减纯进球）：{delta:+.6f}，区间[{lo:+.6f}, {hi:+.6f}]。')
    report += [
        '', '总进球和客让两球的两周区间仍跨零，不能把所有次指标都称为稳定改善。这些区间条件于已保存预测，没有覆盖权重估计、反复研究选择或资料修订的不确定性。历史后季此前已看过，此次不是新的盲测。此次只检验五大联赛的历史联赛比赛，不能据此认定欧冠、欧联等跨联赛比赛或赛事投入因素也已验证。', '',
        '## 改善是否普遍', '',
        '| 联赛 | 场次 | 比分损失差，负值改善 | 胜平负损失差 |', '|---|---:|---:|---:|',
    ]
    names = {'E0': '英超', 'D1': '德甲', 'SP1': '西甲', 'I1': '意甲', 'F1': '法甲'}
    for comp, name in names.items():
        item = result['by_league']['2025-26'][comp]
        d = item['mean_loss_differences']
        report.append(f'| {name} | {item["n"]} | {d["common_blend_minus_common_goal:exact_score"]:+.6f} | {d["common_blend_minus_common_goal:one_x_two"]:+.6f} |')
    report += [
        '', '比分指标在四个联赛改善，意甲略差；胜平负的改善幅度也有差异。不能称每个联赛都已提升，不根据这些后季结果增加联赛专属权重。', '',
        '| 后季月份 | 场次 | 比分损失差 | 胜平负损失差 |', '|---|---:|---:|---:|',
    ]
    for month, item in monthly.items():
        report.append(f'| {month} | {item["n"]} | {item["exact_score_delta"]:+.6f} | {item["one_x_two_delta"]:+.6f} |')
    report += [
        '', '比分与胜平负均在十个月中的八个月改善，但改善并非每月持续。此处完整列出月份只用于描述波动，不用于选择月份、调权或充当额外独立验证。', '',
        '## 下一步优化', '',
        '1. **优先扩展历史年份。** 沿用相同攻防模型、比分分布与单权重学习规则，向更早赛季做时间前推复现，检验优势能否跨年份重复；不围绕61%的权重反复微调。资料可用日期及历史修订的局限继续明确记录。',
        '2. **针对实际预测缺口做诊断。** 关注意甲比分退步及总进球、主客进球率的偏差，先确认是否跨赛季重复；单一联赛或少数失败比赛不触发新增参数。',
        '3. **保留其他线路。** 纯进球、独立xG、市场、LLM和联合线路继续分别保留。球员、伤停、轮换和赛事投入仍需可核实的赛前资料；不把LLM主观判断直接乘入统计概率。需要新增核心结构或深度学习时再讨论。', '',
        '本轮未查询或复核当前比赛赛果；复核等待用户提醒。没有改写已有赛前预测，所有文件与临时文件均在项目目录。', '',
        f'[实际训练与完整结果]({(RUN/"共同训练记录融合说明.md").as_posix()}) · [独立核验]({(RUN/"独立核验与采用结论.md").as_posix()}) · [逐月描述数据]({(RUN/"逐月描述对照.json").as_posix()})', '',
    ]
    (RUN/'本轮结论与优化计划.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({'report': str(RUN/'本轮结论与优化计划.md'),
                      'months': len(monthly), 'exact_improved_months': sum(v['exact_score_delta']<0 for v in monthly.values()),
                      'one_x_two_improved_months': sum(v['one_x_two_delta']<0 for v in monthly.values())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
