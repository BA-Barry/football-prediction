"""Report the completed fixed-total allocation contrast, including negative results."""
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
RUN = ROOT/'outputs/固定总球主客分配研究_20261024T182604417382Z'
MARKET_REPORT = ROOT/'outputs/进球xG历史市场对照_20261024T181414691003Z/历史市场对照结论与优化边界.md'
NAMES = {'E0':'英超','D1':'德甲','SP1':'西甲','I1':'意甲','F1':'法甲'}


def interval(entry,domain,block):
    index = entry['difference_order'].index('allocation_minus_full_blend:'+domain)
    lo,hi = entry['block_intervals'][str(block)]['interval_95'][index]
    return f'[{lo:+.6f}, {hi:+.6f}]'


def main():
    result = json.loads((RUN/'固定总球主客分配结果.json').read_text(encoding='utf-8'))
    report = ['# 固定总进球率的主客分配研究：实际结果', '',
        '**两轮结果一轮略好、一轮略差，没有稳定的新增增益。这条主客分配线不采用，不继续追加权重微调。** 原完整融合仍为研究候选，正式及原赛前预测未改。', '',
        '## 用户同意后实际完成', '',
        '每场总进球率T取原已保存完整融合的主、客两率之和，保持不变。只在纯进球与独立xG给出的主客进球份额之间，学习一个五联赛共享的权重；采用logit份额插值，重新分配后的主客进球率相加仍为T。', '',
        '学习目标是逐场均权的条件Binomial进球损失，包含组合项。训练使用真实主、客进球，实际总球仅作为训练目标的一部分；预测接口只接收保存的预期总率与两条赛前进球率，不接收实际比分。没有按实际总球重新加权，也没有删去0–0场次。', '',
        '两个年份分别只学习较早赛季的1个权重：2022/23→2023/24，以及2024/25→2025/26。共新增2次一元权重估计、0次球队拟合、0个参数网格；两轮目标队列沿用原记录，不增加或删选有利场次。', '',
        '| 估计季→检验季 | 估计场数 | 原完整融合权重 | 新主客分配权重 | 检验场数 |', '|---|---:|---:|---:|---:|']
    for name,v in result['splits'].items():
        period = name[-7:]
        report.append(f'| {name.replace("_to_","→")} | {v["parameters"]["n"]} | {v["old_full_blend_weight"]:.6f} | {v["applied_allocation_weight"]:.6f} | {v["seasons"][period]["n"]} |')
    report += ['', '两轮均可识别，均使用学习值。代码同时处理无信息情况：如果两条线份额相同，或所有有份额差异的训练场均无进球，则明确标记不可识别，应用时维持旧完整融合的份额。', '',
        '## 后季实际结果', '',
        '| 检验季 | 原完整融合比分NLL | 新分配比分NLL | 差值，负数改善 | 原胜平负LL | 新胜平负LL | 差值 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,v in result['splits'].items():
        p = name[-7:]
        e = v['seasons'][p]
        b,a = e['scores']['full_blend'],e['scores']['allocation']
        report.append(f'| {p} | {b["exact_score"]["log_loss"]:.6f} | {a["exact_score"]["log_loss"]:.6f} | {e["mean_loss_differences"]["allocation_minus_full_blend:exact_score"]:+.6f} | {b["one_x_two"]["log_loss"]:.6f} | {a["one_x_two"]["log_loss"]:.6f} | {e["mean_loss_differences"]["allocation_minus_full_blend:one_x_two"]:+.6f} |')
    report += ['', '| 检验季／指标 | 2周95%条件区间 | 4周 | 8周 |', '|---|---|---|---|']
    for name,v in result['splits'].items():
        p = name[-7:]
        e = v['seasons'][p]
        for label,domain in [('比分','exact_score'),('胜平负','one_x_two'),('客让两球','away_minus_two_three_way')]:
            report.append(f'| {p}／{label} | {interval(e,domain,2)} | {interval(e,domain,4)} | {interval(e,domain,8)} |')
    report += ['', '两季比分、胜平负与客让两球的三套区间均跨零。早季条件目标的优化成功，不能据此声称后季预测能力提高；本轮后季增量方向不一致且幅度很小。', '',
        '| 检验季 | 原融合Top1 | 新分配Top1 | 原融合与市场LL差 | 新分配与市场LL差 |', '|---|---:|---:|---:|---:|']
    market = json.loads((MARKET_REPORT.parent/'历史市场对照结果.json').read_text(encoding='utf-8'))
    for name,v in result['splits'].items():
        p = name[-7:]
        e = v['seasons'][p]
        b,a = e['scores']['full_blend'],e['scores']['allocation']
        report.append(f'| {p} | {round(b["one_x_two"]["accuracy"]*e["n"])}/{e["n"]} | {round(a["one_x_two"]["accuracy"]*e["n"])}/{e["n"]} | {market["seasons"][p]["avg_preclosing"]["mean_differences"]["blend_minus_market:log_loss"]:+.6f} | {v["market_reference"]["allocation_minus_market_LL"]:+.6f} |')
    report += ['', '最高概率命中数也一降一升，不能据2025/26多命中8场就选择新结构。新分配仍落后归档平均赔率参考。市场信息可能晚于球队率的更新时点，比较不代表同一信息时点的能力结论，更不能推导可成交收益。', '',
        '## 所有联赛同时报告', '',
        '| 检验季 | 联赛 | 场数 | 比分损失差 | 胜平负损失差 |', '|---|---|---:|---:|---:|']
    for name,v in result['splits'].items():
        p = name[-7:]
        for comp,e in v['by_league'][p].items():
            d = e['mean_loss_differences']
            report.append(f'| {p} | {NAMES[comp]} | {e["n"]} | {d["allocation_minus_full_blend:exact_score"]:+.6f} | {d["allocation_minus_full_blend:one_x_two"]:+.6f} |')
    report += ['', '不按后季表现给联赛选择不同权重，也不只留下有利联赛。完整2／4／8周区间保存在结果JSON中。', '',
        '## 实现与数值验证', '',
        '新增独立主客分配模块的47项测试通过，覆盖梯度／曲率、已知最优和端点、时间边界与未来扰动、0–0与不可识别、总率守恒、总率缩放及主客交换、数值极值和非法输入。', '',
        '实际运行逐场确认原完整融合权重可回放原主客率；早季优化后的条件损失不高于这个可行旧权重。新旧完整比分损失之差完全来自主客分配，理论总球分布未变。网格截断会产生约10⁻¹²级总球概率差，已用解析Poisson分布核对，没有把该数值尾差当成模型增益。', '',
        '独立数值审查通过：两权重重新求解一致；6,990场率值、总率、条件分解与24个范围的六领域评分及全部区间一致。最大评分／区间差约1.84×10⁻¹³，比分差与分配差最大误差约2.66×10⁻¹⁵。元数据明确区分球队拟合截止与分配权重截止，早季标记为拟合内结果，后季标记为固定早季权重检验。', '',
        '## 采用结论与后续方向', '',
        '这版新增份额权重不采用，停止围绕其数值继续微调；原完整融合继续列为研究候选，纯进球、独立xG、市场与LLM线路均保留。它只检验了固定现有总球判断后重新分配主客份额的一个具体结构，不能推出所有主客分配模型或xG资料都无价值。', '',
        '下一步的重点应是能够改变主客相对实力判断的新信息及其赛前可用时间，先评估现有球员、首发、伤停和赛事投入资料能支持哪些历史对照，再讨论实质结构调整。新增复杂度必须有明确数据与检验目的，不用另加参数来补本轮分数，也不直接转向黑盒。', '',
        '本轮仍是已查看历史样本上的探索，条件区间不包含全部权重估计和研究选择不确定性。历史xG发布／修订与赔率时间仍有限制。没有查询或复核当前比赛赛果，也未改原赛前预测。全部文件、测试、缓存和临时文件在项目目录。', '',
        f'[完整结果]({(RUN/"固定总球主客分配结果.json").as_posix()}) · [评分前计划]({(RUN/"分析前计划.json").as_posix()}) · [历史市场对照]({MARKET_REPORT.as_posix()})', '']
    (RUN/'主客分配实跑与采用结论.md').write_text('\n'.join(report),encoding='utf-8')
    print(str(RUN/'主客分配实跑与采用结论.md'))


if __name__ == '__main__':
    main()
