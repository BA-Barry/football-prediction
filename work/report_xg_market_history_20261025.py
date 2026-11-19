"""Report the measured market reference gap without adding a model or bet rule."""
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
RUN = ROOT/'outputs/进球xG历史市场对照_20261024T181414691003Z'
LABELS = {'avg_preclosing':'平均赔率AvgH／D／A','avg_closing':'平均收盘AvgCH／CD／CA',
          'b365_preclosing':'B365H／D／A','b365_closing':'B365CH／CD／CA'}
LEAGUES = {'E0':'英超','D1':'德甲','SP1':'西甲','I1':'意甲','F1':'法甲'}


def interval(entry, metric, block):
    i = entry['difference_order'].index('blend_minus_market:'+metric)
    lo,hi = entry['block_intervals'][str(block)]['interval_95'][i]
    return f'[{lo:+.6f}, {hi:+.6f}]'


def main():
    result = json.loads((RUN/'历史市场对照结果.json').read_text(encoding='utf-8'))
    report = ['# 现有融合模型与历史市场的同场对照', '',
              '**融合改善了原统计模型，但在两个目标赛季中仍未胜过历史市场参考。** 这项负面结果限制了现阶段对模型实用性的判断，不能只报告相对原模型的正向增量。', '',
              '## 本轮实际做了什么', '',
              '复用两轮已保存的纯进球、独立xG、进球＋xG概率，分别取2023/24和2025/26两个检验季，各1,733场。按原Football-Data源文件中的同一比赛ID、日期、主客身份及完整比分重新联结四类1X2赔率。每种报价均独立检查共同覆盖和缺失，本轮四种报价在两季各1,733场均完整。', '',
              '主参考在评分前选定为AvgH／AvgD／AvgA，沿用项目既有比例去水：每项赔率的倒数除以三项倒数之和。平均收盘、B365及B365收盘全部保留为次对照。新增训练0次、权重0个、参数网格0个；模型概率、原预测、市场融合及待讨论结构均未改动。', '',
              '## 主参考的结果', '',
              '| 目标季 | 场数 | 纯进球LL | 独立xG LL | 融合LL | 市场LL | 融合减市场 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for period,markets in result['seasons'].items():
        v = markets['avg_preclosing']
        s = v['scores']
        report.append(f'| {period} | {v["n"]} | {s["goal"]["log_loss"]:.6f} | {s["xg"]["log_loss"]:.6f} | {s["blend"]["log_loss"]:.6f} | {s["market"]["log_loss"]:.6f} | {v["mean_differences"]["blend_minus_market:log_loss"]:+.6f} |')
    report += ['', 'LL越低越好，因此两季融合均落后。比分与总进球的统计改进不能直接推断胜平负概率已超过市场。本轮仅有1X2赔率，未据此评价市场总进球或让球报价。', '',
               '| 目标季 | 融合Top1 | 市场Top1 | 融合Brier | 市场Brier | 融合RPS | 市场RPS |',
               '|---|---:|---:|---:|---:|---:|---:|']
    for period,markets in result['seasons'].items():
        v = markets['avg_preclosing']
        b,m = v['scores']['blend'],v['scores']['market']
        report.append(f'| {period} | {round(b["accuracy"]*v["n"])}/{v["n"]}（{b["accuracy"]:.2%}） | {round(m["accuracy"]*v["n"])}/{v["n"]}（{m["accuracy"]:.2%}） | {b["brier_sum"]:.6f} | {m["brier_sum"]:.6f} | {b["rps"]:.6f} | {m["rps"]:.6f} |')
    report += ['', 'Brier为三类平方误差之和，RPS沿用三类顺序累计误差除以2。两项均越低越好。Top1是每场选择概率最高一项的命中率，正差才表示较好；本轮融合也低于市场。', '',
               '| 融合减市场LL | 2周95%条件区间 | 4周 | 8周 |', '|---|---|---|---|']
    for period,markets in result['seasons'].items():
        v = markets['avg_preclosing']
        report.append(f'| {period} | {interval(v,"log_loss",2)} | {interval(v,"log_loss",4)} | {interval(v,"log_loss",8)} |')
    report += ['', '两季LL差的三套区间均高于零，Brier和RPS方向也一致；Top1差的三套区间均低于零。采用同场、同日历周的联合重采样，两目标季分别分析，未把相隔年份直接拼成连续周序列。区间条件于已保存预测，不能覆盖历史方法选择和全部参数不确定性。', '',
               '## 全部报价和联赛', '',
               '| 目标季 | 报价 | 共同场数 | 融合LL | 市场LL | 差值 |', '|---|---|---:|---:|---:|---:|']
    for period,markets in result['seasons'].items():
        for market,v in markets.items():
            report.append(f'| {period} | {LABELS[market]} | {v["n"]} | {v["scores"]["blend"]["log_loss"]:.6f} | {v["scores"]["market"]["log_loss"]:.6f} | {v["mean_differences"]["blend_minus_market:log_loss"]:+.6f} |')
    report += ['', '四种报价的整体LL均落后，没有根据结果换主报价。各报价自己的覆盖集合与缺失ID保存在结果里；其他数据若出现缺失，应在该报价的同场集合上重算全部模型。', '',
               '| 目标季 | 联赛 | 主报价共同场数 | 融合减市场LL | 4周区间 |', '|---|---|---:|---:|---|']
    for period,leagues in result['by_league'].items():
        for comp,markets in leagues.items():
            v = markets['avg_preclosing']
            report.append(f'| {period} | {LEAGUES[comp]} | {v["n"]} | {v["mean_differences"]["blend_minus_market:log_loss"]:+.6f} | {interval(v,"log_loss",4)} |')
    report += ['', '十个“赛季×联赛”的主报价LL点估计都落后；分组区间逐项保留，不把整体结论扩大成每个小组均得到独立确认。', '',
               '## 这项比较能说明什么', '',
               '现有融合在这些历史队列上的胜平负概率质量尚未达到市场参考水平。这支持继续把融合列为研究候选，分别保留纯进球、xG、市场与LLM，而不能用“比旧统计线更好”替代市场增量验证。', '',
               '模型每28日更新，市场报价可能包含更晚的首发、伤停、球队状态等信息；具体历史报价采集时刻未知。因此，这不是同一信息时点的能力比较，也不能把差距全部归因于算法、刷新频率或某一未接入因素。AvgH／D／A不能直接当成开盘或T−24小时报价，更不能据这四列还原开盘至即时变化。', '',
               '去水概率是采用明确规则构造的市场参考，真实概率仍未知。整体概率评分与某种选择规则的可成交价值、收益是不同问题；本轮未构造交易规则或估算盈利。', '',
               '## 独立核验与下一步', '',
               '独立审查直接从原CSV重建3,466场的四组报价，去水概率完全一致；旧保存的5,199份报价版本一致，三个模型的概率与来源逐条相同。48个范围的四指标、四组对比和全部2／4／8周区间复算通过，最大评分差约2.78×10⁻¹⁵、区间差约6.94×10⁻¹⁸。没有发现需修改计算或重训的问题。', '',
               '此市场对照完成后，用户明确同意固定总球的主客分配研究；该研究已另行实跑完成，结论见下方链接。市场对照本身保持零拟合，没有自动引入市场融合或神经网络。当前比赛赛果复核仍暂停，等待用户提醒。全部文件与临时文件均在项目目录。', '',
               '[后续主客分配实跑结论](../../outputs/固定总球主客分配研究_20261024T182604417382Z/主客分配实跑与采用结论.md)', '',
               f'[完整结果与逐场报价]({(RUN/"历史市场对照结果.json").as_posix()}) · [评分前分析计划]({(RUN/"分析前计划.json").as_posix()})', '']
    (RUN/'历史市场对照结论与优化边界.md').write_text('\n'.join(report),encoding='utf-8')
    print(str(RUN/'历史市场对照结论与优化边界.md'))


if __name__ == '__main__':
    main()
