"""Describe saved fixed-run results; no fitting, new prediction or result query."""
from collections import Counter
import json
from pathlib import Path

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OUT = ROOT / 'outputs/历史比赛模式与平局分支_20261027'


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def main():
    result = read('两季完整结果.json')
    coverage = read('真实来源覆盖.json')
    records = read('逐场概率与纠错.json')['records']
    lines = [
        '# 历史比赛模式接入：本轮结果与优化判断', '',
        '2026-10-27完成一次固定接入：旧121项输入原样保留，增加24项历史比赛模式，'
        '沿用原分类树设置和22个训练窗口，仅复用其平局分量，再统一到原比分网格。'
        '两季共3,452场完整比较。概率评分相对原丰富模型小幅改善，单选命中未改善，'
        '相对上一版平局分支的增益也未跨季保持，因此不替换当前预测、不继续围绕本期调权重。', '',
        '## 做了什么，为什么做', '',
        '旧输入保留近期均值与趋势，本轮补充均值之外的逐场差异：相同平均攻防数据，'
        '可能来自场场均势，也可能来自大幅正负摆动。双方各取最近5／15场同联赛历史，'
        '共同加入npxG净差绝对值均值、npxG总量方差、npxGF/npxGA协方差、实际平局比例、'
        '过去完赛机会分布d均值与方差。LLM原事实包已包含历史平局次数；本轮补的是统计分类分支，'
        '没有把旧LLM信息称为首次发现。', '',
        '这24项是一组联合输入，由同一非线性学习器同时处理与旧信息的关系，'
        '没有逐项试到不增量就判定该方向无价值。没有重新拟合原进球／xG模型，'
        '也没有新校准、阈值、权重、神经网络或参数网格。', '',
        '## 两季完整结果', '',
        'LL与Brier越低越好；AUC用于观察平／非平的相对排序，不能换算成单选准确率。'
        '三个方案使用同一提前3个日历日、同一目标比赛和训练范围。', '',
        '| 赛季 | 方案 | 单选正确/1726 | 命中率 | 胜平负LL | 三类求和Brier | 平局AUC | 平局首选：中/选 |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    tracks = [('原丰富模型', 'pattern_minus_original_rich', 'original'),
              ('上一版平局分支', 'pattern_minus_previous_draw_branch', 'original'),
              ('新增历史模式分支', 'pattern_minus_original_rich', 'draw_branch')]
    for period, comparisons in result['by_period'].items():
        for label, comparison, arm in tracks:
            value = comparisons[comparison]
            score = value['scores'][arm]
            outcome, draw = score['one_x_two'], score['draw_binary']
            lines.append(f"| {period} | {label} | {value['correct_counts'][arm]} | "
                f"{100*outcome['accuracy']:.2f}% | {outcome['log_loss']:.6f} | {outcome['brier_sum']:.6f} | "
                f"{draw['auc']:.4f} | {draw['top1_correct']}/{draw['top1_selected']} |")
    lines += ['', '## 不确定性与纠错代价', '',
        '下表为新分支减参照的胜平负LL；负值为评分改善。4周日历块条件区间只描述已保存历史概率，'
        '未覆盖研发选择、供应商修订或未来赛季变化。两季已多轮用于研发，不能称新的盲测。', '',
        '| 赛季 | 参照 | LL差 | 条件95%区间 | 改首选 | 纠正 | 改坏 | 改前改后均错 |',
        '|---|---|---:|---|---:|---:|---:|---:|']
    for period, comparisons in result['by_period'].items():
        for name, label in [('pattern_minus_original_rich', '原丰富模型'),
                            ('pattern_minus_previous_draw_branch', '上一版分支')]:
            value = comparisons[name]
            lower, upper = value['four_week_conditional_block_interval']['interval_95'][0]
            c = value['changes']
            lines.append(f"| {period} | {label} | {value['mean_loss_differences']['one_x_two']:+.6f} | "
                f"[{lower:+.6f}, {upper:+.6f}] | {c['changed']} | {c['corrected']} | "
                f"{c['damaged']} | {c['both_wrong_after_change']} |")
    lines += ['', '2025/26相对原模型改选平局15场，纠正4场、改坏7场、改前改后均错4场；'
        '440场实际平局仍漏436场。新分支平均平局概率25.4488%，实际比例25.4925%，'
        '这不代表已找到具体哪些比赛会打平；其AUC为0.5715，仍低于原模型0.5768。'
        '2024/25相对旧分支AUC上升，2025/26却回落，不能声称新增表示已产生稳定排序增益。', '',
        '相对原模型，两季平局比赛获得较高概率，非平比赛承担概率下调成本；'
        '后季这两部分对整体LL差的贡献分别约−0.016798与+0.015328，净差仅−0.001470。'
        '这解释了平均概率更贴近、概率损失略降，却未多选对比赛的现象；不据此归因于某一单独输入。', '',
        '## 五联赛全部保留', '',
        '| 赛季 | 联赛 | 场次 | 原模型正确 | 旧分支正确 | 新分支正确 | 新减原LL | 新减旧LL |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    league_labels = {'E0': '英超', 'D1': '德甲', 'SP1': '西甲', 'I1': '意甲', 'F1': '法甲'}
    for period, leagues in result['by_league_and_period'].items():
        for comp, comparisons in leagues.items():
            rich = comparisons['pattern_minus_original_rich']
            previous = comparisons['pattern_minus_previous_draw_branch']
            lines.append(f"| {period} | {league_labels[comp]} | {rich['n']} | "
                f"{rich['correct_counts']['original']} | {previous['correct_counts']['original']} | "
                f"{rich['correct_counts']['draw_branch']} | {rich['mean_loss_differences']['one_x_two']:+.6f} | "
                f"{previous['mean_loss_differences']['one_x_two']:+.6f} |")
    lines += ['', '联赛切片仅供定位，不按这些后验分数挑选上线联赛。', '',
        '## 比分、总球、让球的含义', '',
        '新分支保持原模型主胜／客胜比，仅重新分配平局质量。最终比分仍采用原条件比分形状，'
        '均值、总球、让球均从区域缩放后的网格重新计算，完整比分NLL直接读取实际比分格子。'
        '在这一分工下，逐场完整比分NLL差与胜平负NLL差代数相同，不是两份独立成功证据。', '',
        '| 赛季 | 新减原比分NLL | 新减原总球0/1/2/3/4+ LL | 新减原客让两球三项LL |',
        '|---|---:|---:|---:|']
    for period, comparisons in result['by_period'].items():
        diff = comparisons['pattern_minus_original_rich']['mean_loss_differences']
        lines.append(f"| {period} | {diff['exact_score']:+.6f} | {diff['total_goals']:+.6f} | "
            f"{diff['away_minus_two_three_way']:+.6f} |")
    missing_by_period = Counter(row['period'] for row in records if not row['pattern_history_available'])
    lines += ['', '## 数据、训练和实际检查', '',
        f"- 历史供应商来源精确联结{coverage['provider_source']['matched']:,}场，d缺失"
        f"{coverage['provider_source']['missing']}场；准备{coverage['source_rows']:,}份原历史输入。",
        f"- 历史不足5场的输入保留，24项为None；目标两季分别{missing_by_period['2024-25']}／"
        f"{missing_by_period['2025-26']}场，没有删除来改善分数。15场窗口可用5–14场，来源数量逐项保存。",
        '- 22个模型与旧分类树逐个核对训练ID、日期、参数设置和时间权重，均一致；标签日期严格<P，'
        '每份输入历史日期严格<F。原模型、预处理和新模型都未把当前被预测目标放入自身训练。',
        '- 新增17项测试与旧16项相关测试共33项通过，覆盖目标／未来污染、相同均值不同模式、'
        '缺失来源不填0、窗口和双方身份。独立复算三线指标一致；3,452个比分网格完整唯一，'
        '概率／均值／总球／让球与记录一致。首、中、末三个保存模型重放完整H/D/A误差为0。',
        '- 以上是实现和日期范围检查，不证明供应商底层xG模型从未使用未来资料。原始发布时间、'
        '修订及其模型训练未知，仍是回溯日期级研究。', '',
        '## 保留了什么，下一步改变什么', '',
        '保留24项历史模式组件、逐窗口来源、22个模型和全部结果。研究支持把均值之外的比赛模式'
        '作为可复用表示；本轮没有证明它们在现有学习器和这些历史赛季中稳定提高胜平负命中，'
        '也没有证明行为、波动或机会分布方向没有价值。', '',
        '后续优先区分两个剩余缺口：一是如何在不牺牲原主客判断的情况下识别具体平局，'
        '二是混合不同历史对手／比分状态的机会数据是否真正表示目标对阵。'
        '本轮统计块只提供无条件历史描述，原其他输入虽已有实力信息，仍不能据此宣称上述条件化问题已解决。'
        '下一步先复用现成记录与公开研究确认可补足的具体信息；涉及重新定义学习目标或主客分工时先讨论。', '',
        '本轮停止点已达到：一次完整两季前推完成。当前预测和LLM／市场线路保留，不把同向当加置信度，'
        '未复核新赛果、未新安装依赖、未把旧历史成绩当作当前预测准确率。所有新增文件、模型、缓存均在G盘。', '',
        f'[信息借鉴与解释边界]({OUT.as_posix()}/信息补充依据与解释边界.md) · '
        f'[完整数值]({OUT.as_posix()}/两季完整结果.json) · '
        f'[逐场概率]({OUT.as_posix()}/逐场概率与纠错.json) · '
        f'[实际接入检查]({OUT.as_posix()}/实际接入检查.json)', '',
    ]
    (OUT / '本轮结果与优化判断.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
