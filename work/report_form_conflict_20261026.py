"""Complete the planned saved-forecast uncertainty readout; no estimator fits."""
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np

from football_model.forecast_diagnostics import moving_block_means

OUT=Path('G:/Codex/football-prediction-2026-10-22/outputs/近期长期冲突诊断_20261026T034109989122Z')


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def main():
    source=read(OUT/'完整分组汇总.json')
    uncertainty={}
    for lead in ('3','1'):
        saved=read(OUT/f'提前{lead}日_逐场分组与误差.json')['records']
        uncertainty[lead]={}
        for period in ('2024-25','2025-26'):
            rows=[r for r in saved if r['period']==period and r['signal']['group']=='conflict']
            values=[]
            for r in rows:
                c=r['signal']['recent_supported_class']
                y=int(r['actual_outcome']==c)
                p0=r['goal_probabilities'][c];p1=r['rich_probabilities'][c]
                values.append([100*(p1-y),100*(p0-y),100*(p1-p0)])
            result=moving_block_means([r['date'] for r in rows],np.asarray(values),block_weeks=4)
            result['column_order']=['rich_recent_single_win_bias_pp','goal_recent_single_win_bias_pp',
                                    'rich_minus_goal_probability_pp']
            uncertainty[lead][period]=result
    payload={'created_at':datetime.now(timezone.utc).isoformat(),'source':'完整分组汇总.json',
        'event':'One home/away win event supported by relative recent goal residual;not double chance.',
        'scope':'Same planned conflict groups,conditional4week intervals;no new groups,thresholds or fits.',
        'horizons':uncertainty}
    (OUT/'冲突单胜概率偏差区间.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# 本轮结论：近期信息是否修正过头','',
        '**目前没有发现“近期信号与较长期模型冲突时，丰富模型整体变差”的证据；现有结果也不足以认定近期影响已经最合理。暂不降低近期权重，不增加手写罚分，不改已发布概率。**',
        '', '本轮复用已保存的五联赛2024/25、2025/26逐场赛前概率；主要观察提前3日，提前1日用同一定义核对。没有重新训练、调参或复核当前比赛赛果。',
        '', '## 实际比较','',
        '每队最近5场逐一对照当时赛前预期：真实净胜球减预期净胜球，统一球队方向后取均值。双方近期残差差R与目标较长期进球模型主客差L方向相反，才归为冲突。该定义已部分处理历史主客场和对手强弱，但仍依赖模型，不能等同真实实力变化。',
        '', '| 提前3日的冲突比赛 | 2024/25 | 2025/26 |','|---|---:|---:|',
        '| 比赛数 | 744 | 746 |',
        '| 简单模型胜平负LogLoss | 1.01799 | 1.03017 |',
        '| 丰富候选胜平负LogLoss | 1.00747 | 1.02622 |',
        '| 候选减简单模型（负为更好） | −0.01052 | −0.00395 |',
        '| 概率评分差的95%条件区间 | [−0.01647, −0.00479] | [−0.01066, +0.00019] |',
        '', '较早季改善较清晰；较晚季平均也更好，但区间略跨零，不能说已确认稳定增益。Brier评分的方向一致。提前1日后季也平均改善，两个提前量包含大量相同比赛，不能当作两份独立证明。',
        '', '## 局部概率偏高，与整体评分改善可以同时出现','',
        '提前3日的2025/26冲突组，近期信号支持一方的单独获胜概率，简单模型平均为26.01%，丰富候选为27.08%，实际发生24.93%。候选较简单模型提高1.07个百分点，平均高估2.15个百分点。该数是组内单一事件的平均偏差，不能称为整个模型“准确率低了2.15%”，也不代表完整校准曲线。',
        '', '| 提前量／赛季 | 候选平均偏高（百分点） | 95%条件区间 |','|---|---:|---:|']
    for lead in ('3','1'):
        for season in ('2024-25','2025-26'):
            r=uncertainty[lead][season];ci=r['interval_95'][0]
            lines.append(f"| {lead}日／{season} | {r['mean'][0]:+.2f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] |")
    lines+=['','冲突定义使“近期信号支持的胜方”同时是简单模型原本较弱的一方。随机树把极端概率向中间收缩，也会抬高这个概率；因此不能把这约1个百分点的修正全部归因于追随近期战绩。',
        '', '概率评分同时评价主胜、平局、客胜以及不同比赛中的调整；一个分组中的某个单胜事件平均偏高，仍可能与整体评分改善并存。不能仅凭均值偏高，就统一减掉2个百分点。',
        '', '## 没有支持的调整','',
        '“近期比分转好，但机会质量没有改善就直接降权”也未获得本轮支持。用最近5场与此前不重叠10场NPxG净值作旁证时，这类冲突比赛后季的整体概率评分仍优于简单模型。',
        '', '不足完整15场的冲突组后季表现较差，但仅14—15场，且属于已看历史的细分样本。它只能提醒覆盖不足，不能成为新规则或参数依据。',
        '', '## 本轮决定及后续','',
        '保留现有近期输入和参数；原预测维持不变。近期与较长期表现冲突、尤其只有比分改善的比赛，继续作为前瞻记录中的解释项，仍完整列出主胜、平局、客胜，不人工调概率或抬高置信度。',
        '', '下一步使用已经发布的未来10场记录检验概率表现，并积累多个比赛周；不会因一两场输赢立即改权重。本轮停止扩大历史分组和尝试新阈值，也不重跑已有模型。',
        '', '这次只分析统计线。LLM没有相同口径的长历史概率，不能把统计线结论套到LLM；独立LLM与统计线继续分别记录。',
        '', '## 限制与复现','',
        '历史年份此前已被查看，本轮是回溯诊断。近期残差按5场定义，不是纯粹的最近1—2场因果效应；丰富模型与简单模型还相差xG、非线性映射等。当前预测提前5—7日的适用性，仍不能由本次提前1／3日结果代替。',
        '', '分组原值见提前3日_逐场分组与误差.json、提前1日_逐场分组与误差.json；全部评分见完整分组汇总.json；概率偏差区间见冲突单胜概率偏差区间.json。所有文件位于G盘。本轮0次拟合、0次新预测、0轮调参。']
    # Take displayed point scores from source rather than rounded hand entry.
    for season in ('2024-25','2025-26'):
        m=source['horizons']['3'][season]['groups']['conflict']
        if season=='2024-25': first=m
        else: second=m
    for i,line in enumerate(lines):
        if line.startswith('| 简单模型胜平负LogLoss'):
            lines[i]=f"| 简单模型胜平负LogLoss | {first['goal_log_loss']:.5f} | {second['goal_log_loss']:.5f} |"
        elif line.startswith('| 丰富候选胜平负LogLoss'):
            lines[i]=f"| 丰富候选胜平负LogLoss | {first['rich_log_loss']:.5f} | {second['rich_log_loss']:.5f} |"
    report=OUT/'本轮结论与执行决定.md'
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(report),'bias_intervals':{l:{s:{'mean':r['mean'],
        'interval_95':r['interval_95']} for s,r in seasons.items()} for l,seasons in uncertainty.items()}},ensure_ascii=False))


if __name__=='__main__': main()
