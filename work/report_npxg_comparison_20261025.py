"""Write the readable findings from the saved, completed NP comparison."""
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import json

ROOT = REPOSITORY_ROOT
OUT = ROOT/'outputs/非点球xG对照_20261025T054906945702Z'


def main():
    result = json.loads((OUT/'非点球对照汇总.json').read_text(encoding='utf-8'))
    labels = {'goal': '纯进球', 'xg_raw': '原总xG均值', 'xg_scaled': '总xG＋前季尺度',
              'npxg_scaled': '非点球xG＋前季尺度', 'old_blend': '原进球／总xG融合'}
    lines = ['# 非点球xG独立对照：实跑结果与采用结论', '',
        '**结论：两组后季均有很小的平均改善，但主比较的不确定区间均跨零，尚无足够证据采用非点球线路替换现有模型。保留数据和代码，不继续围绕本结果搜索权重或联赛例外。**', '',
        '用户要求暂停首发后，本轮实际执行较早可用机会信息的首项研究。没有查询或复核当前赛果，没有下载新的比赛资料，原赛前预测、市场和LLM线路均未改动。所有数据、代码、缓存及临时文件在项目目录。', '',
        '## 实际做了什么', '',
        f'原30季资料10,734场中，沿用既有独立比分、日期差异与特殊比赛规则后，可联结{result["source_matches"]:,}场源NPxG。字段完整不改变原目标队列，也不自动解决未知球队。每次NP训练ID、日期、权重和截止与原进球／总xG模型逐一相同。', '',
        '固定设置为正则0.01、半衰期180天、每28天更新、原独立Poisson比分分布。新增214次非点球均值拟合；原214次进球和214次总xG参数原样复用，新增4个闭式尺度估计，无参数网格、无新增融合权重。154项相关测试通过。', '',
        '实跑后另独立核对6,990条校准／检验记录、214组参数的226,801条较早训练引用及4个尺度，错误0项；用独立无限支撑Poisson／Skellam复算五条线路全部评分，与保存结果一致，最大概率差约1.38×10⁻¹²。[独立核对结果](./独立保存结果核对.json)。', '',
        '源NPxG直接取供应商逐场球队历史，并核对球队ID、完整时间、主客身份、双方NPxG／NPxGA及原总xG。没有用简单射门相加或减固定点球值代替，也没有使用首发、同场预测或赛季最终汇总作为输入。', '',
        '## 怎样把非点球信号用于预测全部进球', '',
        '非点球均值不能直接冒充全部进球率。两条机会模型分别在较早赛季的前推预测上估计同一种尺度：', '',
        '`尺度 c = 较早季真实全场进球总和 ÷ 较早季预测机会均值总和`', '',
        '`较晚季全场进球率 = c × 该线路的较早信息预测均值`', '',
        '这是一个主客和五联赛共享的经验均值换算，等价于固定均值下的Poisson似然闭式估计；它不是点球、乌龙球的独立发生模型。总xG接受相同处理，主比较因此能区分去点球信息与一般尺度校准。两条线路各用一个尺度，后季均固定，不按后季结果再估计。', '',
        '| 学习季 → 检验季 | 学习场数 | 检验场数 | 总xG尺度 | 非点球尺度 |',
        '|---|---:|---:|---:|---:|']
    for study in result['studies'].values():
        s = study['scales']
        lines.append(f'| {study["calibration_period"]} → {study["test_period"]} | {s["xg"]["n"]} | {study["overall"]["n"]} | {s["xg"]["scale"]:.6f} | {s["npxg"]["scale"]:.6f} |')
    lines += ['', '后季球队模型继续按原28天规则使用当时已完成的较早比赛，尺度仍固定为前季值。两个检验季共3,466场；两组在方法上复现，但包含正常承接的历史，不能称为完全独立的两次盲测。', '',
        '## 后季完整结果', '',
        'NLL／LL越低越好；最高概率命中数只反映三选一方向，不代表概率质量或盈利。总球为0／1／2／3／4+五档。', '',
        '| 检验季 | 线路 | 完整比分NLL | 胜平负LL | 总进球LL | 最高概率命中 |',
        '|---|---|---:|---:|---:|---:|']
    for study in result['studies'].values():
        v = study['overall']
        for arm, score in v['scores'].items():
            lines.append(f'| {study["test_period"]} | {labels[arm]} | {score["exact_score"]["log_loss"]:.6f} | {score["one_x_two"]["log_loss"]:.6f} | {score["total_goals_0_1_2_3_4plus"]["log_loss"]:.6f} | {round(score["one_x_two"]["accuracy"]*v["n"])}/{v["n"]} |')
    lines += ['', '## 主比较：非点球＋尺度 减 总xG＋尺度', '',
        '负差表示非点球线路较好。主指标是完整比分，胜平负与总球作为次指标；不能挑较好的次指标替代原主指标。', '',
        '| 检验季 | 指标 | 平均损失差 | 2周95%条件区间 | 4周95%条件区间 | 8周95%条件区间 |',
        '|---|---|---:|---|---|---|']
    for study in result['studies'].values():
        v = study['overall']
        for metric, label in [('exact_score', '比分'), ('one_x_two', '胜平负'), ('total_goals_0_1_2_3_4plus', '总进球')]:
            key = f'npxg_scaled_minus_xg_scaled:{metric}'
            index = v['difference_order'].index(key)
            cis = []
            for block in ('2', '4', '8'):
                lo, hi = v['block_intervals'][block]['interval_95'][index]
                cis.append(f'[{lo:+.6f}, {hi:+.6f}]')
            lines.append(f'| {study["test_period"]} | {label} | {v["mean_loss_differences"][key]:+.6f} | '+ ' | '.join(cis)+' |')
    lines += ['', '完整比分的两组差值约−0.00115、−0.00095，所有2／4／8周区间均跨零。胜平负平均差也很小且所有区间跨零。较晚季总球在4／8周区间较好，但2周区间仍跨零，且更早复现未确认，暂不能推广为稳定总球优势。', '',
        '两组逐联赛的方向也不一致：不允许据此给有利联赛专设权重或挑选上线。', '',
        '| 检验季 | 联赛 | 场数 | 比分差 | 胜平负差 |', '|---|---|---:|---:|---:|']
    names = {'E0': '英超', 'D1': '德甲', 'SP1': '西甲', 'I1': '意甲', 'F1': '法甲'}
    for study in result['studies'].values():
        for comp, v in study['by_league'].items():
            lines.append(f'| {study["test_period"]} | {names[comp]} | {v["n"]} | {v["mean_loss_differences"]["npxg_scaled_minus_xg_scaled:exact_score"]:+.6f} | {v["mean_loss_differences"]["npxg_scaled_minus_xg_scaled:one_x_two"]:+.6f} |')
    lines += ['', '## 从本轮能够学到什么', '',
        '1. 去点球信号有小幅有利的平均方向，当前证据不足以改变正式预测。最高概率命中相对校准总xG为929→920、904→909，方向也不一致。',
        '2. 较晚季总xG仅加前季尺度，比分NLL就由2.901540降至2.892157，改善约0.00938；其2／4／8周条件区间均低于零。随后换NP只再改善约0.00095。这说明本轮较大的比分改善主要属于一般进球尺度校准，不能全部归功于新增NP信息。较早季同一尺度校准仅改善约0.00078，区间跨零；尺度本身也不能据单季就当通用提升。',
        '3. 相对现有进球／总xG融合，NP线路更早季比分较差、较晚季略好，胜平负两季均较差，未形成一致替换理由。没有检验新增NP／进球融合，也不因当前小差异马上加一个权重。', '',
        '## 下一步', '',
        '本轮研究到此停止，保留独立NP实现及输出，不进行联赛专属调参。按已提出路线，下一项采用相同早期输入、相同截止和相同目标队列，做一个有限复杂度概率树对照，检验是否存在原线性结构遗漏的交互；不得同时加入PPDA／deep后把算法与新信息的效果混在一起。压迫和深入进攻作为后续单独信息增量。', '',
        '完整跨赛事休息与前后赛程继续作为数据补充方向，持续动态实力另列结构研究。首发路线和当前赛果复核维持暂停，神经网络／强化学习未引入。', '',
        '## 解释范围与复现', '',
        '这是已看过赛果的历史研究，按严格较早日使用资料，不是原始发布时间精确可还原的赛前24小时回放；供应商历史修订未知。2／4／8周日历块共同重采样五联赛，保留空周，但区间条件于当前已估计的模型及尺度，不包含全部模型选择、尺度估计与数据商不确定性，不作盈利或保证命中解释。', '',
        '[分析前计划](./分析前计划.json)、[汇总评分与全部区间](./非点球对照汇总.json)、[数据联结](./逐季数据联结.json)、[更早组逐场预测](./earlier_前推结果.json)、[较近组逐场预测](./recent_前推结果.json)。原排除理由、参数和训练ID均在文件中保留。', '',
        '[运行程序](../../work/run_npxg_comparison_20261025.py)、[源NP读取与均值模型](../../model_v2/src/football_model/npxg_history.py)、[经验尺度换算](../../model_v2/src/football_model/goal_rate_scale.py)。', '']
    (OUT/'非点球实跑与采用结论.md').write_text('\n'.join(lines), encoding='utf-8')
    print(str(OUT/'非点球实跑与采用结论.md'))


if __name__ == '__main__':
    main()
