"""Report the completed broad model study without selecting on later results."""
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
OUT = ROOT/'outputs/多信息多算法前推_20261025T060704977450Z'
NAMES = {'goal': '原纯进球', 'xg_raw': '原总xG', 'old_blend': '原进球／xG融合',
         'glm_basic': '线性／基础信息', 'glm_rich': '线性／丰富信息',
         'hgb_basic': '提升树／基础信息', 'hgb_rich': '提升树／丰富信息',
         'extra_trees_rich': '随机树／丰富信息'}


def main():
    result = json.loads((OUT/'多方案结果汇总_评分剔除未完赛.json').read_text(encoding='utf-8'))
    selected = result['selected']
    validation, test = result['validation'], result['test']
    main_key = f'{selected}_minus_old_blend:exact_score'
    index = test['difference_order'].index(main_key)
    main_delta = test['mean_loss_differences'][main_key]
    all_negative = all(v['interval_95'][index][1] < 0 for v in test['block_intervals'].values())
    conclusion = ('较早选择的方案在较晚比分主指标上优于原融合，仍需更晚未查看比赛与完整概率校准验证，暂作研究候选。'
                  if all_negative else '本轮尚未确认较早选择的方案能稳定改善原融合，不替换当前正式预测，也不根据后季结果改选赢家。')
    lines = ['# 多信息、多算法完整预测方案：探索实跑与核心问题', '',
        f'**结论：{conclusion}**', '',
        '**解释限制：官方资料新增确认两场未完成90分钟的历史比赛。评分已对所有方案统一剔除，但其中蒙彼利埃一场曾进入65／110组训练，以及原球队专家和历史特征。当前分数仅是保留旧拟合后的探索对照，不能称为清理完成后的验证；下一轮重建整条训练链。**', '',
        '按用户要求，本轮扩展为整套预测方案比较，覆盖信息组合、学习方法、时间前推和误差分组。当前赛果复核与首发线路继续暂停，市场、LLM及原赛前概率版本保留。所有代码、模型、数据及缓存位于项目目录。', '',
        '## 本轮扩展范围', '',
        '每个得分侧的基础输入为10项：原球队进球和总xG模型对双方的较早预测、主客场及五联赛类别。丰富输入合计131项，加入近5／15场进失球、xG、非点球xG、PPDA分子分母与汇总比值、深入进攻、对手对应表现、近期减中期趋势、相关主客场子样本、历史覆盖和同联赛日期间隔。它们由原30季独立联结的10,711场产生。', '',
        '较少历史不删除比赛。96个历史样本在不足5场时保留基础输入，并以缺失及可用标记表达丰富信息不足；预处理只从每次更早训练集学习。PPDA比值按窗口分子合计／分母合计计算，零合计分母保持缺失。相关主客场均值来自最近15场中的主／客子集，未虚称15个主场或客场。', '',
        '| 方案 | 学习方式 | 信息范围 | 固定复杂度 |',
        '|---|---|---|---|',
        '| GLM基础 | 正则Poisson回归 | 10项 | alpha0.01 |',
        '| GLM丰富 | 正则Poisson回归 | 131项 | alpha0.01 |',
        '| HGB基础 | 直方图梯度提升树 | 10项 | 150轮、最多7叶、叶至少60样本 |',
        '| HGB丰富 | 直方图梯度提升树 | 131项 | 同上 |',
        '| ExtraTrees丰富 | 多随机树平均 | 131项 | 120树、最多6层、叶至少60样本 |', '',
        '树模型学习多变量交互，未添加神经网络或强化学习。三类回归器都以真实全场进球为目标，输出主客进球均值，再用相同独立Poisson比分分布评分；这没有自动解决过离散、主客相关或全部模型不确定性，随机树平均也不是比分概率混合。', '',
        '[scikit-learn HGB官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)支持Poisson损失及对数连接；本轮关闭默认自动早停及随机验证，迭代数预设。另见[ExtraTrees官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)和[Poisson回归官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html)。使用本地已安装1.9.1，没有新增依赖或整包复制后宣称增准。', '',
        '## 时间推进与选择', '',
        '2022/23与2023/24为初始训练，2024/25逐次前推比较五个方案，随后按完整比分NLL最低选择一次；2025/26检验时继续按原28天规则利用已完成的较早比赛更新，方案结构保持原选择，全部五方案仍完整报告。两个得分侧始终随同一比赛一起进入训练或检验。', '',
        f'历史特征样本共{result["feature_rows"]:,}场，原较早比较1734场、后季1733场，新增完整模型{result["new_model_fits"]}次，参数搜索网格0个。新增完赛核对后评分分别为{validation["n"]:,}场、{test["n"]:,}场；没有把剔除评分误称为重训。每场训练特征保留其原预测截止，目标及训练样本在五方案间一致。', '',
        '原2024/25融合权重曾用该季结果估计，因此从本轮较早比较和选择中排除；2025/26原融合使用前季权重，可作后季参照。基础专家率在2022/23—2023/24从2020年起训练，2024/25—2025/26从2022年起训练；五方案共用这一既定输入，不能把新旧总体差异全部归因于算法。', '',
        f'**原始2024/25比较记录的选择：{NAMES[selected]}。评分剔除后保留这次选择，不依据后季另选方案；下表为剔除评分后的描述，不是清理训练后的新选择。**', '',
        '| 较早方案 | 比分NLL | 胜平负LL | 总球LL |', '|---|---:|---:|---:|']
    for arm, metrics in validation['scores'].items():
        lines.append(f'| {NAMES[arm]} | {metrics["exact_score"]["log_loss"]:.6f} | {metrics["one_x_two"]["log_loss"]:.6f} | {metrics["total_goals_0_1_2_3_4plus"]["log_loss"]:.6f} |')
    lines += ['', '## 较晚赛季结果：全部方案', '',
        '损失越低越好。最高概率命中率是三选一方向表现，不能代替概率评分或解释为收益率。', '',
        '| 方案 | 比分NLL | 胜平负LL | 胜平负Brier | 总球LL | 最高概率命中 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm, metrics in test['scores'].items():
        lines.append(f'| {NAMES[arm]} | {metrics["exact_score"]["log_loss"]:.6f} | {metrics["one_x_two"]["log_loss"]:.6f} | {metrics["one_x_two"]["brier_sum"]:.6f} | {metrics["total_goals_0_1_2_3_4plus"]["log_loss"]:.6f} | {round(metrics["one_x_two"]["accuracy"]*test["n"])}/{test["n"]} |')
    lines += ['', f'### 较早选定方案减原融合：{NAMES[selected]}', '',
        '| 指标 | 损失差 | 2周95%条件区间 | 4周95%条件区间 | 8周95%条件区间 |',
        '|---|---:|---|---|---|']
    for domain, label in [('exact_score', '完整比分'), ('one_x_two', '胜平负'), ('total_goals_0_1_2_3_4plus', '总进球')]:
        key = f'{selected}_minus_old_blend:{domain}'
        idx = test['difference_order'].index(key)
        cells = []
        for block in ('2', '4', '8'):
            lo, hi = test['block_intervals'][block]['interval_95'][idx]
            cells.append(f'[{lo:+.6f}, {hi:+.6f}]')
        lines.append(f'| {label} | {test["mean_loss_differences"][key]:+.6f} | '+' | '.join(cells)+' |')
    lines += ['', '### 信息组合与学习方法', '',
        '以下所有比较使用相同后季比赛。负差表示前者较好，完整2／4／8周区间保存在汇总文件；这些结果用来解释本轮方案，不能据后季再挑配置。', '',
        '| 对比 | 比分差 | 胜平负差 | 总球差 |', '|---|---:|---:|---:|']
    for arm, reference in [('glm_rich', 'glm_basic'), ('hgb_rich', 'hgb_basic'),
                           ('hgb_basic', 'glm_basic'), ('hgb_rich', 'glm_rich'),
                           ('extra_trees_rich', 'hgb_rich')]:
        keys = [f'{arm}_minus_{reference}:{d}' for d in ('exact_score', 'one_x_two', 'total_goals_0_1_2_3_4plus')]
        lines.append(f'| {NAMES[arm]} − {NAMES[reference]} | '+' | '.join(f'{test["mean_loss_differences"][k]:+.6f}' for k in keys)+' |')
    lines += ['', '### 五联赛逐项保留', '',
        '| 联赛 | 场数 | 较早选定方案比分NLL | 原融合比分NLL | 差值 |', '|---|---:|---:|---:|---:|']
    names = {'E0': '英超', 'D1': '德甲', 'SP1': '西甲', 'I1': '意甲', 'F1': '法甲'}
    for comp, v in result['by_league'].items():
        lines.append(f'| {names[comp]} | {v["n"]} | {v["scores"][selected]["exact_score"]["log_loss"]:.6f} | {v["scores"]["old_blend"]["exact_score"]["log_loss"]:.6f} | {v["mean_loss_differences"][main_key]:+.6f} |')
    lines += ['', '## 误差分组', '',
        '在看后季结果前定义了原模型总球低于2、2至3、至少3，以及双方进球率之比低于1.5／至少1.5的分组。下表仅描述预测误差形态，不将其解释为伤停、红牌或战意造成，也不用它来筛比赛或重调参数。', '',
        '| 原模型分组 | 场数 | 真实平均总球 | 原预测总球 | 选定方案预测总球 | 比分损失差（对原纯进球） |',
        '|---|---:|---:|---:|---:|---:|']
    group_names = {'all': '全部', 'low_total_lt2': '原总球<2', 'middle_total_2to3': '原总球2至3',
                   'high_total_ge3': '原总球≥3', 'balanced': '双方率比<1.5', 'clear_favourite': '双方率比≥1.5'}
    for name, v in result['failure_diagnostics'].items():
        lines.append(f'| {group_names[name]} | {v["n"]} | {v["actual_mean_total"]:.3f} | {v["goal_mean_total"]:.3f} | {v["selected_mean_total"]:.3f} | {v["differences"]["exact_score"]:+.6f} |')
    lines += ['', '## 找到的具体预测失误机制', '',
        '丰富GLM对Cremonese–Parma曾输出53.87／0.075球。对应的“前两场已观察同联赛比赛间隔”是812天，较早训练范围只有3至99天；标准化值约110，仅该项对数均值贡献约4.15，相当于乘约63倍。科隆的同类输入463天，对应乘数约10.46。重点样本全部131项有值，缺失指示为0，并非插补造成。', '',
        '这些是球队在当前联赛资料中断档的间隔，不能当成球队实际休息了数百天。当前比赛之前最近一次已观察比赛本身很近，异常出在更前面两次已观察记录之间。丰富线性模型在训练范围外沿对数连接外推，形成极端均值；这解释了数值机制，不证明具体体能、战意或因果影响。', '',
        '[外推失误诊断](./丰富线性外推失误诊断.json)保留原值、训练范围、系数贡献及对照预测。未裁剪进球率、删除坏预测、增加惩罚网格或改选后季赢家。完整全赛事日期可帮助重新定义真实负荷，资料断档则应作为覆盖不确定性，不能混同真实休息。', '',
        '## 官方确认的不完整历史目标', '',
        '[法甲官方关于蒙彼利埃—圣埃蒂安的决定](https://ligue1.com/fr/articles/l1_article_2269-le-point-sur-la-lutte-pour-le-maintien-020425-l1)确认2025-03-16比赛中止后以0-2处理；[南特—图卢兹的官方决定](https://ligue1.com/fr/articles/l1_article_5189-discipline-toutes-les-decisions-l1-270526)确认2026-05-17比赛中止后确认0-0。数字比分一致、来源标记完成都不足以说明踢满90分钟。', '',
        '两场已加入既有SPECIAL_MATCHES例外处理，原CSV及历史预测未改写，未来日期级进球拟合将排除。南特一场没有进入110组训练或历史特征，剔除评分不需重训；蒙彼利埃一场进入65组新模型、原进球和总xG各13组法甲参数，以及12场随后验证样本历史，因此重建需同时清理原专家、历史特征和新学习器。', '',
        '本次只统一剔除两场评分，原1734／1733评分和全部模型仍保留；新1733／1732表仅作评分敏感性。原先几轮xG／融合及NP研究采用旧例外表，同样需要按该新事实复核，不能继续把旧分数当作已完整处理所有异常。', '',
        '[新增不完整比赛引用范围](./新增不完整比赛引用范围.json)明确记录受影响范围，这不是已测量的重训后概率变化。五场西甲旧接口“待赛但已有90分钟比分”的矛盾另保留，不仅凭该标签认定它们实际未完赛或任意删掉训练；需要官方或其他可靠历史依据。', '',
        '## 采用与后续', '', conclusion, '',
        '本轮完整方案不含经过验证的全赛事负荷。并行任务已从历史正式赛程开始扩展五联赛全队资料，单独核对联赛覆盖、杯赛／欧战、友谊赛与日期范围；其数据未混入本次训练，因此不能把本轮分数当作全赛事赛程信息的验证。', '',
        '下一步按新例外规则重建基础球队专家、历史特征及五方案训练；同时用已接通的完整跨赛事前序日期／场数重构负荷，区分真实间隔、资料断档与赛事类型，再整体评估信息组合及非线性学习。杯赛比分的90分钟／加时口径尚未核实，先使用可信日期和场数，不直接拿杯赛全场数字训练进球或臆造加时分钟。更复杂的持续动态状态空间仍另列；需要神经网络时再与用户讨论。', '',
        '保留统计、xG、市场和LLM线路。LLM后续可以基于时间明确的客观赛程、公告整理情境，但本轮没有让LLM读取历史赛果后编赛前概率，也没有用缺精确时点的收盘赔率倒填早期输入。', '',
        '## 验证与局限', '',
        '217项相关测试通过，含两场新例外仍保留原分数、但不进入全场进球拟合的测试。独立重建6,990场的207,071条历史引用，并复算110组模型564,610条较早训练引用、评分和保存模型重放一致。审计证明旧口径下的计算／时间正确，不证明所有资料都踢满90分钟。首次运行的主场标记合并问题已修正并整体重跑，旧部分模型未计入结果。', '',
        '这些历史赛果此前已经查看，属于回溯探索；较早年份选择不使后季重新变成盲测。全部指标和日历块区间条件于当前已估计模型，不含全部选择、参数、资料修订或赛程公告不确定性。仅保证日期级更早资料规则，不声称还原精确赛前24小时。', '',
        '五联赛国内的联赛类别与球队专家输入不等于跨联赛绝对实力尺度，本轮不能直接证明欧战预测已成熟。缺完整伤停、预计轮换、市场早期快照等仍会限制实时应用。', '',
        '[评分剔除后汇总](./多方案结果汇总_评分剔除未完赛.json)、[原始结果汇总](./多方案结果汇总.json)、[原较早选择](./较早选择.json)、[全部原逐场预测](./逐场预测.json)、[历史特征核对](./独立历史特征核对.json)、[模型与评分核对](./独立模型与评分核对.json)、[运行程序](../../work/run_broad_early_models_20261025.py)。完整模型保存在同目录models，未改原赛前预测。', '']
    (OUT/'完整方案对照与后续计划.md').write_text('\n'.join(lines), encoding='utf-8')
    print(str(OUT/'完整方案对照与后续计划.md'))


if __name__ == '__main__':
    main()
