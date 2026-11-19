# GitHub 模型复审：完整方案、可复用部分与比较边界

复审日期：2026-10-26。共核读 12 个不同项目的 README 与关键实现；同时核对公开许可证及维护元信息。本轮没有安装依赖、执行上游代码、训练模型或修改正式预测代码。作者报告的分数均未由本项目复现。

**本轮不能确认我们的模型优于这些项目。** 现有模型的输入、目标、算法和信息截止时间与其他方案并不相同。已有局部实验能说明“该接法在该数据与窗口下没有稳定收益”，不能说明整个球员、战术、射正、深度学习或多模型联合方向无效。把这种局部结论外推到整个方向，需要更正。

## 先改变什么

优先从论文官方仓库的“多路动态强度 → XGBoost 联合预测”完整骨架入手，其次移植独立分类概率与比分矩阵的一致化模块。先形成一个有明确输入、目标和输出的完整候选，再比较整体表现；整体有何长处、哪些输入产生协同效应，可以随后用成组消融解释。不能始终只给现有模型增加一列，失败后就排除这个信息领域。

这也不要求一次堆上所有模型。现阶段重点是：已有真实比分、射门/射正、xG、Elo、市场与上下文，是否通过合适的强度表示、交互学习和概率输出一起发挥作用。LLM 更适合整理可核查的上下文和提出情景；其参与预测的效果，需要与这些概率路线分别记录。

## 项目对照

“已研究”表示原工程曾研究或适配相关思想，不表示原工程已经实现、训练、比较过其整个上游项目。维护时间来自本轮 GitHub API，近期推送本身不证明预测质量。

| 项目与定位 | 实际可借鉴内容 | 时间、数据与效果的限定 | 许可 / 最后推送 UTC |
|---|---|---|---|
| [sparsity-sports-forecasting](https://github.com/spoho-datascience/sparsity-sports-forecasting)，新增，论文官方代码 | 结果、进球、pi、市场、身价、射门/射正分别形成强度，再由 XGBoost 联合；另有每队状态 LSTM；缺失机制、样本覆盖和信息丰富度一起研究 | 12 个赛季每年 4–5 月长提前量检验；有真实实现，但预处理存在必须修复的目标信息覆盖及本机路径，缺完整数据 | Apache-2.0 / 2024-05-31 |
| [Reymes/football-match-prediction](https://github.com/Reymes/football-match-prediction)，新增，联合预测工程 | DC + LightGBM + 市场 → 逻辑 stack → 温度校准；分类概率约束比分矩阵，使各市场内部一致 | 日期 train < val < test；作者自报 7,626 场时间外结果与市场几乎相同，未证明胜过市场；历史赔率必须按我方 D-3 对齐 | MIT / 2026-07-11 |
| [Soccer-Prediction-Challenge-2023](https://github.com/calvinyeungck/Soccer-Prediction-Challenge-2023)，新增，论文配套 | 有真实 PyTorch DNN、数据拆分和预测代码；关联论文比较 CatBoost + pi 与 DNN | 论文比赛任务、联赛和时间窗口不同；仓库主要是 DNN 线，不能把论文全部 CatBoost 实现都说成已提供 | Apache-2.0 / 2024-04-26 |
| [footAI](https://github.com/pmatorras/footAI)，新增，球队与市场联合 | Elo、近况、均势/平局、联赛、开盘至收盘变化联合；分顶级与跨级两个版本 | 实际采用 TimeSeriesSplit；收盘信息不适用于 D-3；文档不同版成绩不一致；未校准为投注用途，杯赛/欧战不在作者范围 | MIT / 2026-10-26 |
| [FootWork](https://github.com/lrivals/FootWork)，新增，134 特征联合比较 | 丰富统计、xG、Elo、赔率隐含概率、主客、疲劳；boosting、校准、分层分类和决策阈值 | 正常路径为较早训练、2020–21 校准、2022 起测试；CatBoost 以测试集作 eval_set 的配置须修正/确认；二分类约 72% 不是 1X2 72% | 未发现明确 LICENSE / 2026-02-28 |
| [SIM2WIN](https://github.com/Mouad-Ze/SIM2WIN)，新增，战术风格交互 | 压迫/夺回球、射门质量、直接性等风格比率；训练期聚类，主客风格对位、跨赛事对照和模拟解释 | 模型 notebook 是随机分层切分；另有 LOCO；本场 starting_formation 在特征中，缺失阵型以本场填充，不符合不等首发/D-3 | 未发现明确 LICENSE / 2026-05-04 |
| [penaltyblog](https://github.com/martineastwood/penaltyblog)，已研究，比分工具库 | 相关比分、低比分修正、层次 Bayes、后验与统一比分/盘口接口 | 不能拿本地一个固定分布修正当作整个库的效果；工具库没有单一通用胜率 | MIT / 2026-10-19 |
| [footBayes](https://github.com/LeoEgidi/footBayes)，已研究，动态层次模型 | 周/赛季动态攻防、联合比分、跨季收缩、不确定性和后验预测 | R/Stan 依赖及迁移成本较高；本地简化平局修正远不等于全套动态 Bayes | DESCRIPTION 为 GPL-2 / 2026-10-14 |
| [kickscore](https://github.com/lucasmaystre/kickscore)，已研究，动态技能 | GP 时间核、潜在实力演化、未来预测与未知状态不确定性 | 固定时间衰减试验不覆盖整个 GP 动态类别；不自动提供球员、战术或全部比分关联 | MIT / 2025-09-05 |
| [panna](https://github.com/peteowen1/panna)，已研究，个人贡献到球队 | 事件/boxscore 球员贡献、预计分钟、历史 fallback 与球队聚合 | 需要丰富球员/事件数据；本地伤停数或 FPL 分钟不是 RAPM；按时点训练仍须适配，用户目前可将其延后 | LICENSE.md 为 MIT / 2026-10-26 |
| [soccer-betting-models](https://github.com/pabsanamono/soccer-betting-models)，新增，可插拔多路线 | 统计/树/神经网络接口、shift(1) 滚动与更新前 Elo、尾部校准和回测 | 默认含 closing odds；未核到足以证明预测优势的同条件成绩；README 许可说明未核到完整顶层 LICENSE | README 称 MIT，文本不完整 / API 403 |
| [MatchBench](https://github.com/Dr-yato/MatchBench)，新增，国际赛 stack | 比较 argmax、平局阈值与概率；时间切分；三种 boosting + DC 的 meta 模型 | 国际赛强弱悬殊场占比大；均势与全体成绩差异很大；pi 跨场地实现与原参考仍需核对；不优先于论文官方代码 | MIT / 2026-06-20 |

## 第一候选：不是再拼一列，而是复用完整多信息骨架

[官方仓库](https://github.com/spoho-datascience/sparsity-sports-forecasting)的 [ELOXGBoostModel.py](https://github.com/spoho-datascience/sparsity-sports-forecasting/blob/main/models/evaluation/ELOXGBoostModel.py) 明确将结果、进球、pi、赔率、市场价值、射门、射正生成不同 rating；[input/utilities.py](https://github.com/spoho-datascience/sparsity-sports-forecasting/blob/main/input/utilities.py) 定义完整特征组合；[Experiments.py](https://github.com/spoho-datascience/sparsity-sports-forecasting/blob/main/models/evaluation/Experiments.py) 实际包含基础、多信息、全信息与多种缺失处理的组合。它可以直接作为完整结构适配的起点，避免重新从一个单指标实验开始。

它与目前主要在固定 ExtraTrees 进球回归器上逐项增加输入的方式不同：先让不同原始比赛量形成有对手和时间含义的强度，再由模型学习它们的共同作用。XGBoost 在树的不同分支可表达“同样近况，在实力差或市场均势不同条件下含义不同”的交互。是否在我们的数据上有效尚未知，但已经有结构与文献依据值得直接采用后适配，不能因我们一次射正修正无增益而排除。

该库的 LSTM 也是实质代码：[network.py](https://github.com/spoho-datascience/sparsity-sports-forecasting/blob/main/models/deepmodel/network.py) 分别维护每队 hidden/cell state，并提供进球和 WDL 任务。深度学习能否发挥作用与样本量、输入丰富度和数据缺失一起相关；不能把其简单视作“黑盒无必要”，也不能因存在 LSTM 就推断会超过当前模型。该阶段若要训练深度线，按用户偏好先讨论。

这里需要把“研究有效”与“公开脚本可直接运行”分开。已查到的具体适配问题如下：

- [input/prediction.py](https://github.com/spoho-datascience/sparsity-sports-forecasting/blob/main/input/prediction.py) 144–174 行原本计算历史射门/射正估计，185–188 行随后却用本场实际 HTS/HST/ATS/AST 覆盖 est_* 列。如果照原样给我方提前预测使用，将输入本场结果统计。应保留从历史计算的估计并只用 F 前数据补缺。这个问题属于公开实现的具体路径，不能由此断言原论文所有实验无效。
- 同文件的部分均值补缺来自整个 data，需改成预测时点前；升降级标记有回写上一场记录的逻辑，必须使用当时已知信息重建。
- [数据切分代码](https://github.com/spoho-datascience/sparsity-sports-forecasting/blob/main/input/_save_data_splits.py) 使用原数据库 CSV；完整数据没有随仓库给出。README 明确说 full dataset 可向作者索取。训练/测试 CSV、Transfermarkt 市场价值、赔率和比赛统计都须映射到现有数据，不能声称已经接通。
- 无根级依赖清单，多个研究脚本含个人 F 盘/相对路径；LSTM 状态处理也需按我方运行方式核对。不会原样执行这些脚本向其他盘写文件。

已将这个小型官方代码包实际保存到本报告目录下的 `upstream/sparsity-sports-forecasting`：69 个源码/文档/JSON/LICENSE，合计 485,692 字节。没有下载数据或模型权重，没有安装或执行。Python 主线需 pandas、numpy、scipy、scikit-learn、xgboost、joblib、statsmodels，深度线另需 PyTorch。其 MongoDB 和抓取层不是必须带入的依赖，可以直接对接我们已有清洗数据。

## 第二候选：联合以后，概率关系应能讲清楚

[Reymes 项目](https://github.com/Reymes/football-match-prediction)把分类预测与比分分布区分后再明确协调，[predict.py](https://github.com/Reymes/football-match-prediction/blob/main/match_predict/pipeline/predict.py) 的 `reconcile_matrix_to_1x2` 按主胜、平局、客胜三个区域缩放比分矩阵。这比简单把几条路线概率平均更具体：可以让分类线专门学习 1X2，同时保留比分线在各结果区域内部的分布形状，再给出一致的比分、总球和让球概率。

这只是一个可解释的一致化假设，不是自动提准。区域缩放会改变原预期进球；该项目也明确从最终矩阵重新读出展示的均值。若采用，必须比较总球和正确比分是否受到影响，不能只看 1X2 好转。用户提出“分别收束后联合”可以在此得到一种实际实现，但不能说它是唯一最佳联合方式。

其 [backtest.py](https://github.com/Reymes/football-match-prediction/blob/main/match_predict/evaluation/backtest.py) 真实分开训练、验证、测试；stack 在基础模型时间外验证预测上训练，再对测试输出。仓库还清楚区分生产训练卡中在 meta 自己拟合集计算的乐观指标与独立测试指标。这些模块比“ensemble 在 README 中写了多少命中率”更值得复用。

作者自报 22 联赛、7,626 场时间外测试，市场 / 校准 ensemble 分别为：LogLoss 1.0026 / 1.0025，Brier 0.5998 / 0.5998，RPS 0.2038 / 0.2036，ECE 0.0135 / 0.0125，accuracy 50.6% / 50.6%；配对 LogLoss 差 -0.00007，95% 区间 [-0.0019,+0.0019]。它没有证明击败市场。本轮未复现这些数值，且其赔率时点仍需对齐我们的 D-3，不能直接将这组成绩与我们比较。[来源：项目 README](https://github.com/Reymes/football-match-prediction/blob/main/README.md)。

## 多指标为什么必须一起看，但不能随意挑有利指标

[FootWork 的真实文档](https://github.com/lrivals/FootWork/blob/main/docs/model-performance.md)直接展示了冲突：多分类总体准确率、macro AUC、平局召回会随决策阈值和模型选择朝不同方向变化。其约 72% 是“客胜 vs 非客胜”二分类，绝不能拿来当胜平负单选 72%。文档中“更高 AUC 就等于更好校准”也不成立，两者需分别看。复审应保留其多目标取舍的长处，同时指出具体限制，不能整包赞成或否定。

[MatchBench](https://github.com/Dr-yato/MatchBench/blob/main/README.md)则显示覆盖的影响：国际赛全部测试 4,600 场的 stack 准确率为 60.3%，其作者划定的均势 1,290 场为 42.8%。这是不同难度集合，不能拿全部成绩替代用户关心的均势比赛表现；也不能把那 42.8% 与另一份论文的 55.8% 直接比高低。平局阈值提高召回，也不自动表示平局概率更准确。其 README 把某一市场 53.6% 写成一般上限，不采用这种未经证明的普遍化说法。

对我们当前任务，应同时记录以下不同问题，且在同一比赛集合与同一信息时点下比较：

- 1X2 单选正确率，以及主胜/平/客胜各类表现，避免只靠多数类提高总数。
- LogLoss、Brier、RPS 等概率质量，以及分档校准和过度自信。准确率上升而错误场更加自信，是实质风险。
- 比分和总球的概率质量、让球分布的一致性。1X2 不能自动代替这些市场。
- 不同联赛、实力差、数据覆盖、赔率区间、提前量的表现与样本量；筛选后命中率必须连覆盖率一起报告。

这些指标不是事后挑一个漂亮分数来宣布成功。可以明确“胜平负是本轮首要改进目标，同时记录概率质量与总球变化”；发生取舍时向用户说明，而不是为了某个指标强行把其余输入或路线判为无价值。

## 对旧结论的校准

| 本地已有结论最多能支持什么 | 不能据此否定什么 |
|---|---|
| 某个历史射正修正或单个回归系数，在两段检验中没有稳定增量 | 射门质量/数量作为多路对手校正强度，与实力、市场和比赛情景发生交互的完整方案 |
| FPL 状态计数或状态×历史分钟的现有接法，效果不稳定 | 球员贡献、预计分钟、替补情景、战术风格的全部价值；它们根本不是同一种表征 |
| 某一个单权重市场融合没有确认增量 | 分路线校准、条件融合、时间外 stack 或比分区域协调；这些结构与单权重不同 |
| 固定分布修正或一套跨季收缩设定无稳定改善 | footBayes 的动态层次/后验、penaltyblog 整个分布族、kickscore 全部 GP 核 |
| 现有样本下一个直接分类器没有改善 | CatBoost+pi、合适表征的 DNN、多信息联合分类的全部研究方向 |

反方向也成立：别人项目有丰富特征、很多算法、理论来源或更高公开分数，不表示可以不做任何本地适配就宣称有效。文献能减少重复验证机制的工作，不能消除球队 ID、输入时间、缺失处理和输出目标的具体差异。

## 下一步的实际范围

建议从 A1 官方 rating + XGBoost 完整主线开始，对接当前有的数据，修掉公开预处理的本场信息覆盖与本机路径；明确缺失的市场价值输入，不能造值或把本场首发价值拿来用。市场信息可保留独立支路，D-3 历史不齐时先用历史市场强度，不能偷换成收盘快照。随后借用 A2 的概率输出一致化，先保留为独立候选，不直接替换正式模型。

本轮研究已有收束点：12 个实质不同项目、源码与许可证据、一个实际保存的官方小代码包、优先适配方案和具体缺口。不会继续堆更多同质项目，也不因某个小试验不显著而无限做参数搜索。下一阶段应出现可运行的完整候选及其同条件对比，而不再只有研究名单或孤立指标测试。

完整逐项目证据、已读文件、输入、依赖、比较边界与作者自报结果见同目录 `GitHub证据.json`；本轮上游原始文件在 `GitHub原始证据`，官方代码包在 `upstream/sparsity-sports-forecasting`。
