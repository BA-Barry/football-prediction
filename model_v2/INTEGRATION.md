# 已有组件整合与调用

## 2026-10-29：英超历史对手条件的实际协作

`work/build_epl_adjusted_context_20261029.py`复用既有I1方法，只读取已存英超源及原表现事实。六队原五场不换样，13个日期各固定拟合一次`DateTeamXgModel(.01,180)`，训练严格date<旧场date。`data/epl_adjusted_context_20261029`保存训练ID、模型、30侧原xG与期待／残差。独立按系数手算及来源、窗口核对通过；未知不填零，残差不是跨队绝对实力，原xG与残差不重复计权。

`work/build_epl_assisted_packet_20261029.py`保留原v2三场事实，加入历史条件上下文和官方有界更新的无变更结论。新LLM只读该包，实际输出三组H/D/A，27引用等值；历史模型估计可见而当前目标统计概率不可见，明确标记为辅助线。`work/record_epl_assisted_forecasts_20261029.py`复用既有`multi_record/score_multi`，保留原10场主键及旧概率，增加3条`llm_model_informed`、结果为空，另存`条件表现协作前瞻记录.json`。原记录不覆盖，旧13场权重不迁用，未产生数值融合或新准确率结论。本轮已在实际输出及登记完成处停止，读取结果不需重训。见[三场判断与边界](G:/Codex/football-prediction-2026-10-22/outputs/英超人员协作前瞻_20261029/对手条件接入与本轮判断.md)。

## 2026-10-29：当前英超人员协作前瞻

`work/intake_fpl_live_20261029.py`真实获取官方bootstrap/fixtures，确认2026/27并导出未来未开场fixture与人员文字。`attach_live_fpl_participation_20261029.py`只为最早三场六队选定的29人请求element-summary，利用fixture/was_home/opponent恢复真实同队身份，145条近5场分钟不混入其他队；缺记录保持null。当前原始资料、来源时刻和官方人员陈述在`data/epl_personnel_live_20261029`。

`work/predict_epl_personnel_context_20261029.py`已实际固定拟合现有goal/xG模型各一次，使用严格早于10月29日的1,600/2,360场，最大训练日10月25日；原121项树只加载推断。统计预测与无率／无概率的表现事实包分别保存到`outputs/英超人员协作前瞻_20261029`，不可把完整统计文件交给独立LLM。

`work/build_live_personnel_packets_20261029.py`连接近5/15场实际表现、原人员文字、同队较早参与和官方陈述，当前入口导出v2（含人工web观察时刻语义）。原独立LLM恰在v2准备时已基于v1完成，原v1、生成时间和概率保留；v2新增时间限定另作后附读取说明，未倒写为原输入。干净新调用完成三场主观H/D/A，57条事实引用可解析，12项独立输入测量及7项统计普通检查通过。LLM身份仅能声明本会话继承模型、具体ID未暴露，暴露信息是调用方声明而非外部认证。

`work/record_live_personnel_forecasts_20261029.py`复用`multi_record/score_multi`保存10场统计＋3场独立LLM的本次版本。沿用原fixture主键，new_unique_fixtures=0，results={}；既存模型／LLM记录不覆盖。`joint_pool`缺失原因明确为新政策无此前配对结果拟合，助手只做独立的定性综合。文件已生成，读取不需重训或重新调用LLM；再次执行记录器保留已有记录。当前代码没有加入新人员强度系数或替代原模型结构。详见[本轮结果](G:/Codex/football-prediction-2026-10-22/outputs/英超人员协作前瞻_20261029/本轮协作结果与判断.md)。

## 2026-10-29：人员新闻语义与同截止参与证据

`football_model.personnel_news_semantics.extract_personnel_news(raw_news, raw_status)`仅提取文字引用，原文／源状态／百分比和返回措辞保留，类别不由status决定；不输出能力损失、确认缺席、诊断或补齐归期年份。2,041条来源目录完成一次LLM原文审查，透明规则修正及60项定向测试记录在`outputs/人员信息语义接入_20261028/LLM文本审查.json`；未分类7条保留未知，语义目录见`data/personnel_semantics_20261028/语义目录.json`。

`work/align_personnel_news_history_20261028.py`按原预测F和原确切近5场ID，使用快照同teamcode／playercode对齐。源新闻ID是status+news联合键，比较时仍将raw_status和raw_news分别判断。全8,753场输出保留旧22项及诊断，1,502场有FPL来源。未知、另一队同code、名单缺席均不充健康。6项时点／身份／投影场景测试通过。

`work/build_personnel_semantic_packets_20261028.py`进一步按同赛季FPL id核对回溯身份映射，复用原较早同队分钟，导出`历史LLM人员事实包.json.gz`。完整窗口的实际0分钟与缺窗口／缺身份不同，分钟不等于能力。原始特征文件含历史目标标签，程序仅读取明确字段用于计算与导出；不声称该源文件未被解码。LLM语义视图剔除全历史目录统计与最晚观测跨度，完整存储包不能直接当单场提示词。`--match-id "<existing match id>"`只物化该场和已引用新闻，不提供其他场次／未引用的未来文本。三项按来源机制选择的历史阅读实例不含目标标签，不能拿来学习真实前瞻融合权重。

读取已有产物不需重跑全量对齐；所有文件在G盘。本轮没有新增训练、参数搜索或当前比赛预测，原213项模型保持。来源覆盖、实测变化、LLM理解和边界见[本轮整合与所学](G:/Codex/football-prediction-2026-10-22/outputs/人员信息语义接入_20261028/本轮整合与所学.md)。

## 2026-10-28：历史情境共同训练

入口`work/integrate_historical_context_20261028.py`已经实际完成两次固定设置拟合。输入是扩样8,753场121项，以及`data/historical_context_20261028/{schedule_sources,availability_sources,coach_sources}`三类同F=D−3上下文；目标不因缺失删去。原表现保留，40赛程＋14源状态＋22较早参与变化＋16较早教练观测共同加入，得到213项。原有训练期中位数与缺失指示处理保留，不将未知强行解释为健康或零负荷。

评分模型截止2025-08-01（7,027训练），复用上一轮121项同截止、同1,726目标预测。最终研究模型截止2026-08-01（8,753训练），不用于这批目标的评分。文件位于`outputs/历史情境共同接入_20261028`，包括两个joblib、`整合训练输入.json.gz`、`共同接入训练结果.json`及独立核对。读结果不需再运行拟合。`work/report_joint_context_20261028.py`仅读既存预测生成变化说明、报告及五个不含目标标签的历史资料理解样例，不拟合、不访问赛果。

教练接入脚本`work/intake_historical_coaches_20261028.py`和`measure_historical_coach_context_20261028.py`仅读作者公开归档；明确俱乐部ID与日期联结，原CSV有一项经官网核实的日期修正另列。特征只取date<F的较早同联赛执教标签；最后已观测教练不保证仍是目标教练，连续同名场数和观测更替距今不是任期／任命日期。7项针对性场景通过；原始出处及8,753行全引用时序记录在coach_sources。

历史状态与赛程是存量资料复用，教练观测为本轮新增；不把8753当作全部情境齐全的场数。整个研究保留原算法与比分分布，不使用目标首发、目标实际分钟、目标教练、LLM预测或市场。原当前统计、LLM及综合预测未被替换。结论见[本轮共同接入与所学](G:/Codex/football-prediction-2026-10-22/outputs/历史情境共同接入_20261028/本轮共同接入与所学.md)。

五个历史资料包另做一次LLM理解试用，输出`历史资料LLM理解试用.json`及`LLM资料理解记录核对.json`。仅理解和区分事实／推断／未知，121个JSON路径引用可解析；不输出目标概率或训练强度系数，不增加真实前瞻数，不是LLM微调或完整事实认证。输入包生成时的`llm_calls:0`描述当时状态，不改写原包冒充之后的调用已发生。该试用没有进入上述213项训练。

## 2026-10-28后续：大样本训练与证据协作

`work/expand_rich_training_20261028.py`已经完成一次实际扩样：原lead3的6,967条完整行保留，新增2021/22的1,786场同语义121项特征，合计8,753；36场明确记录当时未知球队。基础专家按55个联赛／参数日期缓存，固定配置树共3次拟合，无网格。`扩样训练结果.json`保留同截止、同1,726目标的两次评分，最终2026候选另存，未用于该批评分。结果和独立重建核查见[扩样目录](G:/Codex/football-prediction-2026-10-22/outputs/协作预测与大样本方案_20261028/扩样训练/)。读取报告无需再运行训练。

`work/build_collaborative_packet_20261028.py`生成新输入版本，包含原统计概率、当季及重建对手校正资料、官方陈述。未提供赔率、其他LLM或目标结果。新LLM明确已见统计；`work/summarize_collaboration_20261028.py`保存助手综合，保留单独H/D/A并逐场解释共享依据、新情境、分歧、反证与使用限制，不产生新数值融合。数据包发布后不应通过再次执行构造脚本冒充原LLM当时见过更新输入；有新资料另建版本。

`work/record_contextual_llm_20261028.py`复用`multi_record/score_multi`接入该情境LLM，保留原统计与独立LLM的ID、生成时间及概率。新`llm_model_informed`明确当前概率曝光，不借用旧独立LLM的13场权重。初始评分五场全部pending；后续可对其`records`及真实结果调用`score_multi`，不得将新版本计为五场新比赛。原预测、原包和旧辅助LLM均保留。

本轮结束后停止拟合。后续先盘点有可靠事件日期的历史情境覆盖，区分统计大样本、LLM资料提取研究与真实前瞻，见[协作分工与后续训练](G:/Codex/football-prediction-2026-10-22/outputs/协作预测与大样本方案_20261028/协作分工与后续训练.md)。

## v0.52：已学习权重的模型／LLM联合与对手校正上下文

`football_model.learned_pool.LearnedLinearPool().fit(rows,before)`仅使用`result_observed_at < before`的真实赛前双线记录；每个match_id一次，双线生成均早于开球，结果已知时间晚于开球。输入包含kickoff、statistical/llm_generated_at、statistical/llm_probabilities、actual_outcome和结果时间；市场不参与。`llm_track`声明独立／模型辅助／unknown，原样保留模型family与未知身份。单一统计权重w最小化三项胜平负LogLoss，使用凸导数求根并比较两个端点；平坦损失报告不可识别，不造50:50。`predict(...,match_id,kickoff,as_of)`拒绝训练同场与不合法时间。

`multiline.learned_arithmetic_pool`复用原记录／评分入口，保留`earlier_prospective_log_loss_fit`、拟合截止、训练ID、已知结果时间及完整参数。独立LLM训练权重不能套给模型辅助LLM；原`arithmetic_pool`及`explicit_untrained_research`语义保留。现有`multi_record`／`score_multi`均能区分手动与已学权重，不将训练元数据当成提准认证。

执行`work/run_learned_model_llm_pool_20261028.py --mode run`已完成两次标量拟合（较早10场、当前13场），两者均统计1／LLM0。当前5场真正新联合已存档，其概率与原主线相同；`--mode report`只读结果补报告，不重拟合或刷新旧概率。新辅助LLM独立保存cohort并等待它自己的已结算配对，不混用来源政策。详见[报告与边界](G:/Codex/football-prediction-2026-10-22/outputs/模型LLM融合研究_20261028/本轮联合学习与结果.md)。

`work/build_current_adjusted_context_20261028.py`已复用2359场意甲xG，为当前10队近5场的42场历史比赛按16个日期各拟合一次，仅使用该日期以前资料。输出每场原始、期望、攻防残差／净差和训练日期；缓存避免重跑。重建期望是现在计算、源修订日期未还原，不是当时真实发布记录；未知不填0，缺失不改选更早比赛。新辅助LLM仅见此包、不见当前p、赔率或旧LLM，仍因见历史模型估计而明确标模型辅助。资料与解释见[接入说明](G:/Codex/football-prediction-2026-10-22/data/serie_a_adjusted_context_20261028/训练与来源说明.md)。

## v0.51：固定平局概率与条件主客方向

用户明确同意后，`work/run_conditional_direction_20261028.py --mode run`已执行完22个日期窗，保存22个条件学习器及3452场完整H/D/A、比分网格。`football_model.conditional_direction.ConditionalDirectionLearner`复用原ExtraTrees设置，仅从date<P且H/A的历史拟合q；缺失处理也只看这些训练行，目标包含全部实际平局。前期无可用持续状态的窗口保留121项，其余127项。`pH=(1-d)q,pD=d,pA=(1-d)(1-q)`，d复用已有组件，不重训。新网格保留v0.50的条件比分形状，不能将其比分差异全归q。

`--mode inspect`只检查既存输入；`--mode summary`只从保存预测汇总，不训练。实际研究已结束，不需为读取结果再运行训练。两季正确930／889，原923／901；同d参照922／898。新q未确认稳定改善，不用于当前比赛。55项相关测试通过，独立按时间与训练ID重建、3452场保存模型重放最大差7.77e-16。详细接口与限制见[结果与所学](G:/Codex/football-prediction-2026-10-22/outputs/条件主客方向研究_20261028/本轮结果与所学.md)及[独立机制核对](G:/Codex/football-prediction-2026-10-22/outputs/条件主客方向研究_20261028/独立核对与机制说明.md)。

本轮还完成昨日意甲3场原概率结算与今日5场新前瞻。当前统计基础线/xG各真实拟合1次，121项树复用0拟合；独立LLM仅见纯事实包，市场单源去水另列，未平均、未使用新q。新资料、模型及结果全部保存G盘。

## 2026-10-28：当前预测入口修复

`work/run_current_rich_forecasts_20261026.py`现在于资料加载／拟合前检查过期批次，同日复用已有特征接口，真实生成结束后复查开球；原确认快照的观察时间单列保存，不改成now。旧`current_projection`也将其既有截止前置并保留结束检查。88项相关检查通过；算法、已有预测、资料档案均保留，未发布新概率。详见[修复结果与边界](G:/Codex/football-prediction-2026-10-22/outputs/当前预测入口修复_20261028/修复结果与边界.md)。

## 2026-10-28：存量胜平负机制收束

`work/diagnose_reused_outcome_mechanisms_20261028.py`只读取8路既存完整概率，按相同3,452场核对上下文后复用精确损失分解；不拟合、不生成组合。新产物见[既存概率的机制分解](G:/Codex/football-prediction-2026-10-22/outputs/胜平负机制复用收束_20261028/既存概率的机制分解.md)。其条件方向指标只用于实际非平比赛的事后诊断，不能筛选赛前比赛。

`work/prepare_conditional_direction_plan_20261028.py`在准备阶段复用当前6,967行和现存平局组件，仅测量22窗更早非平训练量及可用状态列，不训练、不预测。后续明确同意、22窗实际执行及未采纳结论见本页顶部；原方案保留于[结构讨论](G:/Codex/football-prediction-2026-10-22/outputs/胜平负机制复用收束_20261028/下一步结构讨论.md)。它不能被描述为首次直接三分类，或视为数学分解天然提准。

## 当前可用性与证据收束

`current_availability.compare_fpl_snapshots(current, earlier, current_observed_at=..., earlier_observed_at=...)`按来源球队代码＋球员代码比较。状态、新闻文字、新闻时间元数据与来源百分数字段变化分开；转会／新增／缺失不猜配，u不归伤病，未知不归健康。原始FPL名单及官网陈述均在G盘`data/current_availability/official_20261027/`，4项边界测试通过。

`work/continue_current_availability_20261027.py build`把已有球队表现、完赛／计划赛程、当前FPL及官网可用性投影进同10场无预测概率事实包。一次无历史对话LLM已完成10场，身份无法确知时如实记unknown；`finalize`只适配其原生成时间与回答，复用原统计track，不调用LLM。额外调用既有`evidence_reconciliation.reconcile_evidence`做模型知情的定性解释，不产生球员系数、联合概率或固定伤停扣分。产物见[可用性接入与判断](G:/Codex/football-prediction-2026-10-22/outputs/当前可用性接入与判断_20261027/可用性接入与判断.md)。

新组`next_availability_llm10`与上一计划杯赛LLM按同场比较；本轮9组仍为20场不同英超比赛。可用`finalize --freshness-path <后续来源观察JSON>`在报告附注较晚取数状态；较晚观察不进入旧LLM事实包，不刷新其预测时间。本轮12:22来源再次核对无变化，故没有第二次LLM生成。官网旧伤与当前截至10/25表现可能重叠，下一轮可用性、停赛范围和替代能力仍留未知。

## 已公布计划赛程与独立LLM

`planned_schedule.planned_schedule_context(target, events, as_of=..., source_scope=...)`将已完成、未来计划、状态未明和延期取消记录分开，过滤目标自身、目标以后及观测晚于截止的记录。间隔明确为开球到开球小时，不估计实际休息或轮换，不修改统计概率；空列表只解释为所查范围无记录。

`work/continue_planned_context_20261027.py build`从G盘已归档的三份单日杯赛响应与白名单描述统计构建`data/current_all_competition/planned_20261027/facts_only_llm_packet.json`。完成了一次无历史对话LLM整批判断；`finalize`只适配其原回答、生成时间与原161统计记录，不调用LLM或重训。报告见[计划赛程与独立判断](G:/Codex/football-prediction-2026-10-22/outputs/计划赛程与独立LLM_20261027/计划赛程与独立判断.md)。新LLM原回答、事实包和旧预测均保留。计划赛程3项边界测试与原多线评分70项测试通过。

新组`next_planned_cup_llm10`通过现有评分入口单列，与`next_rich10`的旧LLM按同场比较；同时接入`next_allcomp_repaired10`后为8个版本、20场不同英超比赛，不增加比赛样本数。统计列完整复用原记录及时间；独立LLM只指未见其他路线概率，共用事实可能导致误差相关，没有平均融合或准确率改善结论。

## 已存真实前瞻的统一评价

全赛事取数已新增`allcomp_daily_intake.py`，复用原`AllCompetitionFeatures`：单日响应的结构／日期／赛事验证、冲突不覆盖的事件合并、成功日的连续覆盖区间。`work/refresh_allcomp_daily_20261027.py fetch`实际归档13赛事×28日及球队接口，`build`只从已归档响应计算覆盖。原宽日期请求和旧资料保留。

`work/predict_repaired_allcomp_20261027.py`加载原161模型与原121表现字段，只更新40项赛程输入；新预测的生成时间与原资料日期分开。它不拟合、不调用LLM或结算赛果。已完成10场，旧161重放率值误差0；同161输入修复最大变化0.93个百分点，最高单项改变0场，准确率尚待评价。报告见[资料修复与独立预测](G:/Codex/football-prediction-2026-10-22/outputs/全赛事缺口修复与预测_20261027/修复结果与独立预测.md)。

评价入口可用`--additional-cohort next_allcomp_repaired10=outputs/全赛事缺口修复与预测_20261027/标准前瞻记录.json`接入新版本。新增版本单列，不能覆盖原有组名；已实际运行7个版本，仍为20场不同英超比赛。

`work/continue_prospective_evaluation_20261027.py`复用现有`multiline.score_multi`，连接原轮首次／修订、下一轮原版／资料修订、10月26日简单／丰富候选6个版本。每个版本内只保留一条比赛记录，跨版本只按共同比赛评价，不拼接成更多样本、不择优覆盖首次记录。数据截止与实际概率生成时间分开；新导出时间不作为预测时间。原市场缓存不补入新版本，当前未生成的联合概率仍缺失。

默认读取既存赛果，不联网、拟合或调用LLM；也不将2025/26回测并入前瞻。输出位于[前瞻评价接续](G:/Codex/football-prediction-2026-10-22/outputs/前瞻评价接续_20261027/前瞻评价接续与优化安排.md)。全五线、每对线路的共同覆盖及自身覆盖分别保留，不能混用分母。

## v0.44：保留原概率，只学习增量

上一轮全量概率关系学习产生明显类别取舍退步；已继续实现`football_model.anchored_residual.AnchoredResidualStack`。相同七条成员和人员输入先形成22项自然尺度差分，不再学习原概率/联赛/类别截距，不做中心化或方差标准化。原rich_performance的log概率是固定offset：

`Q = softmax(log(P_reference) + gamma * X @ B)`

B为三类对称L2系数。自然尺度L2改变了先验约束，不应将它视作删除StandardScaler的等价操作。即使无截距，成员差分的非零均值也可能改变整体类别比例。

接口：`fit(rows,before_date)`、`residuals_many(rows)`、`predict_many(rows,attenuation=1.0)`、`parameters()`。行结构与下述七路输入相同。零残差或`attenuation=0`逐位恢复原始raw rich_performance概率；非英超人员差分为0只表示此修正分支未启用。

`fit_attenuation(rows,before_date)`只接收真正较早的联合时间外记录，以NLL学习一个0到1的幅度，显式保留两个端点。本轮学到0.477731；它限制增量，不对整个原概率另施温度。较早小样本模型与最终全季模型的幅度迁移仍有不确定性。

```python
from pathlib import Path
import json
import joblib
from football_model.probability import score_grid, summarize
from football_model.score_reconciliation import reconcile_score_probabilities

output = Path('G:/Codex/football-prediction-2026-10-22/outputs/锚定增量整合_20261026')
model = joblib.load(output / 'models/anchored_residual_2025-08-01.joblib')
gamma = json.loads((output / '较早学习的修正幅度.json').read_text(encoding='utf-8'))['attenuation']
final_hda = model.predict_many([row], attenuation=gamma)[0]
grid = reconcile_score_probabilities(score_grid(home_rate, away_rate), final_hda)
markets = summarize(grid)
```

仍需准备当时可知、同语义的七路概率和原进球率，不能用旧赛季最后一次概率替代当前输入。该示例不生成新比赛预测。运行入口`work/run_anchored_residual_20261026.py`；会执行预定的5次增量拟合和1次幅度学习。后季结果899/1726、LL0.990586：相对上轮恢复27场，尚未超过原901场；概率评分点值改善但区间跨零，总球稍差。详见[本轮结果](G:/Codex/football-prediction-2026-10-22/outputs/锚定增量整合_20261026/增量整合结果与优化结论.md)。

以下保留v0.43原始整合的实现说明；其全量stack与温度不是v0.44的额外必经环节。

本轮是在已有信息和已训练概率之上学习联合关系，没有重训七条基础路线。用户目标是把已有积累用到一起并观察效果；本次已完成整合实跑，但完整后季结果未改善，不覆盖既有比赛记录。

## 流程与边界

```text
既有丰富表现 / 全赛事 / 进球 / xG / GAP / KickScore / 多路评级概率
                    ＋ 英超状态、参与的同设增量
                                  ↓
             29项概率差异、覆盖及少量强弱交互
                                  ↓
                 固定正则多项逻辑联合模型
                                  ↓
             只用较早联合样本外概率学习的温度
                                  ↓
                    最终主胜 / 平局 / 客胜
                                  ↓
          按结果区域重分配原丰富模型比分条件形状
                                  ↓
               一致的比分、总球、让球及新均值
```

该流程不假设七条来源独立。成员高度共享数据，差分表示与L2用于共同学习，但不能把系数解释为独立信息贡献或因果。没有历史LLM概率和可靠D−3报价时，不伪造对应成员；它们继续按独立概率/证据线路保留。

已知首发、实际同场分钟和旧供应商个人系数不进入早期联合；人员残差来自FPL状态/历史参与研究，并非完整球员能力模型。其他联赛的人员资料没有覆盖，以显式无覆盖分支处理。

## 两个可复用模块

`football_model.integrated_stack.IntegratedProbabilityStack`：

- `fit(rows, before_date)`：只接收比赛日早于该日的训练记录。成员概率应由调用者确认是当时较早资料形成的时间外预测。
- `predict_many(rows)`：返回按主/平/客排列的概率；拟合日必须不晚于预测发布日期。
- `parameters()`：特征、标准化、系数、训练范围、版本与收敛记录。

每行保留`match_id, date, forecast_before_date, competition`。`probabilities_by_route`需要`rich_performance, rich_allcomp, goal, xg, gap, kickscore, rating_joint`七组正且归一的H/D/A概率。人员字段为`availability_control, availability_status, availability_usage`，英超可三者齐备或全部未知，其他联赛全部为`None`。训练另需`actual_outcome`；预测不使用标签/比分。

`football_model.score_reconciliation.reconcile_score_probabilities(grid, target_hda)`：

- 从原网格自身计算H/D/A区域概率；按目标重分配，不从另一版本的概率向量取分母。
- 保持每个结果区域内部的条件比分比例，不能因此学习区域内部的1:1/2:2差异。
- 返回新网格并重算均值；总球、让球必须由此派生。
- 原分布某区域没有支持时明确报错；不会编造该区域比分。

## 使用已有整合模型

以下只是调用接口，完整的新预测还需要同提前量、同语义且当时已知的七条成员输入。不能把当前2026/27资料缺失用2025/26最后概率代替。

```python
from pathlib import Path
import json
import joblib
from football_model.prematch_calibration import apply_calibrator
from football_model.probability import score_grid, summarize
from football_model.score_reconciliation import reconcile_score_probabilities

output = Path('G:/Codex/football-prediction-2026-10-22/outputs/现有组件整合_20261026')
model = joblib.load(output / 'models/integrated_stack_2025-08-01.joblib')
calibrator = json.loads((output / '整合温度参数.json').read_text(encoding='utf-8'))

# row遵循上述字段；home_rate/away_rate是该行原丰富模型的赛前进球率。
raw_hda = model.predict_many([row])
final_hda = apply_calibrator(raw_hda, calibrator)[0]
grid = reconcile_score_probabilities(score_grid(home_rate, away_rate), final_hda)
markets = summarize(grid)
```

比分分布按P(比分|结果)保留，因此完整比分NLL增量等于1X2 NLL增量。它提供一致性，不代表修复了更差的H/D/A或增加新信息。跨场之间也没有由此建立独立性保证。

## 复现与实际结果

真实运行入口是G盘`work/run_integrated_components_20261026.py`；该入口会执行预定的四次较早联合拟合、一次最终拟合及一次温度拟合。`work/describe_integrated_components_20261026.py`只读存盘结果作变化定位，不训练。

2024/25成员时间外预测用于学习；内部90日热身、之后60日扩展窗口产生1203场联合时间外概率以学温度。2025-08-01最终拟合后固定，2025/26完整1726场：原丰富表现901场正确，整合872场，校准后LL 1.004480。没有根据后季改参数、删成员或增加变体。历史已多轮查看，不是全新盲测。

详细结果与组件收获见[本轮报告](G:/Codex/football-prediction-2026-10-22/outputs/现有组件整合_20261026/整合结果与所学.md)及[变化定位](G:/Codex/football-prediction-2026-10-22/outputs/现有组件整合_20261026/变化定位与可复用结论.md)。本地只运行了新增16项针对性测试和本轮实际整合；没有重新执行旧全套审计。

## 条件平局分支复用（v0.46.0研究接口）

该接口不拟合参数，也不自动采用分类器的主／客胜判断。调用者需提供同一场比赛、同提前量且确实在预测时点可用的平局概率；目前已完成的历史整合仅为提前3个日历日，不能直接冒充同日预测。

```python
from football_model.draw_branch import apply_draw_branch, replace_draw_probability
from football_model.probability import score_grid, summarize

# home_rate/away_rate来自原模型；d是已核对类别顺序的分类树平局分量。
original = score_grid(home_rate, away_rate, rho=0)
new_hda = replace_draw_probability(original.home_draw_away, d)
combined = apply_draw_branch(original, d)
markets = summarize(combined)
```

`replace_draw_probability`要求原H/D/A严格为正且归一、d严格位于(0,1)，不填造缺失值或平滑零值。新H/D/A为`((1-d)*pH/(pH+pA), d, (1-d)*pA/(pH+pA))`，保持主客胜比例。`apply_draw_branch`直接复用既有区域缩放，返回新网格；均值、总球及让球从该网格重算，不能用新均值重新构造独立Poisson后评分。

完整比分NLL应为`-log(combined.grid[actual_home_goals, actual_away_goals])`。本分工下逐场完整比分NLL差、H/D/A NLL差和二元平／非平NLL差相同；这用于检查接入正确性，不是三项独立增益。

固定重放入口为G盘`work/run_draw_branch_integration_20261027.py`，只读已有两季时间外预测，不训练或学习权重。完整结果与逐场网格保存在[条件平局分支整合](G:/Codex/football-prediction-2026-10-22/outputs/条件平局分支整合_20261027/整合结果与所学.md)。两季表现方向不一致，保留研究资产，未替换当前比赛预测。

## 历史比赛模式输入（研究组件）

`match_pattern_features.MatchPatternFeatures`复用`EarlyTeamObservation`，返回主队视角的24项历史描述和逐窗口来源。它不训练模型，也不直接给出目标比赛的平局概率。

```python
from football_model.match_pattern_features import MatchPatternFeatures, FEATURE_NAMES

# provider_draws[(competition, match_id)]是过去已完成比赛的机会分布d，缺失为None。
# 绝不能传入目标比赛实际机会统计，或把目标比赛的赛后forecast当赛前预测。
builder = MatchPatternFeatures(observations, provider_draws)
snapshot = builder.for_match(home=home, away=away, competition=competition,
                             cutoff=forecast_date, match_date=match_date)
pattern = snapshot['features']
if pattern is None:
    pattern = {name: None for name in FEATURE_NAMES}
match_features = {**archived_121_match_features, **pattern}
```

own/opponent分别代表目标主队／客队，各取严格早于cutoff的最近5／15场同联赛比赛。六种描述为：npxG净差绝对值均值、npxG总量总体方差、npxGF与npxGA总体协方差、实际平局比例、过去机会分布d均值及总体方差。双方各需至少5场；15场窗口未满时用实际数量，并在metadata中记录。任一来源d缺失，该窗口两项d描述均为None，不补0或跳到更早比赛；训练端继续复用仅在较早训练集拟合的imputer。

这些是混合历史对手与主客场的经验描述，协方差不是目标条件比分相关参数。供应商d是过去完赛机会描述，和相应xpts存在代数关系，因此不再重复加入xpts；其原始发布时间、修订和内部模型训练资料未知，不能声称完整复现了历史当时可得的信息。

固定接入入口为G盘`work/run_match_pattern_draw_20261027.py`：在旧121项上增加24项，原分类树设置和训练比赛范围保持一致，22次前推仅取新d，再复用上面的统一比分接口。旧rich／旧draw分支完整保留作同场描述，不自动替换当前预测。结果见[历史比赛模式接入报告](G:/Codex/football-prediction-2026-10-22/outputs/历史比赛模式与平局分支_20261027/本轮结果与优化判断.md)。

## 已保存射门的比分状态描述（研究资料接口）

`football_model.understat_shot_states.reconstruct_shot_states(shots, expected_full_time=None, match_id=None)`只描述已有历史事件，不输出目标比赛的概率，也未进入当前模型。`shots`为原始`h`、`a`两个射门数组。

```python
from football_model.understat_shot_states import reconstruct_shot_states

states = reconstruct_shot_states(payload['shots'],
                                 expected_full_time=independent_match.full_time,
                                 match_id=understat_id)
```

该源没有半场标识和秒级时钟。仅在来源采用全场累计分钟的前提下，解释`minute<45`；其余状态均为未知。更早分钟的进球计入射前比分，同分钟其他进球保留全部可能先后；当前进球不计入自身射前比分，OwnGoal归给射门方的对侧。ID与数组顺序仅保留作来源信息，不用于判定先后。

每条分别返回`possible_scores`、`unique_score`、`possible_own_states`、`unique_own_state`；比分不唯一时，本方领先／持平／落后仍可能唯一。未知为`None`，不能当作0:0或持平。独立终场比分只核对进球记账；冲突时撤下派生状态，不倒推修复。源`h_goals/a_goals`不参与状态重建；每射门xG原样保留，不直接相加冒充源NPxG。

一次760场完整测量已保存20,402条事件，其中8,625条本方状态唯一、127条早段状态有歧义、11,650条时钟范围未知。6项针对性测试通过，完整记账760/760一致。这些是资料覆盖和接入正确性结果，不是预测提升，也不是完整时间暴露或人数状态。实跑入口与结果见[测量说明](G:/Codex/football-prediction-2026-10-22/outputs/平局条件信息定位_20261027/射门比分状态测量说明.md)。

## 丰富信息与持续状态接入（v0.50.0研究组件）

`persistent_rich_integration.plan_windows(rows)`只从`date<P`的训练前缀判断六个新增状态列是否有数值；`window_rows`深拷贝并仅剪去训练中全空的新增列，原121列始终保留。目标值不能决定训练列。首窗全空时复用原模型，避免无新信息时改变随机抽列。

`work/run_persistent_rich_integration_20261028.py`默认`--mode inspect`只实际复用首窗旧模型；`--mode run`沿用原22窗口和原学习器完成一次接入，首窗复用、其余21次拟合。模型、训练范围、逐场H/D/A、比分网格、总球、损失分解、明确改对改坏和CSV均在`outputs/持续状态与丰富信息接入_20261028/`。已实际完整运行；无参数搜索、上游重训、LLM／市场融合或新平局分支。

两季单选931／893，原923／901，合计不变；LL点值略降而条件区间跨0，不作为采用中的新预测。当前原算法、进球目标与独立Poisson结构保留，因此此版本不是直接胜平负训练。全局状态参数的资料截至日仍严格约束早期行的可用性，不回填未来。实际使用与局限见该目录结果报告。

## 持续球队状态与净胜球概率（v0.49.0研究组件）

同日后续新增`PersistentGoalDifferenceFilter.explain_state()`和`persistent_state_features.build_persistent_state_features(static, dynamics, history_after_static, targets)`，用于抽取已学到的信息，不拟合、不生成新概率、不改变旧`forecast`结果。`explain_state`返回按查询F推进的两队攻防坐标、锚点、偏差、已观察次数和日期；未知队公开坐标为None。

桥接函数除观察日期<F外，还要求`dynamics['training_max_date'] < F`，不满足时新增六列全None、原目标仍保留。返回每侧六项本方／对手攻防、基础项与本侧主场项（客队0），参数日期、可用性和历史计数留在元数据。实际6,967行输入中前3,515行不具有当时可用的参数，不能用最终状态回填；后3,452行与旧GAS率重建误差0。真实模型训练发生在2026年，时间条件只描述回测资料前缀，不声称历史真实发布。随后已完成上述v0.50独立接入研究，不能把代码及模型产物接通叫已提准模型。

`persistent_goal_difference.py` 改变独立概率源头，不是给原模型概率加权。`fit_static_panel`在2022/23的净胜球上估计各联赛截距、主场项和居中球队攻防锚点，固定L2=4；`fit_dynamics`在2023/24估计四个共享参数（攻／防更新幅度、攻／防每七日留存率），单起点一次优化，无参数网格。此前静态锚点固定，持续偏差随日历间隔推进，换季不重置；同日比赛用共同日初状态计算梯度后再合并更新。

`forecast_targets(static, dynamics['parameters'], history_after_static, targets)`返回完整原顺序记录及最后目标信息前缀的状态。调用方须保证全局参数的`training_max_date < min(target.F)`；内部只吸收`match.date < F`。目标需要`F/date/competition/home/away/original_hda/match_id`，历史需要实际日期和全场比分。全部历史和目标输入示例见G盘本轮输入资料。预测不读目标比分，且不会推进状态。

输出`gas_probabilities`为主／平／客，`net_goal_distribution`保留逐净胜球概率及左右未列尾；这两个潜在率不宣称是真实预期进球。新线不输出总球／完整比分，也不调用旧三分区网格去伪造相同净胜球尾部。原比分、LLM和市场独立保留。

零历史球队的`gas_available=False`、三概率为null；`deployment_probabilities`在此时逐值回退`original_hda`，其余才取新线。中性新队坐标仅用于第一次观察的算法启动，没有经验先验积分或后验不确定性。结果须同时报告新线覆盖、相同覆盖比较、全覆盖回退规则，不能把回退成绩归给新模型。

`PersistentGoalDifferenceFilter.state_dict()`可JSON保存；`from_state_dict(saved)`恢复后继续观察或预测。状态带真实已吸收日期、计数、锚点和偏差。条件状态不是贝叶斯后验；数值尾容差也不是预测置信度。本研究不替换已保存的当前比赛预测，使用过的历史赛季不称新盲测。实现和一次完整研究结果见`outputs/持续状态净胜球研究_20261028/`。

## 对手预期下的攻防残差模式（v0.48.0研究组件）

`football_model.xg_residual_patterns.summarize_residual_patterns(residual_pairs)`只汇总恰好五场的团队视角残差。每条提供有正负号的`residual_for`和`residual_against`，分别为实际创造／允许的**总xG**减去该旧场较早总xG专家的预期。`against>0`表示允许机会多于预期，不表示防守改善。

```python
from football_model.xg_residual_patterns import summarize_residual_patterns

summary = summarize_residual_patterns([
    {'residual_for': actual_for - expected_for,
     'residual_against': actual_against - expected_against}
    for actual_for, actual_against, expected_for, expected_against in five_verified_matches
])
```

返回`xg_residual_{metric}_5`八种描述：for/against/net/total均值、净差绝对值均值、净差／总量总体方差、for/against总体协方差。负残差、负协方差是合法值；布尔、非有限数拒绝。缺少完整五场时整组为`None`，不能跳过缺场、取更早比赛或按较少场数计算。

这些投影共享信息：净均值=for均值−against均值，总均值=两者之和，协方差=(总方差−净差方差)/4。新增列也会改变固定`max_features`下随机树的分裂机会，不能称为八条独立证据或因果球队实力。

项目适配器位于G盘`work/xg_residual_pattern_features_20261027.py`的`build_residual_pattern_rows(original_rows, actual_xg_records)`。它复用旧`build_adjusted_rows`的身份、来源和时序核查，再在原两侧121项输入上分别加入own/opponent两组，共137列。每条参考保留实际／预期攻防分量、旧fit/F/比赛日及ID，满足旧fit≤旧F<旧比赛日<目标F；目标比赛的实际xG和首发不参与。实际总xG与源NPxG是不同口径，禁止相减混用。

固定整合入口为G盘`work/run_xg_residual_pattern_integration_20261027.py`：沿用同一120棵、深度6、最小叶60的进球树、180天权重和22个历史日期，拟合只用日期早于参数日的成对主客进球标签；中位数与缺失指示只在训练期拟合。现有上游专家和平局分类器不重训。新进球率构造原比分网格后，复用v0.47保存的d及`apply_draw_branch`，最终比分、H/D/A、总球、让球从同一网格读取。

原模型、纯新进球层、已有平局整合、新联合四阶段只用于分工定位，不按较晚成绩选择上线。联合与已有平局整合的d逐场相同，因此平局AUC／二元平局损失不变；主客比例、平局是否最高和区域内比分仍可能变化。两季各1,726场完整结果见[本轮整合结果与所学](G:/Codex/football-prediction-2026-10-22/outputs/对手校正攻防模式整合_20261027/本轮整合结果与所学.md)。本组件不自动适配当前赛季或其他提前量。

## 历史射门与分钟事件条件重建（2026-10-28）

`football_model.archive_shot_context.reconstruct_archive_context(shots, minute_rows, expected_full_time=..., expected_half_time=..., clock_offsets=(0,1,2))`仅生成已完赛历史资料的条件联合外包集合，不输出概率或中性修正后的xG。射门输入使用明确的`side` h/a、整数`minute`、`id`和`result`，可保留原`xG`；分钟输入使用作者`half_id`、整数`minute`、明确`side`以及`goals/red_cards`事件计数。源Score.Diff和RedCard.Diff不参与状态赋值，适配器另存原值及CSV行号。

重复分钟状态行累计事件，不择首尾、不按行数当时间。先按作者半场再按分钟累计；同分钟先后未知时枚举事件前缀，Goal与OwnGoal排除自身。比分或窗口一一关联不一致时撤回整场派生状态，保留原数据。每个候选坐标及其联合状态成对保存，不跨坐标拼接边际集合。任何分支未知均不填`window_unique_*`，`all_branches_covered`也只是声明假设下覆盖，不是精确时钟证明；全局一一匹配仅提供存在性，不保证每个局部候选均能扩展到某个全局关联。

半场标签为作者推断，第一半场最大观察分钟、第二半场从46起均是研究假设；红牌只表示原受罚记录，本队减对手为正表示本队记录更多，不是已核实人数差。0/1/2偏移有地标证据但并非官方时钟。未知标志显式依赖历史射门结果：**不能用于预测同一脚进球，不能只筛唯一状态样本训练。**后续用于更晚赛前的整场机会表示仍须处理结果相关测量偏差。

一次实际重建入口为G盘`work/reconstruct_archive_shot_context_20261028.py`，只读既有1099场身份联结及两份原资料，输出逐射门候选、原分钟事件与分结果覆盖。未改变现有特征接口、拟合、赔率、LLM或当前预测。详见[本轮报告](G:/Codex/football-prediction-2026-10-22/outputs/射门分钟状态重建_20261028/历史射门状态重建与下一步.md)。
