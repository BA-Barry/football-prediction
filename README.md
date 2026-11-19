# 足球赛前预测与模型／LLM协作研究

个人研究项目：以球队历史表现估计完整胜平负与比分分布，保留市场、独立LLM和模型辅助LLM的不同判断，并记录真实赛前生成时间与后续评估依据。

## 仓库内容

| 路径 | 内容 |
|---|---|
| [`model_v2/src/football_model`](model_v2/src/football_model) | 球队进球率、比分分布、历史前推验证、球员与人员上下文、市场概率、LLM多线路记录等实现 |
| [`model_v2/tests`](model_v2/tests) | 概率计算、时间边界、缺失信息、来源对齐及记录评分的测试 |
| [`work`](work) | 数据获取、研究训练、核对与报告脚本；日期表示对应研究任务 |
| [`outputs`](outputs) | 已有研究报告，以及2026-10-29英超示例的完整概率与运行记录 |
| [`data/epl_adjusted_context_20261029`](data/epl_adjusted_context_20261029) | 六队近期表现的对手／主客场条件解释及13个较早日期的小型xG模型 |

原始项目长期保存在 `G:\Codex\football-prediction-2026-10-22`。约11GB原始数据、重复中间产物和大型模型未放入普通Git历史，详见[数据与模型说明](DATA_AND_ARTIFACTS.md)。本仓库不是完整原始数据备份。

## 已实现的主要方法

- 随时间衰减的球队进攻、防守和主场效应，以进球或xG估计比赛强度。
- 泊松、低比分相关及其他比分分布研究，输出完整主胜／平局／客胜、总球和比分概率。
- 按时间前推的分类、跨赛季状态、射正、xG及赛程等研究组件，保留各项结果与限制。
- 有来源和观察时间的人员新闻、较早同队参与记录，以及对手条件下的历史表现解释。
- 独立LLM、模型辅助LLM及市场线路分别保存；融合组件记录其权重来源，助手综合判断另列。

组件已经实现，不代表每一项都已证明提准，也不代表研究候选均已替换当前主线。部分历史数据的首次发布时间和修订时间不能完整重建；历史研究不能自动称为当时真正发布的预测。

## 本地运行

需要 Python 3.12 或更高版本。Windows PowerShell 从项目根目录执行下列命令；把项目放在G盘可使虚拟环境、安装缓存与输出也留在G盘。

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:TEMP = Join-Path $PWD 'work\tmp'
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $PWD 'work\pip-cache'
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e './model_v2[test]'
```

运行不需要下载真实比赛资料的合成演示：

```powershell
.\model_v2\run.ps1 -Action demo
```

运行测试：

```powershell
.\model_v2\run.ps1 -Action test
```

首次上传前已运行合成演示，以及概率计算、多线路记录、赛前记录和日期模型四组专项测试，共147项通过；这不是全测试集执行记录，也不表示模型预测准确率。

演示里的球队、系数及情景权重均为虚构，只演示程序行为。真实训练和研究脚本需要对应数据；部分脚本仍引用原G盘绝对路径，应先阅读脚本中的来源与路径要求。LLM预测是会话中实际保存的输出，仓库没有附带LLM权重或自动调用凭据。

## 已保存的示例

[2026-10-29三场协作说明](outputs/英超人员协作前瞻_20261029/对手条件接入与本轮判断.md)及[前瞻记录](outputs/英超人员协作前瞻_20261029/条件表现协作前瞻记录.json)保留统计、独立LLM和模型辅助LLM的完整三项概率。

这些文件是历史档案。文中的“当前”“未来”和待结算状态均对应文件原生成时间，不能当作阅读当天的最新赛程、预测或战绩。2026-10-29曾开始扩展其余七场，期间工作被中断；本次上传没有将其补称为已完成预测。

## 研究与来源说明

- [详细研究记录](model_v2/README.md)
- [组件整合与调用](model_v2/INTEGRATION.md)
- [第三方代码、项目借鉴与许可证](model_v2/THIRD_PARTY.md)
- [数据和大型产物的本地保存范围](DATA_AND_ARTIFACTS.md)

早期文档保留了原本的G盘链接，可在原机器使用；GitHub上请使用本页的相对链接导航。本项目未新增统一开源许可，第三方代码仍按各自许可使用。概率是研究估计，不承诺固定准确率或投注收益。
