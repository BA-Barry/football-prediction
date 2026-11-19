# 预测记录格式

核心记录由`football_model.multiline.multi_record`生成。以下是字段语义的简化说明，实际约束以代码和测试为准。

```json
{
  "fixture": {
    "match_id": "stable competition/date/teams identity",
    "home": "home team",
    "away": "away team",
    "competition": "competition id",
    "kickoff": "timezone-aware ISO-8601"
  },
  "as_of": "prediction cutoff",
  "tracks": {
    "statistical": {"probabilities": [0.45, 0.28, 0.27]},
    "llm_independent": null,
    "market": null,
    "llm_model_informed": null,
    "joint_pool": null
  },
  "missing_reasons": {
    "llm_independent": "not supplied"
  }
}
```

概率顺序始终是主胜、平局、客胜，三项均为有限非负数且和为1。比赛结算采用常规时间加补时，不含加时和点球大战。每条线路包含独立ID、fixture ID、生成或观察时间和模型／来源说明。

LLM线路另外保存是否见过统计概率和市场。`joint_pool`保存两个原组件、权重及权重来源；若声称由历史样本学习，还需保存更早训练比赛ID、结果最大可知时间和拟合截止。

评分文件保留未结算fixture，不用0分代替未知赛果。赛果记录需要主客进球、完赛时间、赛果可知时间及相同结算口径。
