"""One readable latest-information digest; does not refresh any predictions."""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

from football_model.prospective import _time
from summarize_new_round_20261024 import CHINESE

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
LATEST = ROOT / 'outputs/欧战赛程信息修订_20261024T154941744958Z/当前20场信息版本总表.json'
OBSERVATION = ROOT / 'data/prematch-intelligence/20261024_batch/result_observation_20261024T155023874777Z/result_observations.json'


def main():
    payload = json.loads(LATEST.read_text(encoding='utf-8'))
    observation = json.loads(OBSERVATION.read_text(encoding='utf-8'))
    added = [r for r in payload['records'] if _time(r['fixture']['kickoff'], 'kickoff').date().isoformat() >= '2026-10-31']
    if len(added) != 10 or len({r['fixture']['match_id'] for r in payload['records']}) != 20:
        raise ValueError('Expected ten added unique fixtures and twenty total')
    now, bj = datetime.now(timezone.utc), timezone(timedelta(hours=8))
    target = ROOT / 'outputs' / now.strftime('当前预测简报_%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True, exist_ok=False)
    report = ['# 当前预测与优化进度', '',
              '已新增10场未开赛英超，累计20场不同比赛。已确认结算2场；18场待结算包括正在进行的比赛。新增10场均尚未开球，最早北京时间10月31日20:30、最晚11月3日04:00。', '',
              f"本简报生成于{now.astimezone(bj).isoformat()}，只整理已存预测，没有生成新的概率。旧比赛状态最后核实时间：{_time(observation['recorded_at'], 'observed').astimezone(bj).isoformat()}；阿森纳与考文垂两场源中尚未完赛，不以进行中比分评分。", '',
              '## 新10场当前信息版本', '',
              '每格为主胜／平局／客胜百分比。7场采用另存的欧战资料版、3场保留原新增版；两版均在开球前生成。知情LLM与统计不是独立两票，50:50组合固定为统计与独立LLM平均，未经训练择优。', '',
              '| 北京时间 | 比赛 | 统计 | 独立LLM | 知情LLM | 50:50组合 |',
              '|---|---|---|---|---|---|']
    for record in sorted(added, key=lambda r: r['fixture']['kickoff']):
        f = record['fixture']
        title = CHINESE.get(f['home'], f['home']) + '—' + CHINESE.get(f['away'], f['away'])
        cells = ['/'.join(f'{p * 100:.1f}' for p in record['tracks'][arm]['probabilities'])
                 for arm in ('statistical', 'llm_independent', 'llm_model_informed', 'joint_pool')]
        report.append('| ' + ' | '.join([_time(f['kickoff'], 'kickoff').astimezone(bj).strftime('%m-%d %H:%M'), title, *cells]) + ' |')
    report += ['', '市场另外保留10场第三方署名Bet365缓存去水对照，直接核实最新可成交价格0场、开盘变化未知。缓存资料不进入上述LLM与组合，也不能当即时价值判断依据。', '',
               '## 失误定位与实际改进', '',
               '- 两场已结算：伊普斯维奇1–1森林、维拉2–1曼城。原胜平负最高概率方向均未中；统计给实际结果的原概率分别24.32%和24.79%，两场不足以确定谁更准。',
               '- 原知情LLM原10条及更新9条全部等于统计，没有数值增量。本轮改为两个单独上下文，并要求列依据、反向情景及维持／调整理由；不强迫改数字，也不因数字不同宣布进步。',
               '- 补进周中杯赛与9队下周欧战安排，7场另存信息版本。赛事安排不等于测得的疲劳、战意或确定轮换；长期伤缺可能已反映在历史结果中。LLM采样也可能影响变化，因此新旧差值不能当成欧战因果效应。',
               '- 原比分分布中伊普斯维奇1–1概率11.25%，是最大单格；这与胜平负最高概率落在另一类并不矛盾。仍记原1X2方向失误，比分／总球分开评价，不事后改评分目标。',
               '- 既有两季3478场概率复核提示平局偏低及强热门组偏保守，属于待验证解释；不从两场失利或事后分组直接调参数。', '',
               '本轮统计使用1591场较早日期比赛拟合1次、参数搜索0次。欧战资料版0次拟合；相关模块123项及版本模块36项测试通过。所有新增项目文件、数据、模型与临时目录均在G盘。', '',
               '## 后续安排', '',
               '先按已存原概率累计比赛结果，分别评价各线Log loss、Brier和统计比分NLL；旧版、新版按同一比赛配对，不能增加独立样本数。出现确认首发、真实杯赛用人或可靠伤情变化时才新增下一版本。', '',
               '同一资料不继续反复抽取LLM答案。只有跨轮持续出现明确概率偏差，或特征与误差能稳定对应时，再讨论一项结构改变；此前已经试过的校准和分布方案不换名字重复调同一目标季。', '',
               '[新增10场原版及来源](G:/Codex/football-prediction-2026-10-22/outputs/前瞻新增执行_20261024T152455896607Z/新增预测与优化说明.md) · [欧战资料新旧对照](G:/Codex/football-prediction-2026-10-22/outputs/欧战赛程信息修订_20261024T154941744958Z/欧战赛程修订与后续优化说明.md) · [两场失误复核](G:/Codex/football-prediction-2026-10-22/outputs/前瞻失误复核_20261024T150602478007Z/前瞻失误复核说明.md) · [历史概率误差定位](G:/Codex/football-prediction-2026-10-22/outputs/已存概率误差定位_20261024T151251984500Z/概率误差定位说明.md)。', '']
    path = target / '当前预测与优化进度.md'
    path.write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({'report': str(path), 'new_unique': 10, 'total_unique': 20, 'settled': 2, 'pending': 18}, ensure_ascii=False))


if __name__ == '__main__':
    main()
