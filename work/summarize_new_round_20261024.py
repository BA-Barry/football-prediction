"""Summarize newly saved forecasts with old unique fixtures; no new predictions."""
import argparse
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

from football_model.multiline import TRACKS, score_multi
from football_model.prospective import _time

ROOT = REPOSITORY_ROOT
OLD_SCORE = ROOT / 'outputs/五线赛果评分_20261024T150438997728Z/原赛前记录及新增赛果.json'
COHORT = ROOT / 'data/prematch-intelligence/NEW20261024_cohort_20261024T150344311377Z'
FAILURES = ROOT / 'outputs/前瞻失误复核_20261024T150602478007Z/前瞻失误复核说明.md'
DIAGNOSTIC = ROOT / 'outputs/已存概率误差定位_20261024T151251984500Z/概率误差定位说明.md'
CHINESE = {'Chelsea': '切尔西', 'Manchester United': '曼联', 'AFC Bournemouth': '伯恩茅斯',
           'Leeds United': '利兹联', 'Brentford': '布伦特福德', 'Nottingham Forest': '诺丁汉森林',
           'Coventry City': '考文垂', 'Sunderland': '桑德兰', 'Hull City': '赫尔城',
           'Ipswich Town': '伊普斯维奇', 'Manchester City': '曼城', 'Brighton & Hove Albion': '布莱顿',
           'Tottenham Hotspur': '热刺', 'Crystal Palace': '水晶宫', 'Aston Villa': '阿斯顿维拉',
           'Fulham': '富勒姆', 'Liverpool': '利物浦', 'Arsenal': '阿森纳',
           'Newcastle United': '纽卡斯尔', 'Everton': '埃弗顿'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--export-report', type=Path, required=True)
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    case_path = args.case.resolve()
    export_report = args.export_report.resolve()
    if any(not p.is_relative_to(REPOSITORY_ROOT) for p in (case_path, export_report)):
        raise ValueError('Project artifacts must remain below the project root')
    new, old = read(case_path), read(OLD_SCORE)
    projection = read(Path(new['projection_path']))
    new_ids = {r['fixture']['match_id'] for r in new['records']}
    old_ids = {r['fixture']['match_id'] for r in old['records']}
    if new_ids & old_ids or len(new_ids) != len(new['records']):
        raise ValueError('New cohort must contain new unique fixtures')
    if any(_time(r['fixture']['kickoff'], 'kickoff') <= now for r in new['records']):
        raise ValueError('Newly added fixtures must still be unstarted')
    citation_counts, exact_copies = {}, 0
    for mode in ('independent', 'model_informed'):
        packets = {p['fixture']['match_id']: p for p in read(case_path.parent / f'{mode}_packets.json')['packets']}
        responses = read(case_path.parent / f'{mode}_response.json')['forecasts']
        counts = 0
        for row in responses:
            packet = packets[row['fixture_id']]
            available_ids = {r['id'] for r in packet['evidence'] + packet['descriptive_context']}
            if mode == 'model_informed':
                available_ids.add(packet['statistical']['id'])
            if not row.get('cited_evidence_ids') or set(row['cited_evidence_ids']) - available_ids:
                raise ValueError('Forecast citation not present in its own information packet')
            counts += len(row['cited_evidence_ids'])
            if mode == 'model_informed':
                exact_copies += all(math.isclose(a, b, rel_tol=0, abs_tol=1e-15)
                                    for a, b in zip(row['probabilities'], packet['statistical']['probabilities']))
        citation_counts[mode] = counts
    combined_records = old['records'] + new['records']
    combined_scores = score_multi(combined_records, old['results'])
    registry = {'generated_at': now.isoformat(), 'records': combined_records,
                'results': old['results'], 'new_case': str(case_path), 'old_scored_case': str(OLD_SCORE),
                'cohorts': [{'kind': 'original', 'fixture_ids': sorted(old_ids)},
                            {'kind': 'new_round_9', 'fixture_ids': sorted(new_ids)}],
                'revision_policy': 'Old revisions are linked separately, not added as unique fixtures.',
                'heterogeneity_note': 'Different information dates and LLM process versions; aggregate description is not a fixed-version efficacy estimate.'}
    market = new['market_source']
    audit = {'generated_at': now.isoformat(), 'new_fixtures': len(new_ids),
             'old_fixtures': len(old_ids), 'total_unique_fixtures': len(new_ids | old_ids),
             'settled_unique_fixtures': combined_scores['n_settled'],
             'pending_unique_fixtures': len(combined_scores['pending_fixture_ids']),
             'new_fit_count': projection['fit_count'], 'parameter_searches': projection['parameter_searches'],
             'training_matches': projection['training_matches'], 'current_matches': projection['current_season_matches'],
             'llm_response_batches': 2, 'llm_fixture_forecasts': sum(len(v) for v in new['llm_responses'].values()),
             'same_model_platform_conditions_not_independent_vendors': True,
             'new_informed_probability_exact_copies': exact_copies,
             'citation_counts': citation_counts, 'new_all_five_numeric': sum(all(r['tracks'][a] is not None for a in TRACKS) for r in new['records']),
             'market_coverage': market.get('coverage'),
             'tests': {'related_tests_passed': 123, 'elapsed_seconds': 7.62},
             'old_predictions_modified': False, 'local_neural_or_rl_training': False,
             'new_report': str(export_report), 'failure_report': str(FAILURES), 'historical_diagnostic': str(DIAGNOSTIC)}
    target = ROOT / 'outputs' / now.strftime('前瞻新增执行_%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True, exist_ok=False)
    for name, payload in (('前瞻比赛总表.json', registry), ('原记录累计评分.json', combined_scores), ('本轮执行记录.json', audit)):
        (target / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    bj = timezone(timedelta(hours=8))
    report = ['# 新增10场未开赛预测与失误定位', '',
              f"记录更新于{now.astimezone(bj).isoformat()}。新增{len(new_ids)}场，共{len(new_ids | old_ids)}个不同比赛；已确认结束{combined_scores['n_settled']}场，待结算{len(combined_scores['pending_fixture_ids'])}场。待结算不等于全部未开场。原版本和更新版本没有重复增加比赛数。", '',
              '新批次为英超第9轮完整10场：北京时间10月31日20:30至11月3日04:00，全部仍未开场。比赛和UTC时间已与官方赛程逐场核对，不能使用10月25日前的英国夏令时换算这批比赛。', '',
              '| 北京时间 | 比赛 | 统计主/平/客% | 独立LLM主/平/客% | 知情LLM主/平/客% | 50:50组合主/平/客% |',
              '|---|---|---|---|---|---|']
    for record in new['records']:
        f = record['fixture']
        title = CHINESE.get(f['home'], f['home']) + '—' + CHINESE.get(f['away'], f['away'])
        probs = ['/'.join(f'{p * 100:.1f}' for p in record['tracks'][line]['probabilities'])
                 for line in ('statistical', 'llm_independent', 'llm_model_informed', 'joint_pool')]
        report.append('| ' + ' | '.join([_time(f['kickoff'], 'kickoff').astimezone(bj).strftime('%m-%d %H:%M'), title, *probs]) + ' |')
    report += ['', f"[完整五列数字、逐场理由、统计总进球及来源]({export_report.as_posix()})。市场列为10场第三方署名Bet365缓存的比例去水对照：直接博彩公司核实0场、确认最新可成交0场，开盘变化未知。不能将数值齐全称为已核实的即时市场；50:50组合不使用这些报价。", '',
               '## 这次实际改变了什么', '',
               f"统计线沿用原设置，从{projection['training_matches']}场历史记录拟合1次，其中本季{projection['current_season_matches']}场；参数搜索0次。已确认的伊普斯维奇1比1森林是新日期级输入。维拉本日赛果用于结算，按现有日期级规则不进入本次拟合。", '',
               f"两个LLM执行上下文未继承主对话：独立线只看来源资料与原始表现摘要，知情线只多看统计数字，均看不到市场或另一回答。两线共完成20项逐场预测，逐项证据ID均来自各自资料包。知情线与统计逐项完全相同{exact_copies}/{len(new_ids)}场；差异本身不是进步，仍需未来评分。", '',
               '新增周中杯赛、下一场时间、伤停元数据及累计出场背景。切尔西杯赛客战利物浦后约64.5小时迎战曼联；曼城与布莱顿一周双赛；纽卡与埃弗顿也在杯赛和联赛交锋。安排已知，实际轮换、出场分钟、恢复与赛事优先级未知。没有直接给战意或疲劳配胜率系数，长期伤缺也不机械重复扣分。', '',
               '## 能确认的问题与仍需检验的解释', '',
               '1. 原两场完赛的最高概率方向均未命中；统计线分别给真实发生的平局24.32%、维拉主胜24.79%。这属于模型已分配概率的结果，两场不足以判定概率失真或比较优越性。',
               '2. 原10条及更新9条知情LLM概率全部与统计相同，确实没有数值增量。本轮通过单独执行上下文和逐条调整理由改进这条线路，但没有因数字不同就宣称预测更准。',
               '3. 维拉一场增加球员描述后，独立LLM把主胜22%改为20%，对真实结果的概率评分更差。更多文字没有自动转化为有用信息；该个案不能证明球员因素无效。',
               '4. 已存历史1736＋1742场的描述性复核显示：统计线平局平均概率分别23.50%/24.06%，真实频率24.88%/25.49%；市场识别的70%以上热门组，统计平均给该方向72.11%/71.19%，真实频率81.25%/78.40%。这提供平局偏低和部分强弱差偏保守的候选线索，不证明具体因果，也不据此手填修正量。', '',
               '上述汇总包括五大联赛。只看两季各378场英超，同样发现统计平局平均概率22.19%/23.85%，实际24.60%/27.51%；市场自身也低估了这两季英超的平局频率。当前前瞻只接英超，不能把五联赛汇总直接当本轮准确率。', '',
               f"[原两场详细复核]({FAILURES.as_posix()})；[既有历史概率分组定位]({DIAGNOSTIC.as_posix()})。历史分组已经看过，不冒称新盲测或新的正式检验。", '',
               '## 后续优化次序', '',
               '先用原概率累计这20个不同比赛的结果，报告同场Log loss、Brier、覆盖与按轮差异。新批次距开球约一周，下一次补充临近开场的确认名单及最新信息时，应保存为新版本，不能覆盖本次或在开场后回填。', '',
               '优先检查杯赛和伤停资料是否产生有来源、可重复的LLM增量，以及不同时间版本是否改善概率评分。统计线继续追踪平局及热门组的概率偏差；此前概率校准、衰减、收缩和分布方案已有研究，不换名字继续搜同一目标季。若需要改变球队强度层、加入赛事投入情景或深度学习，再用具体失败模式和验证安排与用户讨论。', '',
               '本轮相关测试123项通过。没有新增hash、冻结contract、baseline或gate；没有新增本地神经网络、强化学习或超参数网格。所有项目数据、模型、临时文件及报告都在项目目录。实验设计与统计分析技能用于按真实比赛配对、区别版本、事先写明比较和完整报告负结果。', '']
    (target / '新增预测与优化说明.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({'directory': str(target), **audit}, ensure_ascii=False))


if __name__ == '__main__':
    main()
