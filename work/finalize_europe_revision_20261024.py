"""Keep raw LLM answers, adapt explicit probability order, assemble a new version."""
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path

from football_model.multiline import TRACKS, score_multi
from football_model.prospective import _probabilities, _time
from football_model.prospective_revision import run

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
PACKETS = ROOT / 'data/prematch-intelligence/20261024_batch/europe_context_20261024T154313760557Z'
BASE = ROOT / 'data/prematch-intelligence/20261024_batch/run_20261024T150900467071Z/batch_case.json'
OLD = ROOT / 'outputs/五线赛果评分_20261024T150438997728Z/原赛前记录及新增赛果.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    adapted_paths, adaptations = {}, []
    for mode in ('independent', 'model_informed'):
        path = PACKETS / f'{mode}_response.json'
        response = read(path)
        packets = {p['fixture']['match_id']: p for p in read(PACKETS / f'{mode}_packets.json')['packets']}
        adapted = deepcopy(response)
        for row in adapted['forecasts']:
            packet = packets[row['fixture_id']]
            ids = {v['id'] for v in packet['evidence'] + packet['descriptive_context']}
            if mode == 'model_informed':
                ids.add(packet['statistical']['id'])
            if not row.get('cited_evidence_ids') or set(row['cited_evidence_ids']) - ids:
                raise ValueError('Invalid evidence citation in its own packet')
            values = row['probabilities']
            if isinstance(values, dict):
                if set(values) != {'home', 'draw', 'away'}:
                    raise ValueError('Mapping must explicitly name home/draw/away')
                row['probabilities'] = [values[k] for k in ('home', 'draw', 'away')]
                adaptations.append({'mode': mode, 'fixture_id': row['fixture_id'],
                    'kind': 'Explicit named keys to [home,draw,away] array; identical numbers and original generation time.'})
            _probabilities(row['probabilities'])
        target = PACKETS / f'{mode}_response_normalized.json'
        adapted['format_adaptation'] = {'raw_response': str(path), 'adapted_at': datetime.now(timezone.utc).isoformat(),
                                      'probabilities_changed': False, 'generation_times_changed': False}
        write(target, adapted)
        adapted_paths[mode] = target
    write(PACKETS / '原回答格式衔接说明.json', {'adaptations': adaptations,
          'raw_answers_untouched': True, 'no_llm_resampling': True, 'no_probability_normalization': True})
    result = run(ROOT, BASE, PACKETS, adapted_paths['independent'], adapted_paths['model_informed'])
    revised = read(Path(result['case']))
    base, old = read(BASE), read(OLD)
    prior = {r['fixture']['match_id']: r for r in base['records']}
    updates = {r['fixture']['match_id']: r for r in revised['records']}
    changes = []
    for fid, record in updates.items():
        before = prior[fid]
        if any(record['tracks'][arm] != before['tracks'][arm] for arm in ('statistical', 'market')):
            raise ValueError('Original statistical/market line changed')
        changes.append({'fixture_id': fid,
            'per_line_max_absolute_probability_change': {
                arm: max(abs(a - b) for a, b in zip(record['tracks'][arm]['probabilities'], before['tracks'][arm]['probabilities']))
                for arm in ('llm_independent', 'llm_model_informed', 'joint_pool')},
            'original': {arm: before['tracks'][arm]['probabilities'] for arm in TRACKS},
            'revised': {arm: record['tracks'][arm]['probabilities'] for arm in TRACKS}})
    latest_records = old['records'] + [updates.get(r['fixture']['match_id'], r) for r in base['records']]
    if len({r['fixture']['match_id'] for r in latest_records}) != 20:
        raise ValueError('Registry counts unique fixtures, not revisions')
    scores = score_multi(latest_records, old['results'])
    now = datetime.now(timezone.utc)
    target = ROOT / 'outputs' / now.strftime('欧战赛程信息修订_%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True, exist_ok=False)
    registry = {'generated_at': now.isoformat(), 'records': latest_records, 'results': old['results'],
                'source_cases': {'original_scored_ten': str(OLD), 'new_round_original_ten': str(BASE), 'new_round_europe_seven_revision': result['case']},
                'scope': 'Latest information-version bookkeeping only, not model promotion. Each original version retained for paired evaluation.'}
    write(target / '当前20场信息版本总表.json', registry)
    write(target / '信息版本变化与原概率.json', {'generated_at': now.isoformat(), 'changes': changes,
        'new_unique_fixtures': 0, 'llm_response_batches': 2, 'new_fixture_forecasts': 14,
        'new_fits': 0, 'new_parameter_searches': 0, 'statistical_and_market_unchanged': True,
        'causal_limit': 'LLM sampling and context effects are not controlled; cannot attribute probability changes solely to European schedule facts.'})
    write(target / '按当前信息版本的描述性评分.json', scores)
    bj = timezone(timedelta(hours=8))
    report = ['# 补齐欧战赛程后的独立信息版本', '',
              f"本次新增9条球队欧战日程事实，涉及已记录的7场比赛。总计仍20个不同比赛，不是27场；已确认结算{scores['n_settled']}场，待结算{len(scores['pending_fixture_ids'])}场。信息修订生成于{now.astimezone(bj).isoformat()}。", '',
              '7场各由新的独立上下文与知情上下文完成一次预测，共14项。两者均未接触市场或另一回答，知情线只多读取原统计参考。另3场没有新增欧战事实，不重复生成。统计、市场原记录逐项未变，模型拟合0次、调参0次。', '',
              '| 比赛 | 原独立LLM主/平/客% | 新独立LLM主/平/客% | 原组合主/平/客% | 新组合主/平/客% |',
              '|---|---|---|---|---|']
    for record in revised['records']:
        fid = record['fixture']['match_id']
        row = [record['fixture']['home'] + '—' + record['fixture']['away']]
        for source, line in ((prior[fid], 'llm_independent'), (record, 'llm_independent'),
                             (prior[fid], 'joint_pool'), (record, 'joint_pool')):
            row.append('/'.join(f'{p * 100:.1f}' for p in source['tracks'][line]['probabilities']))
        report.append('| ' + ' | '.join(row) + ' |')
    largest = max(v['per_line_max_absolute_probability_change']['llm_independent'] for v in changes)
    report += ['', f"独立LLM某一项概率最大变化{largest * 100:.2f}个百分点。这同时提醒我们：资料增加不等于判断必然变好，新的LLM采样及上下文变化也会影响数字，不能将差额解释成测得的欧战影响。", '',
               '曼城一场独立LLM主胜由55%变61%，尽管新增欧战日程并没有直接证明联赛实力上升；应将它保留为未来可检验的上下文敏感性现象，不能选择看起来更有信心的一版宣布升级。维拉有三天后的巴萨客场，但两条LLM均不能从这一安排确认其联赛轮换。', '',
               '欧洲日程包括维拉11月3日客战巴塞罗那、曼联同日主战罗马，利物浦/曼城/阿森纳11月4日欧冠，伯恩茅斯/水晶宫/桑德兰11月5日欧联及布莱顿同日欧协。日期和计划间隔已逐条核算；实际出场分钟、伤停恢复、旅行、教练取舍和优先级仍未知。', '',
               f"[本版五线及旧新版完整对照]({Path(result['report']).as_posix()})；[新增10场原版报告](G:/Codex/football-prediction-2026-10-22/outputs/前瞻新增执行_20261024T152455896607Z/新增预测与优化说明.md)。原版与本版将来分别评分、按同一比赛对齐，不能把版本拼接扩大样本。", '',
               '[原比分分布复核](G:/Codex/football-prediction-2026-10-22/outputs/原比分分布复核_20261024T154517265623Z/原比分分布复核说明.md)显示：伊普斯维奇1比1的原单格概率11.25%，是原分布最大单格，但原胜平负最高概率方向仍未命中。分类、比分、总球各有目标，不能事后挑好的改成绩。', '',
               '失误研究继续以原样本概率的Log loss、Brier和比分NLL为依据。两个完赛样本不能确认哪个算法更优；历史平局和强热门分组偏差仍作为候选问题，已有校准研究不在同一目标季重复搜索。下一次有正式名单或杯赛实际负荷才值得补新的版本，停止同一信息的反复LLM采样。', '',
               '所有缓存市场仍为第三方署名资料，未确认最新可成交，不能作为即时赔率或收益依据。原始独立回答使用home/draw/away键值形式，衔接时按明确键序转为数组，数值与实际生成时间均未变，原始文件完整保留。', '',
               '相关模块此前123项测试通过，本次通用资料版本模块36项测试通过。原预测未覆盖，文件均在G盘；没有新增训练网络、权重搜索或手写战意系数。', '']
    (target / '欧战赛程修订与后续优化说明.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({'directory': str(target), 'revision': result, 'total_unique': 20,
                      'settled': scores['n_settled'], 'pending': len(scores['pending_fixture_ids']),
                      'independent_max_change': largest}, ensure_ascii=False))


if __name__ == '__main__':
    main()
