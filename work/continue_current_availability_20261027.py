"""Connect current source availability to the existing forecast/evidence routes.

One facts-only LLM batch is supplied externally. No player coefficients,
training, probability pooling or target-result retrieval are introduced here.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import argparse
import json

from football_model.current_availability import compare_fpl_snapshots
from football_model.domain import Fixture
from football_model.multiline import build_packet, multi_record, score_multi
from football_model.evidence_reconciliation import reconcile_evidence
from refresh_current_all_competition_20261026 import ROOT, read, write

DATA = ROOT/'data/current_availability/official_20261027'
OUTPUT = ROOT/'outputs/当前可用性接入与判断_20261027'
PREVIOUS = ROOT/'data/current_all_competition/planned_20261027/facts_only_llm_packet.json'
PREVIOUS_FORECAST = ROOT/'outputs/计划赛程与独立LLM_20261027/标准前瞻记录.json'
EARLIER = ROOT/'data/current_rich_inputs/run_20261026T025507813598Z'
BJ = timezone(timedelta(hours=8))
DISPLAY = {'Bournemouth': 'AFC Bournemouth', 'Brighton': 'Brighton & Hove Albion',
    'Leeds': 'Leeds United', 'Man City': 'Manchester City', 'Man Utd': 'Manchester United',
    'Newcastle': 'Newcastle United', "Nott'm Forest": 'Nottingham Forest', 'Spurs': 'Tottenham Hotspur'}


def build():
    bootstrap = read(DATA/'raw/fpl_bootstrap.json')
    receipt = read(DATA/'receipts/fpl_bootstrap.json')
    earlier = read(EARLIER/'official-fpl/bootstrap.json')
    comparison = compare_fpl_snapshots(bootstrap, earlier,
        current_observed_at=receipt['received_at'],
        earlier_observed_at=read(EARLIER/'coverage_report.json')['completed_at'])
    comparison['earlier_time_kind'] = 'retrieval run completion upper bound, not exact individual response time'
    comparison['source_url'] = receipt['source_url']
    comparison['earlier_path'] = str(EARLIER/'official-fpl/bootstrap.json')
    write(DATA/'fpl_snapshot_comparison.json', comparison)
    unpaired_keys = {(r['player']['team_code'], r['player']['code'])
                     for r in comparison['unpaired_current_players']}
    official = [read(DATA/f'{part}10.json') for part in ('first', 'second')]
    news = {r['team']: r for doc in official for r in doc['teams']}
    if len(news) != 20 or sum(len(doc['teams']) for doc in official) != 20:
        raise ValueError('Official research must account for the twenty target clubs once')
    by_name = {DISPLAY.get(r['team']['name'], r['team']['name']): r for r in comparison['teams']}
    by_id = {r['id']: r for r in read(DATA/'raw/fpl_fixtures.json')}
    original = read(PREVIOUS)
    packet = deepcopy(original)
    now = datetime.now(timezone.utc)
    packet.update(assembled_at=now.isoformat(), prior_facts_assembled_at=original['assembled_at'],
        scope='Facts-only next10 packet including current official source availability; no outcome forecasts.')
    packet['sources']['current_fpl_availability'] = {'url': receipt['source_url'],
        'observed_at': receipt['received_at'], 'raw_path': str(DATA/'raw/fpl_bootstrap.json'),
        'earlier_snapshot_path': str(EARLIER/'official-fpl/bootstrap.json'),
        'earlier_observation_upper_bound': comparison['earlier_observed_at'],
        'paired_players': comparison['paired_player_count'],
        'status_changes': len(comparison['status_changes']), 'news_changes': len(comparison['news_changes'])}
    packet['sources']['current_official_target_confirmation'] = {
        'url': 'https://fantasy.premierleague.com/api/fixtures/',
        'observed_at': read(DATA/'receipts/fpl_fixtures.json')['received_at']}
    packet['unknowns'] = [s for s in packet['unknowns'] if 'Verified current injuries/player availability' not in s]
    packet['unknowns'] += ['Official reports mostly concern earlier matches or the cup; next league-round availability is not generally confirmed.',
        'Injury severity, return dates and competition-specific suspension service may remain unresolved.',
        'No player ability or replacement-strength coefficient is supplied; absence and recovery effects are not identified.',
        'Current league statistics end on Oct25 and can already reflect some reported absences; exact per-player overlap was not joined.']
    packet['interpretation'] += ['A report describing a new injury at an earlier announcement is not a newly occurring injury at this observation.',
        'Source status/news are unchanged from the earlier Oct26 snapshot; the new information is its connection to this LLM route.',
        'Status u is other unavailability/registration data, not an injury; a does not guarantee a start or even an appearance.',
        'FPL raw chance percentages are source labels, not calibrated appearance probabilities or match-win probabilities.',
        'Expected-back or suspended-until strings do not establish next-match eligibility, especially with intervening cup fixtures.',
        'Zero season minutes do not prove zero player ability; cumulative minutes are not future minutes or causal contribution.',
        'Recovery training, selection on an earlier bench and confirmed next-league availability are different claims.',
        'Do not count the same player-news assertion from FPL and a club report as independent evidence.']
    excluded = []
    for row in packet['records']:
        f = Fixture(**(row['fixture'] | {'kickoff': datetime.fromisoformat(row['fixture']['kickoff'])}))
        target = by_id[row['official_confirmation']['source_fixture_id']]
        if target.get('started') is not False or datetime.fromisoformat(target['kickoff_time'].replace('Z', '+00:00')) != f.kickoff or f.kickoff <= now:
            raise ValueError('Current official target is not future and unstarted at its archived kickoff')
        row['availability'] = {}
        evidence = []
        for side, team in (('home', f.home), ('away', f.away)):
            own = by_name[team]
            if target['team_h' if side == 'home' else 'team_a'] != own['team']['id']:
                raise ValueError('Current official home/away identity differs from target')
            found = deepcopy(news[team])
            claims = []
            for i, claim in enumerate(found['claims']):
                published = claim.get('published_at')
                day = claim.get('published_date_only')
                if (datetime.fromisoformat(claim['observed_at'].replace('Z', '+00:00')) > now
                    or (published is not None and datetime.fromisoformat(published.replace('Z', '+00:00')) > now)
                    or (day is not None and datetime.fromisoformat(day).date() > now.date())):
                    excluded.append({'team': team, 'claim': claim, 'reason': 'future/unresolved availability time'})
                    continue
                claim['evidence_id'] = f'club:{own["team"]["code"]}:{i}'
                claims.append(claim)
                evidence.append({'id': claim['evidence_id'], 'url': claim['source_url'],
                    'claim': (claim.get('player') or '球队说明')+'：'+claim['claim_summary_zh']+'；适用范围：'+claim['target_scope']+'；限制：'+claim['uncertainty'],
                    'claim_type': 'fact', 'team': team, 'category': 'availability',
                    'status': claim['availability_state'], 'observed_at': claim['observed_at'],
                    'published_at': published, 'source_title': claim['source_title'],
                    'source_kind': claim['source_type'],
                    'source_time_label': day or 'Only retrieval time verified; exact publication unknown.'})
            found['claims'] = claims
            flags = deepcopy(own['flagged_players'])
            for p in flags:
                p['status_and_news_already_in_earlier_snapshot'] = None if (p['team_code'], p['code']) in unpaired_keys else not any(
                    x['player_code'] == p['code'] and x['team_code'] == p['team_code']
                    for x in comparison['status_changes']+comparison['news_changes'])
                p['season_zero_minutes_source_counter'] = p['minutes'] == 0
            row['availability'][side] = {'team': team, 'fpl_observed_at': receipt['received_at'],
                'fpl_team': own['team'], 'source_status_counts': own['current_status_counts'],
                'flagged_players': flags, 'unknown_status_players': own['unknown_status_players'],
                'other_unavailable_count': len(own['unavailable_other_players']),
                'other_unavailable_interpretation': 'Status u is not classified as injury; registration/transfer absence is not a strength penalty.',
                'official_news': found}
            names = '；'.join(f'{p["first_name"]} {p["second_name"]}（来源状态{p["status"]}，本季累计{p["minutes"]}分钟）' for p in flags)
            evidence.append({'id': f'fpl:{own["team"]["code"]}', 'url': receipt['source_url'],
                'claim': '当前FPL来源i/d/s球员：'+(names or '该名单无i/d/s记录')+'。这些是来源标签和累计出场，不是目标缺阵名单或实力效应。',
                'claim_type': 'fact', 'team': team, 'category': 'availability', 'status': 'source_snapshot',
                'observed_at': receipt['received_at'], 'published_at': None,
                'source_title': 'Official FPL bootstrap-static source snapshot', 'source_kind': 'official_fpl'})
        row['news_packet'] = build_packet(f, now.isoformat(), evidence, packet['unknowns'])
    packet['official_research_summary'] = {'teams': len(news),
        'teams_with_official_claims': sum(bool(t['claims']) for t in news.values()),
        'teams_without_extractable_official_claims': [t['team'] for t in news.values() if not t['claims']],
        'source_urls': sorted({c['source_url'] for t in news.values() for c in t['claims']}),
        'claims_before_time_filter': sum(len(t['claims']) for t in news.values()), 'excluded': excluded}
    write(DATA/'facts_only_llm_packet.json', packet)
    print(json.dumps({'packet': str(DATA/'facts_only_llm_packet.json'),
        'targets': len(packet['records']), 'flagged_i_d_s': sum(len(r['flagged_players']) for r in comparison['teams']),
        'status_changes': len(comparison['status_changes']), 'news_changes': len(comparison['news_changes']),
        'official_claims': packet['official_research_summary']['claims_before_time_filter'],
        'official_team_coverage': packet['official_research_summary']['teams_with_official_claims']}))


def finalize(freshness_path=None):
    packet, lm = read(DATA/'facts_only_llm_packet.json'), read(DATA/'llm_response.json')
    responses = {r['fixture_id']: r for r in lm['records']}
    previous = {r['fixture']['match_id']: r for r in read(PREVIOUS_FORECAST)['records']}
    if len(responses) != 10 or set(responses) != {r['fixture']['match_id'] for r in packet['records']}:
        raise ValueError('Exactly one response per target is required')
    if any(lm['exposure'].get(k) is not False for k in ('statistical_probabilities', 'market_probabilities', 'other_llm_probabilities', 'target_results')):
        raise ValueError('Response is not declared independent')
    at = datetime.fromisoformat(lm['generated_at'])
    if at.tzinfo is None or at < datetime.fromisoformat(packet['assembled_at']):
        raise ValueError('Response must follow its factual inputs')
    records, rows, reviews = [], [], []
    assembled = datetime.now(timezone.utc).isoformat()
    for r in packet['records']:
        f = Fixture(**(r['fixture'] | {'kickoff': datetime.fromisoformat(r['fixture']['kickoff'])}))
        answer, old = responses[f.match_id], previous[f.match_id]
        if (answer['home'], answer['away'], answer['kickoff_utc']) != (f.home, f.away, r['fixture']['kickoff']):
            raise ValueError('Response fixture identity or kickoff differs')
        p = answer.get('probabilities')
        tracks = {'statistical': old['tracks']['statistical']}
        missing = {'market': 'No new matching market observation.', 'llm_model_informed': 'Only qualitative evidence review, no informed probability.',
            'joint_pool': 'Original probabilities retained separately; no numeric pooling.'}
        if p is not None:
            tracks['llm_independent'] = {'fixture_id': f.match_id,
                'id': f'llm_availability_20261027:{f.match_id}:{lm["generated_at"]}',
                'model': f'independent subjective LLM; model_id={lm["model_id"]}; official availability facts',
                'generated_at': lm['generated_at'], 'probabilities': p,
                'exposed_to_statistical': False, 'exposed_to_market': False}
        else:
            missing['llm_independent'] = answer['abstention_reason']
        record = multi_record(f, assembled, tracks=tracks, missing_reasons=missing)
        record.update(statistical_source=str(PREVIOUS_FORECAST), statistical_reused_without_change=True,
            llm_facts_source=str(DATA/'facts_only_llm_packet.json'))
        records.append(record)
        rows.append({'fixture': r['fixture'], 'home_chinese': r['home_chinese'], 'away_chinese': r['away_chinese'],
            'statistical_probabilities': old['tracks']['statistical']['probabilities'],
            'previous_llm_probabilities': old['tracks']['llm_independent']['probabilities'],
            'new_llm_probabilities': p, 'judgment': answer, 'availability': r['availability']})
    # The explanation layer sees both probability routes and declares that exposure.
    review_at = datetime.now(timezone.utc).isoformat()
    for record, row in zip(records, packet['records']):
        ids = [e['id'] for e in row['news_packet']['evidence']]
        review = {'fixture_id': record['fixture']['match_id'], 'generated_at': review_at,
            'model': 'assistant model-informed qualitative synthesis; no fitted personnel effects',
            'exposed_to_statistical': True, 'exposed_to_market': False,
            'issues': [{'id': 'overlap-and-source-duplication', 'kind': 'shared_information', 'evidence_ids': ids,
                'summary': 'FPL与官网可能重复同一伤情；截至10月25日球队表现可能已经反映部分缺阵，缺少逐人逐场重叠与替代能力识别，不能叠加固定扣分。',
                'next_question': '确认缺阵影响覆盖了哪些既往比赛、近期角色及替代者信息，保留无法识别的部分。'},
                {'id': 'target-scope-and-unquantified-effect', 'kind': 'unquantified_effect', 'evidence_ids': ids,
                'summary': '此前比赛缺阵、恢复训练、进入替补和下一轮可用性不是同一事实；杯赛后的伤情与停赛执行仍可能改变。统计概率未改，LLM变化没有对应的已训练球员效应。',
                'next_question': '优先等待能明确改变可用性的官方诊断或赛事范围公告；不靠猜测补首发、轮换概率或胜率。'}]}
        reviews.append(reconcile_evidence(record, row['news_packet'], review, as_of=review_at))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write(OUTPUT/'独立LLM原始判断.json', lm)
    write(OUTPUT/'标准前瞻记录.json', {'pilot_id': 'current-availability-independent-llm-20261027',
        'assembled_at': assembled, 'records': records, 'results': {}, 'new_llm_batches': 1,
        'new_statistical_forecasts': 0, 'fits': 0, 'parameter_searches': 0})
    tops = lambda p: {i for i, v in enumerate(p) if v == max(p)}
    available = [r for r in rows if r['new_llm_probabilities'] is not None]
    summary = {'llm_covered': len(available),
        'llm_top_set_changed': sum(tops(r['new_llm_probabilities']) != tops(r['previous_llm_probabilities']) for r in available),
        'stat_and_llm_top_set_equal': sum(tops(r['new_llm_probabilities']) == tops(r['statistical_probabilities']) for r in available),
        'max_single_llm_change_pp': max((abs(100*(a-b)) for r in available for a,b in zip(r['new_llm_probabilities'],r['previous_llm_probabilities'])), default=None),
        'interpretation': 'Changed information plus one fresh subjective generation; not a causal personnel effect or demonstrated accuracy gain.'}
    freshness = None if freshness_path is None else read(freshness_path)
    write(OUTPUT/'逐场对照与可用性.json', {'assembled_at': assembled, 'summary': summary, 'records': rows,
        'later_freshness_observation': freshness, 'later_observation_is_llm_input': False})
    write(OUTPUT/'逐场证据收束.json', {'analyzed_at': review_at, 'records': reviews,
        'new_combined_probabilities': None, 'model_informed_explanation_only': True})
    write(OUTPUT/'初始评分.json', score_multi(records, {}))
    from continue_prospective_evaluation_20261027 import run
    run(ROOT, OUTPUT/'分组评价', additional_cohorts={
        'next_allcomp_repaired10': str(ROOT/'outputs/全赛事缺口修复与预测_20261027/标准前瞻记录.json'),
        'next_planned_cup_llm10': str(PREVIOUS_FORECAST),
        'next_availability_llm10': str(OUTPUT/'标准前瞻记录.json')})
    comparison = read(DATA/'fpl_snapshot_comparison.json')
    lines = ['# 当前可用性接入与独立判断', '',
        '本轮已接通20队官方FPL名单、官网／英超可用性报道、原球队表现及已公布杯赛安排。无需等待目标首发；不加入神经网络、不学习球员能力或新融合权重。原统计路线逐字段复用，全部新增文件在G盘。', '',
        f'FPL于北京时间{datetime.fromisoformat(comparison["current_observed_at"]).astimezone(BJ):%Y-%m-%d %H:%M:%S}获取；{comparison["paired_player_count"]}人可跨快照配对，较10月26日早间旧快照，来源状态变化{len(comparison["status_changes"])}、新闻文字变化{len(comparison["news_changes"])}。本轮是把已取得／已发布资料接入判断，不是宣布今晚出现这些新伤。旧时刻为原取数运行结束上界，未伪造单个响应时间。', '',
        f'官网研究覆盖20队，{packet["official_research_summary"]["teams_with_official_claims"]}队有可提取正文／官方摘要；其余为'+ '、'.join(packet['official_research_summary']['teams_without_extractable_official_claims'])+'。未读到正文的状态保留未知，不视作全员健康。具体网页、日期、适用比赛及限制均保留。', '',
        '## 可用性事实与限制', '', '| 比赛 | 主队FPL i／d／s | 客队FPL i／d／s | 官网取得情况 |', '|---|---:|---:|---|']
    for r in rows:
        counts = lambda side: '／'.join(str(r['availability'][side]['source_status_counts'].get(s,0)) for s in ('i','d','s'))
        coverage = lambda side: '有资料' if r['availability'][side]['official_news']['claims'] else '正文未知'
        lines.append(f'| {r["home_chinese"]}—{r["away_chinese"]} | {counts("home")} | {counts("away")} | {coverage("home")}／{coverage("away")} |')
    if freshness is not None:
        fresh_at = datetime.fromisoformat(freshness['observed_at']).astimezone(BJ)
        lines[2:2] = [f'后续资料核对：北京时间{fresh_at:%Y-%m-%d %H:%M:%S}再次取得FPL和赛程快照，球员状态变化{len(freshness["status_changes"])}、新闻文字变化{len(freshness["news_changes"])}，10场目标对阵／时间与未开赛状态均一致。此为预测生成后的来源观察，不回填为凌晨LLM输入，也不刷新预测生成时间；没有再次调用LLM。', '']
    lines += ['', 'i／d／s是来源的伤病／存疑／停赛标签计数，不是目标比赛已确认缺席人数。u不归伤病；a不保证出场。FPL百分数、预计回归日、累计分钟和首发次数均不自动转换成目标出场概率或实力加减。停赛需保留赛事范围、杯赛执行及剩余场次的未知。', '',
        '## 各路线三项概率', '', f'新独立LLM于北京时间{at.astimezone(BJ):%Y-%m-%d %H:%M:%S}生成，仅读事实包，未提供统计／市场／旧LLM概率。概率均为未经校准的主观分配；“独立”是输入暴露意义，共用事实仍可能导致相关错误。', '',
        '| 北京时间 | 比赛 | 统计主／平／客% | 上版LLM主／平／客% | 本版LLM主／平／客% |', '|---|---|---:|---:|---:|']
    pct = lambda p: '／'.join(f'{100*v:.1f}' for v in p) if p is not None else '弃权'
    for r in rows:
        kickoff = datetime.fromisoformat(r['fixture']['kickoff']).astimezone(BJ)
        lines.append(f'| {kickoff:%m-%d %H:%M} | {r["home_chinese"]}—{r["away_chinese"]} | {pct(r["statistical_probabilities"])} | {pct(r["previous_llm_probabilities"])} | {pct(r["new_llm_probabilities"])} |')
    lines += ['', f'本版LLM首选集合变化{summary["llm_top_set_changed"]}场，与统计首选集合相同{summary["stat_and_llm_top_set_equal"]}场；最高单项变化{summary["max_single_llm_change_pp"]:.1f}个百分点。一次新生成和信息增补同时存在，差值不等于球员因果效应，也不证明准确率提高。未把两线简单合并。', '',
        '## 逐场判断、反证及官网依据', '']
    for r in rows:
        answer = r['judgment']
        lines += [f'### {r["home_chinese"]}—{r["away_chinese"]}', '',
            '判断：'+'；'.join(answer.get('key_facts', [])), '',
            '反证：'+'；'.join(answer.get('counterarguments', [])), '',
            '可用性解释：'+answer.get('availability_interpretation','未提供'), '',
            '不确定性：'+answer.get('uncertainty','未提供'), '']
        for side in ('home','away'):
            news = r['availability'][side]['official_news']
            for claim in news['claims']:
                lines.append(f'- [{claim.get("player") or "球队说明"}：{claim["claim_summary_zh"]}]({claim["source_url"]}) 适用范围：{claim["target_scope"]}；{claim["uncertainty"]}')
        lines.append('')
    lines += ['## 本轮完成与停止点', '',
        '已复用既有多线评分和证据解释层：数值路线单独保留，额外输出有来源的模型知情质疑，不生成联合胜率。前瞻仍为20场不同英超比赛，其中下一轮10场尚无归档结果；版本数量不增加样本。本轮不查询目标赛果、不调参，不重新生成多批LLM择优。', '',
        '球员身份按FPL球队代码＋球员代码配对，转会／新增／缺失单列；4项边界测试通过。新旧状态完全相同也不删除资料：它说明信息接入缺口与伤情变化是两回事。', '',
        '下一步优先取得能明确改变目标可用性的官方新公告，或杯赛后真实新增信息；长缺球员与现有表现的重复计算、替代能力、来源停赛范围仍需保留未知。在这些未识别前，不手工学习“缺一个人扣几点”，也不凭一次命中或失误改变模型结构。', '',
        '[事实包](G:/Codex/football-prediction-2026-10-22/data/current_availability/official_20261027/facts_only_llm_packet.json)；[逐场对照与可用性](G:/Codex/football-prediction-2026-10-22/outputs/当前可用性接入与判断_20261027/逐场对照与可用性.json)；[证据收束](G:/Codex/football-prediction-2026-10-22/outputs/当前可用性接入与判断_20261027/逐场证据收束.json)。']
    (OUTPUT/'可用性接入与判断.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(OUTPUT),'summary':summary},ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['build','finalize'])
    parser.add_argument('--freshness-path', type=Path, help='Later source observation only; never an input to the archived LLM forecast')
    args = parser.parse_args()
    build() if args.action == 'build' else finalize(args.freshness_path)
