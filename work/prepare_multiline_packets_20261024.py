"""Prepare two separately routed actual inputs; no LLM call or fitting here."""
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

from football_model.domain import Fixture
from football_model.multiline import build_packet

ROOT = REPOSITORY_ROOT
CONTEXT = ROOT / 'data/prospective_context/20261023T172314505200Z'
PROJECTION = ROOT / 'outputs/当前球队研究投影_20261023T172816738118Z/球队投影与来源.json'
DEST = ROOT / 'data/prematch-intelligence/20261024_multiline'
DEST.mkdir(parents=True, exist_ok=True)
case = json.loads((ROOT / 'data/prematch-intelligence/20261024_可行性样本.json').read_text(encoding='utf-8'))['cases'][0]
projection = json.loads(PROJECTION.read_text(encoding='utf-8'))
official = json.loads((CONTEXT / 'official_context.json').read_text(encoding='utf-8'))
current = json.loads((CONTEXT / 'descriptive_context.json').read_text(encoding='utf-8'))
cutoff = datetime.now(timezone.utc)
fixture = Fixture(**(case['fixture'] | {'kickoff': datetime.fromisoformat(case['fixture']['kickoff'])}))
facts = [item for item in case['evidence'] if item['id'] != 'schedule-midweek']
midweek = official['bournemouth_midweek_context']
facts.append({'id': 'updated-bournemouth-midweek', 'url': midweek['source_url'],
              'published_at': None, 'observed_at': midweek['observed_at'],
              'claim': midweek['claim'] + ' ' + midweek['interpretation_limit'],
              'claim_type': 'fact', 'team': fixture.away, 'category': 'midweek_context',
              'status': 'official_search_cache_assertion',
              'retrieval_note': '官方域名搜索缓存；直接正文未核验；source date仅URL日历日期，不构造精确发布时间。'})
facts.append({'id': 'updated-united-midweek', 'url': case['evidence'][0]['url'],
              'published_at': None, 'observed_at': case['evidence'][0]['observed_at'],
              'claim': '曼联官网较早预告记载周三欧冠客场1-4负Como；恢复与轮换效应未知，不能由单场失利推定本场能力。',
              'claim_type': 'fact', 'team': fixture.home, 'category': 'midweek_context',
              'status': 'earlier_official_search_cache_assertion',
              'retrieval_note': '同一较早赛前预告的缓存陈述，完整最新伤停仍未知。'})
unknowns = ['完整最新伤停、正式首发、替补、预计分钟和轮换规模',
            '球员缺阵到球队进球率的已验证定量映射',
            '赛事优先级、疲劳/恢复状态及战术变化；未默认相同重视或零影响',
            '当前英超第6-7轮其他16场赛果仍缺，部分对手近期强度资料可能滞后',
            '官方缓存与原页更新时间差异；情报精确首次发布时间未知',
            '跨联赛欧战比分不直接混入英超能力统计；本次没有xG']
context = []
for item in projection['descriptive_team_results']:
    context.append({'id': f"descriptive-{item['team']}-{item['season']}",
                    'observed_at': cutoff.isoformat(),
                    'source': str(PROJECTION) + ' (descriptive results only; no fitted probabilities)',
                    'text': json.dumps(item, ensure_ascii=False)})
for item in current['teams']:
    context.append({'id': f"current-splits-{item['team']}", 'observed_at': cutoff.isoformat(),
                    'source': str(CONTEXT / 'descriptive_context.json'),
                    'text': json.dumps({key: value for key, value in item.items() if key != 'results'}, ensure_ascii=False)})
packets = {
    'independent': build_packet(fixture, cutoff, facts, unknowns, descriptive_context=context),
    'model_informed': build_packet(fixture, cutoff, facts, unknowns, descriptive_context=context,
                                   statistical=projection['track'], mode='model_informed'),
}
for name, packet in packets.items():
    # Two packets differ only by their declared information policy and stat reference.
    packet['run_instructions'] = ('Use only this packet. Do not browse or read other files. '
        'Return one JSON with generated_at_utc, probabilities [home,draw,away] or null, decision, '
        'reasoning_summary, key_scenarios, unknowns, exposed_to_statistical, exposed_to_market, '
        'calibration_status, model_identifier (unknown if runtime exact ID unavailable). '
        'Do not manufacture a probability solely to avoid abstention, and do not promise profitability. '
        'No other LLM responses are provided. Existing descriptive results may support a subjective '
        'research forecast, not a validated probability. Clock may be used to log generation time.')
    (DEST / f'{name}_packet.json').write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding='utf-8')
(DEST / 'preparation.json').write_text(json.dumps({'prepared_at': cutoff.isoformat(),
    'projection_path': str(PROJECTION), 'context_directory': str(CONTEXT),
    'fixture': case['fixture'], 'statistical_track': projection['track'], 'evidence': facts,
    'unknowns': unknowns, 'descriptive_context': context,
    'pool_policy': {'statistical': .5, 'llm': .5},
    'pool_policy_note': 'Explicit equal-weight arithmetic research candidate declared before either new LLM response. Not learned, optimal, or a default production fusion.'},
    ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'packet_directory': str(DEST), 'prepared_at': cutoff.isoformat(),
                  'packet_evidence_rows': len(facts), 'context_rows': len(context), 'fit_count': 0}))
