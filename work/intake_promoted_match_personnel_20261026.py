"""Observe past match personnel with explicit ESPN identities, without fitting."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from math import fsum
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

from probe_espn_personnel_20261026 import OUT as ARCHIVE, read_source
from football_model.historical_relevance import personnel_usage_overlap

ROOT = REPOSITORY_ROOT
OUT = ROOT/'outputs/升班球队人员参考_20261026'
CUTOFF = '2026-10-25'
CLUBS = [('Coventry', '考文垂', '388'), ('Hull', '赫尔城', '306'), ('Ipswich', '伊普斯维奇', '373')]
REGULAR_TYPES = {('eng.2',2025):13531, ('eng.1',2026):14308}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def fetch(request):
    label, url = request
    path = ARCHIVE/(label+'.json')
    receipt_path = ARCHIVE/(label+'_receipt.json')
    if path.exists() and receipt_path.exists():
        receipt = read(receipt_path)
        if receipt.get('status_code') == 200 and receipt.get('url') == url:
            return read(path), receipt
    return read_source(label, url)


def iso(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def schedule(payload, club, year, league):
    events = []
    for event in payload.get('events', []):
        if event.get('season', {}).get('year') != year:
            continue
        # English tier 1 labels the phase with the league name, whereas tier 2
        # uses "Regular Season". Use observed source phase identities instead.
        phase = event.get('seasonType', {})
        if str(phase.get('id')) != '1' or phase.get('type') != REGULAR_TYPES[(league,year)]:
            continue
        if event['date'][:10] >= CUTOFF:
            continue
        comps = event.get('competitions', [])
        if len(comps) != 1:
            raise ValueError('Ambiguous scheduled competition')
        comp = comps[0]
        if not comp.get('status', {}).get('type', {}).get('completed'):
            continue
        sides = {c['homeAway']:str(c['id']) for c in comp['competitors']}
        if set(sides) != {'home', 'away'} or club not in sides.values():
            raise ValueError('Schedule does not identify the requested club')
        if iso(event['date']) != iso(comp['date']):
            raise ValueError('Schedule kickoff mismatch')
        events.append({'event_id':str(event['id']), 'date':event['date'],
            'home_id':sides['home'], 'away_id':sides['away'], 'year':year, 'league':league})
    if len({e['event_id'] for e in events}) != len(events):
        raise ValueError('Repeated schedule event ID')
    return sorted(events, key=lambda e:(iso(e['date']), e['event_id']))


def parse(payload, event):
    header = payload['header']
    comp = header['competitions'][0]
    if (str(header['id']) != event['event_id'] or str(comp['id']) != event['event_id']
            or header['season']['year'] != event['year'] or iso(comp['date']) != iso(event['date'])
            or header['season']['type'] != REGULAR_TYPES[(event['league'],event['year'])]):
        raise ValueError('Summary event, season or kickoff differs from schedule')
    if not comp['status']['type']['completed']:
        raise ValueError('Summary does not mark a completed historical match')
    sides = {c['homeAway']:str(c['id']) for c in comp['competitors']}
    if sides != {'home':event['home_id'], 'away':event['away_id']}:
        raise ValueError('Summary team identities differ from schedule')
    row = {'match_id':'espn:event:'+event['event_id'], 'match_date':event['date'][:10],
        'kickoff':event['date'], 'season':str(event['year'])+'/'+str(event['year']+1)[-2:],
        'source_competition':event['league'], 'home':'espn:team:'+sides['home'],
        'away':'espn:team:'+sides['away'], 'players':{}}
    roster_sides, all_ids = set(), set()
    for roster in payload.get('rosters', []):
        side = roster['homeAway']
        if side in roster_sides or str(roster['team']['id']) != sides.get(side):
            raise ValueError('Roster identity mismatch')
        roster_sides.add(side)
        players = []
        for athlete in roster['roster']:
            pid = 'espn:player:'+str(athlete['athlete']['id'])
            if pid in all_ids:
                raise ValueError('Player repeated across a match roster')
            all_ids.add(pid)
            starter, sub = athlete.get('starter'), athlete.get('subbedIn')
            if not isinstance(starter, bool) or not isinstance(sub, bool):
                raise ValueError('Missing explicit starter or substitution indicator')
            stats = {s['name']:s.get('value') for s in athlete.get('stats', [])}
            appearances = stats.get('appearances')
            if appearances not in (0, 1) or appearances != int(starter or sub):
                raise ValueError('Roster participation disagrees with appearance statistic')
            if stats.get('subIns') != int(sub) or (starter and sub):
                raise ValueError('Roster substitution count differs from explicit flag')
            players.append({'player_id':pid, 'source_name':athlete['athlete'].get('displayName'),
                'starter':starter, 'appeared':bool(appearances), 'subbed_in':sub})
        starters = [p['player_id'] for p in players if p['starter']]
        if len(starters) != 11:
            raise ValueError('Historical lineup does not contain exactly 11 starters')
        row[side+'_starters'] = starters
        row['players'][side] = players
    if roster_sides != {'home', 'away'}:
        raise ValueError('Both historical rosters are required')
    return row


def starts(rows, team):
    counts = Counter()
    for row in rows:
        side = 'home' if row['home'] == team else 'away'
        counts.update(row[side+'_starters'])
    total = sum(counts.values())
    return {p:v/total for p,v in counts.items()} if total else {}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    requests, keys = [], []
    for name, zh, club in CLUBS:
        for year, league in ((2025, 'eng.2'), (2026, 'eng.1')):
            keys.append((name, club, year, league))
            requests.append((f'{name.lower()}_{year}_schedule',
                f'https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/teams/{club}/schedule?season={year}'))
    selections, schedules, errors = {}, {}, []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for key, (payload, receipt) in zip(keys, pool.map(fetch, requests)):
            name, club, year, league = key
            if payload is None:
                errors.append({'stage':'schedule', 'team':name, 'year':year, 'receipt':receipt})
                continue
            events = schedule(payload, club, year, league)
            schedules[(name, year)] = events
            selections[(name, year)] = events[-15:] if year == 2025 else events
    unique = {e['event_id']:e for events in selections.values() for e in events}
    event_requests = [(f"event_{e['event_id']}_summary",
        f"https://site.api.espn.com/apis/site/v2/sports/soccer/{e['league']}/summary?event={e['event_id']}")
        for e in unique.values()]
    observed = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for event, (payload, receipt) in zip(unique.values(), pool.map(fetch, event_requests)):
            try:
                if payload is None:
                    raise ValueError('No successful summary response')
                row = parse(payload, event)
                row.update(source_url=receipt['url'], source_received_at=receipt['received_at'])
                observed[event['event_id']] = row
            except (KeyError, TypeError, ValueError) as error:
                errors.append({'stage':'summary', 'event_id':event['event_id'],
                    'reason':str(error), 'receipt':receipt})
    teams = []
    for name, zh, club in CLUBS:
        team = 'espn:team:'+club
        previous_events = selections.get((name,2025), [])
        current_events = selections.get((name,2026), [])
        previous = [observed[e['event_id']] for e in previous_events if e['event_id'] in observed]
        current = [observed[e['event_id']] for e in current_events if e['event_id'] in observed]
        complete = (len(previous) == 15 and len(previous) == len(previous_events)
                    and len(current) == len(current_events) and len(current) >= 5)
        comparison = personnel_usage_overlap(previous+current, team, CUTOFF)
        old_shares, recent_shares = starts(previous, team), starts(current[-5:], team)
        season_overlap = (fsum(min(v, old_shares.get(p,0)) for p,v in recent_shares.items())
            if complete else None)
        previous_ids = set()
        for row in previous:
            side = 'home' if row['home'] == team else 'away'
            previous_ids.update(p['player_id'] for p in row['players'][side] if p['appeared'])
        recent_known = (fsum(v for p,v in recent_shares.items() if p in previous_ids)
                       if complete else None)
        if not complete:
            comparison['overlap'] = None
            comparison['missing_reason'] = 'selected_match_coverage_incomplete'
        teams.append({'team':name,'team_zh':zh,'source_team_id':team,'coverage_complete':complete,
            'previous_regular_season_schedule_matches':len(schedules.get((name,2025), [])),
            'previous_last15_selected':len(previous_events),'previous_last15_observed':len(previous),
            'current_completed_before_cutoff':len(current_events),'current_observed':len(current),
            'current_recent5_start_shares':recent_shares,'previous_last15_start_shares':old_shares,
            'recent5_vs_previous_season_last15_start_share_overlap':season_overlap,
            'recent5_start_share_from_ids_appearing_in_previous_last15':recent_known,
            'consecutive5_vs_preceding15':comparison,
            'previous_match_ids':[r['match_id'] for r in previous],
            'current_match_ids':[r['match_id'] for r in current]})
    report = {'observed_at':datetime.now(timezone.utc).isoformat(),'match_cutoff_exclusive_utc_date':CUTOFF,
        'teams':teams,'requested_unique_summaries':len(unique),'parsed_unique_summaries':len(observed),
        'errors':errors,'minutes_inferred':False,'target_lineup_used':False,'model_fit_count':0,
        'forecast_probabilities_changed':False,'first_publication_times_known':False,
        'interpretation':'Past starting-appearance distributions, not minute retention or team strength. Prior window is 15 matches, not a complete historical transfer register.',
        'season_aggregate_statistics_used':False,'player_id_cross_provider_join':False}
    (OUT/'逐场同源人员记录.json').write_text(json.dumps(list(observed.values()),ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'逐场同源人员对照.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines = ['# 三支升班球队：逐场人员参考','',
        '使用ESPN逐场已完赛名单和明确球员ID。只取2026-10-25 UTC日期之前的比赛；获取发生在今天，不能冒充原预测已知数据。', '',
        '| 球队 | 上季常规赛目录 | 上季末窗口/本季已观测 | 最近5场对上季末15场首发份额重叠 | 连续最近5场对前15场重叠 |',
        '|---|---:|---:|---:|---:|']
    def pct(value):
        return '未知' if value is None else f'{value:.1%}'
    for r in teams:
        lines.append(f"| {r['team_zh']} | {r['previous_regular_season_schedule_matches']} | {r['previous_last15_observed']} / {r['current_observed']} | {pct(r['recent5_vs_previous_season_last15_start_share_overlap'])} | {pct(r['consecutive5_vs_preceding15']['overlap'])} |")
    lines += ['',
        '每场严格确认比赛ID、日期、赛季、主客队，以及两队各11个首发；出场和替补标记另与该场出场统计核对。未使用active字段区分登场，因为未登场替补也可能标为active。', '',
        '第一项重叠固定比较本季最近5场和上季末15场；第二项比较连续的最近5场与此前15场，参考段会含本季更早比赛。跨级比赛仅用于同队人员使用描述，不能把英冠表现直接视为英超实力。', '',
        '缺少分钟，不估算换人时点；未与Understat球员ID按名字连接，不能与Understat分钟指标混合排名。过去首发不是预测目标比赛的首发。', '',
        '赛季汇总接口存在阶段标签和累计次数口径问题，已退出正式人员比较；本文件改用逐场来源。没有给胜率加减分，也没有重新训练。', '',
        '停止节点：完成三队的明确历史窗口及覆盖测量后停止补抓；不为此扩展到全部球队所有球员生涯。若要把人员变化写入胜率，需要另行讨论可学习的作用及识别条件。', '',
        f"本轮解析{len(observed)} / {len(unique)}个唯一比赛详情；异常{len(errors)}项，详见JSON。", '',
        '来源：[ESPN考文垂赛程](https://www.espn.com/soccer/team/fixtures/_/id/388/coventry-city)、[赫尔城赛程](https://www.espn.com/soccer/team/fixtures/_/id/306/hull-city)、[伊普斯维奇赛程](https://www.espn.com/soccer/team/fixtures/_/id/373/ipswich-town)。逐场API地址、获取时间、原始响应在data/espn/personnel_20261026保留。']
    (OUT/'逐场人员参考与适用范围.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(OUT),'parsed':len(observed),'requested':len(unique),'errors':errors,
        'teams':[{k:r[k] for k in ('team','coverage_complete','current_observed','recent5_vs_previous_season_last15_start_share_overlap')}
                 for r in teams]},ensure_ascii=False))


if __name__ == '__main__':
    main()
