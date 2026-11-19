"""Same-source cross-tier player appearances; never substitute these for minutes."""
from datetime import datetime, timezone
import json
from math import fsum, isfinite
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

from probe_espn_personnel_20261026 import OUT as ARCHIVE, read_source

ROOT = REPOSITORY_ROOT
OUT = ROOT/'outputs/升班球队人员参考_20261026'
CLUBS = [('Coventry', '考文垂', '388'), ('Hull', '赫尔城', '306'), ('Ipswich', '伊普斯维奇', '373')]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def number(value):
    if value is None:
        return None
    result = float(value)
    if isinstance(value, bool) or not isfinite(result) or result < 0 or int(result) != result:
        raise ValueError('Appearance counts must be nonnegative integers')
    return int(result)


def parse(payload, club, year):
    if str(payload.get('team', {}).get('id')) != club or payload.get('season', {}).get('year') != year:
        raise ValueError('Returned team or season differs from requested identity')
    rows, seen, fields = [], set(), set()
    for athlete in payload['athletes']:
        pid = str(athlete['id'])
        if not pid.isdecimal() or pid in seen:
            raise ValueError('Invalid or repeated source player ID')
        seen.add(pid)
        stats = {}
        for category in athlete.get('statistics', {}).get('splits', {}).get('categories', []):
            for stat in category.get('stats', []):
                key = stat['name']
                if key in stats:
                    raise ValueError('Ambiguous repeated player statistic')
                stats[key] = stat.get('value')
                fields.add(key)
        apps, subs = number(stats.get('appearances')), number(stats.get('subIns'))
        if apps is not None and subs is not None and subs > apps:
            raise ValueError('Substitute appearances exceed total appearances')
        rows.append({'player_id':'espn:player:'+pid, 'source_name':athlete.get('displayName'),
            'appearances':apps, 'sub_ins':subs,
            'derived_start_count':apps-subs if apps is not None and subs is not None else None,
            'reported_minutes':None,
            'availability':'source statistic absent' if apps is None else 'source season appearance count'})
    positive = [r for r in rows if r['appearances'] is not None and r['appearances'] > 0]
    return {'source_team_id':club, 'season_metadata':payload['season'], 'source_timestamp':payload.get('timestamp'),
        'listed_players':len(rows), 'positive_appearance_players':len(positive),
        'missing_appearance_statistics':sum(r['appearances'] is None for r in rows),
        'positive_appearance_total':sum(r['appearances'] for r in positive),
        'derived_start_count_total':sum(r['derived_start_count'] or 0 for r in positive),
        'raw_statistic_fields':sorted(fields), 'players':rows,
        'minutes_available':any('minute' in key.lower() for key in fields),
        'coverage_warning':'Season aggregate; phase metadata may name a promotion final while totals span league games. Do not treat these as phase-specific or a last15-match window.'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results, failures = [], []
    for name, zh, club in CLUBS:
        parsed = {}
        for year, league in ((2025, 'eng.2'), (2026, 'eng.1')):
            label = f'{name.lower()}_{year}_roster'
            path = ARCHIVE/(label+'.json')
            if path.exists():
                payload = read(path)
                receipt = read(ARCHIVE/(label+'_receipt.json'))
            else:
                payload, receipt = read_source(label,
                    f'https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/teams/{club}/roster?season={year}')
            if payload is None:
                failures.append({'team':name, 'year':year, 'receipt':receipt})
                break
            current = parse(payload, club, year)
            current.update(source_url=receipt['url'], received_at=receipt['received_at'], raw_path=str(path))
            parsed[year] = current
        if len(parsed) != 2:
            break
        previous, current = parsed[2025], parsed[2026]
        old_positive = {r['player_id'] for r in previous['players'] if (r['appearances'] or 0)>0}
        old_missing = {r['player_id'] for r in previous['players'] if r['appearances'] is None}
        prior = {r['player_id']:r['appearances']/previous['positive_appearance_total']
                 for r in previous['players'] if (r['appearances'] or 0)>0}
        now = {r['player_id']:r['appearances']/current['positive_appearance_total']
               for r in current['players'] if (r['appearances'] or 0)>0}
        overlap = fsum(min(value, prior.get(pid, 0.)) for pid,value in now.items())
        same_ids = fsum(value for pid,value in now.items() if pid in old_positive)
        missing_old = fsum(value for pid,value in now.items() if pid in old_missing)
        results.append({'team':name, 'team_zh':zh, 'source_team_id':'espn:team:'+club,
            'previous':previous, 'current':current,
            'current_known_appearance_share_from_previous_positive_ids':same_ids,
            'current_known_appearance_share_with_missing_previous_statistics':missing_old,
            'whole_season_appearance_distribution_overlap':overlap,
            'current_positive_players_seen_previous_same_team':sum(pid in old_positive for pid in now),
            'current_positive_players':len(now),
            'current_players':[{**row, 'previous_same_team_status':
                'positive_appearances_observed' if row['player_id'] in old_positive else
                'listed_but_appearance_statistic_missing' if row['player_id'] in old_missing else
                'no_positive_appearance_observed_in_saved_previous_roster'}
                for row in current['players'] if (row['appearances'] or 0)>0],
            'understat_identity_join_performed':False, 'minute_overlap':None,
            'numerical_probability_effect':None,
            'metric_warning':'Every appearance has equal weight; one substitute appearance and one full match both count once. Not a minute retention ratio or an ability estimate.'})
    report = {'measured_at':datetime.now(timezone.utc).isoformat(), 'source':'ESPN public season roster statistics',
        'teams':results, 'failures':failures,
        'use_status':'diagnostic_only_not_validated_for_personnel_comparison',
        'aggregate_coverage_issue':'For all three 2025/26 teams, sum(appearances-subIns) exceeds 11*46; phase filtering is not honoured. Match-specific personnel records supersede this aggregate comparison.',
        'historical_roster_first_publication_time_known':False,
        'current_aggregate_observed_today':True, 'model_fit_count':0, 'forecasts_changed':False,
        'scope':'Same ESPN player IDs across previous Championship and current Premier League season; a separate appearance-count comparison, not a replacement for Understat minute metrics.'}
    (OUT/'同源跨级人员对照.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# 三支升班球队：赛季汇总接口诊断（不采用）','', '测量时间：'+report['measured_at'], '',
        '本接口退出正式人员比较。三队上季由出场减替补推导的首发总次数分别为516、536、529，均超过46场×11人=506；来源无法明确保证纯常规赛、同队口径，阶段参数也未被接口正确执行。以下数字仅保留排查依据，不能当作已验证的人员保持率。正式比较改用《逐场人员参考与适用范围》。', '',
        '本轮接通ESPN两季球员ID和出场次数。来源没有分钟，所以下表只描述出场次数，原Understat分钟缺失仍保留。', '',
        '| 球队 | 上季/本季有正出场记录人数 | 本季出场次数来自上季同队正出场ID | 两季全段出场份额重叠 |', '|---|---:|---:|---:|']
    for row in results:
        lines.append(f"| {row['team_zh']} | {row['previous']['positive_appearance_players']} / {row['current']['positive_appearance_players']} | {row['current_known_appearance_share_from_previous_positive_ids']:.1%} | {row['whole_season_appearance_distribution_overlap']:.1%} |")
    lines+=['', '一场替补出场与一场踢满均记为一次出场，不能与之前的名义分钟比例混合或直接排名。', '',
        '上季返回元数据的阶段名可能显示升级决赛，但统计含大量联赛出场；因此保留原值，明确标为赛季汇总，不宣称其为决赛、纯常规赛或末15场记录。', '',
        '本季为本次下载时来源累积值，可能比上轮截至10月24日的Understat记录多一场，不能倒填为原预测输入。缺少出场统计的名单行保留未知。', '',
        '球员身份在ESPN内部按ID连接；没有按名字猜测ESPN与Understat之间的ID对应。', '',
        '不能使用这些有歧义的累计份额收束LLM或修正模型；应以逐场球员ID和明确参与记录替代。', '',
        '## 来源及样本','']
    for row in results:
        for key in ('previous','current'):
            data=row[key]
            lines.append(f"- {row['team_zh']} {data['season_metadata']['displayName']}：[来源]({data['source_url']})；出场次数总和{data['positive_appearance_total']}，由出场减替补推导的首发次数总和{data['derived_start_count_total']}，缺统计名单行{data['missing_appearance_statistics']}。")
    if failures:
        lines+=['','未完成来源请求详见JSON failures；没有用姓名或零值补齐。']
    (OUT/'三队人员依据与局限.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(OUT), 'teams':len(results), 'failures':failures,
        'measurements':[{k:r[k] for k in ('team','current_known_appearance_share_from_previous_positive_ids','whole_season_appearance_distribution_overlap')} for r in results]},ensure_ascii=False))


if __name__ == '__main__':
    main()
