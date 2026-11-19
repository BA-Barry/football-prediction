"""Describe current EPL personnel history from saved, earlier source records.

No network, fitting, score evaluation, target squad, or probability adjustment.
"""
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import argparse
import json
from math import fsum
from pathlib import Path
import re

from football_model.historical_player_usage import extract_usage
from football_model.historical_relevance import personnel_usage_overlap
from football_model.understat_rosters import extract_starting_xi
from audit_understat_roster_seasons_20261025 import canonical_team, normalized_name

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
SOURCE = ROOT/'data/understat/roster_EPL_2025_2026_20261025'
OUT = ROOT/'outputs/当前人员历史代表性_20261025'
CUTOFF = date(2026, 10, 25)
PREVIOUS, CURRENT = '2025/26', '2026/27'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def shares(rows):
    totals = defaultdict(float)
    for row in rows:
        for player, minutes in row['minutes'].items():
            if minutes > 0:
                totals[player] += minutes
    denominator = fsum(totals.values())
    return {p: value/denominator for p, value in sorted(totals.items())} if denominator else {}


def window(rows):
    return {'n_matches': len(rows), 'match_ids': [r['match_id'] for r in rows],
            'date_min': rows[0]['date'] if rows else None,
            'date_max': rows[-1]['date'] if rows else None,
            'source_minute_shares': shares(rows)}


def verified_observation(fixture, payload):
    """Check directory-to-payload identity; this is not an independent FD join."""
    observation = extract_starting_xi(payload, fixture)
    rosters = payload['rosters']
    players = {str(r['player_id']): r for side in ('h', 'a') for r in rosters[side].values()}
    errors, checked = [], 0
    for side, name in (('h', 'home'), ('a', 'away')):
        expected_club = fixture[name+'_source_team_id']
        if any(str(row['team_id']) != expected_club for row in rosters[side].values()):
            errors.append(name+':source_team_id')
        shots = payload.get('shots', {}).get(side)
        if not isinstance(shots, list):
            errors.append(side+':missing_shot_list')
            continue
        for shot in shots:
            checked += 1
            try:
                player = players.get(str(shot['player_id']))
                checks = {
                    'match_id': str(shot['match_id']) == fixture['understat_id'],
                    'date': shot['date'][:10] == fixture['date'],
                    'home': canonical_team(shot['h_team'], 'E0') == fixture['home'],
                    'away': canonical_team(shot['a_team'], 'E0') == fixture['away'],
                    'player_side': player is not None and player['h_a'] == side == shot['h_a'],
                }
                errors.extend(f"shot:{shot.get('id')}:{key}" for key, passed in checks.items() if not passed)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append('malformed_shot:'+str(exc))
        template = payload.get('tmpl', {}).get(name, '')
        if isinstance(template, str):
            match = re.search(r'<h3>\s*<a\b[^>]*>([^<]*)</a>', template)
            if match and canonical_team(match.group(1), 'E0') != fixture[name]:
                errors.append(name+':template_team')
    if not checked:
        errors.append('no_shot_identity_evidence')
    if errors:
        raise ValueError('; '.join(errors))
    observation.update(season=fixture['season'],
        identity_basis='source_league_directory_plus_all_shot_ids_dates_teams_players_and_roster_team_ids',
        independent_fixture_identity_verified=False)
    usage = extract_usage(payload, fixture)
    for source_side, side in (('h', 'home'), ('a', 'away')):
        observation[side+'_appeared'] = [f"understat:player:{r['player_id']}"
            for r in rosters[source_side].values() if float(r['time']) > 0]
        observation[side+'_listed'] = [f"understat:player:{r['player_id']}"
            for r in rosters[source_side].values()]
    names = {f"understat:player:{r['player_id']}": r.get('player') for r in players.values()}
    return observation, usage, names, checked


def team_profiles(selected, observations, usages, names):
    teams = sorted({f[side] for f in selected if f['season'] == CURRENT for side in ('home', 'away')})
    accepted = {r['match_id'] for r in observations}
    by_team = defaultdict(list)
    for row in usages:
        by_team[row['team_id']].append(row)
    known_last_season_anywhere = defaultdict(set)
    for row in usages:
        if row['season'] == PREVIOUS:
            for player, minutes in row['minutes'].items():
                if minutes > 0:
                    known_last_season_anywhere[player].add(row['team_id'])
    result = []
    for team in teams:
        planned = sorted((f for f in selected if team in (f['home'], f['away'])),
                         key=lambda f: (f['date'], f['match_id']))
        history = sorted(by_team[team], key=lambda r: (r['date'], r['match_id']))
        current_planned = [f for f in planned if f['season'] == CURRENT]
        previous_planned = [f for f in planned if f['season'] == PREVIOUS]
        current_rows = [r for r in history if r['season'] == CURRENT]
        previous_rows = [r for r in history if r['season'] == PREVIOUS]
        missing = [f['match_id'] for f in planned if f['match_id'] not in accepted]
        current_complete = bool(current_planned) and all(f['match_id'] in accepted for f in current_planned)
        previous_complete = bool(previous_planned) and all(f['match_id'] in accepted for f in previous_planned)
        recent_ids = {f['match_id'] for f in planned[-5:]}
        recent = [r for r in history if r['match_id'] in recent_ids]
        tail_ids = {f['match_id'] for f in previous_planned[-15:]}
        tail = [r for r in previous_rows if r['match_id'] in tail_ids]
        recent_ready = len(recent_ids) == len(recent) == 5
        tail_ready = len(tail_ids) == len(tail) == 15
        rshares, pshares = shares(recent), shares(tail)
        minute_overlap = (fsum(min(value, pshares.get(player, 0.)) for player, value in rshares.items())
                          if recent_ready and tail_ready else None)
        last_season_players = {p for row in previous_rows for p, m in row['minutes'].items() if m > 0}
        cshares = shares(current_rows)
        retained_share = (fsum(value for player, value in cshares.items() if player in last_season_players)
                          if current_complete and previous_complete else None)
        continuity = personnel_usage_overlap(observations, team, CUTOFF)
        # Missing selected games must not silently turn an older observed window into the latest window.
        required_starter_ids = {f['match_id'] for f in planned[-20:]}
        if not required_starter_ids.issubset(accepted):
            continuity['overlap'] = None
            continuity['reference_start_share_seen_in_recent'] = None
            continuity['recent_start_share_seen_in_reference'] = None
            continuity['missing_reason'] = 'selected_games_missing_from_required_latest_windows'
        per_player = []
        for player, recent_share in sorted(rshares.items(), key=lambda item: (-item[1], item[0])):
            recent_records = [{'match_id': row['match_id'], 'date': row['date'],
                               'source_minutes': row['minutes'].get(player, 0.)}
                              for row in recent if row['minutes'].get(player, 0.) > 0]
            prior_team_minutes = fsum(row['minutes'].get(player, 0.) for row in previous_rows)
            per_player.append({'player_id': player, 'source_names': names.get(player),
                'recent_source_minute_share': recent_share, 'recent_records_with_positive_minutes': recent_records,
                'previous_season_same_team_observed_minutes': prior_team_minutes,
                'previous_season_same_team_observed_positive_matches': sum(row['minutes'].get(player, 0.) > 0 for row in previous_rows),
                'previous_season_observed_team_ids': sorted(known_last_season_anywhere.get(player, [])),
                'previous_season_same_team_status': ('observed' if prior_team_minutes > 0 else
                    'not_observed_in_complete_same_team_season' if previous_complete else 'reference_incomplete_or_team_absent')})
        minute_missing = []
        if not recent_ready:
            minute_missing.append('recent_five_selected_games_incomplete')
        if not previous_planned:
            minute_missing.append('team_absent_from_previous_EPL_season; promotion_is_not_total_personnel_turnover')
        elif not tail_ready:
            minute_missing.append('previous_season_last_fifteen_games_incomplete')
        result.append({'team': team, 'before_date': CUTOFF.isoformat(),
            'planned_matches_by_season': dict(Counter(f['season'] for f in planned)),
            'available_matches_by_season': dict(Counter(r['season'] for r in history)),
            'missing_match_ids': missing, 'current_season_complete_to_selected_cutoff': current_complete,
            'previous_season_complete': previous_complete,
            'starting_appearance_overlap_last5_vs_preceding15': continuity,
            'source_minute_overlap_last5_vs_previous_season_last15': {
                'overlap': minute_overlap, 'recent': window(recent), 'previous_season_last15': window(tail),
                'missing_reasons': minute_missing},
            'current_season_minute_share_by_previously_observed_same_team_players': retained_share,
            'current_season_window': window(current_rows),
            'recent_players': per_player,
            'probability_adjustment': None, 'causal_strength_effect': None,
            'interpretation': 'observed historical EPL usage only; not current registration, transfer confirmation, expected XI or player quality'})
    return result


def pct(value):
    return '未知' if value is None else f'{value:.1%}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.drive.casefold() != 'g:':
        raise ValueError('All outputs must be on G:')
    output.mkdir(parents=True, exist_ok=True)
    selection = read(source/'比赛选择.json')
    selected = selection['selected']
    if any(date.fromisoformat(f['date']) >= CUTOFF for f in selected):
        raise ValueError('Selection includes a match on or after the analysis cutoff')
    independent_path = output/'独立比赛连接.json'
    independent, independent_errors = {}, []
    if independent_path.exists():
        try:
            for item in read(independent_path)['matches']:
                uid = str(item['understat_id'])
                if uid in independent:
                    raise ValueError('Duplicate independently joined Understat ID')
                independent[uid] = item
        except (OSError, ValueError, KeyError, TypeError) as exc:
            independent = {}
            independent_errors.append(str(exc))
    observations, usages, statuses = [], [], []
    names = defaultdict(set)
    total_shots = 0
    for fixture in selected:
        uid = fixture['understat_id']
        status = {**fixture, 'status': 'missing'}
        statuses.append(status)
        try:
            receipt = read(source/'receipts'/f'{uid}.json')
            if receipt.get('status_code') != 200 or receipt.get('error'):
                status.update(status='request_failed', error=receipt.get('error'))
                continue
            payload = read(source/'raw'/f'{uid}.json')
            if json.loads((source/'raw'/f'{uid}_response.bin').read_bytes()) != payload:
                raise ValueError('Saved raw response differs from parsed JSON')
            observation, usage, current_names, shots = verified_observation(fixture, payload)
            observation['receipt'] = {k: receipt[k] for k in ('requested_at', 'received_at', 'status_code', 'final_url') if k in receipt}
            independent_item = independent.get(uid)
            independent_verified = False
            if independent_item is not None:
                identity_fields_match = all(independent_item.get(key) == fixture[key]
                    for key in ('date', 'home', 'away', 'season'))
                independent_verified = independent_item.get('verified') is True and identity_fields_match
                if not identity_fields_match:
                    independent_errors.append(uid+':independent_identity_fields_disagree')
                if independent_verified:
                    observation['independent_fixture_identity_verified'] = True
                    observation['football_data_match_id'] = independent_item['football_data_match_id']
                    observation['identity_basis'] += '_and_independent_FD_exact_join'
            for row in usage:
                row['independent_fixture_identity_verified'] = independent_verified
                if independent_verified:
                    row['football_data_match_id'] = independent_item['football_data_match_id']
            observations.append(observation)
            usages.extend(usage)
            for player, source_name in current_names.items():
                if isinstance(source_name, str):
                    names[player].add(source_name)
            total_shots += shots
            status.update(status='usable_source_identity', shots_checked=shots,
                          independent_fixture_identity_verified=independent_verified)
        except FileNotFoundError:
            continue
        except (OSError, ValueError, KeyError, TypeError) as exc:
            status.update(status='unusable', error=str(exc))
    source_names = {player: {'raw_names': sorted(values),
                            'normalized_names': sorted({normalized_name(value) for value in values})}
                    for player, values in sorted(names.items())}
    profiles = team_profiles(selected, observations, usages, source_names)
    measured_at = datetime.now(timezone.utc).isoformat()
    summary = {'measured_at': measured_at, 'before_date': CUTOFF.isoformat(), 'source': str(source),
        'selected_matches': len(selected), 'usable_matches': len(observations),
        'status_counts': dict(Counter(r['status'] for r in statuses)),
        'usable_matches_by_season': dict(Counter(r['season'] for r in observations)),
        'source_shots_identity_checked': total_shots, 'current_teams': len(profiles),
        'teams_with_complete_starter_windows': sum(p['starting_appearance_overlap_last5_vs_preceding15']['overlap'] is not None for p in profiles),
        'teams_with_crossseason_minute_overlap': sum(p['source_minute_overlap_last5_vs_previous_season_last15']['overlap'] is not None for p in profiles),
        'independent_fixture_identity_verified': bool(observations) and all(r['independent_fixture_identity_verified'] for r in observations),
        'independently_verified_usable_matches': sum(r['independent_fixture_identity_verified'] for r in observations),
        'independent_fixture_join_path': str(independent_path) if independent_path.exists() else None,
        'independent_fixture_join_errors': independent_errors,
        'original_publication_time_known': False, 'model_fit_count': 0,
        'results_evaluation_performed': False, 'probabilities_changed': False,
        'source_minutes_semantics': 'provider nominal minutes; shares use observed totals, not repaired to 990',
        'limitations': ['两季均为英超已存人员记录；杯赛、欧战、国家队和训练缺失。',
            '先核对同源稳定ID、目录/射门/人员块；独立赛程核对另按逐场标记，缺失时不会宣称完成。',
            '上季未在英超出现的球队没有同队基准；不能解释为球员全部更换。',
            '零分钟来源行保留为名单身份，但不当作正分钟出场。',
            '重叠是历史使用分布，未估计球员能力、替代质量或可校准的概率影响。',
            '数据本轮下载时间晚于旧预测；不会倒填为旧预测当时已持有的信息。']}
    write(output/'逐场人员记录状态.json', statuses)
    write(output/'首发身份观察.json', {'observations': observations})
    write(output/'逐场球员使用记录.json', {'observations': usages})
    write(output/'球员身份名称.json', {'identity_basis': 'stable Understat player_id; names are display metadata only',
                                   'players': source_names})
    write(output/'球队历史代表性.json', {'summary': summary, 'teams': profiles})
    lines = ['# 当前英超人员历史代表性', '',
        f'测量时间：{measured_at}；只使用比赛日期早于2026-10-25的记录。', '',
        f'已读取 {len(observations)}/{len(selected)} 场；当前球队 {len(profiles)} 支。未训练、未复核预测赛果、未改变胜率。', '',
        f"其中 {summary['independently_verified_usable_matches']} 场另有Football-Data精确比赛连接；其余仍只完成同源身份核对。", '',
        '三个比例分别描述不同问题，不能互相替代，也不能当作球队胜率。', '',
        '| 球队 | 本季可用/计划 | 最近5场首发 vs 前15场 | 最近5场分钟 vs 上季最后15场 | 本季分钟来自上季同队出现过球员 |',
        '|---|---:|---:|---:|---:|']
    for p in profiles:
        n = p['available_matches_by_season'].get(CURRENT, 0)
        planned = p['planned_matches_by_season'].get(CURRENT, 0)
        lines.append(f"| {p['team'].split(':',2)[-1]} | {n}/{planned} | "
            f"{pct(p['starting_appearance_overlap_last5_vs_preceding15']['overlap'])} | "
            f"{pct(p['source_minute_overlap_last5_vs_previous_season_last15']['overlap'])} | "
            f"{pct(p['current_season_minute_share_by_previously_observed_same_team_players'])} |")
    lines += ['', '“最近5场首发 vs 前15场”是互不重叠窗口的首发次数份额重叠；总计不足20场返回未知。', '',
        '“最近5场分钟 vs 上季最后15场”固定比较换季两侧的历史使用结构；分钟是数据商原始名义计数。', '',
        '“本季分钟来自上季同队出现过球员”只说明ID曾在同队有正分钟记录，不说明这些球员状态、角色和能力保持不变。', '',
        '## 使用边界', '']
    lines.extend('- '+limit for limit in summary['limitations'])
    if any(r['status'] not in ('usable_source_identity', 'missing') for r in statuses):
        lines += ['', '存在不能使用的来源记录，详见逐场人员记录状态.json；相关窗口的缺失不会填成零。']
    (output/'当前人员历史代表性说明.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(output), **summary}, ensure_ascii=False))


if __name__ == '__main__':
    main()
