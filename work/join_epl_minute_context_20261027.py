"""Exact historical minute-package/FD/Understat identity join, without fitting.

An ESPN ID is retained as provenance only. Scores are checked after exact
date/home/away linkage and never used to select or repair a match identity.
Rows are state/event aggregates, not independent elapsed-minute exposures.
"""
from collections import Counter, defaultdict
import csv
from datetime import date
import gzip
import io
import json
from pathlib import Path
import zipfile

from football_model.xg_history import COMPETITION_TEAM_NAMES
from run_npxg_comparison_20261025 import load_source


ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OUT = ROOT / 'outputs/平局条件信息定位_20261027'
ARCHIVE = ROOT / 'work/research/game_context_20261027/PremierLeague_Sami_minute_by_minute_df_w_red_cards_WITH_BOOKING_ODDS.zip'
YEARS = {'2021': '2020/21', '2022': '2021/22', '2023': '2022/23'}
# Explicit identity aliases, decided from team names rather than match results.
ALIASES = {
    **COMPETITION_TEAM_NAMES['E0'],
    'AFC Bournemouth': 'Bournemouth',
    'Brighton & Hove Albion': 'Brighton',
    'Leeds United': 'Leeds',
    'Leicester City': 'Leicester',
    'Norwich City': 'Norwich',
    'Sheffield United': 'Sheffield United',
    'Tottenham Hotspur': 'Tottenham',
    'West Ham United': 'West Ham',
}


def whole(raw):
    number = float(raw)
    if not number.is_integer():
        raise ValueError(f'Expected integer count/state: {raw}')
    return int(number)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parsed, independent, source_joins, _, directories = load_source()
    fd_index, pair_dates, known_teams = {}, defaultdict(list), set()
    for record in independent.values():
        if record.competition != 'E0' or record.season not in YEARS.values():
            continue
        key = (record.season, record.match_date.isoformat(), record.home_label, record.away_label)
        if key in fd_index:
            raise ValueError('Repeated independent date/home/away fixture')
        fd_index[key] = record
        pair_dates[(record.season, record.home_label, record.away_label)].append(record.match_date.isoformat())
        known_teams.update((record.home_label, record.away_label))
    games = {}
    with zipfile.ZipFile(ARCHIVE) as archive:
        members = [m for m in archive.namelist() if m.endswith('.csv')]
        if len(members) != 1:
            raise ValueError('Expected one source CSV')
        member = members[0]
        with archive.open(member) as source:
            for row in csv.DictReader(io.TextIOWrapper(source, encoding='utf-8-sig')):
                if row['season'] not in YEARS:
                    continue
                season = YEARS[row['season']]
                game_id = row['gameId']
                played = date.fromisoformat(row['gamedDate']).isoformat()
                game = games.setdefault(game_id, {
                    'espn_game_id': game_id, 'source_id': row['ID'], 'season': season,
                    'date': played, 'teams': {}, 'rows': 0, 'goals': Counter(),
                    'half_goals': Counter(), 'minute_rows': Counter(), 'minute_states': defaultdict(set),
                })
                if (game['season'], game['date'], game['source_id']) != (season, played, row['ID']):
                    raise ValueError('ESPN game ID has conflicting season/date/source ID')
                side = row['HomeAway']
                if side not in ('Home', 'Away'):
                    raise ValueError('Unrecognized source HomeAway')
                raw_team = row['Team']
                if side in game['teams'] and game['teams'][side] != raw_team:
                    raise ValueError('Conflicting source side/team identity')
                game['teams'][side] = raw_team
                half, minute = whole(row['half_id']), whole(row['Minute.clean'])
                if half not in (1, 2):
                    raise ValueError('Invalid declared source half')
                goal = whole(row['Goals'])
                if goal < 0:
                    raise ValueError('Negative source goal event count')
                key = (side, half, minute)
                game['rows'] += 1
                game['goals'][side] += goal
                if half == 1:
                    game['half_goals'][side] += goal
                game['minute_rows'][key] += 1
                game['minute_states'][key].add((whole(row['Score.Diff']), whole(row['RedCard.Diff'])))
    by_season = {}
    for season in YEARS.values():
        by_season[season] = {
            'package_games': 0, 'exact_independent_matches': 0, 'exact_trusted_understat_matches': 0,
            'goal_count_matches_full_time': 0, 'goal_count_conflicts_full_time': 0,
            'goal_count_matches_half_time': 0, 'goal_count_conflicts_half_time': 0,
            'games_with_repeated_team_half_minute': 0, 'extra_repeated_minute_rows': 0,
            'games_with_multiple_states_same_team_half_minute': 0, 'multiple_state_groups': 0,
            'unique_team_half_minute_keys': 0, 'package_rows': 0,
            'unknown_team_names': [], 'unmatched_fixtures': [], 'goal_conflicts': [],
        }
    details, used_ids, unknown_names = [], set(), Counter()
    for game in sorted(games.values(), key=lambda g: (g['date'], g['espn_game_id'])):
        if set(game['teams']) != {'Home', 'Away'}:
            raise ValueError('Source fixture lacks both identified sides')
        stats = by_season[game['season']]
        stats['package_games'] += 1
        raw_home, raw_away = game['teams']['Home'], game['teams']['Away']
        home, away = ALIASES.get(raw_home, raw_home), ALIASES.get(raw_away, raw_away)
        for raw, canonical in ((raw_home, home), (raw_away, away)):
            if canonical not in known_teams:
                unknown_names[raw] += 1
                stats['unknown_team_names'].append(raw)
        key = (game['season'], game['date'], home, away)
        record = fd_index.get(key)
        repeated = [k for k, n in game['minute_rows'].items() if n > 1]
        multi_states = [k for k, states in game['minute_states'].items() if len(states) > 1]
        extras = sum(n-1 for n in game['minute_rows'].values())
        stats['games_with_repeated_team_half_minute'] += bool(repeated)
        stats['extra_repeated_minute_rows'] += extras
        stats['games_with_multiple_states_same_team_half_minute'] += bool(multi_states)
        stats['multiple_state_groups'] += len(multi_states)
        stats['unique_team_half_minute_keys'] += len(game['minute_rows'])
        stats['package_rows'] += game['rows']
        full = [game['goals'][side] for side in ('Home', 'Away')]
        half = [game['half_goals'][side] for side in ('Home', 'Away')]
        item = {
            'espn_game_id': game['espn_game_id'], 'source_id': game['source_id'],
            'season': game['season'], 'date': game['date'],
            'home_raw': raw_home, 'away_raw': raw_away, 'home': home, 'away': away,
            'package_rows': game['rows'], 'unique_team_half_minute_keys': len(game['minute_rows']),
            'extra_repeated_minute_rows': extras,
            'multiple_state_groups': [
                {'side': k[0], 'half_id': k[1], 'minute_clean': k[2],
                 'row_count': game['minute_rows'][k],
                 'score_red_diff_states': sorted(map(list, game['minute_states'][k]))}
                for k in sorted(multi_states)
            ],
            'package_goal_event_sum': full, 'package_half_1_goal_event_sum': half,
            'football_data_match_id': None, 'trusted_understat_id': None,
        }
        if record is None:
            mismatch = {
                'espn_game_id': game['espn_game_id'], 'source_date': game['date'],
                'home': home, 'away': away,
                'same_pair_independent_dates': pair_dates.get((game['season'], home, away), []),
                'reason': 'no_exact_date_home_away_match_no_date_repair',
            }
            stats['unmatched_fixtures'].append(mismatch)
            item['unmatched'] = mismatch
        else:
            if record.match_id in used_ids:
                raise ValueError('More than one ESPN source fixture maps to one independent fixture')
            used_ids.add(record.match_id)
            stats['exact_independent_matches'] += 1
            item['football_data_match_id'] = record.match_id
            item['independent_full_time'] = list(record.full_time)
            item['independent_half_time'] = list(record.half_time) if record.half_time is not None else None
            full_ok = full == list(record.full_time)
            half_ok = record.half_time is not None and half == list(record.half_time)
            item['full_time_goal_check'] = full_ok
            item['half_time_goal_check'] = half_ok
            stats['goal_count_matches_full_time' if full_ok else 'goal_count_conflicts_full_time'] += 1
            stats['goal_count_matches_half_time' if half_ok else 'goal_count_conflicts_half_time'] += 1
            if not full_ok or not half_ok:
                stats['goal_conflicts'].append({k: item[k] for k in (
                    'espn_game_id', 'date', 'home', 'away', 'package_goal_event_sum',
                    'package_half_1_goal_event_sum', 'independent_full_time', 'independent_half_time')})
            trusted = parsed.get(record.match_id)
            if trusted is not None:
                if (trusted.match_date, trusted.home, trusted.away, trusted.goals) != (
                        record.match_date, record.home, record.away, record.full_time):
                    raise ValueError('Existing trusted independent/source link identity conflict')
                stats['exact_trusted_understat_matches'] += 1
                item['trusted_understat_id'] = trusted.understat_id
        details.append(item)
    for season, stats in by_season.items():
        stats['unknown_team_names'] = sorted(set(stats['unknown_team_names']))
        stats['independent_games_not_in_package'] = sorted(
            r.match_id for r in fd_index.values() if r.season == season
            and r.match_id not in used_ids)
    summary = {
        'purpose': 'historical exact identity linkage and measurement only; no new model fitting',
        'archive': str(ARCHIVE), 'csv_member': member,
        'join_key': ['declared_season', 'unmodified_calendar_date', 'explicit_canonical_home', 'explicit_canonical_away'],
        'espn_game_id_used_as_cross_provider_key': False,
        'aliases': ALIASES, 'unknown_team_names': dict(unknown_names),
        'by_season': by_season,
        'total_package_games': len(games),
        'total_exact_independent_matches': sum(s['exact_independent_matches'] for s in by_season.values()),
        'total_exact_trusted_understat_matches': sum(s['exact_trusted_understat_matches'] for s in by_season.values()),
        'full_time_goal_matches': sum(s['goal_count_matches_full_time'] for s in by_season.values()),
        'full_time_goal_conflicts': sum(s['goal_count_conflicts_full_time'] for s in by_season.values()),
        'half_time_goal_matches': sum(s['goal_count_matches_half_time'] for s in by_season.values()),
        'half_time_goal_conflicts': sum(s['goal_count_conflicts_half_time'] for s in by_season.values()),
        'games_with_multiple_states_same_team_half_minute': sum(s['games_with_multiple_states_same_team_half_minute'] for s in by_season.values()),
        'multiple_state_groups': sum(s['multiple_state_groups'] for s in by_season.values()),
        'extra_repeated_minute_rows': sum(s['extra_repeated_minute_rows'] for s in by_season.values()),
        'trusted_source_join_reports': {k: v for k, v in source_joins.items() if k in [f'E0:{y}' for y in YEARS.values()]},
        'independent_source_directories': {k: v for k, v in directories.items() if k in YEARS.values()},
        'limitations': [
            'No xG in the minute package; full-match xG cannot be distributed across minute states.',
            'No season overlap with existing 2023/24 and 2024/25 760-match shot collection.',
            'One source row is not one elapsed minute; same minute may have multiple score/red-card states.',
            'Source half_id is author-derived rather than independently verified official event period.',
            'Score and red-card difference do not establish true playing-time exposure or all on-field personnel.',
            'Full/half goal accounting agreement does not validate within-minute order or every score-state transition.',
            'Bookmaker columns were not used; original publication/revisions are not established.',
            'Unmatched dates and score conflicts are reported without repairing, dropping or selecting by score.',
        ],
        'new_fits': 0, 'new_network_requests': 0, 'current_match_results_checked': False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'EPL分钟资料精确联结.json', summary)
    with gzip.open(OUT / 'EPL分钟资料精确联结明细.json.gz', 'wt', encoding='utf-8') as dest:
        json.dump(details, dest, ensure_ascii=False, allow_nan=False)
    lines = [
        '# EPL分钟资料与本地历史记录精确联结', '',
        '2026-10-27。本轮只做历史数据接入与测量，不训练、不调整预测、不查询当前赛果。', '',
        '按原始日历日期、明确队名别名和主客身份联结；ESPN gameId只保留来源身份。比分在身份联结完成后核对，不用于挑选匹配或更改日期。', '',
        '|赛季|包内场数|精确联结独立FD|精确联结可信Understat|进球合计吻合终场|吻合半场|同分钟多状态场数|',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for season, stats in by_season.items():
        lines.append(f"|{season}|{stats['package_games']}|{stats['exact_independent_matches']}|{stats['exact_trusted_understat_matches']}|{stats['goal_count_matches_full_time']}|{stats['goal_count_matches_half_time']}|{stats['games_with_multiple_states_same_team_half_minute']}|")
    lines += ['',
        f"合计{summary['total_package_games']}场，精确联结{summary['total_exact_independent_matches']}场；未知球队名{len(unknown_names)}个、无法精确联结{sum(len(s['unmatched_fixtures']) for s in by_season.values())}场。终场进球计数冲突{summary['full_time_goal_conflicts']}场，半场计数冲突{summary['half_time_goal_conflicts']}场。所有冲突仍保留于结果，不强行修正。", '',
    ]
    for stats in by_season.values():
        for conflict in stats['goal_conflicts']:
            source_score = ':'.join(map(str, conflict['package_goal_event_sum']))
            fd_score = ':'.join(map(str, conflict['independent_full_time']))
            lines += [f"具体冲突：{conflict['date']} {conflict['home']}—{conflict['away']}（ESPN {conflict['espn_game_id']}），包内Goals事件累加{source_score}，独立终场比分{fd_score}。这说明精确联结不等于分钟事件已清洗正确；当前保留身份联结和异常记录，该场暂停用于后续状态修正，不据终场结果回填事件状态。本轮保存的明细不含该场完整Goals及Score.Diff行，未据此判断内部转移是否自洽，也不推定冲突原因。", '']
    lines += [
        f"{summary['games_with_multiple_states_same_team_half_minute']}场存在同队、同半场、同分钟的多种比分差／红牌差状态，共{summary['multiple_state_groups']}个键；相对唯一队伍半场分钟键，多出{summary['extra_repeated_minute_rows']}行。因此不能一行按1分钟累计暴露，也没有把状态行顺序解释成秒级顺序。", '',
        '分钟包不含xG，止于2022/23，与已有760场逐射门的2023/24及2024/25没有赛季重叠。它可以补较早聚合记录的比分／红牌状态线索，却不能据此给较新760场补时间线，也不能把全场xG摊到每分钟。', '',
        '身份及进球总数匹配只确认可联结与事件计数相容，不证明半场时钟、每个状态转移或真实场上人数准确。源half_id由作者清洗产生、缺秒与独立终场标志；当前保留研究资料身份，不接入正式预测。', '',
        '[汇总与全部异常](G:/Codex/football-prediction-2026-10-22/outputs/平局条件信息定位_20261027/EPL分钟资料精确联结.json) · [逐场联结与多状态明细](G:/Codex/football-prediction-2026-10-22/outputs/平局条件信息定位_20261027/EPL分钟资料精确联结明细.json.gz)', '',
        '到此停止本轮数据接入，不下载其余联赛包、不拟合状态修正系数。',
    ]
    (OUT / 'EPL分钟资料精确联结说明.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: summary[k] for k in (
        'total_package_games', 'total_exact_independent_matches', 'total_exact_trusted_understat_matches',
        'full_time_goal_matches', 'full_time_goal_conflicts', 'half_time_goal_matches', 'half_time_goal_conflicts',
        'games_with_multiple_states_same_team_half_minute', 'multiple_state_groups', 'extra_repeated_minute_rows')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
