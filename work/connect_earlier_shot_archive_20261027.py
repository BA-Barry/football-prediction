"""Acquire one EPL shot archive and link only explicitly identified old games.

No model training or current match results. Files, dependencies and temporary
storage for this run remain on G. IDs never identify games across providers.
"""
from datetime import datetime, timezone
import ast
from collections import Counter
import gzip
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import shutil
import urllib.request

ROOT = REPOSITORY_ROOT
DATA = ROOT / 'data/understat/earlier_shot_archive_20261027'
OUT = ROOT / 'outputs/较早射门归档接通_20261027'
RELEASE_API = 'https://api.github.com/repos/JaseZiv/worldfootballR_data/releases/tags/understat_shots'
RELEASE_PAGE = 'https://github.com/JaseZiv/worldfootballR_data/releases/tag/understat_shots'
ASSET = 'epl_shot_data.rds'


def request(url):
    return urllib.request.Request(url, headers={'User-Agent': 'football-personal-history-research'})


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def acquire():
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = DATA / '来源记录.json'
    if metadata.exists() and (DATA / ASSET).exists():
        return json.loads(metadata.read_text(encoding='utf-8'))
    release = json.load(urllib.request.urlopen(request(RELEASE_API), timeout=30))
    assets = [a for a in release['assets'] if a['name'] == ASSET]
    if len(assets) != 1:
        raise ValueError('Expected exactly one author-published EPL archive')
    asset = assets[0]
    target = DATA / ASSET
    if not target.exists():
        partial = DATA / (ASSET + '.partial')
        with urllib.request.urlopen(request(asset['browser_download_url']), timeout=60) as source:
            with partial.open('wb') as stream:
                shutil.copyfileobj(source, stream, length=1024*1024)
        partial.replace(target)
    provenance = {
        'downloaded_at': datetime.now(timezone.utc).isoformat(),
        'release_page': RELEASE_PAGE, 'release_api': RELEASE_API,
        'asset': {k: asset[k] for k in ('name', 'size', 'browser_download_url', 'updated_at')},
        'local_file': str(target), 'local_bytes': target.stat().st_size,
        'only_epl_asset_downloaded': True,
        'data_license': 'No independently verified data license; software license does not establish source-data rights.',
        'purpose': 'Personal historical data preparation; not a new prediction or model fit.',
    }
    write(metadata, provenance)
    return provenance


def inspect():
    import pyreadr
    provenance = acquire()
    provenance['reader'] = {'name': 'pyreadr', 'version': pyreadr.__version__,
                            'module_file': str(Path(pyreadr.__file__).resolve()),
                            'purpose': 'Read this author-published RDS archive into a dataframe; no R installation or training.',
                            'install_cache': str(ROOT / 'work/pip-cache'),
                            'temporary_directory': str(ROOT / 'work/tmp')}
    write(DATA / '来源记录.json', provenance)
    loaded = pyreadr.read_r(str(DATA / ASSET))
    print('RDS objects', list(loaded), flush=True)
    frames = []
    for name, frame in loaded.items():
        info = {'object_name': str(name), 'rows': len(frame), 'columns': list(frame.columns),
                'dtypes': {str(k): str(v) for k, v in frame.dtypes.items()},
                'first_rows': json.loads(frame.head(3).to_json(orient='records', date_format='iso'))}
        frames.append(info)
    write(OUT / '归档字段检查.json', {'provenance': provenance, 'reader': 'pyreadr',
          'reader_version': pyreadr.__version__, 'objects': frames})
    if list(loaded) != [None]:
        raise ValueError('Expected one standalone dataframe in RDS')
    return loaded[None], provenance


def integer(value, field):
    number = float(value)
    if not math.isfinite(number) or number != int(number):
        raise ValueError(f'Non-integral {field}: {value!r}')
    return int(number)


def main():
    import pandas as pd
    frame, provenance = inspect()
    manifest_path = ROOT / 'outputs/平局条件信息定位_20261027/EPL分钟资料精确联结明细.json.gz'
    with gzip.open(manifest_path, 'rt', encoding='utf-8') as stream:
        manifest = json.load(stream)
    # Reuse the existing explicit Understat -> Football-Data aliases, without
    # importing or fitting any model. No score/date-assisted matching is used.
    alias_file = ROOT / 'model_v2/src/football_model/xg_history.py'
    tree = ast.parse(alias_file.read_text(encoding='utf-8'))
    aliases = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'TEAM_NAMES' for t in node.targets))
    wanted = {str(row['trusted_understat_id']): row for row in manifest}
    if len(wanted) != len(manifest):
        raise ValueError('Repeated trusted Understat ID in input manifest')
    frame['match_id'] = frame['match_id'].map(lambda value: str(integer(value, 'match_id')))
    source_seasons = []
    for season, group in frame.groupby('season'):
        source_seasons.append({'season_start_year': integer(season, 'season'), 'shots': len(group),
                               'matches': group['match_id'].nunique(),
                               'first_date': str(group['date'].min()), 'last_date': str(group['date'].max())})
    selected = frame[frame['match_id'].isin(wanted)].copy()
    groups = {str(key): group for key, group in selected.groupby('match_id')}
    linked = []
    missing = []
    identity_conflicts = []
    accounting_conflicts = []
    global_goal = Counter()
    side_sources = Counter()
    side_conflicts = []
    side_missing = 0
    # The release concatenates source schemas. The populated side field changes
    # between h_a and home_away; both carry explicit h/a labels. Resolve only
    # these literal values, never the match score, player name, or row order.
    def shooting_side(shot):
        a = shot['h_a'] if pd.notna(shot['h_a']) else None
        b = shot['home_away'] if pd.notna(shot['home_away']) else None
        if a is not None and b is not None and a != b:
            return None, 'conflict'
        if a in ('h', 'a'):
            return a, 'h_a'
        if b in ('h', 'a'):
            return b, 'home_away'
        return None, 'missing_or_invalid'
    for _, shot in selected.iterrows():
        side, origin = shooting_side(shot)
        side_sources[origin] += 1
        if origin == 'conflict':
            side_conflicts.append({'match_id': shot['match_id'], 'shot_id': integer(shot['id'], 'id'),
                                   'h_a': str(shot['h_a']), 'home_away': str(shot['home_away'])})
        if side is None:
            side_missing += 1
    for source_id, target in wanted.items():
        if source_id not in groups:
            missing.append({k: target[k] for k in ['season', 'date', 'home', 'away', 'trusted_understat_id']})
            continue
        group = groups[source_id]
        home = sorted(set(str(v) for v in group['home_team']))
        away = sorted(set(str(v) for v in group['away_team']))
        dates = sorted(set(str(v)[:10] for v in group['date']))
        seasons = sorted(set(integer(v, 'season') for v in group['season']))
        home_fd = [aliases.get(v, v) for v in home]
        away_fd = [aliases.get(v, v) for v in away]
        identity_ok = (home_fd == [target['home']] and away_fd == [target['away']]
                       and dates == [target['date']] and seasons == [int(target['season'][:4])]
                       and set(group['league']) == {'EPL'})
        goals = [0, 0]
        own_goals = [0, 0]
        unknown_goal_sides = []
        goal_rows = group[group['result'].isin(['Goal', 'OwnGoal'])]
        for _, shot in goal_rows.iterrows():
            side_label, origin = shooting_side(shot)
            if side_label is None:
                unknown_goal_sides.append(str(shot['id']))
                continue
            side = 0 if side_label == 'h' else 1
            if shot['result'] == 'Goal':
                goals[side] += 1
                global_goal['Goal'] += 1
            else:
                # h_a identifies the shooting player's team: an own goal is
                # credited to the opponent, never silently discarded.
                own_goals[1-side] += 1
                global_goal['OwnGoal'] += 1
        credited = [goals[i] + own_goals[i] for i in (0, 1)]
        independent = [integer(v, 'independent_full_time') for v in target['independent_full_time']]
        final_totals = sorted(set(zip(group['home_goals'].map(lambda v: integer(v, 'home_goals')),
                                     group['away_goals'].map(lambda v: integer(v, 'away_goals')))))
        record = {k: target[k] for k in ['season', 'date', 'home', 'away', 'trusted_understat_id', 'espn_game_id', 'football_data_match_id']}
        record.update({'shot_rows': len(group), 'identity_ok': identity_ok,
                       'source_identity': {'home': home, 'away': away, 'date': dates, 'season': seasons},
                       'Goal_by_credited_side': goals, 'OwnGoal_by_credited_side': own_goals,
                       'Goal_plus_OwnGoal': credited, 'independent_full_time': independent,
                       'shot_goal_count_agrees': credited == independent and not unknown_goal_sides,
                       'source_final_totals': [list(v) for v in final_totals],
                       'source_final_totals_agree': final_totals == [tuple(independent)],
                       'unknown_goal_side_ids': unknown_goal_sides,
                       'duplicate_shot_ids': int(group['id'].duplicated().sum()),
                       'minute_min': float(group['minute'].min()), 'minute_max': float(group['minute'].max()),
                       'minute_missing': int(group['minute'].isna().sum()),
                       'xG_missing': int(group['xG'].isna().sum()),
                       'author_minute_goal_count_agrees': target['full_time_goal_check']})
        linked.append(record)
        if not identity_ok:
            identity_conflicts.append(record)
        if not record['shot_goal_count_agrees'] or not record['source_final_totals_agree']:
            accounting_conflicts.append(record)
    by_season = []
    for season in sorted({row['season'] for row in manifest}):
        original = [row for row in manifest if row['season'] == season]
        joined = [row for row in linked if row['season'] == season]
        by_season.append({'season': season, 'requested_matches': len(original), 'id_linked_matches': len(joined),
                          'identity_agree_matches': sum(row['identity_ok'] for row in joined),
                          'shot_rows': sum(row['shot_rows'] for row in joined),
                          'shot_goal_count_agree_matches': sum(row['shot_goal_count_agrees'] for row in joined),
                          'source_final_totals_agree_matches': sum(row['source_final_totals_agree'] for row in joined)})
    fields_missing = [field for field in ['period', 'second', 'seconds', 'red_card', 'score_before_shot'] if field not in frame.columns]
    minute_values = selected['minute'].dropna()
    invalid_minutes = int((~minute_values.map(lambda v: math.isfinite(float(v)) and float(v).is_integer() and v >= 0)).sum())
    summary = {'purpose': 'Historical shot archive connection and goal accounting only; zero model fits.',
               'provenance': provenance, 'input_manifest': str(manifest_path), 'explicit_alias_file': str(alias_file),
               'matching_rule': 'Exact trusted Understat match_id only, followed by date/season/home/away identity checks; scores never choose matches.',
               'archive': {'rows': len(frame), 'matches': frame['match_id'].nunique(), 'seasons': source_seasons,
                           'result_counts': {str(k): int(v) for k, v in frame['result'].value_counts().items()},
                           'nulls_by_field': {str(k): int(v) for k, v in frame.isna().sum().items()},
                           'duplicate_shot_ids': int(frame['id'].duplicated().sum())},
               'requested_matches': len(manifest), 'id_linked_matches': len(linked),
               'identity_agree_matches': sum(row['identity_ok'] for row in linked),
               'selected_shot_rows': len(selected), 'by_season': by_season,
               'missing_matches': missing, 'identity_conflicts': identity_conflicts,
               'goal_counts': dict(global_goal),
               'shot_goal_count_agree_matches': sum(row['shot_goal_count_agrees'] for row in linked),
               'source_final_totals_agree_matches': sum(row['source_final_totals_agree'] for row in linked),
               'accounting_conflicts': accounting_conflicts,
               'selected_duplicate_shot_ids': int(selected['id'].duplicated().sum()),
               'selected_result_counts': {str(k): int(v) for k, v in selected['result'].value_counts().items()},
               'selected_h_a_values': sorted(set(str(v) for v in selected['h_a'])),
               'side_normalization': {'rule': 'Literal h/a from h_a; if null use explicit home_away h/a; reject contradictory populated values.',
                                      'sources': dict(side_sources), 'unknown_or_conflict_rows': side_missing,
                                      'conflicts': side_conflicts},
               'selected_nulls_by_field': {str(k): int(v) for k, v in selected.isna().sum().items()},
               'invalid_minutes': invalid_minutes, 'missing_state_fields': fields_missing,
               'limitations': ['No period or seconds; minute values do not resolve half-boundary or same-minute ordering.',
                               'No author half_id is attached to shots; no halftime or red-card timeline has been reconstructed.',
                               'Each row is a shot, not one minute. Final score columns are final totals, not score at shot time.',
                               'Goal and OwnGoal bookkeeping agreement does not prove the author minute-state timeline reliable.',
                               'Per-shot xG is not declared equivalent to provider NPxG; no state weights or new model were trained.']}
    write(OUT / '接通与进球记账汇总.json', summary)
    with gzip.open(OUT / '1099场逐场联结与记账.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump(linked, stream, ensure_ascii=False, allow_nan=False)
    # Source values and all 22 original fields remain available for reuse.
    # This export includes ID-linked games even if an identity/accounting conflict
    # is present; its metadata report must accompany later use.
    with gzip.open(DATA / '1099场ID联结射门原字段.json.gz', 'wt', encoding='utf-8') as stream:
        stream.write(selected.to_json(orient='records', date_format='iso', double_precision=15))
    table = '\n'.join(f"|{s['season']}|{s['requested_matches']}|{s['id_linked_matches']}|{s['identity_agree_matches']}|{s['shot_rows']}|{s['shot_goal_count_agree_matches']}|" for s in by_season)
    report = f'''# 较早英超射门归档实际接通（2026-10-27）

本轮仅下载并读取一个 EPL 包，完成既有分钟资料的同场射门联结与进球记账；没有训练、调参，也没有查当前赛果。

来源为作者 [worldfootballR_data 的 understat_shots 发布页]({RELEASE_PAGE})，确切文件 [{ASSET}]({provenance['asset']['browser_download_url']})，{provenance['local_bytes']:,} 字节，作者更新时间 {provenance['asset']['updated_at']}。读取器 pyreadr {provenance['reader']['version']} 安装在 项目目录项目虚拟环境，缓存和临时目录也在 项目目录，没有安装系统 R。软件许可不等于底层数据许可；本轮未独立确认数据授权条款。

归档实际含 {len(frame):,} 条射门、{frame['match_id'].nunique():,} 场比赛、{len(source_seasons)} 个赛季。既有 1,099 场通过已可信的 Understat ID 联结，再核对原日期、主客队与赛季；未靠比分选择、模糊匹配或放宽日期。

|赛季|请求场次|ID联结|身份一致|射门条数|进球记账一致|
|---|---:|---:|---:|---:|---:|
{table}

合计 ID 联结 {len(linked)}/{len(manifest)}，身份一致 {summary['identity_agree_matches']}/{len(manifest)}，射门 {len(selected):,} 条。Goal {global_goal['Goal']} 条；OwnGoal {global_goal['OwnGoal']} 条按射门方的相反一侧计入得分。该记账与独立全场比分一致 {summary['shot_goal_count_agree_matches']}/{len(linked)}，归档终场总分列一致 {summary['source_final_totals_agree_matches']}/{len(linked)}。未联结 {len(missing)} 场，身份冲突 {len(identity_conflicts)} 场，记账或终场列冲突 {len(accounting_conflicts)} 场；明细保存在汇总 JSON 和逐场压缩 JSON。

源字段有年代差异：2020/21、2021/22 的射门方保存在 h_a，2022/23 保存在 home_away，二者均为明确 h/a 标签。本集合按 h_a 读取 {side_sources['h_a']:,} 条、按互补的 home_away 读取 {side_sources['home_away']:,} 条；不明或冲突 {side_missing} 条。仅统一原有明确标签，没有依据比分或球员身份补造射门方。原始字段保留不变。初次仅解析 h_a 时出现的记账冲突属于字段读取错误，已按互补字段修正并重新实跑，不能作为源数据缺失或比分冲突。

关键边界：归档有 minute、xG、match_id、Goal/OwnGoal，但没有 period、秒数、红牌或射门前比分。不能仅凭分钟恢复上半场补时与下半场起点的顺序，也不能恢复同一分钟事件先后；本轮未把作者 half_id 自动贴到射门。home_goals/away_goals 是终场总分，不能当作当时比分。一行是一脚射门，不是一分钟。射门 xG 累加未被宣称等于数据商 NPxG。

已知作者分钟包的 2022-09-03 Newcastle—Crystal Palace 进球累计异常继续保留；即使本归档该场记账通过，也不能据此宣布原分钟状态可靠。此次接通不意味着完整事件顺序已经恢复，也不构成状态修正效果证据。

复现：在项目目录使用 项目目录虚拟环境运行 `python -B work/connect_earlier_shot_archive_20261027.py`。已下载文件存在时不重复下载。原始 RDS、来源记录及原字段压缩导出均在 `data/understat/earlier_shot_archive_20261027`；输出汇总与逐场记账在本报告同目录。
'''
    (OUT / '较早射门归档接通说明.md').write_text(report, encoding='utf-8')
    print(json.dumps({key: summary[key] for key in ['requested_matches', 'id_linked_matches', 'identity_agree_matches', 'selected_shot_rows', 'by_season', 'goal_counts', 'shot_goal_count_agree_matches', 'source_final_totals_agree_matches', 'selected_duplicate_shot_ids', 'missing_state_fields']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
