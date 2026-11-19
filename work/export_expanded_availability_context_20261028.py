"""Reuse dated FPL observations for the expanded historical cohort.

This is a context-only join. It does not fit effects, infer clinical health,
invent earlier snapshots, or read target scores/outcomes into context fields.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from intake_fpl_availability_20261026 import DATA, MEASUREMENTS, archive_index, utc

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
BASE = ROOT/'outputs/协作预测与大样本方案_20261028/扩样训练/扩样合并特征.json.gz'
PREVIOUS = ROOT/'outputs/FPL历史可用性接入_20261026/逐场赛前状态特征.json.gz'
EXPOSURE = ROOT/'outputs/FPL可用性变化与参与接入_20261026/逐场变化参与特征.json.gz'
DEST = ROOT/'data/historical_context_20261028/availability_sources'
META_KEYS = ('match_id', 'date', 'period', 'competition', 'home', 'away',
             'fit_before_date', 'forecast_before_date', 'horizon_days')


def read(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')


def context_for(row, prior, exposure, snapshots, earliest):
    result = {key: row[key] for key in META_KEYS}
    ident = row['match_id']
    old = prior.get(ident)
    delta = exposure.get(ident)
    cutoff = utc(row['forecast_before_date']+'T00:00:00Z')
    observation = None
    reason = None
    if row['competition'] != 'E0':
        reason = 'source_scope_excludes_competition'
    elif old is None:
        reason = ('forecast_precedes_archive_start' if cutoff <= earliest
                  else 'no_existing_snapshot_join_for_this_target')
    elif any(row[key] != old[key] for key in META_KEYS):
        raise ValueError('Expanded and existing availability fixture metadata differ')
    else:
        archived = old['archive_observation']
        if archived is None:
            reason = 'existing_join_has_no_archive_observation'
        elif archived.get('status') == 'failed':
            reason = 'existing_snapshot_download_failed'
        else:
            observation = snapshots[archived['path']]
            if (utc(observation['filename_observed_at']) >= cutoff or
                    utc(observation['repository_recorded_at']) >= cutoff or
                    utc(observation['availability_proxy_at']) >= cutoff):
                raise ValueError('Snapshot observation proxy must be strictly before forecast cutoff')
            if any(archived.get(k) != observation.get(k) for k in
                   ('path', 'commit', 'filename_observed_at', 'repository_recorded_at',
                    'availability_proxy_at', 'source_url')):
                raise ValueError('Joined snapshot differs from its existing source manifest')
    result['archive_observation'] = (None if observation is None else {
        key: observation.get(key) for key in ('path', 'commit', 'filename_observed_at',
        'repository_recorded_at', 'availability_proxy_at', 'source_url', 'received_at', 'local_path')})
    result['availability_proxy_interpretation'] = (
        'max(dated cache filename, first-added Git commit time); retrospective source-version '
        'proxy, not independent evidence of push time or first medical announcement')
    result['missing_reason'] = reason
    result['teams'] = []
    for side, team in enumerate((row['home'], row['away'])):
        found = None if observation is None else observation['teams'].get(team)
        measures = {key: None for key in MEASUREMENTS}
        players = []
        if found and not found['empty_roster']:
            measures.update(found['measurements'])
            measures['snapshot_age_hours'] = old['snapshot_age_hours']
            for player in found['flagged_players']:
                item = deepcopy(player)
                news_at = item.get('news_added')
                item['news_added_after_snapshot_proxy'] = bool(news_at and
                    utc(news_at) > utc(observation['availability_proxy_at']))
                item['status_interpretation'] = (
                    'Raw FPL source state; not a clinical diagnosis, confirmed absence, '
                    'starting selection, expected minutes or a fitted ability effect')
                players.append(item)
        result['teams'].append({'team':team, 'side':'home' if side == 0 else 'away',
            'observed': bool(found and not found['empty_roster']),
            'source_name': None if found is None else found['source_name'],
            'source_team_id': None if found is None else found['source_team_id'],
            'source_team_code': None if found is None else found['source_team_code'],
            'measurements':measures,
            'raw_status_counts':None if found is None else found['raw_status_counts'],
            'flagged_players':players,
            'missing_reason': reason if found is None else None})
    result['availability_count_features'] = []
    for side in range(2):
        own = result['teams'][side]['measurements']
        opponent = result['teams'][1-side]['measurements']
        features = {'fpl_own_'+key:own[key] for key in MEASUREMENTS}
        features.update({'fpl_opponent_'+key:opponent[key] for key in MEASUREMENTS})
        if observation is not None:
            if any(old['side_features'][side][key] != value for key,value in features.items()):
                raise ValueError('Existing availability count features changed')
        result['availability_count_features'].append(features)
    result['existing_participation_change_features'] = None
    result['existing_participation_diagnostics'] = None
    if delta is not None:
        if any(row[key] != delta[key] for key in META_KEYS):
            raise ValueError('Existing participation context identifies another cutoff')
        result['existing_participation_change_features'] = [
            {key:value for key,value in side.items() if key not in row['fresh_performance_features'][i]}
            for i,side in enumerate(delta['side_features'])]
        result['existing_participation_diagnostics'] = delta['availability_diagnostics']
    return result


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    base = read(BASE)['rows']
    old = read(PREVIOUS)['rows']
    previous = {row['match_id']:row for row in old}
    delta = {row['match_id']:row for row in read(EXPOSURE)['rows']}
    snapshot_list = read(DATA/'selected_snapshot_manifest.json')['observations']
    snapshots = {row['path']:row for row in snapshot_list}
    index = archive_index()
    earliest = min(utc(row['availability_proxy_at']) for row in index)
    rows = [context_for(row, previous, delta, snapshots, earliest) for row in base]
    if len({row['match_id'] for row in rows}) != len(base):
        raise ValueError('Expanded cohort identifiers are not unique')
    if set(previous)-{row['match_id'] for row in rows}:
        raise ValueError('An existing available fixture was lost')
    known_rows = [r for r in rows if all(t['observed'] for t in r['teams'])]
    if len(known_rows) != len(old):
        raise ValueError('Expanded join silently altered existing availability coverage')
    for position in (next(i for i,r in enumerate(base) if r['match_id'] in previous),
                     next(i for i,r in enumerate(base) if r['competition']=='E0' and r['period']=='2021-22'),
                     next(i for i,r in enumerate(base) if r['competition']!='E0')):
        changed = dict(base[position], actual_score=[77,78], actual_outcome=0)
        if context_for(changed,previous,delta,snapshots,earliest) != rows[position]:
            raise ValueError('Target outcome affected exported context')
    encoded_keys = set(rows[0])
    if {'actual_score','actual_outcome'} & encoded_keys:
        raise ValueError('Context export contains target labels')
    document = {'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'feature_source':str(BASE), 'previous_count_source':str(PREVIOUS),
        'previous_participation_source':str(EXPOSURE),
        'scope':'Context-only exact-ID join; English FPL only, no new effects, fit or probability updates',
        'rows':rows}
    path = DEST/'8753场历史可用性上下文.json.gz'
    with gzip.open(path,'wt',encoding='utf-8') as handle:
        json.dump(document,handle,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    by_period = {}
    for period in sorted({r['period'] for r in rows}):
        group = [r for r in rows if r['period'] == period]
        by_period[period] = {'n':len(group),
            'epl_n':sum(r['competition']=='E0' for r in group),
            'both_teams_observed':sum(all(t['observed'] for t in r['teams']) for r in group),
            'missing_reasons':dict(Counter(r['missing_reason'] for r in group if r['missing_reason'])),
            'existing_participation_context_n':sum(r['existing_participation_change_features'] is not None for r in group)}
    flagged = [p for r in rows for t in r['teams'] for p in t['flagged_players']]
    summary = {'created_at_utc':document['created_at_utc'],
        'expanded_fixture_n':len(base),'original_fixture_n':6967,'added_fixture_n':1786,
        'all_competitions':dict(Counter(r['competition'] for r in rows)),
        'availability_both_teams_observed_n':len(known_rows),
        'availability_added_coverage_n':0,
        'available_original_count_features_preserved':True,
        'source_snapshots_reused':len({r['archive_observation']['path'] for r in known_rows}),
        'earliest_archive_availability_proxy':earliest.isoformat(),
        'archive_index_n':len(index),'by_period':by_period,
        'flagged_player_contexts':len(flagged),
        'flagged_status_counts':dict(Counter(p['status'] for p in flagged)),
        'news_after_snapshot_proxy_n':sum(p['news_added_after_snapshot_proxy'] for p in flagged),
        'count_features_per_side':14, 'participation_delta_features_per_side':22,
        'target_mutation_context_unchanged':True,'target_labels_exported':False,
        'new_model_fits':0,'new_raw_snapshot_downloads':0,'current_results_queried':False,
        'current_forecasts_changed':False,
        'source_repository':'https://github.com/Randdalf/fplcache',
        'primary_source_review': {'repository_readme':'https://github.com/Randdalf/fplcache#readme',
            'license_url':'https://raw.githubusercontent.com/Randdalf/fplcache/main/LICENSE',
            'review_completed_at_utc':document['created_at_utc'],
            'time_basis':'Report generation after the finite primary-page review; not an invented per-request retrieval second',
            'software_license':'Unlicense; does not itself license upstream FPL data'},
        'limits':['The 2021/22 added cohort predates this archive; do not fill with 2022/23 or season-end states.',
            'Other four leagues are outside FPL scope, retained as unknown rather than healthy/zero.',
            'Commit timestamps are retrospective version proxies, not independently certified push or announcement times.',
            'Status u includes loan/transfer/other unavailability, n remains unknown; non-a is not synonymous with injury.',
            'FPL chance is a source availability signal, not team win probability or verified calibration.',
            'Repeated flagged-player contexts are not independent people or additional fixtures.',
            'Observed minute share is participation, not skill, replacement ability or expected future minutes.',
            'No target current lineup or actual same-match minutes are used.']}
    write(DEST/'覆盖与复用实测.json',summary)
    examples = []
    for period in ('2021-22','2022-23','2023-24','2024-25','2025-26'):
        record = next(r for r in rows if r['competition']=='E0' and r['period']==period)
        examples.append(record)
    write(DEST/'按季上下文实例.json',{'examples':examples})
    lines=['# 扩样后的历史球员可用性上下文','','本轮复用已接通的Randdalf/fplcache归档和既存参与变化资料，没有重复下载历史球员包、没有拟合或修改概率。', '',
        '|赛季|全部比赛|英超比赛|双队状态可观测|','|---|---:|---:|---:|']
    for period,c in by_period.items():
        lines.append(f"|{period}|{c['n']}|{c['epl_n']}|{c['both_teams_observed']}|")
    lines += ['', f'8,753场完整导出，其中{len(known_rows)}场可复用既有状态，459份缓存；新增2021/22共1,786场没有新增可用性覆盖。档案最早代理为{earliest.isoformat()}，晚于这一整季；其余四联赛超出FPL范围。', '',
        '每场保留相同forecast_before_date，只有文件名时间与首次added提交时间均严格早于该截点才提供状态。每队包含原始状态计数、带news/news_added和身份编号的逐人非a资料、14项自己／对手计数特征；已有22项参与变化特征单独附带，未知保持None。输出不包含目标比分和胜负。', '',
        'status a仅代表FPL源状态，不证明临床健康；i/d/s/u/n不混为伤病。u包括转会和外租，n保留其他未知。chance不用于球队胜率。新闻更新时间不是缺阵起点，个人分钟份额不是个人能力或替代损失。', '',
        '软件来源[Randdalf/fplcache](https://github.com/Randdalf/fplcache)说明每天四次缓存FPL bootstrap；[LICENSE](https://raw.githubusercontent.com/Randdalf/fplcache/main/LICENSE)为Unlicense。底层FPL数据的使用权不能仅由软件许可证推出，本轮按已授权个人研究保留。', '',
        '本轮曾先核过vaastav旧版本字段，但发现本地已有更完整归档后停止该路线，没有另下载重复原始球员包。Git记录只作为历史版本可见性代理，不提供原新闻首次发布或Git push时刻的独立证明。', '',
        f'检查：旧可用计数特征保持一致；全部{len(known_rows)}个可用目标时点严格向前；目标标签改动不影响上下文。非a逐人情境{len(flagged)}条不是独立训练比赛；news时间晚于快照代理{summary["news_after_snapshot_proxy_n"]}条。', '',
        '停止节点：完成一轮精确联结、完整缺失覆盖和按季实例即交付。不按缺失状态训练默认零伤停，不因覆盖未扩张而重复采集同类项目。']
    (DEST/'复用结果与边界.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('expanded_fixture_n','availability_both_teams_observed_n','availability_added_coverage_n','by_period','news_after_snapshot_proxy_n')},ensure_ascii=False))


if __name__ == '__main__':
    main()
