"""Prepare full trusted local league history and the saved direct-1X2 cohorts.

Reads only the explicit 2022/23--2025/26 sources. It neither downloads data nor
uses old feature availability to choose the training history.
"""
from collections import Counter
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

from football_model.native_experiment import load_seasons, season_directories


ROOT = REPOSITORY_ROOT
OUT = ROOT / 'outputs/动态胜平负接入_20261025/data'
OLD = ROOT / 'outputs/直接胜平负分类研究_20261025T091329913808Z'
EARLY_AUDIT = ROOT / 'outputs/更早年份xG源核对_20261024T180024700032Z/逐季源差异.json'
RECENT_AUDIT = ROOT / 'outputs/跨联赛xG源核对_20261024T172403681186Z/逐季源差异.json'
SEASONS = ('2022/23', '2023/24', '2024/25', '2025/26')
PERIODS = ('2024-25', '2025-26')
COMPS = ('E0', 'D1', 'SP1', 'I1', 'F1')
CONTEXT = ('match_id', 'date', 'period', 'competition', 'home', 'away',
           'actual_outcome', 'actual_score', 'fit_before_date',
           'forecast_before_date', 'horizon_days')
HISTORICAL = ('match_id', 'date', 'competition', 'home', 'away', 'actual_outcome', 'actual_score')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def by_season(rows):
    return {s: {'n': len(group := [r for r in rows if r['season'] == s]),
                'by_competition': dict(Counter(r['competition'] for r in group)),
                'date_min': min((r['date'] for r in group), default=None),
                'date_max': max((r['date'] for r in group), default=None)} for s in SEASONS}


def prepare_history():
    directories = season_directories(ROOT)
    if set(directories) != set(SEASONS):
        raise ValueError('Read exactly the four authorized historical seasons')
    records, reports = load_seasons(directories)
    completeness = {}
    for season in SEASONS:
        completeness[season] = {}
        for comp in COMPS:
            local = [r for r in records if r.season == season and r.competition == comp]
            teams = {t for r in local for t in (r.home, r.away)}
            pairs = {(r.home, r.away) for r in local}
            expected_pairs = {(h, a) for h in teams for a in teams if h != a}
            completeness[season][comp] = {
                'source_rows': len(local), 'teams': len(teams),
                'expected_double_round_robin_rows': len(expected_pairs),
                'missing_home_away_pairs': [list(p) for p in sorted(expected_pairs-pairs)],
                'complete_double_round_robin': len(local) == len(expected_pairs) and pairs == expected_pairs}
    source_conflicts = {}
    # Reuse the recorded source conflict decisions from the preceding research;
    # no new date preference, outcome-based choice, or fresh source fetch.
    audits = ((EARLY_AUDIT, read(EARLY_AUDIT)['seasons']),
              (RECENT_AUDIT, read(RECENT_AUDIT)))
    for source, groups in audits:
        for key, group in groups.items():
            if not isinstance(group, dict) or 'differences' not in group:
                continue
            comp, season = key.split(':', 1)
            if comp not in COMPS or season not in SEASONS:
                continue
            for conflict in group['differences']:
                identifier = conflict['football_data_match_id']
                if not identifier:
                    raise ValueError('Unresolved historical source conflict identity')
                source_conflicts.setdefault(identifier, []).append({'audit': str(source), **conflict})
    independent = {r.match_id: r for r in records}
    if set(source_conflicts) - set(independent):
        raise ValueError('A recorded historical source conflict is absent from the selected source')
    rows, excluded = [], []
    for record in records:
        if record.season not in SEASONS or record.competition not in COMPS:
            raise ValueError('Unexpected source season or competition')
        if not date(int(record.season[:4]), 7, 1) <= record.match_date < date(int(record.season[:4])+1, 7, 1):
            raise ValueError('Fixture date is outside its declared historical season')
        h, a = record.full_time
        row = {'match_id': record.match_id, 'date': record.match_date.isoformat(),
               'competition': record.competition, 'home': record.home, 'away': record.away,
               'actual_outcome': 0 if h > a else 2 if h < a else 1,
               'actual_score': [h, a], 'season': record.season}
        reasons = []
        if record.goal_fit_exclusion:
            reasons.append(record.goal_fit_exclusion)
        if record.match_id in source_conflicts:
            reasons.append('previously_recorded_cross_source_conflict')
        if reasons:
            excluded.append({**row, 'reasons': reasons,
                             'source_conflicts': source_conflicts.get(record.match_id, [])})
        else:
            rows.append(row)
    rows.sort(key=lambda r: (r['date'], r['match_id']))
    if len({r['match_id'] for r in rows}) != len(rows):
        raise ValueError('Canonical history must have unique fixture identities')
    metadata = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'class_order': ['home', 'draw', 'away'],
        'actual_outcome_mapping': {'home': 0, 'draw': 1, 'away': 2},
        'seasons': list(SEASONS), 'competitions': list(COMPS), 'n': len(rows),
        'source_records': len(records), 'source_directories': {s: str(p) for s, p in directories.items()},
        'sources': {s: {c: {'path': v['source_path'], 'url': v['source_url']}
                       for c, v in report['leagues'].items()} for s, report in reports.items()},
        'by_season': by_season(rows),
        'raw_source_completeness': completeness,
        'excluded_n': len(excluded),
        'source_special_exclusions_n': sum(bool(r.goal_fit_exclusion) for r in records),
        'cross_source_conflict_unique_n': len(source_conflicts),
        'history_selection': 'all parsed local Football-Data fixtures in the four seasons, excluding existing source special cases and already-recorded source conflicts; no feature, target or old-model availability filter',
        'identity_policy': 'unchanged Football-Data league-local IDs and labels; no fuzzy matching or new team aliases',
        'trust_policy': 'reuse football_data.parse_row/load_directory validation and native_experiment.load_seasons invalid/duplicate refusal; retain previous conservative cross-source conflict exclusions',
        'time_semantics': 'historical source calendar dates; unknown first publication/revision times; no timezone or exact 24/72-hour claim',
        'network_requests': 0, 'current_season_results_read': False,
        'source_conflict_audits': [str(EARLY_AUDIT), str(RECENT_AUDIT)],
    }
    # Write immediately so the model run can proceed while the alignment report
    # and saved comparator copies are being prepared.
    write(OUT / 'canonical.json', {'rows': rows, 'metadata': metadata})
    write(OUT / 'source_reports.json', reports)
    write(OUT / 'excluded_history.json', {'rows': excluded, 'n': len(excluded),
          'selection': metadata['history_selection']})
    print(json.dumps({'canonical_ready': str(OUT / 'canonical.json'), 'n': len(rows),
                      'excluded_n': len(excluded), 'by_season': metadata['by_season']}, ensure_ascii=False), flush=True)
    return rows, metadata, excluded, reports


def prepare_references(canonical):
    index = {r['match_id']: r for r in canonical}
    leads, checks = {}, {}
    for lead in (1, 3):
        path = OLD / f'lead_{lead}/逐场胜平负概率.json'
        tracks = read(path)
        if 'regression_allcomp' not in tracks:
            raise ValueError('Saved reference anchor is missing')
        targets = [{k: row[k] for k in CONTEXT} for row in tracks['regression_allcomp']]
        if not targets or len({r['match_id'] for r in targets}) != len(targets):
            raise ValueError('Saved target cohort has empty or duplicate identities')
        if targets != sorted(targets, key=lambda r: (r['date'], r['match_id'])):
            raise ValueError('Saved cohort ordering differs from its chronological order')
        for row in targets:
            source = index.get(row['match_id'])
            if source is None or any(row[k] != source[k] for k in HISTORICAL):
                raise ValueError('Saved target identity/score does not match trusted full history')
            if row['period'] not in PERIODS or row['period'] != source['season'].replace('/', '-'):
                raise ValueError('Saved target period differs from historical season')
            played, fitted, forecast = map(date.fromisoformat,
                (row['date'], row['fit_before_date'], row['forecast_before_date']))
            if row['horizon_days'] != lead or not fitted <= forecast < played or (played-forecast).days != lead:
                raise ValueError('Saved target has inconsistent publication timing')
        per_track = {}
        for track, track_rows in tracks.items():
            expected = [r for r in targets if track != 'clean_blend' or r['period'] == '2025-26']
            actual = [{k: row[k] for k in CONTEXT} for row in track_rows]
            if actual != expected:
                raise ValueError(f'Saved {lead}/{track} fixture identities/timing/order do not align')
            for row in track_rows:
                p = row['probabilities']
                if (len(p) != 3 or any(not isinstance(v, (int, float)) or not math.isfinite(v)
                                      or not 0 < v < 1 for v in p)
                        or abs(math.fsum(p)-1) > 1e-12):
                    raise ValueError(f'Saved {lead}/{track} has invalid native probabilities')
            per_track[track] = {'n': len(track_rows), 'by_period': dict(Counter(r['period'] for r in track_rows)),
                                'context_and_order_equal': True, 'probabilities_native_unchanged': True}
        periods = {p: {'n': len(local := [r for r in targets if r['period'] == p]),
                       'match_ids': [r['match_id'] for r in local],
                       'by_competition': dict(Counter(r['competition'] for r in local)),
                       'parameter_dates': sorted({r['fit_before_date'] for r in local}),
                       'date_min': min(r['date'] for r in local), 'date_max': max(r['date'] for r in local)}
                   for p in PERIODS}
        leads[str(lead)] = {'targets': targets, 'references': tracks, 'periods': periods,
                            'old_probability_path': str(path)}
        checks[str(lead)] = {'n': len(targets), 'by_period': {p: v['n'] for p, v in periods.items()},
                             'tracks': per_track, 'independent_full_history_identity_score_equal': True,
                             'legal_publication_timing': True, 'target_order': 'date,match_id'}
        write(OUT / f'targets_lead{lead}.json', {'rows': targets, 'horizon_days': lead,
                                               'old_study_path': str(OLD), 'periods': periods})
        write(OUT / f'refs_lead{lead}.json', tracks)
        print(json.dumps({'references_ready': lead, 'n': len(targets),
                          'by_period': checks[str(lead)]['by_period']}, ensure_ascii=False), flush=True)
    cohorts = {l: {r['match_id']: r for r in v['targets']} for l, v in leads.items()}
    common = set(cohorts['1']) & set(cohorts['3'])
    for identifier in common:
        if any(cohorts['1'][identifier][k] != cohorts['3'][identifier][k] for k in HISTORICAL):
            raise ValueError('Horizons disagree about shared target identity or observed result')
    cross = {'common_n': len(common),
             'lead1_only_match_ids': sorted(set(cohorts['1'])-set(cohorts['3'])),
             'lead3_only_match_ids': sorted(set(cohorts['3'])-set(cohorts['1'])),
             'shared_identity_score_equal': True,
             'comparison_rule': 'retain each horizon\'s original complete cohort; do not intersect leads'}
    references = {'old_study_path': str(OLD), 'class_order': ['home', 'draw', 'away'],
                  'leads': leads, 'cross_horizon_check': cross,
                  'identity_fields': list(CONTEXT),
                  'clean_blend_rule': 'only 2025-26, using its already-saved probabilities'}
    write(OUT / 'references.json', references)
    return checks, cross


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    canonical, metadata, excluded, reports = prepare_history()
    checks, cross = prepare_references(canonical)
    coverage = read(OLD / '覆盖与不可预测.json')
    old_unavailable = {lead: {'n': len(c['unavailable']),
                             'included_in_full_training_history_n': sum(r['match_id'] in {x['match_id'] for x in canonical}
                                                                         for r in c['unavailable']),
                             'note': 'old model forecast availability does not filter the canonical history'}
                       for lead, c in coverage.items()}
    result = {'metadata': metadata, 'alignment_checks': checks, 'cross_horizon_check': cross,
              'excluded_history_n': len(excluded), 'old_unavailable_training_history_check': old_unavailable,
              'source_validation': {s: {'rejected_rows': r['rejected_rows'],
                                        'duplicate_keys': r['duplicate_keys'],
                                        'field_issues_n': len(r['field_issues'])} for s, r in reports.items()},
              'files': {'canonical': 'canonical.json', 'references': 'references.json',
                        'targets': ['targets_lead1.json', 'targets_lead3.json'],
                        'refs': ['refs_lead1.json', 'refs_lead3.json'],
                        'source_reports': 'source_reports.json', 'excluded_history': 'excluded_history.json'},
              'old_files_modified': False, 'new_data_downloaded': False, 'current_results_reviewed': False}
    write(OUT / 'preparation_report.json', result)
    lines = ['# KickScore 历史与旧结果准备报告', '',
             f"完整本地 Football-Data 原始记录 {metadata['source_records']} 场；沿用既有特殊赛及来源冲突排除 {len(excluded)} 场；可信完整历史 {len(canonical)} 场。", '',
             '历史并未按旧特征可用性或旧预测可用性筛选。常规时间结果编码为 0=主胜、1=平、2=客胜；球队身份完整沿用 Football-Data 的联赛内标识。', '',
             '|赛季|可信场次|E0|D1|SP1|I1|F1|', '|---|---:|---:|---:|---:|---:|---:|']
    for season, row in metadata['by_season'].items():
        lines.append('|'+ '|'.join([season, str(row['n']), *(str(row['by_competition'].get(c, 0)) for c in COMPS)])+'|')
    lines.extend(['', '原始数据及下载来源清单来自 native_experiment.season_directories；解析和无效/重复检查复用 football_data.load_directory / native_experiment.load_seasons。逐文件本地路径与原始 URL 见 canonical.json 的 metadata.sources、完整解析报告见 source_reports.json。', '',
                  f"既有特殊赛排除 {metadata['source_special_exclusions_n']} 场；跨源冲突涉及 {metadata['cross_source_conflict_unique_n']} 个唯一比赛 ID，与特殊赛有重合。未选择更有利的新日期规则，未重新解释比分；详细排除原因、原来源差异和已存核对报告位置见 excluded_history.json。", '',
                  '20个联赛赛季原始记录均覆盖联赛内每个有向主客队配对一次，确认双循环赛程完整（2022/23法甲20队，其后三季法甲18队，德甲18队，其余20队）；排除后训练历史不再是完整双循环。仅比分/赛果用于训练。CSV 的半场或赔率缺失继续保留为来源字段问题，不用于筛除可用全场赛果；所有季无拒绝解析行和重复主键。', '',
                  '|提前天数|2024-25共同目标|2025-26共同目标|轨道数|', '|---|---:|---:|---:|'])
    for lead, check in checks.items():
        lines.append(f"|{lead}|{check['by_period']['2024-25']}|{check['by_period']['2025-26']}|{len(check['tracks'])}|")
    lines.extend(['', '每条旧轨道完整复制并逐行核对身份、比分、日期、拟合截止日、发布截止日和提前天数；各提前天数内目标顺序一致。clean_blend 只在较晚赛季存在，未补造较早概率；原始概率均为正、有限且归一，未剪裁、平滑或重算。', '',
                  'targets_lead1/3.json 的 rows 提供共同目标及原始时序；refs_lead1/3.json 直接保持旧文件的 {track: rows} 格式。references.json 汇总每个 lead 的 targets、references、periods（含完整 match_ids）。', '',
                  f"两个提前天数共用 {cross['common_n']} 个比赛身份；仍分别保留原完整队列，未取交集丢弃目标。", '',
                  '日期语义限于原始历史日历日期，无法证明首次发表或后续修订时点，提前1/3天不声称精确24/72小时。本次无网络请求、无新下载、无旧文件修改，未读取或复核2026/27赛果。', ''])
    (OUT / '数据准备与对齐说明.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'report_ready': str(OUT / 'preparation_report.json'),
                      'cross_horizon': cross, 'old_unavailable': old_unavailable}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
