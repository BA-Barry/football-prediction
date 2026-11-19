"""Reuse historical schedule inputs for the expanded cohort, without fitting.

The export contains no target scores or future fixture pressure. Source event
dates can precede the forecast cutoff while the archived data was retrieved
later; this is explicitly a reconstructed historical context, not an original
publication-time replay.
"""
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import gzip
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import time

from football_model.all_competition_features import AllCompetitionFeatures, WINDOWS

ROOT = REPOSITORY_ROOT
SOURCE = ROOT/'outputs/协作预测与大样本方案_20261028/扩样训练/扩样合并特征.json.gz'
SCHEDULE = ROOT/'outputs/全赛事赛程可用性_20261025'
OUT = ROOT/'data/historical_context_20261028/schedule_sources'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n',encoding='utf-8')


def raw_gap(log_gap):
    return round(math.expm1(log_gap)) if log_gap is not None else None


def numeric_export(rows, created_at):
    """A small ID-linked training input; detailed references stay in the export."""
    compact = []
    for row in rows:
        features = [{k:v for k,v in side.items() if k != 'allcomp_log1p_forecast_lead_days'}
                    for side in row['schedule_feature_vectors']]
        if any(len(side) != 40 for side in features):
            raise ValueError('Expected the existing forty incremental schedule features')
        unknown = {}
        for side in ('home', 'away'):
            context = row[side+'_context']
            reasons = []
            if not context['scope_contains_cutoff']:
                reasons.append('requested_member_scope_does_not_cover_cutoff')
            for n in WINDOWS:
                if not context['windows'][str(n)]['usable']:
                    reasons.append(f'{n}d_event_count_unusable_or_uncovered')
            if context['last_usable_observed_kickoff_day_to_cutoff_days'] is None:
                reasons.append('no_usable_last_observed_formal_event_gap')
            reasons.append('original_future_schedule_announcements_not_archived')
            unknown[side] = reasons
        compact.append({'match_id':row['match_id'], 'forecast_cutoff':row['forecast_cutoff'],
            'context_numeric_side_features':features, 'context_unknowns':unknown})
    document = {'created_at_utc':created_at,
        'reference_file':str(OUT/'8753场历史全赛事上下文.json.gz'),
        'reference_join_key':'match_id', 'source_first_publication_reconstructed':False,
        'actual_counts_are_null_when_uncovered':True,
        'binary_coverage_flags_are_not_match_counts':True,
        'context_feature_count_per_side':40, 'rows':compact}
    with gzip.open(OUT/'8753场赛程数值特征.json.gz', 'wt', encoding='utf-8') as handle:
        json.dump(document,handle,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    return len(compact)


def side_context(features, metadata, lead_days):
    windows = {str(n): {'formal_match_count': features[f'own_allcomp_formal_count_{n}d'],
        'nonleague_match_count': features[f'own_allcomp_nonleague_count_{n}d'],
        'unresolved_event_count': features[f'own_allcomp_unresolved_count_{n}d'],
        'usable': bool(features[f'own_allcomp_window_usable_{n}d']),
        'event_ids': metadata['windows'][str(n)]['selected_event_ids'],
        'event_dates': metadata['windows'][str(n)]['selected_event_dates'],
        'unresolved_events': metadata['windows'][str(n)]['excluded_or_unresolved_events']}
        for n in WINDOWS}
    gap = raw_gap(features['own_allcomp_last_log1p_gap_to_cutoff_days'])
    return {'team': metadata['team'], 'source_team_id':metadata['source_team_id'],
        'requested_member_scope': metadata['contiguous_member_scope'],
        'scope_contains_cutoff': bool(features['own_allcomp_scope_contains_cutoff']),
        'windows':windows, 'last_completed_event':metadata['last_completed_event'],
        'last_usable_observed_kickoff_day_to_cutoff_days':gap,
        'last_usable_observed_kickoff_day_to_target_days':gap+lead_days if gap is not None else None,
        'gap_interpretation':'Calendar kickoff-day gaps, not final-whistle recovery, actual minutes or fatigue.',
        'previous3_events': metadata['previous3_events'],
        'previous3_usable': metadata['previous3_usable'],
        'planned_fixtures_between_cutoff_and_target':None,
        'future_fixture_pressure':None,
        'planned_information_reason':'Original announcement and revision times not archived; actual later events are not backfilled as known plans.'}


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(SOURCE, 'rt', encoding='utf-8') as handle:
        original = json.load(handle)['rows']
    if len(original) != 8753 or len({r['match_id'] for r in original}) != 8753:
        raise ValueError('Expected the existing unique expanded cohort')
    builder = AllCompetitionFeatures.from_directory(SCHEDULE)
    inventory = read(SCHEDULE/'四季覆盖汇总.json')
    rows, per_group, examples = [], defaultdict(Counter), {}
    old_comparisons, old_differences = 0, []
    reference_n = 0
    for row in original:
        cutoff, target = date.fromisoformat(row['forecast_before_date']), date.fromisoformat(row['date'])
        if (target-cutoff).days != 3:
            raise ValueError('This export preserves the existing lead-three context')
        context = builder.for_match(home=row['home'], away=row['away'], competition=row['competition'],
            cutoff=cutoff, match_date=target)
        features = [context['home_features'], context['away_features']]
        sides = {side:side_context(features[i], context['metadata']['history'][side], 3)
                 for i,side in enumerate(('home', 'away'))}
        for side, meta in context['metadata']['history'].items():
            refs = list(meta['previous3_events'])
            if meta['last_completed_event']:
                refs.append(meta['last_completed_event'])
            for window in meta['windows'].values():
                refs.extend({'event_id':eid, 'date':day} for eid,day in zip(
                    window['selected_event_ids'], window['selected_event_dates'], strict=True))
                refs.extend(window['excluded_or_unresolved_events'])
            if any(date.fromisoformat(r['date']) >= cutoff for r in refs):
                raise ValueError('A schedule observation reaches its own forecast cutoff')
            reference_n += len(refs)
            if not sides[side]['scope_contains_cutoff']:
                for n in WINDOWS:
                    if sides[side]['windows'][str(n)]['formal_match_count'] is not None:
                        raise ValueError('Unknown coverage was turned into a zero event count')
        available_both = all(sides[s]['scope_contains_cutoff'] for s in sides)
        both_usable = {str(n):all(sides[s]['windows'][str(n)]['usable'] for s in sides) for n in WINDOWS}
        output = {'match_id':row['match_id'], 'date':row['date'], 'period':row['period'],
            'competition':row['competition'], 'home':row['home'], 'away':row['away'],
            'forecast_cutoff':cutoff.isoformat(), 'parameter_cutoff':row['fit_before_date'],
            'horizon_days':3, 'schedule_feature_vectors':features, 'home_context':sides['home'],
            'away_context':sides['away'], 'both_member_scopes_cover_cutoff':available_both,
            'both_window_usable':both_usable, 'source_first_publication_reconstructed':False}
        rows.append(output)
        for key in ('all',f"season:{row['period']}",f"league:{row['competition']}",
                    f"season_league:{row['period']}:{row['competition']}"):
            counts = per_group[key]
            counts['fixtures'] += 1
            counts['team_observations'] += 2
            counts['both_member_scopes_cover_cutoff'] += available_both
            counts['side_member_scopes_cover_cutoff'] += sum(sides[s]['scope_contains_cutoff'] for s in sides)
            for n in WINDOWS:
                counts[f'both_{n}d_usable'] += both_usable[str(n)]
                counts[f'side_{n}d_usable'] += sum(sides[s]['windows'][str(n)]['usable'] for s in sides)
                counts[f'side_{n}d_unresolved'] += sum(bool(sides[s]['windows'][str(n)]['unresolved_events']) for s in sides)
                counts[f'side_{n}d_observed_nonleague_present'] += sum(
                    sides[s]['windows'][str(n)]['usable'] and sides[s]['windows'][str(n)]['nonleague_match_count'] > 0
                    for s in sides)
            counts['both_last_day_gap_available'] += all(sides[s]['last_usable_observed_kickoff_day_to_cutoff_days'] is not None for s in sides)
            counts['side_previous3_usable'] += sum(sides[s]['previous3_usable'] for s in sides)
        if 'fresh_allcomp_features' in row:
            for i in range(2):
                for key,value in features[i].items():
                    if key == 'allcomp_log1p_forecast_lead_days':
                        continue
                    old_comparisons += 1
                    old = row['fresh_allcomp_features'][i][key]
                    if old != value:
                        old_differences.append({'match_id':row['match_id'],'side':i,'field':key,
                            'previous':old,'new':value})
        if not available_both:
            examples.setdefault('unknown_earlier_season',output)
        if available_both and all(both_usable.values()) and any(
                sides[s]['windows']['7']['nonleague_match_count'] > 0 for s in sides):
            examples.setdefault('recent_nonleague_event',output)
        if any(sides[s]['windows']['28']['unresolved_events'] for s in sides):
            examples.setdefault('partial_or_unresolved_event',output)
    payload = {'created_at_utc':datetime.now(timezone.utc).isoformat(), 'source_feature_file':str(SOURCE),
        'source_schedule_directory':str(SCHEDULE), 'class':'historical_date_reconstruction_not_publication_replay',
        'observation_unit':'one row per existing fixture; home and away views are not additional matches',
        'target_scores_exported':False, 'rows':rows}
    with gzip.open(OUT/'8753场历史全赛事上下文.json.gz', 'wt',encoding='utf-8') as handle:
        json.dump(payload,handle,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    numeric_export(rows,payload['created_at_utc'])
    write(OUT/'实例.json',examples)
    summary = {'created_at_utc':payload['created_at_utc'], 'source_feature_file':str(SOURCE),
        'source_schedule_directory':str(SCHEDULE), 'source_metadata':builder.source_metadata,
        'existing_source_acquisition_min_utc':inventory['source_retrieval_time_utc_min'],
        'existing_source_acquisition_max_utc':inventory['source_retrieval_time_utc_max'],
        'source_inventory_totals':inventory['totals'], 'counts':dict(per_group),
        'exported_schedule_features_per_side':len(rows[0]['schedule_feature_vectors'][0]),
        'new_fields_when_joined_with_121':40, 'original_schedule_field_comparisons':old_comparisons,
        'original_schedule_field_differences':old_differences,
        'chronology_reference_checks':reference_n, 'chronology_violation_n':0,
        'target_scores_exported':False, 'future_plans_reconstructed':False,
        'original_schedule_announcement_time_known':False, 'source_revision_history_known':False,
        'all_competition_completeness_certified':False, 'new_network_requests':0,
        'new_models_fitted':0, 'current_forecasts_modified':False,
        'elapsed_seconds':time.perf_counter()-start}
    write(OUT/'覆盖与复用测量.json',summary)
    table=[]
    for period in sorted({r['period'] for r in rows}):
        c=per_group['season:'+period]
        table.append(f"|{period}|{c['fixtures']}|{c['both_member_scopes_cover_cutoff']}|{c['both_7d_usable']}|{c['both_14d_usable']}|{c['both_28d_usable']}|{c['both_last_day_gap_available']}|")
    report = f'''# 历史全赛事上下文复用实测

本轮不联网、不训练、不查询新赛果。直接复用2026-10-25已下载的ESPN历史目录、逐队全赛事及原`AllCompetitionFeatures`实现，向扩样后同一8,753场比赛导出赛前3日截止的日期／计数上下文。逐场输出不包含目标比分或胜负。

## 实际覆盖

|赛季|目标场次|双方请求身份范围含截止日|双方7日可用|双方14日可用|双方28日可用|双方末次开球日间隔可用|
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

分母始终是8,753场、17,506个球队观察，按联赛／赛季交叉分母在`覆盖与复用测量.json`。2021/22无相应全赛事历史包，实际计数保持null；可用性指示为0仅表示未覆盖，不表示零负荷。请求身份范围和窗口完整性不等于独立认证每项杯赛都完整。

现有四季来源共{inventory['totals']['global_unique_event_ids']}个去重事件，其中{inventory['totals']['unique_competitive_completed_events']}个来源标为正式且已完成。覆盖来自当季五联赛身份386俱乐部赛季，不保证球队降级后所在较低联赛的全年赛程。来源取得时间{inventory['source_retrieval_time_utc_min']}至{inventory['source_retrieval_time_utc_max']}；历史原始公告与改期首次发布时间未保存，所以本产物是**按事件发生日期重建的历史观察**，不是发布时点的完全回放。

## 导出的字段和限制

- 使用现有41项每侧赛程向量：自身／对手各20项，加已知预测提前量。与原121项合并时，提前量已有，净新增40项，即原161项语义；本轮没有新增算法或训练。
- 7/14/28日正式赛事计数、其中非联赛计数、已知不完整事件计数、窗口可用性；带逐条事件ID和发生日期。
- 最后可用已观察正式赛事开球**日期**距截止日及距目标日的日历间隔；最近3场事件及间隔。不是终场恢复时长，也不是球员分钟、旅行或因果疲劳。
- 已知行政判罚、部分／复赛和来源完成状态冲突仍排除／标缺。窗口中存在未决事件时相关计数null，不将其静默忽略成正常零值。
- 没有历史原公告／改期快照，所以“截止至目标之间已安排比赛”和未来压力为null；**不会读取后来已经踢完的赛事再假装它在历史截止时已公布**。

对已有6,967行的赛程分量逐字段比较{old_comparisons:,}次，差异{len(old_differences)}项；旧行本身未改动。所有导出历史引用{reference_n:,}次均严格早于各自forecast cutoff。此处验证复用一致性和日期语义，不证明预测性能提高。

## 可直接调用

`8753场历史全赛事上下文.json.gz`的`rows`按`match_id`与既有丰富扩样特征联结；每场的`forecast_cutoff`保持原`forecast_before_date`，不是目标比赛日期。`schedule_feature_vectors`为主／客两视图；`home_context`、`away_context`保留解释用事件引用。原161特征模型已接过这批较晚赛季资料，不能把本轮导出再算成新信息发现或新提准。

实例见`实例.json`：无覆盖早季、可见非联赛负荷，以及已知部分／未决比赛导致窗口不可用。若后来补2021/22真实资料，可经同一适配器产生该季分量；在此之前保持缺失，不靠联赛赛程冒充全赛事。

原来源和采集说明见[四季全赛事接通与限制](../../outputs/全赛事赛程可用性_20261025/四季全赛事接通与使用限制.md)。实际来源例：[ESPN历史参赛身份](https://sports.core.api.espn.com/v2/sports/soccer/leagues/eng.1/seasons/2024/teams?limit=1000)。所有原始请求地址、取得时间与逐队赛事保留在原目录，本轮未重新下载。

停止点：一次覆盖导出和原字段一致性测量完成；新增联网请求0、模型训练0、当前预测更改0。
'''
    (OUT/'历史赛程覆盖与接入说明.md').write_text(report,encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('counts','original_schedule_field_comparisons',
        'original_schedule_field_differences','elapsed_seconds')},ensure_ascii=False))


if __name__ == '__main__':
    main()
