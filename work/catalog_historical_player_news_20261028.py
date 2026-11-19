"""Catalogue exact archived FPL status/news text without semantic labels.

Reads only existing source snapshots and label-free contextual rows. Repeated
player observations are references, not independent football matches.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
import lzma
from pathlib import Path
import time

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
CONTEXT = ROOT/'data/historical_context_20261028/availability_sources/8753场历史可用性上下文.json.gz'
ARCHIVE = ROOT/'data/fpl_availability_archive'
MANIFEST = ARCHIVE/'selected_snapshot_manifest.json'
OUT = ROOT/'data/personnel_semantics_20261028'


def read(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    if path.suffix == '.gz':
        with gzip.open(path, 'wt', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def instant(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(timezone.utc)


def empty_type(news):
    if news is None:
        return 'null'
    if news == '':
        return 'empty_string'
    if not news.strip():
        return 'whitespace_only'
    return 'nonempty'


def sort_key(key):
    status, news = key
    return (status is not None, '' if status is None else status,
            news is not None, '' if news is None else news)


def main():
    tick = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    contexts = read(CONTEXT)['rows']
    if len(contexts) != 8753 or len({r['match_id'] for r in contexts}) != 8753:
        raise ValueError('Expected the complete unique expanded context cohort')
    selected = [r for r in contexts if r['archive_observation'] is not None]
    # Do not read any target scores or outcomes in this cataloguing path.
    if any(r['competition'] != 'E0' for r in selected):
        raise ValueError('FPL snapshots are scoped to EPL targets only')
    wanted = defaultdict(list)
    for row in selected:
        source = row['archive_observation']
        cutoff = instant(row['forecast_before_date']+'T00:00:00+00:00')
        if instant(source['availability_proxy_at']) >= cutoff:
            raise ValueError('Archived snapshot is not earlier than its target context cutoff')
        wanted[source['path']].append(row)
    manifest = {r['path']:r for r in read(MANIFEST)['observations']}
    if len(manifest) != 459 or set(wanted) - manifest.keys():
        raise ValueError('Existing selected snapshot manifest is incomplete or changed')
    sources = sorted((manifest[path] for path in wanted),
        key=lambda r:(instant(r['availability_proxy_at']), r['path']))
    raw_refs, fixture_refs, snapshot_index = [], [], []
    stats, target_stats = Counter(), Counter()
    catalog = {}
    source_field_missing = Counter()
    future_news_metadata = []
    existing_flagged_n = sum(len(t['flagged_players']) for r in selected for t in r['teams'])
    for snap_number, meta in enumerate(sources, 1):
        snapshot_id = f'snapshot_{snap_number:04d}'
        local = ARCHIVE/'raw'/meta['path'].removeprefix('cache/')
        if not local.is_file():
            raise ValueError(f'Existing source snapshot is missing: {local}')
        payload = json.loads(lzma.decompress(local.read_bytes()))
        teams = {r['id']:r for r in payload['teams']}
        if len(teams) != len(payload['teams']):
            raise ValueError('Duplicate source team ID')
        source_players = sorted(payload['elements'], key=lambda p:p['id'])
        if len({p['id'] for p in source_players}) != len(source_players):
            raise ValueError('Duplicate player ID in one archived snapshot')
        by_team = defaultdict(list)
        player_index = {}
        proxy = instant(meta['availability_proxy_at'])
        for player in source_players:
            for field in ('id','code','team','status','news','news_added'):
                source_field_missing[field] += field not in player
            status, news = player.get('status'), player.get('news')
            if status is not None and not isinstance(status, str):
                raise ValueError('Status must be source text or null')
            if news is not None and not isinstance(news, str):
                raise ValueError('News must be source text or null')
            if player['team'] not in teams:
                raise ValueError('Player points to an absent snapshot team ID')
            key = (status, news)
            if key not in catalog:
                catalog[key] = {'raw_status':status, 'raw_news':news,
                    'news_empty_type':empty_type(news), 'snapshot_ids':set(), 'player_codes':set(),
                    'first_snapshot_proxy_at':meta['availability_proxy_at'],
                    'last_snapshot_proxy_at':meta['availability_proxy_at'],
                    'source_player_observation_count':0, 'target_player_context_count':0,
                    'target_non_a_player_context_count':0, 'target_match_ids':set()}
            item = catalog[key]
            item['snapshot_ids'].add(snapshot_id)
            item['player_codes'].add(player.get('code'))
            item['last_snapshot_proxy_at'] = meta['availability_proxy_at']
            item['source_player_observation_count'] += 1
            team = teams[player['team']]
            observation_id = f'player_observation_{len(raw_refs)+1:07d}'
            ref = {'observation_id':observation_id, 'snapshot_id':snapshot_id,
                '_news_key':key, 'source_player_id':player['id'], 'source_player_code':player.get('code'),
                'source_team_id':team['id'], 'source_team_code':team.get('code'),
                'source_team_name':team.get('name'), 'first_name':player.get('first_name'),
                'second_name':player.get('second_name'), 'web_name':player.get('web_name'),
                'element_type':player.get('element_type'), 'news_added':player.get('news_added'),
                'chance_of_playing_this_round':player.get('chance_of_playing_this_round'),
                'chance_of_playing_next_round':player.get('chance_of_playing_next_round')}
            raw_refs.append(ref)
            by_team[player['team']].append(ref)
            player_index[player['id']] = player
            stats[status] += 1
            if player.get('news_added') is not None and instant(player['news_added']) > proxy:
                future_news_metadata.append({'observation_id':observation_id,
                    'news_added':player['news_added'],'snapshot_proxy_at':meta['availability_proxy_at']})
        for target in wanted[meta['path']]:
            for side, target_team in enumerate(target['teams']):
                if not target_team['observed']:
                    raise ValueError('Expected both target teams to be present for selected snapshots')
                source_team = teams[target_team['source_team_id']]
                if source_team.get('code') != target_team['source_team_code']:
                    raise ValueError('Target team source code changed at the same snapshot')
                # Preserve and confirm the old non-a subset, then additionally
                # retain source-a observations (including source-a nonempty news).
                original_players = {p['id']:p for p in target_team['flagged_players']}
                non_a_ids = {r['source_player_id'] for r in by_team[source_team['id']]
                             if r['_news_key'][0] != 'a'}
                if set(original_players) != non_a_ids:
                    raise ValueError('Original non-a player cohort differs from the raw snapshot')
                for identifier, old in original_players.items():
                    raw = player_index[identifier]
                    if any(old.get(k) != raw.get(k) for k in ('code','status','news','news_added')):
                        raise ValueError('An old non-a source field was changed')
                for ref in by_team[source_team['id']]:
                    key = ref['_news_key']
                    item = catalog[key]
                    item['target_player_context_count'] += 1
                    item['target_non_a_player_context_count'] += key[0] != 'a'
                    item['target_match_ids'].add(target['match_id'])
                    target_stats[key[0]] += 1
                    fixture_refs.append({'match_id':target['match_id'],
                        'forecast_before_date':target['forecast_before_date'], 'period':target['period'],
                        'team':target_team['team'], 'side':'home' if side==0 else 'away',
                        'observation_id':ref['observation_id'], 'snapshot_id':snapshot_id,
                        'source_player_code':ref['source_player_code'], 'source_player_id':ref['source_player_id'],
                        '_news_key':key})
        snapshot_index.append({'snapshot_id':snapshot_id,
            **{k:meta.get(k) for k in ('path','commit','filename_observed_at','repository_recorded_at',
                'availability_proxy_at','source_url','received_at')},
            'local_path':str(local), 'raw_player_count':len(source_players),
            'target_match_ids':sorted(r['match_id'] for r in wanted[meta['path']]),
            'target_periods':sorted({r['period'] for r in wanted[meta['path']]})})
        if snap_number%100==0 or snap_number==len(sources):
            print(json.dumps({'read_snapshots':snap_number,'unique_status_news':len(catalog),
                'raw_player_observations':len(raw_refs),'target_player_contexts':len(fixture_refs)}),flush=True)
    key_ids = {key:f'news_{number:05d}' for number,key in enumerate(sorted(catalog,key=sort_key),1)}
    catalogue_rows = []
    for key in sorted(catalog,key=sort_key):
        source = catalog[key]
        catalogue_rows.append({**{k:v for k,v in source.items() if k not in
            ('snapshot_ids','player_codes','target_match_ids')}, 'news_id':key_ids[key],
            'snapshot_count':len(source['snapshot_ids']),
            'source_player_code_count':len(source['player_codes']),
            'target_match_count':len(source['target_match_ids']),
            'raw_status_counts':{str(key[0]):source['source_player_observation_count']},
            'reference_lookup':'Use news_id in 快照球员新闻引用.json.gz and 比赛球员新闻引用.json.gz; snapshot IDs resolve via 来源快照目录.json'})
    for ref in raw_refs+fixture_refs:
        ref['news_id'] = key_ids[ref.pop('_news_key')]
    used_keys = [key for key,item in catalog.items() if item['target_player_context_count']]
    flagged_keys = [key for key,item in catalog.items() if item['target_non_a_player_context_count']]
    normalization = defaultdict(list)
    for key in catalog:
        if isinstance(key[1],str):
            normalization[(key[0], ' '.join(key[1].split()).casefold())].append(key_ids[key])
    normalization_candidates = [{'raw_status':key[0], 'candidate_news_ids':values,
        'normalization_for_review_only':'Unicode casefold and collapsed whitespace; no semantic or source rows were merged'}
        for key,values in normalization.items() if len(values)>1]
    report = {'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_context_file':str(CONTEXT), 'source_manifest_file':str(MANIFEST),
        'context_matches_total':len(contexts), 'epl_matches_with_source':len(selected),
        'matches_without_source':len(contexts)-len(selected), 'snapshot_n':len(sources),
        'raw_player_observation_n':len(raw_refs),'unique_exact_status_news_n':len(catalog),
        'unique_exact_status_news_with_target_context_n':len(used_keys),
        'unique_exact_non_a_status_news_with_target_context_n':len(flagged_keys),
        'unique_raw_news_string_or_null_n':len({key[1] for key in catalog}),
        'source_empty_news_observations':{kind:sum(r['source_player_observation_count']
            for r in catalogue_rows if r['news_empty_type']==kind)
            for kind in ('null','empty_string','whitespace_only','nonempty')},
        'target_player_context_n':len(fixture_refs),
        'target_non_a_player_context_n':sum(item['target_non_a_player_context_count'] for item in catalog.values()),
        'existing_export_non_a_player_context_n':existing_flagged_n,
        'raw_snapshot_player_status_counts':{str(k):v for k,v in stats.items()},
        'target_player_status_counts':{str(k):v for k,v in target_stats.items()},
        'catalog_news_empty_type_counts':dict(Counter(r['news_empty_type'] for r in catalogue_rows)),
        'source_field_missing_counts':dict(source_field_missing),
        'news_added_after_proxy_n':len(future_news_metadata),
        'normalization_candidate_groups_n':len(normalization_candidates),
        'by_period':{period:{'all_context_matches':sum(r['period']==period for r in contexts),
            'source_context_matches':sum(r['period']==period for r in selected)} for period in sorted({r['period'] for r in contexts})},
        'semantic_classification_performed':False,'model_fits':0,'network_requests':0,
        'target_scores_or_outcomes_read':False,'original_raw_text_changed':False,
        'new_raw_snapshot_downloads':0, 'catalog_key':['exact raw status','exact raw news including null/empty distinction'],
        'cataloguing_before_output_seconds':time.perf_counter()-tick,
        'limits':['FPL coverage is England only; 459 snapshots and many player references are not independent match samples.',
            'Source statuses are labels: u/n are not automatically injury, a is not confirmed fitness.',
            'news_added is source metadata, not medical onset or confirmed return.',
            'Raw chance fields are source values, not calibrated appearance or win probabilities.',
            'Filename time and repository commit are retrospective availability proxies, not independently established original publication/push times.',
            'Catalogue includes all players in selected snapshots; only target_context_count>0 applies to the1502 fixture contexts.',
            'Null/empty/whitespace news and different statuses remain separate exact records; normalization candidates are not merged.',
            'No outcomes, performance scores, manual injury deductions, or future team strength labels are used.']}
    if report['target_non_a_player_context_n'] != existing_flagged_n:
        raise ValueError('Old target non-a cohort changed')
    write(OUT/'来源快照目录.json',{'snapshots':snapshot_index})
    write(OUT/'源新闻目录.json', {'created_at_utc':report['created_at_utc'],
        'key_policy':report['catalog_key'],'semantic_classification_performed':False,'records':catalogue_rows})
    write(OUT/'快照球员新闻引用.json.gz', {'observations':raw_refs})
    write(OUT/'比赛球员新闻引用.json.gz', {'references':fixture_refs})
    write(OUT/'仅大小写空白差异候选.json',{'merged':False,'groups':normalization_candidates})
    write(OUT/'新闻时间异常引用.json',{'references':future_news_metadata})
    lines=['# FPL历史原文目录与引用覆盖','',
        f"全量{len(contexts)}场中，{len(selected)}场英超有既存快照；仅读取{len(sources)}个原始xz文件，未重复下载或拟合。",'',
        f"原快照球员观测{len(raw_refs)}条，exact(status,news)目录{len(catalog)}条，其中本1502场实际两队上下文涉及{len(used_keys)}条、非a上下文涉及{len(flagged_keys)}条。",'',
        f"两队全名单上下文{len(fixture_refs)}条；其中非a为{existing_flagged_n}条，与上一轮导出逐项一致。球员观测、新闻条目和快照均不当作新增比赛样本。",'',
        '目录保留原始status和news：不同状态、空串、null和仅空白原文不合并。news_id与snapshot_id只是普通顺序编号，无内容hash。语义分类尚未执行。', '',
        '- 源新闻目录.json：原文、源状态、首次/末次快照代理时刻、各类次数。',
        '- 快照球员新闻引用.json.gz：每份快照的球员code、local ID、球队、news_added及源chance原值；通过news_id取回原文。',
        '- 比赛球员新闻引用.json.gz：每场同F的球队/球员到observation_id、news_id引用，含全部a，非a子集不变。',
        '- 来源快照目录.json：源URL、已有Git版本、文件名/commit代理时间、本轮接收时间和原本地路径。', '',
        'u/n不能一概称伤病；a不证明健康。news_added不是起病时间或医学归期。Git记录仅为回溯版本代理，原始首次发布时间/推送/修订仍有不确定性。未读取目标比分或胜负来选择文本。', '',
        '来源：[Randdalf/fplcache](https://github.com/Randdalf/fplcache)，既有本地缓存；本轮未调用互联网。']
    (OUT/'来源与覆盖.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    report['elapsed_seconds'] = time.perf_counter()-tick
    write(OUT/'来源与覆盖.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
