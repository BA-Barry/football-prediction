"""Finite independent read-only audit; no fits, outcome inspection or predictions."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
from urllib.parse import urlparse
import csv
import gc
import gzip
import json
import lzma
import re

from align_personnel_news_history_20261028 import load_original_windows
from build_personnel_semantic_packets_20261028 import materialize_match

ROOT = REPOSITORY_ROOT
DATA = ROOT/'data/personnel_semantics_20261028'
OUT = ROOT/'outputs/人员信息语义接入_20261028'
errors = []
counts = Counter()


def read(path):
    with (gzip.open(path, 'rt', encoding='utf-8') if str(path).endswith('.gz') else path.open(encoding='utf-8-sig')) as h:
        return json.load(h)


def check(ok, detail):
    if not ok and len(errors) < 40:
        errors.append(detail)


def utc(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def before(a, b):
    return bool(a and b and utc(a) < utc(b))


def forbidden_keys(document):
    banned = {'actual_score', 'actual_outcome', 'actual_home_score', 'actual_away_score',
              'target_lineup', 'target_starting_xi', 'actual_target_lineup',
              'target_player_minutes', 'actual_target_minutes', 'target_result'}
    found = Counter()
    stack = [document]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            found.update(k for k in item if k in banned)
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return dict(found)


def observation_equal(view, source):
    fields = ('observation_id', 'snapshot_id', 'source_player_id', 'source_player_code',
              'source_team_code', 'news_id', 'news_added', 'chance_of_playing_this_round',
              'chance_of_playing_next_round')
    return all(view.get(k) == source.get(k) for k in fields) and all(
        view[k] == raw_news[source['news_id']][k] for k in ('raw_news', 'raw_status'))


print('Loading original source references and alignment.', flush=True)
snapshots = {r['snapshot_id']: r for r in read(DATA/'来源快照目录.json')['snapshots']}
raw_news = {r['news_id']: r for r in read(DATA/'源新闻目录.json')['records']}
obs = {r['observation_id']: r for r in read(DATA/'快照球员新闻引用.json.gz')['observations']}
current_refs = defaultdict(set)
for ref in read(DATA/'比赛球员新闻引用.json.gz')['references']:
    current_refs[(ref['match_id'], ref['side'])].add(ref['observation_id'])
windows = load_original_windows()
aligned = read(DATA/'逐场人员新闻上下文.json.gz')
check(not forbidden_keys(aligned), 'Target labels or actual lineup/minutes in alignment')
registry = {}
all_classifications = Counter()
for row in aligned['rows']:
    ident, cutoff = row['match_id'], row['forecast_before_date']
    check(ident not in registry, 'Duplicate match ID '+ident)
    registry[ident] = {k: row[k] for k in ('date', 'forecast_before_date', 'period', 'competition', 'home', 'away')}
    registry[ident]['teams'] = []
    for team in row['teams']:
        side = team['side']
        counts['aligned_team_sides'] += 1
        check(team['exact_reference_match_ids'] == windows[ident][side], 'Original five-game IDs differ '+ident)
        check([r['match_id'] for r in team['reference_sources']] == team['exact_reference_match_ids'], 'Reference order differs '+ident)
        if not team['observed']:
            counts['unobserved_team_sides'] += 1
            check(team['current_snapshot'] is None and team['players'] == [] and team['descriptive_comparison_counts'] is None,
                  'Unknown team imputed as observed '+ident)
        else:
            counts['observed_team_sides'] += 1
            snapshot = team['current_snapshot']
            original_snapshot = snapshots[snapshot['snapshot_id']]
            check(before(snapshot['availability_proxy_at'], cutoff), 'Current snapshot crosses F '+ident)
            check(ident in original_snapshot['target_match_ids'], 'Snapshot-target association differs '+ident)
            check(all(snapshot.get(k) == original_snapshot.get(k) for k in ('path','local_path','commit','source_url','availability_proxy_at')),
                  'Current source fields differ '+ident)
            check(set(p['current']['observation_id'] for p in team['players']) == current_refs[(ident,side)], 'Current source roster differs '+ident)
        for reference in team['reference_sources']:
            if reference['date']:
                check(before(reference['date'], cutoff), 'Reference fixture crosses target F '+ident)
            if reference['observed']:
                check(before(reference['availability_proxy_at'], reference['forecast_before_date']), 'Reference snapshot crosses old F '+ident)
        for player in team['players']:
            counts['aligned_current_player_references'] += 1
            current = player['current']
            check(observation_equal(current,obs[current['observation_id']]), 'Current observation differs '+ident)
            for reference, source in zip(player['references'],team['reference_sources']):
                category = reference['classification']
                all_classifications[category] += 1
                old = reference['observation']
                if old is None:
                    check(category not in ('same_status_and_news','same_status_news_changed','same_news_status_changed','both_status_and_news_changed'), 'Unknown treated as paired '+ident)
                    continue
                check(observation_equal(old,obs[old['observation_id']]), 'Old observation differs '+ident)
                check(old['snapshot_id'] == source['snapshot_id'] and old['source_player_code'] == current['source_player_code']
                      and old['source_team_code'] == current['source_team_code'] == team['source_team_code'], 'Old source/code/team differs '+ident)
                same_status = current['raw_status'] == old['raw_status']
                same_text = current['raw_news'] == old['raw_news']
                expected = ('same_status_and_news' if same_status and same_text else 'same_status_news_changed' if same_status
                            else 'same_news_status_changed' if same_text else 'both_status_and_news_changed')
                check(category == expected, 'State and text classification differs '+ident)
        registry[ident]['teams'].append({'team':team['team'],'side':side,'observed':team['observed'],
            'source_team_code':team['source_team_code'],'current_snapshot':team['current_snapshot'],
            'exact_reference_match_ids':team['exact_reference_match_ids'], 'reference_sources':team['reference_sources']})
check(len(registry) == 8753 and counts['aligned_current_player_references'] == 111608, 'Expected cohort/reference total differs')
for snap in snapshots.values():
    url = urlparse(snap['source_url'])
    check(url.scheme == 'https' and bool(url.netloc) and snap['commit'] in url.path and snap['path'] in url.path, 'Unresolvable source URL '+snap['snapshot_id'])
    check(Path(snap['local_path']).is_file(), 'Missing local source '+snap['snapshot_id'])
    utc(snap['availability_proxy_at'])
counts['parseable_snapshot_sources'] = len(snapshots)
del aligned
gc.collect()

print('Loading final fact packet and validating every exported current player.', flush=True)
packet = read(DATA/'历史LLM人员事实包.json.gz')
examples = read(OUT/'人员新闻与参与联合实例.json')
summary = read(OUT/'事实包与参与汇总.json')
check(not forbidden_keys(packet) and not forbidden_keys(examples), 'Target labels/actual lineup/minutes exported')
semantic_keys = {'news_id','raw_status','raw_news','semantics'}
check(all(set(v) == semantic_keys for v in packet['news_semantics'].values()), 'Future catalogue metadata remains in LLM semantic views')
for key, value in packet['news_semantics'].items():
    check(all(value[k] == raw_news[key][k] for k in ('raw_status','raw_news')), 'Semantic source text differs '+key)
packet_by_id = {r['match_id']:r for r in packet['rows']}
check(set(packet_by_id) == set(registry), 'Packet cohort differs')

with (ROOT/'data/fpl_availability_archive/id_mapping/Master.csv').open(encoding='utf-8-sig',newline='') as h:
    mapping_rows = list(csv.DictReader(h))
by_code, reverse = defaultdict(list), defaultdict(set)
for r in mapping_rows:
    if r['code'].isdigit() and r['understat'].isdigit():
        by_code[int(r['code'])].append(r)
        reverse[r['understat']].add(int(r['code']))
mapping = {c:rs[0] for c,rs in by_code.items() if len(rs)==1 and len(reverse[rs[0]['understat']])==1}

candidates = defaultdict(list)
usage_counts = Counter()
for row in packet['rows']:
    ident, cutoff = row['match_id'], row['forecast_before_date']
    expected = registry[ident]
    check(all(row[k] == expected[k] for k in expected if k != 'teams'), 'Packet metadata differs '+ident)
    params = row['earlier_statistical_context']
    check(not params['parameter_training_max_date'] or before(params['parameter_training_max_date'],params['parameter_before_date']), 'Parameter training boundary differs '+ident)
    check(utc(params['parameter_before_date']) <= utc(cutoff), 'Parameters later than forecast '+ident)
    for side, team in enumerate(row['personnel_context']['teams']):
        check(all(team.get(k) == v for k,v in expected['teams'][side].items()), 'Packet source alignment differs '+ident)
        perf = row['recent_performance'][side]
        check(not perf['history']['latest_used_date'] or before(perf['history']['latest_used_date'], cutoff), 'Recent performance crosses F '+ident)
        if not team['observed']:
            check(team['same_team_participation'] is None, 'Unknown team has participation value '+ident)
            continue
        usage = team['same_team_participation']
        check(usage['exact_reference_match_ids'] == windows[ident][team['side']], 'Participation uses another window '+ident)
        usage_counts['complete_minute_windows'] += usage['minute_window_complete']
        check(not usage['latest_minute_reference_date'] or before(usage['latest_minute_reference_date'],cutoff), 'Participation crosses F '+ident)
        for player in team['players']:
            usage_counts['current_player_references'] += 1
            u = player['prior_same_team_usage']
            mapped = mapping.get(player['source_player_code'])
            col = row['period'][2:4]+'-'+row['period'][-2:]
            matched = bool(mapped and mapped.get(col,'').isdigit() and int(mapped[col]) == player['current']['source_player_id'])
            check(u['identity_matched_by_code_and_same_season_id'] == matched, 'Code and same-season ID matching differs '+ident)
            check(u['understat_player_id'] == ('understat:player:'+mapped['understat'] if matched else None), 'Wrong Understat identity '+ident)
            if not matched or not usage['minute_window_complete']:
                check(u['window_minutes'] is None and u['window_minute_share'] is None, 'Unknown imputed as zero '+ident)
            else:
                check(u['window_minutes'] >= 0 and abs(u['window_minute_share']-u['window_minutes']/usage['actual_observed_team_minutes']) < 1e-12, 'Minute share differs '+ident)
            if not matched:
                check(u['prior_positive_same_team_minutes_observed'] is None, 'Unknown identity assigned old minutes '+ident)
                state = 'identity_unknown'
            elif u['window_minutes'] is None:
                state = 'window_unknown'
            elif u['window_minutes'] > 0:
                state = 'positive_window_minutes'
            elif u['prior_positive_same_team_minutes_observed']:
                state = 'zero_window_but_earlier_positive_observed'
            else:
                state = 'zero_window_no_prior_positive_observed'
            usage_counts[state] += 1
            categories = packet['news_semantics'][player['current']['news_id']]['semantics']['text_categories']
            if row['period'] != '2024-25':
                continue
            candidate = (row,side,player)
            if 'medical_reference' in categories and state == 'positive_window_minutes' and any(r['classification']=='same_status_news_changed' for r in player['references']):
                candidates['same_status_different_medical_text_positive_usage'].append(candidate)
            if 'medical_reference' in categories and state == 'zero_window_but_earlier_positive_observed':
                candidates['medical_text_zero_recent_but_older_participation'].append(candidate)
            if 'transfer_reference' in categories and state == 'positive_window_minutes':
                candidates['transfer_text_positive_prior_participation'].append(candidate)
check(usage_counts['current_player_references']==111608, 'Packet current player total differs')
check(all(summary['counts'].get(k)==v for k,v in usage_counts.items()), 'Summary usage counts differ')

selected, chosen = set(), []
for mechanism,pool in candidates.items():
    options = sorted(pool,key=lambda c:(c[0]['date'],c[0]['match_id'],str(c[2]['source_player_code'])),reverse=True)
    pick = next((c for c in options if c[0]['match_id'] not in selected),None)
    if pick:
        selected.add(pick[0]['match_id'])
        chosen.append((mechanism,*pick))
check(len(chosen)==len(examples['records'])==3, 'Expected three declared mechanism examples')
for (mechanism,row,side,player),example in zip(chosen,examples['records']):
    check(example['mechanism_for_selection']==mechanism and example['match_id']==row['match_id'] and example['player']==player,
          'Declared label-free candidate selection differs')
    referenced = {player['current']['news_id']} | {r['observation']['news_id'] for r in player['references'] if r['observation']}
    check(set(example['news_semantics'])==referenced, 'Example includes unreferenced future text')

print('Independently recomputing three example usage windows and raw snapshots.', flush=True)
minute_rows = {}
minute_paths = [ROOT/'outputs/历史球员使用输入_20261025/历史出场分钟.json', ROOT/'outputs/当前人员历史代表性_20261025/逐场球员使用记录.json']
for path in minute_paths:
    for r in read(path)['observations']:
        key = (r.get('football_data_match_id',r['match_id']),r['team_id'])
        if key in minute_rows:
            check(minute_rows[key]['minutes']==r['minutes'], 'Overlapping minute input differs')
        minute_rows[key]=r
raw_cache = {}
samples = []
for mechanism,row,side,player in chosen:
    team = row['personnel_context']['teams'][side]
    ids,cutoff = windows[row['match_id']][team['side']], row['forecast_before_date']
    refs = [minute_rows.get((ident,team['team'])) for ident in ids]
    complete = len(ids)==5 and all(r is not None for r in refs)
    summed = Counter()
    if complete:
        for r in refs:
            check(before(r['date'],cutoff) and r['team_id']==team['team'], 'Sample minute source crosses team/F')
            summed.update(r['minutes'])
    denominator = sum(summed.values()) if complete else None
    check(denominator==team['same_team_participation']['actual_observed_team_minutes'], 'Sample team minute denominator differs')
    prior_positive = {p for r in minute_rows.values() if r['team_id']==team['team'] and before(r['date'],cutoff) for p,v in r['minutes'].items() if v>0}
    checked_players=0
    for p in team['players']:
        u=p['prior_same_team_usage']; key=u['understat_player_id']
        expected_minutes=summed.get(key,0) if complete and key else None
        check(u['window_minutes']==expected_minutes, 'Sample player minutes differ '+str(p['source_player_code']))
        check(u['prior_positive_same_team_minutes_observed']==(key in prior_positive if key else None), 'Sample old-positive flag differs')
        checked_players+=1
    checked_raw=0
    pairs=[(team['current_snapshot'],player['current'])]+[(s,r['observation']) for s,r in zip(team['reference_sources'],player['references']) if r['observation']]
    for source,view in pairs:
        path=source['local_path']
        if path not in raw_cache:
            raw_cache[path]=json.loads(lzma.decompress(Path(path).read_bytes()))
        raw=raw_cache[path]
        team_codes={t['id']:t['code'] for t in raw['teams']}
        hit=[p for p in raw['elements'] if p['id']==view['source_player_id'] and p.get('code')==view['source_player_code'] and team_codes[p['team']]==view['source_team_code']]
        check(len(hit)==1 and hit[0].get('news')==view['raw_news'] and hit[0].get('status')==view['raw_status'], 'Sample raw snapshot text/state differs')
        checked_raw+=1
    rendered=materialize_match(packet,row['match_id'])
    needed={p['current']['news_id'] for t in rendered['record']['personnel_context']['teams'] for p in t['players']}
    needed.update(r['observation']['news_id'] for t in rendered['record']['personnel_context']['teams'] for p in t['players'] for r in p['references'] if r['observation'])
    check(set(rendered['news_semantics'])==needed and 'rows' not in rendered and not forbidden_keys(rendered), 'Single-match local rendering differs')
    check(all(set(v)==semantic_keys for v in rendered['news_semantics'].values()), 'Single-match future catalogue metadata')
    samples.append({'match_id':row['match_id'],'mechanism':mechanism,'team':team['team'],'forecast_before_date':cutoff,
        'selected_source_player_code':player['source_player_code'],'window_match_ids':ids,'minute_source_dates':[r['date'] if r else None for r in refs],
        'team_minutes_independently_recomputed':denominator,'all_team_players_recomputed':checked_players,
        'selected_player_minutes':player['prior_same_team_usage']['window_minutes'],'raw_current_and_old_snapshots_checked':checked_raw,
        'current_snapshot':team['current_snapshot']['snapshot_id'],'current_and_old_source_text_state_match':True,
        'single_match_semantic_entries':len(needed)})

report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'verdict':'pass' if not errors else 'findings',
    'scope':'Finite independent audit of source alignment and final reading inputs; no training, scoring, current results or prediction changes.',
    'counts':dict(counts),'matches':len(registry),'packet_matches':len(packet['rows']),
    'source_observations':len(obs),'news_semantic_entries':len(packet['news_semantics']),
    'classification_counts':dict(all_classifications),'participation_counts':dict(usage_counts),
    'candidate_counts_independently_reproduced':{k:len(v) for k,v in candidates.items()},
    'sample_recomputations':samples,'unresolved_findings':errors,
    'resolved_review_finding':'Removed all-history first/last observation times and coverage frequencies from final LLM semantic views; only requested match and its cited current/paired old texts are materialized.',
    'label_boundary':'Historical base source contains target scores/outcomes and builder json.load decodes the mixed file. Explicit output projection, candidate selection and usage calculations do not reference target-label columns. This audit loaded only original five-match IDs from that source via projection. No current match result was requested.',
    'interpretation':'Exact state and text equality are separate. Existing participation and recent performance are context, not additional loss coefficients; unchanged absence is not automatically charged again.',
    'limits':['This is a retrospective evidence-reading/development artifact, not a prospective historical LLM forecast or performance evaluation.',
        'Understat identity uses a retrospectively assembled mapping table with stable source code plus same-season local ID agreement; no assertion that the mapping itself existed at F.',
        'Archive filename/commit proxy is version evidence, not independently proved first public announcement time.',
        'Complete-window zero means observed earlier league participation only. Unknown identity or incomplete window remains null.',
        'The aggregate multi-match storage file and the original alignment news catalog are research assets; per-match materialization is the reading input.',
        'No new model was fitted and text-to-ability, replacement quality, target absence and causal text effects remain unestimated.']}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'独立复核.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
md=f"""# 人员新闻与参与事实包：独立复核

结论：{'通过，未留未解决问题' if not errors else '存在待解决问题'}。

全量核对 {len(registry)} 场、{counts['aligned_current_player_references']} 条当前球员引用、{counts['observed_team_sides']} 个已观测球队侧和 {len(snapshots)} 个来源快照。来源引用、当前/旧快照、原始近五场ID、严格F前边界、状态与原文分别比较均一致。身份使用稳定source code与同季source ID双重匹配；未知不转成零。

事实包与三个实例没有导出目标比分、胜平负、实际目标阵容或目标分钟。独立重现三个机制的候选选择，未使用结果；三场逐一重算所选球队全体球员的近五场分钟，并从本地原始xz核对所选球员当前与旧快照的原文及状态。

审查发现并修正：原语义整条记录会携带全历史观察时间和频次。最终LLM语义视图只保留news_id、raw_status、raw_news、semantics；单场读取仅物化请求比赛及其引用原文。全历史存储包不应整体作为一场的LLM输入。

历史base源文件确实含赛果，builder会用json.load解码整份文件；计算、选择与输出未引用目标标签列。本复核仅从该混合源投影读取原始近五场ID，没有查询当前比赛结果。回溯Master身份表和档案commit时间不能证明这些映射或新闻在F时已首次公开；本产物也不是前瞻预测检验。重复缺阵仅作上下文，不自动另扣实力。

抽样复算：
"""
for s in samples:
    md+=f"\n- {s['match_id']}：重算 {s['all_team_players_recomputed']} 名球员，球队分钟 {s['team_minutes_independently_recomputed']}，核对 {s['raw_current_and_old_snapshots_checked']} 个当前/旧来源引用。\n"
if errors:
    md+='\n待解决：\n'+''.join('- '+e+'\n' for e in errors)
(OUT/'独立复核.md').write_text(md,encoding='utf-8')
print(json.dumps({'verdict':report['verdict'],'matches':len(registry),'current_player_references':counts['aligned_current_player_references'],
    'findings':errors,'samples':samples},ensure_ascii=False),flush=True)
