"""Align exact source text and state at existing forecast cutoffs.

This is a descriptive join, not a new availability model. Stable FPL player
code AND same stable source team code are required. A changed string is never
treated as injury onset, recovery, ability, lost strength or expected minutes.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import re


ROOT = REPOSITORY_ROOT
DATA = ROOT/'data/personnel_semantics_20261028'
CONTEXT = ROOT/'data/historical_context_20261028/availability_sources/8753场历史可用性上下文.json.gz'
HISTORY = ROOT/'outputs/协作预测与大样本方案_20261028/扩样训练/扩样合并特征.json.gz'
OUT = DATA/'逐场人员新闻上下文.json.gz'
SUMMARY = DATA/'对齐汇总.json'
SIDES = ('home', 'away')
PAIRED = ('same_status_news_changed', 'same_news_status_changed',
          'both_status_and_news_changed', 'same_status_and_news')


def read(path):
    with (gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz'
          else path.open(encoding='utf-8-sig')) as handle:
        return json.load(handle)


def utc(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def strictly_before(proxy, cutoff):
    return bool(proxy and cutoff and utc(proxy) < utc(cutoff))


# Decode only selected paths from the mixed feature/target source. Unselected
# values are skipped as JSON tokens, without decoding target scores/outcomes.
# The allowlist also makes this useful for projecting a synthetic fixture with
# changed labels, rather than introducing a second input representation.
_STRING = re.compile(r'"(?:[^"\\]|\\.)*"', re.S)
_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|[{}\[\]]|[^{}\[\]"]+', re.S)
_SPACE = re.compile(r'\s*')
_PRIMITIVE = re.compile(r'[^,\]}\s]+')


def project_json(source, selection):
    decoder = json.JSONDecoder()
    pos = 0

    def space():
        nonlocal pos
        pos = _SPACE.match(source, pos).end()

    def skip():
        nonlocal pos
        space()
        first = source[pos]
        if first == '"':
            pos = _STRING.match(source, pos).end()
        elif first in '[{':
            depth = 0
            for match in _TOKEN.finditer(source, pos):
                token = match.group()
                if token in ('{', '['):
                    depth += 1
                elif token in ('}', ']'):
                    depth -= 1
                if depth == 0:
                    pos = match.end()
                    break
        else:
            pos = _PRIMITIVE.match(source, pos).end()

    def parse(wanted):
        nonlocal pos
        space()
        if wanted is True:
            value, pos = decoder.raw_decode(source, pos)
            return value
        if source[pos] == '[':
            pos += 1
            result = []
            space()
            while source[pos] != ']':
                result.append(parse(wanted['*']))
                space()
                if source[pos] == ',':
                    pos += 1
                    space()
                else:
                    break
            if source[pos] != ']':
                raise ValueError('Invalid selected JSON array')
            pos += 1
            return result
        if source[pos] != '{':
            raise ValueError('Selected JSON path is not an object')
        pos += 1
        result = {}
        space()
        while source[pos] != '}':
            key, pos = decoder.raw_decode(source, pos)
            space()
            if source[pos] != ':':
                raise ValueError('Invalid selected JSON object')
            pos += 1
            if key in wanted:
                result[key] = parse(wanted[key])
            else:
                skip()
            space()
            if source[pos] == ',':
                pos += 1
                space()
            else:
                break
        if source[pos] != '}':
            raise ValueError('Invalid selected JSON object end')
        pos += 1
        return result

    result = parse(selection)
    space()
    if pos != len(source):
        raise ValueError('Trailing JSON data')
    return result


def load_original_windows(path=HISTORY):
    side = {'window_match_ids': {'5': True}}
    selection = {'rows': {'*': {'match_id': True, 'fresh_feature_history': {
        'history': {'home': side, 'away': side}}}}}
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        rows = project_json(handle.read(), selection)['rows']
    result = {}
    for row in rows:
        ident = row['match_id']
        if ident in result:
            raise ValueError('Original history match ID is duplicated')
        result[ident] = {side: row['fresh_feature_history']['history'][side]['window_match_ids']['5']
                         for side in SIDES}
    return result


def build_indexes(observations):
    by_id, teams, codes, duplicates = {}, defaultdict(lambda: defaultdict(list)), defaultdict(list), []
    for item in observations:
        ident = item['observation_id']
        if ident in by_id:
            raise ValueError('Duplicated observation ID: '+ident)
        by_id[ident] = item
        teams[(item['snapshot_id'], item['source_team_code'])][item['source_player_code']].append(item)
        if item['source_player_code'] is not None:
            codes[(item['snapshot_id'], item['source_player_code'])].append(item)
    for (snapshot, team), players in teams.items():
        for code, items in players.items():
            if len(items) > 1:
                duplicates.append({'snapshot_id': snapshot, 'source_team_code': team,
                    'source_player_code': code, 'observation_ids': [i['observation_id'] for i in items]})
    return by_id, dict(teams), dict(codes), duplicates


def source_comparison(current, old, raw_catalog):
    """The news key also includes status: compare raw_news, never news_id."""
    now, then = raw_catalog[current['news_id']], raw_catalog[old['news_id']]
    same_status = now['raw_status'] == then['raw_status']
    same_news = now['raw_news'] == then['raw_news']
    if same_status and same_news:
        return 'same_status_and_news'
    if same_status:
        return 'same_status_news_changed'
    if same_news:
        return 'same_news_status_changed'
    return 'both_status_and_news_changed'


def classify_pair(current, reference_source, team_players, by_code, raw_catalog):
    if current['source_player_code'] is None:
        return 'missing_stable_player_code', None, []
    if not reference_source['observed']:
        return 'reference_record_missing', None, []
    snapshot, team, code = (reference_source['snapshot_id'], current['source_team_code'],
                            current['source_player_code'])
    if reference_source['source_team_code'] != team:
        return 'team_identity_changed_unknown', None, []
    candidates = team_players.get((snapshot, team), {}).get(code, [])
    if len(candidates) > 1:
        return 'duplicate_reference_identity', None, [p['observation_id'] for p in candidates]
    if candidates:
        return source_comparison(current, candidates[0], raw_catalog), candidates[0], []
    other_teams = [p['observation_id'] for p in by_code.get((snapshot, code), [])
                   if p['source_team_code'] != team]
    if other_teams:
        return 'other_team_same_code_transfer_unknown', None, other_teams
    return 'player_new_or_historical_identity_unobserved', None, []


def observation_view(item, catalog, semantics):
    raw = catalog[item['news_id']]
    return {key: item.get(key) for key in ('observation_id', 'snapshot_id', 'source_player_id',
        'source_player_code', 'source_team_code', 'news_id', 'news_added',
        'chance_of_playing_this_round', 'chance_of_playing_next_round')} | {
        'raw_status': raw['raw_status'], 'raw_news': raw['raw_news'],
        'semantic_news_id': item['news_id'] if item['news_id'] in semantics else None}


def snapshot_view(snapshot, forecast):
    return {'snapshot_id': snapshot['snapshot_id'], 'forecast_before_date': forecast,
        **{key: snapshot.get(key) for key in ('availability_proxy_at', 'path', 'local_path',
            'source_url', 'commit', 'filename_observed_at', 'repository_recorded_at')}}


def reference_source(ident, team, target_cutoff, context_by_id, snapshots_by_path):
    old = context_by_id.get(ident)
    result = {'match_id': ident, 'date': None, 'forecast_before_date': None,
        'observed': False, 'source_team_code': None, 'snapshot_id': None,
        'availability_proxy_at': None, 'missing_reason': 'reference_match_not_in_exported_cohort'}
    if old is None:
        return result
    result.update(date=old['date'], forecast_before_date=old['forecast_before_date'])
    if not strictly_before(old['date'], target_cutoff):
        raise ValueError('Reference match date must be strictly before the target F')
    old_team = next((t for t in old['teams'] if t['team'] == team), None)
    if old_team is None:
        result['missing_reason'] = 'team_not_in_original_reference_match'
        return result
    result['source_team_code'] = old_team['source_team_code']
    if not old_team['observed'] or old['archive_observation'] is None:
        result['missing_reason'] = old_team.get('missing_reason') or old.get('missing_reason') or 'reference_team_unobserved'
        return result
    snap = snapshots_by_path[old['archive_observation']['path']]
    if not strictly_before(snap['availability_proxy_at'], old['forecast_before_date']):
        raise ValueError('Historical source proxy must be strictly before historical F')
    result.update(snapshot_view(snap, old['forecast_before_date']))
    result.update(observed=True, missing_reason=None)
    return result


def team_context(row, side, windows, context_by_id, snapshots_by_path,
                 team_players, by_code, catalog, semantics):
    team = next(t for t in row['teams'] if t['side'] == side)
    ident, cutoff = row['match_id'], row['forecast_before_date']
    ids = windows[ident][side]
    diagnostics = row.get('existing_participation_diagnostics')
    if diagnostics and diagnostics[side]['exact_reference_match_ids'] != ids:
        raise ValueError('Original past-five window differs from existing participation context')
    refs = [reference_source(ref, team['team'], cutoff, context_by_id, snapshots_by_path) for ref in ids]
    result = {key: team.get(key) for key in ('team', 'side', 'observed', 'source_team_code', 'missing_reason')}
    result.update(current_snapshot=None, exact_reference_match_ids=ids, reference_sources=refs,
                  players=[], descriptive_comparison_counts=None)
    if not team['observed']:
        return result
    archived = row['archive_observation']
    snap = snapshots_by_path[archived['path']]
    if not strictly_before(snap['availability_proxy_at'], cutoff):
        raise ValueError('Current source proxy must be strictly before target F')
    if snap['availability_proxy_at'] != archived['availability_proxy_at']:
        raise ValueError('Current news and existing source states identify different proxies')
    if ident not in snap['target_match_ids']:
        raise ValueError('Current source snapshot does not list the target match')
    result['current_snapshot'] = snapshot_view(snap, cutoff)
    players = team_players.get((snap['snapshot_id'], team['source_team_code']), {})
    if not players:
        raise ValueError('Observed team has no player observations')
    counts = Counter()
    for code, candidates in sorted(players.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)):
        if len(candidates) != 1:
            # Keep every candidate; never choose an arbitrary same-team/code row.
            result['players'].append({'source_player_code': code, 'current': None,
                'current_identity_ambiguous': True,
                'current_candidates': [observation_view(p, catalog, semantics) for p in candidates],
                'references': [{'match_id': ref['match_id'], 'classification': 'duplicate_current_identity',
                    'observation': None, 'other_or_ambiguous_observation_ids': []} for ref in refs]})
            counts['duplicate_current_identity'] += len(refs)
            continue
        current = candidates[0]
        out = {'source_player_code': code, 'source_player_id': current['source_player_id'],
            **{key: current.get(key) for key in ('first_name', 'second_name', 'web_name', 'element_type')},
            'current_identity_ambiguous': False,
            'current': observation_view(current, catalog, semantics), 'references': []}
        for ref in refs:
            classification, old, other = classify_pair(current, ref, team_players, by_code, catalog)
            counts[classification] += 1
            out['references'].append({'match_id': ref['match_id'], 'classification': classification,
                'observation': None if old is None else observation_view(old, catalog, semantics),
                'other_or_ambiguous_observation_ids': other})
        result['players'].append(out)
    current_codes = set(players)
    for ref in refs:
        ref['reference_only_players'] = None
        if ref['observed'] and ref['source_team_code'] == team['source_team_code']:
            old_players = team_players.get((ref['snapshot_id'], team['source_team_code']), {})
            missing_now = [p for code, old in old_players.items() if code not in current_codes for p in old]
            ref['reference_only_players'] = [{'source_player_code': p['source_player_code'],
                'observation_id': p['observation_id'], 'news_id': p['news_id'],
                'classification': 'absent_from_current_same_team_roster_unknown'} for p in missing_now]
    result['descriptive_comparison_counts'] = dict(counts)
    return result


def load_semantics():
    path = DATA/'语义目录.json'
    if not path.exists():
        return {}, {'path': str(path), 'available': False, 'record_count': 0}
    document = read(path)
    if isinstance(document, list):
        rows = document
    else:
        rows = next((document[key] for key in ('records', 'semantics', 'news', 'entries')
                     if isinstance(document.get(key), list)), None)
    if rows is None:
        raise ValueError('Semantic catalog needs a record list with news_id')
    records = {}
    for item in rows:
        if item['news_id'] in records:
            raise ValueError('Semantic news ID duplicated')
        records[item['news_id']] = item
    return records, {'path': str(path), 'available': True, 'record_count': len(records)}


def synthetic_checks():
    # Exact equality is excluded at BOTH cutoff boundaries.
    assert not strictly_before('2024-08-02T00:00:00Z', '2024-08-02')
    assert strictly_before('2024-08-01T23:59:59Z', '2024-08-02')
    catalogs = {'a': {'raw_status': 'i', 'raw_news': 'Thigh injury'},
        'b': {'raw_status': 'i', 'raw_news': 'Hip injury'},
        'c': {'raw_status': 'd', 'raw_news': 'Thigh injury'},
        'd': {'raw_status': 'd', 'raw_news': 'Hip injury'},
        'e': {'raw_status': 'i', 'raw_news': 'thigh injury'}}
    now = {'news_id': 'a', 'source_team_code': 3, 'source_player_code': 111}
    assert source_comparison(now, {'news_id': 'b'}, catalogs) == PAIRED[0]
    assert source_comparison(now, {'news_id': 'c'}, catalogs) == PAIRED[1]
    assert source_comparison(now, {'news_id': 'd'}, catalogs) == PAIRED[2]
    assert source_comparison(now, {'news_id': 'a'}, catalogs) == PAIRED[3]
    assert source_comparison(now, {'news_id': 'e'}, catalogs) == PAIRED[0]
    old_other_team = {'snapshot_id': 'old', 'source_team_code': 6, 'source_player_code': 111,
                     'observation_id': 'other', 'news_id': 'b'}
    _, teams, codes, duplicates = build_indexes([old_other_team])
    ref = {'snapshot_id': 'old', 'observed': True, 'source_team_code': 3}
    assert classify_pair(now, ref, teams, codes, catalogs) == (
        'other_team_same_code_transfer_unknown', None, ['other'])
    assert classify_pair(now, dict(ref, observed=False), teams, codes, catalogs)[0] == 'reference_record_missing'
    old_same_team = dict(old_other_team, source_team_code=3, observation_id='one')
    repeated = dict(old_same_team, observation_id='two')
    _, teams, codes, duplicates = build_indexes([old_same_team, repeated])
    assert len(duplicates) == 1
    assert classify_pair(now, ref, teams, codes, catalogs)[0] == 'duplicate_reference_identity'
    synthetic = {'rows': [{'match_id': 'm', 'actual_score': [8, 9], 'actual_outcome': 2,
                           'fresh_feature_history': {'history': {'home': {'window_match_ids': {'5': ['old']}}}}}]}
    selected = {'rows': {'*': {'match_id': True, 'fresh_feature_history': True}}}
    original = project_json(json.dumps(synthetic), selected)
    synthetic['rows'][0].update(actual_score=[0, 0], actual_outcome=0)
    assert project_json(json.dumps(synthetic), selected) == original
    return {'cutoff_equal_excluded': True, 'same_news_across_different_status_news_ids': True,
        'text_change_not_status_change': True, 'case_change_remains_text_change': True,
        'cross_team_same_code_not_paired': True, 'missing_reference_not_healthy_or_zero': True,
        'duplicate_same_team_code_explicitly_unknown': True,
        'target_label_mutation_does_not_change_projected_inputs': True}


def main():
    checks = synthetic_checks()
    contexts = read(CONTEXT)['rows']
    context_by_id = {r['match_id']: r for r in contexts}
    if len(context_by_id) != len(contexts):
        raise ValueError('Context match ID duplicated')
    windows = load_original_windows()
    if set(windows) != set(context_by_id):
        raise ValueError('Original history and full cohort differ')
    snapshots = read(DATA/'来源快照目录.json')['snapshots']
    snapshots_by_path = {s['path']: s for s in snapshots}
    raw_catalog = {r['news_id']: r for r in read(DATA/'源新闻目录.json')['records']}
    semantics, semantic_source = load_semantics()
    if set(semantics) - set(raw_catalog):
        raise ValueError('Semantics contains an unknown raw news reference')
    by_id, team_players, by_code, duplicates = build_indexes(read(DATA/'快照球员新闻引用.json.gz')['observations'])
    match_references = read(DATA/'比赛球员新闻引用.json.gz')['references']
    current_refs = defaultdict(set)
    for ref in match_references:
        item = by_id[ref['observation_id']]
        row = context_by_id[ref['match_id']]
        team = next(t for t in row['teams'] if t['side'] == ref['side'])
        snapshot = snapshots_by_path[row['archive_observation']['path']]
        if (ref['forecast_before_date'] != row['forecast_before_date'] or ref['team'] != team['team']
                or ref['snapshot_id'] != snapshot['snapshot_id']
                or item['snapshot_id'] != ref['snapshot_id']
                or item['source_team_code'] != team['source_team_code']
                or item['source_player_code'] != ref['source_player_code']
                or item['source_player_id'] != ref['source_player_id']
                or item['news_id'] != ref['news_id']):
            raise ValueError('Current match reference disagrees with source observation')
        key = (ref['match_id'], ref['side'])
        if ref['observation_id'] in current_refs[key]:
            raise ValueError('Current match/player reference duplicated')
        current_refs[key].add(ref['observation_id'])
    prepared = []
    coverage = defaultdict(Counter)
    overall = Counter()
    transition_examples = {key: [] for key in PAIRED}
    for row in contexts:
        out = {key: row[key] for key in ('match_id', 'date', 'period', 'competition', 'home', 'away', 'forecast_before_date')}
        out['teams'] = [team_context(row, side, windows, context_by_id, snapshots_by_path,
            team_players, by_code, raw_catalog, semantics) for side in SIDES]
        # These are the existing measurements and notes, unchanged, not new research.
        out['existing_participation_change_features'] = row['existing_participation_change_features']
        out['existing_participation_diagnostics'] = row['existing_participation_diagnostics']
        for team in out['teams']:
            c = Counter(team_sides=1, observed_team_sides=int(team['observed']))
            if team['observed']:
                obs_ids = {p['current']['observation_id'] for p in team['players'] if p['current']}
                obs_ids.update(p['observation_id'] for pl in team['players']
                               for p in pl.get('current_candidates', []))
                if obs_ids != current_refs[(row['match_id'], team['side'])]:
                    raise ValueError('Current roster and target news references differ')
                c['current_player_contexts'] = len(team['players'])
                c['reference_sources_observed'] = sum(r['observed'] for r in team['reference_sources'])
                c['all_five_reference_sources_observed_team_sides'] = int(
                    len(team['reference_sources']) == 5 and all(r['observed'] for r in team['reference_sources']))
                c['reference_only_player_records'] = sum(len(r['reference_only_players'] or []) for r in team['reference_sources'])
                c.update(team['descriptive_comparison_counts'])
                c['all_five_paired_player_contexts'] = sum(len(p['references']) == 5 and
                    all(r['classification'] in PAIRED for r in p['references']) for p in team['players'])
                for player in team['players']:
                    for ref in player['references']:
                        category = ref['classification']
                        if category in transition_examples and len(transition_examples[category]) < 3:
                            transition_examples[category].append({'match_id': row['match_id'],
                                'forecast_before_date': row['forecast_before_date'], 'team': team['team'],
                                'source_team_code': team['source_team_code'],
                                'source_player_code': player['source_player_code'], 'web_name': player.get('web_name'),
                                'current': player['current'], 'reference': ref})
            else:
                c['unknown_team_sides'] = 1
            coverage[row['period']].update(c)
            overall.update(c)
        prepared.append(out)
    created = datetime.now(timezone.utc).isoformat()
    document = {'created_at_utc': created,
        'scope': 'Descriptive exact-code/same-team source-state and exact-news alignment at original forecast cutoffs; no model fit',
        'sources': {'existing_context': str(CONTEXT), 'original_window_projection': str(HISTORY),
            'snapshot_observations': str(DATA/'快照球员新闻引用.json.gz'),
            'match_references': str(DATA/'比赛球员新闻引用.json.gz'),
            'snapshot_catalog': str(DATA/'来源快照目录.json'), 'raw_news_catalog': str(DATA/'源新闻目录.json'),
            'semantic_catalog': semantic_source},
        'news_semantics': semantics,
        'comparison_policy': {'identity': 'same source_player_code AND same source_team_code; never names or cross-season local IDs',
            'news': 'exact raw_news equality, preserving case/whitespace; news_id also encodes raw_status and is not a text-equality key',
            'states': 'exact raw source status equality, including n/u; source a is not confirmed fitness',
            'time': 'current proxy < target F; reference proxy < reference F; reference match date < target F',
            'missing': 'Unobserved team retains observed=false and counts=null; no healthy/zero imputation',
            'counts': 'Descriptive repeated source-pair counts; never illness incidence, independent players, strength or model features'},
        'rows': prepared}
    with gzip.open(OUT, 'wt', encoding='utf-8') as handle:
        json.dump(document, handle, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    summary = {'created_at_utc': created, 'rows': len(prepared), **dict(overall),
        'paired_comparison_counts': {key: overall[key] for key in PAIRED},
        'paired_comparisons_total': sum(overall[key] for key in PAIRED),
        'by_period': {period: dict(c) for period, c in sorted(coverage.items())},
        'source_observation_count': len(by_id), 'source_snapshot_count': len(snapshots),
        'current_news_reference_count': len(match_references),
        'same_snapshot_team_player_code_duplicate_groups': duplicates,
        'missing_stable_source_player_codes': sum(p['source_player_code'] is None for p in by_id.values()),
        'existing_participation_context_rows_preserved': sum(r['existing_participation_change_features'] is not None for r in prepared),
        'existing_participation_features_per_side': 22,
        'semantic_catalog': semantic_source,
        'semantic_player_references_joined': sum(p['current'] is not None and p['current']['semantic_news_id'] is not None
             for r in prepared for t in r['teams'] for p in t['players']),
        'directed_checks': checks, 'examples': transition_examples,
        'sources': document['sources'],
        'target_labels_used_in_alignment': False, 'target_labels_exported': False,
        'feature_inputs_projection': 'Only match_id and home/away original past-five IDs are decoded from the mixed feature/target file',
        'format_inspection_limitation': 'An initial raw-prefix format inspection exposed the first historical row target labels; no labels enter alignment computation or outputs',
        'new_model_fits': 0, 'prediction_updates': 0, 'target_actual_lineups_used': False,
        'limits': ['Text changes are source wording changes, not dated onset or recovery.',
            'Semantic labels reference source text; they are not clinical diagnoses or player ability.',
            'Neither current nor historical FPL source a establishes health or selection.',
            'Missing/new/other-team records remain unknown and never become healthy or zero.',
            'No cross-team local-ID or name match; other-team same-code references only document identity uncertainty.',
            'Counts repeat players and snapshots across matches and five references; they are not independent sample size.',
            'Existing 22 participation measurements and diagnostics are retained; observed minutes are not re-estimated contributions.',
            'Archive proxy is retrospective version evidence, not independently verified first public announcement time.',
            'Only the existing English FPL coverage is observed; earlier seasons and other competitions remain unobserved.']}
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: summary[key] for key in ('rows', 'observed_team_sides', 'unknown_team_sides',
        'current_player_contexts', 'paired_comparison_counts', 'paired_comparisons_total',
        'semantic_catalog', 'semantic_player_references_joined')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
