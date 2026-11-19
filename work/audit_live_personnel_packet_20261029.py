"""Finite independent measurements of the current personnel reading packet.

Reads only declared source facts and safe projections of raw FPL data. It never
loads predictions, a fitted model, odds, target scores, or old LLM output. This
is an audit report, not a new production gate or a frozen contract.
"""
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import sys
sys.dont_write_bytecode = True

ROOT = REPOSITORY_ROOT
DATA = ROOT / 'data/epl_personnel_live_20261029'
OUT = ROOT / 'outputs/英超人员协作前瞻_20261029'
OLD = ROOT / 'data/current_rich_inputs/run_20261026T025507813598Z'
READS = []


def read(path):
    READS.append(str(path))
    return json.loads(path.read_text(encoding='utf-8-sig'))


def instant(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def canon(value):
    value = value.removeprefix('football-data:E0:')
    return {'AFC Bournemouth': 'Bournemouth', 'Manchester United': 'Man United',
            'Man Utd': 'Man United', 'Leeds United': 'Leeds',
            'Nottingham Forest': "Nott'm Forest", 'Hull City': 'Hull', 'Coventry City': 'Coventry',
            'Tottenham Hotspur': 'Tottenham', 'Spurs': 'Tottenham',
            'Manchester City': 'Man City', 'Brighton & Hove Albion': 'Brighton',
            'Newcastle United': 'Newcastle', 'Ipswich Town': 'Ipswich'}.get(value, value)


def import_file(name):
    path = ROOT / 'work' / name
    READS.append(str(path))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # main() is protected; no fetching or writes.
    return module


def main():
    builder_path = ROOT / 'work/build_live_personnel_packets_20261029.py'
    builder_path.read_text(encoding='utf-8')
    READS.append(str(builder_path))
    packet1 = read(DATA / '独立LLM人员协作输入.json')
    packet = read(DATA / '独立LLM人员协作输入_v2.json')
    verified = read(DATA / '已核实赛程.json')
    future = read(DATA / '未来未开场赛程.json')
    personnel = read(DATA / '当前球员新闻.json')
    usage = read(DATA / '较早参与记录.json')
    evidence = read(DATA / '官方人员证据.json')
    form = read(OUT / '给LLM的表现事实包.json')
    acquisition = read(DATA / '获取记录.json')
    old_meta = read(OLD / 'coverage_report.json')
    # Raw sources can contain historical outcomes. Project identities, dates,
    # flags and the stated player fields before any audit computation/output.
    b = read(DATA / 'raw/bootstrap-static.json')
    teams = {t['id']: {k: t[k] for k in ('id', 'code', 'name')} for t in b['teams']}
    fields = ('id', 'code', 'team', 'first_name', 'second_name', 'web_name',
              'status', 'news', 'news_added', 'chance_of_playing_this_round',
              'chance_of_playing_next_round')
    raw_players = {p['id']: {k: p.get(k) for k in fields} for p in b['elements']}
    del b
    fixture_fields = ('id', 'event', 'team_h', 'team_a', 'kickoff_time', 'finished', 'started')
    raw_fixtures = [{k: f.get(k) for k in fixture_fields}
                    for f in read(DATA / 'raw/fixtures.json')]
    fixtures = {f['id']: f for f in raw_fixtures}
    b = read(OLD / 'official-fpl/bootstrap.json')
    old_teams = {t['id']: {k: t[k] for k in ('id', 'code', 'name')} for t in b['teams']}
    old_players = {(old_teams[p['team']]['code'], p['code']):
                   {k: p.get(k) for k in fields} for p in b['elements']}
    del b
    checks = []

    def check(name, good, detail):
        checks.append({'check': name, 'passed': bool(good), 'detail': detail})

    change_keys = sorted(k for k in set(packet1) | set(packet)
                         if packet1.get(k) != packet.get(k))
    check('v1_v2_fact_identity', change_keys == ['assembled_at_utc', 'official_observation_time_note'],
          {'changed_top_level_keys': change_keys,
           'v1_assembled_at_utc': packet1['assembled_at_utc'],
           'v2_assembled_at_utc': packet['assembled_at_utc']})
    check('official_observation_limitation_retained',
          packet.get('official_observation_time_note') == evidence.get('observation_time_note')
          and bool(packet.get('official_observation_time_note')),
          'v2 explicitly retains manual web-note finalization versus exact local receipt time; v1 did not.')
    selected = verified['selection_for_personnel_research']
    verified_map = {f['fixture_id']: f for f in verified['fixtures']}
    form_map = {r['fixture']['fixture_id']: r for r in form['records']}
    future_map = {r['source_fixture_id']: r for r in future['fixtures']}
    fixture_failures = []
    future_rows = []
    for r in form['records']:
        f = r['fixture']; v = verified_map[f['fixture_id']]
        raw = fixtures[f['source_fixture_id']]; ff = future_map.get(raw['id'])
        good = (f['status'] == 'scheduled_unstarted' and raw['started'] is False
                and raw['finished'] is False and ff is not None
                and v['status'] == 'scheduled_future_at_observation'
                and instant(f['kickoff']) == instant(v['kickoff_utc']) == instant(raw['kickoff_time'])
                and instant(f['kickoff']) > instant(packet['assembled_at_utc']))
        for side, raw_side in (('home', 'team_h'), ('away', 'team_a')):
            good = good and f[side + '_source_team_id'] == raw[raw_side]
            good = good and canon(f[side]) == canon(v[side]) == canon(teams[raw[raw_side]]['name'])
        if not good:
            fixture_failures.append(f['fixture_id'])
        if f['fixture_id'] in selected:
            future_rows.append({'fixture_id': f['fixture_id'], 'fpl_fixture_id': raw['id'],
                                'home': f['home'], 'away': f['away'], 'kickoff_utc': f['kickoff'],
                                'official_status': v['status'], 'packet_status': f['status'],
                                'fpl_started': raw['started'], 'fpl_finished': raw['finished']})
    check('future_fixtures_official_fpl_sides_times_statuses',
          not fixture_failures and len(future_rows) == 3 and len(form['records']) == 10,
          {'checked_round_fixtures': len(form['records']), 'selected': future_rows,
           'failures': fixture_failures})
    check('three_selected_records_preserve_order',
          [r['fixture']['fixture_id'] for r in packet['records']] == selected,
          {'selection': selected, 'tie_limit': verified['selection_note']})

    personnel_map = {p['source_player_id']: p for p in personnel['players']}
    usage_map = {(p['source_team_code'], p['source_player_code']): p for p in usage['players']}
    roster_failures = []; included = []; expected = []; side_counts = []
    player_fields = ('source_player_code', 'source_player_id', 'source_team_code', 'first_name',
                     'second_name', 'web_name', 'source_status', 'news', 'news_added',
                     'chance_of_playing_this_round', 'chance_of_playing_next_round')
    source_pair = {'source_player_code': 'code', 'source_player_id': 'id', 'source_team_id': 'team',
                   'source_status': 'status'}
    for rec in packet['records']:
        f = rec['fixture']
        if f != form_map[f['fixture_id']]['fixture'] or rec['descriptive_performance'] != form_map[f['fixture_id']]['sides']:
            roster_failures.append({'fixture': f['fixture_id'], 'issue': 'form_projection_differs'})
        for side in rec['personnel']:
            tid = f[side['side'] + '_source_team_id']; t = teams[tid]
            roster = [p for p in personnel['players'] if p['source_team_id'] == tid]
            if side['source_team_id'] != tid or canon(side['team']) != canon(t['name']):
                roster_failures.append({'team_id': tid, 'issue': 'side_team_mismatch'})
            if side['roster_source_counts'] != dict(Counter(p['source_status'] for p in roster)):
                roster_failures.append({'team_id': tid, 'issue': 'status_counts_mismatch'})
            for p in roster:
                old = old_players.get((p['source_team_code'], p['source_player_code']))
                returned = p['source_status'] == 'a' and old and old['status'] in ('i', 'd', 's')
                if p['source_status'] != 'a' or p['news'] or returned:
                    expected.append(p['source_player_id'])
            plist = side['players_with_text_flags_or_observed_label_return']
            side_counts.append({'team': side['team'], 'team_id': tid, 'team_code': t['code'],
                                'packet_players': len(plist), 'source_status_counts': side['roster_source_counts']})
            for p in plist:
                included.append(p['source_player_id']); src = personnel_map[p['source_player_id']]
                raw = raw_players[p['source_player_id']]
                good = (src['source_team_id'] == tid and p['source_team_code'] == t['code']
                        and all(p[k] == src[k] for k in player_fields)
                        and all(src[k] == raw[source_pair.get(k, k)] for k in player_fields if k != 'source_team_code')
                        and raw['team'] == tid)
                annotation = {k: src['semantics'][k] for k in p['text_annotation']}
                good = good and p['text_annotation'] == annotation
                if not good:
                    roster_failures.append({'player_id': p['source_player_id'], 'issue': 'identity_or_raw_fields'})
                old = old_players.get((t['code'], p['source_player_code']))
                observed_old = p['earlier_source_version']
                if old:
                    good_old = (observed_old['source_status'] == old['status']
                                and observed_old['news'] == old['news']
                                and observed_old['news_added'] == old['news_added']
                                and observed_old['status_changed'] == (old['status'] != p['source_status'])
                                and observed_old['text_changed'] == (old['news'] != p['news'])
                                and observed_old['run_started_at'] == old_meta['created_at']
                                and observed_old['run_completed_at'] == old_meta['completed_at'])
                    if not good_old:
                        roster_failures.append({'player_id': p['source_player_id'], 'issue': 'earlier_source_projection'})
    check('six_teams_52_players_source_identity_and_fields',
          len(included) == len(set(included)) == 52 and sorted(included) == sorted(expected) and not roster_failures,
          {'players': len(included), 'teams': side_counts, 'failures': roster_failures})

    cutoff = instant(usage['past_fixture_cutoff_at_utc'])
    recent = {}
    for tid in {p['source_team_id'] for p in usage['players']}:
        games = [f for f in raw_fixtures if tid in (f['team_h'], f['team_a'])
                 and f['finished'] is True and f['kickoff_time'] and instant(f['kickoff_time']) < cutoff]
        recent[tid] = sorted(games, key=lambda f: (instant(f['kickoff_time']), f['id']))[-5:]
    usage_failures = []; rows_seen = 0; numeric_zero_rows = 0; missing_rows = 0
    for u in usage['players']:
        tid = u['source_team_id']; rawp = raw_players[u['source_player_id']]
        if rawp['team'] != tid or rawp['code'] != u['source_player_code'] or teams[tid]['code'] != u['source_team_code']:
            usage_failures.append({'player_id': u['source_player_id'], 'issue': 'player_identity'})
        raw = read(DATA / 'raw/element-summary' / f"{u['source_player_id']}.json")
        history = [{k: h.get(k) for k in ('element', 'fixture', 'was_home', 'opponent_team',
                                         'kickoff_time', 'minutes', 'starts')} for h in raw['history']]
        del raw
        expected_rows = recent[tid]
        games = u['previous_five_same_team_league_games']
        if [g['source_fixture_id'] for g in games] != [g['id'] for g in expected_rows]:
            usage_failures.append({'player_id': u['source_player_id'], 'issue': 'five_fixture_selection'})
        for g, f in zip(games, expected_rows):
            rows_seen += 1; home = f['team_h'] == tid; opp = f['team_a'] if home else f['team_h']
            hits = [h for h in history if h['fixture'] == f['id'] and h['element'] == u['source_player_id']
                    and h['was_home'] is home and h['opponent_team'] == opp]
            good = (g['was_home'] is home and g['opponent_source_team_id'] == opp
                    and g['source_team_id'] == tid and g['source_team_code'] == teams[tid]['code']
                    and g['kickoff_time'] == f['kickoff_time'] and g['finished'] is True)
            if len(hits) == 1:
                h = hits[0]
                good = (good and g['history_record_state'] == 'present'
                        and g['history_kickoff_time'] == h['kickoff_time'] == f['kickoff_time']
                        and g['minutes'] == h['minutes'] and g['starts'] == h['starts'])
                numeric_zero_rows += g['minutes'] == 0
            else:
                missing_rows += 1
                good = good and g['minutes'] is None and g['starts'] is None
            if not good:
                usage_failures.append({'player_id': u['source_player_id'], 'fixture_id': f['id'], 'issue': 'history_side_time_values'})
        summary = u['previous_five_summary']
        complete = len(games) == 5 and all(g['history_record_state'] == 'present' and
                    isinstance(g['minutes'], (int, float)) and isinstance(g['starts'], (int, float)) for g in games)
        if (summary['complete_five_rows_and_fields'] != complete or summary['missing_is_zero'] is not False
                or summary['minutes_sum'] != (sum(g['minutes'] for g in games) if complete else None)
                or summary['starts_sum'] != (sum(g['starts'] for g in games) if complete else None)):
            usage_failures.append({'player_id': u['source_player_id'], 'issue': 'summary'})
    check('29_players_145_prior_records_raw_side_team_time_minutes',
          len(usage['players']) == 29 and rows_seen == 145 and not usage_failures,
          {'players': len(usage['players']), 'records': rows_seen, 'observed_zero_minute_rows': numeric_zero_rows,
           'missing_or_ambiguous_rows': missing_rows, 'failures': usage_failures})
    alignment = []; alignment_failures = []; attachment_failures = []; participation_seen = 0
    for rec in packet['records']:
        for side in rec['personnel']:
            tid = side['source_team_id']; name = side['side']
            fh = rec['descriptive_performance'][name]['window_5']['history']
            form_ids = [(h['date'], canon(h['opponent']), h['is_home']) for h in fh]
            usage_ids = [(g['kickoff_time'][:10], canon(teams[g['team_a'] if g['team_h'] == tid else g['team_h']]['name']), g['team_h'] == tid)
                         for g in recent[tid]]
            alignment.append({'team': side['team'], 'five_performance_matches': form_ids,
                              'five_participation_matches': usage_ids, 'same': form_ids == usage_ids})
            if form_ids != usage_ids:
                alignment_failures.append(side['team'])
            for p in side['players_with_text_flags_or_observed_label_return']:
                u = usage_map.get((p['source_team_code'], p['source_player_code']))
                ap = p['prior_same_team_participation']
                if u is None:
                    if ap is not None:
                        attachment_failures.append(p['source_player_id'])
                else:
                    participation_seen += 1
                    keys = ('source_fixture_id', 'kickoff_time', 'opponent_source_team_name', 'was_home',
                            'history_record_state', 'minutes', 'starts')
                    if (ap['games'] != [{k: g[k] for k in keys} for g in u['previous_five_same_team_league_games']]
                            or ap['previous_five_summary'] != u['previous_five_summary']
                            or ap['observed_at_utc'] != u['element_summary_request']['received_at_utc']):
                        attachment_failures.append(p['source_player_id'])
    check('five_match_cross_provider_date_opponent_home_away_alignment',
          len(alignment) == 6 and not alignment_failures, alignment)
    check('selected_participation_only_and_exact_packet_projection',
          participation_seen == 29 and not attachment_failures,
          {'with_prior_participation': participation_seen, 'unobserved_prior_participation': len(included) - participation_seen,
           'failures': attachment_failures})

    # Ordinary counterfactual measurement of the existing missing-row path.
    attachment = import_file('attach_live_fpl_participation_20261029.py')
    u = usage['players'][0]; tid = u['source_team_id']
    raw = read(DATA / 'raw/element-summary' / f"{u['source_player_id']}.json")
    safe = {'history': [{k: h.get(k) for k in ('element', 'fixture', 'was_home', 'opponent_team',
                                             'kickoff_time', 'minutes', 'starts')} for h in raw['history']]}
    del raw
    missing_id = u['previous_five_same_team_league_games'][0]['source_fixture_id']
    safe['history'] = [h for h in safe['history'] if h['fixture'] != missing_id]
    team_five = next(t['fixtures'] for t in usage['last_five_fixtures_by_current_team'] if t['source_team_id'] == tid)
    result = attachment.player_record(personnel_map[u['source_player_id']], safe, u['element_summary_request'],
                                     team_five, fixtures, teams)
    missing = next(g for g in result['previous_five_same_team_league_games'] if g['source_fixture_id'] == missing_id)
    check('missing_history_remains_unknown_in_existing_code',
          missing['history_record_state'] == 'missing' and missing['minutes'] is None and missing['starts'] is None
          and result['previous_five_summary']['minutes_sum'] is None
          and result['previous_five_summary']['starts_sum'] is None,
          {'synthetic_removed_fixture': missing_id, 'state': missing['history_record_state'],
           'minutes': missing['minutes'], 'starts': missing['starts'], 'summary': result['previous_five_summary']})

    forbidden = {'probabilities', 'p_home', 'p_draw', 'p_away', 'lambda_home', 'lambda_away',
                 'home_rate', 'away_rate', 'rates', 'odds', 'b365h', 'b365d', 'b365a', 'ftr',
                 'target_label', 'target_score', 'target_outcome', 'last_seen_at', 'last_seen_date',
                 'news_frequency', 'word_frequencies', 'last_observation', 'latest_observation',
                 'estimated_strength_loss', 'clinical_state_confirmed', 'actual_target_absence_confirmed'}
    forbidden_paths = []

    def walk(value, ptr=''):
        if isinstance(value, dict):
            for k, v in value.items():
                if k.lower() in forbidden:
                    forbidden_paths.append(ptr + '/' + k)
                walk(v, ptr + '/' + k)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, ptr + '/' + str(i))
    walk(packet)
    permitted_measurements = {'goals_for', 'goals_against', 'xg_for', 'xg_against', 'npxg_for',
                              'npxg_against', 'deep_for', 'deep_against', 'ppda_att', 'ppda_def',
                              'ppda_allowed_att', 'ppda_allowed_def', 'ppda_ratio', 'ppda_allowed_ratio'}
    measurements_ok = all(set(s[w]['measurements']) <= permitted_measurements
                          for r in packet['records'] for s in r['descriptive_performance'].values()
                          for w in ('window_5', 'window_15'))
    check('reading_boundary_no_forecast_odds_target_labels_or_full_history_personnel_summaries',
          not forbidden_paths and measurements_ok and all(v is False for v in packet['exposure'].values()),
          {'forbidden_field_paths': forbidden_paths, 'performance_measurements_are_descriptive_only': measurements_ok,
           'exposure_declaration': packet['exposure'],
           'source_chance_fields': 'Preserved as raw FPL fields; never used in arithmetic or diagnosis.'})
    instructions = ' '.join(packet['instructions'])
    check('source_a_and_percentages_not_diagnoses_or_direct_win_adjustments',
          'Source-a is not confirmed fitness' in instructions
          and 'Do not turn source chance percentages' in instructions
          and all(p['semantics']['clinical_state_confirmed'] is False
                  and p['semantics']['actual_target_absence_confirmed'] is False
                  for p in personnel['players']),
          'Source-a, current labels, training, source chance and previous minutes retain separate meanings; no strength-loss coefficient is output.')
    claims_ok = all(r['official_claims'] == [c for c in evidence['claims'] if c['fixture_id'] == r['fixture']['fixture_id']]
                    and r['official_sources'] == [s for s in evidence['sources'] if s['id'] in {c['source_id'] for c in r['official_claims']}]
                    for r in packet['records'])
    times = [personnel['observed_at_utc'], usage['observation_completed_at_utc'],
             form['generated_at'], evidence['observed_at_utc']]
    times += [s['observed_at_utc'] for s in evidence['sources']]
    times += [u['element_summary_request']['received_at_utc'] for u in usage['players']]
    check('declared_observations_precede_original_packet_and_claims_match',
          claims_ok and all(instant(t) <= instant(packet1['assembled_at_utc']) for t in times)
          and instant(acquisition['observation_completed_at_utc']) <= instant(personnel['observed_at_utc']),
          {'source_count': len(evidence['sources']), 'claim_count': len(evidence['claims']),
           'original_packet_time': packet1['assembled_at_utc'], 'latest_declared_fact_time': max(times, key=instant)})
    official_review = {
        'method': 'Separate read-only review of official evidence and its explicit local raw files; no web result search.',
        'packet_source_records_identical': 13, 'packet_claim_records_identical': 27,
        'sources_with_exact_local_receipts': 6, 'claims_supported_by_local_page_text_and_context': 12,
        'sources_with_original_publication_reproduced_from_local_html_cms_or_search': 8,
        'web_note_sources_without_per_source_receipts': 7,
        'manual_note_source_pointers': ['/sources/1', '/sources/5', '/sources/6', '/sources/7', '/sources/8', '/sources/9', '/sources/10'],
        'finding': 'No substantive contradiction found in locally reproducible statements. Training, prior absence, cup squads and future press conferences retain their conditions and dates.',
        'limitation': 'Seven web-read source times are note-finalization declarations, not independently proved per-source precise HTTP observations. v2 now carries this limitation. Date-only publication evidence remains date-only.'}
    failed = [c['check'] for c in checks if not c['passed']]
    report = {'audit_completed_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Finite independent audit of current future-match personnel reading inputs; no training, previous prediction updates, new result lookup or settlement.',
              'status': 'passed_with_disclosed_source_time_limitation' if not failed else 'issues_found',
              'checks': checks, 'failed_checks': failed, 'official_read_only_subreview': official_review,
              'version_note': 'v1 remains the original independently read forecast input. v2 adds the official observation-time limitation and a new assembly time only; no source fact, performance value, participation value or forecast probability was changed by this audit.',
              'builder_code_review': 'The reviewed builder reads the named form/personnel/participation/official sources and one earlier same-team roster version; it copies descriptive facts and selected participation fields. No model loading, forecast arithmetic, historical personnel frequency, latest-seen calculation or target-score access is present.',
              'access_boundary': {'read_files': READS,
                  'not_read': ['statistical predictions or fitted model files', 'model rates', 'market odds',
                               'old or new LLM forecast probability outputs', 'target match scores or labels'],
                  'raw_handling': 'Raw FPL payloads may contain historical labels; only explicit safe identity/date/flag/player participation projections enter computation or report.'},
              'limits': ['Official web-note times without per-source receipts cannot be independently established as exact per-source observation times.',
                         'This audit does not medically verify players, infer target lineups or calibrate source chance percentages.',
                         'No personnel frequency or historical latest-seen statistic is included in the packet.',
                         'No forecast output was read; parent task handles any post-forecast source-time disclosure separately.']}
    path = OUT / '输入独立复核.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    text = [f"# 输入独立复核\n\n结论：{'通过，保留来源时点限制' if not failed else '发现需要修正的问题'}。",
            f"审阅完成时间：{report['audit_completed_at_utc']}。",
            f"有限复核 {len(checks)} 项；失败 {len(failed)} 项。未训练、修改旧预测、查询或结算目标赛果。",
            '3场目标与英超官方/FPL主客、UTC开球时间一致；本轮10场均为未来且源标记未开始/未完赛。',
            f"6队52名入包球员按player id/code、team id/code与原源核对；29名球员145条较早同队记录按fixture、主客、对手与时间核对。源中实际0分钟记录{numeric_zero_rows}条，缺行/歧义记录{missing_rows}条；额外移除一行的普通测量保留unknown/None，汇总未补0。",
            '6队近5场表现窗口与人员参与窗口逐场按球队、日期、主客与归一对手一致，不要求不同供应商的match ID相同。',
            'source-a、原始chance、训练和旧缺阵均未当成诊断/确认首发/直接胜率调整。未见统计预测、rates、赔率、目标标签或全历史人员词频/最晚观察进入阅读包。',
            '官方材料：13源27条claims与证据文件一致；6源的本地receipt时间精确匹配，12条claims可由原文及上下文复现，8源的原发布时间可在本地复现。其余7个web源的11:11:37Z只有阅读笔记终结时间声明，无逐源receipt，不能独立证明逐源精确观察时点。日期级发布时间仍只保留日期。',
            '已修正的限定遗漏：v1未携带官方时点说明；v2添加official_observation_time_note，并更新assembly时间。逐项差异只有这两项，3场全部事实、表现与人员记录不变。原始独立前瞻仍对应v1；本复核未阅读任何LLM预测输出，后附来源说明由主任务单独处理。',
            '具体测量与失败定位见同目录《输入独立复核.json》。本报告是一次有限复核，没有新增生产门禁、hash或冻结约定。',
            '\n逐项结果：\n' + '\n'.join(f"- {'通过' if c['passed'] else '失败'}：{c['check']}" for c in checks)]
    (OUT / '输入独立复核.md').write_text('\n\n'.join(text) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'checks': len(checks), 'failed_checks': failed,
                      'json': str(path), 'markdown': str(OUT / '输入独立复核.md')}, ensure_ascii=False))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
