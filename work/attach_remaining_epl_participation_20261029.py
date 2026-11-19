"""Extend existing same-team participation intake to the other seven EPL fixtures.

The bootstrap/news/fixture identities retain their original observation times.
Only the selected player's current-season participation endpoint is requested.
No score, future lineup, prediction, fitted player effect or market is output.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import sys

from attach_live_fpl_participation_20261029 import (
    OTHER_CATEGORIES, fetch, fixture_side, now, player_record, read, utc, write)

ROOT = REPOSITORY_ROOT
SOURCE = ROOT / 'data/epl_personnel_live_20261029'
OUT = ROOT / 'data/epl_remaining_personnel_20261029'
OLD = ROOT / 'data/current_rich_inputs/run_20261026T025507813598Z'
FACTS = ROOT / 'outputs/英超人员协作前瞻_20261029/给LLM的表现事实包.json'


def scope():
    bootstrap = read(SOURCE / 'raw/bootstrap-static.json')
    personnel = read(SOURCE / '当前球员新闻.json')
    old = read(OLD / 'official-fpl/bootstrap.json')
    old_report = read(OLD / 'coverage_report.json')
    fixtures = [r['fixture'] for r in read(FACTS)['records'][3:]]
    if len(fixtures) != 7:
        raise ValueError('This extension retains the original other seven fixtures.')
    teams = {t['id']: t for t in bootstrap['teams']}
    wanted_ids = {f[side+'_source_team_id'] for f in fixtures for side in ('home', 'away')}
    wanted_codes = {teams[k]['code'] for k in wanted_ids}
    if len(wanted_codes) != 14:
        raise ValueError('Expected fourteen distinct source teams.')
    old_teams = {t['id']: t for t in old['teams']}
    old_players = {(old_teams[p['team']]['code'], p['code']): p for p in old['elements']}
    selected, excluded = [], []
    for p in personnel['players']:
        if p['source_team_code'] not in wanted_codes:
            continue
        if p['source_team_id'] not in wanted_ids or teams[p['source_team_id']]['code'] != p['source_team_code']:
            raise ValueError('Current source team ID and stable code differ.')
        previous = old_players.get((p['source_team_code'], p['source_player_code']))
        status = None if previous is None else previous.get('status')
        reason = ('current_source_i_d_s' if p['source_status'] in ('i','d','s') else
                  'current_a_after_earlier_i_d_s' if p['source_status'] == 'a' and status in ('i','d','s') else
                  'other_nonmedical_wording' if OTHER_CATEGORIES.intersection(p['semantics']['text_categories']) else None)
        if reason is None:
            if p['source_status'] != 'a':
                excluded.append({'source_player_code': p['source_player_code'],
                    'source_team_code': p['source_team_code'], 'source_status': p['source_status'],
                    'text_categories': p['semantics']['text_categories'],
                    'reason': 'No declared selection condition; transfer-only source-u is not an injury.'})
            continue
        selected.append(dict(p, selection_reason=reason, earlier_snapshot_observation={
            'path': str(OLD / 'official-fpl/bootstrap.json'),
            'run_started_at_utc': old_report['created_at'], 'run_completed_at_utc': old_report['completed_at'],
            'same_team_code_and_player_code': previous is not None,
            'earlier_status': status, 'earlier_news': None if previous is None else previous.get('news'),
            'earlier_news_added': None if previous is None else previous.get('news_added'),
            'current_status': p['source_status'],
            'interpretation': 'Two observed source versions, not medical onset, recovery, strength loss or actual return.'}))
    identities = [(p['source_team_code'], p['source_player_code']) for p in selected]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate selected player identity.')
    summary = {'eligible_before_endpoint_limit': len(selected), 'selected_endpoint_count': len(selected),
        'other_nonmedical_excluded_because_over_40': False,
        'by_selection_reason': dict(Counter(p['selection_reason'] for p in selected)),
        'by_team': dict(Counter(p['source_team_name'] for p in selected)),
        'source_team_ids': sorted(wanted_ids), 'source_team_codes': sorted(wanted_codes),
        'original_fixture_ids': [f['fixture_id'] for f in fixtures],
        'selected': [{'id':p['source_player_id'], 'code':p['source_player_code'], 'name':p['web_name'],
            'team':p['source_team_name'], 'source_status':p['source_status'], 'reason':p['selection_reason']} for p in selected],
        'other_source_flags_excluded': excluded,
        'selection_note': 'Same source rules as earlier intake; no artificial forty-player cap in this fourteen-team extension.'}
    return selected, summary, teams, wanted_ids


def main():
    selected, selection, teams, wanted_ids = scope()
    if '--count-only' in sys.argv:
        print(json.dumps(selection, ensure_ascii=False, indent=2))
        return
    destination = OUT / '较早参与记录.json'
    if destination.exists():
        print('Existing observation retained; create a new version for another observation.')
        return
    started = now()
    cutoff = utc(started)
    fixtures = read(SOURCE / 'raw/fixtures.json')
    past = [f for f in fixtures if f.get('finished') is True and f.get('kickoff_time')
            and utc(f['kickoff_time']) < cutoff]
    by_id = {f['id']:f for f in past}
    recent_by_team = {}
    for team_id in sorted(wanted_ids):
        rows = [f for f in past if team_id in (f['team_h'], f['team_a'])]
        rows.sort(key=lambda f:(f['kickoff_time'],f['id']), reverse=True)
        recent_by_team[team_id] = [fixture_side(f,team_id,teams) for f in reversed(rows[:5])]
    observed = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {pool.submit(fetch, p, OUT):p for p in selected}
        for job in as_completed(jobs):
            p = jobs[job]
            observed[p['source_player_id']] = job.result()
    players = [player_record(p,*observed[p['source_player_id']],recent_by_team[p['source_team_id']],by_id,teams)
               for p in selected]
    completed = now()
    rows = [r for p in players for r in p['previous_five_same_team_league_games']]
    for p in players:
        if any(r['source_team_code'] != p['source_team_code'] for r in p['previous_five_same_team_league_games']):
            raise ValueError('A current-team fixture references a different team code.')
        for r in p['previous_five_same_team_league_games']:
            if r['history_record_state'] != 'present' and (r['minutes'] is not None or r['starts'] is not None):
                raise ValueError('Unknown rows must remain unknown rather than zero.')
    stats = {'selected_endpoints':len(players), 'request_attempt_count':len(players),
        'http_status_counts':dict(Counter(p['element_summary_request'].get('http_status') for p in players)),
        'download_status_counts':dict(Counter(p['element_summary_request']['status'] for p in players)),
        'team_fixture_window_count':len(recent_by_team), 'side_fixture_rows':len(rows),
        'history_record_states':dict(Counter(r['history_record_state'] for r in rows)),
        'real_source_zero_minute_rows':sum(r['history_record_state']=='present' and r['minutes']==0 for r in rows),
        'unknown_minute_rows':sum(r['minutes'] is None for r in rows),
        'complete_five_players':sum(p['previous_five_summary']['complete_five_rows_and_fields'] for p in players),
        'rejected_history_mapping_rows':sum(len(p['rejected_history_mapping_rows']) for p in players)}
    value = {'batch_started_at_utc':started, 'observation_completed_at_utc':completed,
        'past_fixture_cutoff_at_utc':started,
        'bootstrap_observation_at_utc':read(SOURCE/'获取记录.json')['observation_completed_at_utc'],
        'bootstrap_news_source_observation_at_utc':read(SOURCE/'当前球员新闻.json')['observed_at_utc'],
        'source_files':{'bootstrap':str(SOURCE/'raw/bootstrap-static.json'), 'news':str(SOURCE/'当前球员新闻.json'),
            'fixtures':str(SOURCE/'raw/fixtures.json'), 'original_descriptive_facts':str(FACTS)},
        'source_urls':['https://fantasy.premierleague.com/api/bootstrap-static/','https://fantasy.premierleague.com/api/fixtures/']+
            [p['element_summary_request']['source_url'] for p in players],
        'selection':selection,
        'last_five_fixtures_by_current_team':[{'source_team_id':tid,'source_team_code':teams[tid]['code'],
            'source_team_name':teams[tid]['name'],'fixtures':r} for tid,r in recent_by_team.items()],
        'players':players, 'observed_counts':stats,
        'limits':['Source bootstrap/news identities retain their 11:05:23 UTC observation; endpoint history was observed at the separately recorded later time.',
            'No refreshed bootstrap/news observation or target-match availability confirmation is claimed.',
            'Source flags and changes between snapshots do not confirm health, onset, recovery, lineup or future availability.',
            'Prior same-team league history maps through fixture, was_home, opponent and player identity; rows for another team are excluded.',
            'Missing or ambiguous rows and fields stay null; zero minutes is a real recorded value, not a missing-value replacement.',
            'Totals require exactly five valid same-team fixture rows with minutes and starts.',
            'No target result, target lineup, market, source chance calibration, current model probability or fitted personnel effect is projected.']}
    write(destination,value)
    write(OUT/'参与观察汇总.json',{'observation_completed_at_utc':completed,**stats,
        'by_team':selection['by_team'], 'by_selection_reason':selection['by_selection_reason']})
    print(json.dumps({'path':str(destination),'observation_completed_at_utc':completed,**stats},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
