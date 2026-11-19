"""Add sourced upcoming European schedules to affected future packets only."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

from football_model.batch_multiline import _fixture
from football_model.multiline import build_packet
from football_model.prospective import _time

ROOT = REPOSITORY_ROOT
ORIGINAL = ROOT / 'data/prematch-intelligence/20261024_batch/run_20261024T150900467071Z'
SUPPLEMENT = ROOT / 'data/prematch-intelligence/NEW20261024_cohort_20261024T150344311377Z/supplemental_europe_schedule_observation.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, payload):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    now = datetime.now(timezone.utc)
    old = read(ORIGINAL / 'batch_case.json')
    records = {r['fixture']['match_id']: r for r in old['records']}
    prior = {p['fixture']['match_id']: p for p in read(ORIGINAL / 'independent_packets.json')['packets']}
    supplement = read(SUPPLEMENT)
    additions = {}
    for row in supplement['fixtures']:
        fid = row['affected_prospective_fixture_id']
        fixture = _fixture(records[fid]['fixture'])
        following = _time(row['kickoff'], 'European kickoff')
        gap = (following - fixture.kickoff).total_seconds() / 3600
        if row['team'] not in (fixture.home, fixture.away) or row['team'] not in (row['home'], row['away']):
            raise ValueError('European source team identity differs')
        if gap <= 0 or abs(gap - row['scheduled_kickoff_gap_hours_after_epl']) > 1e-10:
            raise ValueError('European scheduled gap does not match actual fixture clocks')
        if _time(row['observed_at'], 'European observation') > now:
            raise ValueError('Future observation')
        additions.setdefault(fid, []).append({
            'id': f"europe-next:{row['team']}:{row['kickoff']}", 'url': row['source_url'],
            'published_at': row.get('published_at'), 'observed_at': row['observed_at'],
            'claim_type': 'fact', 'team': row['team'], 'category': 'schedule',
            'status': 'official_source_assertion', 'source_kind': row['source_kind'],
            'claim': row['claim'], 'limits': row['limits'],
            'retrieval_note': 'Newly saved official European schedule supplement; original packet omitted this information. Not confirmed rotation or a measured effect.'})
    destination = ROOT / 'data/prematch-intelligence/20261024_batch' / now.strftime('europe_context_%Y%m%dT%H%M%S%fZ')
    destination.mkdir(parents=True, exist_ok=False)
    packets = {'independent': [], 'model_informed': []}
    for fid in sorted(additions):
        old_packet = prior[fid]
        fixture = _fixture(old_packet['fixture'])
        if now >= fixture.kickoff:
            raise ValueError('New packet must precede kickoff')
        facts = deepcopy(old_packet['evidence']) + additions[fid]
        unknowns = deepcopy(old_packet['unknowns']) + [
            'Nov3-5 next European commitments are now explicitly sourced. Previous unknown full calendar warning remains: this is not an exhaustive all-competition schedule.',
            'Upcoming European travel and time gaps do not prove a team will prioritise Europe or rotate in this EPL fixture. Confirmed lineup, coach statement and actual minutes remain unknown.',
            'This is an information-version study. LLM sampling/context variation can also change forecasts; differences cannot be attributed causally to schedule evidence alone.']
        for mode in packets:
            packets[mode].append(build_packet(fixture, now, facts, unknowns,
                descriptive_context=old_packet['descriptive_context'], mode=mode,
                statistical=records[fid]['tracks']['statistical'] if mode == 'model_informed' else None))
    for mode, rows in packets.items():
        write(destination / f'{mode}_packets.json', {'prepared_at': now.isoformat(), 'mode': mode, 'packets': rows})
    audit = {'prepared_at': now.isoformat(), 'old_case': str(ORIGINAL / 'batch_case.json'),
             'supplement': str(SUPPLEMENT), 'affected_fixtures': sorted(additions),
             'european_team_facts': sum(map(len, additions.values())),
             'unchanged_fixtures_not_rerun': sorted(set(records) - set(additions)),
             'new_unique_fixtures': 0, 'fits': 0, 'parameter_changes': 0,
             'market_refreshed': False, 'original_forecasts_overwritten': False,
             'scope': 'Only affected fixtures are re-forecast in separately dated contexts; not a causal ablation or automatic adoption.'}
    write(destination / '准备与新增赛程说明.json', audit)
    print(json.dumps({'directory': str(destination), **audit}, ensure_ascii=False))


if __name__ == '__main__':
    main()
