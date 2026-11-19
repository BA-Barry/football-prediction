"""Record the first actual five-line case after both one-shot LLM runs."""
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

from football_model.domain import Fixture, OddsSnapshot
from football_model.multiline import arithmetic_pool, multi_record
from football_model.odds_history import SourcedOddsSnapshot, select_asof
from football_model.prospective import market_track

ROOT = REPOSITORY_ROOT
DEST = ROOT / 'data/prematch-intelligence/20261024_multiline'
preparation = json.loads((DEST / 'preparation.json').read_text(encoding='utf-8'))
market = json.loads((DEST / 'market_observation.json').read_text(encoding='utf-8'))
fixture = Fixture(**(preparation['fixture'] | {'kickoff': datetime.fromisoformat(preparation['fixture']['kickoff'])}))
responses, tracks, reasons = {}, {'statistical': preparation['statistical_track']}, {}
for name, arm, mode in [('independent', 'llm_independent', False), ('model_informed', 'llm_model_informed', True)]:
    response = json.loads((DEST / f'{name}_response.json').read_text(encoding='utf-8'))
    responses[arm] = response
    if response['probabilities'] is None:
        reasons[arm] = 'abstained: ' + response['reasoning_summary']
        tracks[arm] = None
    else:
        tracks[arm] = {'id': f"{arm}:{response['generated_at_utc']}", 'fixture_id': fixture.match_id,
                       'generated_at': response['generated_at_utc'], 'probabilities': response['probabilities'],
                       'model': 'Codex agent; exact runtime model identifier unknown',
                       'exposed_to_statistical': response['exposed_to_statistical'],
                       'exposed_to_market': response['exposed_to_market'],
                       'description': response['calibration_status']}
        assert response['exposed_to_statistical'] is mode
snapshot = OddsSnapshot(**market['decimal_prices'], observed_at=datetime.fromisoformat(market['observed_at']),
                        bookmaker=market['bookmaker'], fixture_id=fixture.match_id)
sourced = SourcedOddsSnapshot(snapshot, market['source'], market['source_url'], market['source_event_id'])
calculation_start = datetime.now(timezone.utc)
history = select_asof([sourced], fixture=fixture, bookmaker=market['bookmaker'],
                      source=market['source'], as_of=calculation_start)
tracks['market'] = market_track([sourced], fixture=fixture, bookmaker=market['bookmaker'],
                                source=market['source'], as_of=calculation_start)
tracks['market']['id'] = f"market:{market['bookmaker']}:{market['observed_at']}"
tracks['joint_pool'], pool_reason = arithmetic_pool(tracks['statistical'], tracks['llm_independent'],
                                                     preparation['pool_policy'], fixture, calculation_start)
if pool_reason:
    reasons['joint_pool'] = pool_reason
record = multi_record(fixture, datetime.now(timezone.utc), tracks=tracks, missing_reasons=reasons)
payload = {'pilot_id': '20261024-five-line-first-prospective-case', 'records': [record], 'results': {},
           'llm_responses': responses, 'statistical_projection_path': preparation['projection_path'],
           'market_observation': market, 'market_history': history,
           'packet_paths': {name: str(DEST / f'{name}_packet.json') for name in ('independent', 'model_informed')},
           'policy': {'pool_weights': preparation['pool_policy'], 'weight_note': preparation['pool_policy_note'],
                      'llm_runs_with_new_data': 2, 'old_abstention_preserved': True,
                      'same_vendor_model_comparison': 'Two isolated information policies on existing Codex agents; exact runtime model IDs unknown, not a multi-vendor comparison.'},
           'notes': ['This replaces no previous record: the earlier one-shot abstention remains in the v0.28 case.',
                     'New descriptive seven-round and prior-season evidence was supplied before these two new runs, not a same-prompt resample until numerical output.',
                     'All five predictions precede kickoff. A shared recording deadline does not mean news was exhaustively monitored between input preparation and this deadline.',
                     '54 current league scores plus 1520 historical league matches; other16 current results missing. No player or motivation numerical effects were fitted.',
                     'Only1X2 is comparable across all lines. LLM1X2 does not identify a unique score, totals or half/full-time distribution.',
                     'No efficacy score or profitability is available before the real match is settled.']}
path = DEST / 'multiline_case.json'
with path.open('x', encoding='utf-8') as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
print(json.dumps({'case': str(path), 'record_as_of': record['as_of'],
                  'tracks': {name: None if item is None else item['probabilities'] for name, item in record['tracks'].items()},
                  'pool_weights': preparation['pool_policy'], 'source_quote_age_seconds': history['latest_observed']['age_seconds']}, ensure_ascii=False))
