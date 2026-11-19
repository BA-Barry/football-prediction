"""Add actual assisted forecasts to existing prospective fixture records."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

from football_model.domain import Fixture
from football_model.multiline import multi_record, score_multi

ROOT = REPOSITORY_ROOT
OUT = ROOT / 'outputs/英超人员协作前瞻_20261029'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def resolve(node, pointer):
    if not pointer.startswith('/'):
        raise ValueError('Citations need absolute JSON Pointers.')
    for item in pointer.split('/')[1:]:
        key = item.replace('~1', '/').replace('~0', '~')
        node = node[int(key)] if isinstance(node, list) else node[key]
    return node


def main():
    destination = OUT / '条件表现协作前瞻记录.json'
    if destination.exists():
        raise FileExistsError('Retain the existing prospective publication.')
    original = read(OUT / '人员协作前瞻记录.json')
    prediction_path = OUT / '模型辅助LLM条件表现预测.json'
    llm = read(prediction_path)
    packet_path = ROOT / 'data/epl_personnel_live_20261029/模型辅助LLM条件表现输入.json'
    packet = read(packet_path)
    facts = read(ROOT / 'data/epl_personnel_live_20261029/独立LLM人员协作输入_v2.json')
    review = read(OUT / '对手条件接入独立复核.json')
    if Path(llm['input_packet']) != packet_path:
        raise ValueError('Forecast source differs from the actual packet.')
    expected_exposure = {'current_statistical_probabilities': False,
        'historical_model_estimates': True, 'market': False, 'other_llm': False,
        'target_results': False}
    if packet['exposure'] != expected_exposure or llm['exposure'] != expected_exposure:
        raise ValueError('The new forecast must disclose the historical model context.')
    if llm['forecast_type'] != 'model_assisted_llm':
        raise ValueError('Do not relabel assisted forecasts as independent.')
    if len(packet['records']) != 3 or len(llm['records']) != 3:
        raise ValueError('This integration covers exactly the original three fixtures.')
    for old, new in zip(facts['records'], packet['records']):
        if any(new[k] != v for k, v in old.items()):
            raise ValueError('Original observed performance/personnel facts changed.')
    fixtures = {r['fixture']['fixture_id']: r['fixture'] for r in packet['records']}
    answers = {r['fixture_id']: r for r in llm['records']}
    if set(answers) != set(fixtures):
        raise ValueError('Prediction IDs differ from the supplied fixture set.')
    cited = 0
    for key, answer in answers.items():
        if answer['source_fixture_id'] != fixtures[key]['source_fixture_id']:
            raise ValueError('Source fixture identity differs.')
        probabilities = [answer['probabilities'][k] for k in ('home_win', 'draw', 'away_win')]
        if any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities):
            raise ValueError('Invalid full 1X2 probabilities.')
        if not math.isclose(math.fsum(probabilities), 1., abs_tol=1e-12):
            raise ValueError('Full 1X2 probabilities must sum to one.')
        for reference in answer['cited_facts']:
            if resolve(packet, reference['pointer']) != reference['value']:
                raise ValueError('A forecast citation does not match its source value.')
            cited += 1
    now = datetime.now(timezone.utc)
    generated = datetime.fromisoformat(llm['generated_at_utc'])
    if not datetime.fromisoformat(packet['assembled_at_utc']) <= generated <= now:
        raise ValueError('Forecast generation must follow its input and precede publication.')
    records = []
    for old in original['records']:
        fixture_data = deepcopy(old['fixture'])
        fixture_data['kickoff'] = datetime.fromisoformat(fixture_data['kickoff'])
        fixture = Fixture(**fixture_data)
        tracks = {k: deepcopy(v) for k, v in old['tracks'].items() if v is not None}
        reasons = deepcopy(old['missing_reasons'])
        answer = answers.get(old['revision']['official_fixture_id'])
        if answer is not None:
            tracks['llm_model_informed'] = {
                'id': 'conditioned-form-llm:'+fixture.match_id+':'+llm['generated_at_utc'],
                'fixture_id': fixture.match_id, 'generated_at': llm['generated_at_utc'],
                'probabilities': [answer['probabilities'][k] for k in ('home_win', 'draw', 'away_win')],
                'model': 'Uncalibrated model-assisted LLM; historical opponent/venue-conditioned xG plus original performance/personnel; '+llm['model_id'],
                'exposed_to_statistical': True, 'exposed_to_market': False}
            reasons.pop('llm_model_informed', None)
        else:
            reasons['llm_model_informed'] = 'Only the original earliest three fixtures received assisted context.'
        reasons['joint_pool'] = 'No numeric pooling: this assisted policy has no earlier prospective paired-result fit. Original independent/statistical tracks remain separate.'
        row = multi_record(fixture, now, tracks=tracks, missing_reasons=reasons)
        for name, track in old['tracks'].items():
            if track is not None and row['tracks'][name] != track:
                raise ValueError('An existing statistical or independent prediction was altered.')
        for name in ('revision', 'source_files', 'additional_statistical_research'):
            row[name] = deepcopy(old[name])
        if answer is not None:
            row['source_files'].update(assisted_forecast=str(prediction_path),
                assisted_input=str(packet_path), historical_context=packet['source_files']['historical_context'])
            row['assisted_exposure_details'] = expected_exposure
            row['assisted_policy'] = 'Historical model estimates are visible; current target statistical probabilities are not. This track is model-assisted, not independent.'
        records.append(row)
    if len(records) != 10 or len({r['fixture']['match_id'] for r in records}) != 10:
        raise ValueError('All ten existing fixture identities must remain distinct.')
    score = score_multi(records, {})
    document = {'published_at_utc': now.isoformat(), 'records': records, 'results': {},
        'unique_existing_fixtures': 10, 'new_unique_fixtures': 0,
        'new_assisted_llm_versions': 3, 'unchanged_statistical_versions': 10,
        'unchanged_independent_llm_versions': 3, 'llm_citations_resolved': cited,
        'source_record': str(OUT / '人员协作前瞻记录.json'),
        'input_policy': 'existing_descriptive_personnel_facts_plus_strictly_earlier_opponent_venue_conditioned_xg',
        'note': 'Three new forecast versions for existing matches; no target results, probability averaging, hyperparameter selection or accuracy claim.'}
    write(destination, document)
    write(OUT / '条件表现协作初始评分.json', score)
    write(OUT / '条件表现接入检查.json', {'completed_at_utc': now.isoformat(),
        'independent_context_review_status': review['status'],
        'original_three_fact_records_unchanged': True, 'original_forecast_tracks_unchanged': True,
        'historical_model_exposure_disclosed': True, 'all_cited_values_resolved': cited,
        'same_ten_fixture_ids': True, 'new_unique_fixtures': 0, 'results_queried': 0,
        'new_probability_pools': 0, 'parameter_searches': 0})
    print(json.dumps({'path': str(destination), 'published_at_utc': now.isoformat(),
        'assisted_forecasts': 3, 'citations_resolved': cited, 'new_unique_fixtures': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
