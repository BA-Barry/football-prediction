"""Record refreshed future forecasts against existing fixture identities."""
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

from football_model.domain import Fixture
from football_model.multiline import multi_record, score_multi

ROOT = REPOSITORY_ROOT
OUT = ROOT/'outputs/英超人员协作前瞻_20261029'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    destination = OUT/'人员协作前瞻记录.json'
    if destination.exists():
        print('Existing record retained; read its stored publication rather than generating a duplicate.')
        return
    stats = read(OUT/'统计预测.json')
    llm = read(OUT/'独立LLM预测.json')
    packet = read(Path(llm['input_packet']))
    supplemental = read(OUT/'LLM来源限定补充.json')
    answers = {r['fixture_id']: r for r in llm['records']}
    now = datetime.now(timezone.utc)
    cited_facts = 0
    for answer in answers.values():
        for reference in answer['cited_facts']:
            node = packet
            for part in reference['pointer'].split('/')[1:]:
                key = part.replace('~1', '/').replace('~0', '~')
                node = node[int(key)] if isinstance(node, list) else node[key]
            if node != reference['value']:
                raise ValueError('A forecast source citation does not resolve to the cited value')
            cited_facts += 1
    if any(value is not False for value in llm['exposure'].values()):
        raise ValueError('This publication expects a clean non-probability-exposed LLM input')
    rows = []
    for record in stats['records']:
        f = record['fixture']
        references = record['revision']['matched_previous_fixture_references']
        original = references[0]['fixture']
        if datetime.fromisoformat(original['kickoff']) != datetime.fromisoformat(f['kickoff']):
            raise ValueError('A revision must keep the real original kickoff')
        fixture = Fixture(**(original | {'kickoff': datetime.fromisoformat(original['kickoff'])}))
        stat = {'id': 'goal-refresh:'+fixture.match_id+':'+stats['generated_at'],
            'fixture_id': fixture.match_id, 'generated_at': stats['generated_at'],
            'probabilities': record['goal_simple']['probabilities'],
            'model': 'Existing EPL DateTeamGoalModel regularization=.01 half_life=180; independent Poisson rho=0; strictly earlier data refreshed 2026-10-29'}
        tracks = {'statistical': stat}
        answer = answers.get(f['fixture_id'])
        if answer is not None:
            if answer['source_fixture_id'] != f['source_fixture_id']:
                raise ValueError('LLM source fixture identity differs')
            tracks['llm_independent'] = {
                'id': 'personnel-llm:'+fixture.match_id+':'+llm['generated_at_utc'],
                'fixture_id': fixture.match_id, 'generated_at': llm['generated_at_utc'],
                'probabilities': [answer['probabilities'][key] for key in ('home_win', 'draw', 'away_win')],
                'model': 'Uncalibrated clean-context LLM; descriptive past form + sourced personnel; '+llm['model_id'],
                'exposed_to_statistical': False, 'exposed_to_market': False}
        reasons = {'market': 'No market observations in this personnel-context run.',
                   'llm_model_informed': 'Assistant synthesis is separate qualitative analysis, not a second numeric LLM forecast.',
                   'joint_pool': 'New context input policy has no earlier paired result fit; qualitative synthesis does not create a numerical pool.'}
        if answer is None:
            reasons['llm_independent'] = 'This context policy was run on the earliest three fixtures only.'
        row = multi_record(fixture, now, tracks=tracks, missing_reasons=reasons)
        row['revision'] = {'same_existing_fixture': True, 'new_unique_fixture': False,
            'original_fixture_sources': references, 'source_fixture_id': f['source_fixture_id'],
            'official_fixture_id': f['fixture_id']}
        row['source_files'] = {'statistics': str(OUT/'统计预测.json'),
            'llm': str(OUT/'独立LLM预测.json') if answer is not None else None,
            'llm_input': llm['input_packet'] if answer is not None else None,
            'later_source_time_annotation': str(OUT/'LLM来源限定补充.json') if answer is not None else None}
        row['additional_statistical_research'] = {key: {k: record[key][k] for k in (
            'rates', 'probabilities', 'over_2_5', 'top_scores')} for key in ('independent_xg', 'rich_121_research')}
        rows.append(row)
    document = {'published_at_utc': now.isoformat(), 'records': rows, 'results': {},
        'unique_existing_fixtures': len(rows), 'new_unique_fixtures': 0,
        'new_context_llm_versions': len(answers), 'llm_citations_resolved': cited_facts,
        'input_policy': 'personnel_semantics_same_team_prior5_official_news_descriptive_form_no_other_probabilities',
        'llm_input_observation_note': supplemental,
        'note': 'New source-time note was read after the original LLM forecast; original v1 input and generation time are retained. These are forecast versions of already recorded fixtures, not additional independent matches.'}
    write(destination, document)
    write(OUT/'前瞻初始评分.json', score_multi(rows, {}))
    print(json.dumps({'path': str(destination), 'records': len(rows),
        'llm_versions': len(answers), 'new_unique_fixtures': 0, 'settled_results': 0,
        'citations_resolved': cited_facts, 'published_at_utc': now.isoformat()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
