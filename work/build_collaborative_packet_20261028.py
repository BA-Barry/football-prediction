"""Assemble a new, explicitly model-informed pre-match discussion input.

Existing statistical and LLM publications are read-only. No forecast averaging,
target outcomes, market data or earlier LLM answers are supplied to this input.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
DEST = ROOT/'data/serie_a_collaborative_20261028'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def build(stat, context, evidence, now):
    statistical = {r['fixture']['match_id']: r for r in stat['records']}
    facts = {r['fixture']['match_id']: r for r in context['records']}
    fixtures = {r['match_id']: r for r in evidence['fixtures']}
    if statistical.keys() != facts.keys() or facts.keys() != fixtures.keys():
        raise ValueError('The three inputs must cover the same fixtures')
    if any(datetime.fromisoformat(r['kickoff_utc']) <= now for r in fixtures.values()):
        raise ValueError('This input is only for unstarted matches')
    if datetime.fromisoformat(stat['generated_at']) > now:
        raise ValueError('Statistical publication is not available yet')
    for source in evidence['sources']:
        if datetime.fromisoformat(source['observed_at_utc']) > now:
            raise ValueError('Evidence observation is not available yet')
    records = []
    for ident, original in statistical.items():
        if original.get('actual_score') is not None or original.get('actual_outcome') is not None:
            raise ValueError('Outcome-bearing records are not pre-match discussion inputs')
        fixture = original['fixture']
        if fixture['kickoff'] != fixtures[ident]['kickoff_utc']:
            raise ValueError('Fixture kickoff disagrees')
        lines = {name: {key: deepcopy(original[name][key]) for key in
                 ('rates', 'class_order', 'probabilities', 'top_scores', 'over_2_5')}
                 for name in ('goal_simple', 'independent_xg', 'rich_121_research')}
        records.append({'fixture': deepcopy(fixture),
            'statistical_publication_at': stat['generated_at'],
            'statistical_estimates': lines,
            'performance_context': deepcopy(facts[ident]),
            'official_claims': [deepcopy(c) for c in evidence['claims'] if c['match_id'] == ident]})
    return {'assembled_at_utc': now.isoformat(),
        'purpose': 'New model-informed LLM analysis, followed by a separate assistant synthesis',
        'class_order': ['home_win', 'draw', 'away_win'],
        'exposure': {'current_statistical_probabilities': True,
            'historical_model_estimates': True, 'official_context': True,
            'market': False, 'other_llm_predictions': False, 'target_results': False},
        'source_files': {'statistical': str(ROOT/'outputs/今日意甲预测_20261028/统计预测与比分分布.json'),
            'performance': str(ROOT/'data/serie_a_adjusted_context_20261028/带对手校正的事实包.json'),
            'official_context': str(DEST/'补充赛前证据.json')},
        'statistical_scope': {key: stat['model_metadata'][key] for key in
            ('base_fit_cutoff', 'regularization', 'half_life_days', 'goal_fit_used_n', 'xg_fit_used_n')},
        'statistical_limits': deepcopy(stat['limits']),
        'opponent_adjustment_method': deepcopy(context['opponent_adjustment_method']),
        'official_sources': deepcopy(evidence['sources']),
        'official_source_conflicts': deepcopy(evidence['conflicts']),
        'official_unknowns': deepcopy(evidence['unknowns']),
        'interpretation_rules': [
            'The original statistical models contain no new official news. A later LLM must not imply otherwise.',
            'Original forecasts and previous LLM publications remain unchanged. This input belongs to a new version.',
            'Statistics, residuals and recent records overlap; agreement is not independent corroboration.',
            'A training status or squad inclusion does not establish starting status, ability, minutes or a fitted causal effect.',
            'A coach statement is attributed speech; intended tactics, match priority and actual execution remain distinct.',
            'Already reflected in earlier matches may be unknown; do not apply a second mechanical injury penalty.',
            'Use complete H/D/A probabilities if issuing a new subjective forecast; disclose model exposure and lack of calibration.',
            'For every game distinguish shared evidence, incremental context, disagreement, strongest counterargument and material unknowns.',
            'Do not consult results, market prices or other LLM outputs. Do not invent a numerical effect for an unquantified factor.',
            'The rich 121-feature candidate has a lead-three versus same-day input limitation. Do not treat it as a validated current upgrade.'
        ], 'records': records}


def main():
    now = datetime.now(timezone.utc)
    packet = build(read(ROOT/'outputs/今日意甲预测_20261028/统计预测与比分分布.json'),
        read(ROOT/'data/serie_a_adjusted_context_20261028/带对手校正的事实包.json'),
        read(DEST/'补充赛前证据.json'), now)
    path = DEST/'统计与情境协作输入.json'
    path.write_text(json.dumps(packet, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'path': str(path), 'fixtures': len(packet['records']),
        'claims': sum(len(r['official_claims']) for r in packet['records']),
        'sources': len(packet['official_sources']), 'assembled_at': packet['assembled_at_utc']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
