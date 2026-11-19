"""Reuse existing match facts and historical conditional xG for assisted reading."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
DATA = ROOT / 'data/epl_personnel_live_20261029'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    original_path = DATA / '独立LLM人员协作输入_v2.json'
    context_path = ROOT / 'data/epl_adjusted_context_20261029/对手条件表现.json'
    update_path = DATA / '更新核对_1144UTC/最新官方状态.json'
    destination = DATA / '模型辅助LLM条件表现输入.json'
    if destination.exists():
        raise FileExistsError('Keep the published input; create an explicit new version if needed.')
    original = read(original_path)
    context = read(context_path)
    update = read(update_path)
    if update['changes']:
        raise ValueError('New personnel claims need an explicit factual projection before use.')
    packet = deepcopy(original)
    packet['assembled_at_utc'] = datetime.now(timezone.utc).isoformat()
    packet['purpose'] = 'Prospective model-assisted LLM forecast with descriptive form, sourced personnel and historical opponent/venue-conditioned xG context.'
    packet['exposure']['historical_model_estimates'] = True
    packet['source_files'] = {'original_facts': str(original_path),
        'historical_context': str(context_path), 'official_update_check': str(update_path)}
    packet['historical_context_method'] = context['method']
    packet['historical_context_limits'] = context['limits']
    packet['historical_context_reconstructed_at_utc'] = context['generated_at_utc']
    packet['official_update_check'] = {
        'checked_at_utc': update['checked_at_utc'], 'changes': [],
        'conclusion': update['conclusion'],
        'meaning': 'No claim-changing update was found within this bounded check; this does not confirm target-match availability.'}
    n = 0
    for row in packet['records']:
        row['historical_conditioned_performance'] = {}
        for side, facts in row['descriptive_performance'].items():
            item = deepcopy(context['teams'][facts['team']])
            expected_ids = [x['match_id'] for x in facts['window_5']['history']]
            if item['original_window_match_ids'] != expected_ids:
                raise ValueError('Historical context must retain the exact original five-match window.')
            for observation in item['observations']:
                if not observation['past_training_max_date'] < observation['date']:
                    raise ValueError('Historical expectation includes same-day or later information.')
                observation.pop('past_model_file', None)
                n += 1
            row['historical_conditioned_performance'][side] = item
    packet['instructions'] = [
        'Output separate home-win, draw and away-win probabilities summing to 1 for all three fixtures.',
        'This is a MODEL-ASSISTED forecast because historical model estimates are visible. Do not call it independent.',
        'Read only this packet. Do not browse, consult target probabilities, market odds, other LLM predictions or target results.',
        'Use the original descriptive facts and personnel uncertainties together with the historical conditional context. Explain competing evidence; do not force agreement.',
        'A residual is deviation from that team\'s own earlier model expectation conditional on opponent and venue. It is not an absolute cross-team strength ranking or proof of weak opponents.',
        'Raw xG and the corresponding residual are the SAME observations expressed differently. Do not treat them as independent votes or mechanically add them.',
        'Positive defense_xga_residual means more xG conceded than expected (worse defense). These are total-xG expectations, not goals or NPxG.',
        'Five-match context is small; do not let one recent match mechanically determine the forecast. The 15-match window can cross seasons and is not fully opponent-adjusted.',
        'Historical expectations were reconstructed now using strictly earlier match dates. Historical source publication/revision times are not fully reconstructible.',
        'No automated personnel probability penalty, assumed lineup, tactical certainty, calibration or causal accuracy gain is provided.',
        'Official news retains its publication/observation limits. Return to training is not medical clearance; zero recent minutes is not a missing row.',
        'Cite selected supporting and opposing facts using exact JSON Pointers and copied values from this packet.'
    ]
    if n != 30:
        raise ValueError('Expected six original five-match contexts.')
    destination.write_text(json.dumps(packet, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'path': str(destination), 'assembled_at_utc': packet['assembled_at_utc'],
        'fixtures': len(packet['records']), 'historical_side_rows': n, 'exposure': packet['exposure']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
