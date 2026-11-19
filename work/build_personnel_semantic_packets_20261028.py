"""Join archived text with prior participation for source-grounded LLM reading.

No fitting, scoring, actual target lineup or target outcome is exported. Existing
historical files contain outcomes; only the explicit fields below are projected.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import gzip
import json

from prepare_availability_exposure_20261026 import load_mapping, load_minutes

ROOT = REPOSITORY_ROOT
DATA = ROOT / 'data/personnel_semantics_20261028'
OUT = ROOT / 'outputs/人员信息语义接入_20261028'
BASE = ROOT / 'outputs/协作预测与大样本方案_20261028/扩样训练/扩样合并特征.json.gz'
PERFORMANCE_KEYS = tuple(
    'own_' + field + '_' + str(window)
    for window in (5, 15)
    for field in ('history_count', 'goals_for_mean', 'goals_against_mean',
                  'xg_for_mean', 'xg_against_mean', 'npxg_for_mean', 'npxg_against_mean')
) + tuple('own_venue_' + f + '_15' for f in (
    'count', 'goals_for_mean', 'goals_against_mean', 'xg_for_mean', 'xg_against_mean'))


def read(path):
    with (gzip.open(path, 'rt', encoding='utf-8') if str(path).endswith('.gz')
          else open(path, encoding='utf-8')) as handle:
        return json.load(handle)


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with (gzip.open(path, 'wt', encoding='utf-8') if str(path).endswith('.gz')
          else open(path, 'w', encoding='utf-8')) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=None if str(path).endswith('.gz') else 2)


def llm_semantic_view(record):
    """Only text evidence, never future catalogue frequencies or observation spans."""
    return {key: record[key] for key in ('news_id', 'raw_status', 'raw_news', 'semantics')}


def usage_evidence(period, cutoff, team, reference_ids, players, mapping, minutes, histories):
    """Actual earlier same-team participation; zero and unknown are distinct."""
    reference_rows = [minutes.get((identifier, team)) for identifier in reference_ids]
    if any(r is not None and r['date'] >= cutoff for r in reference_rows):
        raise ValueError('Participation is not strictly before the target forecast')
    if any(r is not None and r['team_id'] != team for r in reference_rows):
        raise ValueError('Participation belongs to a different team')
    complete = len(reference_ids) == 5 and all(r is not None for r in reference_rows)
    summed = Counter()
    if complete:
        for row in reference_rows:
            summed.update(row['minutes'])
    denominator = sum(summed.values()) if complete else None
    if complete and denominator <= 0:
        raise ValueError('Observed complete window has no positive team minutes')
    earlier = [r for r in histories.get(team, []) if r['date'] < cutoff]
    observed_positive = {key for row in earlier for key, value in row['minutes'].items() if value > 0}
    season_column = period[2:4] + '-' + period[-2:]
    results = {}
    for player in players:
        code = player['source_player_code']
        current = player['current']
        mapped = mapping.get(code)
        matched = bool(mapped and mapped.get(season_column, '').isdigit()
                       and int(mapped[season_column]) == current['source_player_id'])
        key = 'understat:player:' + mapped['understat'] if matched else None
        results[code] = {
            'identity_matched_by_code_and_same_season_id': matched,
            'understat_player_id': key,
            'prior_positive_same_team_minutes_observed': key in observed_positive if matched else None,
            'window_minutes': summed.get(key, 0) if complete and matched else None,
            'window_minute_share': summed.get(key, 0) / denominator if complete and matched else None,
            'meaning': 'Observed earlier league minutes, not ability, expected minutes, fitness or a loss coefficient.'}
    return {
        'exact_reference_match_ids': reference_ids,
        'minute_reference_rows_observed': sum(r is not None for r in reference_rows),
        'minute_window_complete': complete,
        'actual_observed_team_minutes': denominator,
        'latest_minute_reference_date': max((r['date'] for r in reference_rows if r is not None), default=None),
        'source_files': [str(ROOT / 'outputs/历史球员使用输入_20261025/历史出场分钟.json'),
                         str(ROOT / 'outputs/当前人员历史代表性_20261025/逐场球员使用记录.json')],
        'players': results,
        'limits': ['A zero in a complete observed window describes past participation only.',
                   'No matched identity or incomplete window remains unknown.',
                   'Earlier participation is not the target lineup or replacement quality.']}


def packet_record(base, aligned, mapping, minutes, histories):
    if base['match_id'] != aligned['match_id'] or base['forecast_before_date'] != aligned['forecast_before_date']:
        raise ValueError('Different match or forecast cutoff')
    cutoff = base['forecast_before_date']
    if base['base_training_max_date'] and base['base_training_max_date'] >= base['fit_before_date']:
        raise ValueError('Historical parameter training crosses its own cutoff')
    record = {k: base[k] for k in ('match_id', 'date', 'period', 'competition', 'home', 'away',
                                    'fit_before_date', 'forecast_before_date', 'horizon_days')}
    record['earlier_statistical_context'] = {
        'goal_rates': base['goal_rates'], 'xg_rates': base['xg_rates'],
        'parameter_before_date': base['fit_before_date'],
        'parameter_training_max_date': base['base_training_max_date'],
        'meaning': 'Existing earlier-parameter estimates; not observed xG, new LLM probabilities or new forecasts.'}
    record['recent_performance'] = []
    record['personnel_context'] = aligned
    for side_index, side in enumerate(('home', 'away')):
        values = base['fresh_performance_features'][side_index]
        record['recent_performance'].append({
            'team': base[side], 'side': side,
            'values': {key: values.get(key) for key in PERFORMANCE_KEYS},
            'history': {k: base['fresh_feature_history']['history'][side][k]
                        for k in ('history_count', 'earliest_used_date', 'latest_used_date', 'window_match_ids')},
            'meaning': 'Descriptive strictly earlier same-league performance; these means are not opponent-adjusted effects.'})
        team_context = aligned['teams'][side_index]
        if team_context['side'] != side or team_context['team'] != base[side]:
            raise ValueError('Team-side alignment differs')
        if not team_context['observed']:
            team_context['same_team_participation'] = None
            continue
        usage = usage_evidence(base['period'], cutoff, base[side], team_context['exact_reference_match_ids'],
                               team_context['players'], mapping, minutes, histories)
        team_context['same_team_participation'] = {k: v for k, v in usage.items() if k != 'players'}
        for player in team_context['players']:
            player['prior_same_team_usage'] = usage['players'][player['source_player_code']]
    return record


def materialize_example(record, side, player, semantics):
    team = record['personnel_context']['teams'][side]
    news_ids = {player['current']['news_id']}
    for reference in player['references']:
        if reference.get('observation'):
            news_ids.add(reference['observation']['news_id'])
    return {
        'match_id': record['match_id'], 'date': record['date'],
        'forecast_before_date': record['forecast_before_date'],
        'earlier_statistical_context': record['earlier_statistical_context'],
        'team': team['team'], 'side': team['side'],
        'recent_performance': record['recent_performance'][side],
        'current_snapshot': team['current_snapshot'],
        'reference_sources': team['reference_sources'],
        'same_team_participation': team['same_team_participation'],
        'player': player,
        'news_semantics': {identifier: semantics[identifier] for identifier in sorted(news_ids)}}


def materialize_match(payload, match_id):
    """Render just one historical fixture and its cited text, not future rows."""
    record = next(row for row in payload['rows'] if row['match_id'] == match_id)
    news_ids = set()
    for team in record['personnel_context']['teams']:
        for source in team.get('reference_sources', []):
            for older_player in source.get('reference_only_players') or []:
                news_ids.add(older_player['news_id'])
        for player in team['players']:
            news_ids.add(player['current']['news_id'])
            for reference in player['references']:
                if reference.get('observation'):
                    news_ids.add(reference['observation']['news_id'])
    return {'reconstructed_at_utc': payload['created_at_utc'], 'purpose': payload['purpose'],
            'instructions': payload['instructions'], 'target_labels_exported': False,
            'record': record,
            'news_semantics': {key: llm_semantic_view(payload['news_semantics'][key]) for key in sorted(news_ids)}}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match-id', help='Render one stored fixture as a self-contained LLM reading input.')
    arguments = parser.parse_args()
    if arguments.match_id:
        print(json.dumps(materialize_match(read(DATA / '历史LLM人员事实包.json.gz'), arguments.match_id), ensure_ascii=False))
        return
    aligned_payload = read(DATA / '逐场人员新闻上下文.json.gz')
    aligned = {r['match_id']: r for r in aligned_payload['rows']}
    semantics = {r['news_id']: llm_semantic_view(r) for r in read(DATA / '语义目录.json')['records']}
    bases = read(BASE)['rows']
    mapping = load_mapping()
    minutes, histories = load_minutes()
    rows = [packet_record(row, aligned[row['match_id']], mapping, minutes, histories) for row in bases]
    counts = Counter()
    same_status_text_change_details = Counter()
    per_category = defaultdict(Counter)
    candidates = defaultdict(list)
    for row in rows:
        for side, team in enumerate(row['personnel_context']['teams']):
            counts['team_contexts'] += 1
            if not team['observed']:
                counts['team_contexts_without_source'] += 1
                continue
            counts['team_contexts_with_source'] += 1
            counts['complete_minute_windows'] += team['same_team_participation']['minute_window_complete']
            for player in team['players']:
                counts['current_player_references'] += 1
                usage = player['prior_same_team_usage']
                categories = semantics[player['current']['news_id']]['semantics']['text_categories']
                if not usage['identity_matched_by_code_and_same_season_id']:
                    state = 'identity_unknown'
                elif usage['window_minutes'] is None:
                    state = 'window_unknown'
                elif usage['window_minutes'] > 0:
                    state = 'positive_window_minutes'
                elif usage['prior_positive_same_team_minutes_observed']:
                    state = 'zero_window_but_earlier_positive_observed'
                else:
                    state = 'zero_window_no_prior_positive_observed'
                counts[state] += 1
                for category in categories:
                    per_category[category][state] += 1
                current_semantics = semantics[player['current']['news_id']]['semantics']
                for reference in player['references']:
                    if reference['classification'] != 'same_status_news_changed':
                        continue
                    old_semantics = semantics[reference['observation']['news_id']]['semantics']
                    same_status_text_change_details['paired_references'] += 1
                    same_status_text_change_details['same_text_categories'] += (
                        set(current_semantics['text_categories']) == set(old_semantics['text_categories']))
                    same_status_text_change_details['reason_segment_changed'] += (
                        current_semantics['reason_segment'] != old_semantics['reason_segment'])
                    same_status_text_change_details['source_percentage_mentions_changed'] += (
                        current_semantics['source_percentage_mentions'] != old_semantics['source_percentage_mentions'])
                    same_status_text_change_details['return_wording_changed'] += (
                        current_semantics['expected_return_wording'] != old_semantics['expected_return_wording'])
                if row['period'] != '2024-25':
                    continue
                if 'medical_reference' in categories and state == 'positive_window_minutes' and any(
                        r['classification'] == 'same_status_news_changed' for r in player['references']):
                    candidates['same_status_different_medical_text_positive_usage'].append((row, side, player))
                if 'medical_reference' in categories and state == 'zero_window_but_earlier_positive_observed':
                    candidates['medical_text_zero_recent_but_older_participation'].append((row, side, player))
                if 'transfer_reference' in categories and state == 'positive_window_minutes':
                    candidates['transfer_text_positive_prior_participation'].append((row, side, player))
    examples, selected = [], set()
    for mechanism, pool in candidates.items():
        options = sorted(pool, key=lambda x: (x[0]['date'], x[0]['match_id'], str(x[2]['source_player_code'])), reverse=True)
        chosen = next((c for c in options if c[0]['match_id'] not in selected), None)
        if chosen:
            row, side, player = chosen
            selected.add(row['match_id'])
            example = materialize_example(row, side, player, semantics)
            example['mechanism_for_selection'] = mechanism
            examples.append(example)
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        'created_at_utc': now,
        'purpose': 'Historical evidence-reading inputs, not genuine historical LLM forecasts or a trained text-to-strength model.',
        'source_paths': {'base': str(BASE), 'personnel': str(DATA / '逐场人员新闻上下文.json.gz'),
                         'semantics': str(DATA / '语义目录.json')},
        'target_labels_exported': False,
        'storage_not_single_llm_prompt': True,
        'rendering_policy': 'Render one selected match and only its referenced text IDs; never send the multi-match storage bundle or full historical dictionary as a match prompt.',
        'instructions': ['Separate cited source fact, conditional interpretation and unknown.',
                         'Do not count the same absence twice against a model already using recent performance.',
                         'Do not infer numerical ability, replacement quality, actual target absence, outcome or probabilities.',
                         'Text changes are not medical onset or recovery; source percentages are not calibrated probabilities.'],
        'news_semantics': semantics, 'rows': rows}
    write(DATA / '历史LLM人员事实包.json.gz', payload)
    write(OUT / '人员新闻与参与联合实例.json', {
        'created_at_utc': now, 'purpose': payload['purpose'], 'target_labels_exported': False,
        'selection': '2024/25 latest distinct fixture for each declared source/participation mechanism; no outcomes used.',
        'instructions': payload['instructions'], 'records': examples})
    summary = {
        'created_at_utc': now, 'matches': len(rows), 'counts': dict(counts),
        'text_category_by_past_participation': {k: dict(v) for k, v in sorted(per_category.items())},
        'same_status_news_changed_details': dict(same_status_text_change_details),
        'mechanism_candidates': {k: len(v) for k, v in candidates.items()},
        'examples': [{'match_id': r['match_id'], 'team': r['team'],
                      'mechanism': r['mechanism_for_selection']} for r in examples],
        'model_fits': 0, 'parameter_searches': 0,
        'current_result_requests': 0, 'current_predictions_changed': False,
        'limits': ['Player references are repeated across fixtures, not independent samples.',
                   'Text-to-performance effects and replacement quality remain unestimated.',
                   'The historical base file contains outcomes; explicit projection excludes them from this output and all calculations.',
                   'Historical records are already studied development material, not a prospective LLM performance test.']}
    write(OUT / '事实包与参与汇总.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
