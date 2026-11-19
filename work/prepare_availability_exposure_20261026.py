"""FPL source-state changes with the existing model's exact past-five usage.

All weights describe observed league participation, never ability, expected
future minutes, clinical recovery or an independently applied goal-rate loss.
"""
from collections import Counter, defaultdict
import csv
from datetime import date, datetime, timezone
import gzip
import json
import lzma

from intake_fpl_availability_20261026 import ROOT, DATA, ALIASES, write


BASE = ROOT/'outputs/固定发布日期信息刷新_20261025T071806561562Z/features_lead3.json.gz'
PREVIOUS = ROOT/'outputs/FPL历史可用性接入_20261026/逐场赛前状态特征.json.gz'
OUT = ROOT/'outputs/FPL可用性变化与参与接入_20261026'
FLAGS = ('i', 'd', 's')
INTERPRETABLE = ('a', 'i', 'd', 's')
MEASUREMENTS = tuple('current_'+s+'_observed_minute_share' for s in FLAGS) + tuple(
    'change_'+s+'_same_weight_observed_share' for s in FLAGS) + (
    'current_interpretable_minute_coverage', 'five_reference_paired_minute_coverage',
    'flagged_without_observed_prior_same_team_count',
    'flagged_with_prior_but_zero_window_minutes_count',
    'now_a_after_old_flag_zero_window_minutes_count')


def exposure(weights, current, references):
    """Observed mass on fixed weights; unknown states never become source a."""
    result = {key: None for key in MEASUREMENTS[:8]}
    if weights is None:
        return result
    known = {p for p in weights if current.get(p) in INTERPRETABLE}
    paired = {p for p in known if len(references) == 5 and all(r.get(p) in INTERPRETABLE for r in references)}
    current_mass = sum(weights[p] for p in known)
    paired_mass = sum(weights[p] for p in paired)
    result['current_interpretable_minute_coverage'] = current_mass
    result['five_reference_paired_minute_coverage'] = paired_mass
    for status in FLAGS:
        result['current_'+status+'_observed_minute_share'] = (
            sum(weights[p] for p in known if current[p] == status) if current_mass > 0 else None)
        result['change_'+status+'_same_weight_observed_share'] = (
            sum(weights[p]*((current[p] == status)-sum(r[p] == status for r in references)/5)
                for p in paired) if paired_mass > 0 else None)
    return result


def load_mapping():
    with (DATA/'id_mapping/Master.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    codes, reverse = defaultdict(list), defaultdict(set)
    for row in rows:
        if row['code'].isdigit() and row['understat'].isdigit():
            codes[int(row['code'])].append(row)
            reverse[row['understat']].add(int(row['code']))
    return {code: values[0] for code, values in codes.items()
            if len(values) == 1 and len(reverse[values[0]['understat']]) == 1}


def load_minutes():
    observed = {}
    for path in (ROOT/'outputs/历史球员使用输入_20261025/历史出场分钟.json',
                 ROOT/'outputs/当前人员历史代表性_20261025/逐场球员使用记录.json'):
        for row in json.loads(path.read_text(encoding='utf-8'))['observations']:
            match_id = row.get('football_data_match_id', row['match_id'])
            key = (match_id, row['team_id'])
            if key in observed and row['minutes'] != observed[key]['minutes']:
                raise ValueError('Overlapping historical usage differs')
            observed[key] = row
    histories = defaultdict(list)
    for row in observed.values():
        histories[row['team_id']].append(row)
    for values in histories.values():
        values.sort(key=lambda r: (r['date'], r['match_id']))
    return observed, histories


class States:
    def __init__(self, prior_rows, mapping):
        self.prior = {r['match_id']: r for r in prior_rows}
        self.mapping, self.cache = mapping, {}

    def for_row(self, row):
        prior = self.prior.get(row['match_id'])
        if not prior or not prior.get('archive_observation'):
            return None
        observation = prior['archive_observation']
        if observation['status'] == 'failed':
            return None
        key = (observation['path'], row['period'])
        if key not in self.cache:
            path = DATA/'raw'/observation['path'].removeprefix('cache/')
            payload = json.loads(lzma.decompress(path.read_bytes()))
            team_keys = {r['id']: 'football-data:E0:'+ALIASES.get(r['name'], r['name']) for r in payload['teams']}
            players = defaultdict(list)
            season_column = row['period'][2:4]+'-'+row['period'][-2:]
            raw_codes = Counter(p.get('code') for p in payload['elements'])
            for player in payload['elements']:
                code = player.get('code')
                mapped = self.mapping.get(code)
                agreement = bool(mapped and mapped.get(season_column, '').isdigit()
                                 and int(mapped[season_column]) == player['id'] and raw_codes[code] == 1)
                players[team_keys[player['team']]].append({
                    'code': code, 'fpl_id': player['id'], 'status': player.get('status'),
                    'name': player.get('web_name'),
                    'understat_key': 'understat:player:'+mapped['understat'] if agreement else None})
            self.cache[key] = dict(players)
        return {'players': self.cache[key], 'observation': observation,
                'forecast_before_date': prior['forecast_before_date']}


def team_input(row, side, by_id, states, minutes, histories):
    team = row[side]
    history = row['fresh_feature_history']['history'][side]
    ids = history['window_match_ids']['5']
    minute_rows = [minutes.get((match_id, team)) for match_id in ids]
    complete = len(ids) == 5 and all(r is not None for r in minute_rows)
    if any(r is not None and not r['date'] < row['forecast_before_date'] for r in minute_rows):
        raise ValueError('Minute reference is not earlier than the forecast')
    weights = None
    team_minutes = None
    if complete:
        summed = Counter()
        for r in minute_rows:
            summed.update(r['minutes'])
        team_minutes = sum(summed.values())
        if team_minutes <= 0:
            raise ValueError('Complete usage window needs positive observed team minutes')
        weights = {p: value/team_minutes for p, value in summed.items() if value > 0}
    earlier = [r for r in histories[team] if r['date'] < row['forecast_before_date']]
    previously_positive = {p for r in earlier for p, v in r['minutes'].items() if v > 0}
    current_payload = states.for_row(row)
    current_players = [] if current_payload is None else current_payload['players'].get(team, [])
    current = {p['understat_key']: p['status'] for p in current_players if p['understat_key'] is not None}
    references, source_notes = [], []
    for identifier in ids:
        old = by_id.get(identifier)
        source = None if old is None else states.for_row(old)
        old_players = [] if source is None else source['players'].get(team, [])
        references.append({p['understat_key']: p['status'] for p in old_players if p['understat_key'] is not None})
        source_notes.append({'match_id': identifier, 'forecast_before_date': None if source is None else source['forecast_before_date'],
            'snapshot': None if source is None else source['observation']['path'], 'team_observed': bool(old_players)})
    result = exposure(weights, current, references)
    flagged = [p for p in current_players if p['status'] in FLAGS]
    result['flagged_without_observed_prior_same_team_count'] = (sum(
        p['understat_key'] not in previously_positive for p in flagged) if current_players else None)
    result['flagged_with_prior_but_zero_window_minutes_count'] = (sum(
        p['understat_key'] in previously_positive and weights.get(p['understat_key'], 0) == 0
        for p in flagged) if weights is not None and current_players else None)
    comparable_references = len(references) == 5 and all(n['team_observed'] for n in source_notes)
    zero_returns = []
    transitions, examples = Counter(), []
    for player in current_players:
        key = player['understat_key']
        old_states = [r.get(key) for r in references]
        paired_known = len(old_states) == 5 and all(s in INTERPRETABLE for s in old_states)
        classification = None
        if player['status'] in FLAGS:
            if not paired_known:
                classification = 'old_source_states_not_fully_known'
            elif all(s == player['status'] for s in old_states):
                classification = 'same_flag_all_five_sources'
            elif all(s == 'a' for s in old_states):
                classification = 'flag_now_all_five_sources_a'
            else:
                classification = 'flag_now_mixed_or_other_flag_sources'
        elif player['status'] == 'a' and paired_known and any(s in FLAGS for s in old_states):
            classification = 'source_now_a_after_any_old_flag'
            if weights is not None and weights.get(key, 0) == 0:
                zero_returns.append(key)
        if classification:
            transitions[classification] += 1
            examples.append(player | {'reference_statuses': old_states, 'classification': classification,
                'observed_prior_same_team_positive_minutes': key in previously_positive,
                'window_minute_share': None if weights is None else weights.get(key, 0) if key else None})
    result['now_a_after_old_flag_zero_window_minutes_count'] = (
        len(zero_returns) if weights is not None and current_players and comparable_references else None)
    if tuple(result) != MEASUREMENTS:
        raise ValueError('Measurement definitions differ from the declared block')
    latest_date = max((r['date'] for r in minute_rows if r is not None), default=None)
    diagnostics = {'team': team, 'exact_reference_match_ids': ids,
        'minute_reference_rows_observed': sum(r is not None for r in minute_rows),
        'minute_window_complete': complete, 'actual_observed_team_minutes': team_minutes,
        'latest_minute_reference_date': latest_date,
        'days_since_latest_minute_reference': None if latest_date is None else (
            date.fromisoformat(row['forecast_before_date'])-date.fromisoformat(latest_date)).days,
        'reference_status_source_count': sum(n['team_observed'] for n in source_notes),
        'reference_status_sources': source_notes,
        'current_source_players': len(current_players), 'current_flagged_players': len(flagged),
        'observed_transition_counts': dict(transitions), 'flag_and_source_return_examples': examples,
        'measurements': result}
    return result, diagnostics


def synthetic_checks():
    w = {'A': .4, 'B': .3, 'C': .2, 'D': .1}
    now = {'A': 'i', 'B': 'a', 'C': 'd', 'D': 's'}
    old = [{'A': 'a', 'B': 'i', 'C': 'd', 'D': 's'} for _ in range(5)]
    r = exposure(w, now, old)
    assert abs(r['current_i_observed_minute_share']-.4) < 1e-12
    assert abs(r['change_i_same_weight_observed_share']-.1) < 1e-12
    assert r['change_d_same_weight_observed_share'] == r['change_s_same_weight_observed_share'] == 0
    assert exposure({'A': 1.}, {'A': 'i'}, [{'A': 'i'}]*5)['change_i_same_weight_observed_share'] == 0
    unknown = exposure({'A': 1.}, {'A': 'i'}, [{'A': 'n'}]*5)
    assert unknown['change_i_same_weight_observed_share'] is None and unknown['five_reference_paired_minute_coverage'] == 0
    assert exposure(None, now, old)['current_i_observed_minute_share'] is None
    return {'hand_calculated_new_continuing_and_source_a_cases': True,
            'all_unknown_paired_state_kept_none': True, 'missing_window_kept_none': True}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT/'分析前输入方法.json', {
        'created_at': datetime.now(timezone.utc).isoformat(), 'base_source': str(BASE),
        'new_measurements_per_team': list(MEASUREMENTS), 'feature_columns': 121+2*len(MEASUREMENTS),
        'reference': 'exact original last-five match IDs; never replace missing rows with other matches',
        'weights': 'positive observed same-team minutes summed across exact five games / their actual total team minutes; fixed across current and old states',
        'status_reference': 'each of those five games original F=D-3 archived source observation, not its actual lineup',
        'change': 'sum fixed w times [I(current=s) - mean of five historical source I(state=s)] on same fully paired subset',
        'coverage': 'unrenormalized observed mass plus current/paired minute coverage; partial observations are not full lost strength',
        'zero_return_case': 'source now a after an earlier flag can have zero recent participation; separate observed count prevents silently losing this source change',
        'missing': 'missing complete team-match minute row => window measures None; player absent from a complete usage list => zero participation, not zero ability',
        'not_used': ['target starting XI', 'target player minutes', 'target actual appearance', 'FPL chance as probability', 'old foreign-provider player coefficients', 'manual goal-rate subtraction'],
        'limits': ['a is not confirmed fitness; i/d/s are source labels',
            'FPL observation changes are not clinical onset or recovery times',
            'old FPL states are three days before old games, not guaranteed match-time states',
            'past-five window may span a summer break; ages retained',
            'only EPL usage is observed; lack of earlier records is not proof of no professional history',
            'historical source and retrospective identity reconciliation assumptions from existing intake remain'],
        'current_results_checked': False})
    with gzip.open(BASE, 'rt', encoding='utf-8') as handle:
        rows = [r for r in json.load(handle)['rows'] if r['competition'] == 'E0']
    with gzip.open(PREVIOUS, 'rt', encoding='utf-8') as handle:
        previous = json.load(handle)['rows']
    by_id = {r['match_id']: r for r in rows}
    states = States(previous, load_mapping())
    minutes, histories = load_minutes()
    prepared, coverage = [], defaultdict(Counter)
    for row in rows:
        values, diagnostics = [], {}
        for side in ('home', 'away'):
            measurements, note = team_input(row, side, by_id, states, minutes, histories)
            values.append(measurements)
            diagnostics[side] = note
            c = coverage[row['period']]
            c['team_sides'] += 1
            c['complete_exact_minute_window'] += note['minute_window_complete']
            c['five_old_status_sources_observed'] += note['reference_status_source_count'] == 5
            c['sides_with_positive_paired_coverage'] += bool(measurements['five_reference_paired_minute_coverage'])
            c['current_flagged_player_contexts'] += note['current_flagged_players']
            for k, v in note['observed_transition_counts'].items():
                c[k] += v
            c['source_a_after_old_flag_zero_window_minutes'] += measurements['now_a_after_old_flag_zero_window_minutes_count'] or 0
            c['flagged_without_observed_prior_same_team'] += measurements['flagged_without_observed_prior_same_team_count'] or 0
        features = []
        for side in range(2):
            extra = {'avail_own_'+k: v for k, v in values[side].items()}
            extra.update({'avail_opponent_'+k: v for k, v in values[1-side].items()})
            features.append(dict(row['fresh_performance_features'][side]) | extra)
        context = {k: row[k] for k in ('match_id', 'date', 'period', 'competition', 'home', 'away',
                   'fit_before_date', 'forecast_before_date', 'horizon_days', 'actual_score', 'actual_outcome')}
        prepared.append(context | {'side_features': features,
            'reference_features': row['fresh_performance_features'],
            'goal_rates': row['goal_rates'], 'xg_rates': row['xg_rates'],
            'availability_diagnostics': diagnostics})
    with gzip.open(OUT/'逐场变化参与特征.json.gz', 'wt', encoding='utf-8') as handle:
        json.dump({'rows': prepared}, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    checks = synthetic_checks()
    changed = dict(rows[-1], actual_score=[19, 19], actual_outcome=1)
    checks['target_result_mutation_measurements_unchanged'] = all(
        team_input(changed, side, by_id, states, minutes, histories)[0] ==
        prepared[-1]['availability_diagnostics'][side]['measurements'] for side in ('home', 'away'))
    report = {'rows': len(prepared), 'feature_columns': 143,
        'by_period': {p: dict(c) for p, c in coverage.items()}, 'checks': checks,
        'new_downloads': 0, 'new_model_fits': 0, 'current_forecasts_replaced': False}
    write(OUT/'接入实测.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
