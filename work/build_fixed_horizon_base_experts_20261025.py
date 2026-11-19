"""Legal base rates for fixed one/three-calendar-day forecast publication.

Only unchanged earlier designs reuse old parameters. Unknown teams remain
explicitly unavailable; no absent coefficient is treated as a known zero prior.
"""
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import time

from football_model.date_model import DateTeamGoalModel
from football_model.xg_history import DateTeamXgModel
from football_model.goal_xg_blend import fit_goal_xg_weight, predict_goal_xg_rates
from football_model.team_model import team_features
from run_npxg_comparison_20261025 import load_source, ROOT, read
from run_shot_increment_20261025 import write


BASE = ROOT/'outputs/干净基础专家重建_20261025T064408733954Z'


def replay(parameters, row):
    required = {row['home'], row['away']}
    if not required.issubset(parameters['teams']):
        raise ValueError('Unknown teams need an estimated prior, not absent zero coefficients')
    return [math.exp(parameters['intercept']+math.fsum(parameters['coefficients'].get(k, 0)*v
        for k, v in vector.items())) for vector in team_features(row['home'], row['away'], row['competition'])]


def parameter_date(played, period, lead):
    forecast = played-timedelta(days=lead)
    anchor = date(int(period[:4]), 8, 1)
    fitted = anchor+timedelta(days=((forecast-anchor).days//28)*28)
    if not fitted <= forecast < played:
        raise ValueError('Both model and observations must be available at publication')
    return forecast, fitted


def main():
    started = time.perf_counter()
    now = datetime.now(timezone.utc)
    out = ROOT/'outputs'/now.strftime('固定提前日期基础专家_%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    original = read(BASE/'clean_base_rows.json')
    saved_parameters = read(BASE/'干净基础专家参数.json')
    source, independent, joins, fd_reports, directories = load_source()
    population = {group: sorted({identifier for fit in parameters.values()
                               for identifier in fit['goal']['reference_match_ids']})
                  for group, parameters in saved_parameters.items()}
    for group, identifiers in population.items():
        if any(identifier not in source or independent[identifier].goal_fit_exclusion for identifier in identifiers):
            raise ValueError('Source population changed beyond clean expert design')
        for key, pair in saved_parameters[group].items():
            comp, cutoff = key.split(':', 1)
            expected = [r.match_id for r in sorted((source[i] for i in identifiers
                if source[i].competition == comp and source[i].match_date < date.fromisoformat(cutoff)),
                key=lambda r: (r.match_date, r.match_id))]
            if any(pair[response]['reference_match_ids'] != expected for response in ('goal', 'xg')):
                raise ValueError('Recorded expert designs do not share the reconstructed source range')
    write(out/'实施说明.json', {
        'created_at': now.isoformat(), 'purpose': 'base experts available no later than fixed forecast publication day',
        'original_base_directory': str(BASE), 'forecast_leads_days': [1, 3],
        'rule': 'F=fixture_date-L; P=seasonAug1+28*floor((F-seasonAug1)/28); P<=F<fixture_date',
        'source_ranges': {'earlier': 'original clean2020-origin expert reference universe',
                          'recent': 'original clean2022-origin expert reference universe'},
        'reuse_rule': 'same group, competition, cutoff and complete earlier reference-ID design',
        'missing_team_rule': 'unavailable with exact identity and reason; not zero-filled',
        'preset': {'regularization': .01, 'half_life_days': 180, 'parameter_update_days': 28},
        'new_parameter_grid': False, 'current_results_reviewed': False, 'lineups_used': False,
    })
    params, horizons, new_fits, reused = {}, {}, 0, 0
    for lead in (1, 3):
        rows, unavailable, shifted = [], [], []
        for old in original['rows']:
            played = date.fromisoformat(old['date'])
            forecast, fitted = parameter_date(played, old['period'], lead)
            group, comp = old['base_generation_group'], old['competition']
            local_key = f'{comp}:{fitted.isoformat()}'
            key = f'{group}:{local_key}'
            if key not in params:
                history = sorted((source[i] for i in population[group] if source[i].competition == comp
                                  and source[i].match_date < fitted), key=lambda r: (r.match_date, r.match_id))
                identifiers = [r.match_id for r in history]
                saved = saved_parameters[group].get(local_key)
                if saved:
                    if any(saved[response]['reference_match_ids'] != identifiers for response in ('goal', 'xg')):
                        raise ValueError('Unchanged cutoff does not have an unchanged reference design')
                    pair = {'goal': saved['goal'], 'xg': saved['xg'], 'action': 'reused_same_earlier_design'}
                    reused += 2
                else:
                    goal = DateTeamGoalModel(.01, 180).fit([independent[i] for i in identifiers], fitted).parameters()
                    xg = DateTeamXgModel(.01, 180).fit(history, fitted).parameters()
                    pair = {'goal': goal, 'xg': xg, 'action': 'fitted_missing_earlier_cutoff'}
                    new_fits += 2
                for response in ('goal', 'xg'):
                    p = pair[response]
                    if (p['reference_match_ids'] != identifiers or p['before_date'] != fitted.isoformat()
                            or p['training_max_date'] >= fitted.isoformat()):
                        raise ValueError('Base parameter temporal design differs from declared cutoff')
                params[key] = pair
                print(json.dumps({'parameter': key, 'action': pair['action'], 'n': len(identifiers)}, ensure_ascii=False), flush=True)
            pair = params[key]
            missing = {response: sorted({old['home'], old['away']}-set(pair[response]['teams']))
                       for response in ('goal', 'xg')}
            context = {k: old[k] for k in ('match_id', 'date', 'period', 'competition', 'home', 'away',
                                         'actual_score', 'actual_outcome', 'base_generation_group')}
            context.update(fit_before_date=fitted.isoformat(), forecast_before_date=forecast.isoformat(),
                           horizon_days=lead, original_parameter_date=old['fit_before_date'], base_parameter_key=key)
            if fitted.isoformat() != old['fit_before_date']:
                shifted.append(old['match_id'])
            if any(missing.values()):
                unavailable.append({**context, 'reason': 'unseen_team_in_legal_earlier_base_parameters',
                                    'unseen_teams': missing})
                continue
            gr, xr = replay(pair['goal'], old), replay(pair['xg'], old)
            rows.append({**context, 'rates': gr, 'goal_rates': gr, 'xg_rates': xr,
                         'training_max_date': pair['goal']['training_max_date'],
                         'base_action': pair['action']})
        blend_reports = {}
        for group, fit_period, test_period, learned in (
                ('earlier', '2022-23', '2023-24', date(2023, 8, 1)),
                ('recent', '2024-25', '2025-26', date(2025, 8, 1))):
            calibration = [r for r in rows if r['period'] == fit_period]
            fit = fit_goal_xg_weight(calibration, learned)
            test = [r for r in rows if r['period'] == test_period]
            if any(learned > date.fromisoformat(r['forecast_before_date']) for r in test):
                raise ValueError('Blend weight arrives after forecast publication')
            blend_reports[group] = fit
            for row in (r for r in rows if r['base_generation_group'] == group):
                # An apparent-fit calibration blend is kept only as provenance;
                # the new scheme runner must not score/use it for selection.
                row['blend_rates'] = list(predict_goal_xg_rates(row['goal_rates'], row['xg_rates'], fit['weight']))
                row['blend_weight'] = fit['weight']
                row['blend_weight_before_date'] = fit['before_date']
                row['blend_role'] = ('apparent_calibration_fit_do_not_use_for_selection'
                    if row['period'] == fit_period else 'earlier_calibration_fixed_before_publication')
        horizons[str(lead)] = {'rows': rows, 'unavailable': unavailable,
            'coverage': {'original_n': len(original['rows']), 'available_n': len(rows), 'unavailable_n': len(unavailable),
                'available_by_period': dict(Counter(r['period'] for r in rows)),
                'unavailable_by_period': dict(Counter(r['period'] for r in unavailable)),
                'shifted_parameter_n': len(shifted), 'shifted_parameter_match_ids': shifted},
            'blend_parameters': blend_reports}
        print(json.dumps({'horizon': lead, 'coverage': {k: v for k, v in horizons[str(lead)]['coverage'].items()
            if k != 'shifted_parameter_match_ids'}}, ensure_ascii=False), flush=True)
    references = 0
    for key, pair in params.items():
        for response in ('goal', 'xg'):
            p = pair[response]
            for identifier in p['reference_match_ids']:
                record = independent[identifier]
                if record.goal_fit_exclusion or record.match_date.isoformat() >= p['before_date']:
                    raise ValueError('Ineligible or later match entered base training')
            references += len(p['reference_match_ids'])
    result = {'created_at': now.isoformat(), 'finished_at': datetime.now(timezone.utc).isoformat(),
        'source_directory': str(BASE), 'new_base_fits': new_fits, 'reused_base_fits': reused,
        'unique_parameter_pairs': len(params), 'horizons': horizons,
        'checked_training_reference_entries': references,
        'parameter_file': str(out/'固定提前日期基础参数.json'),
        'source_report': str(out/'源范围与联结.json'), 'elapsed_seconds': time.perf_counter()-started}
    write(out/'源范围与联结.json', {'population_ids_by_group': population, 'joins': joins,
        'fd_reports': fd_reports, 'directories': directories, 'source_matches': len(source)})
    write(out/'固定提前日期基础参数.json', params)
    write(out/'fixed_horizon_base_rows.json', result)
    print(json.dumps({'directory': str(out), 'new_fits': new_fits, 'reused': reused,
                      'seconds': result['elapsed_seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
