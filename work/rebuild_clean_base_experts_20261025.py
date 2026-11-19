"""Repair both base experts after newly evidenced incomplete fixtures.

Reuse readable parameters only when the exact earlier source design is unchanged;
refit both responses when its design changes. Original outputs stay untouched.
"""
from datetime import date, datetime, timezone
import math
from pathlib import Path
import time

import numpy as np

from football_model.date_model import DateTeamGoalModel
from football_model.xg_history import DateTeamXgModel
from football_model.goal_xg_blend import fit_goal_xg_weight, predict_goal_xg_rates
from football_model.team_model import team_features
from run_npxg_comparison_20261025 import load_source, pairs, ROOT
from run_shot_increment_20261025 import write


def replay(parameters, row):
    return [math.exp(parameters['intercept'] + math.fsum(parameters['coefficients'].get(k, 0)*v
        for k, v in vector.items())) for vector in team_features(row['home'], row['away'], row['competition'])]


def main():
    started = time.perf_counter()
    now = datetime.now(timezone.utc)
    out = ROOT/'outputs'/now.strftime('干净基础专家重建_%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    plan = {
        'created_at': now.isoformat(), 'purpose': 'remove newly evidenced incomplete matches from both base experts and target samples',
        'source_policy': 'current independent source join and registered incomplete-match exclusions; preserve old conflict exclusions',
        'preset': {'regularization': .01, 'half_life_days': 180, 'update_days': 28},
        'reuse': 'only exact unchanged eligible earlier match-ID design; coefficients replayed against original rates',
        'refit': 'changed design refits BOTH full-goal and total-xG experts',
        'cohort': 'original four-season fixture cohort minus currently excluded fixtures; no outcome-based drops',
        'blend': 'one weight from each earlier calibration season, fixed to its following season; calibration-season blend is apparent fit only',
        'new_current_results_requests': 0, 'lineups_used': False, 'parameter_grids': 0,
    }
    write(out/'重建计划.json', plan)
    source, independent, joins, fd_reports, directories = load_source()
    excluded_independent = {k: r.goal_fit_exclusion for k, r in independent.items() if r.goal_fit_exclusion}
    write(out/'新口径源联结.json', {'n': len(source), 'joins': joins, 'fd_reports': fd_reports,
        'directories': directories, 'excluded_independent': excluded_independent})
    all_rows, group_reports, all_parameters = [], {}, {}
    new_count, reused_count = 0, 0
    max_old_replay_error = 0.
    for label, group in pairs().items():
        gp, xp = group['goal_parameters'], group['xg_parameters']
        if set(gp) != set(xp):
            raise ValueError('Base expert cutoff designs differ')
        repaired = {}
        local_changes = []
        for key in sorted(gp):
            original_goal, original_xg = gp[key], xp[key]
            for field in ('reference_match_ids', 'n_matches', 'before_date', 'training_min_date',
                          'training_max_date', 'regularization', 'half_life_days'):
                if original_goal[field] != original_xg[field]:
                    raise ValueError(f'Original expert histories differ: {key}:{field}')
            original_ids = original_goal['reference_match_ids']
            ids = [identifier for identifier in original_ids if identifier in source
                   and independent[identifier].goal_fit_exclusion is None]
            removed = [identifier for identifier in original_ids if identifier not in ids]
            if any(not independent[identifier].goal_fit_exclusion for identifier in removed):
                raise ValueError('Unexpected source removal beyond newly registered exceptions')
            cutoff = date.fromisoformat(original_goal['before_date'])
            comp = key.split(':', 1)[0]
            history = [source[identifier] for identifier in ids]
            if (len(set(ids)) != len(ids) or any(r.match_date >= cutoff or r.competition != comp for r in history)
                    or ids != [r.match_id for r in sorted(history, key=lambda r: (r.match_date, r.match_id))]):
                raise ValueError('Repaired base history chronology conflict')
            if removed:
                goal = DateTeamGoalModel(.01, 180).fit([independent[identifier] for identifier in ids], cutoff).parameters()
                xg = DateTeamXgModel(.01, 180).fit(history, cutoff).parameters()
                action = 'refitted_changed_design'
                new_count += 2
            else:
                goal, xg = original_goal, original_xg
                action = 'reused_exact_unchanged_design'
                reused_count += 2
            for fitted in (goal, xg):
                if (fitted['reference_match_ids'] != ids or fitted['n_matches'] != len(ids)
                        or fitted['before_date'] != cutoff.isoformat()
                        or fitted['training_max_date'] >= cutoff.isoformat()
                        or any(identifier in excluded_independent for identifier in fitted['reference_match_ids'])):
                    raise ValueError('Repaired expert parameter history conflict')
            repaired[key] = {'goal': goal, 'xg': xg, 'action': action, 'removed_match_ids': removed}
            if removed:
                local_changes.append({'key': key, 'removed_match_ids': removed})
            print(f'{label} {key} {action}', flush=True)
        xg_map = {row['match_id']: row for row in group['xg']}
        local_rows, dropped = [], []
        for original in sorted(group['goal'], key=lambda r: (r['date'], r['match_id'])):
            identifier = original['match_id']
            if identifier not in source or independent[identifier].goal_fit_exclusion:
                dropped.append({'match_id': identifier, 'reason': independent[identifier].goal_fit_exclusion})
                continue
            record, old_xg = independent[identifier], xg_map[identifier]
            key = f'{original["competition"]}:{original["fit_before_date"]}'
            fit = repaired[key]
            for field in ('date', 'period', 'competition', 'home', 'away', 'fit_before_date', 'actual_score'):
                if original[field] != old_xg[field]:
                    raise ValueError('Original target tracks differ')
            if (record.match_date.isoformat() != original['date'] or record.home != original['home']
                    or record.away != original['away'] or list(record.full_time) != original['actual_score']):
                raise ValueError('Independent target identity or score conflict')
            gr, xr = replay(fit['goal'], original), replay(fit['xg'], original)
            if fit['action'].startswith('reused'):
                error = max(abs(a-b) for actual, saved in ((gr, original['rates']), (xr, old_xg['rates']))
                            for a,b in zip(actual, saved, strict=True))
                max_old_replay_error = max(max_old_replay_error, error)
                if error > 1e-10:
                    raise ValueError('Reused base expert failed original rate replay')
            h, a = original['actual_score']
            local_rows.append({
                'match_id': identifier, 'date': original['date'], 'period': original['period'],
                'competition': original['competition'], 'home': original['home'], 'away': original['away'],
                'fit_before_date': original['fit_before_date'], 'actual_score': [h, a],
                'actual_outcome': 0 if h > a else 2 if h < a else 1,
                'rates': gr, 'goal_rates': gr, 'xg_rates': xr,
                'training_max_date': fit['goal']['training_max_date'], 'base_parameter_key': f'{label}:{key}',
                'base_action': fit['action'], 'base_generation_group': label,
            })
        calibration = [row for row in local_rows if row['period'] == group['fit_period']]
        weight = fit_goal_xg_weight(calibration, group['scale_cutoff'])
        for row in local_rows:
            row['blend_rates'] = list(predict_goal_xg_rates(row['rates'], row['xg_rates'], weight['weight']))
            row['blend_weight'] = weight['weight']
            row['blend_role'] = ('apparent_calibration_fit_do_not_use_for_selection'
                if row['period'] == group['fit_period'] else 'earlier_season_weight_fixed')
            row['blend_weight_before_date'] = weight['before_date']
        group_reports[label] = {'newly_changed_designs': local_changes, 'dropped_targets': dropped,
            'by_period': {period: sum(row['period'] == period for row in local_rows)
                          for period in (group['fit_period'], group['test_period'])}, 'blend_parameters': weight}
        all_parameters[label] = repaired
        all_rows.extend(local_rows)
    all_rows.sort(key=lambda r: (r['date'], r['match_id']))
    if len({row['match_id'] for row in all_rows}) != len(all_rows):
        raise ValueError('Duplicate base target cohort')
    if any(row['match_id'] in excluded_independent for row in all_rows):
        raise ValueError('Incomplete match remains a target')
    result = {'plan': plan, 'finished_at': datetime.now(timezone.utc).isoformat(), 'n': len(all_rows),
        'source_matches': len(source), 'new_base_fits': new_count, 'reused_base_fits': reused_count,
        'max_unchanged_rate_replay_error': max_old_replay_error, 'groups': group_reports,
        'source_report': str(out/'新口径源联结.json'), 'parameters': str(out/'干净基础专家参数.json'),
        'all_reference_exclusion_check': 'passed; every source reference is strictly earlier and no registered incomplete fixture remains',
        'elapsed_seconds': time.perf_counter()-started, 'rows': all_rows}
    write(out/'干净基础专家参数.json', all_parameters)
    write(out/'clean_base_rows.json', result)
    print({'output': str(out), 'n': len(all_rows), 'new_fits': new_count, 'reused': reused_count,
           'max_replay_error': max_old_replay_error, 'elapsed_seconds': result['elapsed_seconds']}, flush=True)


if __name__ == '__main__':
    main()
