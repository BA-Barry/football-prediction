"""Add one earlier season to existing lead-3 features; three fixed tree fits.

Historical seasons are already examined research data, not a new blind test.
No network calls, search, current-season labels, or live forecast replacement.
"""
from collections import Counter
from datetime import date, datetime, timezone
import gzip
import json
import math
from pathlib import Path
import time

import joblib
import numpy as np

from football_model.all_competition_features import AllCompetitionFeatures
from football_model.date_model import DateTeamGoalModel
from football_model.early_goal_learners import EarlyGoalLearner
from football_model.fixed_horizon_features import PreparedPredictionFeatures, derive_timing
from football_model.xg_history import DateTeamXgModel
from run_broad_early_models_20261025 import observations, predict_row
from run_npxg_comparison_20261025 import ROOT, load_source
from run_shot_increment_20261025 import score_details

OUT = ROOT/'outputs/协作预测与大样本方案_20261028/扩样训练'
SOURCE = ROOT/'outputs/固定发布日期信息刷新_20261025T071806561562Z/features_lead3.json.gz'
ORIGINAL_MODEL = ROOT/'outputs/当前丰富信息预测_20261026T030805654404Z/丰富信息候选模型.joblib'
FEATURE = 'fresh_performance_features'
HELD_CUTOFF = date(2025, 8, 1)
FINAL_CUTOFF = date(2026, 8, 1)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def load_gzip(path):
    with gzip.open(path, 'rt', encoding='utf8') as handle:
        return json.load(handle)


def dump_gzip(path, value):
    with gzip.open(path, 'wt', encoding='utf8') as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def calendar_audit(rows):
    """Reject row identity/time/schema drift before it reaches a learner."""
    if len({r['match_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate training fixture')
    for row in rows:
        timing = derive_timing(date.fromisoformat(row['date']), 3,
                               season_start_year=int(row['period'][:4]))
        if (row['fit_before_date'] != timing['parameter_cutoff'].isoformat()
                or row['forecast_before_date'] != timing['forecast_cutoff'].isoformat()
                or row['horizon_days'] != 3
                or any(len(side) != 121 for side in row[FEATURE])):
            raise ValueError('Prepared feature schema or lead changed')
        h, a = row['actual_score']
        if row['actual_outcome'] != (0 if h > a else 2 if h < a else 1):
            raise ValueError('Outcome does not match its independently checked goals')


def weight_summary(rows, cutoff):
    w = np.asarray([math.exp(-math.log(2)*(cutoff-date.fromisoformat(r['date'])).days/180)
                    for r in rows])
    by_period = {}
    for period in sorted({r['period'] for r in rows}):
        local = w[[r['period'] == period for r in rows]]
        by_period[period] = {'n': len(local), 'weight_sum': float(local.sum()),
                             'weight_share': float(local.sum()/w.sum())}
    return {'n_matches': len(rows), 'n_scoring_sides': 2*len(rows),
            'half_life_days': 180, 'sum_match_weights': float(w.sum()),
            'effective_match_n_kish': float(w.sum()**2/(w@w)),
            'effective_n_note': 'Weight concentration only; correlated matches are not independent effective observations.',
            'by_period': by_period,
            'preprocessing_note': 'Existing learner retains unweighted training medians and count-based min_samples_leaf; old rows can affect them despite small outcome weights.'}


def prepare_added(source, independent, expected_names):
    destination = OUT/'2021-22新增特征.json.gz'
    if destination.exists():
        stored = load_gzip(destination)
        calendar_audit(stored['rows'])
        if stored['source_feature_file'] != str(SOURCE):
            raise ValueError('Prepared expansion belongs to another feature source')
        return stored, 0
    targets = sorted((r for r in source.values() if independent[r.match_id].season == '2021/22'
                      and independent[r.match_id].goal_fit_exclusion is None),
                     key=lambda r: (r.match_date, r.match_id))
    eligible_source = {mid: r for mid, r in source.items()
                       if independent[mid].goal_fit_exclusion is None}
    # No schedule features are used in the 121-field candidate. Empty explicit
    # schedule coverage supplies only the discarded allcomp branch.
    builder = PreparedPredictionFeatures(observations(source), AllCompetitionFeatures([]))
    groups = {}
    for target in targets:
        timing = derive_timing(target.match_date, 3, season_start_year=2021)
        key = (target.competition, timing['parameter_cutoff'])
        groups.setdefault(key, []).append((target, timing))
    rows, unavailable, reports, new_fits = [], [], {}, 0
    for (comp, cutoff), held in sorted(groups.items()):
        history = sorted((r for r in eligible_source.values() if r.competition == comp
                          and r.match_date < cutoff), key=lambda r: (r.match_date, r.match_id))
        identifiers = [r.match_id for r in history]
        path = OUT/'基础模型'/f'{comp}_{cutoff.isoformat()}.joblib'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            pair = joblib.load(path)
            action = 'reused'
        else:
            pair = {'goal': DateTeamGoalModel(.01, 180).fit([independent[mid] for mid in identifiers], cutoff),
                    'xg': DateTeamXgModel(.01, 180).fit(history, cutoff)}
            joblib.dump(pair, path, compress=3)
            new_fits += 2
            action = 'fitted'
        parameters = {arm: model.parameters() for arm, model in pair.items()}
        for arm, p in parameters.items():
            if (p['reference_match_ids'] != identifiers or p['before_date'] != cutoff.isoformat()
                    or not p['training_max_date'] < cutoff.isoformat()
                    or p['regularization'] != .01 or p['half_life_days'] != 180):
                raise ValueError('Cached earlier base design changed')
        key = f'{comp}:{cutoff.isoformat()}'
        reports[key] = {'goal': parameters['goal'], 'xg': parameters['xg'], 'cache_action': action}
        for record, timing in held:
            independent_record = independent[record.match_id]
            missing = {arm: sorted({record.home, record.away}-set(p['teams']))
                       for arm, p in parameters.items()}
            context = {'match_id': record.match_id, 'date': record.match_date.isoformat(),
                'period': '2021-22', 'competition': comp, 'home': record.home, 'away': record.away,
                'fit_before_date': cutoff.isoformat(),
                'forecast_before_date': timing['forecast_cutoff'].isoformat(), 'horizon_days': 3,
                'actual_score': list(independent_record.full_time),
                'actual_outcome': 0 if record.goals[0] > record.goals[1] else 2 if record.goals[0] < record.goals[1] else 1}
            if any(missing.values()):
                unavailable.append(context | {'reason': 'unseen_team_in_legal_earlier_base_parameters',
                                               'unseen_teams': missing})
                continue
            if (record.match_id in identifiers or record.goals != independent_record.full_time
                    or record.match_date != independent_record.match_date):
                raise ValueError('Target entered its own base training or source identity changed')
            gr, xr = (pair[arm].predict(record.home, record.away, comp, record.match_date)
                      for arm in ('goal', 'xg'))
            fresh = builder.for_match(home=record.home, away=record.away, competition=comp,
                parameter_cutoff=cutoff, forecast_cutoff=timing['forecast_cutoff'],
                match_date=record.match_date, goal_rates=gr, xg_rates=xr, refresh=True)
            if any(set(side) != expected_names for side in fresh['performance_features']):
                raise ValueError('Expansion feature names do not match original candidate')
            rows.append(context | {FEATURE: fresh['performance_features'],
                'fresh_feature_history': fresh['feature_history'], 'fresh_timing': fresh['timing'],
                'fresh_feature_before_date': fresh['feature_before_date'],
                'goal_rates': list(gr), 'xg_rates': list(xr), 'base_parameter_key': key,
                'base_training_max_date': parameters['goal']['training_max_date']})
        write(OUT/'基础生成进度.json', {'completed_parameter_pairs': len(reports),
            'planned_parameter_pairs': len(groups), 'added_rows_so_far': len(rows),
            'unavailable_so_far': len(unavailable), 'new_base_fits_this_execution': new_fits})
        print(json.dumps({'parameter': key, 'action': action, 'history_n': len(history),
                          'added_so_far': len(rows), 'completed': len(reports), 'planned': len(groups)}), flush=True)
    rows.sort(key=lambda r: (r['date'], r['match_id']))
    calendar_audit(rows)
    stored = {'rows': rows, 'unavailable': unavailable, 'source_feature_file': str(SOURCE),
        'target_source_n': len(targets), 'base_parameter_pairs': len(groups),
        'feature_n': 121, 'historical_first_publication_reconstructed': False}
    write(OUT/'新增基础模型参数.json', reports)
    dump_gzip(destination, stored)
    return stored, new_fits


def fit_one(name, rows, cutoff, expected_names, expected_configuration):
    selected = sorted((r for r in rows if date.fromisoformat(r['date']) < cutoff),
                      key=lambda r: (r['date'], r['match_id']))
    path = OUT/f'{name}.joblib'
    if path.exists():
        model, action = joblib.load(path), 'reused'
    else:
        model = EarlyGoalLearner('extra_trees', 'rich').fit(
            [{**row, 'side_features': row[FEATURE]} for row in selected], cutoff)
        joblib.dump(model, path, compress=3)
        action = 'fitted'
    p = model.parameters()
    if (p['reference_match_ids'] != [r['match_id'] for r in selected]
            or p['before_date'] != cutoff.isoformat() or set(p['feature_names']) != expected_names
            or p['configuration'] != expected_configuration):
        raise ValueError('Saved candidate does not match fixed earlier data/architecture')
    write(OUT/f'{name}_训练说明.json', p | {'action_this_execution': action,
        'weight_summary': weight_summary(selected, cutoff), 'source_feature_file': str(SOURCE)})
    print(json.dumps({'meta_fit': name, 'action': action, 'n': len(selected)}, ensure_ascii=False), flush=True)
    return model, action


def evaluate(model, rows):
    rates = model.predict_many([r[FEATURE] for r in rows], [date.fromisoformat(r['date']) for r in rows])
    predictions = [predict_row({k: r[k] for k in ('match_id', 'date', 'period', 'competition',
        'home', 'away', 'actual_score', 'actual_outcome')}, mean) for r, mean in zip(rows, rates, strict=True)]
    scores, _ = score_details(predictions)
    leagues = {comp: score_details([r for r in predictions if r['competition'] == comp])[0]
               for comp in sorted({r['competition'] for r in predictions})}
    return predictions, scores, leagues


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {'created_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'Add 2021/22 same-semantic lead3 features; preserve original 6967 rows verbatim.',
        'new_meta_fit_limit': 3, 'new_parameter_grids': 0,
        'algorithm': 'Existing ExtraTrees 120trees/depth6/leaf60/features0.8/seed20261025,180day weights.',
        'base_source_range': 'Exact joined eligible goal/xG IDs from 2020/21, per competition, strictly earlier parameter dates.',
        'test_comparison': 'Two single-cutoff fits before2025-08-01, same1726 2025/26 targets. Not comparable with earlier22window-refit scores.',
        'final_candidate': 'Expanded historical rows before2026-08-01; research only, not deployed.',
        'limits': ['Previously examined historical seasons, not untouched blind data.',
            'Date-level availability only; source initial publication/revisions unknown.',
            'Additional old rows receive small180day weights; larger row count is not proportional new information.',
            'No historical LLM/context rows are invented; no current results or current-season labelled training.'],
        'lineups_used': False, 'neural_networks_added': False, 'current_forecasts_replaced': False}
    if not (OUT/'实施计划.json').exists():
        write(OUT/'实施计划.json', plan)
    old = load_gzip(SOURCE)['rows']
    original_parameters = joblib.load(ORIGINAL_MODEL).parameters()
    expected_names = set(original_parameters['feature_names'])
    if len(old) != 6967 or len(expected_names) != 121:
        raise ValueError('Original training dimensions changed')
    calendar_audit(old)
    if set(original_parameters['reference_match_ids']) != {r['match_id'] for r in old}:
        raise ValueError('Original saved model cohort differs from original feature file')
    source, independent, joins, fd_reports, directories = load_source()
    write(OUT/'数据来源与规模.json', {'independent_fd_n': len(independent),
        'goal_eligible_n': sum(r.goal_fit_exclusion is None for r in independent.values()),
        'joined_xg_n': len(source), 'rich_original_n': len(old),
        'rich_original_by_period': dict(Counter(r['period'] for r in old)),
        'joined_by_competition': dict(Counter(r.competition for r in source.values())),
        'joins': joins, 'fd_reports': fd_reports, 'directories': directories})
    added, base_fits = prepare_added(source, independent, expected_names)
    combined = sorted([*added['rows'], *old], key=lambda r: (r['date'], r['match_id']))
    calendar_audit(combined)
    originals = {r['match_id']: r for r in old}
    if any(originals[r['match_id']] != r for r in combined if r['match_id'] in originals):
        raise ValueError('An original prepared row changed')
    dump_gzip(OUT/'扩样合并特征.json.gz', {'rows': combined, 'original_source': str(SOURCE),
        'added_source': str(OUT/'2021-22新增特征.json.gz')})
    held = [r for r in old if r['period'] == '2025-26']
    if len(held) != 1726 or any(r['forecast_before_date'] < HELD_CUTOFF.isoformat() for r in held):
        raise ValueError('Fixed-cutoff held cohort changed or arrives before learner fit')
    tracks, actions, scores, by_league = {}, {}, {}, {}
    for name, rows in (('原覆盖_20250801', old), ('扩样_20250801', combined)):
        model, action = fit_one(name, rows, HELD_CUTOFF, expected_names, original_parameters['configuration'])
        tracks[name], scores[name], by_league[name] = evaluate(model, held)
        actions[name] = action
        write(OUT/f'{name}_2025-26预测.json', tracks[name])
    final, action = fit_one('扩样_20260801_研究候选', combined, FINAL_CUTOFF,
                            expected_names, original_parameters['configuration'])
    actions['扩样_20260801_研究候选'] = action
    reloaded = joblib.load(OUT/'扩样_20260801_研究候选.joblib')
    example = combined[-1][FEATURE]
    if final.predict(example, FINAL_CUTOFF) != reloaded.predict(example, FINAL_CUTOFF):
        raise ValueError('Serialized final model does not replay')
    summary = {'finished_at': datetime.now(timezone.utc).isoformat(), 'original_n': len(old),
        'added_n': len(added['rows']), 'expanded_n': len(combined),
        'expanded_by_period': dict(Counter(r['period'] for r in combined)),
        'source_2021_22_n': added['target_source_n'], 'unseen_team_unavailable_n': len(added['unavailable']),
        'base_parameter_pairs': added['base_parameter_pairs'], 'new_base_fits_this_execution': base_fits,
        'meta_fit_actions': actions, 'new_meta_fits_this_execution': sum(a == 'fitted' for a in actions.values()),
        'fixed_cutoff': HELD_CUTOFF.isoformat(), 'held_n': len(held), 'scores': scores,
        'scores_by_league': by_league, 'original_rows_unchanged': True,
        'original_121_feature_semantics_preserved': True, 'target_ids_excluded_from_meta_training': True,
        'final_model_reloaded_prediction_exact': True, 'final_fit_before': FINAL_CUTOFF.isoformat(),
        'final_weight_summary': weight_summary(combined, FINAL_CUTOFF),
        'no_parameter_search': True, 'adopted': False, 'live_forecasts_changed': False,
        'scope': 'Previously examined retrospective data; single-cutoff comparison not a new blind or current accuracy estimate.',
        'elapsed_seconds': time.perf_counter()-start}
    write(OUT/'扩样训练结果.json', summary)
    print(json.dumps({k: summary[k] for k in ('original_n', 'added_n', 'expanded_n', 'held_n',
        'new_base_fits_this_execution', 'new_meta_fits_this_execution', 'scores', 'elapsed_seconds')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
