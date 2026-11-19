"""Clean end-to-end early prediction schemes, with matched all-competition arms.

Run one predeclared comparison; no parameter grid or prospective settlement.
"""
import argparse
from collections import Counter
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import time

import joblib
import numpy as np

from football_model.all_competition_features import AllCompetitionFeatures
from football_model.early_history_features import EarlyHistoryFeatures
from football_model.early_goal_learners import EarlyGoalLearner
from football_model.forecast_diagnostics import moving_block_means
from run_broad_early_models_20261025 import observations, base_features, predict_row, failure_diagnostics
from run_npxg_comparison_20261025 import load_source, ROOT, read
from run_shot_increment_20261025 import score_details, write


ARMS = {
    'glm_basic': ('glm', 'basic'), 'hgb_basic': ('hgb', 'basic'),
    'glm_performance': ('glm', 'performance'), 'hgb_performance': ('hgb', 'performance'),
    'extra_trees_performance': ('extra_trees', 'performance'),
    'glm_allcomp': ('glm', 'allcomp'), 'hgb_allcomp': ('hgb', 'allcomp'),
    'extra_trees_allcomp': ('extra_trees', 'allcomp'),
}
COMPS = ('E0', 'D1', 'SP1', 'I1', 'F1')
# These describe league-source gaps, not physical rest. Both rich variants omit
# them, so the schedule contrast is not confounded by removing a broken input.
OMITTED_SUFFIXES = (
    'cutoff_staleness_days', 'previous_league_interval_days',
    'known_league_matches_before_cutoff_7d', 'known_league_matches_before_cutoff_14d',
    'known_league_matches_before_cutoff_28d', 'same_league_gap_to_match_days',
)


def prepare_rows(base_path, schedule_directory):
    base = read(base_path)
    np_records, independent, joins, fd_reports, directories = load_source()
    league = EarlyHistoryFeatures(observations(np_records))
    schedule = AllCompetitionFeatures.from_directory(schedule_directory)
    contexts = []
    omitted = set()
    for original in base['rows']:
        record = independent[original['match_id']]
        if (record.goal_fit_exclusion or original['match_id'] not in np_records
                or original['date'] != record.match_date.isoformat()
                or original['actual_score'] != list(record.full_time)
                or original['home'] != record.home or original['away'] != record.away):
            raise ValueError('Clean base target differs from current independently joined source')
        kwargs = dict(home=original['home'], away=original['away'], competition=original['competition'],
                      cutoff=date.fromisoformat(original['fit_before_date']),
                      match_date=date.fromisoformat(original['date']))
        history = league.for_match(**kwargs)
        load = schedule.for_match(**kwargs)
        contexts.append((original, history, load))
    template = next(h['home_features'] for _, h, _ in contexts if h['home_features'] is not None)
    raw_keys = set(template)
    omitted = {k for k in raw_keys if any(k.endswith(s) for s in OMITTED_SUFFIXES)}
    performance_keys = sorted(raw_keys-omitted)
    rows, missing = [], []
    for original, history, load in contexts:
        basic = [base_features(original, i, original['xg_rates']) for i in (0, 1)]
        performance, integrated = [], []
        for i, side in enumerate(('home', 'away')):
            values = history[f'{side}_features']
            pf = {k: None if values is None else values[k] for k in performance_keys}
            pf.update(basic[i])
            pf['history_features_available'] = float(values is not None)
            # Forecast lead is known to BOTH rich schemes. Otherwise the
            # matched schedule contrast would also be a lead-time contrast.
            pf['forecast_log1p_lead_days'] = math.log1p(
                (date.fromisoformat(original['date'])-date.fromisoformat(original['fit_before_date'])).days)
            extra = dict(load[f'{side}_features'])
            if extra.pop('allcomp_log1p_forecast_lead_days') != pf['forecast_log1p_lead_days']:
                raise ValueError('Both rich schemes must share identical forecast lead time')
            if not isinstance(extra, dict) or set(extra) & set(pf):
                raise ValueError('All-competition feature namespace must be explicit and distinct')
            performance.append(pf)
            integrated.append({**pf, **extra})
        if history['home_features'] is None:
            missing.append(original['match_id'])
        h, a = original['actual_score']
        rows.append({k: original[k] for k in ('match_id', 'date', 'period', 'competition', 'home', 'away',
                                              'fit_before_date', 'actual_score')}
                    | {'actual_outcome': 0 if h > a else 2 if h < a else 1,
                       'basic_features': basic, 'performance_features': performance,
                       'allcomp_features': integrated, 'feature_history': history['metadata'],
                       'allcomp_history': load['metadata'], 'goal_rates': original['rates'],
                       'xg_rates': original['xg_rates'], 'clean_blend_rates': original['blend_rates']})
    rows.sort(key=lambda r: (r['date'], r['match_id']))
    if len(rows) != len(base['rows']) or len({r['match_id'] for r in rows}) != len(rows):
        raise ValueError('Every independently supported clean target must be retained once')
    sizes = {name: sorted({len(r[f'{name}_features'][i]) for r in rows for i in (0, 1)})
             for name in ('basic', 'performance', 'allcomp')}
    return rows, {'base_path': str(base_path), 'schedule_directory': str(schedule_directory),
        'source_matches': len(np_records), 'joins': joins, 'fd_reports': fd_reports,
        'independent_directories': directories, 'feature_counts': sizes,
        'omitted_league_schedule_fields': sorted(omitted), 'insufficient_league_histories_retained': missing,
        'by_period': dict(Counter(r['period'] for r in rows))}


def compare(tracks, selected=None, intervals=False):
    ids = [r['match_id'] for r in tracks['goal']]
    if any([r['match_id'] for r in rows] != ids for rows in tracks.values()):
        raise ValueError('Every whole-scheme comparison must use identical ordered targets')
    scores, losses = {}, {}
    for arm, rows in tracks.items():
        scores[arm], losses[arm] = score_details(rows)
    contrasts = [(f'{kind}_allcomp', f'{kind}_performance') for kind in ('glm', 'hgb', 'extra_trees')]
    contrasts += [('glm_performance', 'glm_basic'), ('hgb_performance', 'hgb_basic'),
                  ('hgb_basic', 'glm_basic'), ('extra_trees_allcomp', 'hgb_allcomp')]
    if selected is not None:
        contrasts.extend([(selected, 'goal'), (selected, 'clean_blend')])
    columns, values = [], []
    for arm, reference in contrasts:
        for domain in losses[arm]:
            columns.append(f'{arm}_minus_{reference}:{domain}')
            values.append(losses[arm][domain]-losses[reference][domain])
    matrix = np.column_stack(values)
    return {'n': len(ids), 'scores': scores, 'difference_order': columns,
        'mean_loss_differences': dict(zip(columns, matrix.mean(axis=0).tolist())), 'match_ids': ids,
        'block_intervals': {str(b): moving_block_means([r['date'] for r in tracks['goal']], matrix,
                              block_weeks=b) for b in (2, 4, 8)} if intervals else {}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--schedule', type=Path, default=ROOT/'outputs/全赛事赛程可用性_20261025')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--prepared', type=Path, help='Continue an already prepared, not yet fitted study directory')
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    out = args.prepared or ROOT/'outputs'/now.strftime('完整清理与全赛事联合模型_%Y%m%dT%H%M%S%fZ')
    if args.prepared is None:
        out.mkdir(parents=True)
        (out/'models').mkdir()
    elif list((out/'models').glob('*.joblib')) or (out/'完整方案结果汇总.json').exists():
        raise ValueError('Use a fresh prepared study; do not overwrite existing fitted models')
    plan = {'created_at': now.isoformat(), 'arms': ARMS,
        'purpose': 'clean whole-pipeline prediction with matched performance and all-competition inputs',
        'prior_validation': '2024-25', 'later_retrospective_test': '2025-26',
        'initial_training_examples': ['2022-23', '2023-24'],
        'selection': 'lowest earlier2024/25 exact-score NLL among all8 preset schemes; no later reselection',
        'data_correction': 'new incomplete-match exceptions applied to original experts, league histories, new learners, targets and scoring',
        'information_contrast': 'both rich variants omit the same raw league-gap/date/load fields; allcomp adds past complete competitive schedule summaries',
        'allcomp_timing': 'only observed events strictly before original28day fit cutoff; windows relative to cutoff, target lead time explicit',
        'negative_information': 'no fixture after cutoff, no current result, no actual or predicted lineup, no same-match goals in features',
        'allcomp_caveats': 'no complete playing-minute reconstruction or exact historical publication timestamps; missing coverage is unknown, not rest',
        'distribution': 'common independent Poisson full-score law after positive conditional full-goal means',
        'algorithm_settings': 'existing fixed GLM/HGB/ExtraTrees settings unchanged;180day half-life',
        'budget': '8 schemes x22 pooled cutoffs=176 fits;0 hyperparameter grids;26 affected original expert fits rebuilt separately',
        'primary': 'earlier-selected scheme minus clean earlier-trained goal/xG blend later full-score NLL',
        'secondary': ['1X2LL/Brier/Top1', 'total0/1/2/3/4+LL', 'away-minus2LL',
                      'matchedallcomp versus performance by learner', 'all5leagues and descriptive error groups'],
        'uncertainty': 'paired calendar2/4/8week block resampling,2000 repeats,conditional retrospective intervals',
        'historical_research': 'these later seasons have already been viewed; not an untouched blind test',
        'current_results_reviewed': False, 'lineups_used': False, 'neural_networks_added': False,
        'new_dependencies': False, 'adopted': False}
    if args.prepared is None:
        write(out/'分析前计划.json', plan)
        rows, source = prepare_rows(args.base, args.schedule)
        write(out/'数据接入与覆盖.json', source)
        write(out/'完整历史特征.json', {'rows': rows})
    else:
        plan = read(out/'分析前计划.json')
        source = read(out/'数据接入与覆盖.json')
        rows = read(out/'完整历史特征.json')['rows']
        if Path(source['base_path']).resolve() != args.base.resolve():
            raise ValueError('Prepared rows must retain their original base experts')
    print(json.dumps({'directory': str(out), 'rows': len(rows), 'features': source['feature_counts'],
                      'periods': source['by_period']}, ensure_ascii=False), flush=True)
    if args.prepare_only:
        return
    tracks = {arm: [] for arm in ARMS}
    tracks.update(goal=[], xg_raw=[], clean_blend=[])
    parameters, windows, selected = {}, [], None
    for period in ('2024-25', '2025-26'):
        targets = [r for r in rows if r['period'] == period]
        for cutoff in sorted({r['fit_before_date'] for r in targets}):
            started = time.perf_counter()
            held = [r for r in targets if r['fit_before_date'] == cutoff]
            history = [r for r in rows if r['date'] < cutoff]
            fits = {}
            for arm, (kind, feature_set) in ARMS.items():
                model = EarlyGoalLearner(kind, 'basic' if feature_set == 'basic' else 'rich')
                model.fit([{**r, 'side_features': r[f'{feature_set}_features']} for r in history],
                          date.fromisoformat(cutoff))
                key = f'{arm}:{cutoff}'
                parameters[key] = model.parameters() | {'pipeline_feature_set': feature_set}
                joblib.dump(model, out/'models'/f'{arm}_{cutoff}.joblib', compress=3)
                predicted = model.predict_many([r[f'{feature_set}_features'] for r in held],
                                               [date.fromisoformat(r['date']) for r in held])
                for row, rates in zip(held, predicted, strict=True):
                    context = {k: row[k] for k in ('match_id', 'date', 'period', 'competition', 'home', 'away',
                                                  'fit_before_date', 'actual_score', 'actual_outcome')}
                    tracks[arm].append(predict_row(context, rates))
                fits[arm] = {'training_n': len(history), 'prediction_n': len(held), 'parameters_key': key}
            for row in held:
                context = {k: row[k] for k in ('match_id', 'date', 'period', 'competition', 'home', 'away',
                                              'fit_before_date', 'actual_score', 'actual_outcome')}
                for arm, rate_key in (('goal', 'goal_rates'), ('xg_raw', 'xg_rates')):
                    tracks[arm].append(predict_row(context, row[rate_key]))
                if period == '2025-26':
                    tracks['clean_blend'].append(predict_row(context, row['clean_blend_rates']))
            windows.append({'period': period, 'cutoff': cutoff, 'fits': fits,
                            'seconds': time.perf_counter()-started})
            write(out/'训练进度.json', {'windows': windows, 'fits': len(parameters), 'selected': selected})
            print(json.dumps({'period': period, 'cutoff': cutoff, 'n': len(held), 'fits': len(parameters),
                              'seconds': windows[-1]['seconds']}, ensure_ascii=False), flush=True)
        for arm in tracks:
            tracks[arm].sort(key=lambda r: (r['date'], r['match_id']))
        local = {a: [r for r in values if r['period'] == period] for a, values in tracks.items()
                 if a != 'clean_blend' or period == '2025-26'}
        if period == '2024-25':
            validation = compare(local)
            selected = min(ARMS, key=lambda a: (validation['scores'][a]['exact_score']['log_loss'], a))
            write(out/'较早方案选择.json', {'selected': selected, 'rule': plan['selection'], 'validation': validation})
            print(json.dumps({'validation_selected': selected, 'scores': {
                a: v['exact_score']['log_loss'] for a,v in validation['scores'].items()}}, ensure_ascii=False), flush=True)
        else:
            test = compare(local, selected=selected, intervals=True)
            by_league = {c: compare({a: [r for r in values if r['competition'] == c]
                                     for a,values in local.items()}, selected=selected) for c in COMPS}
            diagnostics = failure_diagnostics(local, selected)
    write(out/'全部模型参数.json', parameters)
    write(out/'逐场完整预测.json', tracks)
    summary = {'plan': plan, 'finished_at': datetime.now(timezone.utc).isoformat(), 'selected': selected,
        'rows': len(rows), 'feature_counts': source['feature_counts'], 'new_scheme_fits': len(parameters),
        'validation': validation, 'test': test, 'by_league': by_league,
        'descriptive_failure_groups': diagnostics, 'windows': windows, 'adopted': False}
    write(out/'完整方案结果汇总.json', summary)
    print(json.dumps({'directory': str(out), 'selected': selected, 'fits': len(parameters),
        'test_scores': test['scores'], 'test_differences': test['mean_loss_differences']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
