"""Fixed publication dates, matched fresh/locked whole-pipeline experiments."""
import argparse
from collections import Counter
from datetime import date, datetime, timezone
import gc
import gzip
import json
from pathlib import Path
import time

import joblib
import numpy as np

from football_model.all_competition_features import AllCompetitionFeatures
from football_model.fixed_horizon_features import PreparedPredictionFeatures, derive_timing
from football_model.early_goal_learners import EarlyGoalLearner
from football_model.forecast_diagnostics import moving_block_means
from run_broad_early_models_20261025 import observations, predict_row
from run_npxg_comparison_20261025 import load_source, read, ROOT
from run_shot_increment_20261025 import score_details, write


ARMS = ('fresh_performance', 'fresh_allcomp', 'locked_allcomp')
COMPS = ('E0', 'D1', 'SP1', 'I1', 'F1')
CONTRASTS = [('fresh_allcomp', 'locked_allcomp'), ('fresh_allcomp', 'fresh_performance'),
             ('fresh_allcomp', 'clean_blend'), ('fresh_performance', 'clean_blend')]
CONTEXT_KEYS = ('match_id', 'date', 'period', 'competition', 'home', 'away', 'fit_before_date',
                'forecast_before_date', 'horizon_days', 'actual_score', 'actual_outcome')


def dump_features(path, rows):
    # Preserve every history reference without writing several giant pretty
    # printed copies of each feature dictionary. Ordinary gzip, no new format.
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        json.dump({'rows': rows}, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def read_features(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)['rows']


def prepare(group, lead, builder, independent):
    rows, counts = [], Counter()
    for base in group['rows']:
        identifier = base['match_id']
        record = independent[identifier]
        played = date.fromisoformat(base['date'])
        timing = derive_timing(played, lead, season_start_year=int(base['period'][:4]))
        parameter, forecast = timing['parameter_cutoff'], timing['forecast_cutoff']
        if (base['fit_before_date'] != parameter.isoformat()
                or base['forecast_before_date'] != forecast.isoformat()
                or record.goal_fit_exclusion or played != record.match_date
                or base['actual_score'] != list(record.full_time)
                or base['home'] != record.home or base['away'] != record.away
                or not base['training_max_date'] < parameter.isoformat() <= forecast.isoformat() < base['date']):
            raise ValueError('Fixed-horizon base input conflicts with source or publication timing')
        kwargs = dict(home=base['home'], away=base['away'], competition=base['competition'],
                      parameter_cutoff=parameter, forecast_cutoff=forecast, match_date=played,
                      goal_rates=base['rates'], xg_rates=base['xg_rates'])
        fresh, locked = builder.for_match(**kwargs, refresh=True), builder.for_match(**kwargs, refresh=False)
        if any(base.get(k) is None for k in CONTEXT_KEYS):
            raise ValueError('Each target needs explicit publication, parameter and observed identities')
        context = {k: base[k] for k in CONTEXT_KEYS}
        row = context | {
            'fresh_performance_features': fresh['performance_features'],
            'fresh_allcomp_features': fresh['allcomp_features'],
            'locked_allcomp_features': locked['allcomp_features'],
            'fresh_feature_history': fresh['feature_history'], 'fresh_allcomp_history': fresh['allcomp_history'],
            'locked_feature_history': locked['feature_history'], 'locked_allcomp_history': locked['allcomp_history'],
            'fresh_timing': fresh['timing'], 'locked_timing': locked['timing'],
            'fresh_feature_before_date': fresh['feature_before_date'],
            'locked_feature_before_date': locked['feature_before_date'],
            'goal_rates': base['rates'], 'xg_rates': base['xg_rates'], 'clean_blend_rates': base['blend_rates'],
            'blend_role': base['blend_role'], 'blend_weight_before_date': base['blend_weight_before_date'],
            'base_parameter_key': base['base_parameter_key'], 'base_training_max_date': base['training_max_date']}
        for arm in ARMS:
            columns = row[arm+'_features']
            expected = 121 if arm == 'fresh_performance' else 161
            if any(len(side) != expected for side in columns):
                raise ValueError('The matched rich schemes need the same declared feature definitions')
            if columns[0]['history_features_available'] == 0:
                counts[arm+'_insufficient_history'] += 1
        if parameter != forecast:
            counts['publication_after_parameter_day'] += 1
        rows.append(row)
    rows.sort(key=lambda r: (r['date'], r['match_id']))
    if len({r['match_id'] for r in rows}) != len(rows):
        raise ValueError('Target identities must remain unique')
    return rows, {'n': len(rows), 'by_period': dict(Counter(r['period'] for r in rows)),
                  'diagnostic_counts': dict(counts), 'unavailable': group['unavailable']}


def compare(tracks, intervals=False, allow_blend=True):
    ids = [r['match_id'] for r in tracks['goal']]
    if any([r['match_id'] for r in values] != ids for values in tracks.values()):
        raise ValueError('All forecast schemes must share target identities and order within a horizon')
    scores, losses = {}, {}
    for arm, rows in tracks.items():
        scores[arm], losses[arm] = score_details(rows)
    contrasts = [(a, b) for a, b in CONTRASTS if allow_blend or b != 'clean_blend']
    columns, values = [], []
    for arm, reference in contrasts:
        for domain in losses[arm]:
            columns.append(f'{arm}_minus_{reference}:{domain}')
            values.append(losses[arm][domain]-losses[reference][domain])
    matrix = np.column_stack(values)
    return {'n': len(ids), 'scores': scores, 'difference_order': columns,
        'mean_loss_differences': dict(zip(columns, matrix.mean(axis=0).tolist())), 'match_ids': ids,
        'block_intervals': {str(w): moving_block_means([r['date'] for r in tracks['goal']], matrix,
                              block_weeks=w) for w in (2, 4, 8)} if intervals else {}}


def train_horizon(out, lead, rows, plan):
    folder = out/f'lead_{lead}'
    (folder/'models').mkdir(parents=True, exist_ok=True)
    tracks = {a: [] for a in (*ARMS, 'goal', 'xg_raw', 'clean_blend')}
    parameters, windows = {}, []
    new_fits, reused_fits = 0, 0
    for period in ('2024-25', '2025-26'):
        targets = [r for r in rows if r['period'] == period]
        for cutoff in sorted({r['fit_before_date'] for r in targets}):
            started = time.perf_counter()
            held = [r for r in targets if r['fit_before_date'] == cutoff]
            history = [r for r in rows if r['date'] < cutoff]
            for arm in ARMS:
                path = folder/'models'/f'{arm}_{cutoff}.joblib'
                if path.exists():
                    model = joblib.load(path)
                    metadata = model.parameters()
                    if (metadata['reference_match_ids'] != [r['match_id'] for r in history]
                            or metadata['before_date'] != cutoff
                            or metadata['feature_names'] != sorted(history[0][arm+'_features'][0])
                            or metadata['configuration'] != EarlyGoalLearner('extra_trees', 'rich')._estimator().get_params()
                            or metadata['half_life_days'] != 180):
                        raise ValueError('Saved model does not match this unchanged prepared experiment')
                    reused_fits += 1
                else:
                    model = EarlyGoalLearner('extra_trees', 'rich').fit(
                        [{**r, 'side_features': r[arm+'_features']} for r in history], date.fromisoformat(cutoff))
                    joblib.dump(model, path, compress=3)
                    new_fits += 1
                key = f'{arm}:{cutoff}'
                parameters[key] = model.parameters() | {'pipeline': arm, 'horizon_days': lead}
                predicted = model.predict_many([r[arm+'_features'] for r in held],
                                               [date.fromisoformat(r['date']) for r in held])
                for row, rates in zip(held, predicted, strict=True):
                    if not cutoff <= row['forecast_before_date'] < row['date']:
                        raise ValueError('A fitted model cannot arrive after the forecast publication date')
                    context = {k: row[k] for k in CONTEXT_KEYS}
                    context['feature_before_date'] = row[
                        'locked_feature_before_date' if arm == 'locked_allcomp' else 'fresh_feature_before_date']
                    tracks[arm].append(predict_row(context, rates))
            for row in held:
                context = {k: row[k] for k in CONTEXT_KEYS}
                for arm, rate in (('goal', 'goal_rates'), ('xg_raw', 'xg_rates')):
                    tracks[arm].append(predict_row(context, row[rate]))
                if period == '2025-26':
                    if (row['blend_role'] != 'earlier_calibration_fixed_before_publication'
                            or row['blend_weight_before_date'] > row['forecast_before_date']):
                        raise ValueError('Comparator blend must be learned before the publication date')
                    tracks['clean_blend'].append(predict_row(context, row['clean_blend_rates']))
            windows.append({'period': period, 'cutoff': cutoff, 'training_n': len(history), 'target_n': len(held),
                            'seconds': time.perf_counter()-started, 'finished_fits': len(parameters)})
            write(folder/'训练进度.json', {'windows': windows, 'fits': len(parameters),
                                          'new_fits_this_execution': new_fits, 'reused_completed_fits': reused_fits})
            print(json.dumps({'lead': lead, **windows[-1]}, ensure_ascii=False), flush=True)
        for arm in tracks:
            tracks[arm].sort(key=lambda r: (r['date'], r['match_id']))
        local = {a: [r for r in values if r['period'] == period] for a,values in tracks.items()
                 if a != 'clean_blend' or period == '2025-26'}
        if period == '2024-25':
            validation = compare(local, allow_blend=False)
            write(folder/'较早完整比较.json', validation)
            print(json.dumps({'lead': lead, 'validation_scores': {a: v['exact_score']['log_loss']
                for a,v in validation['scores'].items()}, 'selection': 'none; retain all preset schemes'}, ensure_ascii=False), flush=True)
        else:
            test = compare(local, intervals=True)
            leagues = {c: compare({a: [r for r in values if r['competition'] == c]
                                   for a,values in local.items()}) for c in COMPS}
    write(folder/'全部模型参数.json', parameters)
    write(folder/'逐场完整预测.json', tracks)
    summary = {'lead_days': lead, 'validation': validation, 'test': test, 'by_league': leagues,
               'new_scheme_fits': len(parameters), 'new_fits_this_execution': new_fits,
               'reused_completed_fits': reused_fits, 'windows': windows, 'adopted': False}
    write(folder/'完整结果汇总.json', summary)
    print(json.dumps({'lead': lead, 'finished_fits': len(parameters), 'test_scores': test['scores'],
                      'test_differences': test['mean_loss_differences']}, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--schedule', type=Path, default=ROOT/'outputs/全赛事赛程可用性_20261025')
    parser.add_argument('--resume', type=Path, help='Continue this prepared study after an interrupted execution')
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    out = args.resume.resolve() if args.resume else ROOT/'outputs'/now.strftime('固定发布日期信息刷新_%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=bool(args.resume))
    plan = {'created_at': now.isoformat(), 'base_path': str(args.base.resolve()), 'horizons': [1, 3],
        'arms_per_horizon': ARMS, 'parameter_policy': 'Aug1 anchored28days, P<=F=D-L; every layer respects publication day',
        'feature_policy': 'fresh uses strictly earlier than F;locked uses strictly earlier than P; own historical training examples follow same L',
        'learner': 'unchanged ExtraTrees preset120trees depth6 leaf60 featurefraction.8 seed20261025;180day weights',
        'features': {'fresh_performance': 121, 'fresh_allcomp': 161, 'locked_allcomp': 161},
        'preset_origin': 'earlier2024/25 choice in preceding whole-scheme research; no new learner/lead selection',
        'primary': 'each horizon separately: fresh_allcomp minus locked_allcomp complete-score NLL in2025/26',
        'secondary': ['freshallcomp minus performance', 'earliertrained goal/xGblend', '1X2LL/Brier/Top1',
                      'total0/1/2/3/4+LL', 'awayminus2LL', 'fiveleague descriptions'],
        'earlier_comparison': '2024-25; no choosing a best arm or best lead after later results',
        'later_comparison': '2025-26;already viewed historical seasons, retrospective not untouched test',
        'fairness': 'same targets and base means within horizon; both fresh lines share performance/timing, only40schedule inputs differ',
        'missing': 'retain source/history uncertainty; unknown teams in legal earlier experts explicitly unavailable, not zero',
        'blend': 'separate earlier2024/25 weight per horizon fixed later; calibration-year blend excluded from selection/scoring',
        'budget': '3schemes x2horizons x22 pooled parameterdates=132fits;0grids;alldeclared metrics kept',
        'uncertainty': 'paired2/4/8calendarweek conditional block intervals;2000repeats;not causal fatigue effects',
        'scope_limits': ['datelevel availability assumption; historicalpublication/revisions unknown',
            '1/3calendar days not exact24/72hours', 'source allcompetition completeness not certified',
            'domesticleague strength scales not universal UEFA scale', 'conditionalmeans with common independentPoisson law'],
        'lineups_used': False, 'current_results_reviewed': False, 'new_dependencies': False,
        'neural_networks_added': False, 'adopted': False}
    if args.resume:
        plan = read(out/'分析前计划.json')
        if Path(plan['base_path']).resolve() != args.base.resolve():
            raise ValueError('Resume must use the same earlier base rates')
    else:
        write(out/'分析前计划.json', plan)
    base = read(args.base)
    np_records, independent, joins, reports, directories = load_source()
    write(out/'当前源接入.json', {'n': len(np_records), 'joins': joins, 'fd_reports': reports,
                                  'independent_directories': directories})
    summaries = {}
    coverage = read(out/'覆盖与不可预测.json') if args.resume else {}
    for lead in (1, 3):
        # Rebuild the cache for each horizon to keep peak memory bounded.
        feature_path = out/f'features_lead{lead}.json.gz'
        if args.resume and feature_path.exists():
            rows = read_features(feature_path)
            cover = read(out/'覆盖与不可预测.json')[str(lead)]
        else:
            builder = PreparedPredictionFeatures(observations(np_records), AllCompetitionFeatures.from_directory(args.schedule))
            rows, cover = prepare(base['horizons'][str(lead)], lead, builder, independent)
            dump_features(feature_path, rows)
            del builder
        coverage[str(lead)] = cover
        write(out/'覆盖与不可预测.json', coverage)
        print(json.dumps({'directory': str(out), 'prepared_lead': lead, 'coverage': {
            k:v for k,v in cover.items() if k != 'unavailable'}}, ensure_ascii=False), flush=True)
        gc.collect()
        summaries[str(lead)] = train_horizon(out, lead, rows, plan)
        del rows
        gc.collect()
    result = {'plan': plan, 'finished_at': datetime.now(timezone.utc).isoformat(), 'coverage': coverage,
              'horizons': summaries, 'new_scheme_fits': sum(s['new_scheme_fits'] for s in summaries.values()),
              'adopted': False}
    write(out/'固定发布日期完整汇总.json', result)
    print(json.dumps({'directory': str(out), 'finished_fits': result['new_scheme_fits'], 'adopted': False}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
