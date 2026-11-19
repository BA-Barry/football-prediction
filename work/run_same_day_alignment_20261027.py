"""Align existing rich-model training and live inference to the same match day.

No feature/learner search. Reuse genuine earlier base means; regenerate all
performance histories at F=D with observations strictly before D. Historical
evaluation is retrospective, never an untouched test or a current win rate.
"""
from collections import Counter
from datetime import date, datetime, timezone
from dataclasses import replace
import gzip
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import time

import joblib
import numpy as np

from football_model.all_competition_features import AllCompetitionFeatures
from football_model.date_model import DateTeamGoalModel
from football_model.early_goal_learners import EarlyGoalLearner
from football_model.fixed_horizon_features import PreparedPredictionFeatures
from football_model.football_data import load_directory
from football_model.forecast_diagnostics import moving_block_means
from football_model.xg_history import DateTeamXgModel
from run_broad_early_models_20261025 import observations, predict_row
from run_npxg_comparison_20261025 import load_source
from run_shot_increment_20261025 import score_details
from current_rich_model_helpers_20261026 import load_current_rich_sources
from predict_serie_a_20261027 import distribution

ROOT = REPOSITORY_ROOT
OUT = ROOT / 'outputs/同日赛前模型一致化_20261027'
BASE = ROOT / 'outputs/干净基础专家重建_20261025T064408733954Z/clean_base_rows.json'
OLD = ROOT / 'outputs/固定发布日期信息刷新_20261025T071806561562Z/lead_3'
LIVE = ROOT / 'data/serie_a_today_20261027'
CURRENT_DATE = date(2026, 10, 27)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def parameter_day(day, season_year):
    anchor = date(season_year, 8, 1)
    from datetime import timedelta
    return anchor + timedelta(days=((day - anchor).days // 28) * 28)


def context(row):
    return {k: row[k] for k in ('match_id', 'date', 'period', 'competition', 'home', 'away',
            'fit_before_date', 'forecast_before_date', 'horizon_days', 'actual_score', 'actual_outcome')}


def prepare():
    base = read(BASE)
    source, independent, joins, _, _ = load_source()
    builder = PreparedPredictionFeatures(observations(source), AllCompetitionFeatures([]))
    rows = []
    for old in base['rows']:
        day = date.fromisoformat(old['date'])
        p = parameter_day(day, int(old['period'][:4]))
        record = independent[old['match_id']]
        if (old['fit_before_date'] != p.isoformat()
                or not old['training_max_date'] < p.isoformat() <= old['date']
                or record.match_date != day or record.goal_fit_exclusion
                or list(record.full_time) != old['actual_score']
                or record.home != old['home'] or record.away != old['away']):
            raise ValueError('Earlier base means differ from the same-day source/timing design')
        result = builder.for_same_day_match(
            home=old['home'], away=old['away'], competition=old['competition'],
            parameter_cutoff=p, forecast_cutoff=day, match_date=day,
            goal_rates=old['goal_rates'], xg_rates=old['xg_rates'], refresh=True)
        h, a = old['actual_score']
        if old['actual_outcome'] != (0 if h > a else 2 if h < a else 1):
            raise ValueError('Target outcome direction differs from its score')
        rows.append({k: old[k] for k in ('match_id', 'date', 'period', 'competition',
                     'home', 'away', 'fit_before_date', 'actual_score', 'actual_outcome',
                     'goal_rates', 'xg_rates', 'base_parameter_key', 'training_max_date')} | {
            'forecast_before_date': day.isoformat(), 'horizon_days': 0,
            'side_features': result['performance_features'],
            'feature_history': result['feature_history'], 'timing': result['timing']})
    rows.sort(key=lambda r: (r['date'], r['match_id']))
    if len({r['match_id'] for r in rows}) != len(rows):
        raise ValueError('Repeated historical fixture')
    with gzip.open(OUT / '同日历史输入.json.gz', 'wt', encoding='utf-8') as f:
        json.dump({'rows': rows}, f, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    write(OUT / '历史覆盖.json', {
        'n': len(rows), 'periods': dict(Counter(r['period'] for r in rows)),
        'feature_count_per_side': 121, 'base_means_reused': len(rows),
        'new_historical_base_fits': 0, 'source_join_reports': joins,
        'feature_rule': 'observation.date < forecast_date = target_date',
        'parameter_rule': 'Aug1-anchored 28 days; labels strictly before parameter date',
        'historical_publication_and_revision_reconstructed': False})
    return rows


def evaluate(rows):
    targets = [r for r in rows if r['period'] == '2025-26']
    tracks = {k: [] for k in ('same_day_rich', 'same_day_goal', 'same_day_xg',
                             'lead3_model_on_same_day_inputs')}
    fits = []
    for cutoff in sorted({r['fit_before_date'] for r in targets}):
        started = time.perf_counter()
        history = [r for r in rows if r['date'] < cutoff]
        held = [r for r in targets if r['fit_before_date'] == cutoff]
        if {r['match_id'] for r in history} & {r['match_id'] for r in held}:
            raise ValueError('Target entered its own training set')
        path = OUT / 'models' / f'same_day_{cutoff}.joblib'
        if path.exists():
            model = joblib.load(path)
            meta = model.parameters()
            if (meta['reference_match_ids'] != [r['match_id'] for r in history]
                    or meta['before_date'] != cutoff
                    or meta['configuration'] != EarlyGoalLearner('extra_trees', 'rich')._estimator().get_params()):
                raise ValueError('Saved fit differs from this input and preset')
            action = 'reused_completed_same_run'
        else:
            model = EarlyGoalLearner('extra_trees', 'rich').fit(history, date.fromisoformat(cutoff))
            joblib.dump(model, path, compress=3)
            action = 'fitted_once'
        new = model.predict_many([r['side_features'] for r in held],
                                 [date.fromisoformat(r['date']) for r in held])
        # Same target input vectors and parameter date, unchanged old learner;
        # only historical feature snapshots/training coverage differ.
        prior = joblib.load(OLD / 'models' / f'fresh_performance_{cutoff}.joblib')
        if prior.parameters()['before_date'] != cutoff:
            raise ValueError('Prior model date differs from the declared comparison')
        transferred = prior.predict_many([r['side_features'] for r in held],
                                         [date.fromisoformat(r['date']) for r in held])
        for row, nr, tr in zip(held, new, transferred, strict=True):
            c = context(row)
            for key, rates in (('same_day_rich', nr), ('same_day_goal', row['goal_rates']),
                               ('same_day_xg', row['xg_rates']), ('lead3_model_on_same_day_inputs', tr)):
                tracks[key].append(predict_row(c, rates))
        fits.append({'parameter_date': cutoff, 'training_n': len(history), 'target_n': len(held),
                     'training_max_date': max(r['date'] for r in history), 'action': action,
                     'seconds': time.perf_counter() - started, 'training_target_overlap': 0})
        write(OUT / '执行进度.json', {'windows': fits})
        print(json.dumps({'stage': 'historical_window', **fits[-1]}, ensure_ascii=False), flush=True)
    for values in tracks.values():
        values.sort(key=lambda r: (r['date'], r['match_id']))
    scores, losses = {}, {}
    for key, values in tracks.items():
        scores[key], losses[key] = score_details(values)
    # Use actual available loss names instead of assuming an external key label.
    contrasts = {}
    for ref in ('same_day_goal', 'same_day_xg', 'lead3_model_on_same_day_inputs'):
        names = list(losses['same_day_rich'])
        diff = np.column_stack([losses['same_day_rich'][name] - losses[ref][name] for name in names])
        contrasts[ref] = {'loss_order': names, 'means': dict(zip(names, diff.mean(axis=0).tolist())),
                         'four_week_conditional_block_interval': moving_block_means(
                             [r['date'] for r in tracks['same_day_rich']], diff, block_weeks=4)}
    old = [r for r in read(OLD / '逐场完整预测.json')['fresh_performance'] if r['period'] == '2025-26']
    ids = {r['match_id'] for r in old}
    common = [r for r in tracks['same_day_rich'] if r['match_id'] in ids]
    if len(common) != len(old):
        raise ValueError('Old retrospective support is not a subset of this same-day support')
    common_scores, _ = score_details(common)
    old_scores, _ = score_details(old)
    result = {
        'n': len(targets), 'scores': scores, 'contrasts_same_inputs': contrasts,
        'by_league': {comp: {key: score_details([r for r in values if r['competition'] == comp])[0]
                            for key, values in tracks.items()} for comp in ('E0', 'D1', 'SP1', 'I1', 'F1')},
        'old_lead3_publication_description': {'common_n': len(common), 'same_day_rich': common_scores,
             'old_original_lead3': old_scores,
             'not_controlled_same_publication': True,
             'additional_same_day_ids': sorted(set(r['match_id'] for r in targets) - ids)},
        'fits': fits, 'untouched_holdout': False, 'historical_results_previously_viewed': True,
        'parameter_searches': 0, 'current_results_queried': False,
        'full_season_league_results_not_for_model_selection': True}
    write(OUT / '历史逐场预测.json', tracks)
    write(OUT / '历史结果.json', result)
    return result


def current(rows):
    now = datetime.now(timezone.utc)
    if now.date() != CURRENT_DATE:
        raise ValueError('Explicit live date must match the real UTC publication day')
    fixtures = read(LIVE / 'confirmed_fixtures.json')['fixtures']
    for f in fixtures:
        if datetime.fromisoformat(f['kickoff']) <= now or f['status'] != 'scheduled':
            raise ValueError('Target must remain a future scheduled fixture')
    p = parameter_day(CURRENT_DATE, 2026)
    history = [r for r in rows if r['date'] < p.isoformat()]
    tree = EarlyGoalLearner('extra_trees', 'rich').fit(history, p)
    records, _ = load_directory(LIVE / 'raw', season='2026/27')
    records = [replace(r, markets={}) for r in records if r.competition == 'I1']
    sources = load_current_rich_sources(LIVE / 'raw/Serie_A_2026.json', records,
                                      before_date=CURRENT_DATE, season='2026/27', competition='I1')
    goal_history = [replace(r, markets={}) for r in sources['independent'].values()
                    if r.competition == 'I1' and r.match_date < p]
    xg_history = [r for r in sources['np_records'].values() if r.competition == 'I1' and r.match_date < p]
    goal = DateTeamGoalModel(.01, 180).fit(goal_history, p)
    xg = DateTeamXgModel(.01, 180).fit(xg_history, p)
    builder = PreparedPredictionFeatures(sources['observations'], AllCompetitionFeatures([]))
    output, pairs = [], []
    for f in fixtures:
        home, away = ('football-data:I1:' + f[k] for k in ('home_model_label', 'away_model_label'))
        gr = goal.predict(home, away, 'I1', CURRENT_DATE)
        xr = xg.predict(home, away, 'I1', CURRENT_DATE)
        features = builder.for_same_day_match(
            home=home, away=away, competition='I1', parameter_cutoff=p,
            forecast_cutoff=CURRENT_DATE, match_date=CURRENT_DATE, goal_rates=gr, xg_rates=xr, refresh=True)
        rate = tree.predict(features['performance_features'], CURRENT_DATE)
        pairs.append(features['performance_features'])
        output.append({'fixture': f, 'actual_score': None, 'actual_outcome': None,
                       'forecast_before_date': CURRENT_DATE.isoformat(), 'parameter_before_date': p.isoformat(),
                       'model': distribution(rate), 'goal_parameter_rates': list(gr), 'xg_parameter_rates': list(xr),
                       'feature_history': features['feature_history'], 'timing': features['timing']})
    model_path = OUT / '同日当前模型.joblib'
    joblib.dump({'rich': tree, 'goal': goal, 'xg': xg}, model_path, compress=3)
    restored = joblib.load(model_path)['rich'].predict_many(pairs, [CURRENT_DATE] * len(fixtures))
    if not np.array_equal(np.asarray(restored), np.asarray([r['model']['rates'] for r in output])):
        raise ValueError('Saved model probabilities do not reproduce issued means')
    generated = datetime.now(timezone.utc).isoformat()
    result = {'generated_at': generated, 'class_order': ['home', 'draw', 'away'],
              'model_file': str(model_path), 'records': output,
              'base_parameter_date': p.isoformat(), 'historical_training_lead_days': 0,
              'new_fits': {'rich_current': 1, 'goal_current': 1, 'xg_current': 1},
              'rich_training_n': len(history), 'rich_parameters': tree.parameters(),
              'goal_parameters': goal.parameters(), 'xg_parameters': xg.parameters(),
              'current_season_labels_added_to_tree_training': False,
              'schedule_features_used': False, 'availability_used_numerically': False,
              'target_result_input': False, 'old_predictions_overwritten': False,
              'role': 'separate prospective research record, not a proven accuracy upgrade'}
    write(OUT / '意甲同日新版本.json', result)
    print(json.dumps({'stage': 'current_ready', 'generated_at': generated,
                     'p': p.isoformat(), 'probabilities': [r['model']['probabilities'] for r in output]},
                     ensure_ascii=False), flush=True)
    return result


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / 'models').mkdir(exist_ok=True)
    plan = {'created_at': datetime.now(timezone.utc).isoformat(),
            'problem': 'lead3 historical training applied to lead0 inputs and unaligned parameter ages',
            'repair': 'genuine lead0 historical features; unchanged 28-day parameter cadence at training and inference',
            'unchanged': '121 features,120 ExtraTrees,depth6,leaf60,.8 feature fraction,seed20261025,180-day weights,independent Poisson',
            'historical_target': '2025/26 full five-league eligible 1732 fixtures; earlier dates only for each fit',
            'fit_plan': '11 chronological rich fits +1 current rich +2 current base fits;0 historical base fits;0 search',
            'stop': 'finish this full fixed application check; do not tune or add algorithms based on its scores',
            'new_lineups_neural_models_or_market_pooling': False,
            'historical_test_status': 'previously viewed retrospective seasons, not an untouched holdout'}
    if not (OUT / '执行安排.json').exists():
        write(OUT / '执行安排.json', plan)
    cache = OUT / '同日历史输入.json.gz'
    if cache.exists():
        with gzip.open(cache, 'rt', encoding='utf-8') as f:
            rows = json.load(f)['rows']
    else:
        rows = prepare()
    print(json.dumps({'stage': 'prepared', 'n': len(rows)}, ensure_ascii=False), flush=True)
    historical = evaluate(rows)
    live = current(rows)
    write(OUT / '完成记录.json', {'finished_at': datetime.now(timezone.utc).isoformat(),
           'historical_target_n': historical['n'], 'new_historical_tree_fits': sum(r['action'] == 'fitted_once' for r in historical['fits']),
           'reused_historical_tree_fits': sum(r['action'] != 'fitted_once' for r in historical['fits']),
           'new_current_fits': live['new_fits'], 'new_historical_base_fits': 0,
           'parameter_searches': 0, 'new_llm_generations': 0, 'current_results_queried': False})


if __name__ == '__main__':
    main()
