"""Independent retrospective audit for the fixed direct-outcome comparison.

This script never fits a learner or constructs production features. It checks
the dated home-view projection, recomputes preprocessing and classification
scores, and replays saved models when complete study outputs are available.
"""
from datetime import date, timedelta
import argparse
import gzip
import json
import math
from pathlib import Path
import warnings

import joblib
import numpy as np
from scipy.special import softmax


ROOT = Path(r'G:\Codex\football-prediction-2026-10-22')
SOURCE = ROOT/'outputs'/'固定发布日期信息刷新_20261025T071806561562Z'
CLASS_NAMES = ('home', 'draw', 'away')
ARMS = {'logistic_performance': ('logistic', 'performance'),
        'logistic_allcomp': ('logistic', 'allcomp'),
        'extra_trees_performance': ('extra_trees', 'performance'),
        'extra_trees_allcomp': ('extra_trees', 'allcomp')}
CONTEXT = ('match_id', 'date', 'period', 'competition', 'home', 'away', 'actual_score',
           'actual_outcome', 'fit_before_date', 'forecast_before_date', 'horizon_days')


def read(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def write(path, payload):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def load_rows(source, lead):
    with gzip.open(Path(source)/f'features_lead{lead}.json.gz', 'rt', encoding='utf-8') as handle:
        return json.load(handle)['rows']


def parameter_day(played, lead):
    publication = played-timedelta(days=lead)
    anchor = date(played.year if played.month >= 7 else played.year-1, 8, 1)
    return anchor+timedelta(days=((publication-anchor).days//28)*28)


def projection_audit(rows, lead):
    """Verify that exactly one complete home-perspective row is sufficient."""
    assert len({row['match_id'] for row in rows}) == len(rows)
    assert [(row['date'], row['match_id']) for row in rows] == sorted(
        (row['date'], row['match_id']) for row in rows)
    shared = {'history_features_available', 'forecast_log1p_lead_days',
              'parameter_log1p_age_at_forecast_days'} | {
                  f'league_{league}' for league in ('E0', 'D1', 'SP1', 'I1', 'F1')}
    swapped = {'log_goal_mean_for': 'log_goal_mean_against',
               'log_goal_mean_against': 'log_goal_mean_for',
               'log_xg_mean_for': 'log_xg_mean_against',
               'log_xg_mean_against': 'log_xg_mean_for'}
    keys_by_family, pairs, history_dates = {}, 0, 0
    for row in rows:
        played = date.fromisoformat(row['date'])
        publication = date.fromisoformat(row['forecast_before_date'])
        parameter = date.fromisoformat(row['fit_before_date'])
        assert publication == played-timedelta(days=lead)
        assert parameter == parameter_day(played, lead)
        assert parameter <= publication < played
        assert row['horizon_days'] == lead
        assert row['fresh_feature_before_date'] == publication.isoformat()
        assert row['base_training_max_date'] < parameter.isoformat()
        home_score, away_score = row['actual_score']
        assert row['actual_outcome'] == (0 if home_score > away_score else 2 if home_score < away_score else 1)
        for family, count in (('fresh_performance', 121), ('fresh_allcomp', 161)):
            both = row[family+'_features']
            assert len(both) == 2
            home, away = both
            names = set(home)
            assert names == set(away) and len(names) == count
            if family not in keys_by_family:
                keys_by_family[family] = names
            assert names == keys_by_family[family]
            assert home['is_home'] == 1 and away['is_home'] == 0
            accounted = {'is_home'} | shared | set(swapped)
            for name in names:
                if name.startswith('own_'):
                    counterpart = 'opponent_'+name[4:]
                elif name.startswith('opponent_'):
                    counterpart = 'own_'+name[9:]
                elif name in swapped:
                    counterpart = swapped[name]
                elif name in shared:
                    counterpart = name
                else:
                    assert name == 'is_home', (family, name, 'unaccounted feature')
                    continue
                assert home[name] == away[counterpart], (row['match_id'], family, name)
                accounted.add(name)
                pairs += 1
            assert names <= accounted
            for value in home.values():
                assert value is None or (isinstance(value, (int, float)) and math.isfinite(value))
            assert home['forecast_log1p_lead_days'] == math.log1p(lead)
            assert home['parameter_log1p_age_at_forecast_days'] == math.log1p((publication-parameter).days)
        performance = row['fresh_performance_features'][0]
        allcomp = row['fresh_allcomp_features'][0]
        assert set(performance) < set(allcomp)
        assert all(allcomp[name] == value for name, value in performance.items())
        assert all(name.startswith(('own_allcomp_', 'opponent_allcomp_'))
                   for name in set(allcomp)-set(performance))
        feature_history = row['fresh_feature_history']
        assert feature_history['cutoff'] == publication.isoformat()
        for side in ('home', 'away'):
            history = feature_history['history'][side]
            assert history['team'] == row[side]
            latest = history['latest_used_date']
            if latest is not None:
                assert latest < publication.isoformat()
                history_dates += 1
            assert row['match_id'] not in history['used_match_ids']
            schedule = row['fresh_allcomp_history']['history'][side]
            assert schedule['team'] == row[side]
            assert schedule['cutoff'] == publication.isoformat()
            for window in schedule['windows'].values():
                assert all(day < publication.isoformat() for day in window['selected_event_dates'])
                history_dates += len(window['selected_event_dates'])
            last = schedule['last_completed_event']
            if last is not None:
                assert last['date'] < publication.isoformat()
            assert all(event['date'] < publication.isoformat() for event in schedule['previous3_events'])
    return {'status': 'passed', 'horizon_days': lead, 'fixtures': len(rows),
            'home_feature_counts': {key: len(value) for key, value in keys_by_family.items()},
            'mirrored_feature_values_compared': pairs,
            'strictly_earlier_documented_history_dates_checked': history_dates,
            'one_home_view_contains_both_teams': True,
            'extra_allcompetition_features': 40,
            'away_mirror_duplicated_as_training_label': False,
            'scope': 'projection/timing of previously built inputs; no new raw feature reconstruction'}


def metrics(probabilities, labels):
    p, y = np.asarray(probabilities, dtype=float), np.asarray(labels, dtype=int)
    assert p.shape == (len(y), 3)
    assert np.isfinite(p).all() and np.all(p > 0) and np.all(p <= 1)
    assert np.max(np.abs(p.sum(axis=1)-1)) < 1e-10
    assert set(y) <= {0, 1, 2}
    truth = np.eye(3)[y]
    vectors = {'log_loss': -np.log(p[np.arange(len(y)), y]),
               'brier_sum': ((p-truth)**2).sum(axis=1),
               'rps': ((np.cumsum(p, axis=1)[:, :2]-np.cumsum(truth, axis=1)[:, :2])**2).sum(axis=1)/2,
               'misclassification': (np.argmax(p, axis=1) != y).astype(float)}
    scores = {name: float(values.mean()) for name, values in vectors.items()}
    scores['accuracy'] = float(np.mean(np.argmax(p, axis=1) == y))
    scores['n'] = len(y)
    del scores['misclassification']
    return scores, vectors


def calibration(probabilities, labels):
    p, y = np.asarray(probabilities, dtype=float), np.asarray(labels, dtype=int)
    bins = []
    for c, label in enumerate(CLASS_NAMES):
        assignments = np.minimum((p[:, c]*10).astype(int), 9)
        for number in range(10):
            selected = assignments == number
            n = int(selected.sum())
            if n:
                bins.append({'outcome': label, 'bin': number, 'n': n,
                             'mean_probability': float(p[selected, c].mean()),
                             'observed_frequency': float(np.mean(y[selected] == c))})
    return bins


def independent_blocks(dates, matrix, weeks, repeats=2000, seed=20261023):
    """Calendar-week blocks including empty weeks; no production helper used."""
    days = [date.fromisoformat(value) for value in dates]
    first = min(days)-timedelta(days=min(days).weekday())
    indices = np.asarray([(value-first).days//7 for value in days])
    span = int(max(indices))+1
    counts = np.asarray([int(np.sum(indices == i)) for i in range(span)])
    sums = np.vstack([matrix[indices == i].sum(axis=0) for i in range(span)])
    starts = np.random.default_rng(seed).integers(0, span-weeks+1,
        size=(repeats, (span+weeks-1)//weeks))
    samples = []
    for row in starts:
        selected = np.asarray([i for start in row for i in range(start, start+weeks)][:span])
        denominator = int(counts[selected].sum())
        if denominator:
            samples.append(sums[selected].sum(axis=0)/denominator)
    return np.quantile(np.asarray(samples), [.025, .975], axis=0).T


def assert_close(expected, saved, label, tolerance=1e-9):
    first, second = np.asarray(expected, dtype=float), np.asarray(saved, dtype=float)
    assert first.shape == second.shape, (label, first.shape, second.shape)
    error = float(np.abs(first-second).max(initial=0))
    assert np.allclose(first, second, atol=tolerance, rtol=1e-11), (label, error)
    return error


def preprocessing_audit(rows, parameter, family):
    """Recompute earlier-only imputation/weighted scaling without sklearn fit."""
    before = parameter['before_date']
    history = [row for row in rows if row['date'] < before]
    expected_ids = [row['match_id'] for row in history]
    assert parameter['reference_match_ids'] == expected_ids
    assert parameter['n_matches'] == parameter['n_training_rows'] == len(history)
    assert parameter['training_min_date'] == history[0]['date']
    assert parameter['training_max_date'] == history[-1]['date'] < before
    assert all(row['fit_before_date'] <= row['forecast_before_date'] < row['date'] < before
               for row in history)
    assert all(row['forecast_before_date'] < before for row in history)
    targets = [row['actual_outcome'] for row in history]
    assert parameter['class_counts'] == {name: targets.count(i) for i, name in enumerate(CLASS_NAMES)}
    assert parameter['classes'] == [0, 1, 2] and parameter['class_names'] == list(CLASS_NAMES)
    names = sorted({name for row in history for name in row[f'fresh_{family}_features'][0]})
    assert names == parameter['feature_names']
    matrix = np.asarray([[np.nan if feature[name] is None else feature[name] for name in names]
                         for feature in (row[f'fresh_{family}_features'][0] for row in history)], dtype=float)
    missing = np.isnan(matrix)
    empty = missing.all(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        medians = np.nanmedian(matrix, axis=0)
    medians[empty] = 0.
    imputation = parameter['imputation']
    error = assert_close(medians, imputation['statistics'], 'training medians', tolerance=1e-12)
    indicators = np.flatnonzero(missing.any(axis=0))
    assert imputation['indicator_feature_indices'] == indicators.tolist()
    assert imputation['all_missing_training_features'] == [names[i] for i in np.flatnonzero(empty)]
    assert imputation['strategy'] == 'unweighted training median'
    assert imputation['keep_empty_features'] is True and imputation['add_indicator'] is True
    expanded = names+['missingindicator_'+names[i] for i in indicators]
    assert parameter['expanded_feature_names'] == expanded
    design = np.column_stack((np.where(missing, medians, matrix), missing[:, indicators]))
    assert parameter['half_life_days'] == 180
    raw_weights = np.asarray([math.exp(-math.log(2)*(date.fromisoformat(before)-date.fromisoformat(row['date'])).days/180)
                              for row in history])
    factor = len(history)/float(raw_weights.sum())
    weights = raw_weights*factor
    weighting = parameter['weighting']
    error = max(error, assert_close(raw_weights.sum(), weighting['raw_sum'], 'raw time weights'),
                assert_close(factor, weighting['normalizing_factor'], 'time normalization'),
                assert_close(weights.sum(), weighting['sum'], 'normalized time weights'))
    assert weighting['n_fixture_weights'] == len(history)
    assert abs(weights.sum()-len(history)) < 1e-8
    scaling = parameter['scaling']
    if parameter['kind'] == 'logistic':
        mean = np.sum(design*weights[:, None], axis=0)/weights.sum()
        variance = np.sum(weights[:, None]*(design-mean)**2, axis=0)/weights.sum()
        # sklearn treats numerically constant columns as unit-scaled. This
        # floating-point bound avoids turning a constant into a tiny divisor.
        epsilon = np.finfo(np.float64).eps
        constant = variance <= weights.sum()*epsilon*variance+(weights.sum()*mean*epsilon)**2
        scale = np.sqrt(variance)
        scale[constant] = 1.
        error = max(error, assert_close(mean, scaling['mean'], 'weighted scale means'),
                    assert_close(variance, scaling['variance'], 'weighted scale variances'),
                    assert_close(scale, scaling['scale'], 'weighted scale factors'))
    else:
        assert scaling is None
    return {'references': len(history), 'statistics': len(medians), 'max_numeric_error': error}, medians, indicators


def expected_difference_order(include_blend):
    pairs = [('extra_trees_allcomp', 'regression_allcomp'),
             ('extra_trees_performance', 'regression_performance'),
             ('logistic_allcomp', 'regression_allcomp'),
             ('logistic_performance', 'regression_performance'),
             ('extra_trees_allcomp', 'extra_trees_performance'),
             ('logistic_allcomp', 'logistic_performance'),
             ('extra_trees_allcomp', 'logistic_allcomp')]
    if include_blend:
        pairs += [('extra_trees_allcomp', 'clean_blend'), ('logistic_allcomp', 'clean_blend')]
    return [f'{arm}_minus_{reference}:{statistic}' for arm, reference in pairs
            for statistic in ('log_loss', 'brier_sum', 'rps', 'misclassification')]


def score_section_audit(section, tracks):
    assert set(section['scores']) == set(tracks)
    ids = [row['match_id'] for row in tracks['regression_allcomp']]
    assert section['match_ids'] == ids and section['n'] == len(ids)
    losses, error, bins_checked = {}, 0., 0
    for arm, track in tracks.items():
        assert [row['match_id'] for row in track] == ids
        p = np.asarray([row['probabilities'] for row in track])
        y = np.asarray([row['actual_outcome'] for row in track])
        scores, losses[arm] = metrics(p, y)
        saved = section['scores'][arm]
        for statistic, value in scores.items():
            error = max(error, assert_close(value, saved[statistic], (arm, statistic), tolerance=1e-10))
        error = max(error, assert_close(p.mean(axis=0), saved['mean_probabilities'], (arm, 'mean p')),
                    assert_close(np.bincount(y, minlength=3)/len(y), saved['observed_frequencies'], (arm, 'obs freq')))
        expected_bins = calibration(p, y)
        assert expected_bins == saved['class_probability_bins'], (arm, 'calibration bins')
        bins_checked += len(expected_bins)
        for c, name in enumerate(CLASS_NAMES):
            actual = saved['by_actual_class'][name]
            selected = y == c
            assert actual['n'] == int(selected.sum())
            error = max(error, assert_close(losses[arm]['log_loss'][selected].mean(), actual['log_loss'], (arm, name)))
    assert section['difference_order'] == expected_difference_order('clean_blend' in tracks)
    matrices = []
    for column in section['difference_order']:
        contrast, statistic = column.split(':', 1)
        arm, reference = contrast.split('_minus_', 1)
        delta = losses[arm][statistic]-losses[reference][statistic]
        error = max(error, assert_close(delta.mean(), section['mean_loss_differences'][column], column, tolerance=1e-10))
        matrices.append(delta)
    matrix = np.column_stack(matrices)
    max_interval_error = 0.
    for length, saved in section['block_intervals'].items():
        assert int(length) in (2, 4, 8)
        expected = independent_blocks([row['date'] for row in tracks['regression_allcomp']], matrix, int(length))
        max_interval_error = max(max_interval_error,
            assert_close(expected, saved['interval_95'], ('block intervals', length), tolerance=1e-10))
        assert saved['seed'] == 20261023 and saved['repeats'] == saved['valid_repeats'] == 2000
        assert saved['zero_count_repeats'] == 0 and saved['n_matches'] == len(ids)
        error = max(error, assert_close(matrix.mean(axis=0), saved['mean'], ('block means', length)))
        days = [date.fromisoformat(row['date']) for row in tracks['regression_allcomp']]
        monday = min(days)-timedelta(days=min(days).weekday())
        span = (max(days)-monday).days//7+1
        counts = [sum((day-monday).days//7 == index for day in days) for index in range(span)]
        assert saved['n_calendar_weeks'] == span and saved['n_nonempty_weeks'] == sum(value > 0 for value in counts)
        assert saved['weekly_match_counts'] == counts
        assert saved['calendar_week_starts'] == [(monday+timedelta(weeks=i)).isoformat() for i in range(span)]
    return {'score_tracks': len(tracks), 'calibration_bins': bins_checked,
            'max_score_error': error, 'max_interval_error': max_interval_error}


def model_audit(directory, source, lead, rows):
    directory, source = Path(directory), Path(source)
    folder = directory/f'lead_{lead}'
    parameters = read(folder/'全部分类模型参数.json')
    summary = read(folder/'直接分类结果汇总.json')
    tracks = read(folder/'逐场胜平负概率.json')
    source_tracks = read(source/f'lead_{lead}'/'逐场完整预测.json')
    expected_windows = [(period, cutoff) for period in ('2024-25', '2025-26')
                        for cutoff in sorted({row['fit_before_date'] for row in rows if row['period'] == period})]
    assert len(parameters) == 4*len(expected_windows) == summary['new_classifier_fits']
    assert {(arm, cutoff) for arm in ARMS for _, cutoff in expected_windows} == {
        tuple(key.split(':', 1)) for key in parameters}
    progress = read(folder/'训练进度.json')
    assert progress['fits'] == len(parameters) and progress['windows'] == summary['windows']
    assert [(window['period'], window['cutoff']) for window in summary['windows']] == expected_windows
    for window in summary['windows']:
        assert window['training_n'] == sum(row['date'] < window['cutoff'] for row in rows)
        assert window['target_n'] == sum(row['period'] == window['period'] and row['fit_before_date'] == window['cutoff'] for row in rows)
    assert {path.stem for path in (folder/'models').glob('*.joblib')} == {
        key.replace(':', '_', 1) for key in parameters}
    row_map = {row['match_id']: row for row in rows}
    for new, old in (('regression_performance', 'fresh_performance'), ('regression_allcomp', 'fresh_allcomp'),
                     ('goal', 'goal'), ('xg_raw', 'xg_raw'), ('clean_blend', 'clean_blend')):
        expected = [{key: row[key] for key in (*CONTEXT, 'probabilities')} for row in source_tracks[old]]
        assert tracks[new] == expected, (new, 'reference outputs changed')
    assert all(row['period'] == '2025-26' for row in tracks['clean_blend'])
    references = statistics = replayed = 0
    max_preprocess_error = max_replay_error = max_manual_error = 0.
    for key, parameter in parameters.items():
        arm, cutoff = key.split(':', 1)
        kind, family = ARMS[arm]
        assert parameter['kind'] == kind and parameter['pipeline_feature_set'] == family
        assert parameter['horizon_days'] == lead and parameter['before_date'] == cutoff
        assert parameter['n_matches'] > 0 and parameter['new_parameter_grid'] is False
        config = parameter['configuration']
        assert config['class_weight'] is None and config['random_state'] == 20261025
        if kind == 'logistic':
            assert config['C'] == 1. and config['solver'] == 'lbfgs'
            assert config['max_iter'] == 1000 and config['tol'] == 1e-8
        else:
            assert config['criterion'] == 'log_loss' and config['n_estimators'] == 120
            assert config['max_depth'] == 6 and config['min_samples_leaf'] == 30
            assert config['max_features'] == .8 and config['bootstrap'] is False and config['n_jobs'] == 1
        checked, medians, indicators = preprocessing_audit(rows, parameter, family)
        references += checked['references']
        statistics += checked['statistics']
        max_preprocess_error = max(max_preprocess_error, checked['max_numeric_error'])
        model = joblib.load(folder/'models'/f'{arm}_{cutoff}.joblib')
        assert model.parameters() == {k: v for k, v in parameter.items() if k not in ('pipeline_feature_set', 'horizon_days')}
        assert model.estimator.classes_.tolist() == [0, 1, 2]
        predicted = [row for row in tracks[arm] if row['fit_before_date'] == cutoff]
        assert predicted
        features = [row_map[row['match_id']][f'fresh_{family}_features'][0] for row in predicted]
        probabilities = np.asarray([row['probabilities'] for row in predicted])
        replay = model.predict_many(features, [date.fromisoformat(row['forecast_before_date']) for row in predicted])
        max_replay_error = max(max_replay_error, assert_close(replay, probabilities, (key, 'saved estimator replay'), tolerance=1e-10))
        names = parameter['feature_names']
        matrix = np.asarray([[np.nan if feature[name] is None else feature[name] for name in names]
                             for feature in features], dtype=float)
        missing = np.isnan(matrix)
        design = np.column_stack((np.where(missing, medians, matrix), missing[:, indicators]))
        if kind == 'logistic':
            design = (design-np.asarray(parameter['scaling']['mean']))/np.asarray(parameter['scaling']['scale'])
            expanded = parameter['expanded_feature_names']
            coefficients = np.asarray([[parameter['coefficients'][name][feature] for feature in expanded] for name in CLASS_NAMES])
            intercept = np.asarray([parameter['intercepts'][name] for name in CLASS_NAMES])
            manual = softmax(design @ coefficients.T+intercept, axis=1)
        else:
            manual = np.mean([tree.predict_proba(design) for tree in model.estimator.estimators_], axis=0)
        max_manual_error = max(max_manual_error, assert_close(manual, probabilities, (key, 'independent serialized computation'), tolerance=1e-10))
        replayed += len(predicted)
        del model
    section_counts = []
    for period, section in (('2024-25', summary['validation']), ('2025-26', summary['test'])):
        expected = [row['match_id'] for row in rows if row['period'] == period]
        local = {arm: [row for row in values if row['period'] == period] for arm, values in tracks.items()
                 if arm != 'clean_blend' or period == '2025-26'}
        for arm, values in local.items():
            assert [row['match_id'] for row in values] == expected
            for predicted in values:
                original = row_map[predicted['match_id']]
                assert all(predicted[key] == original[key] for key in CONTEXT)
                assert predicted['fit_before_date'] <= predicted['forecast_before_date'] < predicted['date']
                if arm in ARMS:
                    assert predicted['feature_before_date'] == original['forecast_before_date']
                    assert predicted['output_scope'] == 'one_x_two_only'
                    assert 'rates' not in predicted and 'total_goals' not in predicted
                if arm == 'clean_blend':
                    assert original['blend_role'] == 'earlier_calibration_fixed_before_publication'
                    assert original['blend_weight_before_date'] <= original['forecast_before_date']
        assert set(section['block_intervals']) == ({'2', '4', '8'} if period == '2025-26' else set())
        section_counts.append(score_section_audit(section, local))
    assert set(summary['by_league']) == {'E0', 'D1', 'SP1', 'I1', 'F1'}
    for competition, section in summary['by_league'].items():
        local = {arm: [row for row in values if row['period'] == '2025-26' and row['competition'] == competition]
                 for arm, values in tracks.items()}
        assert section['block_intervals'] == {}
        section_counts.append(score_section_audit(section, local))
    assert 'selected' not in summary and summary['adopted'] is False
    result = {'status': 'passed', 'horizon_days': lead, 'saved_models': len(parameters),
        'training_reference_entries': references, 'training_imputation_statistics': statistics,
        'all_classifier_prediction_replays': replayed,
        'independent_logistic_softmax_and_tree_ensemble_replays': replayed,
        'score_tracks_recomputed': sum(item['score_tracks'] for item in section_counts),
        'calibration_bins_recomputed': sum(item['calibration_bins'] for item in section_counts),
        'max_preprocessing_error': max_preprocess_error, 'max_estimator_replay_error': max_replay_error,
        'max_independent_serialized_computation_error': max_manual_error,
        'max_score_error': max(item['max_score_error'] for item in section_counts),
        'max_block_interval_error': max(item['max_interval_error'] for item in section_counts),
        'reference_predictions_unchanged': True, 'one_fixture_one_label': True,
        'all_training_labels_strictly_earlier_than_parameter_date': True,
        'calibration_year_clean_blend_excluded': True, 'no_post_comparison_arm_or_lead_choice': True,
        'logistic_regularization_explanation': {
            'objective': '(1/n)*sum_i(w_i * negative_log_probability_of_actual_class) + ||coefficient_matrix||_F^2/(2*C*n); intercepts unpenalized',
            'current_C': 1., 'sum_normalized_weights': 'n fixtures',
            'equivalent_sum_objective': 'sum_i(w_i * NLL_i) + ||B||_F^2/(2*C)',
            'limitation': 'mean-one normalization removes an arbitrary common decay-weight scale; average-loss L2 coefficient is 1/(C*n), not constant across expanding cutoffs',
            'local_source': ['.venv/Lib/site-packages/sklearn/linear_model/_logistic.py:463-474,580',
                             '.venv/Lib/site-packages/sklearn/linear_model/_linear_loss.py:52-58']},
        'scope': 'independent date/preprocessing/serialized-model/probability arithmetic audit; no fresh fit or causal attribution; retrospective source availability assumptions remain'}
    write(directory/f'独立直接分类模型与评分核对_lead{lead}.json', result)
    return result


def completed_study_audit(directory, source):
    directory, source = Path(directory), Path(source)
    main = read(directory/'直接胜平负完整汇总.json')
    plan = read(directory/'分析前计划.json')
    audits = [read(directory/f'独立直接分类模型与评分核对_lead{lead}.json') for lead in (1, 3)]
    assert all(item['status'] == 'passed' for item in audits)
    assert set(main['horizons']) == {'1', '3'}
    assert main['new_classifier_fits'] == sum(item['saved_models'] for item in audits) == 176
    assert plan['horizons'] == [1, 3]
    assert plan['arms'] == {name: list(values) for name, values in ARMS.items()}
    assert plan['class_order'] == list(CLASS_NAMES)
    for field in ('lineups_used', 'current_results_reviewed', 'new_dependencies', 'neural_networks_added', 'adopted'):
        assert plan[field] is False
    assert main['adopted'] is False and Path(plan['source_study']).resolve() == source.resolve()
    assert read(directory/'覆盖与不可预测.json') == read(source/'覆盖与不可预测.json')
    for lead in (1, 3):
        assert main['horizons'][str(lead)] == read(directory/f'lead_{lead}'/'直接分类结果汇总.json')
        summary = main['horizons'][str(lead)]
        assert summary['validation']['difference_order'] == expected_difference_order(False)
        assert summary['test']['difference_order'] == expected_difference_order(True)
        assert set(summary['by_league']) == {'E0', 'D1', 'SP1', 'I1', 'F1'}
        assert all(section['difference_order'] == expected_difference_order(True)
                   for section in summary['by_league'].values())
    horizon_support = {}
    for section in ('validation', 'test'):
        first = set(main['horizons']['1'][section]['match_ids'])
        third = set(main['horizons']['3'][section]['match_ids'])
        horizon_support[section] = {'lead1_fixtures': len(first), 'lead3_fixtures': len(third),
                                    'intersection': len(first & third), 'union': len(first | third),
                                    'horizons_are_independent_replications': False}
    result = {'status': 'passed', 'saved_classifier_models': 176,
              'source_coverage_and_reference_outputs_preserved': True,
              'audits': audits, 'horizon_target_support': horizon_support,
              'training_reference_count_unit': 'entries repeated across cutoff and arm, not independent fixture count',
              'uncertainty_scope': 'paired calendar-block intervals conditional on saved predictions; not uncertainty from all historical modeling choices or source revisions',
              'current_results_reviewed': False, 'adopted': False}
    write(directory/'独立直接分类总核对.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--projection-only', action='store_true')
    parser.add_argument('--final-only', action='store_true')
    parser.add_argument('--lead', type=int, choices=(1, 3), action='append')
    arguments = parser.parse_args()
    if arguments.final_only:
        print(json.dumps(completed_study_audit(arguments.directory, arguments.source), ensure_ascii=False))
        raise SystemExit(0)
    projection, models = [], []
    for lead in arguments.lead or (1, 3):
        input_rows = load_rows(arguments.source, lead)
        projection.append(projection_audit(input_rows, lead))
        if arguments.directory and not arguments.projection_only:
            models.append(model_audit(arguments.directory, arguments.source, lead, input_rows))
    if arguments.directory:
        projection_path = arguments.directory/'独立单场双队投影核对.json'
        previous = read(projection_path) if projection_path.exists() else []
        merged = {item['horizon_days']: item for item in previous+projection}
        write(projection_path, [merged[key] for key in sorted(merged)])
        if not arguments.projection_only and not arguments.lead:
            completed_study_audit(arguments.directory, arguments.source)
    print(json.dumps({'projection': projection, 'models': models}, ensure_ascii=False))
