"""Fixed-method source-NPxG comparison on two saved historical study pairs."""
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from football_model.native_experiment import load_seasons, season_directories
from football_model.npxg_history import join_understat_npxg, DateTeamNpxgModel
from football_model.team_model import team_features
from football_model.goal_rate_scale import fit_goal_scale, apply_goal_scale
from football_model.forecast_diagnostics import moving_block_means
from football_model.probability import score_grid, summarize
from run_shot_increment_20261025 import score_details, write

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
EARLY = ROOT/'outputs/进球xG更早年份复现_20261024T180238377843Z/更早年份复现结果.json'
RECENT = ROOT/'outputs/进球xG共同训练记录_20261024T174247963857Z/共同训练记录融合结果.json'
EARLY_AUDIT = ROOT/'outputs/更早年份xG源核对_20261024T180024700032Z/逐季源差异.json'
RECENT_AUDIT = ROOT/'outputs/跨联赛xG源核对_20261024T172403681186Z/逐季源差异.json'
COVERAGE = ROOT/'outputs/早期预测研究_20261025/现有历史数据覆盖.json'
SLUGS = {'EPL': 'E0', 'Bundesliga': 'D1', 'La_Liga': 'SP1', 'Serie_A': 'I1', 'Ligue_1': 'F1'}
COMPS = ('E0', 'D1', 'SP1', 'I1', 'F1')
CONTRASTS = (('npxg_scaled', 'xg_scaled'), ('npxg_scaled', 'goal'),
             ('npxg_scaled', 'old_blend'), ('xg_scaled', 'xg_raw'))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_source():
    early_audit, recent_audit = read(EARLY_AUDIT), read(RECENT_AUDIT)
    directories = {**season_directories(ROOT), **{k: Path(v) for k, v in early_audit['fd_directories'].items()}}
    records, fd_reports = load_seasons(directories)
    independent = {r.match_id: r for r in records}
    parsed, reports = {}, {}
    for source in read(COVERAGE)['by_season']:
        comp, year = SLUGS[source['league']], source['season_start']
        season = f'{year}/{str(year+1)[-2:]}'
        key = f'{comp}:{season}'
        discrepancies = []
        if key in early_audit['seasons']:
            discrepancies.extend(early_audit['seasons'][key]['differences'])
        if key in recent_audit:
            discrepancies.extend(recent_audit[key]['differences'])
        # Reuse recorded source differences; never choose exclusions by score.
        bad_raw = {str(r['understat_id']) for r in discrepancies}
        bad_fd = {r['football_data_match_id'] for r in discrepancies}
        if None in bad_fd:
            raise ValueError('Unresolved independent source identity')
        payload = read(source['source'])
        raw_n = len(payload['dates'])
        payload['dates'] = [r for r in payload['dates'] if str(r['id']) not in bad_raw]
        local_fd = [r for r in records if r.competition == comp and r.season == season and r.match_id not in bad_fd]
        local, report = join_understat_npxg(payload, local_fd, season, comp)
        if report['missing_independent_match_ids'] or report['missing_special_match_ids']:
            raise ValueError(f'Unexplained source coverage: {key}')
        for record in local:
            if record.match_id in parsed:
                raise ValueError('Repeated source fixture')
            parsed[record.match_id] = record
        reports[key] = {**report, 'source_path': source['source'], 'raw_n': raw_n,
                        'excluded_source_ids': sorted(bad_raw), 'excluded_independent_ids': sorted(bad_fd)}
    return parsed, independent, reports, fd_reports, {k: str(v) for k, v in directories.items()}


def replay(parameters, row):
    return tuple(math.exp(parameters['intercept']+math.fsum(parameters['coefficients'].get(k, 0)*v
        for k, v in vector.items())) for vector in team_features(row['home'], row['away'], row['competition']))


def pairs():
    early, recent = read(EARLY), read(RECENT)
    cross, epl = (read(path) for path in recent['source_results'][:2])
    recent_xg_parameters = dict(cross['new_xg_parameters'])
    recent_xg_parameters.update({f'E0:{k}': v for k, v in epl['xg_parameters'].items()})
    return {
        'earlier': {'source': str(EARLY), 'fit_period': '2022-23', 'test_period': '2023-24',
                    'scale_cutoff': date(2023, 8, 1), 'expected_calibration_n': 1790,
                    'goal_parameters': early['goal_parameters'], 'xg_parameters': early['xg_parameters'],
                    'goal': early['predictions']['goal'], 'xg': early['predictions']['xg'],
                    'old_blend': early['predictions']['blend']},
        'recent': {'source': str(RECENT), 'fit_period': '2024-25', 'test_period': '2025-26',
                   'scale_cutoff': date(2025, 8, 1), 'expected_calibration_n': 1734,
                   'goal_parameters': {k: v['parameters'] for k, v in recent['common_goal_parameters'].items()},
                   'xg_parameters': recent_xg_parameters, 'goal': recent['common_goal_predictions'],
                   'xg': cross['new_predictions']+epl['new_predictions'],
                   'old_blend': recent['common_blend_predictions']},
    }


def prediction(context, rates):
    forecast = summarize(score_grid(*rates, rho=0))
    return {**context, 'rates': list(rates), 'forecast': forecast,
            'probabilities': list(forecast['one_x_two'].values())}


def compare(tracks, *, intervals):
    ids = [r['match_id'] for r in tracks['goal']]
    if any([r['match_id'] for r in rows] != ids for rows in tracks.values()):
        raise ValueError('Paired scoring requires identical fixture order')
    scores, losses = {}, {}
    for arm, rows in tracks.items():
        scores[arm], losses[arm] = score_details(rows)
    columns, values = [], []
    for arm, reference in CONTRASTS:
        for domain in losses[arm]:
            columns.append(f'{arm}_minus_{reference}:{domain}')
            values.append(losses[arm][domain]-losses[reference][domain])
    matrix = np.column_stack(values)
    return {'n': len(ids), 'scores': scores, 'difference_order': columns,
            'mean_loss_differences': dict(zip(columns, matrix.mean(axis=0).tolist())),
            'block_intervals': {str(b): moving_block_means([r['date'] for r in tracks['goal']], matrix, block_weeks=b)
                                for b in (2, 4, 8)} if intervals else {},
            'match_ids': ids}


def main():
    now = datetime.now(timezone.utc)
    target = ROOT/'outputs'/now.strftime('非点球xG对照_%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True, exist_ok=False)
    plan = {'created_at': now.isoformat(), 'purpose': 'NP signal versus total-xG signal with identical empirical full-goal conversion',
        'sources': [str(EARLY), str(RECENT), str(COVERAGE)],
        'pairs': [{'calibration': '2022-23', 'test': '2023-24', 'calibration_n': 1790, 'test_n': 1733},
                  {'calibration': '2024-25', 'test': '2025-26', 'calibration_n': 1734, 'test_n': 1733}],
        'preset': {'regularization': .01, 'half_life_days': 180, 'update_days': 28, 'rho': 0},
        'fit_history': 'exact original goal/xG reference IDs per cutoff, same calendar weights; no source-based expansion',
        'npxg_source': 'supplier team-match history; not raw shot sum or fixed penalty subtraction',
        'calibration': 'one global c=sum earlier full goals/sum earlier out-of-time signal means; each arm same rule; fixed next season',
        'calibration_interpretation': 'empirical mean conversion, not distinct penalty or own-goal process',
        'budget': {'new_npxg_team_fits': 214, 'reused_goal_fits': 214, 'reused_total_xg_fits': 214,
                   'new_closed_form_scales': 4, 'parameter_grids': 0, 'new_blend_weights': 0},
        'primary': 'each later season scaled NP minus scaled total-xG full-score NLL',
        'secondary': ['1X2 LL/Brier/Top1', 'total 0/1/2/3/4+ LL', 'away-minus-two three-way LL',
                      'context: goals and earlier-trained goal/total-xG blend'],
        'uncertainty': 'paired 2/4/8-week calendar blocks,2000 repeats; conditional descriptive intervals',
        'availability': 'strict earlier calendar dates assumed; exact historical publication/revisions unknown',
        'scope': 'already-viewed retrospective seasons, not blind or fully independent validation pairs',
        'unchanged': ['original prediction versions', 'market', 'LLM', 'current match results', 'paused lineup routes'],
        'current_results_reviewed': False, 'new_network_requests': 0, 'adoption': 'research only; no automatic replacement'}
    write(target/'分析前计划.json', plan)
    np_records, fd, joins, fd_reports, directories = load_source()
    write(target/'逐季数据联结.json', {'sources': joins, 'fd_directories': directories, 'fd_reports': fd_reports})
    study_results, fit_count = {}, 0
    for label, group in pairs().items():
        gp, xp = group['goal_parameters'], group['xg_parameters']
        if set(gp) != set(xp):
            raise ValueError('Saved goal and total-xG cutoffs differ')
        models, np_parameters = {}, {}
        for key, parameters in sorted(xp.items()):
            goal_parameters = gp[key]
            for field in ('reference_match_ids', 'n_matches', 'before_date', 'training_min_date',
                          'training_max_date', 'regularization', 'half_life_days'):
                if parameters[field] != goal_parameters[field]:
                    raise ValueError('Existing total-xG and goal histories differ')
            ids = parameters['reference_match_ids']
            history = [np_records[identifier] for identifier in ids]
            model = DateTeamNpxgModel(.01, 180).fit(history, date.fromisoformat(parameters['before_date']))
            fitted = model.parameters()
            for field in ('reference_match_ids', 'n_matches', 'before_date', 'training_min_date',
                          'training_max_date', 'regularization', 'half_life_days'):
                if fitted[field] != parameters[field]:
                    raise ValueError('NP weighted history differs from original')
            models[key], np_parameters[key] = model, fitted
            fit_count += 1
        write(target/f'{label}_非点球参数.json', np_parameters)
        goal_rows = sorted(group['goal'], key=lambda r: (r['date'], r['match_id']))
        xg_map = {r['match_id']: r for r in group['xg']}
        blend_map = {r['match_id']: r for r in group['old_blend']}
        if set(xg_map) != {r['match_id'] for r in goal_rows} or set(blend_map) != set(xg_map):
            raise ValueError('Saved target cohorts differ')
        observations = []
        for row in goal_rows:
            identifier = row['match_id']
            record, xg, blend = fd[identifier], xg_map[identifier], blend_map[identifier]
            key = f'{row["competition"]}:{row["fit_before_date"]}'
            for comparison_row in (xg, blend):
                if any(comparison_row[field] != row[field] for field in
                       ('date', 'home', 'away', 'competition', 'period', 'fit_before_date')):
                    raise ValueError('Saved comparator fixture metadata differs')
            h, a = row['actual_score']
            if row['actual_outcome'] != (0 if h > a else 2 if h < a else 1):
                raise ValueError('Outcome differs from independently verified score')
            if (row['date'] != record.match_date.isoformat() or row['home'] != record.home
                    or row['away'] != record.away or row['actual_score'] != list(record.full_time)
                    or xg['actual_score'] != row['actual_score'] or blend['actual_score'] != row['actual_score']
                    or not gp[key]['training_max_date'] < row['fit_before_date'] <= row['date']):
                raise ValueError('Target identity or temporal isolation conflict')
            for arm, parameters in ((row, gp[key]), (xg, xp[key])):
                if not np.allclose(replay(parameters, row), arm['rates'], rtol=0, atol=1e-10):
                    raise ValueError('Saved original rates fail coefficient replay')
            mean = models[key].predict(row['home'], row['away'], row['competition'], date.fromisoformat(row['date']))
            observations.append({'match_id': identifier, 'date': row['date'], 'period': row['period'],
                'competition': row['competition'], 'fit_before_date': row['fit_before_date'],
                'home': row['home'], 'away': row['away'], 'actual_score': row['actual_score'],
                'actual_outcome': row['actual_outcome'], 'training_max_date': gp[key]['training_max_date'],
                'goal_rates': row['rates'], 'xg_signal_rates': xg['rates'], 'npxg_signal_rates': list(mean),
                'old_blend_rates': blend['rates']})
        training = [r for r in observations if r['period'] == group['fit_period']]
        testing = [r for r in observations if r['period'] == group['test_period']]
        if len(training) != group['expected_calibration_n'] or len(testing) != 1733:
            raise ValueError('Original study cohort changed')
        scales = {}
        for arm in ('npxg', 'xg'):
            scales[arm] = fit_goal_scale([{**r, 'signal_rates': r[f'{arm}_signal_rates']} for r in training],
                group['scale_cutoff'], signal_label=f'provider_{arm}_mean')
        tracks = {arm: [] for arm in ('goal', 'xg_raw', 'xg_scaled', 'npxg_scaled', 'old_blend')}
        for row in testing:
            played = date.fromisoformat(row['date'])
            for arm, rates in (
                ('goal', row['goal_rates']), ('xg_raw', row['xg_signal_rates']),
                ('xg_scaled', apply_goal_scale(row['xg_signal_rates'], scales['xg'], played)),
                ('npxg_scaled', apply_goal_scale(row['npxg_signal_rates'], scales['npxg'], played)),
                ('old_blend', row['old_blend_rates'])):
                tracks[arm].append(prediction(row, rates))
        overall = compare(tracks, intervals=True)
        by_league = {comp: compare({arm: [r for r in rows if r['competition'] == comp]
                                   for arm, rows in tracks.items()}, intervals=False) for comp in COMPS}
        result = {'calibration_period': group['fit_period'], 'test_period': group['test_period'],
            'original_source': group['source'], 'new_npxg_fits': len(models), 'scales': scales,
            'scale_training_observations': training, 'test_predictions': tracks,
            'overall': overall, 'by_league': by_league, 'adopted': False}
        study_results[label] = result
        write(target/f'{label}_前推结果.json', result)
        print(json.dumps({'pair': label, 'fits': len(models), 'scales': {k: v['scale'] for k,v in scales.items()},
            'test_n': overall['n'], 'primary_difference': overall['mean_loss_differences']['npxg_scaled_minus_xg_scaled:exact_score']}, ensure_ascii=False), flush=True)
    if fit_count != plan['budget']['new_npxg_team_fits']:
        raise ValueError('Investigate fit coverage; do not silently expand the study')
    compact = {label: {k: v for k, v in study.items() if k not in ('test_predictions', 'scale_training_observations')}
               for label, study in study_results.items()}
    result = {'plan': plan, 'finished_at': datetime.now(timezone.utc).isoformat(),
        'source_matches': len(np_records), 'source_seasons': len(joins), 'new_npxg_fits': fit_count,
        'reused_goal_fits': fit_count, 'reused_xg_fits': fit_count, 'new_shared_scales': 4,
        'parameter_grids': 0, 'studies': compact, 'adopted': False}
    write(target/'非点球对照汇总.json', result)
    print(json.dumps({'directory': str(target), 'new_fits': fit_count,
        'held_tests': {k: {'n': v['overall']['n'], 'scores': v['overall']['scores'],
                          'differences': v['overall']['mean_loss_differences']}
                       for k, v in compact.items()}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
