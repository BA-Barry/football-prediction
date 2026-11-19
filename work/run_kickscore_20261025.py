"""Directly reuse KickScore on trusted chronological football histories.

No upstream-paper reproduction, kernel grid, or evaluation-season tuning.
Run with the G: environment variables in run_kickscore_20261025.ps1.
"""
import argparse
from datetime import date, datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
from scipy.optimize import minimize

from football_model.kickscore_model import KickScoreOutcomeModel

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OUT = ROOT/'outputs/动态胜平负接入_20261025'
INITIAL = {'cvar': .5, 'dvar': .25, 'lscale_days': 180., 'margin': .4, 'havar': 1.}
BOUNDS = {'cvar': (.001, 4.), 'dvar': (.001, 4.), 'lscale_days': (7., 3000.), 'margin': (.03, 1.5)}
TRAIN_END = date(2024, 7, 1)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def utc():
    return datetime.now(timezone.utc).isoformat()


def compact(model):
    return {k: v for k,v in model.parameters().items() if k != 'reference_match_ids'}


def observe_fit(kind, params, history, cutoff):
    started = time.perf_counter()
    model = KickScoreOutcomeModel(kind, params).fit(history, cutoff)
    return model, compact(model) | {'seconds': time.perf_counter()-started}


def benchmark(history, targets):
    """Real earlier-data fit and one real beginning-of-evaluation prefix."""
    windows = []
    for cutoff in (date(2023, 1, 1), date(2024, 8, 15)):
        for kind in ('static', 'dynamic'):
            model, meta = observe_fit(kind, INITIAL, history, cutoff)
            if not model.converged:
                raise RuntimeError(f'Initial real fit failed: {kind} {cutoff}')
            held = [r for r in targets if r['forecast_before_date'] == cutoff.isoformat()]
            predictions = model.predict_many(held, cutoff)
            windows.append(meta | {'prediction_n': len(predictions), 'predictions': predictions})
            write(OUT/'benchmark.json', {'windows': windows, 'fit_calls': len(windows),
                'timing_note': 'Python/import/JIT startup before function timer excluded; first fit may still compile kernels'})
            print(json.dumps({'phase': 'benchmark', 'kind': kind, 'before_date': str(cutoff),
                'n': meta['n_matches'], 'converged': model.converged, 'seconds': meta['seconds']}, ensure_ascii=False), flush=True)
    fs = sorted({r['forecast_before_date'] for r in targets})
    prefixes = {sum(r['date'] < f for r in history) for f in fs}
    return {'windows': windows, 'fit_calls': len(windows), 'publication_dates': len(fs),
        'distinct_history_prefixes': len(prefixes), 'conservative_full_refit_seconds':
            len(prefixes)*sum(w['seconds'] for w in windows[-2:]),
        'timing_note': 'first-run JIT can dominate; warm-start measurement follows if supported'}


def tune(kind, history):
    keys = ('cvar', 'dvar', 'lscale_days', 'margin') if kind == 'dynamic' else ('cvar', 'margin')
    calls, started = [], time.perf_counter()
    best = None

    def objective(x):
        nonlocal best
        params = dict(INITIAL) | dict(zip(keys, np.exp(x).tolist()))
        begin = time.perf_counter()
        record = {'evaluation': len(calls)+1, 'parameters': params, 'before_date': str(TRAIN_END)}
        try:
            model = KickScoreOutcomeModel(kind, params).fit(history, TRAIN_END)
            meta = compact(model)
            record.update(meta)
            if not model.converged or meta['log_likelihood'] is None:
                raise ArithmeticError('upstream EP convergence or approximate likelihood failed')
            loss = -meta['log_likelihood']
            record['objective'] = loss
            if best is None or loss < best['objective']:
                best = {'objective': loss, 'parameters': params, 'evaluation': record['evaluation']}
        except (ArithmeticError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            # Failed evaluations are explicit and consume the same finite budget.
            loss = 1e100
            record.update({'error': f'{type(error).__name__}: {error}', 'objective': None})
        record['seconds'] = time.perf_counter()-begin
        calls.append(record)
        write(OUT/f'hyperparameter_{kind}_progress.json', {'kind': kind, 'calls': calls, 'best': best})
        print(json.dumps({'phase': 'earlier_hyperparameter_fit', 'kind': kind, 'evaluation': len(calls),
            'objective': record.get('objective'), 'converged': record.get('converged'),
            'error': record.get('error'), 'seconds': record['seconds']}, ensure_ascii=False), flush=True)
        return loss

    result = minimize(objective, np.log([INITIAL[k] for k in keys]), method='Nelder-Mead',
        bounds=[tuple(np.log(BOUNDS[k])) for k in keys],
        options={'maxfev': 40, 'maxiter': 40, 'xatol': .01, 'fatol': .01})
    if best is None:
        raise RuntimeError(f'No valid earlier-data {kind} fit; see progress file')
    answer = {'kind': kind, 'parameters': best['parameters'], 'objective': best['objective'],
        'selected_evaluation': best['evaluation'], 'objective_evaluations': len(calls), 'underlying_fit_calls': len(calls),
        'optimizer_converged': bool(result.success), 'converged': bool(result.success), 'message': str(result.message),
        'seconds': time.perf_counter()-started, 'fit_before_date': str(TRAIN_END),
        'optimizer': 'one bounded log-parameter Nelder-Mead start; max40 full objectives; no grid/restart',
        'parameters_optimized': list(keys), 'bounds': {k: BOUNDS[k] for k in keys},
        'initial': INITIAL, 'evaluations': calls, 'evaluation_season_scores_used': False}
    write(OUT/f'hyperparameter_{kind}.json', answer)
    return answer


def forward(history, targets, hyper):
    begin = time.perf_counter()
    tracks = {k: [] for k in ('dynamic', 'static', 'dynamic_mean_only')}
    windows, failures, call_count = [], [], 0
    all_dates = sorted({r['forecast_before_date'] for r in targets})
    models, counts = {}, {}
    histories = {c: [r for r in history if r['competition'] == c]
                 for c in ('E0','D1','SP1','I1','F1')}
    for i,f in enumerate(all_dates):
        cutoff = date.fromisoformat(f)
        held = [r for r in targets if r['forecast_before_date'] == f]
        # Fixed shared hyperparameters do not couple the disjoint league items.
        # Refit only the relevant competition, reusing identical legal prefixes.
        jobs = [(c, kind) for c in sorted({r['competition'] for r in held})
                for kind in ('static', 'dynamic')]
        for comp, kind in jobs:
            local_held = [r for r in held if r['competition'] == comp]
            fixtures = {r['match_id']: r for r in local_held}
            local_history = histories[comp]
            n = sum(r['date'] < f for r in local_history)
            key = (kind, comp)
            start = time.perf_counter()
            if counts.get(key) != n:
                if key in models and hasattr(models[key], 'advance'):
                    models[key].advance(local_history, cutoff)
                    mode = 'append_earlier_history_and_refit_EP'
                else:
                    models[key] = KickScoreOutcomeModel(kind, hyper[kind]['parameters']).fit(local_history, cutoff)
                    mode = 'fresh_prefix_EP'
                counts[key] = n
                call_count += 1
            else:
                mode = 'identical_earlier_history_reused'
            model = models[key]
            meta = compact(model)
            if mode == 'append_earlier_history_and_refit_EP':
                mode = meta.get('fit_mode', 'full_prefix_refit_via_advance')
            window = {'before_date': f, 'kind': kind, 'competition': comp, 'n_matches': n,
                'training_max_date': meta['training_max_date'], 'model_before_date': meta['before_date'],
                'converged': model.converged, 'seconds': time.perf_counter()-start, 'mode': mode,
                'n_targets': len(local_held), 'log_likelihood': meta['log_likelihood']}
            windows.append(window)
            if not model.converged:
                failures.append(window | {'stage': 'fit', 'match_ids': [r['match_id'] for r in local_held]})
                write(OUT/'forward_progress.json', {'windows': windows, 'fit_calls': call_count, 'failures': failures})
                raise RuntimeError(f'Forward EP not converged at {f} {kind}; no targets silently omitted')
            output = model.predict_many(list(fixtures.values()), cutoff)
            lookup = {r['match_id']: r for r in output}
            for r in local_held:
                p = lookup[r['match_id']]
                common = {k: r[k] for k in ('match_id', 'date', 'period', 'competition', 'home', 'away',
                    'actual_score', 'actual_outcome', 'forecast_before_date', 'horizon_days')}
                common |= {'fit_before_date': meta['before_date'], 'parameter_before_date': str(TRAIN_END),
                    'training_max_date': meta['training_max_date'], 'n_training_matches': n,
                    'source_model': kind, 'unknown_teams': p['unknown_teams'],
                    'output_scope': '1X2 only'}
                tracks[kind].append(common | {k: p[k] for k in ('probabilities', 'mean_difference',
                    'variance_difference', 'raw_upstream_probabilities', 'raw_upstream_probability_error')})
                if kind == 'dynamic':
                    tracks['dynamic_mean_only'].append(common | {'probabilities': p['mean_only_probabilities'],
                        'mean_difference': p['mean_difference'], 'variance_difference': 0.,
                        'diagnostic_only': True, 'refitted': False})
        progress = {'publication_index': i+1, 'publication_count': len(all_dates), 'before_date': f,
            'fit_calls': call_count, 'n_predictions_per_fitted_arm': len(tracks['dynamic']),
            'elapsed_seconds': time.perf_counter()-begin, 'windows': windows, 'failures': failures}
        write(OUT/'forward_progress.json', progress)
        if i % 10 == 0 or i+1 == len(all_dates):
            write(OUT/'forecasts.json', {'tracks': tracks, 'completed': i+1 == len(all_dates)})
            for (kind,comp), model in models.items():
                joblib.dump(model, OUT/'models'/f'{kind}_{comp}_latest.joblib', compress=3)
            print(json.dumps({k: v for k,v in progress.items() if k not in ('windows','failures')}, ensure_ascii=False), flush=True)
    for values in tracks.values():
        values.sort(key=lambda r: (r['horizon_days'], r['date'], r['match_id']))
    write(OUT/'forecasts.json', {'tracks': tracks, 'completed': True})
    for (kind,comp), model in models.items():
        joblib.dump(model, OUT/'models'/f'{kind}_{comp}_latest.joblib', compress=3)
        write(OUT/'models'/f'{kind}_{comp}_latest_parameters.json', model.parameters())
    return {'fit_calls': call_count, 'windows': windows, 'seconds': time.perf_counter()-begin,
        'failures': failures, 'n_publication_dates': len(all_dates),
        'n_predictions_per_arm': len(tracks['dynamic']), 'converged': all(w['converged'] for w in windows),
        'execution': 'independent competition pools; identical shared earlier-fit hyperparameters'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('benchmark','tune','forward','all'), default='all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'models').mkdir(exist_ok=True)
    history = read(OUT/'data/canonical.json')['rows']
    targets = sum((read(OUT/'data'/f'targets_lead{l}.json')['rows'] for l in (1,3)), [])
    if any(r['date'] >= '2026-07-01' for r in history):
        raise ValueError('Current season is outside this authorized historical study')
    plan = {'created_at': utc(), 'user_approved': 'direct reuse upstream implementation and train on our earlier histories',
        'source': 'https://github.com/lucasmaystre/kickscore', 'dependencies': {k: version(k) for k in ('kickscore','numba','llvmlite','numpy','scipy')},
        'initial': INITIAL, 'bounds': BOUNDS, 'maximum_objective_evaluations_per_kind': 40,
        'data_period': '2022/23 through2025/26', 'hyperparameter_training_before': str(TRAIN_END),
        'forward': 'all existing targets; legal publication F=D-lead calendar days; history date strictly<F',
        'selection': 'no evaluation-season selection; fixed structure; retain all arms',
        'scope': '1X2 only', 'lineups_used': False, 'current_results_reviewed': False, 'adopted': False}
    if not (OUT/'implementation_plan.json').exists():
        write(OUT/'implementation_plan.json', plan)
    import kickscore
    license_src = Path(kickscore.__file__).parent.parent/'kickscore-0.2.0.dist-info/licenses/LICENSE'
    if license_src.exists():
        shutil.copyfile(license_src, OUT/'KickScore_MIT_LICENSE.txt')
    if args.stage in ('benchmark','all'):
        write(OUT/'benchmark.json', benchmark(history, targets))
    if args.stage in ('tune','all'):
        for kind in ('static','dynamic'):
            tune(kind, history)
    if args.stage in ('forward','all'):
        hyper = {kind: read(OUT/f'hyperparameter_{kind}.json') for kind in ('static','dynamic')}
        summary = {'phase': 'running', 'started_at': utc(), 'hyperparameter_search': hyper,
            'benchmark': read(OUT/'benchmark.json'), 'dependencies': plan['dependencies'],
            'lineups_used': False, 'current_results_reviewed': False, 'adopted': False}
        old_progress = OUT/'initial_pooled_runtime/forward_progress.json'
        if old_progress.exists():
            old = read(old_progress)
            summary['initial_pooled_runtime'] = {'completed_publications': old['publication_index'],
                'recorded_completed_fit_calls': old['fit_calls'], 'seconds': old['elapsed_seconds'],
                'note': 'terminated for runtime; up to two additional fit calls at interruption not fully recorded; retained separately, not evaluated',
                'used_for_selection': False}
        write(OUT/'training_summary.json', summary)
        from run_kickscore_parallel_20261025 import run as parallel_forward
        summary['forward'] = parallel_forward(history, targets, hyper)
        if (OUT/'partition_check.json').exists():
            summary['partition_check'] = read(OUT/'partition_check.json')
        summary.update({'phase': 'completed', 'finished_at': utc()})
        write(OUT/'training_summary.json', summary)
    print(json.dumps({'finished_stage': args.stage, 'output': str(OUT)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
