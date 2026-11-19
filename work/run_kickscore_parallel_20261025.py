"""Parallel disjoint league pools; same fixed models and legal histories."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import time

import run_kickscore_20261025 as study


def _worker(comp, history, targets, hyper, run_folder):
    # A persistent process can execute more than one competition task.
    study.OUT = Path(run_folder)/comp
    (study.OUT/'models').mkdir(parents=True, exist_ok=True)
    result = study.forward(history, targets, hyper)
    study.write(study.OUT/'finished.json', result)
    return comp, result


def run(history, targets, hyper, resume_checkpoint=None, *, merge_only=False):
    started = time.perf_counter()
    out = study.OUT
    names = ('dynamic','static','dynamic_mean_only')
    existing = {k: [] for k in names}
    previous = None
    if resume_checkpoint is not None:
        saved = study.read(resume_checkpoint)
        existing = saved['tracks']
        if set(existing) != set(names):
            raise ValueError('Resume requires exactly the same three study arms')
        reference = {(r['match_id'],r['horizon_days']): r for r in targets}
        idsets = []
        for arm in names:
            keys = [(r['match_id'],r['horizon_days']) for r in existing[arm]]
            if len(set(keys)) != len(keys):
                raise ValueError('Duplicate resume predictions')
            idsets.append(set(keys))
            for r in existing[arm]:
                orig = reference[(r['match_id'],r['horizon_days'])]
                fields = ('date','period','competition','home','away','actual_score','actual_outcome',
                          'forecast_before_date')
                if any(r[k] != orig[k] for k in fields):
                    raise ValueError('Resume identity disagrees with fixed target')
                if not r['training_max_date'] < r['fit_before_date'] <= r['forecast_before_date']:
                    raise ValueError('Resume forecast includes later history')
        if any(ids != idsets[0] for ids in idsets):
            raise ValueError('Resume must preserve every arm of each prediction')
        previous_path = Path(resume_checkpoint).parent/'forward_progress.json'
        if previous_path.exists():
            previous = study.read(previous_path)
    completed = {(r['match_id'],r['horizon_days']) for r in existing['dynamic']}
    pending = [r for r in targets if (r['match_id'],r['horizon_days']) not in completed]
    results = {}
    run_folder = out/'parallel' if merge_only else out/datetime.now(timezone.utc).strftime('parallel_%Y%m%dT%H%M%S%fZ')
    if merge_only:
        for path in run_folder.rglob('finished.json'):
            comp = path.parent.name
            if comp in results:
                raise ValueError('Duplicate completed competition artifacts')
            results[comp] = study.read(path)
    else:
        with ProcessPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(_worker, comp,
                [r for r in history if r['competition'] == comp],
                [r for r in pending if r['competition'] == comp], hyper, run_folder)
                for comp in ('E0','D1','SP1','I1','F1') if any(r['competition'] == comp for r in pending)]
            for fut in as_completed(futures):
                comp, result = fut.result()
                results[comp] = result
                print(f'Completed league {comp}: {result["n_predictions_per_arm"]} new rows per arm', flush=True)
    for comp in results:
        sources = [p.parent for p in run_folder.rglob('finished.json') if p.parent.name == comp]
        if len(sources) != 1:
            raise ValueError(f'Need exactly one completed output for {comp}')
        source = sources[0]
        tracks = study.read(source/'forecasts.json')['tracks']
        for name in names:
            existing[name].extend(tracks[name])
        import shutil
        for model_path in (source/'models').iterdir():
            shutil.copyfile(model_path, out/'models'/model_path.name)
    expected = {(r['match_id'],r['horizon_days']) for r in targets}
    for arm, rows in existing.items():
        actual = [(r['match_id'],r['horizon_days']) for r in rows]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError(f'Incomplete/duplicate parallel merged targets: {arm}')
        rows.sort(key=lambda r: (r['horizon_days'],r['date'],r['match_id']))
    study.write(out/'forecasts.json', {'tracks': existing, 'completed': True,
        'execution': 'parallel independent league pools with fixed shared hyperparameters',
        'resumed_rows_per_arm': len(completed)})
    windows = (previous or {}).get('windows', []) + [w for r in results.values() for w in r['windows']]
    return {'fit_calls': (previous or {}).get('fit_calls',0) + sum(r['fit_calls'] for r in results.values()),
        'windows': windows, 'seconds': None if merge_only else time.perf_counter()-started,
        'merge_only_seconds': time.perf_counter()-started if merge_only else None,
        'prior_sequential_seconds': (previous or {}).get('elapsed_seconds',0),
        'prior_interruption_note': 'Checkpoint at complete publication; up to ten fit calls after final progress may be unrecorded at process interruption.',
        'parallel_worker_seconds_sum': sum(r['seconds'] for r in results.values()),
        'failures': [f for r in results.values() for f in r['failures']],
        'n_publication_dates': len({r['forecast_before_date'] for r in targets}),
        'n_predictions_per_arm': len(targets), 'resumed_rows_per_arm': len(completed),
        'converged': all(w['converged'] for w in windows),
        'execution': 'three processes; disjoint competition pools; same fixed hyperparameters; complete target cohort',
        'run_folder': str(run_folder),
        'by_competition': {c: {k:v for k,v in result.items() if k != 'windows'} for c,result in results.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--merge-only', action='store_true')
    args = parser.parse_args()
    out = study.OUT
    hyper = {k: study.read(out/f'hyperparameter_{k}.json') for k in ('static','dynamic')}
    history = study.read(out/'data/canonical.json')['rows']
    targets = sum((study.read(out/'data'/f'targets_lead{l}.json')['rows'] for l in (1,3)), [])
    summary = study.read(out/'training_summary.json')
    summary['phase'] = 'merging' if args.merge_only else 'running_parallel'
    summary['merge_started_at' if args.merge_only else 'parallel_started_at'] = datetime.now(timezone.utc).isoformat()
    study.write(out/'training_summary.json', summary)
    summary['forward'] = run(history, targets, hyper, args.resume_checkpoint, merge_only=args.merge_only)
    if summary['forward']['seconds'] is None:
        summary['forward']['seconds'] = (datetime.now(timezone.utc)-datetime.fromisoformat(summary['parallel_started_at'])).total_seconds()
        summary['forward']['wall_seconds_note'] = 'parallel execution through final merge; includes process startup and merge repair, not sum of worker CPU seconds'
    summary.update({'phase': 'completed', 'finished_at': datetime.now(timezone.utc).isoformat()})
    summary['partition_check'] = study.read(out/'partition_check.json')
    study.write(out/'training_summary.json', summary)
    print(f'Complete: {out}', flush=True)


if __name__ == '__main__':
    main()
