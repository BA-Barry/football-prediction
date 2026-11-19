"""One real-prefix pooled-versus-league partition check; no retuning."""
from datetime import date, datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import time

from football_model.kickscore_model import KickScoreOutcomeModel


ROOT = REPOSITORY_ROOT
OUT = ROOT / 'outputs/动态胜平负接入_20261025'
CUTOFF = date(2024, 8, 15)
COMPS = ('E0', 'D1', 'SP1', 'I1', 'F1')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(result):
    (OUT / 'partition_check.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def fit_one(kind, params, history, targets):
    start = time.perf_counter()
    model = KickScoreOutcomeModel(kind, params).fit(history, CUTOFF)
    elapsed = time.perf_counter()-start
    meta = model.parameters()
    result = {'seconds': elapsed, 'converged': model.converged,
              'n_matches': meta['n_matches'], 'training_max_date': meta['training_max_date'],
              'n_teams': meta['n_teams'], 'log_likelihood': meta['log_likelihood'],
              'fit_controls': meta['fit_controls'], 'target_n': len(targets)}
    predictions = model.predict_many(targets, CUTOFF) if model.converged else []
    return result, {r['match_id']: r['probabilities'] for r in predictions}


def main():
    history = read(OUT / 'data/canonical.json')['rows']
    targets = sum((read(OUT / 'data' / f'targets_lead{lead}.json')['rows'] for lead in (1, 3)), [])
    targets = [r for r in targets if r['forecast_before_date'] == CUTOFF.isoformat()]
    if len({r['match_id'] for r in targets}) != len(targets):
        raise ValueError('This check needs unique fixtures at the shared publication date')
    prefix = [r for r in history if r['date'] < CUTOFF.isoformat()]
    result = {'created_at': datetime.now(timezone.utc).isoformat(), 'before_date': str(CUTOFF),
              'purpose': 'one real-prefix check that independent competition fits retain pooled probabilities',
              'new_hyperparameter_fits': 0, 'n_history': len(prefix), 'n_targets': len(targets),
              'hyperparameter_sources': {}, 'kinds': {}, 'underlying_fit_calls': 0,
              'timing_note': 'fits run sequentially in one fresh Python process, static pooled first; first-fit JIT may dominate and these times are not an unbiased speed comparison',
              'note': 'same already-saved fixed shared parameters in every competition; no evaluation-score selection'}
    save(result)
    for kind in ('static', 'dynamic'):
        hyper_path = OUT / f'hyperparameter_{kind}.json'
        hyper = read(hyper_path)
        params = hyper['parameters']
        result['hyperparameter_sources'][kind] = {
            'path': str(hyper_path), 'fit_before_date': hyper['fit_before_date'], 'parameters': params}
        pooled, pooled_p = fit_one(kind, params, history, targets)
        result['underlying_fit_calls'] += 1
        local, split_p = {}, {}
        result['kinds'][kind] = {'pooled': pooled, 'by_competition': local}
        save(result)
        print(json.dumps({'kind': kind, 'scope': 'pooled', **pooled}), flush=True)
        for comp in COMPS:
            rows = [r for r in history if r['competition'] == comp]
            held = [r for r in targets if r['competition'] == comp]
            local[comp], p = fit_one(kind, params, rows, held)
            split_p.update(p)
            result['underlying_fit_calls'] += 1
            save(result)
            print(json.dumps({'kind': kind, 'scope': comp, **local[comp]}), flush=True)
        details = []
        if pooled['converged'] and all(v['converged'] for v in local.values()):
            if set(pooled_p) != set(split_p) or set(pooled_p) != {r['match_id'] for r in targets}:
                raise ValueError('Pooled and split prediction fixture identities differ')
            for row in targets:
                identifier = row['match_id']
                difference = [a-b for a, b in zip(split_p[identifier], pooled_p[identifier])]
                details.append({k: row[k] for k in ('match_id', 'competition', 'date', 'horizon_days')}
                    | {'pooled_probabilities': pooled_p[identifier], 'split_probabilities': split_p[identifier],
                       'split_minus_pooled': difference, 'max_abs_difference': max(map(abs, difference))})
        result['kinds'][kind].update({
            'split_total_seconds': sum(v['seconds'] for v in local.values()),
            'all_converged': pooled['converged'] and all(v['converged'] for v in local.values()),
            'partition_training_n_equal': sum(v['n_matches'] for v in local.values()) == pooled['n_matches'],
            'max_abs_probability_difference': max((r['max_abs_difference'] for r in details), default=None),
            'predictions': details})
        save(result)
    result['completed'] = True
    result['difference_interpretation'] = (
        'Competitions have disjoint team/home items and independent GP priors conditional on the same fixed '
        'hyperparameters, so partitioning changes no statistical information. A local EP fit stops when '
        'its own site changes satisfy the existing tolerance; a pooled fit waits for the slowest competition. '
        'Their numerical stop points can therefore differ slightly, with the observed probability differences '
        'reported directly rather than forced equal or used to retune parameters.')
    save(result)
    print(json.dumps({'completed': True, 'underlying_fit_calls': result['underlying_fit_calls'],
                      'max_abs_differences': {k: v['max_abs_probability_difference']
                                              for k, v in result['kinds'].items()}}), flush=True)


if __name__ == '__main__':
    main()
