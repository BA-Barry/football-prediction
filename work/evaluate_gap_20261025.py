"""Paired historical scoring for one saved GAP 1X2 forecast line.

No fitting, tuning, live results, or probability calibration occurs here. Each
reference comparison uses its own exact fixture intersection and discloses the
remaining fixtures; unlike averages from different samples, these deltas are
comparable within a pair. This does not create a new blind test season.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import numpy as np

from football_model.forecast_diagnostics import moving_block_means


ROOT = REPOSITORY_ROOT
REFERENCE = ROOT/'outputs/直接胜平负分类研究_20261025T091329913808Z/lead_3/逐场胜平负概率.json'
PERIODS = ('2024-25', '2025-26')
COMPS = ('E0', 'D1', 'SP1', 'I1', 'F1')
REFERENCES = ('goal', 'xg_raw', 'regression_allcomp', 'logistic_allcomp', 'clean_blend')
IDENTITY = ('date', 'period', 'competition', 'home', 'away', 'actual_score',
            'actual_outcome', 'forecast_before_date', 'horizon_days', 'fit_before_date')
METRICS = ('log_loss', 'brier_sum', 'rps', 'accuracy')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')


def _mapped(rows, label, identities):
    if not isinstance(rows, list):
        raise ValueError(f'{label} requires a row list')
    result = {}
    for row in rows:
        fid = row.get('match_id')
        if not isinstance(fid, str) or not fid or fid in result:
            raise ValueError(f'Missing or duplicate fixture identity in {label}: {fid}')
        if row['competition'] not in COMPS or row['period'] not in PERIODS:
            raise ValueError(f'Unexpected competition or period: {fid}')
        played = date.fromisoformat(row['date'])
        published = date.fromisoformat(row['forecast_before_date'])
        fitted = date.fromisoformat(row['fit_before_date'])
        if (row['horizon_days'] != 3 or isinstance(row['horizon_days'], bool)
                or (played-published).days != 3 or not fitted <= published < played):
            raise ValueError(f'GAP/reference chronology differs from lead 3: {fid}')
        if (not isinstance(row['home'], str) or not isinstance(row['away'], str)
                or not row['home'] or row['home'] == row['away']):
            raise ValueError(f'Invalid team identity: {fid}')
        score = row['actual_score']
        if (not isinstance(score, list) or len(score) != 2
                or any(type(g) is not int or g < 0 for g in score)):
            raise ValueError(f'Invalid actual score: {fid}')
        y = 0 if score[0] > score[1] else 2 if score[0] < score[1] else 1
        if type(row['actual_outcome']) is not int or row['actual_outcome'] != y:
            raise ValueError(f'Outcome disagrees with score: {fid}')
        p = np.asarray(row['probabilities'], dtype=float)
        if (p.shape != (3,) or not np.isfinite(p).all() or np.any(p < 0)
                or np.any(p > 1) or not np.isclose(p.sum(), 1, rtol=0, atol=1e-10)):
            raise ValueError(f'Invalid native home/draw/away probabilities: {fid}')
        identity = {k: row[k] for k in IDENTITY}
        if fid in identities and identities[fid] != identity:
            raise ValueError(f'Conflicting fixture identity, label, or cutoff in {label}: {fid}')
        identities[fid] = identity
        result[fid] = row
    return result


def _score(rows):
    """Score native probabilities without tiny-value clipping or smoothing."""
    if not rows:
        return {'n': 0, **{m: None for m in METRICS}}, {}
    p = np.asarray([r['probabilities'] for r in rows], dtype=float)
    y = np.asarray([r['actual_outcome'] for r in rows], dtype=int)
    truth = np.eye(3)[y]
    with np.errstate(divide='ignore'):
        loss = -np.log(p[np.arange(len(rows)), y])
    per = {'log_loss': loss,
           'brier_sum': np.square(p-truth).sum(axis=1),
           'rps': np.square(np.cumsum(p, axis=1)[:, :2]-
                            np.cumsum(truth, axis=1)[:, :2]).sum(axis=1)/2,
           'accuracy': (p.argmax(axis=1) == y).astype(float)}
    mean = {m: float(v.mean()) if np.isfinite(v).all() else None for m, v in per.items()}
    return {'n': len(rows), **mean,
            'infinite_log_loss_count': int(np.isinf(loss).sum()),
            'log_loss_is_infinite': bool(np.isinf(loss).any()),
            'top1_tie_count': int(((p == p.max(axis=1, keepdims=True)).sum(axis=1) > 1).sum()),
            'mean_probabilities': p.mean(axis=0).tolist(),
            'observed_frequencies': truth.mean(axis=0).tolist()}, per


def _pair(gap, reference, period, name, *, with_interval=False):
    left = {k: r for k, r in gap.items() if r['period'] == period}
    right = {k: r for k, r in reference.items() if r['period'] == period}
    ids = sorted(set(left)&set(right), key=lambda k: (left[k]['date'], k))
    gap_scores, gap_loss = _score([left[k] for k in ids])
    ref_scores, ref_loss = _score([right[k] for k in ids])
    differences = {}
    for metric in METRICS:
        a, b = gap_scores[metric], ref_scores[metric]
        differences[metric] = a-b if a is not None and b is not None else None
    result = {'period': period, 'reference': name, 'n_common': len(ids),
              'gap_available_n': len(left), 'reference_available_n': len(right),
              'common_match_ids': ids,
              'gap_only_match_ids': sorted(set(left)-set(right)),
              'reference_only_match_ids': sorted(set(right)-set(left)),
              'gap': gap_scores, 'reference_scores': ref_scores,
              'gap_minus_reference': differences,
              'delta_interpretation': 'negative favours GAP for losses; positive favours GAP for accuracy'}
    if with_interval:
        if ids and np.isfinite(gap_loss['log_loss']).all() and np.isfinite(ref_loss['log_loss']).all():
            delta = gap_loss['log_loss']-ref_loss['log_loss']
            result['conditional_four_week_log_loss_interval'] = moving_block_means(
                [left[k]['date'] for k in ids], delta[:, None], block_weeks=4,
                repeats=2000, seed=20261025)
        else:
            result['conditional_four_week_log_loss_interval'] = {
                'interval_95': None, 'unavailable_reason': 'empty pair or native infinite log loss'}
    return result


def evaluate(rows, saved):
    identities = {}
    gap = _mapped(rows, 'GAP', identities)
    if not gap:
        raise ValueError('No GAP predictions supplied')
    references = {name: _mapped(saved[name], name, identities) for name in REFERENCES}
    result = {'class_order': ['home', 'draw', 'away'], 'horizon_days': 3,
              'primary_metric': 'paired 1X2 natural-log loss',
              'native_probability_policy': 'no clipping, smoothing, or renormalization; infinite means saved as null plus explicit flags',
              'top1_tie_rule': 'numpy argmax, same as saved direct-classification historical study',
              'periods': {}, 'n_gap_predictions': len(gap)}
    for period in PERIODS:
        local = {k:r for k,r in gap.items() if r['period'] == period}
        own_score, _ = _score(sorted(local.values(), key=lambda r: (r['date'], r['match_id'])))
        refs = REFERENCES if period == '2025-26' else tuple(n for n in REFERENCES if n != 'clean_blend')
        comparisons = {}
        for name in refs:
            interval = ((period == '2025-26' and name == 'clean_blend') or
                        (period == '2024-25' and name == 'regression_allcomp'))
            pair = _pair(gap, references[name], period, name, with_interval=interval)
            pair['by_league'] = {comp: _pair(
                {k:r for k,r in gap.items() if r['competition'] == comp},
                {k:r for k,r in references[name].items() if r['competition'] == comp},
                period, name) for comp in COMPS}
            comparisons[name] = pair
        result['periods'][period] = {'gap_own_available': own_score,
            'gap_own_by_league': {comp: _score([r for r in local.values() if r['competition'] == comp])[0]
                                  for comp in COMPS},
            'paired_comparisons': comparisons}
    result['limits'] = [
        'Previously inspected historical seasons; this is local integration comparison, not a new blind test.',
        'Each reference uses its displayed intersection; means from different intersections do not establish a model ranking.',
        'Four-week intervals condition on saved forecasts and do not include parameter or model-selection uncertainty.',
        'No effect-size tuning, betting returns, current 2026/27 results, or target lineups are used.',
    ]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forecast', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    forecast, output = args.forecast.resolve(), args.output.resolve()
    if not output.is_relative_to(REPOSITORY_ROOT):
        raise ValueError('Evaluation output must remain below the project root')
    payload = _read(forecast)
    results = evaluate(payload['rows'], _read(REFERENCE))
    results.update(generated_at=datetime.now(timezone.utc).isoformat(),
                   forecast_source=str(forecast), reference_source=str(REFERENCE),
                   evaluation_fits=0, parameter_changes=0)
    output.mkdir(parents=True, exist_ok=True)
    _write(output/'GAP同场评价.json', results)
    concise = {'forecast_source': str(forecast), 'n_gap_predictions': results['n_gap_predictions'],
               'evaluation_path': str(output/'GAP同场评价.json'), 'periods': {}}
    for period, data in results['periods'].items():
        concise['periods'][period] = {
            'own_gap': data['gap_own_available'],
            'paired': {name:{k:pair[k] for k in ('n_common', 'gap_available_n',
                'reference_available_n', 'gap', 'reference_scores', 'gap_minus_reference')}
                | ({'conditional_four_week_log_loss_interval':pair['conditional_four_week_log_loss_interval']}
                   if 'conditional_four_week_log_loss_interval' in pair else {})
                for name,pair in data['paired_comparisons'].items()}}
    _write(output/'GAP评价简表.json', concise)
    print(json.dumps(concise, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
