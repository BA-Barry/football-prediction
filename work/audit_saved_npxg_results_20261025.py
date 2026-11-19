"""Read saved NPxG study outputs; independently recompute chronology and scores."""
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import numpy as np
from scipy.stats import poisson, skellam

from football_model.native_experiment import load_seasons

ROOT = REPOSITORY_ROOT
OUT = ROOT/'outputs/非点球xG对照_20261025T054906945702Z'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    joins = read(OUT/'逐季数据联结.json')
    matches, _ = load_seasons({k: Path(v) for k, v in joins['fd_directories'].items()})
    source = {r.match_id: r for r in matches}
    old_early = read(ROOT/'outputs/进球xG更早年份复现_20261024T180238377843Z/更早年份复现结果.json')
    old_recent = read(ROOT/'outputs/进球xG共同训练记录_20261024T174247963857Z/共同训练记录融合结果.json')
    old_cross, old_epl = (read(p) for p in old_recent['source_results'][:2])
    cases = {
        'earlier': {'goal': old_early['predictions']['goal'], 'xg': old_early['predictions']['xg'],
                    'blend': old_early['predictions']['blend'], 'parameters': old_early['xg_parameters']},
        'recent': {'goal': old_recent['common_goal_predictions'],
                   'xg': old_cross['new_predictions']+old_epl['new_predictions'],
                   'blend': old_recent['common_blend_predictions'],
                   'parameters': {**old_cross['new_xg_parameters'], **{f'E0:{k}': v for k, v in old_epl['xg_parameters'].items()}}},
    }
    audit = {'measured_at': datetime.now(timezone.utc).isoformat(), 'new_fits': 0,
             'network_requests': 0, 'current_results_reviewed': False,
             'method': 'Saved outputs versus independent stored Football-Data fixtures; closed-form scales; infinite-support Poisson/Skellam scoring independent of model grid.',
             'studies': {}}
    for label, old in cases.items():
        study = read(OUT/f'{label}_前推结果.json')
        params = read(OUT/f'{label}_非点球参数.json')
        maps = {arm: {r['match_id']: r for r in old[arm]} for arm in ('goal', 'xg', 'blend')}
        identity_checks = 0
        for key, p in params.items():
            q = old['parameters'][key]
            assert p['reference_match_ids'] == q['reference_match_ids']
            assert p['n_matches'] == len(p['reference_match_ids'])
            assert len(set(p['reference_match_ids'])) == p['n_matches']
            history = [source[k] for k in p['reference_match_ids']]
            assert all(r.match_date < date.fromisoformat(p['before_date']) and r.competition == p['competition'] for r in history)
            assert min(r.match_date for r in history).isoformat() == p['training_min_date']
            assert max(r.match_date for r in history).isoformat() == p['training_max_date']
            assert p['regularization'] == .01 and p['half_life_days'] == 180
            identity_checks += len(history)
        all_rows = study['scale_training_observations'] + study['test_predictions']['goal']
        for row in all_rows:
            mid = row['match_id']
            s = source[mid]
            assert row['date'] == s.match_date.isoformat()
            assert (row['home'], row['away'], row['competition']) == (s.home, s.away, s.competition)
            assert row['actual_score'] == list(s.full_time)
            h, a = row['actual_score']
            assert row['actual_outcome'] == (0 if h > a else 2 if h < a else 1)
            key = f'{row["competition"]}:{row["fit_before_date"]}'
            assert params[key]['training_max_date'] < row['fit_before_date'] <= row['date']
            for arm in maps:
                original = maps[arm][mid]
                for field in ('date', 'home', 'away', 'competition', 'fit_before_date', 'actual_score', 'actual_outcome'):
                    assert row[field] == original[field], (label, mid, arm, field)
        scales = {}
        training = study['scale_training_observations']
        for arm, p in study['scales'].items():
            assert p['n'] == len(training)
            assert p['reference_match_ids'] == [r['match_id'] for r in training]
            assert all(r['date'] < p['before_date'] for r in training)
            assert all(r['date'] >= p['before_date'] for r in study['test_predictions']['goal'])
            total_goals = math.fsum(g for r in training for g in r['actual_score'])
            total_signal = math.fsum(g for r in training for g in r[f'{arm}_signal_rates'])
            c = total_goals/total_signal
            assert abs(c-p['scale']) < 1e-14
            scales[arm] = c
        scores = {}
        losses = {}
        maximum_probability_difference = 0.
        for arm, rows in study['test_predictions'].items():
            assert [r['match_id'] for r in rows] == study['overall']['match_ids']
            mu = np.asarray([r['rates'] for r in rows])
            goals = np.asarray([r['actual_score'] for r in rows])
            assert mu.shape == goals.shape == (1733, 2)
            assert np.isfinite(mu).all() and (mu > 0).all()
            expected_mu = np.asarray([r['npxg_signal_rates'] if arm == 'npxg_scaled' else r['xg_signal_rates'] if arm in ('xg_raw', 'xg_scaled') else r['goal_rates'] if arm == 'goal' else r['old_blend_rates'] for r in rows])
            if arm in ('npxg_scaled', 'xg_scaled'):
                expected_mu *= scales[arm.split('_')[0]]
            assert np.max(np.abs(mu-expected_mu)) < 1e-12
            one = np.column_stack([skellam.sf(0, mu[:, 0], mu[:, 1]), skellam.pmf(0, mu[:, 0], mu[:, 1]), skellam.cdf(-1, mu[:, 0], mu[:, 1])])
            total = np.column_stack([poisson.pmf(i, mu.sum(axis=1)) for i in range(4)] + [poisson.sf(3, mu.sum(axis=1))])
            hc = np.column_stack([skellam.sf(2, mu[:, 1], mu[:, 0]), skellam.pmf(2, mu[:, 1], mu[:, 0]), skellam.cdf(1, mu[:, 1], mu[:, 0])])
            saved_one = np.asarray([r['probabilities'] for r in rows])
            saved_total = np.asarray([[r['forecast']['total_goals'][k] for k in ('0', '1', '2', '3', '4+')] for r in rows])
            saved_hc = np.asarray([[r['forecast']['away_minus_two_jincai'][k] for k in ('win', 'draw', 'lose')] for r in rows])
            err = max(np.max(np.abs(one-saved_one)), np.max(np.abs(total-saved_total)), np.max(np.abs(hc-saved_hc)))
            assert err < 1e-10
            maximum_probability_difference = max(maximum_probability_difference, float(err))
            outcome = np.where(goals[:, 0] > goals[:, 1], 0, np.where(goals[:, 0] < goals[:, 1], 2, 1))
            hcdiff = goals[:, 1]-goals[:, 0]-2
            hcoutcome = np.where(hcdiff > 0, 0, np.where(hcdiff == 0, 1, 2))
            ix = np.arange(len(rows))
            arm_losses = {'exact_score': -poisson.logpmf(goals, mu).sum(axis=1),
                          'one_x_two': -np.log(one[ix, outcome]),
                          'total_goals_0_1_2_3_4plus': -np.log(total[ix, np.minimum(goals.sum(axis=1), 4)]),
                          'away_minus_two_three_way': -np.log(hc[ix, hcoutcome])}
            losses[arm] = arm_losses
            for domain, values in arm_losses.items():
                assert abs(float(values.mean())-study['overall']['scores'][arm][domain]['log_loss']) < 1e-10
            assert abs(float((one.argmax(axis=1) == outcome).mean())-study['overall']['scores'][arm]['one_x_two']['accuracy']) < 1e-14
            scores[arm] = {domain: float(values.mean()) for domain, values in arm_losses.items()}
        for name, saved in study['overall']['mean_loss_differences'].items():
            contrast, domain = name.split(':')
            arm, reference = contrast.split('_minus_')
            assert abs(float((losses[arm][domain]-losses[reference][domain]).mean())-saved) < 1e-10
        audit['studies'][label] = {'calibration_n': len(training), 'held_n': 1733, 'fit_count': len(params),
            'training_reference_rows_checked': identity_checks, 'metadata_rows_checked': len(all_rows),
            'scales_recomputed': scales, 'maximum_infinite_support_probability_difference': maximum_probability_difference,
            'analytic_scores': scores, 'errors': 0}
    (OUT/'独立保存结果核对.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == '__main__':
    main()
