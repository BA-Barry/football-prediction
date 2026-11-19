"""Replay the same goal/xG method two years earlier; never inspect live scores."""
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import numpy as np

from football_model.date_model import DateTeamGoalModel
from football_model.native_experiment import load_seasons
from football_model.xg_history import join_understat, DateTeamXgModel
from football_model.goal_xg_blend import fit_goal_xg_weight, predict_goal_xg_rates
from football_model.probability import score_grid, summarize
from football_model.forecast_diagnostics import moving_block_means
from run_shot_increment_20261025 import score_details, write

ROOT = REPOSITORY_ROOT
AUDIT = ROOT/'outputs/更早年份xG源核对_20261024T180024700032Z/逐季源差异.json'
COMPS = ('E0','D1','SP1','I1','F1')
CONTRASTS = (('blend','goal'),('blend','xg'),('xg','goal'))


def compare(tracks):
    ids = [r['match_id'] for r in tracks['goal']]
    if any([r['match_id'] for r in rows] != ids for rows in tracks.values()):
        raise ValueError('All scores require identical ordered target fixtures')
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
                                for b in (2,4,8)}, 'match_ids': ids}


def prediction(context, rates):
    forecast = summarize(score_grid(*rates, rho=0))
    return {**context, 'rates': list(rates), 'forecast': forecast,
            'probabilities': list(forecast['one_x_two'].values())}


def main():
    now = datetime.now(timezone.utc)
    target = ROOT/'outputs'/now.strftime('进球xG更早年份复现_%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True, exist_ok=False)
    plan = {
        'created_at': now.isoformat(), 'purpose': 'fixed-method retrospective replication two target years earlier',
        'warmup_seasons': ['2020/21','2021/22'], 'weight_estimation_season': '2022/23',
        'held_weight_season': '2023/24', 'training_origin': '2020/21 throughout both target years',
        'presets': {'regularization': .01, 'half_life_days': 180, 'update_days': 28, 'rho': 0},
        'source_audit': str(AUDIT),
        'source_policy': 'exact identity/date/full-score joins; all source conflicts and registered specials jointly excluded; no guessed dates or scores',
        'unknown_teams': 'joint target exclusion with reason, never zero effect; no new priors',
        'budget': {'new_goal_fits': 104, 'new_xg_fits': 104, 'new_shared_weight_fits': 1, 'parameter_grids': 0},
        'empty_calendar_windows': 'skip empty target windows, including World Cup break; derive leap-year dates dynamically',
        'weight_rule': 'one global log-rate coefficient in [0,1], uniform earlier-season out-of-time full-score goal NLL, fixed later',
        'primary': '2023/24 same-fixture blend minus goal complete-score NLL',
        'secondary': ['1X2 LL/Brier/Top1','total 0/1/2/3/4+ LL','away-minus-two three-way LL','pure xG comparison'],
        'uncertainty': 'paired noncircular calendar blocks,2/4/8 weeks,2000 repeats,conditional descriptive intervals',
        'limitations': ['method and presets chosen after later historical research, not new blind test',
                        'historical xG publication and revisions unknown, earlier-calendar-date convention only',
                        'intervals do not include coefficient estimation or complete model-selection uncertainty',
                        'Ligue 1 changes 20 to 18 teams; same per-fixture weighting but different league mix',
                        'pandemic warmup can affect home advantage, no new pandemic parameter'],
        'new_live_results_reviewed': False, 'adoption': 'research only, retain all leagues and arms, no later winner selection',
    }
    write(target/'分析前计划.json', plan)
    audit = json.loads(AUDIT.read_text(encoding='utf-8'))
    records, fd_reports = load_seasons({k: Path(v) for k,v in audit['fd_directories'].items()})
    source = {r.match_id: r for r in records}
    xg_records, joins, exclusions = [], {}, []
    for key, entry in audit['seasons'].items():
        comp, season = key.split(':')
        if entry['independent_pairs_missing_from_source']:
            raise ValueError('Resolve missing source identities before the common-history measurement')
        differences = entry['differences']
        if any(d['football_data_match_id'] is None for d in differences):
            raise ValueError('Unresolved historical team identity')
        bad_raw = {d['understat_id'] for d in differences}
        bad_fd = {d['football_data_match_id'] for d in differences}
        raw = json.loads(Path(entry['source_path']).read_text(encoding='utf-8'))['dates']
        local = [r for r in records if r.competition == comp and r.season == season and r.match_id not in bad_fd]
        parsed, report = join_understat([r for r in raw if str(r['id']) not in bad_raw], local, season, comp)
        if report['missing_independent_match_ids'] or report['missing_special_match_ids']:
            raise ValueError('Unexplained exact-join coverage difference')
        joins[key] = {'raw_n': len(raw), 'joint_exclusions': differences, 'join_report': report}
        xg_records.extend(parsed)
        exclusions.extend({'competition': comp, 'season': season, **d} for d in differences)
    write(target/'逐季联结与共同排除.json', joins)
    available = {r.match_id: r for r in xg_records}
    if len(available) != len(xg_records):
        raise ValueError('Duplicate xG match identity')
    goal_parameters, xg_parameters = {}, {}
    predictions = {'goal': [], 'xg': [], 'blend': []}
    skipped, empty_windows = [], []
    weight_parameters = None
    weight_rows = []
    for year in (2022,2023):
        period, season = f'{year}-{str(year+1)[-2:]}', f'{year}/{str(year+1)[-2:]}'
        for comp in COMPS:
            local = [r for r in records if r.competition == comp and r.season == season]
            xg_history = [r for r in xg_records if r.competition == comp]
            cutoff, end = date(year,8,1), date(year+1,7,1)
            while cutoff < end:
                stop = min(cutoff+timedelta(days=28), end)
                calendar_targets = sorted((r for r in local if cutoff <= r.match_date < stop),
                                          key=lambda r: (r.match_date, r.match_id))
                if not calendar_targets:
                    empty_windows.append({'competition': comp, 'period': period, 'cutoff': cutoff.isoformat()})
                    cutoff = stop
                    continue
                history = sorted((r for r in xg_history if r.match_date < cutoff),
                                 key=lambda r: (r.match_date, r.match_id))
                known = {t for r in history for t in (r.home, r.away)}
                targets = []
                for match in calendar_targets:
                    reason = (match.goal_fit_exclusion or 'source_date_or_score_conflict') if match.match_id not in available else (
                        'unseen_team_no_estimated_prior' if {match.home,match.away}-known else None)
                    if reason:
                        skipped.append({'match_id': match.match_id, 'competition': comp, 'period': period,
                                        'date': match.match_date.isoformat(), 'cutoff': cutoff.isoformat(),
                                        'reason': reason, 'unknown_teams': sorted({match.home,match.away}-known)})
                    else:
                        targets.append(match)
                if targets:
                    goal_history = [source[r.match_id] for r in history]
                    gm = DateTeamGoalModel(.01,180).fit(goal_history, cutoff)
                    xm = DateTeamXgModel(.01,180).fit(history, cutoff)
                    gp, xp = gm.parameters(), xm.parameters()
                    for field in ('reference_match_ids','n_matches','before_date','training_min_date',
                                  'training_max_date','regularization','half_life_days'):
                        if gp[field] != xp[field]:
                            raise ValueError('Goal and xG fits have different weighted histories')
                    fit_key = f'{comp}:{cutoff.isoformat()}'
                    goal_parameters[fit_key], xg_parameters[fit_key] = gp, xp
                    for match in targets:
                        h,a = match.full_time
                        context = {'match_id': match.match_id, 'competition': comp, 'period': period,
                            'date': match.match_date.isoformat(), 'home': match.home, 'away': match.away,
                            'fit_before_date': cutoff.isoformat(), 'training_matches': gp['n_matches'],
                            'training_max_date': gp['training_max_date'], 'actual_score': [h,a],
                            'actual_outcome': 0 if h>a else 2 if h<a else 1,
                            'source_url': match.source_url, 'xg_source_url': available[match.match_id].source_url}
                        gr = gm.predict(match.home, match.away, comp, match.match_date)
                        xr = xm.predict(match.home, match.away, comp, match.match_date)
                        predictions['goal'].append(prediction(context, gr))
                        predictions['xg'].append(prediction(context, xr))
                        if year == 2022:
                            weight_rows.append({'match_id': match.match_id, 'date': context['date'],
                                                'goal_rates': list(gr), 'xg_rates': list(xr), 'actual_score': [h,a]})
                        if weight_parameters is not None:
                            w = weight_parameters['weight']
                            predictions['blend'].append(prediction({**context,'goal_rates':list(gr),'xg_rates':list(xr),
                                'shared_xg_weight':w}, predict_goal_xg_rates(gr,xr,w)))
                cutoff = stop
            print(f'Completed {period} {comp}', flush=True)
        if year == 2022:
            weight_parameters = fit_goal_xg_weight(weight_rows, date(2023,8,1))
            for gr, xr in zip(predictions['goal'], predictions['xg'], strict=True):
                if gr['match_id'] != xr['match_id']:
                    raise ValueError('Goal/xG prediction identity difference')
                w = weight_parameters['weight']
                predictions['blend'].append(prediction({**gr,'goal_rates':gr['rates'],'xg_rates':xr['rates'],
                    'shared_xg_weight':w}, predict_goal_xg_rates(gr['rates'],xr['rates'],w)))
    if len(goal_parameters) != 104 or len(xg_parameters) != 104:
        raise ValueError('Actual nonempty training groups differ from premeasured 104, investigate without budget expansion')
    for arm in predictions:
        predictions[arm].sort(key=lambda r: (r['date'], r['match_id']))
    seasons, leagues = {}, {}
    for period in ('2022-23','2023-24'):
        tracks = {arm: [r for r in rows if r['period'] == period] for arm,rows in predictions.items()}
        seasons[period] = compare(tracks)
        seasons[period]['role'] = 'weight estimation apparent fit' if period == '2022-23' else 'held earlier weight retrospective replication'
        leagues[period] = {c: compare({arm:[r for r in rows if r['competition'] == c]
                                       for arm,rows in tracks.items()}) for c in COMPS}
    result = {'plan': plan, 'finished_at': datetime.now(timezone.utc).isoformat(),
              'fd_reports': fd_reports, 'join_reports': joins, 'source_exclusions': exclusions,
              'raw_source_matches': sum(v['raw_n'] for v in joins.values()),
              'usable_source_matches': len(xg_records), 'skipped_targets': skipped, 'empty_windows': empty_windows,
              'new_goal_fits': len(goal_parameters), 'new_xg_fits': len(xg_parameters), 'new_scalar_weight_fits': 1,
              'parameter_grids': 0, 'parameters': weight_parameters, 'weight_training_rows': weight_rows,
              'goal_parameters': goal_parameters, 'xg_parameters': xg_parameters,
              'predictions': predictions, 'seasons': seasons, 'by_league': leagues, 'adopted': False}
    write(target/'更早年份复现结果.json', result)
    late = seasons['2023-24']
    print(json.dumps({'directory': str(target), 'raw_n': result['raw_source_matches'],
        'usable_n': len(xg_records), 'new_goal_fits': len(goal_parameters), 'new_xg_fits': len(xg_parameters),
        'weight': weight_parameters['weight'], 'early_n': seasons['2022-23']['n'], 'late_n': late['n'],
        'skipped': len(skipped), 'late_scores': late['scores'],
        'main_deltas': {k:v for k,v in late['mean_loss_differences'].items() if k.startswith('blend_minus_goal:')},
        'by_league': {c:{'n':v['n'],'deltas':{k:x for k,x in v['mean_loss_differences'].items()
                    if k.startswith('blend_minus_goal:')}} for c,v in leagues['2023-24'].items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
