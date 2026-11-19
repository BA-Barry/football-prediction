"""Independent fixed-publication-date feature and saved-model audit.

No fitting, source downloads, current-score requests or production feature builders.
"""
import argparse
from bisect import bisect_left
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import json
import gzip
import math
import warnings
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import joblib
import numpy as np

from audit_clean_integrated_study_20261025 import independent_scores
from run_npxg_comparison_20261025 import load_source, COVERAGE
from scipy.special import gammaln
from scipy.stats import poisson, skellam


ROOT = REPOSITORY_ROOT
MEASUREMENTS = ('goals_for', 'goals_against', 'xg_for', 'xg_against',
    'npxg_for', 'npxg_against', 'deep_for', 'deep_against',
    'ppda_att', 'ppda_def', 'ppda_allowed_att', 'ppda_allowed_def')
SLUGS = {'E0': 'eng.1', 'D1': 'ger.1', 'SP1': 'esp.1', 'I1': 'ita.1', 'F1': 'fra.1'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def parameter_day(played, lead):
    publication = played-timedelta(days=lead)
    anchor = date(played.year if played.month >= 7 else played.year-1, 8, 1)
    return publication, anchor+timedelta(days=28*((publication-anchor).days//28))


def replay_base(parameter, row):
    coefficients = parameter['coefficients']
    home, away, comp = row['home'], row['away'], row['competition']
    result = []
    for own, other, is_home in ((home, away, True), (away, home, False)):
        log_mean = math.fsum((parameter['intercept'], coefficients.get(f'attack:{own}', 0.),
            -coefficients.get(f'defense:{other}', 0.), coefficients.get(f'competition:{comp}', 0.),
            coefficients.get(f'home:{comp}', 0.) if is_home else 0.))
        result.append(math.exp(log_mean))
    return result


def base_audit(directory):
    directory = Path(directory)
    payload = read(directory/'fixed_horizon_base_rows.json')
    parameters = read(directory/'固定提前日期基础参数.json')
    population = read(directory/'源范围与联结.json')['population_ids_by_group']
    original = read(Path(payload['source_directory'])/'clean_base_rows.json')
    old_parameters = read(Path(payload['source_directory'])/'干净基础专家参数.json')
    source, independent, _, _, _ = load_source()
    original_rows = {r['match_id']:r for r in original['rows']}
    references, max_replay_error = 0, 0.
    for group, identifiers in population.items():
        expected_universe = sorted({mid for pair in old_parameters[group].values() for mid in pair['goal']['reference_match_ids']})
        assert expected_universe == identifiers
    for key, pair in parameters.items():
        group, comp, cutoff = key.split(':',2)
        expected = sorted((source[mid] for mid in population[group]
                           if source[mid].competition == comp and source[mid].match_date.isoformat() < cutoff),
                          key=lambda r:(r.match_date,r.match_id))
        ids = [r.match_id for r in expected]
        for response in ('goal','xg'):
            parameter = pair[response]
            assert parameter['reference_match_ids'] == ids
            assert parameter['n_matches'] == len(ids)
            assert parameter['before_date'] == cutoff
            assert parameter['training_min_date'] == expected[0].match_date.isoformat()
            assert parameter['training_max_date'] == expected[-1].match_date.isoformat() < cutoff
            assert parameter['regularization'] == .01 and parameter['half_life_days'] == 180
            assert set(parameter['teams']) == {team for r in expected for team in (r.home,r.away)}
            assert all(not independent[mid].goal_fit_exclusion for mid in ids)
            if pair['action'] == 'reused_same_earlier_design':
                assert parameter == old_parameters[group][f'{comp}:{cutoff}'][response]
            references += len(ids)
    summaries = {}
    for horizon, group in payload['horizons'].items():
        rows, unavailable = group['rows'], group['unavailable']
        assert len(rows)+len(unavailable) == len(original_rows)
        assert {r['match_id'] for r in rows}.isdisjoint({r['match_id'] for r in unavailable})
        assert {r['match_id'] for r in rows+unavailable} == set(original_rows)
        for row in rows+unavailable:
            old = original_rows[row['match_id']]
            target = date.fromisoformat(row['date'])
            forecast, parameter_date = parameter_day(target,int(horizon))
            assert row['forecast_before_date'] == forecast.isoformat()
            assert row['fit_before_date'] == parameter_date.isoformat()
            assert parameter_date <= forecast < target
            assert row['base_parameter_key'] == f'{old["base_generation_group"]}:{row["competition"]}:{parameter_date.isoformat()}'
            for key in ('date','period','competition','home','away','actual_score','actual_outcome'):
                assert row[key] == old[key]
            assert row['actual_score'] == list(independent[row['match_id']].full_time)
            pair = parameters[row['base_parameter_key']]
            missing = {response: sorted({row['home'],row['away']}-set(pair[response]['teams']))
                       for response in ('goal','xg')}
            if 'reason' in row:
                assert row['reason'] == 'unseen_team_in_legal_earlier_base_parameters'
                assert row['unseen_teams'] == missing and any(missing.values())
                assert 'rates' not in row and 'xg_rates' not in row
            else:
                assert not any(missing.values())
                for response, field in (('goal','goal_rates'),('xg','xg_rates')):
                    rates = replay_base(pair[response],row)
                    error = max(abs(a-b) for a,b in zip(rates,row[field],strict=True))
                    max_replay_error = max(max_replay_error,error)
                    assert error < 1e-10
                assert row['rates'] == row['goal_rates']
                w = row['blend_weight']
                blend = [math.exp((1-w)*math.log(g)+w*math.log(x)) for g,x in zip(row['goal_rates'],row['xg_rates'],strict=True)]
                assert max(abs(a-b) for a,b in zip(blend,row['blend_rates'],strict=True)) < 1e-10
                if row['period'] in ('2023-24','2025-26'):
                    assert row['blend_role'] == 'earlier_calibration_fixed_before_publication'
                    assert row['blend_weight_before_date'] <= row['forecast_before_date']
                else:
                    assert row['blend_role'] == 'apparent_calibration_fit_do_not_use_for_selection'
        blends = {}
        for label, period in (('earlier','2022-23'),('recent','2024-25')):
            calibration = [r for r in rows if r['period'] == period]
            parameter = group['blend_parameters'][label]
            assert parameter['n'] == len(calibration)
            assert parameter['reference_match_ids'] == [r['match_id'] for r in calibration]
            goals = np.array([r['actual_score'] for r in calibration])
            lg = np.log(np.array([r['goal_rates'] for r in calibration]))
            lx = np.log(np.array([r['xg_rates'] for r in calibration]))
            w = parameter['weight']
            log_mu = (1-w)*lg+w*lx
            mu = np.exp(log_mu)
            value = float((mu-goals*log_mu+gammaln(goals+1)).sum()/len(calibration))
            gradient = float(((mu-goals)*(lx-lg)).sum()/len(calibration))
            assert abs(gradient) < 1e-7
            assert abs(value-parameter['objective']) < 1e-10
            blends[label] = {'weight':w,'n':len(calibration),'independent_objective':value,'gradient':gradient}
        summaries[horizon] = {'available':len(rows),'unavailable':len(unavailable),
            'by_period':dict(Counter(r['period'] for r in rows)), 'blend':blends}
    result = {'status':'passed','checked_base_parameter_responses':2*len(parameters),
        'training_reference_entries':references,'max_independent_coefficient_replay_error':max_replay_error,
        'horizons':summaries,'known_incomplete_matches_excluded':True,
        'unknown_teams_remain_unavailable':True,'future_blends_not_accepted_as_available':True}
    write(directory/'固定提前日期基础独立核对.json',result)
    return result


def close_dict(saved, expected, label, counts):
    assert set(saved) == set(expected), (label, 'columns', set(saved)^set(expected))
    for key, value in expected.items():
        current = saved[key]
        if value is None:
            assert current is None, (label, key, current, value)
        else:
            assert current is not None and math.isfinite(float(current)), (label, key)
            difference = abs(float(current)-float(value))
            counts['feature_values'] += 1
            counts['max_feature_error'] = max(counts['max_feature_error'], difference)
            assert difference <= 1e-10, (label, key, current, value)


def loss_vectors(track):
    means = np.asarray([r['rates'] for r in track])
    actual = np.asarray([r['actual_score'] for r in track])
    home, away = means[:,0], means[:,1]
    y = np.where(actual[:,0]>actual[:,1],0,np.where(actual[:,0]<actual[:,1],2,1))
    p = np.column_stack((skellam.sf(0,home,away),skellam.pmf(0,home,away),skellam.cdf(-1,home,away)))
    total = np.column_stack([poisson.pmf(k,home+away) for k in range(4)]+[poisson.sf(3,home+away)])
    hc = np.column_stack((skellam.cdf(-3,home,away),skellam.pmf(-2,home,away),skellam.sf(-2,home,away)))
    handicap = np.where(actual[:,1]-actual[:,0]>2,0,np.where(actual[:,1]-actual[:,0]==2,1,2))
    return {'exact_score': -poisson.logpmf(actual,means).sum(axis=1),
        'one_x_two': -np.log(p[np.arange(len(y)),y]),
        'total_goals_0_1_2_3_4plus': -np.log(total[np.arange(len(y)),np.minimum(actual.sum(axis=1),4)]),
        'away_minus_two_three_way': -np.log(hc[np.arange(len(y)),handicap])}


def independent_blocks(dates, matrix, weeks, *, repeats=2000, seed=20261023):
    days = [date.fromisoformat(value) for value in dates]
    first_monday = min(days)-timedelta(days=min(days).weekday())
    labels = np.asarray([(day-first_monday).days//7 for day in days])
    span = int(max(labels))+1
    counts = np.asarray([sum(labels == i) for i in range(span)])
    sums = np.vstack([matrix[labels == i].sum(axis=0) for i in range(span)])
    starts = np.random.default_rng(seed).integers(0,span-weeks+1,
        size=(repeats,(span+weeks-1)//weeks))
    boot = []
    for starts_row in starts:
        sample = np.asarray([i for start in starts_row for i in range(start,start+weeks)][:span])
        denominator = counts[sample].sum()
        if denominator:
            boot.append(sums[sample].sum(axis=0)/denominator)
    return np.quantile(np.asarray(boot),[.025,.975],axis=0).T


class RawSources:
    def __init__(self, schedule_directory):
        records, independent, _, _, _ = load_source()
        self.records, self.independent = records, independent
        fixtures, raw_history = {}, {}
        for entry in read(COVERAGE)['by_season']:
            payload = read(entry['source'])
            for fixture in payload['dates']:
                fixtures[str(fixture['id'])] = fixture
            for team_id, team in payload['teams'].items():
                for observation in team['history']:
                    raw_history[(str(team_id), observation['date'], observation['h_a'])] = observation
        self.league, self.league_dates, self.performance_cache = {}, {}, {}
        for record in records.values():
            fixture = fixtures[record.understat_id]
            for i, side in enumerate(('h', 'a')):
                source = raw_history[(str(fixture[side]['id']), fixture['datetime'], side)]
                row = {'match_id': record.match_id, 'date': record.match_date, 'is_home': i == 0,
                    'goals_for': float(record.goals[i]), 'goals_against': float(record.goals[1-i]),
                    'xg_for': record.xg[i], 'xg_against': record.xg[1-i],
                    'npxg_for': record.npxg[i], 'npxg_against': record.npxg[1-i],
                    'deep_for': float(source['deep']), 'deep_against': float(source['deep_allowed']),
                    'ppda_att': float(source['ppda']['att']), 'ppda_def': float(source['ppda']['def']),
                    'ppda_allowed_att': float(source['ppda_allowed']['att']),
                    'ppda_allowed_def': float(source['ppda_allowed']['def'])}
                self.league.setdefault((record.competition, (record.home, record.away)[i]), []).append(row)
        for key, rows in self.league.items():
            rows.sort(key=lambda r: (r['date'], r['match_id']))
            self.league_dates[key] = [r['date'] for r in rows]
        self.schedule, self.schedule_cache = {}, {}
        blocked = set()
        seasons = read(Path(schedule_directory)/'四季覆盖汇总.json')['seasons']
        for season in seasons:
            blocked.update(str(row['event_id']) for row in read(season['summary_file']).get('known_specials', []))
        official = Path(schedule_directory)/'官方部分比赛补充说明_20261025.json'
        if official.exists():
            blocked.update(str(r['event_id']) for r in read(official)['matches'])
        for season in seasons:
            year = int(season['season'][:4])
            payload = read(Path(season['events_file']).with_name('相邻赛季补全_逐队历史观察.json'))
            for club in payload['clubs'].values():
                key = (club['competition'], f"football-data:{club['competition']}:{club['independent_label']}")
                result = self.schedule.setdefault(key, {'coverage': [], 'events': {}})
                result['coverage'].append((date(year, 7, 1), date(year+1, 7, 1)))
                for event in club['events']:
                    identifier = str(event['event_id'])
                    when = datetime.fromisoformat(event['date_utc_text'].replace('Z', '+00:00')).date()
                    result['events'][identifier] = {'date': when, 'id': identifier,
                        'good': event['completed'] and identifier not in blocked,
                        'competitive': event['category'] == 'competitive',
                        'nonleague': event['competition_slug'] != SLUGS[club['competition']]
                                     or not event['is_regular_league_fixture']}
        for club in self.schedule.values():
            merged = []
            for start, end in sorted(club['coverage']):
                if merged and merged[-1][1] >= start:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            club['coverage'] = merged
            club['events'] = sorted(club['events'].values(), key=lambda r: (r['date'], r['id']))

    def performance(self, competition, team, cutoff, venue):
        key = (competition, team, cutoff, venue)
        if key in self.performance_cache:
            return self.performance_cache[key]
        source_key = (competition, team)
        stop = bisect_left(self.league_dates.get(source_key, []), cutoff)
        history = self.league.get(source_key, [])[:stop]
        recent = history[-15:]
        features = {'history_count': float(len(history))}
        for width in (5, 15):
            selected = recent[-width:]
            features[f'history_count_{width}'] = float(len(selected))
            for name in MEASUREMENTS:
                values = [r[name] for r in selected]
                features[f'{name}_mean_{width}'] = math.fsum(values)/len(values) if values and all(v is not None for v in values) else None
            for name, numerator, denominator in (('ppda_ratio', 'ppda_att', 'ppda_def'),
                                                ('ppda_allowed_ratio', 'ppda_allowed_att', 'ppda_allowed_def')):
                top = [r[numerator] for r in selected]
                bottom = [r[denominator] for r in selected]
                features[f'{name}_{width}'] = (math.fsum(top)/math.fsum(bottom)
                    if top and all(v is not None for v in top+bottom) and math.fsum(bottom) > 0 else None)
        for name in MEASUREMENTS[:8]+('ppda_ratio', 'ppda_allowed_ratio'):
            names = [f'{name}_mean_{width}' if name in MEASUREMENTS else f'{name}_{width}' for width in (5,15)]
            values = [features[k] for k in names]
            features[f'trend_{name}_5_minus_15'] = values[0]-values[1] if all(v is not None for v in values) else None
        selected = [r for r in recent if r['is_home'] == venue]
        features['venue_count_15'] = float(len(selected))
        for name in MEASUREMENTS:
            values = [r[name] for r in selected]
            features[f'venue_{name}_mean_15'] = (math.fsum(values)/len(values)
                if len(values) >= 3 and all(v is not None for v in values) else None)
        result = features, history
        self.performance_cache[key] = result
        return result


    def schedule_features(self, competition, team, cutoff):
        key = (competition, team, cutoff)
        if key in self.schedule_cache:
            return self.schedule_cache[key]
        club = self.schedule.get((competition, team))
        interval = next(((a,b) for a,b in club['coverage'] if a <= cutoff <= b), None) if club else None
        events = [r for r in club['events'] if interval and interval[0] <= r['date'] < cutoff
                  and r['competitive']] if club else []
        good, bad = [r for r in events if r['good']], [r for r in events if not r['good']]
        features = {'identity_available': float(club is not None), 'scope_contains_cutoff': float(interval is not None),
            'covered_days_within_28': float(min(28, (cutoff-interval[0]).days)) if interval else 0.}
        windows = {}
        for width in (7,14,28):
            left = cutoff-timedelta(days=width)
            covered = interval is not None and interval[0] <= left
            chosen, issues = [r for r in good if r['date'] >= left], [r for r in bad if r['date'] >= left]
            usable = covered and not issues
            features[f'window_usable_{width}d'] = float(usable)
            features[f'formal_count_{width}d'] = float(len(chosen)) if usable else None
            features[f'nonleague_count_{width}d'] = float(sum(r['nonleague'] for r in chosen)) if usable else None
            features[f'unresolved_count_{width}d'] = float(len(issues)) if covered else None
            windows[str(width)] = chosen
        last = good[-1] if good else None
        usable = last is not None and not any(r['date'] >= last['date'] for r in bad)
        features['last_log1p_gap_to_cutoff_days'] = math.log1p((cutoff-last['date']).days) if usable else None
        features['last_is_nonleague'] = float(last['nonleague']) if usable else None
        previous = good[-3:]
        usable = len(previous) == 3 and not any(r['date'] >= previous[0]['date'] for r in bad)
        features['previous3_count'] = float(len(previous)) if interval else None
        features['previous3_usable'] = float(usable)
        features['previous3_log1p_mean_interval_days'] = (math.log1p(sum(
            (b['date']-a['date']).days for a,b in zip(previous,previous[1:]))/2) if usable else None)
        result = features, windows
        self.schedule_cache[key] = result
        return result


def feature_audit(directory,lead):
    directory = Path(directory)
    plan = read(directory/'分析前计划.json')
    base = read(plan['base_path'])['horizons'][str(lead)]
    with gzip.open(directory/f'features_lead{lead}.json.gz','rt',encoding='utf-8') as handle:
        rows = json.load(handle)['rows']
    base_map = {r['match_id']:r for r in base['rows']}
    assert [r['match_id'] for r in rows] == sorted(base_map,key=lambda mid:(base_map[mid]['date'],mid))
    source = RawSources(ROOT/'outputs/全赛事赛程可用性_20261025')
    counts = Counter(max_feature_error=0.)
    excluded_keys = ('cutoff_staleness_days','previous_league_interval_days',
        'known_league_matches_before_cutoff_7d','known_league_matches_before_cutoff_14d',
        'known_league_matches_before_cutoff_28d','same_league_gap_to_match_days')
    keys_by_arm = {}
    for row in rows:
        mid, target = row['match_id'], date.fromisoformat(row['date'])
        forecast, parameter = parameter_day(target,lead)
        assert row['forecast_before_date'] == forecast.isoformat()
        assert row['fit_before_date'] == parameter.isoformat()
        assert parameter <= forecast < target
        base_row = base_map[mid]
        for field in ('date','period','competition','home','away','actual_score','actual_outcome',
                      'base_parameter_key','blend_role','blend_weight_before_date'):
            assert row[field] == base_row[field]
        assert row['goal_rates'] == base_row['rates'] and row['xg_rates'] == base_row['xg_rates']
        assert row['clean_blend_rates'] == base_row['blend_rates']
        assert row['base_training_max_date'] < row['fit_before_date']
        assert row['actual_score'] == list(source.independent[mid].full_time)
        assert not source.independent[mid].goal_fit_exclusion
        for mode, cutoff in (('fresh',forecast),('locked',parameter)):
            assert row[mode+'_feature_before_date'] == cutoff.isoformat()
            metadata = row[mode+'_timing']
            assert metadata['refresh'] == (mode == 'fresh')
            assert metadata['parameter_before_date'] == parameter.isoformat()
            assert metadata['forecast_before_date'] == forecast.isoformat()
            assert metadata['feature_before_date'] == cutoff.isoformat()
            assert metadata['publication_lead_days'] == lead
            assert metadata['input_lead_days'] == (target-cutoff).days
            assert metadata['parameter_age_at_publication_days'] == (forecast-parameter).days
            histories = []
            for label, team, venue in (('home',row['home'],True),('away',row['away'],False)):
                features, history = source.performance(row['competition'],team,cutoff,venue)
                histories.append((features,history))
                meta = row[mode+'_feature_history']['history'][label]
                assert meta['history_count'] == len(history)
                assert meta['used_match_ids'] == [r['match_id'] for r in history[-15:]]
                assert meta['latest_used_date'] == (history[-1]['date'].isoformat() if history else None)
                for width in (5,15):
                    assert meta['window_match_ids'][str(width)] == [r['match_id'] for r in history[-width:]]
                counts['performance_reference_entries'] += min(15,len(history))
            available = min(len(history) for _,history in histories) >= 5
            for side,(own,other) in enumerate(((histories[0][0],histories[1][0]),(histories[1][0],histories[0][0]))):
                expected = {**{'own_'+k:v if available else None for k,v in own.items()},
                            **{'opponent_'+k:v if available else None for k,v in other.items()}}
                expected.update({f'league_{comp}':float(row['competition']==comp) for comp in SLUGS})
                expected.update(is_home=float(side==0), log_goal_mean_for=math.log(row['goal_rates'][side]),
                    log_goal_mean_against=math.log(row['goal_rates'][1-side]),
                    log_xg_mean_for=math.log(row['xg_rates'][side]),
                    log_xg_mean_against=math.log(row['xg_rates'][1-side]),
                    history_features_available=float(available),
                    forecast_log1p_lead_days=math.log1p((target-cutoff).days),
                    parameter_log1p_age_at_forecast_days=math.log1p((forecast-parameter).days))
                if mode == 'fresh':
                    close_dict(row['fresh_performance_features'][side],expected,f'{mid}:{mode}:performance',counts)
                extra = {}
                teams = ((row['home'],row['away']) if side == 0 else (row['away'],row['home']))
                for prefix,team in zip(('own_','opponent_'),teams,strict=True):
                    features, windows = source.schedule_features(row['competition'],team,cutoff)
                    extra.update({prefix+'allcomp_'+k:v for k,v in features.items()})
                    if prefix == 'own_':
                        label = 'home' if side == 0 else 'away'
                        meta = row[mode+'_allcomp_history']['history'][label]
                        for width,chosen in windows.items():
                            assert meta['windows'][width]['selected_event_ids'] == [r['id'] for r in chosen]
                            assert meta['windows'][width]['selected_event_dates'] == [r['date'].isoformat() for r in chosen]
                            counts['schedule_reference_entries'] += len(chosen)
                arm = mode+'_allcomp'
                full = expected|extra
                close_dict(row[arm+'_features'][side],full,f'{mid}:{arm}',counts)
                keys = set(full)
                assert not any(any(key.endswith(suffix) for suffix in excluded_keys) for key in keys)
                assert not any('blend' in key or 'actual_' in key for key in keys)
                keys_by_arm.setdefault(arm,keys)
                assert keys_by_arm[arm] == keys
        assert all(row['fresh_allcomp_features'][i][k] == v for i in (0,1)
                   for k,v in row['fresh_performance_features'][i].items())
    result = {'status':'passed','horizon_days':lead,'rows':len(rows),'by_period':dict(Counter(r['period'] for r in rows)),
        'feature_counts':{'fresh_performance':121,'fresh_allcomp':161,'locked_allcomp':161},
        'counts':dict(counts),'all_row_parameter_forecast_feature_times_verified':True,
        'independent_method':'original provider team histories and original member schedules; no production feature builder imported',
        'fresh_pair_only_differs_by40schedule_inputs':True,'current_results_reviewed':False}
    write(directory/f'独立特征核对_lead{lead}.json',result)
    return result


def model_audit(directory,lead):
    directory = Path(directory)
    folder = directory/f'lead_{lead}'
    with gzip.open(directory/f'features_lead{lead}.json.gz','rt',encoding='utf-8') as handle:
        rows = json.load(handle)['rows']
    row_map = {r['match_id']:r for r in rows}
    summary = read(folder/'完整结果汇总.json')
    parameters = read(folder/'全部模型参数.json')
    predictions = read(folder/'逐场完整预测.json')
    plan = read(directory/'分析前计划.json')
    arms = tuple(plan['arms_per_horizon'])
    expected_count = 3*sum(len({r['fit_before_date'] for r in rows if r['period'] == period})
                         for period in ('2024-25','2025-26'))
    assert len(parameters) == summary['new_scheme_fits'] == expected_count
    progress = read(folder/'训练进度.json')
    assert progress['fits'] == expected_count
    assert summary['new_fits_this_execution'] + summary['reused_completed_fits'] == expected_count
    assert summary['new_fits_this_execution'] == progress['new_fits_this_execution']
    assert summary['reused_completed_fits'] == progress['reused_completed_fits']
    expected_windows = [(period,cutoff) for period in ('2024-25','2025-26')
        for cutoff in sorted({r['fit_before_date'] for r in rows if r['period'] == period})]
    assert [(w['period'],w['cutoff']) for w in summary['windows']] == expected_windows
    assert summary['windows'] == progress['windows']
    assert {p.stem for p in (folder/'models').glob('*.joblib')} == {
        key.replace(':','_',1) for key in parameters}
    _, independent, _, _, _ = load_source()
    feature_names = {arm:sorted(rows[0][arm+'_features'][0]) for arm in arms}
    matrices = {arm:np.asarray([[np.nan if side[k] is None else side[k]
        for k in feature_names[arm]] for row in rows for side in row[arm+'_features']],dtype=float)
        for arm in arms}
    references = imputation_values = 0
    for key,parameter in parameters.items():
        arm, cutoff = key.split(':',1)
        history = [r for r in rows if r['date'] < cutoff]
        assert parameter['reference_match_ids'] == [r['match_id'] for r in history]
        assert parameter['n_matches'] == len(history)
        assert parameter['n_scoring_sides'] == 2*len(history)
        assert parameter['before_date'] == cutoff
        assert parameter['training_max_date'] < cutoff
        assert parameter['pipeline'] == arm and parameter['horizon_days'] == lead
        assert parameter['feature_names'] == sorted(history[0][arm+'_features'][0])
        assert all(not independent[r['match_id']].goal_fit_exclusion for r in history)
        assert parameter['kind'] == 'extra_trees' and parameter['half_life_days'] == 180
        config = parameter['configuration']
        assert config['n_estimators'] == 120 and config['max_depth'] == 6
        assert config['min_samples_leaf'] == 60 and config['max_features'] == .8
        assert config['criterion'] == 'poisson' and config['random_state'] == 20261025
        held = [r for r in rows if r['period'] in ('2024-25','2025-26') and r['fit_before_date'] == cutoff]
        assert held and all(cutoff <= r['forecast_before_date'] < r['date'] for r in held)
        assert all(r['forecast_before_date'] < cutoff for r in history)
        saved_model = joblib.load(folder/'models'/f'{arm}_{cutoff}.joblib')
        assert saved_model.parameters() == {k:v for k,v in parameter.items()
            if k not in ('pipeline','horizon_days')}
        # Recompute the training-only preprocessing directly from earlier rows.
        matrix = matrices[arm][:2*len(history)]
        missing = np.isnan(matrix)
        empty = missing.all(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            medians = np.nanmedian(matrix,axis=0)
        medians[empty] = 0.
        imputation = parameter['imputation']
        assert np.max(np.abs(medians-np.asarray(imputation['statistics']))) < 1e-12
        assert imputation['indicator_feature_indices'] == np.flatnonzero(missing.any(axis=0)).tolist()
        assert imputation['all_missing_training_features'] == [feature_names[arm][i]
            for i in np.flatnonzero(empty)]
        assert imputation['strategy'] == 'unweighted training median'
        assert imputation['keep_empty_features'] is True and imputation['add_indicator'] is True
        imputation_values += len(medians)
        references += len(history)
    del matrices
    max_score_error = max_probability_error = max_interval_error = max_difference_error = 0.
    recomputed_tracks = 0
    for period,section in (('2024-25',summary['validation']),('2025-26',summary['test'])):
        expected = [r['match_id'] for r in rows if r['period'] == period]
        assert section['match_ids'] == expected
        assert section['n'] == len(expected)
        local = {arm:[r for r in values if r['period'] == period] for arm,values in predictions.items()
                 if arm != 'clean_blend' or period == '2025-26'}
        assert set(local) == set(section['scores'])
        analytic_losses = {}
        for arm,track in local.items():
            assert [r['match_id'] for r in track] == expected
            for predicted in track:
                original = row_map[predicted['match_id']]
                for key in ('match_id','date','period','competition','home','away','fit_before_date',
                            'forecast_before_date','horizon_days','actual_score','actual_outcome'):
                    assert predicted[key] == original[key]
                assert predicted['fit_before_date'] <= predicted['forecast_before_date'] < predicted['date']
                if arm in arms:
                    field = 'locked_feature_before_date' if arm == 'locked_allcomp' else 'fresh_feature_before_date'
                    assert predicted['feature_before_date'] == original[field]
                if arm == 'clean_blend':
                    assert original['blend_role'] == 'earlier_calibration_fixed_before_publication'
                    assert original['blend_weight_before_date'] <= original['forecast_before_date']
            values, probability_error = independent_scores(track)
            max_probability_error = max(max_probability_error, probability_error)
            for domain,metrics in values.items():
                for statistic,value in metrics.items():
                    error = abs(value-section['scores'][arm][domain][statistic])
                    max_score_error = max(max_score_error,error)
                    assert error < 1e-10, (lead,period,arm,domain,statistic,value)
            analytic_losses[arm] = loss_vectors(track)
            recomputed_tracks += 1
        matrix = []
        for column in section['difference_order']:
            contrast, domain = column.split(':',1)
            arm, reference = contrast.split('_minus_',1)
            difference = analytic_losses[arm][domain]-analytic_losses[reference][domain]
            error = abs(float(difference.mean())-section['mean_loss_differences'][column])
            max_difference_error = max(max_difference_error,error)
            assert error < 1e-10
            matrix.append(difference)
        matrix = np.column_stack(matrix)
        for length,saved in section['block_intervals'].items():
            intervals = independent_blocks([r['date'] for r in local['goal']],matrix,int(length))
            error = float(np.abs(intervals-np.asarray(saved['interval_95'])).max())
            max_interval_error = max(max_interval_error,error)
            assert error < 1e-10
        if period == '2024-25':
            assert 'clean_blend' not in section['scores']
    recomputed_league_tracks = 0
    for competition,section in summary['by_league'].items():
        expected = [r['match_id'] for r in rows if r['period'] == '2025-26'
                    and r['competition'] == competition]
        assert section['match_ids'] == expected and section['n'] == len(expected)
        assert not section['block_intervals']
        local_losses = {}
        assert set(section['scores']) == set(predictions)
        for arm,track in predictions.items():
            track = [r for r in track if r['period'] == '2025-26' and r['competition'] == competition]
            assert [r['match_id'] for r in track] == expected
            scores,error = independent_scores(track)
            max_probability_error = max(max_probability_error,error)
            for domain,metrics in scores.items():
                for statistic,value in metrics.items():
                    error = abs(value-section['scores'][arm][domain][statistic])
                    max_score_error = max(max_score_error,error)
                    assert error < 1e-10
            local_losses[arm] = loss_vectors(track)
            recomputed_league_tracks += 1
        for column in section['difference_order']:
            contrast, domain = column.split(':',1)
            arm, reference = contrast.split('_minus_',1)
            difference = local_losses[arm][domain]-local_losses[reference][domain]
            error = abs(float(difference.mean())-section['mean_loss_differences'][column])
            max_difference_error = max(max_difference_error,error)
            assert error < 1e-10
    # Saved-estimator replay includes both end seasons, not a fresh fit.
    max_replay_error, replay_count = 0., 0
    for arm in arms:
        for period in ('2024-25','2025-26'):
            cutoffs = sorted({r['fit_before_date'] for r in predictions[arm] if r['period'] == period})
            for cutoff in (cutoffs[0],cutoffs[-1]):
                track = [r for r in predictions[arm] if r['fit_before_date'] == cutoff and r['period'] == period][:3]
                model = joblib.load(folder/'models'/f'{arm}_{cutoff}.joblib')
                replay = model.predict_many([row_map[r['match_id']][arm+'_features'] for r in track],
                    [date.fromisoformat(r['date']) for r in track])
                for rates,predicted in zip(replay,track,strict=True):
                    error = max(abs(a-b) for a,b in zip(rates,predicted['rates'],strict=True))
                    max_replay_error = max(max_replay_error,error)
                    assert error < 1e-10
                    replay_count += 1
    # Keep all declared schemes and both horizons; no post-test winner field.
    assert 'selected' not in summary and summary['adopted'] is False
    result = {'status':'passed','horizon_days':lead,'saved_models':len(parameters),
        'training_reference_entries':references,'score_tracks_recomputed':recomputed_tracks,
        'training_only_imputation_statistics_recomputed':imputation_values,
        'league_score_tracks_recomputed':recomputed_league_tracks,
        'new_fits_this_execution':summary['new_fits_this_execution'],
        'reused_completed_fits':summary['reused_completed_fits'],
        'max_score_error':max_score_error,'max_probability_error':max_probability_error,
        'max_mean_difference_error':max_difference_error,'max_interval_error':max_interval_error,
        'replayed_predictions':replay_count,'max_replay_error':max_replay_error,
        'all_models_available_by_publication':True,'calibration_year_blend_excluded':True,
        'new_scheme_or_horizon_selection':False,
        'scope':'independent retrospective date/computation audit; no claim precise24/72h or complete publication timestamps'}
    write(directory/f'独立模型与概率评分核对_lead{lead}.json',result)
    return result



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',type=Path)
    parser.add_argument('--directory',type=Path)
    parser.add_argument('--lead',type=int,choices=(1,3))
    parser.add_argument('--features-only',action='store_true')
    args = parser.parse_args()
    if args.base:
        print(json.dumps(base_audit(args.base),ensure_ascii=False),flush=True)
    elif args.directory and args.lead:
        if args.features_only:
            print(json.dumps(feature_audit(args.directory,args.lead),ensure_ascii=False),flush=True)
        else:
            print(json.dumps(model_audit(args.directory,args.lead),ensure_ascii=False),flush=True)
    else:
        raise SystemExit('Provide --base or wait for the completed prepared-output interface; no fitting is performed.')
