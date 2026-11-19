"""Five current Serie A forecasts from fixed existing statistical methods.

Read local score/xG sources and confirmed fixtures only. Fit two league-specific
base experts once; reuse the saved five-league 121-feature ExtraTrees model.
Same-day composition uses the explicit PreparedPredictionFeatures match-day
API. Lead-zero application of the saved lead-three tree remains research only;
this script does not establish new historical performance.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import poisson

from football_model.all_competition_features import AllCompetitionFeatures
from football_model.date_model import DateTeamGoalModel
from football_model.fixed_horizon_features import PreparedPredictionFeatures
from football_model.football_data import HistoricalMatch, load_directory
from football_model.probability import score_grid, summarize
from football_model.xg_history import DateTeamXgModel
from current_rich_model_helpers_20261026 import load_current_rich_sources


ROOT = Path('G:/Codex/football-prediction-2026-10-22')
INPUT = ROOT / 'data/serie_a_today_20261028'
OUTPUT = ROOT / 'outputs/今日意甲预测_20261028'
SAVED_TREE = ROOT / 'outputs/当前丰富信息预测_20261026T030805654404Z/丰富信息候选模型.joblib'
FIT_DATE = date(2026, 10, 28)
EXPECTED_PAIRS = (('Milan', 'Bologna'), ('Parma', 'Udinese'), ('Venezia', 'Inter'), ('Genoa', 'Juventus'), ('Monza', 'Napoli'))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )


def same_day_performance_features(prepared_builder, *, home, away, competition,
                                  parameter_cutoff, forecast_cutoff, match_date,
                                  goal_rates, xg_rates):
    """Use the common feature formula without duplicating the 121 fields."""
    method = (prepared_builder.for_same_day_match if forecast_cutoff == match_date
              else prepared_builder.for_match)
    return method(
        home=home, away=away, competition=competition,
        parameter_cutoff=parameter_cutoff, forecast_cutoff=forecast_cutoff,
        match_date=match_date, goal_rates=goal_rates, xg_rates=xg_rates, refresh=True)


def compare_original_formula(observations, fixtures):
    """Check five real identities x two sides x 121 values, without a refit."""
    early = PreparedPredictionFeatures(observations, AllCompetitionFeatures([]))
    original = PreparedPredictionFeatures(observations, AllCompetitionFeatures([]))
    field_count = 0
    for fixture in fixtures:
        # Fixed positive scalars isolate formula correctness; these are not
        # predictions issued for the previous date or historical backtest rows.
        kwargs = dict(
            home='football-data:I1:' + fixture['home_model_label'],
            away='football-data:I1:' + fixture['away_model_label'],
            competition='I1', parameter_cutoff=date(2026, 10, 27),
            forecast_cutoff=date(2026, 10, 27), match_date=FIT_DATE,
            goal_rates=(1.25, .95), xg_rates=(1.15, 1.05),
        )
        reference = original.for_match(**kwargs, refresh=True)['performance_features']
        adapted = same_day_performance_features(early, **kwargs)['performance_features']
        for left, right in zip(reference, adapted, strict=True):
            if set(left) != set(right) or len(left) != 121:
                raise ValueError('Adapted performance feature names differ from the original')
            for name in left:
                if left[name] != right[name]:
                    raise ValueError(f'Adapted performance value differs: {name}')
                field_count += 1
    return {
        'matched': True, 'checked_fixtures': len(fixtures), 'checked_scoring_sides': 2*len(fixtures),
        'checked_feature_values': field_count, 'feature_count_per_side': 121,
        'comparison_parameter_cutoff': '2026-10-27',
        'comparison_forecast_cutoff': '2026-10-27', 'target_date': FIT_DATE.isoformat(),
        'input_goal_rates': [1.25, .95], 'input_xg_rates': [1.15, 1.05],
        'equality': 'exact names and values, including None',
        'purpose': 'input-formula correctness only; no historical scoring or validation claim',
        'core_prepared_feature_checks_modified': False,
    }


def distribution(rates):
    grid = score_grid(*rates, rho=0)
    matrix = grid.grid
    total = grid.total_goals_distribution()
    summary = summarize(grid)
    p = [summary['one_x_two'][key] for key in ('home', 'draw', 'away')]
    if (not np.isfinite(matrix).all() or np.any(matrix < 0)
            or abs(float(matrix.sum()) - 1) > 1e-12
            or abs(math.fsum(p) - 1) > 1e-12):
        raise ValueError('Invalid score/class probabilities')
    top = sorted(
        ({'home_goals': h, 'away_goals': a, 'score': f'{h}-{a}',
          'probability': float(matrix[h, a])}
         for h in range(matrix.shape[0]) for a in range(matrix.shape[1])),
        key=lambda row: (-row['probability'], row['home_goals'], row['away_goals']),
    )[:10]
    omitted = float(1 - poisson.cdf(matrix.shape[0] - 1, rates[0])
                    * poisson.cdf(matrix.shape[1] - 1, rates[1]))
    return {
        'rates': list(rates), 'class_order': ['home', 'draw', 'away'],
        'probabilities': p, 'summary': summary,
        'score_grid': {
            'home_goal_axis': list(range(matrix.shape[0])),
            'away_goal_axis': list(range(matrix.shape[1])),
            'orientation': 'rows=home_goals; columns=away_goals',
            'probabilities': matrix.tolist(), 'sum': float(matrix.sum()),
            'untruncated_law': 'independent Poisson with the two reported rates; rho=0',
            'omitted_probability_before_normalization': omitted,
            'normalized_after_negligible_truncation': True,
        },
        'top_scores': top, 'over_2_5': summary['over_2_5'],
        'total_goals_0_to_7_plus': {
            **{str(k): float(total[k]) for k in range(7)},
            '7+': float(total[7:].sum()),
        },
    }


def source_scope(records):
    return {
        'n': len(records), 'date_min': min(r.match_date for r in records).isoformat(),
        'date_max': max(r.match_date for r in records).isoformat(),
        'by_season': dict(Counter(r.match_id.split(':')[2] for r in records)),
    }


def main():
    started = datetime.now(timezone.utc)
    if started.date() != FIT_DATE:
        raise ValueError('This explicit 2026-10-28 run must execute on its actual UTC day')
    confirmed = read(INPUT / 'confirmed_fixtures.json')
    fixtures = confirmed['fixtures']
    if tuple((r['home_model_label'], r['away_model_label']) for r in fixtures) != EXPECTED_PAIRS:
        raise ValueError('The requested five fixture identities differ')
    for row in fixtures:
        kickoff = datetime.fromisoformat(row['kickoff'])
        if (row['status'] != 'scheduled' or row['neutral'] is not False
                or kickoff.tzinfo is None or kickoff <= started
                or kickoff.astimezone(timezone.utc).date() != FIT_DATE):
            raise ValueError('Need the confirmed future nonneutral target fixtures on the true UTC date')
    current, current_report = load_directory(INPUT / 'raw', season='2026/27')
    if current_report['rejected_rows'] or current_report['duplicate_keys']:
        raise ValueError('Resolve invalid/duplicate independent current rows before fitting')
    # Existing FD loaders also expose auxiliary source fields. Remove all market
    # fields immediately; only identities, dates and scores reach model inputs.
    current = [replace(r, markets={}) for r in current if r.competition == 'I1']
    # Football-Data's current file still ends Oct25. Complete the earlier-date
    # independent score side using real ESPN Oct27 results, never today's labels.
    extra_path = INPUT / 'raw/espn_ita1_20261027_completed.json'
    extra_receipt = read(INPUT / 'receipts/espn_ita1_20261027_completed.json')
    current_by_pair = {(r.match_date, r.home_label, r.away_label): r for r in current}
    supplement = []
    aliases = {'AS Roma': 'Roma', 'AC Milan': 'Milan', 'Internazionale': 'Inter'}
    for event in read(extra_path)['events']:
        contest = event['competitions'][0]
        status = contest.get('status', event.get('status'))['type']
        played = datetime.fromisoformat(event['date'].replace('Z', '+00:00')).date()
        if played >= FIT_DATE or status['completed'] is not True or status['name'] not in ('STATUS_FULL_TIME', 'STATUS_FINAL'):
            raise ValueError('Independent supplement needs strictly earlier completed regulation-time games')
        sides = {r['homeAway']: r for r in contest['competitors']}
        home = aliases.get(sides['home']['team']['displayName'], sides['home']['team']['displayName'])
        away = aliases.get(sides['away']['team']['displayName'], sides['away']['team']['displayName'])
        scores = tuple(int(sides[key]['score']) for key in ('home', 'away'))
        key = (played, home, away)
        if key in current_by_pair:
            if current_by_pair[key].full_time != scores:
                raise ValueError('Independent completed score disagreement')
            continue
        record = HistoricalMatch(
            match_id=f'football-data:I1:2026/27:{home}|{away}', competition='I1', season='2026/27',
            match_date=played, source_time_text=event['date'],
            home='football-data:I1:'+home, away='football-data:I1:'+away,
            home_label=home, away_label=away, full_time=scores, half_time=None, markets={},
            source_path=str(extra_path), source_url=extra_receipt['source_url'], goal_fit_exclusion=None)
        current.append(record)
        current_by_pair[key] = record
        supplement.append({'match_id': record.match_id, 'date': played.isoformat(), 'home': home, 'away': away,
                           'full_time': list(scores), 'source_url': record.source_url,
                           'observed_at': extra_receipt['received_at']})
    sources = load_current_rich_sources(
        INPUT / 'raw/Serie_A_2026.json', current,
        before_date=FIT_DATE, season='2026/27', competition='I1',
    )
    goal_history = [replace(r, markets={}) for r in sources['independent'].values()
                    if r.competition == 'I1' and r.match_date < FIT_DATE]
    xg_history = [r for r in sources['np_records'].values()
                  if r.competition == 'I1' and r.match_date < FIT_DATE]
    observations = [r for r in sources['observations']
                    if r.competition == 'I1' and r.match_date < FIT_DATE]
    target_keys = {(FIT_DATE, 'football-data:I1:' + r['home_model_label'],
                    'football-data:I1:' + r['away_model_label']) for r in fixtures}
    if any((r.match_date, r.home, r.away) in target_keys for r in goal_history + xg_history):
        raise ValueError('A target label entered earlier model history')
    print(json.dumps({'stage': 'sources_joined', 'current': len(current),
                      'goal_history': len(goal_history), 'xg_history': len(xg_history)},
                     ensure_ascii=False), flush=True)
    comparison = compare_original_formula(observations, fixtures)
    goal = DateTeamGoalModel(.01, 180).fit(goal_history, FIT_DATE)
    xg = DateTeamXgModel(.01, 180).fit(xg_history, FIT_DATE)
    tree = joblib.load(SAVED_TREE)
    tree_parameters = tree.parameters()
    tree_leagues = dict(Counter(k.split(':')[1] for k in tree_parameters['reference_match_ids']))
    if 'I1' not in tree_leagues or tree_leagues['I1'] <= 0:
        raise ValueError('Saved rich model lacks an established Italian training scope')
    if datetime.fromisoformat(tree_parameters['before_date']).date() > FIT_DATE:
        raise ValueError('Saved rich model parameters are from after the forecast cutoff')
    builder = PreparedPredictionFeatures(observations, AllCompetitionFeatures([]))
    records, feature_pairs, dates = [], [], []
    for fixture in fixtures:
        home = 'football-data:I1:' + fixture['home_model_label']
        away = 'football-data:I1:' + fixture['away_model_label']
        day = datetime.fromisoformat(fixture['kickoff']).astimezone(timezone.utc).date()
        goal_rates = goal.predict(home, away, 'I1', day)
        xg_rates = xg.predict(home, away, 'I1', day)
        features = same_day_performance_features(
            builder, home=home, away=away, competition='I1',
            parameter_cutoff=FIT_DATE, forecast_cutoff=FIT_DATE, match_date=day,
            goal_rates=goal_rates, xg_rates=xg_rates,
        )
        if any(set(pair) != set(tree_parameters['feature_names'])
               for pair in features['performance_features']):
            raise ValueError('Current feature names do not match the saved rich model')
        for side in features['feature_history']['history'].values():
            if side['latest_used_date'] is not None and side['latest_used_date'] >= FIT_DATE.isoformat():
                raise ValueError('A target-day or future observation entered rich features')
        feature_pairs.append(features['performance_features'])
        dates.append(day)
        records.append({
            'fixture': fixture, 'home_model_identity': home, 'away_model_identity': away,
            'as_of': started.isoformat(), 'actual_score': None, 'actual_outcome': None,
            'goal_simple': distribution(goal_rates), 'independent_xg': distribution(xg_rates),
            'rich_feature_details': features,
        })
    rich_rates = tree.predict_many(feature_pairs, dates)
    for record, rates in zip(records, rich_rates, strict=True):
        record['rich_121_research'] = distribution(rates)
    finished = datetime.now(timezone.utc)
    if any(datetime.fromisoformat(r['fixture']['kickoff']) <= finished for r in records):
        raise ValueError('A target started before the forecast was saved')
    goal_parameters, xg_parameters = goal.parameters(), xg.parameters()
    goal_parameters['current_application'] = (
        'Explicit current I1 research forecast using local independently joined earlier-date scores; '
        'the generic model class retains its original retrospective limitation metadata.'
    )
    metadata = {
        'base_fit_cutoff': FIT_DATE.isoformat(), 'competition': 'I1',
        'regularization': .01, 'half_life_days': 180,
        'new_fits': {'goal_simple': 1, 'independent_xg': 1, 'rich_121': 0},
        'parameter_searches': 0, 'historical_backtests_run': 0,
        'goal_history_input': source_scope(goal_history),
        'goal_fit_used_n': goal_parameters['n_matches'],
        'goal_exclusions': [
            {'match_id': r.match_id, 'reason': r.goal_fit_exclusion}
            for r in goal_history if r.goal_fit_exclusion is not None
        ],
        'xg_history_input': source_scope(xg_history),
        'xg_fit_used_n': xg_parameters['n_matches'],
        'goal_parameters': goal_parameters, 'xg_parameters': xg_parameters,
        'rich_121': {
            'source_model_file': str(SAVED_TREE), 'new_fit': False,
            'training_league_counts': tree_leagues,
            'total_training_n': tree_parameters['n_matches'],
            'italian_training_n': tree_leagues['I1'],
            'parameters': tree_parameters, 'historical_training_lead_days': 3,
            'current_lead_days': 0, 'current_base_parameter_age_days': 0,
            'same_day_generalization_validated': False,
            'training_source_feature_file': str(ROOT / 'outputs/固定发布日期信息刷新_20261025T071806561562Z/features_lead3.json.gz'),
            'note': 'Existing pooled five-league goal regressor including I1; same-day lead-zero research extrapolation from lead-three training snapshots.',
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT / '本次意甲统计模型.joblib'
    joblib.dump({'goal': goal, 'xg': xg, 'rich_121': tree, 'metadata': metadata}, model_path)
    restored = joblib.load(model_path)
    for index, record in enumerate(records):
        home, away = record['home_model_identity'], record['away_model_identity']
        for key, model_name in (('goal_simple', 'goal'), ('independent_xg', 'xg')):
            again = restored[model_name].predict(home, away, 'I1', dates[index])
            if any(abs(a - b) > 1e-12 for a, b in zip(again, record[key]['rates'], strict=True)):
                raise ValueError('Saved base model does not reproduce its forecast')
    restored_rich = restored['rich_121'].predict_many(feature_pairs, dates)
    if any(any(abs(a - b) > 1e-12 for a, b in zip(original, saved, strict=True))
           for original, saved in zip(rich_rates, restored_rich, strict=True)):
        raise ValueError('Saved rich model does not reproduce its forecast')
    final_generated = datetime.now(timezone.utc)
    if any(datetime.fromisoformat(r['fixture']['kickoff']) <= final_generated for r in records):
        raise ValueError('A target started before final probability publication')
    result = {
        'as_of': started.isoformat(), 'generated_at': final_generated.isoformat(),
        'competition': 'I1', 'target_count': len(records),
        'class_order': ['home', 'draw', 'away'], 'model_file': str(model_path),
        'source_files': {
            'confirmed_fixtures': str(INPUT / 'confirmed_fixtures.json'),
            'current_independent_scores': str(INPUT / 'raw/I1.csv'),
            'current_xg': str(INPUT / 'raw/Serie_A_2026.json'),
        },
        'model_metadata': metadata,
        'current_source_report': {
            'independent_matches': len(current),
            'earlier_espn_independent_supplement': supplement,
            'independent_min_date': min(r.match_date for r in current).isoformat(),
            'independent_max_date': max(r.match_date for r in current).isoformat(),
            'rejected_rows': current_report['rejected_rows'],
            'duplicate_keys': current_report['duplicate_keys'],
            'exact_current_join': sources['reports']['current_join'],
            'full_xg_source_report': sources['reports'],
        },
        'feature_formula_comparison': comparison,
        'records': records,
        'checks': {
            'saved_models_reproduce': True, 'target_labels_used': False,
            'target_day_or_future_events_used_in_features': False,
            'all_feature_observation_dates_strictly_before_cutoff': True,
            'same_day_dates_not_fabricated': True,
            'core_prepared_feature_checks_modified': False,
            'market_features_used': False, 'llm_features_used': False,
            'new_network_requests': 0,
        },
        'limits': [
            'All fits and feature histories use strictly earlier calendar dates; historical first-publication times and revisions are not reconstructed.',
            'The simple and xG models use their respective complete eligible I1 histories. Their source exclusions and counts are reported separately, not asserted to be identical.',
            'The independent xG line uses fractional-xG quasi-loss means as future goal rates; that conversion and the shared independent Poisson score law remain research assumptions.',
            'The reused 121-feature rich model was trained on lead-three snapshots, while this run uses a zero-calendar-day lead and current base experts. Same-day generalization has not been validated.',
            'No player availability, expected lineup, tactical news, market prices, LLM judgments or current all-competition workload features enter these statistical forecasts.',
            'The three statistical lines remain separate; no new fusion weights, parameter selection or historical backtest is performed.',
        ],
    }
    write(OUTPUT / '统计预测与比分分布.json', result)
    print(json.dumps({
        'output': str(OUTPUT / '统计预测与比分分布.json'), 'model_file': str(model_path),
        'new_fits': metadata['new_fits'], 'goal_history_input_n': len(goal_history),
        'goal_fit_n': goal_parameters['n_matches'], 'xg_fit_n': xg_parameters['n_matches'],
        'rich_i1_training_n': tree_leagues['I1'], 'rich_training_n': tree_parameters['n_matches'],
        'feature_formula_comparison': comparison,
        'predictions': [
            {'home': r['fixture']['home_model_label'], 'away': r['fixture']['away_model_label'],
             **{key: {'rates': r[key]['rates'], 'probabilities': r[key]['probabilities'],
                      'top_score': r[key]['top_scores'][0]}
                for key in ('goal_simple', 'independent_xg', 'rich_121_research')}}
            for r in records
        ],
    }, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
