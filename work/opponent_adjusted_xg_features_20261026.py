"""Add recent xG surprises using each past fixture's saved earlier forecast.

This helper does not fit a model, regenerate an old forecast, download data, or
drop rows.  Source exclusions remain the responsibility of the unchanged
``run_npxg_comparison.load_source`` join and its original reports.
"""

from collections import Counter, defaultdict
from datetime import date, datetime
import math


EXTRA_FEATURE_NAMES = (
    'own_xg_net_residual_mean_5',
    'opponent_xg_net_residual_mean_5',
)


def _day(value, label):
    if isinstance(value, datetime):
        raise ValueError(f'Expected a date without a time: {label}')
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'Invalid date: {label}') from error


def _nonnegative_pair(value, label):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f'Expected two measurements: {label}')
    if any(isinstance(x, bool) for x in value):
        raise ValueError(f'Boolean measurement: {label}')
    try:
        pair = tuple(float(x) for x in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'Invalid measurement: {label}') from error
    if any(not math.isfinite(x) or x < 0 for x in pair):
        raise ValueError(f'Expected finite nonnegative measurements: {label}')
    return pair


def _team_history(target, side, index, actual_xg_records):
    team = target[side]
    target_cutoff = _day(target['forecast_before_date'], 'target forecast')
    history = target.get('fresh_feature_history', {}).get('history', {}).get(side)
    if history is not None and (history.get('team') != team or
                               history.get('competition') != target['competition']):
        raise ValueError(f'History team or competition differs: {target["match_id"]}/{side}')
    ids = None if history is None else history.get('window_match_ids', {}).get('5')
    if ids is None:
        ids = []
        window_reason = 'recent_five_window_missing'
    else:
        if not isinstance(ids, (tuple, list)) or any(not isinstance(mid, str) for mid in ids):
            raise ValueError('Recent-five reference IDs must be an explicit list of strings')
        ids = list(ids)
        if len(ids) > 5 or len(set(ids)) != len(ids):
            raise ValueError('Recent-five window contains duplicate or excess references')
        window_reason = 'fewer_than_five_observations' if len(ids) < 5 else None

    observations, missing = [], []
    for mid in ids:
        past = index.get(mid)
        if past is None:
            missing.append({'match_id': mid, 'reason': 'earlier_match_forecast_missing'})
            continue
        required = ('fit_before_date', 'forecast_before_date', 'date', 'actual_score')
        if any(past.get(key) is None for key in required):
            missing.append({'match_id': mid, 'reason': 'earlier_forecast_metadata_missing'})
            continue
        fit = _day(past['fit_before_date'], f'{mid}/fit')
        forecast = _day(past['forecast_before_date'], f'{mid}/forecast')
        played = _day(past['date'], f'{mid}/date')
        if not fit <= forecast < played < target_cutoff:
            raise ValueError(f'Past forecast or result crosses target cutoff: {mid}')
        if past['competition'] != target['competition'] or team not in (past['home'], past['away']):
            raise ValueError(f'Reference team or competition differs: {mid}')
        if past.get('xg_rates') is None or any(x is None for x in past['xg_rates']):
            missing.append({'match_id': mid, 'reason': 'earlier_xg_forecast_missing'})
            continue
        expected = _nonnegative_pair(past['xg_rates'], f'{mid}/saved xG forecast')
        actual = actual_xg_records.get(mid)
        if actual is None:
            missing.append({'match_id': mid, 'reason': 'exact_actual_xg_source_missing'})
            continue
        if (actual.match_id != mid or _day(actual.match_date, f'{mid}/source date') != played or
                actual.home != past['home'] or actual.away != past['away'] or
                actual.competition != past['competition']):
            raise ValueError(f'Exact actual-xG identity differs: {mid}')
        goals = _nonnegative_pair(actual.goals, f'{mid}/source goals')
        saved_goals = _nonnegative_pair(past['actual_score'], f'{mid}/saved actual score')
        if any(x != int(x) for x in goals + saved_goals) or goals != saved_goals:
            raise ValueError(f'Exact actual-xG score differs: {mid}')
        measured = _nonnegative_pair(actual.xg, f'{mid}/actual xG')
        direction = 1 if team == past['home'] else -1
        actual_net = direction * (measured[0] - measured[1])
        expected_net = direction * (expected[0] - expected[1])
        observations.append({
            'match_id': mid,
            'date': played.isoformat(),
            'fit_before_date': fit.isoformat(),
            'forecast_before_date': forecast.isoformat(),
            'team_was_home': direction == 1,
            'actual_team_xg_net': actual_net,
            'earlier_expected_team_xg_net': expected_net,
            'team_xg_net_residual': actual_net - expected_net,
        })

    reasons = ([window_reason] if window_reason is not None else [])
    reasons.extend(sorted({item['reason'] for item in missing}))
    mean = None if reasons else math.fsum(item['team_xg_net_residual'] for item in observations) / 5
    return {
        'team': team,
        'reference_match_ids': ids,
        'expected_reference_count': 5,
        'reference_count': len(ids),
        'available_residual_count': len(observations),
        'unavailable_reasons': reasons,
        'missing_references': missing,
        'residual_mean': mean,
        'observations': observations,
    }


def build_adjusted_rows(rows, actual_xg_records):
    """Return all rows with copied 123-column side features and provenance.

    ``rows`` is the unfiltered saved ``features_lead3`` row list.  The source
    mapping is the first return value (``parsed``) of ``load_source()``; values
    expose match_id, match_date, home, away, competition, goals, and xg.

    Missing earlier forecasts or actual-xG records leave the entire five-game
    team mean as None, even when some references are usable.  Reference IDs
    are never replaced with older games.  A contradictory identity or unsafe
    time ordering raises ValueError.  None must be handled by the existing
    training-only imputer/missing indicators, without imputing from test rows.
    """
    rows = list(rows)
    index = {}
    feature_names = None
    for row in rows:
        mid = row['match_id']
        if mid in index:
            raise ValueError(f'Duplicate saved feature match ID: {mid}')
        index[mid] = row
        features = row['fresh_performance_features']
        if len(features) != 2:
            raise ValueError(f'Expected two side feature dictionaries: {mid}')
        for values in features:
            keys = set(values)
            if len(keys) != 121 or keys.intersection(EXTRA_FEATURE_NAMES):
                raise ValueError(f'Expected unchanged 121-column original features: {mid}')
            if feature_names is None:
                feature_names = keys
            elif keys != feature_names:
                raise ValueError(f'Original feature columns differ: {mid}')

    adjusted = []
    counts = Counter()
    reasons = Counter()
    by_period = defaultdict(Counter)
    for row in rows:
        histories = {side: _team_history(row, side, index, actual_xg_records)
                     for side in ('home', 'away')}
        home_mean, away_mean = (histories[side]['residual_mean'] for side in ('home', 'away'))
        derived = dict(row)
        derived['adjusted_performance_features'] = [
            dict(row['fresh_performance_features'][0], **{
                EXTRA_FEATURE_NAMES[0]: home_mean, EXTRA_FEATURE_NAMES[1]: away_mean}),
            dict(row['fresh_performance_features'][1], **{
                EXTRA_FEATURE_NAMES[0]: away_mean, EXTRA_FEATURE_NAMES[1]: home_mean}),
        ]
        derived['adjusted_xg_residual_history'] = {
            'forecast_before_date': row['forecast_before_date'],
            'history': histories,
        }
        adjusted.append(derived)
        available_sides = sum(histories[side]['residual_mean'] is not None for side in ('home', 'away'))
        label = ('neither_side_available', 'one_side_available', 'both_sides_available')[available_sides]
        counts[label] += 1
        counts['rows'] += 1
        by_period[row['period']][label] += 1
        by_period[row['period']]['rows'] += 1
        for history in histories.values():
            counts['team_windows'] += 1
            counts['available_team_windows'] += history['residual_mean'] is not None
            reasons.update(history['unavailable_reasons'])
            by_period[row['period']]['available_team_windows'] += history['residual_mean'] is not None

    report = {
        'input_rows': len(rows),
        'output_rows': len(adjusted),
        'dropped_rows': 0,
        'original_side_feature_count': 121,
        'adjusted_side_feature_count': 123,
        'added_feature_names': list(EXTRA_FEATURE_NAMES),
        'definition': 'Mean over exactly the existing last 5 fixtures of team-oriented '
                      '(actual xG net - that fixture saved earlier xG expected net)',
        'counts': dict(counts),
        'unavailable_team_window_reason_counts': dict(reasons),
        'by_period': {key: dict(value) for key, value in sorted(by_period.items())},
        'source_exclusions': 'No new exclusions. Caller retains load_source original source-difference reports.',
        'missing_handling': 'Keep None. Use the existing training-only imputer and missing indicators; '
                            'do not impute with test rows, zero, fewer fixtures, or regenerated old forecasts.',
        'limitation': 'Earliest saved season has more unknown windows because its earlier fixture '
                      'forecast cache is unavailable; this is missing information, not zero surprise.',
    }
    return adjusted, report
