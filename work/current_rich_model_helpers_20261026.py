"""Reuse existing rich-model research with independently checked current inputs.

No download, experiment loop, parameter search, or write happens on import.
Historical feature snapshots are retained verbatim when fitting the candidate.
"""
from collections import Counter
from datetime import date, datetime
import gzip
import json
import math
from pathlib import Path

from football_model.all_competition_features import AllCompetitionFeatures
from football_model.early_goal_learners import EarlyGoalLearner
from football_model.early_history_features import EarlyTeamObservation
from football_model.npxg_history import join_understat_npxg
from run_broad_early_models_20261025 import observations as historical_observations
from run_npxg_comparison_20261025 import ROOT, load_source


TODAY = date(2026, 10, 26)
HISTORICAL_FEATURES = (
    ROOT / 'outputs/固定发布日期信息刷新_20261025T071806561562Z/features_lead3.json.gz'
)
HISTORICAL_SCHEDULE = ROOT / 'outputs/全赛事赛程可用性_20261025'


def _payload(value):
    if isinstance(value, (str, Path)):
        return json.loads(Path(value).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Need the parsed Understat payload or its local JSON path')
    return value


def _date_text(text):
    return datetime.strptime(text, '%Y-%m-%d %H:%M:%S').date()


def _measurement(value, label):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f'Boolean is not a measurement: {label}')
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f'Need a finite nonnegative measurement: {label}')
    return result


def _current_observations(records, payload):
    fixtures = {str(row['id']): row for row in payload['dates']}
    histories = {}
    for team_id, team in payload['teams'].items():
        for row in team['history']:
            key = (str(team_id), row['date'], row['h_a'])
            if key in histories:
                raise ValueError(f'Duplicate current team/date/side history: {key}')
            histories[key] = row
    result, missing = [], []
    for record in records:
        fixture = fixtures[record.understat_id]
        for index, side in enumerate(('h', 'a')):
            key = (str(fixture[side]['id']), fixture['datetime'], side)
            if key not in histories:
                raise ValueError(f'Current match lacks its exact team history: {key}')
            source = histories[key]
            optional = {}
            for field, container, key_name in (
                ('ppda_att', 'ppda', 'att'), ('ppda_def', 'ppda', 'def'),
                ('ppda_allowed_att', 'ppda_allowed', 'att'),
                ('ppda_allowed_def', 'ppda_allowed', 'def'),
                ('deep_for', None, 'deep'), ('deep_against', None, 'deep_allowed'),
            ):
                raw = source.get(key_name) if container is None else (source.get(container) or {}).get(key_name)
                optional[field] = _measurement(raw, field)
                if optional[field] is None:
                    missing.append({'match_id': record.match_id, 'side': side, 'field': field})
            result.append(EarlyTeamObservation(
                match_id=record.match_id, match_date=record.match_date,
                team=(record.home, record.away)[index],
                opponent=(record.away, record.home)[index],
                competition=record.competition, is_home=(side == 'h'),
                goals_for=record.goals[index], goals_against=record.goals[1-index],
                xg_for=record.xg[index], xg_against=record.xg[1-index],
                npxg_for=record.npxg[index], npxg_against=record.npxg[1-index],
                **optional,
            ))
    return result, missing


def load_current_rich_sources(current_payload, current_independent_records, *,
                              before_date=TODAY, season='2026/27', competition='E0'):
    """Return historical plus current NP records, FD records and team histories.

    ``current_independent_records`` contains HistoricalMatch objects. Official
    FPL supplements must already use the Football-Data canonical team identities,
    independent fixture date and verified full-time score. A stale FD-only input
    is rejected when recent completed Understat matches have no independent pair.
    No nearest-date join, fabricated missing measurements or silent drop occurs.
    """
    if type(before_date) is not date:
        raise ValueError('Need an explicit calendar-date cutoff')
    payload = _payload(current_payload)
    completed, excluded_after_date, unplayed = [], [], 0
    for row in payload['dates']:
        if row.get('isResult') is True:
            if _date_text(row['datetime']) < before_date:
                completed.append(row)
            else:
                excluded_after_date.append(str(row['id']))
        elif row.get('isResult') is False:
            unplayed += 1
        else:
            raise ValueError('Every current source fixture needs an explicit result flag')
    eligible, input_outside_scope = [], []
    for record in current_independent_records:
        if (record.competition == competition and record.season == season
                and record.match_date < before_date):
            eligible.append(record)
        else:
            input_outside_scope.append(record.match_id)
    if len({r.match_id for r in eligible}) != len(eligible):
        raise ValueError('Duplicate independent current fixture ID')
    scoped_payload = {**payload, 'dates': completed}
    current, current_report = join_understat_npxg(scoped_payload, eligible, season, competition)
    if (current_report['missing_independent_match_ids']
            or current_report['missing_special_match_ids']):
        raise ValueError('Current independent fixtures lack a source match: '
                         + json.dumps(current_report, ensure_ascii=False))
    exclusions = current_report['excluded_special_matches']
    if len(completed) != len(current) + len(exclusions):
        raise ValueError('Every earlier completed source match must be joined or explicitly excluded')
    fresh_observations, missing = _current_observations(current, scoped_payload)
    np_records, independent, joins, fd_reports, directories = load_source()
    old_observations = historical_observations(np_records)
    for record in eligible:
        if record.match_id in independent:
            raise ValueError(f'Current fixture duplicates historical independent ID: {record.match_id}')
        independent[record.match_id] = record
    for record in current:
        if record.match_id in np_records:
            raise ValueError(f'Current fixture duplicates historical source ID: {record.match_id}')
        np_records[record.match_id] = record
    historical_exclusions = {
        key: {name: report.get(name, []) for name in (
            'excluded_source_ids', 'excluded_independent_ids', 'excluded_special_matches')}
        for key, report in joins.items()
    }
    source_report = {
        'before_date': before_date.isoformat(), 'competition': competition, 'season': season,
        'current_completed_source_n': len(completed), 'current_independent_n': len(eligible),
        'current_npxg_n': len(current), 'current_observation_n': len(fresh_observations),
        'current_date_min': min((r.match_date.isoformat() for r in current), default=None),
        'current_date_max': max((r.match_date.isoformat() for r in current), default=None),
        'unplayed_source_n_ignored': unplayed,
        'completed_on_or_after_cutoff_ignored': excluded_after_date,
        'independent_outside_requested_scope': input_outside_scope,
        'current_join': current_report, 'current_optional_measurements_missing': missing,
        'historical_join_reports': joins,
        'historical_source_difference_exclusions': historical_exclusions,
        'historical_fd_reports': fd_reports, 'historical_fd_directories': directories,
        'source_first_publication_or_revision_reconstructed': False,
    }
    return {
        'np_records': np_records, 'independent': independent,
        'observations': old_observations + fresh_observations,
        'current_np_records': current, 'current_independent_records': eligible,
        'reports': source_report,
    }


def train_current_candidate(features_path=HISTORICAL_FEATURES, *, before_date=TODAY,
                            feature_key='fresh_allcomp_features'):
    """Fit exactly one existing ExtraTrees preset to intact lead3 snapshots.

    This does not add current-season labelled training examples, select a model,
    score historical targets, or assert validation at the new publication lead.
    The returned learner can predict PreparedPredictionFeatures allcomp pairs.
    """
    if type(before_date) is not date:
        raise ValueError('Need an explicit calendar-date cutoff')
    with gzip.open(features_path, 'rt', encoding='utf-8') as handle:
        rows = json.load(handle)['rows']
    if feature_key not in ('fresh_allcomp_features','fresh_performance_features'):
        raise ValueError('Only existing research feature sets are supported')
    expected=161 if feature_key=='fresh_allcomp_features' else 121
    training = []
    for row in rows:
        if date.fromisoformat(row['date']) >= before_date:
            continue
        parameter = date.fromisoformat(row['fit_before_date'])
        forecast = date.fromisoformat(row['forecast_before_date'])
        played = date.fromisoformat(row['date'])
        if not parameter <= forecast < played or row['horizon_days'] != 3:
            raise ValueError('Saved training row violates its own earlier publication timing')
        pairs = row[feature_key]
        if len(pairs) != 2 or any(len(side) != expected for side in pairs):
            raise ValueError('Need the existing fixed-horizon feature schema')
        training.append({**row, 'side_features': pairs})
    model = EarlyGoalLearner('extra_trees', 'rich').fit(training, before_date)
    metadata = model.parameters() | {
        'source_feature_file': str(Path(features_path)),
        'source_feature_key': feature_key,
        'historical_training_horizon_days': 3,
        'historical_training_period_counts': dict(Counter(r['period'] for r in training)),
        'historical_snapshots_recomputed_with_current_information': False,
        'current_season_labels_added_to_meta_training': False,
        'current_initial_fit_before_date': before_date.isoformat(),
        'applicability_note': (
            'Prospective research candidate: historical inputs were published three calendar days '
            'before fixtures with 28-day parameters. Current forecasts may have a different '
            'publication lead and use an initial parameter fit on the publication day. Neither '
            'change has been established by the historical lead3 comparison.'
        ),
        'new_parameter_grid': False,
    }
    return model, metadata


def load_historical_schedule(directory=HISTORICAL_SCHEDULE):
    """Load the existing schedule builder for explicit extension by orchestration.

    Existing requested member-season coverage ends in July 2026. It cannot claim
    current workload is known. Add current competitive ScheduleClub events and
    real coverage, or retain the resulting unknown values and report the gap.
    """
    return AllCompetitionFeatures.from_directory(directory)
