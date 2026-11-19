"""Compose date-level feature snapshots for an explicit forecast publication day.

Model parameter dates and feature dates are separate. All input measurements
must still be supplied from eligible histories; source publication/revision
times are not reconstructed by this date-only composition.
"""
from datetime import date, timedelta
import math
from numbers import Real

from .early_history_features import (
    EarlyHistoryFeatures, MEASUREMENTS, TREND_MEASUREMENTS, WINDOWS,
)


COMPETITIONS = ('E0', 'D1', 'SP1', 'I1', 'F1')
OMITTED_LEAGUE_DATE_SUFFIXES = (
    'cutoff_staleness_days', 'previous_league_interval_days',
    'known_league_matches_before_cutoff_7d',
    'known_league_matches_before_cutoff_14d',
    'known_league_matches_before_cutoff_28d',
    'same_league_gap_to_match_days',
)


def _performance_history_names():
    names = ['history_count']
    for window in WINDOWS:
        names.append(f'history_count_{window}')
        names.extend(f'{measurement}_mean_{window}' for measurement in MEASUREMENTS)
        names.extend(f'{measurement}_{window}' for measurement in ('ppda_ratio', 'ppda_allowed_ratio'))
    names.extend(f'trend_{measurement}_5_minus_15' for measurement in TREND_MEASUREMENTS)
    names.extend(f'trend_{measurement}_5_minus_15' for measurement in ('ppda_ratio', 'ppda_allowed_ratio'))
    names.append('venue_count_15')
    names.extend(f'venue_{measurement}_mean_15' for measurement in MEASUREMENTS)
    return ('is_home', *(f'{side}_{name}' for side in ('own', 'opponent') for name in names))


PERFORMANCE_HISTORY_NAMES = _performance_history_names()


def derive_timing(match_date, lead_days, *, season_start_year=None):
    """Return D, F and the Aug-1 anchored 28-day parameter date P <= F.

    Explicit season_start_year is preferred when a caller has a season label.
    Without one, the usual July-to-June member-season scope is used. Floor
    division intentionally permits July dates before that season's Aug 1.
    """
    if type(match_date) is not date or type(lead_days) is not int or lead_days < 1:
        raise ValueError('Need a calendar target date and a positive integer lead')
    if season_start_year is None:
        season_start_year = match_date.year if match_date.month >= 7 else match_date.year-1
    if type(season_start_year) is not int:
        raise ValueError('Need an explicit calendar season start year')
    anchor = date(season_start_year, 8, 1)
    forecast = match_date-timedelta(days=lead_days)
    parameter = anchor+timedelta(days=((forecast-anchor).days//28)*28)
    return {'match_date': match_date, 'forecast_cutoff': forecast,
            'parameter_cutoff': parameter, 'lead_days': lead_days,
            'anchor_date': anchor}


def _rates(values, label):
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise ValueError(f'Need two earlier positive {label} means')
    if any(isinstance(v, bool) or not isinstance(v, Real)
           or not math.isfinite(float(v)) or v <= 0 for v in values):
        raise ValueError(f'Need two earlier positive finite {label} means')
    return tuple(float(v) for v in values)


class PreparedPredictionFeatures:
    """Prepare matched rich feature sets without receiving a target outcome.

    Both rich sets share performance measurements, publication/parameter age
    information, and basic goal/xG means. The all-competition set adds only its
    40 own/opponent schedule measurements. Unknown league histories stay
    missing while the known basic means and home indicator remain available.
    """

    def __init__(self, observations, schedule_builder):
        self.league = EarlyHistoryFeatures(observations)
        self.schedule = schedule_builder

    def for_match(self, *, home, away, competition, parameter_cutoff,
                  forecast_cutoff, match_date, goal_rates, xg_rates, refresh):
        if (any(type(value) is not date for value in (parameter_cutoff, forecast_cutoff, match_date))
                or not parameter_cutoff <= forecast_cutoff < match_date):
            raise ValueError('Parameter date must be at or before publication date, before the match')
        return self._compose(
            home=home, away=away, competition=competition,
            parameter_cutoff=parameter_cutoff, forecast_cutoff=forecast_cutoff,
            match_date=match_date, goal_rates=goal_rates, xg_rates=xg_rates, refresh=refresh)

    def for_same_day_match(self, *, home, away, competition, parameter_cutoff,
                           forecast_cutoff, match_date, goal_rates, xg_rates, refresh):
        """Prepare a match-day snapshot using strictly earlier calendar dates.

        This deliberately excludes even completed fixtures on the target day.
        The caller must separately establish that a live target has not started;
        calendar dates cannot establish its current kickoff/status or historical
        source publication times. Existing positive-lead calls retain their
        stricter publication-before-match-day requirement in ``for_match``.
        """
        if (any(type(value) is not date for value in (parameter_cutoff, forecast_cutoff, match_date))
                or forecast_cutoff != match_date or parameter_cutoff > forecast_cutoff):
            raise ValueError('Same-day publication must equal the match date, with parameter date no later')
        return self._compose(
            home=home, away=away, competition=competition,
            parameter_cutoff=parameter_cutoff, forecast_cutoff=forecast_cutoff,
            match_date=match_date, goal_rates=goal_rates, xg_rates=xg_rates, refresh=refresh)

    def _compose(self, *, home, away, competition, parameter_cutoff,
                 forecast_cutoff, match_date, goal_rates, xg_rates, refresh):
        if type(refresh) is not bool:
            raise ValueError('Choose explicitly whether historical inputs refresh')
        if (competition not in COMPETITIONS or not isinstance(home, str) or not home
                or not isinstance(away, str) or not away or home == away):
            raise ValueError('Need two distinct explicit teams in a supported competition')
        goals, xg = _rates(goal_rates, 'goal'), _rates(xg_rates, 'xG')
        feature_cutoff = forecast_cutoff if refresh else parameter_cutoff
        kwargs = dict(home=home, away=away, competition=competition,
                      cutoff=feature_cutoff, match_date=match_date)
        history = self.league.for_match(**kwargs)
        load = self.schedule.for_match(**kwargs)
        feature_lead = math.log1p((match_date-feature_cutoff).days)
        parameter_age = math.log1p((forecast_cutoff-parameter_cutoff).days)
        performance, allcomp = [], []
        for index, side in enumerate(('home', 'away')):
            values = history[f'{side}_features']
            rich = {name: None if values is None else values[name]
                    for name in PERFORMANCE_HISTORY_NAMES}
            basic = {**{f'league_{comp}': float(competition == comp) for comp in COMPETITIONS},
                'is_home': float(index == 0),
                'log_goal_mean_for': math.log(goals[index]),
                'log_goal_mean_against': math.log(goals[1-index]),
                'log_xg_mean_for': math.log(xg[index]),
                'log_xg_mean_against': math.log(xg[1-index])}
            rich.update(basic)
            rich.update(history_features_available=float(values is not None),
                        forecast_log1p_lead_days=feature_lead,
                        parameter_log1p_age_at_forecast_days=parameter_age)
            extra = dict(load[f'{side}_features'])
            if extra.pop('allcomp_log1p_forecast_lead_days') != feature_lead:
                raise ValueError('Schedule and performance snapshots must share the same date')
            if set(extra) & set(rich):
                raise ValueError('Schedule measurements must keep their own feature names')
            performance.append(rich)
            allcomp.append({**rich, **extra})
        return {'performance_features': performance, 'allcomp_features': allcomp,
            'feature_history': history['metadata'], 'allcomp_history': load['metadata'],
            'feature_before_date': feature_cutoff.isoformat(),
            'parameter_fit_before_date': parameter_cutoff.isoformat(),
            'forecast_before_date': forecast_cutoff.isoformat(),
            'timing': {'refresh': refresh, 'parameter_before_date': parameter_cutoff.isoformat(),
                'forecast_before_date': forecast_cutoff.isoformat(),
                'feature_before_date': feature_cutoff.isoformat(), 'match_date': match_date.isoformat(),
                'publication_lead_days': (match_date-forecast_cutoff).days,
                'input_lead_days': (match_date-feature_cutoff).days,
                'parameter_age_at_publication_days': (forecast_cutoff-parameter_cutoff).days,
                'assumption': 'strictly earlier calendar dates; exact historical publication/revision time unknown'}}
