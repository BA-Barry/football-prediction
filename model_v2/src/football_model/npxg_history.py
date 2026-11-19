"""Source non-penalty xG means, kept distinct from future full-goal rates.

Only source per-match team history is joined. No penalty subtraction, season
aggregate, current lineup, or same-match forecast is used as a predictor.
"""
from dataclasses import dataclass, field, replace
from datetime import date
import math

import numpy as np

from .xg_history import HistoricalXgMatch, DateTeamXgModel, join_understat


TOLERANCE = 1e-4


@dataclass
class HistoricalNpxgMatch(HistoricalXgMatch):
    npxg: tuple[float, float] = field(kw_only=True)


def _number(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'Boolean is not a measurement: {label}')
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'Invalid numeric measurement: {label}') from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f'Need a finite nonnegative measurement: {label}')
    return result


def _identifier(value):
    if isinstance(value, (bool, np.bool_)) or not str(value).isdecimal():
        raise ValueError('Need an explicit provider team ID')
    return str(value)


def join_understat_npxg(payload, football_data_records, season, competition='E0'):
    """Keep the original independent join, then exactly join both NP histories.

    A team history row has no fixture ID. Identity therefore requires the
    provider team ID, full unmodified datetime text, and home/away side. Both
    views must agree with the fixture xG and each other's non-penalty xG.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get('dates'), list):
        raise ValueError('Need a source payload with individual dates')
    teams = payload.get('teams')
    if not isinstance(teams, dict):
        raise ValueError('Need source per-match team histories')
    originals, report = join_understat(payload['dates'], football_data_records, season, competition)
    rows = {str(row['id']): row for row in payload['dates']}
    # Index only identity fields here. Other games' measurements are read only
    # if that exact independently joined fixture needs them.
    history_index = {}
    for team_id, team in teams.items():
        if not isinstance(team, dict) or not isinstance(team.get('history'), list):
            raise ValueError('Need individual history arrays for source teams')
        for entry in team['history']:
            if not isinstance(entry, dict):
                raise ValueError('Invalid source team history row')
            key = (str(team_id), entry.get('date'), entry.get('h_a'))
            history_index.setdefault(key, []).append(entry)
    parsed = []
    for original in originals:
        row = rows[original.understat_id]
        values, against = [], []
        for position, side in enumerate(('h', 'a')):
            try:
                team_id = _identifier(row[side]['id'])
                team = teams[team_id]
                if _identifier(team['id']) != team_id or team['title'] != row[side]['title']:
                    raise ValueError('Source team ID or title disagrees with fixture')
                candidates = history_index.get((team_id, row['datetime'], side), [])
                if len(candidates) != 1:
                    raise ValueError('Need one exact provider datetime/side team history')
                history = candidates[0]
                xg = _number(history['xG'], 'history xG')
                xga = _number(history['xGA'], 'history xGA')
                np_xg = _number(history['npxG'], 'history npxG')
                np_xga = _number(history['npxGA'], 'history npxGA')
            except (KeyError, TypeError) as error:
                raise ValueError('Missing source team identity or per-match xG/NPxG') from error
            if (abs(xg-original.xg[position]) > TOLERANCE
                    or abs(xga-original.xg[1-position]) > TOLERANCE):
                raise ValueError('Per-team history xG disagrees with source fixture')
            if np_xg > xg+TOLERANCE or np_xga > xga+TOLERANCE:
                raise ValueError('Non-penalty xG exceeds corresponding total xG')
            values.append(np_xg)
            against.append(np_xga)
        if any(abs(values[i]-against[1-i]) > TOLERANCE for i in (0, 1)):
            raise ValueError('Opposing non-penalty xG histories disagree')
        parsed.append(HistoricalNpxgMatch(**vars(original), npxg=tuple(values)))
    report = {**report,
              'npxg_exact_matches': len(parsed),
              'npxg_source': 'teams[provider_team_id].history exact date/h_a',
              'team_history_fields_used': ['date', 'h_a', 'xG', 'xGA', 'npxG', 'npxGA'],
              'cross_view_tolerance': TOLERANCE,
              'ignored_columns': ['forecast', 'players', 'rosters', 'season aggregates',
                                  'team history fields other than the six listed'],
              'availability': 'calendar-date research only; original publication times and revisions unknown'}
    return parsed, report


class DateTeamNpxgModel:
    """Same weighted quasi-loss as total xG, returning an NP mean only.

    A caller wishing to predict all goals must explicitly provide and evaluate
    a conversion learned from earlier data. This class never treats NP mean
    as an already calibrated all-goal rate.
    """
    def __init__(self, regularization=.01, half_life_days=180):
        self._model = DateTeamXgModel(regularization, half_life_days)
        self.regularization = regularization
        self.half_life_days = half_life_days

    def fit(self, records, before_date):
        if type(before_date) is not date:
            raise ValueError('Use a calendar cutoff')
        history = [record for record in records if record.match_date < before_date]
        targets = []
        for record in history:
            raw = np.asarray(record.npxg, dtype=object)
            if raw.shape != (2,):
                raise ValueError('Need two non-penalty xG values')
            values = tuple(_number(v, 'npxG') for v in raw)
            totals = np.asarray(record.xg, dtype=object)
            if totals.shape != (2,):
                raise ValueError('Need two source total xG values')
            if any(np_xg > _number(total, 'source total xG')+TOLERANCE
                   for np_xg, total in zip(values, totals)):
                raise ValueError('Non-penalty xG exceeds corresponding total xG')
            targets.append(replace(record, xg=values))
        self._model.fit(targets, before_date)
        self.teams = set(self._model.teams)
        self.before_date = self._model.before_date
        self.competition = self._model.competition
        return self

    def predict(self, home, away, competition, match_date):
        return self._model.predict(home, away, competition, match_date)

    def parameters(self):
        result = self._model.parameters()
        result.update(model_type='date_level_fractional_npxg_quasi_poisson',
                      response='provider_match_npxg',
                      source_total_xg_preserved=True,
                      rate_use='predicted non-penalty xG mean only; not a full-goal rate without an explicit earlier-data conversion')
        return result
