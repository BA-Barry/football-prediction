"""Multivariate historical league snapshots available before a calendar cutoff.

These are descriptive trailing measurements, not causal effects, opponent-
adjusted strengths, lineup estimates, or complete cross-competition workload.
The caller supplies independently joined observations. Source publication times
and revisions are not reconstructed by this calendar-date feature builder.
"""
from bisect import bisect_left
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, timedelta
import math


@dataclass
class EarlyTeamObservation:
    match_id: str
    match_date: date
    team: str
    opponent: str
    competition: str
    is_home: bool
    goals_for: int
    goals_against: int
    xg_for: float
    xg_against: float
    npxg_for: float
    npxg_against: float
    ppda_att: float | None
    ppda_def: float | None
    ppda_allowed_att: float | None
    ppda_allowed_def: float | None
    deep_for: float | None
    deep_against: float | None


MEASUREMENTS = (
    'goals_for', 'goals_against', 'xg_for', 'xg_against',
    'npxg_for', 'npxg_against', 'deep_for', 'deep_against',
    'ppda_att', 'ppda_def', 'ppda_allowed_att', 'ppda_allowed_def',
)
TREND_MEASUREMENTS = MEASUREMENTS[:8]
OPTIONAL_MEASUREMENTS = set(MEASUREMENTS[6:])
WINDOWS = (5, 15)


def _measurement(value, label):
    if value is None and label in OPTIONAL_MEASUREMENTS:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'Need a numeric historical measurement: {label}')
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f'Need a finite nonnegative historical measurement: {label}')
    if label.startswith('goals_') and not result.is_integer():
        raise ValueError('Historical goals must be integer counts')
    return result


def _mean(values):
    # Missing observations are retained rather than skipped to reach older data.
    return None if not values or any(v is None for v in values) else math.fsum(values)/len(values)


def _ratio(numerators, denominators):
    if any(v is None for v in (*numerators, *denominators)):
        return None
    denominator = math.fsum(denominators)
    return math.fsum(numerators)/denominator if denominator > 0 else None


class EarlyHistoryFeatures:
    """Cache strictly earlier team snapshots across many target fixtures.

    `mean_5` and `mean_15` use at most the last 5/15 league matches, retaining
    their actual counts. Relevant-venue features use the matching venue subset
    of the same last-15 window; fewer than three such observations remain
    missing. PPDA is sum(att)/sum(def), never the mean of individual ratios.
    A zero sum(def) remains None. No missing value is filled with zero.

    Treat supplied observation objects as stable during this instance's life.
    Only identity/date fields are indexed at construction; measurements are
    validated and read only when their strictly earlier rows enter a snapshot.
    """

    def __init__(self, observations, min_history=5):
        if type(min_history) is not int or not 1 <= min_history <= WINDOWS[0]:
            raise ValueError('Need a minimum history between one and five')
        self.min_history = min_history
        self._rows, self._dates, self._cache = {}, {}, {}
        for row in observations:
            if type(row.match_date) is not date:
                raise ValueError('Use historical calendar dates')
            if (not isinstance(row.match_id, str) or not row.match_id
                    or not all(isinstance(v, str) and v for v in (row.team, row.opponent, row.competition))
                    or row.team == row.opponent or type(row.is_home) is not bool):
                raise ValueError('Need explicit distinct team identities and venue')
            self._rows.setdefault((row.competition, row.team), []).append(row)
        for key, rows in self._rows.items():
            rows.sort(key=lambda r: (r.match_date, r.match_id))
            self._dates[key] = [r.match_date for r in rows]

    def _snapshot(self, competition, team, cutoff):
        key = (competition, team, cutoff)
        if key in self._cache:
            return self._cache[key]
        team_key = (competition, team)
        rows = self._rows.get(team_key, [])
        stop = bisect_left(self._dates.get(team_key, []), cutoff)
        history = rows[:stop]
        identifiers = [r.match_id for r in history]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError('Duplicate earlier team/match observation')
        recent = history[-WINDOWS[-1]:]
        numbers = []
        for row in recent:
            values = {name: _measurement(getattr(row, name), name) for name in MEASUREMENTS}
            if any(values['npxg_'+direction] > values['xg_'+direction]+1e-4
                   for direction in ('for', 'against')):
                raise ValueError('Historical non-penalty xG exceeds total xG')
            numbers.append(values)
        features = {'history_count': float(len(history)),
                    'cutoff_staleness_days': float((cutoff-history[-1].match_date).days) if history else None,
                    'previous_league_interval_days': float((history[-1].match_date-history[-2].match_date).days)
                        if len(history) >= 2 else None}
        for days in (7, 14, 28):
            start = bisect_left(self._dates.get(team_key, []), cutoff-timedelta(days=days), hi=stop)
            features[f'known_league_matches_before_cutoff_{days}d'] = float(stop-start)
        for window in WINDOWS:
            selected = numbers[-window:]
            features[f'history_count_{window}'] = float(len(selected))
            for name in MEASUREMENTS:
                features[f'{name}_mean_{window}'] = _mean([v[name] for v in selected])
            for output, att, defense in (
                ('ppda_ratio', 'ppda_att', 'ppda_def'),
                ('ppda_allowed_ratio', 'ppda_allowed_att', 'ppda_allowed_def'),
            ):
                features[f'{output}_{window}'] = _ratio([v[att] for v in selected],
                                                        [v[defense] for v in selected]) if selected else None
        for name in TREND_MEASUREMENTS:
            short, medium = features[f'{name}_mean_5'], features[f'{name}_mean_15']
            features[f'trend_{name}_5_minus_15'] = None if short is None or medium is None else short-medium
        for name in ('ppda_ratio', 'ppda_allowed_ratio'):
            short, medium = features[f'{name}_5'], features[f'{name}_15']
            features[f'trend_{name}_5_minus_15'] = None if short is None or medium is None else short-medium
        venue_features = {}
        for is_home in (True, False):
            values = [v for row, v in zip(recent, numbers) if row.is_home == is_home]
            venue = {'venue_count_15': float(len(values))}
            for name in MEASUREMENTS:
                venue[f'venue_{name}_mean_15'] = _mean([v[name] for v in values]) if len(values) >= 3 else None
            venue_features[is_home] = venue
        metadata = {
            'team': team, 'competition': competition, 'history_count': len(history),
            'used_match_ids': [r.match_id for r in recent],
            'earliest_used_date': recent[0].match_date.isoformat() if recent else None,
            'latest_used_date': recent[-1].match_date.isoformat() if recent else None,
            'unavailable_reason': 'insufficient_league_history' if len(history) < self.min_history else None,
            'window_match_ids': {str(w): [r.match_id for r in recent[-w:]] for w in WINDOWS},
        }
        snapshot = (features, venue_features, metadata)
        self._cache[key] = snapshot
        return snapshot

    def for_match(self, *, home, away, competition, cutoff, match_date):
        if type(cutoff) is not date or type(match_date) is not date or cutoff > match_date:
            raise ValueError('Need a calendar cutoff at or before the target match')
        if home == away or not all(isinstance(v, str) and v for v in (home, away, competition)):
            raise ValueError('Need distinct team identities and competition')
        home_summary, home_venues, home_meta = self._snapshot(competition, home, cutoff)
        away_summary, away_venues, away_meta = self._snapshot(competition, away, cutoff)
        metadata = {
            'cutoff': cutoff.isoformat(), 'match_date': match_date.isoformat(), 'competition': competition,
            'history': {'home': deepcopy(home_meta), 'away': deepcopy(away_meta)},
            'windows': list(WINDOWS), 'minimum_history': self.min_history,
            'venue_minimum': 3, 'missing_policy': 'retain None; never zero-fill or skip a missing window row',
            'clock_basis': 'only rows on strictly earlier calendar dates than cutoff',
            'schedule_scope': 'same-league observed matches only; gaps are not all-competition rest',
            'strength_adjustment': 'descriptive histories, not historical-opponent-adjusted effects',
        }
        if any(v['unavailable_reason'] for v in (home_meta, away_meta)):
            return {'home_features': None, 'away_features': None, 'metadata': metadata}

        def side_features(own, other, own_venue, other_venue, is_home, own_meta, other_meta):
            own_values, other_values = {**own, **own_venue}, {**other, **other_venue}
            for values, meta in ((own_values, own_meta), (other_values, other_meta)):
                latest = date.fromisoformat(meta['latest_used_date'])
                values['same_league_gap_to_match_days'] = float((match_date-latest).days)
            return {'is_home': float(is_home),
                    **{f'own_{k}': v for k, v in own_values.items()},
                    **{f'opponent_{k}': v for k, v in other_values.items()}}

        home_features = side_features(home_summary, away_summary, home_venues[True], away_venues[False],
                                      True, home_meta, away_meta)
        away_features = side_features(away_summary, home_summary, away_venues[False], home_venues[True],
                                      False, away_meta, home_meta)
        return {'home_features': home_features, 'away_features': away_features, 'metadata': metadata}


def build_early_features(observations, *, home, away, competition, cutoff, match_date, min_history=5):
    return EarlyHistoryFeatures(observations, min_history=min_history).for_match(
        home=home, away=away, competition=competition, cutoff=cutoff, match_date=match_date)
