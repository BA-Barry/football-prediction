"""Earlier match-distribution descriptions, distinct from trailing means.

Provider draw probabilities describe already played matches, not the target
fixture's prematch probability. Only observations before the supplied calendar
cutoff are read. Historical initial publication times remain an external data
availability assumption, as with the existing early-history features.
"""
from bisect import bisect_left
from collections.abc import Mapping
from copy import deepcopy
from datetime import date
import math
from numbers import Real


WINDOWS = (5, 15)
MIN_HISTORY = 5
METRICS = (
    'mean_abs_npxg_difference',
    'npxg_total_population_variance',
    'npxg_for_against_population_covariance',
    'actual_draw_fraction',
    'provider_draw_probability_mean',
    'provider_draw_probability_population_variance',
)
FEATURE_NAMES = tuple(f'pattern_{role}_{metric}_{window}'
                      for role in ('own', 'opponent')
                      for window in WINDOWS for metric in METRICS)


def _number(value, label, *, integer=False, probability=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'Need a numeric historical measurement: {label}')
    result = float(value)
    if (not math.isfinite(result) or result < 0
            or integer and not result.is_integer()
            or probability and result > 1):
        raise ValueError(f'Invalid historical measurement: {label}')
    return result


def _mean(values):
    return math.fsum(values)/len(values) if values else None


def _population_variance(values):
    if not values:
        return None
    center = _mean(values)
    return math.fsum((value-center)**2 for value in values)/len(values)


def _population_covariance(first, second):
    if not first:
        return None
    first_mean, second_mean = _mean(first), _mean(second)
    return math.fsum((a-first_mean)*(b-second_mean)
                     for a, b in zip(first, second))/len(first)


class MatchPatternFeatures:
    """Build 24 home-perspective match features using last 5/15 league games.

    ``observations`` are existing ``EarlyTeamObservation`` objects. Provider
    probabilities are a mapping ``(competition, match_id) -> float | None``.
    Both inputs must remain unchanged during this instance's life. Construction
    indexes identities and dates only; selected earlier measurements are
    validated lazily. No target or future probability/score is inspected.

    Each team needs five earlier games. A 15-game window with 5--14 games uses
    its actual count. Missing provider probabilities leave both provider
    summaries missing for that window, without changing its other four values.
    These are descriptive patterns, not opponent-adjusted causal effects or an
    estimate of conditional score dispersion.
    """

    def __init__(self, observations, provider_draw_probabilities):
        if not isinstance(provider_draw_probabilities, Mapping):
            raise ValueError('Need a mapping of historical provider draw probabilities')
        self._provider = provider_draw_probabilities
        self._rows, self._dates, self._cache = {}, {}, {}
        for row in observations:
            if type(row.match_date) is not date:
                raise ValueError('Use historical calendar dates')
            if (not isinstance(row.match_id, str) or not row.match_id
                    or not all(isinstance(value, str) and value
                               for value in (row.team, row.opponent, row.competition))
                    or row.team == row.opponent or type(row.is_home) is not bool):
                raise ValueError('Need explicit distinct team identities and venue')
            self._rows.setdefault((row.competition, row.team), []).append(row)
        for key, rows in self._rows.items():
            rows.sort(key=lambda row: (row.match_date, row.match_id))
            self._dates[key] = [row.match_date for row in rows]

    def _snapshot(self, competition, team, cutoff):
        cache_key = competition, team, cutoff
        if cache_key in self._cache:
            return self._cache[cache_key]
        team_key = competition, team
        rows = self._rows.get(team_key, [])
        stop = bisect_left(self._dates.get(team_key, []), cutoff)
        history = rows[:stop]
        if len({row.match_id for row in history}) != len(history):
            raise ValueError('Duplicate earlier team/match observation')
        recent = history[-WINDOWS[-1]:]
        values = []
        for row in recent:
            goals_for = _number(row.goals_for, 'goals_for', integer=True)
            goals_against = _number(row.goals_against, 'goals_against', integer=True)
            probability = self._provider.get((competition, row.match_id))
            if probability is not None:
                probability = _number(probability, 'provider_draw_probability', probability=True)
            values.append({
                'npxg_for': _number(row.npxg_for, 'npxg_for'),
                'npxg_against': _number(row.npxg_against, 'npxg_against'),
                'actual_draw': float(goals_for == goals_against),
                'provider_draw_probability': probability,
            })
        features, windows = {}, {}
        for window in WINDOWS:
            selected, selected_values = recent[-window:], values[-window:]
            first = [value['npxg_for'] for value in selected_values]
            second = [value['npxg_against'] for value in selected_values]
            probabilities = [value['provider_draw_probability'] for value in selected_values]
            available = sum(value is not None for value in probabilities)
            complete = bool(probabilities) and available == len(probabilities)
            metrics = {
                'mean_abs_npxg_difference': _mean([abs(a-b) for a, b in zip(first, second)]),
                'npxg_total_population_variance': _population_variance(
                    [a+b for a, b in zip(first, second)]),
                'npxg_for_against_population_covariance': _population_covariance(first, second),
                'actual_draw_fraction': _mean([value['actual_draw'] for value in selected_values]),
                'provider_draw_probability_mean': _mean(probabilities) if complete else None,
                'provider_draw_probability_population_variance': (
                    _population_variance(probabilities) if complete else None),
            }
            features.update({f'{metric}_{window}': metrics[metric] for metric in METRICS})
            windows[str(window)] = {
                'count': len(selected),
                'match_ids': [row.match_id for row in selected],
                'dates': [row.match_date.isoformat() for row in selected],
                'provider_available_count': available,
                'provider_missing_count': len(selected)-available,
                'provider_missing_match_ids': [row.match_id for row, value in
                    zip(selected, selected_values) if value['provider_draw_probability'] is None],
            }
        metadata = {
            'team': team, 'competition': competition, 'history_count': len(history),
            'windows': windows,
            'unavailable_reason': ('insufficient_league_history'
                                   if len(history) < MIN_HISTORY else None),
        }
        self._cache[cache_key] = features, metadata
        return self._cache[cache_key]

    def for_match(self, *, home, away, competition, cutoff, match_date):
        if type(cutoff) is not date or type(match_date) is not date or cutoff > match_date:
            raise ValueError('Need a calendar cutoff at or before the target match')
        if (home == away or not all(isinstance(value, str) and value
                                   for value in (home, away, competition))):
            raise ValueError('Need distinct team identities and competition')
        own, own_meta = self._snapshot(competition, home, cutoff)
        opponent, opponent_meta = self._snapshot(competition, away, cutoff)
        metadata = {
            'cutoff': cutoff.isoformat(), 'match_date': match_date.isoformat(),
            'competition': competition,
            'history': {'home': deepcopy(own_meta), 'away': deepcopy(opponent_meta)},
            'windows': list(WINDOWS), 'minimum_history': MIN_HISTORY,
            'clock_basis': 'only rows on strictly earlier calendar dates than cutoff',
            'window_policy': 'last 5/15 same-league matches; use actual available count',
            'variance_convention': 'population denominator n for variance and covariance',
            'provider_missing_policy': 'one missing probability makes both window provider summaries None; no skipping or zero-fill',
            'provider_interpretation': 'earlier played-match opportunity descriptions, never target prematch forecasts',
            'strength_adjustment': 'descriptive patterns, not historical-opponent-adjusted effects',
        }
        features = None
        if not own_meta['unavailable_reason'] and not opponent_meta['unavailable_reason']:
            features = {**{f'pattern_own_{name}': value for name, value in own.items()},
                        **{f'pattern_opponent_{name}': value for name, value in opponent.items()}}
        return {'features': features, 'metadata': metadata}
