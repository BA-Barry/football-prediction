"""Descriptions of five earlier total-xG forecast residual pairs.

These representations share information: mean(net)=mean(for)-mean(against),
mean(total)=mean(for)+mean(against), and cov(for,against)=(var(total)-var(net))/4.
They are not independent new measurements, causal team strengths, or score-
distribution parameters. Keeping a tree's max_features unchanged still changes
its randomized split views when columns are added. Identity, source units and
timing are the caller's responsibility.
"""
from collections.abc import Mapping
import math
from numbers import Real


WINDOW = 5
METRICS = (
    'for_mean', 'against_mean', 'net_mean', 'total_mean', 'mean_abs_net',
    'net_population_variance', 'total_population_variance',
    'for_against_population_covariance',
)
FEATURE_NAMES = tuple(f'xg_residual_{metric}_{WINDOW}' for metric in METRICS)


def _residual(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError('Residuals must be finite real numbers, not booleans')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Residuals must be finite real numbers')
    return result


def summarize_residual_patterns(residual_pairs):
    """Return eight summaries of exactly five team-oriented xG residuals.

    A pair maps ``residual_for`` and ``residual_against`` to signed finite
    numbers. None or a shorter window returns the complete missing group,
    never statistics over a substituted or reduced history. The population
    variance/covariance denominator is five. No input is modified.
    """
    missing = dict.fromkeys(FEATURE_NAMES)
    if residual_pairs is None:
        return missing
    pairs = list(residual_pairs)
    if len(pairs) > WINDOW:
        raise ValueError('Use exactly the existing five-match window')
    first, second = [], []
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ValueError('Need explicit residual_for/residual_against pairs')
        try:
            first.append(_residual(pair['residual_for']))
            second.append(_residual(pair['residual_against']))
        except KeyError as error:
            raise ValueError('Need explicit residual_for/residual_against pairs') from error
    if len(pairs) != WINDOW:
        return missing
    own_mean, against_mean = math.fsum(first)/WINDOW, math.fsum(second)/WINDOW
    net = [a-b for a, b in zip(first, second)]
    total = [a+b for a, b in zip(first, second)]
    net_mean, total_mean = math.fsum(net)/WINDOW, math.fsum(total)/WINDOW
    values = (
        own_mean, against_mean, net_mean, total_mean,
        math.fsum(abs(value) for value in net)/WINDOW,
        math.fsum((value-net_mean)**2 for value in net)/WINDOW,
        math.fsum((value-total_mean)**2 for value in total)/WINDOW,
        math.fsum((a-own_mean)*(b-against_mean) for a, b in zip(first, second))/WINDOW,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Residual summaries overflowed the finite numeric range')
    return dict(zip(FEATURE_NAMES, values))
