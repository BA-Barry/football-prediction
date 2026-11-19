"""Explain a limitation of an independent-Poisson 1X2 decision.

These are mathematical properties of a supplied law, not fitted probabilities,
confidence intervals, empirical accuracy limits, or forecast adjustments.
"""
import math
from numbers import Real

from scipy.special import i0e


def independent_poisson_draw_limit(home_rate, away_rate):
    """Report the largest draw mass at a fixed sum of two Poisson means.

For independent scores, P(draw)=exp(-T)*I0(2*sqrt(h*a)). At fixed T=h+a,
AM-GM bounds the Bessel argument by T; I0 is increasing on [0,infinity),
so exp(-T)*I0(T)=i0e(T) is the attainable maximum, at h=a=T/2.
When that maximum is below 1/3, some winner has more probability than a draw
at every split of the same T. This says nothing about dependent score laws.
"""
    for value in (home_rate, away_rate):
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(float(value)) or value <= 0):
            raise ValueError('Need two positive finite independent Poisson means')
    h, a = float(home_rate), float(away_rate)
    total = h + a
    if not math.isfinite(total):
        raise ValueError('The combined goal mean must be finite')
    argument = min(total, 2 * math.sqrt(h) * math.sqrt(a))
    actual = math.exp(argument-total) * float(i0e(argument))
    maximum = float(i0e(total))
    gap = maximum - 1/3
    return {
        'law': 'untruncated independent Poisson; no Dixon-Coles or other adjustment',
        'total_goal_mean': total,
        'draw_probability_under_supplied_means': actual,
        'max_draw_probability_over_all_splits_of_same_total': maximum,
        'maximizing_home_and_away_mean': total/2,
        'draw_unique_top1_impossible_at_fixed_total': gap < -1e-12,
        'draw_unique_top1_possible_at_fixed_total': gap > 1e-12,
        'near_numerical_one_third_boundary': abs(gap) <= 1e-12,
        'is_empirical_accuracy_bound': False,
        'modifies_forecast': False,
    }
