import math

import numpy as np
import pytest

from football_model.draw_branch import apply_draw_branch, replace_draw_probability
from football_model.probability import score_grid, summarize


def test_full_score_class_and_draw_binary_loss_differences_are_identical():
    source = score_grid(1.8, .8, rho=0)
    old = source.grid.copy()
    p = np.asarray(source.home_draw_away)
    for d in (.05, p[1], .65):
        result = apply_draw_branch(source, d)
        q = np.asarray(result.home_draw_away)
        for h, a, label in ((2, 1, 0), (1, 1, 1), (1, 2, 2)):
            score_difference = math.log(source.grid[h, a]/result.grid[h, a])
            outcome_difference = math.log(p[label]/q[label])
            binary_difference = math.log(p[1]/d) if label == 1 else math.log((p[0]+p[2])/(1-d))
            assert score_difference == pytest.approx(outcome_difference, abs=1e-12)
            assert score_difference == pytest.approx(binary_difference, abs=1e-12)
        assert np.array_equal(source.grid, old)


def test_draw_branch_can_select_draw_without_flipping_the_conditional_winner():
    for rates in ((1.8, .8), (.8, 1.8)):
        source = score_grid(*rates, rho=0)
        p = np.asarray(source.home_draw_away)
        result = apply_draw_branch(source, .6)
        q = np.asarray(result.home_draw_away)
        assert q.argmax() == 1
        assert q[0]/q[2] == pytest.approx(p[0]/p[2], rel=1e-12)
        # The score means must be those of the new matrix, not the old means.
        summary = summarize(result)
        h, a = np.indices(result.grid.shape)
        assert summary['expected_goals']['home'] == pytest.approx((h*result.grid).sum())
        assert summary['expected_goals']['away'] == pytest.approx((a*result.grid).sum())
        assert summary['expected_goals']['home'] != pytest.approx(rates[0], abs=1e-5)


def test_absent_or_invalid_draw_signal_is_not_replaced_with_a_default():
    for bad in (None, True, 0., 1., float('nan'), float('inf')):
        with pytest.raises(ValueError, match='Draw probability'):
            replace_draw_probability([.5, .25, .25], bad)
    with pytest.raises(ValueError, match='Source probabilities'):
        replace_draw_probability([.5, 0., .5], .3)
