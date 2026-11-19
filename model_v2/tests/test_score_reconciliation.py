import numpy as np
import pytest

from football_model.probability import summarize
from football_model.score_reconciliation import reconcile_score_probabilities
from football_model.vendor.penaltyblog_grid import FootballProbabilityGrid


def asymmetric_grid():
    # Rectangular and asymmetric to expose H/A transposition mistakes. Stored
    # means are deliberately stale: the adapter must read the final matrix.
    matrix = np.array([[.08, .12, .02], [.14, .10, .04],
                       [.20, .05, .05], [.10, .07, .03]])
    return FootballProbabilityGrid(matrix, 99., 88., normalize=False)


def test_hda_direction_and_conditional_scores_are_preserved_without_mutation():
    source = asymmetric_grid()
    original = source.grid.copy()
    target = [.18, .27, .55]
    result = reconcile_score_probabilities(source, target)
    assert isinstance(result, FootballProbabilityGrid)
    assert result.home_draw_away == pytest.approx(target, abs=1e-14)
    h, a = np.indices(original.shape)
    for mask in (h > a, h == a, h < a):
        assert result.grid[mask] / result.grid[mask].sum() == pytest.approx(
            original[mask] / original[mask].sum(), abs=1e-14)
    assert np.array_equal(source.grid, original)


def test_means_and_total_goals_are_recomputed_from_the_same_matrix():
    result = reconcile_score_probabilities(asymmetric_grid(), [.60, .25, .15])
    h, a = np.indices(result.grid.shape)
    expected_home = float((h * result.grid).sum())
    expected_away = float((a * result.grid).sum())
    assert result.home_goal_expectation == pytest.approx(expected_home)
    assert result.away_goal_expectation == pytest.approx(expected_away)
    assert result.home_goal_expectation != 99.
    assert result.away_goal_expectation != 88.
    totals = result.total_goals_distribution()
    assert totals.sum() == pytest.approx(1.)
    assert np.dot(np.arange(len(totals)), totals) == pytest.approx(expected_home + expected_away)
    for goals, probability in enumerate(totals):
        assert probability == pytest.approx(result.grid[h+a == goals].sum())
    summary = summarize(result)
    assert list(summary['one_x_two'].values()) == pytest.approx([.60, .25, .15])
    assert sum(summary['total_goals'].values()) == pytest.approx(1.)
    assert sum(summary['away_minus_two_jincai'].values()) == pytest.approx(1.)


def test_missing_outcome_support_is_rejected_instead_of_fabricating_scores():
    source = FootballProbabilityGrid(np.array([[.4, .2], [0., .4]]), 0., 0., normalize=False)
    with pytest.raises(ValueError, match='zero structural support'):
        reconcile_score_probabilities(source, [.3, .4, .3])


@pytest.mark.parametrize('target', [
    [.4, .4], [.3, .3, .3], [0., .4, .6], [-.1, .5, .6],
    [.3, np.nan, .7], [.3, np.inf, .7],
])
def test_invalid_target_probabilities_are_not_silently_repaired(target):
    with pytest.raises(ValueError, match='Target'):
        reconcile_score_probabilities(asymmetric_grid(), target)


@pytest.mark.parametrize('cell', [-.01, np.nan])
def test_invalid_source_cells_are_rejected_even_when_source_container_allows_them(cell):
    source = asymmetric_grid()
    source.grid[0, 0] = cell
    with pytest.raises(ValueError, match='Source'):
        reconcile_score_probabilities(source, [.4, .3, .3])
