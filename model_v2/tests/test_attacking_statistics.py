from football_model.attacking_statistics import parse_attacking_counts


def test_source_difference_retains_zeros_without_interpreting_blocks():
    result = parse_attacking_counts({'HS': '10', 'AS': '0', 'HST': '4', 'AST': '0', 'HC': '3', 'AC': '0'})
    assert result['counts']['non_target_shots'] == [6, 0]
    assert result['counts']['corners'] == [3, 0]
    assert result['issues'] == []


def test_partial_unknown_and_negative_are_not_zero_filled():
    result = parse_attacking_counts({'HS': '10', 'AS': '', 'HST': '4', 'AST': '1', 'HC': '-1', 'AC': '2'})
    assert result['counts']['shots'] is None
    assert result['counts']['non_target_shots'] is None
    assert result['counts']['corners'] is None
    assert result['counts']['shots_on_target'] == [4, 1]


def test_contradictory_shots_preserve_raw_and_do_not_discard_corners():
    result = parse_attacking_counts({'HS': '2', 'AS': '10', 'HST': '3', 'AST': '4', 'HC': '5', 'AC': '6'})
    assert result['counts']['shots_on_target'] is None
    assert result['counts']['non_target_shots'] is None
    assert result['counts']['corners'] == [5, 6]
    assert result['raw_statistic_cells']['HST'] == '3'
    assert result['issues'][0]['reason'] == 'on_target_exceeds_total'
