from copy import deepcopy
import json
import math

import pytest

from football_model import market_research as research


def record(mid="a", outcome=0, markets=None):
    return {"match_id": mid, "date": "2025-08-20", "competition": "E0",
            "actual_outcome": outcome, "source_url": "https://example.test/history.csv",
            "market_prices": {"avg_preclosing": [1.7, 4.0, 5.0]} if markets is None else markets}


def test_real_library_three_methods_same_quote_and_input_not_mutated():
    records = [record(), record("b", 1), record("c", 2)]
    before = deepcopy(records)
    summary, rows = research.analyse_quotes(records)
    assert records == before
    assert len(rows) == 12 and summary["matches"] == 3
    assert summary["quote_method_failures"] == []
    primary = summary["markets"]["avg_preclosing"]
    assert primary["common_three_method_matches"] == 3
    assert primary["not_in_common_match_ids"] == []
    expected = [1/x for x in records[0]["market_prices"]["avg_preclosing"]]
    expected = [x/sum(expected) for x in expected]
    assert primary["common_scores"]["multiplicative"]["log_loss"] == pytest.approx(-sum(map(math.log, expected))/3)
    assert primary["common_scores"]["power"]["n"] == 3
    assert primary["common_scores"]["shin"]["n"] == 3
    assert primary["difference_from_multiplicative_on_common"]["power"]["maximum_absolute_probability_change"] > 0
    json.dumps(summary, allow_nan=False)


def test_failure_explicit_not_replaced_and_common_coverage_not_hidden():
    # A true underround: Shin's installed solver cannot recover a valid z.
    records = [record(), record("underround", markets={"avg_preclosing": [4., 4., 4.]})]
    summary, rows = research.analyse_quotes(records)
    primary = summary["markets"]["avg_preclosing"]
    assert primary["quotes_available"] == 2
    assert primary["common_three_method_matches"] == 1
    assert primary["not_in_common_match_ids"] == ["underround"]
    assert primary["method_available_scores"]["power"]["n"] == 2
    assert primary["method_available_scores"]["shin"]["n"] == 1
    assert primary["common_scores"]["power"]["n"] == 1
    assert summary["quote_method_failures"][0]["method"] == "shin"
    flat = list(research._csv_rows(rows))
    damaged = next(r for r in flat if r["match_id"] == "underround" and r["market"] == "avg_preclosing")
    assert not damaged["shin_available"] and damaged["shin_p_home"] is None
    assert json.loads(damaged["source_decimal_odds"]) == [4., 4., 4.]


def test_missing_market_preserved_without_fabricating_quotes():
    summary, rows = research.analyse_quotes([record(markets={})])
    assert len(rows) == 4
    assert summary["quote_method_failures"] == []
    for detail in summary["markets"].values():
        assert detail["quotes_available"] == 0 and detail["common_three_method_matches"] == 0
        assert detail["missing_quote_match_ids"] == ["a"]
        assert all(x is None for x in detail["common_scores"].values())


def test_methods_and_markets_do_not_increase_independent_match_count():
    markets = {key: [1.7, 4., 5.] for key in research.MARKET_COLUMNS}
    summary, rows = research.analyse_quotes([record(markets=markets)])
    assert summary["matches"] == 1 and len(rows) == 4
    assert all(d["common_three_method_matches"] == 1 for d in summary["markets"].values())


def test_zero_observed_probability_reported_as_infinite_not_clipped():
    score = research._scores([[1., 0., 0.]], [2])
    assert score["log_loss"] is None
    assert score["log_loss_status"] == "positive_infinity"
    assert score["zero_observed_probability_count"] == 1
    assert score["brier_sum"] == 2
    json.dumps(score, allow_nan=False)


@pytest.mark.parametrize("records", [[], [record(), record()], [record(outcome=True)], [record(outcome=3)]])
def test_invalid_cohorts_not_silently_reduced(records):
    with pytest.raises(ValueError):
        research.analyse_quotes(records)


def test_order_does_not_change_results():
    records = [record(), record("b", 2)]
    a, _ = research.analyse_quotes(records)
    b, _ = research.analyse_quotes(records[::-1])
    assert a == b


def write_trials(parent, season="2025-26"):
    """Small archive fixtures retain genuine validator semantics, not a mock."""
    directory = parent/season
    directory.mkdir(parents=True)
    start = int(season[:4])
    paths = {}
    for competition in research.LEAGUES:
        row = record(mid=f"{competition}:{season}", markets={})
        row.update({"competition": competition, "date": f"{start}-08-20",
                    "home": "H", "away": "A", "actual_score": [1, 0],
                    "probabilities": [.5, .25, .25], "market_probabilities": {},
                    "fit_before_date": f"{start}-08-01", "training_max_date": f"{start}-05-20"})
        data = {"start": f"{start}-08-01", "end_exclusive": f"{start+1}-07-01", "predictions": [row]}
        path = directory/f"{competition}_original.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        paths[competition] = path
    return paths


def update_trial(path, update):
    data = json.loads(path.read_text(encoding="utf-8"))
    update(data)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize("season", research.SEASONS)
def test_load_season_records_checks_real_row_metadata_and_retains_all_sources(tmp_path, season):
    paths = write_trials(tmp_path, season)
    rows, sources = research.load_season_records(tmp_path, season)
    assert len(rows) == len(research.LEAGUES)
    assert {r["competition"] for r in rows} == set(research.LEAGUES)
    assert sources == [str(paths[competition].resolve()) for competition in research.LEAGUES]


@pytest.mark.parametrize("field,value", [
    ("start", "2024-08-01"), ("end_exclusive", "2025-07-01"),
    ("start", "2025-08-02"), ("end_exclusive", "2026-07-02"),
])
def test_misplaced_or_wrong_declared_trial_intervals_are_rejected(tmp_path, field, value):
    paths = write_trials(tmp_path)
    update_trial(paths["E0"], lambda data: data.update({field: value}))
    with pytest.raises(ValueError, match="trial interval"):
        research.load_season_records(tmp_path, "2025-26")


def test_swapped_other_seasons_valid_trial_is_not_relabelled(tmp_path):
    paths = write_trials(tmp_path, "2025-26")
    earlier = write_trials(tmp_path, "2024-25")
    paths["E0"].write_text(earlier["E0"].read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="trial interval"):
        research.load_season_records(tmp_path, "2025-26")


@pytest.mark.parametrize("day", ["2025-07-31", "2026-07-01", "2026-07-02"])
def test_valid_chronology_does_not_allow_target_dates_outside_season(tmp_path, day):
    paths = write_trials(tmp_path)

    def change(data):
        row = data["predictions"][0]
        row.update({"date": day, "fit_before_date": "2025-07-01", "training_max_date": "2025-06-01"})

    update_trial(paths["E0"], change)
    with pytest.raises(ValueError, match="outside declared research season"):
        research.load_season_records(tmp_path, "2025-26")


@pytest.mark.parametrize("day", ["2025-08-01", "2026-06-30"])
def test_season_endpoints_are_start_inclusive_end_exclusive(tmp_path, day):
    paths = write_trials(tmp_path)
    update_trial(paths["E0"], lambda data: data["predictions"][0].update({"date": day}))
    rows, _ = research.load_season_records(tmp_path, "2025-26")
    assert next(row for row in rows if row["competition"] == "E0")["date"] == day


def test_misplaced_league_and_duplicate_matches_remain_explicit_failures(tmp_path):
    paths = write_trials(tmp_path)
    update_trial(paths["E0"], lambda data: data["predictions"][0].update({"competition": "D1"}))
    with pytest.raises(ValueError, match="Source league"):
        research.load_season_records(tmp_path, "2025-26")
    update_trial(paths["E0"], lambda data: data["predictions"][0].update({"competition": "E0"}))
    update_trial(paths["E0"], lambda data: data["predictions"].append(deepcopy(data["predictions"][0])))
    with pytest.raises(ValueError, match="Duplicate match"):
        research.load_season_records(tmp_path, "2025-26")


def test_cross_league_duplicate_id_is_not_overwritten(tmp_path):
    paths = write_trials(tmp_path)
    update_trial(paths["D1"], lambda data: data["predictions"][0].update({"match_id": "E0:2025-26"}))
    with pytest.raises(ValueError, match="Duplicate match"):
        research.load_season_records(tmp_path, "2025-26")


def test_runner_rejects_repeated_identity_across_declared_seasons(tmp_path):
    parent = tmp_path/"parent"
    earlier = write_trials(parent, "2024-25")
    write_trials(parent, "2025-26")
    update_trial(earlier["E0"], lambda data: data["predictions"][0].update({"match_id": "E0:2025-26"}))
    with pytest.raises(ValueError, match="Duplicate match across research seasons"):
        research.run_research(tmp_path, parent)


def test_log_loss_report_distinguishes_positive_infinity_from_no_common_quote():
    infinite = research._scores([[1., 0., 0.]], [2])
    finite = research._scores([[.5, .25, .25]], [0])
    assert research._log_loss_cell(infinite) == "+∞（观测结果概率0）"
    assert research._log_loss_cell(None) == "无共同报价"
    assert research._log_loss_cell(finite) == f"{math.log(2):.6f}"
