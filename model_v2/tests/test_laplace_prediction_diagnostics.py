"""Synthetic saved-record diagnostics; no fits or real-season execution."""

import copy
import json
import math

import pytest

from football_model import laplace_prediction_diagnostics as diagnostic
from football_model.gaussian_goal_mixture import METRICS, score


def saved_row(mid, day, actual, mean, known, unknown, cohort):
    covariance = [[known[i][j] + unknown[i][j] for j in range(2)] for i in range(2)]
    distribution = {"match_id": mid, "match_date": day, "log_mean": list(mean), "log_cov": covariance,
                    "known_parameter_log_cov": copy.deepcopy(known), "unknown_player_log_cov": copy.deepcopy(unknown)}
    return {"match_id": mid, "match_date": day, "actual": list(actual), "status": "predicted",
            "old_v36_cohort": cohort, "distribution": distribution,
            **score(mean, covariance, actual, order=3)}


def records(zero=False):
    team, point = [], []
    players = {scale: [] for scale in diagnostic.PRIOR_SCALES}
    for mid, day, actual, cohort in (("a", "2024-01-07", [1, 0], "known"),
                                      ("b", "2024-01-08", [0, 2], "unknown"),
                                      ("c", "2024-01-14", [2, 2], "known")):
        mean = [math.log(1.4), math.log(.9)]
        known = [[0., 0.], [0., 0.]] if zero else [[.02, .001], [.001, .04]]
        t = saved_row(mid, day, actual, mean, known, [[0., 0.], [0., 0.]], cohort)
        team.append(t)
        point.append({"match_id": mid, "match_date": day, "actual": list(actual), "status": "predicted",
                      **score(mean, [[0., 0.], [0., 0.]], actual, order=1)})
        for scale in diagnostic.PRIOR_SCALES:
            p_known = [[value * 1.5 for value in row] for row in known]
            variance = scale ** 2 / 2 if cohort == "unknown" and not zero else 0.
            unknown = [[variance, 0.], [0., variance]]
            players[scale].append(saved_row(mid, day, actual, [mean[0] + scale, mean[1] - scale], p_known, unknown, cohort))
    return team, point, players


def test_eight_metric_identity_and_independent_equal_match_aggregation():
    result, ledgers = diagnostic.diagnose(*records())
    assert result["model_fits"] == result["hyperparameter_searches"] == result["new_prospective_predictions"] == 0
    assert result["main_prior_sd"] == .10
    assert result["fixed_prior_scales"] == [.05, .10, .20]
    for scale, rows in ledgers.items():
        for row in rows:
            for metric in METRICS:
                d = row["decomposition"][metric]
                m = row["metrics"]
                assert d["centre_difference"] == m["player_map_counterfactual"][metric] - m["team_map"][metric]
                assert d["expansion_contrast"] == d["player_expansion"] - d["team_expansion"]
                assert d["total_difference"] == pytest.approx(d["centre_difference"] + d["expansion_contrast"], abs=1e-14)
                assert abs(d["identity_residual"]) < 1e-14
        group = result["scales"][scale]["cohorts"]["all"]
        for metric in METRICS:
            for part in diagnostic.PARTS:
                expected = math.fsum(row["decomposition"][metric][part] for row in rows) / len(rows)
                assert group["mean_decomposition"][metric][part] == expected


def test_zero_covariance_has_zero_expansion_and_jensen_increase():
    result, ledgers = diagnostic.diagnose(*records(zero=True))
    for rows in ledgers.values():
        for row in rows:
            for metric in METRICS:
                d = row["decomposition"][metric]
                assert d["player_expansion"] == pytest.approx(0., abs=1e-14)
                assert d["team_expansion"] == pytest.approx(0., abs=1e-14)
                assert d["expansion_contrast"] == pytest.approx(0., abs=1e-14)
            for track in ("team", "players"):
                assert row["rates_and_variances"][track]["jensen_total_rate_increase"] == pytest.approx(0.)
    assert result["target_matches"] == 3


def test_cohorts_and_monday_weeks_do_not_mix_and_empty_groups_are_null():
    result, _ = diagnostic.diagnose(*records())
    for scale in result["scales"].values():
        assert scale["cohorts"]["old_known"]["match_ids"] == ["a", "c"]
        assert scale["cohorts"]["old_unknown"]["match_ids"] == ["b"]
        assert set(scale["weeks"]) == {"2024-01-01", "2024-01-08"}
        assert scale["weeks"]["2024-01-01"]["all"]["match_ids"] == ["a"]
        empty = scale["weeks"]["2024-01-01"]["old_unknown"]
        assert empty["n"] == 0
        assert empty["mean_metrics"]["team_map"]["full_score_nll"] is None
        assert empty["mean_rates_and_variances"]["players"]["map_total_rate"] is None
        assert empty["maximum_absolute_identity_residual"]["full_score_nll"] is None


def test_saved_laplace_scores_read_directly_and_only_player_zero_covariance_scored(monkeypatch):
    team, point, players = records()
    team[0]["full_score_nll"] += 7.
    players[.10][0]["full_score_nll"] += 11.
    calls = []
    def counterfactual(mean, covariance, actual, order):
        calls.append((covariance, order))
        assert covariance == diagnostic.ZERO_COV and order == 1
        return score(mean, covariance, actual, order=order)
    monkeypatch.setattr(diagnostic, "mixture_score", counterfactual)
    _, ledgers = diagnostic.diagnose(team, point, players)
    assert len(calls) == 9
    first = ledgers["s0.10"][0]
    assert first["metrics"]["team_laplace"]["full_score_nll"] == team[0]["full_score_nll"]
    assert first["metrics"]["player_laplace"]["full_score_nll"] == players[.10][0]["full_score_nll"]


def test_rates_jensen_and_known_unknown_variance_are_separate():
    team, point, players = records()
    result, ledgers = diagnostic.diagnose(team, point, players)
    row = ledgers["s0.10"][1]
    rates = row["rates_and_variances"]["players"]
    assert rates["unknown_log_variance_home"] == pytest.approx(.005)
    assert rates["unknown_log_variance_away"] == pytest.approx(.005)
    assert rates["known_log_variance_home"] == pytest.approx(.03)
    assert rates["known_log_variance_away"] == pytest.approx(.06)
    assert rates["map_total_rate"] == pytest.approx(1.4 * math.exp(.1) + .9 * math.exp(-.1))
    assert rates["integrated_total_rate"] == players[.10][1]["forecast"]["expected_goals"]["total"]
    assert rates["jensen_total_rate_increase"] > 0
    assert rates["actual_total_goals"] == 2
    summary = result["scales"]["s0.10"]["cohorts"]["all"]["mean_rates_and_variances"]["players"]
    assert summary["unknown_log_variance_home"] == pytest.approx(.005 / 3)
    assert summary["actual_total_goals"] == pytest.approx(7 / 3)


def test_inputs_unchanged_and_output_ledger_does_not_alias_actual():
    args = records()
    before = copy.deepcopy(args)
    _, ledgers = diagnostic.diagnose(*args)
    ledgers["s0.10"][0]["actual"].append(999)
    assert args == before


@pytest.mark.parametrize("track", ["team", "point", .05, .10, .20])
def test_duplicate_ids_rejected(track):
    team, point, players = records()
    target = team if track == "team" else point if track == "point" else players[track]
    target.append(copy.deepcopy(target[0]))
    with pytest.raises(ValueError, match="Duplicate"):
        diagnostic.diagnose(team, point, players)


@pytest.mark.parametrize("track", ["point", .05, .10, .20])
def test_missing_or_extra_target_rejected(track):
    team, point, players = records()
    target = point if track == "point" else players[track]
    target.pop()
    with pytest.raises(ValueError, match="Target match set"):
        diagnostic.diagnose(team, point, players)


@pytest.mark.parametrize("field,value,message", [("actual", [4, 3], "actual differs"),
                                                ("old_v36_cohort", "unknown", "cohort differs"),
                                                ("status", "skipped", "predicted"),
                                                ("actual", [True, 0], "integer actual")])
def test_inconsistent_target_and_status_rejected(field, value, message):
    team, point, players = records()
    players[.10][0][field] = value
    with pytest.raises(ValueError, match=message):
        diagnostic.diagnose(team, point, players)


def test_date_mismatch_rejected_even_if_distribution_identity_changes_too():
    team, point, players = records()
    players[.10][0]["match_date"] = "2024-01-06"
    players[.10][0]["distribution"]["match_date"] = "2024-01-06"
    with pytest.raises(ValueError, match="date/actual differs"):
        diagnostic.diagnose(team, point, players)


@pytest.mark.parametrize("problem", ["identity", "scored_mean", "components", "team_point", "missing_scale"])
def test_wrong_saved_distribution_or_scale_set_rejected(problem):
    team, point, players = records()
    if problem == "identity":
        players[.10][0]["distribution"]["match_id"] = "different"
    elif problem == "scored_mean":
        players[.10][0]["log_mean"] = [0., 0.]
    elif problem == "components":
        players[.10][0]["distribution"]["unknown_player_log_cov"][0][0] = .2
    elif problem == "team_point":
        point[0]["log_mean"] = [0., 0.]
    else:
        players.pop(.05)
    with pytest.raises(ValueError):
        diagnostic.diagnose(team, point, players)


def test_synthetic_run_writes_new_directory_preserves_sources_and_report_limits(tmp_path):
    team, point, players = records()
    research = tmp_path / "saved"
    research.mkdir()
    payloads = {"team_laplace_predictions.json": team, "team_map_predictions.json": point,
                **{f"players_s{scale:.2f}_predictions.json": rows for scale, rows in players.items()},
                "固定研究设置.json": {"target_matches": 3, "old_known": 2, "old_unknown": 1}}
    for name, payload in payloads.items():
        (research / name).write_text(json.dumps(payload), encoding="utf-8")
    before = {path: path.read_bytes() for path in research.iterdir()}
    first = diagnostic.run(tmp_path, research)
    second = diagnostic.run(tmp_path, research)
    assert first != second and first.parent == tmp_path / "outputs"
    assert all(path.read_bytes() == content for path, content in before.items())
    result = json.loads((first / "诊断结果.json").read_text(encoding="utf-8"))
    assert result["target_matches"] == 3 and result["research_directory"] == str(research.resolve())
    assert set(result["scales"]) == {"s0.05", "s0.10", "s0.20"}
    assert len(list(first.glob("*_逐场分解.json"))) == 3
    report = (first / "中心扩张诊断说明.md").read_text(encoding="utf-8")
    for text in ("不按结果选赢家", "不是因果归因或独立验证", "不表示未知球员已知零效应", "不据本结果修改方差"):
        assert text in report


@pytest.mark.parametrize("bad_count", ["target_matches", "old_known", "old_unknown"])
def test_shared_incomplete_or_wrong_cohort_counts_rejected_before_output(tmp_path, bad_count):
    team, point, players = records()
    research = tmp_path / "saved"
    research.mkdir()
    policy = {"target_matches": 3, "old_known": 2, "old_unknown": 1}
    policy[bad_count] += 1
    payloads = {"team_laplace_predictions.json": team, "team_map_predictions.json": point,
                **{f"players_s{scale:.2f}_predictions.json": rows for scale, rows in players.items()},
                "固定研究设置.json": policy}
    for name, payload in payloads.items():
        (research / name).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="original fixed research settings"):
        diagnostic.run(tmp_path, research)
    assert not (tmp_path / "outputs").exists()
