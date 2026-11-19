from copy import deepcopy
from datetime import date
import json
from types import SimpleNamespace

import pytest

from football_model.date_model import DateTeamGoalModel
from football_model import season_state_research as research
from football_model.season_state_model import season_for_date


def row(mid, day, home="A", away="B", goals=(1, 0), competition="E0"):
    return SimpleNamespace(match_id=mid, match_date=day, home=home, away=away,
        full_time=goals, competition=competition, season=season_for_date(day), goal_fit_exclusion=None)


def source():
    return [row("initial", date(2022, 8, 5)), row("initial2", date(2022, 9, 1), "C", "D"),
            row("early-1", date(2023, 8, 11)), row("early-2", date(2023, 9, 4), "C", "D"),
            row("late-1", date(2024, 8, 16)), row("late-2", date(2024, 9, 5), "C", "D"),
            row("last-1", date(2025, 8, 15)), row("last-2", date(2025, 9, 5), "C", "D")]


class Fake:
    def __init__(self, arm, q, calls, on_fit=None):
        self.arm, self.q, self.calls, self.on_fit = arm, q, calls, on_fit

    def fit(self, records, cutoff):
        assert records and all(r.match_date < cutoff for r in records)
        assert all(r.competition == "E0" for r in records)
        if self.on_fit:
            self.on_fit(cutoff)
        self.teams = {t for r in records for t in (r.home, r.away)}
        self.cutoff = cutoff
        self.ids = [r.match_id for r in records]
        self.calls.append((self.arm, self.q, cutoff, list(self.ids)))
        return self

    def parameters(self):
        return {"fake_arm": self.arm, "q": self.q, "teams": sorted(self.teams),
                "ids": self.ids, "cutoff": self.cutoff.isoformat(), "optimizer_iterations": 1}

    def predict(self, home, away, competition, day):
        assert getattr(self, "restored", False), "Must use saved restoration, not fitted instance"
        assert day >= self.cutoff and competition == "E0"
        assert {home, away} <= self.teams
        # At outcome (1,0), q=.5 has the best full-score loss.
        return (1+abs((self.q or 0)-.5), .5)


def fakes(monkeypatch, on_fit=None):
    calls = []
    monkeypatch.setattr(research, "_new_model", lambda arm, q: Fake(arm, q, calls, on_fit))

    def restore(arm, payload):
        value = Fake(arm, payload["q"], calls)
        value.teams, value.ids = set(payload["teams"]), list(payload["ids"])
        value.cutoff = date.fromisoformat(payload["cutoff"])
        value.restored = True
        return value

    monkeypatch.setattr(research, "_restore_model", restore)
    return calls


def test_complete_early_save_selection_then_two_fixed_later_seasons(monkeypatch, tmp_path):
    records = source()
    before = deepcopy([vars(r) for r in records])

    def earlier_selection_exists(cutoff):
        if cutoff >= date(2024, 8, 1):
            paths = list((tmp_path/"outputs").glob("*/早期尺度选择.json"))
            assert len(paths) == 1
            saved = json.loads(paths[0].read_text(encoding="utf-8"))
            assert saved["selected_scale"] == .5
            assert saved["training_match_ids"] == ["early-1", "early-2"]

    calls = fakes(monkeypatch, earlier_selection_exists)
    monkeypatch.setattr(research, "load_inspections", lambda directory: (records, {"synthetic": True}))
    sentinel = tmp_path/"unchanged-case.json"
    sentinel.write_bytes(b"old prospective probabilities")
    output = research.run(tmp_path, tmp_path/"inspection")
    result = json.loads((output/"跨赛季研究结果.json").read_text(encoding="utf-8"))
    assert result["selection"]["selected_scale"] == .5
    assert result["early"]["fit_count"] == 6
    assert result["total_fits"] == 14
    assert result["later_selections"] == 0
    assert result["later_combined"]["target_count"] == 4
    assert result["later_combined"]["all_paired"]["n"] == 4
    assert before == [vars(r) for r in records]
    assert sentinel.read_bytes() == b"old prospective probabilities"
    for arm, q, cutoff, ids in calls:
        if cutoff < date(2024, 7, 1):
            assert not any(mid.startswith(("late", "last")) for mid in ids)
        elif arm == "seasonal":
            assert q == .5
    last_fit = [c for c in calls if c[2] == date(2025, 8, 1)]
    assert all("late-2" in c[3] for c in last_fit), "Finished prior evaluation season is valid history"


def test_changing_later_scores_cannot_change_earlier_selection(monkeypatch, tmp_path):
    fakes(monkeypatch)
    original = source()
    monkeypatch.setattr(research, "load_inspections", lambda _: (original, {}))
    first = research.run(tmp_path/"first", tmp_path/"input")
    changed = deepcopy(original)
    for r in changed:
        if r.match_date >= date(2024, 7, 1):
            r.full_time = (7, 3)
    monkeypatch.setattr(research, "load_inspections", lambda _: (changed, {}))
    second = research.run(tmp_path/"second", tmp_path/"input")
    a = json.loads((first/"早期尺度选择.json").read_text(encoding="utf-8"))
    b = json.loads((second/"早期尺度选择.json").read_text(encoding="utf-8"))
    assert a == b


def test_same_date_results_and_unknown_team_not_filled(monkeypatch, tmp_path):
    calls = fakes(monkeypatch)
    records = [row("old", date(2022, 8, 5)), row("cutoff", date(2023, 8, 1), "NEW", "A"),
               row("target", date(2023, 8, 11), "NEW", "B"),
               row("later", date(2023, 9, 1), "NEW", "B")]
    result = research.run_season(records, "2023-24", dict(research.SCALES), tmp_path/"season")
    assert result["summary"]["target_count"] == 3
    assert result["summary"]["all_paired"]["n"] == 1
    for arm in research.SCALES:
        assert result["per_fixture"][0]["arms"][arm]["reason"] == "unseen_team_no_estimated_prior"
        assert result["per_fixture"][1]["arms"][arm]["rates"] is None
    assert all("cutoff" not in ids for _, _, cutoff, ids in calls if cutoff == date(2023, 8, 1))
    assert result["summary"]["by_prior_epl_status"]["local_first_seen"]["target_count"] == 3


def test_identity_labels_first_returning_continuing_only_describe(monkeypatch, tmp_path):
    fakes(monkeypatch)
    records = [row("older", date(2022, 8, 5), "RETURN", "A"),
               row("prior", date(2023, 8, 5), "A", "B"),
               row("c", date(2024, 8, 16), "A", "B"),
               row("r", date(2024, 8, 17), "RETURN", "A"),
               row("f", date(2024, 8, 18), "FIRST", "RETURN")]
    result = research.run_season(records, "2024-25", {"original": None, "seasonal": .5}, tmp_path/"s")
    assert [r["prior_epl_status"] for r in result["per_fixture"]] == ["continuing", "returning", "local_first_seen"]
    assert result["summary"]["all_paired"]["n"] == 2
    assert result["summary"]["by_previous_epl_presence"]["at_least_one_not_previous"]["target_count"] == 2


def test_exact_selection_tie_prefers_small_scale_and_rejects_empty():
    value = {"season": research.EARLY, "arm_scales": dict(research.SCALES),
             "summary": {"all_paired": {"n": 1, "paired_match_ids": ["early"],
                 "scores": {a: {"full_score_nll": 2.} for a in research.SCALES}}}}
    assert research.select_scale(value)["selected_scale"] == 0
    value["summary"]["all_paired"]["n"] = 0
    with pytest.raises(ValueError, match="No common"):
        research.select_scale(value)
    value["season"] = "2024-25"
    with pytest.raises(ValueError, match="earlier"):
        research.select_scale(value)


def test_common_cohort_not_different_arm_means(monkeypatch, tmp_path):
    fakes(monkeypatch)
    records = source()[:4]
    result = research.run_season(records, "2023-24", dict(research.SCALES), tmp_path/"s")
    rows = result["per_fixture"]
    rows[0]["arms"]["q1"] = {"status": "missing", "reason": "synthetic"}
    summary = research.summarize_season(rows, list(research.SCALES))
    assert summary["arms"]["q0"]["coverage"] == 2
    assert summary["arms"]["q1"]["coverage"] == 1
    assert summary["all_paired"]["n"] == 1
    assert all(s["n"] == 1 for s in summary["all_paired"]["scores"].values())


def test_original_restore_is_exact_and_never_fits(monkeypatch):
    records = source()[:4]
    fitted = DateTeamGoalModel().fit(records, date(2024, 8, 1))
    payload = json.loads(json.dumps(fitted.parameters()))
    monkeypatch.setattr(DateTeamGoalModel, "fit", lambda *args: pytest.fail("restore must not fit"))
    restored = research.restore_original(payload)
    assert restored.parameters() == payload
    assert restored.predict("A", "B", "E0", date(2024, 8, 16)) == fitted.predict("A", "B", "E0", date(2024, 8, 16))
    payload["coefficients"].clear()
    assert restored.coefficients


@pytest.mark.parametrize("change", ["duplicate", "season", "self", "negative", "float", "old_season"])
def test_invalid_epl_records(change):
    records = source()
    if change == "duplicate":
        records.append(records[0])
    elif change == "season":
        records[0].season = "2022/23"
    elif change == "self":
        records[0].away = records[0].home
    elif change == "negative":
        records[0].full_time = (-1, 0)
    elif change == "float":
        records[0].full_time = (1., 0)
    else:
        records[0].match_date, records[0].season = date(2021, 8, 1), "2021-22"
    with pytest.raises(ValueError):
        research.validate_records(records)


def test_e1_is_not_training_or_evaluation_and_inputs_sorted():
    records = source()
    records.append(row("E1", date(2022, 8, 5), competition="E1"))
    result = research.validate_records(records[::-1])
    assert len(result) == 8
    assert all(r.competition == "E0" for r in result)
    assert result[0].match_id == "initial"
    with pytest.raises(ValueError):
        research.validate_records([])


def test_empty_targets_duplicate_and_fit_failure_propagate(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="No target"):
        research.run_season(source()[:2], "2023-24", dict(research.SCALES), tmp_path/"none")
    with pytest.raises(ValueError, match="Duplicate"):
        research.run_season(source()+[source()[0]], "2023-24", dict(research.SCALES), tmp_path/"dup")

    def broken(*args):
        raise RuntimeError("actual optimizer failure")

    monkeypatch.setattr(research, "_new_model", broken)
    with pytest.raises(RuntimeError, match="actual optimizer"):
        research.run_season(source(), "2023-24", dict(research.SCALES), tmp_path/"failed")
