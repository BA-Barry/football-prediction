"""Recompute saved historical shot/league aggregation differences; no network or fit.

The rebound grouping below is an empirical reconstruction of these snapshots,
not a permanent provider contract. Rosters, starting XIs and players are unused.
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "data/understat/history_20261024T170717354684Z"
SHOTS = ROOT / "data/understat/roster_seasons_EPL_2023_2024_20261025/raw"
OUTPUT = ROOT / "outputs/早期预测研究_20261025/补射链与源NP核对.json"
TOLERANCE = 1e-4


def read(path):
    return json.loads(path.read_bytes())


def chains(shots):
    result = []
    for shot in shots:
        if shot["lastAction"] == "Rebound" and result:
            result[-1].append(shot)
        else:
            result.append([shot])
    return result


def chain_value(chain):
    return 1 - math.prod(1 - float(shot["xG"]) for shot in chain)


def summarize(values):
    return {
        "count": len(values),
        "abs_delta_gt_1e_4": sum(abs(value) > TOLERANCE for value in values),
        "max_abs_delta": max(map(abs, values), default=0.0),
        "mean_abs_delta": sum(map(abs, values)) / len(values) if values else None,
    }


def main():
    counts, per_season = Counter(), Counter()
    errors, raw_differences, penalty_cases = defaultdict(list), [], []
    penalty_values = Counter()
    for year in (2023, 2024):
        source = read(HISTORY / f"EPL_{year}.json")
        # Access only saved dates and team histories, never the players block.
        saved_response = read(HISTORY / f"EPL_{year}_response.bin")
        counts["league_dates_raw_response_mismatches"] += int(source["dates"] != saved_response["dates"])
        history = {}
        for team_id, team in source["teams"].items():
            counts["league_team_history_raw_response_mismatches"] += int(
                team["history"] != saved_response["teams"][team_id]["history"])
            for row in team["history"]:
                key = (str(team_id), row["date"], row["h_a"])
                counts["duplicate_source_history_keys"] += int(key in history)
                history[key] = row
        for match in source["dates"]:
            counts["matches"] += 1
            uid = str(match["id"])
            # Deliberately select shots only: roster/starting-XI blocks are unused.
            shots = read(SHOTS / f"{uid}.json")["shots"]
            counts["shots_raw_response_mismatches"] += int(
                shots != read(SHOTS / f"{uid}_response.bin")["shots"])
            for side in ("h", "a"):
                counts["team_observations"] += 1
                key = (str(match[side]["id"]), match["datetime"], side)
                if key not in history:
                    counts["missing_source_history_matches"] += 1
                    continue
                counts["source_history_matches"] += 1
                row, events = history[key], shots[side]
                grouped = chains(events)
                total = sum(chain_value(chain) for chain in grouped)
                raw_sum = sum(float(shot["xG"]) for shot in events)
                penalties = [shot for shot in events if shot["situation"] == "Penalty"]
                raw_penalty = sum(float(shot["xG"]) for shot in penalties)
                penalty_values.update(shot["xG"] for shot in penalties)
                counts["penalty_shots"] += len(penalties)
                counts["same_side_duplicate_shot_ids"] += len(events) - len({shot["id"] for shot in events})
                counts["shot_identity_mismatches"] += sum(
                    shot["h_a"] != side or str(shot["match_id"]) != uid
                    or shot["date"] != match["datetime"]
                    or shot["h_team"] != match["h"]["title"]
                    or shot["a_team"] != match["a"]["title"] for shot in events)
                counts["own_goal_events"] += sum(shot["result"] == "OwnGoal" for shot in events)
                counts["own_goal_nonzero_xg"] += sum(
                    shot["result"] == "OwnGoal" and float(shot["xG"]) != 0 for shot in events)
                counts["first_shot_labelled_rebound"] += int(bool(events) and events[0]["lastAction"] == "Rebound")
                for chain in grouped:
                    if len(chain) > 1:
                        counts["multiple_shot_chains"] += 1
                        counts["appended_rebound_shots"] += len(chain) - 1
                        counts["chains_at_least_3_shots"] += int(len(chain) > 2)
                        counts["cross_situation_chains"] += int(len({shot["situation"] for shot in chain}) > 1)
                penalty_chains = [chain for chain in grouped if any(shot["situation"] == "Penalty" for shot in chain)]
                np_exclude_chain = sum(chain_value(chain) for chain in grouped if chain not in penalty_chains)
                np_remove_shot = sum(chain_value(chain) for chain in chains(
                    [shot for shot in events if shot["situation"] != "Penalty"]))
                source_xg, source_np = float(row["xG"]), float(row["npxG"])
                index_xg = float(match["xG"][side])
                errors["source_history_xg_minus_fixture_index"].append(source_xg - index_xg)
                errors["chain_xg_minus_source_xg"].append(total - source_xg)
                errors["chain_minus_raw_penalty_minus_source_npxg"].append(total - raw_penalty - source_np)
                errors["remove_penalty_shot_rechain_minus_source_npxg"].append(np_remove_shot - source_np)
                errors["exclude_entire_penalty_chain_minus_source_npxg"].append(np_exclude_chain - source_np)
                errors["raw_nonpenalty_shot_sum_minus_source_npxg"].append(raw_sum - raw_penalty - source_np)
                delta = raw_sum - index_xg
                errors["raw_shot_sum_minus_source_xg"].append(delta)
                if abs(delta) > TOLERANCE:
                    per_season[f"{year}/{str(year + 1)[-2:]}"] += 1
                    raw_differences.append({"understat_id": uid, "date": match["datetime"], "side": side,
                        "team": match[side]["title"], "raw_shot_xg_sum": raw_sum,
                        "source_fixture_xg": index_xg, "delta": delta})
                rebound_penalties = [chain for chain in penalty_chains if len(chain) > 1]
                counts["penalty_rebound_chains"] += len(rebound_penalties)
                if rebound_penalties:
                    penalty_cases.append({"understat_id": uid, "side": side, "team": match[side]["title"],
                        "date": match["datetime"], "source_xg": source_xg, "source_npxg": source_np,
                        "raw_penalty_shot_xg_sum": raw_penalty,
                        "all_penalty_chains_xg": sum(map(chain_value, penalty_chains)),
                        "exclude_entire_penalty_chain_xg": np_exclude_chain,
                        "remove_penalty_shot_rechain_xg": np_remove_shot,
                        "penalty_rebound_chains": [[{key: shot[key] for key in
                            ("id", "minute", "situation", "xG", "lastAction")} for shot in chain]
                            for chain in rebound_penalties]})
    raw_differences.sort(key=lambda row: abs(row["delta"]), reverse=True)
    report = {
        "scope": "Saved EPL 2023/24 and 2024/25 snapshots; retrospective measurement only; no training or network requests.",
        "interpretation": "Rebound and penalty-chain rules empirically reconstruct these saved source aggregates; not an official or permanent provider contract.",
        "rule": "Keep source array order. Append lastAction=Rebound to preceding same-side chain, otherwise start a chain; sum 1-product(1-shot_xG). For the source NPxG reconstruction exclude every chain containing a Penalty shot.",
        "limits": "A first Rebound without a preceding shot starts a chain. Historical publication timestamps and past provider revisions remain unknown. Raw shot sums and source aggregates are distinct fields.",
        "source_paths": {"history": str(HISTORY), "shots": str(SHOTS)},
        "counts": dict(counts), "error_summaries": {key: summarize(values) for key, values in errors.items()},
        "raw_difference_counts": {"team_observations": len(raw_differences),
            "distinct_matches": len({row["understat_id"] for row in raw_differences}),
            "per_season": dict(per_season), "negative": sum(row["delta"] < 0 for row in raw_differences)},
        "penalty_xg_value_counts": dict(penalty_values),
        "penalty_rebound_cases": penalty_cases,
        "raw_sum_difference_cases_above_1e_4": raw_differences,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "counts": report["counts"],
        "raw_difference_counts": report["raw_difference_counts"],
        "chain_errors": report["error_summaries"]["chain_xg_minus_source_xg"],
        "npxg_errors": report["error_summaries"]["exclude_entire_penalty_chain_minus_source_npxg"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
