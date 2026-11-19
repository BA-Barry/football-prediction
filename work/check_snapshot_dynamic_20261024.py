"""Read-only stdlib replay; no fitting, no new historical evidence."""
from collections import defaultdict
import json
import math
from pathlib import Path

folder = Path(__file__).resolve().parents[1] / "outputs/限额动态攻防_20261023T164923023794Z"
read = lambda name: json.loads((folder / name).read_text(encoding="utf-8"))
result = read("快照内动态攻防结果.json")
early, later = (read(name) for name in ("较早训练预测.json", "较晚动态预测.json"))


def replay(rows, eta):
    epochs, rates, nlls = {}, [], []
    by_day = defaultdict(list)
    for row in rows:
        by_day[row["date"]].append(row)
    for day, matches in sorted(by_day.items()):
        changes = defaultdict(lambda: [[], []])
        for row in matches:
            comp, cutoff = row["competition"], row["fit_before_date"]
            if comp not in epochs or epochs[comp][0] != cutoff:
                epochs[comp] = (cutoff, {})
        for row in matches:
            states = epochs[row["competition"]][1]
            ah, dh = states.get(row["home"], (0., 0.))
            aa, da = states.get(row["away"], (0., 0.))
            current = [row["rates"][0]*math.exp(ah-da), row["rates"][1]*math.exp(aa-dh)]
            rates.append(current)
            nlls.append(math.fsum(mu-y*math.log(mu)+math.lgamma(y+1) for mu,y in zip(current,row["actual_score"])))
            for scorer, conceder, mu, y in ((row["home"], row["away"], current[0], row["actual_score"][0]),
                                           (row["away"], row["home"], current[1], row["actual_score"][1])):
                step = eta*(y-mu)/2
                changes[row["competition"],scorer][0].append(step)
                changes[row["competition"],conceder][1].append(-step)
        for (comp, team), increments in changes.items():
            states = epochs[comp][1]
            previous = states.get(team, (0., 0.))
            states[team] = [math.fsum((previous[i], *increments[i])) for i in (0,1)]
    return rates, nlls


rate_error, nll_error, objective_error = 0., 0., 0.
for track in (early, later):
    rates, nlls = replay(track["original"], result["parameter"]["eta"])
    assert len(rates) == len(track["selected"])
    for mu, nll, saved, source in zip(rates, nlls, track["selected"], track["original"]):
        rate_error = max(rate_error, *(abs(a-b) for a,b in zip(mu,saved["rates"])))
        nll_error = max(nll_error, abs(nll-saved["exact_score_nll"]))
        for field in ("match_id","actual_score","market_prices","market_probabilities"):
            assert saved[field] == source[field]
        assert saved["training_max_date"] < saved["fit_before_date"] == saved["date"]
for item in result["parameter"]["evaluations"]:
    _, nlls = replay(early["original"], item["eta"])
    objective_error = max(objective_error,abs(math.fsum(nlls)/len(nlls)-item["nll"]))
_, late_nlls = replay(later["original"], result["parameter"]["eta"])
aggregate_error = abs(math.fsum(late_nlls)/len(late_nlls)-result["later"]["scores"]["selected"]["exact_score"]["mean_nll"])
assert max(rate_error,nll_error,objective_error,aggregate_error) < 1e-12
print(json.dumps({"unique_training_matches":len(early["original"]),"unique_later_matches":len(later["original"]),
    "objective_values_replayed":len(result["parameter"]["evaluations"]), "maximum_rate_error":rate_error,
    "maximum_NLL_error":nll_error,"maximum_objective_error":objective_error,"aggregate_exact_NLL_error":aggregate_error,
    "readonly_not_new_fits_or_validation_matches":True}))
