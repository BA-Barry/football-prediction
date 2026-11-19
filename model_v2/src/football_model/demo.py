"""Synthetic lineup sensitivity demonstration; never a real-match prediction."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from .domain import Fixture, GoalRates, Lineup, LineupScenario, PlayerEffect
from .probability import forecast_scenarios


def synthetic_inputs():
    as_of = datetime(2026, 10, 22, 8, tzinfo=timezone.utc)
    positions = ["GK"] + ["DF"] * 4 + ["MF"] * 3 + ["FW"] * 3
    effects = {}

    def reference(prefix):
        ids = [f"{prefix}_{position}_{i}" for i, position in enumerate(positions)]
        roles = dict(zip(ids, positions))
        minutes = {p: 90.0 for p in ids}
        sub = f"{prefix}_FW_sub"
        roles[sub] = "FW"
        minutes[ids[-1]], minutes[sub] = 75, 15
        for p in roles:
            effects[p] = PlayerEffect(p, 0, 0, as_of - timedelta(days=1), 1500)
        team = "Synthetic H" if prefix == "H" else "Synthetic A"
        return Lineup(minutes, roles, as_of - timedelta(days=1), "synthetic training reference", team), ids[-1], sub

    home, striker, substitute = reference("H")
    away, _, _ = reference("A")
    effects[striker] = replace(effects[striker], attack_log_rate=0.24)
    effects[substitute] = replace(effects[substitute], attack_log_rate=0.02)
    away_gk = "A_GK_0"
    effects[away_gk] = replace(effects[away_gk], defense_log_rate=0.14)
    effects["A_GK_sub"] = PlayerEffect("A_GK_sub", 0, -0.03, as_of - timedelta(days=1), 450)

    missing_minutes = dict(home.minutes)
    missing_minutes[striker], missing_minutes[substitute] = 0, 90
    home_missing = Lineup(missing_minutes, home.positions, as_of, "synthetic striker unavailable", home.team)
    backup_minutes, backup_roles = dict(away.minutes), dict(away.positions)
    backup_minutes[away_gk], backup_minutes["A_GK_sub"] = 0, 90
    backup_roles["A_GK_sub"] = "GK"
    away_backup = Lineup(backup_minutes, backup_roles, as_of, "synthetic goalkeeper replacement", away.team)
    fixture = Fixture("SYNTHETIC-NOT-A-REAL-MATCH", "Synthetic H", "Synthetic A",
                      "SYNTHETIC", as_of + timedelta(hours=12))
    rates = GoalRates(1.65, 1.2, as_of - timedelta(days=1), "synthetic assumed conditional-reference rates, not fitted",
                      fixture.match_id)
    scenarios = [
        LineupScenario("reference_expected_minutes", 0.55, home, away),
        LineupScenario("home_striker_out_replacement_plays", 0.30, home_missing, away),
        LineupScenario("striker_out_and_away_backup_goalkeeper", 0.15, home_missing, away_backup),
    ]
    return fixture, as_of, rates, home, away, effects, scenarios


def run_demo(output_dir: Path):
    fixture, as_of, rates, home, away, effects, scenarios = synthetic_inputs()
    result = forecast_scenarios(fixture, as_of, rates, home, away, effects, scenarios, rho=-0.05)
    result = {
        "status": "SYNTHETIC DEMONSTRATION ONLY — NO REAL TRAINING OR VALIDATED PREDICTIVE SKILL",
        "parameter_note": "Rates, player effects and scenario weights are invented to exercise the implementation",
        **result,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "阵容情景演示_合成数据.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 阵容情景演示（合成数据）", "",
             "全部球队、进球率、球员系数与情景权重均为虚构，用于验证程序行为。不是本轮比赛预测，也不证明预测效果。", "",
             "| 情景 | 权重 | 主胜 | 平局 | 客胜 | 主队预期进球 |", "|---|---:|---:|---:|---:|---:|"]
    for s in result["scenarios"]:
        p = s["one_x_two"]
        lines.append(f"| {s['label']} | {s['weight']:.0%} | {p['home']:.2%} | {p['draw']:.2%} | {p['away']:.2%} | {s['expected_goals']['home']:.3f} |")
    p = result["summary"]["one_x_two"]
    lines += [f"| 概率加权后的完整比分分布 | 100% | {p['home']:.2%} | {p['draw']:.2%} | {p['away']:.2%} | {result['summary']['expected_goals']['home']:.3f} |", "",
              "程序先分别计算各情景比分分布，再加权合并；没有先平均进球率再假装成单一情景。", "",
              "本演示的情景内部使用预计分钟近似，条件参考进球率也是直接假定值；没有把未校准的球队历史平均率当成条件率。新接口另提供固定预计换人路径的分时段积分，但未学习现实换人规律。", "",
              "情景最小/最大概率仅表示阵容假设变化范围，不是统计置信区间。半全场和参数置信区间当前留空，等待独立训练和验证。"]
    (output_dir / "阵容情景演示_说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result, destination


if __name__ == "__main__":
    task_root = Path(__file__).resolve().parents[3]
    result, destination = run_demo(task_root / "outputs" / "model-v2")
    print(json.dumps({"output": str(destination), "status": result["status"],
                      "one_x_two": result["summary"]["one_x_two"]}, ensure_ascii=False))
