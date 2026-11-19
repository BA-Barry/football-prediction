"""Re-evaluate original saved Poisson rates against confirmed scores, no fit."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
PROJECTION = ROOT / 'outputs/批量球队投影_20261023T175223326233Z/批量球队投影与来源.json'
SETTLED = ROOT / 'outputs/五线赛果评分_20261024T150438997728Z/原赛前记录及新增赛果.json'


def main():
    projection = json.loads(PROJECTION.read_text(encoding='utf-8'))
    settled = json.loads(SETTLED.read_text(encoding='utf-8'))
    by_id = {r['fixture']['match_id']: r for r in projection['fixtures']}
    rows = []
    for fid, result in settled['results'].items():
        original = by_id[fid]
        h, a = result['home_goals'], result['away_goals']
        lh, la = original['rates']['home'], original['rates']['away']
        nll = lh + la - h * math.log(lh) - a * math.log(la) + math.lgamma(h + 1) + math.lgamma(a + 1)
        p = math.exp(-nll)
        total = h + a
        total_prob = math.exp(-(lh + la) + total * math.log(lh + la) - math.lgamma(total + 1))
        mode_h, mode_a = math.floor(lh), math.floor(la)
        mode_prob = math.exp(-(lh + la) + mode_h * math.log(lh) + mode_a * math.log(la)
                             - math.lgamma(mode_h + 1) - math.lgamma(mode_a + 1))
        order = ('home', 'draw', 'away')
        label = order[0 if h > a else 1 if h == a else 2]
        probs = original['forecast']['one_x_two']
        outcome_prob = probs[label]
        outcome_loss = -math.log(outcome_prob)
        row = {'fixture_id': fid, 'fixture': original['fixture'], 'original_rates': original['rates'],
               'original_generated_at': original['track']['generated_at'], 'actual_score': [h, a],
               'exact_score_probability': p, 'full_score_nll': nll,
               'actual_outcome_probability': outcome_prob, 'outcome_nll': outcome_loss,
               'score_given_outcome_nll': nll - outcome_loss,
               'actual_total_goals_probability': total_prob,
               'modal_score_derived_from_original_rates': [mode_h, mode_a],
               'modal_score_probability': mode_prob,
               'original_published_one_x_two': probs,
               'interpretation': 'Derived now from unchanged original saved rates; modal score was not a newly generated prematch recommendation.'}
        if not math.isclose(outcome_loss + row['score_given_outcome_nll'], nll, abs_tol=1e-12):
            raise ValueError('Score/outcome likelihood accounting differs')
        rows.append(row)
    now = datetime.now(timezone.utc)
    target = ROOT / 'outputs' / now.strftime('原比分分布复核_%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True, exist_ok=False)
    (target / '原分布与真实比分.json').write_text(json.dumps({
        'generated_at': now.isoformat(), 'source_projection': str(PROJECTION), 'source_results': str(SETTLED),
        'new_fits': 0, 'new_predictions': 0, 'parameter_changes': 0, 'rows': rows},
        ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    report = ['# 两场1X2方向未中，原比分分布是否也异常？', '',
              '只复算原赛前保存的主客进球率，不重训、不修改概率、不把赛后计算冒充赛前比分推荐。原1X2最高概率方向两场未中这一事实保持不变。', '',
              '| 比赛 | 真实比分 | 原主/客进球率 | 真实比分原概率 | 真实总进球原概率 | 原分布最大单格比分 |',
              '|---|---|---|---:|---:|---|']
    for r in rows:
        title = r['fixture']['home'] + '—' + r['fixture']['away']
        report.append(f"| {title} | {r['actual_score'][0]}-{r['actual_score'][1]} | {r['original_rates']['home']:.3f}/{r['original_rates']['away']:.3f} | {r['exact_score_probability']:.2%} | {r['actual_total_goals_probability']:.2%} | {r['modal_score_derived_from_original_rates'][0]}-{r['modal_score_derived_from_original_rates'][1]}（{r['modal_score_probability']:.2%}） |")
    report += ['', '原独立Poisson分布中，伊普斯维奇1比1森林正是最大概率的单个比分，但所有客胜比分加在一起的概率可以大于全部平局概率。这不矛盾：1X2先把不同比分归为三类，再选最大类；最大单格不一定属于最大类。该复算不能把原1X2方向失误改记为命中，也不能事后改评分目标。', '',
               '维拉主胜的原概率24.79%，其中2比1只是主胜条件下的一种比分。仅凭其发生，无法分清是强度层、比分分布、输入缺失或正常比赛随机性造成了误差。', '',
               '完整比分NLL = 真实胜平负NLL + 给定胜平负后的真实比分NLL，是严格的概率账本。它可以把失误定位到需要研究的预测目标，但不是把误差分成已证明的因果份额。当前LLM只给1X2，不能用其三项概率补造比分或总进球成绩。', '',
               '后续按原先目标同时保留完整比分、总进球和1X2评分；不能因为某个玩法事后更好看就换成主成绩。两场不足以确认当前比分分布已经校准，也不足以证明应更换为神经网络或手动增加平局。', '']
    (target / '原比分分布复核说明.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({'directory': str(target), 'matches': len(rows),
                      'derived_rows': [{k: r[k] for k in ('fixture_id', 'exact_score_probability', 'actual_total_goals_probability', 'modal_score_derived_from_original_rates')} for r in rows]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
