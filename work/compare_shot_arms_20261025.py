"""Describe already-fitted shot arms on identical fixtures; no new selection."""
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import json
import numpy as np

from run_shot_increment_20261025 import score_details, write

ROOT = REPOSITORY_ROOT
TARGET = ROOT / 'outputs/射正攻防校正研究_20261024T163823918831Z'


def main():
    raw = json.loads((ROOT/'outputs/历史射正增量研究_20261024T161730592852Z/射正研究结果.json').read_text(encoding='utf-8'))
    adjusted = json.loads((TARGET/'射正攻防研究结果.json').read_text(encoding='utf-8'))
    tracks = {
        'original': {r['match_id']: r for r in adjusted['original_with_auxiliary'] if r['period']=='2025-26'},
        'raw_last10': {r['match_id']: r for r in raw['new_predictions']},
        'opponent_adjusted': {r['match_id']: r for r in adjusted['new_predictions']},
    }
    ids = sorted(set.intersection(*(set(t) for t in tracks.values())), key=lambda k: (tracks['original'][k]['date'], k))
    for key in ids:
        entries = [track[key] for track in tracks.values()]
        if any((r['date'],r['actual_score'],r['fit_before_date']) != (entries[0]['date'],entries[0]['actual_score'],entries[0]['fit_before_date']) for r in entries):
            raise ValueError('Same-fixture identity conflict')
    scores = {arm: score_details([rows[k] for k in ids])[0] for arm, rows in tracks.items()}
    full_rows = adjusted['new_predictions']
    margins = [r['actual_score'][1]-r['actual_score'][0] for r in full_rows]
    handicap_counts = {'away_wins_by_3plus': sum(m>2 for m in margins),
                       'away_wins_by_exactly_2': sum(m==2 for m in margins),
                       'away_does_not_win_by_2': sum(m<2 for m in margins)}
    result = {'purpose': 'descriptive same-fixture comparison of already-fitted arms; no new selection or blind validation',
              'new_fits': 0, 'parameter_searches': 0, 'n': len(ids), 'match_ids': ids,
              'scores': scores, 'later_378_handicap_actual_counts': handicap_counts,
              'new_versus_old_goal_rate_ratios': {
                  'min': float(np.min([np.asarray(r['rates'])/r['offset_rates'] for r in full_rows])),
                  'max': float(np.max([np.asarray(r['rates'])/r['offset_rates'] for r in full_rows]))}}
    write(TARGET/'射正两种结构同场对照.json', result)
    report = ['# 两种射正结构的同场描述', '',
              f'仅重读已拟合结果，{len(ids)}场交集，新增拟合0次、搜索0次。新结构原378场与简单均值375场总分不能直接比较，因此另列本表；不按本表再选参数。', '',
              '| 指标 | 原模型 | 简单最近10场 | 射正攻防校正 |', '|---|---:|---:|---:|']
    for domain in ('exact_score','one_x_two','total_goals_0_1_2_3_4plus','away_minus_two_three_way'):
        report.append('| '+domain+' LL | '+' | '.join(f'{scores[a][domain]["log_loss"]:.6f}' for a in tracks)+' |')
    report += ['', '| 1X2最高概率命中率 | '+' | '.join(f'{scores[a]["one_x_two"]["accuracy"]:.2%}' for a in tracks)+' |', '',
               f'完整378场客队让两球实际三类为：净胜至少3球{handicap_counts["away_wins_by_3plus"]}场、净胜正好2球{handicap_counts["away_wins_by_exactly_2"]}场、未净胜2球{handicap_counts["away_does_not_win_by_2"]}场。高“不穿盘”命中率不能当投注收益或客让两球让胜命中率。', '',
               '校正后射正增量仍未改善事前完整比分主指标，因此不以让两球的局部改善替换原模型。完整378场让球LL差4／8周区间低于0，但2周区间跨0；总球LL差8周区间反而高于0。它们是多项次指标的描述性结果，不能挑单一块长宣称可靠优势。', '']
    (TARGET/'射正两种结构同场对照说明.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps({'n':len(ids),'scores':scores,'handicap_actual_counts':handicap_counts,'ratio':result['new_versus_old_goal_rate_ratios']},ensure_ascii=False))


if __name__=='__main__':
    main()
