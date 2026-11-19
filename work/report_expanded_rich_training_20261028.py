"""Read-only checks and Chinese report for the completed expansion; no fits."""
from collections import Counter
from datetime import date, datetime, timezone
import json

import joblib

from expand_rich_training_20261028 import (
    OUT, SOURCE, FEATURE, HELD_CUTOFF, FINAL_CUTOFF, load_gzip, write, weight_summary,
)
from run_npxg_comparison_20261025 import load_source


def table_counts(rows, season_key, comp_key):
    counts = Counter((season_key(r), comp_key(r)) for r in rows)
    return {season: {comp: counts[(season, comp)] for comp in ('E0', 'D1', 'SP1', 'I1', 'F1')}
            for season in sorted({key[0] for key in counts})}


def main():
    source, independent, _, _, _ = load_source()
    original = load_gzip(SOURCE)['rows']
    expansion = load_gzip(OUT/'2021-22新增特征.json.gz')
    merged = load_gzip(OUT/'扩样合并特征.json.gz')['rows']
    summary = json.loads((OUT/'扩样训练结果.json').read_text(encoding='utf8'))
    original_map = {r['match_id']: r for r in original}
    merged_map = {r['match_id']: r for r in merged}
    assert len(merged_map) == len(merged) == 8753
    assert all(merged_map[mid] == row for mid, row in original_map.items())
    assert len(original_map) == 6967
    assert set(merged_map)-set(original_map) == {r['match_id'] for r in expansion['rows']}
    held = {r['match_id'] for r in original if r['period'] == '2025-26'}
    audit = {'checked_at': datetime.now(timezone.utc).isoformat(), 'new_model_fits': 0,
        'original_full_rows_checked_object_equal': len(original), 'merged_unique_matches': len(merged),
        'held_match_n': len(held), 'held_training_overlap': {},
        'feature_history_match_references_checked': 0, 'base_reference_entries_checked': 0}
    for name, expected_n in (('原覆盖_20250801', 5241), ('扩样_20250801', 7027)):
        model = joblib.load(OUT/f'{name}.joblib')
        p = model.parameters()
        assert p['n_matches'] == expected_n
        assert p['training_max_date'] < HELD_CUTOFF.isoformat()
        overlap = held & set(p['reference_match_ids'])
        assert not overlap
        audit['held_training_overlap'][name] = len(overlap)
    for row in merged:
        fd = independent[row['match_id']]
        assert row['actual_score'] == list(fd.full_time)
        assert row['date'] == fd.match_date.isoformat()
        assert row['home'] == fd.home and row['away'] == fd.away
        assert len(row[FEATURE][0]) == len(row[FEATURE][1]) == 121
        assert row['fit_before_date'] <= row['forecast_before_date'] < row['date']
        for history in row['fresh_feature_history']['history'].values():
            for mid in history['used_match_ids']:
                assert independent[mid].match_date.isoformat() < row['forecast_before_date']
                assert mid != row['match_id']
                audit['feature_history_match_references_checked'] += 1
    params = json.loads((OUT/'新增基础模型参数.json').read_text(encoding='utf8'))
    for key, pair in params.items():
        comp, cutoff = key.split(':', 1)
        gp, xp = pair['goal'], pair['xg']
        assert gp['reference_match_ids'] == xp['reference_match_ids']
        for mid in gp['reference_match_ids']:
            fd = independent[mid]
            assert fd.competition == comp and fd.match_date.isoformat() < cutoff
            assert fd.goal_fit_exclusion is None and mid in source
            audit['base_reference_entries_checked'] += 1
    excluded_counts = Counter(r['competition'] for r in expansion['unavailable'])
    missing_teams = Counter(team for r in expansion['unavailable']
                            for team in set(r['unseen_teams']['goal'] + r['unseen_teams']['xg']))
    for row in expansion['unavailable']:
        key = f"{row['competition']}:{row['fit_before_date']}"
        for arm in ('goal', 'xg'):
            required = {row['home'], row['away']}
            assert sorted(required-set(params[key][arm]['teams'])) == row['unseen_teams'][arm]
            assert row['unseen_teams'][arm]
    assert len(expansion['unavailable']) == 36
    assert len(expansion['rows'])+len(expansion['unavailable']) == expansion['target_source_n'] == 1822
    audit['excluded_unknown_teams_by_league'] = dict(excluded_counts)
    audit['excluded_unknown_team_occurrences'] = dict(missing_teams)
    audit['all_checks_passed'] = True
    write(OUT/'不重训复核.json', audit)
    counts = {
        'fd_all_by_season_league': table_counts(independent.values(), lambda r:r.season, lambda r:r.competition),
        'fd_goal_eligible_by_season_league': table_counts(
            [r for r in independent.values() if r.goal_fit_exclusion is None], lambda r:r.season, lambda r:r.competition),
        'exact_xg_joined_by_season_league': table_counts(source.values(),
            lambda r: independent[r.match_id].season, lambda r:r.competition),
        'original_rich_lead3_by_period_league': table_counts(original, lambda r:r['period'], lambda r:r['competition']),
        'expanded_rich_lead3_by_period_league': table_counts(merged, lambda r:r['period'], lambda r:r['competition']),
        'old_final_weight_summary': weight_summary(original, FINAL_CUTOFF),
        'expanded_final_weight_summary': summary['final_weight_summary'],
    }
    write(OUT/'按联赛赛季规模与权重.json', counts)
    old_s, new_s = (summary['scores'][name] for name in ('原覆盖_20250801', '扩样_20250801'))
    lines = ['# 本轮扩样训练与所学', '',
        '已实际完成一次同语义扩样及三次固定树拟合，全流程92.26秒。没有搜索参数，没有查询当前赛果，没有改变已发布预测。单次拟合耗时本轮未单独计时，不补造。', '',
        '## 数据规模与用途', '',
        f"- 五联赛2020/21—2025/26 Football-Data唯一记录：{len(independent):,}场，其中进球模型合格记录{sum(r.goal_fit_exclusion is None for r in independent.values()):,}场。",
        f"- 与Understat逐场精确联结的xG/NPxG共同记录：{len(source):,}场；它们是基础模型和历史特征的来源，并非全部都已有合格的提前3天元模型训练快照。",
        '- 原丰富121特征的提前3天训练快照6,967场；本次新增2021/22赛季1,786场，合计8,753场、17,506个主客进球目标。',
        '- 2021/22来源1,822场，36场因当时的基础模型未见至少一方球队而明确不可用；没有把未知球队系数填0。其余缺少足量历史的特征仍保留None。', '',
        '| 快照赛季 | E0英超 | D1德甲 | SP1西甲 | I1意甲 | F1法甲 | 合计 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for period, row in counts['expanded_rich_lead3_by_period_league'].items():
        lines.append('| '+period+' | '+' | '.join(str(row[c]) for c in ('E0','D1','SP1','I1','F1'))+f" | {sum(row.values())} |")
    lines += ['', '## 如何训练与隔离', '',
        '新增赛季采用原有定义：F=比赛日−3天；P为赛季8月1日锚定的28天参数日期，P≤F；球队进球与xG基础模型仅使用P之前、同联赛、同一组精确联结的合格记录；丰富历史特征仅使用F之前的比赛。基础模型设置仍为alpha=0.01、180天衰减。55组联赛/日期参数，共110次轻量基础拟合已保存。', '',
        '三次元模型拟合均使用原ExtraTrees设置（120棵树、深度6、叶最少60、特征比例0.8、原随机种子及180天衰减），没有试权重或参数网格：', '',
        '1. 截止2025-08-01，原覆盖5,241场训练，预测2025/26同一组1,726场。',
        '2. 同一截止、加入早一季后7,027场训练，预测相同1,726场。',
        '3. 截止2026-08-01，以扩样8,753场拟合最终研究候选；这份最终模型包含2025/26标签，不能拿来计算2025/26持出成绩。', '',
        '独立读取复核确认：旧6,967个完整对象逐行一致；两份2025比较模型与持出1,726场的训练ID交集均为0；全部丰富特征引用和新增基础训练引用都早于各自截止。最终模型保存后重载预测一致。原始来源的历史首次发布时间、后续修订时间未重建，因此仍是日期级回溯研究。', '',
        '## 固定截止结果', '',
        '| 指标 | 原覆盖 | 扩样 |', '|---|---:|---:|',
        f"| 胜平负首选命中 | {old_s['one_x_two']['accuracy']:.2%} | {new_s['one_x_two']['accuracy']:.2%} |",
        f"| 胜平负LogLoss（低好） | {old_s['one_x_two']['log_loss']:.6f} | {new_s['one_x_two']['log_loss']:.6f} |",
        f"| 胜平负Brier总和（低好） | {old_s['one_x_two']['brier_sum']:.6f} | {new_s['one_x_two']['brier_sum']:.6f} |",
        f"| 完整比分LogLoss（低好） | {old_s['exact_score']['log_loss']:.6f} | {new_s['exact_score']['log_loss']:.6f} |",
        f"| 总球0/1/2/3/4+ LogLoss（低好） | {old_s['total_goals_0_1_2_3_4plus']['log_loss']:.6f} | {new_s['total_goals_0_1_2_3_4plus']['log_loss']:.6f} |", '',
        '胜平负只多对3场，其余指标变化也很小；不能据此宣称整体提准。这些年份已多次用于研发，不是全新盲测，也不是当前比赛准确率；本次固定截止一整季不更新元模型，不能直接与此前22窗滚动成绩作高低比较。完整各联赛结果全部保留，未挑选有利联赛。', '',
        '## 扩样后最值得注意的限制', '',
        f"截至2026-08-01，180天权重下，原6,967场的Kish权重有效N为{counts['old_final_weight_summary']['effective_match_n_kish']:.1f}，扩至8,753场后为{summary['final_weight_summary']['effective_match_n_kish']:.1f}；新增2021/22只占全体目标权重{summary['final_weight_summary']['by_period']['2021-22']['weight_share']:.3%}。有效N仅描述权重集中度，并未处理比赛相关性，不能等同独立样本量。",
        '', '旧赛季仍会改变未加权中位数填补，以及按行数控制的树叶分割；不能把它理解为完全没有作用。不过，继续单纯堆更早年份，在现有短衰减下很难获得成比例信息增量。要让更早数据更多地学习通用规律、又让当前球队状态保持较新，涉及训练目标或结构的选择，需要另行讨论，不能通过暗改半衰期完成。', '',
        '本轮停止节点已完成：同语义扩样、三次固定拟合、持出核对及保存研究候选。后续优先补齐赛前信息覆盖与结构化LLM协作记录，不反复换参数追逐这组旧赛季分数。', '',
        'LLM没有在这8,753场上训练：尚缺同提前量、可追溯的历史新闻与真正赛前LLM记录。历史比分/特征可以训练统计模型，不能代替历史伤停、教练表述与赛程情境。当前模型、LLM、来源证据和人工综合分析应分别保留。', '']
    (OUT/'扩样训练与所学.md').write_text('\n'.join(lines), encoding='utf8')
    print(json.dumps({'audit': audit, 'original_kish': counts['old_final_weight_summary']['effective_match_n_kish'],
        'expanded_kish': summary['final_weight_summary']['effective_match_n_kish'],
        'added_weight_share': summary['final_weight_summary']['by_period']['2021-22']['weight_share']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
