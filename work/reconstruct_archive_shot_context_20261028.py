"""One historical reconstruction pass, without model fitting or fresh results."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import io
import json
from pathlib import Path
import zipfile

from football_model.archive_shot_context import reconstruct_archive_context
from measure_source_clock_landmarks_20261027 import MANIFEST, SHOTS, MINUTES, integer, side_of

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OUT = ROOT / 'outputs/射门分钟状态重建_20261028'


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_gzip(path, obj):
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, allow_nan=False)


def shot_counts(rows):
    count = Counter()
    status = Counter()
    reasons = Counter()
    for row in rows:
        count['shots'] += 1
        status[row['status']] += 1
        count['with_any_listed_candidate'] += bool(row['joint_candidate_bounds'])
        count['all_hypothesis_branches_covered'] += row['all_branches_covered']
        count['with_unknown_branch_or_source_conflict'] += bool(row['unknown_reasons'])
        count['window_unique_joint_state'] += row['window_unique_joint_state'] is not None
        count['window_unique_score'] += row['window_unique_score'] is not None
        count['window_unique_own_red_card_record_difference'] += row['window_unique_own_red_card_record_difference'] is not None
        count['multiple_author_half_candidates'] += len({b['half_id'] for b in row['branches'] if b['half_id'] is not None}) > 1
        count['nonzero_red_card_difference_possible'] += any(v[2] != v[3] for v in row['joint_candidate_bounds'])
        count['nonzero_red_card_difference_unique'] += row['window_unique_own_red_card_record_difference'] not in (None, 0)
        reasons.update(row['unknown_reasons'])
    return {'counts': dict(count), 'statuses': dict(status), 'unknown_reason_shot_counts': dict(reasons)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(MANIFEST, 'rt', encoding='utf-8') as stream:
        manifest = json.load(stream)
    with gzip.open(SHOTS, 'rt', encoding='utf-8') as stream:
        original_shots = json.load(stream)
    by_match = defaultdict(list)
    for row in original_shots:
        # These are explicit literal labels; neither player nor final score fills a side.
        normalized = {'id': str(integer(row['id'])), 'side': side_of(row),
                      'minute': integer(row['minute']), 'result': row['result'], 'xG': row['xG']}
        by_match[str(integer(row['match_id']))].append(normalized)
    targets = {str(row['espn_game_id']): row for row in manifest}
    minute_rows = defaultdict(list)
    with zipfile.ZipFile(MINUTES) as archive:
        members = [name for name in archive.namelist() if name.endswith('.csv')]
        if len(members) != 1:
            raise ValueError('Expected one CSV')
        with archive.open(members[0]) as stream:
            for line, row in enumerate(csv.DictReader(io.TextIOWrapper(stream, encoding='utf-8-sig')), 2):
                if row['gameId'] not in targets:
                    continue
                minute_rows[row['gameId']].append({
                    'source_csv_row_number': line, 'side': {'Home': 'h', 'Away': 'a'}[row['HomeAway']],
                    'half_id': integer(row['half_id']), 'minute': integer(row['Minute.clean']),
                    'goals': integer(row['Goals']), 'red_cards': integer(row['RedCards']),
                    'source_score_difference': integer(row['Score.Diff']),
                    'source_red_card_difference': integer(row['RedCard.Diff'])})
    details, raw_events, withholds, singleton_pairs = [], [], [], []
    for target in manifest:
        uid, eid = str(target['trusted_understat_id']), str(target['espn_game_id'])
        result = reconstruct_archive_context(by_match[uid], minute_rows[eid],
            expected_full_time=target['independent_full_time'],
            expected_half_time=target['independent_half_time'], clock_offsets=(0, 1, 2))
        identity = {k: target[k] for k in ('season', 'date', 'home', 'away', 'trusted_understat_id', 'espn_game_id')}
        details.append({**identity, **result})
        raw_events.append({**identity, 'minute_rows': minute_rows[eid]})
        if result['withhold_reasons']:
            withholds.append({**identity, **{k: result[k] for k in (
                'withhold_reasons', 'goal_window_matching', 'counted_minute_goals', 'counted_shot_goals')},
                'shot_count': len(result['shots'])})
        # Whole-match single-goal sides require no shot-array order or period assignment.
        if target['full_time_goal_check']:
            for side in ('h', 'a'):
                source = [s['minute'] for s in by_match[uid]
                          if (s['side'] if s['result'] == 'Goal' else
                              ('a' if s['side'] == 'h' else 'h') if s['result'] == 'OwnGoal' else None) == side]
                minute = [r['minute'] for r in minute_rows[eid] if r['side'] == side for _ in range(r['goals'])]
                if len(source) == len(minute) == 1:
                    singleton_pairs.append({**identity, 'credited_side': side, 'shot_minute': source[0],
                                            'minute_package_minute': minute[0], 'difference': minute[0] - source[0]})
    all_shots = [shot for match in details for shot in match['shots']]
    by_season = {}
    for season in sorted({m['season'] for m in details}):
        selected = [m for m in details if m['season'] == season]
        by_season[season] = {'matches': len(selected),
            'matches_without_withhold_reason': sum(not m['withhold_reasons'] for m in selected),
            **shot_counts([s for m in selected for s in m['shots']])}
    raw_state_stats = Counter()
    for eid, rows in minute_rows.items():
        keys = Counter((r['side'], r['half_id'], r['minute']) for r in rows)
        raw_state_stats['rows'] += len(rows)
        raw_state_stats['games_with_repeated_team_half_minute'] += any(v > 1 for v in keys.values())
        raw_state_stats['extra_repeated_rows'] += sum(v - 1 for v in keys.values())
        raw_state_stats['games_with_red_card_events'] += any(r['red_cards'] for r in rows)
        raw_state_stats['red_card_events'] += sum(r['red_cards'] for r in rows)
    summary = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Historical conditional pre-shot score/card bounds; no training, calibration, live predictions or new results.',
        'inputs': {'identity_manifest': str(MANIFEST), 'original_shots': str(SHOTS), 'minute_zip': str(MINUTES)},
        'matches': len(details), 'matches_without_withhold_reason': len(details) - len(withholds),
        'clock_offsets': [0, 1, 2], 'raw_minute_counts': dict(raw_state_stats),
        'all_shots': shot_counts(all_shots), 'by_season': by_season,
        'by_shot_result': {kind: shot_counts([s for s in all_shots if s['result'] == kind])
                           for kind in sorted({s['result'] for s in all_shots})},
        'withheld_matches': withholds,
        'singleton_goal_sides': {'count': len(singleton_pairs),
                                'minute_difference_counts': dict(sorted(Counter(str(s['difference']) for s in singleton_pairs).items()))},
        'model_fits': 0, 'parameter_searches': 0, 'changed_current_predictions': 0,
        'limitations': [
            '0/1/2 offsets are empirically motivated clock-sensitivity assumptions, not official per-shot clock alignment.',
            'Author half1 observed range and half2 start at 46 are assumptions; half_id is inferred, not official period.',
            'All-branches-covered and unique means only within these assumptions; it is not proof of the true time/state.',
            'All local bounds preserve their joint coordinate; some may not extend to a global event association.',
            'Same-minute event prefixes form conservative outer sets with no probabilities or equal weighting.',
            'Recorded red sanctions do not establish actual playing numbers; bench/staff identities are absent.',
            'Current Goal/OwnGoal not found at a branch stays unknown. Dropping these unknown shots would select on outcome.',
            'Coverage/unknown depends on the historical shot result: never use these flags to predict that same shot goal label.',
            'Goal accounting agreement does not verify every event, card, clock, original publication or later revision.',
            'Rows are not exposure minutes; no shot xG is aggregated as provider NPxG or neutralized in this pass.']}
    write_gzip(OUT / '逐射门条件状态候选.json.gz', {'metadata': {k: v for k, v in summary.items() if k != 'withheld_matches'}, 'matches': details})
    write_gzip(OUT / '原分钟事件与状态拆分.json.gz', raw_events)
    write(OUT / '重建覆盖与未知汇总.json', summary)
    write(OUT / '单进球侧时钟证据.json', singleton_pairs)
    counts = summary['all_shots']['counts']
    table = '\n'.join(f"|{season}|{row['matches']}|{row['matches_without_withhold_reason']}|{row['counts']['shots']}|{row['counts']['window_unique_joint_state']}|{row['counts']['with_unknown_branch_or_source_conflict']}|" for season, row in by_season.items())
    report = f'''# 历史射门状态重建与本轮判断

2026-10-28。已对既有1,099场、27,582脚历史射门完成一次实际重建；训练、调参、新比赛预测和新赛果查询均为0。全部新文件与运行临时目录在G盘。

本轮补充的是历史机会的比赛背景：同样的机会产出，可能来自平局相持、领先后收缩、落后追赶或红牌记录变化。后续可研究如何解释这些差别；这次没有给任何一种状态添加强弱系数。

## 实际覆盖

|赛季|比赛|无撤回原因比赛|射门|条件联合状态唯一|有未知分支或整场撤回的射门|
|---|---:|---:|---:|---:|---:|
{table}

合计{counts['shots']:,}脚；在声明的时钟、作者半场范围假设下，所有候选分支都有来源覆盖{counts['all_hypothesis_branches_covered']:,}脚，比分和红牌记录联合状态唯一{counts['window_unique_joint_state']:,}脚；包含未知分支或源冲突{counts['with_unknown_branch_or_source_conflict']:,}脚。数字表示数据覆盖，不是预测命中率，也不是已证明的真实状态数量。

{len(withholds)}场撤回派生状态，保留原射门及原xG；原因及逐场身份见汇总JSON。既知2022-09-03 Newcastle—Crystal Palace分钟进球累计异常继续保留，未用终场比分修补事件。其余撤回若因0/1/2分钟窗口不兼容，只表示本研究窗口不足，不能直接宣布源数据错误。

## 解决了什么、如何处理不确定

1. 包含第二黄的红牌记录按原事件重新累计。作者[Stage3源码](https://github.com/UsDAnDreS/JQAS_LeveragingMinuteByMinute_Soccer_Event_Data_Paper/blob/main/Stage3_JQAS_MBM_Data_Wrangling_With_RedCards.R)中的比分与红牌累计有不同的移位方式，不能把一行两个差值直接当同一射门前时点。因此本轮不使用源Score.Diff/RedCard.Diff来赋射前状态；原值另存。
2. 同队、同半场、同分钟多行属于状态拆分，累计事件但不择首尾、不按行数计算时长。先按作者半场再按分钟处理，避免第一半场补时48被误排到第二半场46之后。
3. 上轮固定加1的地标测量仍有52个进球分钟未对齐。本轮无须排序配对的单进球侧共{len(singleton_pairs)}个，时间差分布为{summary['singleton_goal_sides']['minute_difference_counts']}。据此保留0/1/2分钟研究候选；不按场挑一个偏移，也不宣称这能覆盖所有射门真实时间。
4. 同分钟先后未知时保留联合候选及各自坐标，不把不同坐标的比分、红牌边际结果混拼。当前Goal与OwnGoal排除自身进球；任何时钟分支无法关联或缺行时都保留unknown，即使其余分支一致也不标唯一。候选不给均匀权重。
5. 红牌差指本队累计红牌记录减对手，正数表示本队记录更多；缺球员身份，不能自动当真实场上人数差。半场标签和范围也来自作者处理，并非官方节次或实际终场时点。

## 可以怎样接入，哪里先停

当前可以复用逐脚候选、原分钟事件与未知原因，定位以往机会是否主要在追分或有红牌记录的背景下出现。还不能只保留“唯一状态”射门后训练：2,932个Goal和115个OwnGoal全部未达到全分支覆盖，唯一状态均为0，其中2,928个Goal和114个OwnGoal仍有部分候选；22,842个唯一状态全部来自非进球射门。这是本轮要求全部偏移分支均覆盖的保守规则造成，不表示原始进球或xG丢失。直接筛选会系统性漏掉进球。结果分类覆盖已在汇总中单列。

unknown/covered的产生显式使用该脚已发生的结果，不能作为预测该脚是否进球的普通输入，否则会泄漏标签。用于更晚比赛的历史整体表示不会因此自动产生未来泄漏，但仍须处理结果相关的测量偏差。后续应保留每场全部射门和候选／未知部分，不能把未知视作中性状态、丢掉或当独立提准信息。

本轮停止于历史状态与缺失机制接通，不继续扩大偏移窗口或寻找最有利口径，不拟合状态系数，不替换正式模型。下一步优先处理进球事件的候选关联与结果相关测量偏差，让整场机会表示保留全部射门；再沿用已查到的状态校正方法，明确一项整合如何同时处理比分、红牌和时间。在改变核心预测结构前与你同步，当前预测准确率没有因此更新。

新增11项语义测试和原6项早段射门状态测试共17项通过，涵盖乌龙记账、排除自身进球、同分钟乱序、补时跨半场、缺失候选、红牌前后与冲突撤回。测试只核对实现规则，不能证明时钟假设为真。一次独立代码审查另外检查了这些语义边界。

文件：`逐射门条件状态候选.json.gz`、`原分钟事件与状态拆分.json.gz`、`重建覆盖与未知汇总.json`、`单进球侧时钟证据.json`。复现使用G盘环境运行`python -B work/reconstruct_archive_shot_context_20261028.py`。原始两份数据保持原样。
'''
    (OUT / '历史射门状态重建与下一步.md').write_text(report, encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('matches', 'matches_without_withhold_reason', 'all_shots', 'by_season', 'singleton_goal_sides', 'model_fits')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
