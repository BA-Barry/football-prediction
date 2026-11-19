"""One predeclared shot-information arm; no search or prospective replacement."""
import csv
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import numpy as np
from scipy.special import gammaln

from football_model.football_data import parse_row
from football_model.native_experiment import season_directories
from football_model.forecast_diagnostics import moving_block_means
from football_model.probability import score_grid, summarize
from football_model.refresh_experiment import scores_for, prediction_arrays
from football_model.shot_increment import ShotObservation, history_features, OffsetShotModel

ROOT = REPOSITORY_ROOT
SHAPES = ROOT / 'outputs/比分形态诊断_20261023T123154423712Z'
SAVED = ROOT / 'outputs/统计市场固定融合研究_20261024T120045031283Z/统计市场融合结果.json'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sources():
    observations, matches = [], {}
    for season, directory in sorted(season_directories(ROOT).items()):
        path = directory / 'E0.csv'
        for raw in csv.DictReader(path.open(encoding='utf-8-sig', newline='')):
            match, _ = parse_row(raw, competition='E0', season=season,
                                 source_path=str(path),
                                 source_url=f'https://www.football-data.co.uk/mmz4281/{season[2:4]+season[-2:]}/E0.csv')
            if match.match_id in matches:
                raise ValueError('Duplicate source match')
            counts = []
            for key in ('HST', 'AST'):
                text = raw.get(key, '').strip()
                if text and not text.isdecimal():
                    raise ValueError('Invalid shot count')
                counts.append(int(text) if text else None)
            for key, value in zip(('HS', 'AS'), counts):
                if value is not None and raw.get(key, '').strip() and value > int(raw[key]):
                    raise ValueError('Shots on target exceed total shots')
            shots = tuple(counts) if all(v is not None for v in counts) else None
            observations.append(ShotObservation(match.match_id, match.match_date,
                                                 match.home, match.away, shots))
            matches[match.match_id] = match
    return observations, matches


def load_rows(observations, matches):
    saved = json.loads(SAVED.read_text(encoding='utf-8'))
    rows, excluded = [], []
    for period in ('2024-25', '2025-26'):
        scores = {r['match_id']: r for r in saved['seasons'][period]['paired_matches']}
        anchor = date(int(period[:4]), 8, 1)
        with (SHAPES/f'{period}_逐场形态.csv').open(encoding='utf-8-sig', newline='') as handle:
            for raw in csv.DictReader(handle):
                if raw['competition'] != 'E0':
                    continue
                record, score = matches[raw['match_id']], scores[raw['match_id']]
                played = date.fromisoformat(raw['date'])
                cutoff = anchor + timedelta(days=((played-anchor).days//28)*28)
                # This is the original August-1/28-day policy, not a new cadence.
                if (played != record.match_date or raw['home'] != record.home_label
                        or raw['away'] != record.away_label
                        or [int(raw['original_home_goals']), int(raw['original_away_goals'])] != list(record.full_time)):
                    raise ValueError('Source/shape fixture identity conflict')
                rates = [float(raw['original_home_rate']), float(raw['original_away_rate'])]
                forecast = summarize(score_grid(*rates, rho=0))
                p = list(forecast['one_x_two'].values())
                if not np.allclose(p, score['probabilities']['statistical'], rtol=0, atol=1e-10):
                    raise ValueError('Original offset does not replay saved probabilities')
                context = history_features(observations, record.home, record.away, cutoff)
                row = {'match_id': record.match_id, 'date': played.isoformat(),
                       'competition': 'E0', 'home': record.home, 'away': record.away,
                       'period': period, 'fit_before_date': cutoff.isoformat(),
                       'actual_score': list(record.full_time), 'actual_outcome': score['actual_outcome'],
                       'rates': rates, 'forecast': forecast, 'probabilities': p,
                       'shot_features': context['features'], 'shot_context': context,
                       'source_url': record.source_url}
                rows.append(row)
                if context['features'] is None:
                    excluded.append({'match_id': row['match_id'], 'date': row['date'],
                                     'period': period, 'context': context})
    if len(rows) != 756 or len({r['match_id'] for r in rows}) != len(rows):
        raise ValueError('Unexpected original EPL cohort')
    return sorted(rows, key=lambda r: (r['date'], r['match_id'])), excluded


def score_details(rows):
    arrays = prediction_arrays(rows)
    losses = {key: -np.log(p[np.arange(len(y)), y]) for key, (p, y) in arrays.items()}
    nll = []
    for row in rows:
        nll.append(math.fsum(rate-goal*math.log(rate)+float(gammaln(goal+1))
                             for goal, rate in zip(row['actual_score'], row['rates'])))
    losses['exact_score'] = np.asarray(nll)
    summary = scores_for(rows)
    summary['exact_score'] = {'n': len(rows), 'log_loss': float(np.mean(nll))}
    return summary, losses


def main():
    now = datetime.now(timezone.utc)
    target = ROOT / 'outputs' / now.strftime('历史射正增量研究_%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True, exist_ok=False)
    plan = {
        'created_at': now.isoformat(), 'user_approved': '2026-10-25 first historical shot-information contrast',
        'scope': 'EPL only; already-viewed historical seasons, exploratory not untouched validation',
        'train_initial': '2024/25 original out-of-time goal rates and earlier-cutoff shots',
        'evaluate': '2025/26; refit two beta every original 28-day cutoff using strictly earlier target dates',
        'fixed_settings': {'window_matches': 10, 'min_history': 5, 'regularization': .01,
                           'half_life_days': 180, 'new_intercept': False, 'rho': 0},
        'feature_information_cutoff': 'strictly before each original fit_before_date, not the later match date',
        'primary': 'paired complete-score negative log likelihood',
        'secondary': ['1X2 LL/Brier/top1', 'total 0/1/2/3/4+ LL', 'away-minus-two three-way LL'],
        'uncertainty': 'paired noncircular calendar moving blocks 2/4/8 weeks, 2000 repeats; descriptive sensitivity',
        'missing': 'no zero fill; common subset plus original full cohort, retain all exclusion reasons',
        'fit_budget': 'one two-coefficient fit per original 2025/26 cutoff, 0 new team fits, 0 parameter searches',
        'unchanged': 'all saved original probabilities and all 20 prospective records',
        'limitations': ['earlier calendar-date availability assumed, publication times unknown',
                        'past shots not fully historical-opponent-adjusted; current opponent controlled by fixed offset',
                        'EPL-only windows omit cups/Championship and can be stale after promotion',
                        'shots are not xG, goals remain the target; no causal interpretation',
                        'no parameter or model uncertainty mixture; new arm not adopted automatically'],
        'source_definitions': 'https://www.football-data.co.uk/notes.txt',
    }
    write(target/'分析前计划.json', plan)
    observations, matches = sources()
    rows, excluded = load_rows(observations, matches)
    eligible = [r for r in rows if r['shot_features'] is not None]
    full = [r for r in rows if r['period'] == '2025-26']
    old = [r for r in eligible if r['period'] == '2025-26']
    models, revised = {}, []
    for row in old:
        cutoff = row['fit_before_date']
        if cutoff not in models:
            model = OffsetShotModel().fit(eligible, date.fromisoformat(cutoff))
            models[cutoff] = model
        rates = models[cutoff].predict(row['rates'], row['shot_features'])
        forecast = summarize(score_grid(*rates, rho=0))
        revised.append({**row, 'rates': list(rates), 'forecast': forecast,
                        'probabilities': list(forecast['one_x_two'].values()),
                        'offset_rates': row['rates']})
    old_scores, old_losses = score_details(old)
    new_scores, new_losses = score_details(revised)
    full_scores, _ = score_details(full)
    domains = list(new_losses)
    difference = np.column_stack([new_losses[k]-old_losses[k] for k in domains])
    intervals = {str(block): moving_block_means([r['date'] for r in old], difference,
                  block_weeks=block) for block in (2, 4, 8)}
    per_window = []
    for cutoff, model in models.items():
        indices = [i for i, r in enumerate(old) if r['fit_before_date'] == cutoff]
        per_window.append({'cutoff': cutoff, 'n': len(indices),
                           'mean_log_loss_differences': dict(zip(domains, difference[indices].mean(axis=0).tolist())),
                           'parameters': model.parameters()})
    result = {'plan': plan, 'source_matches': len(matches),
              'shots_complete': sum(r.shots_on_target is not None for r in observations),
              'new_team_fits': 0, 'new_two_coefficient_fits': len(models), 'parameter_searches': 0,
              'earlier_original_n': sum(r['period']=='2024-25' for r in rows),
              'earlier_eligible_n': sum(r['period']=='2024-25' for r in eligible),
              'later_original_n': len(full), 'later_common_n': len(old),
              'exclusions': excluded, 'old_full_scores': full_scores, 'old_common_scores': old_scores,
              'new_common_scores': new_scores, 'loss_difference_order': domains,
              'mean_log_loss_differences': dict(zip(domains, difference.mean(axis=0).tolist())),
              'block_intervals': intervals, 'windows': per_window,
              'all_feature_rows': rows, 'new_predictions': revised,
              'max_latest_history_gap_days': max(h['latest_gap_days'] for r in eligible
                                                for h in r['shot_context']['history'].values()),
              'adopted': False}
    write(target/'射正研究结果.json', result)
    report = ['# 历史射正信息增量研究', '',
              '用户批准后的单一研究线：原进球率固定，只加入本队此前射正及对手此前被射正两个共享系数。真实进球为目标，不混入xG，不改比分分布。', '',
              f'源数据英超{len(matches)}场，射正完整{result["shots_complete"]}场。较早2024/25原378场中375场具足历史；较晚2025/26原{len(full)}场中共同{len(old)}场。缺失记录完整保留，未填0。', '',
              f'实际新增两系数拟合{len(models)}次、球队拟合0次、参数搜索0次。沿用正则.01、半衰180天、28天截止。后季每次只用更早赛果更新，属于在线前推，并非后季固定参数。', '',
              '| 指标 | 原模型／共同场次 | 射正增量／共同场次 | 新减旧 |',
              '|---|---:|---:|---:|']
    for key in ('exact_score', 'one_x_two', 'total_goals_0_1_2_3_4plus', 'away_minus_two_three_way'):
        report.append(f'| {key} LL | {old_scores[key]["log_loss"]:.6f} | {new_scores[key]["log_loss"]:.6f} | {result["mean_log_loss_differences"][key]:+.6f} |')
    for metric in ('brier_sum', 'accuracy'):
        report.append(f'| 1X2 {metric} | {old_scores["one_x_two"][metric]:.6f} | {new_scores["one_x_two"][metric]:.6f} | {new_scores["one_x_two"][metric]-old_scores["one_x_two"][metric]:+.6f} |')
    report += ['', 'LL、Brier越低越好；Top1越高越好。完整比分为事前主指标，不能因某个次指标好看而挑选上线。', '',
               f'原完整{len(full)}场1X2 LL为{full_scores["one_x_two"]["log_loss"]:.6f}，不能与新共同{len(old)}场直接比较。', '',
               '| 移动块长度 | 完整比分LL差95%描述区间 | 1X2 LL差95%描述区间 |', '|---|---|---|']
    for length, entry in intervals.items():
        def show(key):
            bounds = entry['interval_95'][domains.index(key)]
            return f'[{bounds[0]:+.6f}, {bounds[1]:+.6f}]'
        report.append(f'| {length}周 | {show("exact_score")} | {show("one_x_two")} |')
    report += ['', '## 时序与局限', '',
               '训练每一行都保留该行当时的OOF进球率；射正窗口严格截止原28天拟合日期。未把当前球队拟合回算给过去样本，也未让新模型比原模型多看截止日后的比赛。训练标准化均值、尺度及系数只来自该次更早训练集。', '',
               f'英超窗口不含英冠和杯赛，升班马可出现陈旧记录；本共同数据最长距窗口截止{result["max_latest_history_gap_days"]}天，不能称为完整近期状态。原进球率控制当场球队／对手与主场，但历史射正窗口的过去对手强度未充分校正。', '',
               '已看过的历史赛季再次用于探索，不能称为全新盲测；移动块区间只反映已保存预测的描述敏感性，没有重演模型选择或参数估计不确定性。无论本次分数方向如何，均不自动改当前20场预测或融合权重。', '',
               '[Football-Data字段说明](https://www.football-data.co.uk/notes.txt)：HST/AST为主客射正数，是赛后记录，本实验只用更早比赛生成输入。', '']
    (target/'射正增量研究说明.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({'directory': str(target), 'fits': len(models), 'common_n': len(old),
                      'differences': result['mean_log_loss_differences'], 'scores': new_scores,
                      'old_scores': old_scores, 'intervals_4_weeks': intervals['4']['interval_95']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
