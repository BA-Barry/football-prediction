"""Approved single-shot-strength increment; fixed budget and chronological folds."""
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from football_model.forecast_diagnostics import moving_block_means
from football_model.probability import score_grid, summarize
from football_model.shot_strength import DateTeamShotModel, ShotRateGoalIncrement
from run_shot_increment_20261025 import sources, score_details, write

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
CORE = ROOT / 'outputs/核心参数训练_20261023T115640092598Z'


def load_originals(matches):
    rows, paths = [], []
    for period, arm in (('2024-25', 'regularized_poisson_h180_a0p01'), ('2025-26', 'original')):
        path = CORE / period / arm / 'E0.json'
        saved = json.loads(path.read_text(encoding='utf-8'))
        paths.append(str(path))
        if saved['failures'] or saved['update_interval_days'] != 28:
            raise ValueError('Original cohort has failures or another update interval')
        fits = {f['before_date']: f for f in saved['fits']}
        for row in saved['predictions']:
            record = matches[row['match_id']]
            fit = fits[row['fit_before_date']]
            if (row['date'] != record.match_date.isoformat()
                    or row['home'] != record.home_label or row['away'] != record.away_label
                    or row['actual_score'] != list(record.full_time)
                    or not fit['training_max_date'] < fit['before_date'] <= row['date']
                    or fit['regularization'] != .01 or fit['half_life_days'] != 180):
                raise ValueError('Original fixture or cutoff does not match source')
            p = list(summarize(score_grid(*row['rates'], rho=0))['one_x_two'].values())
            if not np.allclose(p, row['probabilities'], rtol=0, atol=1e-10):
                raise ValueError('Original rates do not replay the original probabilities')
            rows.append({**row, 'period': period, 'home': record.home, 'away': record.away})
    if len(rows) != 756 or len({r['match_id'] for r in rows}) != len(rows):
        raise ValueError('Original unique cohort is incomplete')
    return sorted(rows, key=lambda r: (r['date'], r['match_id'])), paths


def main():
    started = datetime.now(timezone.utc)
    target = ROOT / 'outputs' / started.strftime('射正攻防校正研究_%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True, exist_ok=False)
    plan = {
        'created_at': started.isoformat(), 'user_approved': '2026-10-25: opponent/venue-adjusted shot-strength contrast',
        'auxiliary': 'Poisson regression of earlier HST/AST with team attack/defense and home features',
        'goal_increment': 'log(mu_new)=log(mu_original)+beta*z(log(predicted_shots_on_target))',
        'single_shared_goal_coefficient': True, 'new_goal_intercept': False,
        'fixed_settings': {'regularization_both_layers': .01, 'half_life_days_both_layers': 180,
                           'update_days': 28, 'rho': 0},
        'information_cutoff': 'both original goal rates and auxiliary shot rates fitted strictly before the same original cutoff',
        'initial_shot_training': '2022/23 and 2023/24 EPL; no recent-10 heuristic or minimum-five requirement',
        'initial_goal_increment_training': '2024/25 out-of-time original goal rates and out-of-time auxiliary shot rates',
        'evaluation': '2025/26 online: beta uses only earlier target dates at each original cutoff',
        'budget': {'auxiliary_shot_fits': 22, 'single_coefficient_goal_fits': 11,
                   'new_original_goal_model_fits': 0, 'hyperparameter_searches': 0},
        'primary': 'same-fixture complete-score NLL',
        'secondary': ['1X2 LL/Brier/top1', 'total 0/1/2/3/4+ LL', 'away-minus-two three-way LL'],
        'uncertainty': 'paired noncircular moving calendar blocks 2/4/8 weeks, 2000 repeats, descriptive only',
        'missing': 'unknown shot teams or unavailable data recorded; never fill a missing shot rate with zero',
        'preserve': 'all original historic and prospective predictions; never automatically adopt',
        'limitations': ['already-viewed seasons, not untouched validation',
                        'date-only availability assumption; historical publication times unknown',
                        'EPL only; Championship/cups missing, promotion and long absence not fully resolved',
                        'shots include goal events, so auxiliary information is correlated with goal information',
                        'fixed settings reused for a bounded experiment, not proven optimal for shots',
                        'point estimates, no posterior player/team/parameter uncertainty integration'],
    }
    write(target/'分析前计划.json', plan)
    observations, matches = sources()
    originals, source_paths = load_originals(matches)
    shot_models, rows, excluded = {}, [], []
    for row in originals:
        cutoff = row['fit_before_date']
        if cutoff not in shot_models:
            shot_models[cutoff] = DateTeamShotModel().fit(observations, date.fromisoformat(cutoff))
        model = shot_models[cutoff]
        if {row['home'], row['away']} - set(model.parameters()['teams']):
            excluded.append({'match_id': row['match_id'], 'date': row['date'], 'period': row['period'],
                             'reason': 'unseen_shot_team'})
            continue
        rates = model.predict(row['home'], row['away'], 'E0', date.fromisoformat(row['date']))
        rows.append({**row, 'shot_rates': list(rates)})
    old = [r for r in rows if r['period'] == '2025-26']
    revised, increment_models = [], {}
    for row in old:
        cutoff = row['fit_before_date']
        if cutoff not in increment_models:
            increment_models[cutoff] = ShotRateGoalIncrement().fit(rows, date.fromisoformat(cutoff))
        rates = increment_models[cutoff].predict(row['rates'], row['shot_rates'])
        forecast = summarize(score_grid(*rates, rho=0))
        revised.append({**row, 'offset_rates': row['rates'], 'rates': list(rates), 'forecast': forecast,
                        'probabilities': list(forecast['one_x_two'].values())})
    if len(shot_models) != plan['budget']['auxiliary_shot_fits'] or len(increment_models) != 11:
        raise ValueError('Unexpected fit count: investigate without expanding the budget')
    old_scores, old_losses = score_details(old)
    new_scores, new_losses = score_details(revised)
    full_scores, _ = score_details([r for r in originals if r['period']=='2025-26'])
    domains = list(new_losses)
    difference = np.column_stack([new_losses[k]-old_losses[k] for k in domains])
    intervals = {str(block): moving_block_means([r['date'] for r in old], difference,
                                              block_weeks=block) for block in (2, 4, 8)}
    windows = []
    for cutoff, model in increment_models.items():
        indices = [i for i, r in enumerate(old) if r['fit_before_date']==cutoff]
        windows.append({'cutoff': cutoff, 'n': len(indices), 'parameters': model.parameters(),
                        'mean_log_loss_differences': dict(zip(domains, difference[indices].mean(axis=0).tolist()))})
    # Describe overlap of the two signals without treating correlation as independence or value.
    signal_description = {}
    for period in ('2024-25', '2025-26'):
        selected = [r for r in rows if r['period']==period]
        shot_values = np.log(np.asarray([r['shot_rates'] for r in selected]).reshape(-1))
        goal_values = np.log(np.asarray([r['rates'] for r in selected]).reshape(-1))
        signal_description[period] = {
            'n_matches': len(selected), 'log_shot_log_goal_correlation': float(np.corrcoef(shot_values, goal_values)[0, 1]),
            'predicted_mean_shots_home_away': np.mean([r['shot_rates'] for r in selected], axis=0).tolist(),
            'observed_mean_shots_home_away': np.mean([next(o.shots_on_target for o in observations
                                                   if o.match_id==r['match_id']) for r in selected], axis=0).tolist()}
    result = {'plan': plan, 'finished_at': datetime.now(timezone.utc).isoformat(),
              'original_trial_paths': source_paths, 'source_matches': len(matches),
              'new_auxiliary_shot_fits': len(shot_models), 'new_single_coefficient_fits': len(increment_models),
              'new_original_goal_model_fits': 0, 'hyperparameter_searches': 0,
              'earlier_original_n': 378, 'earlier_common_n': sum(r['period']=='2024-25' for r in rows),
              'later_original_n': 378, 'later_common_n': len(old), 'exclusions': excluded,
              'old_full_scores': full_scores, 'old_common_scores': old_scores, 'new_common_scores': new_scores,
              'loss_difference_order': domains,
              'mean_log_loss_differences': dict(zip(domains, difference.mean(axis=0).tolist())),
              'block_intervals': intervals, 'increment_windows': windows,
              'auxiliary_parameters': {k: m.parameters() for k, m in shot_models.items()},
              'signal_description': signal_description, 'original_with_auxiliary': rows, 'new_predictions': revised,
              'adopted': False}
    write(target/'射正攻防研究结果.json', result)
    report = ['# 射正攻防校正：单系数进球增量研究', '',
              '本轮按用户同意的结构实跑：先由更早射正学习球队攻击、对手防守与主场差异，再把合成的预期射正作为原进球率的一个辅助输入。最后仅学习一个主客共享系数，无新截距，真实进球仍是最终目标。', '',
              f'共{len(matches)}场源数据。较早2024/25共同{result["earlier_common_n"]}/378场；较晚2025/26共同{len(old)}/378场，缺失{len(excluded)}条均留记录。射正模型{len(shot_models)}次、单系数进球拟合{len(increment_models)}次，原进球模型重训0次、参数搜索0次。', '',
              '两层均固定正则.01、半衰180天；信息截止严格使用原28天日期。较早赛季射正预测自身也是前推输出，后季系数每次只使用更早赛果与其当时的预测，未用同场射正或未来赛果回填。', '',
              '| 指标 | 原模型／共同场次 | 校正射正增量／共同场次 | 新减旧 |', '|---|---:|---:|---:|']
    for key in ('exact_score', 'one_x_two', 'total_goals_0_1_2_3_4plus', 'away_minus_two_three_way'):
        report.append(f'| {key} LL | {old_scores[key]["log_loss"]:.6f} | {new_scores[key]["log_loss"]:.6f} | {result["mean_log_loss_differences"][key]:+.6f} |')
    for metric in ('brier_sum', 'accuracy'):
        report.append(f'| 1X2 {metric} | {old_scores["one_x_two"][metric]:.6f} | {new_scores["one_x_two"][metric]:.6f} | {new_scores["one_x_two"][metric]-old_scores["one_x_two"][metric]:+.6f} |')
    report += ['', 'LL与Brier越低越好；最高概率命中率越高越好。完整比分为事前主指标。', '',
               '| 块长度 | 比分LL差95%描述区间 | 胜平负LL差95%描述区间 |', '|---|---|---|']
    for block, entry in intervals.items():
        def show(key):
            low, high = entry['interval_95'][domains.index(key)]
            return f'[{low:+.6f}, {high:+.6f}]'
        report.append(f'| {block}周 | {show("exact_score")} | {show("one_x_two")} |')
    report += ['', '## 解释范围', '',
               '该模型对射正记录中的对手与主客场作显式统计控制，区别于最近10场的原始均值；没有使用历史窗口的事后最终排名，也没有把当前拟合伪装成当时的前推预测。', '',
               '新结构不再要求最近至少5场，但并未填零：只要球队已在更早射正训练集中出现，就由真实观测和固定收缩估计；未知队仍明确缺失。升班马少量数据以及缺席英超期间的实力变化仍有不确定性。没有英冠射正、杯赛或球员信息，不宣称解决跨级迁移。', '',
               '射正含进球事件，两个信号相关；它是一项辅助特征，不是第二张独立选票。预期射正仅作输入，最终比分仍为进球Poisson分布，没有把射正分布当成比分。', '',
               '这些历史赛季此前已看过，属于探索。移动块区间条件于已保存预测，不重演全部训练和选择，也不积分模型参数不确定性。无论本次结果如何，都不自动替换当前20场预测、不改LLM或市场权重、不继续按后季分数搜索参数。', '',
               '原简单射正均值研究共同375场，这次覆盖若不同，不能直接比较两份总分；需要同场才可对照。22份辅助参数与逐行前推输出均保留于结果文件。', '']
    (target/'射正攻防校正研究说明.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({'directory': str(target), 'shot_fits': len(shot_models), 'increment_fits': len(increment_models),
                      'n': len(old), 'differences': result['mean_log_loss_differences'],
                      'old_scores': old_scores, 'new_scores': new_scores,
                      'intervals_4_weeks': intervals['4']['interval_95'], 'signal': signal_description,
                      'first_beta': windows[0]['parameters']['coefficient'],
                      'last_beta': windows[-1]['parameters']['coefficient']}, ensure_ascii=False))


if __name__=='__main__':
    main()
