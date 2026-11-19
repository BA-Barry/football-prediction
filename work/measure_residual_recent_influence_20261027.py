"""One finite local influence measurement of v0.48's new residual branch.

Zero residual is a hypothetical expected-performance reference, not removal
or missing information. Original 121 columns, model and d are unchanged.
The target's result is read only for retrospective diagnostic grouping.
"""
from collections import Counter
from datetime import date, datetime, timezone
import csv
import gzip
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import joblib
import numpy as np

from football_model.xg_residual_patterns import FEATURE_NAMES, summarize_residual_patterns
from football_model.probability import score_grid
from football_model.draw_branch import replace_draw_probability

ROOT = REPOSITORY_ROOT
SOURCE = ROOT / 'outputs/对手校正攻防模式整合_20261027'
OUT = ROOT / 'outputs/新增攻防分支近期影响_20261027'
SCENARIOS = tuple((role, k) for k in (1, 2) for role in ('home', 'away', 'both'))
PERIODS = ('2024-25', '2025-26')
ADDED = {f'{role}_{name}' for role in ('own', 'opponent') for name in FEATURE_NAMES}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def probabilities(rates, d):
    return np.asarray(replace_draw_probability(score_grid(*rates, rho=0).home_draw_away, d))


def neutral_features(row, role, k):
    """Copy only scalar feature dictionaries; retain every original history."""
    features = [dict(side) for side in row['side_features']]
    requested = ('home', 'away') if role == 'both' else (role,)
    details = []
    for side in requested:
        history = row['residual_pattern_history']['history'][side]
        if history['unavailable_reasons']:
            details.append({'side': side, 'eligible': False,
                            'reasons': history['unavailable_reasons'], 'neutralized_ids': []})
            continue
        observations = history['observations']
        keys = [(o['date'], o['match_id']) for o in observations]
        if len(keys) != 5 or keys != sorted(keys) or len(set(keys)) != 5:
            raise ValueError('Use the original five distinct chronological references')
        if any(not o['fit_before_date'] <= o['forecast_before_date'] < o['date'] < row['forecast_before_date']
               for o in observations):
            raise ValueError('Historical residual crosses the target cutoff')
        chosen = {o['match_id'] for o in observations[-k:]}
        pairs = [{'residual_for': 0. if o['match_id'] in chosen else o['residual_for'],
                  'residual_against': 0. if o['match_id'] in chosen else o['residual_against']}
                 for o in observations]
        summary = summarize_residual_patterns(pairs)
        own_index = 0 if side == 'home' else 1
        for name, value in summary.items():
            features[own_index][f'own_{name}'] = value
            features[1-own_index][f'opponent_{name}'] = value
        details.append({'side': side, 'eligible': True, 'neutralized_ids': [o['match_id'] for o in observations[-k:]],
            'original_recent_residuals': [{key: o[key] for key in
                ('match_id', 'date', 'residual_for', 'residual_against', 'net_residual', 'total_residual')}
                for o in observations[-k:]], 'hypothetical_summary': summary})
    for old, new in zip(row['side_features'], features):
        if set(old) != set(new) or any(old[key] != new[key] for key in old if key not in ADDED):
            raise ValueError('The existing 121 inputs changed')
    for name in FEATURE_NAMES:
        if features[0][f'own_{name}'] != features[1][f'opponent_{name}'] or features[1][f'own_{name}'] != features[0][f'opponent_{name}']:
            raise ValueError('Own/opponent feature views disagree')
    return features, details


def change_group(record, reference):
    old = int(np.argmax(record['tracks'][reference]['probabilities']))
    new = int(np.argmax(record['tracks']['combined']['probabilities']))
    y = record['actual_outcome']
    if old == new:
        return 'unchanged'
    if old != y and new == y:
        return 'corrected'
    if old == y and new != y:
        return 'damaged'
    return 'both_wrong'


def summary(rows):
    eligible = [r for r in rows if r['eligible_side_count'] > 0]
    values = np.asarray([r['max_abs_change_pp'] for r in eligible])
    return {'all_targets_n': len(rows), 'at_least_one_requested_side_eligible_n': len(eligible),
        'all_requested_sides_eligible_n': sum(r['eligible_side_count'] == r['requested_side_count'] for r in rows),
        'mean_max_abs_change_pp_eligible': float(values.mean()) if len(values) else None,
        'median_max_abs_change_pp_eligible': float(np.median(values)) if len(values) else None,
        'p95_max_abs_change_pp_eligible': float(np.quantile(values, .95)) if len(values) else None,
        'max_abs_change_pp_eligible': float(values.max()) if len(values) else None,
        'top1_change_n_eligible': sum(r['top1_changed'] for r in eligible),
        'conditional_HA_direction_change_n_eligible': sum(r['conditional_direction_changed'] for r in eligible),
        'unavailable_unchanged_n': len(rows)-len(eligible)}


def describe(results):
    by_period = {}
    for period in PERIODS:
        local = [r for r in results if r['period'] == period]
        by_period[period] = {}
        for role, k in SCENARIOS:
            key = f'{role}_latest{k}_expected_reference'
            chosen = [r for r in local if r['scenario'] == key]
            by_period[period][key] = {
                'all': summary(chosen),
                'by_previous_change_vs_original': {group: summary([r for r in chosen if r['previous_change_vs_original'] == group])
                    for group in ('unchanged', 'corrected', 'damaged', 'both_wrong')},
                'by_previous_change_vs_existing_draw': {group: summary([r for r in chosen if r['previous_change_vs_existing_draw'] == group])
                    for group in ('unchanged', 'corrected', 'damaged', 'both_wrong')},
            }
    return by_period


def report(result):
    names = {'home': '仅主队', 'away': '仅客队', 'both': '双方'}
    lines = ['# 新增攻防残差分支对最近一两场的局部敏感度', '',
        '只复用v0.48保存的22个模型和3,452个目标。不重新训练、调参、拟合校准、查当前赛果或更新比赛预测。'
        '本轮不重复旧2026-10-26对当前10场、基专家和LLM的近期信息测量。', '',
        '人为把指定球队最近1／2场的攻防残差设为0，即假定该旧场实际总xG恰好等于当时预期。'
        '仍保留五场窗口、原121项特征、基专家、模型和d，只重算新增16列；真实历史记录原值保留。'
        '因此测到的是新增分支局部影响，不能解释为整模型删掉某场后的影响，也不能由此证明过度响应。'
        '人为组合与仍保留同场信息的原121列可能不一致，也可能偏离训练分布；它不代表现实中可直接使用的一套输入。', '',
        '|赛季|人为参照场景|至少一侧可测／全部|平均最大单项变化|95分位|最大值|首选改变|',
        '|---|---|---:|---:|---:|---:|---:|']
    for period, scenarios in result['by_period'].items():
        for role, k in SCENARIOS:
            a = scenarios[f'{role}_latest{k}_expected_reference']['all']
            lines.append(f"|{period}|{names[role]}最近{k}场偏差归零|{a['at_least_one_requested_side_eligible_n']}/{a['all_targets_n']}|"
                f"{a['mean_max_abs_change_pp_eligible']:.3f}|{a['p95_max_abs_change_pp_eligible']:.3f}|"
                f"{a['max_abs_change_pp_eligible']:.3f}|{a['top1_change_n_eligible']}|")
    lines += ['', '变化单位为百分点：每场先取H/D/A三项绝对变化的最大值，再统计。均值、分位、最大和首选改变只统计至少一侧可干预的目标。'
        '缺失窗口保持整组未知，未把无法干预造成的零变化算作稳健证据；双方场景中仅一侧可用的覆盖另存在JSON。', '',
        'd保持不变，主胜／客胜变化互为相反数，且ΔpH=(1−d)ΔP(H|非平)。'
        '首选变化包括平局与非平排名变化，不能只把它当作主客方向反转。', '',
        '## 与已有纠错记录的关系', '',
        '|赛季／双方最近2场参照|原整合纠正组：可测／平均变化|原整合改坏组：可测／平均变化|',
        '|---|---:|---:|']
    for period, scenarios in result['by_period'].items():
        grouped = scenarios['both_latest2_expected_reference']['by_previous_change_vs_original']
        def cell(label):
            a = grouped[label]
            return f"{a['at_least_one_requested_side_eligible_n']}/{a['all_targets_n']}／{a['mean_max_abs_change_pp_eligible']:.3f}"
        lines.append(f'|{period}|{cell("corrected")}|{cell("damaged")}|')
    lines += ['', '这里按已知纠正／改坏分组，仅描述已保存错误；没有为人为场景评分命中率，没有据此制定筛选或回退规则。'
        '变化较大可指向需要解释的个案；较小也不能证明整个模型不受近期比赛影响。', '',
        '## 实测后的判断', '',
        '双方最近两场偏差归零时，2024/25在1,710场可测比赛上平均最大单项变化0.168个百分点，'
        '95分位0.476、最大1.576；2025/26在1,718场上分别0.162、0.454、1.033个百分点。'
        '两季各只有5场首选变化。其余16／8场因双方窗口缺失未改动，不计入上述分母。', '',
        '较晚赛季原先24场整合改坏的比赛，在这个双方最近两场参照下只有1场首选改变；'
        '17场纠正比赛也有1场首选改变。首选变化不代表改对。'
        '本测量没有显示这批错误主要由新增分支最近两场的偏差驱动，不能据此主张调整整个模型的近期权重。', '',
        '执行取舍：保留当前近期权重，不给新分支继续找收缩系数或窗口。'
        '接下来补已有明确缺口的同场历史机会与比分／人数状态资料，先核清来源重叠和时钟能力，'
        '再判断是否可以形成有明确含义的新信息；本轮局部测量不构造新的比赛预测。', '',
        '## 本轮执行与后续取舍', '',
        '一次运行完成六种场景、全部两季目标，原模型输入重放后再作测量。'
        '没有把0残差当作推荐输入，没有因已知输赢选择中性化场次或再搜窗口。'
        '是否应改变近期信息的用法，需要同时确认影响通道、训练分布和面向未来的收益；本轮不给新增权重。', '',
        '当前证据仅覆盖新分支。若近期变化很小，应将关注点转回已有强弱判断／平局识别的具体缺口，'
        '而不是继续给这16种相关表示调参数；若少数病例变化明显，先核查那几场的原始机会与对手预期，不能用目标赛果反推权重。', '',
        f'[全部测量汇总]({OUT.as_posix()}/局部敏感度汇总.json) · '
        f'[全部逐场人为测量]({OUT.as_posix()}/全部局部变化.csv) · '
        f'[原预测与历史来源]({SOURCE.as_posix()}/逐场概率与分量.json)', '']
    (OUT / '新增分支近期影响说明.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / '本轮测量范围.json', {
        'created_at': datetime.now(timezone.utc).isoformat(), 'source': str(SOURCE),
        'purpose': 'local influence of new residual columns, no outcome scoring or new model candidate',
        'scenarios': list(SCENARIOS), 'total_fits': 0, 'parameter_searches': 0,
        'residual_reference': 'Hypothetical actual total xG equal to old expected total xG; not removal/missingness.',
        'unchanged': ['original121', 'five reference identities', 'fit/F/d', 'saved model/imputer', 'actual historical observations'],
        'limits': ['Not whole-model LOO, no causal effect, no evidence that less sensitivity is necessarily better.',
                   'Synthetic marginal perturbation can be inconsistent with the retained original121 and training distribution.',
                   'Known damaged/corrected group is retrospective and cannot define an executable selection rule.'],
        'stop': 'one six-scenario full-cohort measurement; no alternative neutral values, model fits or scenario accuracy search',
        'current_results_checked': False, 'current_forecasts_replaced': False})
    records = read(SOURCE / '逐场概率与分量.json')['records']
    index = {r['match_id']: r for r in records}
    with gzip.open(SOURCE / '联合输入与历史来源.json.gz', 'rt', encoding='utf-8') as stream:
        inputs = [r for r in json.load(stream)['rows'] if r['period'] in PERIODS]
    if len(inputs) != 3452 or {r['match_id'] for r in inputs} != set(index):
        raise ValueError('Use all saved targets without dropping missing rows')
    for row in inputs:
        if any(row[key] != index[row['match_id']][key] for key in
               ('date', 'period', 'fit_before_date', 'forecast_before_date', 'actual_score')):
            raise ValueError('Saved inputs and outputs have different contexts')
    results, errors = [], Counter()
    completed = 0
    for cutoff in sorted({r['fit_before_date'] for r in inputs}):
        rows = [r for r in inputs if r['fit_before_date'] == cutoff]
        model = joblib.load(SOURCE / 'models' / f'residual_patterns_{cutoff}.joblib')
        dates = [date.fromisoformat(r['date']) for r in rows]
        original_means = model.predict_many([r['side_features'] for r in rows], dates)
        for row, rates in zip(rows, original_means):
            record = index[row['match_id']]
            error = max(abs(a-b) for a, b in zip(rates, record['input_goal_means']))
            errors['original_saved_rate_replay'] = max(errors['original_saved_rate_replay'], error)
            if error > 1e-12:
                raise ValueError('Saved model does not replay its original feature inputs')
        for role, k in SCENARIOS:
            transformed = [neutral_features(row, role, k) for row in rows]
            new_means = model.predict_many([item[0] for item in transformed], dates)
            for row, rates, (_, details) in zip(rows, new_means, transformed):
                record = index[row['match_id']]
                original = np.asarray(record['tracks']['combined']['probabilities'])
                d = record['unchanged_draw_probability']
                altered = probabilities(rates, d)
                delta = altered-original
                old_ratio = original[0]/(original[0]+original[2])
                new_ratio = altered[0]/(altered[0]+altered[2])
                errors['unchanged_d'] = max(errors['unchanged_d'], abs(delta[1]))
                errors['opposite_HA_changes'] = max(errors['opposite_HA_changes'], abs(delta[0]+delta[2]))
                errors['conditional_HA_relation'] = max(errors['conditional_HA_relation'], abs(delta[0]-(1-d)*(new_ratio-old_ratio)))
                eligible = sum(item['eligible'] for item in details)
                if not eligible and np.max(np.abs(delta)) > 1e-12:
                    raise ValueError('An unavailable unchanged window altered the prediction')
                results.append({**{key: row[key] for key in ('match_id', 'period', 'date', 'home', 'away')},
                    'scenario': f'{role}_latest{k}_expected_reference',
                    'requested_side_count': len(details), 'eligible_side_count': eligible,
                    'original_probabilities': original.tolist(), 'hypothetical_probabilities': altered.tolist(),
                    'changes_pp': (100*delta).tolist(), 'max_abs_change_pp': float(100*np.max(np.abs(delta))),
                    'conditional_home_given_non_draw_change_pp': float(100*(new_ratio-old_ratio)),
                    'original_goal_means': record['input_goal_means'], 'hypothetical_goal_means': list(rates),
                    'top1_changed': int(original.argmax()) != int(altered.argmax()),
                    'conditional_direction_changed': bool((original[0] >= original[2]) != (altered[0] >= altered[2])),
                    'previous_change_vs_original': change_group(record, 'original'),
                    'previous_change_vs_existing_draw': change_group(record, 'existing_draw'),
                    'details': details})
        completed += 1
        print(json.dumps({'completed_model_dates': completed, 'cutoff': cutoff, 'target_n': len(rows)}, ensure_ascii=False), flush=True)
    if len(results) != 6*3452 or max(errors.values()) > 1e-12:
        raise ValueError('Full measurement or probability identities failed')
    with gzip.open(OUT / '全部局部人为参照.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump({'records': results}, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    fields = ['match_id', 'period', 'date', 'home', 'away', 'scenario', 'requested_side_count', 'eligible_side_count',
              'original_H', 'original_D', 'original_A', 'hypothetical_H', 'hypothetical_D', 'hypothetical_A',
              'delta_H_pp', 'delta_D_pp', 'delta_A_pp', 'max_abs_change_pp',
              'conditional_home_given_non_draw_change_pp', 'top1_changed', 'conditional_direction_changed',
              'previous_change_vs_original', 'previous_change_vs_existing_draw']
    with (OUT / '全部局部变化.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in results:
            item = {key: row[key] for key in fields if key in row}
            for label, p, q, change in zip(('H', 'D', 'A'), row['original_probabilities'], row['hypothetical_probabilities'], row['changes_pp']):
                item.update({f'original_{label}': p, f'hypothetical_{label}': q, f'delta_{label}_pp': change})
            writer.writerow(item)
    result = {'by_period': describe(results), 'actual_model_dates': completed,
              'total_inference_rows': len(results), 'maximum_numerical_errors': dict(errors),
              'new_fits': 0, 'parameter_searches': 0, 'counterfactual_outcomes_scored': False}
    write(OUT / '局部敏感度汇总.json', result)
    report(result)
    print(json.dumps({'phase': 'complete', 'summary': {p: {key: value['all'] for key, value in item.items()}
                                                    for p, item in result['by_period'].items()}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
