"""Explain saved residual-pattern integration; no fits or new predictions.

The non-draw subset is defined after observing outcomes and is diagnostic only.
No subgroup rules, parameters, deployment choices, or source changes are made.
"""
from collections import Counter
import csv
import gzip
import json
import math
from pathlib import Path

from football_model.xg_residual_patterns import FEATURE_NAMES


ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OUT = ROOT / 'outputs/对手校正攻防模式整合_20261027'
TRACKS = ('original', 'existing_draw', 'residual_only', 'combined')
NAMES = {'original': '原丰富模型', 'existing_draw': '已有平局整合',
         'residual_only': '攻防模式进球层', 'combined': '本轮联合输出'}
LABELS = ('H', 'D', 'A')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def mean(values):
    return math.fsum(values)/len(values) if values else None


def top(probabilities):
    return max(range(3), key=lambda i: probabilities[i])


def status(old, new, actual):
    if old == new:
        return 'unchanged'
    if old != actual and new == actual:
        return 'corrected'
    if old == actual and new != actual:
        return 'damaged'
    return 'both_wrong'


def terms(probabilities, actual):
    if (len(probabilities) != 3 or not all(math.isfinite(p) and p > 0 for p in probabilities)
            or not math.isclose(sum(probabilities), 1., abs_tol=1e-12, rel_tol=0.)):
        raise ValueError('Saved H/D/A probabilities need positive normalized entries')
    h, d, a = probabilities
    draw_loss = -math.log(d if actual == 1 else 1-d)
    direction_loss = 0. if actual == 1 else -math.log((h if actual == 0 else a)/(h+a))
    total = -math.log(probabilities[actual])
    if not math.isclose(total, draw_loss+direction_loss, abs_tol=1e-12, rel_tol=0.):
        raise ValueError('H/D/A loss did not decompose into draw and conditional H/A')
    return total, draw_loss, direction_loss


def describe(records):
    non_draw = [r for r in records if r['actual_outcome'] != 1]
    tracks = {}
    maximum_term_error = 0.
    for name in TRACKS:
        losses = [terms(r['tracks'][name]['probabilities'], r['actual_outcome']) for r in records]
        maximum_term_error = max(maximum_term_error,
            max(abs(total-draw-direction) for total, draw, direction in losses))
        directions = [0 if r['tracks'][name]['probabilities'][0] >= r['tracks'][name]['probabilities'][2] else 2
                      for r in non_draw]
        correct = sum(direction == row['actual_outcome'] for direction, row in zip(directions, non_draw))
        tracks[name] = {
            'non_draw_subset_n': len(non_draw), 'conditional_direction_correct_n': correct,
            'conditional_direction_accuracy_on_observed_non_draws': correct/len(non_draw) if non_draw else None,
            'mean_full_HDA_LL': mean([item[0] for item in losses]),
            'mean_binary_draw_LL': mean([item[1] for item in losses]),
            'mean_HA_LL_contribution_all_targets': mean([item[2] for item in losses]),
            'mean_conditional_HA_LL_on_observed_non_draws': mean([
                terms(r['tracks'][name]['probabilities'], r['actual_outcome'])[2] for r in non_draw]),
        }
    contrasts = {}
    for reference in ('original', 'existing_draw'):
        changes = Counter()
        direction_changes = Counter()
        differences = []
        binary_deltas = []
        direction_deltas = []
        maximum_draw_probability_change = 0.
        for row in records:
            actual = row['actual_outcome']
            p = row['tracks'][reference]['probabilities']
            q = row['tracks']['combined']['probabilities']
            changes[status(top(p), top(q), actual)] += 1
            old, new = terms(p, actual), terms(q, actual)
            delta = [b-a for a, b in zip(old, new)]
            differences.append(delta[0])
            binary_deltas.append(delta[1])
            direction_deltas.append(delta[2])
            maximum_draw_probability_change = max(maximum_draw_probability_change, abs(q[1]-p[1]))
            if actual != 1:
                pdir, qdir = (0 if prob[0] >= prob[2] else 2 for prob in (p, q))
                direction_changes[status(pdir, qdir, actual)] += 1
        remainder = max(abs(full-binary-direction) for full, binary, direction
                        in zip(differences, binary_deltas, direction_deltas))
        contrasts['combined_minus_'+reference] = {
            'top1_change_counts': dict(changes),
            'observed_non_draw_direction_change_counts': dict(direction_changes),
            'non_draw_subset_n': len(non_draw),
            'mean_HDA_LL_delta': mean(differences),
            'mean_binary_draw_LL_delta': mean(binary_deltas),
            'mean_HA_contribution_delta_all_targets': mean(direction_deltas),
            'maximum_pointwise_LL_decomposition_remainder': remainder,
            'maximum_draw_probability_change': maximum_draw_probability_change,
        }
        if reference == 'existing_draw':
            if maximum_draw_probability_change > 1e-12 or max(map(abs, binary_deltas)) > 1e-12:
                raise ValueError('Expected unchanged saved draw component')
            if max(abs(full-direction) for full, direction in zip(differences, direction_deltas)) > 1e-12:
                raise ValueError('Existing-draw contrast should change only conditional H/A loss')
    return {'all_target_n': len(records), 'observed_draw_n': len(records)-len(non_draw),
            'observed_non_draw_n': len(non_draw), 'tracks': tracks,
            'contrasts': contrasts, 'maximum_HDA_LL_decomposition_error': maximum_term_error}


def main():
    records = read(OUT / '逐场概率与分量.json')['records']
    with gzip.open(OUT / '联合输入与历史来源.json.gz', 'rt', encoding='utf-8') as stream:
        sources = json.load(stream)['rows']
    index = {row['match_id']: row for row in sources}
    if len(index) != len(sources) or len({row['match_id'] for row in records}) != len(records):
        raise ValueError('Saved fixture identities are not unique')
    csv_rows = []
    for row in records:
        source = index[row['match_id']]
        if any(row[k] != source[k] for k in ('date', 'period', 'fit_before_date', 'forecast_before_date', 'actual_score')):
            raise ValueError('Saved probabilities and residual history have different contexts')
        selected = {name: top(row['tracks'][name]['probabilities']) for name in TRACKS}
        if len(set(selected.values())) == 1:
            continue
        item = {k: row[k] for k in ('match_id', 'date', 'period', 'competition', 'home', 'away')}
        item['actual_score'] = ':'.join(map(str, row['actual_score']))
        item['actual_outcome'] = LABELS[row['actual_outcome']]
        for name in TRACKS:
            for label, value in zip(LABELS, row['tracks'][name]['probabilities']):
                item[f'{name}_{label}'] = value
            item[f'{name}_top1'] = LABELS[selected[name]]
        for reference in ('original', 'existing_draw'):
            item[f'combined_vs_{reference}'] = status(selected[reference], selected['combined'], row['actual_outcome'])
        item['original_home_rate'], item['original_away_rate'] = row['original_goal_means']
        item['new_home_rate'], item['new_away_rate'] = row['input_goal_means']
        for side in ('home', 'away'):
            history = source['residual_pattern_history']['history'][side]
            item[f'{side}_residual_window_available'] = not history['unavailable_reasons']
            for key in FEATURE_NAMES:
                item[f'{side}_{key}'] = history['summary'][key]
        csv_rows.append(item)
    base_fields = ['match_id', 'date', 'period', 'competition', 'home', 'away', 'actual_score', 'actual_outcome']
    probability_fields = [f'{name}_{label}' for name in TRACKS for label in (*LABELS, 'top1')]
    fields = base_fields+probability_fields+['combined_vs_original', 'combined_vs_existing_draw',
        'original_home_rate', 'original_away_rate', 'new_home_rate', 'new_away_rate']
    fields += [field for side in ('home', 'away')
               for field in [f'{side}_residual_window_available', *[f'{side}_{key}' for key in FEATURE_NAMES]]]
    with (OUT / '全部改首选比赛.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    summary = {
        'scope': 'saved full cohorts only; no new predictions, training or selection rule',
        'changed_csv_definition': 'all rows where any of the four saved stages has a different top1',
        'changed_csv_rows': len(csv_rows),
        'conditional_subset_warning': 'Observed non-draw outcomes define the diagnostic subset retrospectively. Its conditional H/A accuracy is not complete 1X2 accuracy and is not an executable match filter.',
        'decomposition': 'HDA_LL = binary draw/not-draw LL + I(non-draw)*conditional H/A LL. The latter contribution is averaged over all targets unless explicitly conditioned.',
        'by_period': {period: describe([r for r in records if r['period'] == period])
                      for period in ('2024-25', '2025-26')},
        'new_fits': 0, 'current_results_checked': False,
    }
    write(OUT / '方向纠错与损失分解.json', summary)
    lines = ['# 本轮整合的主客方向与损失分解', '',
        '只读取本轮存盘概率和来源，未训练、改参数、生成新预测或查询当前赛果。'
        'H/D/A分别为主胜／平局／客胜。全部改首选CSV保留四个阶段的各自概率、首选和残差摘要。', '',
        '**下表只取历史实际非平比赛，是事后诊断。该条件命中率不能当作完整胜平负准确率，'
        '也不能用于未来预先筛掉平局。**', '',
        '|赛季|实际非平场数|原模型主客方向正确|本轮主客方向正确|方向纠正|方向改坏|',
        '|---|---:|---:|---:|---:|---:|']
    for period, value in summary['by_period'].items():
        change = value['contrasts']['combined_minus_original']['observed_non_draw_direction_change_counts']
        lines.append(f"|{period}|{value['observed_non_draw_n']}|{value['tracks']['original']['conditional_direction_correct_n']}|"
                     f"{value['tracks']['combined']['conditional_direction_correct_n']}|{change.get('corrected',0)}|{change.get('damaged',0)}|")
    lines += ['', '已有平局整合保留原模型主客比例，本轮联合输出保留新进球层主客比例；'
        '因此各自条件方向正确数相同，不是四条独立证据。原首选改变、平局／非平选择改变和条件主客方向改变是不同统计。', '',
        '胜平负对数损失可精确拆为“二元平／非平损失＋实际非平时的条件主／客损失贡献”。'
        '本轮联合与已有平局整合的d逐场相同，所以前一项不变；总体1X2损失增量全部来自后一项。'
        '这不代表完整比分损失也只由主客比例决定：新进球层还改变了各结果区间内的条件比分。', '',
        '|赛季／联合相对|全场1X2 LL差|二元平局LL差|主客条件贡献差（除全部场数）|逐场分解最大误差|',
        '|---|---:|---:|---:|---:|']
    for period, value in summary['by_period'].items():
        for reference in ('original', 'existing_draw'):
            item = value['contrasts']['combined_minus_'+reference]
            lines.append(f"|{period}／{NAMES[reference]}|{item['mean_HDA_LL_delta']:+.9f}|"
                f"{item['mean_binary_draw_LL_delta']:+.9f}|{item['mean_HA_contribution_delta_all_targets']:+.9f}|"
                f"{item['maximum_pointwise_LL_decomposition_remainder']:.3g}|")
    lines += ['', '差为新减原，负数表示该概率损失下降；分解用于确认变化来自哪个职责，'
        '不是输入特征的因果效应。未按这些已知赛果新增分组、阈值、权重或选场规则。', '',
        f'[全部改首选比赛]({OUT.as_posix()}/全部改首选比赛.csv) · '
        f'[完整方向与分解]({OUT.as_posix()}/方向纠错与损失分解.json)', '']
    (OUT / '方向纠错与分解说明.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'changed_csv_rows': len(csv_rows), 'by_period': summary['by_period']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
