"""Account for saved outcome mechanisms; no fitting or probability recombination."""
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

import numpy as np

from diagnose_persistent_probabilities_20261028 import loss_parts, metrics


ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OUT = ROOT / 'outputs/胜平负机制复用收束_20261028'
SOURCES = {
    'direct': '直接胜平负分类研究_20261025T091329913808Z/lead_3/逐场胜平负概率.json',
    'draw': '条件平局分支整合_20261027/逐场整合与纠错.json',
    'history': '历史比赛模式与平局分支_20261027/逐场概率与纠错.json',
    'persistent': '持续状态净胜球研究_20261028/逐场概率与状态.json.gz',
    'rich_state': '持续状态与丰富信息接入_20261028/接入逐场概率.json.gz',
}
LABELS = {
    'original': '原121项进球回归',
    'direct_tree': '原121项直接三分类树',
    'direct_logistic': '原121项多项逻辑回归',
    'draw_tree': '只复用三分类树平局列',
    'history_classifier': '145项历史模式三分类器',
    'history_draw': '只复用历史模式平局列',
    'persistent': '持续状态净胜球似然',
    'rich_state': '121项加持续状态的进球回归',
}
CONTEXT = ('date', 'period', 'competition', 'home', 'away', 'horizon_days',
           'forecast_before_date')


def read(relative):
    path = ROOT / 'outputs' / relative
    with (gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz'
          else path.open(encoding='utf-8')) as stream:
        return json.load(stream)


def indexed(rows):
    result = {row['match_id']: row for row in rows}
    if len(result) != len(rows) or len(result) != 3452:
        raise ValueError('Expected the complete 3452 unique fixture cohort')
    return result


def compare_context(reference, candidate):
    if set(reference) != set(candidate):
        raise ValueError('Saved mechanisms have different fixture sets')
    for key, original in reference.items():
        row = candidate[key]
        if any(original[field] != row[field] for field in CONTEXT):
            raise ValueError('Identity, timing or fixture context differs')
        score = row.get('actual_score', [row.get('hg'), row.get('ag')])
        if score != original['actual_score']:
            raise ValueError('Saved fixture scores differ')
        parameter_date = row.get('fit_before_date', row.get('original_fit_before_date'))
        if parameter_date != original['fit_before_date']:
            raise ValueError('Original model parameter dates differ')
        if not parameter_date <= row['forecast_before_date'] < row['date']:
            raise ValueError('Need parameter date <= forecast date < match date')


def check_probability(p):
    if (p.shape != (3452, 3) or not np.isfinite(p).all() or np.any(p <= 0)
            or np.any(p >= 1) or not np.allclose(p.sum(axis=1), 1., rtol=0, atol=1e-12)):
        raise ValueError('Saved H/D/A must be complete, positive and normalized without clipping')


def run():
    source = {key: read(value) for key, value in SOURCES.items()}
    original = indexed(source['direct']['regression_performance'])
    records = sorted(original.values(), key=lambda row: (row['date'], row['match_id']))
    keys = [row['match_id'] for row in records]
    supplied = {
        'original': (original, lambda row: row['probabilities']),
        'direct_tree': (indexed(source['direct']['extra_trees_performance']), lambda row: row['probabilities']),
        'direct_logistic': (indexed(source['direct']['logistic_performance']), lambda row: row['probabilities']),
        'draw_tree': (indexed(source['draw']['records']), lambda row: row['probabilities']),
        'history_classifier': (indexed(source['history']['records']), lambda row: row['classifier_probabilities_reference_only']),
        'history_draw': (indexed(source['history']['records']), lambda row: row['probabilities']),
        'persistent': (indexed(source['persistent']['records']), lambda row: row['gas_probabilities']),
        'rich_state': (indexed(source['rich_state']['records']), lambda row: row['tracks']['state_rich']['probabilities']),
    }
    arrays = {}
    for name, (bank, extract) in supplied.items():
        compare_context(original, bank)
        arrays[name] = np.asarray([extract(bank[key]) for key in keys], dtype=float)
        check_probability(arrays[name])
    original_matches = {
        'draw': np.asarray([supplied['draw_tree'][0][k]['original_probabilities'] for k in keys]),
        'history': np.asarray([supplied['history_draw'][0][k]['original_probabilities'] for k in keys]),
        'persistent': np.asarray([supplied['persistent'][0][k]['original_hda'] for k in keys]),
        'rich_state': np.asarray([supplied['rich_state'][0][k]['tracks']['original']['probabilities'] for k in keys]),
    }
    original_errors = {key: float(np.max(np.abs(p-arrays['original']))) for key, p in original_matches.items()}
    if max(original_errors.values()) != 0:
        raise ValueError('Source studies did not preserve the exact same original probabilities')
    y_all = np.asarray([row['actual_outcome'] for row in records], dtype=int)
    if any(y != (0 if r['actual_score'][0] > r['actual_score'][1] else
                    2 if r['actual_score'][0] < r['actual_score'][1] else 1)
           for y, r in zip(y_all, records, strict=True)):
        raise ValueError('Outcome labels differ from stored full-time scores')
    groups = {}
    for period in ('2024-25', '2025-26', 'all'):
        mask = np.ones(len(records), dtype=bool) if period == 'all' else np.asarray([r['period'] == period for r in records])
        y = y_all[mask]
        base_parts = loss_parts(arrays['original'][mask], y)[:3]
        tracks = {}
        for name, full_p in arrays.items():
            p = full_p[mask]
            result = metrics(p, y)
            result['top1_correct'] = int((p.argmax(axis=1) == y).sum())
            result['draw_top1_count'] = int((p.argmax(axis=1) == 1).sum())
            result['draw_top1_correct'] = int(((p.argmax(axis=1) == 1) & (y == 1)).sum())
            parts = loss_parts(p, y)[:3]
            result['difference_from_original'] = dict(zip(
                ('full_log_loss', 'draw_binary_log_loss', 'weighted_direction_log_loss'),
                [float(np.mean(a-b)) for a, b in zip(parts, base_parts, strict=True)], strict=True))
            tracks[name] = result
        groups[period] = {'n': len(y), 'tracks': tracks}
    ratio_errors = {}
    original_q = arrays['original'][:, 0]/(arrays['original'][:, 0]+arrays['original'][:, 2])
    for name in ('draw_tree', 'history_draw'):
        p = arrays[name]
        ratio_errors[name] = float(np.max(np.abs(p[:, 0]/(p[:, 0]+p[:, 2])-original_q)))
        if ratio_errors[name] > 1e-12:
            raise ValueError('A draw-only transformation changed the conditional winner ratio')
    result = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Exact accounting of already saved probabilities, not new candidates or model selection',
        'source_files': {key: str(ROOT/'outputs'/value) for key, value in SOURCES.items()},
        'class_order': ['home', 'draw', 'away'], 'cohort_n': len(records),
        'earliest_target': records[0]['date'], 'latest_target': records[-1]['date'],
        'identical_original_probability_errors': original_errors,
        'draw_only_conditional_ratio_max_errors': ratio_errors,
        'new_fits': 0, 'new_forecasts': 0, 'new_probability_combinations': 0,
        'current_results_queries': 0, 'parameter_search': False,
        'interpretation': 'The conditional H/A diagnosis uses actual non-draws retrospectively; these cannot be preselected at forecast time. Historical seasons have already been reused for development.',
        'groups': groups,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'既存概率的机制分解.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = ['# 已有胜平负机制：同场概率损失分解', '',
        '2026-10-28。只读取既有概率；新拟合、新预测、重新拼接概率和当前赛果查询均为0。全部线路保留相同3,452场、原参数日P与发布日期F及标签；不按事后结果择路。', '',
        'LL = 平／非平二分类LL + 实际非平时的条件主／客LL贡献。最后一项按全体比赛平均；不能把实际非平样本的方向准确率当成赛前可执行准确率。', '',
        '|赛季|既有机制|正确/1726|完整LL↓|平／非平LL↓|条件方向贡献↓|平局首选/正确|',
        '|---|---|---:|---:|---:|---:|---:|']
    for period in ('2024-25', '2025-26'):
        for name, m in groups[period]['tracks'].items():
            lines.append(f"|{period}|{LABELS[name]}|{m['top1_correct']}|{m['log_loss']:.6f}|{m['draw_binary_log_loss']:.6f}|{m['conditional_direction_weighted_log_loss']:.6f}|{m['draw_top1_count']}/{m['draw_top1_correct']}|")
    lines += ['', '## 直接可复用的认识', '',
        '1. 既有直接三分类已经不受独立Poisson平局上限约束；“改成分类”不是未做过的新方向。',
        '2. 两个平局分支保留了原条件主客比例；其条件方向损失变化应仅为浮点误差。这是明确的设计限制，不能要求它纠正主客反选。',
        '3. 持续状态、丰富回归、三分类器均保留其信息与实现，不因整体差值小就判断整条研究无价值，也不按事后哪条正确拼成新预测。',
        '4. 条件分解是概率恒等式，本身不提供新信息。未来若分头训练，实质变化是各头的样本、函数形式和共享约束，不能宣称数学拆分天然提准。',
        '5. 本页到存量机制诊断停止；不追加区间、参数或新候选以寻找有利结果。旧赛季已参与研发，所列数字不是新盲测或当前准确率。', '']
    (OUT/'既存概率的机制分解.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'n': len(records), 'original_errors': original_errors,
        'ratio_errors': ratio_errors, 'by_period': {period: {name: {
            'correct': m['top1_correct'], 'll': m['log_loss'],
            'draw_delta': m['difference_from_original']['draw_binary_log_loss'],
            'direction_delta': m['difference_from_original']['weighted_direction_log_loss']}
            for name, m in groups[period]['tracks'].items()}
            for period in ('2024-25', '2025-26')},
        'new_fits': 0, 'new_predictions': 0}, ensure_ascii=False))


if __name__ == '__main__':
    run()
