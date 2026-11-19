"""Paired historical 1X2 evaluation of saved KickScore forecasts; never fits.

Input: forecasts.json with tracks dynamic/static/dynamic_mean_only and
data/references.json prepared from the previous saved direct-outcome study.
Only evaluation.json and the Chinese report are written, under the G: project.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from football_model.evaluation import calibration_bins, probability_scores
from football_model.forecast_diagnostics import moving_block_means


ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OLD_STUDY = ROOT/'outputs/直接胜平负分类研究_20261025T091329913808Z'
PERIODS = ('2024-25', '2025-26')
LEADS = (1, 3)
COMPS = ('E0', 'D1', 'SP1', 'I1', 'F1')
NEW_TRACKS = ('dynamic', 'static', 'dynamic_mean_only')
OLD_TRACKS = ('goal', 'xg_raw', 'clean_blend', 'regression_allcomp', 'extra_trees_allcomp')
CLASSES = ('home', 'draw', 'away')
IDENTITY = ('date', 'period', 'competition', 'home', 'away', 'actual_score', 'actual_outcome')
TIMING = ('forecast_before_date', 'horizon_days')
METRICS = ('log_loss', 'brier_sum', 'rps', 'misclassification')
BLOCKS = (2, 4, 8)
REPEATS = 2000
SEED = 20261025
MODEL_LABELS = {'dynamic': '动态 KickScore', 'static': '静态 KickScore',
    'dynamic_mean_only': '动态均值消融', 'goal': '进球基线', 'xg_raw': '原始 xG 基线',
    'clean_blend': '较早年份确定的融合', 'regression_allcomp': '全赛事回归',
    'extra_trees_allcomp': '全赛事 ExtraTrees'}
METRIC_LABELS = {'log_loss': 'Log loss', 'brier_sum': 'Brier（求和）',
    'rps': 'RPS', 'misclassification': '分类错误率'}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _inside_project(path: Path) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f'Evaluation output must remain inside {ROOT}: {resolved}')
    return resolved


def _validate_row(row: dict, *, expected_lead: int | None = None) -> None:
    for field in ('match_id', 'home', 'away', *IDENTITY, *TIMING):
        if field not in row:
            raise ValueError(f'Missing {field} in forecast/target row')
    for field in ('match_id', 'home', 'away'):
        if not isinstance(row[field], str) or not row[field]:
            raise ValueError(f'Invalid fixture {field}')
    if row['home'] == row['away'] or row['competition'] not in COMPS:
        raise ValueError(f'Invalid fixture identity: {row["match_id"]}')
    if row['period'] not in PERIODS:
        raise ValueError(f'Unexpected evaluation period: {row["period"]}')
    played = date.fromisoformat(row['date'])
    publication = date.fromisoformat(row['forecast_before_date'])
    lead = row['horizon_days']
    if type(lead) is not int or lead not in LEADS or (expected_lead is not None and lead != expected_lead):
        raise ValueError(f'Unexpected horizon for {row["match_id"]}: {lead}')
    if publication >= played or (played-publication).days != lead:
        raise ValueError(f'Publication day disagrees with horizon: {row["match_id"]}')
    season_start = int(row['period'][:4])
    if not date(season_start, 7, 1) <= played < date(season_start+1, 7, 1):
        raise ValueError(f'Fixture date disagrees with period: {row["match_id"]}')
    if 'fit_before_date' in row:
        fitted = date.fromisoformat(row['fit_before_date'])
        if fitted > publication:
            raise ValueError(f'Fit cutoff after forecast publication: {row["match_id"]}')
        if 'training_max_date' in row and row['training_max_date'] is not None:
            if date.fromisoformat(row['training_max_date']) >= fitted:
                raise ValueError(f'Training includes fit cutoff or later: {row["match_id"]}')
    if 'feature_before_date' in row and row['feature_before_date'] != row['forecast_before_date']:
        raise ValueError(f'Feature and publication date disagree: {row["match_id"]}')
    score = row['actual_score']
    if not isinstance(score, list) or len(score) != 2 or any(type(g) is not int or g < 0 for g in score):
        raise ValueError(f'Invalid actual score: {row["match_id"]}')
    expected = 0 if score[0] > score[1] else 2 if score[0] < score[1] else 1
    if type(row['actual_outcome']) is not int or row['actual_outcome'] != expected:
        raise ValueError(f'Outcome disagrees with score: {row["match_id"]}')


def _row_map(rows: list[dict], label: str, *, lead: int,
             identity_registry: dict[str, dict]) -> dict[str, dict]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'Nonempty row list required: {label}')
    mapped = {}
    for row in rows:
        _validate_row(row, expected_lead=lead)
        match_id = row['match_id']
        if match_id in mapped:
            raise ValueError(f'Duplicate match ID in {label}: {match_id}')
        identity = {key: row[key] for key in IDENTITY}
        if match_id in identity_registry and identity_registry[match_id] != identity:
            raise ValueError(f'Conflicting fixture identity/label in {label}: {match_id}')
        identity_registry[match_id] = identity
        mapped[match_id] = row
    return mapped


def _require_exact_ids(mapped: dict, expected: dict, label: str) -> None:
    missing = sorted(set(expected)-set(mapped))
    extra = sorted(set(mapped)-set(expected))
    if missing or extra:
        raise ValueError(f'Incomplete same-match cohort for {label}: '
                         f'missing={len(missing)} {missing[:5]}; extra={len(extra)} {extra[:5]}')
    for match_id, row in mapped.items():
        for field in (*IDENTITY, *TIMING):
            if row[field] != expected[match_id][field]:
                raise ValueError(f'Conflicting {field} in {label}: {match_id}')


def _binary_auc(probability: np.ndarray, positive: np.ndarray) -> float | None:
    """Mann-Whitney AUC with average ranks for exact probability ties."""
    positives = int(positive.sum())
    negatives = len(positive)-positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(probability, kind='stable')
    sorted_p = probability[order]
    _, starts, counts = np.unique(sorted_p, return_index=True, return_counts=True)
    average_ranks = starts+(counts+1)/2
    ranks = np.empty(len(probability), dtype=float)
    ranks[order] = np.repeat(average_ranks, counts)
    return float((ranks[positive].sum()-positives*(positives+1)/2)/(positives*negatives))


def score_track(rows: list[dict]) -> tuple[dict, dict[str, np.ndarray]]:
    """Strict native-probability losses plus per-class AUC and calibration."""
    p = np.asarray([r['probabilities'] for r in rows], dtype=float)
    y = np.asarray([r['actual_outcome'] for r in rows], dtype=int)
    if (p.shape != (len(rows), 3) or not len(rows) or not np.isfinite(p).all()
            or np.any(p <= 0) or np.any(p >= 1)
            or not np.allclose(p.sum(axis=1), 1, atol=1e-12, rtol=0)):
        raise ValueError('Need native positive normalized 1X2 probabilities; no clipping or renormalization')
    truth = np.eye(3)[y]
    losses = {'log_loss': -np.log(p[np.arange(len(y)), y]),
        'brier_sum': np.square(p-truth).sum(axis=1),
        'rps': np.square(np.cumsum(p, axis=1)[:, :2]-np.cumsum(truth, axis=1)[:, :2]).sum(axis=1)/2,
        'misclassification': (p.argmax(axis=1) != y).astype(float)}
    bins = calibration_bins(p, y, bins=10)
    auc, calibration, by_actual = {}, {}, {}
    for c, name in enumerate(CLASSES):
        positive = y == c
        class_bins = [b for b in bins if b['outcome'] == name]
        value = _binary_auc(p[:, c], positive)
        auc[name] = {'auc': value, 'n_positive': int(positive.sum()),
            'n_negative': int((~positive).sum()),
            'undefined_reason': 'only_one_binary_class_present' if value is None else None}
        calibration[name] = {'mean_probability': float(p[:, c].mean()),
            'observed_frequency': float(positive.mean()),
            'mean_probability_minus_frequency': float((p[:, c]-truth[:, c]).mean()),
            'binary_brier': float(np.square(p[:, c]-truth[:, c]).mean()),
            'ece_equal_width_10_bins': float(sum(b['n']*abs(b['mean_probability']-b['observed_frequency'])
                                               for b in class_bins)/len(y)),
            'maximum_bin_gap': float(max(abs(b['mean_probability']-b['observed_frequency'])
                                         for b in class_bins))}
        by_actual[name] = {'n': int(positive.sum()),
            'log_loss': float(losses['log_loss'][positive].mean()) if positive.any() else None}
    auc_values = [v['auc'] for v in auc.values()]
    scores = probability_scores(p, y) | {'misclassification': float(losses['misclassification'].mean()),
        'mean_probabilities': p.mean(axis=0).tolist(),
        'observed_frequencies': truth.mean(axis=0).tolist(), 'one_vs_rest_auc': auc,
        'macro_one_vs_rest_auc': float(np.mean(auc_values)) if all(v is not None for v in auc_values) else None,
        'calibration': {'classes': calibration, 'equal_width_10_bins': bins,
            'note': 'Descriptive only; no posthoc probability calibration or model selection.'},
        'by_actual_class': by_actual}
    return scores, losses


def _score_cohort(tracks: dict[str, dict], ids: list[str]) -> dict:
    if not ids:
        raise ValueError('Cannot score an empty declared cohort')
    scores, losses = {}, {}
    for arm, mapped in tracks.items():
        scores[arm], losses[arm] = score_track([mapped[k] for k in ids])
    references = ['static', 'dynamic_mean_only', 'regression_allcomp']
    if 'clean_blend' in tracks:
        references.append('clean_blend')
    columns, deltas = [], []
    for reference in references:
        for metric in METRICS:
            columns.append((f'dynamic_minus_{reference}', metric))
            deltas.append(losses['dynamic'][metric]-losses[reference][metric])
    matrix = np.column_stack(deltas)
    dates = [tracks['dynamic'][k]['date'] for k in ids]
    contrasts = {f'dynamic_minus_{reference}': {} for reference in references}
    for index, (contrast, metric) in enumerate(columns):
        contrasts[contrast][metric] = {'mean': float(matrix[:, index].mean()), 'interval_95_by_block_weeks': {}}
    bootstrap = {}
    for weeks in BLOCKS:
        boot = moving_block_means(dates, matrix, block_weeks=weeks, repeats=REPEATS, seed=SEED)
        interval = boot.pop('interval_95')
        boot.pop('mean')
        boot['difference_order'] = [f'{contrast}:{metric}' for contrast, metric in columns]
        for index, (contrast, metric) in enumerate(columns):
            contrasts[contrast][metric]['interval_95_by_block_weeks'][str(weeks)] = (
                interval[index] if interval is not None else None)
        bootstrap[str(weeks)] = boot
    return {'n': len(ids), 'first_date': min(dates), 'last_date': max(dates), 'match_ids': ids,
        'scores': scores, 'contrasts': contrasts, 'bootstrap': bootstrap}


def _load_tracks(out: Path, reference_path: Path) -> tuple[dict, dict, dict]:
    forecast_document = _read(out/'forecasts.json')
    if not isinstance(forecast_document, dict) or not isinstance(forecast_document.get('tracks'), dict):
        raise ValueError('forecasts.json must contain a tracks dictionary')
    new = forecast_document['tracks']
    if set(new) != set(NEW_TRACKS):
        raise ValueError(f'Expected exactly the three declared KickScore tracks: {NEW_TRACKS}')
    references = _read(reference_path)
    if references.get('class_order') != list(CLASSES):
        raise ValueError('Reference class order must be home/draw/away')
    if set(references.get('leads', {})) != {'1', '3'}:
        raise ValueError('References must declare both leads 1 and 3')
    if Path(references['old_study_path']).resolve() != OLD_STUDY.resolve():
        raise ValueError('References point to a different historical direct-outcome study')
    registry, loaded, coverage = {}, {}, {}
    for arm, rows in new.items():
        if not isinstance(rows, list) or not rows:
            raise ValueError(f'Nonempty forecast list required: {arm}')
        for row in rows:
            _validate_row(row)
        seen = set()
        for row in rows:
            key = (row['match_id'], row['horizon_days'])
            if key in seen:
                raise ValueError(f'Duplicate match/horizon in {arm}: {key}')
            seen.add(key)
    for lead in LEADS:
        record = references['leads'][str(lead)]
        target = _row_map(record['targets'], f'lead{lead} targets', lead=lead, identity_registry=registry)
        declared_periods = record['periods']
        if set(declared_periods) != set(PERIODS):
            raise ValueError(f'References must declare both evaluation periods, lead {lead}')
        declared_ids = []
        for period in PERIODS:
            ids = declared_periods[period]['match_ids']
            expected_ids = {k for k, row in target.items() if row['period'] == period}
            if not isinstance(ids, list) or len(ids) != len(set(ids)) or set(ids) != expected_ids:
                raise ValueError(f'Explicit period target IDs conflict, lead {lead} period {period}')
            declared_ids.extend(ids)
        if len(declared_ids) != len(target):
            raise ValueError('Declared season cohorts must cover every target exactly once')
        local = {}
        for arm, rows in new.items():
            mapped = _row_map([r for r in rows if r['horizon_days'] == lead],
                              f'{arm} lead{lead}', lead=lead, identity_registry=registry)
            _require_exact_ids(mapped, target, f'{arm} lead{lead}')
            local[arm] = mapped
        old = record['references']
        if not set(OLD_TRACKS).issubset(old):
            raise ValueError(f'Expected all five historical comparison tracks, lead {lead}')
        for arm in OLD_TRACKS:
            mapped = _row_map(old[arm], f'{arm} lead{lead}', lead=lead, identity_registry=registry)
            expected = {k: r for k, r in target.items() if arm != 'clean_blend' or r['period'] == '2025-26'}
            _require_exact_ids(mapped, expected, f'{arm} lead{lead}')
            local[arm] = mapped
        loaded[str(lead)] = local
        coverage[str(lead)] = {'n_targets': len(target),
            'period_counts': {period: sum(r['period'] == period for r in target.values()) for period in PERIODS},
            'track_counts': {arm: len(rows) for arm, rows in local.items()},
            'source_declared_reference_tracks': sorted(old),
            'missing_ids': [], 'extra_ids': [],
            'alignment': 'Exact complete target IDs and fixture/label/publication-day identity; no intersection deletion.'}
    return loaded, coverage, forecast_document


def _number(value: Any, digits: int = 5) -> str:
    return '未定义' if value is None else f'{value:.{digits}f}'


def _interval(value: Any) -> str:
    return '不可估计' if value is None else f'[{value[0]:.5f}, {value[1]:.5f}]'


def _metadata_lines(metadata: dict | None) -> list[str]:
    if metadata is None:
        return ['未提供 training_summary.json；本报告不推测拟合次数、运行耗时或收敛状态。']
    lines = [f'训练记录状态：{metadata.get("phase", "未记录")}。']
    for kind, search in metadata.get('hyperparameter_search', {}).items():
        if isinstance(search, dict):
            fields = [('objective_evaluations', '目标函数评估'), ('underlying_fit_calls', '底层拟合'),
                      ('seconds', '秒数'), ('converged', '收敛记录')]
            details = [f'{label}={json.dumps(search[key], ensure_ascii=False)}'
                       for key, label in fields if key in search]
            lines.append(f'- {kind} 超参数学习：' + ('；'.join(details) if details else '未记录上述计数') + '。')
    forward = metadata.get('forward', {})
    if isinstance(forward, dict):
        details = []
        for key, label in (('fit_calls', '已记账完成拟合次数'), ('seconds', '本阶段总秒数')):
            if key in forward:
                details.append(f'{label}={forward[key]}')
        windows = forward.get('windows', [])
        if isinstance(windows, list):
            explicit = [w['converged'] for w in windows if isinstance(w, dict) and 'converged' in w]
            if explicit:
                details.append(f'明确记录收敛={sum(v is True for v in explicit)}/{len(explicit)}')
            if windows:
                details.append(f'记录窗口数={len(windows)}')
        if 'failures' in forward:
            failures = forward['failures']
            details.append(f'失败记录数={len(failures) if isinstance(failures, list) else json.dumps(failures, ensure_ascii=False)}')
        lines.append('前推运行：' + ('；'.join(details) if details else '未记录上述计数') + '。')
    if metadata.get('dependencies'):
        lines.append('运行依赖版本：' + json.dumps(metadata['dependencies'], ensure_ascii=False) + '。')
    return lines


def _make_report(result: dict) -> str:
    lines = ['# 动态胜平负历史回放实跑结论', '',
        '本研究比较动态 KickScore、静态 KickScore、动态均值消融与已保存的五类胜平负参考概率。'
        '这是已被查看过的 2024/25、2025/26 历史回放，不是新的盲测；两个赛季分别报告，不用较晚赛季选择模型或权重。', '',
        'KickScore 是概率技能模型，并非神经网络。本次只评价主胜／平局／客胜（1X2），'
        '不能从三分类概率扩展出比分、大小球或让球预测。本报告不计算或声称投注收益，不保证未来效果；保留旧正式模型，本次研究未自动替换它。', '',
        'lead 1 与 lead 3 表示比赛日期前 1／3 个日历日可用的信息快照，属于日期级可用性，'
        '不等同于精确开球前 24／72 小时。未知的原始发布时间、修订和历史研究选择仍是限制。'
        '模型拟合和超参数学习的具体范围以训练记录为准。', '',
        '所有模型逐场匹配完整目标 ID，身份、比分、标签和发表日期必须一致；缺失、重复或冲突会报错，'
        '不通过取交集删掉难预测的比赛。较早赛季不评价 clean_blend，该参考仅在较晚赛季存在。', '',
        '## 实际运行记录', '', *_metadata_lines(result['training_metadata']), '',
        '## 主要比较读数', '',
        '以下只陈述本次历史回放中的 Log loss 差值。负值表示动态模型平均损失更低，'
        '区间跨 0 表示该块长下差异方向未被区间区分，不能据此保证未来优势。', '']
    for lead in LEADS:
        for period in PERIODS:
            cohort = result['leads'][str(lead)]['periods'][period]
            parts = []
            for reference in ('static', 'dynamic_mean_only', 'regression_allcomp', 'clean_blend'):
                contrast = cohort['contrasts'].get(f'dynamic_minus_{reference}')
                if contrast is None:
                    continue
                values = contrast['log_loss']
                interval = values['interval_95_by_block_weeks']['4']
                if interval is None:
                    direction = '该区间不可估计'
                elif interval[1] < 0:
                    direction = '该区间全低于 0'
                elif interval[0] > 0:
                    direction = '该区间全高于 0'
                else:
                    direction = '该区间含 0'
                parts.append(f'相对{MODEL_LABELS[reference]} Δ={values["mean"]:.5f}，'
                    f'4 周区间 {_interval(interval)}（{direction}）')
            lines.append(f'- lead {lead}，{period}：'+'；'.join(parts)+'。')
    lines += ['', '## 两季主要评分', '',
        'Log loss、Brier（3 类平方误差之和）、RPS（主／平／客顺序，除以 2）和分类错误率越低越好；'
        'AUC 按各类 one-vs-rest 计算，越高越好。AUC 不能代替概率校准；逐类频率、偏差、Brier、10 个等宽概率箱与 ECE 见 evaluation.json。', '']
    for lead in LEADS:
        for period in PERIODS:
            cohort = result['leads'][str(lead)]['periods'][period]
            lines += [f'### lead {lead}，{period}（{cohort["n"]} 场）', '',
                '| 模型 | Log loss | Brier | RPS | 分类错误率 | 主胜 AUC | 平局 AUC | 客胜 AUC |',
                '|---|---:|---:|---:|---:|---:|---:|---:|']
            for arm, scores in cohort['scores'].items():
                values = [scores[m] for m in METRICS]
                values += [scores['one_vs_rest_auc'][c]['auc'] for c in CLASSES]
                lines.append('| ' + MODEL_LABELS[arm] + ' | ' + ' | '.join(_number(v) for v in values) + ' |')
            lines += ['', '动态模型逐场损失减去参考模型：负值有利于动态模型。以下为配对日历周移动块的 95% 百分位区间，'
                '每个块长 2000 次；区间条件于这些已保存预测，未包含历史研究选择或再次拟合的不确定性。', '',
                '| 参考 | 指标 | 平均差 | 2 周区间 | 4 周区间 | 8 周区间 |',
                '|---|---|---:|---|---|---|']
            for contrast, metrics in cohort['contrasts'].items():
                reference = contrast.removeprefix('dynamic_minus_')
                for metric, values in metrics.items():
                    intervals = values['interval_95_by_block_weeks']
                    lines.append(f'| {MODEL_LABELS[reference]} | {METRIC_LABELS[metric]} | '
                        f'{_number(values["mean"])} | ' + ' | '.join(_interval(intervals[str(w)]) for w in BLOCKS) + ' |')
            lines.append('')
    lines += ['## 分联赛结果', '',
        '每个分联赛单元继续保留该赛季的完整共同场次；完整逐类 AUC、校准与全部配对区间存于 evaluation.json。', '',
        '| lead | 赛季 | 联赛 | 场次 | 模型 | Log loss | Brier | RPS | 分类错误率 |',
        '|---:|---|---|---:|---|---:|---:|---:|---:|']
    for lead in LEADS:
        for period in PERIODS:
            for comp, cohort in result['leads'][str(lead)]['periods'][period]['by_league'].items():
                for arm, scores in cohort['scores'].items():
                    lines.append(f'| {lead} | {period} | {comp} | {cohort["n"]} | {MODEL_LABELS[arm]} | '
                        + ' | '.join(_number(scores[m]) for m in METRICS) + ' |')
    lines += ['', '分联赛、分赛季和多指标结果用于描述异质性，没有为多重比较另作显著性声明。'
        '动态均值消融只隔离预测时是否积分潜在技能不确定性，其与静态模型、旧参考模型的差异不等于单一机制的因果效应。', '',
        f'评分输入：`{result["source_forecasts"]}`；参考输入：`{result["source_references"]}`。',
        '完整结果：`evaluation.json`。']
    return '\n'.join(lines)+'\n'


def evaluate(out: Path, *, reference_path: Path | None = None) -> dict:
    """Validate and score a completed run, returning and saving its evaluation.

    ``evaluate(Path(run_directory))`` reads run_directory/forecasts.json and
    run_directory/data/references.json; an explicit reference_path is accepted.
    No training, tuning, old-output edits, outcome fetching, or adoption occurs.
    """
    out = _inside_project(out)
    reference_path = Path(reference_path) if reference_path is not None else out/'data/references.json'
    tracks, coverage, forecast_document = _load_tracks(out, reference_path)
    metadata_path = out/'training_summary.json'
    metadata = _read(metadata_path) if metadata_path.exists() else None
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError('training_summary.json must be a dictionary')
    result = {'created_at': datetime.now(timezone.utc).isoformat(),
        'source_forecasts': str(out/'forecasts.json'), 'source_references': str(reference_path.resolve()),
        'old_study_path': str(OLD_STUDY), 'class_order': list(CLASSES), 'coverage': coverage,
        'scope': {'periods': list(PERIODS), 'horizon_days': list(LEADS), 'output': 'one_x_two_only',
            'historical_replay_not_new_blind_test': True, 'availability': 'calendar_date_only_not_exact_24_72_hours',
            'neural_network': False, 'repeats': REPEATS, 'block_weeks': list(BLOCKS),
            'uncertainty': 'paired_saved_predictions_conditional_non_circular_calendar_week_percentile_95',
            'brier': 'sum_across_three_classes', 'rps': 'home_draw_away_order_divide_by_two',
            'no_clipping_or_renormalization': True, 'no_betting_returns_claimed': True,
            'old_formal_model_retained': True},
        'training_metadata': metadata, 'forecast_metadata': {k: v for k, v in forecast_document.items() if k != 'tracks'},
        'leads': {}, 'adopted': False}
    for lead in LEADS:
        result['leads'][str(lead)] = {'horizon_days': lead, 'periods': {}}
        for period in PERIODS:
            local = {arm: mapped for arm, mapped in tracks[str(lead)].items()
                     if arm != 'clean_blend' or period == '2025-26'}
            ids = sorted((k for k, r in local['dynamic'].items() if r['period'] == period),
                         key=lambda k: (local['dynamic'][k]['date'], k))
            cohort = _score_cohort(local, ids)
            cohort['by_league'] = {comp: _score_cohort(local,
                [k for k in ids if local['dynamic'][k]['competition'] == comp]) for comp in COMPS}
            result['leads'][str(lead)]['periods'][period] = cohort
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    report = _make_report(result)
    (out/'evaluation.json').write_text(encoded, encoding='utf-8')
    (out/'动态胜平负实跑结论.md').write_text(report, encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--reference-path', type=Path)
    args = parser.parse_args()
    summary = evaluate(args.out, reference_path=args.reference_path)
    print(json.dumps({'evaluation': str(args.out/'evaluation.json'), 'adopted': summary['adopted'],
        'counts': {lead: summary['coverage'][lead]['period_counts'] for lead in summary['leads']}}, ensure_ascii=False))
