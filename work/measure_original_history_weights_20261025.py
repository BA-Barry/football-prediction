"""Reconstruct and measure saved forecast sample weights, without any fit."""
from collections import Counter
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path

from football_model.batch_projection import _history
from football_model.current_projection import load_current_context

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
SOURCE = ROOT/'outputs/批量球队投影_20261024T150805722459Z/批量球队投影与来源.json'
OUTPUT = ROOT/'outputs/当前人员历史代表性_20261025'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def measure(rows, before_date, half_life, seasons):
    weighted = [(row, math.exp(-math.log(2) * (before_date-row.match_date).days / half_life))
                for row in rows]
    total = math.fsum(weight for _, weight in weighted)
    counts = Counter(row.season for row, _ in weighted)
    by_season = []
    for season in seasons:
        values = [weight for row, weight in weighted if row.season == season]
        mass = math.fsum(values)
        by_season.append({'season': season, 'matches': counts[season], 'weight_sum': mass,
                          'share_of_weight_sum': mass/total if total else None,
                          'min_match_weight': min(values, default=None),
                          'max_match_weight': max(values, default=None)})
    return {'matches': len(rows), 'weight_sum': total, 'by_season': by_season}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    params = source['model_parameters']
    as_of = datetime.fromisoformat(source['fit_input_cutoff'])
    before_date = date.fromisoformat(params['before_date'])
    half_life = params['half_life_days']
    report = {'measured_at': datetime.now(timezone.utc).isoformat(), 'source_projection': str(SOURCE),
        'original_fit_cutoff': as_of.isoformat(), 'before_date': before_date.isoformat(),
        'half_life_days': half_life, 'formula': 'exp(-log(2) * calendar_age_days / half_life_days)',
        'model_fit_count': 0, 'prediction_calls': 0, 'outcome_evaluation_count': 0,
        'downloads': 0, 'current_team_count_expected': 20,
        'interpretation': 'Sample weight mass only; not coefficient, player-strength or prediction contribution.',
        'goal_row_note': 'Original fit duplicates each match weight for its home and away goal observations. Ratios are identical whether summing match weights or both goal-row weights.'}
    errors = []
    try:
        historical, sources = _history(ROOT, as_of)
        context = Path(source['source_inspection']).parent
        current, inspection, supplements = load_current_context(context, as_of)
        all_rows = historical + current
        eligible = sorted((row for row in all_rows if row.match_date < before_date
                           and row.goal_fit_exclusion is None), key=lambda row: (row.match_date, row.match_id))
        saved_ids = params['reference_match_ids']
        rebuilt_ids = [row.match_id for row in eligible]
        report['reconstruction'] = {'historical_sources': sources, 'context_directory': str(context),
            'historical_rows': len(historical), 'current_rows': len(current), 'total_input_rows': len(all_rows),
            'eligible_training_rows': len(eligible), 'saved_training_rows': source['training_matches'],
            'saved_reference_ids_count': len(saved_ids), 'reference_ids_exact_order_equal': rebuilt_ids == saved_ids,
            'current_source_records': len(inspection['records']), 'current_supplements': len(supplements['records']),
            'input_season_counts': dict(sorted(Counter(row.season for row in all_rows).items()))}
        if len(eligible) != 1591 or source['training_matches'] != 1591:
            errors.append('Training history does not reproduce the requested 1591 rows')
        if rebuilt_ids != saved_ids:
            errors.append('Rebuilt training IDs/order differ from saved original reference_match_ids')
            report['reconstruction']['missing_saved_ids'] = sorted(set(saved_ids)-set(rebuilt_ids))
            report['reconstruction']['additional_rebuilt_ids'] = sorted(set(rebuilt_ids)-set(saved_ids))
        if before_date != date(2026, 10, 24) or half_life != 180:
            errors.append('Saved date or half-life differs from the requested original setup')
        if sources != source['sources']:
            errors.append('Original historical source paths/observations/counts were not reproduced exactly')
        teams = sorted(set(source['alias_map'].values()))
        if len(teams) != 20:
            errors.append('Current team labels are not exactly the original twenty explicit aliases')
        if not errors:
            seasons = sorted({row.season for row in eligible})
            report['league_sample_weights'] = measure(eligible, before_date, half_life, seasons)
            report['current_team_sample_weights'] = []
            for team in teams:
                team_id = f'football-data:E0:{team}'
                sample = [row for row in eligible if team_id in (row.home, row.away)]
                report['current_team_sample_weights'].append({'team': team_id, 'label': team,
                    **measure(sample, before_date, half_life, seasons)})
            report['status'] = 'exact_original_history_reconstructed_and_measured'
        else:
            report['status'] = 'reconstruction_gap_no_weight_claims'
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(str(exc))
        report['status'] = 'reconstruction_gap_no_weight_claims'
    report['errors'] = errors
    write(OUTPUT/'原预测历史权重.json', report)
    lines = ['# 原预测历史样本权重实测', '',
        '本次只重建原始输入并计算样本权重，没有重新拟合、预测、评估赛果或下载。', '',
        f'原预测文件：{SOURCE}', '', f'状态：{report["status"]}', '']
    if errors:
        lines += ['未能精确重建，不报告推测的权重占比。', ''] + [f'- {error}' for error in errors]
    else:
        league = report['league_sample_weights']
        lines += ['原1591场逐条ID及顺序与保存的训练记录完全一致：四个完整赛季各380场，加本季71场。', '',
            '原公式为 `exp(-log(2) × 距2026-10-24的日数 / 180)`；每过180日，单场权重减半。', '',
            '| 赛季 | 比赛数 | 权重总和 | 全部样本权重占比 |', '|---|---:|---:|---:|']
        for row in league['by_season']:
            lines.append(f'| {row["season"]} | {row["matches"]} | {row["weight_sum"]:.6f} | {row["share_of_weight_sum"]:.2%} |')
        lines += ['', '以下分母改为该球队参加比赛的权重总和；全联赛各球队表不能相加。', '',
            '| 球队 | 本队场数 | 2022/23 | 2023/24 | 2024/25 | 2025/26 | 2026/27 |', '|---|---:|---:|---:|---:|---:|---:|']
        for team in report['current_team_sample_weights']:
            shares = ' | '.join(f'{row["share_of_weight_sum"]:.2%}' for row in team['by_season'])
            lines.append(f'| {team["label"]} | {team["matches"]} | {shares} |')
        lines += ['', '这只是训练样本权重占比，不等于参数贡献，也不等于球员实力的保留比例。', '',
            '攻防系数还受到对手、主客场、全联赛联合拟合和正则化影响；不能从本表直接推出某季对某场胜率贡献多少。', '',
            '原模型没有显式的球员更替或转会特征。时间衰减能减轻旧赛季的直接样本权重，但不能自动识别当前阵容结构改变。', '',
            '原拟合将每场权重重复用于主、客队两条进球观测；是否重复计数不会改变本报告各赛季比例。']
    (OUTPUT/'原预测历史权重.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'reconstruction': report.get('reconstruction'),
        'league_sample_weights': report.get('league_sample_weights'), 'errors': errors,
        'output': str(OUTPUT/'原预测历史权重.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
