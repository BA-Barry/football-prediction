"""Describe saved out-of-time scores by post-match red-card labels; no fit."""
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import re

from football_model.football_data import LEAGUES, parse_row
from football_model.native_experiment import season_directories
from football_model.probability import score_grid, summarize


ROOT = Path('G:/Codex/football-prediction-2026-10-22')
TARGET = ROOT / 'outputs/红牌事后分组诊断_20261024T161044Z'
SOURCE = ROOT / 'outputs/统计市场固定融合研究_20261024T120045031283Z/统计市场融合结果.json'
SHAPES = ROOT / 'outputs/比分形态诊断_20261023T123154423712Z'
GROUPS = ('all', 'no_red', 'home_only', 'away_only', 'both', 'unknown')


def same(actual, expected, name):
    if actual != expected:
        raise ValueError(f'{name}: {actual!r} != {expected!r}')


def close(actual, expected, name):
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError(f'{name}: {actual!r} != {expected!r}')


def unique(rows, label):
    result = {}
    for row in rows:
        if row['match_id'] in result:
            raise ValueError(f'Duplicate {label}: {row["match_id"]}')
        result[row['match_id']] = row
    return result


def card(text):
    text = (text or '').strip()
    if not text:
        return None, 'missing'
    if not re.fullmatch(r'[0-9]+', text):
        return None, 'invalid'
    return int(text), 'known'


def group(home, away):
    if home is None or away is None:
        return 'unknown'
    if home == 0 and away == 0:
        return 'no_red'
    if home > 0 and away == 0:
        return 'home_only'
    if home == 0 and away > 0:
        return 'away_only'
    return 'both'


def independent_scores(p, outcome):
    if len(p) != 3 or not all(math.isfinite(x) and x >= 0 for x in p):
        raise ValueError('Invalid archived probability')
    close(sum(p), 1., 'probability mass')
    if p[outcome] <= 0:
        raise ValueError('This diagnostic requires finite saved log losses')
    maxima = [i for i, value in enumerate(p) if value == max(p)]
    return {'log_loss': -math.log(p[outcome]),
            'brier_sum': sum((value - float(i == outcome)) ** 2 for i, value in enumerate(p)),
            'top1': float(outcome in maxima) / len(maxima)}


def summary(rows):
    n = len(rows)
    result = {'n': n, 'missing_red_labels': sum(r['red_group'] == 'unknown' for r in rows),
              'invalid_red_labels': sum(r['HR_status'] == 'invalid' or r['AR_status'] == 'invalid'
                                        for r in rows),
              'penalty_event_missing_n': n}
    if not rows:
        return {**result, 'lines': None, 'goals': None}
    result['lines'] = {line: {metric: math.fsum(r['scores'][line][metric] for r in rows) / n
        for metric in ('log_loss', 'brier_sum', 'top1')} for line in ('statistical', 'market')}
    result['statistical_minus_market'] = {metric: result['lines']['statistical'][metric]
        - result['lines']['market'][metric] for metric in ('log_loss', 'brier_sum', 'top1')}
    result['goals'] = {side: {'actual_sum': sum(r['goals'][i] for r in rows),
        'raw_rate_sum': math.fsum(r['rates'][i] for r in rows),
        'actual_minus_rate_mean': math.fsum(r['goals'][i] - r['rates'][i] for r in rows) / n}
        for i, side in enumerate(('home', 'away'))}
    return result


def summarize_scope(rows):
    groups = {name: summary(rows if name == 'all' else [r for r in rows if r['red_group'] == name])
              for name in GROUPS}
    same(sum(groups[name]['n'] for name in GROUPS if name != 'all'), len(rows), 'group partition')
    return groups


def main():
    plan = TARGET / '分析计划.md'
    if not plan.is_file():
        raise ValueError('Save the analysis plan before running this diagnostic')
    started = datetime.now(timezone.utc)
    if plan.stat().st_mtime >= started.timestamp():
        raise ValueError('Analysis plan must precede this run')
    study = json.loads(SOURCE.read_text(encoding='utf-8'))
    source_directories = season_directories(ROOT)
    joined = []
    audits = {}
    raw_headers = {}
    for season, n_expected in (('2024-25', 1736), ('2025-26', 1742)):
        pairs = unique(study['seasons'][season]['paired_matches'], 'saved probability')
        same(len(pairs), n_expected, 'expected season cohort')
        with (SHAPES / f'{season}_逐场形态.csv').open(encoding='utf-8-sig', newline='') as handle:
            shapes = unique(list(csv.DictReader(handle)), 'saved shape')
        same(set(pairs), set(shapes), 'probability/shape match IDs')
        label = season[:4] + '/' + season[-2:]
        raw = {}
        for league in sorted(LEAGUES):
            path = source_directories[label] / f'{league}.csv'
            source_url = f'https://football-data.co.uk/mmz4281/{label[2:4]+label[-2:]}/{league}.csv'
            with path.open(encoding='utf-8-sig', newline='') as handle:
                reader = csv.DictReader(handle)
                raw_headers[f'{season}:{league}'] = reader.fieldnames
                for row in reader:
                    record, _ = parse_row(row, competition=league, season=label,
                        source_path=str(path), source_url=source_url)
                    if record.match_id in raw:
                        raise ValueError(f'Duplicate raw CSV identity {record.match_id}')
                    raw[record.match_id] = (record, row)
        if set(pairs) - set(raw):
            raise ValueError('Saved forecast lacks exact raw CSV identity')
        for match_id, pair in pairs.items():
            shape = shapes[match_id]
            record, original = raw[match_id]
            for name, expected in (('date', record.match_date.isoformat()),
                                   ('competition', record.competition),
                                   ('home', record.home_label), ('away', record.away_label)):
                same(pair[name], expected, f'pair/{name}/{match_id}')
                same(shape[name], expected, f'shape/{name}/{match_id}')
            goals = tuple(int(shape[f'original_{side}_goals']) for side in ('home', 'away'))
            same(goals, record.full_time, f'raw/shape score/{match_id}')
            outcome = 0 if goals[0] > goals[1] else 2 if goals[0] < goals[1] else 1
            same(pair['actual_outcome'], outcome, f'outcome/{match_id}')
            rates = [float(shape[f'original_{side}_rate']) for side in ('home', 'away')]
            expected_p = list(summarize(score_grid(*rates, rho=0))['one_x_two'].values())
            for actual, expected in zip(pair['probabilities']['statistical'], expected_p):
                close(actual, expected, f'raw rate probability/{match_id}')
            scores = {}
            for line in ('statistical', 'market'):
                scores[line] = independent_scores(pair['probabilities'][line], outcome)
                for metric in scores[line]:
                    saved = pair['scores'][line][metric]
                    if metric == 'log_loss':
                        same(saved['status'], 'finite', 'saved log-loss status')
                        saved = saved['value']
                    close(scores[line][metric], saved, f'saved score/{line}/{metric}/{match_id}')
            hr, hr_status = card(original.get('HR'))
            ar, ar_status = card(original.get('AR'))
            joined.append({'match_id': match_id, 'season': season, 'date': pair['date'],
                'competition': record.competition, 'home': record.home_label, 'away': record.away_label,
                'goals': list(goals), 'rates': rates, 'probabilities': {
                    line: pair['probabilities'][line] for line in ('statistical', 'market')},
                'scores': scores, 'HR': hr, 'AR': ar, 'HR_status': hr_status, 'AR_status': ar_status,
                'red_group': group(hr, ar), 'raw_source_path': record.source_path,
                'source_url': record.source_url})
        audits[season] = {'saved_probability_n': len(pairs), 'saved_shape_n': len(shapes),
            'raw_csv_n': len(raw), 'joined_n': len(pairs), 'missing_raw_identity_n': 0,
            'duplicate_identity_n': 0, 'date_team_goal_outcome_mismatches': 0,
            'recomputed_score_mismatches': 0, 'raw_rate_probability_mismatches': 0}
    same(len({r['match_id'] for r in joined}), 3478, 'pooled unique identities')
    scopes = {}
    for season in ('2024-25', '2025-26', 'pooled'):
        selected = joined if season == 'pooled' else [r for r in joined if r['season'] == season]
        scopes[season] = {scope: summarize_scope(selected if scope == 'five_leagues'
            else [r for r in selected if r['competition'] == 'E0'])
            for scope in ('five_leagues', 'EPL')}
    result = {'generated_at': started.isoformat(), 'analysis_plan': str(plan),
        'saved_score_source': str(SOURCE), 'raw_rate_source': str(SHAPES),
        'raw_source_directories': {k: str(v) for k, v in source_directories.items()},
        'new_fits': 0, 'parameter_searches': 0, 'new_forecasts': 0,
        'unique_matches': len(joined), 'join_audits': audits, 'groups': scopes,
        'labels_are_post_match': True, 'red_card_times_verified': False,
        'penalty_event_data_available': False,
        'top1_rule': 'Expected hit under uniform selection among exact maximum probabilities.',
        'interpretation': 'Previously viewed historical exploratory description; no causal attribution, no red-card exclusion/refit, no pre-match red-card input.',
        'raw_csv_columns': raw_headers}
    (TARGET / '红牌分组结果.json').write_text(json.dumps(result, ensure_ascii=False, indent=2,
        allow_nan=False), encoding='utf-8')
    (TARGET / '逐场联结与评分.json').write_text(json.dumps(joined, ensure_ascii=False, indent=2,
        allow_nan=False), encoding='utf-8')
    report = ['# 红牌事后分组：已存概率误差诊断', '',
        '复用两季3,478场既有时间外预测；拟合0次、搜索0次、新预测0条。先保存分析计划，再联结和评分。'
        '唯一ID、日期、主客、比分、赛果、原始率派生概率及评分全部核对一致。', '',
        '红牌只是同场事后标签，缺可靠出牌时间；不能作为赛前特征，不能把分组差异解释成红牌的因果效应。'
        '点球事件数据未接入，以下不是点球诊断。all保留所有场次，未知标签单列；没有剔除红牌比赛重训。', '']
    for scope, scope_label in (('five_leagues', '五联赛总体'), ('EPL', '英超')):
        report += [f'## {scope_label}', '',
            '| 赛季/分组 | 场数 | 红牌未知 | 统计LL | 市场LL | 统计Brier | 市场Brier | 统计Top1 | 市场Top1 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for season in ('2024-25', '2025-26', 'pooled'):
            for name in GROUPS:
                s = scopes[season][scope][name]
                if s['n']:
                    st, mk = s['lines']['statistical'], s['lines']['market']
                    report.append(f'| {season}/{name} | {s["n"]} | {s["missing_red_labels"]} | '
                        f'{st["log_loss"]:.6f} | {mk["log_loss"]:.6f} | {st["brier_sum"]:.6f} | '
                        f'{mk["brier_sum"]:.6f} | {st["top1"]:.2%} | {mk["top1"]:.2%} |')
                else:
                    report.append(f'| {season}/{name} | 0 | 0 | — | — | — | — | — | — |')
        report += ['', '| 赛季/分组 | 场数 | 主实际进球减原rate均值 | 客实际进球减原rate均值 |',
                   '|---|---:|---:|---:|']
        for season in ('2024-25', '2025-26', 'pooled'):
            for name in GROUPS:
                s = scopes[season][scope][name]
                if s['n']:
                    report.append(f'| {season}/{name} | {s["n"]} | '
                        f'{s["goals"]["home"]["actual_minus_rate_mean"]:+.6f} | '
                        f'{s["goals"]["away"]["actual_minus_rate_mean"]:+.6f} |')
                else:
                    report.append(f'| {season}/{name} | 0 | — | — |')
        report.append('')
    report += ['LL、Brier越低越好；Brier为三类平方误差之和。Top1在精确并列时按均匀选择的期望计分，'
        '并独立复算原评分。分组样本不随机、人数不等，双方红牌组较少，不能按命中率高低推断结构或训练策略。', '',
        '两季均已查看，合并结果仅描述样本，不是新增独立验证；未做显著性筛选或新增概率修正。'
        '即使no_red中仍有误差，也不能自动认定原因是收缩、缺球员或缺射正。市场历史均价同样可能有错，'
        '且不是已核实可执行的赛前固定时点报价。全部样本、缺失和原始来源路径见JSON。', '']
    (TARGET / '红牌事后诊断说明.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({'directory': str(TARGET), 'unique_matches': len(joined), 'audits': audits,
        'pooled': scopes['pooled'], 'new_fits': 0, 'parameter_searches': 0},
        ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
