"""Describe existing same-fixture probability errors; no fitting or tuning."""
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
SOURCE = ROOT / 'outputs/统计市场固定融合研究_20261024T120045031283Z/统计市场融合结果.json'


def summary(rows):
    if not rows:
        return {'n': 0}
    n = len(rows)
    values = {'n': n, 'lines': {}}
    for line in ('statistical', 'market'):
        values['lines'][line] = {
            key: sum(r['scores'][line][key]['value'] if key == 'log_loss'
                     else r['scores'][line][key] for r in rows) / n
            for key in ('log_loss', 'brier_sum', 'top1')}
    values['log_loss_statistical_minus_market'] = values['lines']['statistical']['log_loss'] - values['lines']['market']['log_loss']
    values['outcome_calibration'] = []
    for index, outcome in enumerate(('home', 'draw', 'away')):
        observed = sum(r['actual_outcome'] == index for r in rows) / n
        values['outcome_calibration'].append({
            'outcome': outcome, 'observed_frequency': observed,
            **{line: {'mean_probability': sum(r['probabilities'][line][index] for r in rows) / n,
                       'observed_minus_probability': observed - sum(r['probabilities'][line][index] for r in rows) / n}
               for line in ('statistical', 'market')}})
    return values


def main():
    study = json.loads(SOURCE.read_text(encoding='utf-8'))
    now = datetime.now(timezone.utc)
    result = {'generated_at': now.isoformat(), 'source': str(SOURCE), 'new_fits': 0,
              'parameter_searches': 0, 'purpose': 'Exploratory description of previously viewed historical scores, not new validation or causal attribution.',
              'seasons': {}}
    report = ['# 从已保存逐场概率定位误差', '',
              '只重读此前已看过的历史前推输出，新增拟合0次、参数搜索0次。原预测、原权重和新前瞻预测均未修改。分组用于定位候选问题，未根据本次分组选参数；不能作为新的独立验证。', '',
              '| 赛季／范围 | 场数 | 统计LL | 市场LL | 统计减市场 | 统计Top1 | 市场Top1 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for season, data in study['seasons'].items():
        rows = data['paired_matches']
        if len({r['match_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate historical fixture')
        grouped = {'all': summary(rows), 'by_league': {}, 'by_market_favourite_band': {}, 'by_actual_outcome': {}}
        for league in sorted({r['competition'] for r in rows}):
            grouped['by_league'][league] = summary([r for r in rows if r['competition'] == league])
        for low, high in ((0, .4), (.4, .5), (.5, .6), (.6, .7), (.7, 1.000001)):
            selected = [r for r in rows if low <= max(r['probabilities']['market']) < high]
            entry = summary(selected)
            if selected:
                st = mk = hit = 0.
                for row in selected:
                    p = row['probabilities']['market']
                    choices = [i for i, v in enumerate(p) if v == max(p)]
                    st += sum(row['probabilities']['statistical'][i] for i in choices) / len(choices)
                    mk += max(p)
                    hit += (row['actual_outcome'] in choices) / len(choices)
                entry['market_favourite_condition'] = {
                    'statistical_mean_probability_for_that_outcome': st / len(selected),
                    'market_mean_probability': mk / len(selected),
                    'actual_frequency_fractional_tie': hit / len(selected)}
            grouped['by_market_favourite_band'][f'{low:.1f}..{min(high, 1):.1f}'] = entry
        for index, label in enumerate(('home', 'draw', 'away')):
            grouped['by_actual_outcome'][label] = summary([r for r in rows if r['actual_outcome'] == index])
        result['seasons'][season] = grouped
        for label, entry in [('all', grouped['all']), *grouped['by_league'].items()]:
            st, mk = entry['lines']['statistical'], entry['lines']['market']
            report.append(f"| {season}/{label} | {entry['n']} | {st['log_loss']:.6f} | {mk['log_loss']:.6f} | {entry['log_loss_statistical_minus_market']:+.6f} | {st['top1']:.2%} | {mk['top1']:.2%} |")
    report += ['', '## 强弱差与热门概率的描述性检查', '',
               '分组由同场市场最高概率定义，仅用于解释市场相对统计线的差异，不把市场当真值。精确并列最高时按均匀选择的期望计数。', '',
               '| 赛季 | 市场热门概率范围 | 场数 | 统计给该方向 | 市场给该方向 | 实际该方向频率 | 全三类LL差 |',
               '|---|---|---:|---:|---:|---:|---:|']
    for season, grouped in result['seasons'].items():
        for band, entry in grouped['by_market_favourite_band'].items():
            if not entry['n']:
                continue
            c = entry['market_favourite_condition']
            report.append(f"| {season} | {band} | {entry['n']} | {c['statistical_mean_probability_for_that_outcome']:.2%} | {c['market_mean_probability']:.2%} | {c['actual_frequency_fractional_tie']:.2%} | {entry['log_loss_statistical_minus_market']:+.6f} |")
    report += ['', '## 主／平／客整体概率偏差', '',
               '| 赛季 | 结果 | 真实频率 | 统计平均概率 | 市场平均概率 |',
               '|---|---|---:|---:|---:|']
    for season, grouped in result['seasons'].items():
        for row in grouped['all']['outcome_calibration']:
            report.append(f"| {season} | {row['outcome']} | {row['observed_frequency']:.2%} | {row['statistical']['mean_probability']:.2%} | {row['market']['mean_probability']:.2%} |")
    report += ['', 'LL越低越好；整体平均概率接近真实频率不代表逐场校准充分。实际结果分组会天然偏向该结果，因此其损失只能作描述，不能用于选择新的概率权重。', '',
               '这些输出能检验是否反复存在强弱差过小、整体主客偏差或平局概率不足的现象；无法单独区分收缩、数据遗漏、独立Poisson结构或球员因素的因果贡献。市场也可能错误，且历史均价并非可核实的当前可成交价格。', '',
               '没有从分组中挑一项自动升级。后续若改强度层、分布层或LLM提示，保留原线路，只做一项有解释的候选，并用之后的真实比赛配对验证。当前两场前瞻失利不能单独证明任何一种结构需要更换。', '']
    target = ROOT / 'outputs' / now.strftime('已存概率误差定位_%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True, exist_ok=False)
    (target / '逐组概率误差.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (target / '概率误差定位说明.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({'directory': str(target), 'seasons': {s: g['all']['n'] for s, g in result['seasons'].items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
