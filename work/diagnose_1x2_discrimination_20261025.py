"""Describe saved historical discrimination; no fitting or new result requests."""
from pathlib import Path
import json

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path('G:/Codex/football-prediction-2026-10-22')
SOURCE = ROOT/'outputs/直接胜平负分类研究_20261025T091329913808Z'
OUT = ROOT/'outputs/胜平负核心调研_20261025'
NAMES = {
    'goal': '纯进球专家', 'xg_raw': '独立xG专家',
    'regression_allcomp': '丰富信息进球回归',
    'extra_trees_allcomp': '丰富信息直接分类树',
    'logistic_allcomp': '丰富信息多项逻辑回归',
    'clean_blend': '较早学得的进球/xG融合',
}


def pair_rank_auc(y, p):
    """Mann-Whitney AUC with half credit for exactly tied scores."""
    y, p = np.asarray(y, dtype=bool), np.asarray(p, dtype=float)
    positives, negatives = int(y.sum()), int((~y).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(p, kind='stable')
    y, p = y[order], p[order]
    rank_sum, start = 0., 0
    while start < len(p):
        end = start+1
        while end < len(p) and p[end] == p[start]:
            end += 1
        rank_sum += float(y[start:end].sum())*(start+1+end)/2
        start = end
    return (rank_sum-positives*(positives+1)/2)/(positives*negatives)


def describe(rows):
    p = np.asarray([row['probabilities'] for row in rows], dtype=float)
    y = np.asarray([row['actual_outcome'] for row in rows], dtype=int)
    if p.shape != (len(rows), 3) or not np.isfinite(p).all():
        raise ValueError('Need the saved finite three-class probabilities')
    result = {'n': len(rows), 'by_class': {}}
    for k, name in enumerate(('home', 'draw', 'away')):
        indicator = y == k
        auc = pair_rank_auc(indicator, p[:, k])
        check = float(roc_auc_score(indicator, p[:, k]))
        if abs(auc-check) > 1e-12:
            raise ValueError('Independent rank and library AUC differ')
        result['by_class'][name] = {
            'n_actual': int(indicator.sum()), 'auc': auc,
            'mean_probability': float(p[:, k].mean()),
            'actual_frequency': float(indicator.mean()),
            'probability_sd': float(p[:, k].std()),
            'mean_probability_when_actual': float(p[indicator, k].mean()),
            'mean_probability_when_not_actual': float(p[~indicator, k].mean()),
            'n_top1': int((p.argmax(axis=1) == k).sum()),
        }
    decisive = y != 1
    home_conditional = p[decisive, 0]/(p[decisive, 0]+p[decisive, 2])
    result['decisive_historical_subset_only'] = {
        'n': int(decisive.sum()),
        'home_vs_away_auc': pair_rank_auc(y[decisive] == 0, home_conditional),
        'note': 'Conditioned on realised non-draw only for diagnosis; not a deployable forecast filter',
    }
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {'source': SOURCE.as_posix(), 'fitted_models': 0,
              'new_results_queried': 0, 'current_results_reviewed': False,
              'interpretation': 'Descriptive AUC, not probability loss, accuracy, causal attribution or an adoption decision; no interval estimated',
              'horizons': {}}
    lines = ['# 已保存胜平负概率：辨别能力诊断', '',
             '只读取上轮已评分的2024/25、2025/26历史概率，没有训练、校准、选阈值或查询新赛果。'
             'AUC衡量该类真实发生的比赛是否通常获更高概率；0.5对应无排序辨别，'
             '它不是命中率，也不能替代主概率损失。以下均为描述量，没有区间或因果结论。', '',
             '前1日与前3日比赛高度重叠，不能视为独立重复。后季此前已查看，本诊断不是新盲测。', '']
    for lead in (1, 3):
        saved = json.loads((SOURCE/f'lead_{lead}'/'逐场胜平负概率.json').read_text(encoding='utf-8'))
        result['horizons'][str(lead)] = {}
        for period in ('2024-25', '2025-26'):
            section = result['horizons'][str(lead)][period] = {}
            identity = None
            lines += [f'## 前{lead}日，{period}', '',
                      '| 方案 | 场数 | 主胜AUC | 平局AUC | 客胜AUC | 平局平均概率 | 实际平局比例 | 平局为首选场数 |',
                      '|---|---:|---:|---:|---:|---:|---:|---:|']
            for arm, name in NAMES.items():
                rows = [row for row in saved[arm] if row['period'] == period]
                if not rows:
                    continue
                ids = [(row['match_id'], row['actual_outcome']) for row in rows]
                if identity is not None and ids != identity:
                    raise ValueError('Comparison needs the same ordered historical fixtures')
                identity = ids
                value = section[arm] = describe(rows)
                a = value['by_class']
                d = a['draw']
                lines.append(f'| {name} | {value["n"]} | {a["home"]["auc"]:.4f} | '
                             f'{d["auc"]:.4f} | {a["away"]["auc"]:.4f} | '
                             f'{d["mean_probability"]:.2%} | {d["actual_frequency"]:.2%} | {d["n_top1"]} |')
            lines.append('')
    lines += ['## 如何用于下一步设计', '',
              '提高平局的平均概率、让更多比赛把平局列首选、识别真正容易平局的比赛，是不同问题。'
              '下一项方法需同时报告平局辨别与完整三类概率评分，不通过强制类别比例来宣称增准。'
              '若某条新线仅改善总体频率或分箱，而不改善逐场区分，继续增加该类输出次数没有充分依据。', '',
              '本次没有估计可提升上限；AUC接近0.5不能证明足球平局无法预测，也不能证明某一新增状态结构必然有效。'
              '既有输入、算法复杂度、历史范围及资料可用时间都会影响观察结果。', '',
              '评分、可靠性与辨别应分开审视的方法背景可参见'
              '[Foulley原预印本](https://arxiv.org/abs/2106.14345)；'
              '[Wheatcroft原论文稿](https://arxiv.org/abs/1908.08980)讨论了足球概率评分的选择。'
              '本报告保留既定Log loss主指标，未因调研改换有利指标。', '']
    (OUT/'历史胜平负辨别诊断.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT/'历史胜平负辨别诊断.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines[:21]))


if __name__ == '__main__':
    main()
