"""Measure final-point stationarity, without any fit or parameter change."""
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path

from football_model.persistent_goal_difference import PersistentGoalDifferenceFilter, PARAMETER_NAMES, DYNAMIC_BOUNDS

OUT = Path('G:/Codex/football-prediction-2026-10-22/outputs/持续状态净胜球研究_20261028')
static = json.loads((OUT / '静态初始化.json').read_text(encoding='utf-8'))
fit = json.loads((OUT / '动态参数拟合.json').read_text(encoding='utf-8'))
payload = json.load(gzip.open(OUT / '输入资料.json.gz', 'rt', encoding='utf-8'))
days = defaultdict(list)
for row in payload['matches']:
    if row['season'] == '2023/24':
        days[row['date']].append(row)


def nll(params):
    state = PersistentGoalDifferenceFilter(static, params)
    return -math.fsum(value for rows in days.values() for value in state.observe_day(rows)['log_likelihoods'])


params = fit['parameters']
value = nll(params)
gradient, projected = {}, {}
epsilon = 1e-7
for name, (lo, hi) in zip(PARAMETER_NAMES, DYNAMIC_BOUNDS):
    step = -epsilon if params[name] == hi else epsilon
    perturbed = {**params, name: params[name] + step}
    derivative = (nll(perturbed) - value) / step
    gradient[name] = derivative
    projected[name] = (0. if (params[name] == hi and derivative < 0)
                       or (params[name] == lo and derivative > 0) else derivative)
result = {'purpose': 'post-fit final-point derivative measurement; no optimization or parameter changes',
          'training_matches': sum(map(len, days.values())), 'epsilon': epsilon,
          'reevaluated_nll': value, 'saved_nll_difference': value - fit['optimizer']['nll'],
          'gradient_one_sided': gradient, 'projected_gradient': projected,
          'max_abs_projected_gradient': max(map(abs, projected.values()))}
(OUT / '最终点数值诊断.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
