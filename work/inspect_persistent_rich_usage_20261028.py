"""Read completed estimators to distinguish state use from added-column effects.

No fitting, counterfactual prediction, calibration, or branch selection.
Training impurity importance is descriptive, never time-out causal contribution.
"""
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import joblib
import numpy as np

ROOT = REPOSITORY_ROOT
OUT = ROOT / 'outputs/持续状态与丰富信息接入_20261028'
parameters = json.loads((OUT / '接入模型参数.json').read_text(encoding='utf-8'))
with gzip.open(OUT / '接入逐场概率.json.gz', 'rt', encoding='utf-8') as stream:
    records = json.load(stream)['records']

result = []
for cutoff, entry in parameters.items():
    if not entry['active_state_features']:
        continue
    model_params = entry['model']
    importance = model_params['impurity_feature_importance']
    raw_sum = math.fsum(value for name, value in importance.items() if name.startswith('gas_'))
    indicator_sum = math.fsum(value for name, value in importance.items() if name.startswith('missingindicator_gas_'))
    local = [r for r in records if r['fit_before_date'] == cutoff]
    row = {'cutoff': cutoff, 'target_count': len(local),
           'training_impurity_importance_raw_states_sum': raw_sum,
           'training_impurity_importance_state_missing_sum': indicator_sum,
           'changed_top1': sum(r['old_top1'] != r['new_top1'] for r in local),
           'corrected': sum(r['change_type'] == 'corrected' for r in local),
           'damaged': sum(r['change_type'] == 'damaged' for r in local),
           'mean_hda_ll_difference': math.fsum(-math.log(r['tracks']['state_rich']['probabilities'][r['actual_outcome']])
                 +math.log(r['tracks']['original']['probabilities'][r['actual_outcome']]) for r in local)/len(local)}
    # A zero-importance result alone does not prove the absence of every split;
    # inspect the actual fitted tree feature indices in those concrete cases.
    if raw_sum == 0:
        model = joblib.load(entry['model_path'])
        names = model_params['expanded_feature_names']
        raw_indices = [i for i, name in enumerate(names) if name.startswith('gas_')]
        indicator_indices = [i for i, name in enumerate(names) if name.startswith('missingindicator_gas_')]
        row['actual_tree_split_nodes_raw_state'] = sum(int(np.isin(tree.tree_.feature, raw_indices).sum())
                                                      for tree in model.estimator.estimators_)
        row['actual_tree_split_nodes_state_missing'] = sum(int(np.isin(tree.tree_.feature, indicator_indices).sum())
                                                          for tree in model.estimator.estimators_)
    result.append(row)

payload = {'inspected_at_utc': datetime.now(timezone.utc).isoformat(), 'windows': result,
    'new_fits': 0, 'new_probabilities': 0,
    'interpretation': 'Actual split references in zero-importance cases; other entries are training impurity summaries, not held-out or causal information value.',
    'limit': 'Added columns change randomized tree split opportunities even when no final tree references them; no rescue seed, ablation or extra learner is fitted.'}
(OUT / '状态列实际使用检查.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps([r for r in result if 'actual_tree_split_nodes_raw_state' in r], ensure_ascii=False, indent=2))
