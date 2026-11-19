"""Independent arithmetic/history audit of the repaired base export."""
from collections import Counter
from datetime import date
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln

from run_npxg_comparison_20261025 import pairs, load_source
from run_shot_increment_20261025 import write

OUT = REPOSITORY_ROOT / Path('outputs/干净基础专家重建_20261025T064408733954Z')


def main():
    export = json.loads((OUT/'clean_base_rows.json').read_text(encoding='utf-8'))
    params = json.loads((OUT/'干净基础专家参数.json').read_text(encoding='utf-8'))
    rows = export['rows']
    source, independent, *_ = load_source()
    excluded = {k for k,r in independent.items() if r.goal_fit_exclusion}
    counts, ref_count, changed, max_delta = Counter(), 0, [], 0.
    all_originals = {}
    originals = pairs()
    for group, saved in originals.items():
        for r in saved['goal']:
            all_originals[r['match_id']] = r
        for key, pair in params[group].items():
            gp, xp = pair['goal'], pair['xg']
            assert gp['reference_match_ids'] == xp['reference_match_ids']
            ids = gp['reference_match_ids']
            assert not excluded.intersection(ids)
            assert all(k in source and source[k].match_date.isoformat() < gp['before_date'] for k in ids)
            assert len(ids) == gp['n_matches'] == xp['n_matches']
            assert ids == [r.match_id for r in sorted((source[k] for k in ids), key=lambda r:(r.match_date,r.match_id))]
            assert pair['removed_match_ids'] == [k for k in saved['goal_parameters'][key]['reference_match_ids'] if k not in ids]
            ref_count += 2*len(ids)
            counts[pair['action']] += 2
            if pair['removed_match_ids']:
                assert pair['action'] == 'refitted_changed_design'
                changed.append({'key': f'{group}:{key}', 'removed': pair['removed_match_ids']})
    assert len(set(r['match_id'] for r in rows)) == len(rows) == export['n']
    assert not excluded.intersection(r['match_id'] for r in rows)
    assert len(all_originals)-len(rows) == 2
    for row in rows:
        r = independent[row['match_id']]
        assert row['actual_score'] == list(r.full_time)
        assert row['fit_before_date'] <= row['date']
        assert all(math.isfinite(x) and x > 0 for x in row['rates']+row['xg_rates']+row['blend_rates'])
        max_delta=max(max_delta,max(abs(a-b) for a,b in zip(row['rates'],all_originals[row['match_id']]['rates'])))
    weight_checks = {}
    for label, saved in originals.items():
        training=[r for r in rows if r['period'] == saved['fit_period']]
        parameter = export['groups'][label]['blend_parameters']
        assert set(parameter['reference_match_ids']) == {r['match_id'] for r in training}
        y=np.array([r['actual_score'] for r in training],dtype=float)
        g=np.log(np.array([r['rates'] for r in training]))
        x=np.log(np.array([r['xg_rates'] for r in training]))
        derivative=lambda w: float(np.mean(np.sum((np.exp(g+w*(x-g))-y)*(x-g),axis=1)))
        a,b=derivative(0),derivative(1)
        w=0. if a>=0 else 1. if b<=0 else brentq(derivative,0,1,xtol=1e-14)
        assert abs(w-parameter['weight']) < 1e-12
        nll=float(np.mean(np.sum(np.exp(g+w*(x-g))-y*(g+w*(x-g))+gammaln(y+1),axis=1)))
        assert abs(nll-parameter['full_nll'])<1e-12
        weight_checks[label]={'n':len(training),'weight':w,'objective':nll,'gradient':derivative(w)}
    summary={'status':'passed','n':len(rows),'by_period':dict(Counter(r['period'] for r in rows)),
        'base_fits_by_action':dict(counts),'checked_reference_entries':ref_count,
        'changed_base_designs':changed,'clean_blend_checks':weight_checks,
        'max_changed_goal_rate_delta':max_delta,'max_unchanged_replay_error':export['max_unchanged_rate_replay_error'],
        'current_results_requests':0,'limitations':'current registered exceptions removed; does not establish unknown-source perfect 90minute coverage'}
    write(OUT/'基础重建独立核对.json',summary)
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    main()
