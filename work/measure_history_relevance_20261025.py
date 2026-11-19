"""One descriptive real-history measurement; no training or threshold search."""
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from football_model.historical_relevance import personnel_usage_overlap


ROOT=Path('G:/Codex/football-prediction-2026-10-22')
OUT=ROOT/'outputs/历史代表性与证据联合_20261025/history'
LINEUPS=ROOT/'outputs/英超跨季首发覆盖_20261025/首发身份观察.json'
PREDICTIONS=ROOT/'outputs/直接胜平负分类研究_20261025T091329913808Z/lead_3/逐场胜平负概率.json'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def summary(rows):
    if not rows:return {'n':0,'accuracy':None,'log_loss':None}
    return {'n':len(rows),'correct':sum(r['top1_correct'] for r in rows),
            'accuracy':sum(r['top1_correct'] for r in rows)/len(rows),
            'log_loss':sum(r['log_loss'] for r in rows)/len(rows)}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    observations=read(LINEUPS)['observations']
    index={r['match_id']:r for r in observations}
    if len(index)!=len(observations):raise ValueError('Source fixture identity repeats')
    predictions=[r for r in read(PREDICTIONS)['regression_allcomp'] if r['period']=='2024-25' and r['competition']=='E0']
    paired=[];missing=[]
    for r in predictions:
        ident=r['match_id'];source=index.get(ident)
        if source is None:
            missing.append({'match_id':ident,'reason':'no_exact_source_identity'});continue
        if any(source[k]!=r[k] for k in ('home','away')) or source['match_date']!=r['date']:
            raise ValueError('Source and prediction fixture identity disagree')
        profiles={side:personnel_usage_overlap(observations,r[side],r['forecast_before_date']) for side in ('home','away')}
        y=r['actual_outcome'];p=r['probabilities']
        if p[y]<=0:raise ValueError('Native saved loss is infinite; do not silently clip')
        both=all(profiles[s]['overlap'] is not None for s in profiles)
        paired.append({**r,'personnel_history':profiles,
            'pair_min_overlap':min(profiles[s]['overlap'] for s in profiles) if both else None,
            'log_loss':-math.log(p[y]),'top1_correct':int(max(range(3),key=lambda c:p[c])==y)})
    complete=[r for r in paired if r['pair_min_overlap'] is not None]
    # Cutpoints are prescribed in the task; no search or adoption by bin score.
    bins={'below_0.50':[r for r in complete if r['pair_min_overlap']<.5],
          '0.50_to_below_0.75':[r for r in complete if .5<=r['pair_min_overlap']<.75],
          '0.75_and_above':[r for r in complete if r['pair_min_overlap']>=.75]}
    values=np.asarray([r['pair_min_overlap'] for r in complete])
    losses=np.asarray([r['log_loss'] for r in complete])
    correlation=float(np.corrcoef(values,losses)[0,1]) if len(complete)>1 and values.std()>0 and losses.std()>0 else None
    distribution={'n':len(values),'min':float(values.min()),'q25':float(np.quantile(values,.25)),
                  'median':float(np.median(values)),'q75':float(np.quantile(values,.75)),
                  'max':float(values.max())} if len(values) else {'n':0}
    result={'generated_at':datetime.now(timezone.utc).isoformat(),'sources':[str(LINEUPS),str(PREDICTIONS)],
        'measurement':'sum min(normalized observed starter shares), last 5 vs preceding disjoint 15, per team',
        'pair_summary':'minimum of the two team overlap values; a descriptive summary, not learned match risk',
        'horizon_days':3,'target_period':'2024-25','competition':'E0','route':'regression_allcomp',
        'source_matches':len(observations),'target_predictions':len(predictions),'matched_predictions':len(paired),
        'missing_exact_identities':missing,'complete_disjoint_windows':len(complete),
        'incomplete_disjoint_windows':len(paired)-len(complete),'all_matched_scores':summary(paired),
        'complete_window_scores':summary(complete),'overlap_distribution':distribution,
        'pearson_overlap_vs_log_loss':correlation,'fixed_bins':{name:summary(rows) for name,rows in bins.items()},
        'incomplete_window_reasons':dict(Counter('+'.join(s for s in ('home','away')
            if r['personnel_history'][s]['overlap'] is None)
            for r in paired if r['pair_min_overlap'] is None)),
        'inference_limits':['Historical original publication times unknown: strictly earlier calendar-date exploration only.',
            'Observed XI usage is not actual minutes, transfers, official current roster or player strength.',
            'Early new-season cutoffs may still contain only previous-season matches; no future summer roster is inferred.',
            'The overlap can change from rotation, injury or tactics: it does not identify their causes.',
            'Bin results are descriptive; no statistical significance, causality, future accuracy or improvement claim.',
            'No model fitted, threshold searched, weights changed or existing forecasts overwritten.'],
        'model_fits':0,'threshold_searches':0,'current_2026_27_results_read':False}
    write(OUT/'逐场人员使用重叠.json',{'rows':paired})
    write(OUT/'历史代表性测量结果.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
