"""Make model-layer dates explicit without refitting or changing any numbers."""
from datetime import datetime, timezone
import json
from pathlib import Path

PATH = Path('G:/Codex/football-prediction-2026-10-22/outputs/固定总球主客分配研究_20261024T182604417382Z/固定总球主客分配结果.json')


def main():
    result = json.loads(PATH.read_text(encoding='utf-8'))
    labels = {'model_type','allocation_fit_before_date','allocation_training_matches','prediction_role','shared_xg_weight'}
    count = 0
    for name,entry in result['splits'].items():
        early = name[:7]
        for row in entry['predictions']:
            before = {k:v for k,v in row.items() if k not in labels}
            row.pop('shared_xg_weight',None)
            row.update({'model_type':'fixed_total_conditional_allocation',
                'allocation_fit_before_date':entry['parameters']['before_date'],
                'allocation_training_matches':entry['parameters']['n'],
                'prediction_role':('allocation_estimation_apparent_fit_not_out_of_time' if row['period']==early
                                   else 'held_earlier_allocation_weight')})
            if before != {k:v for k,v in row.items() if k not in labels}:
                raise ValueError('Label update changed prediction numbers')
            if row['prediction_role']=='held_earlier_allocation_weight' and not row['allocation_fit_before_date']<=row['date']:
                raise ValueError('Allocation parameter date is later than held prediction')
            count += 1
    result['metadata_clarification'] = {'at':datetime.now(timezone.utc).isoformat(),'rows':count,
        'purpose':'separate old full-rate weight from new allocation weight and distinguish early apparent fits from held predictions',
        'probabilities_and_scores_changed':False,'new_fits':0}
    PATH.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result['metadata_clarification'],ensure_ascii=False))


if __name__ == '__main__':
    main()
