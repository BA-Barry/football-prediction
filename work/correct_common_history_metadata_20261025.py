"""Correct copied training descriptions; do not refit or change predictions."""
from datetime import datetime, timezone
import json
from pathlib import Path

RUN = Path('G:/Codex/football-prediction-2026-10-22/outputs/进球xG共同训练记录_20261024T174247963857Z')


def main():
    path = RUN/'共同训练记录融合结果.json'
    result = json.loads(path.read_text(encoding='utf-8'))
    corrections = {}
    for arm in ('common_goal_predictions', 'common_blend_predictions', 'fixed_old_weight_predictions'):
        changed_count, changed_max = 0, 0
        for row in result[arm]:
            parameters = result['common_goal_parameters'][f'{row["competition"]}:{row["fit_before_date"]}']['parameters']
            old_count, old_max = row['training_matches'], row['training_max_date']
            changed_count += old_count != parameters['n_matches']
            changed_max += old_max != parameters['training_max_date']
            row['original_training_matches'] = old_count
            row['original_training_max_date'] = old_max
            row['training_matches'] = parameters['n_matches']
            row['training_max_date'] = parameters['training_max_date']
        corrections[arm] = {'rows': len(result[arm]), 'training_matches_changed': changed_count,
                            'training_max_date_changed': changed_max}
    result['metadata_correction'] = {
        'at': datetime.now(timezone.utc).isoformat(),
        'cause': 'copied original prediction training descriptions, although common-history rates were already correct',
        'corrections': corrections,
        'preserved': ['all rates', 'all probabilities and forecasts', 'all scores and intervals',
                      'all fitted parameters', 'all original external experiment files'],
        'refits': 0,
    }
    allowed_fields = {'original_training_matches', 'original_training_max_date',
                      'training_matches', 'training_max_date'}
    original = json.loads(path.read_text(encoding='utf-8'))
    for arm in corrections:
        for before, after in zip(original[arm], result[arm], strict=True):
            if {k:v for k,v in before.items() if k not in allowed_fields} != {
                    k:v for k,v in after.items() if k not in allowed_fields}:
                raise ValueError('Metadata correction unexpectedly changed prediction content')
            p = result['common_goal_parameters'][f'{after["competition"]}:{after["fit_before_date"]}']['parameters']
            if after['training_matches'] != p['n_matches'] or after['training_max_date'] != p['training_max_date']:
                raise ValueError('Training description still differs from actual fit')
    for key in original.keys()-corrections.keys():
        if result[key] != original[key]:
            raise ValueError('Non-prediction experiment content changed')
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result['metadata_correction'], ensure_ascii=False))


if __name__ == '__main__':
    main()
