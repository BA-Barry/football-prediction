"""Equal evaluation exclusions; explicitly does not repair contaminated fits."""
from datetime import datetime, timezone
import json
from pathlib import Path

from run_broad_early_models_20261025 import compare, failure_diagnostics
from run_shot_increment_20261025 import write

OUT = Path('G:/Codex/football-prediction-2026-10-22/outputs/多信息多算法前推_20261025T060704977450Z')
EXCLUSIONS = {
    'football-data:F1:2024/25:Montpellier|St Etienne':
        'https://ligue1.com/fr/articles/l1_article_2269-le-point-sur-la-lutte-pour-le-maintien-020425-l1',
    'football-data:F1:2025/26:Nantes|Toulouse':
        'https://ligue1.com/fr/articles/l1_article_5189-discipline-toutes-les-decisions-l1-270526',
}


def main():
    result = json.loads((OUT/'多方案结果汇总.json').read_text(encoding='utf-8'))
    predictions = json.loads((OUT/'逐场预测.json').read_text(encoding='utf-8'))
    selected = result['selected']  # Recorded earlier choice; never replaced from later outcomes.
    validation = {arm: [r for r in rows if r['period'] == '2024-25' and r['match_id'] not in EXCLUSIONS]
                  for arm, rows in predictions.items() if arm != 'old_blend'}
    testing = {arm: [r for r in rows if r['period'] == '2025-26' and r['match_id'] not in EXCLUSIONS]
               for arm, rows in predictions.items()}
    result['validation'] = compare(validation)
    result['test'] = compare(testing, selected=selected, intervals=True)
    result['by_league'] = {comp: compare({a: [r for r in rows if r['competition'] == comp]
                                        for a, rows in testing.items()}, selected=selected)
                           for comp in ('E0', 'D1', 'SP1', 'I1', 'F1')}
    result['failure_diagnostics'] = failure_diagnostics(testing, selected)
    result['evaluation_correction'] = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'exclusions': EXCLUSIONS,
        'original_summary_preserved': str(OUT/'多方案结果汇总.json'),
        'original_validation_n': 1734, 'original_test_n': 1733,
        'evaluation_validation_n': 1733, 'evaluation_test_n': 1732,
        'selected_from_original_earlier_validation': selected,
        'new_fit_count': 0, 'prior_selection_replaced': False,
        'training_not_repaired': True,
        'training_issue': 'Montpellier incomplete result occurs in65/110 broadfits and13each oldF1goal/xGfits plus12later validation feature histories; a clean rebuild of experts/features/learners is still required',
        'interpretation': 'exploratory same-prediction score sensitivity only; not a complete ninety-minute-clean validation',
    }
    result['adopted'] = False
    write(OUT/'多方案结果汇总_评分剔除未完赛.json', result)
    print(json.dumps({'validation_n': result['validation']['n'], 'test_n': result['test']['n'],
                     'retained_prior_choice': selected,
                     'later_primary_difference': result['test']['mean_loss_differences'][selected+'_minus_old_blend:exact_score'],
                     'training_still_requires_rebuild': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
