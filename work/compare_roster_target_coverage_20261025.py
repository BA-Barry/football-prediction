"""Compare the full new historical cohort to already saved team predictions."""
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
ARCHIVE = ROOT/'data/understat/roster_seasons_EPL_2023_2024_20261025'
PREDICTIONS = ROOT/'outputs/进球xG共同训练记录_20261024T174247963857Z/共同训练记录融合结果.json'
OUT = ROOT/'outputs/英超跨季首发覆盖_20261025'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    selection = read(ARCHIVE/'比赛选择.json')['selected']
    targets = [r for r in selection if r['season'] == '2024/25']
    saved = read(PREDICTIONS)
    arms = {}
    for key in ('common_goal_predictions', 'common_blend_predictions'):
        subset = [r for r in saved[key] if r['competition'] == 'E0' and r['period'] == '2024-25']
        arms[key] = {r['match_id']: r for r in subset}
        if len(arms[key]) != len(subset):
            raise ValueError('Duplicate saved target predictions')
    goal, blend = arms.values()
    if set(goal) != set(blend):
        raise ValueError('Saved team and blend coverage differ')
    rows = []
    for target in targets:
        played = date.fromisoformat(target['date'])
        cutoff = date(2024, 8, 1) + timedelta(days=28*((played-date(2024, 8, 1)).days//28))
        parameter_key = f'E0:{cutoff.isoformat()}'
        params = saved['common_goal_parameters'][parameter_key]['parameters']
        unknown = sorted({target['home'], target['away']} - set(params['teams']))
        covered = target['match_id'] in goal
        if covered and goal[target['match_id']]['fit_before_date'] != cutoff.isoformat():
            raise ValueError('Reconstructed date differs from saved prediction cutoff')
        rows.append({'match_id': target['match_id'], 'date': target['date'],
            'fit_before_date': cutoff.isoformat(), 'saved_parameter_key': parameter_key,
            'saved_team_prediction_available': covered, 'saved_blend_prediction_available': target['match_id'] in blend,
            'unseen_teams_in_saved_fit': unknown,
            'coverage_explanation': 'saved_prediction_present' if covered else
                'unseen_team_in_saved_fit' if unknown else 'unresolved_missing_prediction'})
    result = {'checked_at': datetime.now(timezone.utc).isoformat(), 'source_archive': str(ARCHIVE),
        'saved_prediction_source': str(PREDICTIONS), 'target_season': '2024/25',
        'full_target_matches': len(targets), 'saved_team_predictions': len(goal),
        'saved_blend_predictions': len(blend),
        'missing_saved_predictions': [r for r in rows if not r['saved_team_prediction_available']],
        'unresolved_missing': sum(r['coverage_explanation'] == 'unresolved_missing_prediction' for r in rows),
        'per_match': rows, 'new_model_fits': 0, 'prediction_numbers_changed': False,
        'current_results_reviewed': False,
        'limits': ['Player roster coverage is distinct from whether the team model has a forecast.',
                   'No probabilities are fabricated for the two unseen-team cases.',
                   'Historical lineup-known conditions do not establish T-24h availability.',
                   '2024/25 has already been viewed in past research; this is not a new blind test.']}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'现有球队预测覆盖.json').open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('full_target_matches','saved_team_predictions',
        'saved_blend_predictions','missing_saved_predictions','unresolved_missing')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
