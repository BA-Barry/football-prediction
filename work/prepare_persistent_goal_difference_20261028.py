"""Export trusted saved full-time goals and the unchanged 3-day target cohort.

No download, fitting, scoring, odds, xG predictor, or current-results query.
The existing source loader retains all its recorded exceptional exclusions.
"""
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import gzip
import json
import math
from pathlib import Path

from run_npxg_comparison_20261025 import load_source


ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OUT = ROOT / 'outputs/持续状态净胜球研究_20261028'
TARGET_SOURCE = ROOT / 'outputs/对手校正攻防模式整合_20261027/逐场概率与分量.json'
SEASONS = ('2022/23', '2023/24', '2024/25', '2025/26')
COMPETITIONS = ('E0', 'D1', 'SP1', 'I1', 'F1')


def _score(values):
    if (not isinstance(values, (tuple, list)) or len(values) != 2
            or any(type(v) is not int or v < 0 for v in values)):
        raise ValueError('Full-time goals must be two nonnegative integers')
    return tuple(values)


def main():
    parsed, independent, reports, _, directories = load_source()
    original = json.loads(TARGET_SOURCE.read_text(encoding='utf-8'))
    if not isinstance(original, dict) or not isinstance(original.get('records'), list):
        raise ValueError('Expected saved target object with records array')
    matches = []
    for match_id, source in parsed.items():
        record = independent.get(match_id)
        if record is None:
            raise ValueError('Trusted source match lacks its independent identity')
        if record.competition not in COMPETITIONS or record.season not in SEASONS:
            continue
        if (source.match_id != match_id or record.match_id != match_id
                or source.competition != record.competition or source.match_date != record.match_date
                or source.home != record.home or source.away != record.away
                or _score(source.goals) != _score(record.full_time)
                or record.goal_fit_exclusion):
            raise ValueError(f'Unexpected trusted history identity or goal conflict: {match_id}')
        hg, ag = _score(record.full_time)
        matches.append({'match_id': match_id, 'date': record.match_date.isoformat(),
                        'season': record.season, 'competition': record.competition,
                        'home': record.home, 'away': record.away,
                        'home_label': record.home_label, 'away_label': record.away_label,
                        'hg': hg, 'ag': ag, 'understat_id': source.understat_id})
    matches.sort(key=lambda r: (r['date'], r['match_id']))
    ids = {r['match_id'] for r in matches}
    if len(ids) != len(matches):
        raise ValueError('Duplicate trusted historical match IDs')
    fixtures = [(r['competition'], r['season'], r['date'], r['home'], r['away']) for r in matches]
    if len(set(fixtures)) != len(fixtures):
        raise ValueError('Duplicate historical date/home/away identity')
    targets, unknowns, seen_targets = [], [], set()
    for index, row in enumerate(original['records']):
        missing = []
        identifier = row.get('match_id')
        if not isinstance(identifier, str) or not identifier:
            missing.append('match_id_missing')
        if identifier in seen_targets:
            missing.append('duplicate_target_id')
        seen_targets.add(identifier)
        record = independent.get(identifier)
        if record is None:
            missing.append('independent_match_absent')
        if identifier not in ids:
            missing.append('trusted_history_match_absent')
        probabilities = row.get('tracks', {}).get('original', {}).get('probabilities')
        if (not isinstance(probabilities, list) or len(probabilities) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in probabilities)
                or abs(sum(probabilities)-1) > 1e-10):
            missing.append('original_hda_invalid_or_missing')
        feature_date, forecast_date = row.get('feature_before_date'), row.get('forecast_before_date')
        original_fit = row.get('fit_before_date')
        if feature_date != forecast_date:
            missing.append('feature_forecast_dates_differ')
        try:
            played = date.fromisoformat(row['date'])
            available = date.fromisoformat(forecast_date)
            fitted = date.fromisoformat(original_fit)
            if available != played - timedelta(days=3) or row.get('horizon_days') != 3:
                missing.append('not_original_three_calendar_day_horizon')
            if not fitted <= available < played:
                missing.append('original_fit_feature_match_date_order')
        except (KeyError, TypeError, ValueError):
            missing.append('missing_or_invalid_target_dates')
        if record is not None:
            if (record.match_date.isoformat() != row.get('date') or record.home != row.get('home')
                    or record.away != row.get('away') or record.competition != row.get('competition')
                    or list(record.full_time) != row.get('actual_score')):
                missing.append('original_target_independent_identity_or_score_conflict')
            if record.season.replace('/', '-') != row.get('period'):
                # Saved period is YYYY-YY, whereas the independent season is YYYY/YY.
                missing.append('target_period_season_conflict')
        target = {'source_row_index': index, 'match_id': identifier, 'date': row.get('date'),
                  'F': forecast_date, 'forecast_before_date': forecast_date,
                  'feature_before_date': feature_date, 'competition': row.get('competition'),
                  'season': record.season if record is not None else None, 'period': row.get('period'),
                  'home': row.get('home'), 'away': row.get('away'),
                  'hg': record.full_time[0] if record is not None else None,
                  'ag': record.full_time[1] if record is not None else None,
                  'original_hda': probabilities, 'original_fit_before_date': original_fit,
                  'horizon_days': row.get('horizon_days'), 'unknown_fields': sorted(set(missing))}
        targets.append(target)
        if missing:
            unknowns.append({'source_row_index': index, 'match_id': identifier,
                             'unknown_fields': target['unknown_fields']})
    selected_reports = {key: value for key, value in reports.items()
                        if value['competition'] in COMPETITIONS and value['season'] in SEASONS}
    by_season_competition = []
    for season in SEASONS:
        for competition in COMPETITIONS:
            group = [r for r in matches if r['season'] == season and r['competition'] == competition]
            comparison = [r for r in independent.values()
                          if r.season == season and r.competition == competition]
            by_season_competition.append({'season': season, 'competition': competition,
                'independent_rows': len(comparison), 'trusted_rows': len(group),
                'excluded_independent_ids': sorted(r.match_id for r in comparison if r.match_id not in ids)})
    summary = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
               'purpose': 'Existing trusted four-season full-goal history and unchanged 3452 target probability records.',
               'source_loader': str(ROOT / 'work/run_npxg_comparison_20261025.py'),
               'source_target_file': str(TARGET_SOURCE), 'football_data_directories': directories,
               'seasons': list(SEASONS), 'competitions': list(COMPETITIONS),
               'matches': len(matches), 'unique_match_ids': len(ids),
               'history_min_date': min(r['date'] for r in matches),
               'history_max_date': max(r['date'] for r in matches),
               'targets': len(targets), 'expected_target_count': 3452,
               'target_count_agrees': len(targets) == 3452,
               'target_unique_ids': len(seen_targets), 'target_rows_omitted': 0,
               'target_unknown_rows': unknowns,
               'target_by_period': dict(Counter(r['period'] for r in targets)),
               'target_by_competition': dict(Counter(r['competition'] for r in targets)),
               'by_season_competition': by_season_competition,
               'source_join_reports': selected_reports,
               'probability_order': ['home', 'draw', 'away'],
               'original_hda_rule': 'Copied exactly from tracks.original.probabilities, without calibration, blending or renormalization.',
               'history_rule': 'Only IDs in the existing parsed trusted source join; no independent-only additions or relaxed exclusions.',
               'date_rule': 'Calendar-date retrospective observations; fitting must select match date < P and predictors date < F, P <= F < D.',
               'limitations': ['No exact historical publication, revision or timezone reconstruction.',
                              'These seasons have already been viewed during development; not a fresh blind test.',
                              'All history goals are labels, not target-time features; future rows must remain excluded by downstream dated fitting.',
                              'No xG, shot-state, lineup, odds or LLM values are exported as new model inputs.'],
               'model_fits': 0, 'network_requests': 0, 'current_results_queries': 0}
    payload = {'metadata': summary, 'matches': matches, 'targets': targets}
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT / '输入资料.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
    (OUT / '输入资料说明.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({key: summary[key] for key in (
        'matches', 'unique_match_ids', 'history_min_date', 'history_max_date', 'targets',
        'target_unique_ids', 'target_count_agrees', 'target_rows_omitted', 'target_unknown_rows',
        'target_by_period', 'target_by_competition')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
