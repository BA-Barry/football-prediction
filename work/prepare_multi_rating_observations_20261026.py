"""Prepare real, already-accepted historical observations for multi-rating reuse.

Only existing G-drive files are read. The preparation does not fetch results,
fit models or infer individual publication timestamps. Execution is in main.
"""
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from football_model.attacking_statistics import parse_attacking_counts
from football_model.football_data import LEAGUES, parse_row
from run_npxg_comparison_20261025 import ROOT, load_source


OUT = ROOT / 'outputs/多路评级联合接入_20261026'


def write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n', encoding='utf-8')


def prepare_observations():
    parsed, independent, joins, fd_reports, directories = load_source()
    source_difference_ids = {
        identifier for report in joins.values()
        for identifier in report['excluded_independent_ids']
    }
    if not set(parsed) <= set(independent):
        raise ValueError('Parsed provider matches must retain exact independent identities')
    eligible = {
        identifier: record for identifier, record in parsed.items()
        if independent[identifier].goal_fit_exclusion is None
        and identifier not in source_difference_ids
    }
    found, observations, exclusions, files = set(), [], [], []
    by_season_league = []
    for season, directory_text in sorted(directories.items()):
        directory = Path(directory_text)
        if directory.drive.casefold() != 'g:':
            raise ValueError('Use only the existing G-drive historical archive')
        for competition in LEAGUES:
            path = directory / f'{competition}.csv'
            if not path.exists():
                raise ValueError(f'Existing historical source missing: {path}')
            counts = Counter()
            dates = []
            with path.open(encoding='utf-8-sig', newline='') as handle:
                reader = csv.DictReader(handle)
                fields = set(reader.fieldnames or [])
                for line, raw in enumerate(reader, start=2):
                    if not any(value and str(value).strip() for value in raw.values()):
                        continue
                    counts['source_rows'] += 1
                    home = (raw.get('HomeTeam') or '').strip()
                    away = (raw.get('AwayTeam') or '').strip()
                    identifier = f'football-data:{competition}:{season}:{home}|{away}'
                    independent_record = independent.get(identifier)
                    if independent_record is None:
                        raise ValueError(f'Existing source row lacks independent record: {identifier}')
                    checked, _ = parse_row(raw, competition=competition, season=season,
                        source_path=str(path), source_url=independent_record.source_url)
                    if (checked.match_id != identifier
                            or checked.match_date != independent_record.match_date
                            or checked.full_time != independent_record.full_time
                            or checked.home != independent_record.home
                            or checked.away != independent_record.away
                            or checked.goal_fit_exclusion != independent_record.goal_fit_exclusion):
                        raise ValueError(f'Raw CSV and accepted independent record differ: {identifier}')
                    reason = None
                    if independent_record.goal_fit_exclusion is not None:
                        reason = 'existing_goal_fit_exclusion'
                    elif identifier in source_difference_ids:
                        reason = 'existing_cross_source_discrepancy_exclusion'
                    elif identifier not in eligible:
                        reason = 'not_in_existing_accepted_provider_join'
                    if reason is not None:
                        counts[reason] += 1
                        exclusions.append({
                            'match_id': identifier, 'date': checked.match_date.isoformat(),
                            'league': competition, 'season': season, 'reason': reason,
                            'original_goal_fit_exclusion': independent_record.goal_fit_exclusion,
                            'also_recorded_source_difference': identifier in source_difference_ids,
                            'source_path': str(path), 'source_line_number': line,
                        })
                        continue
                    joined = eligible[identifier]
                    if (identifier in found or joined.match_date != checked.match_date
                            or joined.home != checked.home or joined.away != checked.away
                            or tuple(joined.goals) != checked.full_time
                            or joined.competition != competition):
                        raise ValueError(f'Rating row differs from exact accepted join: {identifier}')
                    found.add(identifier)
                    statistics = parse_attacking_counts(raw)
                    shots = statistics['counts']['shots']
                    on_target = statistics['counts']['shots_on_target']
                    observations.append({
                        'match_id': identifier, 'date': checked.match_date.isoformat(),
                        'league': competition, 'home_team': checked.home, 'away_team': checked.away,
                        'home_goals': checked.full_time[0], 'away_goals': checked.full_time[1],
                        'home_shots': None if shots is None else shots[0],
                        'away_shots': None if shots is None else shots[1],
                        'home_shots_on_target': None if on_target is None else on_target[0],
                        'away_shots_on_target': None if on_target is None else on_target[1],
                        'season': season, 'source_path': str(path),
                        'source_url': checked.source_url, 'source_line_number': line,
                        'statistic_issues': statistics['issues'],
                        'raw_statistic_cells': statistics['raw_statistic_cells'],
                    })
                    counts['accepted_observations'] += 1
                    counts['shots_pairs_available'] += shots is not None
                    counts['shots_on_target_pairs_available'] += on_target is not None
                    counts['statistic_issue_rows'] += bool(statistics['issues'])
                    dates.append(checked.match_date.isoformat())
            source = {
                'season': season, 'league': competition, 'source_path': str(path),
                'columns_present': sorted(fields & {'HS', 'AS', 'HST', 'AST', 'HC', 'AC'}),
                'date_min': min(dates) if dates else None, 'date_max': max(dates) if dates else None,
                **dict(counts),
            }
            by_season_league.append(source)
            files.append(str(path))
    if found != set(eligible):
        raise ValueError(f'Accepted history not fully found in existing CSVs: {sorted(set(eligible)-found)}')
    observations.sort(key=lambda row: (row['date'], row['match_id']))
    coverage = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'purpose': 'accepted six-season real observations for opponent-adjusted multi-rating inputs',
        'source_loader': 'work/run_npxg_comparison_20261025.py::load_source',
        'independent_records': len(independent), 'accepted_provider_join_records': len(parsed),
        'accepted_goal_fit_rating_observations': len(observations),
        'date_min': observations[0]['date'], 'date_max': observations[-1]['date'],
        'source_files_read': len(files), 'directories': directories,
        'shots_pairs_available': sum(row['home_shots'] is not None for row in observations),
        'shots_on_target_pairs_available': sum(row['home_shots_on_target'] is not None for row in observations),
        'statistic_issue_rows': sum(bool(row['statistic_issues']) for row in observations),
        'by_season_league': by_season_league,
        'excluded_reason_counts': dict(Counter(row['reason'] for row in exclusions)),
        'excluded_rows': exclusions,
        'recorded_cross_source_exclusions': {
            key: {'independent_ids': report['excluded_independent_ids'],
                  'provider_ids': report['excluded_source_ids']}
            for key, report in joins.items()
        },
        'independent_load_reports': fd_reports,
        'identity_policy': 'exact season/league/home/away ID, independent date and score; no fuzzy names',
        'statistics_policy': 'missing/invalid pairs remain None; source issues and original cells retained; no target-data imputation',
        'availability_assumption': 'Only observation calendar dates strictly before each forecast date may enter rating states. Exact initial publication/revision timestamps are unknown.',
        'new_downloads': 0, 'new_model_fits': 0, 'current_results_requested': False,
        'current_forecasts_changed': False,
    }
    return observations, coverage


def main():
    observations, coverage = prepare_observations()
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / '历史评级观测.json', {'observations': observations})
    write(OUT / '观测来源.json', coverage)
    print(json.dumps({key: coverage[key] for key in (
        'accepted_goal_fit_rating_observations', 'date_min', 'date_max', 'source_files_read',
        'shots_pairs_available', 'shots_on_target_pairs_available', 'statistic_issue_rows',
        'excluded_reason_counts', 'new_downloads', 'new_model_fits')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
