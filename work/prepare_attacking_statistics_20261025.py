"""Join unused historical shot/corner columns to existing accepted fixtures.

Reads only the existing canonical source list, not current or future seasons.
No model training, current-results query, odds integration, or new download.
"""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from football_model.attacking_statistics import parse_attacking_counts
from football_model.football_data import _date

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
CANONICAL = ROOT/'outputs/动态胜平负接入_20261025/data/canonical.json'
OUT = ROOT/'outputs/多项进攻统计接入_20261025'
REQUIRED = ('shots_on_target', 'non_target_shots', 'corners')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    canonical = read(CANONICAL)
    expected = {row['match_id']: row for row in canonical['rows']}
    found, rows, sources, excluded = set(), [], [], []
    for season, leagues in sorted(canonical['metadata']['sources'].items()):
        for competition, metadata in sorted(leagues.items()):
            path = Path(metadata['path'])
            if path.drive.casefold() != 'g:':
                raise ValueError('Read the G-drive source archive')
            counts = Counter()
            with path.open(encoding='utf-8-sig', newline='') as handle:
                reader = csv.DictReader(handle)
                columns = set(reader.fieldnames)
                for line_number, raw in enumerate(reader, start=2):
                    if not raw.get('HomeTeam') and not raw.get('AwayTeam'):
                        continue
                    counts['source_rows'] += 1
                    mid = f"football-data:{competition}:{season}:{raw['HomeTeam'].strip()}|{raw['AwayTeam'].strip()}"
                    if mid not in expected:
                        excluded.append({'match_id': mid, 'source_path': str(path), 'line_number': line_number,
                                         'reason': 'not_in_existing_accepted_canonical_history'})
                        continue
                    fixture = expected[mid]
                    if (mid in found or raw.get('Div') != competition
                            or _date(raw['Date']).isoformat() != fixture['date']):
                        raise ValueError('Statistical row differs from the existing exact fixture identity')
                    found.add(mid)
                    parsed = parse_attacking_counts(raw)
                    rows.append({**fixture, **parsed, 'source_path': str(path),
                                 'source_url': metadata['url'], 'source_line_number': line_number})
                    counts['accepted_fixtures'] += 1
                    for name, pair in parsed['counts'].items():
                        counts[name+'_pairs_available'] += pair is not None
                    counts['complete_three_statistics'] += all(parsed['counts'][name] is not None for name in REQUIRED)
                    counts['fixtures_with_statistic_issues'] += bool(parsed['issues'])
            sources.append({'season': season, 'competition': competition, 'path': str(path),
                            'columns_present': sorted(columns & {'HS', 'AS', 'HST', 'AST', 'HC', 'AC'}),
                            **counts})
    if found != set(expected):
        raise ValueError('Some accepted historical identities lack a source row')
    rows.sort(key=lambda r: (r['date'], r['match_id']))
    issues = [{key: row[key] for key in ('match_id', 'date', 'issues', 'raw_statistic_cells', 'source_path')}
              for row in rows if row['issues']]
    summary = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'canonical_source': str(CANONICAL),
        'source_files_read': len(sources), 'source_rows': sum(r['source_rows'] for r in sources),
        'accepted_historical_matches': len(rows), 'existing_exclusions_retained': len(excluded),
        'date_min': rows[0]['date'], 'date_max': rows[-1]['date'],
        'available_pairs': {name: sum(row['counts'][name] is not None for row in rows)
                            for name in ('shots', 'shots_on_target', 'non_target_shots', 'corners')},
        'complete_three_statistics': sum(all(row['counts'][name] is not None for name in REQUIRED) for row in rows),
        'fixtures_with_statistic_issues': len(issues), 'sources': sources,
        'current_results_requested': False, 'target_lineups_used': False, 'model_fits': 0,
        'first_publication_times_known': False,
        'availability_convention_for_later_research': 'only strictly earlier calendar dates, not exact T-24/72h or original publication reconstruction',
        'new_model_structure_approval': 'pending; this run prepares data only',
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT/'历史多项统计.json', {'rows': rows})
    write(OUT/'接入结果.json', summary)
    write(OUT/'字段问题与原有排除.json', {'statistic_issues': issues, 'existing_exclusions': excluded})
    print(json.dumps({k: v for k, v in summary.items() if k != 'sources'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
