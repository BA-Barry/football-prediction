"""Compare fitted results with later received official FPL fixtures; no refit."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

from football_model.current_projection import ALIASES, load_current_context

ROOT = REPOSITORY_ROOT
CONTEXT = ROOT / 'data/prospective_context/20261023T175004Z_completed'
SOURCE = ROOT / 'data/prematch-intelligence/20261024_batch'


def main():
    current, _, _ = load_current_context(CONTEXT, datetime.now(timezone.utc))
    teams = json.loads((SOURCE / 'fpl_raw_source_metadata.json').read_text(encoding='utf-8'))['teams']
    explicit_labels = {
        'Arsenal': 'Arsenal', 'Aston Villa': 'Aston Villa', 'Bournemouth': 'Bournemouth',
        'Brentford': 'Brentford', 'Brighton': 'Brighton', 'Chelsea': 'Chelsea',
        'Coventry City': 'Coventry', 'Crystal Palace': 'Crystal Palace', 'Everton': 'Everton',
        'Fulham': 'Fulham', 'Hull City': 'Hull', 'Ipswich Town': 'Ipswich',
        'Leeds': 'Leeds', 'Liverpool': 'Liverpool', 'Man City': 'Man City', 'Man Utd': 'Man United',
        'Newcastle': 'Newcastle', "Nott'm Forest": "Nott'm Forest", 'Sunderland': 'Sunderland',
        'Spurs': 'Tottenham',
    }
    team_map = {row['id']: explicit_labels[row['name']] for row in teams}
    fixtures = json.loads((SOURCE / 'fpl_fixtures_raw.json').read_text(encoding='utf-8'))
    reported = {}
    for row in fixtures:
        if row['finished']:
            key = (row['kickoff_time'][:10], team_map[row['team_h']], team_map[row['team_a']])
            if key in reported:
                raise ValueError('Duplicate official fixture')
            reported[key] = (row['team_h_score'], row['team_a_score'])
    fitted = {(row.match_date.isoformat(), row.home_label, row.away_label): row.full_time for row in current}
    differences = [{'key': key, 'fitted': fitted.get(key), 'official': reported.get(key)}
                   for key in sorted(set(fitted) | set(reported)) if fitted.get(key) != reported.get(key)]
    counts = Counter(t for row in current for t in (row.home_label, row.away_label))
    result = {'checked_at': datetime.now(timezone.utc).isoformat(), 'fitted_current_results': len(fitted),
              'later_official_finished_results': len(reported), 'differences': differences,
              'team_match_counts': dict(counts), 'new_fits': 0,
              'time_note': 'Official API received after fit; corroboration only, never relabelled as an earlier fit input.'}
    target = SOURCE / 'completed_results_official_crosscheck.json'
    with target.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))
    if differences or len(fitted) != 70 or set(counts.values()) != {7}:
        raise ValueError('Result coverage or score disagreement needs explicit resolution')


if __name__ == '__main__':
    main()
