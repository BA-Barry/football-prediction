"""Compare exact historical identities before the earlier-years replication."""
from datetime import datetime, timezone
import json
from pathlib import Path

from football_model.native_experiment import load_seasons
from football_model.xg_history import COMPETITION_TEAM_NAMES

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
FD_DIRS = {
    f'{year}/{str(year+1)[-2:]}': ROOT/f'data/football-data/{year}-{str(year+1)[-2:]}/{stamp}'
    for year, stamp in [(2020, '20261024T175549025471Z'), (2021, '20261024T175549025471Z'),
                        (2022, '20261022T113804786593Z'), (2023, '20261022T113804786593Z')]
}
SLUGS = {'E0': 'EPL', 'D1': 'Bundesliga', 'SP1': 'La_Liga', 'I1': 'Serie_A', 'F1': 'Ligue_1'}


def source_path(comp, year):
    stamp = ('20261024T175519794963Z' if year < 2022 else
             '20261024T170717354684Z' if comp == 'E0' else '20261024T172015108538Z')
    return ROOT/f'data/understat/history_{stamp}/{SLUGS[comp]}_{year}.json'


def main():
    records, reports = load_seasons(FD_DIRS)
    seasons = {}
    for comp in SLUGS:
        for year in range(2020, 2024):
            season = f'{year}/{str(year+1)[-2:]}'
            local = [r for r in records if r.competition == comp and r.season == season]
            index = {(r.home_label, r.away_label): r for r in local}
            if len(index) != len(local):
                raise ValueError('Ambiguous independent home/away identity')
            path = source_path(comp, year)
            raw = json.loads(path.read_text(encoding='utf-8'))['dates']
            differences, seen_pairs = [], set()
            aliases = COMPETITION_TEAM_NAMES[comp]
            for row in raw:
                home = aliases.get(row['h']['title'], row['h']['title'])
                away = aliases.get(row['a']['title'], row['a']['title'])
                pair = home, away
                if pair in seen_pairs or row.get('isResult') is not True:
                    raise ValueError('Duplicate or uncompleted source fixture')
                seen_pairs.add(pair)
                fd = index.get(pair)
                played = datetime.strptime(row['datetime'], '%Y-%m-%d %H:%M:%S').date()
                goals = [int(row['goals'][k]) for k in ('h', 'a')]
                reasons = []
                if fd is None:
                    reasons.append('unresolved_fixture_identity')
                else:
                    if played != fd.match_date:
                        reasons.append('date_conflict')
                    if goals != list(fd.full_time):
                        reasons.append('score_conflict')
                    if fd.goal_fit_exclusion:
                        reasons.append(fd.goal_fit_exclusion)
                if reasons:
                    differences.append({'home': home, 'away': away, 'understat_id': str(row['id']),
                        'understat_home': row['h']['title'], 'understat_away': row['a']['title'],
                        'understat_datetime_text': row['datetime'], 'understat_goals': goals,
                        'football_data_match_id': fd.match_id if fd else None,
                        'football_data_date': fd.match_date.isoformat() if fd else None,
                        'football_data_goals': list(fd.full_time) if fd else None,
                        'reasons': reasons, 'understat_source': f'https://understat.com/match/{row["id"]}',
                        'independent_source': fd.source_url if fd else None})
            missing = [r.match_id for pair, r in index.items() if pair not in seen_pairs]
            seasons[f'{comp}:{season}'] = {'source_path': str(path), 'raw_n': len(raw),
                'independent_n': len(local), 'differences': differences,
                'independent_pairs_missing_from_source': missing}
    target = ROOT/'outputs'/datetime.now(timezone.utc).strftime('更早年份xG源核对_%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True, exist_ok=False)
    result = {'fd_directories': {k: str(v) for k,v in FD_DIRS.items()}, 'fd_reports': reports,
              'seasons': seasons, 'new_live_results_reviewed': False}
    (target/'逐季源差异.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'directory': str(target), 'raw_n': sum(v['raw_n'] for v in seasons.values()),
                      'difference_counts': {k: len(v['differences']) for k,v in seasons.items() if v['differences']},
                      'unresolved_fixture_count': sum('unresolved_fixture_identity' in x['reasons'] for v in seasons.values() for x in v['differences']),
                      'missing_pair_counts': {k: len(v['independent_pairs_missing_from_source']) for k,v in seasons.items()
                                             if v['independent_pairs_missing_from_source']}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
