"""Reuse the fixed team goal model for three requested ticket fixtures.

No target results, current starting lineups, parameter search, or model promotion.
"""
from collections import Counter
from datetime import date
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

from football_model.date_model import DateTeamGoalModel
from football_model.football_data import HistoricalMatch
from football_model.probability import score_grid, summarize

ROOT = REPOSITORY_ROOT
CUTOFF = date(2026, 10, 25)
SNAPSHOT = ROOT / 'data/openfootball/2026-27/20261023T105013041188Z'
OUTPUT = ROOT / 'outputs/用户六串一分析_20261025/欧洲统计线路.json'
# Explicit spelling bridges; current-only promoted clubs retain their own identity.
ALIASES = {
    'D1': {
        '1. FC Köln': 'FC Koln', '1. FC Union Berlin': 'Union Berlin',
        '1. FSV Mainz 05': 'Mainz', 'Bayer 04 Leverkusen': 'Leverkusen',
        'Borussia Dortmund': 'Dortmund', 'Borussia Mönchengladbach': "M'gladbach",
        'Eintracht Frankfurt': 'Ein Frankfurt', 'FC Augsburg': 'Augsburg',
        'FC Bayern München': 'Bayern Munich', 'FC Schalke 04': 'Schalke 04',
        'Hamburger SV': 'Hamburg', 'RB Leipzig': 'RB Leipzig',
        'SC Freiburg': 'Freiburg', 'SC Paderborn 07': 'Paderborn',
        'SV 07 Elversberg': 'Elversberg', 'SV Werder Bremen': 'Werder Bremen',
        'TSG 1899 Hoffenheim': 'Hoffenheim', 'VfB Stuttgart': 'Stuttgart',
    },
    'SP1': {
        'Athletic Club': 'Ath Bilbao', 'CA Osasuna': 'Osasuna',
        'Club Atlético de Madrid': 'Ath Madrid', 'Deportivo Alavés': 'Alaves',
        'Elche CF': 'Elche', 'FC Barcelona': 'Barcelona', 'Getafe CF': 'Getafe',
        'Levante UD': 'Levante', 'Málaga CF': 'Malaga', 'RC Celta de Vigo': 'Celta',
        'RC Deportivo La Coruña': 'La Coruna', 'RCD Espanyol de Barcelona': 'Espanol',
        'Rayo Vallecano de Madrid': 'Vallecano', 'Real Betis Balompié': 'Betis',
        'Real Madrid CF': 'Real Madrid', 'Real Racing Club de Santander': 'Santander',
        'Real Sociedad de Fútbol': 'Sociedad', 'Sevilla FC': 'Sevilla',
        'Valencia CF': 'Valencia', 'Villarreal CF': 'Villarreal',
    },
    'I1': {
        'AC Milan': 'Milan', 'AC Monza': 'Monza', 'ACF Fiorentina': 'Fiorentina',
        'AS Roma': 'Roma', 'Atalanta BC': 'Atalanta', 'Bologna FC 1909': 'Bologna',
        'Cagliari Calcio': 'Cagliari', 'Como 1907': 'Como',
        'FC Internazionale Milano': 'Inter', 'Frosinone Calcio': 'Frosinone',
        'Genoa CFC': 'Genoa', 'Juventus FC': 'Juventus', 'Parma Calcio 1913': 'Parma',
        'SS Lazio': 'Lazio', 'SSC Napoli': 'Napoli', 'Torino FC': 'Torino',
        'US Lecce': 'Lecce', 'US Sassuolo Calcio': 'Sassuolo',
        'Udinese Calcio': 'Udinese', 'Venezia FC': 'Venezia',
    },
}
CONFIGS = [
    ('D1', 'de.1.json', 'Freiburg', 'Bayern Munich', '2026-10-26T00:30:00+08:00'),
    ('SP1', 'es.1.json', 'Sociedad', 'Levante', '2026-10-26T01:30:00+08:00'),
    ('I1', 'it.1.json', 'Udinese', 'Milan', '2026-10-26T03:45:00+08:00'),
]


def key(competition, name):
    return f'football-data:{competition}:{name}'


def make_record(match_id, competition, season, day, home, away, score, path, url):
    if len(score) != 2 or any(type(x) is not int or x < 0 for x in score):
        raise ValueError(f'Invalid observed score {match_id}')
    return HistoricalMatch(match_id, competition, season, day, None,
        key(competition, home), key(competition, away), home, away,
        tuple(score), None, {}, str(path), url)


def run():
    canonical_path = ROOT / 'outputs/动态胜平负接入_20261025/data/canonical.json'
    canonical = json.loads(canonical_path.read_text(encoding='utf-8'))['rows']
    results = []
    for competition, filename, home, away, kickoff in CONFIGS:
        records = []
        for row in canonical:
            if row['competition'] != competition:
                continue
            records.append(make_record(row['match_id'], competition, row['season'],
                date.fromisoformat(row['date']), row['home'].split(':', 2)[2],
                row['away'].split(':', 2)[2], row['actual_score'], canonical_path,
                'https://www.football-data.co.uk/data.php'))
        old_count = len(records)
        old_teams = {t for row in records for t in (row.home, row.away)}
        source_path = SNAPSHOT / filename
        source = json.loads(source_path.read_text(encoding='utf-8'))['matches']
        missing, current = [], []
        for row in source:
            day = date.fromisoformat(row['date'])
            if day >= CUTOFF:
                continue  # never inspect target/current-day scores
            h = ALIASES[competition][row['team1']]
            a = ALIASES[competition][row['team2']]
            observed = row.get('score')
            if isinstance(observed, dict):
                observed = observed.get('ft')
            if observed is None:
                missing.append({'date': row['date'], 'home': h, 'away': a})
                continue
            record = make_record(f'openfootball:{competition}:2026/27:{h}|{a}',
                competition, '2026/27', day, h, a, observed, source_path,
                f'https://github.com/openfootball/football.json/blob/master/2026-27/{filename}')
            current.append(record)
        records.extend(current)
        if len({r.match_id for r in records}) != len(records):
            raise ValueError('Duplicate training identities')
        model = DateTeamGoalModel(regularization=.01, half_life_days=180).fit(records, CUTOFF)
        rates = model.predict(key(competition, home), key(competition, away), competition, CUTOFF)
        summary = summarize(score_grid(*rates))
        probabilities = summary['one_x_two']
        if abs(sum(probabilities.values()) - 1) > 1e-10 or min(probabilities.values()) < 0:
            raise ValueError('Invalid exported probabilities')
        coverage = {}
        for team in (home, away):
            included = [r for r in current if key(competition, team) in (r.home, r.away)]
            missing_team = [r for r in missing if team in (r['home'], r['away'])]
            coverage[team] = {'known_current_season_matches': len(included),
                'last_known_result_date': max((r.match_date.isoformat() for r in included), default=None),
                'missing_past_scheduled_matches': missing_team,
                'historical_matches': sum(key(competition, team) in (r.home,r.away) for r in records[:old_count])}
        parameters = model.parameters()
        parameters['this_run_usage'] = ('Exploratory 2026 extrapolation only; original model not_for '
            'metadata remains intact. Incomplete current-season coverage; injuries and European travel absent.')
        results.append({'competition': competition, 'home': home, 'away': away,
            'kickoff_beijing': kickoff, 'fit_before_calendar_date': CUTOFF.isoformat(),
            'old_historical_matches': old_count, 'current_season_known_matches': len(current),
            'current_season_past_schedule_matches': len(current)+len(missing),
            'current_season_missing_past_results': missing,
            'new_current_only_club_identities': sorted({t for r in current for t in (r.home,r.away)} - old_teams),
            'target_team_coverage': coverage, 'goal_rates': {'home':rates[0], 'away':rates[1]},
            'probabilities': summary, 'model': parameters,
            'sources': {'canonical':str(canonical_path), 'current_snapshot':str(source_path)},
            'explicit_current_name_aliases': ALIASES[competition]})
    output = {'model': 'existing DateTeamGoalModel fixed .01 / 180 days + independent Poisson grid',
        'fit_calls': 3, 'parameter_searches':0, 'target_results_read':False,
        'limitations':['Current source is a saved 2026-10-23 partial snapshot; missing observed results remain missing.',
            'This is an extrapolating statistical comparison, not a calibrated current forecast or the xG blended route.',
            'Injuries, rest/travel and current starting lineups are not input to this fixed goal model.',
            'Cross-source spelling bridges are explicit; recently promoted clubs lack the older top-flight history.'],
        'fixtures':results}
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    for row in results:
        print(json.dumps({k:row[k] for k in ('competition','home','away','old_historical_matches',
            'current_season_known_matches','current_season_past_schedule_matches',
            'target_team_coverage','goal_rates')} | {'1x2':row['probabilities']['one_x_two'],
                'optimizer_iterations':row['model']['optimizer_iterations']},ensure_ascii=False))


if __name__ == '__main__':
    run()
