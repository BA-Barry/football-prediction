"""Same fixed team goal model, refreshed observed current-season CSV results."""
from datetime import date
import csv
import json
from pathlib import Path

from football_model.date_model import DateTeamGoalModel
from football_model.football_data import _date, parse_row
from football_model.probability import score_grid, summarize
from ticket_europe_statistics_20261025 import ROOT, CUTOFF, CONFIGS, make_record, key

OUT = ROOT / 'outputs/用户六串一分析_20261025'
CURRENT = OUT / '当前FootballData'


def run():
    canonical_path = ROOT / 'outputs/动态胜平负接入_20261025/data/canonical.json'
    canonical = json.loads(canonical_path.read_text(encoding='utf-8'))['rows']
    receipt = json.loads((CURRENT/'下载来源.json').read_text(encoding='utf-8'))
    results=[]
    for competition, _, home, away, kickoff in CONFIGS:
        records=[]
        for row in canonical:
            if row['competition'] != competition:
                continue
            records.append(make_record(row['match_id'], competition, row['season'],
                date.fromisoformat(row['date']), row['home'].split(':',2)[2],
                row['away'].split(':',2)[2], row['actual_score'], canonical_path,
                'https://www.football-data.co.uk/data.php'))
        old_count=len(records)
        source_path=CURRENT/f'{competition}.csv'
        source_url=f'https://www.football-data.co.uk/mmz4281/2627/{competition}.csv'
        issues=[]; current=[]; excluded=[]; ignored_same_day_or_future=0
        with source_path.open(encoding='utf-8-sig',newline='') as stream:
            reader=csv.DictReader(stream)
            columns=reader.fieldnames
            for row in reader:
                if not (row.get('Date') or '').strip():
                    continue
                if _date(row['Date']) >= CUTOFF:
                    ignored_same_day_or_future += 1
                    continue  # ignore all such outcomes without parsing/printing them
                record, row_issues=parse_row(row,competition=competition,season='2026/27',
                    source_path=str(source_path),source_url=source_url)
                issues.extend(row_issues)
                if record.goal_fit_exclusion is not None:
                    excluded.append({'id':record.match_id,'reason':record.goal_fit_exclusion})
                current.append(record)
        records.extend(current)
        if len({r.match_id for r in records})!=len(records):
            raise ValueError('Duplicate current/historical match identities')
        model=DateTeamGoalModel(.01,180).fit(records,CUTOFF)
        rates=model.predict(key(competition,home),key(competition,away),competition,CUTOFF)
        probabilities=summarize(score_grid(*rates))
        one=probabilities['one_x_two']
        if min(one.values())<0 or abs(sum(one.values())-1)>1e-10:
            raise ValueError('Invalid probability export')
        coverage={}
        for team in (home,away):
            games=[r for r in current if key(competition,team) in (r.home,r.away)]
            coverage[team]={'current_season_included_matches':len(games),
                'last_included_date':max((r.match_date.isoformat() for r in games),default=None),
                'matches':[{'id':r.match_id,'date':r.match_date.isoformat(),
                    'home':r.home_label,'away':r.away_label,'score':list(r.full_time)} for r in games]}
        parameters=model.parameters()
        parameters['this_run_usage']='Exploratory current-season extrapolation; original not_for metadata retained.'
        results.append({'competition':competition,'home':home,'away':away,
            'kickoff_beijing':kickoff,'old_historical_matches':old_count,
            'current_season_included_matches':len(current),'training_matches':model.n_matches,
            'current_season_date_max':max((r.match_date.isoformat() for r in current),default=None),
            'target_team_coverage':coverage,'goal_rates':{'home':rates[0],'away':rates[1]},
            'probabilities':probabilities,'model':parameters,
            'source_columns':columns,'source_path':str(source_path),'source_url':source_url,
            'field_issues':issues,'current_goal_exclusions':excluded,
            'same_day_or_future_rows_ignored_before_outcome_parse':ignored_same_day_or_future})
    output={'model':'Existing fixed DateTeamGoalModel .01 / 180 + independent Poisson score grid',
        'additional_fit_calls':3,'parameter_searches':0,'target_outcomes_parsed_or_scored':False,
        'current_source_download_receipts':receipt,'fixtures':results,
        'limitations':['Current CSV source is updated through its recorded last-modified date, not a live odds snapshot.',
            'Current team injuries, European travel, and subjective motivation are absent from the goal model.',
            'This is the existing pure goals route, not the xG blended route or a newly calibrated 2026 model.',
            'Original date-level model usage caveats remain in saved model metadata.']}
    (OUT/'欧洲统计线路_资料刷新.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    for row in results:
        print(json.dumps({k:row[k] for k in ('competition','home','away','old_historical_matches',
            'current_season_included_matches','training_matches','current_season_date_max','goal_rates')}
            | {'one_x_two':row['probabilities']['one_x_two'],
               'target_coverage':{k:{'n':v['current_season_included_matches'],'latest':v['last_included_date']} for k,v in row['target_team_coverage'].items()},
               'optimizer_iterations':row['model']['optimizer_iterations']},ensure_ascii=False))


if __name__=='__main__':
    run()
