"""Measure source discrepancies before fitting the replicated xG model."""
from datetime import datetime,timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

from football_model.native_experiment import load_seasons,season_directories
from football_model.xg_history import COMPETITION_TEAM_NAMES
from run_shot_increment_20261025 import write

ROOT=REPOSITORY_ROOT
SOURCE=ROOT/'data/understat/history_20261024T172015108538Z'
SLUGS={'D1':'Bundesliga','SP1':'La_Liga','I1':'Serie_A','F1':'Ligue_1'}


def main():
    records,_=load_seasons(season_directories(ROOT))
    out={}
    for competition,slug in SLUGS.items():
        aliases=COMPETITION_TEAM_NAMES[competition]
        for year in (2022,2023,2024,2025):
            season=f'{year}/{str(year+1)[-2:]}'
            independent=[r for r in records if r.competition==competition and r.season==season]
            index={(r.home_label,r.away_label):r for r in independent}
            if len(index)!=len(independent):
                raise ValueError('Ambiguous source home-away pair')
            raw=json.loads((SOURCE/f'{slug}_{year}.json').read_text(encoding='utf-8'))['dates']
            differences=[]
            for row in raw:
                home=aliases.get(row['h']['title'],row['h']['title'])
                away=aliases.get(row['a']['title'],row['a']['title'])
                record=index.get((home,away))
                played=datetime.strptime(row['datetime'],'%Y-%m-%d %H:%M:%S').date()
                goals=[int(row['goals'][s]) for s in ('h','a')]
                if record is None or played!=record.match_date or goals!=list(record.full_time) or record.goal_fit_exclusion:
                    differences.append({'home':home,'away':away,'understat_id':row['id'],
                        'understat_datetime_text':row['datetime'],'understat_goals':goals,
                        'football_data_date':record.match_date.isoformat() if record else None,
                        'football_data_goals':list(record.full_time) if record else None,
                        'known_exclusion':record.goal_fit_exclusion if record else None,
                        'football_data_match_id':record.match_id if record else None,
                        'understat_source':f'https://understat.com/match/{row["id"]}',
                        'independent_source':record.source_url if record else None})
            out[f'{competition}:{season}']={'raw_n':len(raw),'independent_n':len(independent),'differences':differences}
    target=ROOT/'outputs'/datetime.now(timezone.utc).strftime('跨联赛xG源核对_%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True,exist_ok=False)
    write(target/'逐季源差异.json',out)
    print(json.dumps({'directory':str(target),'differences':{k:v['differences'] for k,v in out.items() if v['differences']},
                      'raw_n':sum(v['raw_n'] for v in out.values())},ensure_ascii=False))


if __name__=='__main__':
    main()
