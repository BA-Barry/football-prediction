"""Historical Understat xG joined to exact independently stored fixtures.

Only individual completed matches are read. Same-match Understat forecasts and
season aggregates are deliberately ignored, not used as prematch predictors.
"""
from dataclasses import dataclass
from datetime import date, datetime
import math
import re

import numpy as np

from .team_model import fit_weighted_poisson, team_features
from .football_data import LEAGUES, SPECIAL_MATCHES


TEAM_NAMES = {
    'Manchester City': 'Man City', 'Manchester United': 'Man United',
    'Newcastle United': 'Newcastle', 'Nottingham Forest': "Nott'm Forest",
    'Wolverhampton Wanderers': 'Wolves',
    'West Bromwich Albion': 'West Brom',
}
COMPETITION_TEAM_NAMES = {
    'E0':TEAM_NAMES,
    'D1':{'Arminia Bielefeld':'Bielefeld','Greuther Fuerth':'Greuther Furth',
          'Bayer Leverkusen':'Leverkusen','Borussia Dortmund':'Dortmund',
          'Borussia M.Gladbach':"M'gladbach",'Eintracht Frankfurt':'Ein Frankfurt',
          'FC Cologne':'FC Koln','FC Heidenheim':'Heidenheim','Hamburger SV':'Hamburg',
          'Hertha Berlin':'Hertha','Mainz 05':'Mainz','RasenBallsport Leipzig':'RB Leipzig',
          'St. Pauli':'St Pauli','VfB Stuttgart':'Stuttgart'},
    'SP1':{'Athletic Club':'Ath Bilbao','Atletico Madrid':'Ath Madrid','Celta Vigo':'Celta',
           'Espanyol':'Espanol','Rayo Vallecano':'Vallecano','Real Betis':'Betis',
           'Real Oviedo':'Oviedo','Real Sociedad':'Sociedad','Real Valladolid':'Valladolid',
           'SD Huesca':'Huesca'},
    'I1':{'AC Milan':'Milan','Parma Calcio 1913':'Parma'},
    'F1':{'Clermont Foot':'Clermont','Paris Saint Germain':'Paris SG','Saint-Etienne':'St Etienne'},
}


@dataclass
class HistoricalXgMatch:
    match_id: str
    match_date: date
    home: str
    away: str
    xg: tuple[float, float]
    goals: tuple[int, int]
    understat_id: str
    source_url: str
    competition: str = 'E0'


def _number(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'Boolean is not a measurement: {label}')
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'Invalid numeric measurement: {label}') from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f'Need a finite nonnegative measurement: {label}')
    return result


def join_understat(rows, football_data_records, season, competition='E0'):
    """Exact date/home/away join with explicit aliases and full-score equality."""
    if not re.fullmatch(r'[0-9]{4}/[0-9]{2}', season):
        raise ValueError('Need a YYYY/YY season')
    if competition not in LEAGUES:
        raise ValueError('Unsupported competition')
    aliases=COMPETITION_TEAM_NAMES[competition]
    year = int(season[:4])
    index = {}
    for record in football_data_records:
        if record.competition != competition or record.season != season:
            continue
        key = (record.match_date, record.home_label, record.away_label)
        if key in index:
            raise ValueError('Duplicate Football-Data date/team fixture')
        index[key] = record
    if not index:
        raise ValueError('Need independent league fixtures for this season')
    # Existing source-specific exceptional fixtures remain excluded from both fits.
    # A known resumed match may have its initial date in one provider; this is an
    # explicit pair exclusion, never a date relaxation for ordinary fixtures.
    special_pairs={}
    for key,record in index.items():
        special=SPECIAL_MATCHES.get((competition,*key))
        if special is not None and record.goal_fit_exclusion==special[0]:
            special_pairs[(record.home_label,record.away_label)]=(key,record,special)
    seen_ids, used, parsed, excluded = set(), set(), [], []
    for row in rows:
        if row.get('isResult') is not True:
            raise ValueError('Historical xG requires a completed source result')
        identifier = str(row['id'])
        if not identifier.isdecimal() or identifier in seen_ids:
            raise ValueError('Invalid or duplicate Understat match ID')
        seen_ids.add(identifier)
        played = datetime.strptime(row['datetime'], '%Y-%m-%d %H:%M:%S').date()
        if not date(year, 7, 1) <= played < date(year+1, 7, 1):
            raise ValueError('Match date outside declared season')
        # The provider clock has no verified timezone; only its calendar date is joined.
        home_raw, away_raw = row['h']['title'], row['a']['title']
        home, away = aliases.get(home_raw, home_raw), aliases.get(away_raw, away_raw)
        key = (played, home, away)
        goals_raw = [row['goals'][side] for side in ('h','a')]
        goals = tuple(_number(v, 'goals') for v in goals_raw)
        if any(v != math.floor(v) for v in goals):
            raise ValueError('Goals must be whole counts')
        xg = tuple(_number(row['xG'][side], 'xG') for side in ('h','a'))
        if (home,away) in special_pairs:
            independent_key,record,special=special_pairs[(home,away)]
            if independent_key in used:
                raise ValueError('Duplicate known special fixture')
            used.add(independent_key)
            excluded.append({'match_id':record.match_id,'reason':special[0], 'source_url':special[1],
                             'understat_id':identifier,'understat_date':played.isoformat(),
                             'football_data_date':record.match_date.isoformat(),
                             'understat_goals':list(map(int,goals)),
                             'football_data_goals':list(record.full_time), 'understat_xg':list(xg),
                             'identity':'known_special_home_away_pair_exclusion_not_an_ordinary_date_join'})
            continue
        if key not in index or key in used:
            raise ValueError(f'No unique exact independent fixture: {key}')
        used.add(key)
        record = index[key]
        if goals != record.full_time:
            raise ValueError('Independent full-time score disagrees')
        parsed.append(HistoricalXgMatch(record.match_id, played, record.home, record.away,
                                        xg, tuple(map(int, goals)), identifier,
                                        f'https://understat.com/match/{identifier}', competition))
    special_ids={record.match_id for _,record,_ in special_pairs.values()}
    missing = sorted(record.match_id for key, record in index.items() if key not in used and record.match_id not in special_ids)
    missing_special=sorted(record.match_id for key,record in index.items() if key not in used and record.match_id in special_ids)
    return sorted(parsed,key=lambda r:(r.match_date,r.match_id)), {
        'season':season, 'competition':competition, 'understat_rows':len(rows), 'independent_rows':len(index),
        'exact_matches':len(parsed), 'missing_independent_match_ids':missing,
        'aliases':dict(aliases), 'goal_disagreements':0,
        'excluded_special_matches':excluded, 'missing_special_match_ids':missing_special,
        'clock_basis':'provider date text only, not an asserted timezone or publication time',
        'ignored_columns':['forecast','teams','players','season aggregates'],
    }


class DateTeamXgModel:
    """Fractional-xG Poisson quasi-loss; its rate-to-goals use is experimental.

    This does not assert xG is an integer Poisson count. It estimates a mean
    from nonnegative fractional targets using the same log-link and preset.
    """
    def __init__(self, regularization=.01, half_life_days=180):
        if not all(math.isfinite(v) and v>0 for v in (regularization,half_life_days)):
            raise ValueError('Positive regularization and half-life required')
        self.regularization,self.half_life_days=regularization,half_life_days

    def fit(self, records, before_date):
        if type(before_date) is not date:
            raise ValueError('Use a calendar cutoff')
        history=sorted((r for r in records if r.match_date<before_date),
                       key=lambda r:(r.match_date,r.match_id))
        if not history or len({r.match_id for r in history})!=len(history):
            raise ValueError('Need unique nonempty earlier xG history')
        competitions={r.competition for r in history}
        if len(competitions)!=1 or not competitions.issubset(LEAGUES):
            raise ValueError('Use exactly one supported competition per independent fit')
        competition=next(iter(competitions))
        features,target,weights=[],[],[]
        for record in history:
            raw=np.asarray(record.xg,dtype=object)
            if raw.shape!=(2,) or record.home==record.away:
                raise ValueError('Need distinct teams and two xG values')
            values=[_number(v,'xG') for v in raw]
            features.extend(team_features(record.home,record.away,competition))
            target.extend(values)
            weight=math.exp(-math.log(2)*(before_date-record.match_date).days/self.half_life_days)
            weights.extend((weight,weight))
        vectorizer,model=fit_weighted_poisson(features,target,weights,self.regularization)
        coefficients=dict(zip(vectorizer.get_feature_names_out(),map(float,model.coef_)))
        intercept=float(model.intercept_)
        design=vectorizer.transform(features)
        mu=np.exp(intercept+design @ model.coef_)
        normalized=np.asarray(weights)/sum(weights)
        residual=normalized*(mu-np.asarray(target))
        gradient=np.r_[float(residual.sum()),np.asarray(design.T @ residual).ravel()+self.regularization*model.coef_]
        if not np.isfinite(gradient).all() or np.max(np.abs(gradient))>1e-6:
            raise ValueError('xG quasi-loss optimizer did not converge sufficiently')
        self.coefficients,self.intercept=coefficients,intercept
        self.teams={team for r in history for team in (r.home,r.away)}
        self.before_date=before_date
        self.competition=competition
        self.metadata={
            'model_type':'date_level_fractional_xg_quasi_poisson', 'competition':competition,
            'response':'provider_match_xg', 'loss':'Poisson deviance/quasi-loss for nonnegative fractional targets',
            'regularization':self.regularization,'half_life_days':self.half_life_days,
            'before_date':before_date.isoformat(),'training_min_date':history[0].match_date.isoformat(),
            'training_max_date':history[-1].match_date.isoformat(),'n_matches':len(history),
            'reference_match_ids':[r.match_id for r in history], 'teams':sorted(self.teams),
            'intercept':intercept,'coefficients':coefficients,'optimizer_iterations':int(model.n_iter_),
            'gradient_max_abs':float(np.max(np.abs(gradient))),
            'availability':'strictly earlier calendar dates assumed available; provider historical revisions unknown',
            'rate_use':'independent research line; treating predicted xG mean as future goal rate is evaluated, not assumed true',
        }
        return self

    def predict(self, home, away, competition, match_date):
        if not hasattr(self,'metadata'):
            raise ValueError('Fit earlier xG before predicting')
        if type(match_date) is not date or match_date<self.before_date:
            raise ValueError('Prediction must follow the calendar fit cutoff')
        if competition!=self.competition or home==away or {home,away}-self.teams:
            raise ValueError('Need the fitted competition and two known distinct teams; unknown is not zero')
        rates=tuple(math.exp(self.intercept+math.fsum(self.coefficients.get(k,0)*v for k,v in row.items()))
                    for row in team_features(home,away,competition))
        if not all(math.isfinite(v) and v>0 for v in rates):
            raise ValueError('Invalid xG-derived rate')
        return rates

    def parameters(self):
        if not hasattr(self,'metadata'):
            raise ValueError('Model not fitted')
        return dict(self.metadata)
