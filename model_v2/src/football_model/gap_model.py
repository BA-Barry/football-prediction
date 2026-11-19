"""Four-state Generalised Attacking Performance ratings.

Local adaptation of Wheatcroft (2021), DOI 10.3233/JSA-200462.  Each statistic
has one shared learning rate and two role-transfer fractions.  Team states are
league-local, and all eight updates in a match use the same pre-match snapshot.
This module produces expected statistics, not scores or 1X2 probabilities.
"""
from copy import deepcopy
from datetime import date
import math
from collections.abc import Mapping

import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import expit


STATISTICS = ('goals', 'shots_on_target', 'non_target_shots', 'corners')
STATE_ROLES = ('home_attack', 'home_defence', 'away_attack', 'away_defence')


def _identity(row):
    fields = ('competition', 'season', 'home', 'away', 'date')
    if any(not isinstance(row.get(k), str) or not row[k] for k in fields):
        raise ValueError('GAP needs explicit league, season, teams and calendar date')
    if row['home'] == row['away']:
        raise ValueError('GAP teams must differ')
    try:
        played = date.fromisoformat(row['date'])
    except (TypeError, ValueError) as error:
        raise ValueError('GAP dates must use ISO calendar dates') from error
    if played.isoformat() != row['date']:
        raise ValueError('GAP dates must use ISO calendar dates')
    season = row['season']
    if (len(season) != 7 or season[4] != '/' or not season[:4].isdigit()
            or not season[5:].isdigit()
            or int(season[5:]) != (int(season[:4])+1) % 100):
        raise ValueError('GAP season must be YYYY/YY')
    start = int(season[:4])
    if not date(start, 7, 1) <= played < date(start+1, 7, 1):
        raise ValueError('GAP date is outside its season')
    return row['competition'], season, row['home'], row['away'], played


def _counts(row):
    values = []
    counts = row.get('counts')
    for statistic in STATISTICS:
        pair = (row.get('actual_score') if statistic == 'goals'
                else counts.get(statistic) if isinstance(counts, Mapping) else None)
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(type(x) is not int or x < 0 for x in pair)):
            raise ValueError(f'Observed {statistic} needs two nonnegative integer counts; no imputation')
        values.append(pair)
    return np.asarray(values, dtype=float)


def _parameters(parameters):
    if not isinstance(parameters, Mapping) or set(parameters) != set(STATISTICS):
        raise ValueError('One parameter triplet is required per GAP statistic')
    result = {}
    for statistic in STATISTICS:
        triplet = parameters[statistic]
        if not isinstance(triplet, Mapping) or set(triplet) != {'lambda', 'phi_home', 'phi_away'}:
            raise ValueError('GAP parameter names are lambda, phi_home and phi_away')
        if any(isinstance(v, bool) or not isinstance(v, (int, float, np.number))
               or not math.isfinite(float(v)) for v in triplet.values()):
            raise ValueError('GAP parameters must be finite real numbers')
        rate, home, away = (float(triplet[k]) for k in ('lambda', 'phi_home', 'phi_away'))
        if rate <= 0 or not 0 < home < 1 or not 0 < away < 1:
            raise ValueError('GAP needs lambda>0 and transfer fractions strictly between 0 and 1')
        result[statistic] = {'lambda': rate, 'phi_home': home, 'phi_away': away}
    return result


class GAPState:
    """Mutable chronological GAP state, with explicit schedule-roster priors.

    ``season_rosters`` is {competition: {season: iterable(team IDs)}}.  It must
    represent known season participants, not future performance.  Without that
    information a new-season team borrows the previous league-state mean.
    ``predict`` can initialize a new season/team, but never reads match counts.
    The caller is responsible for advancing only observations dated before its
    forecast cutoff; ``last_update_date`` exposes the latest observed date.
    """

    def __init__(self, params_by_stat, season_rosters=None):
        self.params_by_stat = _parameters(params_by_stat)
        self.season_rosters = {}
        for competition, seasons in (season_rosters or {}).items():
            if not isinstance(competition, str) or not isinstance(seasons, Mapping):
                raise ValueError('Schedule rosters need explicit competition and season keys')
            self.season_rosters[competition] = {}
            for season, teams in seasons.items():
                roster = set(teams)
                if not roster or any(not isinstance(t, str) or not t for t in roster):
                    raise ValueError('A known season roster must contain explicit team IDs')
                self.season_rosters[competition][season] = roster
        self.states = {}
        self.seasons = {}
        self.team_observations = {}
        self.initializations = {}
        self.last_update_date = {}
        self._season_seeds = {}

    def clone(self):
        return deepcopy(self)

    def _season(self, competition, season):
        previous = self.seasons.get(competition)
        if previous == season:
            return
        if previous is not None and int(season[:4]) <= int(previous[:4]):
            raise ValueError('GAP seasons cannot move backwards')
        old = self.states.get(competition, {})
        old_counts = self.team_observations.get(competition, {})
        roster = self.season_rosters.get(competition, {}).get(season)
        departed = sorted(set(old)-roster) if roster is not None else []
        if departed:
            seed = np.mean([old[t] for t in departed], axis=0)
            method = 'previous_season_departed_group_mean'
            donors = departed
        elif old:
            donors = sorted(old)
            seed = np.mean([old[t] for t in donors], axis=0)
            method = 'previous_season_league_state_mean'
        else:
            donors = []
            seed = np.zeros((len(STATISTICS), len(STATE_ROLES)), dtype=float)
            method = 'initial_zero_state'
        self._season_seeds[competition] = (seed.copy(), {
            'method': method, 'source_season': previous, 'donor_team_ids': donors,
            'roster_membership_assumed_known': roster is not None,
            'interpretation': 'heuristic point initialization, not learned cross-tier ability or uncertainty'})
        # Pre-create known participants so no later result changes the season's
        # common prior. Retained teams carry state; new teams copy the seed.
        new, counts, initializations = {}, {}, {}
        for team in sorted(roster or ()):
            if team in old:
                new[team] = old[team].copy()
                counts[team] = old_counts[team]
                initializations[team] = {'method':'carried_from_previous_season',
                    'source_season':previous, 'donor_team_ids':[team],
                    'roster_membership_assumed_known':True}
            else:
                new[team] = seed.copy()
                counts[team] = 0
                initializations[team] = deepcopy(self._season_seeds[competition][1])
        # With no roster we retain earlier teams; the cached mean is used for
        # an unseen participant rather than being recomputed after new games.
        if roster is None:
            for team in old:
                new[team] = old[team].copy()
                counts[team] = old_counts[team]
                initializations[team] = {'method':'carried_without_known_season_roster',
                    'source_season':previous, 'donor_team_ids':[team],
                    'roster_membership_assumed_known':False}
        self.states[competition], self.team_observations[competition] = new, counts
        self.initializations[competition], self.seasons[competition] = initializations, season

    def _teams(self, row):
        competition, season, home, away, played = _identity(row)
        self._season(competition, season)
        roster = self.season_rosters.get(competition, {}).get(season)
        for team in (home, away):
            if roster is not None and team not in roster:
                raise ValueError('Fixture team is absent from its explicit season roster')
            if team not in self.states[competition]:
                seed, metadata = self._season_seeds[competition]
                self.states[competition][team] = seed.copy()
                self.team_observations[competition][team] = 0
                self.initializations[competition][team] = deepcopy(metadata)
        return competition, season, home, away, played

    def predict(self, row):
        competition, _, home, away, _ = self._teams(row)
        h, a = self.states[competition][home], self.states[competition][away]
        expected = np.column_stack(((h[:,0]+a[:,3])/2, (a[:,2]+h[:,1])/2))
        return {
            'expected_statistics': {s: expected[i].tolist() for i,s in enumerate(STATISTICS)},
            'differences': {s: float(expected[i,0]-expected[i,1]) for i,s in enumerate(STATISTICS)},
            'team_observations': {side:self.team_observations[competition][team]
                                  for side,team in (('home',home),('away',away))},
            'initialization': {side:deepcopy(self.initializations[competition][team])
                               for side,team in (('home',home),('away',away))},
            'last_update_date': (self.last_update_date[competition].isoformat()
                                 if competition in self.last_update_date else None),
        }

    def update(self, row):
        # Check the observation before making even an initialization mutation.
        observed = _counts(row)
        competition, _, home, away, played = _identity(row)
        if competition in self.last_update_date and played < self.last_update_date[competition]:
            raise ValueError('GAP observations must be updated in calendar-date order')
        prediction = self.predict(row)
        old_home = self.states[competition][home].copy()
        old_away = self.states[competition][away].copy()
        expected = np.asarray([prediction['expected_statistics'][s] for s in STATISTICS])
        residual = observed-expected
        new_home, new_away = old_home.copy(), old_away.copy()
        for i, statistic in enumerate(STATISTICS):
            p = self.params_by_stat[statistic]
            rate, ph, pa = p['lambda'], p['phi_home'], p['phi_away']
            eh, ea = residual[i]
            new_home[i] += rate*np.array([ph*eh, ph*ea, (1-ph)*eh, (1-ph)*ea])
            new_away[i] += rate*np.array([(1-pa)*ea, (1-pa)*eh, pa*ea, pa*eh])
        new_home, new_away = np.maximum(new_home, 0), np.maximum(new_away, 0)
        if not np.isfinite(new_home).all() or not np.isfinite(new_away).all():
            raise ValueError('Nonfinite GAP state update')
        self.states[competition][home], self.states[competition][away] = new_home, new_away
        self.team_observations[competition][home] += 1
        self.team_observations[competition][away] += 1
        self.last_update_date[competition] = played
        return prediction


@njit(cache=False)
def _sequential_mae(home, away, observed, n_teams, rate, phi_home, phi_away):
    """One-statistic objective, identical old-state residuals to GAPState."""
    states = np.zeros((n_teams, 4), dtype=np.float64)
    absolute_error = 0.0
    for i in range(len(home)):
        h, a = home[i], away[i]
        expected_home = (states[h,0]+states[a,3])/2
        expected_away = (states[a,2]+states[h,1])/2
        eh, ea = observed[i,0]-expected_home, observed[i,1]-expected_away
        absolute_error += abs(eh)+abs(ea)
        states[h,0] = max(0.0, states[h,0]+rate*phi_home*eh)
        states[h,1] = max(0.0, states[h,1]+rate*phi_home*ea)
        states[h,2] = max(0.0, states[h,2]+rate*(1-phi_home)*eh)
        states[h,3] = max(0.0, states[h,3]+rate*(1-phi_home)*ea)
        states[a,0] = max(0.0, states[a,0]+rate*(1-phi_away)*ea)
        states[a,1] = max(0.0, states[a,1]+rate*(1-phi_away)*eh)
        states[a,2] = max(0.0, states[a,2]+rate*phi_away*ea)
        states[a,3] = max(0.0, states[a,3]+rate*phi_away*eh)
        if not math.isfinite(absolute_error):
            return np.inf
    return absolute_error/(2*len(home))


def optimize_gap_updates(rows):
    """Four single-start MAE fits on 2022/23, with no outcomes-based selection.

    Initial transformed parameters correspond to lambda=.3 and phi=.6/.6.
    SciPy Nelder-Mead defaults provide numerical stopping, not a parameter grid
    or a promise of global optimality. Non-convergence is returned explicitly;
    the caller must decide whether to stop, rather than claiming fit success.
    """
    selected = sorted(list(rows), key=lambda r:(r['date'],r['match_id']))
    if not selected or any(r.get('season') != '2022/23' for r in selected):
        raise ValueError('GAP updating parameters learn only from nonempty 2022/23 observations')
    identities = [_identity(r) for r in selected]
    if len({r['match_id'] for r in selected}) != len(selected):
        raise ValueError('Duplicate fixture in GAP parameter learning')
    observations = np.asarray([_counts(r) for r in selected])
    teams = sorted({(competition,team) for competition,_,h,a,_ in identities for team in (h,a)})
    index = {team:i for i,team in enumerate(teams)}
    home = np.asarray([index[(c,h)] for c,_,h,_,_ in identities],dtype=np.int64)
    away = np.asarray([index[(c,a)] for c,_,_,a,_ in identities],dtype=np.int64)
    parameters, reports = {}, {}
    start = np.asarray([math.log(.3), math.log(.6/.4), math.log(.6/.4)])
    for i,statistic in enumerate(STATISTICS):
        counts = np.ascontiguousarray(observations[:,i,:])
        objective_calls = 0
        def objective(z):
            nonlocal objective_calls
            objective_calls += 1
            if not np.isfinite(z).all() or z[0] >= math.log(np.finfo(float).max):
                return math.inf
            rate = math.exp(float(z[0]))
            ph, pa = map(float, expit(z[1:]))
            if rate <= 0 or not 0 < ph < 1 or not 0 < pa < 1:
                return math.inf
            return float(_sequential_mae(home, away, counts, len(teams), rate, ph, pa))
        initial_error = objective(start)
        fit = minimize(objective, start.copy(), method='Nelder-Mead')
        rate = math.exp(float(fit.x[0]))
        ph, pa = map(float,expit(fit.x[1:]))
        triplet = {'lambda':rate, 'phi_home':ph, 'phi_away':pa}
        if not math.isfinite(float(fit.fun)) or not rate>0 or not 0<ph<1 or not 0<pa<1:
            raise ValueError(f'Nonfinite or invalid optimized GAP parameters for {statistic}')
        parameters[statistic] = triplet
        reports[statistic] = {
            'success':bool(fit.success),'message':str(fit.message),
            'objective_mae':float(fit.fun),'initial_mae':initial_error,
            'objective_evaluations':objective_calls,
            'optimizer_objective_evaluations':int(fit.nfev),
            'initial_mae_evaluations':1,'iterations':int(fit.nit),
            'parameters':triplet,'start_parameters':{'lambda':.3,'phi_home':.6,'phi_away':.6},
            'optimizer':'scipy Nelder-Mead with default stopping, one start, no restart/grid',
            'parameter_sharing':'one triplet per statistic shared across leagues; states remain league-local',
            'initialization':'all zero states in the first historical season',
            'objective_rows':'all 2022/23 fixtures, both sides; no first-six-match omission',
            'scope':'count MAE optimum is local; not fitted to later 1X2 outcomes',
        }
    return {'parameters':parameters,'optimization':reports,'n_matches':len(selected),
            'parameter_learning_season':'2022/23','n_teams':len(teams),
            'date_min':selected[0]['date'],'date_max':selected[-1]['date'],
            'all_converged':all(r['success'] for r in reports.values())}
