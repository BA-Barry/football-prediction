"""Fixed-complexity full-goal learners for earlier-information comparisons.

Both scoring sides of a fixture always enter the same historical fit. Feature
construction and source availability remain the caller's responsibility; this
adapter learns preprocessing from earlier rows only. It returns two conditional
goal means, not a claim that independent Poisson scores are already calibrated.
"""
from datetime import date
import math

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler


HALF_LIFE_DAYS = 180
RANDOM_STATE = 20261025


def _calendar(value, label):
    if type(value) is date:
        return value
    if not isinstance(value, str):
        raise ValueError(f'Need an ISO calendar date: {label}')
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f'Need an ISO calendar date: {label}') from error


def _features(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('Need home and away feature dictionaries')
    for side in value:
        if not isinstance(side, dict) or any(not isinstance(k, str) or not k for k in side):
            raise ValueError('Feature names must be nonempty strings')
        for measurement in side.values():
            if measurement is None:
                continue
            if (isinstance(measurement, (bool, np.bool_))
                    or not isinstance(measurement, (int, float, np.integer, np.floating))
                    or not math.isfinite(float(measurement))):
                raise ValueError('Features must be finite numeric measurements or None')
    return value


def _counts(value):
    raw = np.asarray(value, dtype=object)
    if raw.shape != (2,):
        raise ValueError('Need two full-goal counts')
    values = []
    for number in raw:
        if (isinstance(number, (bool, np.bool_))
                or not isinstance(number, (int, float, np.integer, np.floating))):
            raise ValueError('Full goals must be nonnegative integer counts')
        number = float(number)
        if not math.isfinite(number) or number < 0 or not number.is_integer():
            raise ValueError('Full goals must be nonnegative integer counts')
        values.append(number)
    return values


class EarlyGoalLearner:
    """One preset learner; basic/rich identifies the caller-supplied feature set."""

    def __init__(self, kind, feature_set):
        if kind not in ('glm', 'hgb', 'extra_trees'):
            raise ValueError('Unknown full-goal learner')
        if feature_set not in ('basic', 'rich'):
            raise ValueError('Feature set must be basic or rich')
        self.kind, self.feature_set = kind, feature_set

    def _estimator(self):
        if self.kind == 'glm':
            return PoissonRegressor(alpha=.01, max_iter=1000, tol=1e-8)
        if self.kind == 'hgb':
            return HistGradientBoostingRegressor(
                loss='poisson', learning_rate=.05, max_iter=150,
                max_leaf_nodes=7, min_samples_leaf=60, l2_regularization=1.,
                early_stopping=False, random_state=RANDOM_STATE)
        return ExtraTreesRegressor(
            criterion='poisson', n_estimators=120, max_depth=6,
            min_samples_leaf=60, max_features=.8, bootstrap=False,
            random_state=RANDOM_STATE, n_jobs=1)

    @staticmethod
    def _matrix(sides, names):
        return np.asarray([[np.nan if side.get(k) is None else float(side[k])
                            for k in names] for side in sides], dtype=float)

    def fit(self, rows, before_date):
        if type(before_date) is not date:
            raise ValueError('Use a calendar-date fit cutoff')
        selected = []
        for row in rows:
            played = _calendar(row['date'], 'fixture date')
            if played < before_date:
                selected.append((played, row))
        selected.sort(key=lambda item: (item[0], item[1]['match_id']))
        ids = [r['match_id'] for _, r in selected]
        if (not ids or any(not isinstance(k, str) or not k for k in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('Need unique nonempty earlier fixtures')
        sides, targets, weights = [], [], []
        for played, row in selected:
            if _calendar(row['fit_before_date'], 'feature cutoff') > played:
                raise ValueError('Historical features cannot use a cutoff after their fixture')
            values = _features(row['side_features'])
            sides.extend(values)
            targets.extend(_counts(row['actual_score']))
            weight = math.exp(-math.log(2)*(before_date-played).days/HALF_LIFE_DAYS)
            weights.extend((weight, weight))
        if math.fsum(targets) <= 0:
            raise ValueError('Need positive total goals to estimate positive goal means')
        names = sorted({key for side in sides for key in side})
        if not names:
            raise ValueError('Need at least one named feature')
        weights = np.asarray(weights, dtype=float)
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError('Time weights must be finite and positive')
        matrix = self._matrix(sides, names)
        imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
        design = imputer.fit_transform(matrix)
        expanded_names = list(imputer.get_feature_names_out(names))
        scaler = None
        if self.kind == 'glm':
            scaler = StandardScaler().fit(design, sample_weight=weights)
            design = scaler.transform(design)
            if (not np.isfinite(scaler.scale_).all() or np.any(scaler.scale_ <= 0)
                    or not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.var_).all()):
                raise ValueError('Training feature scaling must remain finite')
        if not np.isfinite(design).all():
            raise ValueError('Training feature preprocessing must remain finite')
        estimator = self._estimator()
        estimator.fit(design, np.asarray(targets), sample_weight=weights)
        self.before_date = before_date
        self.feature_names = names
        self.imputer, self.scaler, self.estimator = imputer, scaler, estimator
        self._parameters = {
            'model_type': 'fixed_preset_earlier_information_full_goal_regression',
            'kind': self.kind, 'feature_set': self.feature_set,
            'response': 'real_full_time_goals',
            'before_date': before_date.isoformat(),
            'training_min_date': selected[0][0].isoformat(),
            'training_max_date': selected[-1][0].isoformat(),
            'n_matches': len(ids), 'n_scoring_sides': len(targets),
            'reference_match_ids': ids, 'half_life_days': HALF_LIFE_DAYS,
            'feature_names': names, 'expanded_feature_names': expanded_names,
            'feature_cutoff_rule': 'each historical row retains its original feature cutoff, no later than its fixture date',
            'fixture_time_rule': 'both scoring sides enter only when fixture date is strictly earlier than fit cutoff',
            'configuration': estimator.get_params(),
            'imputation': {'strategy': 'unweighted training median',
                'add_indicator': True, 'keep_empty_features': True,
                'statistics': imputer.statistics_.tolist(),
                'indicator_feature_indices': imputer.indicator_.features_.tolist(),
                'all_missing_training_features': [names[i] for i in range(len(names)) if np.isnan(matrix[:, i]).all()],
                'empty_feature_handling': 'retained with zero substitution and a missing indicator; zero is not asserted to be an observed measurement'},
            'scaling': None if scaler is None else {
                'method': 'training sample-weighted StandardScaler',
                'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                'variance': scaler.var_.tolist()},
            'prediction_interpretation': 'two positive conditional goal means; score distribution and dependence remain caller assumptions',
            'new_parameter_grid': False,
        }
        if self.kind == 'glm':
            self._parameters.update(intercept=float(estimator.intercept_),
                                    coefficients=dict(zip(expanded_names, estimator.coef_.tolist())),
                                    optimizer_iterations=int(estimator.n_iter_))
        elif self.kind == 'extra_trees':
            self._parameters['impurity_feature_importance'] = dict(zip(expanded_names, estimator.feature_importances_.tolist()))
            self._parameters['importance_note'] = 'training impurity importance; neither causal importance nor held-out predictive contribution'
        else:
            self._parameters['fitted_boosting_iterations'] = int(estimator.n_iter_)
        return self

    def predict(self, side_features, match_date):
        return self.predict_many([side_features], [match_date])[0]

    def predict_many(self, feature_pairs, match_dates):
        """Transform and predict a group of fixtures once without changing means."""
        if not hasattr(self, '_parameters'):
            raise ValueError('Fit earlier full goals before prediction')
        try:
            pairs, dates = list(feature_pairs), list(match_dates)
        except TypeError as error:
            raise ValueError('Need feature pairs and corresponding calendar dates') from error
        if not pairs or len(pairs) != len(dates):
            raise ValueError('Need one calendar date for each nonempty feature pair')
        if any(type(played) is not date or played < self.before_date for played in dates):
            raise ValueError('Prediction must be at or after the fit cutoff')
        sides = [side for pair in pairs for side in _features(pair)]
        unknown = sorted({key for side in sides for key in side} - set(self.feature_names))
        if unknown:
            raise ValueError(f'Unseen feature names: {unknown}')
        design = self.imputer.transform(self._matrix(sides, self.feature_names))
        if self.scaler is not None:
            design = self.scaler.transform(design)
        if not np.isfinite(design).all():
            raise ValueError('Prediction feature preprocessing must remain finite')
        rates = np.asarray(self.estimator.predict(design), dtype=float)
        if rates.shape != (2*len(pairs),) or not np.isfinite(rates).all() or np.any(rates <= 0):
            raise ValueError('Learner must return two positive finite goal means')
        return [tuple(map(float, pair)) for pair in rates.reshape(-1, 2)]

    def parameters(self):
        if not hasattr(self, '_parameters'):
            raise ValueError('Learner is not fitted')
        # Return independent readable metadata; fitted estimator remains accessible.
        import copy
        return copy.deepcopy(self._parameters)
