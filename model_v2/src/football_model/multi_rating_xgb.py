"""Fixed XGBoost 1X2 adapter for jointly supplied pre-match information.

The rating -> XGBClassifier architecture is adapted from Fabian Wunderlich and
coauthors' ``spoho-datascience/sparsity-sports-forecasting`` (Apache-2.0), notably
models/probability_models/ProbabilityModels.py. The upstream license is retained
with the downloaded source in outputs/全网模型比较复审_20261026/upstream/.
This independent adapter fixes the modern estimator configuration explicitly,
changes outcome coding to the local H=0/D=1/A=2 convention, and accepts richer,
already dated feature dictionaries. It is not an exact paper reproduction.

Feature availability is the caller's responsibility. This class selects earlier
fixtures and checks each selected row's supplied dates; it cannot establish a
source's publication time from a numeric feature. It supplies 1X2 only, with no
automatic goal distribution, calibration, stacking, or time decay.
"""
from copy import deepcopy
from datetime import date
import json
import math

import numpy as np
import xgboost as xgb


CLASS_NAMES = ('home', 'draw', 'away')
RANDOM_STATE = 20261026
PROBABILITY_SUM_TOLERANCE = 1e-6
FIXED_CONFIGURATION = {
    'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.3,
    'min_child_weight': 1, 'gamma': 0,
    'subsample': 1., 'colsample_bytree': 1.,
    'reg_lambda': 1., 'reg_alpha': 0.,
    'objective': 'multi:softprob', 'eval_metric': 'mlogloss',
    'tree_method': 'hist', 'device': 'cpu',
    'n_jobs': 1, 'random_state': RANDOM_STATE,
}


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
    if (not isinstance(value, dict)
            or any(not isinstance(k, str) or not k for k in value)):
        raise ValueError('Need match feature dictionaries with nonempty string names')
    for number in value.values():
        if number is None:
            continue
        if (isinstance(number, (bool, np.bool_))
                or not isinstance(number, (int, float, np.integer, np.floating))
                or math.isinf(float(number))):
            raise ValueError('Features must be numeric measurements, None or NaN; infinity is invalid')
    return value


def _outcome(score):
    if not isinstance(score, (list, tuple)) or len(score) != 2:
        raise ValueError('Need home and away nonnegative integer full-goal counts')
    goals = []
    for number in score:
        if (isinstance(number, (bool, np.bool_))
                or not isinstance(number, (int, float, np.integer, np.floating))):
            raise ValueError('Full goals must be nonnegative integer counts')
        number = float(number)
        if not math.isfinite(number) or number < 0 or not number.is_integer():
            raise ValueError('Full goals must be nonnegative integer counts')
        goals.append(number)
    return 0 if goals[0] > goals[1] else 1 if goals[0] == goals[1] else 2


class FixedMultiRatingXGB:
    """One fixed, unweighted, native-missing three-class XGBoost candidate."""

    @staticmethod
    def _matrix(features, names):
        return np.asarray([[np.nan if row.get(name) is None else float(row[name])
                            for name in names] for row in features], dtype=float)

    def fit(self, rows, before_date, feature_key='match_features'):
        if type(before_date) is not date:
            raise ValueError('Use a calendar-date fit cutoff')
        if not isinstance(feature_key, str) or not feature_key:
            raise ValueError('Need a named feature dictionary field')
        selected = []
        # Labels, features and other dates of excluded fixtures are never read.
        for row in rows:
            played = _calendar(row['date'], 'fixture date')
            if played < before_date:
                selected.append((played, row))
        identities = [row['match_id'] for _, row in selected]
        if (not identities
                or any(not isinstance(k, str) or not k for k in identities)
                or len(identities) != len(set(identities))):
            raise ValueError('Need unique nonempty earlier fixture identities')
        selected.sort(key=lambda item: (item[0], item[1]['match_id']))
        features, targets = [], []
        for played, row in selected:
            parameter = _calendar(row['fit_before_date'], 'historical parameter date')
            forecast = _calendar(row['forecast_before_date'], 'historical publication date')
            if not parameter <= forecast < played:
                raise ValueError('Historical parameter date must be at or before publication, before fixture')
            features.append(_features(row[feature_key]))
            targets.append(_outcome(row['actual_score']))
        if set(targets) != {0, 1, 2}:
            raise ValueError('All three earlier outcome classes are required; absent classes are not fabricated')
        names = sorted({name for row in features for name in row})
        if not names:
            raise ValueError('Need at least one named feature')
        matrix = self._matrix(features, names)
        estimator = xgb.XGBClassifier(**FIXED_CONFIGURATION)
        estimator.fit(matrix, np.asarray(targets, dtype=int))
        if estimator.classes_.tolist() != [0, 1, 2]:
            raise ValueError('Classifier must retain ordered home/draw/away classes')
        self.before_date, self.feature_names = before_date, names
        self.feature_key, self.estimator = feature_key, estimator
        missing = np.isnan(matrix)
        self._parameters = {
            'model_type': 'fixed_multi_rating_rich_information_xgboost_classifier',
            'response': 'one_fixture_one_home_draw_away_label_from_full_goals',
            'classes': [0, 1, 2], 'class_names': list(CLASS_NAMES),
            'upstream_class_mapping': {'upstream_0_away': 2, 'upstream_1_draw': 1, 'upstream_2_home': 0},
            'before_date': before_date.isoformat(),
            'training_min_date': selected[0][0].isoformat(),
            'training_max_date': selected[-1][0].isoformat(),
            'reference_match_ids': [row['match_id'] for _, row in selected],
            'n_matches': len(selected), 'n_training_rows': len(selected),
            'feature_key': feature_key, 'feature_names': names,
            'class_counts': {name: targets.count(i) for i, name in enumerate(CLASS_NAMES)},
            'configuration': deepcopy(FIXED_CONFIGURATION),
            'xgboost_version': xgb.__version__,
            'resolved_booster_configuration': json.loads(estimator.get_booster().save_config()),
            'boosting_rounds': estimator.get_booster().num_boosted_rounds(),
            'weighting': {'method': 'uniform one fixture one weight', 'half_life_days': None,
                          'note': 'upstream-style unweighted classification; differs from existing 180-day decay'},
            'missing': {
                'strategy': 'XGBoost native NaN routing; no fitted imputer or zero replacement',
                'counts_by_feature': dict(zip(names, missing.sum(axis=0).astype(int).tolist())),
                'all_missing_training_features': [name for i, name in enumerate(names) if missing[:, i].all()],
                'rows_with_any_missing': int(missing.any(axis=1).sum()),
                'unobserved_key_policy': 'missing feature keys are NaN; all-missing columns remain in the design',
            },
            'fixture_time_rule': 'fixture date strictly earlier than fit cutoff',
            'historical_time_rule': 'parameter date <= forecast publication date < fixture date',
            'probability_policy': {
                'order': list(CLASS_NAMES), 'source': 'raw predict_proba, no clipping or smoothing',
                'float_sum_tolerance': PROBABILITY_SUM_TOLERANCE,
                'roundoff_handling': 'divide a valid row by its sum only within tolerance; no other normalization',
            },
            'prediction_roundoff': {'rows_seen': 0, 'rows_corrected': 0, 'maximum_raw_sum_deviation': 0.},
            'source': {
                'repository': 'https://github.com/spoho-datascience/sparsity-sports-forecasting',
                'file': 'models/probability_models/ProbabilityModels.py',
                'license': 'Apache-2.0',
                'attribution': 'Fabian Wunderlich and coauthors',
                'adaptation': 'explicit modern fixed configuration, local label coding, dated richer features, native missing handling',
                'reproduction_claim': False,
            },
            'early_stopping': False, 'parameter_search': False, 'calibration': None,
            'output_scope': '1X2 only; no automatic score, total-goals or handicap distribution',
        }
        return self

    def predict(self, match_features, forecast_date):
        return self.predict_many([match_features], [forecast_date])[0]

    def predict_many(self, features, forecast_dates):
        if not hasattr(self, '_parameters'):
            raise ValueError('Fit earlier outcomes before predicting')
        features, dates = list(features), list(forecast_dates)
        if not features or len(features) != len(dates):
            raise ValueError('Need one publication date for each nonempty prediction batch')
        dates = [_calendar(value, 'forecast publication date') for value in dates]
        if any(value < self.before_date for value in dates):
            raise ValueError('Forecast publication must be at or after the fit cutoff')
        features = [_features(row) for row in features]
        unknown = sorted({name for row in features for name in row} - set(self.feature_names))
        if unknown:
            raise ValueError(f'Unseen feature names: {unknown}')
        raw = np.asarray(self.estimator.predict_proba(self._matrix(features, self.feature_names)), dtype=float)
        if (raw.shape != (len(features), 3) or not np.isfinite(raw).all()
                or np.any(raw < 0) or np.any(raw > 1)):
            raise ValueError('Classifier returned invalid ordered class probabilities')
        sums = raw.sum(axis=1)
        deviations = np.abs(sums - 1.)
        if np.any(deviations > PROBABILITY_SUM_TOLERANCE):
            raise ValueError('Classifier probability sum exceeds floating point tolerance')
        normalized = raw / sums[:, None]
        diagnostics = self._parameters['prediction_roundoff']
        diagnostics['rows_seen'] += len(features)
        diagnostics['rows_corrected'] += int(np.count_nonzero(deviations))
        diagnostics['maximum_raw_sum_deviation'] = max(
            diagnostics['maximum_raw_sum_deviation'], float(deviations.max()))
        return normalized.tolist()

    def parameters(self):
        if not hasattr(self, '_parameters'):
            raise ValueError('Fit earlier outcomes before reading parameters')
        return deepcopy(self._parameters)
