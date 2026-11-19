"""Fixed-complexity direct three-class outcome learners.

Each historical fixture contributes one row and one home/draw/away label. The
caller supplies an explicitly dated match-level feature dictionary. This module
does not infer a score, total-goals law, or handicap distribution from 1X2.
"""
from copy import deepcopy
from datetime import date
import math
import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


HALF_LIFE_DAYS = 180
RANDOM_STATE = 20261025
CLASS_NAMES = ('home', 'draw', 'away')


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
    if (not isinstance(value, dict) or not value
            or any(not isinstance(k, str) or not k for k in value)):
        raise ValueError('Need one nonempty match feature dictionary per fixture')
    for measurement in value.values():
        if measurement is None:
            continue
        if (isinstance(measurement, (bool, np.bool_))
                or not isinstance(measurement, (int, float, np.integer, np.floating))
                or not math.isfinite(float(measurement))):
            raise ValueError('Features must be finite numeric measurements or None')
    return value


def _outcome(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or int(value) not in (0, 1, 2)):
        raise ValueError('Outcome must be integer 0=home, 1=draw or 2=away')
    return int(value)


class EarlyOutcomeLearner:
    """One preset direct classifier, with strictly earlier preprocessing.

    ExtraTrees leaf30 counts fixtures. The related goal-regression preset uses
    leaf60 scoring-side rows; these broadly align the minimum fixture scale,
    but classification and two-side regression remain different objectives and
    training representations, not a causal isolation of the target alone.
    """

    def __init__(self, kind):
        if kind not in ('logistic', 'extra_trees'):
            raise ValueError('Unknown direct-outcome learner')
        self.kind = kind

    def _estimator(self):
        if self.kind == 'logistic':
            return LogisticRegression(C=1., solver='lbfgs', max_iter=1000,
                tol=1e-8, random_state=RANDOM_STATE)
        return ExtraTreesClassifier(criterion='log_loss', n_estimators=120,
            max_depth=6, min_samples_leaf=30, max_features=.8, bootstrap=False,
            random_state=RANDOM_STATE, n_jobs=1)

    @staticmethod
    def _matrix(features, names):
        return np.asarray([[np.nan if row.get(name) is None else float(row[name])
                            for name in names] for row in features], dtype=float)

    def fit(self, rows, before_date):
        if type(before_date) is not date:
            raise ValueError('Use a calendar-date parameter cutoff')
        selected = []
        # Do not inspect labels, publication dates or measurements from a row
        # that cannot belong to this earlier fit.
        for row in rows:
            played = _calendar(row['date'], 'fixture date')
            if played < before_date:
                selected.append((played, row))
        ids = [row['match_id'] for _, row in selected]
        if (not ids or any(not isinstance(k, str) or not k for k in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('Need unique nonempty earlier fixture identities')
        selected.sort(key=lambda item: (item[0], item[1]['match_id']))
        ids = [row['match_id'] for _, row in selected]
        features, targets, raw_weights = [], [], []
        for played, row in selected:
            parameter = _calendar(row['fit_before_date'], 'historical parameter date')
            forecast = _calendar(row['forecast_before_date'], 'historical publication date')
            if not parameter <= forecast < played:
                raise ValueError('Historical parameter date must be at or before publication, before fixture')
            features.append(_features(row['match_features']))
            targets.append(_outcome(row['actual_outcome']))
            raw_weights.append(math.exp(-math.log(2)*(before_date-played).days/HALF_LIFE_DAYS))
        if set(targets) != {0, 1, 2}:
            raise ValueError('All three earlier outcome classes are required; missing classes are not fabricated')
        names = sorted({name for row in features for name in row})
        raw_weights = np.asarray(raw_weights, dtype=float)
        if not np.isfinite(raw_weights).all() or np.any(raw_weights <= 0):
            raise ValueError('Historical time weights must remain finite and positive')
        factor = len(ids)/float(raw_weights.sum())
        weights = raw_weights*factor
        matrix = self._matrix(features, names)
        imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
        design = imputer.fit_transform(matrix)
        expanded_names = list(imputer.get_feature_names_out(names))
        scaler = None
        if self.kind == 'logistic':
            scaler = StandardScaler().fit(design, sample_weight=weights)
            design = scaler.transform(design)
            if (not np.isfinite(scaler.scale_).all() or np.any(scaler.scale_ <= 0)
                    or not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.var_).all()):
                raise ValueError('Training feature scaling must remain finite')
        if not np.isfinite(design).all():
            raise ValueError('Training preprocessing must remain finite')
        estimator = self._estimator()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                estimator.fit(design, np.asarray(targets, dtype=int), sample_weight=weights)
        except ConvergenceWarning as error:
            raise ValueError('Preset direct-outcome optimizer did not converge; report the failed fit') from error
        if estimator.classes_.tolist() != [0, 1, 2]:
            raise ValueError('Classifier must retain ordered home/draw/away classes')
        self.before_date, self.feature_names = before_date, names
        self.imputer, self.scaler, self.estimator = imputer, scaler, estimator
        self._parameters = {
            'model_type': 'fixed_preset_direct_three_class_earlier_outcome', 'kind': self.kind,
            'response': 'one_fixture_one_home_draw_away_label', 'classes': [0, 1, 2],
            'class_names': list(CLASS_NAMES), 'before_date': before_date.isoformat(),
            'training_min_date': selected[0][0].isoformat(),
            'training_max_date': selected[-1][0].isoformat(),
            'n_matches': len(ids), 'n_training_rows': len(targets),
            'reference_match_ids': ids, 'half_life_days': HALF_LIFE_DAYS,
            'feature_names': names, 'expanded_feature_names': expanded_names,
            'configuration': estimator.get_params(),
            'class_counts': {name: targets.count(i) for i, name in enumerate(CLASS_NAMES)},
            'weighting': {'method': 'exp(-log2*age/180) normalized to mean1',
                'raw_sum': float(raw_weights.sum()), 'normalizing_factor': factor,
                'sum': float(weights.sum()), 'n_fixture_weights': len(weights),
                'note': 'fixed logistic C applies after weight normalization; tree splits invariant to common positive scaling'},
            'fixture_time_rule': 'fixture date strictly earlier than parameter cutoff; both teams are in one match row',
            'historical_time_rule': 'each row parameter date <= its publication date < its own fixture date',
            'imputation': {'strategy': 'unweighted training median', 'add_indicator': True,
                'keep_empty_features': True, 'statistics': imputer.statistics_.tolist(),
                'indicator_feature_indices': imputer.indicator_.features_.tolist(),
                'all_missing_training_features': [names[i] for i in range(len(names)) if np.isnan(matrix[:, i]).all()],
                'empty_feature_handling': 'retained with imputer zero substitution and missing indicator; zero is not an observed measurement'},
            'scaling': None if scaler is None else {'method': 'training weighted StandardScaler',
                'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'variance': scaler.var_.tolist()},
            'probability_policy': 'raw ordered predict_proba; no clipping, smoothing or manual renormalization; zero probabilities reported as failed predictions',
            'output_scope': '1X2 only; does not identify full scores, totals or handicaps',
            'new_parameter_grid': False,
        }
        if self.kind == 'logistic':
            self._parameters.update(optimizer_iterations=estimator.n_iter_.tolist(),
                intercepts=dict(zip(CLASS_NAMES, estimator.intercept_.tolist())),
                coefficients={name: dict(zip(expanded_names, coefficients.tolist()))
                    for name, coefficients in zip(CLASS_NAMES, estimator.coef_, strict=True)},
                objective='multinomial logistic probability loss with fixed regularization')
        else:
            self._parameters.update(impurity_feature_importance=dict(zip(expanded_names, estimator.feature_importances_.tolist())),
                importance_note='training impurity importance; not causal or held-out contribution',
                comparison_note='leaf30 fixture rows versus related goal-regression leaf60 scoring-side rows; different objectives and representations')
        return self

    def predict(self, match_features, forecast_date):
        return self.predict_many([match_features], [forecast_date])[0]

    def predict_many(self, match_features, forecast_dates):
        if not hasattr(self, '_parameters'):
            raise ValueError('Fit earlier direct outcomes before predicting')
        try:
            features, dates = list(match_features), list(forecast_dates)
        except TypeError as error:
            raise ValueError('Need feature dictionaries and publication dates') from error
        if not features or len(features) != len(dates):
            raise ValueError('Need one publication date for each nonempty match feature dictionary')
        if any(type(value) is not date or value < self.before_date for value in dates):
            raise ValueError('Forecast publication must be at or after the parameter cutoff')
        features = [_features(row) for row in features]
        unknown = sorted({name for row in features for name in row}-set(self.feature_names))
        if unknown:
            raise ValueError(f'Unseen feature names: {unknown}')
        design = self.imputer.transform(self._matrix(features, self.feature_names))
        if self.scaler is not None:
            design = self.scaler.transform(design)
        if not np.isfinite(design).all():
            raise ValueError('Prediction preprocessing must remain finite')
        if self.estimator.classes_.tolist() != [0, 1, 2]:
            raise ValueError('Classifier class order differs from home/draw/away')
        probabilities = np.asarray(self.estimator.predict_proba(design), dtype=float)
        if (probabilities.shape != (len(features), 3) or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or np.any(probabilities > 1)
                or not np.allclose(probabilities.sum(axis=1), 1., rtol=0, atol=1e-10)):
            raise ValueError('Classifier must return three raw finite probabilities summing to one')
        if np.any(probabilities == 0):
            raise ValueError('Zero class probability from preset classifier; report failed prediction without smoothing')
        return [tuple(map(float, values)) for values in probabilities]

    def parameters(self):
        if not hasattr(self, '_parameters'):
            raise ValueError('Learner is not fitted')
        return deepcopy(self._parameters)
