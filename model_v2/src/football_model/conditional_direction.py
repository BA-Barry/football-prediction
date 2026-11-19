"""One fixed conditional H/A learner, retaining the existing tree preset.

Earlier draws are legitimate labels for selecting the conditional training
sample. Forecast targets must never be selected using their actual outcomes.
This component returns q=P(H|non-draw), not an executable non-draw probability.
"""
from copy import deepcopy
from datetime import date
import math

import numpy as np
from sklearn.impute import SimpleImputer

from .early_outcome_learners import (
    EarlyOutcomeLearner, HALF_LIFE_DAYS, _calendar, _features, _outcome,
)


class ConditionalDirectionLearner:
    """Fit only earlier H/A rows, with no additional algorithm or hyperparameters."""

    def fit(self, rows, before_date):
        if type(before_date) is not date:
            raise ValueError('Use a calendar-date parameter cutoff')
        earlier = []
        for row in rows:
            played = _calendar(row['date'], 'fixture date')
            if played < before_date:
                earlier.append((played, row))
        identities = [row['match_id'] for _, row in earlier]
        if (not identities or any(not isinstance(key, str) or not key for key in identities)
                or len(set(identities)) != len(identities)):
            raise ValueError('Need unique nonempty earlier fixture identities')
        earlier.sort(key=lambda item: (item[0], item[1]['match_id']))
        selected, excluded_draws = [], []
        for played, row in earlier:
            parameter = _calendar(row['fit_before_date'], 'historical parameter date')
            forecast = _calendar(row['forecast_before_date'], 'historical publication date')
            if not parameter <= forecast < played:
                raise ValueError('Historical parameter date must be at or before publication, before fixture')
            outcome = _outcome(row['actual_outcome'])
            if outcome == 1:
                # These features cannot influence conditional imputation, names,
                # missingness indicators or sample weights.
                excluded_draws.append(row['match_id'])
                continue
            selected.append((played, row, outcome))
        if {outcome for _, _, outcome in selected} != {0, 2}:
            raise ValueError('Both earlier home and away wins are required')
        features = [_features(row['match_features']) for _, row, _ in selected]
        names = sorted({name for row in features for name in row})
        matrix = EarlyOutcomeLearner._matrix(features, names)
        imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
        design = imputer.fit_transform(matrix)
        raw_weights = np.asarray([
            math.exp(-math.log(2)*(before_date-played).days/HALF_LIFE_DAYS)
            for played, _, _ in selected], dtype=float)
        if not np.isfinite(raw_weights).all() or np.any(raw_weights <= 0):
            raise ValueError('Historical time weights must remain finite and positive')
        factor = len(selected)/float(raw_weights.sum())
        weights = raw_weights*factor
        if not np.isfinite(design).all() or not np.isfinite(weights).all():
            raise ValueError('Conditional training design and weights must remain finite')
        estimator = EarlyOutcomeLearner('extra_trees')._estimator()
        estimator.fit(design, np.asarray([outcome for _, _, outcome in selected]),
                      sample_weight=weights)
        if estimator.classes_.tolist() != [0, 2]:
            raise ValueError('Conditional classes must retain ordered home=0, away=2')
        self.before_date, self.feature_names = before_date, names
        self.imputer, self.estimator = imputer, estimator
        self._parameters = {
            'model_type': 'fixed_existing_tree_conditional_home_away',
            'response': 'home_given_non_draw_earlier_outcomes_only',
            'classes': [0, 2], 'class_names': ['home', 'away'],
            'before_date': before_date.isoformat(), 'half_life_days': HALF_LIFE_DAYS,
            'n_matches': len(selected), 'n_earlier_all_matches': len(earlier),
            'reference_match_ids': [row['match_id'] for _, row, _ in selected],
            'earlier_all_match_ids': [row['match_id'] for _, row in earlier],
            'excluded_draw_match_ids': excluded_draws,
            'training_min_date': selected[0][0].isoformat(),
            'training_max_date': selected[-1][0].isoformat(),
            'feature_names': names,
            'expanded_feature_names': list(imputer.get_feature_names_out(names)),
            'configuration': estimator.get_params(),
            'class_counts': {'home': sum(y == 0 for _, _, y in selected),
                             'away': sum(y == 2 for _, _, y in selected)},
            'weighting': {'method': 'exp(-log2*age/180), conditional training weights normalized to mean1',
                          'raw_sum': float(raw_weights.sum()), 'normalizing_factor': factor,
                          'sum': float(weights.sum()),
                          'common_scaling_note': 'Common positive scaling does not change relative tree split objectives'},
            'imputation': {'strategy': 'unweighted conditional training median',
                'add_indicator': True, 'keep_empty_features': True,
                'statistics': imputer.statistics_.tolist(),
                'indicator_feature_indices': imputer.indicator_.features_.tolist(),
                'all_missing_training_features': [names[i] for i in range(len(names)) if np.isnan(matrix[:, i]).all()],
                'empty_feature_handling': 'Internal zero plus missing indicator; not an observed zero effect'},
            'impurity_feature_importance': dict(zip(imputer.get_feature_names_out(names), estimator.feature_importances_.tolist())),
            'importance_note': 'Training split importance, not causal or held-out contribution',
            'parameter_search': False,
            'probability_policy': 'raw ordered predict_proba home column; no clipping, smoothing or target-outcome filtering',
            'scope': 'q=P(home|non-draw); requires a separately supplied draw probability for full H/D/A',
            'sample_scale_note': 'leaf30 is now 30 conditional non-draw fixtures; not equivalent to 30 fixtures in the old ternary learner',
        }
        return self

    def predict_many(self, match_features, forecast_dates):
        if not hasattr(self, '_parameters'):
            raise ValueError('Fit earlier conditional outcomes before predicting')
        features, dates = list(match_features), list(forecast_dates)
        if not features or len(features) != len(dates):
            raise ValueError('Need one publication date for each nonempty target feature dictionary')
        if any(type(value) is not date or value < self.before_date for value in dates):
            raise ValueError('Forecast publication cannot precede the parameter cutoff')
        features = [_features(row) for row in features]
        unknown = sorted({name for row in features for name in row}-set(self.feature_names))
        if unknown:
            raise ValueError(f'Unseen feature names: {unknown}')
        design = self.imputer.transform(EarlyOutcomeLearner._matrix(features, self.feature_names))
        if not np.isfinite(design).all():
            raise ValueError('Conditional prediction design must remain finite')
        if self.estimator.classes_.tolist() != [0, 2]:
            raise ValueError('Conditional class order differs from home/away')
        probabilities = np.asarray(self.estimator.predict_proba(design), dtype=float)
        if (probabilities.shape != (len(features), 2) or not np.isfinite(probabilities).all()
                or np.any(probabilities <= 0) or np.any(probabilities >= 1)
                or not np.allclose(probabilities.sum(axis=1), 1., rtol=0, atol=1e-12)):
            raise ValueError('Need native positive conditional probabilities without smoothing')
        return probabilities[:, 0].tolist()

    def predict(self, match_features, forecast_date):
        return self.predict_many([match_features], [forecast_date])[0]

    def parameters(self):
        if not hasattr(self, '_parameters'):
            raise ValueError('Conditional learner is not fitted')
        return deepcopy(self._parameters)
