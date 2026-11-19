"""Earlier-outcome ordered-logit mapping for four GAP statistic differences.

This directly reuses statsmodels' likelihood and prediction implementation.
It is a local adaptation, not the original paper's least-squares mapping.
"""
from datetime import date
import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels
from statsmodels.miscmodels.ordinal_model import OrderedModel


FEATURES = ('goals', 'shots_on_target', 'non_target_shots', 'corners')


def feature_matrix(rows):
    matrix = np.asarray([[r['gap_differences'][name] for name in FEATURES] for r in rows], dtype=float)
    if matrix.shape != (len(rows), len(FEATURES)) or not np.isfinite(matrix).all():
        raise ValueError('Four finite pre-match GAP differences are required')
    return matrix


class GAPOutcomeModel:
    """One unweighted maximum-likelihood mapping; no regularization grid."""

    def fit(self, rows, before_date):
        if type(before_date) is not date:
            raise ValueError('Use a calendar parameter date')
        rows = sorted(rows, key=lambda r: (r['date'], r['match_id']))
        if not rows or len({r['match_id'] for r in rows}) != len(rows):
            raise ValueError('Nonempty unique historical fixtures are required')
        for r in rows:
            if not r['forecast_before_date'] < r['date'] < before_date.isoformat():
                raise ValueError('Historical features and labels must precede their respective cutoffs')
            if r['feature_before_date'] != r['forecast_before_date']:
                raise ValueError('Use the original earlier feature snapshot')
        y_project = np.asarray([r['actual_outcome'] for r in rows], dtype=int)
        if set(y_project.tolist()) != {0, 1, 2}:
            raise ValueError('All three observed outcome classes are required')
        x = feature_matrix(rows)
        self.scaler = StandardScaler().fit(x)
        if np.any(self.scaler.var_ <= 0):
            raise ValueError('Constant GAP inputs cannot enter ordered regression as implicit intercepts')
        x = self.scaler.transform(x)
        if np.linalg.matrix_rank(np.column_stack([np.ones(len(x)), x])) != len(FEATURES)+1:
            raise ValueError('GAP inputs contain a constant or linear dependency')
        # OrderedModel returns ascending away/draw/home; the project exports H/D/A.
        native_y = 2-y_project
        model = OrderedModel(native_y, x, distr='logit')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = model.fit(method='bfgs', maxiter=1000, gtol=1e-7, disp=False)
        if not result.mle_retvals.get('converged', False):
            raise ValueError('Preset ordered-logit optimizer did not converge')
        if not np.isfinite(result.params).all() or not np.isfinite(result.llf):
            raise ValueError('Ordered-logit fit returned nonfinite estimates')
        self.result, self.before_date = result, before_date
        thresholds = model.transform_threshold_params(result.params)[1:-1]
        self._parameters = {
            'model_type':'GAP_four_difference_ordered_logit_ML',
            'statsmodels_version':statsmodels.__version__, 'distr':'logit',
            'before_date':before_date.isoformat(), 'n_training_matches':len(rows),
            'training_date_min':rows[0]['date'], 'training_date_max':rows[-1]['date'],
            'reference_match_ids':[r['match_id'] for r in rows],
            'feature_names':list(FEATURES), 'feature_scaler_mean':self.scaler.mean_.tolist(),
            'feature_scaler_scale':self.scaler.scale_.tolist(),
            'native_class_order':['away','draw','home'], 'export_class_order':['home','draw','away'],
            'standardized_coefficients':dict(zip(FEATURES, result.params[:4].tolist())),
            'raw_threshold_parameters':result.params[4:].tolist(),
            'latent_thresholds':thresholds.tolist(),
            'converged':True, 'log_likelihood':float(result.llf),
            'optimizer':{k:(v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else v)
                         for k,v in result.mle_retvals.items()},
            'warnings':[{'category':w.category.__name__, 'message':str(w.message)} for w in caught],
            'weights':'all earlier 2023/24+ mapping fixtures equally weighted; no unsupported likelihood weights',
            'scope':'one_x_two_only; no score, total-goals or handicap distribution',
            'paper_adaptation':'Maximum likelihood instead of reported least-squares mapping; no burn-in exclusion',
        }
        return self

    def predict_many(self, rows):
        if not hasattr(self, 'result') or not rows:
            raise ValueError('Fit a model before predicting nonempty rows')
        for r in rows:
            if not self.before_date.isoformat() <= r['forecast_before_date'] < r['date']:
                raise ValueError('Prediction must follow fitting and precede kickoff date')
            if r['feature_before_date'] != r['forecast_before_date']:
                raise ValueError('Prediction features must have the declared publication date')
        x = self.scaler.transform(feature_matrix(rows))
        native = np.asarray(self.result.model.predict(self.result.params, exog=x),dtype=float)
        p = native[:, ::-1]
        if (p.shape != (len(rows),3) or not np.isfinite(p).all() or np.any(p<0)
                or np.any(p>1) or not np.allclose(p.sum(axis=1),1,atol=1e-12,rtol=0)):
            raise ValueError('Invalid native ordered-logit probabilities')
        return p

    def parameters(self):
        if not hasattr(self, '_parameters'):
            raise ValueError('No ordered-logit fit has been performed')
        return self._parameters
