"""Learn one model/LLM probability weight from earlier prospective outcomes.

The caller resolves source identities, LLM exposure and forecast-version choice.
Timestamps here are availability claims, not source authentication. One match is
one observation; fitting this component is not evidence of predictive improvement.
No market, intercept, class-specific weight, probability smoothing or grid is used.
"""
from collections import Counter
from copy import deepcopy
import math

import numpy as np
from scipy.optimize import brentq

from .prospective import _probabilities, _text, _time


OUTCOMES = ('home', 'draw', 'away')


def _outcome(value):
    if isinstance(value, str) and value in OUTCOMES:
        return OUTCOMES.index(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value in (0, 1, 2):
        return int(value)
    raise ValueError('actual_outcome must be home/draw/away or integer 0/1/2')


def _objective(weight, statistical_truth, llm_truth):
    """Convex mean log loss and its derivatives, without clipping zero mass."""
    if weight == 0:
        mass = llm_truth
    elif weight == 1:
        mass = statistical_truth
    else:
        mass = weight*statistical_truth + (1-weight)*llm_truth
    delta = statistical_truth-llm_truth
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        value = float(-np.log(mass).mean())
        gradient = float(-(delta/mass).mean())
        curvature = float(np.square(delta/mass).mean())
    return value, gradient, curvature


def _json_number(value):
    return float(value) if math.isfinite(value) else None


class LearnedLinearPool:
    """P = w*P(statistical) + (1-w)*P(LLM), one equal-weight loss per fixture.

    fit(rows, before) takes mappings with match_id, kickoff,
    statistical_generated_at, llm_generated_at, statistical_probabilities,
    llm_probabilities, result_observed_at, and actual_outcome. Optional
    statistical_family/llm_family are recorded, never inferred. Optional llm_track
    declares llm_independent, llm_model_informed, or unknown (the default). Mixed
    declarations may be studied but are not a uniform target-route allocation.
    A None result observation denotes an unsettled fixture. Result availability is filtered
    strictly before `before`, before future labels/probabilities are inspected.
    """

    def fit(self, rows, before):
        cutoff = _time(before, 'fit before')
        history = []
        for row in rows:
            observed = row.get('result_observed_at')
            if observed is None:
                continue
            observed = _time(observed, 'result_observed_at')
            if observed >= cutoff:
                continue
            identity = _text(row.get('match_id'), 'match_id')
            kickoff = _time(row.get('kickoff'), 'kickoff')
            stat_at = _time(row.get('statistical_generated_at'), 'statistical_generated_at')
            llm_at = _time(row.get('llm_generated_at'), 'llm_generated_at')
            if not max(stat_at, llm_at) < kickoff < observed:
                raise ValueError('Both real forecasts must precede kickoff and the observed result must follow it')
            stat = _probabilities(row.get('statistical_probabilities'))
            llm = _probabilities(row.get('llm_probabilities'))
            actual = _outcome(row.get('actual_outcome'))
            families = {
                key: _text(row.get(key, 'unknown'), key)
                for key in ('statistical_family', 'llm_family')
            }
            llm_track = row.get('llm_track', 'unknown')
            if llm_track not in ('llm_independent', 'llm_model_informed', 'unknown'):
                raise ValueError('llm_track must explicitly identify an allowed LLM exposure route or unknown')
            history.append({'match_id': identity, 'kickoff': kickoff,
                'result_observed_at': observed, 'statistical_generated_at': stat_at,
                'llm_generated_at': llm_at, 'statistical_probabilities': stat,
                'llm_probabilities': llm, 'actual_outcome': actual, 'llm_track': llm_track, **families})
        if not history:
            raise ValueError('Need earlier settled prospective forecast pairs to estimate a weight')
        identities = [row['match_id'] for row in history]
        if len(set(identities)) != len(identities):
            raise ValueError('Duplicate match_id: repeated forecast versions are not independent observations')
        history.sort(key=lambda row:(row['result_observed_at'], row['kickoff'], row['match_id']))
        stat_y = np.asarray([row['statistical_probabilities'][row['actual_outcome']] for row in history])
        llm_y = np.asarray([row['llm_probabilities'][row['actual_outcome']] for row in history])
        if np.any((stat_y == 0) & (llm_y == 0)):
            raise ValueError('Both routes assign zero to a realised outcome: no finite log loss at any weight')
        evaluations = 0

        def evaluate(weight):
            nonlocal evaluations
            evaluations += 1
            return _objective(weight, stat_y, llm_y)

        left, right = evaluate(0.), evaluate(1.)
        identified = not np.array_equal(stat_y, llm_y)
        if not identified:
            weight, optimum, method = None, left, 'empirical_loss_flat_weight_unidentified'
        else:
            if left[1] >= 0:
                weight, method = 0., 'convex_llm_endpoint'
            elif right[1] <= 0:
                weight, method = 1., 'convex_statistical_endpoint'
            else:
                weight = float(brentq(lambda w:evaluate(w)[1], 0., 1., xtol=1e-14, rtol=1e-14))
                method = 'brentq_convex_gradient_root'
            optimum = evaluate(weight)
            # Compare both closed endpoints explicitly; retain a genuine endpoint
            # optimum instead of forcing a blend through an open-domain optimiser.
            if left[0] < optimum[0]:
                weight, optimum, method = 0., left, 'compared_llm_endpoint'
            if right[0] < optimum[0]:
                weight, optimum, method = 1., right, 'compared_statistical_endpoint'
        if not math.isfinite(optimum[0]):
            raise ValueError('No finite empirical log-loss solution; no probability smoothing is applied')
        ids = [row['match_id'] for row in history]
        self.fit_before = cutoff
        self.weight = weight
        self.reference_match_ids = set(ids)
        track_counts = dict(sorted(Counter(row['llm_track'] for row in history).items()))
        self._parameters = {
            'model_type': 'one_shared_linear_probability_pool',
            'weight_source': 'earlier_prospective_log_loss_fit',
            'fit_before': cutoff.isoformat(), 'class_order': list(OUTCOMES),
            'n_matches': len(history), 'reference_match_ids': ids,
            'weight': weight,
            'weights': None if weight is None else {'statistical': weight, 'llm': 1-weight},
            'identified': identified,
            'identified_weight_interval': [weight, weight] if identified else [0., 1.],
            'interval_meaning': 'minimizers of empirical loss, not a confidence or credible interval',
            'objective': 'uniform mean fixture 1X2 log loss; no grid or additional loss selection',
            'training_log_loss': optimum[0],
            'endpoint_log_loss': {'llm': _json_number(left[0]), 'statistical': _json_number(right[0])},
            'endpoint_log_loss_is_finite': {'llm': math.isfinite(left[0]), 'statistical': math.isfinite(right[0])},
            'endpoint_gradients': [_json_number(left[1]), _json_number(right[1])],
            'gradient_at_solution': _json_number(optimum[1]),
            'curvature_at_solution': _json_number(optimum[2]),
            'optimizer_function_evals': evaluations, 'method': method,
            'training_results_min_observed_at': history[0]['result_observed_at'].isoformat(),
            'training_results_max_observed_at': history[-1]['result_observed_at'].isoformat(),
            'training_kickoff_max': max(row['kickoff'] for row in history).isoformat(),
            'statistical_family': dict(sorted(Counter(row['statistical_family'] for row in history).items())),
            'llm_family': dict(sorted(Counter(row['llm_family'] for row in history).items())),
            'training_llm_track': next(iter(track_counts)) if len(track_counts) == 1 else 'mixed',
            'training_llm_track_counts': track_counts,
            'training_forecast_times': [{key: row[key].isoformat() if key != 'match_id' else row[key]
                for key in ('match_id', 'kickoff', 'statistical_generated_at', 'llm_generated_at', 'result_observed_at')}
                for row in history],
            'limits': [
                'An empirical fitted weight is not a verified best future allocation or a gain estimate.',
                'Source/exposure and forecast-version provenance must be independently checked by the caller.',
                'Mixed route families and small samples may make the weight unstable or poorly transferable.',
                'Unknown or mixed LLM exposure routes do not identify a matching deployment route.',
                'Result-observation times are supplied claims; they are not reconstructed final-whistle timestamps.',
                'Flat empirical loss does not identify a weight even if unobserved class probabilities differ.',
            ],
            'parameter_grid': False, 'market_used': False,
        }
        return self

    def predict(self, statistical_probabilities, llm_probabilities, *, match_id, kickoff, as_of):
        if not hasattr(self, '_parameters'):
            raise ValueError('Fit on earlier prospective pairs before predicting')
        instant, kickoff = _time(as_of, 'as_of'), _time(kickoff, 'kickoff')
        if not self.fit_before <= instant < kickoff:
            raise ValueError('Prediction must follow weight fitting and precede kickoff')
        if _text(match_id, 'match_id') in self.reference_match_ids:
            raise ValueError('Do not evaluate a fitted weight on its own training fixture')
        stat, llm = _probabilities(statistical_probabilities), _probabilities(llm_probabilities)
        if self.weight is None:
            raise ValueError('Empirical loss does not identify a fusion weight; no arbitrary allocation is supplied')
        if self.weight == 0:
            return llm
        if self.weight == 1:
            return stat
        return _probabilities([math.fsum((self.weight*a, (1-self.weight)*b)) for a,b in zip(stat,llm)])

    def parameters(self):
        if not hasattr(self, '_parameters'):
            raise ValueError('Weight learner is not fitted')
        return deepcopy(self._parameters)
