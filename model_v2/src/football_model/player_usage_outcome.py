"""Existing direct-outcome learner with a larger numerical iteration allowance.

The objective, C, tolerance, preprocessing and history rules are inherited
unchanged. This importable class also makes saved research models reusable
outside the one-off runner's __main__ namespace.
"""
from .early_outcome_learners import EarlyOutcomeLearner


class PlayerUsageOutcomeLearner(EarlyOutcomeLearner):
    def _estimator(self):
        return super()._estimator().set_params(max_iter=4000)
