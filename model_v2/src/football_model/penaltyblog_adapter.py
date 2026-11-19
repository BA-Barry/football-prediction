"""Date-aware adapter to penaltyblog's installed, complete native goal models.

Dates deliberately remain dates: this is not a reconstruction of precise
historical quote/publication times. Native fitted outputs are team-marginal
score distributions and must not be relabelled as conditional lineup rates.
"""

from datetime import date
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import penaltyblog as pb

from .football_data import HistoricalMatch


class NativeFitError(RuntimeError):
    """The native optimizer failed or produced unusable fitted parameters."""


class NativeGoalModel:
    MODEL_TYPES = {
        "poisson": pb.models.PoissonGoalsModel,
        "dixon_coles": pb.models.DixonColesGoalModel,
    }

    def __init__(self, model_name: str = "poisson", half_life_days: float | None = 180):
        if model_name not in self.MODEL_TYPES:
            raise ValueError("model_name must be poisson or dixon_coles")
        if half_life_days is not None and (
            isinstance(half_life_days, bool)
            or not math.isfinite(half_life_days) or half_life_days <= 0
        ):
            raise ValueError("half_life_days must be positive and finite, or None")
        self.model_name = model_name
        self.half_life_days = half_life_days

    def fit(self, records: Sequence[HistoricalMatch], before_date: date):
        if type(before_date) is not date:
            raise ValueError("Use an explicit calendar date, not a fabricated timestamp")
        for record in records:
            if type(record.match_date) is not date:
                raise ValueError("Record match_date must be a calendar date")
        history = sorted((r for r in records if r.match_date < before_date
                          and r.goal_fit_exclusion is None),
                         key=lambda r: (r.match_date, r.match_id))
        if not history:
            raise ValueError("No eligible earlier-date results")
        if len({r.match_id for r in history}) != len(history):
            raise ValueError("Duplicate match identifiers in native training history")
        competitions = {r.competition for r in history}
        if len(competitions) != 1:
            raise ValueError("Native goal models require one league per fit")
        for record in history:
            if (not isinstance(record.full_time, (tuple, list)) or len(record.full_time) != 2
                    or any(type(g) is not int or g < 0 for g in record.full_time)):
                raise ValueError("Goals must be non-negative integers, not truncated floats or booleans")
            if (not isinstance(record.home, str) or not isinstance(record.away, str)
                    or not record.home or not record.away or record.home == record.away):
                raise ValueError("Need two distinct nonempty team identities")
        if self.half_life_days is None:
            weights = np.ones(len(history), dtype=float)
        else:
            weights = pb.models.dixon_coles_weights(
                [r.match_date for r in history],
                xi=math.log(2) / self.half_life_days,
                base_date=before_date,
            )
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("Time weights must remain finite and positive")
        native = self.MODEL_TYPES[self.model_name](
            [r.full_time[0] for r in history],
            [r.full_time[1] for r in history],
            [r.home for r in history],
            [r.away for r in history],
            weights=weights,
        )
        # Native fit returns None; no replacement optimizer, manual rho, or
        # fallback model is used when the original implementation fails.
        try:
            native.fit(minimizer_options={"maxiter": 1000, "disp": False}, use_gradient=True)
        except (ValueError, RuntimeError, FloatingPointError) as error:
            raise NativeFitError(f"penaltyblog {self.model_name} fit failed: {error}") from error
        optimization = getattr(native, "_res", None)
        if not native.fitted or optimization is None or not bool(optimization.success):
            raise NativeFitError(f"penaltyblog {self.model_name} optimizer did not report success")
        params = native.get_params()
        if (not all(math.isfinite(float(value)) for value in params.values())
                or not math.isfinite(float(native.loglikelihood))):
            raise NativeFitError(f"penaltyblog {self.model_name} fitted non-finite parameters")
        self.native = native
        self.competition = next(iter(competitions))
        self.teams = {t for r in history for t in (r.home, r.away)}
        self.before_date = before_date
        self.n_matches = len(history)
        self.training_min_date = history[0].match_date
        self.training_max_date = history[-1].match_date
        self.reference_match_ids = [r.match_id for r in history]
        self.weights = np.asarray(weights).copy()
        self.optimizer_diagnostics = {
            "success": bool(optimization.success),
            "status": int(optimization.status),
            "message": str(optimization.message),
            "iterations": int(optimization.nit),
            "objective": float(optimization.fun),
            "function_evaluations": int(optimization.nfev),
        }
        return self

    def predict(self, home: str, away: str, competition: str, match_date: date):
        if not hasattr(self, "native"):
            raise ValueError("Fit earlier-date results before predicting")
        if type(match_date) is not date or match_date < self.before_date:
            raise ValueError("Prediction must not precede the native fit cutoff date")
        if competition != self.competition:
            raise ValueError("Native model fitted to a different league")
        if home == away:
            raise ValueError("Teams must differ")
        if {home, away} - self.teams:
            raise ValueError("Unseen teams require an explicitly estimated prior")
        raw = self.native.predict(home, away, max_goals=30, normalize=False)
        matrix = np.asarray(raw.grid, dtype=float)
        if not np.isfinite(matrix).all() or np.any(matrix < 0):
            raise ValueError("Native prediction contains non-finite or negative probabilities")
        total = float(matrix.sum())
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"Native prediction omitted material mass or is not normalized: {total}")
        if (not math.isfinite(raw.home_goal_expectation)
                or not math.isfinite(raw.away_goal_expectation)
                or raw.home_goal_expectation <= 0 or raw.away_goal_expectation <= 0):
            raise ValueError("Native prediction has invalid goal expectations")
        return pb.models.FootballProbabilityGrid(
            matrix, float(raw.home_goal_expectation), float(raw.away_goal_expectation), normalize=True)

    def parameters(self):
        if not hasattr(self, "native"):
            raise ValueError("Native model has not been fitted")
        return {
            "model_type": f"penaltyblog_native_{self.model_name}",
            "penaltyblog_version": str(pb.__version__),
            "half_life_days": self.half_life_days,
            "before_date": self.before_date.isoformat(),
            "competition": self.competition,
            "teams": sorted(self.teams),
            "n_matches": self.n_matches,
            "training_min_date": self.training_min_date.isoformat(),
            "training_max_date": self.training_max_date.isoformat(),
            "reference_match_ids": self.reference_match_ids,
            "native_parameters": {str(k): float(v) for k, v in self.native.get_params().items()},
            "optimizer": self.optimizer_diagnostics,
            "basis": "team_marginal",
            "availability_convention": "strictly_earlier_calendar_date_results_assumed_available",
            "native_max_goals_exclusive": 30,
            "prediction_tail_tolerance": 1e-8,
            "persistence_note": "Native pickle; only load self-generated trusted local models",
            "not_for": "precise T-24h, lineup-confirmed, or unvalidated cross-league predictions",
        }

    def save(self, path: str | Path):
        """Save the original native object; only trusted self-generated files may be loaded."""
        if not hasattr(self, "native"):
            raise ValueError("Native model has not been fitted")
        self.native.save(str(Path(path)))
