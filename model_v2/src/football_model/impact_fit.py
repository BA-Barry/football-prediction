"""Experimental, time-aware player attack/defence log-rate fitting.

This is an original regularised Poisson implementation, not a conversion of
additive RAPM xG/90 ratings.  Both teams' player deviations enter each stint's
design together.  Positive attack increases own scoring intensity; positive
defence reduces the opponent's.  Deviations are relative to the reference
lineup represented by the supplied pre-match conditional-reference intensity,
rather than adding an entire average lineup again. Centering alone does not
calibrate a marginal team rate into a conditional rate: exp nonlinearity can
still shift the marginal mean. Joint fitting or training-window conditional
normalization is required and is not implemented here. References and offsets must have been
available by kickoff; callers must also ensure their underlying data really
respect those timestamps.  A timestamp alone cannot prove data provenance.

For an actual constant-lineup stint, player fractions are 1 for those on the
pitch, not stint_minutes / 90: the exposure already handles stint duration.
Reference fractions are historical/expected full-match minutes / 90.  A
substitute therefore replaces a specific player's fraction rather than adding
an arbitrary injury penalty.  Pre-match applications must supply projected
minutes, not future actual substitutions, and use zero pre-match score/man
difference or simulate their evolution separately.

With goals, this is a Poisson likelihood.  With nonnegative xG targets, the
same objective is a quasi-likelihood for the conditional mean, not an integer
goal distribution or a calibrated uncertainty model.  Ridge coefficients are
conditional associations, not causal effects.  Persistent teammate pairing,
weak opponent connectivity, red-card selection, bad team offsets and sparse
minutes remain limitations; goalkeepers' shot stopping is not identified by
xG allowed.  No real player data have been trained or validated here.  Ridge
strength needs time-forward selection outside this module; no confidence
intervals or claimed predictive performance are produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite, log
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import minimize


@dataclass
class StintObservation:
    """One non-overlapping, constant-lineup match interval.

    ``base_*_lambda`` are calibrated full-90 conditional-reference scoring
    expectations before this lineup's deviations, already including
    team/opponent/home strength. Raw marginal team-model rates do not suffice.
    ``home_target``/``away_target`` are goals or summed xG *in this stint*.
    Score and man differences are at the start, never the end, of the stint.
    All IDs must be stable across clubs and observations.  Team reference
    player fractions must correspond to the reference behind the offsets.
    """

    match_id: str
    kickoff: datetime
    result_available_at: datetime
    offset_asof: datetime
    reference_asof: datetime
    duration_minutes: float
    home_players: Mapping[str, float]
    away_players: Mapping[str, float]
    home_reference: Mapping[str, float]
    away_reference: Mapping[str, float]
    base_home_lambda: float
    base_away_lambda: float
    home_target: float
    away_target: float
    home_score_difference: float = 0.0
    home_man_difference: float = 0.0


@dataclass
class FittedPlayerEffects:
    """Ridge-shrunk log-rate effects; not xG/90 or causal player values."""

    effects: dict[str, tuple[float, float]]
    context_effects: dict[str, float]
    training_cutoff: datetime
    response_kind: str
    n_stints: int
    n_matches: int
    ridge: float
    context_ridge: float
    objective: float
    iterations: int
    training_note: str = (
        "Experimental conditional associations. No formal real-data validation "
        "or confidence intervals; unknown players require an explicit prior."
    )


def _utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _fractions(values: Mapping[str, float], name: str) -> None:
    for player, value in values.items():
        if not isinstance(player, str) or not player:
            raise ValueError(f"{name} requires nonempty stable string player IDs")
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}[{player!r}] must be a finite fraction in [0, 1]")


def _validate(obs: StintObservation, cutoff: datetime, response_kind: str) -> None:
    kickoff = _utc(obs.kickoff, "kickoff")
    available = _utc(obs.result_available_at, "result_available_at")
    if kickoff >= cutoff or available >= cutoff:
        raise ValueError(f"{obs.match_id}: result must be available strictly before cutoff")
    if available < kickoff:
        raise ValueError(f"{obs.match_id}: result_available_at precedes kickoff")
    for name in ("offset_asof", "reference_asof"):
        if _utc(getattr(obs, name), name) > kickoff:
            raise ValueError(f"{obs.match_id}: {name} must not be later than kickoff")
    for name in ("duration_minutes", "base_home_lambda", "base_away_lambda"):
        value = getattr(obs, name)
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{obs.match_id}: {name} must be finite and positive")
    for name in ("home_target", "away_target"):
        value = getattr(obs, name)
        if not isfinite(value) or value < 0:
            raise ValueError(f"{obs.match_id}: {name} must be finite and nonnegative")
        if response_kind == "goals" and value != int(value):
            raise ValueError("fractional targets require response_kind='xg'")
    for name in ("home_score_difference", "home_man_difference"):
        if not isfinite(getattr(obs, name)):
            raise ValueError(f"{obs.match_id}: {name} must be finite")
    for name in ("home_players", "away_players", "home_reference", "away_reference"):
        _fractions(getattr(obs, name), name)
    if set(obs.home_players) & set(obs.away_players):
        raise ValueError(f"{obs.match_id}: a player cannot be on both teams in one stint")


def fit_player_effects(
    observations: Sequence[StintObservation],
    *,
    cutoff: datetime,
    ridge: float = 10.0,
    context_ridge: float = 1.0,
    response_kind: str = "goals",
    max_iterations: int = 1000,
) -> FittedPlayerEffects:
    """Fit sparse joint attack/defence effects with L-BFGS.

    For home scoring, ``log(mean) = log(duration/90 * base_home_lambda)
    + sum(home_fraction-reference)*attack
    - sum(away_fraction-reference)*defence + score/man controls``.
    Away scoring reverses the teams and the context signs.  The summed loss
    uses ``ridge/2 * ||player_coefficients||^2`` (not a sample-averaged loss),
    so sample size matters when selecting ridge.  All input outcomes must be
    strictly before cutoff: future rows are rejected, not silently included.
    No automatic tuning, downloading, uncertainty estimation or causal
    identification is performed.  Overlapping/duplicated stints must be
    removed upstream because no period/event identifiers are inferred here.
    """

    cutoff = _utc(cutoff, "cutoff")
    if response_kind not in {"goals", "xg"}:
        raise ValueError("response_kind must be 'goals' or 'xg'")
    if not observations:
        raise ValueError("at least one stint is required")
    for name, value in (("ridge", ridge), ("context_ridge", context_ridge)):
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    for obs in observations:
        _validate(obs, cutoff, response_kind)

    players = sorted({
        player
        for obs in observations
        for lineup in (obs.home_players, obs.away_players, obs.home_reference, obs.away_reference)
        for player in lineup
    })
    if not players:
        raise ValueError("at least one identified player is required")
    player_index = {player: index for index, player in enumerate(players)}
    n_players = len(players)
    n_rows = 2 * len(observations)
    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []
    offsets = np.empty(n_rows)
    targets = np.empty(n_rows)

    def add(row: int, column: int, value: float) -> None:
        if value != 0.0:
            row_indices.append(row)
            col_indices.append(column)
            values.append(value)

    for index, obs in enumerate(observations):
        h, a = 2 * index, 2 * index + 1
        for current, reference, attack_row, defence_row in (
            (obs.home_players, obs.home_reference, h, a),
            (obs.away_players, obs.away_reference, a, h),
        ):
            for player in set(current) | set(reference):
                delta = current.get(player, 0.0) - reference.get(player, 0.0)
                column = player_index[player]
                add(attack_row, column, delta)
                add(defence_row, n_players + column, -delta)
        for row, sign in ((h, 1.0), (a, -1.0)):
            add(row, 2 * n_players, sign * obs.home_score_difference)
            add(row, 2 * n_players + 1, sign * obs.home_man_difference)
        log_exposure = log(obs.duration_minutes / 90.0)
        offsets[h] = log_exposure + log(obs.base_home_lambda)
        offsets[a] = log_exposure + log(obs.base_away_lambda)
        targets[h], targets[a] = obs.home_target, obs.away_target

    design = sparse.csr_matrix(
        (values, (row_indices, col_indices)), shape=(n_rows, 2 * n_players + 2)
    )
    penalties = np.full(design.shape[1], ridge, dtype=float)
    penalties[-2:] = context_ridge

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        eta = offsets + design @ coefficients
        # No hidden clipping/bounds: overflow is an optimisation failure, not
        # a silently capped effect or an imposed football probability.
        with np.errstate(over="ignore", invalid="ignore"):
            means = np.exp(eta)
            loss = np.sum(means - targets * eta) + 0.5 * np.dot(
                penalties * coefficients, coefficients
            )
            gradient = np.asarray(design.T @ (means - targets)).ravel()
            gradient += penalties * coefficients
        return float(loss), gradient

    result = minimize(
        objective,
        np.zeros(design.shape[1]),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iterations, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success or not np.all(np.isfinite(result.x)) or not isfinite(result.fun):
        raise RuntimeError(f"player-effect fit did not converge: {result.message}")
    coefficients = result.x
    return FittedPlayerEffects(
        effects={
            player: (float(coefficients[index]), float(coefficients[n_players + index]))
            for player, index in player_index.items()
        },
        context_effects={
            "own_score_difference": float(coefficients[-2]),
            "own_man_difference": float(coefficients[-1]),
        },
        training_cutoff=cutoff,
        response_kind=response_kind,
        n_stints=len(observations),
        n_matches=len({obs.match_id for obs in observations}),
        ridge=ridge,
        context_ridge=context_ridge,
        objective=float(result.fun),
        iterations=int(result.nit),
    )
