"""Alternative margin allocations for an ordered three-outcome odds quote.

This is a small adapter around the installed penaltyblog implementation, not a
new team model or a claim to recover true probabilities.  The caller must keep
the bookmaker/source, fixture, settlement rules and actual availability time
with the input quote.  This module neither assigns an as-of time nor checks
that a historical price was available before a prediction.
"""

from importlib.metadata import version
import math
from numbers import Real
from typing import Sequence

from penaltyblog import implied as penaltyblog_implied


IMPLIED_METHODS = ("multiplicative", "power", "shin")


def dewater_1x2(
    odds: Sequence[float],
    method: str = "multiplicative",
    *,
    labels: Sequence[str] = ("home", "draw", "away"),
) -> dict:
    """Preserve quote order and return one explicitly named margin assumption.

    Odds must be three finite decimal prices greater than one.  Labels describe
    the same order; they are never sorted.  ``overround`` is sum(1/odds)-1,
    not a measured bookmaker profit or a model uncertainty estimate.

    No outcome is clipped and no returned distribution is renormalized.  A
    library solver failure, invalid result or invalid method-specific parameter
    raises ValueError with the method name.  In particular, Shin need not be
    defined for an underround quote, and power's finite solver interval can fail
    for extreme prices.  The caller may record the failure, but must not silently
    substitute another method or drop its match from a common comparison.

    The installed dependency is declared as penaltyblog==1.13.1 by the project;
    actual package version is returned as provenance, not enforced here.
    """
    if not isinstance(method, str) or method not in IMPLIED_METHODS:
        raise ValueError(f"Unsupported 1X2 margin method: {method!r}")
    if isinstance(odds, (str, bytes)):
        raise ValueError(f"{method}: odds must be three decimal numbers")
    try:
        quote = tuple(odds)
    except TypeError as exc:
        raise ValueError(f"{method}: odds must be three decimal numbers") from exc
    if len(quote) != 3 or any(
        isinstance(x, bool) or not isinstance(x, Real)
        or not math.isfinite(x) or x <= 1
        for x in quote
    ):
        raise ValueError(f"{method}: require three finite decimal odds greater than 1")
    prices = [float(x) for x in quote]

    if isinstance(labels, (str, bytes)):
        raise ValueError(f"{method}: labels must be three distinct nonempty strings")
    try:
        names = list(labels)
    except TypeError as exc:
        raise ValueError(f"{method}: labels must be three distinct nonempty strings") from exc
    if (len(names) != 3 or any(not isinstance(x, str) or not x.strip() for x in names)
            or len(set(names)) != 3):
        raise ValueError(f"{method}: labels must be three distinct nonempty strings")

    raw_implied = [1 / x for x in prices]
    # Use the same three-term NumPy summation as the old market implementation
    # through the library; fsum below only measures consistency of its metadata.
    overround = math.fsum(raw_implied) - 1
    try:
        result = penaltyblog_implied.calculate_implied(
            prices, method=method, market_names=names,
        )
    except (ValueError, RuntimeError, FloatingPointError, OverflowError,
            ZeroDivisionError) as exc:
        raise ValueError(f"{method}: penaltyblog margin calculation failed: {exc}") from exc

    try:
        probabilities = [float(x) for x in result.probabilities]
        returned_method = result.method.value
        returned_margin = float(result.margin)
        returned_names = list(result.market_names)
        parameters = {
            str(key): float(value)
            for key, value in (result.method_params or {}).items()
        }
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{method}: malformed penaltyblog result") from exc
    if returned_method != method or returned_names != names:
        raise ValueError(f"{method}: library changed method or outcome order")
    if (len(probabilities) != 3 or any(
            not math.isfinite(x) or not 0 <= x <= 1 for x in probabilities)
            or not math.isclose(math.fsum(probabilities), 1, rel_tol=0, abs_tol=1e-9)):
        raise ValueError(f"{method}: library returned invalid or unnormalized probabilities")
    if (not math.isfinite(returned_margin)
            or not math.isclose(returned_margin, overround, rel_tol=0, abs_tol=1e-12)):
        raise ValueError(f"{method}: library overround disagrees with original quote")
    if any(not math.isfinite(value) for value in parameters.values()):
        raise ValueError(f"{method}: library returned non-finite method parameters")
    if method == "power" and ("k" not in parameters or parameters["k"] <= 0):
        raise ValueError(f"{method}: power exponent k must be finite and positive")
    if method == "shin" and ("z" not in parameters or not 0 <= parameters["z"] < 1):
        raise ValueError(f"{method}: Shin z must be in [0, 1)")

    return {
        "odds": prices,
        "labels": names,
        "raw_implied": raw_implied,
        "overround": overround,
        "method": method,
        "method_params": parameters,
        "probabilities": probabilities,
        "probabilities_by_label": dict(zip(names, probabilities)),
        "implementation": {
            "package": "penaltyblog",
            "version": version("penaltyblog"),
            "api": "penaltyblog.implied.calculate_implied",
        },
    }
