"""Closed-form Black-Scholes under a continuous dividend yield.

This module is the only source of truth in the repository. The Monte Carlo
engine is judged against it (M5), the control variate's exact expectation comes
from it (M3), and the pathwise and bumped Greeks are scored against its delta
and vega (M6). It is therefore written to be obviously correct rather than
fast, and it has no dependency on anything else in the package.

Conventions, stated once and used everywhere:

* ``sigma`` and ``T`` are annualised; ``r`` and ``q`` are continuously
  compounded annual rates.
* **Vega is per 1.00 of volatility**, not per vol point. A vega of 39.7 means
  the price moves 39.7 if volatility goes from 0.20 to 1.20 — to first order.
  Divide by 100 for the per-point figure desks usually quote. This is the kind
  of unit that has to be said out loud (trap 16).
* Delta and gamma are with respect to spot, not to the forward.

The formulae::

    d1 = (ln(S/K) + (r - q + sigma^2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    Call = S * exp(-qT) * N(d1) - K * exp(-rT) * N(d2)
    Put  = K * exp(-rT) * N(-d2) - S * exp(-qT) * N(-d1)

    Delta_call = exp(-qT) * N(d1)          Delta_put = -exp(-qT) * N(-d1)
    Vega       = S * exp(-qT) * phi(d1) * sqrt(T)
    Gamma      = exp(-qT) * phi(d1) / (S * sigma * sqrt(T))
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.special import ndtr

__all__ = [
    "OptionKind",
    "d1_d2",
    "price",
    "call_price",
    "put_price",
    "delta",
    "vega",
    "gamma",
    "greeks",
    "forward",
    "digital_price",
    "digital_delta",
    "digital_vega",
]

OptionKind = Literal["call", "put"]

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return _INV_SQRT_2PI * np.exp(-0.5 * np.square(x))


def _check_kind(kind: str) -> str:
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    return kind


def _prepare(S, K, r, q, sigma, T) -> tuple[np.ndarray, ...]:
    """Broadcast inputs to a common shape and reject nonsense (R4)."""
    arrays = [np.asarray(x, dtype=np.float64) for x in (S, K, r, q, sigma, T)]
    S, K, r, q, sigma, T = np.broadcast_arrays(*arrays)

    if np.any(~np.isfinite(np.stack([S, K, r, q, sigma, T]))):
        raise ValueError("inputs must all be finite")
    if np.any(S <= 0):
        raise ValueError("spot must be strictly positive")
    if np.any(K <= 0):
        raise ValueError("strike must be strictly positive")
    if np.any(sigma < 0):
        raise ValueError("volatility must be non-negative")
    if np.any(T < 0):
        raise ValueError("time to maturity must be non-negative")
    return S, K, r, q, sigma, T


def _scalarise(result: np.ndarray, *inputs: Any) -> Any:
    """Return a float when every input was a scalar, else the array."""
    if all(np.ndim(x) == 0 for x in inputs):
        return float(result)
    return result


def forward(S, K=None, r=0.0, q=0.0, T=0.0):
    """Forward price ``S * exp((r - q) * T)``. ``K`` is ignored; kept for symmetry."""
    return np.asarray(S, dtype=np.float64) * np.exp(
        (np.asarray(r, dtype=np.float64) - np.asarray(q, dtype=np.float64))
        * np.asarray(T, dtype=np.float64)
    )


def d1_d2(S, K, r, q, sigma, T):
    """Return ``(d1, d2)``.

    Where ``sigma * sqrt(T) == 0`` both are undefined — the distribution of
    ``S_T`` is a point mass — and are returned as NaN rather than as a large
    finite number. Every function below branches on that degenerate case
    explicitly instead of relying on the arithmetic to work out.
    """
    S, K, r, q, sigma, T = _prepare(S, K, r, q, sigma, T)
    vol_sqrt_t = sigma * np.sqrt(T)
    degenerate = vol_sqrt_t <= 0.0
    denom = np.where(degenerate, 1.0, vol_sqrt_t)

    d1 = (np.log(S / K) + (r - q + 0.5 * np.square(sigma)) * T) / denom
    d1 = np.where(degenerate, np.nan, d1)
    d2 = d1 - vol_sqrt_t
    return d1, d2


def _core(S, K, r, q, sigma, T):
    """Shared intermediates: everything below is assembled from these."""
    S, K, r, q, sigma, T = _prepare(S, K, r, q, sigma, T)
    vol_sqrt_t = sigma * np.sqrt(T)
    degenerate = vol_sqrt_t <= 0.0
    denom = np.where(degenerate, 1.0, vol_sqrt_t)

    d1 = (np.log(S / K) + (r - q + 0.5 * np.square(sigma)) * T) / denom
    d2 = d1 - vol_sqrt_t

    df_r = np.exp(-r * T)  # discount factor
    df_q = np.exp(-q * T)  # dividend-yield factor
    return S, K, sigma, T, d1, d2, vol_sqrt_t, degenerate, df_r, df_q


def price(S, K, r, q, sigma, T, kind: OptionKind = "call"):
    """European option price.

    Degenerate limits are returned exactly, not approximated:

    * ``sigma == 0``: the payoff is deterministic, so the value is the
      discounted intrinsic value **of the forward**,
      ``max(S*exp(-qT) - K*exp(-rT), 0)`` for a call.
    * ``T == 0``: the same expression with both factors equal to 1, i.e. the
      spot intrinsic ``max(S - K, 0)``.

    One expression covers both, which is why they are handled by a single
    branch.
    """
    _check_kind(kind)
    S_, K_, _, _, d1, d2, _, degenerate, df_r, df_q = _core(S, K, r, q, sigma, T)

    if kind == "call":
        regular = S_ * df_q * ndtr(d1) - K_ * df_r * ndtr(d2)
        limit = np.maximum(S_ * df_q - K_ * df_r, 0.0)
    else:
        regular = K_ * df_r * ndtr(-d2) - S_ * df_q * ndtr(-d1)
        limit = np.maximum(K_ * df_r - S_ * df_q, 0.0)

    out = np.where(degenerate, limit, regular)
    return _scalarise(out, S, K, r, q, sigma, T)


def call_price(S, K, r, q, sigma, T):
    return price(S, K, r, q, sigma, T, kind="call")


def put_price(S, K, r, q, sigma, T):
    return price(S, K, r, q, sigma, T, kind="put")


def delta(S, K, r, q, sigma, T, kind: OptionKind = "call"):
    """dPrice/dSpot.

    Dimensionless: a call delta lives in ``[0, exp(-qT)]``, so an absolute
    agreement of 1e-4 on delta is a far tighter statement than 1e-4 on vega.
    M6 has to say which quantity a tolerance is absolute in.

    In the degenerate case the payoff is a step function of spot, so delta is
    the indicator that the forward finishes in the money, scaled by
    ``exp(-qT)``. It is genuinely undefined exactly at the money; the
    convention here is that the strict inequality decides it.
    """
    _check_kind(kind)
    S_, K_, _, _, d1, _, _, degenerate, df_r, df_q = _core(S, K, r, q, sigma, T)

    itm = (S_ * df_q - K_ * df_r) > 0.0
    if kind == "call":
        regular = df_q * ndtr(d1)
        limit = np.where(itm, df_q, 0.0)
    else:
        regular = -df_q * ndtr(-d1)
        limit = np.where(itm, 0.0, -df_q)

    out = np.where(degenerate, limit, regular)
    return _scalarise(out, S, K, r, q, sigma, T)


def vega(S, K, r, q, sigma, T):
    """dPrice/dSigma, per 1.00 of volatility. Identical for calls and puts.

    Identical because put-call parity's right-hand side does not contain sigma,
    so differentiating it gives ``vega_call - vega_put = 0``. The same argument
    gives the shared gamma.

    The degenerate case is not uniformly zero, which is easy to get wrong. As
    ``sigma -> 0`` with the forward strictly in or out of the money, ``d1``
    diverges and ``phi(d1)`` vanishes, so vega goes to zero. But exactly at the
    money forward, ``d1 = sigma*sqrt(T)/2 -> 0`` and vega tends to
    ``S*exp(-qT)*phi(0)*sqrt(T)``, which is not zero. That single point is
    handled explicitly.
    """
    S_, K_, _, T_, d1, _, _, degenerate, df_r, df_q = _core(S, K, r, q, sigma, T)

    regular = S_ * df_q * _norm_pdf(d1) * np.sqrt(T_)
    atm_forward = (S_ * df_q - K_ * df_r) == 0.0
    limit = np.where(atm_forward, S_ * df_q * _INV_SQRT_2PI * np.sqrt(T_), 0.0)

    out = np.where(degenerate, limit, regular)
    return _scalarise(out, S, K, r, q, sigma, T)


def gamma(S, K, r, q, sigma, T):
    """d2Price/dSpot2. Identical for calls and puts, by the parity argument in vega.

    In the degenerate case the price is a kink in spot, so the second
    derivative is zero away from the money and a delta function at it. Returned
    as ``inf`` at exactly the money forward: that is the honest limit, and it is
    better that a caller sees an infinity than a large finite number that looks
    usable.
    """
    S_, K_, sigma_, T_, d1, _, vol_sqrt_t, degenerate, df_r, df_q = _core(
        S, K, r, q, sigma, T
    )

    denom = np.where(degenerate, 1.0, S_ * vol_sqrt_t)
    regular = df_q * _norm_pdf(d1) / denom
    atm_forward = (S_ * df_q - K_ * df_r) == 0.0
    limit = np.where(atm_forward, np.inf, 0.0)

    out = np.where(degenerate, limit, regular)
    return _scalarise(out, S, K, r, q, sigma, T)


def greeks(S, K, r, q, sigma, T, kind: OptionKind = "call") -> dict[str, Any]:
    """Price, delta, vega and gamma in one call.

    Convenience only — it recomputes the intermediates per quantity. M6 is
    where the cost of computing sensitivities actually matters, and there the
    comparison is between Monte Carlo estimators, not against this.
    """
    _check_kind(kind)
    return {
        "price": price(S, K, r, q, sigma, T, kind),
        "delta": delta(S, K, r, q, sigma, T, kind),
        "vega": vega(S, K, r, q, sigma, T),
        "gamma": gamma(S, K, r, q, sigma, T),
    }


# --------------------------------------------------------------------------
# Cash-or-nothing digital (added at M6)
# --------------------------------------------------------------------------
# A digital is the sharpest available test case for a Greeks estimator. Its
# payoff is a step function, so it is precisely the situation in which the
# pathwise method is invalid — and unlike a knock-out barrier it still has a
# clean closed form to validate a likelihood-ratio estimator against.


def digital_price(S, K, r, q, sigma, T, kind: OptionKind = "call"):
    """Cash-or-nothing digital paying one unit if it finishes in the money."""
    _check_kind(kind)
    S_, K_, _, _, _, d2, _, degenerate, df_r, df_q = _core(S, K, r, q, sigma, T)
    itm = (S_ * df_q - K_ * df_r) > 0.0

    if kind == "call":
        regular, limit = df_r * ndtr(d2), np.where(itm, df_r, 0.0)
    else:
        regular, limit = df_r * ndtr(-d2), np.where(itm, 0.0, df_r)

    return _scalarise(np.where(degenerate, limit, regular), S, K, r, q, sigma, T)


def digital_delta(S, K, r, q, sigma, T, kind: OptionKind = "call"):
    """dPrice/dSpot for a digital.

    Since ``d(d2)/dS = 1 / (S*sigma*sqrt(T))``, the delta is a scaled normal
    density: sharply peaked at the money and vanishing in both tails. This is
    the quantity M6's likelihood-ratio estimator is validated against, and also
    the quantity the pathwise estimator silently reports as exactly zero.

    In the degenerate case the price is a step in spot, so the derivative is
    zero away from the money forward and infinite at it — the same structure as
    gamma's degenerate limit, and returned as ``inf`` for the same reason.
    """
    _check_kind(kind)
    S_, K_, _, _, _, d2, vol_sqrt_t, degenerate, df_r, df_q = _core(S, K, r, q, sigma, T)

    denom = np.where(degenerate, 1.0, S_ * vol_sqrt_t)
    magnitude = df_r * _norm_pdf(d2) / denom
    regular = magnitude if kind == "call" else -magnitude

    at_forward = (S_ * df_q - K_ * df_r) == 0.0
    limit = np.where(at_forward, np.inf if kind == "call" else -np.inf, 0.0)

    return _scalarise(np.where(degenerate, limit, regular), S, K, r, q, sigma, T)


def digital_vega(S, K, r, q, sigma, T, kind: OptionKind = "call"):
    """dPrice/dSigma for a digital, per 1.00 of volatility.

    Uses the identity ``d(d2)/dsigma = -d1/sigma``. Note the sign: a digital
    call's vega is **negative** when it is in the money, because more
    volatility makes a payoff that is already likely less likely. That sign
    flip is a useful check that a Monte Carlo estimator is producing the right
    quantity rather than merely plausible positive numbers.
    """
    _check_kind(kind)
    _, _, sigma_, _, d1, d2, _, degenerate, df_r, _ = _core(S, K, r, q, sigma, T)
    magnitude = -df_r * _norm_pdf(d2) * d1 / np.where(degenerate, 1.0, sigma_)
    regular = magnitude if kind == "call" else -magnitude
    return _scalarise(np.where(degenerate, 0.0, regular), S, K, r, q, sigma, T)
