"""European payoffs, as functions of simulated prices.

Payoff functions own their own discounting, because only the payoff knows when
it pays. A path-dependent product that pays a coupon at an early observation
date discounts that coupon from that date, not from maturity, and pushing the
discount factor out to the driver would make that impossible to express.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from quantdesk.analytics.blackscholes import OptionKind

__all__ = [
    "european_payoff",
    "arithmetic_asian_payoff",
    "on_terminal_column",
    "linear_in_z_payoff",
]


def european_payoff(
    strike: float,
    rate: float,
    maturity: float,
    kind: OptionKind = "call",
) -> Callable[[np.ndarray], np.ndarray]:
    """Discounted European payoff as a function of terminal price."""
    if strike <= 0:
        raise ValueError(f"strike must be positive, got {strike}")
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    discount = float(np.exp(-rate * maturity))

    def payoff(terminal: np.ndarray) -> np.ndarray:
        terminal = np.asarray(terminal, dtype=np.float64)
        intrinsic = (
            terminal - strike if kind == "call" else strike - terminal
        )
        return discount * np.maximum(intrinsic, 0.0)

    payoff.__doc__ = f"Discounted European {kind}, K={strike}, T={maturity}"
    return payoff


def arithmetic_asian_payoff(
    strike: float,
    rate: float,
    maturity: float,
    kind: OptionKind = "call",
) -> Callable[[np.ndarray], np.ndarray]:
    """Discounted arithmetic-average Asian payoff, as a function of full paths.

    The average runs over the simulated observation columns and **excludes
    column 0**, which is the spot and is known at inception. Averaging a known
    constant into the observation set would damp the payoff's variance for a
    reason that has nothing to do with the contract, and would make the
    variance-reduction study flatter than it should be.

    There is no closed form for the arithmetic average of lognormals, which is
    exactly why this is the target of M3's study: the European call on the same
    paths *does* have one, so it can serve as a control whose expectation is
    known exactly.
    """
    if strike <= 0:
        raise ValueError(f"strike must be positive, got {strike}")
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    discount = float(np.exp(-rate * maturity))

    def payoff(paths: np.ndarray) -> np.ndarray:
        paths = np.asarray(paths, dtype=np.float64)
        if paths.ndim != 2:
            raise ValueError(f"expected paths of shape (n, steps+1), got {paths.shape}")
        if paths.shape[1] < 2:
            raise ValueError("an Asian payoff needs at least one observation after inception")
        average = paths[:, 1:].mean(axis=1)
        intrinsic = average - strike if kind == "call" else strike - average
        return discount * np.maximum(intrinsic, 0.0)

    payoff.__doc__ = f"Discounted arithmetic Asian {kind}, K={strike}, T={maturity}"
    return payoff


def on_terminal_column(
    payoff: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    """Adapt a terminal-price payoff to run on full paths.

    Lets the identical payoff be priced through the one-step engine and through
    the stepped engine, which is how M2 shows the two agree and how M5 prices a
    European through the same code path the structured products use.
    """

    def on_paths(paths: np.ndarray) -> np.ndarray:
        paths = np.asarray(paths, dtype=np.float64)
        if paths.ndim != 2:
            raise ValueError(f"expected paths of shape (n, steps+1), got {paths.shape}")
        return payoff(paths[:, -1])

    return on_paths


def linear_in_z_payoff(market) -> Callable[[np.ndarray], np.ndarray]:
    """``log(S_T / S_0)``, which is exactly linear in the normal draw ``Z``.

    Not a traded product. It exists to test the antithetic estimator, and it is
    the sharpest available test of it. Takes no maturity: the payoff is a pure
    function of the terminal price, and the maturity is already baked into how
    those terminal prices were generated. Accepting one here would suggest it
    changed the answer, which it does not.

    Since ``log(S_T/S_0) = mu*T + sigma*sqrt(T)*Z`` is affine in ``Z``, the
    antithetic pair average is

        (f(Z) + f(-Z)) / 2 = mu * T

    for *every* pair — a constant. The sample variance across pairs is
    therefore exactly zero, so a correctly computed standard error is zero to
    floating-point noise. A standard error wrongly pooled over the 2N
    individual paths is not: it picks up the full variance of the underlying
    draws. The trap is visible rather than a matter of a few percent.
    """

    def payoff(terminal: np.ndarray) -> np.ndarray:
        return np.log(np.asarray(terminal, dtype=np.float64) / market.spot)

    return payoff
