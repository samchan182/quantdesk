"""Autocallable note: early redemption on an up-barrier, unconditional downside.

The observation-schedule machinery lives here and is shared with the snowball.

**The relationship between the two products, which is the point of having both.**
An autocallable pays an accruing coupon, redeems early if the underlying is at
or above the knock-out level on an observation date, and if it never redeems
early the investor takes the index's downside at maturity. The investor is, in
effect, short a put.

A snowball is the same note with that put made *conditional*: the downside
applies only if a lower knock-in barrier was breached at some point. A
conditional put is worth less than an unconditional one, so the issuer can pay a
higher coupon for it — which is exactly why snowball coupons look attractive,
and exactly where the risk is buried.

That relationship is not just narrative. `snowball_payoff` with the knock-in
level at infinity — always knocked in — must equal `autocallable_payoff` path
for path, and a test asserts it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Literal

import numpy as np

__all__ = [
    "Frequency",
    "AutocallableTerms",
    "observation_indices",
    "observation_times",
    "autocallable_payoff",
]

Frequency = Literal["daily", "monthly", "quarterly"]

_PER_YEAR: dict[str, int] = {"daily": 0, "monthly": 12, "quarterly": 4}


def observation_indices(
    frequency: Frequency, tenor_years: float, steps_per_year: int
) -> np.ndarray:
    """Column indices into a path array at which a barrier is observed.

    Column 0 is inception and is never an observation date: a barrier that is
    already breached at t=0 is a term-sheet error, not a market event.

    Daily observation means every simulated step, so the simulation grid and
    the contract's observation grid coincide — which is the fact M7 turns on.
    Monthly and quarterly dates are placed on the nearest simulated step.
    """
    n_steps = int(round(tenor_years * steps_per_year))
    if n_steps < 1:
        raise ValueError(f"tenor {tenor_years} at {steps_per_year}/yr gives no steps")

    if frequency == "daily":
        return np.arange(1, n_steps + 1)

    if frequency not in _PER_YEAR:
        raise ValueError(f"unknown frequency {frequency!r}; expected one of {sorted(_PER_YEAR)}")

    per_year = _PER_YEAR[frequency]
    n_obs = int(round(tenor_years * per_year))
    if n_obs < 1:
        raise ValueError(f"tenor {tenor_years} gives no {frequency} observations")

    steps = np.rint(np.arange(1, n_obs + 1) * (n_steps / n_obs)).astype(int)
    steps = np.clip(steps, 1, n_steps)
    if len(np.unique(steps)) != len(steps):
        raise ValueError(
            f"{frequency} observations collide on the {steps_per_year}/yr grid; "
            "use more steps per year"
        )
    return steps


def observation_times(indices: np.ndarray, steps_per_year: int) -> np.ndarray:
    """Observation dates in years, from column indices."""
    return np.asarray(indices, dtype=np.float64) / float(steps_per_year)


@dataclass(frozen=True)
class AutocallableTerms:
    """Terms of an autocallable note, as fractions of the initial level.

    Levels are quoted relative to the initial fixing, which is how term sheets
    quote them: `knock_out_level = 1.00` means "at or above the initial level".
    """

    tenor_years: float = 2.0
    knock_out_level: float = 1.00
    knock_out_frequency: Frequency = "monthly"
    coupon_rate: float = 0.15
    downside_participation: float = 1.0
    steps_per_year: int = 252

    def __post_init__(self) -> None:
        if self.tenor_years <= 0:
            raise ValueError(f"tenor must be positive, got {self.tenor_years}")
        if self.knock_out_level < 0:
            raise ValueError(f"knock-out level must be non-negative, got {self.knock_out_level}")
        if self.downside_participation < 0:
            raise ValueError("downside participation must be non-negative")
        if self.steps_per_year < 1:
            raise ValueError("steps_per_year must be at least 1")

    @property
    def n_steps(self) -> int:
        return int(round(self.tenor_years * self.steps_per_year))

    def as_dict(self) -> dict:
        return asdict(self)


def redemption_amounts(
    paths: np.ndarray,
    spot: float,
    *,
    ko_indices: np.ndarray,
    ko_times: np.ndarray,
    ko_barrier: float,
    tenor_years: float,
    coupon_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve the early-redemption leg shared by both products.

    Returns ``(knocked_out, redemption_time, coupon_payment)`` where
    ``coupon_payment`` is the undiscounted amount paid at ``redemption_time``
    for the paths that knock out, and the maturity amount for those that do not.

    The coupon accrues simple, pro rata to the redemption date: a note that
    redeems after 7 months at a 15% rate pays `1 + 0.15 * 7/12`. Simple rather
    than compounded, which is the market convention for these notes and is
    stated here because it changes the number.
    """
    observed = paths[:, ko_indices]
    above = observed >= ko_barrier
    knocked_out = above.any(axis=1)

    # argmax on a boolean row returns the first True, or 0 if there are none —
    # which is why the result is only consulted where knocked_out is True.
    first = np.argmax(above, axis=1)
    redemption_time = np.where(knocked_out, ko_times[first], tenor_years)
    coupon_payment = 1.0 + coupon_rate * redemption_time
    return knocked_out, redemption_time, coupon_payment


def autocallable_payoff(
    terms: AutocallableTerms, spot: float, rate: float
) -> Callable[[np.ndarray], np.ndarray]:
    """Discounted autocallable payoff per unit notional, as a function of paths.

    Two branches:

    1. **Knocked out** at observation `t_k`: pay `1 + c*t_k`, discounted from
       `t_k`.
    2. **Survived to maturity**: the investor is short an unconditional put on
       the index performance, so pay `1 - p*(1 - S_T/S_0)` capped at par,
       discounted from `T`.
    """
    ko_indices = observation_indices(
        terms.knock_out_frequency, terms.tenor_years, terms.steps_per_year
    )
    ko_times = observation_times(ko_indices, terms.steps_per_year)
    ko_barrier = terms.knock_out_level * spot
    tenor = terms.tenor_years
    participation = terms.downside_participation

    def payoff(paths: np.ndarray) -> np.ndarray:
        paths = np.asarray(paths, dtype=np.float64)
        if paths.ndim != 2 or paths.shape[1] != terms.n_steps + 1:
            raise ValueError(
                f"expected paths of shape (n, {terms.n_steps + 1}), got {paths.shape}"
            )

        knocked_out, redemption_time, coupon = redemption_amounts(
            paths, spot,
            ko_indices=ko_indices, ko_times=ko_times, ko_barrier=ko_barrier,
            tenor_years=tenor, coupon_rate=terms.coupon_rate,
        )

        performance = paths[:, -1] / spot
        downside = 1.0 - participation * (1.0 - performance)
        amount = np.where(knocked_out, coupon, np.minimum(downside, 1.0))
        return np.exp(-rate * redemption_time) * amount

    return payoff
