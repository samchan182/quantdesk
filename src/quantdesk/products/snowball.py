"""Snowball note — an autocallable whose downside is conditional on a knock-in.

**The contract in plain language.** The investor buys a two-year note on an
index. Every month the level is checked: if it is at or above the initial
level, the note redeems early and pays back par plus coupon accrued at 15% a
year, pro rata to that date. If it never redeems early, what happens at
maturity depends on whether the index ever closed at or below 75% of its
initial level:

* it never did — the investor receives par plus the full two years of coupon;
* it did, but the index finished at or above the initial level — the investor
  receives par, and no coupon at all;
* it did, and the index finished below the initial level — the investor takes
  the index's loss from the initial level, one for one.

The coupon looks large because the investor has sold a knock-in put. Most of
the time it pays; when it does not, the loss is unbounded to the downside and
the coupon is gone at the same moment.

**The four branches, written out.** `snowball_payoff` computes exactly these,
which is why they are enumerated rather than folded into one expression:

1. knocked out early;
2. never knocked out and never knocked in;
3. knocked in, but recovered to at or above the strike at maturity;
4. knocked in and below the strike at maturity.

**Conventions that change the number, so they are stated.**

* *Knock-out takes precedence.* A path that knocks in and later knocks out pays
  branch 1. The knock-in only matters for paths that survive to maturity.
* *The coupon accrues simple, not compounded*, pro rata to the redemption date.
* *The strike is a recovery threshold, not the redemption basis.* Branch 4
  redeems at the index's performance from the **initial level**, not from the
  strike. With `strike_level = 1.00` the payoff is continuous at the strike:
  branch 3 pays par and branch 4 pays par at `S_T = S_0`.
* *No lock-up.* Real snowballs usually forbid knocking out for the first three
  months. This one does not, and that makes it worth more than the market
  equivalent.
* *Barriers are observed on closing levels*, at the frequencies in the terms —
  daily for the knock-in, monthly for the knock-out. Since the simulation steps
  daily, the daily knock-in observation grid **is** the simulation grid, so the
  simulation of that barrier is exact. M7 turns on this.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from quantdesk.products.autocallable import (
    Frequency,
    observation_indices,
    observation_times,
    redemption_amounts,
)

__all__ = ["SnowballTerms", "snowball_payoff", "fixed_coupon_note_value", "BRANCHES"]

BRANCHES = (
    "knocked_out_early",
    "no_knock_out_no_knock_in",
    "knocked_in_recovered",
    "knocked_in_below_strike",
)


@dataclass(frozen=True)
class SnowballTerms:
    """Snowball terms, with levels quoted as fractions of the initial fixing."""

    tenor_years: float = 2.0
    knock_out_level: float = 1.00
    knock_out_frequency: Frequency = "monthly"
    knock_in_level: float = 0.75
    knock_in_frequency: Frequency = "daily"
    coupon_rate: float = 0.15
    strike_level: float = 1.00
    downside_participation: float = 1.0
    steps_per_year: int = 252

    def __post_init__(self) -> None:
        if self.tenor_years <= 0:
            raise ValueError(f"tenor must be positive, got {self.tenor_years}")
        if self.knock_out_level < 0:
            raise ValueError("knock-out level must be non-negative")
        if self.knock_in_level < 0:
            raise ValueError("knock-in level must be non-negative")
        if self.strike_level < 0:
            raise ValueError("strike level must be non-negative")
        if self.downside_participation < 0:
            raise ValueError("downside participation must be non-negative")
        if self.steps_per_year < 1:
            raise ValueError("steps_per_year must be at least 1")
        if np.isfinite(self.knock_in_level) and np.isfinite(self.knock_out_level):
            if self.knock_in_level > self.knock_out_level:
                raise ValueError(
                    f"knock-in level {self.knock_in_level} is above knock-out level "
                    f"{self.knock_out_level}; the note would knock out before it could "
                    "knock in, which is not a product"
                )

    @property
    def n_steps(self) -> int:
        return int(round(self.tenor_years * self.steps_per_year))

    def as_dict(self) -> dict:
        return asdict(self)


def fixed_coupon_note_value(terms: SnowballTerms, rate: float) -> float:
    """Value of the degenerate note that can neither knock out nor knock in.

    `exp(-r*T) * (1 + c*T)` — the discounted certain redemption. Used by the
    first degenerate acceptance test, and worth having as a function because a
    test that recomputes the formula it is testing proves nothing.
    """
    return float(np.exp(-rate * terms.tenor_years) * (1.0 + terms.coupon_rate * terms.tenor_years))


def snowball_payoff(
    terms: SnowballTerms, spot: float, rate: float, *, return_branches: bool = False
) -> Callable[[np.ndarray], np.ndarray]:
    """Discounted snowball payoff per unit notional, as a function of paths.

    Parameters
    ----------
    return_branches:
        When True the returned callable yields ``(payoff, branch_index)``,
        with branch indices matching :data:`BRANCHES`. Used by the acceptance
        tests to assert that the degenerate cases land in the branch they are
        supposed to, rather than merely producing the right number by luck.
    """
    ko_indices = observation_indices(
        terms.knock_out_frequency, terms.tenor_years, terms.steps_per_year
    )
    ko_times = observation_times(ko_indices, terms.steps_per_year)
    ki_indices = observation_indices(
        terms.knock_in_frequency, terms.tenor_years, terms.steps_per_year
    )

    ko_barrier = terms.knock_out_level * spot
    ki_barrier = terms.knock_in_level * spot
    strike = terms.strike_level * spot
    tenor = terms.tenor_years
    participation = terms.downside_participation
    full_coupon = 1.0 + terms.coupon_rate * tenor

    def evaluate(paths: np.ndarray):
        paths = np.asarray(paths, dtype=np.float64)
        if paths.ndim != 2 or paths.shape[1] != terms.n_steps + 1:
            raise ValueError(
                f"expected paths of shape (n, {terms.n_steps + 1}), got {paths.shape}"
            )

        knocked_out, redemption_time, ko_coupon = redemption_amounts(
            paths, spot,
            ko_indices=ko_indices, ko_times=ko_times, ko_barrier=ko_barrier,
            tenor_years=tenor, coupon_rate=terms.coupon_rate,
        )
        # A knock-in level of +inf means "always knocked in"; of 0, "never".
        knocked_in = (paths[:, ki_indices] <= ki_barrier).any(axis=1)

        terminal = paths[:, -1]
        recovered = terminal >= strike
        performance = terminal / spot

        alive = ~knocked_out
        branch = np.empty(paths.shape[0], dtype=np.int8)
        branch[knocked_out] = 0
        branch[alive & ~knocked_in] = 1
        branch[alive & knocked_in & recovered] = 2
        branch[alive & knocked_in & ~recovered] = 3

        amount = np.select(
            [branch == 0, branch == 1, branch == 2],
            [ko_coupon, full_coupon, 1.0],
            default=1.0 - participation * (1.0 - performance),
        )
        payoff = np.exp(-rate * redemption_time) * amount
        return (payoff, branch) if return_branches else payoff

    return evaluate
