"""Seeds, market parameters, and (from M4) contract terms.

Everything here is a *declared* input, not a calibrated one. Nothing in this
repository is fitted to market prices, so every parameter below is a modelling
assumption the author chose and must be able to defend as such. They are
gathered in one file so that a benchmark's inputs can be recorded verbatim in
its result JSON.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

from quantdesk.products.autocallable import AutocallableTerms as _AutocallableTerms
from quantdesk.products.snowball import SnowballTerms as _SnowballTerms

__all__ = [
    "Market",
    "BASE_MARKET",
    "SEEDS",
    "seed_for",
    "SNOWBALL_TERMS",
    "AUTOCALLABLE_TERMS",
]


# --------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------
# One fixed seed per benchmark, registered here rather than passed on the
# command line, so that "run the benchmark" and "reproduce the reported number"
# are the same action (R5). A benchmark may accept a seed override for
# sensitivity work, but its headline run uses the registered value.
SEEDS: Final[dict[str, int]] = {
    "determinism_selftest": 20260915,
    "variance_reduction": 31415926,
    "closed_form_agreement": 27182818,
}


def seed_for(name: str) -> int:
    """Return the registered seed for a benchmark, or raise.

    Deliberately raises instead of falling back to a default: an unregistered
    benchmark producing numbers from an ad-hoc seed is exactly the kind of
    irreproducible result R5 exists to prevent.
    """
    try:
        return SEEDS[name]
    except KeyError:
        raise KeyError(
            f"no seed registered for benchmark {name!r}; "
            f"add one to SEEDS in config.py. Known: {sorted(SEEDS)}"
        ) from None


# --------------------------------------------------------------------------
# Market parameters
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Market:
    """A flat Black-Scholes world.

    Attributes
    ----------
    spot:
        Underlying price at t=0.
    rate:
        Continuously compounded risk-free rate, annualised.
    div_yield:
        Continuous dividend yield ``q``, annualised.
    vol:
        Lognormal volatility ``sigma``, annualised.

    Flat and constant in all three of rate, yield and vol. That is a real
    limitation, stated here rather than buried: there is no term structure and
    no smile, so nothing in this repository can speak to skew risk.
    """

    spot: float
    rate: float
    div_yield: float
    vol: float

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError(f"spot must be positive, got {self.spot}")
        if self.vol < 0:
            raise ValueError(f"vol must be non-negative, got {self.vol}")

    def as_dict(self) -> dict[str, float]:
        """Inputs in a form the result writer can serialise verbatim."""
        return asdict(self)


# The base case for every pricing benchmark unless a benchmark states otherwise.
# 20% vol and a 2% rate are round numbers chosen for legibility, not calibrated
# to any instrument; spot 100 makes strikes readable as percentage moneyness.
BASE_MARKET: Final[Market] = Market(spot=100.0, rate=0.02, div_yield=0.0, vol=0.20)


# --------------------------------------------------------------------------
# Contract terms
# --------------------------------------------------------------------------
# !! NOT YET CONFIRMED BY THE AUTHOR !!
#
# R7 makes contract terms the author's decision. The author was asked at M4 and
# selected the monitoring convention (daily closes for the knock-in) but did not
# pick a term sheet, so the structure below is the one that was *recommended* to
# them, recorded here rather than assumed silently. It is a conventional
# onshore-China snowball: the shape behind the concentrated dealer losses of
# 2022, which is what §6 question 23 asks about.
#
# Changing it is one edit here. Every number downstream of M4 moves when it
# changes, which is why it lives in config rather than in a benchmark.
#
# Deliberate simplifications, each of which makes this note worth *more* than
# the market equivalent, and each of which must be disclosed when the number is
# quoted:
#   * no lock-up period — real snowballs forbid knocking out for the first
#     three months, which this one permits from month one;
#   * no bid-offer, no funding spread, no dividend basis risk;
#   * flat volatility and flat rates, from BASE_MARKET.

SNOWBALL_TERMS: Final = _SnowballTerms(
    tenor_years=2.0,
    knock_out_level=1.00,  # of the initial fixing, observed monthly
    knock_out_frequency="monthly",
    knock_in_level=0.75,  # of the initial fixing, observed on daily closes
    knock_in_frequency="daily",
    coupon_rate=0.15,  # annual, simple accrual to the redemption date
    strike_level=1.00,  # recovery threshold at maturity
    downside_participation=1.0,  # one-for-one with the index loss
    steps_per_year=252,
)

AUTOCALLABLE_TERMS: Final = _AutocallableTerms(
    tenor_years=SNOWBALL_TERMS.tenor_years,
    knock_out_level=SNOWBALL_TERMS.knock_out_level,
    knock_out_frequency=SNOWBALL_TERMS.knock_out_frequency,
    coupon_rate=SNOWBALL_TERMS.coupon_rate,
    downside_participation=SNOWBALL_TERMS.downside_participation,
    steps_per_year=SNOWBALL_TERMS.steps_per_year,
)
