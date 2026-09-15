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

__all__ = ["Market", "BASE_MARKET", "SEEDS", "seed_for"]


# --------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------
# One fixed seed per benchmark, registered here rather than passed on the
# command line, so that "run the benchmark" and "reproduce the reported number"
# are the same action (R5). A benchmark may accept a seed override for
# sensitivity work, but its headline run uses the registered value.
SEEDS: Final[dict[str, int]] = {
    "determinism_selftest": 20260915,
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
# Deliberately absent until M4. The autocallable and snowball terms are the
# author's decision (R7) and inventing a product to make the code run would
# violate R4. M4 adds them here once confirmed.
