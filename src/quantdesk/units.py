"""Basis-point conventions, in one place.

Trap 16 is "basis points quoted without stating basis points *of what*". The
defence is to make the base an argument that cannot be omitted, and to give the
two bases different function names so a call site says which it meant.

The distinction is not pedantic. For an at-the-money one-year call at 20% vol
on a spot of 100, the premium is about 8.92 against a notional of 100. The same
absolute price difference is therefore **eleven times larger** in basis points
of premium than in basis points of notional. A pricing error of "3 basis
points" is either 0.0027 or 0.03 depending on which was meant, and only one of
those is worth mentioning.

This repository quotes:

* **pricing** differences (M5, M7) in basis points **of the premium**, because
  the question is how accurate the price is;
* **execution** shortfall (M9) in basis points **of the arrival price**,
  because the question is how much the trade cost against a benchmark.
"""

from __future__ import annotations

import numpy as np

__all__ = ["bp_of_premium", "bp_of_notional", "bp_of", "BASIS_POINT"]

BASIS_POINT = 1e-4


def bp_of(difference, base):
    """Express ``difference`` in basis points of ``base``.

    Raises on a zero base rather than returning infinity: a basis-point figure
    against nothing is not a number, and returning `inf` would let it travel
    into a report.
    """
    difference = np.asarray(difference, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    if np.any(base == 0.0):
        raise ValueError("cannot express a difference in basis points of zero")
    result = difference / base / BASIS_POINT
    return float(result) if result.ndim == 0 else result


def bp_of_premium(difference, premium):
    """Basis points of the option premium. The convention for pricing accuracy."""
    return bp_of(difference, premium)


def bp_of_notional(difference, notional):
    """Basis points of notional. Not used for pricing differences here."""
    return bp_of(difference, notional)
