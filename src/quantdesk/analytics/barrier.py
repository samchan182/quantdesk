"""Continuous-barrier closed forms, and the Broadie-Glasserman-Kou correction.

Two independent references for M7, which exists to compare them.

**Continuous barrier options** (Merton; Reiner and Rubinstein) have closed
forms under GBM when the barrier is monitored *continuously*. Real contracts
almost never are — they monitor on closes — so these formulas are a reference
point rather than a price, and knowing which of the two situations you are in
is the whole of M7.

**The continuity correction.** Broadie, Glasserman and Kou showed that a
discretely monitored barrier option is closely approximated by a *continuously*
monitored one whose barrier has been shifted **away from the spot** by

    B_shifted = B * exp(+/- beta * sigma * sqrt(dt)),    beta ~ 0.5826

plus for an upper barrier, minus for a lower one. The constant is
`-zeta(1/2)/sqrt(2*pi)`, and the intuition is that a discretely monitored
barrier is *effectively further away*: the price can cross it and return
between observations without being seen, so the contract behaves as though the
barrier sat past where it is written.

That single fact fixes the sign of everything in M7. Discrete monitoring sees
fewer crossings, so more paths survive, so a knock-out is worth **more** when
monitored discretely than continuously.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.special import ndtr

from quantdesk.analytics.blackscholes import _norm_pdf  # noqa: F401  (kept for parity of imports)
from quantdesk.analytics.blackscholes import call_price

__all__ = [
    "BETA_BGK",
    "Direction",
    "shifted_barrier",
    "down_and_out_call",
    "down_and_in_call",
]

#: -zeta(1/2)/sqrt(2*pi), the Broadie-Glasserman-Kou continuity-correction constant.
BETA_BGK = 0.5825971579390106

Direction = Literal["down", "up"]


def shifted_barrier(
    barrier: float, vol: float, dt: float, direction: Direction = "down"
) -> float:
    """The Broadie-Glasserman-Kou effective barrier for discrete monitoring.

    Shifted **away from the spot**: down for a lower barrier, up for an upper
    one. Getting this direction backwards produces a correction of the right
    magnitude and the wrong sign, which is trap 9.
    """
    if dt < 0:
        raise ValueError(f"dt must be non-negative, got {dt}")
    if direction == "down":
        return float(barrier * np.exp(-BETA_BGK * vol * np.sqrt(dt)))
    if direction == "up":
        return float(barrier * np.exp(+BETA_BGK * vol * np.sqrt(dt)))
    raise ValueError(f"direction must be 'down' or 'up', got {direction!r}")


def _lambda(r: float, q: float, vol: float) -> float:
    return (r - q + 0.5 * vol**2) / vol**2


def down_and_out_call(S, K, H, r, q, sigma, T):
    """Continuously monitored down-and-out call, in closed form.

    Two regimes, because the barrier can sit either side of the strike:

    * ``H <= K`` — the knock-in is worthless above the barrier, and the
      standard reflection argument gives the in-option directly.
    * ``H > K`` — paths finishing between the barrier and the strike matter,
      and the formula carries four terms rather than two.

    Both are covered; a formula written for only one regime silently produces
    nonsense in the other, which is why the split is explicit.
    """
    S, K, H, r, q, sigma, T = (float(x) for x in (S, K, H, r, q, sigma, T))
    if H <= 0 or S <= 0 or K <= 0:
        raise ValueError("spot, strike and barrier must be positive")
    if sigma <= 0 or T <= 0:
        raise ValueError("down_and_out_call needs positive volatility and maturity")
    if S <= H:
        return 0.0  # already knocked out at inception

    lam = _lambda(r, q, sigma)
    vol_root_t = sigma * np.sqrt(T)
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    ratio = H / S

    vanilla = call_price(S, K, r, q, sigma, T)

    if H <= K:
        y = np.log(H * H / (S * K)) / vol_root_t + lam * vol_root_t
        knock_in = S * df_q * ratio ** (2 * lam) * ndtr(y) - K * df_r * ratio ** (
            2 * lam - 2
        ) * ndtr(y - vol_root_t)
        return float(vanilla - knock_in)

    x1 = np.log(S / H) / vol_root_t + lam * vol_root_t
    y1 = np.log(H / S) / vol_root_t + lam * vol_root_t
    out = (
        S * df_q * ndtr(x1)
        - K * df_r * ndtr(x1 - vol_root_t)
        - S * df_q * ratio ** (2 * lam) * ndtr(y1)
        + K * df_r * ratio ** (2 * lam - 2) * ndtr(y1 - vol_root_t)
    )
    return float(out)


def down_and_in_call(S, K, H, r, q, sigma, T):
    """Continuously monitored down-and-in call. In-out parity: in + out = vanilla."""
    return float(call_price(S, K, r, q, sigma, T) - down_and_out_call(S, K, H, r, q, sigma, T))
