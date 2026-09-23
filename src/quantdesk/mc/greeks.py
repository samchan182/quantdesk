"""Monte Carlo sensitivities: pathwise, likelihood ratio, and bump-and-revalue.

Three estimators, and the whole point of the module is knowing **which one is
legitimate for which payoff**. Using the wrong one does not raise; it returns a
confident wrong number.

**Pathwise.** Differentiate the payoff along each path and average::

    Delta_pathwise = exp(-rT) * E[ 1{S_T > K} * dS_T/dS_0 ],   dS_T/dS_0 = S_T/S_0
    Vega_pathwise  = exp(-rT) * E[ 1{S_T > K} * dS_T/dsigma ],
                     dS_T/dsigma = S_T * (sqrt(T)*Z - sigma*T)

Cheap and low-variance, and it produces every sensitivity from one simulation.
**Its validity condition is the interview question**: interchanging the
derivative and the expectation requires the payoff to be Lipschitz continuous
in the parameter and differentiable almost everywhere. A vanilla call qualifies
— the kink at the strike is a single point of measure zero, and the payoff is
Lipschitz either side of it. A **digital or a knock-out does not**: the payoff
jumps, so it is not Lipschitz at the barrier, and the interchange is invalid.

What makes that dangerous rather than merely wrong is the failure mode. The
derivative of a step function is zero almost everywhere, so the naive pathwise
estimator for a digital returns **exactly 0.0** — silently, with a standard
error of 0.0, against a true delta of about 0.0196. It does not error. It
reports a precise answer that is entirely wrong, and `pathwise_digital_delta`
exists in this module purely to demonstrate that.

**Likelihood ratio (Route A, the route taken here).** Differentiate the
*density* instead of the payoff. The payoff can then be as discontinuous as it
likes, since it is never differentiated::

    Delta_LR = exp(-rT) * E[ payoff * Z / (S_0*sigma*sqrt(T)) ]
    Vega_LR  = exp(-rT) * E[ payoff * ((Z^2 - 1)/sigma - Z*sqrt(T)) ]

The cost is variance: the score function grows without bound in the tails, so
the likelihood ratio estimator is materially noisier than pathwise wherever
pathwise is legal. That is the trade, and it is why the two are used on
different parts of different products rather than one being adopted everywhere.

**Bump-and-revalue.** Reprice at bumped parameters and difference. Correct for
any payoff, and the slowest. It must be run on **common random numbers** —
identical seed, identical draws — or the difference of two independently noisy
prices is dominated by the noise rather than the signal.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from quantdesk.config import Market
from quantdesk.mc.paths import terminal_prices

__all__ = [
    "GreekEstimate",
    "summarise",
    "pathwise_european",
    "pathwise_digital_delta",
    "likelihood_ratio",
    "bump_samples",
    "central_difference_greek",
]


@dataclass(frozen=True)
class GreekEstimate:
    """An estimate with the standard error that qualifies it.

    The two never travel apart. A Greek quoted without its standard error
    cannot be compared to anything, which is trap 4 applied to sensitivities.
    """

    value: float
    standard_error: float
    n_samples: int
    label: str = ""

    def agrees_with(self, other: float, n_sigma: float = 2.0) -> bool:
        if self.standard_error == 0.0:
            return self.value == other
        return abs(self.value - other) <= n_sigma * self.standard_error

    def as_dict(self) -> dict:
        return {
            "value": self.value,
            "standard_error": self.standard_error,
            "n_samples": self.n_samples,
            "label": self.label,
        }


def summarise(samples: np.ndarray, label: str = "") -> GreekEstimate:
    """Mean and standard error of per-path estimator samples."""
    samples = np.asarray(samples, dtype=np.float64).ravel()
    n = samples.size
    if n < 2:
        raise ValueError(f"need at least 2 samples, got {n}")
    return GreekEstimate(
        value=float(samples.mean()),
        standard_error=float(samples.std(ddof=1) / np.sqrt(n)),
        n_samples=n,
        label=label,
    )


# --------------------------------------------------------------------------
# Pathwise
# --------------------------------------------------------------------------
def pathwise_european(
    z: np.ndarray, market: Market, maturity: float, strike: float, kind: str = "call"
) -> dict[str, np.ndarray]:
    """Per-path price, delta and vega samples for a European, from one set of draws.

    Returns all three from a single pass, which is the efficiency argument for
    the method: bump-and-revalue needs five pricing runs to produce the same
    three quantities.

    Legitimate here because the payoff is Lipschitz in both `S_0` and `sigma`.
    The kink at the strike is differentiable everywhere except a single point,
    which has probability zero under a continuous distribution.
    """
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")

    z = np.asarray(z, dtype=np.float64)
    terminal = terminal_prices(z, market, maturity)
    discount = float(np.exp(-market.rate * maturity))
    root_t = np.sqrt(maturity)

    # dS_T/dS_0 = S_T/S_0 and dS_T/dsigma = S_T*(sqrt(T)*Z - sigma*T), both
    # read straight off S_T = S_0*exp((r-q-sigma^2/2)T + sigma*sqrt(T)*Z).
    dst_dspot = terminal / market.spot
    dst_dvol = terminal * (root_t * z - market.vol * maturity)

    if kind == "call":
        in_the_money = terminal > strike
        intrinsic = np.maximum(terminal - strike, 0.0)
        sign = 1.0
    else:
        in_the_money = terminal < strike
        intrinsic = np.maximum(strike - terminal, 0.0)
        sign = -1.0

    return {
        "price": discount * intrinsic,
        "delta": discount * sign * np.where(in_the_money, dst_dspot, 0.0),
        "vega": discount * sign * np.where(in_the_money, dst_dvol, 0.0),
    }


def pathwise_digital_delta(
    z: np.ndarray, market: Market, maturity: float, strike: float
) -> np.ndarray:
    """The invalid estimator, implemented deliberately so it can be shown failing.

    A digital's payoff is `1{S_T > K}`. Its derivative with respect to `S_T` is
    zero everywhere except at the strike, where it does not exist. Applying the
    pathwise recipe therefore multiplies `dS_T/dS_0` by a derivative that is
    identically zero, and every sample is exactly 0.0.

    This is not an implementation shortcut. It is what the pathwise method
    genuinely produces for a discontinuous payoff, and the reason CLAUDE.md
    warns that the estimator "will silently return a wrong number rather than
    error". Never use it for anything but the demonstration.
    """
    z = np.asarray(z, dtype=np.float64)
    return np.zeros_like(z)


# --------------------------------------------------------------------------
# Likelihood ratio
# --------------------------------------------------------------------------
def likelihood_ratio(
    z: np.ndarray, market: Market, maturity: float, payoff_values: np.ndarray
) -> dict[str, np.ndarray]:
    """Score-function sensitivities. Tolerates any payoff, including jumps.

    Differentiates the log-density of `log S_T ~ N(log S_0 + (r-q-sigma^2/2)T,
    sigma^2 T)` with respect to each parameter:

        d log p / d S_0   = Z / (S_0 * sigma * sqrt(T))
        d log p / d sigma = (Z^2 - 1)/sigma - Z*sqrt(T)

    `payoff_values` must already be discounted, and must come from the same
    draws `z`.

    The variance cost is real and visible: the delta score is linear in `Z` and
    the vega score quadratic, so both grow without bound in the tails while the
    payoff they multiply does not shrink fast enough to compensate.
    """
    z = np.asarray(z, dtype=np.float64)
    payoff_values = np.asarray(payoff_values, dtype=np.float64)
    if z.shape != payoff_values.shape:
        raise ValueError(
            f"payoff values must come from the same draws: {payoff_values.shape} vs {z.shape}"
        )
    if market.vol <= 0 or maturity <= 0:
        raise ValueError("likelihood ratio requires positive volatility and maturity")

    root_t = np.sqrt(maturity)
    score_spot = z / (market.spot * market.vol * root_t)
    score_vol = (np.square(z) - 1.0) / market.vol - z * root_t

    return {
        "delta": payoff_values * score_spot,
        "vega": payoff_values * score_vol,
    }


# --------------------------------------------------------------------------
# Bump and revalue
# --------------------------------------------------------------------------
def bump_samples(
    z: np.ndarray,
    market: Market,
    maturity: float,
    payoff: Callable[[np.ndarray], np.ndarray],
    *,
    parameter: str,
    bump: float,
    z_down: np.ndarray | None = None,
) -> np.ndarray:
    """Central-difference samples for one parameter.

    Parameters
    ----------
    parameter:
        ``"spot"`` or ``"vol"``.
    bump:
        Relative for spot (1e-2 means a 1% move), absolute for volatility.
    z_down:
        Draws for the down leg. Leave as ``None`` for **common random
        numbers** — the same draws on both legs, which is what makes the
        difference measure signal rather than the difference of two noises.
        Passing independent draws here is how the no-CRN comparison is run.

    Returns per-path difference quotients, so the standard error of the
    resulting Greek is the standard error of these samples.
    """
    z = np.asarray(z, dtype=np.float64)
    z_down = z if z_down is None else np.asarray(z_down, dtype=np.float64)

    if parameter == "spot":
        step = market.spot * bump
        up = replace(market, spot=market.spot + step)
        down = replace(market, spot=market.spot - step)
    elif parameter == "vol":
        step = bump
        # Checked before constructing the bumped Market, so the error names the
        # bump rather than surfacing Market's generic non-negativity message.
        if market.vol - step <= 0:
            raise ValueError(
                f"vol bump {bump} drives volatility non-positive "
                f"(sigma = {market.vol}); a central difference needs both legs valid"
            )
        up = replace(market, vol=market.vol + step)
        down = replace(market, vol=market.vol - step)
    else:
        raise ValueError(f"parameter must be 'spot' or 'vol', got {parameter!r}")

    up_values = payoff(terminal_prices(z, up, maturity))
    down_values = payoff(terminal_prices(z_down, down, maturity))
    return (up_values - down_values) / (2.0 * step)


def central_difference_greek(
    z: np.ndarray,
    market: Market,
    maturity: float,
    payoff: Callable[[np.ndarray], np.ndarray],
    *,
    parameter: str,
    bump: float,
    z_down: np.ndarray | None = None,
    label: str = "",
) -> GreekEstimate:
    """`bump_samples` reduced to an estimate and its standard error."""
    samples = bump_samples(
        z, market, maturity, payoff, parameter=parameter, bump=bump, z_down=z_down
    )
    return summarise(samples, label=label or f"bump {parameter} {bump:g}")
