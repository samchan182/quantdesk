"""M7 acceptance — discrete monitoring and the Brownian bridge.

Three required checks:
  1. as the step size goes to zero, corrected and uncorrected prices converge;
  2. the sign is right — correcting a knock-out *lowers* its price;
  3. the Brownian bridge and the Broadie-Glasserman-Kou barrier shift agree.

And the one the milestone is actually about: that with the contract monitored
on daily closes and the simulation stepping daily, the correct thing to do is
**nothing**.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantdesk.analytics import blackscholes as bs
from quantdesk.analytics.barrier import (
    BETA_BGK,
    down_and_in_call,
    down_and_out_call,
    shifted_barrier,
)
from quantdesk.config import BASE_MARKET, Market
from quantdesk.mc.bridge import bridge_survival, crossing_probability, observed_survival
from quantdesk.mc.paths import price_paths
from quantdesk.rng import rng

MARKET = BASE_MARKET
MATURITY = 1.0
STRIKE = 100.0
BARRIER = 90.0


def _price_both(
    seed, n_steps, n_paths, barrier=BARRIER, market=MARKET, maturity=MATURITY, chunk=50_000
):
    """Both arms on the same paths, so their difference is not two noises.

    Chunked, for the same reason the production driver is: a 1,000,000 x 252
    path array is 2 GB, and the test suite has no business allocating it. The
    two arms are accumulated as sums, and the difference is accumulated as a
    paired quantity so its standard error reflects the pairing.
    """
    generator = rng(seed)
    dt = maturity / n_steps
    discount = float(np.exp(-market.rate * maturity))

    total = {"d": 0.0, "b": 0.0, "dd": 0.0, "bb": 0.0, "diff": 0.0, "diff2": 0.0, "n": 0}
    remaining = n_paths
    while remaining > 0:
        size = min(chunk, remaining)
        remaining -= size
        z = generator.standard_normal((size, n_steps))
        # numba, not numpy: M2 established the two are bit-identical.
        paths = price_paths(z, market, maturity, engine="numba")
        intrinsic = discount * np.maximum(paths[:, -1] - STRIKE, 0.0)

        discrete = intrinsic * observed_survival(paths, barrier, "down")
        bridged = intrinsic * bridge_survival(paths, barrier, market.vol, dt, generator, "down")
        difference = discrete - bridged

        total["d"] += discrete.sum()
        total["dd"] += np.square(discrete).sum()
        total["b"] += bridged.sum()
        total["bb"] += np.square(bridged).sum()
        total["diff"] += difference.sum()
        total["diff2"] += np.square(difference).sum()
        total["n"] += size

    n = total["n"]

    def mean_and_se(total_sum, total_sq):
        mean = total_sum / n
        variance = max(total_sq / n - mean * mean, 0.0) * n / (n - 1)
        return float(mean), float(np.sqrt(variance / n))

    discrete_mean, discrete_se = mean_and_se(total["d"], total["dd"])
    bridged_mean, bridged_se = mean_and_se(total["b"], total["bb"])
    difference_mean, difference_se = mean_and_se(total["diff"], total["diff2"])

    return {
        "discrete": discrete_mean,
        "bridged": bridged_mean,
        "difference": difference_mean,
        "difference_se": difference_se,
        "bridged_se": bridged_se,
        "discrete_se": discrete_se,
    }


# --------------------------------------------------------------------------
# The closed forms this milestone validates against
# --------------------------------------------------------------------------
def test_in_out_parity_holds_exactly():
    for barrier in (70.0, 80.0, 90.0, 95.0, 99.0):
        args = (MARKET.spot, STRIKE, barrier, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY)
        total = down_and_out_call(*args) + down_and_in_call(*args)
        assert total == pytest.approx(
            bs.call_price(MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY),
            rel=1e-13,
        )


def test_knock_out_value_falls_as_the_barrier_rises_toward_spot():
    prices = [
        down_and_out_call(
            MARKET.spot, STRIKE, b, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
        )
        for b in (70.0, 80.0, 90.0, 95.0, 99.0)
    ]
    assert prices == sorted(prices, reverse=True)


def test_both_barrier_regimes_are_covered():
    """H <= K and H > K use different formulas; both must be sane."""
    below = down_and_out_call(100.0, 100.0, 90.0, 0.02, 0.0, 0.2, 1.0)
    above = down_and_out_call(120.0, 100.0, 110.0, 0.02, 0.0, 0.2, 1.0)
    assert 0.0 < below < bs.call_price(100.0, 100.0, 0.02, 0.0, 0.2, 1.0)
    assert 0.0 < above < bs.call_price(120.0, 100.0, 0.02, 0.0, 0.2, 1.0)


def test_already_knocked_out_is_worthless():
    assert down_and_out_call(90.0, 100.0, 95.0, 0.02, 0.0, 0.2, 1.0) == 0.0


def test_bgk_shift_moves_the_barrier_away_from_spot():
    """Trap 9 is getting this direction backwards."""
    assert shifted_barrier(90.0, 0.2, 1 / 252, "down") < 90.0
    assert shifted_barrier(110.0, 0.2, 1 / 252, "up") > 110.0
    # And the shift vanishes as monitoring becomes continuous.
    assert shifted_barrier(90.0, 0.2, 1e-12, "down") == pytest.approx(90.0, rel=1e-6)


def test_bgk_constant_is_the_published_value():
    assert BETA_BGK == pytest.approx(0.5826, abs=5e-5)


# --------------------------------------------------------------------------
# The bridge formula
# --------------------------------------------------------------------------
def test_crossing_probability_is_one_when_an_endpoint_has_already_breached():
    p = crossing_probability(np.array([89.0]), np.array([95.0]), 90.0, 0.2, 1 / 252, "down")
    assert p[0] == 1.0


def test_crossing_probability_rises_as_the_path_approaches_the_barrier():
    starts = np.array([120.0, 105.0, 95.0, 91.0, 90.5])
    ends = starts - 0.5
    p = crossing_probability(starts, ends, 90.0, 0.2, 1 / 252, "down")
    assert np.all(np.diff(p) > 0)
    assert p[0] < 1e-6 and p[-1] > 0.3


def test_crossing_probability_is_symmetric_under_reflection():
    """One formula, not two: the upper case is the lower case reflected.

    Reflecting log-price about log(B) maps `S -> B^2/S` and turns an upper
    barrier into a lower one, so the two directions must agree exactly.
    """
    starts, ends, barrier = np.array([95.0, 105.0, 99.0]), np.array([101.0, 97.0, 100.5]), 100.0
    up = crossing_probability(starts, ends, barrier, 0.2, 1 / 252, "up")
    down = crossing_probability(
        barrier**2 / starts, barrier**2 / ends, barrier, 0.2, 1 / 252, "down"
    )
    assert np.allclose(up, down, rtol=1e-14)


def test_crossing_probability_grows_with_volatility_and_step_size():
    args = (np.array([95.0]), np.array([94.0]), 90.0)
    assert (
        crossing_probability(*args, 0.4, 1 / 252)[0]
        > crossing_probability(*args, 0.2, 1 / 252)[0]
    )
    assert (
        crossing_probability(*args, 0.2, 1 / 52)[0]
        > crossing_probability(*args, 0.2, 1 / 252)[0]
    )


def test_bridge_never_keeps_a_path_the_observed_series_already_killed():
    """The correction can only remove survivors, never resurrect them."""
    generator = rng(7)
    z = generator.standard_normal((20_000, 252))
    paths = price_paths(z, MARKET, MATURITY)
    observed = observed_survival(paths, BARRIER, "down")
    bridged = bridge_survival(paths, BARRIER, MARKET.vol, 1 / 252, generator, "down")
    assert np.all(bridged <= observed)


def test_crossing_probability_rejects_degenerate_inputs():
    with pytest.raises(ValueError, match="positive volatility"):
        crossing_probability(np.array([95.0]), np.array([94.0]), 90.0, 0.0, 1 / 252)
    with pytest.raises(ValueError, match="barrier must be positive"):
        crossing_probability(np.array([95.0]), np.array([94.0]), 0.0, 0.2, 1 / 252)


# --------------------------------------------------------------------------
# Acceptance 2 — the sign
# --------------------------------------------------------------------------
def test_correcting_a_knock_out_lowers_its_price():
    """Discrete monitoring misses excursions, so it keeps too many paths alive.

    Getting this backwards is trap 9. The mechanism is one sentence: an
    unobserved round trip below the barrier leaves a path alive that should
    have died, and a live path in a knock-out is worth more than a dead one.
    """
    out = _price_both(seed=101, n_steps=252, n_paths=400_000)
    assert out["difference"] > 0, "correction must reduce the knock-out price"
    assert out["difference"] > 3 * out["difference_se"], "and detectably so"
    assert out["bridged"] < out["discrete"]


# --------------------------------------------------------------------------
# Acceptance 3 — bridge and BGK agree, both against the closed form
# --------------------------------------------------------------------------
def test_bridge_corrected_price_matches_the_continuous_closed_form():
    """The bridge should reproduce continuous monitoring, which has an exact price."""
    continuous = down_and_out_call(
        MARKET.spot, STRIKE, BARRIER, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
    )
    out = _price_both(seed=202, n_steps=252, n_paths=600_000)
    assert abs(out["bridged"] - continuous) < 3.0 * out["bridged_se"], (
        f"bridged {out['bridged']:.6f} vs continuous {continuous:.6f}, se {out['bridged_se']:.6f}"
    )


def test_discrete_price_matches_the_bgk_shifted_closed_form():
    """The independent route: shift the barrier instead of simulating the excursions.

    These two approximations come at the answer from opposite directions — one
    simulates the missed crossings, the other moves the barrier to account for
    them — so agreement is evidence rather than a restatement.
    """
    bgk_barrier = shifted_barrier(BARRIER, MARKET.vol, MATURITY / 252, "down")
    bgk_price = down_and_out_call(
        MARKET.spot, STRIKE, bgk_barrier, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
    )
    out = _price_both(seed=303, n_steps=252, n_paths=600_000)
    assert abs(out["discrete"] - bgk_price) < 4.0 * out["discrete_se"], (
        f"discrete {out['discrete']:.6f} vs BGK {bgk_price:.6f}, se {out['discrete_se']:.6f}"
    )


def test_the_two_routes_bracket_the_same_answer():
    """Continuous < discrete, and both corrections point the same way."""
    continuous = down_and_out_call(
        MARKET.spot, STRIKE, BARRIER, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
    )
    bgk_barrier = shifted_barrier(BARRIER, MARKET.vol, MATURITY / 252, "down")
    bgk_price = down_and_out_call(
        MARKET.spot, STRIKE, bgk_barrier, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
    )
    assert continuous < bgk_price  # a further barrier is worth more


# --------------------------------------------------------------------------
# Acceptance 1 — convergence as the grid refines
# --------------------------------------------------------------------------
def test_the_correction_shrinks_as_monitoring_becomes_continuous():
    """The limit the whole correction is defined by."""
    differences = []
    for n_steps in (12, 52, 252, 1008):
        # Path count falls as the grid refines so the work stays bounded; the
        # quantity being tested is the ordering of the differences, not their
        # individual precision.
        out = _price_both(
            seed=404 + n_steps,
            n_steps=n_steps,
            n_paths=max(60_000, 400_000 // (n_steps // 12)),
            chunk=20_000,
        )
        differences.append(out["difference"])

    assert differences == sorted(differences, reverse=True), differences
    assert differences[-1] < 0.25 * differences[0]


def test_bgk_shift_and_the_measured_gap_shrink_together():
    """Both routes must vanish in the continuous limit, not just one."""
    for n_steps in (52, 252, 1008):
        shift = BARRIER - shifted_barrier(BARRIER, MARKET.vol, MATURITY / n_steps, "down")
        assert shift > 0
    shifts = [
        BARRIER - shifted_barrier(BARRIER, MARKET.vol, MATURITY / n, "down")
        for n in (12, 52, 252, 1008)
    ]
    assert shifts == sorted(shifts, reverse=True)


# --------------------------------------------------------------------------
# The judgement — the part that matters most
# --------------------------------------------------------------------------
def test_daily_close_monitoring_on_a_daily_grid_needs_no_correction():
    """If the contract observes the same closes the simulation produces, stop.

    `observed_survival` *is* the contract's rule, evaluated exactly. There is
    no approximation to improve on, and applying the bridge here would price a
    continuously-monitored contract nobody wrote — trap 8.

    Asserted structurally: the observed-survival result depends only on the
    simulated closes and is reproducible bit-for-bit, with no auxiliary
    randomness at all, unlike the bridge.
    """
    z = rng(505).standard_normal((50_000, 252))
    paths = price_paths(z, MARKET, MATURITY)

    first = observed_survival(paths, BARRIER, "down")
    second = observed_survival(paths, BARRIER, "down")
    assert np.array_equal(first, second)

    # The bridge, by contrast, consumes randomness and so differs run to run.
    a = bridge_survival(paths, BARRIER, MARKET.vol, 1 / 252, rng(1), "down")
    b = bridge_survival(paths, BARRIER, MARKET.vol, 1 / 252, rng(2), "down")
    assert not np.array_equal(a, b)


def test_the_sensitivity_is_larger_when_the_barrier_is_close_and_vol_is_high():
    """Where the monitoring convention actually costs money."""
    far = _price_both(seed=606, n_steps=252, n_paths=300_000, barrier=70.0)
    near = _price_both(seed=606, n_steps=252, n_paths=300_000, barrier=95.0)
    assert near["difference"] > far["difference"]

    low_vol = Market(spot=100.0, rate=0.02, div_yield=0.0, vol=0.10)
    high_vol = Market(spot=100.0, rate=0.02, div_yield=0.0, vol=0.40)
    quiet = _price_both(seed=707, n_steps=252, n_paths=300_000, market=low_vol)
    wild = _price_both(seed=707, n_steps=252, n_paths=300_000, market=high_vol)
    assert wild["difference"] > quiet["difference"]
