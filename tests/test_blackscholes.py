"""M1 acceptance — the closed-form ground truth.

Three required checks: put-call parity to machine precision, analytic delta and
vega against a central finite difference to 1e-6, and the degenerate limits.
Everything else here exists because this module is what six later milestones
measure themselves against, so an error in it would not look like an error in
it — it would look like a bias in the Monte Carlo engine.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from quantdesk.analytics import blackscholes as bs

# A grid of moneyness, maturities and volatilities, plus non-zero rates and
# dividend yields so that q is actually exercised rather than defaulting away.
SPOTS = [50.0, 90.0, 100.0, 110.0, 200.0]
STRIKES = [80.0, 100.0, 125.0]
RATES = [-0.01, 0.0, 0.02, 0.06]
YIELDS = [0.0, 0.015, 0.05]
VOLS = [0.05, 0.20, 0.45, 0.80]
MATURITIES = [1.0 / 252.0, 0.25, 1.0, 5.0]

GRID = list(itertools.product(SPOTS, STRIKES, RATES, YIELDS, VOLS, MATURITIES))


def _grid_arrays():
    """The grid as broadcast arrays, for the vectorised checks."""
    return [np.array(col, dtype=float) for col in zip(*GRID, strict=True)]


# --------------------------------------------------------------------------
# Acceptance 1 — put-call parity
# --------------------------------------------------------------------------
def test_put_call_parity_to_machine_precision():
    S, K, r, q, sigma, T = _grid_arrays()
    lhs = bs.call_price(S, K, r, q, sigma, T) - bs.put_price(S, K, r, q, sigma, T)
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)

    err = np.abs(lhs - rhs)
    # Scale by the magnitude of the *terms*, not by their difference. Parity is
    # an algebraic identity — N(d1) + N(-d1) = 1 cancels exactly in exact
    # arithmetic — so the only surviving error is rounding on quantities of
    # order S*exp(-qT) and K*exp(-rT). Those two can very nearly cancel (at
    # S=90, K=100, r=2%, T=5 the difference is -0.48 from terms near 90), and
    # dividing by that cancelled residual would measure the conditioning of the
    # subtraction rather than the accuracy of the formula.
    scale = np.maximum(S * np.exp(-q * T), K * np.exp(-r * T))
    worst = np.max(err / scale)
    assert worst < 1e-15, f"worst parity error {worst:.3e} relative to term size"
    # Restated in ulps, which is what "machine precision" actually means here.
    assert np.max(err / np.spacing(scale)) < 4.0


def test_parity_holds_in_the_degenerate_cases_too():
    S, K, r, q = 100.0, 95.0, 0.03, 0.01
    for sigma, T in [(0.0, 1.0), (0.2, 0.0), (0.0, 0.0)]:
        lhs = bs.call_price(S, K, r, q, sigma, T) - bs.put_price(S, K, r, q, sigma, T)
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs(lhs - rhs) < 1e-13, f"parity broken at sigma={sigma}, T={T}"


# --------------------------------------------------------------------------
# Acceptance 2 — analytic sensitivities against central finite differences
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["call", "put"])
def test_delta_matches_central_difference(kind):
    S, K, r, q, sigma, T = _grid_arrays()
    h = 1e-5 * S  # relative bump; absolute would be wrong across a 4x spot range

    up = bs.price(S + h, K, r, q, sigma, T, kind)
    down = bs.price(S - h, K, r, q, sigma, T, kind)
    fd = (up - down) / (2.0 * h)
    analytic = bs.delta(S, K, r, q, sigma, T, kind)

    worst = np.max(np.abs(fd - analytic))
    # Absolute, and delta is dimensionless in [-1, 1], so this is a tight claim.
    assert worst < 1e-6, f"worst absolute delta error {worst:.3e}"


def test_vega_matches_central_difference():
    """Bumped at h = 1e-6, the measured optimum for this grid.

    The sweep is unambiguous about what dominates where. At h = 1e-3, 1e-4 and
    1e-5 the worst error is 8.054e-3, 8.055e-5 and 8.060e-7 — falling by
    exactly 100x per tenfold cut, the O(h^2) signature of a central difference,
    so truncation is in charge throughout. It bottoms at 2.8e-8 near h = 1e-6
    and rises again to 3.5e-7 at h = 1e-7 as cancellation takes over. The worst
    point while truncation dominates is S=110, K=125, sigma=5%, T=5, a
    long-dated low-vol option where volga is large.
    """
    S, K, r, q, sigma, T = _grid_arrays()
    h = 1e-6  # absolute in vol; vega is quoted per 1.00 of vol

    up = bs.call_price(S, K, r, q, sigma + h, T)
    down = bs.call_price(S, K, r, q, sigma - h, T)
    fd = (up - down) / (2.0 * h)
    analytic = bs.vega(S, K, r, q, sigma, T)

    worst = np.max(np.abs(fd - analytic))
    # Absolute, in price units per 1.00 of vol, on spots up to 200.
    assert worst < 1e-6, f"worst absolute vega error {worst:.3e}"


def test_gamma_matches_a_difference_of_analytic_delta():
    """Gamma against dDelta/dS — the well-conditioned route.

    Delta has already been checked against the price above, so differencing it
    validates gamma without paying the conditioning cost of a second difference
    of the price.
    """
    S, K, r, q, sigma, T = _grid_arrays()
    h = 1e-7 * S  # the measured optimum for this grid; see the sweep test below

    fd = (
        bs.delta(S + h, K, r, q, sigma, T, "call") - bs.delta(S - h, K, r, q, sigma, T, "call")
    ) / (2.0 * h)
    analytic = bs.gamma(S, K, r, q, sigma, T)

    worst = np.max(np.abs(fd - analytic))
    assert worst < 1e-9, f"worst absolute gamma error {worst:.3e}"


def test_bump_size_sweep_on_gamma_is_u_shaped():
    """The bias-variance trade-off in finite differencing, measured.

    M6 has to produce this curve for the Monte Carlo Greeks and find where its
    minimum sits. Here the true answer is known in closed form, so the shape
    can be established without any Monte Carlo noise confusing it: truncation
    error falls as h^2 while cancellation error grows as 1/h, so the total has
    an interior minimum. Establishing the shape against analytic gamma first is
    the cheap way to know the M6 sweep is measuring what it claims to.
    """
    S, K, r, q, sigma, T = _grid_arrays()
    analytic = bs.gamma(S, K, r, q, sigma, T)

    errors = []
    for exponent in (5, 6, 7, 8):
        h = 10.0 ** (-exponent) * S
        fd = (
            bs.delta(S + h, K, r, q, sigma, T, "call")
            - bs.delta(S - h, K, r, q, sigma, T, "call")
        ) / (2.0 * h)
        errors.append(float(np.max(np.abs(fd - analytic))))

    assert errors[0] > errors[1] > errors[2], "truncation should fall as h shrinks"
    assert errors[3] > errors[2], "cancellation should take over at the smallest h"
    # Truncation is O(h^2): a tenfold cut in h should buy about a hundredfold
    # in error while truncation still dominates.
    assert 30.0 < errors[0] / errors[1] < 300.0


def test_gamma_matches_second_difference_of_price_within_truncation_error():
    """The same check the hard way, and a lesson M6 will need.

    A second difference divides by h^2, so the usable window is narrow: small h
    is destroyed by cancellation, large h by the fourth derivative. Even at the
    near-optimal h = eps^(1/4) * S the worst point on this grid — a one-day
    at-the-money option, where gamma is 1.27 and the price is sharply curved —
    retains about 1e-4 of relative truncation error. That U-shaped trade-off is
    exactly what M6's bump-size sweep measures, so it is worth seeing here
    first, on a function whose true derivative is known.
    """
    S, K, r, q, sigma, T = _grid_arrays()
    h = 1e-4 * S

    fd = (
        bs.call_price(S + h, K, r, q, sigma, T)
        - 2.0 * bs.call_price(S, K, r, q, sigma, T)
        + bs.call_price(S - h, K, r, q, sigma, T)
    ) / np.square(h)
    analytic = bs.gamma(S, K, r, q, sigma, T)

    # Restricted to points where the second difference can resolve gamma at
    # all. Cancellation puts an absolute floor of about eps * price / h^2 ~
    # 1e-11 on this grid, so for deep in-the-money options whose gamma is
    # genuinely ~1e-12 the measurement is all noise and a relative error of 59x
    # means nothing. Relative error is only a meaningful statistic above that
    # floor — itself a lesson for M6, where a relative error on a Greek that
    # has underflowed is not a number worth reporting.
    live = analytic > 1e-6
    worst = np.max(np.abs(fd - analytic)[live] / analytic[live])
    assert worst < 1e-3, f"worst relative gamma error {worst:.3e} over {live.sum()} points"


def test_vega_and_gamma_are_the_same_for_calls_and_puts():
    """A consequence of parity: its right-hand side contains neither sigma nor
    a second power of S, so both sensitivities must cancel exactly."""
    S, K, r, q, sigma, T = _grid_arrays()
    h = 1e-5
    call_vega = (
        bs.call_price(S, K, r, q, sigma + h, T) - bs.call_price(S, K, r, q, sigma - h, T)
    ) / (2 * h)
    put_vega = (
        bs.put_price(S, K, r, q, sigma + h, T) - bs.put_price(S, K, r, q, sigma - h, T)
    ) / (2 * h)
    assert np.max(np.abs(call_vega - put_vega)) < 1e-6


# --------------------------------------------------------------------------
# Acceptance 3 — degenerate cases
# --------------------------------------------------------------------------
def test_zero_vol_collapses_to_discounted_intrinsic():
    """With sigma = 0 the payoff is deterministic, so the option is worth the
    discounted intrinsic value of the *forward*, not of spot."""
    S, K, r, q, _, T = _grid_arrays()
    expected_call = np.maximum(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    expected_put = np.maximum(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)

    assert np.allclose(bs.call_price(S, K, r, q, 0.0, T), expected_call, atol=0, rtol=0)
    assert np.allclose(bs.put_price(S, K, r, q, 0.0, T), expected_put, atol=0, rtol=0)


def test_small_vol_approaches_the_zero_vol_limit():
    """The sigma = 0 branch is a limit, not a special case bolted on: the
    continuous formula must walk into it.

    Probed at the money forward, where the convergence is O(sigma) and so
    visible. Away from the money it is far faster than that — at K = 95 with a
    forward of 102 the price already equals the limit to the last bit by
    sigma = 1e-3, because N(d1) and N(d2) have both saturated — which makes
    that a useless place to watch a limit being approached.
    """
    S, r, q, T = 100.0, 0.03, 0.01, 1.0
    K_atmf = S * np.exp((r - q) * T)

    limit = bs.call_price(S, K_atmf, r, q, 0.0, T)
    assert limit == pytest.approx(0.0, abs=1e-13)  # forward = strike

    errs = [abs(bs.call_price(S, K_atmf, r, q, s, T) - limit) for s in (1e-2, 1e-3, 1e-4)]
    assert errs[0] > errs[1] > errs[2]
    # Price at the money forward is ~ S*exp(-qT)*phi(0)*sigma*sqrt(T), i.e.
    # linear in sigma with slope ~39.5 here; check the ratio, not just the order.
    assert errs[0] / errs[1] == pytest.approx(10.0, rel=1e-3)
    assert errs[1] / errs[2] == pytest.approx(10.0, rel=1e-3)


def test_far_from_the_money_the_zero_vol_limit_is_reached_exactly():
    """Documents the behaviour the previous test avoids leaning on."""
    S, K, r, q, T = 100.0, 95.0, 0.03, 0.01, 1.0
    limit = bs.call_price(S, K, r, q, 0.0, T)
    assert bs.call_price(S, K, r, q, 1e-3, T) == limit


def test_zero_maturity_collapses_to_intrinsic():
    S, K, r, q, sigma, _ = _grid_arrays()
    assert np.allclose(bs.call_price(S, K, r, q, sigma, 0.0), np.maximum(S - K, 0.0), rtol=0, atol=0)
    assert np.allclose(bs.put_price(S, K, r, q, sigma, 0.0), np.maximum(K - S, 0.0), rtol=0, atol=0)


def test_short_maturity_approaches_intrinsic():
    S, K, r, q, sigma = 100.0, 95.0, 0.03, 0.01, 0.2
    errs = [abs(bs.call_price(S, K, r, q, sigma, t) - max(S - K, 0.0)) for t in (1e-4, 1e-6, 1e-8)]
    assert errs[0] > errs[1] > errs[2]


def test_degenerate_delta_is_the_forward_moneyness_indicator():
    r, q, T = 0.03, 0.01, 1.0
    # Forward = 100 * exp(0.02) = 102.02; strikes either side of it.
    assert bs.delta(100.0, 90.0, r, q, 0.0, T, "call") == pytest.approx(np.exp(-q * T))
    assert bs.delta(100.0, 120.0, r, q, 0.0, T, "call") == 0.0
    assert bs.delta(100.0, 120.0, r, q, 0.0, T, "put") == pytest.approx(-np.exp(-q * T))
    # T = 0: the indicator is on spot.
    assert bs.delta(100.0, 99.0, r, q, 0.2, 0.0, "call") == pytest.approx(1.0)
    assert bs.delta(100.0, 101.0, r, q, 0.2, 0.0, "call") == 0.0


def test_degenerate_vega_is_not_uniformly_zero_at_the_money_forward():
    """The easy mistake is returning 0 everywhere when sigma = 0.

    Away from the money vega does vanish, because d1 diverges and phi(d1) dies
    faster than 1/d1. Exactly at the money forward d1 -> 0 instead, and vega
    tends to S*exp(-qT)*phi(0)*sqrt(T).
    """
    S, r, q, T = 100.0, 0.03, 0.01, 1.0
    K_atmf = S * np.exp((r - q) * T)  # strike equal to the forward

    assert bs.vega(S, 90.0, r, q, 0.0, T) == 0.0
    expected = S * np.exp(-q * T) * (1.0 / np.sqrt(2 * np.pi)) * np.sqrt(T)
    assert bs.vega(S, K_atmf, r, q, 0.0, T) == pytest.approx(expected, rel=1e-12)

    # And the continuous formula converges to that same value.
    assert bs.vega(S, K_atmf, r, q, 1e-6, T) == pytest.approx(expected, rel=1e-6)


def test_degenerate_gamma_is_infinite_at_the_money_forward():
    S, r, q, T = 100.0, 0.03, 0.01, 1.0
    K_atmf = S * np.exp((r - q) * T)
    assert bs.gamma(S, 90.0, r, q, 0.0, T) == 0.0
    assert np.isinf(bs.gamma(S, K_atmf, r, q, 0.0, T))


# --------------------------------------------------------------------------
# Structural properties
# --------------------------------------------------------------------------
def test_prices_respect_no_arbitrage_bounds():
    S, K, r, q, sigma, T = _grid_arrays()
    call = bs.call_price(S, K, r, q, sigma, T)
    put = bs.put_price(S, K, r, q, sigma, T)
    fwd_pv, strike_pv = S * np.exp(-q * T), K * np.exp(-r * T)

    assert np.all(call >= np.maximum(fwd_pv - strike_pv, 0.0) - 1e-12)
    assert np.all(call <= fwd_pv + 1e-12)
    assert np.all(put >= np.maximum(strike_pv - fwd_pv, 0.0) - 1e-12)
    assert np.all(put <= strike_pv + 1e-12)


def test_delta_bounds_and_signs():
    S, K, r, q, sigma, T = _grid_arrays()
    cap = np.exp(-q * T)
    call_d = bs.delta(S, K, r, q, sigma, T, "call")
    put_d = bs.delta(S, K, r, q, sigma, T, "put")
    assert np.all((call_d >= 0.0) & (call_d <= cap + 1e-15))
    assert np.all((put_d <= 0.0) & (put_d >= -cap - 1e-15))
    # The two differ by exactly the parity delta.
    assert np.max(np.abs((call_d - put_d) - cap)) < 1e-14


def test_vega_and_gamma_are_non_negative_and_prices_are_monotone():
    S, K, r, q, sigma, T = _grid_arrays()
    # Non-negative, not strictly positive — see the underflow test below.
    assert np.all(bs.vega(S, K, r, q, sigma, T) >= 0.0)
    assert np.all(bs.gamma(S, K, r, q, sigma, T) >= 0.0)

    # Monotone increasing in vol, and calls monotone increasing in spot.
    #
    # Asserted to within a few ulps rather than exactly. Two grid points are
    # one-day options 20+ points in the money, where vega is ~1e-13: the true
    # price gain from a 1% vol bump is ~3e-16, three orders of magnitude below
    # the 1.4e-14 granularity of a double near 20. Rounding therefore decides
    # the comparison, and demanding exact monotonicity there would be asserting
    # something float64 cannot represent.
    base = bs.call_price(S, K, r, q, sigma, T)
    tol = 16.0 * np.spacing(np.maximum(base, 1.0))
    assert np.all(bs.call_price(S, K, r, q, sigma * 1.01, T) >= base - tol)
    assert np.all(bs.call_price(S * 1.01, K, r, q, sigma, T) >= base - tol)

    put_base = bs.put_price(S, K, r, q, sigma, T)
    put_tol = 16.0 * np.spacing(np.maximum(put_base, 1.0))
    assert np.all(bs.put_price(S * 1.01, K, r, q, sigma, T) <= put_base + put_tol)

    # Where the effect is genuinely resolvable, monotonicity is strict.
    resolvable = bs.vega(S, K, r, q, sigma, T) * 0.01 * sigma > 1e3 * np.spacing(base)
    assert resolvable.sum() > 2000  # most of the grid, not a token subset
    assert np.all(bs.call_price(S, K, r, q, sigma * 1.01, T)[resolvable] > base[resolvable])


def test_vega_and_gamma_underflow_to_zero_far_out_of_the_money():
    """Analytically strictly positive; in float64, exactly zero. Know which.

    At S=50, K=80, sigma=5%, T=1/252 the option is about 150 standard
    deviations out of the money, so phi(d1) = exp(-11135)/sqrt(2*pi). The true
    vega is around 1e-4836 and the smallest representable double is 5e-324, so
    it underflows to exactly 0 — 180 of this grid's 2880 points do.

    This is not a defect and must not be patched over with a floor: zero is the
    nearest representable value, and it is the correct answer to give a caller.
    It matters for M6, where a relative error on a Greek is meaningless once
    the Greek itself has underflowed, and for any test tempted to assert strict
    positivity.
    """
    S, K, r, q, sigma, T = 50.0, 80.0, -0.01, 0.0, 0.05, 1.0 / 252.0
    d1, _ = bs.d1_d2(S, K, r, q, sigma, T)
    assert d1 < -100.0
    assert bs.vega(S, K, r, q, sigma, T) == 0.0
    assert bs.gamma(S, K, r, q, sigma, T) == 0.0
    # The price underflows to zero here too, consistently.
    assert bs.call_price(S, K, r, q, sigma, T) == 0.0

    # Wherever d1 is in a range float64 can represent, vega is strictly positive.
    Sg, Kg, rg, qg, sg, Tg = _grid_arrays()
    d1_grid, _ = bs.d1_d2(Sg, Kg, rg, qg, sg, Tg)
    representable = np.abs(d1_grid) < 35.0  # exp(-612) is still well above 1e-324
    assert np.all(bs.vega(Sg, Kg, rg, qg, sg, Tg)[representable] > 0.0)
    assert np.all(bs.gamma(Sg, Kg, rg, qg, sg, Tg)[representable] > 0.0)


def test_deep_in_and_out_of_the_money_limits():
    r, q, sigma, T = 0.02, 0.0, 0.2, 1.0
    assert bs.call_price(1e-6 * 100, 100.0, r, q, sigma, T) == pytest.approx(0.0, abs=1e-12)
    deep_itm = bs.call_price(10_000.0, 100.0, r, q, sigma, T)
    assert deep_itm == pytest.approx(10_000.0 - 100.0 * np.exp(-r * T), rel=1e-12)


def test_scalar_inputs_return_floats_and_vectorised_agrees():
    scalar = bs.call_price(100.0, 100.0, 0.02, 0.0, 0.2, 1.0)
    assert isinstance(scalar, float)

    vector = bs.call_price(np.full(4, 100.0), 100.0, 0.02, 0.0, 0.2, 1.0)
    assert isinstance(vector, np.ndarray) and vector.shape == (4,)
    assert np.all(vector == scalar)


def test_d1_d2_are_nan_where_undefined():
    d1, d2 = bs.d1_d2(100.0, 100.0, 0.02, 0.0, 0.0, 1.0)
    assert np.isnan(d1) and np.isnan(d2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"S": -1.0},
        {"S": 0.0},
        {"K": 0.0},
        {"sigma": -0.1},
        {"T": -1.0},
        {"S": np.inf},
        {"sigma": np.nan},
    ],
)
def test_invalid_inputs_raise(kwargs):
    args = {"S": 100.0, "K": 100.0, "r": 0.02, "q": 0.0, "sigma": 0.2, "T": 1.0} | kwargs
    with pytest.raises(ValueError):
        bs.call_price(**args)


def test_bad_kind_raises():
    with pytest.raises(ValueError, match="call.*put"):
        bs.price(100.0, 100.0, 0.02, 0.0, 0.2, 1.0, kind="straddle")


def test_greeks_bundle_matches_individual_functions():
    args = (100.0, 105.0, 0.02, 0.01, 0.25, 1.5)
    g = bs.greeks(*args, kind="put")
    assert g["price"] == bs.put_price(*args)
    assert g["delta"] == bs.delta(*args, kind="put")
    assert g["vega"] == bs.vega(*args)
    assert g["gamma"] == bs.gamma(*args)


def test_known_values_from_an_independent_high_precision_computation():
    """One absolute anchor, independent of this implementation.

    The property tests above are all internal consistency checks: parity,
    finite differences, monotonicity. They would all still pass if the whole
    module were wrong by a common factor. This test pins the actual values.

    Reference figures for S=100, K=100, r=5%, q=0, sigma=20%, T=1 were computed
    at 50 decimal digits with mpmath, using erf directly rather than SciPy's
    ndtr, and are quoted here to 20 significant figures:

        python3 -c "
        import mpmath as mp; mp.mp.dps = 50
        S,K,r,q,s,T = mp.mpf(100),mp.mpf(100),mp.mpf('0.05'),mp.mpf(0),mp.mpf('0.20'),mp.mpf(1)
        N = lambda x: (1 + mp.erf(x/mp.sqrt(2)))/2
        d1 = (mp.log(S/K) + (r-q+s**2/2)*T)/(s*mp.sqrt(T)); d2 = d1 - s*mp.sqrt(T)
        print(mp.nstr(S*mp.e**(-q*T)*N(d1) - K*mp.e**(-r*T)*N(d2), 20))"

    mpmath is not a dependency of this package; it was used once, at the
    terminal, to generate the constants below.
    """
    args = (100.0, 100.0, 0.05, 0.0, 0.20, 1.0)
    assert bs.call_price(*args) == pytest.approx(10.450583572185566782, rel=1e-15)
    assert bs.put_price(*args) == pytest.approx(5.5735260222569676908, rel=1e-15)
    assert bs.delta(*args, kind="call") == pytest.approx(0.63683065117561907122, rel=1e-15)
    assert bs.vega(*args[:5], args[5]) == pytest.approx(37.524034691693787837, rel=1e-15)
    assert bs.gamma(*args[:5], args[5]) == pytest.approx(0.018762017345846893919, rel=1e-15)
