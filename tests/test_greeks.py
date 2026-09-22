"""M6 acceptance — pathwise, likelihood ratio, and bump-and-revalue.

Three required checks:
  1. pathwise and bump agree on the vanilla to a stated tolerance, scored
     against M1's exact delta and vega, with the units of the tolerance named;
  2. the likelihood-ratio delta on a digital matches its closed form within
     noise;
  3. the bump-size sweep shows the expected U shape.

Plus the one that matters most for the interview: that the pathwise estimator
applied across a discontinuity returns a confident wrong answer rather than
raising, demonstrated rather than described.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantdesk.analytics import blackscholes as bs
from quantdesk.config import BASE_MARKET, Market
from quantdesk.mc.greeks import (
    bump_samples,
    central_difference_greek,
    likelihood_ratio,
    pathwise_digital_delta,
    pathwise_european,
    summarise,
)
from quantdesk.mc.paths import terminal_prices
from quantdesk.rng import rng

MARKET = BASE_MARKET
MATURITY = 1.0
STRIKE = 100.0
N = 2_000_000

TRUE_DELTA = bs.delta(MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY)
TRUE_VEGA = bs.vega(MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY)
TRUE_DIGITAL_DELTA = bs.digital_delta(
    MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
)


def _discount() -> float:
    return float(np.exp(-MARKET.rate * MATURITY))


def call_payoff(terminal):
    return _discount() * np.maximum(terminal - STRIKE, 0.0)


def digital_payoff(terminal):
    return _discount() * (terminal > STRIKE)


# --------------------------------------------------------------------------
# The digital closed forms themselves, since M6 validates against them
# --------------------------------------------------------------------------
def test_digital_price_and_greeks_match_finite_differences():
    args = (MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY)
    h = 1e-6 * MARKET.spot
    fd_delta = (
        bs.digital_price(MARKET.spot + h, *args[1:]) - bs.digital_price(MARKET.spot - h, *args[1:])
    ) / (2 * h)
    assert bs.digital_delta(*args) == pytest.approx(fd_delta, rel=1e-6)

    hv = 1e-6
    fd_vega = (
        bs.digital_price(*args[:4], MARKET.vol + hv, MATURITY)
        - bs.digital_price(*args[:4], MARKET.vol - hv, MATURITY)
    ) / (2 * hv)
    assert bs.digital_vega(*args) == pytest.approx(fd_vega, rel=1e-5)


def test_digital_is_bounded_by_the_discount_factor_and_monotone():
    spots = np.array([50.0, 80.0, 100.0, 120.0, 200.0])
    prices = bs.digital_price(spots, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY)
    assert np.all((prices >= 0) & (prices <= np.exp(-MARKET.rate * MATURITY) + 1e-15))
    assert np.all(np.diff(prices) > 0)  # a digital call rises with spot


def test_digital_call_vega_is_negative_when_in_the_money():
    """A sign worth pinning: more vol makes an already-likely payoff less likely."""
    deep_itm = bs.digital_vega(150.0, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY)
    deep_otm = bs.digital_vega(60.0, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY)
    assert deep_itm < 0 < deep_otm


def test_digital_call_and_put_sum_to_the_discount_factor():
    args = (MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY)
    total = bs.digital_price(*args, kind="call") + bs.digital_price(*args, kind="put")
    assert total == pytest.approx(np.exp(-MARKET.rate * MATURITY), rel=1e-14)


# --------------------------------------------------------------------------
# Acceptance 1 — pathwise and bump agree on the vanilla
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def vanilla_estimates():
    z = rng(24680).standard_normal(N)
    pathwise = pathwise_european(z, MARKET, MATURITY, STRIKE)
    return {
        "pw_delta": summarise(pathwise["delta"]),
        "pw_vega": summarise(pathwise["vega"]),
        "bump_delta": summarise(
            bump_samples(z, MARKET, MATURITY, call_payoff, parameter="spot", bump=0.01)
        ),
        "bump_vega": summarise(
            bump_samples(z, MARKET, MATURITY, call_payoff, parameter="vol", bump=0.01)
        ),
        "price": summarise(pathwise["price"]),
    }


def test_pathwise_delta_and_vega_match_the_closed_form(vanilla_estimates):
    """Scored against M1, not against the other estimator."""
    pw_delta, pw_vega = vanilla_estimates["pw_delta"], vanilla_estimates["pw_vega"]
    assert pw_delta.agrees_with(TRUE_DELTA, n_sigma=3.0), (
        f"delta {pw_delta.value:.8f} vs {TRUE_DELTA:.8f}, se {pw_delta.standard_error:.2e}"
    )
    assert pw_vega.agrees_with(TRUE_VEGA, n_sigma=3.0)


def test_bumped_delta_and_vega_match_the_closed_form(vanilla_estimates):
    """Bump carries truncation bias as well as noise, so it is scored absolutely.

    At a 1% relative bump the central-difference truncation error on delta is
    O(h^2) times the third derivative, which is small but not zero — so the
    tolerance here is an absolute one on a dimensionless quantity rather than a
    multiple of the standard error.
    """
    assert abs(vanilla_estimates["bump_delta"].value - TRUE_DELTA) < 1e-3
    assert abs(vanilla_estimates["bump_vega"].value - TRUE_VEGA) / TRUE_VEGA < 1e-3


def test_pathwise_and_bump_agree_with_each_other_and_the_units_are_named(vanilla_estimates):
    """The headline agreement, with its units stated.

    Delta is dimensionless and lives in [0, 1], so an absolute agreement of
    1e-4 on delta is a far stronger claim than 1e-4 on a vega whose scale is
    about 37 per 1.00 of volatility. The vega comparison is therefore also
    expressed relative to the vega itself.
    """
    delta_gap = abs(vanilla_estimates["pw_delta"].value - vanilla_estimates["bump_delta"].value)
    vega_gap = abs(vanilla_estimates["pw_vega"].value - vanilla_estimates["bump_vega"].value)

    assert delta_gap < 1e-3, f"delta gap {delta_gap:.2e} (absolute, dimensionless)"
    assert vega_gap / TRUE_VEGA < 1e-3, f"vega gap {vega_gap:.2e} (absolute, per 1.00 vol)"


def test_pathwise_price_agrees_with_the_closed_form(vanilla_estimates):
    truth = bs.call_price(
        MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
    )
    assert vanilla_estimates["price"].agrees_with(truth, n_sigma=3.0)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_pathwise_works_for_puts_too(kind):
    z = rng(1357).standard_normal(500_000)
    out = pathwise_european(z, MARKET, MATURITY, STRIKE, kind=kind)
    truth_delta = bs.delta(
        MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY, kind
    )
    assert summarise(out["delta"]).agrees_with(truth_delta, n_sigma=3.5)
    assert summarise(out["vega"]).agrees_with(TRUE_VEGA, n_sigma=3.5)


def test_pathwise_deltas_of_call_and_put_obey_parity_exactly_path_by_path():
    """The identity is exact per path, and only *in expectation* on the mean.

    Per path the call and put delta samples are `disc*1{S_T>K}*S_T/S_0` and
    `-disc*1{S_T<K}*S_T/S_0`, so their difference is `disc*S_T/S_0` exactly —
    the indicators partition the sample space. Averaging that gives
    `exp(-rT) * mean(S_T)/S_0`, which equals `exp(-qT)` only in expectation and
    carries sampling error like any other Monte Carlo mean. Asserting exactness
    on the mean would be asserting that a random variable equals its own
    expectation.
    """
    z = rng(999).standard_normal(400_000)
    call = pathwise_european(z, MARKET, MATURITY, STRIKE, "call")["delta"]
    put = pathwise_european(z, MARKET, MATURITY, STRIKE, "put")["delta"]

    terminal = terminal_prices(z, MARKET, MATURITY)
    expected = np.exp(-MARKET.rate * MATURITY) * terminal / MARKET.spot
    assert np.allclose(call - put, expected, rtol=0, atol=1e-15)

    # And the mean lands on exp(-qT) within its own sampling error.
    difference = summarise(call - put)
    assert difference.agrees_with(float(np.exp(-MARKET.div_yield * MATURITY)), n_sigma=3.0)


# --------------------------------------------------------------------------
# Acceptance 2 — the validity condition, and the likelihood ratio
# --------------------------------------------------------------------------
def test_pathwise_on_a_digital_returns_exactly_zero_and_does_not_raise():
    """The single most important test in this module.

    The derivative of a step function is zero almost everywhere, so the naive
    pathwise estimator produces exactly 0.0 for a digital — with a standard
    error of exactly 0.0, which makes it look *more* trustworthy rather than
    less. The true delta is about 0.0196. Nothing raises.
    """
    z = rng(111).standard_normal(200_000)
    estimate = summarise(pathwise_digital_delta(z, MARKET, MATURITY, STRIKE))

    assert estimate.value == 0.0
    assert estimate.standard_error == 0.0
    assert TRUE_DIGITAL_DELTA > 0.015  # the answer it should have given
    assert abs(estimate.value - TRUE_DIGITAL_DELTA) > 0.015


def test_likelihood_ratio_delta_on_a_digital_matches_the_closed_form():
    """Acceptance: the score-function estimator handles the jump correctly."""
    z = rng(222).standard_normal(N)
    payoff = digital_payoff(terminal_prices(z, MARKET, MATURITY))
    estimate = summarise(likelihood_ratio(z, MARKET, MATURITY, payoff)["delta"])

    assert estimate.agrees_with(TRUE_DIGITAL_DELTA, n_sigma=3.0), (
        f"LR delta {estimate.value:.8f} vs closed form {TRUE_DIGITAL_DELTA:.8f}, "
        f"se {estimate.standard_error:.2e}"
    )


def test_likelihood_ratio_vega_on_a_digital_matches_the_closed_form():
    z = rng(333).standard_normal(N)
    payoff = digital_payoff(terminal_prices(z, MARKET, MATURITY))
    estimate = summarise(likelihood_ratio(z, MARKET, MATURITY, payoff)["vega"])
    truth = bs.digital_vega(
        MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
    )
    assert estimate.agrees_with(truth, n_sigma=3.5)


def test_likelihood_ratio_also_works_where_pathwise_does_but_costs_variance():
    """The trade-off, measured. This is why both estimators exist."""
    z = rng(444).standard_normal(N)
    terminal = terminal_prices(z, MARKET, MATURITY)

    pathwise = summarise(pathwise_european(z, MARKET, MATURITY, STRIKE)["delta"])
    lr = summarise(likelihood_ratio(z, MARKET, MATURITY, call_payoff(terminal))["delta"])

    # Both unbiased ...
    assert pathwise.agrees_with(TRUE_DELTA, n_sigma=3.0)
    assert lr.agrees_with(TRUE_DELTA, n_sigma=3.5)
    # ... but the score function is materially noisier.
    assert lr.standard_error > 2.0 * pathwise.standard_error


def test_likelihood_ratio_rejects_mismatched_draws():
    z = rng(1).standard_normal(100)
    with pytest.raises(ValueError, match="same draws"):
        likelihood_ratio(z, MARKET, MATURITY, np.zeros(50))


def test_likelihood_ratio_rejects_zero_volatility():
    flat = Market(spot=100.0, rate=0.02, div_yield=0.0, vol=0.0)
    z = rng(1).standard_normal(100)
    with pytest.raises(ValueError, match="positive volatility"):
        likelihood_ratio(z, flat, MATURITY, np.zeros(100))


# --------------------------------------------------------------------------
# Common random numbers
# --------------------------------------------------------------------------
def test_common_random_numbers_are_dramatically_better_than_independent_draws():
    """Trap 6, quantified rather than asserted.

    At a 1% bump the measured penalty is about 17x in standard error, which is
    roughly 280x in paths for the same accuracy. The threshold below is set
    from that measurement rather than guessed.
    """
    z = rng(555).standard_normal(400_000)
    z_independent = rng(556).standard_normal(400_000)

    with_crn = summarise(
        bump_samples(z, MARKET, MATURITY, call_payoff, parameter="spot", bump=0.01)
    )
    without = summarise(
        bump_samples(
            z, MARKET, MATURITY, call_payoff, parameter="spot", bump=0.01, z_down=z_independent
        )
    )

    ratio = without.standard_error / with_crn.standard_error
    assert ratio > 10.0, f"CRN advantage only {ratio:.1f}x"
    assert abs(with_crn.value - TRUE_DELTA) < abs(without.value - TRUE_DELTA)


def test_the_crn_advantage_grows_as_the_bump_shrinks():
    """The mechanism, not just the fact.

    Without common random numbers the numerator is the difference of two
    independent noisy prices, whose standard error does not shrink with `h`,
    while the `1/(2h)` divisor blows it up. With common random numbers the two
    legs share their noise and the numerator shrinks with `h` as the signal
    does. So the penalty scales like `1/h`: halve the bump and the gap roughly
    doubles.
    """
    z = rng(559).standard_normal(200_000)
    z_independent = rng(560).standard_normal(200_000)

    ratios = []
    for bump in (0.04, 0.01):
        crn = summarise(bump_samples(z, MARKET, MATURITY, call_payoff, parameter="spot", bump=bump))
        free = summarise(
            bump_samples(
                z, MARKET, MATURITY, call_payoff, parameter="spot", bump=bump, z_down=z_independent
            )
        )
        ratios.append(free.standard_error / crn.standard_error)

    assert ratios[1] > 2.0 * ratios[0], f"penalty did not grow as the bump shrank: {ratios}"


def test_without_common_random_numbers_the_estimate_is_still_unbiased():
    """It is noise, not bias — which is exactly why it is so easy to miss."""
    z = rng(557).standard_normal(400_000)
    z_independent = rng(558).standard_normal(400_000)
    without = summarise(
        bump_samples(
            z, MARKET, MATURITY, call_payoff, parameter="spot", bump=0.01, z_down=z_independent
        )
    )
    assert without.agrees_with(TRUE_DELTA, n_sigma=3.0)


# --------------------------------------------------------------------------
# Acceptance 3 — the bump-size sweep is U-shaped
# --------------------------------------------------------------------------
BUMPS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def _sweep(z, z_down=None, n=1_000_000):
    out = []
    for bump in BUMPS:
        est = summarise(
            bump_samples(
                z, MARKET, MATURITY, call_payoff, parameter="spot", bump=bump, z_down=z_down
            )
        )
        out.append((abs(est.value - TRUE_DELTA), est.standard_error))
    return [e for e, _ in out], [s for _, s in out]


def test_without_common_random_numbers_the_sweep_is_u_shaped_as_predicted():
    """The U shape CLAUDE.md predicts, reproduced — in the regime it applies to.

    Differencing two independently noisy prices gives a numerator whose
    standard error barely depends on `h`, so dividing by `2h` amplifies it as
    `1/h`. That is the rising left arm of the U; truncation supplies the right.
    """
    # 4M paths, not 400k, and the reason is the shape itself. Without common
    # random numbers the noise arm scales as 1/(h*sqrt(N)) while truncation
    # scales as h^2 and does not shrink with N, so the minimum sits at
    # h ~ (c/sqrt(N))^(1/3). At 400k paths that is about 0.05 and the noise arm
    # still dominates the whole 1e-4..1e-1 grid, putting the minimum on the
    # endpoint. At 4M it moves to about 0.034 and the U is interior. The sweep
    # needs enough paths to see its own right arm.
    z = rng(666).standard_normal(4_000_000)
    z_independent = rng(667).standard_normal(4_000_000)
    errors, ses = _sweep(z, z_down=z_independent)

    # Standard error scales as 1/h: a 3x larger bump gives a 3x smaller error.
    assert ses[0] / ses[-1] > 100.0, f"no amplification: {ses}"
    for i in range(len(BUMPS) - 1):
        ratio = ses[i] / ses[i + 1]
        expected = BUMPS[i + 1] / BUMPS[i]
        assert ratio == pytest.approx(expected, rel=0.1), f"step {i}: {ratio} vs {expected}"

    best = int(np.argmin(errors))
    assert 0 < best < len(BUMPS) - 1, f"minimum at an endpoint: {dict(zip(BUMPS, errors))}"
    assert errors[0] > 10 * errors[best]  # steep left arm
    assert errors[-1] > errors[best]  # truncation right arm


def test_with_common_random_numbers_the_sweep_is_not_u_shaped():
    """The finding that contradicts the spec's expectation, asserted.

    Sharing draws makes the numerator shrink with `h` exactly as the signal
    does, so the quotient converges to the pathwise derivative and its variance
    stays bounded as `h -> 0`. The left arm of the U therefore does not exist:
    the standard error flattens onto the pathwise noise floor. What remains is
    a flat region followed by truncation — a hockey stick.

    Since correct practice *requires* common random numbers, this is the shape
    the repository actually obtains, and the report says so.
    """
    z = rng(668).standard_normal(400_000)
    errors, ses = _sweep(z)
    pathwise_floor = summarise(
        pathwise_european(z, MARKET, MATURITY, STRIKE)["delta"]
    ).standard_error

    # No amplification at all, and the floor is the pathwise standard error.
    assert ses[0] / ses[-1] < 1.5, f"unexpected amplification: {ses}"
    assert ses[0] == pytest.approx(pathwise_floor, rel=0.02)

    # Truncation still dominates at the largest bump.
    assert errors[-1] > 5 * min(errors)


def test_standard_error_of_the_bumped_delta_falls_as_the_bump_grows():
    """The noise half of the U, isolated from the truncation half."""
    z = rng(777).standard_normal(400_000)
    ses = [
        summarise(
            bump_samples(z, MARKET, MATURITY, call_payoff, parameter="spot", bump=b)
        ).standard_error
        for b in (1e-4, 1e-3, 1e-2)
    ]
    assert ses[0] > ses[1] > ses[2]


def test_bump_rejects_a_vol_bump_that_goes_non_positive():
    z = rng(1).standard_normal(100)
    with pytest.raises(ValueError, match="drives volatility non-positive"):
        bump_samples(z, MARKET, MATURITY, call_payoff, parameter="vol", bump=0.5)


def test_bump_rejects_an_unknown_parameter():
    z = rng(1).standard_normal(100)
    with pytest.raises(ValueError, match="spot.*vol"):
        bump_samples(z, MARKET, MATURITY, call_payoff, parameter="rho", bump=0.01)


# --------------------------------------------------------------------------
# GreekEstimate plumbing
# --------------------------------------------------------------------------
def test_greek_estimate_carries_its_standard_error():
    est = summarise(np.array([1.0, 2.0, 3.0]), "x")
    assert est.value == pytest.approx(2.0)
    assert est.n_samples == 3
    assert est.label == "x"
    assert est.agrees_with(2.0)
    assert not est.agrees_with(100.0)


def test_summarise_refuses_a_single_sample():
    with pytest.raises(ValueError, match="at least 2"):
        summarise(np.array([1.0]))


def test_central_difference_greek_labels_itself():
    z = rng(1).standard_normal(10_000)
    est = central_difference_greek(
        z, MARKET, MATURITY, call_payoff, parameter="spot", bump=0.01
    )
    assert "bump spot" in est.label
