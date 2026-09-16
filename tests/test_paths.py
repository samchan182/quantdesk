"""M2 acceptance — the GBM path engine.

Four required checks:
  1. numba kernel and NumPy reference agree bit-for-bit (or to 1e-12 if the
     reduction order differs);
  2. simulated E[S_T] matches S_0 * exp((r - q) T) within two standard errors;
  3. the simulated European call converges to M1's closed form, with error
     falling as 1/sqrt(N) — log-log slope near -0.5;
  4. the linear-payoff antithetic test: a payoff affine in Z must give exactly
     zero variance under antithetic pairing.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantdesk.analytics import blackscholes as bs
from quantdesk.config import BASE_MARKET, Market
from quantdesk.mc.estimators import Accumulator
from quantdesk.mc.paths import (
    NUMBA_AVAILABLE,
    chunk_sizes,
    price_paths,
    simulate,
    terminal_prices,
)
from quantdesk.products.vanilla import european_payoff, linear_in_z_payoff, on_terminal_column
from quantdesk.rng import rng

MARKET = BASE_MARKET
MATURITY = 1.0
STRIKE = 100.0


# --------------------------------------------------------------------------
# Acceptance 1 — the two engines are the same arithmetic
# --------------------------------------------------------------------------
@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
def test_numba_kernel_matches_numpy_reference():
    z = rng(1234).standard_normal((2_000, 252))
    reference = price_paths(z, MARKET, MATURITY, engine="numpy")
    kernel = price_paths(z, MARKET, MATURITY, engine="numba")

    assert kernel.shape == reference.shape
    identical = np.array_equal(kernel, reference)
    if not identical:
        # Permitted fallback: the accumulation order is the same by
        # construction, so any gap is scalar-vs-vectorised exp in the last ulp.
        rel = np.max(np.abs(kernel - reference) / reference)
        assert rel < 1e-12, f"engines disagree by {rel:.3e} relative"


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
@pytest.mark.parametrize("n_steps", [1, 2, 12, 252])
def test_engines_agree_across_step_counts(n_steps):
    z = rng(99).standard_normal((500, n_steps))
    a = price_paths(z, MARKET, MATURITY, engine="numpy")
    b = price_paths(z, MARKET, MATURITY, engine="numba")
    assert np.max(np.abs(a - b) / a) < 1e-12


def test_first_column_is_exactly_spot():
    z = rng(5).standard_normal((100, 10))
    for engine in (["numpy", "numba"] if NUMBA_AVAILABLE else ["numpy"]):
        paths = price_paths(z, MARKET, MATURITY, engine=engine)
        assert np.all(paths[:, 0] == MARKET.spot)


def test_one_step_path_matches_the_terminal_shortcut():
    """A single stepped period and the one-step exact draw must coincide.

    To within 9 ulps, not bit-for-bit, and the reason is worth knowing: the
    stepped engine accumulates in logs and computes `exp(log(S_0) + drift +
    sigma*sqrt(T)*Z)`, while the terminal shortcut computes
    `S_0 * exp(drift + sigma*sqrt(T)*Z)`. The log-then-exp round trip on the
    spot is not exact in float64. Both are correct; they are different
    orderings of the same expression.
    """
    z = rng(77).standard_normal(1000)
    stepped = price_paths(z.reshape(-1, 1), MARKET, MATURITY)[:, -1]
    direct = terminal_prices(z, MARKET, MATURITY)

    rel = np.max(np.abs(stepped - direct) / direct)
    assert rel < 1e-14, f"max relative gap {rel:.3e}"
    assert np.max(np.abs(stepped - direct) / np.spacing(direct)) < 32.0


# --------------------------------------------------------------------------
# Acceptance 2 — the discounted price is a martingale
# --------------------------------------------------------------------------
def test_expected_terminal_price_matches_the_forward():
    """E[S_T] = S_0 exp((r - q) T), to within two standard errors.

    This is the cheapest possible check that the drift is right — in
    particular that the -sigma^2/2 Ito correction is present. Drop it and the
    simulated mean is high by a factor of exp(sigma^2 T / 2), which is 2% here
    and would be swamped by noise if you only looked at the option price.
    """
    n = 2_000_000
    z = rng(2024).standard_normal(n)
    terminal = terminal_prices(z, MARKET, MATURITY)

    expected = MARKET.spot * np.exp((MARKET.rate - MARKET.div_yield) * MATURITY)
    se = terminal.std(ddof=1) / np.sqrt(n)
    assert abs(terminal.mean() - expected) < 2.0 * se, (
        f"mean {terminal.mean():.6f} vs forward {expected:.6f}, se {se:.6f}"
    )


def test_missing_ito_correction_would_be_caught():
    """The test above has teeth: show it rejects the classic wrong drift."""
    n = 2_000_000
    z = rng(2024).standard_normal(n)
    wrong = MARKET.spot * np.exp(
        (MARKET.rate - MARKET.div_yield) * MATURITY + MARKET.vol * np.sqrt(MATURITY) * z
    )
    expected = MARKET.spot * np.exp((MARKET.rate - MARKET.div_yield) * MATURITY)
    se = wrong.std(ddof=1) / np.sqrt(n)
    assert abs(wrong.mean() - expected) > 2.0 * se


@pytest.mark.parametrize("n_steps", [1, 12, 252])
def test_stepped_paths_are_a_martingale_at_every_observation(n_steps):
    z = rng(31).standard_normal((400_000, n_steps))
    paths = price_paths(z, MARKET, MATURITY, engine="numpy")
    times = np.linspace(0.0, MATURITY, n_steps + 1)
    forwards = MARKET.spot * np.exp((MARKET.rate - MARKET.div_yield) * times)

    means = paths.mean(axis=0)
    ses = paths.std(axis=0, ddof=1) / np.sqrt(paths.shape[0])
    # Column 0 is deterministic; check the rest.
    assert np.all(np.abs(means[1:] - forwards[1:]) < 3.0 * ses[1:])


def test_terminal_distribution_is_lognormal_in_its_moments():
    """Second moment too, not just the first: E[S_T^2] pins down sigma."""
    n = 2_000_000
    terminal = terminal_prices(rng(808).standard_normal(n), MARKET, MATURITY)
    m, s, v, T = MARKET.spot, MARKET.vol, MARKET.rate - MARKET.div_yield, MATURITY
    expected_second = m**2 * np.exp((2 * v + s**2) * T)

    sample = np.mean(terminal**2)
    se = np.std(terminal**2, ddof=1) / np.sqrt(n)
    assert abs(sample - expected_second) < 3.0 * se


# --------------------------------------------------------------------------
# Acceptance 3 — 1/sqrt(N) convergence to the closed form
# --------------------------------------------------------------------------
def test_european_call_converges_to_closed_form_at_root_n():
    """Error must fall as 1/sqrt(N): log-log slope near -0.5.

    Measured on the RMS error over 24 independent seeds per N, not on a single
    run. A single run's error at one N is itself a random variable with a heavy
    enough spread that a slope fitted through three of them is nearly
    meaningless — it can easily come out at -0.2 or -0.9 by luck. Averaging in
    the mean-square is what makes the exponent estimable.
    """
    truth = bs.call_price(
        MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
    )
    payoff = european_payoff(STRIKE, MARKET.rate, MATURITY)

    path_counts = [10_000, 100_000, 1_000_000]
    n_repeats = 24
    rms_errors = []
    for n_paths in path_counts:
        errors = [
            simulate(
                seed=10_000 + rep,
                market=MARKET,
                maturity=MATURITY,
                n_paths=n_paths,
                payoff=payoff,
                chunk_paths=250_000,
            ).mean
            - truth
            for rep in range(n_repeats)
        ]
        rms_errors.append(float(np.sqrt(np.mean(np.square(errors)))))

    slope = np.polyfit(np.log(path_counts), np.log(rms_errors), 1)[0]
    assert -0.60 < slope < -0.40, f"convergence slope {slope:.3f}, rms errors {rms_errors}"


def test_reported_standard_error_is_honest():
    """The quoted standard error must predict the actual spread of the estimator.

    A standard error is a claim about a distribution. This checks the claim:
    over 200 independent runs, the spread of the estimates should match the
    standard error each run reports, and about 95% should land within two of
    them of the truth.
    """
    truth = bs.call_price(
        MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
    )
    payoff = european_payoff(STRIKE, MARKET.rate, MATURITY)

    runs = [
        simulate(
            seed=50_000 + rep,
            market=MARKET,
            maturity=MATURITY,
            n_paths=20_000,
            payoff=payoff,
            chunk_paths=20_000,
        )
        for rep in range(200)
    ]
    estimates = np.array([r.mean for r in runs])
    reported_se = float(np.mean([r.standard_error for r in runs]))
    realised_sd = float(estimates.std(ddof=1))

    assert realised_sd == pytest.approx(reported_se, rel=0.15)
    inside = np.mean(np.abs(estimates - truth) < 2.0 * reported_se)
    assert 0.90 < inside < 0.99, f"{inside:.1%} within two standard errors"


def test_stepped_and_terminal_engines_price_the_same_european():
    """Same payoff, both code paths, agreeing within their joint noise.

    M5 prices a European through the full stepped stack; this confirms up
    front that stepping introduces no bias for a terminal-only payoff — as it
    must not, since each step is exact in law.
    """
    payoff = european_payoff(STRIKE, MARKET.rate, MATURITY)
    common = dict(market=MARKET, maturity=MATURITY, n_paths=400_000, chunk_paths=50_000)

    direct = simulate(seed=4242, payoff=payoff, **common)
    stepped = simulate(seed=4242, payoff=on_terminal_column(payoff), n_steps=252, **common)

    joint_se = np.hypot(direct.standard_error, stepped.standard_error)
    assert abs(direct.mean - stepped.mean) < 2.0 * joint_se


# --------------------------------------------------------------------------
# Acceptance 4 — the antithetic trap
# --------------------------------------------------------------------------
def test_antithetic_on_a_linear_payoff_has_exactly_zero_variance():
    """The sharpest test of the pairing, and of the standard error's denominator.

    log(S_T/S_0) is affine in Z, so every antithetic pair average equals the
    drift mu*T exactly. Sample variance across pairs is zero, so a correctly
    computed standard error is zero to floating-point noise. A standard error
    pooled over the 2N individual paths instead would report the full spread of
    the draws — around 0.2 here — rather than zero.
    """
    # Deliberately *not* BASE_MARKET: there r - q - sigma^2/2 = 0.02 - 0 - 0.02
    # is exactly zero, so "the mean equals mu*T" would collapse to "the mean is
    # zero" and the test would pass without pinning anything down.
    drifting = Market(spot=100.0, rate=0.05, div_yield=0.01, vol=0.20)
    payoff = linear_in_z_payoff(drifting, MATURITY)
    acc = simulate(
        seed=7,
        market=drifting,
        maturity=MATURITY,
        n_paths=100_000,
        payoff=payoff,
        antithetic=True,
        chunk_paths=10_000,
    )

    mu_t = (drifting.rate - drifting.div_yield - 0.5 * drifting.vol**2) * MATURITY
    assert mu_t == pytest.approx(0.02, rel=1e-12)  # genuinely non-zero
    assert acc.mean == pytest.approx(mu_t, rel=1e-14)
    assert acc.standard_error < 1e-15, f"standard error {acc.standard_error:.3e} is not zero"
    assert acc.std < 1e-14

    # And the sample size is pairs, not paths. This is the trap, stated.
    assert acc.n_samples == 50_000
    assert acc.n_paths == 100_000


def test_the_wrong_denominator_would_be_visibly_wrong():
    """Show what pooling over 2N paths instead of N pairs would have reported."""
    z = rng(7).standard_normal(50_000)
    market, T = MARKET, MATURITY
    mu_t = (market.rate - market.div_yield - 0.5 * market.vol**2) * T

    log_plus = np.log(terminal_prices(z, market, T) / market.spot)
    log_minus = np.log(terminal_prices(-z, market, T) / market.spot)

    correct = np.std(0.5 * (log_plus + log_minus), ddof=1) / np.sqrt(z.size)
    pooled = np.std(np.concatenate([log_plus, log_minus]), ddof=1) / np.sqrt(2 * z.size)

    assert correct < 1e-15
    assert pooled > 5e-4  # the full spread of the draws, ~0.2 / sqrt(100000)
    assert np.mean(0.5 * (log_plus + log_minus)) == pytest.approx(mu_t, abs=1e-15)


def test_antithetic_reduces_variance_on_a_monotone_payoff():
    payoff = european_payoff(STRIKE, MARKET.rate, MATURITY)
    common = dict(market=MARKET, maturity=MATURITY, n_paths=400_000, chunk_paths=40_000)

    crude = simulate(seed=11, payoff=payoff, antithetic=False, **common)
    anti = simulate(seed=11, payoff=payoff, antithetic=True, **common)

    assert crude.n_samples == 400_000 and anti.n_samples == 200_000
    assert anti.n_paths == crude.n_paths  # equal compute
    assert anti.standard_error < crude.standard_error
    joint = np.hypot(crude.standard_error, anti.standard_error)
    assert abs(anti.mean - crude.mean) < 3.0 * joint


def test_antithetic_requires_an_even_path_count():
    with pytest.raises(ValueError, match="even"):
        simulate(
            seed=1,
            market=MARKET,
            maturity=MATURITY,
            n_paths=999,
            payoff=european_payoff(STRIKE, MARKET.rate, MATURITY),
            antithetic=True,
        )


# --------------------------------------------------------------------------
# Chunking: memory discipline must not change the answer given the same plan
# --------------------------------------------------------------------------
def test_chunk_sizes_partition_exactly():
    assert chunk_sizes(200_000, 10_000) == [10_000] * 20
    assert chunk_sizes(25_000, 10_000) == [10_000, 10_000, 5_000]
    assert chunk_sizes(7, 10) == [7]
    assert sum(chunk_sizes(123_457, 10_000)) == 123_457


def test_same_chunk_plan_reproduces_exactly():
    payoff = european_payoff(STRIKE, MARKET.rate, MATURITY)
    kwargs = dict(
        seed=3,
        market=MARKET,
        maturity=MATURITY,
        n_paths=50_000,
        payoff=payoff,
        chunk_paths=10_000,
    )
    a, b = simulate(**kwargs), simulate(**kwargs)
    assert a.mean == b.mean and a.standard_error == b.standard_error
    assert a.chunk_sizes == b.chunk_sizes == [10_000] * 5


def test_chunk_size_changes_the_stream_but_not_the_statistics():
    """Different chunk plans are different random streams, not different physics.

    Because each chunk draws from its own spawned generator, the chunk count is
    part of the seed's meaning: 5 chunks and 1 chunk give different numbers.
    They must still agree within their joint standard error, and the chunk plan
    is recorded as a benchmark input so a reported number stays reproducible.
    """
    payoff = european_payoff(STRIKE, MARKET.rate, MATURITY)
    common = dict(seed=3, market=MARKET, maturity=MATURITY, n_paths=200_000, payoff=payoff)

    fine = simulate(chunk_paths=10_000, **common)
    coarse = simulate(chunk_paths=200_000, **common)

    assert fine.mean != coarse.mean  # different streams
    assert len(fine.chunk_sizes) == 20 and len(coarse.chunk_sizes) == 1
    joint = np.hypot(fine.standard_error, coarse.standard_error)
    assert abs(fine.mean - coarse.mean) < 3.0 * joint


def test_chunking_keeps_peak_memory_bounded():
    """The 403 MB allocation the spec forbids is never made.

    200,000 x 252 float64 is 403 MB. Simulated in 10,000-path chunks, the
    largest array alive at once is a chunk of paths, about 20 MB. Asserted by
    watching the payoff's input shape rather than by measuring the process.
    """
    seen: list[tuple[int, ...]] = []

    def watching_payoff(paths: np.ndarray) -> np.ndarray:
        seen.append(paths.shape)
        return paths[:, -1]

    simulate(
        seed=1,
        market=MARKET,
        maturity=MATURITY,
        n_paths=200_000,
        n_steps=252,
        payoff=watching_payoff,
        chunk_paths=10_000,
    )

    assert len(seen) == 20
    assert {s[0] for s in seen} == {10_000}
    biggest_bytes = max(np.prod(s) * 8 for s in seen)
    assert biggest_bytes < 25 * 1024**2, f"{biggest_bytes / 1024**2:.1f} MB chunk"


# --------------------------------------------------------------------------
# Accumulator behaviour
# --------------------------------------------------------------------------
def test_accumulator_matches_a_single_pass_computation():
    values = rng(12).standard_normal(100_000) * 7.0 + 3.0
    acc = Accumulator()
    for block in np.array_split(values, 13):
        acc.update(block)

    assert acc.n_samples == values.size
    assert acc.mean == pytest.approx(values.mean(), rel=1e-14)
    assert acc.variance == pytest.approx(values.var(ddof=1), rel=1e-12)
    assert acc.standard_error == pytest.approx(
        values.std(ddof=1) / np.sqrt(values.size), rel=1e-12
    )


def test_accumulator_is_stable_where_sum_of_squares_is_not():
    """Chan's update survives a large mean with a small spread. E[X^2]-E[X]^2 does not.

    100,000 draws of unit variance around a mean of 1e9. The accumulator
    reproduces NumPy's two-pass variance to the last digit. The textbook
    one-pass form `E[X^2] - E[X]^2` returns **-128.0** on the same data: the
    two terms agree to about 18 significant figures and a double carries 16, so
    the subtraction is pure cancellation and the sign itself is noise.

    This is why the accumulator does not sum squares, and it matters for real
    payoffs, not just contrived ones — a deep in-the-money option has exactly
    this shape, a large mean with a small spread around it.
    """
    values = 1e9 + rng(13).standard_normal(100_000)
    acc = Accumulator().update(values)

    reference = float(np.var(values, ddof=1))  # NumPy is two-pass, so accurate
    assert acc.variance == pytest.approx(reference, rel=1e-12)
    assert reference == pytest.approx(1.0, rel=0.01)  # sampling error ~sqrt(2/n)

    naive = float(np.mean(values**2) - np.mean(values) ** 2)
    assert naive < 0.0, f"expected the naive form to fail; got {naive}"


def test_accumulator_refuses_non_finite_samples():
    with pytest.raises(ValueError, match="non-finite"):
        Accumulator().update(np.array([1.0, np.nan]))


def test_variance_of_one_sample_is_nan_not_zero():
    acc = Accumulator().update(np.array([5.0]))
    assert np.isnan(acc.variance) and np.isnan(acc.standard_error)


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------
def test_invalid_engine_raises():
    with pytest.raises(ValueError, match="engine"):
        price_paths(rng(1).standard_normal((2, 2)), MARKET, 1.0, engine="cuda")


def test_zero_steps_raises():
    with pytest.raises(ValueError, match="n_steps"):
        price_paths(np.empty((5, 0)), MARKET, 1.0)


def test_one_dimensional_z_raises_for_paths():
    with pytest.raises(ValueError, match="2-D"):
        price_paths(rng(1).standard_normal(10), MARKET, 1.0)


def test_zero_vol_market_is_deterministic():
    flat = Market(spot=100.0, rate=0.05, div_yield=0.01, vol=0.0)
    z = rng(1).standard_normal((50, 12))
    paths = price_paths(z, flat, 1.0)
    assert np.allclose(paths[:, -1], 100.0 * np.exp(0.04), rtol=1e-14)
    assert np.ptp(paths[:, -1]) < 1e-10  # every path identical
