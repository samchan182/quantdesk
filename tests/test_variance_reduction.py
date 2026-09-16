"""M3 acceptance — the variance reduction study.

Two required checks:
  1. all four arms agree on the price within two standard errors of the widest;
  2. the realised variance reduction from the control variate matches
     ``1 - rho^2`` computed on the same sample.

Plus the traps the milestone is really about: beta must not be 1, and beta must
not be fitted to the sample it is applied to.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.special import ndtr

from quantdesk.analytics import blackscholes as bs
from quantdesk.config import BASE_MARKET
from quantdesk.mc.estimators import PairedAccumulator, control_variate_samples
from quantdesk.mc.paths import simulate_paired
from quantdesk.products.vanilla import (
    arithmetic_asian_payoff,
    european_payoff,
    on_terminal_column,
)
from quantdesk.rng import rng

MARKET = BASE_MARKET
MATURITY = 1.0
STRIKE = 100.0
N_STEPS = 252


def _payoffs():
    return (
        arithmetic_asian_payoff(STRIKE, MARKET.rate, MATURITY),
        on_terminal_column(european_payoff(STRIKE, MARKET.rate, MATURITY)),
    )


def geometric_asian_call_closed_form(
    spot: float, strike: float, rate: float, div_yield: float, vol: float,
    maturity: float, n_obs: int,
) -> float:
    """Closed form for a discretely-monitored *geometric* average Asian call.

    An independent reference, and a stronger one than it looks. The geometric
    average of lognormals is itself lognormal, so this has an exact price,
    while the arithmetic average does not. Matching it requires the simulated
    path to have the right joint distribution across *all* 252 observation
    dates — its whole covariance structure — not merely the right terminal
    distribution, which is all the M2 checks pinned down.

    log G = (1/n) * sum_i log S(t_i) is normal with

        mean = log S0 + (r - q - sigma^2/2) * mean(t_i)
        var  = (sigma^2 / n^2) * sum_i sum_j min(t_i, t_j)
    """
    times = np.arange(1, n_obs + 1) * (maturity / n_obs)
    mu = np.log(spot) + (rate - div_yield - 0.5 * vol**2) * times.mean()
    var = (vol**2 / n_obs**2) * np.minimum.outer(times, times).sum()
    sd = np.sqrt(var)

    d1 = (mu - np.log(strike) + var) / sd
    d2 = d1 - sd
    return float(np.exp(-rate * maturity) * (np.exp(mu + 0.5 * var) * ndtr(d1) - strike * ndtr(d2)))


# --------------------------------------------------------------------------
# Acceptance 1 — the four arms agree
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def four_arms():
    """Run the study once at a reduced path count and share it across tests."""
    target, control = _payoffs()
    expected_control = bs.call_price(
        MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
    )

    def pilot(seed, antithetic):
        run = simulate_paired(
            seed=seed, market=MARKET, maturity=MATURITY, n_paths=20_000,
            target=target, control=control, n_steps=N_STEPS,
            chunk_paths=10_000, antithetic=antithetic,
        )
        return run.joint.beta_star

    beta_plain, beta_anti = pilot(900_001, False), pilot(900_002, True)

    def arm(seed, antithetic, beta):
        return simulate_paired(
            seed=seed, market=MARKET, maturity=MATURITY, n_paths=100_000,
            target=target, control=control, expected_control=expected_control,
            beta=beta, n_steps=N_STEPS, chunk_paths=10_000, antithetic=antithetic,
        )

    plain, anti = arm(5001, False, beta_plain), arm(5002, True, beta_anti)
    return {
        "A": (plain.joint.mean_x, plain.joint.standard_error_x),
        "B": (anti.joint.mean_x, anti.joint.standard_error_x),
        "C": (plain.controlled.mean, plain.controlled.standard_error),
        "D": (anti.controlled.mean, anti.controlled.standard_error),
        "plain": plain,
        "anti": anti,
        "expected_control": expected_control,
    }


def test_all_four_arms_agree_within_two_standard_errors_of_the_widest(four_arms):
    arms = {k: four_arms[k] for k in "ABCD"}
    widest = max(se for _, se in arms.values())

    for (n1, (e1, _)), (n2, (e2, _)) in itertools.combinations(arms.items(), 2):
        assert abs(e1 - e2) < 2.0 * widest, (
            f"{n1}={e1:.6f} and {n2}={e2:.6f} differ by more than 2 x {widest:.6f}"
        )


def test_every_arm_simulates_the_same_number_of_paths(four_arms):
    """'Equal total path count' is the premise of the comparison, so assert it."""
    assert four_arms["plain"].joint.n_paths == 100_000
    assert four_arms["anti"].joint.n_paths == 100_000
    # ... but the antithetic arms carry half the statistical samples.
    assert four_arms["plain"].joint.n_samples == 100_000
    assert four_arms["anti"].joint.n_samples == 50_000


def test_the_arms_are_ordered_as_theory_predicts(four_arms):
    se = {k: four_arms[k][1] for k in "ABCD"}
    assert se["D"] < se["C"] < se["A"]
    assert se["B"] < se["A"]


# --------------------------------------------------------------------------
# Acceptance 2 — realised reduction matches 1 - rho^2
# --------------------------------------------------------------------------
def test_realised_variance_reduction_matches_one_minus_rho_squared(four_arms):
    """Theory and measurement, on the same sample, to three decimal places."""
    for label, run, uncontrolled in (
        ("plain", four_arms["plain"], four_arms["A"][1]),
        ("antithetic", four_arms["anti"], four_arms["B"][1]),
    ):
        theory = run.joint.theoretical_variance_ratio  # 1 - rho^2
        realised = (run.controlled.standard_error / uncontrolled) ** 2
        assert realised == pytest.approx(theory, rel=2e-3), (
            f"{label}: realised {realised:.6f} vs theory {theory:.6f}"
        )


def test_correlation_is_high_but_not_perfect(four_arms):
    """If it were 1 the study would be vacuous; if it were 0 there'd be no gain."""
    rho = four_arms["plain"].joint.correlation
    assert 0.5 < rho < 0.99


# --------------------------------------------------------------------------
# The beta traps
# --------------------------------------------------------------------------
def test_optimal_beta_is_not_one(four_arms):
    """Hardcoding beta = 1 is trap 2, and here it would be badly wrong."""
    beta = four_arms["plain"].joint.beta_star
    assert 0.3 < beta < 0.7, f"beta* = {beta:.4f}"
    assert abs(beta - 1.0) > 0.25


def test_beta_of_one_is_measurably_worse_than_beta_star():
    """Quantify the trap rather than asserting it.

    Var(X - b(Y - E[Y])) = Var(X) - 2b*Cov + b^2*Var(Y), a parabola in b with
    its minimum at beta*. At b = 1, well to the right of a beta* near 0.46, the
    variance is substantially higher — and if beta* were below 0.5 the whole
    control variate would be *worse than useless* at b = 1.
    """
    target, control = _payoffs()
    run = simulate_paired(
        seed=4242, market=MARKET, maturity=MATURITY, n_paths=60_000,
        target=target, control=control, n_steps=N_STEPS, chunk_paths=10_000,
    )
    var_x, var_y, cov = run.joint.var_x, run.joint.var_y, run.joint.covariance
    beta_star = run.joint.beta_star

    def variance_at(b):
        return var_x - 2.0 * b * cov + b * b * var_y

    assert variance_at(beta_star) < variance_at(1.0)
    assert variance_at(beta_star) == pytest.approx(var_x * (1 - run.joint.correlation ** 2), rel=1e-9)
    # The parabola's minimum really is at beta*.
    for b in (beta_star - 0.2, beta_star - 0.05, beta_star + 0.05, beta_star + 0.2):
        assert variance_at(b) > variance_at(beta_star)

    # And the point that makes the trap dangerous rather than merely suboptimal:
    # beyond b = 2*beta*, the control variate increases variance.
    assert variance_at(2.0 * beta_star) == pytest.approx(var_x, rel=1e-9)


def test_control_variate_is_unbiased_for_any_fixed_beta():
    """E[X - b(Y - E[Y])] = E[X] for every fixed b, since E[Y - E[Y]] = 0.

    Checked at a deliberately terrible beta, because the property has nothing
    to do with beta being good.
    """
    target, control = _payoffs()
    expected_control = bs.call_price(
        MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
    )
    kwargs = dict(
        seed=606, market=MARKET, maturity=MATURITY, n_paths=200_000,
        target=target, control=control, n_steps=N_STEPS, chunk_paths=20_000,
        expected_control=expected_control,
    )
    good = simulate_paired(beta=0.46, **kwargs)
    awful = simulate_paired(beta=-3.0, **kwargs)

    plain_estimate = good.joint.mean_x
    assert good.controlled.mean == pytest.approx(
        plain_estimate, abs=3 * good.controlled.standard_error
    )
    assert awful.controlled.mean == pytest.approx(
        plain_estimate, abs=3 * awful.controlled.standard_error
    )
    # The bad beta is unbiased but far noisier — that is the whole cost of it.
    assert awful.controlled.standard_error > 3 * good.controlled.standard_error


def test_in_sample_beta_is_biased_and_the_bias_shrinks_as_one_over_n():
    """Why beta comes from an independent pilot (trap 3).

    Fitting beta to the same draws it is applied to correlates the coefficient
    with the residual it multiplies, leaving an O(1/N) bias. It is small, and
    it is real: measured here by repeating a small-sample experiment many times
    and comparing the average in-sample estimate against the average
    out-of-sample one, on the same draws.
    """
    market, T = MARKET, MATURITY
    exact_control_mean = bs.call_price(
        market.spot, STRIKE, market.rate, market.div_yield, market.vol, T
    )
    payoff_call = european_payoff(STRIKE, market.rate, T)
    payoff_put = european_payoff(STRIKE, market.rate, T, kind="put")

    def experiment(n_samples, n_trials, seed0):
        in_sample, out_sample = [], []
        for trial in range(n_trials):
            generator = rng(seed0 + trial)
            # Terminal-only: fast, and the bias argument is about estimation,
            # not about the product.
            from quantdesk.mc.paths import terminal_prices

            z = generator.standard_normal(n_samples)
            terminal = terminal_prices(z, market, T)
            x, y = payoff_put(terminal), payoff_call(terminal)

            beta_in = np.cov(x, y, ddof=1)[0, 1] / np.var(y, ddof=1)
            zp = generator.standard_normal(n_samples)
            tp = terminal_prices(zp, market, T)
            beta_out = (
                np.cov(payoff_put(tp), payoff_call(tp), ddof=1)[0, 1]
                / np.var(payoff_call(tp), ddof=1)
            )
            in_sample.append(np.mean(control_variate_samples(x, y, exact_control_mean, beta_in)))
            out_sample.append(np.mean(control_variate_samples(x, y, exact_control_mean, beta_out)))
        return float(np.mean(in_sample)), float(np.mean(out_sample))

    truth = bs.put_price(market.spot, STRIKE, market.rate, market.div_yield, market.vol, T)
    small_in, small_out = experiment(200, 4000, 70_000)
    large_in, large_out = experiment(2000, 4000, 80_000)

    bias_small = abs(small_in - truth)
    bias_large = abs(large_in - truth)

    # Out-of-sample beta is unbiased at both sizes; in-sample is not, and its
    # bias falls roughly as 1/N.
    assert abs(small_out - truth) < bias_small
    assert bias_large < bias_small
    assert bias_small / max(bias_large, 1e-12) > 2.0


def test_perfect_correlation_collapses_the_variance():
    """Sanity bound: a control identical to the target gives exactly zero variance.

    This is why the target of the study must not itself be a European. It also
    confirms the estimator recovers the known answer in the degenerate case.
    """
    generator = rng(31337)
    terminal = np.exp(generator.standard_normal(50_000))
    x = y = terminal
    acc = PairedAccumulator().update(x, y)
    assert acc.correlation == pytest.approx(1.0, abs=1e-12)
    assert acc.beta_star == pytest.approx(1.0, rel=1e-12)

    controlled = control_variate_samples(x, y, float(np.mean(y)), acc.beta_star)
    assert np.std(controlled, ddof=1) < 1e-12


# --------------------------------------------------------------------------
# The paired accumulator
# --------------------------------------------------------------------------
def test_paired_accumulator_matches_numpy_across_chunks():
    generator = rng(24)
    x = generator.standard_normal(80_000) * 3.0 + 5.0
    y = 0.7 * x + generator.standard_normal(80_000)

    acc = PairedAccumulator()
    for bx, by in zip(np.array_split(x, 11), np.array_split(y, 11), strict=True):
        acc.update(bx, by)

    assert acc.n_samples == x.size
    assert acc.mean_x == pytest.approx(x.mean(), rel=1e-13)
    assert acc.var_x == pytest.approx(np.var(x, ddof=1), rel=1e-12)
    assert acc.covariance == pytest.approx(np.cov(x, y, ddof=1)[0, 1], rel=1e-11)
    assert acc.correlation == pytest.approx(np.corrcoef(x, y)[0, 1], rel=1e-11)


def test_paired_accumulator_is_stable_with_a_huge_offset():
    """The co-moment must not be computed as E[XY] - E[X]E[Y] either."""
    generator = rng(25)
    x = 1e9 + generator.standard_normal(50_000)
    y = 1e9 + generator.standard_normal(50_000)
    acc = PairedAccumulator().update(x, y)
    assert acc.covariance == pytest.approx(np.cov(x, y, ddof=1)[0, 1], abs=1e-3)
    assert abs(acc.correlation) < 0.05  # independent by construction


def test_paired_accumulator_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="match in shape"):
        PairedAccumulator().update(np.zeros(3), np.zeros(4))


# --------------------------------------------------------------------------
# The product
# --------------------------------------------------------------------------
def test_asian_average_excludes_the_known_spot():
    payoff = arithmetic_asian_payoff(STRIKE, 0.0, 1.0)
    paths = np.array([[100.0, 200.0, 200.0]])
    # Averaging columns 1: gives 200; including column 0 would give 166.67.
    assert payoff(paths)[0] == pytest.approx(100.0)


def test_asian_price_sits_above_the_geometric_closed_form():
    """AM >= GM pathwise, so the arithmetic Asian call must be worth more.

    A genuine independent check on the path engine's full covariance structure:
    the geometric average has a closed form, and matching it needs the joint
    distribution across all 252 observation dates to be right, not just the
    terminal one.
    """
    geometric_truth = geometric_asian_call_closed_form(
        MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY, N_STEPS
    )

    def geometric_payoff(paths):
        gm = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
        return np.exp(-MARKET.rate * MATURITY) * np.maximum(gm - STRIKE, 0.0)

    arithmetic, _ = _payoffs()
    run = simulate_paired(
        seed=99, market=MARKET, maturity=MATURITY, n_paths=200_000,
        target=arithmetic, control=geometric_payoff, n_steps=N_STEPS,
        chunk_paths=20_000, antithetic=True,
    )
    mc_geometric, se_geo = run.joint.mean_y, np.sqrt(run.joint.var_y / run.joint.n_samples)

    # The engine reproduces the geometric closed form.
    assert abs(mc_geometric - geometric_truth) < 2.5 * se_geo, (
        f"MC geometric {mc_geometric:.6f} vs closed form {geometric_truth:.6f}, se {se_geo:.6f}"
    )
    # And AM >= GM makes the arithmetic price strictly higher.
    assert run.joint.mean_x > mc_geometric


def test_asian_is_worth_less_than_the_european_on_the_same_terms():
    """Averaging damps the effective volatility, so the Asian is cheaper."""
    target, control = _payoffs()
    run = simulate_paired(
        seed=1717, market=MARKET, maturity=MATURITY, n_paths=100_000,
        target=target, control=control, n_steps=N_STEPS, chunk_paths=20_000,
    )
    european_truth = bs.call_price(
        MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
    )
    assert run.joint.mean_x < european_truth


def test_asian_payoff_rejects_a_path_with_no_observations():
    payoff = arithmetic_asian_payoff(STRIKE, 0.0, 1.0)
    with pytest.raises(ValueError, match="at least one observation"):
        payoff(np.array([[100.0]]))


def test_simulate_paired_requires_beta_and_expectation_together():
    target, control = _payoffs()
    with pytest.raises(ValueError, match="together"):
        simulate_paired(
            seed=1, market=MARKET, maturity=MATURITY, n_paths=100,
            target=target, control=control, n_steps=4, beta=0.5,
        )
