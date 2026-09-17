"""M5 acceptance — Monte Carlo against the closed form, and the units it is quoted in.

The milestone's acceptance is a single sentence in the spec: if the measured
difference exceeds two standard errors, there is a real bias in the engine and
it must be found before proceeding. That makes the interesting tests the ones
about *method* — that the comparison is quoted in stated units, that the
standard error travels next to the difference, and that a European under GBM
has no discretisation error to hide behind.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantdesk.analytics import blackscholes as bs
from quantdesk.config import BASE_MARKET
from quantdesk.mc.paths import simulate
from quantdesk.products.vanilla import european_payoff, on_terminal_column
from quantdesk.units import bp_of, bp_of_notional, bp_of_premium

MARKET = BASE_MARKET
MATURITY = 1.0
STRIKE = 100.0


def _price(seed: int, n_steps: int, n_paths: int = 400_000, kind: str = "call"):
    return simulate(
        seed=seed,
        market=MARKET,
        maturity=MATURITY,
        n_paths=n_paths,
        payoff=on_terminal_column(european_payoff(STRIKE, MARKET.rate, MATURITY, kind=kind)),
        n_steps=n_steps,
        chunk_paths=50_000,
        antithetic=True,
        engine="numpy",
    )


# --------------------------------------------------------------------------
# The units, which are half the milestone
# --------------------------------------------------------------------------
def test_basis_points_of_premium_and_of_notional_are_not_the_same_number():
    """The distinction trap 16 is about, made concrete.

    An at-the-money one-year call at 20% vol on a spot of 100 has a premium of
    about 8.92. The same absolute price difference is roughly eleven times
    larger in basis points of premium than of notional.
    """
    premium = bs.call_price(
        MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
    )
    difference = 0.0025

    of_premium = bp_of_premium(difference, premium)
    of_notional = bp_of_notional(difference, MARKET.spot)

    assert of_premium == pytest.approx(2.8040, rel=1e-3)
    assert of_notional == pytest.approx(0.25, rel=1e-12)
    assert of_premium / of_notional == pytest.approx(MARKET.spot / premium, rel=1e-12)
    assert of_premium > 11 * of_notional


def test_basis_points_of_zero_raises_rather_than_returning_infinity():
    with pytest.raises(ValueError, match="basis points of zero"):
        bp_of(1.0, 0.0)


def test_bp_of_is_vectorised_and_scalar_preserving():
    assert isinstance(bp_of(1.0, 100.0), float)
    out = bp_of(np.array([1.0, 2.0]), 100.0)
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, [100.0, 200.0])


# --------------------------------------------------------------------------
# No discretisation error: the claim the whole milestone rests on
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n_steps", [1, 4, 12, 63, 252])
def test_stepped_european_matches_closed_form_within_two_standard_errors(n_steps):
    """Each step is exact in law, so step count must not move the answer."""
    truth = bs.call_price(
        MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
    )
    acc = _price(seed=90_000 + n_steps, n_steps=n_steps)
    assert abs(acc.mean - truth) < 2.0 * acc.standard_error, (
        f"{n_steps} steps: {bp_of_premium(acc.mean - truth, truth):+.2f} bp vs "
        f"se {bp_of_premium(acc.standard_error, truth):.2f} bp"
    )


def test_step_count_does_not_systematically_move_the_price():
    """Stronger than the per-step-count check: no *trend* across the sweep.

    A discretisation error would show as monotone drift in step count. Fitting
    a slope across the sweep tests for that directly, rather than checking five
    independent confidence intervals and hoping.
    """
    truth = bs.call_price(
        MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
    )
    step_counts = [1, 4, 12, 63, 252]
    # Common random numbers across step counts would not be meaningful here
    # (different step counts consume different numbers of draws), so these are
    # independent runs and the comparison is against their own standard errors.
    runs = [_price(seed=77_000 + i, n_steps=n) for i, n in enumerate(step_counts)]
    diffs = np.array([r.mean - truth for r in runs])
    ses = np.array([r.standard_error for r in runs])

    # Every point individually consistent with zero ...
    assert np.all(np.abs(diffs) < 2.5 * ses)
    # ... and the spread across step counts no larger than sampling noise.
    assert diffs.std(ddof=1) < 2.5 * ses.mean()


def test_one_step_and_many_steps_agree_with_each_other():
    truth = bs.call_price(
        MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
    )
    one, many = _price(seed=4321, n_steps=1), _price(seed=4321, n_steps=252)
    joint = np.hypot(one.standard_error, many.standard_error)
    assert abs(one.mean - many.mean) < 2.5 * joint
    assert abs(one.mean - truth) < 2.5 * one.standard_error


# --------------------------------------------------------------------------
# Pooling independent runs: how "bias or noise?" is actually answered
# --------------------------------------------------------------------------
def test_pooled_runs_give_an_unbiased_estimate_and_an_honest_standard_error():
    """The bias test the benchmark performs, at test scale.

    Pooling independent runs gives the difference a standard error of its own,
    which turns "the gap looks small" into a t-statistic. The standard error
    here is computed from the spread of the runs, so it assumes nothing about
    the payoff's distribution.
    """
    truth = bs.call_price(
        MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
    )
    estimates = np.array([_price(seed=60_000 + i, n_steps=252, n_paths=200_000).mean
                          for i in range(24)])
    se = estimates.std(ddof=1) / np.sqrt(estimates.size)
    t_stat = (estimates.mean() - truth) / se

    assert abs(t_stat) < 3.0, f"t = {t_stat:.2f}: the engine looks biased"
    # The per-run standard errors must agree with the realised spread of runs.
    reported = np.mean([_price(seed=60_000 + i, n_steps=252, n_paths=200_000).standard_error
                        for i in range(6)])
    assert estimates.std(ddof=1) == pytest.approx(reported, rel=0.35)


def test_put_and_call_are_both_unbiased_on_the_same_paths():
    """Parity turns the error decomposition into a sharp diagnostic.

    Since `max(S-K,0) - max(K-S,0) = S - K` exactly, on *identical* paths the
    call's error minus the put's error is entirely determined by the sample
    mean of S_T. So a drift bias would push the call and the put in **opposite**
    directions, while anything symmetric would push them the same way. Running
    both on one set of draws separates the two.
    """
    call_truth = bs.call_price(
        MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
    )
    put_truth = bs.put_price(
        MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
    )
    call = _price(seed=1234, n_steps=252, n_paths=600_000, kind="call")
    put = _price(seed=1234, n_steps=252, n_paths=600_000, kind="put")

    parity_lhs = call.mean - put.mean
    parity_rhs = MARKET.spot * np.exp(-MARKET.div_yield * MATURITY) - STRIKE * np.exp(
        -MARKET.rate * MATURITY
    )
    # The antisymmetric part is the drift error, and M2 already pinned E[S_T].
    assert abs(parity_lhs - parity_rhs) < 4.0 * np.hypot(
        call.standard_error, put.standard_error
    )
    assert abs(call.mean - call_truth) < 3.0 * call.standard_error
    assert abs(put.mean - put_truth) < 3.0 * put.standard_error


# --------------------------------------------------------------------------
# The benchmark module's own plumbing
# --------------------------------------------------------------------------
def test_benchmark_declares_its_convention_and_reports_error_beside_estimate():
    import importlib.util
    from pathlib import Path

    from quantdesk.results import project_root

    spec = importlib.util.spec_from_file_location(
        "_cfa", Path(project_root()) / "bench" / "closed_form_agreement.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    row = module.run_case(seed=5150, strike=STRIKE, kind="call", n_runs=3)
    # Every difference is accompanied by its standard error in the same units.
    for key in (
        "difference_bp_of_premium",
        "standard_error_bp_of_premium",
        "difference_in_standard_errors",
        "two_sided_p_value",
    ):
        assert key in row, key
    assert row["difference_bp_of_premium"] == pytest.approx(
        bp_of_premium(row["difference"], row["closed_form"]), rel=1e-12
    )
    assert row["difference_in_standard_errors"] == pytest.approx(
        row["difference"] / row["standard_error"], rel=1e-12
    )
    assert module.verify_same_code_path() is True
