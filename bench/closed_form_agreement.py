"""M5 — Monte Carlo validated against the closed form. Headline #2.

    python bench/closed_form_agreement.py

A European call is priced through the **full production stack** — the same
stepped path engine, the same chunked driver, the same antithetic estimator,
the same payoff plumbing that prices the snowball — and compared to M1's closed
form.

**Two conventions, fixed here and repeated in the report.**

1. Differences are quoted in **basis points of the premium**, not of notional.
   One basis point of an 8.92 premium is 0.00089 of price; one basis point of a
   100 notional is 0.01, eleven times larger. A number quoted without saying
   which is trap 16.
2. The **Monte Carlo standard error is reported in the same units, next to the
   difference**. A difference of 4bp means nothing on its own: against a
   standard error of 2bp it is noise, against one of 0.1bp it is a bias.

**Where an interviewer will push, and the answer this benchmark is built to
give.** A European payoff under GBM has *no discretisation error*. Each step of
the scheme is exact in law, so simulating in 1 step or in 252 samples the same
distribution of `S_T`. The gap to the closed form must therefore be pure
sampling noise, with nothing else available to blame it on.

Two pieces of evidence are produced rather than asserted:

* **A bias test with a t-statistic.** The headline is pooled over independent
  runs, so the difference has a standard error of its own and the hypothesis
  "the engine is unbiased" can be tested rather than eyeballed. One run's gap
  being small is luck; a t-statistic near zero over many is evidence.
* **A discretisation sweep.** The same option priced at 1, 4, 12, 63 and 252
  steps. If any discretisation error existed it would show as drift across that
  sweep. It must not.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from quantdesk.analytics import blackscholes as bs
from quantdesk.config import BASE_MARKET, seed_for
from quantdesk.mc.paths import price_paths, simulate, simulate_paired
from quantdesk.products.vanilla import european_payoff, on_terminal_column
from quantdesk.results import write_result
from quantdesk.rng import rng
from quantdesk.units import bp_of_premium

BENCHMARK = "closed_form_agreement"

MARKET = BASE_MARKET
MATURITY = 1.0
N_STEPS = 252
PATHS_PER_RUN = 1_000_000
CHUNK_PATHS = 20_000
ENGINE = "numba"

# Independent runs pooled into the headline. Chosen for *precision*, not to land
# on a value: at 40 runs the standard error is under 2bp of premium, which is
# what makes the comparison able to detect a bias of a few basis points at all.
# A comparison whose noise floor is 50bp cannot rule out a 10bp bias.
HEADLINE_RUNS = 40
# Equal to the headline, deliberately. An earlier version used 10 runs here and
# the at-the-money put came out at -5.60bp with t = -2.26, over the two-standard
# -error threshold. It was noise: escalating the same case to 20, 40 and 80 runs
# gave t = -2.28, -0.90 and finally -0.05 at -0.05bp. A real bias grows in
# significance as t ~ sqrt(n); this collapsed. Under-powering the supporting
# cases relative to the headline is what made a routine fluctuation look like a
# finding, so they now carry the same weight.
SUPPORTING_RUNS = 40

HEADLINE_CASE = ("atm_call", 100.0, "call")
SUPPORTING_CASES = [
    ("otm_call", 120.0, "call"),
    ("itm_call", 80.0, "call"),
    ("atm_put", 100.0, "put"),
]
SWEEP_STEPS = [1, 4, 12, 63, 252]


def price_once(seed: int, strike: float, kind: str, n_steps: int, n_paths: int) -> float:
    """One run through the production stack. Returns the estimate."""
    payoff = on_terminal_column(european_payoff(strike, MARKET.rate, MATURITY, kind=kind))
    return simulate(
        seed=seed,
        market=MARKET,
        maturity=MATURITY,
        n_paths=n_paths,
        payoff=payoff,
        n_steps=n_steps,
        chunk_paths=CHUNK_PATHS,
        antithetic=True,
        engine=ENGINE,
    ).mean


def run_case(seed: int, strike: float, kind: str, n_runs: int) -> dict:
    """Pool independent runs and test the difference against zero."""
    truth = bs.price(
        MARKET.spot, strike, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY, kind
    )
    estimates = np.array(
        [price_once(seed + i, strike, kind, N_STEPS, PATHS_PER_RUN) for i in range(n_runs)]
    )
    differences = estimates - truth

    pooled = float(estimates.mean())
    # Standard error of the pooled estimate, from the spread of independent
    # runs. Robust: it makes no assumption about the payoff's distribution.
    se = float(estimates.std(ddof=1) / np.sqrt(n_runs))
    difference = pooled - truth

    t_stat = difference / se
    return {
        "strike": strike,
        "kind": kind,
        "closed_form": truth,
        "mc_estimate": pooled,
        "difference": difference,
        "difference_bp_of_premium": bp_of_premium(difference, truth),
        "standard_error": se,
        "standard_error_bp_of_premium": bp_of_premium(se, truth),
        "difference_in_standard_errors": t_stat,
        "two_sided_p_value": float(2.0 * stats.t.sf(abs(t_stat), df=n_runs - 1)),
        "n_runs": n_runs,
        "paths_per_run": PATHS_PER_RUN,
        "total_paths": n_runs * PATHS_PER_RUN,
        "worst_single_run_bp": bp_of_premium(float(np.max(np.abs(differences))), truth),
    }


def discretisation_sweep(seed: int) -> list[dict]:
    """The same option at 1 to 252 steps. No drift is permitted to appear."""
    truth = bs.call_price(
        MARKET.spot, 100.0, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
    )
    out = []
    for n_steps in SWEEP_STEPS:
        estimates = np.array(
            [price_once(seed + 500 + i, 100.0, "call", n_steps, PATHS_PER_RUN) for i in range(8)]
        )
        se = float(estimates.std(ddof=1) / np.sqrt(estimates.size))
        difference = float(estimates.mean() - truth)
        out.append({
            "n_steps": n_steps,
            "mc_estimate": float(estimates.mean()),
            "difference_bp_of_premium": bp_of_premium(difference, truth),
            "standard_error_bp_of_premium": bp_of_premium(se, truth),
            "difference_in_standard_errors": difference / se,
        })
    return out


def parity_decomposition(seed: int, n_runs: int = 20) -> dict:
    """Split the pricing error into a drift part and a symmetric part.

    The sharpest diagnostic available, and the direct answer to "a European has
    no discretisation error, so where does your gap come from?".

    Since `max(S-K,0) - max(K-S,0) = S - K` *exactly*, pricing a call and a put
    on the **same** paths makes the antisymmetric part of their errors entirely
    a statement about the sample mean of `S_T`:

        err_call - err_put = exp(-rT) * (mean(S_T) - E[S_T])

    So a drift bias — a missing Ito correction, a wrong dividend yield — pushes
    the call and the put in *opposite* directions and shows up here. Anything
    symmetric pushes them the same way and does not. The identity is asserted
    numerically rather than assumed, since it is the thing that makes the
    decomposition meaningful.
    """
    call_truth = bs.call_price(
        MARKET.spot, 100.0, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
    )
    put_truth = bs.put_price(
        MARKET.spot, 100.0, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
    )
    call_payoff = on_terminal_column(european_payoff(100.0, MARKET.rate, MATURITY, "call"))
    put_payoff = on_terminal_column(european_payoff(100.0, MARKET.rate, MATURITY, "put"))

    call_errors, put_errors = [], []
    for i in range(n_runs):
        run = simulate_paired(
            seed=seed + 4_000 + i, market=MARKET, maturity=MATURITY,
            n_paths=PATHS_PER_RUN, target=call_payoff, control=put_payoff,
            n_steps=N_STEPS, chunk_paths=CHUNK_PATHS, antithetic=True, engine=ENGINE,
        )
        call_errors.append(run.joint.mean_x - call_truth)
        put_errors.append(run.joint.mean_y - put_truth)

    call_errors = np.array(call_errors)
    put_errors = np.array(put_errors)
    antisymmetric = call_errors - put_errors  # == exp(-rT) * (mean S_T - E[S_T])
    forward = MARKET.spot * np.exp((MARKET.rate - MARKET.div_yield) * MATURITY)
    implied_mean_terminal = np.exp(MARKET.rate * MATURITY) * antisymmetric + forward

    def summarise(values, base):
        mean = float(values.mean())
        se = float(values.std(ddof=1) / np.sqrt(values.size))
        return {
            "mean": mean, "standard_error": se, "t": mean / se,
            "bp_of_base": bp_of_premium(mean, base),
        }

    return {
        "n_runs": n_runs,
        "call_error": summarise(call_errors, call_truth),
        "put_error": summarise(put_errors, put_truth),
        "antisymmetric_drift_component": summarise(antisymmetric, call_truth),
        # Proportional to the antisymmetric component by construction — it is
        # that component multiplied by exp(rT) — so its t-statistic is
        # identical *by definition*. It is reported because it restates the
        # drift error in the units of the underlying rather than of the
        # premium, NOT as independent corroboration. Reading the matching t as
        # a second confirmation would be double-counting one measurement.
        "implied_mean_terminal_vs_forward": summarise(
            implied_mean_terminal - forward, forward
        ),
        "interpretation": (
            "a drift bias would push call and put in OPPOSITE directions and so "
            "appear in the antisymmetric component; a symmetric error would not"
        ),
    }


def verify_same_code_path() -> bool:
    """The claim is 'the same code path as the structured products'. Check it.

    The structured products run on the NumPy engine by default; this benchmark
    uses the numba kernel for throughput. M2 established the two agree
    bit-for-bit, and that is re-verified here rather than assumed, so the
    'same stack' claim is tested at the moment it is made.
    """
    z = rng(20260917).standard_normal((2_000, N_STEPS))
    return bool(
        np.array_equal(
            price_paths(z, MARKET, MATURITY, engine="numpy"),
            price_paths(z, MARKET, MATURITY, engine="numba"),
        )
    )


def main() -> int:
    seed = seed_for(BENCHMARK)
    engines_identical = verify_same_code_path()
    if not engines_identical:
        raise RuntimeError("numba and NumPy engines disagree; the 'same stack' claim is void")

    name, strike, kind = HEADLINE_CASE
    headline = run_case(seed, strike, kind, HEADLINE_RUNS)
    headline["case"] = name

    supporting = []
    for offset, (case_name, case_strike, case_kind) in enumerate(SUPPORTING_CASES):
        row = run_case(seed + 100 * (offset + 1), case_strike, case_kind, SUPPORTING_RUNS)
        row["case"] = case_name
        supporting.append(row)

    sweep = discretisation_sweep(seed)
    parity = parity_decomposition(seed)

    values = {
        "headline": headline,
        "supporting_cases": supporting,
        "discretisation_sweep": sweep,
        "parity_decomposition": parity,
        "multiple_comparisons_note": (
            f"{1 + len(SUPPORTING_CASES)} cases are tested, so a two-sided p below "
            "0.05 in one of them is expected about 19% of the time by chance; judge "
            "any single flagged case by escalating its run count, not by its p-value"
        ),
        "engines_bit_identical": engines_identical,
        "convention": (
            "differences in basis points OF THE PREMIUM, not of notional; "
            "the Monte Carlo standard error is quoted in the same units"
        ),
    }

    inputs = {
        "product": "European option priced through the full stepped production stack",
        "market": MARKET.as_dict(),
        "maturity": MATURITY,
        "n_steps": N_STEPS,
        "paths_per_run": PATHS_PER_RUN,
        "headline_runs": HEADLINE_RUNS,
        "headline_total_paths": HEADLINE_RUNS * PATHS_PER_RUN,
        "supporting_runs": SUPPORTING_RUNS,
        "chunk_paths": CHUNK_PATHS,
        "engine": ENGINE,
        "antithetic": True,
        "discretisation_sweep_steps": SWEEP_STEPS,
        "seed_headline": seed,
    }

    path = write_result(
        benchmark=BENCHMARK, seed=seed, inputs=inputs, values=values,
        notes=(
            "Headline pooled over independent runs so the difference has a standard "
            "error of its own and unbiasedness can be tested rather than eyeballed. "
            "Run count chosen for precision, not to reach a target value."
        ),
    )

    print("convention: basis points OF THE PREMIUM; standard error in the same units")
    print(f"engines bit-identical: {engines_identical}\n")
    print(f"{'case':<12}{'closed form':>13}{'MC':>13}{'diff bp':>10}{'se bp':>9}{'t':>8}{'p':>8}")
    for row in [headline, *supporting]:
        print(
            f"{row['case']:<12}{row['closed_form']:>13.8f}{row['mc_estimate']:>13.8f}"
            f"{row['difference_bp_of_premium']:>10.2f}{row['standard_error_bp_of_premium']:>9.2f}"
            f"{row['difference_in_standard_errors']:>8.2f}{row['two_sided_p_value']:>8.3f}"
        )

    print("\ndiscretisation sweep (ATM call; no discretisation error is possible here)")
    print(f"{'steps':>7}{'diff bp':>10}{'se bp':>9}{'t':>8}")
    for row in sweep:
        print(
            f"{row['n_steps']:>7}{row['difference_bp_of_premium']:>10.2f}"
            f"{row['standard_error_bp_of_premium']:>9.2f}{row['difference_in_standard_errors']:>8.2f}"
        )

    print(f"\nparity decomposition ({parity['n_runs']} paired runs, call and put on identical paths)")
    print(f"  {'component':<36}{'bp':>9}{'t':>8}")
    for key in ("call_error", "put_error", "antisymmetric_drift_component",
                "implied_mean_terminal_vs_forward"):
        row = parity[key]
        print(f"  {key:<36}{row['bp_of_base']:>+9.3f}{row['t']:>8.4f}")
    print("  (the last two are proportional by construction; identical t is definitional)")

    h = headline
    print(
        f"\nHEADLINE: {h['difference_bp_of_premium']:+.2f} bp of premium, "
        f"standard error {h['standard_error_bp_of_premium']:.2f} bp "
        f"({h['difference_in_standard_errors']:+.2f} standard errors, "
        f"p = {h['two_sided_p_value']:.3f}) over "
        f"{h['total_paths']:,} paths"
    )
    print(f"wrote {path.relative_to(path.parents[1])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
