"""M6 — pathwise versus bump-and-revalue. Headline #3.

    python bench/greeks.py

Four things are measured, and the third is the one an interviewer will press on.

**1. Agreement on the vanilla**, where M1 gives the true delta and vega, so both
Monte Carlo estimators can be scored against an exact answer rather than
against each other. Differences are reported as absolute, and the report says
what quantity they are absolute in: delta is dimensionless and lives in [0,1],
so 1e-4 on delta is a far tighter claim than 1e-4 on a vega of order 37.

**2. The pathwise method failing on a discontinuous payoff.** For a digital, the
naive pathwise estimator returns exactly 0.0 with a standard error of 0.0
against a true delta near 0.0196. It does not raise. The likelihood-ratio
estimator handles the same payoff correctly, at a cost in variance that is
measured rather than asserted.

**3. The compute ratio, at matched standard error on the Greek.** Not at matched
path count — that would answer a different and easier question. Central-
difference bumping costs about five pricing runs for price, delta and vega;
pathwise produces all three from one. So the *theoretical* ratio is roughly one
fifth, not one tenth, and a tenth only follows if bumping additionally needs
more paths to reach the same accuracy. Whether it does is exactly what matched-
accuracy measurement settles, so both conventions are computed and reported
side by side.

**4. The bump-size sweep**, from 1e-4 to 1e-1 relative. The error against the
closed form is U-shaped: differencing noise amplified by 1/(2h) dominates at
small bumps, truncation at large ones. The location of the minimum is reported.

Common random numbers are used throughout for bumping, and the benchmark also
runs one comparison *without* them to record how much worse it is.
"""

from __future__ import annotations

import time

import numpy as np

from quantdesk.analytics import blackscholes as bs
from quantdesk.config import BASE_MARKET, seed_for
from quantdesk.mc.greeks import (
    GreekEstimate,
    bump_samples,
    likelihood_ratio,
    pathwise_digital_delta,
    pathwise_european,
    summarise,
)
from quantdesk.mc.paths import terminal_prices
from quantdesk.results import write_result
from quantdesk.rng import rng

BENCHMARK = "greeks"

MARKET = BASE_MARKET
MATURITY = 1.0
STRIKE = 100.0
N_PATHS = 4_000_000
BUMP = 0.01  # relative, 1% of spot; the sweep below justifies the choice
VOL_BUMP = 0.01  # absolute in volatility
SWEEP_BUMPS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

# Five pricing runs is what central differencing costs for price + delta + vega:
# base, spot up, spot down, vol up, vol down.
BUMP_PRICING_RUNS = 5
PATHWISE_PRICING_RUNS = 1


def _call_payoff(terminal: np.ndarray) -> np.ndarray:
    return float(np.exp(-MARKET.rate * MATURITY)) * np.maximum(terminal - STRIKE, 0.0)


def _digital_payoff(terminal: np.ndarray) -> np.ndarray:
    return float(np.exp(-MARKET.rate * MATURITY)) * (terminal > STRIKE)


# --------------------------------------------------------------------------
# 1. Agreement on the vanilla
# --------------------------------------------------------------------------
def vanilla_agreement(seed: int) -> dict:
    true_delta = bs.delta(MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY)
    true_vega = bs.vega(MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY)

    z = rng(seed).standard_normal(N_PATHS)
    pathwise = pathwise_european(z, MARKET, MATURITY, STRIKE)
    pw_delta = summarise(pathwise["delta"], "pathwise delta")
    pw_vega = summarise(pathwise["vega"], "pathwise vega")

    bump_delta = summarise(
        bump_samples(z, MARKET, MATURITY, _call_payoff, parameter="spot", bump=BUMP),
        "bump delta (CRN)",
    )
    bump_vega = summarise(
        bump_samples(z, MARKET, MATURITY, _call_payoff, parameter="vol", bump=VOL_BUMP),
        "bump vega (CRN)",
    )

    def row(estimate: GreekEstimate, truth: float, units: str) -> dict:
        return {
            "estimate": estimate.value,
            "standard_error": estimate.standard_error,
            "truth": truth,
            "absolute_error": abs(estimate.value - truth),
            "error_in_standard_errors": (estimate.value - truth) / estimate.standard_error,
            "absolute_in": units,
        }

    delta_gap = abs(pw_delta.value - bump_delta.value)
    vega_gap = abs(pw_vega.value - bump_vega.value)

    return {
        "pathwise_delta": row(pw_delta, true_delta, "delta, dimensionless in [0,1]"),
        "bump_delta": row(bump_delta, true_delta, "delta, dimensionless in [0,1]"),
        "pathwise_vega": row(pw_vega, true_vega, "vega, price per 1.00 of volatility"),
        "bump_vega": row(bump_vega, true_vega, "vega, price per 1.00 of volatility"),
        "pathwise_vs_bump_delta_absolute": delta_gap,
        "pathwise_vs_bump_vega_absolute": vega_gap,
        "pathwise_vs_bump_vega_relative": vega_gap / true_vega,
        "n_paths": N_PATHS,
        "note": (
            "delta is dimensionless so an absolute tolerance on it is a much "
            "stronger claim than the same absolute tolerance on vega, whose "
            "scale here is about 37 per 1.00 of volatility"
        ),
    }


# --------------------------------------------------------------------------
# 2. Where pathwise is invalid
# --------------------------------------------------------------------------
def discontinuous_payoff_study(seed: int) -> dict:
    true_delta = bs.digital_delta(
        MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
    )
    true_price = bs.digital_price(
        MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
    )

    z = rng(seed).standard_normal(N_PATHS)
    terminal = terminal_prices(z, MARKET, MATURITY)
    payoff = _digital_payoff(terminal)

    invalid = summarise(
        pathwise_digital_delta(z, MARKET, MATURITY, STRIKE), "pathwise digital delta (INVALID)"
    )
    lr = summarise(likelihood_ratio(z, MARKET, MATURITY, payoff)["delta"], "likelihood ratio delta")
    bumped = summarise(
        bump_samples(z, MARKET, MATURITY, _digital_payoff, parameter="spot", bump=BUMP),
        "bump delta (CRN)",
    )

    # How much variance does the likelihood ratio cost where pathwise is legal?
    vanilla_pathwise = summarise(pathwise_european(z, MARKET, MATURITY, STRIKE)["delta"])
    vanilla_lr = summarise(likelihood_ratio(z, MARKET, MATURITY, _call_payoff(terminal))["delta"])

    return {
        "digital_closed_form_price": true_price,
        "digital_closed_form_delta": true_delta,
        "pathwise_invalid": {
            "estimate": invalid.value,
            "standard_error": invalid.standard_error,
            "absolute_error": abs(invalid.value - true_delta),
            "comment": (
                "exactly zero, with a standard error of exactly zero, because the "
                "derivative of a step function is zero almost everywhere. The "
                "estimator does not raise; it reports a confident wrong answer."
            ),
        },
        "likelihood_ratio": {
            "estimate": lr.value,
            "standard_error": lr.standard_error,
            "absolute_error": abs(lr.value - true_delta),
            "error_in_standard_errors": (lr.value - true_delta) / lr.standard_error,
        },
        "bump_crn": {
            "estimate": bumped.value,
            "standard_error": bumped.standard_error,
            "absolute_error": abs(bumped.value - true_delta),
            "error_in_standard_errors": (bumped.value - true_delta) / bumped.standard_error,
        },
        "variance_cost_of_lr_on_a_vanilla": {
            "pathwise_standard_error": vanilla_pathwise.standard_error,
            "likelihood_ratio_standard_error": vanilla_lr.standard_error,
            "standard_error_ratio": vanilla_lr.standard_error / vanilla_pathwise.standard_error,
            "equivalent_path_multiple": (
                vanilla_lr.standard_error / vanilla_pathwise.standard_error
            ) ** 2,
            "comment": (
                "the price of tolerating discontinuities: on a payoff where "
                "pathwise is legal, the likelihood ratio needs this many times "
                "the paths for the same accuracy, so it is used only where "
                "pathwise is invalid"
            ),
        },
    }


# --------------------------------------------------------------------------
# 3. Common random numbers, and what happens without them
# --------------------------------------------------------------------------
def common_random_numbers_study(seed: int) -> dict:
    true_delta = bs.delta(MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY)
    generator = rng(seed)
    z = generator.standard_normal(N_PATHS)
    z_independent = rng(seed + 1).standard_normal(N_PATHS)

    with_crn = summarise(
        bump_samples(z, MARKET, MATURITY, _call_payoff, parameter="spot", bump=BUMP), "CRN"
    )
    without_crn = summarise(
        bump_samples(
            z, MARKET, MATURITY, _call_payoff, parameter="spot", bump=BUMP, z_down=z_independent
        ),
        "independent draws",
    )

    return {
        "with_common_random_numbers": {
            "estimate": with_crn.value,
            "standard_error": with_crn.standard_error,
            "absolute_error": abs(with_crn.value - true_delta),
        },
        "without_common_random_numbers": {
            "estimate": without_crn.value,
            "standard_error": without_crn.standard_error,
            "absolute_error": abs(without_crn.value - true_delta),
        },
        "standard_error_ratio": without_crn.standard_error / with_crn.standard_error,
        "equivalent_path_multiple": (
            without_crn.standard_error / with_crn.standard_error
        ) ** 2,
        "comment": (
            "without common random numbers the estimator differences two "
            "independently noisy prices, and the 1/(2h) amplification turns that "
            "noise into the dominant term"
        ),
    }


# --------------------------------------------------------------------------
# 4. Bump-size sweep
# --------------------------------------------------------------------------
def bump_size_sweep(seed: int) -> dict:
    """The sweep, run twice — with and without common random numbers.

    CLAUDE.md predicts a U shape: noise amplification dominating at small bumps,
    truncation at large ones. That prediction is **only true without common
    random numbers**, and running both sweeps is what shows why.

    Without CRN the numerator differences two independently noisy prices. Its
    standard error is roughly constant in `h`, so dividing by `2h` amplifies it
    as `1/h` and the small-bump side of the curve climbs steeply. That is the U.

    With CRN the two legs share their draws, so the numerator shrinks with `h`
    exactly as the signal does and the quotient converges to the pathwise
    derivative. Its variance is therefore bounded as `h -> 0`, and the
    small-bump side **flattens onto the pathwise noise floor** instead of
    climbing. The curve becomes a hockey stick: flat, then truncation.

    Both are reported. The measured result stands over the expectation (R2).
    """
    true_delta = bs.delta(MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY)
    z = rng(seed).standard_normal(N_PATHS)
    z_independent = rng(seed + 1).standard_normal(N_PATHS)

    pathwise_floor = summarise(
        pathwise_european(z, MARKET, MATURITY, STRIKE)["delta"]
    ).standard_error

    def sweep(z_down):
        rows = []
        for bump in SWEEP_BUMPS:
            estimate = summarise(
                bump_samples(
                    z, MARKET, MATURITY, _call_payoff,
                    parameter="spot", bump=bump, z_down=z_down,
                )
            )
            rows.append({
                "relative_bump": bump,
                "estimate": estimate.value,
                "standard_error": estimate.standard_error,
                "absolute_error": abs(estimate.value - true_delta),
            })
        return rows

    crn_rows = sweep(None)
    free_rows = sweep(z_independent)

    def summarise_shape(rows):
        ses = [r["standard_error"] for r in rows]
        best = min(rows, key=lambda r: r["absolute_error"])
        return {
            "rows": rows,
            "minimum_at_relative_bump": best["relative_bump"],
            "minimum_absolute_error": best["absolute_error"],
            "standard_error_at_smallest_bump": ses[0],
            "standard_error_at_largest_bump": ses[-1],
            "standard_error_amplification_small_over_large": ses[0] / ses[-1],
        }

    return {
        "with_common_random_numbers": summarise_shape(crn_rows),
        "without_common_random_numbers": summarise_shape(free_rows),
        "pathwise_standard_error_floor": pathwise_floor,
        "shape_finding": (
            "The U shape CLAUDE.md predicts appears only WITHOUT common random "
            "numbers, where the standard error is amplified as 1/h at small "
            "bumps. WITH common random numbers the two legs share their noise, "
            "the difference quotient converges to the pathwise derivative, and "
            "the standard error flattens onto the pathwise noise floor instead "
            "of climbing - so the curve is a hockey stick, not a U. Since "
            "correct practice requires common random numbers, the shape actually "
            "obtained in this repository is the hockey stick."
        ),
    }


# --------------------------------------------------------------------------
# 5. Compute ratio, at matched accuracy and at matched path count
# --------------------------------------------------------------------------
def compute_ratio(seed: int) -> dict:
    """Both matching conventions, computed and labelled.

    The deterministic cost unit is a *path evaluation*, not wall-clock: pathwise
    needs one pricing pass for price, delta and vega; central differencing needs
    five. Wall-clock is measured too but flagged non-deterministic under R5.
    """
    z = rng(seed).standard_normal(N_PATHS)

    start = time.perf_counter()
    pathwise = pathwise_european(z, MARKET, MATURITY, STRIKE)
    pw_delta = summarise(pathwise["delta"])
    pw_vega = summarise(pathwise["vega"])
    pathwise_seconds = time.perf_counter() - start

    start = time.perf_counter()
    bump_delta = summarise(
        bump_samples(z, MARKET, MATURITY, _call_payoff, parameter="spot", bump=BUMP)
    )
    bump_vega = summarise(
        bump_samples(z, MARKET, MATURITY, _call_payoff, parameter="vol", bump=VOL_BUMP)
    )
    bump_seconds = time.perf_counter() - start

    # Paths bumping would need so that its standard error on delta equals
    # pathwise's at N_PATHS. Variance scales as 1/N, so the multiple is the
    # square of the standard-error ratio.
    delta_path_multiple = (bump_delta.standard_error / pw_delta.standard_error) ** 2
    vega_path_multiple = (bump_vega.standard_error / pw_vega.standard_error) ** 2

    matched_delta = (PATHWISE_PRICING_RUNS) / (BUMP_PRICING_RUNS * delta_path_multiple)
    matched_vega = (PATHWISE_PRICING_RUNS) / (BUMP_PRICING_RUNS * vega_path_multiple)

    return {
        "matching_convention": "matched standard error on the Greek",
        "pathwise_pricing_runs": PATHWISE_PRICING_RUNS,
        "bump_pricing_runs": BUMP_PRICING_RUNS,
        "bump_pricing_runs_note": "base, spot up, spot down, vol up, vol down",
        "delta": {
            "pathwise_standard_error": pw_delta.standard_error,
            "bump_standard_error": bump_delta.standard_error,
            "paths_bump_needs_to_match": delta_path_multiple,
            "cost_ratio_pathwise_over_bump": matched_delta,
        },
        "vega": {
            "pathwise_standard_error": pw_vega.standard_error,
            "bump_standard_error": bump_vega.standard_error,
            "paths_bump_needs_to_match": vega_path_multiple,
            "cost_ratio_pathwise_over_bump": matched_vega,
        },
        "cost_ratio_at_matched_path_count": PATHWISE_PRICING_RUNS / BUMP_PRICING_RUNS,
        "wall_clock_seconds": {
            "pathwise_all_greeks": pathwise_seconds,
            "bump_delta_and_vega": bump_seconds,
            "note": "wall-clock is not deterministic (R5); the pricing-run counts are",
        },
    }


def main() -> int:
    seed = seed_for(BENCHMARK)

    agreement = vanilla_agreement(seed)
    discontinuous = discontinuous_payoff_study(seed + 10)
    crn = common_random_numbers_study(seed + 20)
    sweep = bump_size_sweep(seed + 30)
    ratio = compute_ratio(seed + 40)

    values = {
        "vanilla_agreement": agreement,
        "discontinuous_payoff": discontinuous,
        "common_random_numbers": crn,
        "bump_size_sweep": sweep,
        "compute_ratio": ratio,
        "estimator_assignment": (
            "Route A (likelihood ratio) is implemented. Pathwise is used for the "
            "vanilla and for any Lipschitz leg; the likelihood ratio is used wherever "
            "the payoff jumps - digitals and knock-out features. Pathwise is never "
            "applied across a discontinuity."
        ),
    }

    inputs = {
        "market": MARKET.as_dict(),
        "strike": STRIKE,
        "maturity": MATURITY,
        "n_paths": N_PATHS,
        "relative_spot_bump": BUMP,
        "absolute_vol_bump": VOL_BUMP,
        "sweep_bumps": SWEEP_BUMPS,
        "seed": seed,
    }

    path = write_result(
        benchmark=BENCHMARK, seed=seed, inputs=inputs, values=values,
        notes=(
            "Compute ratio is reported at matched standard error on the Greek, and "
            "also at matched path count, with both labelled. Cost is counted in "
            "pricing runs, which is deterministic; wall-clock is recorded but flagged."
        ),
    )

    a = agreement
    print("1. AGREEMENT ON THE VANILLA (M1 gives the truth)")
    for key in ("pathwise_delta", "bump_delta", "pathwise_vega", "bump_vega"):
        r = a[key]
        print(f"   {key:<16} {r['estimate']:>12.8f}  err {r['absolute_error']:.3e}"
              f"  ({r['error_in_standard_errors']:+.2f} se)  [{r['absolute_in']}]")
    print(f"   pathwise vs bump: delta {a['pathwise_vs_bump_delta_absolute']:.3e} absolute, "
          f"vega {a['pathwise_vs_bump_vega_absolute']:.3e} absolute "
          f"({a['pathwise_vs_bump_vega_relative']:.2e} relative)")

    d = discontinuous
    print("\n2. A DISCONTINUOUS PAYOFF (digital; closed-form delta "
          f"{d['digital_closed_form_delta']:.8f})")
    print(f"   pathwise (INVALID)  {d['pathwise_invalid']['estimate']:.8f}"
          f"   error {d['pathwise_invalid']['absolute_error']:.3e}   <- silently wrong")
    print(f"   likelihood ratio    {d['likelihood_ratio']['estimate']:.8f}"
          f"   error {d['likelihood_ratio']['absolute_error']:.3e}"
          f"  ({d['likelihood_ratio']['error_in_standard_errors']:+.2f} se)")
    print(f"   bump, CRN           {d['bump_crn']['estimate']:.8f}"
          f"   error {d['bump_crn']['absolute_error']:.3e}"
          f"  ({d['bump_crn']['error_in_standard_errors']:+.2f} se)")
    v = d["variance_cost_of_lr_on_a_vanilla"]
    print(f"   LR costs {v['equivalent_path_multiple']:.1f}x the paths of pathwise "
          f"where pathwise is legal")

    print("\n3. COMMON RANDOM NUMBERS")
    print(f"   with    se {crn['with_common_random_numbers']['standard_error']:.3e}")
    print(f"   without se {crn['without_common_random_numbers']['standard_error']:.3e}")
    print(f"   ratio {crn['standard_error_ratio']:.1f}x worse, "
          f"= {crn['equivalent_path_multiple']:.0f}x the paths for the same accuracy")

    print("\n4. BUMP-SIZE SWEEP")
    print(f"   pathwise standard-error floor: {sweep['pathwise_standard_error_floor']:.3e}")
    for key, label in (("without_common_random_numbers", "WITHOUT CRN (the predicted U)"),
                       ("with_common_random_numbers", "WITH CRN (what we actually use)")):
        d = sweep[key]
        print(f"   {label}   se amplification small/large = "
              f"{d['standard_error_amplification_small_over_large']:.1f}x")
        for r in d["rows"]:
            print(f"      h={r['relative_bump']:<7g} err {r['absolute_error']:.3e}"
                  f"  se {r['standard_error']:.3e}")
        print(f"      minimum at h = {d['minimum_at_relative_bump']:g}")

    print("\n5. COMPUTE RATIO")
    print(f"   at matched path count:      {ratio['cost_ratio_at_matched_path_count']:.3f} "
          f"({PATHWISE_PRICING_RUNS} pricing run vs {BUMP_PRICING_RUNS})")
    for greek in ("delta", "vega"):
        g = ratio[greek]
        print(f"   at matched accuracy, {greek:<5}: {g['cost_ratio_pathwise_over_bump']:.3f} "
              f"(bump needs {g['paths_bump_needs_to_match']:.2f}x the paths)")

    print(f"\nwrote {path.relative_to(path.parents[1])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
