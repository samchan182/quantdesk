"""M3 — variance reduction study. Headline #1.

    python bench/variance_reduction.py

Four arms at **equal total path count** on the same product:

    A  crude Monte Carlo,                 2N paths
    B  antithetic only,                   N pairs   (= 2N paths)
    C  control variate only,              2N paths
    D  antithetic + control variate,      N pairs   (= 2N paths)

**The product.** An arithmetic-average Asian call — the average taken over the
252 daily closes after inception, excluding the known spot. It has no closed
form, which is the point: the European call struck at the same level, on the
same paths, does have one, so it can act as a control whose expectation is
known exactly rather than estimated. A plain European as the *target* would be
a degenerate study, since the control would be perfectly correlated with it and
the variance would collapse to zero for reasons that say nothing about the
technique.

**Beta is estimated on a separate pilot run.** Never hardcoded to 1, and never
fitted to the same draws it is then applied to. See the notes in DEFENSE.md.

**Arms share draws within a pairing convention, deliberately.** A and C come out
of one simulation, B and D out of another. That makes the A-to-C and B-to-D
comparisons differ *only* by the control adjustment, instead of also differing
by sampling noise, which is what makes a variance-reduction percentage
meaningful at this sample size. The two simulations use different seeds.

**Units are reported both ways.** A variance reduction of 90% and a standard
error reduction of 68% are the same fact; quoting a percentage without saying
which is trap 16 in a different costume. `REPORT.md` carries both.
"""

from __future__ import annotations

from quantdesk.analytics import blackscholes as bs
from quantdesk.config import BASE_MARKET, seed_for
from quantdesk.mc.paths import PairedRun, simulate_paired
from quantdesk.products.vanilla import arithmetic_asian_payoff, european_payoff, on_terminal_column
from quantdesk.results import write_result

BENCHMARK = "variance_reduction"

MARKET = BASE_MARKET
MATURITY = 1.0
STRIKE = 100.0
N_STEPS = 252
TOTAL_PATHS = 200_000  # every arm simulates exactly this many paths
CHUNK_PATHS = 10_000
PILOT_PATHS = 20_000  # 10% of the main run, on an independent seed
PILOT_SEED_OFFSET = 7_000_001


def _payoffs():
    target = arithmetic_asian_payoff(STRIKE, MARKET.rate, MATURITY)
    control = on_terminal_column(european_payoff(STRIKE, MARKET.rate, MATURITY))
    return target, control


def pilot_beta(seed: int, antithetic: bool) -> tuple[float, float, int]:
    """Estimate beta on an independent pilot. Returns (beta, correlation, n_samples).

    Run under the *same* pairing convention as the arm it will serve: pairing
    changes the joint distribution of target and control, so the optimal beta
    for pair averages is not the optimal beta for individual paths.
    """
    target, control = _payoffs()
    run = simulate_paired(
        seed=seed,
        market=MARKET,
        maturity=MATURITY,
        n_paths=PILOT_PATHS,
        target=target,
        control=control,
        n_steps=N_STEPS,
        chunk_paths=CHUNK_PATHS,
        antithetic=antithetic,
    )
    return run.joint.beta_star, run.joint.correlation, run.joint.n_samples


def run_arm(seed: int, antithetic: bool, beta: float, expected_control: float) -> PairedRun:
    target, control = _payoffs()
    return simulate_paired(
        seed=seed,
        market=MARKET,
        maturity=MATURITY,
        n_paths=TOTAL_PATHS,
        target=target,
        control=control,
        expected_control=expected_control,
        beta=beta,
        n_steps=N_STEPS,
        chunk_paths=CHUNK_PATHS,
        antithetic=antithetic,
    )


def main() -> int:
    seed = seed_for(BENCHMARK)

    # The control's exact expectation, from M1. This is the whole reason a
    # European can serve as the control: E[Y] is known, not estimated.
    expected_control = bs.call_price(
        MARKET.spot, STRIKE, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
    )

    beta_plain, rho_pilot_plain, pilot_n_plain = pilot_beta(seed + PILOT_SEED_OFFSET, False)
    beta_anti, rho_pilot_anti, pilot_n_anti = pilot_beta(seed + PILOT_SEED_OFFSET + 1, True)

    plain = run_arm(seed, False, beta_plain, expected_control)
    anti = run_arm(seed + 1, True, beta_anti, expected_control)

    arms = {
        "A_crude": {
            "estimate": plain.joint.mean_x,
            "standard_error": plain.joint.standard_error_x,
            "n_samples": plain.joint.n_samples,
            "n_paths": plain.joint.n_paths,
        },
        "B_antithetic": {
            "estimate": anti.joint.mean_x,
            "standard_error": anti.joint.standard_error_x,
            "n_samples": anti.joint.n_samples,
            "n_paths": anti.joint.n_paths,
        },
        "C_control_variate": {
            "estimate": plain.controlled.mean,
            "standard_error": plain.controlled.standard_error,
            "n_samples": plain.controlled.n_samples,
            "n_paths": plain.controlled.n_paths,
        },
        "D_antithetic_plus_cv": {
            "estimate": anti.controlled.mean,
            "standard_error": anti.controlled.standard_error,
            "n_samples": anti.controlled.n_samples,
            "n_paths": anti.controlled.n_paths,
        },
    }

    se_a = arms["A_crude"]["standard_error"]
    se_d = arms["D_antithetic_plus_cv"]["standard_error"]

    # Both framings of the same fact, so neither can be quoted ambiguously.
    se_reduction_pct = 100.0 * (1.0 - se_d / se_a)
    variance_reduction_pct = 100.0 * (1.0 - (se_d / se_a) ** 2)

    # Does the realised control-variate gain match theory on the same sample?
    realised_ratio_plain = (arms["C_control_variate"]["standard_error"] / se_a) ** 2
    realised_ratio_anti = (se_d / arms["B_antithetic"]["standard_error"]) ** 2

    values = {
        "arms": arms,
        "headline_se_reduction_pct_A_to_D": se_reduction_pct,
        "headline_variance_reduction_pct_A_to_D": variance_reduction_pct,
        "speedup_equivalent_paths": (se_a / se_d) ** 2,
        "control_variate": {
            "expected_control_closed_form": expected_control,
            "beta_plain": beta_plain,
            "beta_antithetic": beta_anti,
            "pilot_correlation_plain": rho_pilot_plain,
            "pilot_correlation_antithetic": rho_pilot_anti,
            "main_sample_correlation_plain": plain.joint.correlation,
            "main_sample_correlation_antithetic": anti.joint.correlation,
            "theoretical_variance_ratio_plain": plain.joint.theoretical_variance_ratio,
            "theoretical_variance_ratio_antithetic": anti.joint.theoretical_variance_ratio,
            "realised_variance_ratio_plain": realised_ratio_plain,
            "realised_variance_ratio_antithetic": realised_ratio_anti,
        },
        "antithetic_variance_ratio_B_over_A": (
            arms["B_antithetic"]["standard_error"] / se_a
        ) ** 2,
    }

    inputs = {
        "product": "arithmetic average Asian call, average over daily closes after inception",
        "control": "European call on the same paths, same strike and maturity",
        "market": MARKET.as_dict(),
        "strike": STRIKE,
        "maturity": MATURITY,
        "n_steps": N_STEPS,
        "total_paths_per_arm": TOTAL_PATHS,
        "chunk_paths": CHUNK_PATHS,
        "pilot_paths": PILOT_PATHS,
        "pilot_samples_plain": pilot_n_plain,
        "pilot_samples_antithetic": pilot_n_anti,
        "seed_plain_arms": seed,
        "seed_antithetic_arms": seed + 1,
        "pilot_seed_plain": seed + PILOT_SEED_OFFSET,
        "pilot_seed_antithetic": seed + PILOT_SEED_OFFSET + 1,
    }

    path = write_result(
        benchmark=BENCHMARK,
        seed=seed,
        inputs=inputs,
        values=values,
        notes=(
            "Arms A and C share one simulation; B and D share another, so each "
            "control-variate comparison differs only by the control adjustment "
            "and not by sampling noise. Beta is estimated on independent pilot "
            "runs, one per pairing convention."
        ),
    )

    # --- report to the terminal; the JSON above is the record of it ---
    print(f"product : arithmetic Asian call, K={STRIKE}, T={MATURITY}, {N_STEPS} steps")
    print(f"control : European call, exact E[Y] = {expected_control:.10f}")
    print(f"beta    : plain {beta_plain:.6f}   antithetic {beta_anti:.6f}  (pilot, N={PILOT_PATHS})")
    print(f"          pilot rho {rho_pilot_plain:.6f} / {rho_pilot_anti:.6f}")
    print()
    print(f"{'arm':<24}{'estimate':>14}{'std error':>14}{'samples':>10}{'paths':>10}")
    for name, a in arms.items():
        print(
            f"{name:<24}{a['estimate']:>14.6f}{a['standard_error']:>14.6f}"
            f"{a['n_samples']:>10,}{a['n_paths']:>10,}"
        )
    print()
    print(f"A -> D  standard error reduction : {se_reduction_pct:.2f}%")
    print(f"A -> D  variance reduction       : {variance_reduction_pct:.2f}%")
    print(f"        equivalent path speedup  : {(se_a / se_d) ** 2:.1f}x")
    print()
    print("control variate, checked against 1 - rho^2 on the main sample:")
    print(
        f"  plain      rho {plain.joint.correlation:.6f}  "
        f"theory {plain.joint.theoretical_variance_ratio:.6f}  "
        f"realised {realised_ratio_plain:.6f}"
    )
    print(
        f"  antithetic rho {anti.joint.correlation:.6f}  "
        f"theory {anti.joint.theoretical_variance_ratio:.6f}  "
        f"realised {realised_ratio_anti:.6f}"
    )
    print(f"\nwrote {path.relative_to(path.parents[1])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
