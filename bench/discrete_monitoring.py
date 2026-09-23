"""M7 — discrete monitoring and the Brownian bridge. Headline #4.

    python bench/discrete_monitoring.py

**The judgement call comes first, because it decides what this milestone is.**
The author was asked how the contract monitors its barrier and answered: on
**daily closing prices**, with the correction's effect reported separately as a
sensitivity.

The simulation steps daily. The contract observes daily closes. The two grids
therefore coincide, the simulation of that barrier is **exact**, and the
Brownian bridge correction **must not be applied** to the contract as written.
Applying it would price a continuously-monitored contract that nobody wrote,
and would do so in the direction of a lower price — trap 8, and a mistake that
looks like sophistication.

So this benchmark reports two different things and never mixes them:

1. **The contract price**, on the daily grid the contract specifies, with no
   correction. This is a price, not an approximation to one.
2. **A labelled sensitivity**: what the correction *would* move the price by if
   the contract were continuously monitored. This is the headline number, and
   it is reported as "the cost of a monitoring convention", not as a correction
   to the price above.

Everything is validated on a down-and-out call first, because that has a
continuous closed form and so the machinery can be checked against an exact
answer before it is pointed at a snowball that has none.

**Two independent routes are compared**, as the spec requires: the Brownian
bridge, which simulates the missed excursions directly, and the
Broadie-Glasserman-Kou continuity correction, which shifts the barrier outward
by `B*exp(+/- beta*sigma*sqrt(dt))` with `beta ~ 0.5826` and prices with the
continuous formula. They approach the same answer from different directions, so
agreement between them is real evidence rather than a restatement.
"""

from __future__ import annotations

import numpy as np

from quantdesk.analytics import blackscholes as bs
from quantdesk.analytics.barrier import BETA_BGK, down_and_out_call, shifted_barrier
from quantdesk.config import BASE_MARKET, SNOWBALL_TERMS, Market, seed_for
from quantdesk.mc.bridge import bridge_survival, observed_survival
from quantdesk.mc.estimators import Accumulator
from quantdesk.mc.paths import chunk_sizes, price_paths
from quantdesk.products.snowball import snowball_payoff
from quantdesk.results import write_result
from quantdesk.rng import spawn
from quantdesk.units import bp_of_premium

BENCHMARK = "discrete_monitoring"

MARKET = BASE_MARKET
MATURITY = 1.0
STRIKE = 100.0
BARRIER = 90.0  # 90% of spot, a down-and-out knock-out barrier
N_STEPS = 252  # daily, matching the contract's observation grid
N_PATHS = 2_000_000
CHUNK = 50_000

STEP_SWEEP = [12, 52, 252, 1008, 4032]
VOL_SWEEP = [0.10, 0.20, 0.35, 0.50]
BARRIER_SWEEP = [70.0, 80.0, 90.0, 95.0, 98.0]


def _run_arms(
    seed: int, market: Market, maturity: float, n_steps: int, n_paths: int,
    strike: float, barrier: float, chunk: int = CHUNK,
) -> dict:
    """Price the knock-out under discrete monitoring and with the bridge.

    Both arms use the **same paths** — common random numbers — so their
    difference is the effect of the correction and not the difference of two
    independent Monte Carlo errors. The difference is accumulated directly, so
    its standard error is the standard error of a paired quantity and is far
    smaller than either arm's.
    """
    dt = maturity / n_steps
    discount = float(np.exp(-market.rate * maturity))
    sizes = chunk_sizes(n_paths, chunk)
    generators = spawn(seed, len(sizes))

    discrete, bridged, difference = Accumulator(), Accumulator(), Accumulator()

    for size, generator in zip(sizes, generators, strict=True):
        z = generator.standard_normal((size, n_steps))
        paths = price_paths(z, market, maturity, engine="numba")
        intrinsic = discount * np.maximum(paths[:, -1] - strike, 0.0)

        alive_discrete = observed_survival(paths, barrier, direction="down")
        alive_bridge = bridge_survival(
            paths, barrier, market.vol, dt, generator, direction="down"
        )

        a = intrinsic * alive_discrete
        b = intrinsic * alive_bridge
        discrete.update(a)
        bridged.update(b)
        difference.update(a - b)

    return {
        "discrete": discrete.mean,
        "discrete_se": discrete.standard_error,
        "bridged": bridged.mean,
        "bridged_se": bridged.standard_error,
        "difference": difference.mean,
        "difference_se": difference.standard_error,
        "n_paths": discrete.n_paths,
    }


def validation(seed: int) -> dict:
    """Check the machinery against the continuous closed form before using it."""
    continuous = down_and_out_call(
        MARKET.spot, STRIKE, BARRIER, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
    )
    bgk_barrier = shifted_barrier(BARRIER, MARKET.vol, MATURITY / N_STEPS, "down")
    bgk_price = down_and_out_call(
        MARKET.spot, STRIKE, bgk_barrier, MARKET.rate, MARKET.div_yield, MARKET.vol, MATURITY
    )

    arms = _run_arms(seed, MARKET, MATURITY, N_STEPS, N_PATHS, STRIKE, BARRIER)

    return {
        "continuous_closed_form": continuous,
        "bgk_shifted_barrier": bgk_barrier,
        "bgk_closed_form_price": bgk_price,
        "mc_discrete": arms["discrete"],
        "mc_discrete_se": arms["discrete_se"],
        "mc_bridged": arms["bridged"],
        "mc_bridged_se": arms["bridged_se"],
        "bridge_vs_continuous": arms["bridged"] - continuous,
        "bridge_vs_continuous_in_se": (arms["bridged"] - continuous) / arms["bridged_se"],
        "discrete_vs_bgk": arms["discrete"] - bgk_price,
        "discrete_vs_bgk_in_se": (arms["discrete"] - bgk_price) / arms["discrete_se"],
        "beta_bgk": BETA_BGK,
    }


def headline(seed: int) -> dict:
    """The correction's effect on a daily-monitored knock-out, in bp of premium."""
    arms = _run_arms(seed, MARKET, MATURITY, N_STEPS, N_PATHS, STRIKE, BARRIER)
    base = arms["discrete"]
    return {
        "discrete_price": arms["discrete"],
        "bridge_corrected_price": arms["bridged"],
        "difference": arms["difference"],
        "difference_se": arms["difference_se"],
        "difference_bp_of_premium": bp_of_premium(arms["difference"], base),
        "difference_se_bp_of_premium": bp_of_premium(arms["difference_se"], base),
        "difference_in_standard_errors": arms["difference"] / arms["difference_se"],
        "sign_check_correction_lowers_price": arms["difference"] > 0,
        "n_paths": arms["n_paths"],
        "units": "basis points of the discretely-monitored premium",
    }


def step_convergence(seed: int) -> list[dict]:
    """As the grid refines, discrete and corrected must converge."""
    rows = []
    for n_steps in STEP_SWEEP:
        # Hold the path budget roughly constant in work, not in count.
        n_paths = max(100_000, min(N_PATHS, int(4e8 / n_steps)))
        arms = _run_arms(
            seed + n_steps, MARKET, MATURITY, n_steps, n_paths, STRIKE, BARRIER,
            chunk=max(5_000, min(CHUNK, int(2e7 / n_steps))),
        )
        rows.append({
            "n_steps": n_steps,
            "observations_per_year": n_steps,
            "discrete_price": arms["discrete"],
            "bridge_corrected_price": arms["bridged"],
            "difference": arms["difference"],
            "difference_se": arms["difference_se"],
            "difference_bp_of_premium": bp_of_premium(arms["difference"], arms["discrete"]),
            "n_paths": arms["n_paths"],
        })
    return rows


def sensitivity(seed: int) -> dict:
    """Where the bias is large, and where it is negligible."""
    by_vol = []
    for vol in VOL_SWEEP:
        market = Market(spot=MARKET.spot, rate=MARKET.rate, div_yield=MARKET.div_yield, vol=vol)
        arms = _run_arms(seed + int(vol * 1000), market, MATURITY, N_STEPS, 500_000, STRIKE, BARRIER)
        by_vol.append({
            "vol": vol,
            "difference_bp_of_premium": bp_of_premium(arms["difference"], arms["discrete"]),
            "difference_se_bp": bp_of_premium(arms["difference_se"], arms["discrete"]),
            "discrete_price": arms["discrete"],
        })

    by_barrier = []
    for barrier in BARRIER_SWEEP:
        arms = _run_arms(
            seed + int(barrier), MARKET, MATURITY, N_STEPS, 500_000, STRIKE, barrier
        )
        by_barrier.append({
            "barrier": barrier,
            "barrier_pct_of_spot": barrier / MARKET.spot,
            "difference_bp_of_premium": bp_of_premium(arms["difference"], arms["discrete"]),
            "difference_se_bp": bp_of_premium(arms["difference_se"], arms["discrete"]),
            "discrete_price": arms["discrete"],
        })

    return {"by_volatility": by_vol, "by_barrier_level": by_barrier}


def snowball_sensitivity(seed: int) -> dict:
    """The contract as written, and the sensitivity — kept strictly apart.

    The snowball's knock-in is observed on daily closes and the simulation steps
    daily, so the first number below **is the price**. The second is what the
    price would be if the same barrier were monitored continuously, and it is a
    statement about a different contract.
    """
    terms = SNOWBALL_TERMS
    payoff = snowball_payoff(terms, MARKET.spot, MARKET.rate)
    ki_barrier = terms.knock_in_level * MARKET.spot
    dt = terms.tenor_years / terms.n_steps

    n_paths, chunk = 400_000, 10_000
    sizes = chunk_sizes(n_paths, chunk)
    generators = spawn(seed, len(sizes))

    as_written, if_continuous, difference = Accumulator(), Accumulator(), Accumulator()

    for size, generator in zip(sizes, generators, strict=True):
        z = generator.standard_normal((size, terms.n_steps))
        paths = price_paths(z, MARKET, terms.tenor_years, engine="numba")

        contract = payoff(paths)

        # Continuous monitoring: any path the bridge says touched the knock-in
        # is treated as knocked in, by dragging it to the barrier on the step
        # where it crossed. Implemented by forcing a breach into the observed
        # series so the existing four-branch payoff sees it.
        alive = bridge_survival(paths, ki_barrier, MARKET.vol, dt, generator, direction="down")
        continuous_paths = paths.copy()
        newly_knocked_in = ~alive & np.all(paths[:, 1:] > ki_barrier, axis=1)
        continuous_paths[newly_knocked_in, 1] = ki_barrier
        continuous = payoff(continuous_paths)

        as_written.update(contract)
        if_continuous.update(continuous)
        difference.update(contract - continuous)

    base = as_written.mean
    return {
        "price_as_written_daily_close_monitoring": base,
        "price_as_written_se": as_written.standard_error,
        "price_if_continuously_monitored": if_continuous.mean,
        "difference": difference.mean,
        "difference_se": difference.standard_error,
        "difference_bp_of_premium": bp_of_premium(difference.mean, base),
        "difference_se_bp_of_premium": bp_of_premium(difference.standard_error, base),
        "n_paths": as_written.n_paths,
        "interpretation": (
            "the first figure IS the price of the contract the author specified; "
            "the second prices a different contract, one monitored continuously. "
            "The difference is the value of the monitoring convention, not a "
            "correction to be applied."
        ),
    }


def main() -> int:
    seed = seed_for(BENCHMARK)

    valid = validation(seed)
    head = headline(seed + 1)
    convergence = step_convergence(seed + 2)
    sens = sensitivity(seed + 3)
    snowball = snowball_sensitivity(seed + 4)

    values = {
        "judgement": (
            "The contract monitors its barrier on daily closing prices and the "
            "simulation steps daily, so the two grids coincide, the simulation is "
            "EXACT, and the Brownian bridge correction MUST NOT be applied to the "
            "contract as written. The headline below is a sensitivity - what the "
            "correction would move the price by under continuous monitoring - and "
            "not a correction to the reported price."
        ),
        "validation": valid,
        "headline": head,
        "step_convergence": convergence,
        "sensitivity": sens,
        "snowball": snowball,
    }

    inputs = {
        "product": "down-and-out call, barrier below spot, for the validated headline",
        "market": MARKET.as_dict(),
        "strike": STRIKE,
        "barrier": BARRIER,
        "barrier_pct_of_spot": BARRIER / MARKET.spot,
        "maturity": MATURITY,
        "n_steps": N_STEPS,
        "n_paths": N_PATHS,
        "chunk_paths": CHUNK,
        "step_sweep": STEP_SWEEP,
        "vol_sweep": VOL_SWEEP,
        "barrier_sweep": BARRIER_SWEEP,
        "snowball_terms": SNOWBALL_TERMS.as_dict(),
        "beta_bgk": BETA_BGK,
    }

    path = write_result(
        benchmark=BENCHMARK, seed=seed, inputs=inputs, values=values,
        notes=(
            "Both arms share paths (common random numbers), so the difference is "
            "a paired quantity and its standard error is far smaller than either "
            "arm's. The contract price and the continuous-monitoring sensitivity "
            "are reported separately and must not be combined."
        ),
    )

    print("JUDGEMENT: contract monitors daily closes; simulation steps daily.")
    print("           The simulation is EXACT. No correction is applied to the price.\n")

    print("VALIDATION (down-and-out call, closed form available)")
    print(f"  continuous closed form        {valid['continuous_closed_form']:.6f}")
    print(f"  MC + Brownian bridge          {valid['mc_bridged']:.6f}"
          f"   ({valid['bridge_vs_continuous_in_se']:+.2f} se)")
    print(f"  BGK shifted barrier {valid['bgk_shifted_barrier']:.4f} -> "
          f"{valid['bgk_closed_form_price']:.6f}")
    print(f"  MC discrete (daily)           {valid['mc_discrete']:.6f}"
          f"   ({valid['discrete_vs_bgk_in_se']:+.2f} se vs BGK)")

    print(f"\nHEADLINE: correction moves a daily-monitored knock-out by "
          f"{head['difference_bp_of_premium']:+.2f} bp of premium "
          f"(se {head['difference_se_bp_of_premium']:.2f} bp)")
    print(f"  discrete {head['discrete_price']:.6f} -> bridged "
          f"{head['bridge_corrected_price']:.6f}")
    print(f"  sign: correcting LOWERS the knock-out price = "
          f"{head['sign_check_correction_lowers_price']}")

    print("\nCONVERGENCE AS THE GRID REFINES")
    print(f"  {'steps':>7}{'diff bp':>10}{'se bp':>9}{'paths':>12}")
    for row in convergence:
        print(f"  {row['n_steps']:>7}{row['difference_bp_of_premium']:>10.2f}"
              f"{bp_of_premium(row['difference_se'], row['discrete_price']):>9.2f}"
              f"{row['n_paths']:>12,}")

    print("\nSENSITIVITY")
    print("  by volatility:")
    for row in sens["by_volatility"]:
        print(f"    sigma={row['vol']:<5} {row['difference_bp_of_premium']:>9.2f} bp"
              f"  (se {row['difference_se_bp']:.2f})")
    print("  by barrier level:")
    for row in sens["by_barrier_level"]:
        print(f"    B={row['barrier']:<6g} ({row['barrier_pct_of_spot']:.0%} of spot)"
              f" {row['difference_bp_of_premium']:>9.2f} bp"
              f"  (se {row['difference_se_bp']:.2f})")

    print("\nSNOWBALL (knock-in monitored on daily closes)")
    print(f"  price as written           {snowball['price_as_written_daily_close_monitoring']:.6f}"
          f"  (se {snowball['price_as_written_se']:.6f})   <- THE PRICE")
    print(f"  if continuously monitored  {snowball['price_if_continuously_monitored']:.6f}"
          f"   <- a different contract")
    print(f"  monitoring convention worth {snowball['difference_bp_of_premium']:+.2f} bp "
          f"(se {snowball['difference_se_bp_of_premium']:.2f})")

    print(f"\nwrote {path.relative_to(path.parents[1])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
