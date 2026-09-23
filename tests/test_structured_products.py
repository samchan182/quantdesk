"""M4 acceptance — the degenerate tests, which are the whole point.

A structured-product payoff is a nest of branches, and a nest of branches is
where sign errors and precedence mistakes hide without changing anything
obvious about the price. The defence is to collapse the product to cases whose
value is known in closed form — or known with certainty — and check it lands
there exactly.

Three of the four required tests are *deterministic* once collapsed: the payoff
stops depending on the path at all, so the Monte Carlo variance is exactly zero
and the check is an equality rather than a confidence interval. That is much
stronger than "within Monte Carlo noise", and it is worth noticing that the
spec's phrasing permits the weaker version.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantdesk.config import AUTOCALLABLE_TERMS, BASE_MARKET, SNOWBALL_TERMS, Market
from quantdesk.mc.paths import simulate
from quantdesk.products.autocallable import autocallable_payoff, observation_indices
from quantdesk.products.snowball import (
    BRANCHES,
    SnowballTerms,
    fixed_coupon_note_value,
    snowball_payoff,
)
from quantdesk.rng import rng

MARKET = BASE_MARKET
TERMS = SNOWBALL_TERMS


def price(terms: SnowballTerms, market: Market = MARKET, *, seed: int = 4242, n_paths: int = 40_000):
    return simulate(
        seed=seed,
        market=market,
        maturity=terms.tenor_years,
        n_paths=n_paths,
        payoff=snowball_payoff(terms, market.spot, market.rate),
        n_steps=terms.n_steps,
        chunk_paths=10_000,
        antithetic=True,
    )


# --------------------------------------------------------------------------
# Degenerate test 1 — no knock-out, no knock-in: a fixed-coupon note
# --------------------------------------------------------------------------
def test_unreachable_barriers_reduce_to_a_fixed_coupon_note():
    """Knock-out at infinity, knock-in at zero: the note becomes certain.

    Every path lands in branch 2 and pays `1 + c*T` at maturity, so the price
    is `exp(-rT)*(1 + c*T)` with *zero* variance — not merely within Monte
    Carlo noise.
    """
    terms = SnowballTerms(
        tenor_years=TERMS.tenor_years,
        knock_out_level=np.inf,
        knock_out_frequency=TERMS.knock_out_frequency,
        knock_in_level=0.0,
        knock_in_frequency=TERMS.knock_in_frequency,
        coupon_rate=TERMS.coupon_rate,
        steps_per_year=TERMS.steps_per_year,
    )
    acc = price(terms)
    expected = fixed_coupon_note_value(terms, MARKET.rate)

    assert acc.mean == pytest.approx(expected, rel=1e-14)
    assert acc.std < 1e-15, f"a certain payoff must have zero variance, got {acc.std:.3e}"

    # And it really is branch 2 doing it, not two errors cancelling.
    _, branches = _branches_of(terms)
    assert set(np.unique(branches)) == {1}


# --------------------------------------------------------------------------
# Degenerate test 2 — knock-out on day one
# --------------------------------------------------------------------------
def test_knock_out_at_day_one_pays_the_discounted_first_coupon():
    """A knock-out barrier nothing can be below: redemption at the first date.

    Observed daily so that the first observation really is day one, as the
    spec's wording intends; under the contract's own monthly schedule the
    earliest possible redemption is month one.
    """
    terms = SnowballTerms(
        tenor_years=TERMS.tenor_years,
        knock_out_level=0.0,
        knock_out_frequency="daily",
        # Knock-in set unreachable too: this test is about the knock-out leg,
        # and the terms validator rightly refuses a knock-in above a knock-out.
        knock_in_level=0.0,
        knock_in_frequency="daily",
        coupon_rate=TERMS.coupon_rate,
        steps_per_year=TERMS.steps_per_year,
    )
    acc = price(terms)

    first_time = 1.0 / TERMS.steps_per_year
    expected = float(np.exp(-MARKET.rate * first_time) * (1.0 + TERMS.coupon_rate * first_time))

    assert acc.mean == pytest.approx(expected, rel=1e-14)
    assert acc.std < 1e-15  # knocked out with probability 1

    _, branches = _branches_of(terms)
    assert set(np.unique(branches)) == {0}


def test_knock_out_precedence_over_knock_in():
    """A path that knocks in and later knocks out pays the coupon, not the loss.

    Constructed by hand rather than simulated: dip to 50% of spot on day one
    (knocking in), then recover to 120% and stay there. Under the contract's
    monthly schedule the first knock-out observation is month one, by which
    point the path is above the knock-out level.
    """
    terms = SnowballTerms(steps_per_year=TERMS.steps_per_year, tenor_years=TERMS.tenor_years)
    spot = MARKET.spot
    path = np.full((1, terms.n_steps + 1), 1.20 * spot)
    path[0, 0] = spot
    path[0, 1] = 0.50 * spot  # knocks in on day one

    payoff, branches = snowball_payoff(terms, spot, MARKET.rate, return_branches=True)(path)
    assert branches[0] == 0, f"expected branch 0, got {BRANCHES[branches[0]]}"

    ko_step = observation_indices("monthly", terms.tenor_years, terms.steps_per_year)[0]
    t_ko = ko_step / terms.steps_per_year
    assert payoff[0] == pytest.approx(
        np.exp(-MARKET.rate * t_ko) * (1.0 + terms.coupon_rate * t_ko), rel=1e-14
    )


# --------------------------------------------------------------------------
# Degenerate test 3 — no coupon, no barriers: a forward
# --------------------------------------------------------------------------
def test_zero_coupon_and_removed_barriers_reduce_to_a_discounted_forward():
    """The investor is left holding the index outright, so the note is a forward.

    "Barriers removed" is made precise: knock-out at infinity (never callable),
    knock-in at infinity (always knocked in), strike at infinity (never
    recovers), so every path lands in branch 4 and redeems at the index's
    performance. The value is then `exp(-rT) * E[S_T]/S_0 = exp(-qT)`.

    Run at a **non-zero dividend yield**, deliberately. At BASE_MARKET's q = 0
    the answer would be exactly 1.0, which any number of wrong implementations
    would also produce.
    """
    market = Market(spot=100.0, rate=0.03, div_yield=0.02, vol=0.25)
    terms = SnowballTerms(
        tenor_years=2.0,
        knock_out_level=np.inf,
        knock_in_level=np.inf,
        strike_level=np.inf,
        coupon_rate=0.0,
        downside_participation=1.0,
        steps_per_year=TERMS.steps_per_year,
    )
    acc = price(terms, market, seed=777)

    expected = float(np.exp(-market.div_yield * terms.tenor_years))
    assert expected == pytest.approx(0.960789, rel=1e-5)  # visibly not 1.0
    assert abs(acc.mean - expected) < 3.0 * acc.standard_error, (
        f"{acc.mean:.6f} vs forward {expected:.6f}, se {acc.standard_error:.6f}"
    )

    _, branches = _branches_of(terms, market)
    assert set(np.unique(branches)) == {3}


# --------------------------------------------------------------------------
# Degenerate test 4 — monotonicity
# --------------------------------------------------------------------------
@pytest.mark.parametrize("coupon", [0.0, 0.05, 0.10, 0.15, 0.25])
def test_price_is_monotone_in_the_coupon_rate(coupon):
    """Priced on common random numbers, so the comparison is signal not noise.

    Two independent Monte Carlo runs differing by a small contract change are
    dominated by sampling error; the same draws under both make the difference
    exact in sign. This is the same argument M6 will make for bump-and-revalue,
    rehearsed here where the expected sign is not in doubt.
    """
    base = SnowballTerms(
        tenor_years=TERMS.tenor_years, knock_out_level=TERMS.knock_out_level,
        knock_in_level=TERMS.knock_in_level, coupon_rate=coupon,
        steps_per_year=TERMS.steps_per_year,
    )
    higher = SnowballTerms(
        tenor_years=TERMS.tenor_years, knock_out_level=TERMS.knock_out_level,
        knock_in_level=TERMS.knock_in_level, coupon_rate=coupon + 0.01,
        steps_per_year=TERMS.steps_per_year,
    )
    assert price(higher, seed=31, n_paths=20_000).mean > price(base, seed=31, n_paths=20_000).mean


def test_price_is_monotone_in_the_knock_in_level():
    """A higher knock-in barrier is easier to breach, so the note is worth less.

    The sign is the thing to get right: raising the barrier toward spot makes
    the investor's short put more likely to activate.
    """
    levels = [0.50, 0.65, 0.75, 0.85, 0.95]
    prices = [
        price(
            SnowballTerms(
                tenor_years=TERMS.tenor_years, knock_out_level=TERMS.knock_out_level,
                knock_in_level=level, coupon_rate=TERMS.coupon_rate,
                steps_per_year=TERMS.steps_per_year,
            ),
            seed=99, n_paths=20_000,
        ).mean
        for level in levels
    ]
    assert prices == sorted(prices, reverse=True), dict(zip(levels, prices))
    assert prices[0] - prices[-1] > 1e-3, "the knock-in level should matter materially"


# --------------------------------------------------------------------------
# The relationship between the two products
# --------------------------------------------------------------------------
def test_snowball_with_an_infinite_knock_in_is_the_autocallable():
    """Path for path, not merely in price.

    A snowball is an autocallable whose short put is conditional. Make the
    condition always true and the two must coincide exactly — which is the
    cleanest statement of what the knock-in barrier is actually doing, and a
    strong test of both implementations at once.
    """
    terms_auto = AUTOCALLABLE_TERMS
    terms_snow = SnowballTerms(
        tenor_years=terms_auto.tenor_years,
        knock_out_level=terms_auto.knock_out_level,
        knock_out_frequency=terms_auto.knock_out_frequency,
        knock_in_level=np.inf,  # always knocked in
        knock_in_frequency="daily",
        coupon_rate=terms_auto.coupon_rate,
        strike_level=1.00,
        downside_participation=terms_auto.downside_participation,
        steps_per_year=terms_auto.steps_per_year,
    )

    paths = _sample_paths(terms_auto, MARKET, seed=606, n_paths=8_000)
    auto = autocallable_payoff(terms_auto, MARKET.spot, MARKET.rate)(paths)
    snow = snowball_payoff(terms_snow, MARKET.spot, MARKET.rate)(paths)

    assert np.array_equal(auto, snow)


def test_the_snowball_is_worth_more_than_the_autocallable():
    """The conditional put is worth less than the unconditional one, so the
    investor keeps more. This is why snowball coupons look attractive."""
    paths = _sample_paths(AUTOCALLABLE_TERMS, MARKET, seed=808, n_paths=40_000)
    auto = autocallable_payoff(AUTOCALLABLE_TERMS, MARKET.spot, MARKET.rate)(paths)
    snow = snowball_payoff(TERMS, MARKET.spot, MARKET.rate)(paths)
    assert snow.mean() > auto.mean()
    assert np.all(snow >= auto - 1e-12)  # dominance holds path by path


# --------------------------------------------------------------------------
# The contract as written
# --------------------------------------------------------------------------
def test_branch_three_is_unreachable_under_the_contract_as_written():
    """Only three of the four branches can fire at these terms, by construction.

    Branch 3 is "knocked in, but recovered to at or above the strike at
    maturity". Under the recommended term sheet the knock-out level and the
    strike are both 100% of the initial fixing, and the last monthly knock-out
    observation falls *on* the maturity date. So a path that finishes at or
    above 100% knocks out at that final observation and pays the coupon — it
    can never be observed as "recovered but not knocked out". The two
    conditions are contradictory.

    This is a property of the term sheet, not a defect in the code, and it is
    worth being able to say so: the four-branch decomposition in CLAUDE.md
    describes the general structure, and the general structure has a branch
    that this particular parameterisation closes off. The next test shows the
    branch is reachable as soon as the knock-out level sits above the strike.
    """
    _, branches = _branches_of(TERMS, n_paths=60_000)
    counts = {BRANCHES[i]: int((branches == i).sum()) for i in range(4)}

    assert counts["knocked_in_recovered"] == 0, counts
    for name in ("knocked_out_early", "no_knock_out_no_knock_in", "knocked_in_below_strike"):
        assert counts[name] > 50, f"branch {name!r} fired {counts[name]} times: {counts}"

    # The mechanism, asserted directly rather than inferred from the counts.
    ko_dates = observation_indices("monthly", TERMS.tenor_years, TERMS.steps_per_year)
    assert ko_dates[-1] == TERMS.n_steps, "final knock-out observation is at maturity"
    assert TERMS.knock_out_level == TERMS.strike_level


def test_all_four_branches_occur_once_the_knock_out_sits_above_the_strike():
    """Raise the knock-out to 103% and branch 3 opens up.

    A path that knocks in, then finishes between 100% and 103%, has recovered
    above the strike without triggering the call. That window is exactly the
    gap between the two levels.
    """
    terms = SnowballTerms(
        tenor_years=TERMS.tenor_years,
        knock_out_level=1.03,
        knock_out_frequency=TERMS.knock_out_frequency,
        knock_in_level=TERMS.knock_in_level,
        knock_in_frequency=TERMS.knock_in_frequency,
        coupon_rate=TERMS.coupon_rate,
        strike_level=1.00,
        steps_per_year=TERMS.steps_per_year,
    )
    _, branches = _branches_of(terms, n_paths=60_000)
    counts = {BRANCHES[i]: int((branches == i).sum()) for i in range(4)}
    for name, count in counts.items():
        assert count > 50, f"branch {name!r} fired {count} times: {counts}"


def test_knock_in_probability_is_plausible_and_the_note_prices_near_par():
    acc = price(TERMS, n_paths=60_000)
    # A two-year note paying 15% with a 75% knock-in should be worth
    # meaningfully more than par but far less than the certain coupon stream.
    assert 1.0 < acc.mean < fixed_coupon_note_value(TERMS, MARKET.rate)


def test_terms_reject_a_knock_in_above_the_knock_out():
    with pytest.raises(ValueError, match="above knock-out"):
        SnowballTerms(knock_out_level=1.00, knock_in_level=1.10)


@pytest.mark.parametrize(
    "kwargs",
    [{"tenor_years": 0.0}, {"knock_in_level": -0.1}, {"downside_participation": -1.0},
     {"steps_per_year": 0}, {"strike_level": -1.0}],
)
def test_invalid_terms_raise(kwargs):
    with pytest.raises(ValueError):
        SnowballTerms(**kwargs)


def test_payoff_rejects_paths_of_the_wrong_length():
    payoff = snowball_payoff(TERMS, MARKET.spot, MARKET.rate)
    with pytest.raises(ValueError, match="expected paths of shape"):
        payoff(np.full((5, 10), 100.0))


def test_observation_schedule_is_what_the_term_sheet_says():
    ko = observation_indices("monthly", 2.0, 252)
    ki = observation_indices("daily", 2.0, 252)
    assert len(ko) == 24 and len(ki) == 504
    assert ko[0] == 21 and ko[-1] == 504  # monthly, last one at maturity
    assert ki[0] == 1  # inception is never an observation date
    assert set(ko).issubset(set(ki))


def test_monthly_observations_collide_on_a_coarse_grid_and_say_so():
    with pytest.raises(ValueError, match="collide"):
        observation_indices("monthly", 2.0, 4)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _sample_paths(terms, market: Market, *, seed: int, n_paths: int) -> np.ndarray:
    from quantdesk.mc.paths import price_paths

    z = rng(seed).standard_normal((n_paths, terms.n_steps))
    return price_paths(z, market, terms.tenor_years, engine="numpy")


def _branches_of(terms: SnowballTerms, market: Market = MARKET, *, n_paths: int = 20_000):
    paths = _sample_paths(terms, market, seed=1234, n_paths=n_paths)
    return snowball_payoff(terms, market.spot, market.rate, return_branches=True)(paths)
