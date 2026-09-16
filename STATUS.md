# STATUS

Milestone checklist. The agent maintains this file; §7 of `CLAUDE.md` says each
session starts by reading it and resuming from the first incomplete milestone.

**Current position: M1 complete. Next: M2 — GBM path engine.**

| # | Milestone | State | Headline |
|---|---|---|---|
| M0 | Scaffold and determinism harness | **done** | — |
| M1 | Closed-form Black–Scholes | **done** | — |
| M2 | GBM path engine | not started | — |
| M3 | Variance reduction study | not started | #1 |
| M4 | Structured products | not started | — |
| M5 | MC vs closed form | not started | #2 |
| M6 | Greeks: pathwise vs bump | not started | #3 |
| M7 | Discrete monitoring / Brownian bridge | not started | #4 |
| M8 | Market data and volume curve | not started | — |
| M9 | Execution algorithms and shortfall | not started | #5, #6 |
| M10 | Pre-trade risk and latency | not started | #7 |
| M11 | Trade lifecycle and reconciliation | not started | #8 |
| M12 | Report, defense, CV lines | not started | — |

---

## M0 — done

**Built.** Repository layout per §3. `pyproject.toml` (hatchling, src layout,
pytest wired with `pythonpath = ["src"]`). `docker-compose.yml` bringing up
PostgreSQL 16 with a healthcheck, `sql/` mounted as init DDL, UTC and `C` locale
pinned. `src/quantdesk/rng.py` — the single `rng(seed)` factory over
`numpy.random.default_rng`, plus `spawn(seed, n)` for the chunked simulation M2
needs. `src/quantdesk/results.py` — the result-JSON writer capturing value,
seed, inputs, git commit and dirty flag, Python and library versions, platform
and CPU model. `src/quantdesk/config.py` — seed registry and market parameters.

**Acceptance — both met.**

1. *pytest runs green.* 23 tests pass (`python3 -m pytest`). The suite is not
   empty: it covers the RNG factory, the result writer, and the global-state
   guard below.
2. *Two result JSONs at the same seed are identical apart from the timestamp.*
   Asserted in `tests/test_results_determinism.py` both as a dict comparison and
   as a line-by-line file diff, and demonstrated for real by
   `python bench/determinism_selftest.py`, which writes two files into
   `results/` and diffs them.

**Beyond the letter of the acceptance.** "No module-level global random state
anywhere, ever" is enforced by `tests/test_no_global_random_state.py`, which
walks the AST of every file in `src/` and `bench/` and fails on any use of
`np.random.*` or the stdlib `random` module; `default_rng` itself is confined to
`rng.py`. The guard has its own test proving it catches violations, because a
guard that never fires is indistinguishable from no guard.

**Deviation from the spec, flagged.** §3 lists `REPORT.md`, `DEFENSE.md` and
`CV_LINES.md` in the layout. `REPORT.md` and `CV_LINES.md` do **not** exist yet:
both are generated from headline numbers, the first of which arrives at M3, and
a hand-written placeholder in a file whose contract says "generated, never
edited by hand" is the wrong thing to have in the repository. `DEFENSE.md` does
exist, because M2 requires a design decision to be written into it before any
headline number is produced.

---

## M1 — done

**Built.** `src/quantdesk/analytics/blackscholes.py`: European call and put
price, delta, vega and gamma under a continuous dividend yield `q`. Vectorised
over any broadcastable inputs, returning a float when every input is scalar. No
dependency on anything else in the package — this is the reference the rest of
the repository is measured against, so it is written to be obviously correct
rather than fast.

**Acceptance — all three met, on a 2,880-point grid** (5 spots × 3 strikes ×
4 rates including a negative one × 3 dividend yields × 4 vols × 4 maturities
from one day to five years).

| check | required | measured |
|---|---|---|
| put-call parity | machine precision | **2.0 ulp** (2.84e-14 absolute, 2.88e-16 relative to term size) |
| delta vs central difference | 1e-6 | **5.37e-08** absolute, calls and puts |
| vega vs central difference | 1e-6 | **2.81e-08** absolute, per 1.00 of vol |
| σ → 0 | discounted intrinsic | exact, bit-for-bit |
| T → 0 | intrinsic | exact, bit-for-bit |

Parity is measured relative to the size of the *terms*, not of their
difference. The two terms can nearly cancel — at S=90, K=100, r=2%, T=5 they
are both ≈90 and differ by −0.48 — and scaling by that residual would report
the conditioning of the subtraction rather than the accuracy of the formula.

**Four things the grid exposed that are worth having answers to.**

1. *Vega is not zero at σ = 0.* Away from the money it vanishes, but exactly at
   the money forward `d1 → 0` and vega tends to `S·e^(-qT)·φ(0)·√T`. Handled
   explicitly and tested; the easy version of this function returns zero
   everywhere and is wrong at one point.
2. *Vega and gamma underflow to exactly zero* at 180 of the 2,880 points — deep
   out-of-the-money one-day options where `d1 ≈ −149`, so the true value is
   around 1e-4836 against a smallest representable double of 5e-324. Zero is
   the correct answer to return, and no floor is applied. It also means a
   *relative* error on a Greek is meaningless once the Greek has underflowed,
   which M6 will have to respect.
3. *Monotonicity in vol fails at two points, by 1.4e-14.* Both are one-day
   options 20+ points in the money where vega is ~1e-13, so a 1% vol bump moves
   the true price by ~3e-16 — below the representable granularity of a double
   near 20. Asserted to within ulps there, strictly everywhere it is
   resolvable.
4. *The finite-difference bump-size sweep is U-shaped, as M6 will need.*
   Gamma via a difference of analytic delta: 2.13e-6, 2.12e-8, **5.29e-10**,
   7.73e-9 at h = 1e-5, 1e-6, 1e-7, 1e-8 × S — truncation falling as h², then
   cancellation taking over. Establishing the shape here, where the true answer
   is known exactly and there is no Monte Carlo noise, is the cheap way to know
   the M6 sweep measures what it claims to.

**Emits.** Nothing to `REPORT.md`, per the spec. This is infrastructure.

**One external anchor.** Every other check is internal consistency — parity,
finite differences, monotonicity — and all of them would still pass if the
module were wrong by a common factor. So one test pins absolute values for
S=K=100, r=5%, q=0, σ=20%, T=1, computed independently at 50 decimal digits
with mpmath using `erf` rather than SciPy's `ndtr`, and agreeing to 1e-15
relative: call 10.450583572185566782, put 5.5735260222569676908, delta
0.63683065117561907122, vega 37.524034691693787837, gamma
0.018762017345846893919. mpmath is **not** a dependency; it was used once at
the terminal to generate the constants, and the command is in the test
docstring.

---

## Open questions for the author

None blocking M1 or M2. Raised early so there is time to think; each is asked
again at the milestone that needs it.

- **M4 — contract terms.** Autocallable and snowball: knock-out level and
  observation frequency, knock-in level and frequency, coupon rate, tenor,
  downside participation. R7 makes these the author's call and R4 forbids
  inventing a product, so M4 stops until they are decided.
- **M7 — monitoring convention.** Does the contract monitor the barrier
  continuously (intraday) or on daily closes? If daily closes and the simulation
  steps daily, the Brownian bridge correction **must not be applied** (trap 8),
  and M7 becomes a demonstration of that judgement rather than of the formula.
- **M8 — data source.** `akshare` is installed; `baostock` and `tushare` are
  not. Which 20 names, and which 60 trading days?
- **Market parameters.** `config.BASE_MARKET` is currently spot 100, r 2%, q 0,
  vol 20% — round numbers chosen for legibility, not calibrated to anything.
  Flat vol, flat rates, no smile. Say if you want different ones.
- **Docker.** The CLI is installed but the daemon is not running. Not needed
  until M8.
