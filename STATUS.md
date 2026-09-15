# STATUS

Milestone checklist. The agent maintains this file; §7 of `CLAUDE.md` says each
session starts by reading it and resuming from the first incomplete milestone.

**Current position: M0 complete. Next: M1 — closed-form Black–Scholes.**

| # | Milestone | State | Headline |
|---|---|---|---|
| M0 | Scaffold and determinism harness | **done** | — |
| M1 | Closed-form Black–Scholes | not started | — |
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
