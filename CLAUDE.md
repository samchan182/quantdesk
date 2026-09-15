# quantdesk — Build Specification and Agent Brief

You are building this repository from scratch with the author. Read this entire
file before writing any code. Everything below is binding.

---

## 0. Mission

This repository exists to survive a hostile technical interview.

The author will put a small number of specific quantitative results on a résumé.
An interviewer will pick one of them and ask follow-up questions two or three
layers deep. The repository succeeds if, and only if, every headline number is:

1. produced by code in this repo,
2. reproducible from a fixed seed with a single command,
3. explainable by the author in terms of the mechanism that produced it.

A repository that runs but produces numbers the author cannot defend is a
failure. A repository that produces fewer, smaller, less impressive numbers that
are all real is a success.

---

## 1. Non-negotiable rules

These override any instinct to be helpful, fast, or impressive.

**R1 — Never fabricate a number.** Do not hardcode, estimate, illustrate,
placeholder, or "for example" any quantitative result. Every number that reaches
`REPORT.md` must come from an actual run on the author's machine.

**R2 — Never tune toward the target numbers in this document.** Section 4 lists
expected magnitudes taken from the author's earlier draft. They are sanity
checks, not targets. If the measured variance reduction is 41% instead of 62%,
write 41%. If the barrier correction moves the price by 9bp instead of 31bp,
write 9bp. Changing parameters, path counts, or contract terms in order to land
on a number in this document is the single worst thing you can do here, because
it produces exactly the kind of number that collapses under questioning.

**R3 — No look-ahead bias.** Any execution algorithm that consumes information
not available at the moment of the decision is broken. Flag it, do not ship it.

**R4 — Fail loudly.** If a module is incomplete, it raises. Do not stub, do not
mock, do not return a plausible-looking default. An empty result is honest; a
fabricated one is not.

**R5 — Determinism.** Every benchmark runs from an explicit seed. Two runs of
the same command on the same machine produce identical numbers. If something is
inherently non-deterministic (wall-clock latency), state that clearly and report
the distribution, not a single value.

**R6 — Stay inside the declared stack.** Python, NumPy, SciPy, numba, pandas,
psycopg, PostgreSQL, Docker, pytest. Ask before adding anything else. Do not
pull in QuantLib or any pricing library — the point of this repo is that the
author wrote the pricing.

**R7 — Ask when the specification is ambiguous.** Contract terms, data sources,
and conventions (basis points of what?) are the author's decisions, not yours.

**R8 — One milestone at a time.** Complete it, run its acceptance test, update
`STATUS.md` and `REPORT.md`, commit. Do not work ahead.

---

## 2. Physical foundations

Two facts generate almost every number in this repository. Keep them in view;
they are what makes the results explainable rather than merely measured.

**Monte Carlo error falls as 1/√N.** Buying accuracy with paths is brutally
expensive — four times the compute for half the error. Variance reduction is the
only lever that beats the square root, which is why the variance reduction study
is the first headline result and not a footnote.

**Market impact rises as √(participation).** Cost per share grows with the
square root of the order's size relative to available volume, so total cost grows
with the 3/2 power of order size. Every statement about execution cost rising
superlinearly with participation rate is a direct consequence of this, not a
curve fitted to backtest output.

---

## 3. Repository layout

```
quantdesk/
├── CLAUDE.md                  # this file
├── STATUS.md                  # milestone checklist, you maintain it
├── REPORT.md                  # generated; every headline number lives here
├── DEFENSE.md                 # generated; one entry per headline number
├── CV_LINES.md                # generated; final résumé bullets, real numbers only
├── README.md
├── docker-compose.yml         # postgres
├── pyproject.toml
├── src/quantdesk/
│   ├── config.py              # seeds, market params, contract terms
│   ├── analytics/
│   │   ├── blackscholes.py    # closed form, ground truth
│   │   └── barrier.py         # continuity correction reference formulas
│   ├── mc/
│   │   ├── paths.py           # GBM generator, numpy reference + numba kernel
│   │   ├── estimators.py      # antithetic pairing, control variate, standard error
│   │   ├── greeks.py          # pathwise, likelihood ratio, bump-and-revalue
│   │   └── bridge.py          # Brownian bridge barrier crossing
│   ├── products/
│   │   ├── vanilla.py
│   │   ├── autocallable.py
│   │   └── snowball.py
│   ├── marketdata/
│   │   ├── loader.py          # 5-minute bars into postgres
│   │   └── volume_curve.py    # intraday volume profile
│   ├── execution/
│   │   ├── impact.py          # square-root impact model
│   │   ├── algos.py           # TWAP, VWAP, fixed participation
│   │   └── shortfall.py       # implementation shortfall vs arrival price
│   ├── risk/
│   │   ├── checks.py          # four pre-trade controls, in-memory
│   │   └── latency.py         # measurement harness
│   └── lifecycle/
│       ├── states.py          # order state machine
│       └── recon.py           # independent position and cash reconciliation
├── bench/                     # one module per headline number
├── tests/
├── sql/                       # schema DDL
└── results/                   # timestamped JSON, one per benchmark run
```

Every `bench/` module writes a timestamped JSON into `results/` containing: the
number, the seed, all inputs, the git commit hash, the Python and NumPy
versions, and for any timing measurement the CPU model. `REPORT.md` is
regenerated from the newest JSON of each benchmark, never edited by hand.

---

## 4. Milestones

Work through these in order. Each has a goal, a build spec, an acceptance test
that must pass before moving on, and the artifact it emits.

### M0 — Scaffold and determinism harness

**Build.** Repo layout above. `docker-compose.yml` bringing up PostgreSQL.
`pytest` wired. A `results/` writer that captures seed, git hash, versions,
hardware. A single `rng(seed)` factory using `numpy.random.default_rng` — no
module-level global random state anywhere in the codebase, ever.

**Acceptance.** `pytest` runs green on an empty suite. Writing two result JSONs
with the same seed produces identical payloads apart from the timestamp.

---

### M1 — Closed-form Black–Scholes (the ground truth)

Build this first. It is the only source of truth in the repository, and six
later milestones check themselves against it.

**Build.** European call and put price, plus delta, vega, gamma, under
continuous dividend yield `q`:

```
d1 = (ln(S/K) + (r - q + σ²/2)·T) / (σ·√T)
d2 = d1 - σ·√T

Call = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)
Put  = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)

Δ_call = e^(-qT)·N(d1)
Vega   = S·e^(-qT)·φ(d1)·√T          # per 1.00 of volatility
Γ      = e^(-qT)·φ(d1) / (S·σ·√T)
```

**Acceptance.**
- Put-call parity `C - P = S·e^(-qT) - K·e^(-rT)` holds to machine precision
  across a grid of moneyness, maturities, and volatilities.
- Analytic delta and vega agree with a central finite difference of the price
  function to 1e-6.
- Degenerate cases behave: σ → 0 collapses to the discounted intrinsic value,
  T → 0 collapses to intrinsic.

**Emits.** Nothing to `REPORT.md`. This is infrastructure.

---

### M2 — GBM path engine

**Build.** Write the plain NumPy version first and make it correct. Only then
write the numba kernel, and keep the NumPy version permanently as the reference
that the kernel is tested against. Debugging numba without a known-good
reference is a trap; do not walk into it.

Terminal and path-dependent simulation of

```
S_{t+Δ} = S_t · exp((r - q - σ²/2)·Δ + σ·√Δ·Z)
```

**Memory constraint, handle it explicitly.** 200,000 paths × 252 steps of
float64 is roughly 403 MB in one allocation. Do not allocate it. Simulate in
chunks — 10,000 paths at a time, 20 chunks — accumulating only the statistics
each product needs (running minimum, observation-date prices, payoff, payoff
squared). Record this decision in `DEFENSE.md`; it is a better answer to "how
did you handle scale" than any throughput figure.

**Antithetic sampling, and the trap in it.** Generate `N` normal draws, use both
`+Z` and `-Z`. The pair average is one sample:

```
Y_i = (f(Z_i) + f(-Z_i)) / 2
estimate = mean(Y)
standard error = std(Y, ddof=1) / √N        # N pairs, NOT 2N paths
```

Computing the standard error over 2N individual paths inflates the apparent
variance reduction and is wrong. Write a test that catches it: on a payoff that
is linear in `Z`, antithetic sampling must give exactly zero variance, so the
correctly computed standard error must be 0 to floating-point noise. An
incorrectly pooled standard error will not be.

**Acceptance.**
- numba kernel and NumPy reference agree bit-for-bit given the same seed, or to
  1e-12 if the reduction order differs.
- Simulated `E[S_T]` matches `S_0·e^((r-q)T)` within two standard errors.
- Simulated European call price converges to M1's closed form; error shrinks in
  proportion to 1/√N across N = 10⁴, 10⁵, 10⁶ (fit the slope on a log-log plot
  and check it is near -0.5).
- The linear-payoff antithetic test above passes.

---

### M3 — Variance reduction study → **headline #1**

**Build.** `bench/variance_reduction.py` runs four arms at equal total path
count on the same product:

| arm | description |
|---|---|
| A | crude Monte Carlo, 2N paths |
| B | antithetic only, N pairs |
| C | control variate only, 2N paths |
| D | antithetic + control variate, N pairs |

Control variate: use a European vanilla on the same paths, whose expectation is
known exactly from M1.

```
β* = Cov(X, Y) / Var(Y)
estimate = mean(X) - β*·(mean(Y) - E[Y])
theoretical variance ratio = 1 - ρ²(X, Y)
```

Do **not** hardcode β = 1. It is optimal only when the correlation is exactly 1,
and using it otherwise weakens or reverses the benefit. Estimate β on a small
independent pilot run rather than on the same sample used for the estimate; note
in `DEFENSE.md` that in-sample β estimation introduces an O(1/N) bias, which is
why you separated them.

**Acceptance.**
- All four arms agree on the price within two standard errors of the widest arm.
- Realised variance reduction from the control variate is consistent with
  `1 - ρ²` computed on the same sample.

**Emits.** Standard error for each arm; the percentage reduction from A to D.
Report the measured value. Author's earlier draft expected roughly 62%.

---

### M4 — Structured products

The hard part here is the contract, not the code.

**Build.** Autocallable and snowball, with terms in `config.py` and documented
in plain language in `DEFENSE.md`: knock-out barrier level and observation
frequency (typically monthly), knock-in barrier level and observation frequency
(typically daily close), coupon rate, tenor, participation on the downside.

Ask the author to confirm the terms. Do not invent a product.

Draw the payoff explicitly as four branches: knocked out early; never knocked
out and never knocked in; knocked in but recovered above strike at maturity;
knocked in and below strike at maturity.

**Acceptance — degenerate tests, these are the whole point.**
- Knock-out barrier at infinity and knock-in barrier at zero: the snowball must
  reduce to a fixed-coupon note, priced equal to the discounted coupon stream to
  within Monte Carlo noise.
- Knock-out barrier at the spot on day one: must knock out immediately with
  probability 1 and price to the discounted first coupon.
- Coupon set to zero and barriers removed: must reduce to a discounted forward.
- Price is monotone in coupon rate, and monotone in the knock-in level.

---

### M5 — Monte Carlo validated against closed form → **headline #2**

**Build.** `bench/closed_form_agreement.py`. Price a European payoff through the
full production Monte Carlo stack — same path engine, same estimators, same code
path as the structured products — and compare to M1.

Two conventions you must fix and state explicitly:
- Basis points **of the premium**, not of notional. Say so in the report.
- Report the Monte Carlo standard error in the same units, alongside the
  difference.

**This is where an interviewer will push.** A European payoff under GBM has no
discretisation error, so the observed gap should be pure Monte Carlo noise. If
the measured difference exceeds two standard errors, there is a real bias in the
engine — find it before proceeding. Do not report agreement you cannot justify.

**Emits.** The difference in basis points of premium, and the standard error
next to it. Author's earlier draft expected 1–3bp.

---

### M6 — Greeks: pathwise versus bump-and-revalue → **headline #3**

This milestone carries more mathematical weight than any other. Take the time.

**Pathwise derivatives under GBM.** Since `∂S_T/∂S_0 = S_T/S_0`:

```
Δ_pathwise = e^(-rT)·E[ 1{S_T > K} · S_T/S_0 ]

∂S_T/∂σ = S_T·(√T·Z - σT)
Vega_pathwise = e^(-rT)·E[ 1{S_T > K} · ∂S_T/∂σ ]
```

**The validity condition, which is the interview question.** The pathwise method
requires the payoff to be Lipschitz continuous in the parameter, differentiable
almost everywhere. A knock-out feature is a jump in the payoff — the pathwise
estimator is simply invalid there and will silently return a wrong number rather
than error.

Handle it honestly. Two acceptable routes, and you must implement one and
document which:

*Route A — likelihood ratio for the discontinuous part.* The score function
estimator differentiates the density instead of the payoff, so it tolerates
jumps at the cost of higher variance:

```
Δ_LR = e^(-rT)·E[ payoff · Z / (S_0·σ·√T) ]
```

Validate it on a digital option, where the closed-form delta is known.

*Route B — smoothing.* Replace the indicator at the barrier with a smooth
approximation of controlled width, and show the pricing bias introduced as the
width shrinks.

Whatever the result, `DEFENSE.md` must state plainly which parts of which
product use which estimator and why. "Pathwise holds for the vanilla and
knock-in legs; the knock-out jump uses likelihood ratio" is a strong answer.
Claiming pathwise works everywhere is a weak one that fails on the first
follow-up.

**Bump-and-revalue, done correctly.** The two revaluations must share common
random numbers — identical seed, identical draws. Without them the difference
between two noisy prices is dominated by noise rather than signal; demonstrate
this by running one comparison with independent seeds and recording how much
worse it is. That contrast is itself a good `DEFENSE.md` entry.

Bump size: relative, around 1%. Run a sweep from 1e-4 to 1e-1 and plot the error
against bump size. It is U-shaped — noise amplification dominates at small
bumps, truncation error at large ones. Report where the minimum sits.

**On the compute ratio — read this carefully.** Central-difference bumping costs
2 revaluations for delta, 2 for vega, plus the base price: about 5 pricing runs.
Pathwise produces all sensitivities in a single run, with maybe 10–30% overhead
per path. The honest theoretical ratio is therefore roughly one fifth, not one
tenth. If the author's earlier draft said a tenth, that only holds if the bump
method needs more paths to suppress differencing noise to comparable accuracy —
which it may well. Measure both at *matched standard error on the Greek*, not at
matched path count, and report whatever ratio comes out. State in the report
which matching convention you used.

**Acceptance.**
- Pathwise and bump agree to a stated tolerance on the vanilla, where M1 gives
  the true delta and vega. Report the absolute difference and say what quantity
  it is absolute in — delta is dimensionless and lives in [0,1], so 1e-4 means
  something different than it would for vega.
- Likelihood ratio delta on a digital matches its closed form within noise.
- The bump-size sweep shows the expected U shape.

**Emits.** Agreement tolerance between the two methods; compute ratio at matched
accuracy. Author's earlier draft expected 1e-4 and roughly one tenth.

---

### M7 — Discrete monitoring and the Brownian bridge → **headline #4**

**The idea, in one line.** You observe the price once per step, but it moves
continuously between observations; it can cross a barrier and return unseen.
Discrete monitoring therefore *understates* the crossing probability, leaving too
many paths alive, which *overprices* a knock-out structure.

**First, the judgement call — this matters more than the implementation.**
Ask: what does the contract actually say? If the contract itself is monitored on
daily closing prices and you are simulating with daily steps, your simulation is
exact and **this correction must not be applied**. The correction is only
appropriate when the contract monitors continuously (intraday) and your time grid
is coarser than that.

Determine this with the author before writing the correction. Then write both
cases into `DEFENSE.md`. Knowing when *not* to apply a correction is a stronger
signal of understanding than knowing how to apply it.

**Build.** Conditional on the two endpoints of a step, the probability of
touching an upper barrier `B` has a closed form:

```
upper:  p = exp( -2 · ln(B/S_t) · ln(B/S_{t+Δ}) / (σ²·Δ) )
lower:  p = exp( -2 · ln(S_t/B) · ln(S_{t+Δ}/B) / (σ²·Δ) )
```

At each step, draw a uniform and knock the path out with probability `p`.

**Cross-check.** Independently implement the Broadie–Glasserman–Kou continuity
correction: shift the barrier outward by `B·exp(±β·σ·√Δt)` with
`β ≈ 0.5826` (plus for an upper barrier, minus for a lower one) and price with a
continuous-barrier formula. The two approaches should broadly agree. If they
diverge badly, one of them is wrong — resolve it.

**Sensitivity, because it will be asked.** Sweep the correction's effect across
volatility, barrier distance from spot, and monitoring frequency. The bias grows
when the barrier is close and volatility is high, and shrinks toward zero as
monitoring becomes continuous. Show that limit.

**Acceptance.**
- As step size → 0, the corrected and uncorrected prices converge.
- The sign is right: correcting a knock-out lowers its price.
- The Brownian bridge result and the BGK shift agree within tolerance.

**Emits.** Price difference in basis points, before and after correction, per
structure. Author's earlier draft expected roughly 31bp.

---

### M8 — Market data and the intraday volume curve

**Build.** 20 liquid names, 60 trading days, 5-minute bars. A-share sessions
give 48 bars per day — 24 in the morning, 24 in the afternoon — which is tidier
than Hong Kong's 66. Free sources: akshare, baostock, tushare. Roughly 57,600
rows; PostgreSQL handles this trivially.

Build the intraday volume profile: for each name, take the median share of daily
volume falling in each of the 48 buckets, across the 60 days. The result is the
familiar U shape, heavy at the open and the close.

**Acceptance.**
- Bucket weights sum to 1 per name.
- Data quality report: missing bars, halted sessions, zero-volume buckets, and
  what you did about each.
- The curve is visibly U-shaped. If it is not, something is wrong with the data.

---

### M9 — Execution algorithms and shortfall → **headlines #5 and #6**

**Build.** Three schedules over a parent order:
- **TWAP** — equal slices across all 48 buckets.
- **VWAP** — slices proportional to the **historical** volume curve from M8.
  Using the day's realised volume is look-ahead bias and violates R3.
- **Fixed participation** — target a constant percentage of each bucket's
  realised volume.

**The fill model is the physics of this whole section, and skipping it makes
every number here meaningless.** Do not assume fills at the bar close. Use the
bar's VWAP plus impact:

```
fill_price_i = bar_vwap_i · (1 + side · η · σ_daily · √(q_i / V_i))
η ≈ 0.5 to 1.0     # state your choice and cite that it is a chosen calibration
```

Cost per share grows as √(q/V), so total cost grows as q^{3/2}. That is the
origin of superlinearity in participation rate — a consequence of the model, not
a pattern discovered in the data. Say exactly that in `DEFENSE.md`.

**Implementation shortfall.**

```
arrival_price = open of the bar in which the parent order starts
IS (bp) = side · (avg_fill_price - arrival_price) / arrival_price · 10000
side = +1 buy, -1 sell
```

**Parent order generation.** Define it precisely and write the definition into
the report: how many names, how many days, how many orders per name-day, order
size as a fraction of average daily volume, side assignment, seed. 600 orders
must be reconstructible from that definition alone.

Report the **median**, not the mean — the impact cost distribution has a heavy
right tail.

**Acceptance.**
- A participation-rate sweep shows cost rising superlinearly, and the fitted
  exponent is close to the 1.5 the model implies. If it is not, the impact model
  is not being applied the way you think.
- Setting η = 0 makes shortfall collapse to pure price drift, with a median near
  zero and no systematic ordering between algorithms. This test proves the
  reported spread between VWAP and TWAP comes from impact and not from drift —
  run it, and put the result in the report.
- No function in `execution/` can read a bar with a timestamp later than the
  decision it is informing. Enforce this with a test, not a convention.

**Emits.** Median shortfall for each algorithm; the participation rate above
which cost turns sharply superlinear. Author's earlier draft expected roughly
8.4bp for VWAP, 13.1bp for TWAP, with the turn around 15%.

**One warning to record honestly.** VWAP beating TWAP is partly mechanical:
under a participation-based impact model, VWAP trades more when volume is high
and therefore takes less impact. This is the intended mechanism, not a
coincidence, and not evidence the algorithm is clever. `DEFENSE.md` must say so
in these terms. An interviewer who spots the circularity before the author
acknowledges it will discount everything else in the project.

---

### M10 — Pre-trade risk controls and latency → **headline #7**

**Build.** Four checks at order entry: notional limit, single-name concentration
limit, fat-finger price collar against a reference price, restricted list.

**The data structure choices are the latency budget.** Restricted list as a hash
set, O(1). Current positions as an in-memory dict snapshot. Reference prices in
memory. **Any check that touches the database makes a sub-millisecond p99
impossible** — this is a trap question and the architecture has to make the
answer obvious. The database receives rejection logs after the fact, batched in
blocks of ten thousand via `COPY`, never on the critical path.

**Measurement.** `time.perf_counter_ns()` around each order. Warm up first
— discard the first few thousand. `gc.disable()` around the measured region and
say so in the report. Store per-order durations in a NumPy array and compute the
full distribution: p50, p90, p99, p99.9, max. Reporting only p99 invites "and
your p99.9?"; have the answer.

Record the CPU model and Python version in the result JSON. Latency numbers
without hardware context are not reproducible.

**Replay 1.2 million orders** generated from a seeded distribution, with every
rejection logged against the specific rule that fired.

**Acceptance.**
- Rejection counts per rule match the generator's intent — if 3% of orders are
  constructed to breach the notional limit, roughly 3% are rejected for it.
- Per-check timing breakdown, so "which check is slowest?" has an answer.
- Logging demonstrably off the critical path: timings with logging enabled and
  disabled are statistically indistinguishable.

**Emits.** The full latency distribution across 1.2m orders, with hardware.
Author's earlier draft expected a p99 near 180µs.

---

### M11 — Trade lifecycle and reconciliation → **headline #8**

**Build.** State machine: `NEW → ACKNOWLEDGED → FILLED → CONFIRMED →
PRE_SETTLEMENT_CHECK → SETTLED`, every transition timestamped and persisted, with
illegal transitions rejected.

Daily reconciliation of positions and cash. The reconciliation must recompute
**independently** from the fill blotter and compare against the state machine's
maintained positions. If both sides read the same computed value, it is not
reconciliation and proves nothing.

**Acceptance.**
- Illegal transitions are rejected and tested.
- Reconciliation closes within the stated tolerance on most days.
- State the tolerance's units and justify the threshold.

**On the break.** Run the sixty days honestly. You will almost certainly hit at
least one break; the three common causes are floating-point accumulation in the
cash ledger, corporate action or ex-dividend dates, and the ownership of
unsettled cash across the day boundary. **Whatever you hit, document the root
cause and the trace that found it.** Do not engineer a break to have a story, and
do not paper one over to get a clean sixty. The diagnosis is worth more in an
interview than the clean run.

---

### M12 — Report, defense, and résumé lines

**`REPORT.md`** — generated, never hand-edited. One section per headline number:
the value, the command that produces it, the seed, the git hash, the environment,
and a link to the result JSON.

**`DEFENSE.md`** — one entry per headline number, each covering:
- what the number means, and in what units, of what base
- the mechanism that produces it, in one paragraph of plain language
- what it is most sensitive to
- what would make it wrong, and how you would know
- the honest limitation

**`CV_LINES.md`** — the résumé bullets, written from the actual numbers in
`REPORT.md`. If a milestone was not completed, its bullet does not exist. Half a
result is not a bullet.

---

## 5. Known traps — check these before declaring any milestone done

1. Standard error computed over 2N paths instead of N antithetic pairs.
2. Control variate coefficient hardcoded to 1.
3. Control variate coefficient estimated in-sample without noting the bias.
4. Reporting agreement with closed form without reporting the standard error.
5. Pathwise Greeks applied across a payoff discontinuity.
6. Bump-and-revalue without common random numbers.
7. Compute ratio between Greek methods compared at equal path count rather than
   equal accuracy.
8. Applying the Brownian bridge correction when the contract is genuinely
   discretely monitored.
9. Getting the sign of the discrete-monitoring bias backwards.
10. VWAP built on the day's realised volume instead of the historical curve.
11. Fills assumed at the bar close, with no impact model, making shortfall pure
    drift.
12. Reporting the mean shortfall instead of the median.
13. Any pre-trade check touching the database.
14. Latency reported without hardware, warm-up, or garbage collection handling.
15. Reconciliation that compares a value against itself.
16. Basis points quoted without stating basis points *of what*.
17. Any number in `REPORT.md` that no command reproduces.

---

## 6. Questions this repository must be able to answer

The author will be asked these. After each milestone, confirm the code and
`DEFENSE.md` make the relevant answers available. If one cannot be answered from
what exists, that milestone is not done.

**Pricing and variance reduction**
1. What sample size does your standard error use after antithetic pairing?
2. How did you choose the control variate coefficient, and what happens at 1?
3. Is your closed-form gap a bias or Monte Carlo noise? What is the standard
   error? A European payoff has no discretisation error — so where does the gap
   come from?
4. Are 200,000 paths enough? How do you know?

**Greeks**
5. Does the pathwise method hold for an autocallable? What about the knock-out
   jump?
6. Why is pathwise cheaper than bumping? What is the theoretical ratio, what did
   you measure, and why do they differ?
7. Did you use common random numbers? What happens without them?
8. How did you pick the bump size? What breaks when it is too small, and when it
   is too large?

**Barriers**
9. Does discrete monitoring overprice or underprice a knock-out? Why?
10. Where does the Brownian bridge formula come from, and why is it a product of
    two logarithms?
11. If the contract monitors on daily closes, should you apply the correction?
12. What is your correction most sensitive to?

**Execution**
13. Is your VWAP curve historical or same-day? What breaks if it is same-day?
14. What is your fill model? How much of the shortfall is impact and how much is
    drift?
15. Why is cost superlinear in participation rate?
16. Is VWAP beating TWAP a property of the algorithm, or of your impact model?
17. Why the median rather than the mean?

**Risk and operations**
18. Which of the four checks is slowest, and what is the breakdown?
19. Is logging synchronous? Could you hit that p99 if it were?
20. What was the reconciliation break, and how did you trace it?
21. What are the units of your tolerance, and why that threshold?

**Open-ended**
22. How does this differ from a real sell-side system? (Honest answer: no real
    matching engine, single asset class, no credit or collateral, no capital
    charge calculation, no market data infrastructure.)
23. What is the main risk a dealer carries on a snowball? (Gamma instability
    around the knock-in barrier, and a short vega position — the mechanism behind
    the concentrated dealer losses in 2022.)

---

## 7. Working agreement

At the start of each session, read `STATUS.md` and resume from the first
incomplete milestone. At the end of each milestone: run the acceptance tests,
regenerate `REPORT.md`, update `DEFENSE.md`, tick `STATUS.md`, commit with a
message naming the milestone.

When a measured result contradicts Section 4's expectations, say so directly and
keep the measured result. That is the correct outcome, not a problem to solve.

Nothing goes on the résumé until it has been run.
