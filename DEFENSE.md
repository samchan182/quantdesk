# DEFENSE

One entry per headline number, plus the engineering decisions that headline
numbers rest on. Each entry answers the same five questions: what the number
means and in what units of what base, the mechanism that produces it, what it is
most sensitive to, what would make it wrong and how you would know, and the
honest limitation.

No headline numbers exist yet — the first arrives at M3. What follows are the
design decisions made so far.

---

## D0.1 — All randomness comes from an explicitly seeded Generator

**The decision.** Every random draw in this repository comes from
`quantdesk.rng.rng(seed)`, which returns a fresh `numpy.random.Generator`
(PCG64). There is no module-level random state, and `np.random.seed` is never
called.

**Why it matters.** The legacy `np.random.*` functions all draw from one
process-wide Mersenne Twister. Any import, any library, any change in test
ordering can consume draws from it, so "same seed" quietly stops meaning "same
numbers" — and nothing errors when it happens. A `Generator` object is owned by
its caller, so a benchmark's stream depends on its own seed and nothing else.
This is what makes R5 hold in practice rather than in intention.

**How it is enforced.** Not by convention. `tests/test_no_global_random_state.py`
parses every file in `src/` and `bench/` and fails the build on any use of
`np.random.*` (beyond the `Generator` and `SeedSequence` types) or the stdlib
`random` module; `default_rng` is confined to `rng.py` alone. The guard is
itself tested against known-bad snippets.

**Chunked runs.** M2 simulates in blocks rather than one 403 MB allocation, and
each block needs its own stream. `spawn(seed, n)` derives `n` children through
`SeedSequence.spawn`, which guarantees independence. Sequential seeds
(`seed`, `seed+1`, …) do not: they are a known way to get correlated streams, and
a test asserts `spawn(s, 2)[1]` is not `rng(s+1)`. Because the children depend on
`n`, the chunk count changes the numbers and is therefore recorded as a
benchmark input, not left as a free parameter.

**Honest limitation.** Determinism holds for a fixed machine, NumPy version and
chunk count. A different NumPy could change a distribution's internal algorithm
and move the last digits; this is why the result JSON records library versions
rather than assuming them.

---

## D0.2 — Numbers are written to result JSONs, never printed

**The decision.** A benchmark does not report its answer to the terminal. It
writes a JSON into `results/` carrying the value together with the seed, every
input, the git commit and whether the tree was dirty, the Python and library
versions, the platform and the CPU model. `REPORT.md` is generated from those
files.

**Why it matters.** Trap 17 is "any number in `REPORT.md` that no command
reproduces". Making provenance a side effect of producing the number, rather
than a discipline applied afterwards, is what closes that gap. It also answers
the reproduction question directly: the file says which commit, which seed,
which machine.

**Two specific choices.** The writer refuses to serialise NaN or infinity — a
non-finite result means the run broke, and it must not reach a report looking
like a number. It also refuses unknown types rather than falling back to
`str(obj)`, which would put a value's repr into a report where a number belongs.
Both are R4 applied to the plumbing.

**Honest limitation.** The commit hash is only as useful as the tree is clean.
`git_dirty` records whether it was; a dirty run is reproducible only from the
same working tree, and the report has to say so.

---

## D1.1 — Vega is quoted per 1.00 of volatility

**The convention.** `vega()` returns ∂Price/∂σ with σ in absolute units. A vega
of 37.52 means the price rises by 37.52 if volatility goes from 20% to 120%, to
first order. Desks usually quote vega per *point* of vol, which is this number
divided by 100.

**Why it is stated here.** Trap 16 is basis points quoted without saying of
what, and this is the same failure in a different unit. M6 reports an agreement
tolerance between two ways of computing vega; that tolerance is meaningless
unless the scale of the quantity is fixed. Delta is dimensionless and lives in
[0, 1], so an absolute agreement of 1e-4 on delta is a far stronger claim than
1e-4 on a vega of order 40. The report must say which quantity a tolerance is
absolute in.

---

## D1.2 — The degenerate limits are computed, not approximated

**The decision.** `sigma = 0` and `T = 0` are handled by an explicit branch
returning the exact limit, rather than by letting the arithmetic run with a
tiny denominator. `d1_d2` returns NaN where `sigma·√T = 0`, so anything that
forgets to branch fails loudly instead of silently using a huge finite number.

**The mechanism.** With σ = 0 the terminal price is deterministic, so the option
is worth the discounted intrinsic value **of the forward**:
`max(S·e^(-qT) − K·e^(-rT), 0)`. At T = 0 both factors are 1 and that same
expression collapses to the spot intrinsic `max(S − K, 0)`. One expression
covers both cases, which is why one branch does.

**The subtlety worth knowing.** Vega is *not* uniformly zero as σ → 0. Away from
the money forward, `d1` diverges and `φ(d1)` dies faster than `1/d1`, so vega
→ 0. But exactly at the money forward, `d1 = σ√T/2 → 0`, and vega tends to
`S·e^(-qT)·φ(0)·√T`, which is not zero. Gamma at that same point diverges, and
`inf` is returned rather than a large finite number. These are measure-zero
points, and returning a plausible-looking zero at them would be the kind of
quiet wrongness R4 exists to prevent.

**What would make it wrong, and how you would know.** If the branch were
mis-specified the continuous formula would not walk into it. The tests check
that it does — and check the *rate*: at the money forward the price is linear in
σ, so the gap to the limit falls by exactly 10× per tenfold cut in σ, which is
asserted rather than merely the ordering.

---

## D1.3 — Float64 has a floor, and the Greeks reach it

**The observation.** On the M1 acceptance grid, 180 of 2,880 points return vega
and gamma of exactly zero. These are deep out-of-the-money one-day options where
`d1 ≈ −149`, so `φ(d1) = exp(−11135)/√(2π)`. The true vega is around 1e-4836;
the smallest representable double is 5e-324.

**Why it is not patched.** Zero is the nearest representable value and therefore
the correct answer. Flooring it at some small positive number to preserve strict
positivity would be inventing a number to satisfy a test.

**What it implies downstream.** A *relative* error on a quantity that has
underflowed is not a statistic. M6 compares pathwise and bumped Greeks and must
restrict relative comparisons to the region where the Greek is representable —
and the same argument applies to the second-difference check on gamma, which
carries an absolute cancellation floor of about `eps·Price/h²` ≈ 1e-11 and so
cannot resolve a gamma of 1e-12 at all.

**The related fact about monotonicity.** Two grid points show call prices
falling by 1.4e-14 when volatility is raised 1%. Both are one-day options 20+
points in the money, where vega is ~1e-13 and the true price gain is ~3e-16 —
below the spacing of a double near 20. Rounding decides the comparison. The
correct response is to assert monotonicity to within ulps there and strictly
everywhere the effect is resolvable, not to claim an exactness float64 does not
have.


---

## D2.1 — Why 200,000 × 252 paths is never allocated

**The decision.** `simulate` runs in chunks of 10,000 paths, reducing each chunk
to one payoff per path and discarding the paths before drawing the next chunk.
Peak path memory is about 20 MB regardless of the total path count.

**The arithmetic.** 200,000 paths × 252 steps × 8 bytes is 403 MB in a single
allocation, and under antithetic sampling two such arrays are alive at once.
Nothing needs them: every product here consumes a per-path statistic — a
terminal price, a running minimum, prices on observation dates, a payoff — and
those are what the chunk loop keeps.

**The consequence that has to be recorded.** Each chunk draws from its own
generator, derived through `SeedSequence.spawn` rather than from sequential
seeds, which are a known route to correlated streams. Because the children
depend on how many were requested, **the chunk count changes the numbers**. Run
the same seed with 20 chunks and with 1 and the estimates differ — by less than
their joint standard error, as a test asserts, but they differ. Chunk size is
therefore a recorded benchmark input, not a free tuning knob, and the result
JSON carries it.

**Honest limitation.** This is memory discipline, not a distributed system.
Everything runs in one process on one machine, and the chunk loop is
sequential. The numba kernel parallelises within a chunk across cores; nothing
parallelises across chunks.

---

## D2.2 — The NumPy reference is permanent, and the kernels agree bit-for-bit

**The decision.** The NumPy path generator is not scaffolding that the numba
kernel replaces. It stays, and it is what the kernel is tested against.

**The result.** The two agree **bit-for-bit** on 2,000 × 252 paths, not merely
to 1e-12. That was designed for: the NumPy version puts the log-spot in column 0
and the increments in columns 1..m before a single `cumsum`, so its accumulation
order is `((log S₀ + inc₀) + inc₁) + …`, exactly the kernel's sequential loop.
Adding `log S₀` to a cumsum of the increments would associate differently and
lose it. The kernel is compiled `fastmath=False`; with fastmath on, LLVM may
reassociate and contract to FMA, and the agreement goes.

**Why it is worth the trouble.** An exact reference turns "the kernel is wrong"
into a one-line assertion. Without it, a discrepancy between a compiled kernel
and a closed-form price has at least three candidate causes — the kernel, the
estimator, the payoff — and no way to separate them.

**Honest limitation.** Bit-for-bit is a claim about this machine, this NumPy,
this LLVM. It is asserted with a documented 1e-12 relative fallback precisely
because a different platform's `exp` may differ in the last ulp.

---

## D2.3 — The antithetic sample is the pair, and the standard error says so

**What the number means.** Under antithetic sampling the estimator's sample
size is the number of **pairs**, N, not the number of paths, 2N. The
accumulator tracks `n_samples` and `n_paths` separately and divides by
`n_samples`.

**The mechanism.** `Y_i = (f(Z_i) + f(−Z_i))/2` is one draw from the
distribution of the pair average. The two paths inside it are negatively
correlated by construction — that is the entire point of the technique — so
treating them as 2N independent observations understates the estimator's
variance and inflates the apparent benefit.

**How it is verified, and why this test rather than another.**
`log(S_T/S₀) = μT + σ√T·Z` is affine in `Z`, so every pair average is exactly
`μT` and the variance across pairs is exactly zero. A correct standard error is
zero to floating-point noise: measured **2.21e-19**. A standard error pooled
over the 2N individual paths instead reports the full spread of the draws,
above 5e-4 — six thousand times larger. The failure is not a few percent of
overstated benefit that could hide in noise; it is visible at a glance. Run at
r=5%, q=1%, σ=20% so that μT = 0.02 is genuinely non-zero; at `BASE_MARKET`,
`r − q − σ²/2` happens to be exactly zero and the test would have pinned
nothing down.

**What would make it wrong.** Antithetic sampling helps only when the payoff is
monotone in the driving normal, so that `f(Z)` and `f(−Z)` are negatively
correlated. For a symmetric payoff — a straddle struck at the forward — the
correlation is near zero and the technique buys almost nothing; for some
non-monotone payoffs it can increase variance. M3 measures the benefit rather
than assuming it.

---

## D2.4 — A European priced in one step has no discretisation error

**The decision.** GBM has a closed-form transition density, so `S_T` is
simulated in a single exact draw rather than by stepping. Stepped simulation
exists for path-dependent products, and a test prices the same European both
ways and requires agreement within the joint standard error.

**Why it matters for M5.** M5 measures the gap between Monte Carlo and closed
form and has to attribute it. Under GBM with a European payoff there is no
discretisation error to attribute it to — each step is exact in law, whether
you take one or 252 — so the entire gap must be sampling noise. If the measured
difference exceeds two standard errors, the engine has a real bias and the
right response is to find it, not to report the agreement.

**The cheapest check that the drift is right.** `E[S_T] = S₀e^((r−q)T)`,
measured at **0.82 standard errors** on two million paths. A companion test
shows this has teeth: drop the `−σ²/2` Itô correction and the simulated mean is
high by `exp(σ²T/2)`, 2% here, which the same test rejects. That error is
invisible in an option price, where it would hide inside the noise.

---

## H1 — Variance reduction: 49.62% in standard error, 74.62% in variance

**What the number means, and in what units of what base.** Arm A is crude Monte
Carlo on 200,000 paths; arm D is antithetic pairing plus a control variate on
the same 200,000 paths. Arm D's standard error is 0.008602 against arm A's
0.017075 — a **49.62% reduction in standard error**, which is the same fact as a
**74.62% reduction in variance**. Both are quoted because "a 62% reduction"
alone does not say which, and the two differ by more than twenty points. The
base is the standard error of the Monte Carlo estimate of the price of one
specific product: an arithmetic-average Asian call struck at 100, one year,
averaged over 252 daily closes, at 20% vol and a 2% rate. It is not a statement
about the repository's Monte Carlo in general.

**The mechanism, in plain language.** Two independent effects compose.
Antithetic sampling evaluates the payoff at `+Z` and `-Z` and averages the pair;
because the Asian payoff is monotone in the driving noise, the two legs are
negatively correlated and their average is less variable than two independent
draws would be. The control variate exploits something different — that a
European call on the *same* paths has a price known exactly in closed form. Its
Monte Carlo error is observable, so subtracting `β` times that observed error
from the Asian estimate removes the component of the Asian's error that moves
with it. What is left is the part of the Asian's variation the European cannot
explain, which is `1 − ρ²` of the original variance.

**What it is most sensitive to: the correlation.** Everything about the control
variate's contribution is `ρ = 0.833`. Push it to 0.95 and the remaining
variance would be 10% rather than 31%; drop it to 0.5 and most of the benefit
goes. The correlation is a property of the pairing of *this* target with *this*
control, so the headline does not transfer to a different product.

**What would make it wrong, and how you would know.**

- *Computing the standard error over 2N paths instead of N pairs.* This would
  inflate the apparent antithetic benefit. The accumulator counts pairs and
  reports `n_samples` and `n_paths` separately; the linear-payoff test in M2
  would fail loudly if the denominator were wrong.
- *Hardcoding β = 1.* Measured β is **0.4618**. Since `Var(X − bY)` is a
  parabola in `b` with its minimum at `β*` and its value back at `Var(X)` when
  `b = 2β*`, and `2 × 0.4618 = 0.92 < 1`, β = 1 would make the variance
  **worse than no control variate at all**. A test evaluates the parabola at
  `2β*` and confirms it returns to `Var(X)`.
- *Estimating β in-sample.* Fitting β to the same draws it is applied to
  correlates the coefficient with the residual it multiplies and leaves an
  O(1/N) bias. β is therefore estimated on an independent 20,000-path pilot on
  its own seed. A test measures the in-sample bias directly and shows it
  shrinking with N.
- *A wrong control expectation.* `E[Y]` comes from M1's closed form. If it were
  wrong, the control variate would bias the estimate rather than merely fail to
  help — and the arms would no longer agree. That is what acceptance check 1
  tests.

**The honest limitations.**

1. *This is one product, at one set of parameters.* The percentage is not a
   property of the technique in the abstract.
2. *Arms A and C share draws, as do B and D.* That is deliberate — it isolates
   the control variate's effect from sampling noise — but it means the A-to-C
   comparison is paired, not independent. The two simulations use different
   seeds.
3. *The product choice was mine, not the author's.* An arithmetic Asian is a
   reasonable target for a European control, but it is not a product anyone
   asked for.
4. *A single control variate caps out at `1 − ρ²`.* Getting materially past
   this would need a better-correlated control — a geometric Asian, which also
   has a closed form and would correlate far more tightly — rather than more
   paths.

---

## H1b — Why the arms share a path count rather than a runtime

**The decision.** All four arms simulate exactly 200,000 paths. The comparison
is at equal path count, not at equal wall-clock time.

**Why.** Equal-path-count is the honest framing for a *statistical* claim: it
answers "for the same simulation work, how much less noisy is the answer". The
control variate's extra cost is one payoff evaluation per path on paths already
generated, which is negligible against generating them; antithetic pairing
costs a second payoff evaluation on the negated draws, likewise. So at this
product the two framings nearly coincide.

**Where the distinction bites, and it is flagged now because M6 turns on it.**
M6 compares pathwise Greeks against bump-and-revalue, and there the two methods
have genuinely different costs per unit of accuracy. That comparison must be
made at **matched standard error on the Greek**, not at matched path count, and
the report has to say which convention was used. The same care is not needed
here, but the convention is still stated rather than assumed.

---

## D4.1 — The snowball, in plain language

**What the investor bought.** A two-year note on an index. Every month the level
is checked: if it is at or above where it started, the note redeems early and
pays back par plus coupon accrued at 15% a year, pro rata to that date. If it
never redeems early, what happens at maturity depends on whether the index ever
*closed* at or below 75% of its starting level:

* it never did — par plus the full two years of coupon;
* it did, but the index finished at or above its starting level — par, and no
  coupon at all;
* it did, and the index finished below — the investor takes the index's loss
  from the starting level, one for one.

**Where the coupon comes from.** The investor has sold a knock-in put. The 15%
is not yield; it is the premium on that option, and it is large because the
option is dangerous. Priced at these terms the note is worth **1.000963** per
unit notional — essentially par — which says the coupon is roughly fair
compensation, not free money.

**Conventions that change the number, so they are stated.** Knock-out takes
precedence over knock-in, so a path that knocks in and later knocks out pays the
coupon. The coupon accrues **simple**, not compounded. The strike is a recovery
*threshold*; redemption on the downside is at the index's performance from the
**initial** level, not from the strike, which makes the payoff continuous at the
strike when the two coincide. Barriers are observed on closing levels.

**The honest limitations, each of which flatters the note.** No three-month
lock-up before the first knock-out date, which real snowballs impose and which
would lower the value. No bid-offer, no funding spread, no dividend basis risk.
Flat volatility and flat rates — and since the dealer's risk here is
concentrated in skew and in gamma near the knock-in, a flat-vol model is
precisely the model least able to speak to it.

---

## D4.2 — Branch 3 is unreachable at these terms, and that is the term sheet's doing

**The observation.** The payoff has four branches. At the recommended terms only
three can ever fire: knocked out early 88.69%, survived with no knock-in 1.30%,
knocked in and below strike 10.01%, knocked in and **recovered 0.00%**.

**The mechanism.** Branch 3 is "knocked in, but recovered to at or above the
strike at maturity". The knock-out level and the strike are both 100% of the
initial fixing, and the final monthly knock-out observation falls *on* the
maturity date. A path finishing at or above 100% therefore knocks out at that
final observation and pays the coupon. "Recovered above the strike" and "not
knocked out" are contradictory.

**Why this is worth saying rather than hiding.** It is a property of the
parameterisation, not a defect in the code, and the distinction is the whole
skill being demonstrated. A test asserts the branch is empty *and* asserts the
mechanism directly — that the last observation index equals the step count and
that the two levels are equal — so the claim rests on the structure rather than
on a Monte Carlo count. A second test raises the knock-out to 103% and shows all
four branches fire, the open window being exactly the 100–103% gap.

**What would make it wrong.** If branch 3 were empty for a reason other than
this — a mis-ordered `np.select`, a wrong comparison direction — the mechanism
assertions would still pass while the count was zero for the wrong reason. That
is why the 103% test exists: it proves the branch is reachable code.

---

## D4.3 — The snowball is the autocallable with a conditional put

**The relationship.** An autocallable pays an accruing coupon, redeems early on
an up-barrier, and leaves the investor short an *unconditional* put at maturity.
A snowball makes that put *conditional* on a lower knock-in barrier having been
breached.

**Asserted, not asserted about.** `snowball_payoff` with the knock-in level at
infinity — always knocked in — equals `autocallable_payoff` **path for path**,
bit for bit. And at the real terms the snowball dominates the autocallable path
by path, pricing at 1.000963 against 0.995788.

**Why it matters.** A conditional put is worth less than an unconditional one,
so the issuer can pay a higher coupon for it. That is the entire commercial
logic of the product, and it means the attractive coupon and the buried risk are
the same fact viewed from two sides.

**The risk a dealer carries, which §6 question 23 asks about.** Short the
knock-in put, the dealer is long gamma nowhere useful and short vega, and the
gamma goes unstable as spot approaches the knock-in barrier: the dealer's hedge
ratio changes discontinuously exactly when the market is moving fastest and
liquidity is worst. When a whole market is positioned the same way — as onshore
China was in 2022 — every dealer needs to sell the same underlying at the same
moment. Nothing in this repository models that crowding; it prices one note in
isolation under flat vol, which is the honest limit of what it can say.

---

## H2 — Monte Carlo agrees with the closed form to −1.25bp of premium

**What the number means, and in what units of what base.** An at-the-money
one-year European call, priced through the full production Monte Carlo stack
over 40 million paths, came out **1.25 basis points of the premium** below M1's
closed form, with a Monte Carlo standard error of **1.75 basis points of the
premium**. The base is the premium — 8.916 — not the notional. The same absolute
difference expressed against a notional of 100 would be 0.11bp, eleven times
smaller, and quoting it that way would be flattering rather than informative.
The difference is −0.71 standard errors, p = 0.481.

**The mechanism.** There is nothing for the difference to be except sampling
noise. GBM has a closed-form transition density, so every step of the scheme is
exact *in law*: simulating `S_T` in one step and in 252 steps samples the same
distribution. A European payoff depends only on `S_T`. So no discretisation
error exists to be measured, and the remaining gap is the finite-sample error of
an average — which shrinks as 1/√N and has no preferred sign.

**What it is most sensitive to: the number of paths, and nothing else.** Double
the paths and the standard error falls by √2 and the difference wanders inside
it. The figure is not sensitive to step count, which is the point: the
discretisation sweep at 1, 4, 12, 63 and 252 steps gives t = +0.25, +0.73, −0.81,
−1.51, −1.76, with no drift.

**What would make it wrong, and how you would know.**

- *A drift error* — a missing `−σ²/2` Itô correction, a dividend yield applied
  with the wrong sign. This is caught structurally, not statistically. Because
  `max(S−K,0) − max(K−S,0) = S − K` exactly, a call and a put priced on
  **identical** paths have an antisymmetric error component that is precisely
  `e^(−rT)(mean S_T − E[S_T])`. A drift error pushes call and put in opposite
  directions and appears there and nowhere else. Measured: t = +0.172, against
  call and put errors of t = +0.172 and +0.167. The identity holds to 5.2e-14.
- *Quoting the difference without the standard error.* This is trap 4, and it is
  what makes a number like "1.25bp" meaningless on its own. The two always
  travel together in the result JSON and in the report.
- *Under-powering the comparison.* Covered at length below, because it actually
  happened.

**The honest limitations.**

1. *This validates the engine against a payoff the engine finds easy.* A
   European under GBM is the one case where the Monte Carlo has no
   discretisation error at all. Agreement here says the plumbing is unbiased —
   the drift, the discounting, the antithetic pairing, the chunked accumulation
   — and says nothing about how the same engine handles a barrier, which is
   where discretisation genuinely bites and which is M7's subject.
2. *Absence of evidence.* With a standard error of 1.75bp, this run can rule out
   a bias larger than roughly 3.5bp. It cannot rule out a bias of 0.5bp. The
   honest claim is "no bias detectable at this precision", not "no bias".
3. *The result is machine- and version-specific* in its last digits, as every
   result JSON in this repository records.

---

## H2b — The flagged put, and why under-powering a check is its own failure mode

**What happened.** The first run of this benchmark reported the at-the-money put
at **−5.60bp with t = −2.26** — past the two-standard-error threshold at which
CLAUDE.md says there is a real bias in the engine and it must be found before
proceeding.

**What was done about it.** Not a tolerance change. The case was escalated,
holding everything else fixed and only adding independent runs:

| runs | difference | t | p |
|---|---|---|---|
| 10 | −5.60 bp | −2.26 | 0.050 |
| 20 | −3.80 bp | −2.28 | 0.034 |
| 40 | −1.29 bp | −0.90 | 0.376 |
| 80 | −0.05 bp | −0.05 | 0.960 |

**Why that is decisive.** A bias is a fixed displacement; its t-statistic grows
as `√n`, so a genuine −5.6bp bias would have become *more* significant with
eight times the data, reaching t ≈ −6. Instead the estimate walked to −0.05bp
and t to −0.05. Noise is the only explanation consistent with that pattern. The
parity decomposition independently agrees: the drift component, where a
put-specific bias would have to live, sits at t = +0.17.

**The lesson, which is the part worth carrying.** The supporting cases were run
at 10 runs against the headline's 40 — a quarter of the statistical power —
while being reported in the same table and judged against the same
two-standard-error rule. That asymmetry manufactured the alarm. Four cases were
tested, so the chance of at least one two-sided p below 0.05 under a perfectly
unbiased engine is about **19%**, not 5%. Both the equal run count and the
multiple-comparisons caveat are now in the benchmark and in the result JSON.

**What this would have looked like done badly.** Report the headline call at
−1.25bp, quietly omit the put, and the repository still runs and still shows
agreement. The number would have been true and the process indefensible — which
is precisely the failure CLAUDE.md's mission section describes.

---

## H3 — Pathwise Greeks cost about a fifth of bump-and-revalue, at matched accuracy

**What the number means, and in what units.** Computing price, delta and vega by
pathwise differentiation costs **0.204** of what computing the same three by
central differencing costs, at **equal standard error on the Greek**. The cost
unit is a pricing run — one pass over the paths — which is deterministic and
reproducible; wall-clock is recorded in the result JSON but flagged
non-deterministic under R5.

**The mechanism.** Central differencing needs five pricing runs: a base price,
spot up, spot down, vol up, vol down. Pathwise differentiates the payoff along
each path and produces all three quantities from a single pass. That is the
whole of the five-to-one. With common random numbers the bumped estimator also
turns out to have essentially the *same variance* as the pathwise one — it needs
0.98× the paths to match — so the matched-accuracy ratio and the matched-path-
count ratio coincide at about a fifth.

**Why the matching convention has to be stated.** At matched path count the
comparison answers "how much work per path", which is the easy question. At
matched accuracy it answers "how much work for an answer of the same quality",
which is the one that matters. They differ whenever the two estimators have
different variances — and they would differ a great deal here if common random
numbers were switched off, since bumping's variance would then be 281× higher.

**Against the author's earlier draft.** The draft said one tenth. One tenth
would require bumping to need roughly twice the paths *on top of* its five
pricing runs. With common random numbers it does not. **One fifth is the honest
answer**, and it is the one CLAUDE.md's own reasoning predicts.

**What would make it wrong.** Counting cost in wall-clock on a machine where the
payoff is trivial relative to path generation would flatter pathwise, because
the marginal cost of a second payoff evaluation on already-generated paths is
near zero. Counting in pricing runs avoids that. Conversely, for a product whose
payoff evaluation dominates path generation — a snowball with 504 observation
dates and four branches — the five-to-one would understate pathwise's advantage.

**The honest limitations.** This is one product, a European, at one parameter
set. Pathwise's advantage grows with the number of sensitivities wanted (each
additional Greek costs bumping two more runs and pathwise almost nothing) and
shrinks to nothing for a payoff where pathwise is invalid and the likelihood
ratio's 5.6× variance penalty has to be paid instead.

---

## H3b — The pathwise method is invalid across a discontinuity, and fails silently

**The claim.** Interchanging the derivative and the expectation requires the
payoff to be Lipschitz continuous in the parameter and differentiable almost
everywhere. A vanilla call qualifies: the kink at the strike is a single point
of probability zero. A digital or a knock-out does **not**: the payoff jumps.

**The demonstration, which is the point.** For a digital struck at the money,
the naive pathwise estimator returns **exactly 0.0**, with a standard error of
**exactly 0.0**, against a closed-form delta of **0.01955213**. The reason is
immediate once stated: the derivative of a step function is zero almost
everywhere, so every per-path sample is zero. The estimator does not raise, does
not warn, and reports a tight confidence interval around a completely wrong
number. A zero standard error makes it look *more* trustworthy than a correct
estimator would.

`mc/greeks.py` contains `pathwise_digital_delta` for no purpose other than to
make this reproducible, with a docstring saying so.

**The fix, and what it costs.** Route A, the likelihood ratio: differentiate the
*density* rather than the payoff, so the payoff is never differentiated and may
jump freely. `Delta_LR = e^(−rT) E[payoff · Z/(S₀σ√T)]`. On the digital it
returns 0.01954183, **−0.72 standard errors** from the closed form. The cost is
variance — the score function grows without bound in the tails — measured at
**5.6× the paths** on a payoff where pathwise is legal. That is why the
assignment is per-leg rather than global.

**Which estimator is used where, stated plainly.** Pathwise for the vanilla and
for any Lipschitz leg. Likelihood ratio wherever the payoff jumps: digitals, and
the snowball's knock-out feature. Claiming pathwise works everywhere is the weak
answer that fails on the first follow-up.

**What would make it wrong.** If the likelihood ratio were validated only on a
*continuous* payoff, the test would pass while saying nothing about the case it
exists for. It is therefore validated on the digital, where the closed-form
delta is known and where pathwise demonstrably fails.

---

## H3c — The bump-size sweep is not U-shaped, and the reason is common random numbers

**The expectation, and the measurement.** CLAUDE.md predicts a U: noise
amplification dominating at small bumps, truncation at large ones. Measured:

| | standard-error amplification, smallest/largest bump | minimum |
|---|---|---|
| without common random numbers | **958.6×** | h = 0.03 |
| with common random numbers | **1.1×** | h = 0.01 |

**The mechanism.** Without common random numbers the numerator `P(S+h) − P(S−h)`
differences two *independently* noisy prices. Its standard error barely depends
on `h`, so dividing by `2h` amplifies it as `1/h` — and the measured scaling is
exactly `1/h` across the grid, a factor of three per threefold step. That rising
left arm, meeting truncation's `h²` right arm, is the U.

With common random numbers the two legs share their draws. The numerator then
shrinks with `h` exactly as the signal does, and the quotient converges to the
pathwise derivative as `h → 0`. Its variance is therefore **bounded**, and the
left arm flattens onto the pathwise standard-error floor — measured at 2.944e-04,
matching the pathwise estimator's own standard error. What remains is flat, then
truncation: a hockey stick.

**Why this matters rather than being a curiosity.** The U shape is a symptom of
doing the bumping wrong. Quoting "we chose the bump at the minimum of the
U-shaped error curve" as evidence of care would, in a correctly implemented
bumper, describe a curve that does not exist. The honest statement is that with
common random numbers the bump size barely matters until truncation bites, and
the choice of 1% is about staying safely below that.

**The honest limitation.** The flat region is flat only down to where floating-
point cancellation would eventually bite. That floor was not reached on this
grid — the smallest bump tested was 1e-4 relative, or 0.01 in spot — so the
sweep establishes bounded variance over four decades of `h`, not over all `h`.

---

## H4 — The monitoring convention is worth 295bp of premium, and the correction is not applied

**What the number means, and in what units of what base.** On a down-and-out
call with the barrier at 90% of spot, 20% volatility, one year, monitored
daily, moving from daily-close monitoring to continuous monitoring changes the
price by **295.06 basis points of the premium** (se 2.15 bp). The base is the
discretely-monitored premium, 7.503625. It is **not** a correction applied to
the reported price. It is the value of a contract term.

**The judgement this rests on, which matters more than the number.** The
contract monitors on daily closes. The simulation steps daily. The two grids
coincide, so the simulation is *exact* — there is no discretisation error to
correct, and applying the Brownian bridge here would be pricing a different,
continuously-monitored contract. Trap 8 is doing exactly that. Knowing when not
to apply a correction is the harder half of knowing the correction.

**The mechanism, and the direction.** Between two observed closes the price
moves continuously; it can dip below the barrier and recover unseen. Discrete
monitoring therefore *understates* the crossing probability, leaves too many
paths alive, and **overprices a knock-out**. Correcting lowers the price — which
is what was measured, 7.503625 → 7.282224.

**Where the formula comes from, and why it is a product of two logarithms.**
Conditional on its endpoints a Brownian motion over one step is a Brownian
bridge, and the probability a bridge from `a` to `b` touches zero is
`exp(−2ab/variance)`. In logs the endpoint distances from the barrier are
`ln(S_t/B)` and `ln(S_{t+Δ}/B)` and the variance is `σ²Δ`. The product of the
two logarithms is the product of the two endpoint distances: the excursion has
to reach the barrier *and* come back, so the probability falls off with the
distance at each end.

**What it is most sensitive to: the barrier's distance from spot, overwhelmingly.**

| barrier | 70% | 80% | 90% | 95% | 98% |
|---|---|---|---|---|---|
| effect | 0.21 bp | 21.80 bp | 280.80 bp | 904.86 bp | 2381.17 bp |

and secondarily to volatility, 26.5 bp at σ=10% rising to 1173.6 bp at σ=50%.
Both directions are what the mechanism requires: a closer barrier and a wilder
process both mean more unobserved excursions.

**What would make it wrong, and how you would know.**

- *The sign.* Trap 9. Checked directly, and also structurally: the bridge can
  only ever *remove* survivors, never resurrect them, which a test asserts.
- *The BGK shift direction.* The effective barrier moves **away** from spot,
  because a discretely monitored barrier behaves as though it sat further off.
  Reversing it gives a correction of the right size and the wrong sign.
- *The bridge being wrong altogether.* Cross-checked two ways against an exact
  answer: the bridge reproduces the continuous closed form to **−0.39 se**, and
  the uncorrected discrete price matches the BGK-shifted closed form to
  **−0.41 se**. These are independent routes; agreeing by accident is
  implausible.
- *The limit.* As the grid refines the correction must vanish. It does, and at
  the right **rate** — proportional to √Δt, halving per fourfold refinement:
  1000, 581, 288, 154, 81 bp. A gap that shrank at the wrong rate would mean
  the right sign for the wrong reason.

**Against the author's earlier draft.** The draft expected roughly 31bp;
295.06bp was measured, nearly ten times larger. Nothing was tuned. The
sensitivity sweep accounts for the difference entirely — 31bp corresponds to a
barrier near 80% of spot, not the 90% that was run — and because the sweep
exists, the answer is available for whichever barrier the contract actually
specifies rather than only for the one measured.

**The honest limitations.**

1. *The bridge is exact only for the one-step conditional law.* It assumes GBM
   between observations with the same constant volatility used to simulate
   them. Under a real volatility surface the crossing probability would differ.
2. *It adds Monte Carlo noise of its own*, since it draws a uniform per path
   per step. Two bridge-corrected runs on different seeds differ; the
   uncorrected daily price does not, because it is a deterministic function of
   the simulated closes.
3. *Daily closes are not the only discrete convention.* A contract monitoring
   intraday lows rather than closes would need a different treatment again, and
   none is implemented.

---

## H4b — Why the snowball's number is small, and why that is the right answer

**The figure.** The snowball's knock-in sits at 75% of spot and is observed on
daily closes. Its price is **1.000801** (se 0.000192). Monitored continuously,
the same note would be worth 1.000096 — the monitoring convention is worth
**7.05 bp of premium**.

**Why so much smaller than the 295bp headline.** Because 75% is far from spot.
The barrier sweep above shows the effect collapsing as the barrier retreats —
0.21 bp at 70%, 21.8 bp at 80% — and 7 bp at 75% sits exactly where that curve
says it should. The two numbers are consistent, not in tension, and the sweep is
what makes that visible rather than something to be asserted.

**The point worth making in an interview.** A correction can be technically
correct, correctly signed, correctly validated — and still be irrelevant to the
contract in front of you. Here it is worth 7 basis points on a note trading at
par, against a coupon of 15% a year, and the contract does not call for it
anyway. Reporting it as a labelled sensitivity rather than folding it into the
price is the difference between understanding the correction and merely having
implemented it.
