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
