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
