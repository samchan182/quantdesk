"""The single source of randomness in this repository.

Rule R5 (determinism) has exactly one enforcement point: every random number in
this codebase comes from a ``numpy.random.Generator`` handed out by :func:`rng`,
seeded explicitly by the caller. There is no module-level global random state
anywhere, and ``tests/test_no_global_random_state.py`` fails the build if any is
introduced.

Why ``default_rng`` and not ``numpy.random.seed``: the legacy functions share one
process-wide Mersenne Twister. Any library, any import, any test-ordering change
can consume draws from it, so "same seed" stops meaning "same numbers". A
``Generator`` object is owned by the caller, so a benchmark's stream depends on
its own seed and nothing else.
"""

from __future__ import annotations

import numpy as np

__all__ = ["rng", "spawn"]


def rng(seed: int) -> np.random.Generator:
    """Return a fresh PCG64 generator seeded with ``seed``.

    Two calls with the same seed return independent generator objects that
    produce identical streams.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    return np.random.default_rng(int(seed))


def spawn(seed: int, n: int) -> list[np.random.Generator]:
    """Return ``n`` independent generators derived from one seed.

    Used where work is split into chunks (M2 simulates in blocks of paths) and
    each chunk needs its own stream. ``SeedSequence.spawn`` guarantees the
    children are statistically independent, which sequential seeds
    (``seed``, ``seed+1``, ...) do not.

    The result depends only on ``(seed, n)``, so a chunked run is reproducible
    and — importantly for M2 — independent of how many chunks are used only if
    ``n`` is held fixed. Chunk count is therefore part of a benchmark's recorded
    inputs, not a free parameter.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    children = np.random.SeedSequence(int(seed)).spawn(int(n))
    return [np.random.default_rng(child) for child in children]
