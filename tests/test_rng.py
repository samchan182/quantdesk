"""M0 — the determinism harness itself."""

from __future__ import annotations

import numpy as np
import pytest

from quantdesk.rng import rng, spawn


def test_same_seed_same_stream():
    assert np.array_equal(rng(7).standard_normal(1000), rng(7).standard_normal(1000))


def test_different_seeds_differ():
    assert not np.array_equal(rng(7).standard_normal(1000), rng(8).standard_normal(1000))


def test_generators_are_independent_objects():
    """Two generators from one seed must not share state.

    This is the property the legacy ``np.random.seed`` global lacks, and the
    reason the whole codebase is built on Generator objects.
    """
    a, b = rng(3), rng(3)
    a.standard_normal(500)  # advance one
    assert np.array_equal(b.standard_normal(10), rng(3).standard_normal(10))


def test_spawn_is_deterministic_and_independent():
    first = [g.standard_normal(200) for g in spawn(11, 4)]
    second = [g.standard_normal(200) for g in spawn(11, 4)]
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a, b)

    # Distinct children: no pair of chunk streams coincides.
    for i in range(len(first)):
        for j in range(i + 1, len(first)):
            assert not np.array_equal(first[i], first[j])


def test_spawn_children_are_not_merely_offset_seeds():
    """spawn(s, n) must not degenerate to rng(s), rng(s+1), ...

    Sequential seeds are a classic way to get correlated streams; SeedSequence
    exists precisely to avoid it.
    """
    child = spawn(11, 2)[1].standard_normal(100)
    assert not np.array_equal(child, rng(12).standard_normal(100))


@pytest.mark.parametrize("bad", [-1, 2.5, "7", True, None])
def test_bad_seeds_raise(bad):
    with pytest.raises((TypeError, ValueError)):
        rng(bad)


def test_spawn_rejects_zero():
    with pytest.raises(ValueError):
        spawn(1, 0)
