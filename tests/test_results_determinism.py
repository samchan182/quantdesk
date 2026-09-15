"""M0 acceptance — two result JSONs written at the same seed are identical
apart from the timestamp."""

from __future__ import annotations

import json

import numpy as np
import pytest

from quantdesk.results import (
    SCHEMA_VERSION,
    environment,
    latest_result,
    read_result,
    write_result,
)
from quantdesk.rng import rng


def _run(seed: int) -> dict[str, float]:
    """A stand-in benchmark: a number that depends only on the seed."""
    draws = rng(seed).standard_normal(10_000)
    return {"mean": float(draws.mean()), "std": float(draws.std(ddof=1))}


def test_two_runs_same_seed_identical_apart_from_timestamp(tmp_path):
    seed = 20260915
    p1 = write_result(
        benchmark="determinism_selftest",
        seed=seed,
        inputs={"n_draws": 10_000},
        values=_run(seed),
        directory=tmp_path,
    )
    p2 = write_result(
        benchmark="determinism_selftest",
        seed=seed,
        inputs={"n_draws": 10_000},
        values=_run(seed),
        directory=tmp_path,
    )

    assert p1 != p2, "each run must write its own file"

    a, b = read_result(p1), read_result(p2)
    assert a.pop("timestamp_utc") != b.pop("timestamp_utc")
    assert a == b

    # And the payloads are byte-identical once the timestamp line is gone.
    def _strip(path):
        return [ln for ln in path.read_text().splitlines() if "timestamp_utc" not in ln]

    assert _strip(p1) == _strip(p2)


def test_payload_carries_full_provenance(tmp_path):
    path = write_result(
        benchmark="determinism_selftest",
        seed=1,
        inputs={"n_draws": 10},
        values={"mean": 0.0},
        directory=tmp_path,
    )
    payload = read_result(path)

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["seed"] == 1
    env = payload["environment"]
    # Recorded because a number without these is not reproducible.
    assert env["python_version"]
    assert env["packages"]["numpy"]
    assert env["cpu_model"]
    assert "git_commit" in env and "git_dirty" in env


def test_numpy_scalars_serialise(tmp_path):
    path = write_result(
        benchmark="determinism_selftest",
        seed=1,
        inputs={"n": np.int64(5), "grid": np.array([1.0, 2.0])},
        values={"x": np.float64(1.5), "ok": np.bool_(True)},
        directory=tmp_path,
    )
    payload = json.loads(path.read_text())
    assert payload["inputs"] == {"n": 5, "grid": [1.0, 2.0]}
    assert payload["values"] == {"x": 1.5, "ok": True}


def test_non_finite_values_are_refused(tmp_path):
    """A NaN in a result means the run broke; it must not reach a JSON (R4)."""
    with pytest.raises(ValueError, match="non-finite"):
        write_result(
            benchmark="determinism_selftest",
            seed=1,
            inputs={},
            values={"price": float("nan")},
            directory=tmp_path,
        )


def test_unserialisable_value_raises_rather_than_stringifying(tmp_path):
    with pytest.raises(TypeError):
        write_result(
            benchmark="determinism_selftest",
            seed=1,
            inputs={"market": object()},
            values={"price": 1.0},
            directory=tmp_path,
        )


def test_latest_result_picks_newest_and_raises_when_absent(tmp_path):
    with pytest.raises(FileNotFoundError):
        latest_result("determinism_selftest", directory=tmp_path)

    write_result(
        benchmark="determinism_selftest",
        seed=1,
        inputs={},
        values={"price": 1.0},
        directory=tmp_path,
    )
    newest = write_result(
        benchmark="determinism_selftest",
        seed=2,
        inputs={},
        values={"price": 2.0},
        directory=tmp_path,
    )
    got = latest_result("determinism_selftest", directory=tmp_path)
    assert got["seed"] == 2
    assert got["_path"] == str(newest)


def test_environment_is_stable_within_a_process():
    assert environment() == environment()
