"""M0 acceptance, run for real: two result JSONs at one seed differ only in time.

    python bench/determinism_selftest.py

Emits two files into results/ and diffs them. Nothing here reaches REPORT.md —
M0 is infrastructure — but the two committed files are the evidence that the
provenance layer every later number depends on actually behaves.
"""

from __future__ import annotations

import sys

import numpy as np

from quantdesk.config import seed_for
from quantdesk.results import read_result, write_result
from quantdesk.rng import rng

BENCHMARK = "determinism_selftest"
N_DRAWS = 1_000_000


def measure(seed: int) -> dict[str, float]:
    """A stand-in for a real benchmark: numbers determined entirely by the seed."""
    draws = rng(seed).standard_normal(N_DRAWS)
    return {
        "sample_mean": float(draws.mean()),
        "sample_std": float(draws.std(ddof=1)),
        "standard_error_of_mean": float(draws.std(ddof=1) / np.sqrt(N_DRAWS)),
    }


def main() -> int:
    seed = seed_for(BENCHMARK)
    inputs = {"n_draws": N_DRAWS, "generator": "PCG64 via numpy.random.default_rng"}

    paths = [
        write_result(benchmark=BENCHMARK, seed=seed, inputs=inputs, values=measure(seed))
        for _ in range(2)
    ]

    first, second = (read_result(p) for p in paths)
    t1, t2 = first.pop("timestamp_utc"), second.pop("timestamp_utc")

    identical = first == second
    for path in paths:
        print(f"wrote {path.relative_to(path.parents[1])}")
    print(f"timestamps differ:            {t1 != t2}")
    print(f"payloads otherwise identical: {identical}")
    print(f"sample mean at seed {seed}:  {first['values']['sample_mean']:+.12e}")

    if not identical:
        print("FAIL: payloads differ beyond the timestamp", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
