"""Result JSON writer: the provenance layer for every number in REPORT.md.

Rule R17 of the trap list — "any number in REPORT.md that no command
reproduces" — is prevented here rather than by discipline. A benchmark does not
print its answer; it writes a JSON file carrying the value *together with*
everything needed to re-derive it: the seed, every input, the git commit, the
interpreter and library versions, and the CPU the timings were taken on.
REPORT.md is generated from these files and never hand-edited.

The payload is deterministic apart from ``timestamp_utc``: two runs of the same
benchmark at the same seed on the same commit produce byte-identical content
once that one field is removed. ``tests/test_results_determinism.py`` asserts it.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "SCHEMA_VERSION",
    "project_root",
    "results_dir",
    "environment",
    "write_result",
    "read_result",
    "latest_result",
]

SCHEMA_VERSION = 1

_TRACKED_PACKAGES = ("numpy", "scipy", "numba", "pandas")


# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def project_root() -> Path:
    """Repository root, located by walking up to the directory holding pyproject.toml."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(f"could not locate project root above {here}")


def results_dir() -> Path:
    d = project_root() / "results"
    d.mkdir(exist_ok=True)
    return d


# --------------------------------------------------------------------------
# Environment capture
# --------------------------------------------------------------------------
def _git(*args: str) -> str | None:
    """Run a git command in the project root; return stripped stdout or None."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=project_root(),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _cpu_model() -> str:
    """Best available CPU description. Latency numbers are meaningless without it."""
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "unknown"


@lru_cache(maxsize=1)
def environment() -> dict[str, Any]:
    """Everything about this machine and checkout that a reproducer needs.

    Cached: it is constant for the life of the process, and caching keeps
    repeated benchmark writes from shelling out to git each time.

    ``git_commit`` is ``None`` only before the repository's first commit.
    ``git_dirty`` true means the working tree differed from that commit at the
    moment of the run — the number is then reproducible only from the same
    working tree, which REPORT.md flags.
    """
    versions: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", None)
        except ImportError:
            versions[name] = None

    status = _git("status", "--porcelain")
    commit = _git("rev-parse", "HEAD")

    return {
        "git_commit": commit,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "packages": versions,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
    }


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------
def _jsonable(obj: Any) -> Any:
    """Convert numpy scalars and arrays to JSON types; raise on anything else.

    Raising rather than falling back to ``str(obj)`` is deliberate (R4): a value
    silently serialised as its repr looks like a number in the report and is not
    one.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
            # NaN/inf are not valid JSON; they signal a broken run, so refuse.
            raise ValueError(f"refusing to serialise non-finite value {obj!r}")
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return _jsonable(float(obj))
    if isinstance(obj, np.ndarray):
        return [_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    raise TypeError(
        f"cannot serialise {type(obj).__name__} into a result JSON; "
        "convert it to a plain number, string, list or dict at the call site"
    )


# --------------------------------------------------------------------------
# Write / read
# --------------------------------------------------------------------------
def write_result(
    *,
    benchmark: str,
    seed: int,
    inputs: dict[str, Any],
    values: dict[str, Any],
    notes: str | None = None,
    directory: Path | None = None,
) -> Path:
    """Write one benchmark result and return the path.

    Parameters
    ----------
    benchmark:
        Slug identifying the benchmark; also the filename prefix and the key
        REPORT.md groups by.
    seed:
        The seed the run used. Recorded at top level because it is the first
        thing anyone reproducing the number needs.
    inputs:
        Every parameter that affects the result — path counts, market
        parameters, contract terms, chunk sizes. If changing it changes the
        number, it belongs here.
    values:
        The measured results. Standard errors belong here alongside the
        estimates they qualify (trap 4).
    notes:
        Optional free text recording a judgement made during the run.
    """
    if not benchmark or not all(c.isalnum() or c in "_-" for c in benchmark):
        raise ValueError(
            f"benchmark must be a non-empty slug of alphanumerics, '_' or '-', got {benchmark!r}"
        )

    now = datetime.now(timezone.utc)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": benchmark,
        "timestamp_utc": now.isoformat(),
        "seed": int(seed),
        "inputs": _jsonable(inputs),
        "values": _jsonable(values),
        "notes": notes,
        "environment": environment(),
    }

    target = directory if directory is not None else results_dir()
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{benchmark}__{now.strftime('%Y%m%dT%H%M%S_%f')}Z.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def read_result(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def latest_result(benchmark: str, directory: Path | None = None) -> dict[str, Any]:
    """Newest result for a benchmark, by the timestamp inside the payload.

    Raises if there is none: REPORT.md must not be generatable for a benchmark
    that has never been run (R1).
    """
    target = directory if directory is not None else results_dir()
    candidates = sorted(target.glob(f"{benchmark}__*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"no result JSON for benchmark {benchmark!r} in {target}; run it first"
        )
    payloads = [read_result(p) | {"_path": str(p)} for p in candidates]
    return max(payloads, key=lambda p: p["timestamp_utc"])
