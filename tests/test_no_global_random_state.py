"""Enforces R5's structural half: no module-level global random state, anywhere.

A convention that says "always use quantdesk.rng" is broken the first time
someone writes ``np.random.normal(...)`` in a hurry, and the failure is silent —
the code still runs, the numbers still look plausible, and reproducibility is
gone. So this is a test, not a convention.

Allowed uses of ``numpy.random``:
  * ``default_rng`` — only inside ``quantdesk/rng.py``, the one factory;
  * ``Generator`` and ``SeedSequence`` — anywhere, they are types, not state.
Everything else in the ``numpy.random`` namespace draws from the process-wide
legacy MT19937, and the stdlib ``random`` module is one global generator by
construction.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from quantdesk.results import project_root

SCANNED_DIRS = ("src", "bench")
TYPE_ONLY = {"Generator", "SeedSequence", "BitGenerator", "PCG64"}
FACTORY = "default_rng"
FACTORY_OWNER = Path("src/quantdesk/rng.py")


def _python_files() -> list[Path]:
    root = project_root()
    files: list[Path] = []
    for d in SCANNED_DIRS:
        files.extend(sorted((root / d).rglob("*.py")))
    return files


def _dotted(node: ast.AST) -> str | None:
    """Reconstruct a dotted name from an Attribute/Name chain, or None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _violations(path: Path, rel: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[str] = []

    for node in ast.walk(tree):
        # import random  /  from random import ...
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "random" or alias.name.startswith("random."):
                    found.append(f"{rel}:{node.lineno} imports the stdlib `random` module")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "random":
                found.append(f"{rel}:{node.lineno} imports from the stdlib `random` module")
            elif module in ("numpy.random", "numpy.random.mtrand"):
                for alias in node.names:
                    if alias.name in TYPE_ONLY:
                        continue
                    if alias.name == FACTORY and rel == FACTORY_OWNER:
                        continue
                    found.append(
                        f"{rel}:{node.lineno} imports `{alias.name}` from numpy.random"
                    )
        elif isinstance(node, ast.Attribute):
            dotted = _dotted(node)
            if dotted is None:
                continue
            parts = dotted.split(".")
            if len(parts) < 3 or parts[0] not in ("np", "numpy") or parts[1] != "random":
                continue
            member = parts[2]
            if member in TYPE_ONLY:
                continue
            if member == FACTORY and rel == FACTORY_OWNER:
                continue
            found.append(f"{rel}:{node.lineno} uses `{dotted}`")

    return found


def test_no_global_random_state_in_src_or_bench():
    root = project_root()
    offences: list[str] = []
    for path in _python_files():
        offences.extend(_violations(path, path.relative_to(root)))

    assert not offences, (
        "global random state found — every draw must come from quantdesk.rng.rng(seed):\n  "
        + "\n  ".join(offences)
    )


def test_the_guard_actually_catches_things(tmp_path):
    """A guard that never fires is indistinguishable from no guard."""
    bad = tmp_path / "bad.py"
    bad.write_text("import numpy as np\nx = np.random.normal(size=3)\n")
    assert _violations(bad, Path("src/quantdesk/bad.py"))

    also_bad = tmp_path / "also_bad.py"
    also_bad.write_text("import random\nrandom.seed(1)\n")
    assert _violations(also_bad, Path("src/quantdesk/also_bad.py"))

    ok = tmp_path / "ok.py"
    ok.write_text("from quantdesk.rng import rng\nx = rng(1).standard_normal(3)\n")
    assert not _violations(ok, Path("src/quantdesk/ok.py"))


def _references_factory(path: Path) -> bool:
    """True if the file *uses* default_rng, as opposed to merely naming it.

    AST rather than substring: bench modules record the generator's name in
    their result JSON as a string, and that is documentation, not a call site.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    return any(
        (isinstance(node, ast.Name) and node.id == FACTORY)
        or (isinstance(node, ast.Attribute) and node.attr == FACTORY)
        for node in ast.walk(tree)
    )


def test_default_rng_is_confined_to_the_factory():
    """Only rng.py may call default_rng; anyone else must go through rng()."""
    root = project_root()
    callers = [p.relative_to(root) for p in _python_files() if _references_factory(p)]
    assert callers == [FACTORY_OWNER], f"default_rng used outside the factory: {callers}"


@pytest.mark.parametrize("directory", SCANNED_DIRS)
def test_scanned_directories_exist(directory):
    assert (project_root() / directory).is_dir()
