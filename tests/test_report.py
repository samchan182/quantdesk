"""The report generator is part of the provenance chain, so it is tested.

Trap 17 is "any number in REPORT.md that no command reproduces". The defence
against it is that REPORT.md is rendered from result JSONs and nothing else —
which is only true if the renderer cannot invent, carry over, or silently drop
a value.
"""

from __future__ import annotations

import re

import pytest

from quantdesk.report import HEADLINES, Headline, Row, render
from quantdesk.results import write_result


@pytest.fixture
def one_headline():
    return [
        Headline(
            number=1,
            title="A test headline",
            benchmark="fake_bench",
            command="python bench/fake.py",
            summary="Summary text.",
            rows=[Row("The number", "answer", "{:.3f}"), Row("Missing", "absent")],
            input_rows=[Row("Paths", "n_paths", "{:,}")],
        )
    ]


def test_absent_benchmark_is_listed_as_not_produced(tmp_path, one_headline):
    out = render(one_headline, directory=tmp_path)
    assert "Headline numbers available: **0 of 1**" in out
    assert "## Not yet produced" in out
    assert "A test headline" in out


def test_values_come_from_the_json(tmp_path, one_headline):
    write_result(
        benchmark="fake_bench",
        seed=4242,
        inputs={"n_paths": 200_000},
        values={"answer": 1.23456789},
        directory=tmp_path,
    )
    out = render(one_headline, directory=tmp_path)

    assert "Headline numbers available: **1 of 1**" in out
    assert "`1.235`" in out  # formatted from the payload
    assert "`200,000`" in out
    assert "`4242`" in out
    assert "python bench/fake.py" in out
    assert "## Not yet produced" not in out


def test_a_row_with_no_matching_value_is_dropped_not_faked(tmp_path, one_headline):
    write_result(
        benchmark="fake_bench", seed=1, inputs={"n_paths": 10},
        values={"answer": 1.0}, directory=tmp_path,
    )
    out = render(one_headline, directory=tmp_path)
    assert "The number" in out
    assert "Missing" not in out  # silently absent, never rendered as a blank or zero


def test_the_newest_run_wins(tmp_path, one_headline):
    write_result(benchmark="fake_bench", seed=1, inputs={"n_paths": 1},
                 values={"answer": 111.0}, directory=tmp_path)
    write_result(benchmark="fake_bench", seed=2, inputs={"n_paths": 2},
                 values={"answer": 222.0}, directory=tmp_path)
    out = render(one_headline, directory=tmp_path)
    assert "222.000" in out and "111.000" not in out


def test_environment_and_provenance_are_rendered(tmp_path, one_headline):
    write_result(benchmark="fake_bench", seed=7, inputs={"n_paths": 1},
                 values={"answer": 1.0}, directory=tmp_path)
    out = render(one_headline, directory=tmp_path)

    assert "**Python**:" in out
    assert "**Machine**:" in out
    assert "**Libraries**:" in out
    assert re.search(r"\*\*Commit\*\*: `[0-9a-f]{40}`", out), "commit hash missing"
    assert "**Result JSON**:" in out


def test_a_dirty_working_tree_is_flagged_in_the_report(tmp_path, one_headline, monkeypatch):
    """A number produced from uncommitted code must say so."""
    import quantdesk.report as report_module

    payload_env = {
        "git_commit": "0" * 40, "git_branch": "main", "git_dirty": True,
        "python_version": "3.13.5", "python_implementation": "CPython",
        "packages": {"numpy": "2.1.3"}, "platform": "test", "machine": "arm64",
        "cpu_model": "Test CPU",
    }
    monkeypatch.setattr(
        report_module,
        "latest_result",
        lambda bench, directory=None: {
            "values": {"answer": 1.0}, "inputs": {"n_paths": 1}, "seed": 1,
            "environment": payload_env, "timestamp_utc": "2026-01-01T00:00:00+00:00",
            "notes": None, "_path": str(tmp_path / "x.json"),
        },
    )
    out = render(one_headline, directory=tmp_path)
    assert "Working tree was dirty" in out


def test_real_headlines_are_well_formed():
    """Each declared headline must name a command and a benchmark."""
    assert HEADLINES, "no headlines declared"
    for headline in HEADLINES:
        assert headline.benchmark and headline.command.startswith("python")
        assert headline.rows, f"headline #{headline.number} declares no rows"
        assert headline.summary
