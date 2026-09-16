"""Generate REPORT.md from the newest result JSON of each benchmark.

    python -m quantdesk.report

`REPORT.md` is never edited by hand. Every number in it is read out of a
committed result JSON, together with the seed, the git commit, the environment
and the command that produced it, so that trap 17 — a number in the report that
no command reproduces — is structurally impossible rather than merely avoided.

A headline whose benchmark has never been run does not appear. Half a result is
not a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quantdesk.results import latest_result, project_root, results_dir

__all__ = ["Headline", "HEADLINES", "render", "write_report"]


@dataclass(frozen=True)
class Row:
    label: str
    path: str  # dotted path into the result's "values"
    fmt: str = "{:.6f}"
    note: str = ""


@dataclass(frozen=True)
class Headline:
    number: int
    title: str
    benchmark: str
    command: str
    summary: str
    rows: list[Row] = field(default_factory=list)
    input_rows: list[Row] = field(default_factory=list)


HEADLINES: list[Headline] = [
    Headline(
        number=1,
        title="Variance reduction: antithetic sampling plus a control variate",
        benchmark="variance_reduction",
        command="python bench/variance_reduction.py",
        summary=(
            "Four arms at equal total path count on an arithmetic-average Asian call, "
            "controlled by a European call on the same paths whose expectation is known "
            "exactly in closed form. The reduction is quoted **both** as a standard-error "
            "reduction and as a variance reduction, because they are the same fact in "
            "different units and a bare percentage is ambiguous between them."
        ),
        rows=[
            Row("Standard error reduction, arm A to arm D", "headline_se_reduction_pct_A_to_D", "{:.2f}%"),
            Row("Variance reduction, arm A to arm D", "headline_variance_reduction_pct_A_to_D", "{:.2f}%"),
            Row("Equivalent path-count speedup", "speedup_equivalent_paths", "{:.2f}x",
                "paths the crude estimator would need to match arm D's standard error"),
            Row("Arm A — crude, standard error", "arms.A_crude.standard_error"),
            Row("Arm B — antithetic, standard error", "arms.B_antithetic.standard_error"),
            Row("Arm C — control variate, standard error", "arms.C_control_variate.standard_error"),
            Row("Arm D — both, standard error", "arms.D_antithetic_plus_cv.standard_error"),
            Row("Arm A estimate", "arms.A_crude.estimate"),
            Row("Arm D estimate", "arms.D_antithetic_plus_cv.estimate"),
            Row("Control variate beta (plain arms)", "control_variate.beta_plain", "{:.6f}",
                "estimated on an independent pilot; **not** hardcoded to 1"),
            Row("Control variate beta (antithetic arms)", "control_variate.beta_antithetic", "{:.6f}"),
            Row("Correlation on the main sample (plain)", "control_variate.main_sample_correlation_plain"),
            Row("Theoretical variance ratio 1 - rho^2 (plain)", "control_variate.theoretical_variance_ratio_plain"),
            Row("Realised variance ratio (plain)", "control_variate.realised_variance_ratio_plain"),
            Row("Theoretical variance ratio 1 - rho^2 (antithetic)", "control_variate.theoretical_variance_ratio_antithetic"),
            Row("Realised variance ratio (antithetic)", "control_variate.realised_variance_ratio_antithetic"),
        ],
        input_rows=[
            Row("Product", "product", "{}"),
            Row("Control", "control", "{}"),
            Row("Total paths per arm", "total_paths_per_arm", "{:,}"),
            Row("Time steps", "n_steps", "{:,}"),
            Row("Chunk size (paths)", "chunk_paths", "{:,}"),
            Row("Pilot paths for beta", "pilot_paths", "{:,}"),
        ],
    ),
]


def _dig(payload: dict[str, Any], dotted: str) -> Any:
    node: Any = payload
    for part in dotted.split("."):
        node = node[part]
    return node


def _format(value: Any, fmt: str) -> str:
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


def _environment_block(env: dict[str, Any]) -> list[str]:
    packages = ", ".join(f"{k} {v}" for k, v in sorted(env["packages"].items()) if v)
    dirty = env.get("git_dirty")
    dirty_note = ""
    if dirty:
        dirty_note = (
            "  \n  **Working tree was dirty at run time** — this number is reproducible "
            "only from that tree, not from the commit alone."
        )
    return [
        f"- **Commit**: `{env.get('git_commit') or 'uncommitted'}`"
        f" (branch `{env.get('git_branch') or 'unknown'}`){dirty_note}",
        f"- **Python**: {env['python_version']} ({env['python_implementation']})",
        f"- **Libraries**: {packages}",
        f"- **Machine**: {env['cpu_model']}, {env['machine']}, {env['platform']}",
    ]


def render(headlines: list[Headline] | None = None, directory: Path | None = None) -> str:
    headlines = HEADLINES if headlines is None else headlines
    target = results_dir() if directory is None else directory
    root = project_root()

    lines: list[str] = [
        "# REPORT",
        "",
        "**Generated by `python -m quantdesk.report`. Do not edit by hand.**",
        "",
        "Every number below is read from a committed result JSON in `results/`, "
        "alongside the seed, commit and machine that produced it. Re-running the "
        "stated command on the stated commit reproduces it exactly.",
        "",
        "`DEFENSE.md` carries one entry per number: what it means, the mechanism, "
        "what it is sensitive to, what would make it wrong, and the honest limitation.",
        "",
    ]

    available, missing = [], []
    for headline in headlines:
        try:
            available.append((headline, latest_result(headline.benchmark, directory=target)))
        except FileNotFoundError:
            missing.append(headline)

    lines += [f"Headline numbers available: **{len(available)} of {len(headlines)}**.", ""]

    for headline, payload in available:
        values, inputs = payload["values"], payload["inputs"]
        json_path = Path(payload["_path"])
        try:
            rel = json_path.relative_to(root)
        except ValueError:
            rel = json_path

        lines += [
            "---",
            "",
            f"## Headline #{headline.number} — {headline.title}",
            "",
            headline.summary,
            "",
            "### Result",
            "",
            "| quantity | value |",
            "|---|---|",
        ]
        for row in headline.rows:
            try:
                raw = _dig(values, row.path)
            except KeyError:
                continue
            cell = _format(raw, row.fmt)
            label = f"{row.label}<br><sub>{row.note}</sub>" if row.note else row.label
            lines.append(f"| {label} | `{cell}` |")

        lines += ["", "### Inputs", "", "| input | value |", "|---|---|"]
        for row in headline.input_rows:
            try:
                raw = _dig(inputs, row.path)
            except KeyError:
                continue
            lines.append(f"| {row.label} | `{_format(raw, row.fmt)}` |")
        market = inputs.get("market")
        if isinstance(market, dict):
            terms = ", ".join(f"{k}={v}" for k, v in sorted(market.items()))
            lines.append(f"| Market | `{terms}` |")

        lines += [
            "",
            "### Reproducing it",
            "",
            "```bash",
            headline.command,
            "```",
            "",
            f"- **Seed**: `{payload['seed']}` (registered in `config.SEEDS`)",
            *_environment_block(payload["environment"]),
            f"- **Result JSON**: [`{rel}`]({rel})",
            f"- **Run at**: {payload['timestamp_utc']}",
        ]
        if payload.get("notes"):
            lines += ["", f"> {payload['notes']}"]
        lines.append("")

    if missing:
        lines += [
            "---",
            "",
            "## Not yet produced",
            "",
            "These headline numbers have no result JSON, so they have no value here "
            "and no résumé bullet.",
            "",
        ]
        lines += [f"- **#{h.number}** {h.title} — `{h.command}`" for h in missing]
        lines.append("")

    return "\n".join(lines)


def write_report(path: Path | None = None) -> Path:
    target = project_root() / "REPORT.md" if path is None else path
    target.write_text(render())
    return target


if __name__ == "__main__":
    written = write_report()
    print(f"wrote {written.relative_to(project_root())}")
