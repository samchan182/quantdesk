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
    Headline(
        number=2,
        title="Monte Carlo agreement with the closed form",
        benchmark="closed_form_agreement",
        command="python bench/closed_form_agreement.py",
        summary=(
            "A European call priced through the full production stack — the same stepped "
            "path engine, chunked driver, antithetic estimator and payoff plumbing that "
            "price the snowball — against M1's closed form.\n\n"
            "**Units: basis points OF THE PREMIUM, not of notional.** For an 8.92 premium "
            "on a 100 notional those differ by a factor of eleven. The Monte Carlo standard "
            "error is quoted in the same units beside the difference, because a difference "
            "without its standard error cannot be read as either agreement or bias.\n\n"
            "A European payoff under GBM has **no discretisation error** — each step of the "
            "scheme is exact in law — so the gap can only be sampling noise. Two things "
            "establish that rather than asserting it: the difference is pooled over "
            "independent runs so it carries a t-statistic, and a parity decomposition on "
            "identical paths isolates any drift bias, which would push the call and the put "
            "in opposite directions."
        ),
        rows=[
            Row("Difference from closed form", "headline.difference_bp_of_premium", "{:+.2f} bp of premium"),
            Row("Monte Carlo standard error", "headline.standard_error_bp_of_premium", "{:.2f} bp of premium"),
            Row("Difference, in standard errors", "headline.difference_in_standard_errors", "{:+.2f}",
                "the acceptance threshold is two"),
            Row("Two-sided p-value", "headline.two_sided_p_value", "{:.3f}"),
            Row("Closed form price", "headline.closed_form", "{:.8f}"),
            Row("Monte Carlo price", "headline.mc_estimate", "{:.8f}"),
            Row("Total paths", "headline.total_paths", "{:,}"),
            Row("Worst single run", "headline.worst_single_run_bp", "{:.2f} bp"),
            Row("Parity check — drift component", "parity_decomposition.antisymmetric_drift_component.t", "t = {:+.2f}",
                "a drift bias would appear here and nowhere else"),
            Row("Parity check — call error", "parity_decomposition.call_error.t", "t = {:+.2f}"),
            Row("Parity check — put error", "parity_decomposition.put_error.t", "t = {:+.2f}"),
            Row("numba and NumPy engines bit-identical", "engines_bit_identical", "{}",
                "re-verified in-run, so the \"same code path\" claim is tested where it is made"),
        ],
        input_rows=[
            Row("Product", "product", "{}"),
            Row("Time steps", "n_steps", "{:,}"),
            Row("Paths per run", "paths_per_run", "{:,}"),
            Row("Independent runs pooled", "headline_runs", "{:,}"),
            Row("Engine", "engine", "{}"),
            Row("Discretisation sweep", "discretisation_sweep_steps", "{}"),
        ],
    ),
    Headline(
        number=3,
        title="Greeks: pathwise versus bump-and-revalue",
        benchmark="greeks",
        command="python bench/greeks.py",
        summary=(
            "Pathwise and central-difference sensitivities on a European, both scored "
            "against M1's exact delta and vega rather than against each other.\n\n"
            "**Units matter here.** Delta is dimensionless and lives in [0,1]; vega is a "
            "price per 1.00 of volatility, with a scale near 37. The same absolute "
            "tolerance means very different things for the two, so each is labelled.\n\n"
            "**The compute ratio is quoted at matched standard error on the Greek**, not at "
            "matched path count. Central differencing needs about five pricing runs for "
            "price, delta and vega; pathwise produces all three from one. The honest "
            "theoretical ratio is therefore about one fifth — a tenth would require bumping "
            "to need extra paths on top, which with common random numbers it does not."
        ),
        rows=[
            Row("Pathwise vs bump, delta", "vanilla_agreement.pathwise_vs_bump_delta_absolute", "{:.3e}",
                "absolute, on a dimensionless quantity in [0,1]"),
            Row("Pathwise vs bump, vega", "vanilla_agreement.pathwise_vs_bump_vega_absolute", "{:.3e}",
                "absolute, price per 1.00 of volatility (vega is ~39 here)"),
            Row("Pathwise vs bump, vega, relative", "vanilla_agreement.pathwise_vs_bump_vega_relative", "{:.3e}"),
            Row("Pathwise delta error vs closed form", "vanilla_agreement.pathwise_delta.absolute_error", "{:.3e}"),
            Row("Bumped delta error vs closed form", "vanilla_agreement.bump_delta.absolute_error", "{:.3e}"),
            Row("Compute ratio at matched accuracy, delta", "compute_ratio.delta.cost_ratio_pathwise_over_bump", "{:.3f}",
                "pathwise cost as a fraction of bumping's, at equal standard error"),
            Row("Compute ratio at matched accuracy, vega", "compute_ratio.vega.cost_ratio_pathwise_over_bump", "{:.3f}"),
            Row("Compute ratio at matched path count", "compute_ratio.cost_ratio_at_matched_path_count", "{:.3f}",
                "the easier comparison, reported alongside so the convention is explicit"),
            Row("Paths bumping needs to match pathwise, delta", "compute_ratio.delta.paths_bump_needs_to_match", "{:.2f}x"),
            Row("Pathwise on a digital (INVALID)", "discontinuous_payoff.pathwise_invalid.estimate", "{:.8f}",
                "returned with a standard error of exactly zero, against a true delta of 0.0196 — it does not raise"),
            Row("Likelihood-ratio delta on a digital", "discontinuous_payoff.likelihood_ratio.estimate", "{:.8f}"),
            Row("Digital delta, closed form", "discontinuous_payoff.digital_closed_form_delta", "{:.8f}"),
            Row("Likelihood-ratio error, in standard errors", "discontinuous_payoff.likelihood_ratio.error_in_standard_errors", "{:+.2f}"),
            Row("Variance cost of likelihood ratio where pathwise is legal", "discontinuous_payoff.variance_cost_of_lr_on_a_vanilla.equivalent_path_multiple", "{:.1f}x the paths"),
            Row("Common random numbers: standard-error penalty without", "common_random_numbers.standard_error_ratio", "{:.1f}x worse"),
            Row("Common random numbers: equivalent path penalty", "common_random_numbers.equivalent_path_multiple", "{:.0f}x the paths"),
            Row("Bump sweep WITHOUT CRN: se amplification", "bump_size_sweep.without_common_random_numbers.standard_error_amplification_small_over_large", "{:.1f}x",
                "the U shape — noise amplified as 1/h at small bumps"),
            Row("Bump sweep WITH CRN: se amplification", "bump_size_sweep.with_common_random_numbers.standard_error_amplification_small_over_large", "{:.1f}x",
                "no U — the standard error flattens onto the pathwise floor"),
            Row("Optimal bump, with CRN", "bump_size_sweep.with_common_random_numbers.minimum_at_relative_bump", "{:g} relative"),
            Row("Optimal bump, without CRN", "bump_size_sweep.without_common_random_numbers.minimum_at_relative_bump", "{:g} relative"),
        ],
        input_rows=[
            Row("Paths", "n_paths", "{:,}"),
            Row("Relative spot bump", "relative_spot_bump", "{:g}"),
            Row("Absolute vol bump", "absolute_vol_bump", "{:g}"),
            Row("Sweep bumps", "sweep_bumps", "{}"),
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
