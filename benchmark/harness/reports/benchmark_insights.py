# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Record-driven benchmark insight report."""

from __future__ import annotations

import argparse
import html  # noqa: F401
import json  # noqa: F401
import logging
import re  # noqa: F401
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from ..modes import BENCHMARK_RUNS, mode_names

# Leaf names re-exported for the historical public surface (production + the test suite
# import several of these by name from this module). Kept as explicit re-exports so the
# facade's exported surface stays byte-for-byte the pre-split set.
from ..quality_signals import (  # noqa: F401
    canonical_metric_name,
    is_numeric_metric_value,
    is_plausible_metric_value,
    metric_value_entries,
    reported_metric_payload,
)
from ._context import (  # noqa: F401
    AlgorithmSignal,
    CodeQualitySignal,
    CommandFailureSignal,
    JobExecutionSignal,
    ReportContext,
    StructureView,
)
from ._events import (  # noqa: F401
    _format_command_span,
    _job_rerun_reason,
    _longest_span,
    _span_total_seconds,
    agent_command_spans,
    agent_failure_category,
    as_number,
    bash_permission_denial_count,
    command_error_summary,
    command_failed,
    command_succeeded,
    commands_for_run,
    exit_code,
    failure_evidence,
    fmt_seconds,
    fmt_seconds_with_unit,
    inline_code_text,
    is_dependency_install_command,
    run_activity,
    unsupported_model_message,
)
from ._loader import sanitized_validation_metric  # noqa: F401
from ._loader import collect_benchmark_runs
from ._runs import combined_text, manifest_paths, run_record, run_workspace_delta, unique_paths  # noqa: F401
from ._text import (  # noqa: F401
    _command_count_display,
    _is_file_inspection_segment,
    _shell_command_segments,
    fmt_number,
    markdown_cell,
    strip_ansi,
)
from .evidence import RunEvidence

# Names the orchestration below calls are provided by the star imports above; import them
# explicitly too so they resolve unambiguously (no F405) without changing behavior.
from .insights._charts import *  # noqa: F401,F403
from .insights._charts import embedded_bar_chart, interpretation_section, outcome_metrics_table
from .insights._code_quality import *  # noqa: F401,F403
from .insights._code_quality import fl_algorithm_display, generated_code_quality_section
from .insights._diagnostics import *  # noqa: F401,F403
from .insights._metrics import *  # noqa: F401,F403
from .insights._metrics import benchmark_outcome, run_quality_issues, run_result_metric_status, status_summary

# E2 physical split: the report engine now lives in the ``insights`` package; this module
# is a thin re-export facade so every existing ``from .benchmark_insights import X``
# (production + the test suite, public and private names) keeps working unchanged. Each
# ``insights`` submodule declares ``__all__``; the star imports re-export that surface.
from .insights._plugin_view import *  # noqa: F401,F403
from .insights._plugin_view import _as_run_evidence, _report_context, _result_term, _section_copy
from .insights._sections import *  # noqa: F401,F403
from .insights._sections import (
    activity_insights_table,
    event_mix_table,
    failure_analysis_section,
    fl_algorithm_summary,
    job_execution_summary,
    job_run_status_section,
    missing_result_metrics_section,
    outcome_details_table,
    quality_signal_table,
    status_table,
)
from .insights._spans import *  # noqa: F401,F403
from .insights._structure import *  # noqa: F401,F403
from .insights._structure import (
    artifact_summary,
    output_changes_table,
    run_identity_summary,
    run_identity_table,
    source_input_protection_display,
    structure_correctness_table,
    structure_trees_section,
)
from .insights._why import *  # noqa: F401,F403
from .insights._why import cost_comparison_section, why_section


def _render_plugin_section(section: Any) -> list[str]:
    """Render a plugin ``ReportSection`` to its composed line-slice: the section markdown
    (``title`` + blank + ``body``) followed by the trailing blank every composed block owns."""

    rendered = f"{section.title}\n\n{section.body}" if section.body else section.title
    return [rendered, ""]


def _compose_report(generic_blocks: list[tuple[str, list[str]]], plugin_sections: list[Any]) -> list[str]:
    """Merge plugin-contributed sections into the named generic skeleton (E1b §6).

    Insert-only: the generic blocks are authoritative and never replaced. Each plugin
    section inserts before/after its named anchor block; an unknown anchor (and the
    reserved ``"end"`` anchor) appends at the end, never dropped. Ordering is deterministic
    by ``(anchor, placement, order, id)``. With no plugin sections this is a pure
    concatenation of the generic blocks -> byte-identical.
    """

    known = {block_id for block_id, _ in generic_blocks}
    before: dict[str, list[Any]] = {}
    after: dict[str, list[Any]] = {}
    trailing: list[Any] = []
    seen_ids: set[str] = set()
    for section in sorted(plugin_sections, key=lambda s: (s.anchor, s.placement, s.order, s.id)):
        # The contract is SDK-facing: a malformed section warns loudly rather than being
        # silently coerced. Duplicate ids defeat stable identity (diagnostics/dedup);
        # an unrecognized placement is a plugin bug (we fall back to "after").
        if section.id in seen_ids:
            logger.warning("duplicate plugin ReportSection id %r; section ids should be unique", section.id)
        seen_ids.add(section.id)
        if section.placement not in ("before", "after"):
            logger.warning(
                "plugin ReportSection %r has invalid placement %r; expected 'before'|'after', using 'after'",
                section.id,
                section.placement,
            )
        if section.anchor in known and section.anchor != "end":
            bucket = before if section.placement == "before" else after
            bucket.setdefault(section.anchor, []).append(section)
        else:
            if section.anchor != "end":
                logger.warning(
                    "plugin ReportSection %r targets unknown anchor %r; appending at end",
                    section.id,
                    section.anchor,
                )
            trailing.append(section)
    lines: list[str] = []
    for block_id, block_lines in generic_blocks:
        for section in before.get(block_id, []):
            lines.extend(_render_plugin_section(section))
        lines.extend(block_lines)
        for section in after.get(block_id, []):
            lines.extend(_render_plugin_section(section))
    for section in trailing:
        lines.extend(_render_plugin_section(section))
    return lines


def _executive_summary_section(
    runs: dict[str, RunEvidence],
    modes: list[str],
    ctx: ReportContext,
) -> str:
    algorithm_label = _section_copy(ctx, "exec_summary.algorithm_workflow_label", "Algorithm/workflow")
    status_lines = [
        "### Run Status",
        "",
        "| Run | Overall status | Job run status | Result quality gate | Result metric |",
        "|---|---|---|---|---|",
    ]
    context_lines = [
        "### Run Context",
        "",
        f"| Run | Agent/model | {markdown_cell(algorithm_label)} | Captured generated artifacts |",
        "|---|---|---|---|",
    ]
    for mode in modes:
        run = runs[mode]
        ev = ctx.evidence.get(mode)
        job_exec = ctx.job_execution(mode)
        agent_model = f"agent={run.agent or 'NA'}, model={run.agent_model or 'NA'}"
        job_status = job_exec.status
        if job_exec.status_reason:
            job_status = f"{job_status}: {job_exec.status_reason}"
        status_lines.append(
            f"| {markdown_cell(run.label or mode)} | {markdown_cell(human_readable_status(run, ev))} | "
            f"{markdown_cell(job_status)} | {markdown_cell(benchmark_outcome(run, ev))} | "
            f"{markdown_cell(run_result_metric_status(run, ev))} |"
        )
        context_lines.append(
            f"| {markdown_cell(run.label or mode)} | "
            f"{markdown_cell(agent_model)} | "
            f"{markdown_cell(fl_algorithm_display(run, ctx.algorithm(mode)))} | "
            f"{markdown_cell(artifact_summary(run))} |"
        )
    return "\n".join(["## Executive Summary", "", *status_lines, "", *context_lines])


def benchmark_report(root: Path, runs: dict[str, RunEvidence | dict[str, Any]]) -> str:
    # Resolve the SDK report plugin once by captured identity (§4.2): absent id ->
    # null plugin (Inversion 3). Imported lazily to avoid a module-load import cycle
    # (the registry pulls in the NVFLARE plugin, which imports this module).
    from .. import profile_metadata
    from ..sdks.report_registry import resolve_from_result_root
    from .evidence import SCHEMA_VERSION, ComparisonEvidence

    plugin = resolve_from_result_root(root)
    modes = mode_names(BENCHMARK_RUNS)
    # Contract B is the render spine (Inversion 1): build the typed ComparisonEvidence
    # once from the captured bundles and render from its RunEvidence. Render helpers
    # read the typed fields directly and pass ``run.raw`` only to the dict-based
    # neutral substrate (``_events``/``_runs``/``_logic``).
    cmp = ComparisonEvidence(
        schema_version=SCHEMA_VERSION,
        runs={mode: _as_run_evidence(runs[mode]) for mode in modes},
        modes=list(modes),
        sdk_metadata=profile_metadata.read_profile_metadata_block(root),
    )
    runs = cmp.runs
    # Build the per-mode plugin sidecar once and thread it to the render helpers
    # (ROUTING_PLAN §4d) instead of calling the SDK logic leaf by name.
    ctx = _report_context(runs, modes, plugin)
    missing_metrics = missing_result_metrics_section(runs, modes, ctx)
    cost_comparison = cost_comparison_section(runs, modes)
    # Named generic section blocks (E1b §6). Each block owns its exact slice INCLUDING its
    # trailing "" so the composer is pure concatenation (byte-identical). The one resolved
    # ``ctx`` is threaded into every builder here -- no builder re-resolves a context.
    generic_blocks: list[tuple[str, list[str]]] = [
        (
            "exec_summary",
            [
                "# Agent Benchmark Insights",
                "",
                f"Result root: `{root}`",
                "",
                _executive_summary_section(runs, modes, ctx),
                "",
            ],
        ),
        ("run_identity", ["## Run Identity", "", run_identity_table(runs, modes), ""]),
    ]
    generic_blocks.extend(
        [
            (
                "metrics",
                [
                    "## Metrics",
                    "",
                    embedded_bar_chart({mode: runs[mode] for mode in modes}, ctx),
                    "",
                    outcome_metrics_table(runs, modes, ctx),
                    "",
                ],
            ),
            ("quality_signals", ["## Quality Signals", "", quality_signal_table(runs, modes, ctx), ""]),
        ]
    )
    if missing_metrics:
        generic_blocks.append(("missing_metrics", [missing_metrics, ""]))
    generic_blocks.extend(
        [
            ("failure_analysis", ["## Failure Analysis", "", failure_analysis_section(runs, modes, ctx), ""]),
            ("output_changes", ["## Output Changes", "", output_changes_table(runs, modes), ""]),
            ("outcome_details", ["## Outcome Details", "", outcome_details_table(runs, modes, ctx), ""]),
            (
                "structure_correctness",
                [
                    "## Structure Correctness",
                    "",
                    "The structure checks look for the core converted source files and captured runtime/export artifacts. They are report signals, not a substitute for running the generated job.",
                    "",
                    structure_correctness_table(runs, modes, ctx),
                    "",
                ],
            ),
            ("structure_trees", [structure_trees_section(runs, modes), ""]),
            ("generated_code_quality", [generated_code_quality_section(runs, modes, ctx), ""]),
            ("activity_insights", ["## Activity Insights", "", activity_insights_table(runs, modes, ctx), ""]),
            ("event_mix", ["## Event Mix", "", event_mix_table(runs, modes), ""]),
        ]
    )
    if cost_comparison:
        generic_blocks.append(("cost_comparison", [cost_comparison, ""]))
    why = why_section(runs, modes, ctx)
    if why:
        generic_blocks.append(("why", [why, ""]))
    generic_blocks.append(("interpretation", [interpretation_section(runs, modes, ctx), ""]))
    generic_blocks.append(
        (
            "artifacts",
            ["## Artifacts", "", "- `metrics_report.md`", "- `metrics_report.html`", "- `records/`"],
        )
    )
    # Merge any plugin-contributed sections into the generic skeleton (E1b). The flat/null
    # SDK contributes none -> pure concatenation -> byte-identical.
    lines = _compose_report(generic_blocks, plugin.sections(cmp, ctx.evidence))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs = collect_benchmark_runs(args.root)
    output = args.output or args.root / "benchmark_insights.md"
    output.write_text(benchmark_report(args.root, runs), encoding="utf-8")
    print(output)
