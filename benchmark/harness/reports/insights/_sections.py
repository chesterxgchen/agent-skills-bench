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

"""E2 split module (mechanical, behavior-preserving)."""

from __future__ import annotations

from typing import Any

from .._context import JobExecutionSignal, ReportContext, StructureView
from .._events import (
    _format_command_span,
    _job_rerun_reason,
    _span_total_seconds,
    agent_failure_category,
    bash_permission_denial_count,
    exit_code,
    failure_evidence,
    fmt_seconds_with_unit,
    artifact_validation_metric_evidence,
)
from .._text import markdown_cell
from ..evidence import RunEvidence
from ._code_quality import fl_algorithm_display
from ._diagnostics import (
    bash_blocked_diagnostic,
    command_failure_diagnostics_table,
    failure_root_cause,
    successful_job_evidence,
)
from ._metrics import (
    _metric_value_label,
    additional_or_observed_metric_values_display,
    benchmark_outcome,
    comparable_metric_name,
    dependency_reference_notes,
    final_response_metric_reporting_gap,
    human_readable_status,
    metric_display,
    metric_mismatch_with_reported_scalar,
    metric_names_for_runs,
    metric_reporting_gap_evidence,
    observed_metric_evidence_display,
    quality_signal,
    run_analysis,
    run_quality_issues,
    run_result_metric_status,
    run_status_kind,
)
from ._plugin_view import (
    MODE_LABELS,
    _collect_plugin_evidence,
    _execution_atom,
    _execution_run_noun,
    _participant_model,
    _report_context,
    _result_term,
    _section_copy,
    count_map,
    event_type_count,
    hint_count,
)
from ._spans import _dependency_install_retry_reason, _dependency_install_spans
from ._structure import (
    artifact_summary,
    structure_optional_display,
    structure_required_display,
    workspace_change_display,
)

__all__ = [
    "fl_algorithm_summary",
    "job_run_action",
    "quality_signal_table",
    "missing_result_metrics_section",
    "activity_insights_table",
    "event_mix_table",
    "outcome_details_table",
    "repeated_dependency_install_section",
    "repeated_job_runs_section",
    "status_table",
    "job_run_status_section",
    "job_execution_summary",
    "failure_analysis_section",
]


def fl_algorithm_summary(runs: dict[str, RunEvidence], modes: list[str], ctx: ReportContext | None = None) -> str:
    ctx = ctx or _report_context(runs, modes)
    return "; ".join(
        f"{runs[mode].label or mode}: {fl_algorithm_display(runs[mode], ctx.algorithm(mode))}" for mode in modes
    )


def job_run_action(run: RunEvidence, job_exec: Any = None, ctx: Any = None) -> str:
    if job_exec is None:
        job_exec = _collect_plugin_evidence(run).job_execution or JobExecutionSignal()
    atom = _execution_atom(ctx)
    status = job_exec.status
    if status == "not_started":
        failure_category = agent_failure_category(run.raw)
        if exit_code(run.raw) not in (None, 0) and failure_category and failure_category != "agent_unknown_failure":
            if failure_category == "agent_auth_failure":
                return (
                    "Authenticate the selected agent in the mounted benchmark home, then rerun; the job never started."
                )
            return "Fix the agent startup failure, then rerun; the job never started."
        if bash_permission_denial_count(run.raw):
            return f"Fix agent Bash/tool permissions and rerun; no {_result_term(ctx)}metrics can be trusted until the job executes."
        return f"Require the agent to run the generated job{' or ' + atom if atom else ''} before reporting benchmark metrics."
    if status == "started_failed":
        reason = job_exec.status_reason
        if "missing Python dependency" in reason:
            if "no dependency install command was captured" in reason:
                return f"Install the job requirements in the same Python environment before running the {atom or 'job'}, then rerun the benchmark."
            return f"Inspect the dependency install command output and ensure the {atom or 'job'} uses the environment where requirements were installed."
        return "Inspect the failed job command output, fix the generated job, and rerun the benchmark."
    if status == "background_task_killed":
        return f"Require the agent to wait for the background {atom or 'job'} to exit and verify terminal metrics before finalizing."
    if status == "agent_left_simulation_running":
        return f"Rerun with foreground {atom or 'job'} validation, or make the agent wait for the background task before finalizing."
    if status == "completed":
        if job_exec.recovered_summary:
            return "Use the final successful job logs for metrics, but inspect recovered command failures before drawing conclusions."
        return "Use job logs and reported metrics for quality comparison."
    return "Inspect run artifacts; job execution evidence is unavailable."


def quality_signal_table(runs: dict[str, RunEvidence], modes: list[str], ctx: ReportContext | None = None) -> str:
    ctx = ctx or _report_context(runs, modes)
    lines = [
        "| Run | Expected metric | Reported result | Status | Evidence |",
        "|---|---|---|---|---|",
    ]
    comparable_name = comparable_metric_name(runs)
    for mode in modes:
        run = runs[mode]
        ev = ctx.evidence.get(mode)
        signal = quality_signal(run.record if isinstance(run.record, dict) else {})
        metric = run.validation_metric if isinstance(run.validation_metric, dict) else {}
        expected = signal.get("expected_primary_metric") or metric.get("name")
        result = metric_display(run, comparable_name, ev)
        label = _metric_value_label(run, comparable_name, ev)
        if label:
            result = f"{result} ({label})"
        evidence = artifact_validation_metric_evidence(run.raw) or signal.get("evidence") or "NA"
        status = signal.get("status") or "NA"
        response_gap = final_response_metric_reporting_gap(run, ev)
        if response_gap:
            status = "artifact metric present; final response gap"
            evidence = response_gap
        lines.append(
            f"| {markdown_cell(run.label)} | {markdown_cell(expected)} | {markdown_cell(result)} | "
            f"{markdown_cell(status)} | {markdown_cell(evidence)} |"
        )
    return "\n".join(lines)


def missing_result_metrics_section(
    runs: dict[str, RunEvidence], modes: list[str], ctx: ReportContext | None = None
) -> str:
    ctx = ctx or _report_context(runs, modes)
    issue_modes = [mode for mode in modes if run_quality_issues(runs[mode], ctx.evidence.get(mode))]
    if not issue_modes:
        return ""
    lines = [
        "## Missing, Partial, Or Mismatched Result Metrics",
        "",
        f"A run can complete at the agent/container level and still need review when it omits the requested "
        f"{_result_term(ctx)}validation metric, reports only partial values, or reports a different metric than the "
        "job guidance requested.",
        "",
        "| Run | Result metric status | Final response metric evidence | Why results are missing, partial, or mismatched | Report action |",
        "|---|---|---|---|---|",
    ]
    for mode in issue_modes:
        run = runs[mode]
        issues = run_quality_issues(run, ctx.evidence.get(mode))
        observed_metrics = observed_metric_evidence_display(run, ctx)
        atom = _execution_atom(ctx)
        action = (
            f"Require the final message or benchmark record to include one aggregate {_result_term(ctx)}"
            "validation metric."
        )
        if metric_mismatch_with_reported_scalar(run, ctx.evidence.get(mode)):
            action = (
                "Treat the run as completed with a reported scalar metric, but flag that it did not follow the target "
                "metric instruction."
            )
        if not metric_names_for_runs({mode: run}) and observed_metrics == "none":
            action = "Inspect the final message and generated job logs; no parseable validation metric was reported."
            if successful_job_evidence(run, ctx.evidence.get(mode), ctx):
                action = (
                    f"Job{'/' + atom if atom else ''} logs contain metric evidence, but the final response or "
                    f"benchmark record did not report one aggregate {_result_term(ctx)}validation metric."
                )
        lines.append(
            f"| {markdown_cell(run.label or mode)} | {markdown_cell(run_result_metric_status(run, ctx.evidence.get(mode)))} | "
            f"{markdown_cell(observed_metrics)} | {markdown_cell('; '.join(issues))} | "
            f"{markdown_cell(action)} |"
        )
    return "\n".join(lines)


def activity_insights_table(runs: dict[str, RunEvidence], modes: list[str], ctx: ReportContext | None = None) -> str:
    ctx = ctx or _report_context(runs, modes)
    rows = [
        (
            "File reads (`cat`/`sed`/Read tool)",
            "shell_cat_or_sed",
            "Direct file-read behavior; includes shell cat/sed and Read tool calls.",
        ),
        ("`find` commands", "shell_find", "Filesystem discovery proxy."),
        ("`rg`/`grep` search commands", "shell_search", "Search use proxy; covers rg and grep."),
        (
            f"{_execution_run_noun(ctx).capitalize()} references",
            "simulation",
            "Shows validation effort against generated jobs.",
        ),
        ("Python compile checks", "py_compile", "Shows syntax validation effort."),
        (
            "Skill calls / skill references",
            "skill_references",
            "Only skills-enabled runs should usually show these; includes Skill tool calls.",
        ),
        (
            "Agent / inspect calls",
            "agent_inspect",
            "Shows use of agent inspection commands; includes Agent tool calls.",
        ),
        ("Python job.py references", "python_job_py", "Shows repeated exercise of generated job entry points."),
    ]
    lines = [
        "| Activity signal | " + " | ".join(MODE_LABELS.get(mode, mode) for mode in modes) + " | Interpretation |",
        "|---|" + "|".join("---:" for _ in modes) + "|---|",
    ]
    for label, key, note in rows:
        lines.append(
            f"| {markdown_cell(label)} | "
            + " | ".join(str(hint_count(runs[mode], key)) for mode in modes)
            + f" | {markdown_cell(note)} |"
        )
    return "\n".join(lines)


def event_mix_table(runs: dict[str, RunEvidence], modes: list[str]) -> str:
    keys = sorted({key for mode in modes for key in count_map(runs[mode], "event_types")})
    if not keys:
        keys = ["command_execution", "agent_message", "file_change", "todo_list"]
    lines = [
        "| Event type | " + " | ".join(MODE_LABELS.get(mode, mode) for mode in modes) + " |",
        "|---|" + "|".join("---:" for _ in modes) + "|",
    ]
    for key in keys:
        lines.append(
            f"| `{markdown_cell(key)}` | " + " | ".join(str(event_type_count(runs[mode], key)) for mode in modes) + " |"
        )
    return "\n".join(lines)


def outcome_details_table(runs: dict[str, RunEvidence], modes: list[str], ctx: ReportContext | None = None) -> str:
    ctx = ctx or _report_context(runs, modes)
    comparable_name = comparable_metric_name(runs)
    # Each getter takes (run, ev) where ev is the per-mode PluginEvidence.
    rows = [
        ("Agent/container outcome", lambda run, ev: human_readable_status(run, ev)),
        (f"{_result_term(ctx)}result quality gate", lambda run, ev: benchmark_outcome(run, ev)),
        ("Reported validation metric", lambda run, ev: metric_display(run, comparable_name, ev)),
        (
            "Additional/other validation metric values",
            lambda run, ev: additional_or_observed_metric_values_display(run, comparable_name, ev, ctx),
        ),
        ("Copied workspace changes", lambda run, ev: workspace_change_display(run)),
        ("Captured generated artifacts", lambda run, ev: artifact_summary(run)),
        (
            "Required structure files",
            lambda run, ev: structure_required_display(run, getattr(ev, "structure_view", None) or StructureView()),
        ),
        (
            "Optional structure files",
            lambda run, ev: structure_optional_display(run, getattr(ev, "structure_view", None) or StructureView()),
        ),
    ]
    lines = [
        "| Signal | " + " | ".join(MODE_LABELS.get(mode, mode) for mode in modes) + " |",
        "|---|" + "|".join("---" for _ in modes) + "|",
    ]
    for label, getter in rows:
        lines.append(
            f"| {markdown_cell(label)} | "
            + " | ".join(markdown_cell(getter(runs[mode], ctx.evidence.get(mode))) for mode in modes)
            + " |"
        )
    return "\n".join(lines)


def repeated_dependency_install_section(runs: dict[str, RunEvidence], modes: list[str]) -> str:
    rows = []
    for mode in modes:
        run = runs[mode]
        spans = _dependency_install_spans(run)
        if len(spans) <= 1:
            continue
        attempts = "; ".join(f"{index + 1}. {_format_command_span(span)}" for index, span in enumerate(spans[:4]))
        if len(spans) > 4:
            attempts += f"; +{len(spans) - 4} more"
        rows.append(
            (
                run.label or mode,
                str(len(spans)),
                fmt_seconds_with_unit(_span_total_seconds(spans)),
                attempts,
                _dependency_install_retry_reason(spans),
            )
        )
    if not rows:
        return ""
    lines = [
        "### Repeated Dependency Install Attempts",
        "",
        "These are dependency-install commands captured during the agent run. Repeated attempts usually mean an install failed, was retried with different options, or the agent changed environments.",
        "",
        "| Run | Install attempts | Total captured install time | Attempts | Captured reason/evidence |",
        "|---|---:|---:|---|---|",
    ]
    for label, count, total_time, attempts, reason in rows:
        lines.append(
            f"| {markdown_cell(label)} | {markdown_cell(count)} | {markdown_cell(total_time)} | "
            f"{markdown_cell(attempts)} | {markdown_cell(reason)} |"
        )
    return "\n".join(lines)


def repeated_job_runs_section(runs: dict[str, RunEvidence], modes: list[str], ctx: ReportContext | None = None) -> str:
    ctx = ctx or _report_context(runs, modes)
    rows = []
    for mode in modes:
        run = runs[mode]
        spans = list(ctx.job_execution(mode).successful_job_spans)
        if len(spans) <= 1:
            continue
        execution_list = "; ".join(f"{index + 1}. {_format_command_span(span)}" for index, span in enumerate(spans[:4]))
        if len(spans) > 4:
            execution_list += f"; +{len(spans) - 4} more"
        rows.append(
            (
                run.label or mode,
                str(len(spans)),
                fmt_seconds_with_unit(_span_total_seconds(spans)),
                execution_list,
                _job_rerun_reason(spans, run.raw),
            )
        )
    if not rows:
        return ""
    atom = _execution_atom(ctx)
    run_noun = getattr(_participant_model(ctx), "execution_noun", None)
    header_suffix = f"/{run_noun.capitalize()}" if run_noun else ""
    lines = [
        f"### Repeated Job{header_suffix} Executions",
        "",
        f"These are full successful job{' or ' + atom if atom else ''} executions, excluding export, help, and preflight commands. Repeated runs materially affect elapsed time and usually mean the agent reran after validation, recovery, or configuration changes.",
        "",
        "| Run | Successful executions | Total captured job time | Executions | Captured reason/evidence |",
        "|---|---:|---:|---|---|",
    ]
    for label, count, total_time, executions, reason in rows:
        lines.append(
            f"| {markdown_cell(label)} | {markdown_cell(count)} | {markdown_cell(total_time)} | "
            f"{markdown_cell(executions)} | {markdown_cell(reason)} |"
        )
    return "\n".join(lines)


def status_table(runs: dict[str, RunEvidence], modes: list[str], ctx: ReportContext | None = None) -> str:
    ctx = ctx or _report_context(runs, modes)
    lines = ["| Run | Status | Analysis |", "|---|---|---|"]
    for mode in modes:
        run = runs[mode]
        ev = ctx.evidence.get(mode)
        lines.append(
            f"| {markdown_cell(run.label)} | {markdown_cell(human_readable_status(run, ev))} | "
            f"{markdown_cell(run_analysis(run, ev))} |"
        )
    return "\n".join(lines)


def job_run_status_section(runs: dict[str, RunEvidence], modes: list[str], ctx: ReportContext | None = None) -> str:
    ctx = ctx or _report_context(runs, modes)
    lines = [
        "## Job Run Status",
        "",
        _section_copy(
            ctx,
            "job_run.intro",
            "This section tracks whether the generated job actually ran. Agent/container exit code 0 only means the "
            "agent process finished; it does not prove the generated job executed.",
        ),
        "",
        "| Run | Job run status | Evidence | Action |",
        "|---|---|---|---|",
    ]
    for mode in modes:
        run = runs[mode]
        job_exec = ctx.job_execution(mode)
        lines.append(
            f"| {markdown_cell(run.label or mode)} | {markdown_cell(job_exec.status)} | "
            f"{markdown_cell(job_exec.status_reason)} | {markdown_cell(job_run_action(run, job_exec, ctx))} |"
        )
    return "\n".join(lines)


def job_execution_summary(runs: dict[str, RunEvidence], modes: list[str], ctx: ReportContext | None = None) -> str:
    ctx = ctx or _report_context(runs, modes)
    return "; ".join(
        f"{runs[mode].label or mode}: " f"{ctx.job_execution(mode).status} ({ctx.job_execution(mode).status_reason})"
        for mode in modes
    )


def failure_analysis_section(runs: dict[str, RunEvidence], modes: list[str], ctx: ReportContext | None = None) -> str:
    ctx = ctx or _report_context(runs, modes)
    lines = []
    for mode in modes:
        run = runs[mode]
        ev = ctx.evidence.get(mode)
        label = run.label or mode
        status_kind = run_status_kind(run, ev)
        job_exec = ctx.job_execution(mode)
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"- Job run status: {job_exec.status} — {job_exec.status_reason}")
        if status_kind == "passed":
            metric = metric_display(run, comparable_metric_name(runs), ev)
            label_text = _metric_value_label(run, comparable_metric_name(runs), ev)
            if label_text:
                metric = f"{metric} ({label_text})"
            lines.append(f"- Outcome: passed. {metric}.")
            response_gap = final_response_metric_reporting_gap(run, ev)
            if response_gap:
                lines.append(f"- Reporting note: {response_gap}")
        elif status_kind == "needs review":
            lines.append(
                "- Outcome: needs review. The agent process completed, but benchmark quality checks found issues."
            )
            for issue in run_quality_issues(run, ctx.evidence.get(mode)):
                lines.append(f"- Issue: {issue}")
        elif status_kind == "failed":
            lines.append(f"- Outcome: failed. {failure_root_cause(run)}")
            evidence = failure_evidence(run.raw)
            if evidence:
                lines.append(f"- Evidence: {evidence}")
        else:
            lines.append("- Outcome: missing. No run artifacts were found for this mode.")
        record = run.record if isinstance(run.record, dict) else {}
        signal = quality_signal(record)
        metric_evidence = artifact_validation_metric_evidence(run.raw) or signal.get("evidence")
        if metric_evidence:
            lines.append(f"- Metric evidence: {metric_evidence}")
        if status_kind == "passed":
            bash_blocked = bash_blocked_diagnostic(run, recovered=True, ctx=ctx)
            if bash_blocked:
                lines.append(f"- Recovered Bash/tool issue: {bash_blocked}")
            recovered_commands = command_failure_diagnostics_table(run, recovered_only=True, ev=ctx.evidence.get(mode))
            if recovered_commands:
                lines.append("")
                lines.append("**Recovered Command Evidence**")
                lines.append("")
                lines.append(recovered_commands)
        else:
            bash_blocked = bash_blocked_diagnostic(run, ctx=ctx)
            if bash_blocked:
                lines.append(f"- Bash blocking: {bash_blocked}")
            command_evidence = command_failure_diagnostics_table(run, ev=ctx.evidence.get(mode))
            if command_evidence:
                lines.append("")
                lines.append("**Command Evidence**")
                lines.append("")
                lines.append(command_evidence)
            success_evidence = successful_job_evidence(run, ctx.evidence.get(mode), ctx)
            if success_evidence:
                lines.append(f"- Recovery evidence: {success_evidence}.")
            metric_gap = metric_reporting_gap_evidence(run, ctx.evidence.get(mode), ctx)
            if metric_gap:
                lines.append(f"- {metric_gap}")
        for note in dependency_reference_notes(run):
            lines.append(f"- Dependency reference: {note}")
        lines.append("")
    return "\n".join(lines).rstrip()
