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

import json
import re
from typing import Any

from .._context import JobExecutionSignal, ReportContext
from .._events import (
    _format_command_span,
    _span_total_seconds,
    as_number,
    event_timeline_from_text,
    fmt_seconds,
    fmt_seconds_with_unit,
    inline_code_text,
    is_dependency_install_command,
    predicted_failure_message,
    terminal_failure_anchor,
    truncate,
    run_activity,
)
from .._runs import run_workspace_delta
from .._text import _command_count_display, fmt_number, markdown_cell
from ..evidence import RunEvidence
from ._code_quality import _code_quality_assessment_map
from ._plugin_view import (
    _as_run_evidence,
    _execution_atom,
    _plugin_narrative,
    _report_context,
    _result_term,
    _run_evidence,
    count_map,
    fmt_short,
    run_summary,
)
from ._sections import repeated_dependency_install_section, repeated_job_runs_section
from ._metrics import run_quality_issues, run_result_metric_status
from ._spans import (
    _assistant_turns,
    _command_span_total_seconds,
    _dependency_install_slowdown_note,
    _dependency_install_spans,
    _dependency_install_total_seconds,
    _elapsed_excluding_dependency_install,
    _install_cpu_only_evidence_display,
    _install_stack_evidence,
    _install_strategy_summary,
    _install_total_seconds,
    _non_dependency_command_seconds,
    _run_usage,
    _thinking_token_events,
    _top_command_spans,
)

__all__ = [
    "cost_comparison_section",
    "_elapsed_time_accounting_note",
    "_longest_command_comparison_note",
    "repeated_job_runs_slowdown_section",
    "_code_quality_slowdown_notes",
    "_signed_seconds_delta",
    "_signed_number_delta",
    "_append_time_reason_row",
    "_append_count_reason_row",
    "_slowdown_reason_table",
    "_token_delta_display",
    "_cost_display",
    "_cost_delta_display",
    "_token_usage_comparison_table",
    "_why_slower",
    "_why_more_tokens",
    "why_section",
]


def cost_comparison_section(runs: dict[str, RunEvidence], modes: list[str]) -> str:
    if len(modes) != 2:
        return ""
    left, right = modes
    left_run = runs[left]
    right_run = runs[right]
    left_summary = run_summary(left_run)
    right_summary = run_summary(right_run)
    left_dependency_seconds = _dependency_install_total_seconds(left_run)
    right_dependency_seconds = _dependency_install_total_seconds(right_run)
    rows = [
        ("Total time seconds", left_summary.get("elapsed_seconds"), right_summary.get("elapsed_seconds"), "seconds"),
        (
            "Runtime seconds",
            _elapsed_excluding_dependency_install(left_run),
            _elapsed_excluding_dependency_install(right_run),
            "seconds",
        ),
        ("Dependency install seconds", left_dependency_seconds, right_dependency_seconds, "seconds"),
        (
            "Non-install command seconds",
            _non_dependency_command_seconds(left_run),
            _non_dependency_command_seconds(right_run),
            "seconds",
        ),
        (
            "Agent/model interaction seconds",
            _agent_interaction_seconds(left_run),
            _agent_interaction_seconds(right_run),
            "seconds",
        ),
        ("Total tokens", left_summary.get("token_count"), right_summary.get("token_count"), "short"),
        (
            "Commands",
            run_activity(left_run.raw).get("command_count"),
            run_activity(right_run.raw).get("command_count"),
            "number",
        ),
        (
            "Unique commands",
            run_activity(left_run.raw).get("unique_command_count"),
            run_activity(right_run.raw).get("unique_command_count"),
            "number",
        ),
        (
            "Changed/generated files",
            run_workspace_delta(left_run.raw).get("changed_file_count"),
            run_workspace_delta(right_run.raw).get("changed_file_count"),
            "number",
        ),
        (
            "Runtime artifacts",
            run_workspace_delta(left_run.raw).get("runtime_artifact_count"),
            run_workspace_delta(right_run.raw).get("runtime_artifact_count"),
            "number",
        ),
    ]
    lines = [
        "## Cost And Work Comparison",
        "",
        "Cost numbers are descriptive only. Quality gates decide whether a cost comparison is meaningful.",
        "",
        "`Runtime seconds` is total elapsed time minus captured dependency-install command/background-task time. "
        "`Dependency install seconds` is captured dependency-install command/background-task time. "
        "`Non-install command seconds` is summed duration of captured non-install shell/tool commands, so it can be lower than runtime when the agent spends time reasoning, waiting, or using non-command tools. "
        "`Agent/model interaction seconds` is the remaining runtime after subtracting captured non-install command spans; it is a residual signal for model round trips, tool orchestration, background command gaps, and other time not attributed to command spans.",
        "Command span timing is operation-level evidence, not a strict wall-clock partition; it can differ from total elapsed time when agent event timestamps overlap, are truncated, or come from a different clock than the harness timer.",
        "",
        f"| Signal | {markdown_cell(left_run.label or left)} | {markdown_cell(right_run.label or right)} | Delta right-left |",
        "|---|---:|---:|---:|",
    ]
    for label, left_value, right_value, value_kind in rows:
        left_num = as_number(left_value)
        right_num = as_number(right_value)
        delta = right_num - left_num if left_num is not None and right_num is not None else None
        formatter = fmt_short if value_kind == "short" else fmt_seconds if value_kind == "seconds" else fmt_number
        lines.append(
            f"| {markdown_cell(label)} | {formatter(left_value)} | {formatter(right_value)} | {formatter(delta)} |"
        )
    return "\n".join(lines)


def _elapsed_time_accounting_note(with_run: RunEvidence, base_run: RunEvidence) -> str:
    with_label = with_run.label or "With skills"
    base_label = base_run.label or "No skills baseline"
    rows = [
        (with_label, with_run),
        (base_label, base_run),
    ]
    lines = [
        "**Elapsed time accounting**",
        "",
        "| Run | Total | Dependency install | Runtime after install | Captured non-install commands | Agent/model interaction residual |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, run in rows:
        lines.append(
            f"| {markdown_cell(label)} | "
            f"{fmt_seconds_with_unit(run_summary(run).get('elapsed_seconds'))} | "
            f"{fmt_seconds_with_unit(_dependency_install_total_seconds(run))} | "
            f"{fmt_seconds_with_unit(_elapsed_excluding_dependency_install(run))} | "
            f"{fmt_seconds_with_unit(_non_dependency_command_seconds(run))} | "
            f"{fmt_seconds_with_unit(_agent_interaction_seconds(run))} |"
        )
    lines.extend(
        [
            "",
            "`Runtime after install` is total elapsed time minus captured dependency-install command/background-task time. "
            "Captured command spans identify slow operations but are not guaranteed to add up exactly to total elapsed time. "
            "The residual column is the best available indicator that wall time came from agent/model round trips, "
            "tool orchestration, background command gaps, or other non-command activity.",
        ]
    )
    return "\n".join(lines)


def _agent_interaction_seconds(run: RunEvidence) -> float | None:
    runtime = _elapsed_excluding_dependency_install(run)
    command_seconds = _non_dependency_command_seconds(run)
    if runtime is None or command_seconds is None:
        return None
    return max(0.0, runtime - command_seconds)


def _agent_interaction_slowdown_note(with_run: RunEvidence, base_run: RunEvidence) -> str:
    with_label = with_run.label or "With skills"
    base_label = base_run.label or "No skills baseline"
    with_residual = _agent_interaction_seconds(with_run)
    base_residual = _agent_interaction_seconds(base_run)
    with_commands = _non_dependency_command_seconds(with_run)
    base_commands = _non_dependency_command_seconds(base_run)
    if with_residual is None or base_residual is None:
        return ""
    residual_delta = with_residual - base_residual
    command_delta = None
    if with_commands is not None and base_commands is not None:
        command_delta = with_commands - base_commands
    if residual_delta <= 30:
        return ""
    command_phrase = ""
    if command_delta is not None and command_delta < 0:
        command_phrase = (
            f" Captured non-install command time was lower for {with_label} "
            f"({fmt_seconds_with_unit(with_commands)} vs {fmt_seconds_with_unit(base_commands)}), "
            "so the slowdown is not explained by longer measured shell/job commands."
        )
    return (
        "- **Root cause: more agent/model loop time, not measured command runtime**: "
        f"{with_label} spent {fmt_seconds_with_unit(with_residual)} outside captured dependency and "
        f"non-install command spans vs {fmt_seconds_with_unit(base_residual)} for {base_label} "
        f"(+{fmt_seconds_with_unit(residual_delta)})."
        f"{command_phrase} Read this with the turn/tool rows above: extra assistant turns, skill loading, "
        "tool lookups, file/source inspection, and validation retries compound wall time even when generated "
        "artifact count is smaller."
    )


def _event_text_items(run: RunEvidence) -> list[str]:
    items: list[str] = []
    for line in str(run.agent_events_text or "").splitlines():
        try:
            payload = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        message = payload.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    text = str(item.get("text") or "").strip()
                    if text:
                        items.append(text)
                elif item.get("type") == "tool_result":
                    text = str(item.get("content") or "").strip()
                    if text:
                        items.append(text)
        result = payload.get("tool_use_result")
        if isinstance(result, dict):
            for key in ("stdout", "stderr", "content"):
                text = str(result.get(key) or "").strip()
                if text:
                    items.append(text)
        # Codex stream shape: item.completed events carry agent_message/reasoning
        # text on payload.item.text.
        codex_item = payload.get("item")
        if isinstance(codex_item, dict) and codex_item.get("type") in ("agent_message", "reasoning"):
            text = str(codex_item.get("text") or "").strip()
            if text:
                items.append(text)
    return items


def _skill_contract_mismatch_note(with_run: RunEvidence) -> str:
    """Surface when the agent overrides skill guidance after local API/source evidence.

    This intentionally uses captured event text only. It does not know whether
    the SDK/API claim is true or stable; it reports that the agent spent work
    resolving a conflict between skill guidance and local source/docstring
    evidence.
    """

    if count_map(with_run, "tool_counts").get("Skill", 0) <= 0:
        return ""
    texts = _event_text_items(with_run)
    combined = "\n".join(texts)
    lower = combined.lower()
    override_patterns = (
        "cannot use the lightning `flare.patch",
        "cannot use `flare.patch",
        "cannot use flare.patch",
        "not a plain `flare.patch",
        "cannot express",
        "not the lightning ``flare.patch",
        "rather than a lightning ``flare.patch",
        "switch implementation strategy",
        "switch to a different implementation strategy",
    )
    source_evidence = any(marker in lower for marker in ("docstring", "source/docstring", "client script requirement"))
    if not source_evidence and not re.search(r"/[^\s\"']*nvflare/[^\s\"']*\.py", combined):
        return ""
    if not any(pattern in lower for pattern in override_patterns):
        return ""

    local_source = ""
    source_matches = re.findall(
        r"(/[^\s\"']*nvflare/[^\s\"']*\.py|/[^\s\"']*site-packages/nvflare/[^\s\"']*\.py)",
        combined,
    )
    if source_matches:
        local_source = next(
            (path for path in source_matches if "/recipes/" in path),
            "",
        )
    elif source_evidence:
        local_source = "local SDK source/docstring"
    if not local_source and source_evidence:
        local_source = "local SDK source/docstring"

    decisive_text = _strategy_change_rationale(texts, override_patterns)

    source_phrase = f" after reading {inline_source(local_source)}" if local_source else ""
    decisive_phrase = f" Captured agent rationale: {markdown_cell(decisive_text)}." if decisive_text else ""
    return (
        "- **Skill guidance overridden after local API/source inspection**: the skills-enabled agent loaded a skill, "
        f"then treated local source/docstring evidence{source_phrase} as conflicting with the skill guidance and "
        "changed implementation strategy. This does not assert that the local API/source claim is permanently true; "
        "it surfaces the root cause of this run's extra work: the agent spent turns reading source, reconciling the "
        f"conflict, and recovering from the abandoned path.{decisive_phrase}"
    )


def inline_source(value: str) -> str:
    return f"`{value}`" if value else "local SDK source"


def _strategy_change_rationale(texts: list[str], override_patterns: tuple[str, ...]) -> str:
    matching_texts: list[str] = []
    for text in texts:
        if text.startswith("Base directory for this skill:"):
            continue
        text_lower = text.lower()
        if any(pattern in text_lower for pattern in override_patterns):
            matching_texts.append(text)
    if not matching_texts:
        return ""

    preferred_markers = (
        "switch implementation strategy",
        "switch to a different implementation strategy",
        "so i need to switch",
        "i need to switch",
        "i'll switch",
        "manual",
        "therefore",
    )
    decisive_text = next(
        (text for text in matching_texts if any(marker in text.lower() for marker in preferred_markers)),
        matching_texts[-1],
    )
    return truncate(decisive_text, 260)


def _longest_command_comparison_note(with_run: RunEvidence, base_run: RunEvidence) -> str:
    with_label = with_run.label or "With skills"
    base_label = base_run.label or "No skills baseline"
    with_spans = _top_command_spans(with_run)
    base_spans = _top_command_spans(base_run)
    if not with_spans and not base_spans:
        return ""
    limit = max(len(with_spans), len(base_spans))
    lines = [
        "**Longest command comparison**",
        "",
        f"| Rank | {markdown_cell(with_label)} | {markdown_cell(base_label)} |",
        "|---:|---|---|",
    ]
    for index in range(limit):
        missing = "no timed command span >=30s captured"
        with_display = _format_command_span(with_spans[index]) if index < len(with_spans) else missing
        base_display = _format_command_span(base_spans[index]) if index < len(base_spans) else missing
        lines.append(f"| {index + 1} | {markdown_cell(with_display)} | {markdown_cell(base_display)} |")
    return "\n".join(lines)


def repeated_job_runs_slowdown_section(
    with_run: RunEvidence, base_run: RunEvidence, ctx: ReportContext | None = None
) -> str:
    with_ev = _run_evidence(ctx, with_run)
    with_spans = list((with_ev.job_execution or JobExecutionSignal()).successful_job_spans)
    if len(with_spans) <= 1:
        return ""
    base_spans = list((_run_evidence(ctx, base_run).job_execution or JobExecutionSignal()).successful_job_spans)
    # Re-key the single with-run under "with", reusing the evidence already
    # collected above (Inversion 3: a dropped ctx would re-resolve to the null
    # plugin and silently render an empty table).
    section = repeated_job_runs_section(
        {"with": with_run}, ["with"], ReportContext(evidence={"with": with_ev}, plugin=ctx.plugin if ctx else None)
    )
    if not section:
        return ""
    base_count = len(base_spans)
    base_time = fmt_seconds_with_unit(_span_total_seconds(base_spans)) if base_spans else "NA"
    atom = _execution_atom(ctx)
    note = (
        f"Baseline comparison: {base_run.label or 'No skills baseline'} had "
        f"{_command_count_display(base_count)} classified successful job{'/' + atom if atom else ''} execution"
        f"{'' if base_count == 1 else 's'}"
    )
    if base_spans:
        note += f" totaling {base_time}."
    else:
        note += "."
    return f"{section}\n\n{note}"


def _code_quality_slowdown_notes(
    with_run: RunEvidence, base_run: RunEvidence, ctx: ReportContext | None = None
) -> list[str]:
    with_quality = _code_quality_assessment_map(with_run, ctx.evidence.get(with_run.mode) if ctx else None)
    base_quality = _code_quality_assessment_map(base_run, ctx.evidence.get(base_run.mode) if ctx else None)
    lines = []
    with_runtime = _elapsed_excluding_dependency_install(with_run)
    base_runtime = _elapsed_excluding_dependency_install(base_run)
    runtime_delta = with_runtime - base_runtime if with_runtime is not None and base_runtime is not None else None
    runtime_slower = runtime_delta is not None and runtime_delta > 60

    with_loss_status, with_loss = with_quality.get("Loss/optimizer lifecycle", ("unknown", ""))
    base_loss_status, base_loss = base_quality.get("Loss/optimizer lifecycle", ("unknown", ""))
    if with_loss_status == "poor" and base_loss_status != "poor":
        if runtime_slower:
            lines.append(
                "- **Generated-code efficiency issue aligns with slower non-install runtime**: "
                f"the code-quality signal flags With skills as `{with_loss_status}` for loss/optimizer lifecycle "
                f"({with_loss}), while the baseline is `{base_loss_status}` ({base_loss}). "
                f"Runtime excluding dependency install is {fmt_seconds(with_runtime)}s vs {fmt_seconds(base_runtime)}s, "
                "so repeated setup inside the per-round training boundary is plausible runtime overhead. "
                "This does not prove sole causality, but it is a generated-code issue worth investigating."
            )
        else:
            lines.append(
                "- **Generated-code efficiency issue is not the measured slowdown driver**: "
                f"the code-quality signal flags With skills as `{with_loss_status}` for loss/optimizer lifecycle "
                f"({with_loss}), while the baseline is `{base_loss_status}` ({base_loss}). "
                f"However, runtime excluding dependency install is {fmt_seconds(with_runtime)}s vs "
                f"{fmt_seconds(base_runtime)}s, so this should be read as a code-quality concern, not the cause "
                "of the wall-time slowdown in this run."
            )

    with_metric_status, with_metric = with_quality.get("Per-round metric workload", ("unknown", ""))
    base_metric_status, base_metric = base_quality.get("Per-round metric workload", ("unknown", ""))
    if with_metric_status == "good" and base_metric_status in {"poor", "unknown"}:
        if runtime_slower:
            lines.append(
                "- **Quality-versus-speed tradeoff: useful validation work also adds per-round workload**: "
                f"With skills records `{with_metric}`, while the baseline records `{base_metric}`. "
                "Test/validation evaluation and per-round metric artifacts are desirable quality evidence, "
                f"but they are additional work on every {_result_term(ctx)}round and may explain part of the long per-round wait. "
                "Read this alongside the efficiency issue above: validation work is useful, while rebuilding "
                "setup objects inside the per-round boundary is avoidable overhead."
            )
        else:
            lines.append(
                "- **Quality evidence did not make non-install runtime slower in this run**: "
                f"With skills records `{with_metric}`, while the baseline records `{base_metric}`. "
                "That is useful validation evidence, but the captured runtime excluding dependency install is "
                f"{fmt_seconds(with_runtime)}s vs {fmt_seconds(base_runtime)}s, so it should not be cited as the "
                "wall-time slowdown cause for this run."
            )

    with_dependency_status, with_dependency = with_quality.get("Dependency install strategy", ("unknown", ""))
    if "accelerator-capable dependency stack" in with_dependency:
        lines.append(
            "- **Dependency cost is separate from code efficiency**: "
            f"the code-quality table records `{with_dependency_status}: {with_dependency}`. "
            "That explains install-time cost. Generated-code lifecycle signals remain quality evidence, but they "
            "should only be treated as runtime slowdown evidence when non-install runtime is also slower."
        )
    return lines


def _signed_seconds_delta(with_value: Any, base_value: Any) -> str:
    with_number = as_number(with_value)
    base_number = as_number(base_value)
    if with_number is None or base_number is None:
        return "NA"
    delta = with_number - base_number
    if delta == 0:
        return "0s"
    sign = "+" if delta > 0 else "-"
    return f"{sign}{fmt_seconds_with_unit(abs(delta))}"


def _signed_number_delta(with_value: Any, base_value: Any) -> str:
    with_number = as_number(with_value)
    base_number = as_number(base_value)
    if with_number is None or base_number is None:
        return "NA"
    delta = with_number - base_number
    if delta == 0:
        return "0"
    sign = "+" if delta > 0 else "-"
    return f"{sign}{fmt_number(abs(delta))}"


def _append_time_reason_row(
    rows: list[tuple[str, Any, Any, str, str]],
    label: str,
    with_value: Any,
    base_value: Any,
    interpretation: str,
) -> None:
    with_number = as_number(with_value)
    base_number = as_number(base_value)
    if with_number is None or base_number is None:
        return
    delta = with_number - base_number
    if delta <= 0:
        return
    rows.append(
        (
            label,
            fmt_seconds_with_unit(with_value),
            fmt_seconds_with_unit(base_value),
            _signed_seconds_delta(with_value, base_value),
            interpretation,
        )
    )


def _append_count_reason_row(
    rows: list[tuple[str, Any, Any, str, str]],
    label: str,
    with_value: Any,
    base_value: Any,
    interpretation: str,
) -> None:
    with_number = as_number(with_value)
    base_number = as_number(base_value)
    if with_number is None or base_number is None:
        return
    delta = with_number - base_number
    if delta <= 0:
        return
    rows.append(
        (
            label,
            fmt_number(with_value),
            fmt_number(base_value),
            _signed_number_delta(with_value, base_value),
            interpretation,
        )
    )


def _slowdown_reason_table(
    with_run: RunEvidence,
    base_run: RunEvidence,
    *,
    driver_with_command_seconds: float | None,
    driver_base_command_seconds: float | None,
    command_span_label: str,
    elapsed_is_slower: bool,
) -> str:
    with_label = with_run.label or "With skills"
    base_label = base_run.label or "No skills baseline"
    rows: list[tuple[str, Any, Any, str, str]] = []
    _append_time_reason_row(
        rows,
        "Total elapsed",
        run_summary(with_run).get("elapsed_seconds"),
        run_summary(base_run).get("elapsed_seconds"),
        "overall wall-clock comparison",
    )
    _append_time_reason_row(
        rows,
        "Dependency install",
        _dependency_install_total_seconds(with_run),
        _dependency_install_total_seconds(base_run),
        "dependency setup/download time",
    )
    _append_time_reason_row(
        rows,
        "Runtime after install",
        _elapsed_excluding_dependency_install(with_run),
        _elapsed_excluding_dependency_install(base_run),
        "agent/job runtime after dependency setup",
    )
    _append_time_reason_row(
        rows,
        "Agent/model interaction residual",
        _agent_interaction_seconds(with_run),
        _agent_interaction_seconds(base_run),
        "time not attributed to captured dependency or non-install command spans",
    )
    command_interpretation = (
        "captured command time contributing to wall-clock slowdown"
        if elapsed_is_slower
        else "captured non-install command time contributing to runtime-after-install regression"
    )
    _append_time_reason_row(
        rows,
        command_span_label,
        driver_with_command_seconds,
        driver_base_command_seconds,
        command_interpretation,
    )
    _append_count_reason_row(
        rows,
        "Assistant turns",
        _assistant_turns(with_run),
        _assistant_turns(base_run),
        "extra model round-trips",
    )
    _append_count_reason_row(
        rows,
        "Extended-reasoning events",
        _thinking_token_events(with_run),
        _thinking_token_events(base_run),
        "extra reasoning activity",
    )
    with_tools = count_map(with_run, "tool_counts")
    base_tools = count_map(base_run, "tool_counts")
    for tool_name, interpretation in (
        ("Skill", "skill loading/context overhead"),
        ("Agent", "subagent initialization overhead"),
        ("ToolSearch", "tool schema lookup overhead"),
    ):
        _append_count_reason_row(
            rows,
            f"{tool_name} calls",
            with_tools.get(tool_name, 0),
            base_tools.get(tool_name, 0),
            interpretation,
        )
    if not rows:
        return ""
    lines = [
        "**Slowdown driver comparison**",
        "",
        f"| Driver | {markdown_cell(with_label)} | {markdown_cell(base_label)} | Delta | Interpretation |",
        "|---|---:|---:|---:|---|",
    ]
    for label, with_value, base_value, delta, interpretation in rows:
        lines.append(
            f"| {markdown_cell(label)} | {markdown_cell(with_value)} | {markdown_cell(base_value)} | "
            f"{markdown_cell(delta)} | {markdown_cell(interpretation)} |"
        )
    return "\n".join(lines)


def _token_delta_display(with_value: Any, base_value: Any, formatter=fmt_short) -> str:
    with_number = as_number(with_value)
    base_number = as_number(base_value)
    if with_number is None or base_number is None:
        return "NA"
    delta = with_number - base_number
    if delta == 0:
        return "0"
    sign = "+" if delta > 0 else "-"
    return f"{sign}{formatter(abs(delta))}"


def _cost_display(value: Any) -> str:
    number = as_number(value)
    return "NA" if number is None else f"${number:.4f}"


def _cost_delta_display(with_value: Any, base_value: Any) -> str:
    with_number = as_number(with_value)
    base_number = as_number(base_value)
    if with_number is None or base_number is None:
        return "NA"
    delta = with_number - base_number
    if delta == 0:
        return "$0.0000"
    sign = "+" if delta > 0 else "-"
    return f"{sign}${abs(delta):.4f}"


def _token_usage_comparison_table(with_run: RunEvidence, base_run: RunEvidence) -> str:
    with_label = with_run.label or "With skills"
    base_label = base_run.label or "No skills baseline"
    with_usage = _run_usage(with_run)
    base_usage = _run_usage(base_run)

    def optional_count(run: RunEvidence, map_key: str, count_key: str) -> Any:
        value = run_activity(run.raw).get(map_key)
        if not isinstance(value, dict):
            return None
        return value.get(count_key, 0)

    rows = [
        (
            "Total tokens",
            run_summary(with_run).get("token_count"),
            run_summary(base_run).get("token_count"),
            fmt_short,
            "overall token comparison",
        ),
        (
            "Cache-read tokens",
            with_usage.get("cache_read_input_tokens"),
            base_usage.get("cache_read_input_tokens"),
            fmt_short,
            "cached context re-read across turns",
        ),
        (
            "Cache-creation tokens",
            with_usage.get("cache_creation_input_tokens"),
            base_usage.get("cache_creation_input_tokens"),
            fmt_short,
            "new context written into prompt cache",
        ),
        (
            "Output tokens",
            with_usage.get("output_tokens"),
            base_usage.get("output_tokens"),
            fmt_short,
            "model response text",
        ),
        (
            "Assistant turns",
            optional_count(with_run, "event_types", "assistant"),
            optional_count(base_run, "event_types", "assistant"),
            fmt_number,
            "model round-trips",
        ),
        (
            "Skill calls",
            optional_count(with_run, "tool_counts", "Skill"),
            optional_count(base_run, "tool_counts", "Skill"),
            fmt_number,
            "skill documentation/context loading",
        ),
    ]
    lines = [
        "**Token usage comparison**",
        "",
        f"| Driver | {markdown_cell(with_label)} | {markdown_cell(base_label)} | Delta | Interpretation |",
        "|---|---:|---:|---:|---|",
    ]
    for label, with_value, base_value, formatter, interpretation in rows:
        lines.append(
            f"| {markdown_cell(label)} | {formatter(with_value)} | {formatter(base_value)} | "
            f"{_token_delta_display(with_value, base_value, formatter)} | {markdown_cell(interpretation)} |"
        )
    with_cost = with_usage.get("total_cost_usd")
    base_cost = base_usage.get("total_cost_usd")
    if as_number(with_cost) is not None or as_number(base_cost) is not None:
        lines.append(
            f"| Effective cost | {_cost_display(with_cost)} | {_cost_display(base_cost)} | "
            f"{_cost_delta_display(with_cost, base_cost)} | model/provider reported cost |"
        )
    return "\n".join(lines)


def _job_status_summary(run: RunEvidence, ctx: ReportContext | None) -> str:
    if ctx is None:
        return "not captured"
    job_exec = ctx.job_execution(run.mode)
    status = job_exec.status or "not captured"
    if job_exec.status_reason:
        status = f"{status}: {job_exec.status_reason}"
    return truncate(status, 220)


def _quality_regression_explanation(
    with_run: RunEvidence, base_run: RunEvidence, ctx: ReportContext | None
) -> list[str]:
    with_ev = ctx.evidence.get(with_run.mode) if ctx is not None else None
    base_ev = ctx.evidence.get(base_run.mode) if ctx is not None else None
    with_issues = run_quality_issues(with_run, with_ev)
    base_issues = run_quality_issues(base_run, base_ev)
    if not with_issues or base_issues:
        return []

    result_term = _result_term(ctx)
    with_label = with_run.label or "With skills"
    base_label = base_run.label or "No skills baseline"
    lines = [
        "**Primary result failure**",
        "",
        (
            f"{with_label} did not produce a usable {result_term}result. "
            f"{base_label} did, so faster wall time or lower token use is not a successful benchmark win."
        ),
        "",
        f"| Signal | {markdown_cell(with_label)} | {markdown_cell(base_label)} |",
        "|---|---|---|",
        f"| Job run status | {markdown_cell(_job_status_summary(with_run, ctx))} | "
        f"{markdown_cell(_job_status_summary(base_run, ctx))} |",
        f"| Result quality issue | {markdown_cell(with_issues[0])} | pass |",
        f"| Result metric | {markdown_cell(run_result_metric_status(with_run, with_ev))} | "
        f"{markdown_cell(run_result_metric_status(base_run, base_ev))} |",
    ]
    root_cause_notes = _plugin_narrative(ctx, "why_result_failure")
    if root_cause_notes:
        lines.extend(["", *root_cause_notes])
    return lines


def _job_status_needs_why(run: RunEvidence, ctx: ReportContext | None) -> bool:
    if ctx is None:
        return False
    status = (ctx.job_execution(run.mode).status or "").lower()
    return bool(status and status not in {"completed", "passed", "pass"})


def _run_needs_failure_why(run: RunEvidence, ctx: ReportContext | None) -> bool:
    ev = ctx.evidence.get(run.mode) if ctx is not None else None
    return bool(run_quality_issues(run, ev)) or _job_status_needs_why(run, ctx)


def _comparison_failure_explanation(
    runs_by_mode: dict[str, RunEvidence], modes: list[str], ctx: ReportContext | None
) -> list[str]:
    failing_modes = [mode for mode in modes if _run_needs_failure_why(runs_by_mode[mode], ctx)]
    if not failing_modes:
        return []

    lines = [
        "**Why the comparison needs review**",
        "",
        (
            "At least one run failed the job/result quality gates, so elapsed time, token use, and artifact count "
            "should not be treated as benchmark wins until the result issue is resolved."
        ),
        "",
        "| Run | Job run status | Result quality issue | Result metric |",
        "|---|---|---|---|",
    ]
    for mode in modes:
        run = runs_by_mode[mode]
        ev = ctx.evidence.get(mode) if ctx is not None else None
        issues = run_quality_issues(run, ev)
        lines.append(
            f"| {markdown_cell(run.label or mode)} | {markdown_cell(_job_status_summary(run, ctx))} | "
            f"{markdown_cell(issues[0] if issues else 'pass')} | "
            f"{markdown_cell(run_result_metric_status(run, ev))} |"
        )
    if len(failing_modes) == len(modes):
        lines.extend(
            [
                "",
                (
                    "Both runs need review; neither side is a valid comparison winner until the result metrics "
                    "are fixed."
                ),
            ]
        )
    root_cause_notes = _plugin_narrative(ctx, "why_result_failure")
    if root_cause_notes:
        lines.extend(["", *root_cause_notes])
    return lines


def _run_event_timeline(run: RunEvidence) -> list[dict[str, Any]]:
    """Ordered command/message items from the captured event stream (any agent shape)."""

    return event_timeline_from_text(str(run.agent_events_text or ""))


_REMEDIAL_COMMAND_CHECKS = {
    "missing_python_module": is_dependency_install_command,
}


def _rca_terms(signature: dict[str, str], anchor_command: str) -> set[str]:
    """Search terms derived from the failure itself — nothing situation-specific."""

    terms = {signature["subject"].split(".")[0].lower()}
    terms.update(word.lower() for word in re.findall(r"[A-Za-z_][A-Za-z0-9_-]{3,}", signature["display"])[:4])
    script = re.search(r"([\w-]+\.py)", anchor_command)
    if script:
        terms.add(script.group(1).lower())
    return {term for term in terms if term not in {"module", "named", "error", "exception"}}


def _timeline_item_display(item: dict[str, Any]) -> str:
    if item["kind"] == "command":
        exit_code = item.get("exit_code")
        exit_phrase = f", exit {exit_code}" if exit_code is not None else ""
        return f"ran `{inline_code_text(item.get('command'), 110)}`{exit_phrase}"
    return f"agent: {markdown_cell(truncate(str(item.get('text') or ''), 220))}"


def _failure_root_cause_chain(with_run: RunEvidence, base_run: RunEvidence) -> list[str]:
    """Auto-extracted root-cause chain for the failing run's terminal error.

    Fully evidence-derived: the failure signature is read from the failed
    command's own output, the search terms come from that signature, and every
    chain entry is a verbatim command or agent statement that mentions those
    terms. No situation-specific narrative is encoded here. The chain is only
    built for a genuinely terminal failure — a failure the run recovered from
    (a later command succeeded) produces no chain.
    """

    timeline = _run_event_timeline(with_run)
    anchored = terminal_failure_anchor(timeline)
    if anchored is None:
        return []
    anchor_index, signature = anchored
    anchor = timeline[anchor_index]
    terms = _rca_terms(signature, str(anchor.get("command") or ""))
    chain: list[dict[str, Any]] = []
    for item in timeline[:anchor_index]:
        haystack = " ".join(
            str(item.get(key) or "") for key in ("command", "text", "output")
        ).lower()
        if any(term in haystack for term in terms):
            chain.append(item)
    with_label = with_run.label or "With skills"
    base_label = base_run.label or "No skills baseline"
    lines = [
        f"**Root-cause chain (auto-extracted from {with_label} events)**",
        "",
        f"Terminal failure: `{inline_code_text(anchor.get('command'), 110)}` -> `{markdown_cell(signature['display'])}`. "
        f"Evidence mentioning {', '.join(f'`{term}`' for term in sorted(terms))} before the failure:",
        "",
    ]
    for index, item in enumerate(chain[-8:], start=1):
        lines.append(f"{index}. {_timeline_item_display(item)}")
    remedial_check = _REMEDIAL_COMMAND_CHECKS.get(signature["kind"])
    if remedial_check is not None:
        ran = [
            item
            for item in timeline
            if item["kind"] == "command" and remedial_check(str(item.get("command") or ""))
        ]
        if ran:
            lines.append(
                f"- A remedial command ran but did not prevent the failure: "
                f"`{inline_code_text(ran[-1].get('command'), 110)}`."
            )
        else:
            lines.append(f"- No command that would remediate `{signature['display']}` ran at any point in this run.")
        base_remedial = [
            item
            for item in _run_event_timeline(base_run)
            if item["kind"] == "command" and remedial_check(str(item.get("command") or ""))
        ]
        if base_remedial:
            lines.append(
                f"- {base_label} contrast: it ran `{inline_code_text(base_remedial[0].get('command'), 110)}` "
                "before its job run."
            )
    prediction = predicted_failure_message(str(with_run.agent_events_text or ""))
    if prediction:
        lines.append(
            "- Lint `known_doomed_execution`: the agent predicted this failure before running the command "
            f"anyway — {markdown_cell(truncate(prediction['quote'], 220))} An agent that expects a command to "
            "fail should resolve the blocker or skip the command with an explicit blocker, not execute a "
            "known-doomed run."
        )
    return lines


def _root_cause_lead(with_run: RunEvidence, base_run: RunEvidence, ctx: ReportContext | None = None) -> list[str]:
    """Ranked, concrete root-cause bullets that LEAD the Why section.

    Each bullet names the cause, the attributed seconds, and the evidence in one
    sentence, so the answer is readable before the supporting driver tables.
    """

    with_label = with_run.label or "With skills"
    base_label = base_run.label or "No skills baseline"
    causes: list[tuple[float, str]] = []

    with_installs = _dependency_install_spans(with_run)
    base_installs = _dependency_install_spans(base_run)
    with_install_seconds = _install_total_seconds(with_installs)
    install_delta = with_install_seconds - _install_total_seconds(base_installs)
    if with_installs and install_delta >= 60:
        base_cpu_only = _install_cpu_only_evidence_display(base_installs)
        with_cpu_only = _install_cpu_only_evidence_display(with_installs)
        with_stack = _install_stack_evidence(with_installs)
        detail = (
            f"{with_label} spent {fmt_seconds_with_unit(with_install_seconds)} on "
            f"{_install_strategy_summary(with_installs)}, vs "
            f"{fmt_seconds_with_unit(_install_total_seconds(base_installs))} for {base_label}"
        )
        if base_cpu_only and not with_cpu_only:
            detail += (
                f". The gap is wheel selection, not download luck: the baseline explicitly installed the "
                f"CPU-only framework wheel, while {with_label} resolved an {with_stack} from its requirements "
                "file. For a CPU-only benchmark, pinning CPU wheels in the skill/requirements removes this delta"
            )
        elif "accelerator-capable" in with_stack:
            detail += f". The {with_label} install resolved an {with_stack}"
        causes.append((install_delta, f"**Dependency install +{fmt_seconds(install_delta)}s** — {detail}."))

    with_ev = _run_evidence(ctx, with_run)
    with_job_spans = list((with_ev.job_execution or JobExecutionSignal()).successful_job_spans)
    base_job_spans = list((_run_evidence(ctx, base_run).job_execution or JobExecutionSignal()).successful_job_spans)
    if len(with_job_spans) > 1:
        with_job_seconds = _span_total_seconds(with_job_spans) or 0.0
        longest = max((as_number(span.get("duration_seconds")) or 0.0) for span in with_job_spans)
        rerun_seconds = max(0.0, with_job_seconds - longest)
        if rerun_seconds >= 30:
            atom = _execution_atom(ctx) or "job"
            causes.append(
                (
                    rerun_seconds,
                    f"**Repeated {atom} executions +{fmt_seconds(rerun_seconds)}s** — {with_label} ran "
                    f"{len(with_job_spans)} successful executions (total {fmt_seconds_with_unit(with_job_seconds)}) "
                    f"vs {len(base_job_spans)} for {base_label}; the reruns beyond the first are re-validation "
                    "work (captured rationale in the repeated-executions table below).",
                )
            )

    with_residual = _agent_interaction_seconds(with_run)
    base_residual = _agent_interaction_seconds(base_run)
    if with_residual is not None and base_residual is not None:
        residual_delta = with_residual - base_residual
        if residual_delta > 30:
            with_turns = _assistant_turns(with_run)
            base_turns = _assistant_turns(base_run)
            with_skill_calls = count_map(with_run, "tool_counts").get("Skill", 0)
            base_skill_calls = count_map(base_run, "tool_counts").get("Skill", 0)
            turn_phrase = ""
            if with_turns is not None and base_turns is not None:
                turn_phrase = f" ({with_turns} vs {base_turns} assistant turns"
                if with_skill_calls or base_skill_calls:
                    turn_phrase += f", {with_skill_calls} vs {base_skill_calls} Skill loads"
                turn_phrase += ")"
            causes.append(
                (
                    residual_delta,
                    f"**Agent/model loop +{fmt_seconds(residual_delta)}s** — time outside captured commands: "
                    f"model round-trips, skill loading, and file inspection{turn_phrase}.",
                )
            )

    if not causes:
        return []
    causes.sort(key=lambda item: item[0], reverse=True)
    lines = ["**Root causes (ranked by attributed time)**", ""]
    lines.extend(f"{index}. {text}" for index, (_seconds, text) in enumerate(causes, start=1))
    return lines


def _why_slower(with_run: RunEvidence, base_run: RunEvidence, ctx: ReportContext | None = None) -> list[str]:
    with_label = with_run.label or "With skills"
    base_label = base_run.label or "No skills baseline"
    with_time = as_number(run_summary(with_run).get("elapsed_seconds")) or 0
    base_time = as_number(run_summary(base_run).get("elapsed_seconds")) or 0
    time_delta = with_time - base_time
    pct = round(time_delta / base_time * 100) if base_time > 0 else 0
    with_runtime = _elapsed_excluding_dependency_install(with_run)
    base_runtime = _elapsed_excluding_dependency_install(base_run)
    runtime_delta = with_runtime - base_runtime if with_runtime is not None and base_runtime is not None else None
    runtime_pct = round(runtime_delta / base_runtime * 100) if runtime_delta is not None and base_runtime else 0

    with_command_seconds = _command_span_total_seconds(with_run)
    base_command_seconds = _command_span_total_seconds(base_run)
    elapsed_is_slower = time_delta > 0
    runtime_is_slower = runtime_delta is not None and runtime_delta > 0
    if elapsed_is_slower:
        driver_with_command_seconds = with_command_seconds
        driver_base_command_seconds = base_command_seconds
        command_span_label = "Captured command time"
    else:
        driver_with_command_seconds = _non_dependency_command_seconds(with_run)
        driver_base_command_seconds = _non_dependency_command_seconds(base_run)
        command_span_label = "Captured non-install command time"

    if elapsed_is_slower and runtime_is_slower:
        heading = (
            f"**Why {with_label} is slower and has longer runtime after install** "
            f"(+{fmt_number(time_delta)}s total / +{pct}%; "
            f"+{fmt_seconds(runtime_delta)}s runtime / +{runtime_pct}% vs {base_label}):"
        )
    elif elapsed_is_slower:
        heading = f"**Why {with_label} is slower** (+{fmt_number(time_delta)}s / +{pct}% vs {base_label}):"
    elif runtime_is_slower:
        heading = (
            f"**Why {with_label} has longer runtime after install** "
            f"(+{fmt_seconds(runtime_delta)}s / +{runtime_pct}% vs {base_label}):"
        )
    else:
        heading = f"**Why {with_label} needs more work**:"
    lines = [heading, ""]
    quality_explanation = _quality_regression_explanation(with_run, base_run, ctx)
    include_slowdown_context = elapsed_is_slower or runtime_is_slower or not quality_explanation
    if include_slowdown_context:
        root_causes = _root_cause_lead(with_run, base_run, ctx)
        if root_causes:
            lines.extend([*root_causes, ""])
    if quality_explanation:
        lines.extend([*quality_explanation, ""])
    if with_run.rca_report.strip():
        # An agent-driven RCA investigation ran (benchmark.harness.rca): its
        # report is authoritative over the deterministic seed chain.
        lines.extend(
            [
                f"**Agent root-cause investigation ({with_run.label or 'With skills'})**",
                "",
                with_run.rca_report.strip(),
                "",
            ]
        )
    else:
        root_cause_chain = _failure_root_cause_chain(with_run, base_run)
        if root_cause_chain:
            lines.extend([*root_cause_chain, ""])
            lines.append(
                "_Run `python -m benchmark.harness.rca <result_root>` to have an agent investigate this "
                "failure iteratively and embed its root-cause report here._"
            )
            lines.append("")
    slowdown_table = _slowdown_reason_table(
        with_run,
        base_run,
        driver_with_command_seconds=driver_with_command_seconds,
        driver_base_command_seconds=driver_base_command_seconds,
        command_span_label=command_span_label,
        elapsed_is_slower=elapsed_is_slower,
    )
    if slowdown_table and include_slowdown_context:
        lines.extend(
            [
                slowdown_table,
                "",
            ]
        )
    repeated_dependency_installs = repeated_dependency_install_section(
        {"with": with_run, "base": base_run},
        ["with", "base"],
    )
    if repeated_dependency_installs:
        lines.extend([repeated_dependency_installs, ""])
    repeated_runs = repeated_job_runs_slowdown_section(with_run, base_run, ctx)
    if repeated_runs:
        lines.extend([repeated_runs, ""])
    if include_slowdown_context:
        lines.extend(["", _elapsed_time_accounting_note(with_run, base_run), ""])
    longest_command_note = _longest_command_comparison_note(with_run, base_run)
    if longest_command_note and include_slowdown_context:
        lines.extend([longest_command_note, ""])
    mismatch_note = _skill_contract_mismatch_note(with_run)
    if mismatch_note and include_slowdown_context:
        lines.extend([mismatch_note, ""])
    interaction_note = _agent_interaction_slowdown_note(with_run, base_run)
    if interaction_note and include_slowdown_context:
        lines.extend([interaction_note, ""])
    dependency_note = _dependency_install_slowdown_note(with_run, base_run)
    if dependency_note and include_slowdown_context:
        lines.append(dependency_note)
    if include_slowdown_context:
        # E3 named slot "why_slowdown": the FL runtime-path note now lives in
        # NvflareReportPlugin.explain() and is contributed through this anchor. A
        # flat/absent SDK contributes nothing -> no engine fallback.
        runtime_notes = _plugin_narrative(ctx, "why_slowdown")
        if runtime_notes:
            if lines[-1] != "":
                lines.append("")
            lines.extend(runtime_notes)
            lines.append("")
        lines.extend(_code_quality_slowdown_notes(with_run, base_run, ctx))
    if len(lines) == 2:
        lines.append("- Cause not resolved from available activity signals.")
    return lines


def _why_dependency_install_slower(with_run: RunEvidence, base_run: RunEvidence) -> list[str]:
    dependency_note = _dependency_install_slowdown_note(with_run, base_run)
    if not dependency_note:
        return []
    with_label = with_run.label or "With skills"
    base_label = base_run.label or "No skills baseline"
    with_dependency = _dependency_install_total_seconds(with_run)
    base_dependency = _dependency_install_total_seconds(base_run)
    with_dependency_number = as_number(with_dependency) or 0
    base_dependency_number = as_number(base_dependency) or 0
    dependency_delta = with_dependency_number - base_dependency_number
    pct = round(dependency_delta / base_dependency_number * 100) if base_dependency_number > 0 else 0
    return [
        f"**Why {with_label} has longer dependency install** "
        f"(+{fmt_seconds(dependency_delta)}s / +{pct}% vs {base_label}):",
        "",
        "| Run | Dependency install | Runtime after install | Total elapsed |",
        "|---|---:|---:|---:|",
        (
            f"| {markdown_cell(with_label)} | {fmt_seconds_with_unit(with_dependency)} | "
            f"{fmt_seconds_with_unit(_elapsed_excluding_dependency_install(with_run))} | "
            f"{fmt_seconds_with_unit(run_summary(with_run).get('elapsed_seconds'))} |"
        ),
        (
            f"| {markdown_cell(base_label)} | {fmt_seconds_with_unit(base_dependency)} | "
            f"{fmt_seconds_with_unit(_elapsed_excluding_dependency_install(base_run))} | "
            f"{fmt_seconds_with_unit(run_summary(base_run).get('elapsed_seconds'))} |"
        ),
        "",
        (
            "This isolates dependency setup/download time from the actual job/runtime path; "
            "a run can finish faster overall while still spending more time installing dependencies."
        ),
        "",
        dependency_note,
    ]


def _why_more_tokens(with_run: RunEvidence, base_run: RunEvidence) -> list[str]:
    with_label = with_run.label or "With skills"
    base_label = base_run.label or "No skills baseline"
    with_tokens = as_number(run_summary(with_run).get("token_count")) or 0
    base_tokens = as_number(run_summary(base_run).get("token_count")) or 0
    token_delta = with_tokens - base_tokens
    pct = round(token_delta / base_tokens * 100) if base_tokens > 0 else 0

    with_usage = _run_usage(with_run)
    base_usage = _run_usage(base_run)
    with_cache_read = as_number(with_usage.get("cache_read_input_tokens")) or 0
    base_cache_read = as_number(base_usage.get("cache_read_input_tokens")) or 0
    with_cache_create = as_number(with_usage.get("cache_creation_input_tokens")) or 0
    base_cache_create = as_number(base_usage.get("cache_creation_input_tokens")) or 0
    with_output = as_number(with_usage.get("output_tokens")) or 0
    base_output = as_number(base_usage.get("output_tokens")) or 0
    with_cost = as_number(with_usage.get("total_cost_usd"))
    base_cost = as_number(base_usage.get("total_cost_usd"))
    with_turns = _assistant_turns(with_run)
    base_turns = _assistant_turns(base_run)
    with_tools = count_map(with_run, "tool_counts")
    base_tools = count_map(base_run, "tool_counts")
    skill_calls = with_tools.get("Skill", 0)
    base_skill_calls = base_tools.get("Skill", 0)

    lines = [
        f"**Why {with_label} uses more tokens** (+{fmt_short(token_delta)} / +{pct}% vs {base_label}):",
        "",
        _token_usage_comparison_table(with_run, base_run),
        "",
    ]
    detailed_notes = 0

    cache_read_delta = with_cache_read - base_cache_read
    if cache_read_delta > 0 and with_cache_read > 0 and token_delta > 0:
        cache_pct = round(cache_read_delta / token_delta * 100)
        detailed_notes += 1
        lines.append(
            f"- **Prompt cache re-reads are the dominant driver** "
            f"({fmt_short(with_cache_read)} vs {fmt_short(base_cache_read)}, "
            f"+{fmt_short(cache_read_delta)}, {cache_pct}% of the total token delta): "
            f"cache-read tokens represent context cached from previous turns being re-read on each "
            f"new turn. The {with_label} run accumulated a larger cached context window — primarily "
            f"skill documentation injected via {skill_calls} Skill call(s) — and then re-read that "
            f"context across all {with_turns} turns (vs {base_turns} turns in the {base_label} run)."
        )
    if skill_calls > base_skill_calls:
        detailed_notes += 1
        lines.append(
            f"- **Skill documentation injected into context** ({skill_calls} Skill call(s) vs {base_skill_calls}): "
            f"each Skill invocation adds skill documentation to the context window. "
            f"That content is written into the prompt cache on first use, then re-read as cached context "
            f"on every subsequent turn — compounding the cache-read cost with each additional turn."
        )
    cache_create_delta = with_cache_create - base_cache_create
    if abs(cache_create_delta) > 1000:
        detailed_notes += 1
        if cache_create_delta > 0:
            lines.append(
                f"- **New context written to cache** (+{fmt_short(cache_create_delta)} cache-creation tokens): "
                f"the {with_label} run wrote more new content into the prompt cache "
                f"(skill docs, tool schemas, or conversation history not present in the {base_label} run)."
            )
        else:
            lines.append(
                f"- **Less new context cached** ({fmt_short(abs(cache_create_delta))} fewer cache-creation tokens): "
                f"the {base_label} run actually wrote more fresh content into the cache."
            )
    output_delta = with_output - base_output
    if abs(output_delta) > 500:
        detailed_notes += 1
        if output_delta < 0:
            lines.append(
                f"- **Output tokens decreased** ({fmt_short(with_output)} vs {fmt_short(base_output)}, "
                f"{fmt_short(abs(output_delta))} fewer): "
                f"the {with_label} run generated less text overall — skill guidance focused the agent's "
                f"responses, reducing exploratory output even as context consumption grew."
            )
        else:
            lines.append(
                f"- **Output tokens increased** ({fmt_short(with_output)} vs {fmt_short(base_output)}, "
                f"+{fmt_short(output_delta)}): "
                f"the {with_label} run generated more text, contributing directly to the token delta."
            )
    if with_cost is not None and base_cost is not None:
        cost_delta = with_cost - base_cost
        cost_pct = round(cost_delta / base_cost * 100) if base_cost > 0 else 0
        detailed_notes += 1
        lines.append(
            f"- **Effective cost** (${with_cost:.4f} vs ${base_cost:.4f}, +${cost_delta:.4f} / +{cost_pct}%): "
            f"despite {pct}% more total tokens, the cost premium is much smaller because "
            f"cache-read tokens are priced significantly lower than regular input tokens."
        )
    if detailed_notes == 0:
        lines.append(
            "- Detailed token subcomponents were not available or did not isolate one dominant cause; use the table above "
            "to see which captured token/work drivers changed."
        )
    return lines


def why_section(
    runs: dict[str, RunEvidence | dict[str, Any]], modes: list[str], ctx: ReportContext | None = None
) -> str:
    """Explain why the with-skills run is slower, costlier, or needs more work."""
    from ...modes import WITH_SKILLS_MODE

    ctx = ctx or _report_context(runs, modes)
    if len(modes) != 2:
        return ""
    if WITH_SKILLS_MODE not in modes:
        return ""
    base_mode = next(m for m in modes if m != WITH_SKILLS_MODE)
    with_run = _as_run_evidence(runs.get(WITH_SKILLS_MODE, {}))
    base_run = _as_run_evidence(runs.get(base_mode, {}))
    with_time = as_number(run_summary(with_run).get("elapsed_seconds"))
    base_time = as_number(run_summary(base_run).get("elapsed_seconds"))
    with_runtime = _elapsed_excluding_dependency_install(with_run)
    base_runtime = _elapsed_excluding_dependency_install(base_run)
    with_tokens = as_number(run_summary(with_run).get("token_count"))
    base_tokens = as_number(run_summary(base_run).get("token_count"))
    sections: list[list[str]] = []
    elapsed_is_slower = with_time is not None and base_time is not None and with_time > base_time
    runtime_is_slower = with_runtime is not None and base_runtime is not None and with_runtime > base_runtime
    with_quality_issues = run_quality_issues(with_run, ctx.evidence.get(WITH_SKILLS_MODE))
    base_quality_issues = run_quality_issues(base_run, ctx.evidence.get(base_mode))
    quality_is_worse = bool(with_quality_issues) and not base_quality_issues
    any_failure_needs_why = any(_run_needs_failure_why(_as_run_evidence(runs.get(mode, {})), ctx) for mode in modes)
    if any_failure_needs_why and not quality_is_worse:
        runs_by_mode = {mode: _as_run_evidence(runs.get(mode, {})) for mode in modes}
        sections.append(_comparison_failure_explanation(runs_by_mode, modes, ctx))
    if elapsed_is_slower or runtime_is_slower or quality_is_worse:
        sections.append(_why_slower(with_run, base_run, ctx))
    elif _dependency_install_slowdown_note(with_run, base_run):
        sections.append(_why_dependency_install_slower(with_run, base_run))
    if with_tokens is not None and base_tokens is not None and with_tokens > base_tokens:
        sections.append(_why_more_tokens(with_run, base_run))
    if not sections:
        return ""
    lines = ["## Why", ""]
    for section_lines in sections:
        lines.extend(section_lines)
        lines.append("")
    return "\n".join(lines)
