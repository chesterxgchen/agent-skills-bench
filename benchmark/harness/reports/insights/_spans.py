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

from .._events import (
    _format_command_span,
    _longest_span,
    agent_command_spans,
    as_number,
    command_error_summary,
    command_failed,
    command_succeeded,
    fmt_seconds_with_unit,
    inline_code_text,
    is_dependency_install_command,
    parse_event_timestamp,
)
from .._text import fmt_number, markdown_cell, strip_ansi
from ._plugin_view import dependency_install_attempted, event_type_count, run_summary

__all__ = [
    "_run_usage",
    "_thinking_token_events",
    "_assistant_turns",
    "_command_span_total_seconds",
    "_dependency_install_total_seconds",
    "_non_dependency_command_seconds",
    "_elapsed_excluding_dependency_install",
    "_time_accounting_display",
    "_top_command_spans",
    "_format_command_span_list",
    "_dependency_install_spans",
    "_package_name_from_install_token",
    "_dependency_package_examples",
    "_accelerator_dependency_packages",
    "_targeted_followup_install_span",
    "_failed_requirements_install_span",
    "_install_strategy_label",
    "_install_strategy_summary",
    "_install_tool_label",
    "_install_network_markers",
    "_install_network_markers_for_spans",
    "_install_network_marker_display",
    "_install_network_marker_display_for_spans",
    "_install_total_display",
    "_dependency_install_slowdown_note",
    "_dependency_install_retry_reason",
]


def _run_usage(run: dict[str, Any]) -> dict[str, Any]:
    usage = run.usage
    return usage if isinstance(usage, dict) else {}


def _thinking_token_events(run: dict[str, Any]) -> int:
    return event_type_count(run, "system.thinking_tokens")


def _assistant_turns(run: dict[str, Any]) -> int:
    return event_type_count(run, "assistant")


def _command_span_total_seconds(run: dict[str, Any]) -> float:
    return sum(
        float(span["duration_seconds"])
        for span in _adjusted_command_spans(run)
        if as_number(span.get("duration_seconds")) is not None
    )


def _dependency_install_total_seconds(run: dict[str, Any]) -> float | None:
    spans = _dependency_install_spans(run)
    if not spans:
        return None if dependency_install_attempted(run) else 0.0
    values = [as_number(span.get("duration_seconds")) for span in spans]
    durations = [value for value in values if value is not None]
    return sum(durations) if durations else None


def _non_dependency_command_seconds(run: dict[str, Any]) -> float | None:
    spans = [
        span
        for span in agent_command_spans(run.raw)
        if not is_dependency_install_command(str(span.get("command") or ""))
    ]
    values = [as_number(span.get("duration_seconds")) for span in spans]
    durations = [value for value in values if value is not None]
    return sum(durations) if durations else None


def _elapsed_excluding_dependency_install(run: dict[str, Any]) -> float | None:
    elapsed = as_number(run_summary(run).get("elapsed_seconds"))
    dependency_seconds = _dependency_install_total_seconds(run)
    if elapsed is None or dependency_seconds is None:
        return None
    return max(0.0, elapsed - dependency_seconds)


def _time_accounting_display(run: dict[str, Any]) -> str:
    elapsed = as_number(run_summary(run).get("elapsed_seconds"))
    dependency_seconds = _dependency_install_total_seconds(run)
    runtime_seconds = _elapsed_excluding_dependency_install(run)
    non_install_seconds = _non_dependency_command_seconds(run)
    return (
        f"total {fmt_seconds_with_unit(elapsed)}; "
        f"dependency install {fmt_seconds_with_unit(dependency_seconds)}; "
        f"runtime after install {fmt_seconds_with_unit(runtime_seconds)}; "
        f"captured non-install commands {fmt_seconds_with_unit(non_install_seconds)}"
    )


def _top_command_spans(run: dict[str, Any], *, limit: int = 3, min_seconds: float = 30.0) -> list[dict[str, Any]]:
    spans = [
        span
        for span in _adjusted_command_spans(run)
        if (as_number(span.get("duration_seconds")) or 0) >= min_seconds
        and str(span.get("status") or "") in {"completed", "failed"}
    ]
    return sorted(spans, key=lambda item: as_number(item.get("duration_seconds")) or 0, reverse=True)[:limit]


def _format_command_span_list(label: str, spans: list[dict[str, Any]]) -> str:
    if not spans:
        return f"{label}: no timed command spans >=30s captured"
    return f"{label}: " + "; ".join(_format_command_span(span) for span in spans)


def _background_task_durations_by_tool_id(run: dict[str, Any]) -> dict[str, float]:
    task_by_tool_id: dict[str, str] = {}
    starts = {}
    ends = {}
    for line in str(run.raw.get("agent_events_text") or "").splitlines():
        try:
            payload = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        timestamp = parse_event_timestamp(payload.get("harness_timestamp") or payload.get("timestamp"))
        event_type = str(payload.get("event_type") or payload.get("type") or "")
        if event_type == "system.task_started":
            task_id = str(payload.get("task_id") or "")
            tool_id = str(payload.get("tool_use_id") or "")
            if task_id and tool_id:
                task_by_tool_id[tool_id] = task_id
                if timestamp:
                    starts[task_id] = timestamp
            continue
        if event_type not in {"system.task_updated", "system.task_notification"}:
            continue
        task_id = str(payload.get("task_id") or "")
        if not task_id or not timestamp:
            continue
        patch = payload.get("patch") if isinstance(payload.get("patch"), dict) else {}
        status = str(payload.get("status") or patch.get("status") or "").lower()
        if status in {"completed", "failed", "killed", "stopped"}:
            ends[task_id] = timestamp

    durations = {}
    for tool_id, task_id in task_by_tool_id.items():
        start = starts.get(task_id)
        end = ends.get(task_id)
        if not start or not end:
            continue
        duration = (end - start).total_seconds()
        if duration >= 0:
            durations[tool_id] = duration
    return durations


def _span_uses_background_task(span: dict[str, Any]) -> bool:
    output = str(span.get("output") or "")
    return "Command running in background with ID:" in output


def _adjust_dependency_install_span(span: dict[str, Any], background_durations: dict[str, float]) -> dict[str, Any]:
    if is_dependency_install_command(str(span.get("command") or "")):
        tool_id = str(span.get("id") or "")
        background_duration = background_durations.get(tool_id)
        if background_duration is not None and _span_uses_background_task(span):
            adjusted = dict(span)
            adjusted["duration_seconds"] = background_duration
            adjusted["duration_source"] = "background_task"
            return adjusted
    return span


def _adjusted_command_spans(run: dict[str, Any]) -> list[dict[str, Any]]:
    background_durations = _background_task_durations_by_tool_id(run)
    return [_adjust_dependency_install_span(span, background_durations) for span in agent_command_spans(run.raw)]


def _dependency_install_spans(run: dict[str, Any]) -> list[dict[str, Any]]:
    spans = []
    for span in _adjusted_command_spans(run):
        if not is_dependency_install_command(str(span.get("command") or "")):
            continue
        spans.append(span)
    return spans


def _package_name_from_install_token(token: str) -> str:
    clean = token.strip().strip(",")
    if "==" in clean:
        return clean.split("==", 1)[0]
    match = re.match(r"(.+?)-\d", clean)
    if match:
        return match.group(1)
    return clean


def _dependency_package_examples(output: str, limit: int = 4) -> list[str]:
    downloads: list[tuple[float, str]] = []
    for match in re.finditer(
        r"\bDownloading\s+([A-Za-z0-9_.+-]+)\s+\(([0-9.]+)([KMG]?i?B)\)",
        strip_ansi(output),
        flags=re.IGNORECASE,
    ):
        multiplier = {
            "kb": 1 / 1024,
            "kib": 1 / 1024,
            "mb": 1,
            "mib": 1,
            "gb": 1024,
            "gib": 1024,
        }.get(match.group(3).lower(), 1)
        downloads.append((float(match.group(2)) * multiplier, match.group(1)))
    if downloads:
        examples = []
        for _, name in sorted(downloads, reverse=True):
            if name not in examples:
                examples.append(name)
            if len(examples) >= limit:
                return examples

    examples = []
    for pattern in (
        r"\bDownloading\s+([A-Za-z0-9_.+-]+)",
        r"^\s*\+\s+([A-Za-z0-9_.+-]+)==",
        r"\bSuccessfully installed\s+(.+)",
    ):
        for match in re.finditer(pattern, strip_ansi(output), flags=re.IGNORECASE | re.MULTILINE):
            if pattern.endswith("(.+)"):
                names = [_package_name_from_install_token(part) for part in match.group(1).split()]
            else:
                names = [match.group(1)]
            for name in names:
                clean = name.strip().strip(",")
                if clean and clean not in examples:
                    examples.append(clean)
                if len(examples) >= limit:
                    return examples
    return examples


def _accelerator_dependency_packages(output: str, limit: int = 5) -> list[str]:
    packages = []
    for token in re.findall(r"[A-Za-z0-9_.+-]+", strip_ansi(output)):
        lowered = token.lower()
        if not (lowered.startswith(("nvidia-", "cuda-", "triton-")) or lowered == "triton"):
            continue
        name = _package_name_from_install_token(token)
        if name and name not in packages:
            packages.append(name)
        if len(packages) >= limit:
            break
    return packages


def _targeted_followup_install_span(spans: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        span for span in spans if command_succeeded(span) and "-r" not in str(span.get("command") or "").lower()
    ]
    return _longest_span(candidates)


def _failed_requirements_install_span(spans: list[dict[str, Any]]) -> dict[str, Any] | None:
    for span in spans:
        command = str(span.get("command") or "").lower()
        if "-r" in command and "requirements" in command and command_failed(span):
            return span
    return None


def _install_strategy_label(span: dict[str, Any]) -> str:
    command = str(span.get("command") or "").lower()
    if re.search(r"\s-r\s+\S+|--requirement\s+\S+", command):
        return "requirements-file install"
    return "targeted package install"


def _install_strategy_summary(spans: list[dict[str, Any]]) -> str:
    if not spans:
        return "no dependency install command captured"
    counts: dict[str, int] = {}
    failed = 0
    for span in spans:
        counts[_install_strategy_label(span)] = counts.get(_install_strategy_label(span), 0) + 1
        if command_failed(span):
            failed += 1
    parts = [f"{count} {label}(s)" for label, count in sorted(counts.items())]
    if failed:
        parts.append(f"{failed} failed")
    return ", ".join(parts)


def _install_strategy_phrase(spans: list[dict[str, Any]]) -> str:
    if not spans:
        return "no captured dependency install command"
    counts: dict[str, int] = {}
    failed = 0
    for span in spans:
        counts[_install_strategy_label(span)] = counts.get(_install_strategy_label(span), 0) + 1
        if command_failed(span):
            failed += 1
    parts = [f"{count} {label} command{'s' if count != 1 else ''}" for label, count in sorted(counts.items())]
    if failed:
        parts.append(f"{failed} failed")
    return ", ".join(parts)


def _install_tool_label(span: dict[str, Any] | None) -> str:
    if not span:
        return "no install command"
    command = str(span.get("command") or "").lower()
    if re.search(r"\buv\s+pip\s+install\b", command):
        return "uv pip"
    if re.search(r"\bpython3?\s+-m\s+pip\s+install\b", command):
        return "python -m pip"
    if re.search(r"\bpip\s+install\b", command):
        return "pip"
    return "unknown installer"


def _install_network_markers(span: dict[str, Any] | None) -> list[str]:
    if not span:
        return []
    output = str(span.get("output") or "")
    checks = [
        ("connection timeout", r"connection timed out|read timed out|\btimed out\b"),
        ("broken/incomplete download", r"IncompleteRead|Connection broken|ProtocolError|incomplete download"),
        ("resumed incomplete download", r"attempting to resume incomplete download|resuming download"),
        ("DNS resolution failure", r"NameResolutionError|failed to resolve|temporary failure in name resolution"),
        ("download retry", r"\bRetrying\b|after connection broken"),
    ]
    markers = []
    for label, pattern in checks:
        if re.search(pattern, output, flags=re.IGNORECASE):
            markers.append(label)
    return markers


def _install_network_markers_for_spans(spans: list[dict[str, Any]]) -> list[str]:
    markers = []
    for span in spans:
        for marker in _install_network_markers(span):
            if marker not in markers:
                markers.append(marker)
    return markers


def _install_network_marker_display(label: str, span: dict[str, Any] | None) -> str:
    markers = _install_network_markers(span)
    if markers:
        return f"{label} install log showed {', '.join(markers)}"
    if span:
        return f"{label} install log showed no captured network retry/timeout markers"
    return f"{label} had no captured install log"


def _install_network_marker_display_for_spans(label: str, spans: list[dict[str, Any]]) -> str:
    markers = _install_network_markers_for_spans(spans)
    if markers:
        return f"{label} install logs showed {', '.join(markers)}"
    if spans:
        return f"{label} install logs showed no captured network retry/timeout markers"
    return f"{label} had no captured install log"


def _install_total_display(spans: list[dict[str, Any]]) -> str:
    durations = [as_number(span.get("duration_seconds")) for span in spans]
    total = sum(value for value in durations if value is not None)
    return f"{fmt_number(round(total))}s across {len(spans)} install command(s)"


def _install_total_seconds(spans: list[dict[str, Any]]) -> float:
    return sum(as_number(span.get("duration_seconds")) or 0 for span in spans)


def _install_evidence_text(spans: list[dict[str, Any]]) -> str:
    return "\n".join(f"{span.get('command') or ''}\n{span.get('output') or ''}" for span in spans).lower()


def _install_cpu_only_evidence_display(spans: list[dict[str, Any]]) -> str | None:
    text = _install_evidence_text(spans)
    if "download.pytorch.org/whl/cpu" in text:
        return "the explicit CPU-only PyTorch wheel index"
    if "+cpu" in text:
        return "a CPU-only framework wheel"
    return None


def _install_stack_evidence(spans: list[dict[str, Any]]) -> str:
    text = _install_evidence_text(spans)
    accelerator_packages = _accelerator_dependency_packages(text)
    if "download.pytorch.org/whl/cpu" in text or "+cpu" in text:
        return "CPU-only framework wheel"
    if accelerator_packages:
        return f"accelerator-capable dependency stack ({', '.join(accelerator_packages)})"
    if re.search(r"\btorch\b|\bpytorch(?:[-_]lightning)?\b|\btorchmetrics\b", text):
        return "framework dependency stack"
    if any(_install_strategy_label(span) == "requirements-file install" for span in spans):
        return "requirements-defined training stack, not CPU-only pinned"
    return "not captured"


def _install_path_evidence_phrase(label: str, spans: list[dict[str, Any]], stack_evidence: str) -> str:
    if not spans:
        return f"{label} had no captured dependency install command"
    strategy = _install_strategy_phrase(spans)
    if stack_evidence == "not captured":
        return f"{label} used {strategy}, with stack evidence not captured"
    return f"{label} used {strategy} with {stack_evidence}"


def _run_label(run: dict[str, Any], default: str) -> str:
    label = getattr(run, "label", None)
    if not label and isinstance(run, dict):
        label = run.get("label")
    return str(label or default)


def _install_command_display(span: dict[str, Any] | None) -> str:
    if not span:
        return "not captured"
    return f"`{inline_code_text(span.get('command'), 96)}`"


def _install_runtime_offset_note(with_run: dict[str, Any], base_run: dict[str, Any]) -> str | None:
    with_total = as_number(run_summary(with_run).get("elapsed_seconds"))
    base_total = as_number(run_summary(base_run).get("elapsed_seconds"))
    with_runtime = _elapsed_excluding_dependency_install(with_run)
    base_runtime = _elapsed_excluding_dependency_install(base_run)
    if with_total is None or base_total is None or with_runtime is None or base_runtime is None:
        return None
    if with_total >= base_total or with_runtime >= base_runtime:
        return None
    total_saved = base_total - with_total
    runtime_saved = base_runtime - with_runtime
    return (
        "- **Why With skills is still faster overall**: the dependency install was slower, "
        f"but runtime after install was {fmt_seconds_with_unit(runtime_saved)} faster "
        f"({fmt_seconds_with_unit(with_runtime)} vs {fmt_seconds_with_unit(base_runtime)}). "
        f"That more than offset the setup cost, so total elapsed was {fmt_seconds_with_unit(total_saved)} faster "
        f"({fmt_seconds_with_unit(with_total)} vs {fmt_seconds_with_unit(base_total)})."
    )


def _dependency_install_slowdown_note(with_run: dict[str, Any], base_run: dict[str, Any]) -> str | None:
    with_installs = _dependency_install_spans(with_run)
    base_installs = _dependency_install_spans(base_run)
    with_install = _longest_span(with_installs)
    if not with_install:
        return None
    with_install_total_seconds = _install_total_seconds(with_installs)
    base_install_seconds = _install_total_seconds(base_installs)
    if with_install_total_seconds < 60 or with_install_total_seconds <= base_install_seconds + 60:
        return None
    with_install_output = "\n".join(str(span.get("output") or "") for span in with_installs)
    package_examples = _dependency_package_examples(with_install_output)
    accelerator_packages = _accelerator_dependency_packages(with_install_output)
    base_install = _longest_span(base_installs)
    with_stack_evidence = _install_stack_evidence(with_installs)
    base_stack_evidence = _install_stack_evidence(base_installs)
    with_label = _run_label(with_run, "With skills")
    base_label = _run_label(base_run, "No skills baseline")
    install_reason = (
        "- **Why the install is longer**: "
        f"{_install_path_evidence_phrase(with_label, with_installs, with_stack_evidence)}; "
        f"{_install_path_evidence_phrase(base_label, base_installs, base_stack_evidence)}."
    )
    base_cpu_only = _install_cpu_only_evidence_display(base_installs)
    with_cpu_only = _install_cpu_only_evidence_display(with_installs)
    if base_cpu_only and not with_cpu_only:
        install_reason += (
            f" In this run the baseline used {base_cpu_only}, "
            "which avoids larger accelerator-capable framework packages; the with-skills install logs "
            "did not show CPU-only wheel/index evidence."
        )
    elif "accelerator-capable dependency stack" in with_stack_evidence:
        install_reason += " The with-skills install logs show accelerator-capable framework packages."

    lines = [
        "**Dependency install path differed**",
        "",
        "| Run | Install time | Install scope | Stack evidence | Installer | Representative command |",
        "|---|---:|---|---|---|---|",
        (
            f"| With skills | {fmt_seconds_with_unit(with_install_total_seconds)} | "
            f"{markdown_cell(_install_strategy_summary(with_installs))} | "
            f"{markdown_cell(with_stack_evidence)} | "
            f"{markdown_cell(_install_tool_label(with_install))} | "
            f"{markdown_cell(_install_command_display(with_install))} |"
        ),
        (
            f"| No skills baseline | {fmt_seconds_with_unit(base_install_seconds)} | "
            f"{markdown_cell(_install_strategy_summary(base_installs))} | "
            f"{markdown_cell(base_stack_evidence)} | "
            f"{markdown_cell(_install_tool_label(base_install))} | "
            f"{markdown_cell(_install_command_display(base_install))} |"
        ),
        "",
        install_reason,
    ]
    if package_examples:
        lines.append(f"- **Captured package examples**: {', '.join(package_examples)}.")
    if accelerator_packages:
        lines.append(
            "- **Accelerator dependency evidence**: with-skills install logs included "
            f"{', '.join(accelerator_packages)}; large accelerator/framework wheels can dominate install time."
        )
    if base_install:
        lines.append(
            "- **Installer difference**: "
            f"with-skills used {_install_tool_label(with_install)}, while the baseline used "
            f"{_install_tool_label(base_install)}."
        )
    if _install_network_markers_for_spans(with_installs) or _install_network_markers_for_spans(base_installs):
        lines.append(
            "- **Network/download evidence**: "
            f"{_install_network_marker_display_for_spans('with-skills', with_installs)}; "
            f"{_install_network_marker_display_for_spans('baseline', base_installs)}."
        )
    runtime_note = _install_runtime_offset_note(with_run, base_run)
    if runtime_note:
        lines.append(runtime_note)
    baseline_followup_note = ""
    if len(base_installs) > 1:
        baseline_followup_note = (
            f"- **Baseline follow-up installs**: baseline ran {len(base_installs)} install commands; after its "
            "longest install, later requirements installs mostly reused already-installed packages when the log "
            "reported them as already satisfied."
        )
        lines.append(baseline_followup_note)
    return "\n".join(lines)


def _dependency_install_retry_reason(spans: list[dict[str, Any]]) -> str:
    failed = [span for span in spans if command_failed(span)]
    reason_parts = []
    if failed:
        failed_index = spans.index(failed[0])
        succeeded_later = any(command_succeeded(span) for span in spans[failed_index + 1 :])
        reason = f"first failed: {command_error_summary(str(failed[0].get('output') or ''))}"
        if succeeded_later:
            reason += "; later dependency install succeeded"
        reason_parts.append(reason)
    markers = _install_network_markers_for_spans(spans)
    if markers:
        reason_parts.append(f"network/download evidence: {', '.join(markers)}")
    accelerator_packages = _accelerator_dependency_packages("\n".join(str(span.get("output") or "") for span in spans))
    if accelerator_packages:
        reason_parts.append(f"accelerator package evidence: {', '.join(accelerator_packages)}")
    if not reason_parts:
        reason_parts.append("multiple dependency install commands captured; inspect command output for intent")
    return "; ".join(reason_parts)
