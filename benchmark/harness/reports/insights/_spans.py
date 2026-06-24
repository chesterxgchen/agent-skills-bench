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
    is_dependency_install_command,
)
from .._text import fmt_number, strip_ansi
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
        for span in agent_command_spans(run.raw)
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
        for span in agent_command_spans(run.raw)
        if (as_number(span.get("duration_seconds")) or 0) >= min_seconds
        and str(span.get("status") or "") in {"completed", "failed"}
    ]
    return sorted(spans, key=lambda item: as_number(item.get("duration_seconds")) or 0, reverse=True)[:limit]


def _format_command_span_list(label: str, spans: list[dict[str, Any]]) -> str:
    if not spans:
        return f"{label}: no timed command spans >=30s captured"
    return f"{label}: " + "; ".join(_format_command_span(span) for span in spans)


def _dependency_install_spans(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        span for span in agent_command_spans(run.raw) if is_dependency_install_command(str(span.get("command") or ""))
    ]


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


def _dependency_install_slowdown_note(with_run: dict[str, Any], base_run: dict[str, Any]) -> str | None:
    with_installs = _dependency_install_spans(with_run)
    base_installs = _dependency_install_spans(base_run)
    with_install = _longest_span(with_installs)
    if not with_install:
        return None
    with_install_total_seconds = sum(as_number(span.get("duration_seconds")) or 0 for span in with_installs)
    base_install_seconds = sum(as_number(span.get("duration_seconds")) or 0 for span in base_installs)
    if with_install_total_seconds < 60 or with_install_total_seconds <= base_install_seconds + 60:
        return None
    with_install_output = "\n".join(str(span.get("output") or "") for span in with_installs)
    package_examples = _dependency_package_examples(with_install_output)
    package_note = f"; downloaded packages included {', '.join(package_examples)}" if package_examples else ""
    accelerator_packages = _accelerator_dependency_packages(with_install_output)
    accelerator_note = ""
    if accelerator_packages:
        accelerator_note = (
            " Accelerator dependency evidence: with-skills install logs included "
            f"{', '.join(accelerator_packages)}; large accelerator/framework wheels can dominate install time."
        )
    base_install = _longest_span(base_installs)
    base_note = (
        f"; baseline longest install was {_format_command_span(base_install)}"
        if base_install
        else "; baseline had no captured dependency install command"
    )
    installer_note = ""
    if base_install:
        installer_note = (
            f" Installer form differed: with-skills used {_install_tool_label(with_install)}; "
            f"baseline longest install used {_install_tool_label(base_install)}."
        )
    network_note = ""
    if _install_network_markers_for_spans(with_installs) or _install_network_markers_for_spans(base_installs):
        network_note = (
            " Network/download evidence: "
            f"{_install_network_marker_display_for_spans('with-skills', with_installs)}; "
            f"{_install_network_marker_display_for_spans('baseline', base_installs)}."
        )
    baseline_followup_note = ""
    if len(base_installs) > 1:
        baseline_followup_note = (
            f" Baseline ran {len(base_installs)} install commands; after its longest install, later requirements installs "
            "mostly reused already-installed packages when the log reported them as already satisfied."
        )
    return (
        "- **Dependency install path differed**: "
        f"with-skills spent {_install_total_display(with_installs)} "
        f"({_install_strategy_summary(with_installs)}), while the baseline spent "
        f"{_install_total_display(base_installs)} ({_install_strategy_summary(base_installs)}). "
        f"The longest with-skills install was {_format_command_span(with_install)}{package_note}{base_note}."
        f"{installer_note}{accelerator_note}{network_note}{baseline_followup_note}"
    )


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
