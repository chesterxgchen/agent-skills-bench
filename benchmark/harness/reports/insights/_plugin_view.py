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

"""Base report-utilities + plugin-view layer (E2 split; mechanical, behavior-preserving).

The lowest insights layer: the per-run plugin-sidecar/vocabulary READERS
(``_report_context`` and the ``_result_term``/``_execution_*``/``_section_copy``/
``_scalar_term`` accessors) plus the generic base utilities every other insights module
sits on (formatters/constants re-exports, metric-log parsing, permission-denial
extraction, run helpers). It is pragmatically broad to keep the package acyclic; if it
keeps growing it can be split further. The only SDK touch is the lazy/in-body
``resolve_report_plugin`` default-resolution (relocated verbatim from the pre-split
engine) — no module-load ``insights -> sdks`` edge.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ...modes import BENCHMARK_RUNS
from .._context import ReportContext
from .._events import as_number, commands_for_run, is_dependency_install_command, run_activity
from .._runs import run_record
from .._text import _is_file_inspection_segment, _shell_command_segments, strip_ansi
from ..evidence import RunEvidence

__all__ = [
    "MODE_LABELS",
    "CONFIG_STRUCTURE_SUFFIXES",
    "TREE_SOURCE_SUFFIXES",
    "TREE_RUNTIME_SUFFIXES",
    "OBSERVED_METRIC_NAMES",
    "fmt_short",
    "fmt_percent",
    "is_file_inspection_command",
    "metric_log_lines",
    "_metric_log_patterns",
    "permission_denial_commands",
    "_collect_plugin_evidence",
    "_as_run_evidence",
    "_evidence_or_legacy",
    "_run_evidence",
    "_report_context",
    "_plugin_narratives",
    "_plugin_narrative",
    "_result_term",
    "_participant_model",
    "_execution_atom",
    "_execution_run_noun",
    "_participant_term",
    "_section_copy",
    "_scalar_term",
    "run_summary",
    "workspace_delta_has_artifact_evidence",
    "run_source_input_delta",
    "basename_count_display",
    "count_map",
    "hint_count",
    "event_type_count",
    "dependency_install_attempted",
]


MODE_LABELS = {spec.mode: spec.label for spec in BENCHMARK_RUNS}


CONFIG_STRUCTURE_SUFFIXES = (".cfg", ".ini", ".json", ".toml", ".yaml", ".yml")


TREE_SOURCE_SUFFIXES = (".py",)


TREE_RUNTIME_SUFFIXES = (".py",) + CONFIG_STRUCTURE_SUFFIXES


OBSERVED_METRIC_NAMES = ("AUROC", "accuracy", "loss", "f1")


def fmt_short(value: Any) -> str:
    number = as_number(value)
    if number is None:
        return "NA"
    abs_value = abs(number)
    sign = "-" if number < 0 else ""
    if abs_value >= 1_000_000:
        return f"{sign}{abs_value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{sign}{abs_value / 1_000:.1f}k"
    return str(int(number)) if number.is_integer() else f"{number:.1f}"


def fmt_percent(value: Any) -> str:
    number = as_number(value)
    if number is None:
        return "NA"
    return f"{number * 100:.0f}%"


def is_file_inspection_command(command: str) -> bool:
    return any(_is_file_inspection_segment(segment) for segment in _shell_command_segments(command))


def metric_log_lines(output: str, limit: int = 4, ctx: Any = None) -> list[str]:
    # Generic-ML metric lines (name=value) are recognized on the generic path; any
    # SDK-specific metric-line patterns (e.g. FL aggregated/global/server validation)
    # come from the active plugin (§7.2 D2), so FL interpretation isn't hardcoded here.
    sdk_patterns = _metric_log_patterns(ctx)
    lines = []
    for line in strip_ansi(output).splitlines():
        clean = re.sub(r"^\d{4}-\d{2}-\d{2}[^-]*-\s+(?:INFO|WARNING|ERROR)\s+-\s+", "", line).strip()
        if re.search(r"\b(?:valid|validation|test|train)_[A-Za-z0-9_/-]+\s*=\s*[0-9]+\.[0-9]+", clean):
            lines.append(clean)
        elif any(re.search(pattern, clean, flags=re.IGNORECASE) for pattern in sdk_patterns):
            lines.append(clean)
        if len(lines) >= limit:
            break
    return lines


def _metric_log_patterns(ctx: Any) -> tuple:
    """SDK-specific metric-line regexes from the active plugin (§7.2 D2), or ()."""

    plugin = getattr(ctx, "plugin", None)
    return tuple(plugin.metric_log_patterns()) if plugin is not None else ()


def permission_denial_commands(run: RunEvidence) -> list[str]:
    commands = []
    for line in str(run.agent_events_text or "").splitlines():
        try:
            payload = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        denials = payload.get("permission_denials")
        if not isinstance(denials, list):
            continue
        for denial in denials:
            if not isinstance(denial, dict):
                continue
            tool_input = denial.get("tool_input")
            if isinstance(tool_input, dict):
                command = str(tool_input.get("command") or "").strip()
                if command and command not in commands:
                    commands.append(command)
    return commands


def _collect_plugin_evidence(run: RunEvidence | dict[str, Any], plugin: Any = None):
    """Collect one run's plugin sidecar (``plugin=None`` -> resolved default).

    Since Inversion 3 the absent-id default resolves to the null plugin (NVFLARE
    is selected only by captured identity). The ``evidence`` / registry imports
    are deferred to dodge the module-load cycle (``evidence`` imports this module).
    """

    if plugin is None:
        from ...sdks.report_registry import resolve_report_plugin

        plugin = resolve_report_plugin(None)
    return plugin.collect(_as_run_evidence(run))


def _as_run_evidence(run: Any):
    """Coerce a per-run bundle OR an already-typed ``RunEvidence`` to ``RunEvidence``.

    During the B2 carrier flip the render path threads ``RunEvidence`` while some
    direct/test callers still pass the raw bundle dict; this accepts both.
    """

    from ..evidence import RunEvidence, _run_evidence_from_bundle

    return run if isinstance(run, RunEvidence) else _run_evidence_from_bundle(run)


def _evidence_or_legacy(ev: Any, run: RunEvidence | dict[str, Any]):
    """Per-run derived view for a renderer helper.

    Render path passes the identity plugin's ``PluginEvidence`` (from
    ``ReportContext``); direct/test callers omit it and get the resolved-default
    derived view (the null plugin since Inversion 3 — NVFLARE is selected only by
    captured identity). Either way the renderer reads realized signals — it never
    calls ``plugin.<method>()`` itself.
    """

    return ev if ev is not None else _collect_plugin_evidence(run)


def _run_evidence(ctx: Any, run: RunEvidence):
    """Per-run derived view for a pairwise (with/base) helper that holds ctx + run.

    Looks the run's evidence up in ``ctx`` by its mode, falling back to the
    resolved-default (null) derived view when absent — so the renderer reads
    PluginEvidence, never the plugin directly.
    """

    ev = ctx.evidence.get(run.mode) if ctx is not None else None
    return _evidence_or_legacy(ev, run)


def _report_context(
    runs: dict[str, RunEvidence | dict[str, Any]], modes: list[str], plugin: Any = None
) -> ReportContext:
    """Build the per-mode plugin sidecar once and bundle it for the render path.

    ``benchmark_report`` passes the plugin selected by the captured report
    identity (§4.2); direct callers omit it and get the resolved default (the
    null plugin since Inversion 3, not NVFLARE).
    """

    if plugin is None:
        from ...sdks.report_registry import resolve_report_plugin

        plugin = resolve_report_plugin(None)
    evidence = {mode: _collect_plugin_evidence(runs[mode], plugin) for mode in modes}
    return ReportContext(evidence=evidence, plugin=plugin, narratives=_plugin_narratives(plugin, runs, modes, evidence))


def _plugin_narratives(
    plugin: Any, runs: dict[str, RunEvidence | dict[str, Any]], modes: list[str], evidence: dict[str, Any]
) -> dict:
    """Collect the active plugin's ``explain()`` fragments, grouped by named anchor (E1).

    Fed the already-loaded runs/evidence (never re-reads the result root, per
    ROUTING_PLAN §4b). The engine renders these at fixed interior render slots; a
    flat/absent SDK contributes nothing. ``explain()`` runs read-only over a
    ``ComparisonEvidence`` built from the in-hand runs.
    """

    from ..evidence import SCHEMA_VERSION, ComparisonEvidence

    cmp = ComparisonEvidence(
        schema_version=SCHEMA_VERSION,
        runs={mode: _as_run_evidence(runs[mode]) for mode in modes},
        modes=list(modes),
        sdk_metadata={},
    )
    grouped: dict[str, list] = {}
    for fragment in plugin.explain(cmp, evidence) or ():
        anchor = getattr(fragment, "anchor", None)
        if anchor:
            grouped.setdefault(anchor, []).append(fragment.text)
    return {anchor: tuple(texts) for anchor, texts in grouped.items()}


def _plugin_narrative(ctx: Any, anchor: str) -> list:
    """The plugin fragments contributed to the named render slot ``anchor`` (E1), or []."""

    return ctx.narrative(anchor) if ctx is not None else []


def _result_term(ctx: Any) -> str:
    """The active SDK's result-domain qualifier with a trailing space (Inversion 2).

    Reads ``participant_model().result_term`` on the render path (NVFLARE -> ``"FL "``),
    so the FL adjective on result labels is SDK-owned, not an engine literal. A
    flat/absent SDK yields ``""`` -> the neutral "result quality gate".
    """

    term = getattr(_participant_model(ctx), "result_term", None)
    return f"{term} " if term else ""


def _participant_model(ctx: Any) -> Any:
    """The active plugin's ``ParticipantModel`` (SDK vocabulary), or None if absent."""

    plugin = getattr(ctx, "plugin", None)
    return plugin.participant_model() if plugin is not None else None


def _execution_atom(ctx: Any) -> str | None:
    """The active SDK's runtime-execution atom ('simulator' for NVFLARE), or None.

    The renderer owns the grammar/connective ('job or {atom}', 'job/{atom}',
    '{atom or "job"}'); the plugin supplies only the domain atom (Inversion 2). A
    flat/absent SDK yields None, so the engine falls back to the neutral "job".
    """

    return getattr(_participant_model(ctx), "execution_noun_short", None)


def _execution_run_noun(ctx: Any) -> str:
    """The active SDK's execution-ACTIVITY noun (NVFLARE "simulation"), neutral "run".

    Reads ``participant_model().execution_noun`` (the "-tion" activity form that the
    runtime-atom ``execution_noun_short`` does not cover). A flat/absent SDK yields the
    engine's neutral default "run", so the activity prose carries no FL vocabulary.
    """

    return getattr(_participant_model(ctx), "execution_noun", None) or "run"


def _participant_term(ctx: Any) -> str:
    """The active SDK's participant term (NVFLARE "site"), neutral "participant".

    Reads ``participant_model().participant`` so per-participant prose is SDK-owned;
    a flat/absent SDK yields the engine's neutral "participant".
    """

    return getattr(_participant_model(ctx), "participant", None) or "participant"


def _section_copy(ctx: Any, key: str, default: str) -> str:
    """SDK copy for ``key`` embedded inside a generic section (bounded-vocabulary bridge).

    Reads ``plugin.section_copy(key)`` on the render path (Inversion 2) and falls back
    to neutral ``default`` copy for a flat/absent SDK. This is the durable mechanism for
    SDK strings WITHIN generic sections (the exec-summary algorithm-row label, the job-run
    intro); whole SDK sections are owned by ``sections()`` (E1b), not this bridge.
    """

    plugin = getattr(ctx, "plugin", None)
    copy = plugin.section_copy(key) if plugin is not None else None
    return copy if copy else default


def _scalar_term(ev: Any) -> str:
    """The active SDK's term for the single summary scalar (Inversion 2).

    Reads ``ev.metric.scalar_term`` (NVFLARE "FL-level scalar") with a neutral
    fallback, so partial/missing-metric prose carries no FL vocabulary for a flat SDK.
    """

    assessment = getattr(ev, "metric", None)
    term = getattr(assessment, "scalar_term", None) if assessment is not None else None
    return term or "single result scalar"


def run_summary(run: RunEvidence) -> dict[str, Any]:
    summary = run.summary
    return summary if isinstance(summary, dict) else {}


def workspace_delta_has_artifact_evidence(delta: dict[str, Any]) -> bool:
    changed = as_number(delta.get("changed_file_count")) or 0
    workspace_changes = as_number(delta.get("workspace_change_count")) or 0
    runtime = as_number(delta.get("runtime_artifact_count")) or 0
    copied = as_number(delta.get("copied_file_count")) or 0
    final_structure = as_number(delta.get("final_structure_file_count")) or 0
    final_manifest = as_number(delta.get("final_file_manifest_count")) or 0
    final_structure_files = delta.get("final_structure_files")
    final_files = delta.get("final_files")
    return (
        changed > 0
        or workspace_changes > 0
        or runtime > 0
        or copied > 0
        or final_structure > 0
        or final_manifest > 0
        or (isinstance(final_structure_files, list) and bool(final_structure_files))
        or (isinstance(final_files, list) and bool(final_files))
    )


def run_source_input_delta(run: RunEvidence) -> dict[str, Any]:
    record = run_record(run.raw)
    delta = record.get("source_input_delta") if isinstance(record.get("source_input_delta"), dict) else None
    return delta if isinstance(delta, dict) else {}


def basename_count_display(paths: list[str], limit: int = 6) -> str:
    if not paths:
        return "none"
    counts: dict[str, int] = {}
    for path in paths:
        name = Path(path).name
        counts[name] = counts.get(name, 0) + 1
    entries = []
    for name in sorted(counts)[:limit]:
        count = counts[name]
        entries.append(name if count == 1 else f"{name} ({count} paths)")
    if len(counts) > limit:
        entries.append(f"+{len(counts) - limit} more")
    return ", ".join(entries)


def count_map(run: RunEvidence, key: str) -> dict[str, Any]:
    activity = run_activity(run.raw)
    value = activity.get(key)
    return value if isinstance(value, dict) else {}


def hint_count(run: RunEvidence, key: str) -> int:
    value = count_map(run, "hint_counts").get(key, 0)
    number = as_number(value)
    return int(number) if number is not None else 0


def event_type_count(run: RunEvidence, key: str) -> int:
    value = count_map(run, "event_types").get(key, 0)
    number = as_number(value)
    return int(number) if number is not None else 0


def dependency_install_attempted(run: RunEvidence) -> bool:
    for command in commands_for_run(run.raw):
        if is_dependency_install_command(command):
            return True
    return False
