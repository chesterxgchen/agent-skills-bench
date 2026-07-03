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
    truncate,
    unsupported_model_message,
)
from ._loader import sanitized_validation_metric  # noqa: F401
from ._loader import collect_benchmark_runs
from ._runs import combined_text, manifest_paths, run_record, run_workspace_delta, unique_paths  # noqa: F401
from ._skill_usage import (
    shared_skill_usage_display,
    skill_availability_display,
    skill_inspection_display,
    skill_usage_display,
)
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
from .insights._charts import comparison_scorecard, embedded_bar_chart, interpretation_section, outcome_metrics_table
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
    run_host_os_display,
    run_identity_summary,
    run_identity_table,
    source_input_protection_display,
    structure_correctness_table,
    structure_trees_section,
)
from .insights._why import *  # noqa: F401,F403
from .insights._why import cost_comparison_section, why_section


def _compact_overall_status(status: str) -> str:
    if status.startswith("needs review"):
        return "needs review"
    if status.startswith("completed with metric mismatch"):
        return "warn"
    if status.startswith("failed"):
        return "failed"
    if status.startswith("passed"):
        return "passed"
    if status.startswith("missing"):
        return "missing"
    return status or "NA"


def _compact_result_gate(outcome: str) -> str:
    if outcome.startswith("pass:"):
        return "pass"
    if outcome.startswith("fail:"):
        return "fail"
    if outcome.startswith("warn:"):
        return "warn"
    if outcome.startswith("partial:"):
        return "partial"
    return outcome or "NA"


def _first_run_value(runs: dict[str, RunEvidence], *keys: str) -> str:
    for run in runs.values():
        raw = run.raw if isinstance(run.raw, dict) else {}
        for key in keys:
            value = raw.get(key)
            if value:
                return str(value)
    return ""


def _framework_evidence_text(value: str) -> str:
    text = str(value or "")
    text = re.sub(r'"(?:skills|slash_commands)"\s*:\s*\[[^\]]*\]', " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"/[^\s\"'`]*(?:\.codex|\.claude)/skills/[^\s\"'`]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[\w.-]*skill[\w.-]*\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bnvflare-convert-[a-z0-9_-]+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bslash_commands?\b", " ", text, flags=re.IGNORECASE)
    return text


def _infer_framework_from_text(*values: str) -> str:
    text = " ".join(_framework_evidence_text(value) for value in values if value).lower()
    if not text:
        return ""
    text = re.sub(r"\b(?:not|without|no)\s+(?:pytorch[\s_-]*)?lightning\b", " ", text)
    if re.search(r"\bplain[\s_-]*pytorch\b", text):
        return "PyTorch"
    if "lightning" in text or "pytorch_lightning" in text:
        return "Lightning"
    if "pytorch" in text or re.search(r"(^|[^a-z])torch([^a-z]|$)", text):
        return "PyTorch"
    return ""


def _run_job_name(run: RunEvidence) -> str:
    raw = run.raw if isinstance(run.raw, dict) else {}
    job_name = raw.get("job_name") or raw.get("job_slug")
    if job_name:
        return str(job_name)
    if run.mode_dir:
        for part in run.mode_dir.parts:
            if part.startswith("job="):
                return part.split("=", 1)[1].replace("_", "-")
    return ""


def _run_framework_display(run: RunEvidence) -> str:
    raw = run.raw if isinstance(run.raw, dict) else {}
    target_framework = _infer_framework_from_text(
        str(raw.get("framework") or ""),
        str(raw.get("job_name") or ""),
        str(raw.get("job_slug") or ""),
        str(raw.get("job_path") or ""),
        run.agent_last_message,
    )
    return f"{target_framework} target" if target_framework else ""


def _run_declared_framework_display(run: RunEvidence) -> str:
    raw = run.raw if isinstance(run.raw, dict) else {}
    target_framework = _infer_framework_from_text(
        str(raw.get("framework") or ""),
        str(raw.get("job_name") or ""),
        str(raw.get("job_slug") or ""),
        str(raw.get("job_path") or ""),
    )
    return f"{target_framework} target" if target_framework else ""


def _skill_name_from_mapping(mapping: dict[str, Any]) -> str:
    for key in ("observed_skill_name", "skill_name", "skill"):
        value = mapping.get(key)
        if value:
            return str(value).strip()
    for nested_key, nested_value_key in (
        ("skill_discovery", "selected_skill"),
        ("event_identity_inference", "skill"),
    ):
        nested = mapping.get(nested_key)
        if isinstance(nested, dict):
            value = nested.get(nested_value_key)
            if value:
                return str(value).strip()
    return ""


def _run_skills_enabled(run: RunEvidence) -> bool:
    summary = run.summary if isinstance(run.summary, dict) else {}
    record = run.record if isinstance(run.record, dict) else {}
    raw = run.raw if isinstance(run.raw, dict) else {}
    return (
        any(bool(source.get("skills_enabled")) for source in (summary, record, raw))
        or raw.get("skills") == "with skills"
    )


def _run_skill_available_display(run: RunEvidence) -> str:
    return skill_availability_display(run.skills_list, skills_enabled=_run_skills_enabled(run))


def _run_skill_display(run: RunEvidence) -> str:
    summary = run.summary if isinstance(run.summary, dict) else {}
    record = run.record if isinstance(run.record, dict) else {}
    raw = run.raw if isinstance(run.raw, dict) else {}
    observed_skill_name = ""
    for source in (summary, record, raw):
        skill_name = _skill_name_from_mapping(source)
        if skill_name:
            observed_skill_name = skill_name
            break
    return skill_usage_display(
        events_text=str(raw.get("agent_events_text") or ""),
        observed_skill_name=observed_skill_name,
        skills_enabled=_run_skills_enabled(run),
    )


def _run_skill_inspection_display(run: RunEvidence) -> str:
    raw = run.raw if isinstance(run.raw, dict) else {}
    return skill_inspection_display(str(raw.get("agent_events_text") or ""))


def _run_shared_skill_display(run: RunEvidence) -> str:
    raw = run.raw if isinstance(run.raw, dict) else {}
    return shared_skill_usage_display(str(raw.get("agent_events_text") or ""))


def _first_run_framework(runs: dict[str, RunEvidence]) -> str:
    for run in runs.values():
        framework = _run_declared_framework_display(run).removesuffix(" target")
        if framework:
            return framework
    for run in runs.values():
        framework = _run_framework_display(run).removesuffix(" target")
        if framework:
            return framework
    return ""


def _first_run_job_name(runs: dict[str, RunEvidence]) -> str:
    for run in runs.values():
        job_name = _run_job_name(run)
        if job_name:
            return job_name
    return ""


def _benchmark_target_section(runs: dict[str, RunEvidence]) -> list[str]:
    job_name = _first_run_value(runs, "job_name") or _first_run_value(runs, "job_slug") or _first_run_job_name(runs)
    job_path = _first_run_value(runs, "job_path")
    scenario_name = _first_run_value(runs, "scenario_name")
    framework = _benchmark_target_framework_display(runs)
    if not any((job_name, job_path, framework)):
        return []
    return [
        "### Benchmark Target",
        "",
        "| Job | Framework | Scenario | Job path |",
        "|---|---|---|---|",
        f"| {markdown_cell(job_name or 'NA')} | {markdown_cell(framework or 'NA')} | "
        f"{markdown_cell(scenario_name or 'NA')} | {markdown_cell(job_path or 'NA')} |",
    ]


def _benchmark_target_framework_display(runs: dict[str, RunEvidence]) -> str:
    job_name = _first_run_value(runs, "job_name") or _first_run_value(runs, "job_slug") or _first_run_job_name(runs)
    job_path = _first_run_value(runs, "job_path")
    target_framework = _first_run_framework(runs) or _infer_framework_from_text(
        _first_run_value(runs, "framework"),
        job_name,
        _first_run_value(runs, "job_slug"),
        job_path,
    )
    return f"{target_framework} target" if target_framework else ""


_MAX_PROMPT_DISPLAY_CHARS = 4_000


def _prompt_fence(text: str) -> list[str]:
    body = text.strip()
    if len(body) > _MAX_PROMPT_DISPLAY_CHARS:
        body = body[:_MAX_PROMPT_DISPLAY_CHARS].rstrip() + "\n… (truncated for display)"
    # A prompt containing backtick runs must not close the fence early: the
    # fence is always one backtick longer than the longest run in the body.
    longest_backtick_run = max((len(match) for match in re.findall(r"`+", body)), default=0)
    fence = "`" * max(4, longest_backtick_run + 1)
    return [f"{fence}text", body, fence]


def _benchmark_input_section(runs: dict[str, RunEvidence], modes: list[str]) -> list[str]:
    captured = [(mode, runs[mode]) for mode in modes if runs[mode].available and runs[mode].prompt_text.strip()]
    if not captured:
        return []
    lines = [
        "## Benchmark Input",
        "",
        "The verbatim prompt given to the agent (captured `prompt.txt`; the harness does not inject mode, path, or skill instructions).",
        "",
        "| Run | Prompt SHA-256 | Bytes |",
        "|---|---|---|",
    ]
    for _mode, run in captured:
        meta = run.prompt_metadata or {}
        digest = str(meta.get("prompt_sha256") or run.summary.get("prompt_hash") or "")
        size = meta.get("prompt_bytes")
        size_cell = str(size) if isinstance(size, int) else str(len(run.prompt_text.encode("utf-8")))
        lines.append(f"| {markdown_cell(run.label)} | {markdown_cell(digest[:16] or 'NA')} | {size_cell} |")
    lines.append("")
    texts = {run.prompt_text.strip() for _mode, run in captured}
    if len(texts) == 1:
        lines.extend(_prompt_fence(captured[0][1].prompt_text))
    else:
        for _mode, run in captured:
            lines.extend([f"**{run.label}**", ""])
            lines.extend(_prompt_fence(run.prompt_text))
            lines.append("")
        lines.pop()
    return lines


def _run_status_detail_lines(label: str, overall: str, job_status: str, outcome: str) -> list[str]:
    details: list[tuple[str, str]] = []
    if _compact_overall_status(overall) != overall:
        details.append(("Overall", truncate(overall, 180)))
    if job_status and not job_status.startswith("completed"):
        details.append(("Job", truncate(job_status, 180)))
    if _compact_result_gate(outcome) != "pass":
        details.append(("Result gate", truncate(outcome, 180)))
    if not details:
        return []
    lines = [f"**{label}**"]
    lines.extend(f"- {name}: {detail}" for name, detail in details)
    return lines


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
    target_framework = _benchmark_target_framework_display(runs)
    status_lines = [
        "### Run Status",
        "",
        "| Run | Overall | Job | Result gate | Metric |",
        "|---|---|---|---|---|",
    ]
    status_detail_lines = [
        "### Run Status Details",
        "",
    ]
    context_lines = [
        "### Run Context",
        "",
        f"| Run | Job | Framework | Agent/model | Host OS | {markdown_cell(algorithm_label)} | Captured generated artifacts |",
        "|---|---|---|---|---|---|---|",
    ]
    skill_lines = [
        "### Skill Evidence",
        "",
        "| Run | Skills available | Skills inspected | Skills applied/used | Shared refs read |",
        "|---|---|---|---|---|",
    ]
    for mode in modes:
        run = runs[mode]
        ev = ctx.evidence.get(mode)
        job_exec = ctx.job_execution(mode)
        agent_model = f"agent={run.agent or 'NA'}, model={run.agent_model or 'NA'}"
        job_status = job_exec.status
        if job_exec.status_reason:
            job_status = f"{job_status}: {job_exec.status_reason}"
        overall_status = human_readable_status(run, ev)
        result_gate = benchmark_outcome(run, ev)
        metric_status = run_result_metric_status(run, ev)
        status_lines.append(
            f"| {markdown_cell(run.label or mode)} | {markdown_cell(_compact_overall_status(overall_status))} | "
            f"{markdown_cell(job_exec.status or 'NA')} | {markdown_cell(_compact_result_gate(result_gate))} | "
            f"{markdown_cell(truncate(metric_status, 80))} |"
        )
        detail = _run_status_detail_lines(run.label or mode, overall_status, job_status or "", result_gate)
        if detail:
            if len(status_detail_lines) > 2:
                status_detail_lines.append("")
            status_detail_lines.extend(detail)
        context_lines.append(
            f"| {markdown_cell(run.label or mode)} | "
            f"{markdown_cell(_run_job_name(run) or 'NA')} | "
            f"{markdown_cell(_run_declared_framework_display(run) or target_framework or 'NA')} | "
            f"{markdown_cell(agent_model)} | "
            f"{markdown_cell(run_host_os_display(run))} | "
            f"{markdown_cell(fl_algorithm_display(run, ctx.algorithm(mode)))} | "
            f"{markdown_cell(artifact_summary(run))} |"
        )
        skill_lines.append(
            f"| {markdown_cell(run.label or mode)} | "
            f"{markdown_cell(_run_skill_available_display(run))} | "
            f"{markdown_cell(_run_skill_inspection_display(run))} | "
            f"{markdown_cell(_run_skill_display(run))} | "
            f"{markdown_cell(_run_shared_skill_display(run))} |"
        )
    target_lines = _benchmark_target_section(runs)
    summary_lines = ["## Executive Summary", ""]
    if target_lines:
        summary_lines.extend([*target_lines, ""])
    summary_lines.extend(status_lines)
    if len(status_detail_lines) > 2:
        summary_lines.extend(["", *status_detail_lines])
    summary_lines.extend(["", *context_lines])
    summary_lines.extend(["", *skill_lines])
    return "\n".join(summary_lines)


def _structure_score_value(plugin: Any, run: RunEvidence | dict[str, Any]) -> float | None:
    try:
        score = plugin.score_structure(_as_run_evidence(run)).score
    except Exception:
        score = None
    return float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else None


def _plan_entry_run_bundle(root: Path, entry: dict[str, Any]) -> dict[str, Any] | None:
    """A minimal per-run bundle from one ``run_plan.json`` entry's record dir."""

    from ..common import load_json

    record_dir = entry.get("record_dir")
    mode = str(entry.get("mode") or "")
    if not record_dir or not mode:
        return None
    run_dir = root / str(record_dir)
    record = load_json(run_dir / "benchmark_record.json", None)
    if not isinstance(record, dict):
        record = load_json(run_dir / "records" / f"{mode}_record.json", {}) or {}
    return {
        "available": run_dir.exists(),
        "mode": mode,
        "mode_dir": run_dir,
        "run": load_json(run_dir / "run_summary.json", {}) or {},
        "record": record if isinstance(record, dict) else {},
        "workspace_delta": load_json(run_dir / "workspace_delta_manifest.json", {}) or {},
    }


def persist_quality_summary(root: Path, runs: dict[str, RunEvidence | dict[str, Any]]) -> dict[str, float]:
    """Write per-mode SDK quality scores to a host-owned root sidecar.

    Structure scoring is SDK-specific, but downstream SDK-agnostic tooling (the
    RCA loop, which never imports SDK plugins) needs the number to gate on a
    regression. This persists it once, at report time, from the resolved plugin
    — outside the container-writable mode dirs. Besides the per-mode
    ``structure_score`` map, it writes ``structure_score_by_run`` keyed by
    ``run_plan.json`` run_id, so the RCA gate stays correct in result roots
    holding several runs per mode (where the per-mode map only reflects the
    default run). Returns the ``structure_score`` map written ({} when the
    plugin supplies none, e.g. the null plugin)."""

    from ..common import load_json, write_json
    from ..sdks.report_registry import resolve_from_result_root

    plugin = resolve_from_result_root(root)
    scores: dict[str, float] = {}
    for mode in mode_names(BENCHMARK_RUNS):
        run = runs.get(mode)
        if run is None:
            continue
        score = _structure_score_value(plugin, run)
        if score is not None:
            scores[mode] = score
    run_plan = load_json(root / "run_plan.json", {}) or {}
    entries = run_plan.get("entries") if isinstance(run_plan, dict) else None
    scores_by_run: dict[str, float] = {}
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or not entry.get("run_id"):
            continue
        bundle = _plan_entry_run_bundle(root, entry)
        score = _structure_score_value(plugin, bundle) if bundle is not None else None
        if score is not None:
            scores_by_run[str(entry["run_id"])] = score
    payload: dict[str, Any] = {}
    if scores:
        payload["structure_score"] = scores
    if scores_by_run:
        payload["structure_score_by_run"] = scores_by_run
    if payload:
        write_json(root / "quality_summary.json", payload)
    return scores


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
    benchmark_input = _benchmark_input_section(runs, modes)
    if benchmark_input:
        generic_blocks.append(("benchmark_input", [*benchmark_input, ""]))
    generic_blocks.extend(
        [
            (
                "metrics",
                [
                    "## Metrics",
                    "",
                    comparison_scorecard({mode: runs[mode] for mode in modes}, ctx),
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
