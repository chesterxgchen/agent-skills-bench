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
from pathlib import Path
from typing import Any

from ...modes import BENCHMARK_RUNS, mode_names
from ...metric_artifacts import observed_metric_payloads_from_workspace_delta_manifest
from ...quality_signals import (
    canonical_metric_name,
    is_numeric_metric_value,
    is_plausible_metric_value,
    metric_value_entries,
    reported_metric_payload,
)
from .._context import AlgorithmSignal, JobExecutionSignal, ReportContext
from .._events import exit_code
from .._loader import sanitized_validation_metric
from .._runs import combined_text, run_workspace_delta
from .._text import fmt_number
from ..evidence import RunEvidence
from ._diagnostics import failure_root_cause, successful_job_evidence
from ._plugin_view import (
    MODE_LABELS,
    OBSERVED_METRIC_NAMES,
    _as_run_evidence,
    _evidence_or_legacy,
    _execution_atom,
    _execution_run_noun,
    _participant_term,
    _report_context,
    _scalar_term,
    count_map,
    metric_log_lines,
    run_source_input_delta,
    workspace_delta_has_artifact_evidence,
)

__all__ = [
    "metric_reporting_gap_evidence",
    "referenced_requirements_files",
    "path_list_contains_filename",
    "dependency_file_origin",
    "dependency_reference_notes",
    "metric_mismatch_with_reported_scalar",
    "metric_mismatch_issue",
    "result_metric_scalar_available",
    "final_response_metric_reporting_gap",
    "run_quality_issues",
    "run_status_kind",
    "human_readable_status",
    "run_analysis",
    "status_summary",
    "metric_name_for_runs",
    "metric_names_for_runs",
    "comparable_metric_name",
    "metric_value",
    "_metric_value_label",
    "metric_display",
    "metric_reported_value_count",
    "additional_metric_values_display",
    "metric_payload_display",
    "observed_metric_payloads",
    "observed_metric_evidence_display",
    "additional_or_observed_metric_values_display",
    "run_result_metric_status",
    "benchmark_outcome",
    "quality_signal",
]


def metric_reporting_gap_evidence(run: RunEvidence, ev: Any = None, ctx: Any = None) -> str:
    issues = run_quality_issues(run, ev)
    if not issues:
        return ""
    record = run.record if isinstance(run.record, dict) else {}
    expected = quality_signal(record).get("expected_primary_metric") or "target metric"
    success = successful_job_evidence(run, ev, ctx)
    if success and metric_value(run, canonical_metric_name(expected), ev) is None:
        return (
            f"Metric reporting gap: {success}, but the final response/benchmark record did not include one "
            f"aggregate `{expected}` scalar for comparison."
        )
    return ""


def referenced_requirements_files(text: str) -> list[str]:
    names = []
    for match in re.finditer(r"\bpip\s+install\s+-r\s+([A-Za-z0-9_./-]*requirements[A-Za-z0-9_.-]*\.txt)\b", text):
        name = Path(match.group(1)).name
        if name not in names:
            names.append(name)
    return names


def path_list_contains_filename(items: Any, filename: str) -> bool:
    if not isinstance(items, list):
        return False
    for item in items:
        if isinstance(item, dict) and Path(str(item.get("path") or "")).name == filename:
            return True
    return False


def dependency_file_origin(run: RunEvidence, filename: str) -> str:
    source_delta = run_source_input_delta(run)
    workspace_delta = run_workspace_delta(run.raw)
    if path_list_contains_filename(source_delta.get("final_files"), filename):
        return "original input file"
    for key, label in (
        ("workspace_added_files", "agent-generated file"),
        ("changed_files", "agent-created or modified file"),
        ("workspace_modified_files", "agent-modified original file"),
    ):
        if path_list_contains_filename(workspace_delta.get(key), filename):
            return label
    if path_list_contains_filename(workspace_delta.get("final_files"), filename):
        return "present in final agent workspace"
    return "not found in captured input or workspace manifests"


def dependency_reference_notes(run: RunEvidence) -> list[str]:
    notes = []
    for filename in referenced_requirements_files(combined_text(run.raw)):
        notes.append(f"`{filename}` provenance: {dependency_file_origin(run, filename)}.")
    return notes


def metric_mismatch_with_reported_scalar(run: RunEvidence, ev: Any = None) -> bool:
    record = run.record if isinstance(run.record, dict) else {}
    signal = quality_signal(record)
    if not signal.get("mismatch"):
        return False
    metric = run.validation_metric if isinstance(run.validation_metric, dict) else {}
    actual_name = canonical_metric_name(metric.get("name"))
    return bool(actual_name) and metric_value(run, actual_name, ev) is not None


def metric_mismatch_issue(run: RunEvidence, ev: Any = None) -> str:
    record = run.record if isinstance(run.record, dict) else {}
    signal = quality_signal(record)
    expected = signal.get("expected_primary_metric") or "target metric"
    metric = run.validation_metric if isinstance(run.validation_metric, dict) else {}
    actual_name = canonical_metric_name(metric.get("name"))
    actual = metric_display(run, actual_name or None, ev)
    return f"Metric mismatch `primary_metric_reporting`: expected `{expected}`, reported {actual}."


def result_metric_scalar_available(run: RunEvidence, metric_name: str | None = None, ev: Any = None) -> bool:
    return metric_value(run, canonical_metric_name(metric_name) if metric_name else None, ev) is not None


def _plugin_metric_quality_issue(run: RunEvidence, ev: Any = None) -> str:
    assessment = getattr(ev, "metric", None) if ev is not None else None
    if assessment is None or not getattr(assessment, "gate_phrase", None):
        return ""
    if not (
        getattr(assessment, "reported", False)
        or getattr(assessment, "name", None)
        or getattr(assessment, "scalar_term", None)
    ):
        return ""
    metric = run.validation_metric if isinstance(run.validation_metric, dict) else {}
    metric_name = canonical_metric_name(getattr(assessment, "name", None) or metric.get("name"))
    if result_metric_scalar_available(run, metric_name or None, ev):
        return ""
    scalar_term = getattr(assessment, "scalar_term", None) or "single result scalar"
    if getattr(assessment, "reported", False):
        reported = metric_name or "result metric"
        return f"Failed check `result_metric_scalar`: {reported} was reported, but no {scalar_term} value was found."
    return (
        "Failed check `result_metric_available`: "
        f"{getattr(assessment, 'gate_phrase', None)} was not satisfied; no {scalar_term} value was found."
    )


def final_response_metric_reporting_gap(run: RunEvidence, ev: Any = None) -> str:
    record = run.record if isinstance(run.record, dict) else {}
    signal = quality_signal(record)
    signal_status = str(signal.get("status") or "")
    if signal_status not in {"fail", "missing"} or metric_mismatch_with_reported_scalar(run, ev):
        return ""
    expected = signal.get("expected_primary_metric")
    metric = run.validation_metric if isinstance(run.validation_metric, dict) else {}
    metric_name = canonical_metric_name(expected or metric.get("name"))
    if not result_metric_scalar_available(run, metric_name, ev):
        return ""
    evidence = signal.get("evidence") or "final response did not satisfy the expected validation metric signal"
    metric_text = metric_display(run, metric_name, ev)
    label = _metric_value_label(run, metric_name, ev)
    if label:
        metric_text = f"{metric_text} ({label})"
    return f"Final response reporting gap: artifact/record metric is available ({metric_text}), but {evidence}"


def run_quality_issues(run: RunEvidence, ev: Any = None) -> list[str]:
    issues = []
    job_execution = getattr(ev, "job_execution", None) if ev is not None else None
    job_status = str(getattr(job_execution, "status", "") or "")
    if job_status and job_status != "completed":
        reason = str(getattr(job_execution, "status_reason", "") or "").strip()
        detail = f": {reason}" if reason else ""
        issues.append(f"Failed check `job_execution`: job status is `{job_status}`{detail}.")
    record = run.record if isinstance(run.record, dict) else {}
    signal = quality_signal(record)
    expected = signal.get("expected_primary_metric")
    signal_status = str(signal.get("status") or "")
    if signal_status in {"fail", "missing"}:
        if metric_mismatch_with_reported_scalar(run, ev):
            issues.append(metric_mismatch_issue(run, ev))
        else:
            metric = run.validation_metric if isinstance(run.validation_metric, dict) else {}
            metric_name = canonical_metric_name(expected or metric.get("name"))
            if not result_metric_scalar_available(run, metric_name, ev):
                evidence = (
                    signal.get("evidence") or "final response did not satisfy the expected validation metric signal"
                )
                issues.append(f"Failed check `primary_metric_reporting`: {evidence}")
    metric = run.validation_metric if isinstance(run.validation_metric, dict) else {}
    metric_name = canonical_metric_name(metric.get("name") or expected)
    if expected and metric_value(run, metric_name, ev) is None:
        issues.append(
            f"Failed check `fl_metric_scalar`: no {_scalar_term(ev)} value was found for expected metric `{expected}`."
        )
    plugin_metric_issue = _plugin_metric_quality_issue(run, ev)
    if plugin_metric_issue and not any("metric" in issue for issue in issues):
        issues.append(plugin_metric_issue)
    delta = record.get("workspace_delta") if isinstance(record.get("workspace_delta"), dict) else {}
    if delta and not workspace_delta_has_artifact_evidence(delta):
        issues.append(
            "Failed check `workspace_delta`: no generated workspace files, final job structure, or runtime artifacts "
            "were captured."
        )
    algorithm_mismatch = (_evidence_or_legacy(ev, run).algorithm or AlgorithmSignal()).recipe_mismatch
    if algorithm_mismatch:
        issues.append(f"Failed check `fl_algorithm_recipe_match`: {algorithm_mismatch}")
    return issues


def run_status_kind(run: RunEvidence, ev: Any = None) -> str:
    if not run.available:
        return "missing"
    code = exit_code(run.raw)
    if code not in (None, 0):
        return "failed"
    if run_quality_issues(run, ev):
        return "needs review"
    return "passed"


def human_readable_status(run: RunEvidence, ev: Any = None) -> str:
    if not run.available:
        return "missing"
    code = exit_code(run.raw)
    if code == 0:
        issues = run_quality_issues(run, ev)
        if issues:
            if metric_mismatch_with_reported_scalar(run, ev):
                return f"completed with metric mismatch ({issues[0]})"
            return f"needs review ({issues[0]})"
        return "passed"
    detail = "unknown exit" if code is None else f"container exit {code}"
    return f"failed ({detail}; {failure_root_cause(run)})"


def run_analysis(run: RunEvidence, ev: Any = None) -> str:
    if not run.available:
        return "Run artifacts are missing."
    issues = run_quality_issues(run, ev)
    if exit_code(run.raw) == 0 and issues:
        return issues[0]
    if exit_code(run.raw) == 0:
        return "No failure detected."
    return failure_root_cause(run)


def status_summary(
    runs: dict[str, RunEvidence | dict[str, Any]], modes: list[str] | None = None, ctx: ReportContext | None = None
) -> str:
    modes = modes or mode_names(BENCHMARK_RUNS)
    ctx = ctx or _report_context(runs, modes)
    parts = []
    for mode in modes:
        run = _as_run_evidence(runs.get(mode, {"available": False, "label": MODE_LABELS.get(mode, mode)}))
        parts.append(
            f"{run.label or MODE_LABELS.get(mode, mode)}: {human_readable_status(run, ctx.evidence.get(mode))}"
        )
    return "; ".join(parts)


def metric_name_for_runs(runs: dict[str, RunEvidence]) -> str:
    common_name = comparable_metric_name(runs)
    if common_name:
        return common_name
    if metric_names_for_runs(runs):
        return "mixed validation metrics"
    return "result"


def metric_names_for_runs(runs: dict[str, RunEvidence]) -> list[str]:
    names = []
    for run in runs.values():
        metric = run.validation_metric
        if isinstance(metric, dict) and metric.get("name"):
            name = canonical_metric_name(metric["name"])
            if name and name not in names:
                names.append(name)
    return names


def comparable_metric_name(runs: dict[str, RunEvidence]) -> str | None:
    names = metric_names_for_runs(runs)
    return names[0] if len(names) == 1 else None


def metric_value(run: RunEvidence, metric_name: str | None = None, ev: Any = None) -> Any:
    metric = run.validation_metric
    if not isinstance(metric, dict):
        return None
    if metric_name is not None and canonical_metric_name(metric.get("name")) != canonical_metric_name(metric_name):
        return None
    assessment = getattr(ev, "metric", None) if ev is not None else None
    if getattr(assessment, "value_authoritative", False):
        selected = getattr(assessment, "value", None)
        if is_plausible_metric_value(canonical_metric_name(metric.get("name")), selected):
            return selected
        return None
    value = metric.get("value")
    if is_plausible_metric_value(canonical_metric_name(metric.get("name")), value):
        return value
    # The plain value is absent/implausible: the summary-scalar SELECTION is SDK-owned
    # (e.g. NVFLARE's FL-summary fallback), read from the plugin sidecar's
    # ``MetricAssessment.value``. The generic engine no longer FL-selects here. ``ev``
    # is the per-run PluginEvidence; the metric-name filter above still gates the result.
    if ev is not None:
        selected = getattr(assessment, "value", None) if assessment is not None else None
        if is_plausible_metric_value(canonical_metric_name(metric.get("name")), selected):
            return selected
    return None


def _metric_value_label(run: RunEvidence, metric_name: str | None, ev: Any) -> str:
    """The active plugin's metric value-label (Inversion 2), gated to this metric.

    Reads ``ev.metric.value_label`` (the SDK's label via ``assess_metric``) and applies
    the same metric-name filter the engine used before: empty string when the run's
    metric differs from ``metric_name`` or the plugin supplies no label. The label text
    itself is SDK-owned, so a flat/neutral plugin emits nothing here.
    """

    assessment = getattr(ev, "metric", None)
    label = assessment.value_label if assessment else None
    if not label:
        return ""
    metric = run.validation_metric if isinstance(run.validation_metric, dict) else {}
    if metric_name is not None and canonical_metric_name(metric.get("name")) != canonical_metric_name(metric_name):
        return ""
    return label


def metric_display(run: RunEvidence, metric_name: str | None, ev: Any = None) -> str:
    metric = run.validation_metric
    actual_name = None
    if isinstance(metric, dict) and metric.get("name"):
        actual_name = canonical_metric_name(metric["name"])
    display_name = actual_name if metric_name is None else metric_name
    if not display_name:
        record = run.record if isinstance(run.record, dict) else {}
        display_name = quality_signal(record).get("expected_primary_metric")
    if not display_name:
        display_name = "result"
    value = metric_value(run, metric_name, ev)
    if value is None:
        return f"{display_name} NA"
    return f"{display_name} {fmt_number(value)}"


def metric_reported_value_count(metric: dict[str, Any] | None) -> int:
    if not isinstance(metric, dict):
        return 0
    values = metric.get("reported_values")
    if not isinstance(values, list):
        values = metric.get("site_values")
    if not isinstance(values, list):
        return 0
    return sum(1 for value in values if is_numeric_metric_value(value))


def additional_metric_values_display(run: RunEvidence, metric_name: str | None = None) -> str:
    metric = run.validation_metric
    if not isinstance(metric, dict):
        return "NA"
    if metric_name is not None and canonical_metric_name(metric.get("name")) != canonical_metric_name(metric_name):
        return "NA"
    values = metric.get("reported_values")
    labels = metric.get("reported_value_labels")
    if not isinstance(values, list):
        values = metric.get("site_values")
    if not isinstance(labels, list):
        labels = metric.get("site_value_labels")
    if not isinstance(values, list):
        return "NA"
    entries = []
    for index, value in enumerate(values):
        if not is_numeric_metric_value(value):
            continue
        label = labels[index] if isinstance(labels, list) and index < len(labels) else None
        label_text = str(label).strip() if label else f"value-{index + 1}"
        entries.append(f"{label_text}={fmt_number(value)}")
    if len(entries) <= 1:
        return "NA"
    return ", ".join(entries[:8]) + (f", +{len(entries) - 8} more" if len(entries) > 8 else "")


def metric_payload_display(payload: dict[str, Any]) -> str:
    name = canonical_metric_name(payload.get("name"))
    if not name:
        return "NA"
    value = payload.get("value")
    if is_numeric_metric_value(value):
        return f"{name} {fmt_number(value)}"
    values = payload.get("reported_values")
    if isinstance(values, list):
        numeric_values = [item for item in values if is_numeric_metric_value(item)]
        if numeric_values:
            rendered = ", ".join(fmt_number(item) for item in numeric_values[:4])
            suffix = f", +{len(numeric_values) - 4} more" if len(numeric_values) > 4 else ""
            return f"{name} values: {rendered}{suffix}"
    return f"{name} mentioned without numeric value"


def observed_metric_payloads(run: RunEvidence) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    seen: set[str] = set()
    metric = run.validation_metric
    if isinstance(metric, dict) and metric.get("name"):
        name = canonical_metric_name(metric.get("name"))
        if name:
            payloads.append(metric)
            seen.add(name)
    text = str(run.agent_last_message or "")
    for name in OBSERVED_METRIC_NAMES:
        canonical_name = canonical_metric_name(name)
        if canonical_name in seen:
            continue
        entries = metric_value_entries(name, text)
        if not entries:
            continue
        payload = reported_metric_payload(name, entries)
        if payload.get("reported_values"):
            payloads.append(payload)
            seen.add(canonical_name)
    if run.mode_dir is not None and isinstance(run.workspace_delta, dict):
        artifact_payloads = observed_metric_payloads_from_workspace_delta_manifest(
            run.workspace_delta,
            run.mode_dir / "workspace_delta_manifest.json",
            OBSERVED_METRIC_NAMES,
            skip_names=seen,
        )
        payloads.extend(artifact_payloads)
    return payloads


def observed_metric_evidence_display(run: RunEvidence) -> str:
    payloads = observed_metric_payloads(run)
    if not payloads:
        return "none"
    return "; ".join(metric_payload_display(payload) for payload in payloads)


def additional_or_observed_metric_values_display(
    run: RunEvidence, metric_name: str | None = None, ev: Any = None, ctx: Any = None
) -> str:
    additional = additional_metric_values_display(run, metric_name)
    if additional != "NA":
        return additional
    observed = observed_metric_evidence_display(run)
    if observed != "none":
        return observed
    last_event = (_evidence_or_legacy(ev, run).job_execution or JobExecutionSignal()).last_successful_job_event or {}
    metric_lines = metric_log_lines(str(last_event.get("output") or ""), ctx=ctx)
    participant = _participant_term(ctx)
    if metric_lines:
        return f"Final {participant} metrics=NA; log/per-{participant} evidence: " + "; ".join(metric_lines)
    simulation_refs = count_map(run, "hint_counts").get("simulation", 0)
    if simulation_refs == 0:
        return (
            f"NA (no {_execution_run_noun(ctx)} run detected; per-round/per-{participant} values "
            f"require the agent to run the {_execution_atom(ctx) or 'job'})"
        )
    return "NA"


def run_result_metric_status(run: RunEvidence, ev: Any = None) -> str:
    metric = run.validation_metric
    if not isinstance(metric, dict) or not metric.get("name"):
        return "missing"
    value = metric_value(run, canonical_metric_name(metric.get("name")), ev)
    if value is not None:
        label = _metric_value_label(run, canonical_metric_name(metric.get("name")), ev)
        suffix = f" ({label})" if label else ""
        return f"{canonical_metric_name(metric.get('name'))} {fmt_number(value)}{suffix}"
    count = metric_reported_value_count(metric)
    if count:
        return f"partial: {count} reported values, no {_scalar_term(ev)}"
    return f"missing scalar: {canonical_metric_name(metric.get('name'))} mentioned without value"


def benchmark_outcome(run: RunEvidence, ev: Any = None) -> str:
    if not run.available:
        return "fail: run artifacts missing"
    code = exit_code(run.raw)
    if code not in (None, 0):
        return f"fail: container exit {code}"
    issues = run_quality_issues(run, ev)
    if issues:
        if metric_mismatch_with_reported_scalar(run, ev):
            return "warn: scalar metric reported, but it does not match the target metric instruction"
        return "fail: " + issues[0]
    # The "good result" wording is SDK vocabulary (Inversion 2): the active plugin's
    # MetricAssessment.gate_phrase, with a neutral fallback for a flat/absent SDK.
    metric_assessment = getattr(ev, "metric", None)
    gate_phrase = metric_assessment.gate_phrase if metric_assessment and metric_assessment.gate_phrase else None
    return f"pass: {gate_phrase or 'result metric available'}"


def quality_signal(record: dict[str, Any]) -> dict[str, Any]:
    quality = record.get("quality_signals")
    if not isinstance(quality, dict):
        return {}
    signal = quality.get("job_guidance_primary_validation_metric") or quality.get("readme_primary_validation_metric")
    if not isinstance(signal, dict):
        return {}
    result = dict(signal)
    metric = result.get("reported_validation_metric")
    if isinstance(metric, dict) and metric.get("name"):
        sanitized = sanitized_validation_metric(metric)
        result["reported_validation_metric"] = sanitized
        values = sanitized.get("reported_values")
        if not isinstance(values, list):
            values = []
        has_numeric = is_numeric_metric_value(sanitized.get("value")) or any(
            is_numeric_metric_value(value) for value in values
        )
        expected = result.get("expected_primary_metric") or sanitized.get("name")
        if not has_numeric and sanitized.get("name"):
            result["status"] = "missing"
            result["metric_value_available"] = False
            result["metric_scalar_available"] = False
            result["aligned_with_job_guidance"] = False
            result["aligned_with_readme"] = False
            result["evidence"] = (
                f"Job guidance declares {expected} as the primary metric, and the final response mentioned "
                f"{sanitized.get('name')} but did not report a plausible numeric value."
            )
    return result
