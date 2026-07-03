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

"""Evidence loader — reads a finalized result root into per-run bundles.

A neutral leaf owned by the evidence layer (Contract B): it loads what Stage 3
wrote to disk. It imports only generic substrate (``modes``, ``metric_artifacts``,
``quality_signals``, ``common``, ``reports._runs``) and **never imports the report
engine** (``benchmark_insights``) — so Contract B no longer depends on the report
product (architecture §5, Inversion 1). ``benchmark_insights`` re-imports the
names its render path still uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..agent_identity import MAX_AGENT_EVENTS_TEXT_BYTES, preferred_agent_model, resolve_agent_model
from ..common import load_json
from ..host_environment import host_os_display
from ..metric_artifacts import validation_metric_from_workspace_delta_manifest
from ..modes import BENCHMARK_RUNS
from ..quality_signals import canonical_metric_name, is_numeric_metric_value, reported_metric_payload
from ._runs import read_text


# The verbatim agent input prompt captured by the container run (prompt.txt).
# Generic for all SDKs; capped so a pathological prompt cannot bloat the report.
MAX_PROMPT_TEXT_BYTES = 64_000


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


# Agent-authored RCA reports are post-run interpretation, not Stage-3 capture:
# they ride in the loader bundle (reached via RunEvidence.raw), never as a
# typed Contract B field. Own display cap, unrelated to the prompt cap.
MAX_RCA_REPORT_TEXT_BYTES = 200_000


def _combined_rca_reports(mode_dir: Path) -> str:
    """All agent RCA reports for a run (one per investigated topic), joined."""

    rca_dir = mode_dir / "rca"
    if not rca_dir.is_dir():
        return ""
    reports = []
    for path in sorted(rca_dir.glob("rca_report*.md")):
        report = read_text(path, max_bytes=MAX_RCA_REPORT_TEXT_BYTES).strip()
        if not report:
            continue
        try:
            truncated = path.stat().st_size > MAX_RCA_REPORT_TEXT_BYTES
        except OSError:
            truncated = False
        if truncated:
            report += "\n\n_… RCA report truncated for display._"
        reports.append(report)
    return "\n\n".join(reports)


def mode_dir_for_benchmark(root: Path, mode: str) -> Path:
    legacy = root / mode
    if legacy.exists():
        return legacy

    run_plan = load_json(root / "run_plan.json", {}) or {}
    entries = (
        run_plan.get("entries") if isinstance(run_plan, dict) and isinstance(run_plan.get("entries"), list) else []
    )
    for entry in entries:
        if not isinstance(entry, dict) or str(entry.get("mode")) != mode:
            continue
        record_dir = entry.get("record_dir")
        if not record_dir:
            continue
        candidate = root / str(record_dir)
        if candidate.exists():
            return candidate

    records_root = root / "records"
    if records_root.exists():
        matches = sorted(records_root.glob(f"**/mode={mode}"))
        if len(matches) == 1:
            return matches[0]
    return legacy


def final_record_path(root: Path, mode: str) -> Path:
    mode_dir = mode_dir_for_benchmark(root, mode)
    benchmark_record = mode_dir / "benchmark_record.json"
    if benchmark_record.exists():
        return benchmark_record
    return mode_dir / "records" / f"{mode}_record.json"


def sanitized_validation_metric(metric: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(metric, dict) or not metric.get("name"):
        return metric if isinstance(metric, dict) else {}
    name = canonical_metric_name(metric.get("name"))
    entries = [entry for entry in metric.get("reported_value_entries") or [] if isinstance(entry, dict)]
    if not entries and is_numeric_metric_value(metric.get("value")):
        entry: dict[str, Any] = {"value": metric["value"]}
        if metric.get("summary_value_label"):
            entry["label"] = metric["summary_value_label"]
        entries = [entry]
    sanitized = reported_metric_payload(name, entries)
    sanitized["source"] = metric.get("source") or sanitized.get("source")
    if metric.get("source_path"):
        sanitized["source_path"] = metric.get("source_path")
    return sanitized


def validation_metric_from_record(record: dict[str, Any]) -> dict[str, Any]:
    metric = record.get("validation_metric")
    if isinstance(metric, dict) and metric.get("name"):
        return sanitized_validation_metric(metric)
    metric = record.get("artifact_validation_metric")
    if isinstance(metric, dict) and metric.get("name"):
        return sanitized_validation_metric(metric)
    metric = record.get("reported_validation_metric")
    if isinstance(metric, dict) and metric.get("name"):
        return sanitized_validation_metric(metric)
    quality = record.get("quality_signals")
    if isinstance(quality, dict):
        signal = quality.get("job_guidance_primary_validation_metric") or quality.get(
            "readme_primary_validation_metric"
        )
        if isinstance(signal, dict):
            metric = signal.get("reported_validation_metric")
            if isinstance(metric, dict) and metric.get("name"):
                return sanitized_validation_metric(metric)
    return {}


def expected_validation_metric_name(record: dict[str, Any]) -> str | None:
    policy = record.get("validation_metric_policy")
    if isinstance(policy, dict) and policy.get("expected_primary_metric"):
        return str(policy.get("expected_primary_metric"))
    quality = record.get("quality_signals")
    if isinstance(quality, dict):
        for key in (
            "artifact_validation_metric",
            "job_guidance_primary_validation_metric",
            "readme_primary_validation_metric",
        ):
            signal = quality.get(key)
            if not isinstance(signal, dict):
                continue
            if signal.get("expected_primary_metric"):
                return str(signal.get("expected_primary_metric"))
            metric = signal.get("reported_validation_metric")
            if isinstance(metric, dict) and metric.get("name"):
                return str(metric.get("name"))
    for key in ("validation_metric", "artifact_validation_metric", "reported_validation_metric"):
        metric = record.get(key)
        if isinstance(metric, dict) and metric.get("name"):
            return str(metric.get("name"))
    return None


def filter_mode_console(console_text: str, mode: str) -> str:
    if not console_text:
        return ""
    prefix = f"[{mode}] "
    lines = []
    for line in console_text.splitlines():
        if line.startswith(prefix):
            lines.append(line[len(prefix) :])
    return "\n".join(lines)


def collect_benchmark_runs(root: Path) -> dict[str, dict[str, Any]]:
    console_text = read_text(root / "console_output.log")
    runs: dict[str, dict[str, Any]] = {}
    run_plan = load_json(root / "run_plan.json", {}) or {}
    host_environment = load_json(root / "host_environment.json", {}) or {}
    if not isinstance(host_environment, dict):
        host_environment = {}
    root_host_os = host_os_display(host_environment)
    scenario_name = run_plan.get("scenario_name") if isinstance(run_plan, dict) else None
    entries = (
        run_plan.get("entries") if isinstance(run_plan, dict) and isinstance(run_plan.get("entries"), list) else []
    )
    for spec in BENCHMARK_RUNS:
        mode = spec.mode
        run_plan_entry = next(
            (entry for entry in entries if isinstance(entry, dict) and str(entry.get("mode")) == mode),
            {},
        )
        mode_dir = mode_dir_for_benchmark(root, mode)
        mode_console_text = read_text(root / f"{mode}.console.log") or filter_mode_console(console_text, mode)
        summary = load_json(mode_dir / "run_summary.json", {}) if mode_dir.exists() else {}
        record = load_json(final_record_path(root, mode), {}) if mode_dir.exists() else {}
        workspace_delta_path = mode_dir / "workspace_delta_manifest.json"
        workspace_delta = load_json(workspace_delta_path, {}) if mode_dir.exists() else {}
        skills_list = load_json(mode_dir / "skills_list.json", {}) if mode_dir.exists() else {}
        if not isinstance(summary, dict):
            summary = {}
        if not isinstance(record, dict):
            record = {}
        if not isinstance(workspace_delta, dict):
            workspace_delta = {}
        if not isinstance(skills_list, dict):
            skills_list = {}
        run_host_environment = (
            summary.get("host_environment")
            if isinstance(summary.get("host_environment"), dict)
            else host_environment
        )
        run_host_os = first_non_empty(summary.get("host_os"), run_plan_entry.get("host_os"), root_host_os)
        if run_host_os:
            summary = {**summary, "host_os": run_host_os}
        if run_host_environment:
            summary = {**summary, "host_environment": run_host_environment}
        agent = first_non_empty(summary.get("agent"), record.get("agent"), run_plan_entry.get("agent"))
        configured_agent_model, configured_model_source = preferred_agent_model(
            (summary.get("agent_model"), summary.get("model_source")),
            (record.get("agent_model"), record.get("model_source")),
            (run_plan_entry.get("agent_model"), run_plan_entry.get("model_source")),
        )
        agent_events_text = (
            read_text(mode_dir / "agent_events.jsonl", max_bytes=MAX_AGENT_EVENTS_TEXT_BYTES)
            if mode_dir.exists()
            else ""
        )
        agent_stderr_text = read_text(mode_dir / "agent_stderr.txt", max_bytes=MAX_AGENT_EVENTS_TEXT_BYTES) if mode_dir.exists() else ""
        agent_model, model_source = resolve_agent_model(
            configured_agent_model,
            configured_model_source,
            agent_events_text,
            agent_stderr_text,
        )
        record_metric = validation_metric_from_record(record)
        expected_metric = expected_validation_metric_name(record)
        artifact_metric = validation_metric_from_workspace_delta_manifest(
            workspace_delta,
            workspace_delta_path,
            expected_metric,
        )
        runs[mode] = {
            "available": mode_dir.exists(),
            "mode_dir": mode_dir,
            "mode": mode,
            "label": spec.label,
            "skills": "with skills" if spec.skills_enabled else "without skills",
            "run_plan_entry": dict(run_plan_entry) if isinstance(run_plan_entry, dict) else {},
            "scenario_name": first_non_empty(run_plan_entry.get("scenario_name"), scenario_name),
            "job_name": run_plan_entry.get("job_name"),
            "job_slug": run_plan_entry.get("job_slug"),
            "job_path": run_plan_entry.get("job_path"),
            "agent": agent,
            "agent_model": agent_model,
            "model_source": model_source,
            "run": summary,
            "record": record,
            "container_exit": load_json(mode_dir / "container_exit_code.json", {}) if mode_dir.exists() else {},
            "usage": load_json(mode_dir / "agent_usage.json", {}) if mode_dir.exists() else {},
            "activity": load_json(mode_dir / "agent_activity.json", {}) if mode_dir.exists() else {},
            "workspace_delta": workspace_delta,
            "skills_list": skills_list,
            "prompt_text": read_text(mode_dir / "prompt.txt", max_bytes=MAX_PROMPT_TEXT_BYTES) if mode_dir.exists() else "",
            "prompt_metadata": load_json(mode_dir / "prompt_metadata.json", {}) if mode_dir.exists() else {},
            "rca_report": _combined_rca_reports(mode_dir),
            "runtime_image": load_json(mode_dir / "runtime_image.json", {}) if mode_dir.exists() else {},
            "agent_last_message": read_text(mode_dir / "agent_last_message.txt") if mode_dir.exists() else "",
            "agent_stderr": agent_stderr_text,
            "agent_events_text": agent_events_text,
            "console_text": mode_console_text,
            "validation_metric": artifact_metric or record_metric,
        }
    return runs
