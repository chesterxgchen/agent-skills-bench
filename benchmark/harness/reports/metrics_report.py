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

"""Generate record-driven benchmark metrics reports."""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any

from ..agent_identity import MAX_AGENT_EVENTS_TEXT_BYTES, resolve_agent_model
from ..common import flatten_numbers, load_json, write_json
from ..metric_artifacts import validation_metric_from_workspace_delta_manifest
from ..modes import BENCHMARK_RUNS, NO_SKILLS_MODE, WITH_SKILLS_MODE
from ..quality_signals import canonical_metric_name, is_plausible_metric_value
from ._context import ReportContext
from ._loader import (
    filter_mode_console,
    final_record_path,
    first_non_empty,
    mode_dir_for_benchmark,
    expected_validation_metric_name,
    validation_metric_from_record,
)
from ._runs import read_text
from .benchmark_insights import (
    _report_context,
    _run_skill_available_display,
    _run_skill_inspection_display,
    _run_shared_skill_display,
    _run_skill_display,
    activity_insights_table,
    embedded_bar_chart,
    comparable_metric_name,
    human_readable_status,
    interpretation_section,
    metric_value,
    markdown_cell,
    metric_name_for_runs,
    outcome_metrics_table,
    run_analysis,
    status_summary,
    why_section,
)
from .evidence import RunEvidence, _run_evidence_from_bundle


def _insights_context(root: Path, insight_runs: dict[str, Any]) -> ReportContext:
    """Resolve the captured plugin and build the render context (Inversion 3).

    Identical resolution to ``benchmark_report``: the plugin comes from the
    result root's captured identity, so the metrics report renders through the
    same plugin sidecar instead of an implicit default.
    """

    from ..sdks.report_registry import resolve_from_result_root

    modes = [spec.mode for spec in BENCHMARK_RUNS if spec.mode in insight_runs]
    return _report_context(insight_runs, modes, resolve_from_result_root(root))


def collect_runs(root: Path) -> list[dict[str, Any]]:
    runs = []
    run_plan = load_json(root / "run_plan.json", {}) or {}
    entries = (
        run_plan.get("entries") if isinstance(run_plan, dict) and isinstance(run_plan.get("entries"), list) else []
    )
    for spec in BENCHMARK_RUNS:
        run_plan_entry = next(
            (entry for entry in entries if isinstance(entry, dict) and str(entry.get("mode")) == spec.mode),
            {},
        )
        mode_dir = mode_dir_for_benchmark(root, spec.mode)
        summary = load_json(mode_dir / "run_summary.json", {}) if mode_dir.exists() else {}
        record = load_json(final_record_path(root, spec.mode), {}) if mode_dir.exists() else {}
        activity = load_json(mode_dir / "agent_activity.json", {}) if mode_dir.exists() else {}
        usage = load_json(mode_dir / "agent_usage.json", {}) if mode_dir.exists() else {}
        runtime_image = load_json(mode_dir / "runtime_image.json", {}) if mode_dir.exists() else {}
        workspace_delta_path = mode_dir / "workspace_delta_manifest.json"
        workspace_delta = load_json(workspace_delta_path, {}) if mode_dir.exists() else {}
        skills_list = load_json(mode_dir / "skills_list.json", {}) if mode_dir.exists() else {}
        if not isinstance(summary, dict):
            summary = {}
        if not isinstance(record, dict):
            record = {}
        if not isinstance(activity, dict):
            activity = {}
        if not isinstance(usage, dict):
            usage = {}
        if not isinstance(runtime_image, dict):
            runtime_image = {}
        if not isinstance(workspace_delta, dict):
            workspace_delta = {}
        if not isinstance(skills_list, dict):
            skills_list = {}
        expected_metric = expected_validation_metric_name(record)
        record_metric = validation_metric_from_record(record)
        artifact_metric = validation_metric_from_workspace_delta_manifest(
            workspace_delta,
            workspace_delta_path,
            expected_metric,
        )
        validation_metric = artifact_metric or record_metric
        agent_events_text = (
            read_text(mode_dir / "agent_events.jsonl", max_bytes=MAX_AGENT_EVENTS_TEXT_BYTES)
            if mode_dir.exists()
            else ""
        )
        configured_agent_model = first_non_empty(
            summary.get("agent_model"),
            record.get("agent_model"),
            run_plan_entry.get("agent_model"),
        )
        configured_model_source = first_non_empty(
            summary.get("model_source"),
            record.get("model_source"),
            run_plan_entry.get("model_source"),
        )
        agent_model, model_source = resolve_agent_model(
            configured_agent_model,
            configured_model_source,
            agent_events_text,
        )
        runs.append(
            {
                "mode": spec.mode,
                "label": spec.label,
                "available": mode_dir.exists(),
                "skills_enabled": spec.skills_enabled,
                "agent": first_non_empty(summary.get("agent"), record.get("agent"), run_plan_entry.get("agent")),
                "agent_model": agent_model,
                "model_source": model_source,
                "summary": summary,
                "record": record,
                "activity": activity,
                "usage": usage,
                "runtime_image": runtime_image,
                "skills_list": skills_list,
                "validation_metric": validation_metric,
                "metrics": flatten_numbers(
                    {
                        "summary": summary,
                        "record": record,
                        "activity": activity,
                        "usage": usage,
                        "validation_metric": validation_metric,
                    }
                ),
            }
        )
    return runs


def runs_by_mode_for_insights(root: Path, rows: list[dict[str, Any]]) -> dict[str, RunEvidence]:
    console_text = read_text(root / "console_output.log")
    rows_by_mode = {row["mode"]: row for row in rows if isinstance(row, dict) and isinstance(row.get("mode"), str)}
    runs: dict[str, dict[str, Any]] = {}
    for spec in BENCHMARK_RUNS:
        mode = spec.mode
        row = rows_by_mode.get(mode) or {}
        mode_dir = mode_dir_for_benchmark(root, mode)
        available = bool(row.get("available")) if "available" in row else mode_dir.exists()
        summary = row.get("summary") if isinstance(row.get("summary"), dict) else {}
        record = row.get("record") if isinstance(row.get("record"), dict) else {}
        usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
        activity = row.get("activity") if isinstance(row.get("activity"), dict) else {}
        runtime_image = row.get("runtime_image") if isinstance(row.get("runtime_image"), dict) else {}
        workspace_delta_path = mode_dir / "workspace_delta_manifest.json"
        workspace_delta = (
            row.get("workspace_delta")
            if isinstance(row.get("workspace_delta"), dict)
            else load_json(workspace_delta_path, {}) if available else {}
        )
        skills_list = (
            row.get("skills_list")
            if isinstance(row.get("skills_list"), dict)
            else load_json(mode_dir / "skills_list.json", {}) if available else {}
        )
        if not isinstance(workspace_delta, dict):
            workspace_delta = {}
        if not isinstance(skills_list, dict):
            skills_list = {}
        validation_metric = row.get("validation_metric") if isinstance(row.get("validation_metric"), dict) else {}
        if not validation_metric:
            expected_metric = expected_validation_metric_name(record)
            artifact_metric = validation_metric_from_workspace_delta_manifest(
                workspace_delta if isinstance(workspace_delta, dict) else {},
                workspace_delta_path,
                expected_metric,
            )
            validation_metric = artifact_metric or validation_metric_from_record(record)
        mode_console_text = read_text(root / f"{mode}.console.log") or filter_mode_console(console_text, mode)
        agent_events_text = (
            read_text(mode_dir / "agent_events.jsonl", max_bytes=MAX_AGENT_EVENTS_TEXT_BYTES) if available else ""
        )
        configured_agent_model = first_non_empty(
            row.get("agent_model"),
            summary.get("agent_model"),
            record.get("agent_model"),
        )
        configured_model_source = first_non_empty(
            row.get("model_source"),
            summary.get("model_source"),
            record.get("model_source"),
        )
        agent_model, model_source = resolve_agent_model(
            configured_agent_model,
            configured_model_source,
            agent_events_text,
        )
        runs[mode] = {
            "available": available,
            "mode": mode,
            "label": row.get("label") or spec.label,
            "mode_dir": mode_dir,
            "skills": "with skills" if spec.skills_enabled else "without skills",
            "agent": first_non_empty(row.get("agent"), summary.get("agent"), record.get("agent")),
            "agent_model": agent_model,
            "model_source": model_source,
            "run": summary,
            "record": record,
            "container_exit": load_json(mode_dir / "container_exit_code.json", {}) if available else {},
            "usage": usage,
            "activity": activity,
            "workspace_delta": workspace_delta,
            "skills_list": skills_list,
            "runtime_image": runtime_image,
            "agent_last_message": read_text(mode_dir / "agent_last_message.txt") if available else "",
            "agent_stderr": read_text(mode_dir / "agent_stderr.txt") if available else "",
            "agent_events_text": agent_events_text,
            "console_text": mode_console_text,
            "validation_metric": validation_metric or validation_metric_from_record(record),
        }
    # Render from the typed Contract B spine (Inversion 1): the shared
    # benchmark_insights helpers receive RunEvidence, same as benchmark_report.
    return {mode: _run_evidence_from_bundle(bundle) for mode, bundle in runs.items()}


def _row_validation_metric_value(row: dict[str, Any], metric_name: str) -> Any:
    metric = row.get("validation_metric")
    if not isinstance(metric, dict):
        return None
    if canonical_metric_name(metric.get("name")) != canonical_metric_name(metric_name):
        return None
    value = metric.get("value")
    return value if is_plausible_metric_value(metric_name, value) else None


def _row_comparable_metric_name(rows: list[dict[str, Any]]) -> str | None:
    names = []
    for row in rows:
        metric = row.get("validation_metric") if isinstance(row, dict) else None
        if not isinstance(metric, dict) or not metric.get("name"):
            continue
        name = canonical_metric_name(metric.get("name"))
        if name and name not in names:
            names.append(name)
    return names[0] if len(names) == 1 else None


def numeric_comparison(
    rows: list[dict[str, Any]],
    insight_runs: dict[str, RunEvidence] | None = None,
    ctx: ReportContext | None = None,
) -> dict[str, Any]:
    rows_by_mode = {row.get("mode"): row for row in rows if isinstance(row, dict)}
    without = rows_by_mode.get(NO_SKILLS_MODE)
    with_skills = rows_by_mode.get(WITH_SKILLS_MODE)
    if without is None or with_skills is None:
        return {}
    result: dict[str, Any] = {}
    for key in ("elapsed_seconds", "token_count"):
        left = without["summary"].get(key)
        right = with_skills["summary"].get(key)
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        ):
            result[f"{key}_with_skills_minus_without_skills"] = right - left
    metric_name = (
        comparable_metric_name(insight_runs) if insight_runs is not None else _row_comparable_metric_name(rows)
    )
    if metric_name:
        if insight_runs is not None:
            evidence = ctx.evidence if ctx is not None else {}
            left = metric_value(insight_runs[NO_SKILLS_MODE], metric_name, evidence.get(NO_SKILLS_MODE))
            right = metric_value(insight_runs[WITH_SKILLS_MODE], metric_name, evidence.get(WITH_SKILLS_MODE))
        else:
            left = _row_validation_metric_value(without, metric_name)
            right = _row_validation_metric_value(with_skills, metric_name)
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        ):
            result[f"validation_metric_{metric_name}_with_skills_minus_without_skills"] = right - left
    return result


def report_summary(
    root: Path,
    title: str,
    rows: list[dict[str, Any]] | None = None,
    insight_runs: dict[str, RunEvidence] | None = None,
    ctx: ReportContext | None = None,
) -> dict[str, Any]:
    rows = collect_runs(root) if rows is None else rows
    insight_runs = runs_by_mode_for_insights(root, rows) if insight_runs is None else insight_runs
    ctx = ctx if ctx is not None else _insights_context(root, insight_runs)
    metric_name = metric_name_for_runs(insight_runs)
    return {
        "title": title,
        "result_root": str(root),
        "status": status_summary(insight_runs, ctx=ctx),
        "metric_name": metric_name,
        "runs": rows,
        "comparison": numeric_comparison(rows, insight_runs, ctx),
    }


def fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def comparison_label(key: str) -> str:
    metric_prefix = "validation_metric_"
    metric_suffix = "_with_skills_minus_without_skills"
    if key.startswith(metric_prefix) and key.endswith(metric_suffix):
        metric_name = key[len(metric_prefix) : -len(metric_suffix)]
        return f"Validation metric ({metric_name}) with skills minus without skills"
    return key


def markdown_report(
    summary: dict[str, Any],
    insight_runs: dict[str, RunEvidence] | None = None,
    ctx: ReportContext | None = None,
) -> str:
    root = Path(summary["result_root"])
    insight_runs = runs_by_mode_for_insights(root, summary["runs"]) if insight_runs is None else insight_runs
    ctx = ctx if ctx is not None else _insights_context(root, insight_runs)
    lines = [
        f"# {summary['title']}",
        "",
        f"Result root: `{summary['result_root']}`",
        "",
        f"Status: {summary['status']}",
        "",
        "## Runs",
        "",
        "| Run | Agent | Model | Status | Skills available | Skills inspected | Skills applied/used | Shared refs read | Elapsed seconds | Tokens | Commands | Root cause |",
        "|---|---|---|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in summary["runs"]:
        run = insight_runs[row["mode"]]
        run_summary = row["summary"]
        activity = row["activity"]
        ev = ctx.evidence.get(row["mode"])
        root_cause = "NA" if human_readable_status(run, ev) == "passed" else run_analysis(run, ev)
        lines.append(
            f"| {markdown_cell(row['label'])} | {markdown_cell(run.agent)} | "
            f"{markdown_cell(run.agent_model)} | {markdown_cell(human_readable_status(run, ev))} | "
            f"{markdown_cell(_run_skill_available_display(run))} | "
            f"{markdown_cell(_run_skill_inspection_display(run))} | "
            f"{markdown_cell(_run_skill_display(run))} | {markdown_cell(_run_shared_skill_display(run))} | "
            f"{fmt(run_summary.get('elapsed_seconds'))} | "
            f"{fmt(run_summary.get('token_count'))} | {fmt(activity.get('command_count'))} | "
            f"{markdown_cell(root_cause)} |"
        )
    modes = [spec.mode for spec in BENCHMARK_RUNS if spec.mode in insight_runs]
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            embedded_bar_chart(insight_runs, ctx),
            "",
            outcome_metrics_table(insight_runs, modes, ctx),
        ]
    )
    lines.extend(["", "## Activity Insights", "", activity_insights_table(insight_runs, modes, ctx)])
    lines.extend(["", interpretation_section(insight_runs, modes, ctx)])
    comparison = summary.get("comparison") or {}
    if comparison:
        lines.extend(["", "## Comparison", "", "| Metric | Delta |", "|---|---:|"])
        for key, value in sorted(comparison.items()):
            lines.append(f"| {markdown_cell(comparison_label(str(key)))} | {fmt(value)} |")
    why = why_section(insight_runs, modes, ctx)
    if why:
        lines.extend(["", why])
    return "\n".join(lines) + "\n"


def html_report(
    summary: dict[str, Any],
    insight_runs: dict[str, RunEvidence] | None = None,
    ctx: ReportContext | None = None,
) -> str:
    root = Path(summary["result_root"])
    insight_runs = runs_by_mode_for_insights(root, summary["runs"]) if insight_runs is None else insight_runs
    ctx = ctx if ctx is not None else _insights_context(root, insight_runs)
    rows = []
    for row in summary["runs"]:
        run = insight_runs[row["mode"]]
        run_summary = row["summary"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(row['label'])}</td>"
            f"<td>{html.escape(fmt(run.agent))}</td>"
            f"<td>{html.escape(fmt(run.agent_model))}</td>"
            f"<td>{html.escape(human_readable_status(run, ctx.evidence.get(row['mode'])))}</td>"
            f"<td>{html.escape(_run_skill_available_display(run))}</td>"
            f"<td>{html.escape(_run_skill_inspection_display(run))}</td>"
            f"<td>{html.escape(_run_skill_display(run))}</td>"
            f"<td>{html.escape(_run_shared_skill_display(run))}</td>"
            f"<td>{html.escape(fmt(run_summary.get('elapsed_seconds')))}</td>"
            f"<td>{html.escape(fmt(run_summary.get('token_count')))}</td>"
            "</tr>"
        )
    chart = embedded_bar_chart(insight_runs, ctx)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(summary['title'])}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px; text-align: left; }}
    th {{ background: #f5f7fa; }}
    .bar-row {{ display: grid; grid-template-columns: 180px 1fr 80px; gap: 12px; align-items: center; margin: 8px 0; }}
    .bar-track {{ background: #e4e7eb; height: 14px; position: relative; }}
    .bar-fill {{ display: block; background: #2f80ed; height: 14px; }}
  </style>
</head>
<body>
  <h1>{html.escape(summary['title'])}</h1>
  <p>Result root: <code>{html.escape(summary['result_root'])}</code></p>
  <p>Status: {html.escape(summary['status'])}</p>
  <table>
    <thead><tr><th>Run</th><th>Agent</th><th>Model</th><th>Status</th><th>Skills available</th><th>Skills inspected</th><th>Skills applied/used</th><th>Shared refs read</th><th>Elapsed seconds</th><th>Tokens</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  {chart}
</body>
</html>
"""


def write_reports(root: Path, title: str) -> dict[str, Any]:
    rows = collect_runs(root)
    insight_runs = runs_by_mode_for_insights(root, rows)
    # Resolve the plugin + context once and thread it through every renderer
    # (Inversion 3): all three outputs share one captured-identity plugin.
    ctx = _insights_context(root, insight_runs)
    summary = report_summary(root, title, rows, insight_runs, ctx)
    write_json(root / "metrics_report.json", summary)
    (root / "metrics_report.md").write_text(markdown_report(summary, insight_runs, ctx), encoding="utf-8")
    (root / "metrics_report.html").write_text(html_report(summary, insight_runs, ctx), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--title", default="Agent Skills Benchmark Metrics")
    parser.add_argument(
        "--plots", action="store_true", help="accepted for compatibility; HTML report is always written"
    )
    args = parser.parse_args()
    write_reports(args.root, args.title)
    print(args.root / "metrics_report.html")


if __name__ == "__main__":
    main()
