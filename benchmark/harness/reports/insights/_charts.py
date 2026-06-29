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

import html
from typing import Any

from ...modes import BENCHMARK_RUNS, mode_names
from ...quality_signals import canonical_metric_name
from .._context import CodeQualitySignal, ReportContext, StructureView
from .._events import as_number, fmt_seconds, run_activity
from .._text import fmt_number, markdown_cell
from ..evidence import RunEvidence
from ._metrics import (
    comparable_metric_name,
    metric_display,
    metric_name_for_runs,
    metric_names_for_runs,
    metric_value,
    run_quality_issues,
)
from ._plugin_view import (
    MODE_LABELS,
    _as_run_evidence,
    _evidence_or_legacy,
    _report_context,
    _result_term,
    fmt_short,
    run_summary,
)
from ._spans import _dependency_install_total_seconds, _elapsed_excluding_dependency_install

__all__ = [
    "interpretation_section",
    "mixed_metric_note",
    "chart_number",
    "chart_value_display",
    "benchmark_chart_metrics",
    "chart_mode_label",
    "comparison_scorecard",
    "embedded_bar_chart",
    "outcome_metrics_table",
]


def interpretation_section(runs: dict[str, RunEvidence], modes: list[str], ctx: ReportContext | None = None) -> str:
    ctx = ctx or _report_context(runs, modes)
    failed_quality = [
        runs[mode].label or mode for mode in modes if run_quality_issues(runs[mode], ctx.evidence.get(mode))
    ]
    metric_name = comparable_metric_name(runs) or metric_name_for_runs(runs)
    lines = [
        "## Interpretation",
        "",
    ]
    if failed_quality:
        lines.append(
            "Quality comparison is incomplete because these runs failed a benchmark quality gate: "
            + ", ".join(failed_quality)
            + "."
        )
        lines.append(
            f"For this artifact, the missing/partial signal is `{metric_name}` reporting, not necessarily a Docker or Python execution crash."
        )
    else:
        lines.append("All available runs passed the benchmark quality gates captured by this report.")
    if len(modes) == 2:
        left, right = modes
        left_time = as_number(run_summary(runs[left]).get("elapsed_seconds"))
        right_time = as_number(run_summary(runs[right]).get("elapsed_seconds"))
        left_tokens = as_number(run_summary(runs[left]).get("token_count"))
        right_tokens = as_number(run_summary(runs[right]).get("token_count"))
        if left_time is not None and right_time is not None:
            faster_mode = left if left_time <= right_time else right
            slower_mode = right if left_time <= right_time else left
            faster = runs[faster_mode].label or faster_mode
            time_delta = abs((right_time or 0) - (left_time or 0))
            lines.append(
                f"Runtime winner by wall-clock seconds: {faster} ({fmt_number(min(left_time, right_time))}s vs {fmt_number(max(left_time, right_time))}s, delta {fmt_number(time_delta)}s)."
            )
        if left_tokens is not None and right_tokens is not None:
            cheaper_mode = left if left_tokens <= right_tokens else right
            cheaper = runs[cheaper_mode].label or cheaper_mode
            token_delta = abs((right_tokens or 0) - (left_tokens or 0))
            lines.append(
                f"Token-use winner: {cheaper} ({fmt_short(min(left_tokens, right_tokens))} vs {fmt_short(max(left_tokens, right_tokens))}, delta {fmt_short(token_delta)})."
            )
    lines.append(
        f"Read cost winners only after checking the quality gates; a cheaper run that does not report the requested {_result_term(ctx)}result is not a successful benchmark winner."
    )
    return "\n".join(lines)


def mixed_metric_note(runs: dict[str, RunEvidence]) -> str:
    parts = []
    for run in runs.values():
        metric = run.validation_metric
        name = canonical_metric_name(metric.get("name")) if isinstance(metric, dict) else ""
        if name:
            parts.append(f"{run.label or run.mode}: {name}")
    return "; ".join(parts)


def chart_number(value: Any, kind: str) -> float | None:
    number = as_number(value)
    if number is None:
        return None
    if kind == "percent":
        return max(0.0, min(1.0, number))
    return max(0.0, number)


def chart_value_display(value: Any, kind: str) -> str:
    if value is None:
        return "NA"
    if kind == "seconds":
        return fmt_seconds(value)
    if kind == "short":
        return fmt_short(value)
    if kind == "percent":
        number = as_number(value)
        return "NA" if number is None else f"{number * 100:.0f}%"
    return fmt_number(value)


def benchmark_chart_metrics(
    runs: dict[str, RunEvidence], metric_name: str | None, ctx: ReportContext | None = None
) -> list[dict[str, Any]]:
    # Each value callable takes (run, ev) where ev is the per-run PluginEvidence.
    return [
        {
            "label": "Total time seconds",
            "kind": "seconds",
            "value": lambda run, ev: run_summary(run).get("elapsed_seconds"),
        },
        {
            "label": "Runtime seconds",
            "kind": "seconds",
            "value": lambda run, ev: _elapsed_excluding_dependency_install(run),
        },
        {
            "label": "Dependency install",
            "kind": "seconds",
            "value": lambda run, ev: _dependency_install_total_seconds(run),
        },
        {
            "label": "Total tokens",
            "kind": "short",
            "value": lambda run, ev: run_summary(run).get("token_count"),
        },
        {
            "label": "Commands",
            "kind": "number",
            "value": lambda run, ev: run_activity(run.raw).get("command_count"),
        },
        {
            "label": "Structure score",
            "kind": "percent",
            "value": lambda run, ev: (_evidence_or_legacy(ev, run).structure_view or StructureView()).score,
        },
        {
            "label": "Code quality",
            "kind": "percent",
            "value": lambda run, ev: (_evidence_or_legacy(ev, run).code_quality or CodeQualitySignal()).score,
        },
        {
            "label": f"Metrics ({metric_name or 'result'})",
            "kind": "number",
            "value": lambda run, ev: metric_value(run, metric_name, ev),
        },
    ]


def chart_mode_label(mode: str, run: RunEvidence) -> str:
    if mode == "without_skills":
        return "No skills"
    if mode == "with_skills":
        return "With skills"
    return str(run.label or MODE_LABELS.get(mode, mode))


def _scorecard_delta(left: Any, right: Any, kind: str) -> str:
    left_number = chart_number(left, kind)
    right_number = chart_number(right, kind)
    if left_number is None or right_number is None:
        return "missing"
    if kind == "percent":
        return f"{(right_number - left_number) * 100:+.0f} pts"
    if left_number == 0:
        if right_number == 0:
            return "0"
        return "new"
    delta = right_number - left_number
    percent = delta / left_number * 100
    if abs(percent) >= 10:
        return f"{percent:+.0f}%"
    if kind == "seconds":
        return f"{delta:+.0f}s"
    if kind == "short":
        return fmt_short(delta)
    return fmt_number(delta)


def comparison_scorecard(runs: dict[str, RunEvidence], ctx: ReportContext | None = None) -> str:
    modes = list(runs)
    if not modes:
        return ""
    ctx = ctx or _report_context(runs, modes)
    comparable_name = comparable_metric_name(runs)
    metric_name = comparable_name or metric_name_for_runs(runs)
    metrics = benchmark_chart_metrics(runs, metric_name, ctx)
    labels = [markdown_cell(chart_mode_label(mode, runs[mode])) for mode in modes]
    lines = [
        "### Comparison Scorecard",
        "",
        "Quick view of the same evidence shown in the chart below.",
        "",
        "| Metric | " + " | ".join(labels) + " | Delta |",
        "|---|" + "|".join("---" for _ in modes) + "|---|",
    ]
    for item in metrics:
        label = markdown_cell(str(item["label"]))
        values = [item["value"](runs[mode], ctx.evidence.get(mode)) for mode in modes]
        displays = [markdown_cell(chart_value_display(value, item["kind"])) for value in values]
        delta = "NA"
        if len(values) == 2:
            delta = _scorecard_delta(values[0], values[1], item["kind"])
        lines.append(f"| {label} | " + " | ".join(displays) + f" | {markdown_cell(delta)} |")
    return "\n".join(lines)


def embedded_bar_chart(runs: dict[str, RunEvidence], ctx: ReportContext | None = None) -> str:
    metric_name = comparable_metric_name(runs)
    if metric_name is None and metric_names_for_runs(runs):
        note = markdown_cell(mixed_metric_note(runs))
        return f"<section><h3>Metrics (mixed validation metrics)</h3><p>Not comparable: {note}</p></section>"
    modes = list(runs)
    ctx = ctx or _report_context(runs, modes)
    metrics = benchmark_chart_metrics(runs, metric_name, ctx)
    width = 1180
    margin_x = 32
    top = 104
    panel_gap_x = 24
    panel_gap_y = 52
    panel_columns = 4 if len(metrics) > 4 else max(1, len(metrics))
    panel_rows = (len(metrics) + panel_columns - 1) // panel_columns
    panel_w = (width - margin_x * 2 - panel_gap_x * (panel_columns - 1)) / panel_columns
    panel_h = 250
    chart_h = 145
    height = top + panel_rows * panel_h + max(0, panel_rows - 1) * panel_gap_y + 72
    bar_gap = 24 if len(modes) <= 2 else 12
    available_bar_w = max(40.0, panel_w - 44)
    bar_w = max(18.0, min(38.0, (available_bar_w - bar_gap * max(0, len(modes) - 1)) / max(1, len(modes))))
    colors = {
        "without_skills": "#16a34a",
        "with_skills": "#2563eb",
    }
    fallback_colors = ("#2563eb", "#16a34a", "#7c3aed", "#f97316")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="32" y="35" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">Run comparison</text>',
        '<text x="32" y="58" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">Metrics are mode-local. Missing scalar results are shown as NA instead of drawing a numeric bar.</text>',
    ]
    for metric_index, item in enumerate(metrics):
        row = metric_index // panel_columns
        column = metric_index % panel_columns
        x0 = margin_x + column * (panel_w + panel_gap_x)
        panel_top = top + row * (panel_h + panel_gap_y)
        axis_y = panel_top + panel_h - 28
        title = html.escape(str(item["label"]))
        values = [chart_number(item["value"](runs[mode], ctx.evidence.get(mode)), item["kind"]) for mode in modes]
        numeric_values = [value for value in values if value is not None]
        maximum = max(numeric_values) if numeric_values else 1.0
        if maximum == 0:
            maximum = 1.0
        bar_group_w = len(modes) * bar_w + max(0, len(modes) - 1) * bar_gap
        bar_start_x = x0 + max(14.0, (panel_w - bar_group_w) / 2)
        lines.extend(
            [
                f'<text x="{x0:.1f}" y="{panel_top:.1f}" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#111827">{title}</text>',
                f'<line x1="{x0:.1f}" y1="{axis_y}" x2="{x0 + panel_w:.1f}" y2="{axis_y}" stroke="#d1d5db" stroke-width="1"/>',
                f'<line x1="{x0:.1f}" y1="{axis_y - chart_h}" x2="{x0:.1f}" y2="{axis_y}" stroke="#d1d5db" stroke-width="1"/>',
            ]
        )
        for bar_index, mode in enumerate(modes):
            run = runs[mode]
            value = item["value"](run, ctx.evidence.get(mode))
            numeric_value = values[bar_index]
            bx = bar_start_x + bar_index * (bar_w + bar_gap)
            run_label = html.escape(chart_mode_label(mode, run))
            color = colors.get(mode, fallback_colors[bar_index % len(fallback_colors)])
            if numeric_value is None:
                lines.extend(
                    [
                        f'<rect x="{bx:.1f}" y="{axis_y - 24}" width="{bar_w:.1f}" height="20" fill="#e5e7eb" rx="3"/>',
                        f'<text x="{bx + bar_w / 2:.1f}" y="{axis_y - 9}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="700" fill="#4b5563">NA</text>',
                    ]
                )
            else:
                height_px = max(4.0, numeric_value / maximum * chart_h)
                by = axis_y - height_px
                value_text = html.escape(chart_value_display(value, item["kind"]))
                lines.extend(
                    [
                        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{height_px:.1f}" fill="{color}" rx="3"/>',
                        f'<text x="{bx + bar_w / 2:.1f}" y="{by - 7:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">{value_text}</text>',
                    ]
                )
            lines.append(
                f'<text x="{bx + bar_w / 2:.1f}" y="{axis_y + 19}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">{run_label}</text>'
            )
    legend_x = 32
    legend_y = height - 38
    for index, mode in enumerate(modes):
        run = runs[mode]
        color = colors.get(mode, fallback_colors[index % len(fallback_colors)])
        x = legend_x + index * 220
        label = html.escape(str(run.label or MODE_LABELS.get(mode, mode)))
        lines.extend(
            [
                f'<rect x="{x}" y="{legend_y}" width="14" height="14" fill="{color}" rx="2"/>',
                f'<text x="{x + 22}" y="{legend_y + 12}" font-family="Arial, sans-serif" font-size="13" fill="#111827">{label}</text>',
            ]
        )
    lines.append("</svg>")
    return "\n".join(lines)


def outcome_metrics_table(
    runs: dict[str, RunEvidence | dict[str, Any]], modes: list[str] | None = None, ctx: ReportContext | None = None
) -> str:
    modes = modes or mode_names(BENCHMARK_RUNS)
    ctx = ctx or _report_context(runs, modes)
    comparable_name = comparable_metric_name(runs)
    metric_name = comparable_name or metric_name_for_runs(runs)
    labels = [
        markdown_cell(_as_run_evidence(runs.get(mode, {})).label or MODE_LABELS.get(mode, mode)) for mode in modes
    ]
    values = [
        metric_display(_as_run_evidence(runs.get(mode, {})), comparable_name, ctx.evidence.get(mode)) for mode in modes
    ]
    return "\n".join(
        [
            "| Metric | " + " | ".join(labels) + " |",
            "|---|" + "|".join("---" for _ in modes) + "|",
            f"| Metrics ({markdown_cell(metric_name)}) | "
            + " | ".join(markdown_cell(value) for value in values)
            + " |",
        ]
    )
