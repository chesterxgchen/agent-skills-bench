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

from .._context import CodeQualitySignal, ReportContext
from .._text import fmt_number, markdown_cell
from ..evidence import RunEvidence
from ._plugin_view import MODE_LABELS, _collect_plugin_evidence, _evidence_or_legacy, _report_context

__all__ = [
    "fl_algorithm_display",
    "_status_cell",
    "generated_code_quality_table",
    "generated_code_quality_section",
    "_code_quality_assessment_map",
]


def fl_algorithm_display(run: RunEvidence | dict[str, Any], algorithm: Any = None) -> str:
    if algorithm is None:
        algorithm = _collect_plugin_evidence(run).algorithm
    info = (getattr(algorithm, "info", None)) or {}
    algo_name = info.get("algorithm") or "not captured"
    rounds = info.get("num_rounds")
    if rounds is not None:
        return f"{algo_name} ({fmt_number(rounds)} rounds)"
    return str(algo_name)


def _status_cell(status: str, evidence: str) -> str:
    return f"{status}: {evidence}" if evidence else status


def generated_code_quality_table(
    runs: dict[str, RunEvidence | dict[str, Any]], modes: list[str], ctx: ReportContext | None = None
) -> str:
    ctx = ctx or _report_context(runs, modes)
    cq = {mode: ctx.code_quality(mode) for mode in modes}
    lines = [
        "| Evidence signal | " + " | ".join(MODE_LABELS.get(mode, mode) for mode in modes) + " |",
        "|---|" + "|".join("---" for _ in modes) + "|",
        "| Overall code quality signal | " + " | ".join(markdown_cell(cq[mode].overall) for mode in modes) + " |",
    ]
    # Realized rows (label, status, evidence) are aligned across modes (same spec).
    for i, label in enumerate(row[0] for row in cq[modes[0]].rows):
        lines.append(
            f"| {markdown_cell(label)} | "
            + " | ".join(markdown_cell(_status_cell(cq[mode].rows[i][1], cq[mode].rows[i][2])) for mode in modes)
            + " |"
        )
    for i, label in enumerate(row[0] for row in cq[modes[0]].context_rows):
        lines.append(
            f"| {markdown_cell(label)} | "
            + " | ".join(markdown_cell(_status_cell("context", cq[mode].context_rows[i][2])) for mode in modes)
            + " |"
        )
    return "\n".join(lines)


def generated_code_quality_section(
    runs: dict[str, RunEvidence | dict[str, Any]], modes: list[str], ctx: ReportContext | None = None
) -> str:
    return "\n".join(
        [
            "## Generated Code Quality Signals",
            "",
            "These are evidence signals for interpreting runtime and maintenance quality. They do not change pass/fail quality gates or the winner policy.",
            "",
            generated_code_quality_table(runs, modes, ctx),
            "",
            "Dependency policy note: accelerator-capable framework installs are valid for accelerator-backed training jobs but can dominate benchmark wall time when uncached. CPU-only framework installs are faster, but they should only be treated as comparable when the benchmark is intentionally CPU-only.",
        ]
    )


def _code_quality_assessment_map(run: RunEvidence | dict[str, Any], ev: Any = None) -> dict[str, tuple[str, str]]:
    assessments = (_evidence_or_legacy(ev, run).code_quality or CodeQualitySignal()).assessments
    return {label: (status, evidence) for label, status, evidence in assessments}
