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

"""Render-time transport + plugin-derived signal groups (routing migration).

A neutral leaf (stdlib only) that imports nothing in the package, so BOTH the
SDK plugin (``sdks/``) and the generic report engine (``reports/benchmark_insights``)
can import it without an import cycle. It carries the per-mode ``PluginEvidence``
sidecar from the single resolution point in ``benchmark_report`` to the render
helpers (architecture §6 / ROUTING_PLAN §4d) — so the engine reads neutral typed
signals instead of calling ``sdks.nvflare._logic`` by name.

Signal groups grow one routing increment at a time; each holds the SDK's derived
interpretation for a single run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class JobExecutionSignal:
    """SDK job-run interpretation for one run: did the generated job run, and why.

    ``status`` / ``status_reason`` mirror the strings the report has always shown;
    ``recovered_summary`` is the (possibly empty) note about command failures that
    a later successful run recovered from. ``successful_job_spans`` /
    ``last_successful_job_event`` are the realized job evidence the cost/why and
    metric-evidence sections read.
    """

    status: str | None = None
    status_reason: str | None = None
    recovered_summary: str | None = None
    successful_job_spans: tuple = ()
    last_successful_job_event: Mapping[str, Any] | None = None
    runtime_path: str = ""


@dataclass(frozen=True)
class AlgorithmSignal:
    """SDK FL-algorithm/workflow derived view for one run."""

    info: Mapping[str, Any] | None = None
    recipe_mismatch: Any = None


@dataclass(frozen=True)
class StructureView:
    """SDK generated-code structure derived view for one run.

    Realized file-presence so displays format without asking the SDK: ``score``
    plus the required/optional file-name vocabulary and which are present. Named
    ``StructureView`` to avoid clashing with the score-only ``StructureSignal``
    that ``score_structure`` returns (the plugin's structure-scoring hook).
    """

    score: float | None = None
    required_files: tuple = ()
    optional_files: tuple = ()
    present_required: tuple = ()
    present_optional: tuple = ()
    required_label: str = "Required converted files"
    accepted_required_folders: tuple = ()


@dataclass(frozen=True)
class CodeQualitySignal:
    """SDK generated-code-quality derived view for one run.

    ``rows`` / ``context_rows`` are **realized** (label, status, evidence) tuples
    — no callable specs cross into the engine; the renderer formats them.
    """

    overall: str | None = None
    score: float | None = None
    assessments: tuple = ()
    rows: tuple = ()
    context_rows: tuple = ()


@dataclass(frozen=True)
class CommandFailureSignal:
    """SDK command-failure derived view for one run: realized diagnostic rows.

    Each row is a dict (command, exit, recovery, root_cause, dependency); the
    renderer formats and applies any display limit.
    """

    rows: tuple = ()


@dataclass(frozen=True)
class ReportContext:
    """Immutable render-time bundle: per-mode plugin sidecar + resolved plugin.

    ``evidence`` maps mode -> ``PluginEvidence`` (typed as ``Any`` to keep this a
    neutral leaf). Accessors return an empty signal when a mode or signal is
    absent (e.g. the null plugin), so the engine never special-cases ``None``.
    """

    evidence: Mapping[str, Any]
    plugin: Any = None
    # Plugin-contributed narrative fragments for named render slots (Inversion 2 / E1):
    # anchor -> the fragment texts the active plugin returned from ``explain()``. The
    # engine renders these at fixed interior points in the Why/Cost shells; empty for a
    # flat/absent SDK. Stored as plain strings to keep this a neutral leaf.
    narratives: Mapping[str, tuple] = field(default_factory=dict)

    def narrative(self, anchor: str) -> list:
        """Fragment texts the active plugin contributed to ``anchor`` (or [])."""
        return list(self.narratives.get(anchor, ()))

    def _signal(self, mode: str, attr: str, default):
        ev = self.evidence.get(mode)
        signal = getattr(ev, attr, None) if ev is not None else None
        return signal if signal is not None else default

    def job_execution(self, mode: str) -> JobExecutionSignal:
        return self._signal(mode, "job_execution", JobExecutionSignal())

    def algorithm(self, mode: str) -> AlgorithmSignal:
        return self._signal(mode, "algorithm", AlgorithmSignal())

    def structure_view(self, mode: str) -> StructureView:
        return self._signal(mode, "structure_view", StructureView())

    def code_quality(self, mode: str) -> CodeQualitySignal:
        return self._signal(mode, "code_quality", CodeQualitySignal())

    def command_failure(self, mode: str) -> CommandFailureSignal:
        return self._signal(mode, "command_failure", CommandFailureSignal())
