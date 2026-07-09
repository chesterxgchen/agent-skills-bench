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

"""Stage 4 — the report plugin interface (interpretation CODE, architecture §6).

The plugin runs only at report time, read-only over evidence (Contract B), and
returns a derived ``PluginEvidence`` sidecar. It never mutates ``RunEvidence``.

The interface carries the SDK domain vocabulary (``participant_model``,
``assess_metric``) so the generic engine stays domain-neutral. ``sections()`` is
the FUTURE composition hook (§6) — a deferred stub only this milestone; no
composition/anchor system is built here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..reports._context import (
    AlgorithmSignal,
    CodeQualitySignal,
    CommandFailureSignal,
    JobExecutionSignal,
    StructureView,
)
from ..reports.evidence import ComparisonEvidence, RunEvidence

# --- minimal signal types referenced by §4.2 -------------------------------


@dataclass(frozen=True)
class ParticipantModel:
    """SDK vocabulary for participants and aggregation (§6).

    For FL: ``participant`` = site/client, ``aggregate`` = server/FL-level. A
    null/flat model leaves these ``None``. ``result_term`` is the result-domain
    qualifier the engine prepends to result labels (e.g. NVFLARE's ``"FL"`` ->
    "FL result quality gate"); ``None`` yields the neutral "result quality gate".
    """

    participant: str | None = None
    aggregate: str | None = None
    result_term: str | None = None
    # The SDK's runtime-execution atom (e.g. NVFLARE's "simulator"); the engine owns
    # the grammar (``job or {atom}`` / ``job/{atom}`` / ``{atom or "job"}``). ``None``
    # for a flat SDK with no distinct runtime concept.
    execution_noun_short: str | None = None
    # The SDK's execution-ACTIVITY noun (the "-tion" form, e.g. NVFLARE's "simulation"):
    # the running of the job as an activity, distinct from the runtime tool/atom above.
    # The engine owns the grammar; ``None`` -> the engine's neutral "run".
    execution_noun: str | None = None


@dataclass(frozen=True)
class StructureSignal:
    """Structure-correctness signal (e.g. required-file score)."""

    score: float | None = None


@dataclass(frozen=True)
class MetricAssessment:
    """Assessment of an expected/reported validation metric.

    ``value_label`` is the SDK's label for the reported scalar (read verbatim from
    the captured metric, e.g. NVFLARE's ``summary_value_label``); ``gate_phrase`` is
    the SDK's "what a good result looks like" wording used by the quality gate. Both
    are ``None`` for a flat/neutral SDK, where the engine falls back to neutral copy.
    """

    name: str | None = None
    reported: bool = False
    value: float | None = None
    value_label: str | None = None
    gate_phrase: str | None = None
    # When true, the generic report engine treats ``value`` as the authoritative
    # result scalar and does not fall back to ``RunEvidence.validation_metric.value``.
    # SDKs use this when raw metric payloads may include non-authoritative self-report.
    value_authoritative: bool = False
    # The SDK's term for the single summary scalar (NVFLARE "FL-level scalar"); used
    # in partial/missing-metric prose. None -> engine's neutral "single result scalar".
    scalar_term: str | None = None
    # Some tasks use a non-scalar result artifact as the required output. For those,
    # the absence of a scalar validation metric is expected and should not render as
    # "missing".
    scalar_required: bool = True
    # SDK-owned path/label for a non-scalar result artifact, when one satisfied the
    # task result gate.
    result_artifact: str | None = None


@dataclass(frozen=True)
class SdkActivitySignal:
    """SDK-specific runtime-activity detection (e.g. simulator invoked)."""

    detected: bool = False
    detail: str | None = None


@dataclass(frozen=True)
class NarrativeFragment:
    """A piece of explanatory prose merged into the generic Why/Cost shells."""

    text: str
    anchor: str | None = None


@dataclass(frozen=True)
class ReportSection:
    """A whole report section contributed by a plugin (§6 composition).

    The engine owns the generic section skeleton; a plugin's ``sections()`` returns
    these to be MERGED into it. Insert-only (v1): a plugin cannot replace or remove a
    generic block — the skeleton stays authoritative. Composition is deterministic:
    sections sort by ``(anchor, placement, order, id)`` and each renders as
    ``title`` + blank line + ``body``, inserted relative to the named ``anchor`` block.

    - ``id``: stable identity (tests/diagnostics/dedup/future migration), not the title.
    - ``anchor``: a named generic block id (e.g. ``"exec_summary"``), or ``"end"`` for a
      section with no adjacent generic slot. An unknown anchor is appended at the end.
    - ``placement``: ``"after"`` | ``"before"`` the anchored block.
    - ``order``: tie-break when several sections share an anchor/placement.
    """

    id: str
    title: str
    body: str
    anchor: str
    placement: str = "after"
    order: int = 0


@dataclass(frozen=True)
class PluginEvidence:
    """Derived sidecar, paired with a ``RunEvidence`` at report time (§6).

    NEVER stored on ``RunEvidence`` — Contract B is captured-only (§5).
    """

    structure: StructureSignal | None = None
    sdk_activity: SdkActivitySignal | None = None
    metric: MetricAssessment | None = None
    job_execution: JobExecutionSignal | None = None
    algorithm: AlgorithmSignal | None = None
    structure_view: StructureView | None = None
    code_quality: CodeQualitySignal | None = None
    command_failure: CommandFailureSignal | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


class ReportPlugin(ABC):
    """SDK-specific interpretation, resolved by captured id (§4.2)."""

    # --- current milestone: helpers used by the existing report ---

    @abstractmethod
    def collect(self, run: RunEvidence) -> PluginEvidence:
        """Derive the per-run sidecar (read-only over captured evidence)."""

    @abstractmethod
    def participant_model(self) -> ParticipantModel:
        """Return the SDK participant/aggregation vocabulary."""

    @abstractmethod
    def assess_metric(self, run: RunEvidence, expected: Any) -> MetricAssessment:
        """Assess the expected/reported validation metric for a run."""

    @abstractmethod
    def score_structure(self, run: RunEvidence) -> StructureSignal:
        """Score generated-code structure correctness."""

    @abstractmethod
    def detect_sdk_activity(self, run: RunEvidence) -> SdkActivitySignal:
        """Detect SDK-specific runtime activity (e.g. simulator)."""

    @abstractmethod
    def explain(
        self,
        cmp: ComparisonEvidence,
        plugin: Mapping[str, PluginEvidence],
    ) -> list[NarrativeFragment]:
        """Return narrative fragments merged into the generic Why/Cost shells."""

    # --- TEMPORARY section-copy bridge (Inversion 2; absorbed by sections() in
    # Phase E) -------------------------------------------------------------------

    def metric_log_patterns(self) -> tuple:
        """SDK-specific regexes that mark a log line as a reported metric (§7.2).

        The generic engine recognizes generic-ML ``name=value`` metric lines on its
        own; an SDK supplies any extra domain patterns (e.g. NVFLARE's aggregated/
        global/server validation lines) so that FL-aggregation interpretation lives
        behind the plugin, not on the generic report path. Empty for a flat SDK.
        """

        return ()

    def code_quality_criteria(self, run: RunEvidence) -> list[dict[str, str]]:
        """Generated-code-quality criteria (``{"key", "description"}``) an
        evaluation agent should judge the captured code against.

        Sourced from the SDK's evaluation rules so it stays in sync with the
        report rows. Empty for a plugin without code-quality criteria; those
        SDKs simply get no agent evaluation.
        """

        return []

    def observed_metric_evidence(self, run: RunEvidence) -> str:
        """SDK-specific observed metric evidence that should not satisfy the result gate.

        This is for domain logs that carry useful metric context but are not authoritative
        final scalar artifacts. The generic engine displays the text as supporting
        evidence; ``assess_metric`` remains the gate for successful result metrics.
        """

        return ""

    def section_copy(self, key: str) -> str | None:
        """SDK-specific copy for a known key embedded INSIDE a generic section, or ``None``.

        A bounded-vocabulary mechanism (NOT a temporary bridge) for the SDK strings that
        live within generic sections and so cannot move to ``sections()`` — the
        exec-summary algorithm-row label and the job-run intro. The engine falls back to
        neutral default copy per key when this returns ``None``. Whole SDK-specific
        sections are owned by ``sections()`` (E1b); only those whole-section copy keys
        (e.g. the FL algorithm section title/intro) were retired from this bridge.
        """

        return None

    # --- whole-section composition (§6; E1b) ---

    def sections(
        self,
        cmp: ComparisonEvidence,
        plugin: Mapping[str, PluginEvidence],
    ) -> list[ReportSection]:
        """Plugin-contributed whole report sections, merged into the generic skeleton.

        Returns ``ReportSection``s the engine inserts at their named anchors
        (insert-only; see ``ReportSection``). A plugin builds each body from its OWN
        evidence (the ``PluginEvidence`` sidecar + ``ComparisonEvidence``) and neutral
        formatting leaves — never by importing the report engine (DoD#4). Default: no
        plugin sections (the flat/null SDK contributes none).
        """

        return []


class NullReportPlugin(ReportPlugin):
    """Conceptual default (§4.2): flat participant model, empty signals."""

    def collect(self, run: RunEvidence) -> PluginEvidence:
        return PluginEvidence()

    def participant_model(self) -> ParticipantModel:
        return ParticipantModel()

    def assess_metric(self, run: RunEvidence, expected: Any) -> MetricAssessment:
        # Coherent neutral vocabulary (no SDK assumptions): the engine's neutral
        # fallback mirrors this, so a flat SDK still renders readable copy.
        return MetricAssessment(gate_phrase="result metric available")

    def score_structure(self, run: RunEvidence) -> StructureSignal:
        return StructureSignal()

    def detect_sdk_activity(self, run: RunEvidence) -> SdkActivitySignal:
        return SdkActivitySignal()

    def explain(
        self,
        cmp: ComparisonEvidence,
        plugin: Mapping[str, PluginEvidence],
    ) -> list[NarrativeFragment]:
        return []
