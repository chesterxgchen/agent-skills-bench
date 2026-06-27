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

"""NVFLARE report plugin.

The SDK interpretation is OWNED here and reads only the neutral ``_logic`` leaf
(which never imports ``benchmark_insights`` — no import cycle). ``collect()``
eagerly derives the per-run sidecar (``PluginEvidence``) the render path consumes
through ``ReportContext`` (routing migration); ``score_structure`` and the job-
execution signal both delegate to ``_logic``. This module does NOT import the
report engine: the plugin depends on the SDK logic, not the other way around.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...modes import WITH_SKILLS_MODE
from ...quality_signals import (
    canonical_metric_name,
    is_plausible_metric_value,
    metric_value_label,
    plausible_fl_summary_entry,
)
from ...reports._context import (
    AlgorithmSignal,
    CodeQualitySignal,
    CommandFailureSignal,
    JobExecutionSignal,
    StructureView,
)
from ...reports._events import _format_command_span, _longest_span, _span_total_seconds, fmt_seconds_with_unit
from ...reports._text import _command_count_display, fmt_number, markdown_cell
from ...reports.evidence import ComparisonEvidence, RunEvidence
from ..report_plugin import (
    MetricAssessment,
    NarrativeFragment,
    ParticipantModel,
    PluginEvidence,
    ReportPlugin,
    ReportSection,
    SdkActivitySignal,
    StructureSignal,
)
from . import _logic
from ._logic import completed_job_recovered_issue_summary, job_run_status, job_run_status_reason, structure_score


class NvflareReportPlugin(ReportPlugin):
    """FL interpretation, reading the captured run bundle via the ``_logic`` leaf."""

    def collect(self, run: RunEvidence) -> PluginEvidence:
        structure_view = self._structure_view(run)
        return PluginEvidence(
            structure=StructureSignal(score=structure_view.score),  # reuse the view's score
            sdk_activity=self.detect_sdk_activity(run),
            metric=self.assess_metric(run, None),  # load-bearing: the render path reads ev.metric
            job_execution=self._job_execution(run),
            algorithm=self._algorithm(run),
            code_quality=self._code_quality(run),
            structure_view=structure_view,
            command_failure=CommandFailureSignal(rows=tuple(_logic.command_failure_rows(run.raw))),
        )

    def _structure_view(self, run: RunEvidence) -> StructureView:
        raw = run.raw
        matches = _logic.current_workspace_structure_file_matches
        return StructureView(
            score=_logic.structure_score(raw),
            required_files=_logic.REQUIRED_STRUCTURE_FILES,
            optional_files=_logic.OPTIONAL_STRUCTURE_FILES,
            present_required=tuple(f for f in _logic.REQUIRED_STRUCTURE_FILES if matches(raw, f)),
            present_optional=tuple(f for f in _logic.OPTIONAL_STRUCTURE_FILES if matches(raw, f)),
        )

    def _job_execution(self, run: RunEvidence) -> JobExecutionSignal:
        # Operates on the captured per-run bundle (run.raw) via the neutral leaf.
        raw = run.raw
        return JobExecutionSignal(
            status=job_run_status(raw),
            status_reason=job_run_status_reason(raw),
            recovered_summary=completed_job_recovered_issue_summary(raw),
            successful_job_spans=tuple(_logic._successful_job_spans(raw)),
            last_successful_job_event=_logic.last_successful_job_event(raw),
            runtime_path=_logic.job_runtime_path(_logic.longest_successful_job_span(raw)),
        )

    def _algorithm(self, run: RunEvidence) -> AlgorithmSignal:
        raw = run.raw
        return AlgorithmSignal(
            info=_logic.fl_algorithm_info(raw),
            recipe_mismatch=_logic.fl_algorithm_recipe_mismatch(raw),
        )

    def _code_quality(self, run: RunEvidence) -> CodeQualitySignal:
        raw = run.raw
        rows = tuple(_logic.generated_code_quality_assessments(raw))
        context_rows = tuple(
            (label, "context", evidence_getter(raw)) for label, evidence_getter in _logic.CODE_QUALITY_CONTEXT_ROWS
        )
        return CodeQualitySignal(
            overall=_logic.generated_code_quality_overall(raw),
            score=_logic.generated_code_quality_score(raw),
            assessments=tuple(_logic.generated_code_quality_assessments(raw)),
            rows=rows,
            context_rows=context_rows,
        )

    def participant_model(self) -> ParticipantModel:
        # FL model: participant = site/client, aggregate = server/FL-level; the
        # result-domain qualifier "FL" carries the FL wording on result labels; the
        # runtime-execution atom is the FL simulator.
        return ParticipantModel(
            participant="site",
            aggregate="server",
            result_term="FL",
            execution_noun_short="simulator",
            execution_noun="simulation",
        )

    def assess_metric(self, run: RunEvidence, expected: Any) -> MetricAssessment:
        # Delegate to the SAME source the report consumes: the bundle's
        # validation_metric (artifact_metric or record_metric), not a record-only
        # re-derivation — so routing cannot shift output.
        metric = run.validation_metric or {}
        name = metric.get("name") if isinstance(metric, dict) else None
        value = self._selected_scalar(metric if isinstance(metric, dict) else None)
        return MetricAssessment(
            name=name,
            reported=bool(name),
            value=value if isinstance(value, (int, float)) else None,
            # The SDK's label for the reported scalar, read verbatim from the captured
            # metric (Inversion 2); the engine gates it to the rendered metric name.
            value_label=metric_value_label(metric if isinstance(metric, dict) else None, None) or None,
            # The FL "what a good result looks like" wording for the quality gate
            # (Inversion 2): the engine renders ``pass: {gate_phrase}``.
            gate_phrase="scalar FL result metric available",
            # The FL term for the single summary scalar, used in partial/missing prose.
            scalar_term="FL-level scalar",
        )

    @staticmethod
    def _selected_scalar(metric: dict[str, Any] | None) -> Any:
        # SDK-owned summary-scalar SELECTION (moved out of the generic engine): the
        # plain reported ``value`` when plausible, else the last reported entry whose
        # label is an FL-summary metric label. The generic ``metric_value`` no longer
        # FL-selects; it reads this through ``MetricAssessment.value``.
        if not isinstance(metric, dict):
            return None
        name = canonical_metric_name(metric.get("name"))
        value = metric.get("value")
        if is_plausible_metric_value(name, value):
            return value
        entry = plausible_fl_summary_entry(name, metric.get("reported_value_entries"))
        return entry.get("value") if entry else None

    # Bounded-vocabulary bridge (Inversion 2): SDK copy embedded INSIDE generic sections
    # (the exec-summary algorithm-row label, the job-run intro). Whole sections are owned
    # by sections() (E1b); the algorithm_section.* keys retired there.
    _SECTION_COPY = {
        "exec_summary.algorithm_workflow_label": "FL algorithm/workflow",
        "job_run.intro": (
            "This section tracks whether the generated NVFLARE job or simulator actually ran. Agent/container "
            "exit code 0 only means the agent process finished; it does not prove the generated job executed."
        ),
    }

    # The FL Algorithm/Workflow section copy — owned by the plugin (E1b: the whole section
    # is plugin-contributed, not engine-rendered).
    _ALGORITHM_SECTION_TITLE = "## FL Algorithm / Workflow"
    _ALGORITHM_SECTION_INTRO = (
        "This section reports the FL workflow captured in generated/runtime NVFLARE server config. It is "
        "derived from artifacts such as `config_fed_server.json`; agent planning or final-message text is "
        "not counted as runtime workflow evidence."
    )
    def section_copy(self, key: str) -> str | None:
        return self._SECTION_COPY.get(key)

    def sections(self, cmp: ComparisonEvidence, plugin: Mapping[str, PluginEvidence]) -> list[ReportSection]:
        # The FL Algorithm/Workflow section, built from this plugin's own AlgorithmSignal
        # (+ neutral formatting leaves) and merged after the generic Executive Summary
        # block (E1b §6). The engine no longer renders this section.
        sections: list[ReportSection] = []
        rows = [
            "| Run | Algorithm/workflow | Recipe | Rounds | Evidence |",
            "|---|---|---|---:|---|",
        ]
        for mode in cmp.modes:
            run = cmp.runs[mode]
            evidence = plugin.get(mode)
            algorithm = evidence.algorithm if evidence is not None else None
            info = (algorithm.info if algorithm is not None else None) or {}
            rows.append(
                f"| {markdown_cell(run.label or mode)} | {markdown_cell(info.get('algorithm'))} | "
                f"{markdown_cell(info.get('recipe') or 'not captured')} | "
                f"{markdown_cell(fmt_number(info.get('num_rounds')))} | {markdown_cell(info.get('evidence'))} |"
            )
        body = "\n".join([self._ALGORITHM_SECTION_INTRO, "", *rows])
        sections.append(
            ReportSection(
                id="fl_algorithm",
                title=self._ALGORITHM_SECTION_TITLE,
                body=body,
                anchor="exec_summary",
                placement="after",
            )
        )
        return sections

    def metric_log_patterns(self) -> tuple:
        # FL-aggregation metric lines (aggregated/global/server validation): the
        # SDK-specific log interpretation the generic engine must not own (§7.2).
        return (r"\b(?:best\s+)?(?:aggregated|global|server)\s+validation\b",)

    def score_structure(self, run: RunEvidence) -> StructureSignal:
        # Owned here (step 4): the implementation lives in the neutral _logic
        # leaf, operating on the captured per-run bundle (run.raw).
        return StructureSignal(score=structure_score(run.raw))

    def detect_sdk_activity(self, run: RunEvidence) -> SdkActivitySignal:
        status = job_run_status(run.raw)
        return SdkActivitySignal(detected=status == "completed", detail=status)

    # --- SDK-logic facade: delegate to the neutral _logic leaf (byte-identical
    # to the old direct engine calls; see ROUTING_PLAN R2). ---

    def explain(
        self,
        cmp: ComparisonEvidence,
        plugin: Mapping[str, PluginEvidence],
    ) -> list[NarrativeFragment]:
        # E3: the FL runtime-path narrative is OWNED here, contributed to the
        # engine's named "why_slowdown" slot. The pair mirrors why_section: with =
        # WITH_SKILLS_MODE, base = the other mode. Reads only the in-hand evidence
        # (job_execution) and runs (never re-reads the result root).
        if len(cmp.modes) != 2 or WITH_SKILLS_MODE not in cmp.modes:
            return []
        base_mode = next(mode for mode in cmp.modes if mode != WITH_SKILLS_MODE)
        fragments = []
        result_failure_block = _logic.result_failure_root_cause_block(cmp.runs[WITH_SKILLS_MODE].raw)
        if result_failure_block:
            fragments.append(NarrativeFragment(text=result_failure_block, anchor="why_result_failure"))
        blocks = self._runtime_path_slowdown_blocks(
            with_je=(plugin.get(WITH_SKILLS_MODE) or PluginEvidence()).job_execution or JobExecutionSignal(),
            base_je=(plugin.get(base_mode) or PluginEvidence()).job_execution or JobExecutionSignal(),
            base_run=cmp.runs.get(base_mode),
        )
        fragments.extend(NarrativeFragment(text=block, anchor="why_slowdown") for block in blocks)
        return fragments

    @staticmethod
    def _runtime_path_slowdown_blocks(
        with_je: JobExecutionSignal,
        base_je: JobExecutionSignal,
        base_run: RunEvidence | None,
    ) -> list[str]:
        with_jobs = list(with_je.successful_job_spans)
        base_jobs = list(base_je.successful_job_spans)
        with_job = _longest_span(with_jobs)
        base_job = _longest_span(base_jobs)
        if not with_job:
            return []
        lines: list[str] = []
        with_path = with_je.runtime_path
        base_path = base_je.runtime_path
        if with_path or base_path:
            rows = [
                (
                    "With skills",
                    with_path or "captured job/simulator command",
                    _command_count_display(len(with_jobs)),
                    fmt_seconds_with_unit(_span_total_seconds(with_jobs)),
                    _format_command_span(with_job),
                )
            ]
            if base_job:
                rows.append(
                    (
                        "No skills baseline",
                        base_path or "captured job/simulator command",
                        _command_count_display(len(base_jobs)),
                        fmt_seconds_with_unit(_span_total_seconds(base_jobs)),
                        _format_command_span(base_job),
                    )
                )
            else:
                base_fallback = (
                    _longest_span(_logic._successful_non_install_command_spans(base_run))
                    if base_run is not None
                    else None
                )
                if base_fallback:
                    rows.append(
                        (
                            "No skills baseline",
                            "no classified successful job/simulator command",
                            "0 commands",
                            "NA",
                            f"longest successful non-install command: {_format_command_span(base_fallback)}",
                        )
                    )
                else:
                    rows.append(
                        (
                            "No skills baseline",
                            "no captured successful job/simulator command",
                            "0 commands",
                            "NA",
                            "not captured",
                        )
                    )
            table = [
                "**NVFLARE runtime path diverged**",
                "",
                "| Run | Runtime path | Successful runs | Total captured time | Representative command |",
                "|---|---|---:|---:|---|",
            ]
            for label, path, count, total_time, command in rows:
                table.append(
                    f"| {markdown_cell(label)} | {markdown_cell(path)} | {markdown_cell(count)} | "
                    f"{markdown_cell(total_time)} | {markdown_cell(command)} |"
                )
            lines.append("\n".join(table))
        with_rounds = _logic._round_durations_from_output(str(with_job.get("output") or ""))
        base_rounds = _logic._round_durations_from_output(str(base_job.get("output") or "")) if base_job else []
        if with_rounds:
            with_round, with_max = max(with_rounds, key=lambda item: item[1])
            base_max = max((duration for _, duration in base_rounds), default=None)
            if with_max >= 300 and (base_max is None or with_max > base_max * 5):
                base_text = f" vs baseline max round ~{fmt_number(round(base_max))}s" if base_max is not None else ""
                lines.append(
                    f"- **Slow FL round evidence**: with-skills Round {with_round} took ~{fmt_number(round(with_max / 60))} "
                    f"minutes before all client results returned{base_text}. This elapsed round time can include useful "
                    "training/validation work, NVFLARE result transfer, synchronization wait, or a mixture of those."
                )
        with_tx = _logic._max_download_tx_elapsed(str(with_job.get("output") or ""))
        base_tx = _logic._max_download_tx_elapsed(str(base_job.get("output") or "")) if base_job else None
        if with_tx is not None and with_tx >= 120 and (base_tx is None or with_tx > base_tx * 5):
            base_text = f" vs baseline max transfer {fmt_number(round(base_tx))}s" if base_tx is not None else ""
            lines.append(
                f"- **Transfer/wait evidence**: with-skills logged NVFLARE download transactions up to "
                f"{fmt_number(round(with_tx))}s{base_text}. This points to runtime transfer/synchronization wait that "
                "should be investigated separately from generated-code efficiency."
            )
        return lines
