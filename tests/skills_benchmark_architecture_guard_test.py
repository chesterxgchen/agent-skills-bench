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

"""Architecture guard — measure the ARCHITECTURE, not just the output.

The golden snapshot locks NVFLARE output byte-identical; the routing acceptance
test proves the engine makes no NVFLARE ``_logic`` call under the Null plugin.
Neither can see the architectural invariants the design is actually about:

- a real *second* (non-FL) SDK can drive the report through the plugin boundary
  alone (no ``sdks/nvflare/_logic`` touched), and
- the rendered report carries NO FL/NVFLARE vocabulary when a non-FL plugin is
  active.

This module renders under a TOY non-FL plugin and asserts both — both now PASS
(the Phase D domain carve moved FL vocabulary behind the plugin). The whole-report
guard asserts section presence + FL-vocab absence non-vacuously; the leaf-builder
guards below cover the conditional sections the 2-mode whole-report can't reach.
``_FL_VOCABULARY`` is the banned-term contract — keep it in sync with the domain
terms the engine could emit (it is widened as carves land, e.g. site/per-site).
"""

from __future__ import annotations

import re

from _neutral_fixtures import build_neutral_result_root

from benchmark.harness.reports._context import (
    AlgorithmSignal,
    CodeQualitySignal,
    JobExecutionSignal,
    ReportContext,
    StructureView,
)
from benchmark.harness.sdks.report_plugin import (
    MetricAssessment,
    ParticipantModel,
    PluginEvidence,
    ReportPlugin,
    ReportSection,
    SdkActivitySignal,
    StructureSignal,
)

# Vocabulary that must NOT appear in a non-FL report (case-insensitive word/sub
# matches). These are the FL/NVFLARE domain terms the generic engine still emits.
_FL_VOCABULARY = (
    "nvflare",
    "fedavg",
    "fl algorithm",
    "fl result",
    "fl-level",
    "federated",
    "simulator",
    r"\bsite-\d",
    "aggregated",
    # FL-meaningful forms (not bare common words): the SDK's execution-activity noun
    # ("simulation"), the participant qualifiers, and the FL aggregation phrasing. Bare
    # "site"/"global"/"server" are intentionally NOT banned (they false-positive on
    # generic prose); only the FL-meaningful collocations are.
    "simulation",
    "per-site",
    "site-level",
    "site metric",
    r"(?:aggregated|global|server)\s+validation",
)


class ToyReportPlugin(ReportPlugin):
    """A minimal NON-FL SDK plugin: its own vocabulary, no FL assumptions."""

    def collect(self, run):
        available = bool(run.raw.get("available"))
        return PluginEvidence(
            structure=StructureSignal(score=1.0 if available else None),
            sdk_activity=SdkActivitySignal(detected=available, detail="toy task ran" if available else None),
            job_execution=JobExecutionSignal(
                status="ran" if available else "absent",
                status_reason="the toy task executed" if available else "no toy task",
            ),
            algorithm=AlgorithmSignal(info={"algorithm": "ToyRoutine", "recipe": "toy", "num_rounds": None}),
            structure_view=StructureView(
                score=1.0 if available else None,
                required_files=("main.py",),
                present_required=("main.py",) if available else (),
            ),
            code_quality=CodeQualitySignal(overall="acceptable"),
        )

    def participant_model(self):
        # Non-FL vocabulary on purpose: worker/coordinator, not site/server.
        return ParticipantModel(participant="worker", aggregate="coordinator")

    def assess_metric(self, run, expected):
        metric = run.validation_metric or {}
        name = metric.get("name") if isinstance(metric, dict) else None
        return MetricAssessment(name=name, reported=bool(name))

    def score_structure(self, run):
        return StructureSignal(score=1.0 if run.raw.get("available") else None)

    def detect_sdk_activity(self, run):
        return SdkActivitySignal(detected=bool(run.raw.get("available")), detail="toy")

    def explain(self, cmp, plugin):
        return []

    def sections(self, cmp, plugin):
        # The toy SDK contributes its own algorithm/workflow section (neutral vocabulary),
        # mirroring how NVFLARE contributes the FL one (E1b): no engine-owned algorithm
        # section, each SDK owns its own. Anchored after the generic Executive Summary block.
        rows = ["| Run | Algorithm/workflow | Recipe | Rounds | Evidence |", "|---|---|---|---:|---|"]
        for mode in cmp.modes:
            run = cmp.runs[mode]
            evidence = plugin.get(mode)
            algorithm = evidence.algorithm if evidence is not None else None
            info = (algorithm.info if algorithm is not None else None) or {}
            rows.append(
                f"| {run.label or mode} | {info.get('algorithm') or 'NA'} | "
                f"{info.get('recipe') or 'not captured'} | {info.get('num_rounds') or 'NA'} | "
                f"{info.get('evidence') or 'NA'} |"
            )
        body = "\n".join(
            [
                "This section reports the algorithm/workflow captured from the SDK's generated "
                "configuration; agent planning or final-message text is not counted as runtime workflow evidence.",
                "",
                *rows,
            ]
        )
        return [
            ReportSection(
                id="algorithm",
                title="## Algorithm / Workflow",
                body=body,
                anchor="exec_summary",
                placement="after",
            )
        ]


def _render_under_toy_plugin(tmp_path, monkeypatch):
    from benchmark.harness import profile_metadata
    from benchmark.harness.reports import benchmark_insights
    from benchmark.harness.sdks import report_registry

    root = build_neutral_result_root(tmp_path / "result_root")
    profile_metadata.write_json(
        root / profile_metadata.ROOT_DESCRIPTOR_FILENAME,
        {"report_plugin_id": "__toy__"},
    )
    # Register the toy plugin for the test only (no shared-file edit; Phase C adds
    # the real registration seam).
    monkeypatch.setitem(report_registry._REGISTRY, "__toy__", ToyReportPlugin)
    assert isinstance(report_registry.resolve_from_result_root(root), ToyReportPlugin)

    runs = benchmark_insights.collect_benchmark_runs(root)
    return benchmark_insights.benchmark_report(root, runs)


def test_second_sdk_renders_without_touching_nvflare_logic(tmp_path, monkeypatch):
    """A real non-FL plugin drives the whole report through the plugin boundary."""

    from benchmark.harness.reports import benchmark_insights
    from benchmark.harness.sdks.nvflare import _logic, plugin  # noqa: F401

    # Import the NVFLARE plugin BEFORE poisoning ``_logic``: the plugin binds names
    # via ``from ._logic import ...`` at import time, so if its first import happened
    # while ``_logic`` is patched, those bindings would capture ``_boom`` and survive
    # monkeypatch's revert (which only restores ``_logic``), leaking into later tests.
    def _boom(*_args, **_kwargs):
        raise AssertionError("second-SDK render reached sdks.nvflare._logic")

    for name in dir(_logic):
        if not name.startswith("__") and callable(getattr(_logic, name)):
            monkeypatch.setattr(_logic, name, _boom)

    report = _render_under_toy_plugin(tmp_path, monkeypatch)
    assert "# Agent Benchmark Insights" in report
    # The toy plugin's own vocabulary made it into the report.
    assert "ToyRoutine" in report
    # And benchmark_insights is the consumer, not coupled to _logic at all.
    assert not hasattr(benchmark_insights, "fl_algorithm_info")


def test_non_fl_report_has_no_fl_vocabulary(tmp_path, monkeypatch):
    """Architecture invariant (Phase D done): a non-FL report carries no FL vocabulary.

    Renders the SDK-NEUTRAL fixture under the toy plugin and asserts BOTH that the
    always-on sections are present (so the pass is NON-vacuous — it can't go green by
    sections silently failing to render) AND that no FL/NVFLARE term leaks. The domain
    vocabulary now lives behind ``participant_model``/``assess_metric``/``section_copy``.
    """

    report = _render_under_toy_plugin(tmp_path, monkeypatch)

    # Non-vacuous: the always-on sections actually rendered under the toy plugin.
    for section in ("## Executive Summary", "### Run Status", "Algorithm / Workflow", "ToyRoutine"):
        assert section in report, f"expected section missing (vacuous render?): {section!r}"

    leaks = [term for term in _FL_VOCABULARY if re.search(term, report.lower())]
    assert not leaks, f"FL/NVFLARE vocabulary leaked into a non-FL report: {leaks}"


# --- leaf-builder guards (Q6): the conditional sections the 2-mode whole-report
# fixture can't reach, rendered directly under the toy plugin and asserted neutral ---


def _toy_ctx(evidence=None):
    return ReportContext(evidence=evidence or {}, plugin=ToyReportPlugin())


def _run(bundle):
    from benchmark.harness.reports.evidence import _run_evidence_from_bundle

    return _run_evidence_from_bundle(bundle)


def _assert_no_fl_vocab(text, where):
    leaks = [term for term in _FL_VOCABULARY if re.search(term, text.lower())]
    assert not leaks, f"FL/NVFLARE vocabulary leaked from {where}: {leaks}"


def test_repeated_job_section_is_neutral_under_non_fl_plugin():
    """Repeated-job section: 'job executions' (atom dropped), no 'simulator'."""

    from benchmark.harness.reports import benchmark_insights

    spans = (
        {"command": "python train.py", "duration_seconds": 60, "exit_code": 0, "description": "run"},
        {"command": "python train.py", "duration_seconds": 65, "exit_code": 0, "description": "rerun after fix"},
    )
    ctx = _toy_ctx({"m": PluginEvidence(job_execution=JobExecutionSignal(successful_job_spans=spans))})
    section = benchmark_insights.repeated_job_runs_section({"m": _run({"available": True, "label": "Run"})}, ["m"], ctx)
    # Neutral header: no "/Simulation" suffix when the SDK has no execution-activity noun.
    assert "### Repeated Job Executions" in section  # present (non-vacuous)
    assert "job executions" in section  # neutral grammar, no ' or simulator'
    _assert_no_fl_vocab(section, "repeated_job_runs_section")


def test_job_run_action_not_started_is_neutral_under_non_fl_plugin():
    """Not-started action: 'run the generated job', no 'simulator'/'FL metrics'."""

    from benchmark.harness.reports import benchmark_insights

    action = benchmark_insights.job_run_action(
        _run({"available": True}), JobExecutionSignal(status="not_started"), _toy_ctx()
    )
    assert "run the generated job before" in action
    _assert_no_fl_vocab(action, "job_run_action(not_started)")


def test_additional_metric_values_no_simulation_is_neutral_under_non_fl_plugin():
    """No-run metric note: neutral run/participant nouns, no 'simulation'/'per-site'."""

    from benchmark.harness.reports import benchmark_insights

    text = benchmark_insights.additional_or_observed_metric_values_display(
        _run({"available": True}), "score", None, _toy_ctx()
    )
    # Neutral wording after carve: the SDK's "simulation" -> the engine's "run"; the
    # FL "per-site" -> the toy participant "per-worker".
    assert "no run run detected" in text
    assert "per-round/per-worker values" in text
    _assert_no_fl_vocab(text, "additional_or_observed_metric_values_display(no-sim)")


def test_additional_metric_values_log_evidence_is_neutral_under_non_fl_plugin():
    """The log/per-site metric-evidence branch renders neutral participant wording."""

    from benchmark.harness.reports import benchmark_insights

    # A run whose last successful job event carries a generic-ML metric line so the
    # log-evidence branch (the "Final site metrics=NA; log/per-site evidence:" site)
    # fires; under the toy plugin it must use the neutral participant term.
    ev = PluginEvidence(job_execution=JobExecutionSignal(last_successful_job_event={"output": "valid_acc=0.9\n"}))
    text = benchmark_insights.additional_or_observed_metric_values_display(
        _run({"available": True}), "score", ev, _toy_ctx()
    )
    assert "Final worker metrics=NA; log/per-worker evidence:" in text  # neutral (non-vacuous)
    _assert_no_fl_vocab(text, "additional_or_observed_metric_values_display(log-evidence)")


def test_bash_blocked_unrecovered_is_neutral_under_non_fl_plugin():
    """Unrecovered Bash diagnostic: the engine's neutral run noun, no 'simulation'."""

    from benchmark.harness.reports import benchmark_insights

    run = _run(
        {
            "available": True,
            "agent_events_text": "the agent requested permissions to use Bash but was denied",
        }
    )
    text = benchmark_insights.bash_blocked_diagnostic(run, recovered=False, ctx=_toy_ctx())
    assert text  # non-vacuous (denial count > 0)
    assert "The run was never run as a result." in text  # neutral run noun
    _assert_no_fl_vocab(text, "bash_blocked_diagnostic(unrecovered)")


def test_activity_insights_table_is_neutral_under_non_fl_plugin():
    """Activity table run-noun label is neutral: 'Run references', not 'Simulation references'."""

    from benchmark.harness.reports import benchmark_insights

    ctx = _toy_ctx({"m": PluginEvidence()})
    table = benchmark_insights.activity_insights_table({"m": _run({"available": True, "label": "Run"})}, ["m"], ctx)
    assert "Run references" in table  # neutral run-noun label (non-vacuous)
    _assert_no_fl_vocab(table, "activity_insights_table")


def test_metric_log_lines_fl_aggregation_is_plugin_owned():
    """D2 report-path: the generic engine keeps generic-ML metric lines; the
    FL-aggregation pattern is recognized only when the SDK plugin supplies it."""

    from benchmark.harness.reports import benchmark_insights
    from benchmark.harness.sdks.nvflare.plugin import NvflareReportPlugin

    out = "2026-01-01 00:00:00 - INFO - aggregated validation AUROC = 0.80\nvalid_acc=0.9\n"
    # Under a non-FL plugin: generic-ML line kept, FL-aggregation line NOT surfaced.
    toy_lines = benchmark_insights.metric_log_lines(out, ctx=_toy_ctx())
    assert any("valid_acc=0.9" in line for line in toy_lines)
    assert not any("aggregated validation" in line for line in toy_lines)
    _assert_no_fl_vocab("\n".join(toy_lines), "metric_log_lines(toy)")
    # NVFLARE supplies the aggregation pattern -> that line is recognized.
    nv_ctx = ReportContext(evidence={}, plugin=NvflareReportPlugin())
    assert any("aggregated validation" in line for line in benchmark_insights.metric_log_lines(out, ctx=nv_ctx))


def test_missing_metrics_section_is_neutral_under_non_fl_plugin():
    """Missing-metrics section action copy carries no FL vocabulary (Phase-D straggler)."""

    from benchmark.harness.reports import benchmark_insights

    # A run with an expected metric whose scalar is absent -> a quality issue ->
    # the missing-metrics section renders the default "aggregate <result> metric" action.
    run = _run(
        {
            "available": True,
            "label": "Run",
            "record": {
                "quality_signals": {"job_guidance_primary_validation_metric": {"expected_primary_metric": "score"}}
            },
            "validation_metric": {"name": "score", "reported_values": [0.1, 0.2]},
        }
    )
    ctx = _toy_ctx({"m": PluginEvidence()})
    section = benchmark_insights.missing_result_metrics_section({"m": run}, ["m"], ctx)
    assert "Missing, Partial, Or Mismatched Result Metrics" in section  # present (non-vacuous)
    assert "aggregate validation metric" in section  # neutral: no "FL" qualifier
    _assert_no_fl_vocab(section, "missing_result_metrics_section")


def test_successful_job_evidence_is_neutral_under_non_fl_plugin():
    """Recovery evidence: 'job command exited 0'/'workflow reached a Finished state', no FL vocab."""

    from benchmark.harness.reports import benchmark_insights

    ev = PluginEvidence(
        job_execution=JobExecutionSignal(
            last_successful_job_event={"output": "Round 1\nFinished\nResult workspace: /tmp/ws"}
        )
    )
    text = benchmark_insights.successful_job_evidence(_run({"available": True}), ev, _toy_ctx())
    assert text  # non-vacuous
    assert "job command exited 0" in text  # neutral atom (no 'simulator/job')
    assert "workflow reached a Finished state" in text  # neutral (no 'FL workflow')
    _assert_no_fl_vocab(text, "successful_job_evidence")


def test_bash_blocked_diagnostic_recovered_is_neutral_under_non_fl_plugin():
    """Recovered Bash diagnostic: 'a later job command completed', no 'simulator/job'."""

    from benchmark.harness.reports import benchmark_insights

    run = _run(
        {
            "available": True,
            "agent_events_text": "the agent requested permissions to use Bash but was denied",
        }
    )
    text = benchmark_insights.bash_blocked_diagnostic(run, recovered=True, ctx=_toy_ctx())
    assert text  # non-vacuous (denial count > 0)
    assert "a later job command completed" in text  # neutral atom (no 'simulator/job')
    _assert_no_fl_vocab(text, "bash_blocked_diagnostic(recovered)")


def _runtime_path_divergent_pair():
    """A with/base pair that triggers the NVFLARE runtime-path divergence note.

    With-skills runs an in-process FL recipe (a runtime path); the baseline has no
    classified successful job/simulator command (only an inspection command), so the
    note renders the divergence table with the baseline fallback row.
    """

    import json

    from benchmark.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE

    def command_events(command, start, end, item_id, output="ok"):
        return [
            {
                "timestamp": start,
                "type": "item.started",
                "item": {"command": command, "id": item_id, "status": "in_progress", "type": "command_execution"},
            },
            {
                "timestamp": end,
                "type": "item.completed",
                "item": {
                    "aggregated_output": output,
                    "command": command,
                    "exit_code": 0,
                    "id": item_id,
                    "status": "completed",
                    "type": "command_execution",
                },
            },
        ]

    with_run = {
        "label": "With skills",
        "mode": WITH_SKILLS_MODE,
        "agent_events_text": "\n".join(
            json.dumps(event)
            for event in command_events(
                "python job.py",
                "2026-06-13T08:00:00Z",
                "2026-06-13T08:14:02Z",
                "with_job",
                output="PTInProcessClientAPIExecutor - INFO - result received\nFinished FedAvg.\n",
            )
        ),
    }
    base_run = {
        "label": "No skills baseline",
        "mode": NO_SKILLS_MODE,
        "agent_events_text": "\n".join(
            json.dumps(event)
            for event in command_events(
                "nl -ba nvflare_scaffold_job.py | sed -n '1,240p'",
                "2026-06-13T08:04:00Z",
                "2026-06-13T08:04:01Z",
                "inspect_job_source",
                output='print("Finished FedAvg.")\n',
            )
        ),
    }
    return WITH_SKILLS_MODE, NO_SKILLS_MODE, with_run, base_run


def test_runtime_path_note_is_nvflare_owned_and_absent_under_non_fl_plugin():
    """Leaf-builder guard (E3): the FL runtime-path note is plugin-owned.

    Under the NVFLARE plugin the divergent pair renders the runtime-path note in the
    Why-slower slot; under the toy plugin the SAME pair contributes NO ``why_slowdown``
    narrative and the rendered Why output carries no FL vocabulary. This proves the
    engine slot is empty without an SDK contributor (no engine fallback after E3).
    """

    from benchmark.harness.reports import benchmark_insights
    from benchmark.harness.sdks.nvflare.plugin import NvflareReportPlugin

    with_mode, base_mode, with_bundle, base_bundle = _runtime_path_divergent_pair()
    runs = {with_mode: _run(with_bundle), base_mode: _run(base_bundle)}
    modes = [with_mode, base_mode]

    # NVFLARE: the note renders (slot has a contributor).
    nv_ctx = benchmark_insights._report_context(runs, modes, NvflareReportPlugin())
    assert nv_ctx.narrative("why_slowdown"), "NVFLARE plugin must contribute the why_slowdown narrative"
    nv_why = "\n".join(benchmark_insights._why_slower(runs[with_mode], runs[base_mode], nv_ctx))
    assert "runtime path diverged" in nv_why.lower()

    # Toy: the SAME pair contributes nothing to the slot and leaks no FL vocabulary.
    toy_ctx = benchmark_insights._report_context(runs, modes, ToyReportPlugin())
    assert not toy_ctx.narrative("why_slowdown"), "non-FL plugin must not contribute the why_slowdown narrative"
    toy_why = "\n".join(benchmark_insights._why_slower(runs[with_mode], runs[base_mode], toy_ctx))
    assert "runtime path diverged" not in toy_why.lower()
    _assert_no_fl_vocab(toy_why, "_why_slower(toy)")


def test_summary_scalar_selection_is_sdk_owned_not_engine_fl_selected():
    """Leaf-builder guard (F1): the summary-scalar SELECTION is plugin-owned.

    A run with NO plain ``value`` but reported entries carrying summary labels (one
    FL-pattern, one NON-FL-pattern). The engine's ``metric_value`` no longer FL-selects
    on the generic path; the scalar must come from the active plugin's ``assess_metric``
    via the sidecar (``MetricAssessment.value``):

      * NVFLARE: the FL-pattern entry ("Best aggregated validation AUROC") is selected.
      * Toy (non-FL): ``assess_metric`` does not select a summary scalar, so the generic
        engine surfaces NOTHING -- it must NOT FL-select the FL-pattern entry, and must
        not surface the NON-FL "coordinator summary" entry either.
    """

    from benchmark.harness.reports import benchmark_insights
    from benchmark.harness.sdks.nvflare.plugin import NvflareReportPlugin

    metric = {
        "name": "AUROC",
        "source": "metrics_artifact",
        "source_path": "workspace_delta/runtime_artifacts/server/simulate_job/metrics/metrics_summary.json",
        "value": None,  # no plain scalar -> selection must come from the plugin sidecar
        "reported_value_entries": [
            {"label": "coordinator summary", "value": 0.9},  # NON-FL-pattern label
            {"label": "Best aggregated validation AUROC", "value": 0.7623},  # FL-pattern label
        ],
        "reported_values": [0.9, 0.7623],
    }
    runs = {"m": _run({"available": True, "label": "Run", "validation_metric": metric})}
    modes = ["m"]
    run_ev = runs["m"]

    # NVFLARE: the plugin's assess_metric FL-selects the FL-pattern summary entry.
    nv_ctx = benchmark_insights._report_context(runs, modes, NvflareReportPlugin())
    nv_ev = nv_ctx.evidence["m"]
    assert nv_ev.metric.value == 0.7623
    assert benchmark_insights.metric_value(run_ev, "AUROC", nv_ev) == 0.7623

    # Toy: assess_metric selects no scalar; the generic engine must NOT FL-select.
    # Build the toy sidecar from its OWN assess_metric (its decision, not collect()'s
    # default) so the guard exercises the SDK-owned selection path directly.
    toy_plugin = ToyReportPlugin()
    toy_ev = PluginEvidence(metric=toy_plugin.assess_metric(run_ev, None))
    assert toy_ev.metric.value is None  # toy's assess_metric selected no summary scalar
    assert benchmark_insights.metric_value(run_ev, "AUROC", toy_ev) is None
    # And neither summary entry is surfaced as the result scalar via the generic path.
    assert "0.7623" not in benchmark_insights.metric_display(run_ev, "AUROC", toy_ev)
    assert "0.9" not in benchmark_insights.metric_display(run_ev, "AUROC", toy_ev)


def test_plugin_selected_scalar_clears_scalar_availability_checks():
    """F1 threading: a plugin-selected summary scalar must clear the scalar-AVAILABILITY
    checks (run_quality_issues / benchmark_outcome), not just metric_value.

    Regression for the bug where result_metric_scalar_available()/run_quality_issues()
    called metric_value WITHOUT the sidecar `ev`, so an NVFLARE run whose scalar lived
    only in a labeled entry was falsely flagged `fl_metric_scalar` even though
    assess_metric had selected it.
    """

    from benchmark.harness.reports import benchmark_insights
    from benchmark.harness.sdks.nvflare.plugin import NvflareReportPlugin

    run = _run(
        {
            "available": True,
            "label": "Run",
            # expected metric set so the fl_metric_scalar availability check fires:
            "record": {
                "quality_signals": {"job_guidance_primary_validation_metric": {"expected_primary_metric": "AUROC"}}
            },
            # no plain value; the summary scalar is only in a labeled entry:
            "validation_metric": {
                "name": "AUROC",
                "source": "metrics_artifact",
                "source_path": "workspace_delta/runtime_artifacts/server/simulate_job/metrics/metrics_summary.json",
                "value": None,
                "reported_value_entries": [{"label": "Best aggregated validation AUROC", "value": 0.7623}],
                "reported_values": [0.7623],
            },
        }
    )
    # NVFLARE: assess_metric selects 0.7623 -> the availability checks must see it.
    nv_ev = NvflareReportPlugin().collect(run)
    assert nv_ev.metric.value == 0.7623
    assert benchmark_insights.run_quality_issues(run, nv_ev) == []
    assert benchmark_insights.benchmark_outcome(run, nv_ev).startswith("pass")
    # Toy (non-FL): no scalar inferred -> the checks STILL flag (correct: it genuinely has none).
    toy_ev = PluginEvidence(metric=ToyReportPlugin().assess_metric(run, None))
    assert benchmark_insights.run_quality_issues(run, toy_ev)  # non-empty


def test_nvflare_runtime_log_artifact_scalar_is_plugin_accepted():
    """Captured runtime logs are runtime evidence, but the NVFLARE plugin owns that call.

    The generic artifact parser can recover a scalar from final site logs. This guard
    keeps the SDK decision at the plugin boundary: NVFLARE accepts captured runtime-log
    artifacts as result evidence, while still relying on the generic engine only for
    rendering the selected sidecar scalar.
    """

    from benchmark.harness.reports import benchmark_insights
    from benchmark.harness.sdks.nvflare.plugin import NvflareReportPlugin

    run = _run(
        {
            "available": True,
            "label": "Run",
            "validation_metric": {
                "name": "AUROC",
                "source": "runtime_log_artifact",
                "source_path": "workspace_delta/runtime_artifacts/runtime_workspaces/job/site-1/log.txt",
                "value": 0.7698,
                "summary_value_label": "artifact aggregated validation metric final log mean AUROC",
                "reported_value_entries": [
                    {"label": "artifact site validation metric site-1 final log AUROC", "value": 0.7904},
                    {"label": "artifact aggregated validation metric final log mean AUROC", "value": 0.7698},
                ],
                "reported_values": [0.7904, 0.7698],
            },
        }
    )

    nv_ev = NvflareReportPlugin().collect(run)

    assert nv_ev.metric.value == 0.7698
    assert benchmark_insights.metric_display(run, "AUROC", nv_ev) == "AUROC 0.7698"


def test_server_selected_best_metric_satisfies_fl_scalar_gate(tmp_path):
    """A metrics-less run still passes the FL-scalar gate via the model selector.

    Regression: a recipe run that writes NO metrics_summary.json (only simulator
    logs + configs) reported its AUROC in the final message and the gate failed
    with "AUROC was reported, but no FL-level scalar value was found" — even
    though the captured server config declares the model selector's key_metric
    and the server log carries its selected best value. The plugin must promote
    that structured (config name + log value) pair to the authoritative FL-level
    scalar; the metric NAME is never grepped out of log lines, so it follows
    whatever metric the job declares.
    """

    import json

    from benchmark.harness.reports import benchmark_insights
    from benchmark.harness.sdks.nvflare.plugin import NvflareReportPlugin

    delta_dir = tmp_path / "workspace_delta" / "runtime_artifacts" / "nvflare_runtime"
    config_rel = "nvflare_runtime/job_config/app/config/config_fed_server.json"
    log_rel = "nvflare_runtime/simulation/job/server/log.txt"
    config_path = delta_dir / "job_config" / "app" / "config" / "config_fed_server.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps({"components": [{"id": "model_selector", "args": {"key_metric": "auroc"}}]}),
        encoding="utf-8",
    )
    log_path = delta_dir / "simulation" / "job" / "server" / "log.txt"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        "2026-07-05 03:59:24,691 - IntimeModelSelector - INFO - new best validation metric at round 1: 0.7139\n"
        "2026-07-05 03:59:40,836 - IntimeModelSelector - INFO - new best validation metric at round 2: 0.7359736256742272\n",
        encoding="utf-8",
    )
    run = _run(
        {
            "available": True,
            "label": "Run",
            "mode_dir": tmp_path,
            "record": {
                "quality_signals": {"job_guidance_primary_validation_metric": {"expected_primary_metric": "AUROC"}}
            },
            "workspace_delta": {
                "runtime_artifacts": [
                    {"path": config_rel, "artifact_path": f"runtime_artifacts/{config_rel}"},
                    {"path": log_rel, "artifact_path": f"runtime_artifacts/{log_rel}"},
                ]
            },
            # The agent's final-message self-report alone must NOT satisfy the gate;
            # the plugin recovers the FL-level value from the server-side evidence.
            "validation_metric": {
                "name": "AUROC",
                "source": "agent_last_message",
                "value": 0.7359736257,
                "value_scope": "reported_scalar",
                "reported_values": [0.7359736257],
                "reported_value_entries": [{"label": "Best selected global-model AUROC", "value": 0.7359736257}],
            },
        }
    )

    nv_ev = NvflareReportPlugin().collect(run)

    assert nv_ev.metric.value == 0.7359736256742272
    assert "round 2" in (nv_ev.metric.value_label or "")
    # The fixture carries no job-run evidence, so only the metric checks matter here.
    issues = benchmark_insights.run_quality_issues(run, nv_ev)
    assert not [issue for issue in issues if "result_metric" in issue or "FL-level scalar" in issue], issues


def test_server_selected_metric_consumed_without_any_validation_metric_payload(tmp_path):
    """The recovered scalar must satisfy the gate even with NO validation_metric.

    Regression (review #582): metric_value() gated on run.validation_metric.name
    and never consulted the plugin sidecar when the run payload was missing, so
    valid server config + server log evidence still failed the FL scalar gate
    unless the agent also self-reported the metric in its final message.
    """

    import json

    from benchmark.harness.reports import benchmark_insights
    from benchmark.harness.reports.insights._metrics import metric_value
    from benchmark.harness.sdks.nvflare.plugin import NvflareReportPlugin

    delta_dir = tmp_path / "workspace_delta" / "runtime_artifacts" / "nvflare_runtime"
    config_path = delta_dir / "job_config" / "app" / "config" / "config_fed_server.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps({"components": [{"id": "model_selector", "args": {"key_metric": "auroc"}}]}),
        encoding="utf-8",
    )
    log_path = delta_dir / "simulation" / "job" / "server" / "log.txt"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("INFO - new best validation metric at round 3: 0.81\n", encoding="utf-8")
    run = _run(
        {
            "available": True,
            "label": "Run",
            "mode_dir": tmp_path,
            "record": {
                "quality_signals": {"job_guidance_primary_validation_metric": {"expected_primary_metric": "AUROC"}}
            },
            "workspace_delta": {
                "runtime_artifacts": [
                    {
                        "path": "nvflare_runtime/job_config/app/config/config_fed_server.json",
                        "artifact_path": "runtime_artifacts/nvflare_runtime/job_config/app/config/config_fed_server.json",
                    },
                    {
                        "path": "nvflare_runtime/simulation/job/server/log.txt",
                        "artifact_path": "runtime_artifacts/nvflare_runtime/simulation/job/server/log.txt",
                    },
                ]
            },
            # No validation_metric anywhere: not self-reported, no metrics artifact.
        }
    )

    nv_ev = NvflareReportPlugin().collect(run)

    assert nv_ev.metric.name == "AUROC"
    assert nv_ev.metric.value == 0.81
    assert metric_value(run, "AUROC", nv_ev) == 0.81
    assert metric_value(run, None, nv_ev) == 0.81
    # A DIFFERENT requested metric must not be satisfied by the AUROC recovery.
    assert metric_value(run, "accuracy", nv_ev) is None
    issues = benchmark_insights.run_quality_issues(run, nv_ev)
    assert not [issue for issue in issues if "result_metric" in issue or "FL-level scalar" in issue], issues


def test_server_selected_best_metric_rejects_mismatched_key_metric(tmp_path):
    """The model-selector fallback must not grade a DIFFERENT metric as the result.

    When the job expects AUROC but the captured server config selects models by
    loss, the server log's best value is a loss — promoting it would report a
    wrong-metric scalar. The gate must keep failing instead.
    """

    import json

    from benchmark.harness.sdks.nvflare._logic import server_selected_best_metric

    delta_dir = tmp_path / "workspace_delta" / "runtime_artifacts" / "nvflare_runtime"
    config_path = delta_dir / "job_config" / "app" / "config" / "config_fed_server.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps({"components": [{"id": "model_selector", "args": {"key_metric": "loss"}}]}),
        encoding="utf-8",
    )
    log_path = delta_dir / "simulation" / "job" / "server" / "log.txt"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("INFO - new best validation metric at round 2: 0.42\n", encoding="utf-8")
    raw = {
        "mode_dir": tmp_path,
        "workspace_delta": {
            "runtime_artifacts": [
                {
                    "path": "nvflare_runtime/job_config/app/config/config_fed_server.json",
                    "artifact_path": "runtime_artifacts/nvflare_runtime/job_config/app/config/config_fed_server.json",
                },
                {
                    "path": "nvflare_runtime/simulation/job/server/log.txt",
                    "artifact_path": "runtime_artifacts/nvflare_runtime/simulation/job/server/log.txt",
                },
            ]
        },
    }

    assert server_selected_best_metric(raw, "AUROC") == {}
    selected = server_selected_best_metric(raw, "loss")
    assert selected["value"] == 0.42 and selected["name"] == "loss"


def test_plugin_selected_scalar_renders_value_in_sections_not_na():
    """F1 render-path threading: metric_display() inside the section renderers must
    receive the sidecar `ev`, so a plugin-selected summary scalar renders its VALUE,
    not NA.

    Regression for quality_signal_table() and final_response_metric_reporting_gap()
    calling metric_display WITHOUT `ev` -- an NVFLARE run whose scalar lives only in a
    labeled FL entry (selected by assess_metric) rendered "AUROC NA" in those tables.
    """

    from benchmark.harness.reports import benchmark_insights
    from benchmark.harness.sdks.nvflare.plugin import NvflareReportPlugin

    runs = {
        "m": _run(
            {
                "available": True,
                "label": "Run",
                # signal fails AND the scalar lives only in a labeled FL entry:
                "record": {
                    "quality_signals": {
                        "job_guidance_primary_validation_metric": {
                            "expected_primary_metric": "AUROC",
                            "status": "fail",
                            "evidence": "final response omitted the aggregate scalar",
                        }
                    }
                },
                "validation_metric": {
                    "name": "AUROC",
                    "source": "metrics_artifact",
                    "source_path": "workspace_delta/runtime_artifacts/server/simulate_job/metrics/metrics_summary.json",
                    "value": None,
                    "reported_value_entries": [{"label": "Best aggregated validation AUROC", "value": 0.7623}],
                    "reported_values": [0.7623],
                },
            }
        )
    }
    modes = ["m"]

    # NVFLARE: ctx carries the plugin sidecar (assess_metric selected 0.7623).
    nv_ctx = benchmark_insights._report_context(runs, modes, NvflareReportPlugin())
    table = benchmark_insights.quality_signal_table(runs, modes, nv_ctx)
    assert "0.7623" in table and "AUROC NA" not in table
    gap = benchmark_insights.final_response_metric_reporting_gap(runs["m"], nv_ctx.evidence["m"])
    assert "0.7623" in gap and "NA" not in gap

    # Toy (non-FL): assess_metric selects no scalar -> renders NA (correct; it has none).
    toy_ctx = benchmark_insights._report_context(runs, modes, ToyReportPlugin())
    assert "AUROC NA" in benchmark_insights.quality_signal_table(runs, modes, toy_ctx)


def test_metrics_table_uses_resolved_plugin_ctx_not_a_fresh_null_one():
    """The Metrics table (outcome_metrics_table) must render from the ONE resolved
    ReportContext, not rebuild a fresh null-plugin context.

    Regression: benchmark_report()/metrics_report.py called outcome_metrics_table(runs,
    modes) WITHOUT ctx, so it fell back to `_report_context(...)` (null plugin) and a
    plugin-selected-only scalar rendered "AUROC NA" in the Metrics table even though
    NVFLARE's assess_metric had selected it.
    """

    from benchmark.harness.reports import benchmark_insights
    from benchmark.harness.sdks.nvflare.plugin import NvflareReportPlugin

    runs = {
        "m": _run(
            {
                "available": True,
                "label": "Run",
                "validation_metric": {
                    "name": "AUROC",
                    "source": "metrics_artifact",
                    "source_path": "workspace_delta/runtime_artifacts/server/simulate_job/metrics/metrics_summary.json",
                    "value": None,  # scalar lives only in a labeled FL entry -> needs the sidecar
                    "reported_value_entries": [{"label": "Best aggregated validation AUROC", "value": 0.7623}],
                    "reported_values": [0.7623],
                },
            }
        )
    }
    modes = ["m"]

    # Threaded resolved NVFLARE ctx -> the Metrics table shows the selected value.
    nv_ctx = benchmark_insights._report_context(runs, modes, NvflareReportPlugin())
    assert "0.7623" in benchmark_insights.outcome_metrics_table(runs, modes, nv_ctx)
    # Dropping ctx (the bug) rebuilds a null-plugin context -> NA. This is what must NOT
    # happen in the report entrypoints; the contrast pins the threading requirement.
    assert "0.7623" not in benchmark_insights.outcome_metrics_table(runs, modes)


# --- E1b composition guards (§6 / proposal step 4) ---------------------------------


def test_compose_inserts_plugin_section_at_its_anchor():
    """A plugin section is inserted relative to its named anchor block; the generic blocks
    stay intact and authoritative (insert-only)."""

    from benchmark.harness.reports.benchmark_insights import _compose_report

    blocks = [("exec_summary", ["## Executive Summary", ""]), ("failure_analysis", ["## Failure Analysis", ""])]
    after = ReportSection(id="x", title="## X", body="bx", anchor="exec_summary", placement="after")
    out = "\n".join(_compose_report(blocks, [after]))
    assert out.index("## Executive Summary") < out.index("## X") < out.index("## Failure Analysis")
    before = ReportSection(id="y", title="## Y", body="by", anchor="failure_analysis", placement="before")
    out2 = "\n".join(_compose_report(blocks, [before]))
    assert out2.index("## Executive Summary") < out2.index("## Y") < out2.index("## Failure Analysis")
    # Generic blocks are never dropped/replaced.
    assert "## Executive Summary" in out2 and "## Failure Analysis" in out2


def test_compose_orders_multiple_sections_at_one_anchor_deterministically():
    """Several sections sharing an anchor/placement order by ``order`` regardless of input
    order (the deterministic tie-break)."""

    from benchmark.harness.reports.benchmark_insights import _compose_report

    blocks = [("a", ["## A", ""])]
    s2 = ReportSection(id="s2", title="## S2", body="b", anchor="a", order=2)
    s1 = ReportSection(id="s1", title="## S1", body="b", anchor="a", order=1)
    out = "\n".join(_compose_report(blocks, [s2, s1]))  # input shuffled
    assert out.index("## A") < out.index("## S1") < out.index("## S2")


def test_compose_appends_unknown_anchor_at_end_with_warning(caplog):
    """An unknown anchor is appended at the end (never silently dropped) and warned."""

    import logging

    from benchmark.harness.reports.benchmark_insights import _compose_report

    blocks = [("a", ["## A", ""]), ("b", ["## B", ""])]
    section = ReportSection(id="z", title="## Z", body="bz", anchor="does_not_exist")
    with caplog.at_level(logging.WARNING):
        out = "\n".join(_compose_report(blocks, [section]))
    assert "## Z" in out and out.index("## Z") > out.index("## B")  # at end, not dropped
    assert any("unknown anchor" in record.getMessage() for record in caplog.records)


def test_sdk_plugins_do_not_import_the_report_engine():
    """DoD#4 (plugin -> engine direction): no SDK plugin imports the generic report engine
    (``reports/insights/*`` or the ``benchmark_insights`` facade). Plugins read only neutral
    leaves + the report contracts. This is the inverse of the routing-acceptance guard
    (engine -> plugin) and protects the section-composition boundary E1b introduced."""

    import pathlib

    import benchmark.harness.sdks as sdks_pkg

    sdks_dir = pathlib.Path(sdks_pkg.__file__).resolve().parent
    offenders = []
    for path in sorted(sdks_dir.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if not (stripped.startswith("from ") or stripped.startswith("import ")):
                continue
            if (
                "reports.insights" in stripped
                or "reports import insights" in stripped
                or "benchmark_insights" in stripped
            ):
                offenders.append(f"{path.relative_to(sdks_dir)}:{lineno}: {stripped}")
    assert not offenders, f"SDK plugin imports the report engine (DoD#4 violation): {offenders}"


def test_compose_warns_on_invalid_placement(caplog):
    """The SDK-facing contract is not silently permissive: an unrecognized ``placement``
    warns loudly (and falls back to 'after') rather than being silently coerced."""

    import logging

    from benchmark.harness.reports.benchmark_insights import _compose_report

    blocks = [("a", ["## A", ""])]
    section = ReportSection(id="x", title="## X", body="b", anchor="a", placement="sideways")
    with caplog.at_level(logging.WARNING):
        out = "\n".join(_compose_report(blocks, [section]))
    assert "## X" in out and out.index("## A") < out.index("## X")  # rendered, fell back to "after"
    assert any("invalid placement" in record.getMessage() for record in caplog.records)


def test_compose_warns_on_duplicate_section_ids(caplog):
    """``id`` is the stable identity; duplicate ids across a plugin's section list warn
    (they defeat diagnostics/dedup/migration)."""

    import logging

    from benchmark.harness.reports.benchmark_insights import _compose_report

    blocks = [("a", ["## A", ""])]
    s1 = ReportSection(id="dup", title="## X", body="b", anchor="a", order=1)
    s2 = ReportSection(id="dup", title="## Y", body="b", anchor="a", order=2)
    with caplog.at_level(logging.WARNING):
        _compose_report(blocks, [s1, s2])
    assert any("duplicate plugin ReportSection id" in record.getMessage() for record in caplog.records)
