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

"""Plugin-routing acceptance test (the milestone's regression guard).

The routing DoD: the generic report engine makes **no rendering call into
NVFLARE plugin logic** (``sdks/nvflare/_logic``). This was a strict-xfail red bar
during the migration; routing is now complete, so the test is GREEN and guards
against regressions:

1. **Structural (DoD #1):** none of the ``_logic`` names the engine used to
   import by name are bound on the ``benchmark_insights`` module anymore — the
   engine reaches the SDK only through the resolved plugin.
2. **Behavioural (DoD #2):** with the **Null** plugin resolved (selected via a
   present-but-unknown ``report_plugin_id`` — an *absent* id would fall back to
   the NVFLARE legacy plugin) and every ``_logic`` function poisoned to raise on
   call, the report still renders. So no engine render path reaches ``_logic``
   when a non-NVFLARE plugin is active.
"""

from __future__ import annotations

from _report_fixtures import build_result_root

# Names the engine used to import from sdks.nvflare._logic before routing.
_FORMERLY_IMPORTED_LOGIC_NAMES = (
    "CODE_QUALITY_CONTEXT_ROWS",
    "CODE_QUALITY_ROWS",
    "REQUIRED_STRUCTURE_FILES",
    "_successful_job_spans",
    "completed_job_recovered_issue_summary",
    "current_workspace_structure_file_matches",
    "fl_algorithm_info",
    "fl_algorithm_recipe_mismatch",
    "generated_code_quality_assessments",
    "generated_code_quality_overall",
    "generated_code_quality_score",
    "invokes_nvflare_simulator",
    "is_material_failed_command",
    "job_run_status",
    "job_run_status_reason",
    "last_successful_job_event",
    "recovered_by_later_success",
    "recovered_by_later_successful_job",
)


def _select_null_plugin(root):
    from benchmark.harness import profile_metadata

    profile_metadata.write_json(
        root / profile_metadata.ROOT_DESCRIPTOR_FILENAME,
        {"report_plugin_id": "__test_unknown__"},
    )


def test_engine_does_not_bind_logic_names_by_name():
    """DoD #1: the SDK logic is no longer imported by name into the engine."""

    from benchmark.harness.reports import benchmark_insights

    still_bound = [name for name in _FORMERLY_IMPORTED_LOGIC_NAMES if hasattr(benchmark_insights, name)]
    assert not still_bound, f"engine still binds sdks.nvflare._logic names: {still_bound}"


def test_null_plugin_render_does_not_call_nvflare_logic(tmp_path, monkeypatch):
    """DoD #2: under the Null plugin, no render path reaches sdks.nvflare._logic."""

    from benchmark.harness.reports import benchmark_insights
    from benchmark.harness.sdks.nvflare import _logic
    from benchmark.harness.sdks.report_plugin import NullReportPlugin
    from benchmark.harness.sdks.report_registry import resolve_from_result_root

    root = build_result_root(tmp_path / "result_root")
    _select_null_plugin(root)
    assert isinstance(resolve_from_result_root(root), NullReportPlugin)

    runs = benchmark_insights.collect_benchmark_runs(root)

    # Poison every _logic callable: if the null render path reaches any of them,
    # the render raises instead of silently using NVFLARE logic.
    def _boom(*_args, **_kwargs):
        raise AssertionError("null-plugin render reached sdks.nvflare._logic")

    for name in dir(_logic):
        if name.startswith("__"):
            continue
        attr = getattr(_logic, name)
        if callable(attr):
            monkeypatch.setattr(_logic, name, _boom)

    report = benchmark_insights.benchmark_report(root, runs)

    assert "# Agent Benchmark Insights" in report


def _poison_logic_callables(monkeypatch):
    from benchmark.harness.sdks.nvflare import _logic

    def _boom(*_args, **_kwargs):
        raise AssertionError("null-plugin render reached sdks.nvflare._logic")

    for name in dir(_logic):
        if not name.startswith("__") and callable(getattr(_logic, name)):
            monkeypatch.setattr(_logic, name, _boom)


def test_null_plugin_failure_path_does_not_call_nvflare_logic(monkeypatch):
    """Regression: the failure-analysis branch must also route through the plugin.

    ``metric_reporting_gap_evidence`` previously dropped the plugin when calling
    ``successful_job_evidence``, falling back to the NVFLARE legacy plugin and
    reaching ``_logic`` even under the Null plugin. The standard fixture renders
    only passing runs, so this needs/needs-review path was not exercised.
    """

    from benchmark.harness.reports import benchmark_insights
    from benchmark.harness.reports.evidence import _run_evidence_from_bundle
    from benchmark.harness.sdks.report_plugin import NullReportPlugin

    # A run that flows through the non-passing branch with a quality issue, so
    # metric_reporting_gap_evidence reaches successful_job_evidence. The render
    # path consumes typed RunEvidence, so build it as such.
    run = _run_evidence_from_bundle(
        {
            "available": True,
            "label": "needs review",
            # A truthy workspace_delta with no artifact evidence makes
            # run_quality_issues non-empty, so the gap-evidence path is reached.
            "record": {"workspace_delta": {"changed_file_count": 0}},
        }
    )
    runs = {"with_skills": run}
    plugin = NullReportPlugin()
    ctx = benchmark_insights._report_context(runs, ["with_skills"], plugin)

    _poison_logic_callables(monkeypatch)

    # The renderer reads the per-run derived view (PluginEvidence), not the plugin.
    ev = ctx.evidence["with_skills"]
    # None of these may reach _logic when the active plugin is the Null plugin.
    assert benchmark_insights.run_quality_issues(run, ev)  # issue present -> path is exercised
    benchmark_insights.metric_reporting_gap_evidence(run, ev, ctx)
    benchmark_insights.failure_analysis_section(runs, ["with_skills"], ctx)
