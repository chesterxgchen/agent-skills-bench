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

"""Migration step 3: prove the plugin seam and resolution without changing output.

Covers (a) registry resolution, (b) the NVFLARE shell delegating to the existing
``benchmark_insights`` functions, and (c) ``build_comparison_evidence`` mapping.
"""

from __future__ import annotations

from _report_fixtures import build_result_root

from benchmark.harness.reports import benchmark_insights
from benchmark.harness.reports.evidence import (
    SCHEMA_VERSION,
    ComparisonEvidence,
    RunEvidence,
    build_comparison_evidence,
)
from benchmark.harness.sdks.nvflare._logic import job_run_status, structure_score
from benchmark.harness.sdks.nvflare.plugin import NvflareReportPlugin
from benchmark.harness.sdks.report_plugin import NullReportPlugin, ReportPlugin
from benchmark.harness.sdks.report_registry import resolve_from_result_root, resolve_report_plugin

# --- (a) registry resolution -----------------------------------------------


def test_resolve_nvflare_id_returns_nvflare_plugin():
    plugin = resolve_report_plugin("nvflare")
    assert isinstance(plugin, NvflareReportPlugin)


def test_resolve_none_id_returns_null_plugin():
    # Inversion 3: an absent id resolves to the null plugin; NVFLARE is selected
    # only by explicit captured identity, never assumed for an unidentified tree.
    plugin = resolve_report_plugin(None)
    assert isinstance(plugin, NullReportPlugin)
    assert not isinstance(plugin, NvflareReportPlugin)


def test_resolve_unknown_id_returns_null_plugin():
    plugin = resolve_report_plugin("bogus")
    assert isinstance(plugin, NullReportPlugin)
    assert not isinstance(plugin, NvflareReportPlugin)


def test_resolve_from_result_root_uses_captured_id(tmp_path):
    build_result_root(tmp_path)
    # The fixture stamps report_plugin_id="nvflare" -> resolves NVFLARE by
    # explicit captured identity (Inversion 3: not by an absence fallback).
    assert isinstance(resolve_from_result_root(tmp_path), NvflareReportPlugin)


def test_resolve_from_result_root_absent_identity_returns_null(tmp_path):
    # A finalized tree with no §4.3 identity block resolves to the null plugin.
    from benchmark.harness import profile_metadata

    build_result_root(tmp_path)
    (tmp_path / profile_metadata.ROOT_DESCRIPTOR_FILENAME).unlink()
    assert isinstance(resolve_from_result_root(tmp_path), NullReportPlugin)


def test_all_resolved_plugins_are_report_plugins():
    for plugin_id in ("nvflare", None, "bogus"):
        assert isinstance(resolve_report_plugin(plugin_id), ReportPlugin)


# --- (b) the NVFLARE shell delegates to existing benchmark_insights ---------


def test_shell_score_structure_matches_benchmark_insights(tmp_path):
    build_result_root(tmp_path)
    cmp = build_comparison_evidence(tmp_path)
    bundles = benchmark_insights.collect_benchmark_runs(tmp_path)
    plugin = NvflareReportPlugin()

    for mode, run in cmp.runs.items():
        expected = structure_score(bundles[mode])
        assert plugin.score_structure(run).score == expected


def test_shell_detect_sdk_activity_matches_job_run_status(tmp_path):
    build_result_root(tmp_path)
    cmp = build_comparison_evidence(tmp_path)
    bundles = benchmark_insights.collect_benchmark_runs(tmp_path)
    plugin = NvflareReportPlugin()

    for mode, run in cmp.runs.items():
        expected = job_run_status(bundles[mode])
        assert plugin.detect_sdk_activity(run).detail == expected


def test_shell_assess_metric_matches_report_validation_metric(tmp_path):
    build_result_root(tmp_path)
    cmp = build_comparison_evidence(tmp_path)
    bundles = benchmark_insights.collect_benchmark_runs(tmp_path)
    plugin = NvflareReportPlugin()

    for mode, run in cmp.runs.items():
        # The report consumes run["validation_metric"] (= artifact_metric or
        # record_metric); the shell must reproduce that exact source.
        expected = bundles[mode].get("validation_metric") or {}
        assessment = plugin.assess_metric(run, expected=None)
        assert assessment.name == (expected.get("name") if expected else None)
        assert assessment.reported == bool(expected.get("name"))


def test_shell_participant_model_is_fl_vocabulary():
    model = NvflareReportPlugin().participant_model()
    assert model.participant == "site"
    assert model.aggregate == "server"


def test_null_plugin_is_flat_and_empty(tmp_path):
    build_result_root(tmp_path)
    cmp = build_comparison_evidence(tmp_path)
    plugin = NullReportPlugin()

    model = plugin.participant_model()
    assert model.participant is None and model.aggregate is None

    for run in cmp.runs.values():
        sidecar = plugin.collect(run)
        assert sidecar.structure is None
        assert sidecar.sdk_activity is None
        assert sidecar.metric is None
        assert dict(sidecar.extra) == {}
    assert plugin.explain(cmp, {}) == []
    assert plugin.sections(cmp, {}) == []


def test_shell_collect_returns_sidecar(tmp_path):
    build_result_root(tmp_path)
    cmp = build_comparison_evidence(tmp_path)
    plugin = NvflareReportPlugin()
    bundles = benchmark_insights.collect_benchmark_runs(tmp_path)

    for mode, run in cmp.runs.items():
        sidecar = plugin.collect(run)
        assert sidecar.structure is not None
        assert sidecar.structure.score == structure_score(bundles[mode])
        assert sidecar.sdk_activity is not None


# --- (c) build_comparison_evidence maps collect_benchmark_runs --------------


def test_build_comparison_evidence_maps_fields(tmp_path):
    build_result_root(tmp_path)
    bundles = benchmark_insights.collect_benchmark_runs(tmp_path)
    cmp = build_comparison_evidence(tmp_path)

    assert isinstance(cmp, ComparisonEvidence)
    assert cmp.schema_version == SCHEMA_VERSION
    assert cmp.modes == list(bundles.keys())
    assert set(cmp.runs) == set(bundles)

    for mode, run in cmp.runs.items():
        bundle = bundles[mode]
        assert isinstance(run, RunEvidence)
        assert run.mode == bundle["mode"]
        assert run.label == bundle["label"]
        assert run.available == bundle["available"]
        assert run.agent == bundle["agent"]
        assert run.agent_model == bundle["agent_model"]
        assert run.summary == bundle["run"]
        assert run.record == bundle["record"]
        assert run.workspace_delta == bundle["workspace_delta"]
        assert run.validation_metric == bundle["validation_metric"]
        # Structured captured-text artifacts (B5) map from the bundle.
        assert run.agent_last_message == (bundle.get("agent_last_message") or "")
        assert run.agent_events_text == (bundle.get("agent_events_text") or "")
        # raw keeps the original per-run dict accessible for delegation.
        assert run.raw == bundle


def test_build_comparison_evidence_carries_sdk_metadata(tmp_path):
    from benchmark.harness import profile_metadata

    build_result_root(tmp_path)
    # The fixture stamps the §4.3 identity block; ComparisonEvidence carries it
    # (Contract A).
    sdk_metadata = build_comparison_evidence(tmp_path).sdk_metadata
    assert sdk_metadata["report_plugin_id"] == "nvflare"
    assert sdk_metadata["schema_version"] == 1
    # A tree with no §4.3 block -> empty sdk_metadata.
    (tmp_path / profile_metadata.ROOT_DESCRIPTOR_FILENAME).unlink()
    assert build_comparison_evidence(tmp_path).sdk_metadata == {}


def test_run_evidence_is_frozen(tmp_path):
    build_result_root(tmp_path)
    cmp = build_comparison_evidence(tmp_path)
    run = next(iter(cmp.runs.values()))
    import dataclasses

    try:
        run.mode = "changed"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:  # pragma: no cover - guards against a non-frozen dataclass
        raise AssertionError("RunEvidence must be frozen")
