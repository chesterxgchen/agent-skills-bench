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

"""Migration step 6: declarative EvidenceCaptureSpec + un-hardcoded Stage-3 capture."""

from __future__ import annotations

from pathlib import Path

from benchmark.harness.artifacts import capture_workspace_delta, write_workspace_baseline
from benchmark.harness.common import load_json
from benchmark.harness.sdks.capture_spec import EvidenceCaptureSpec, capture_spec_from_metadata, resolve_capture_spec
from benchmark.harness.sdks.nvflare.capture import NVFLARE_CAPTURE_SPEC


def _final_structure_names(manifest_path: Path) -> set[str]:
    manifest = load_json(manifest_path, {}) or {}
    return {Path(entry["path"]).name for entry in manifest.get("final_structure_files", [])}


def _capture(tmp_path: Path, structure_file_names) -> Path:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "client.py").write_text("x = 1\n", encoding="utf-8")
    (workspace / "model.py").write_text("y = 2\n", encoding="utf-8")
    (workspace / "helper.py").write_text("z = 3\n", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    write_workspace_baseline(workspace, baseline)
    manifest = tmp_path / "workspace_delta_manifest.json"
    capture_workspace_delta(
        workspace,
        baseline,
        tmp_path / "delta",
        manifest,
        tmp_path / "runtime",
        delta_scope="agent_workspace",
        include_runtime_artifacts=False,
        structure_file_names=structure_file_names,
    )
    return manifest


# --- capture_workspace_delta honors the structure_file_names param ----------


def test_capture_flags_structure_files_from_spec(tmp_path):
    manifest = _capture(tmp_path, ("client.py", "model.py", "job.py"))
    assert _final_structure_names(manifest) == {"client.py", "model.py"}


def test_capture_flags_nothing_with_empty_spec(tmp_path):
    manifest = _capture(tmp_path, ())
    assert _final_structure_names(manifest) == set()


# --- EvidenceCaptureSpec serialization + resolution -------------------------


def test_evidence_capture_spec_roundtrip():
    spec = EvidenceCaptureSpec(
        structure_file_names=("client.py", "job.py"),
        runtime_sources=(("runtime_workspaces", "/tmp/nvflare/workspaces"),),
    )
    assert EvidenceCaptureSpec.from_payload(spec.to_payload()) == spec


def test_resolve_capture_spec_nvflare_and_fallbacks():
    assert resolve_capture_spec("nvflare") == NVFLARE_CAPTURE_SPEC
    # absent name -> NVFLARE legacy fallback; unknown SDK -> empty.
    assert resolve_capture_spec(None) == NVFLARE_CAPTURE_SPEC
    assert resolve_capture_spec("other") == EvidenceCaptureSpec()


def test_nvflare_spec_matches_prior_hardcoded_values():
    assert NVFLARE_CAPTURE_SPEC.structure_file_names == (
        "client.py",
        "model.py",
        "job.py",
        "prepare_data.py",
        "download_data.py",
    )
    assert NVFLARE_CAPTURE_SPEC.runtime_sources == (("runtime_workspaces", "/tmp/nvflare/workspaces"),)


def test_capture_spec_from_metadata_prefers_serialized_then_sdk_name():
    serialized = EvidenceCaptureSpec(structure_file_names=("a.py",))
    assert capture_spec_from_metadata({"capture_spec": serialized.to_payload()}) == serialized
    # No serialized spec -> resolve by sdk_name (legacy image).
    assert capture_spec_from_metadata({"sdk_name": "nvflare"}) == NVFLARE_CAPTURE_SPEC
    # Neither -> legacy NVFLARE fallback.
    assert capture_spec_from_metadata({}) == NVFLARE_CAPTURE_SPEC


# --- S1: artifact_globs round-trip + NVFLARE carries them -------------------


def test_artifact_globs_roundtrip():
    spec = EvidenceCaptureSpec(
        structure_file_names=("client.py",),
        runtime_sources=(("runtime_workspaces", "/tmp/nvflare/workspaces"),),
        artifact_globs=("**/*.log", "**/config_fed_*.json"),
    )
    restored = EvidenceCaptureSpec.from_payload(spec.to_payload())
    assert restored == spec
    assert restored.artifact_globs == ("**/*.log", "**/config_fed_*.json")


def test_from_payload_ignores_non_string_globs():
    payload = {"artifact_globs": ["**/*.log", 7, None, "**/x.json"]}
    assert EvidenceCaptureSpec.from_payload(payload).artifact_globs == ("**/*.log", "**/x.json")


def test_nvflare_spec_carries_artifact_globs():
    assert NVFLARE_CAPTURE_SPEC.artifact_globs == (
        "**/log.txt",
        "**/*.log",
        "**/config_fed_*.json",
    )


# --- S2: capture-spec versioning + degrade ----------------------------------


def test_payload_carries_version():
    assert EvidenceCaptureSpec().to_payload()["version"] == 1


def test_legacy_payload_without_version_is_v1():
    payload = {"structure_file_names": ["a.py"], "runtime_sources": [], "artifact_globs": []}
    spec = EvidenceCaptureSpec.from_payload(payload)
    assert spec.version == 1
    assert spec.structure_file_names == ("a.py",)


def test_capture_spec_from_metadata_degrades_on_future_version():
    future = EvidenceCaptureSpec(
        structure_file_names=("client.py", "model.py"),
        runtime_sources=(("runtime_workspaces", "/tmp/nvflare/workspaces"),),
        artifact_globs=("**/*.log",),
        version=2,
    )
    degraded = capture_spec_from_metadata({"capture_spec": future.to_payload()})
    # Structure-only safe minimal spec: drops runtime sources + globs.
    assert degraded == EvidenceCaptureSpec(structure_file_names=("client.py", "model.py"))
    assert degraded.runtime_sources == ()
    assert degraded.artifact_globs == ()


def test_capture_spec_from_metadata_known_version_proceeds_normally():
    spec = EvidenceCaptureSpec(structure_file_names=("a.py",), version=1)
    assert capture_spec_from_metadata({"capture_spec": spec.to_payload()}) == spec


# --- S1: agent_run glob resolution to runtime sources -----------------------


def test_resolve_artifact_glob_sources(tmp_path):
    from benchmark.harness.container.agent_run import resolve_artifact_glob_sources

    workspace = tmp_path / "ws"
    sim = workspace / "simulate_job" / "app_site-1"
    sim.mkdir(parents=True)
    (sim / "log.txt").write_text("ran\n", encoding="utf-8")
    (sim / "config_fed_client.json").write_text("{}\n", encoding="utf-8")
    (workspace / "noise.txt").write_text("x\n", encoding="utf-8")

    sources = resolve_artifact_glob_sources(
        ("**/log.txt", "**/config_fed_*.json"),
        workspace,
    )
    # Both globs match files in the same dir -> a single deduped (label, dir) source.
    assert sources == [("simulate_job/app_site-1", sim)]


def test_resolve_artifact_glob_sources_dedupes_against_existing(tmp_path):
    from benchmark.harness.container.agent_run import resolve_artifact_glob_sources

    workspace = tmp_path / "ws"
    sub = workspace / "logs"
    sub.mkdir(parents=True)
    (sub / "a.log").write_text("x\n", encoding="utf-8")

    sources = resolve_artifact_glob_sources(
        ("**/*.log",),
        workspace,
        existing_sources=[("prior", sub)],
    )
    assert sources == []


def test_resolve_artifact_glob_sources_empty_when_no_globs(tmp_path):
    from benchmark.harness.container.agent_run import resolve_artifact_glob_sources

    workspace = tmp_path / "ws"
    workspace.mkdir()
    assert resolve_artifact_glob_sources((), workspace) == []
