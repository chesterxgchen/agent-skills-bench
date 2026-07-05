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


def test_capture_includes_content_of_unchanged_structure_files(tmp_path):
    # baseline == workspace, so NOTHING is "changed" — the structure file is
    # reused as-is (the nested-conversion case: FL wrapper added, train.py kept).
    # Its CONTENT must still be captured so quality detectors can scan it.
    manifest_path = _capture(tmp_path, ("client.py",))
    manifest = load_json(manifest_path, {})
    assert manifest.get("changed_files") == []
    entry = next(e for e in manifest["final_structure_files"] if Path(e["path"]).name == "client.py")
    assert entry.get("artifact_path"), "unchanged structure file must have captured content"
    content = (tmp_path / "delta" / entry["artifact_path"]).read_text(encoding="utf-8")
    assert content == "x = 1\n"


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
        # train.py: capture the reused training source so quality detectors can
        # scan it even when a nested conversion leaves it unchanged.
        "train.py",
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


# --- runtime_source_globs: private run roots outside the workspace ----------


def test_runtime_source_globs_roundtrip():
    spec = EvidenceCaptureSpec(
        runtime_sources=(("runtime_workspaces", "/tmp/nvflare/workspaces"),),
        runtime_source_globs=("/tmp/nvflare-*/run-*",),
    )
    restored = EvidenceCaptureSpec.from_payload(spec.to_payload())
    assert restored == spec
    assert restored.runtime_source_globs == ("/tmp/nvflare-*/run-*",)


def test_from_payload_ignores_non_string_runtime_source_globs():
    payload = {"runtime_source_globs": ["/tmp/nvflare-*/run-*", 7, None]}
    assert EvidenceCaptureSpec.from_payload(payload).runtime_source_globs == ("/tmp/nvflare-*/run-*",)


def test_legacy_payload_without_runtime_source_globs_is_empty():
    assert EvidenceCaptureSpec.from_payload({"artifact_globs": []}).runtime_source_globs == ()


# --- runtime_output_markers: find the run folder by output structure ---------


def test_runtime_output_markers_roundtrip():
    spec = EvidenceCaptureSpec(
        structure_file_names=("client.py",),
        runtime_output_markers=("**/simulate_job/metrics/metrics_summary.json", "**/config_fed_server.json"),
    )
    restored = EvidenceCaptureSpec.from_payload(spec.to_payload())
    assert restored == spec
    assert restored.runtime_output_markers == (
        "**/simulate_job/metrics/metrics_summary.json",
        "**/config_fed_server.json",
    )


def test_from_payload_ignores_non_string_runtime_output_markers():
    payload = {"runtime_output_markers": ["**/a.json", 7, None, "**/b.json"]}
    assert EvidenceCaptureSpec.from_payload(payload).runtime_output_markers == ("**/a.json", "**/b.json")


def test_legacy_payload_without_runtime_output_markers_is_empty():
    assert EvidenceCaptureSpec.from_payload({"artifact_globs": []}).runtime_output_markers == ()


def test_nvflare_spec_finds_run_folder_by_output_structure_not_path():
    # The run/export folder location is not the harness's to assume, so the spec
    # must NOT bake in an absolute path prefix; it identifies the run root by its
    # OUTPUT STRUCTURE (metrics_summary.json etc.) wherever it landed.
    assert NVFLARE_CAPTURE_SPEC.runtime_source_globs == ()
    assert "**/simulate_job/metrics/metrics_summary.json" in NVFLARE_CAPTURE_SPEC.runtime_output_markers
    assert all(not m.startswith("/") for m in NVFLARE_CAPTURE_SPEC.runtime_output_markers)


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


# --- runtime_source_globs resolution: private run roots ----------------------


def test_resolve_runtime_source_glob_dirs_matches_run_roots(tmp_path):
    from benchmark.harness.container.agent_run import resolve_runtime_source_glob_dirs

    run_a = tmp_path / "nvflare-9999" / "run-abc123"
    run_b = tmp_path / "nvflare-9999" / "run-def456"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    (run_a / "run-manifest.json").write_text("{}\n", encoding="utf-8")
    # A stray FILE matching the pattern must not become a source.
    (tmp_path / "nvflare-9999" / "run-notadir").write_text("x\n", encoding="utf-8")

    sources = resolve_runtime_source_glob_dirs((f"{tmp_path}/nvflare-*/run-*",))
    assert sources == [
        (run_a.as_posix().lstrip("/"), run_a),
        (run_b.as_posix().lstrip("/"), run_b),
    ]


def test_resolve_runtime_source_glob_dirs_dedupes_against_existing(tmp_path):
    from benchmark.harness.container.agent_run import resolve_runtime_source_glob_dirs

    run_dir = tmp_path / "nvflare-9999" / "run-abc123"
    run_dir.mkdir(parents=True)
    sources = resolve_runtime_source_glob_dirs(
        (f"{tmp_path}/nvflare-*/run-*",),
        existing_sources=[("prior", run_dir)],
    )
    assert sources == []


def test_resolve_runtime_source_glob_dirs_rejects_unsafe_patterns(tmp_path):
    from benchmark.harness.container.agent_run import resolve_runtime_source_glob_dirs

    run_dir = tmp_path / "nvflare-9999" / "run-abc123"
    run_dir.mkdir(parents=True)
    # Relative patterns and patterns with ".." segments are ignored.
    assert resolve_runtime_source_glob_dirs(("nvflare-*/run-*",)) == []
    assert resolve_runtime_source_glob_dirs((f"{tmp_path}/nvflare-*/../nvflare-*/run-*",)) == []


def test_resolve_runtime_source_glob_dirs_skips_symlinked_match(tmp_path):
    from benchmark.harness.container.agent_run import resolve_runtime_source_glob_dirs

    real = tmp_path / "elsewhere"
    real.mkdir()
    root = tmp_path / "nvflare-9999"
    root.mkdir()
    (root / "run-linked").symlink_to(real)
    assert resolve_runtime_source_glob_dirs((f"{tmp_path}/nvflare-*/run-*",)) == []


# --- runtime_output_markers resolution: find the run root by structure -------

_NVFLARE_MARKERS = ("**/simulate_job/metrics/metrics_summary.json", "**/config_fed_server.json")


def test_resolve_runtime_output_roots_finds_run_root_by_structure(tmp_path):
    # Reproduces the observed failing layout: the skill ran in a private temp root
    # `/tmp/nvflare-<job>.<rand>/workspace/<job>/server/simulate_job/metrics/...`
    # (no `run-*` segment), so the old absolute glob matched nothing.
    from benchmark.harness.container.agent_run import resolve_runtime_output_roots

    run_root = tmp_path / "nvflare-ames-fedavg.g1o4Nv"
    metrics = run_root / "workspace" / "ames" / "server" / "simulate_job" / "metrics"
    metrics.mkdir(parents=True)
    (metrics / "metrics_summary.json").write_text('{"best_metrics": [{"value": 0.7578}]}\n', encoding="utf-8")

    sources = resolve_runtime_output_roots(_NVFLARE_MARKERS, [tmp_path])
    # The enclosing run root (not the deep metrics dir) becomes the source, so
    # capture walks the whole run folder (metrics, logs, configs, exported job).
    assert sources == [(run_root.as_posix().lstrip("/"), run_root)]


def test_resolve_runtime_output_roots_dedupes_multiple_markers_in_one_root(tmp_path):
    from benchmark.harness.container.agent_run import resolve_runtime_output_roots

    run_root = tmp_path / "nvflare-run.xyz"
    metrics = run_root / "workspace" / "job" / "server" / "simulate_job" / "metrics"
    metrics.mkdir(parents=True)
    (metrics / "metrics_summary.json").write_text("{}\n", encoding="utf-8")
    cfg = run_root / "workspace" / "job" / "app_server" / "config"
    cfg.mkdir(parents=True)
    (cfg / "config_fed_server.json").write_text("{}\n", encoding="utf-8")

    # Two different markers match under the same run root -> one source, not two.
    sources = resolve_runtime_output_roots(_NVFLARE_MARKERS, [tmp_path])
    assert sources == [(run_root.as_posix().lstrip("/"), run_root)]


def test_resolve_runtime_output_roots_dedupes_against_existing(tmp_path):
    from benchmark.harness.container.agent_run import resolve_runtime_output_roots

    run_root = tmp_path / "nvflare-run.xyz"
    metrics = run_root / "server" / "simulate_job" / "metrics"
    metrics.mkdir(parents=True)
    (metrics / "metrics_summary.json").write_text("{}\n", encoding="utf-8")
    assert resolve_runtime_output_roots(_NVFLARE_MARKERS, [tmp_path], existing_sources=[("prior", run_root)]) == []


def test_resolve_runtime_output_roots_prefers_most_specific_nested_root(tmp_path):
    # When a designated root is nested under a broader temp root (e.g.
    # BENCHMARK_JOB_RUN_DIR under /tmp), the same marker resolves to a specific
    # run root via the inner root and to its ancestor via the outer root. Only the
    # specific one must be kept, or the ancestor would drag in sibling runs.
    from benchmark.harness.container.agent_run import resolve_runtime_output_roots

    designated = tmp_path / "designated"
    run_root = designated / "nvflare-run.xyz"
    metrics = run_root / "workspace" / "job" / "server" / "simulate_job" / "metrics"
    metrics.mkdir(parents=True)
    (metrics / "metrics_summary.json").write_text("{}\n", encoding="utf-8")
    # A sibling run under the same designated dir must NOT be captured via an
    # ancestor match.
    sibling = designated / "nvflare-other.abc" / "server" / "simulate_job" / "metrics"
    sibling.mkdir(parents=True)
    (sibling / "metrics_summary.json").write_text("{}\n", encoding="utf-8")

    # Inner (designated) searched first, then the broader ancestor (tmp_path).
    sources = resolve_runtime_output_roots(_NVFLARE_MARKERS, [designated, tmp_path])
    roots = {run_root for _label, run_root in sources}
    assert roots == {designated / "nvflare-run.xyz", designated / "nvflare-other.abc"}
    # The ancestor `designated` itself must never be returned as a run root.
    assert all(run_root.name.startswith("nvflare-") for _label, run_root in sources)


def test_runtime_output_search_roots_excludes_workspace_and_dedupes(tmp_path, monkeypatch):
    from benchmark.harness.container.agent_run import runtime_output_search_roots

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()
    designated = tmp_path / "designated"
    designated.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    monkeypatch.setenv("BENCHMARK_JOB_RUN_DIR", str(designated))

    roots = runtime_output_search_roots(workspace)
    resolved = {r.resolve() for r in roots}
    assert designated.resolve() in resolved and tmpdir.resolve() in resolved
    # The workspace is captured by the delta already; it must not be searched.
    assert workspace.resolve() not in resolved


def test_in_workspace_run_root_is_runtime_evidence_not_generated_source(tmp_path):
    """A skill-mandated run root INSIDE the workspace splits runtime from source.

    Regression: the agent runs the simulation under `<workspace>/nvflare_runtime/`
    (per the skill's instructions), and the workspace source walk swept the whole
    run tree into changed_files/ — misreporting simulator output as agent-generated
    source. The run root must be discovered by its output structure (simulate_job),
    captured wholesale under runtime_artifacts/, and excluded from changed_files.
    """

    import json

    from benchmark.harness.container.agent_run import resolve_runtime_output_roots

    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "client.py").write_text("x = 1\n", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    write_workspace_baseline(workspace, baseline)

    # The agent's actual source change...
    (workspace / "job.py").write_text("job = True\n", encoding="utf-8")
    # ...and the skill-mandated in-workspace run root with simulator output.
    run_root = workspace / "nvflare_runtime"
    meta = run_root / "simulation" / "job" / "site-1" / "simulate_job" / "meta.json"
    meta.parent.mkdir(parents=True)
    meta.write_text("{}", encoding="utf-8")
    server_log = run_root / "simulation" / "job" / "server" / "log.txt"
    server_log.parent.mkdir(parents=True)
    server_log.write_text("new best validation metric at round 1: 0.7\n", encoding="utf-8")
    exported_config = run_root / "job_config" / "app" / "config" / "config_fed_server.json"
    exported_config.parent.mkdir(parents=True)
    exported_config.write_text("{}", encoding="utf-8")

    run_output_markers = tuple(
        marker for marker in NVFLARE_CAPTURE_SPEC.runtime_output_markers if "simulate_job" in marker
    )
    workspace_runtime_sources = [
        (root.relative_to(workspace).as_posix(), root)
        for _label, root in resolve_runtime_output_roots(run_output_markers, [workspace])
    ]
    assert [label for label, _root in workspace_runtime_sources] == ["nvflare_runtime"]

    manifest_path = tmp_path / "workspace_delta_manifest.json"
    capture_workspace_delta(
        workspace,
        baseline,
        tmp_path / "delta",
        manifest_path,
        tmp_path / "runtime",
        delta_scope="agent_workspace",
        extra_runtime_artifact_sources=workspace_runtime_sources,
        structure_file_names=("client.py", "job.py"),
        exclude_source_dirs=[root for _label, root in workspace_runtime_sources],
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    changed = [entry["path"] for entry in manifest["changed_files"]]
    assert changed == ["job.py"], f"run-root files leaked into changed_files: {changed}"
    assert all(not p.startswith("nvflare_runtime/") for p in [e["path"] for e in manifest["final_files"]])
    runtime_paths = [entry["path"] for entry in manifest["runtime_artifacts"]]
    assert "nvflare_runtime/simulation/job/server/log.txt" in runtime_paths
    assert "nvflare_runtime/simulation/job/site-1/simulate_job/meta.json" in runtime_paths
    assert "nvflare_runtime/job_config/app/config/config_fed_server.json" in runtime_paths
