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

import json
from pathlib import Path
from types import SimpleNamespace


def test_docker_build_args_reject_embedded_equals():
    from benchmark.harness.host.build import render_agent_build_args

    try:
        render_agent_build_args({"AGENT_CLI_VERSION": "1.0=bad"})
    except ValueError as exc:
        assert "must not contain '='" in str(exc)
    else:
        raise AssertionError("Docker build arg values with embedded '=' should fail before docker build")

    try:
        render_agent_build_args({"AGENT=CLI_VERSION": "1.0"})
    except ValueError as exc:
        assert "key must not contain '='" in str(exc)
    else:
        raise AssertionError("Docker build arg keys with embedded '=' should fail before docker build")


def test_prepare_build_context_cleans_temp_dir_on_internal_failure(tmp_path, monkeypatch):
    from benchmark.harness.host import build

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(build, "copy_harness", lambda _src, _dst: (_ for _ in ()).throw(RuntimeError("copy failed")))

    try:
        build.prepare_build_context()
    except RuntimeError as exc:
        assert "copy failed" in str(exc)
    else:
        raise AssertionError("prepare_build_context should propagate internal failures")

    assert list(tmp_path.iterdir()) == []


def test_prepare_build_context_stages_benchmark_package_for_docker_copy(tmp_path, monkeypatch):
    from benchmark.harness.host import build

    monkeypatch.setenv("TMPDIR", str(tmp_path))

    context = build.prepare_build_context()
    try:
        assert (context / "config" / "agents" / "codex.yaml").is_file()
        assert (context / "benchmark" / "__init__.py").is_file()
        assert (context / "benchmark" / "harness" / "container" / "agent_run.py").is_file()
        dockerignore = (context / ".dockerignore").read_text(encoding="utf-8")
        assert "!config/" in dockerignore
        assert "!benchmark/" in dockerignore
    finally:
        build.shutil.rmtree(context, ignore_errors=True)


def test_build_copytree_calls_dereference_symlinks(tmp_path, monkeypatch):
    from benchmark.harness.host import build

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    src.joinpath("package").mkdir()
    calls = []

    def fake_copytree(source, target, **kwargs):
        calls.append((Path(source).name, Path(target).name, kwargs))

    monkeypatch.setattr(build.shutil, "copytree", fake_copytree)

    build.copy_directory_contents(src, dst)
    build.copy_harness(src / "package", dst / "package")

    assert [(source, target) for source, target, _kwargs in calls] == [("package", "package"), ("package", "package")]
    assert all(kwargs["symlinks"] is False for _source, _target, kwargs in calls)
    assert all(callable(kwargs["ignore"]) for _source, _target, kwargs in calls)


def test_write_wheel_metadata_uses_atomic_json_helper(tmp_path, monkeypatch):
    from benchmark.harness.host import build

    wheel = tmp_path / "example-1.0.0-py3-none-any.whl"
    wheel.write_text("wheel\n", encoding="utf-8")
    calls = []

    def fake_write_json(path, payload):
        calls.append((Path(path), payload))

    monkeypatch.setattr(build, "write_json", fake_write_json)

    build.write_wheel_metadata(
        sdk=SimpleNamespace(
            build_env_name="EXAMPLE_BUILD",
            import_name="example",
            metadata=lambda: {"name": "example"},
            name="example",
            package_name="example",
        ),
        variant=SimpleNamespace(build_env_value="1", name="skills"),
        wheel_build=SimpleNamespace(build_type="uv_wheel"),
        prepared=build.PreparedSdkWheel(wheel=wheel, source_type="wheel", source_path=tmp_path),
        out_dir=tmp_path / "out",
        evaluation=build.PreparedEvaluationCriteria(
            source_path=tmp_path / "rules.yaml",
            source_type="explicit",
            sha256="criteria-hash",
            staged_entrypoint="rules.yaml",
        ),
    )

    assert calls[0][0] == tmp_path / "out" / "sdk_wheel_metadata.json"
    payload = calls[0][1]
    assert payload["filename"] == wheel.name
    assert payload["sdk_name"] == "example"
    # §4.3 versioned identity/metadata block stamped at build time.
    assert payload["schema_version"] == 1
    assert payload["capture_spec_version"] == "1"
    assert payload["benchmark_profile_id"] == "example"
    assert payload["report_plugin_id"] == "example"
    # §4.1 declarative capture spec serialized at build time (empty for a
    # non-NVFLARE SDK — no product-specific capture leaks into generic code).
    assert payload["capture_spec"] == {
        "version": 1,
        "structure_file_names": [],
        "runtime_sources": [],
        "artifact_globs": [],
        "runtime_source_globs": [],
        "runtime_output_markers": [],
    }
    assert payload["evaluation_criteria"]["sha256"] == "criteria-hash"
    assert payload["evaluation_criteria"]["source_format"] == "harness_yaml"


def test_sdk_repo_evaluation_criteria_are_validated_staged_and_hashed(tmp_path):
    from benchmark.harness.host import build
    from benchmark.harness.sdks.base import SdkSource
    from benchmark.harness.sdks.config import ConfigurableSdkAdapter

    repo = tmp_path / "sdk"
    criteria = repo / "quality" / "rules.yaml"
    criteria.parent.mkdir(parents=True)
    criteria.write_text(
        """
schema_version: 1
sdk: example
task: conversion
default_verdict: caution
signals:
  structure:
    label: Structure
    rules:
      - contains: complete
        verdict: good
""".lstrip(),
        encoding="utf-8",
    )
    profile = tmp_path / "sdk.yaml"
    profile.write_text(
        """
name: example
display_name: Example
package_name: example
import_name: example
source:
  type: repo
  path: "{repo_root}"
  markers: [pyproject.toml]
build:
  type: uv_wheel
  variants:
    skills: {wheel_globs: [example-*.whl]}
    baseline: {wheel_globs: [example-baseline-*.whl]}
docker: {}
skills:
  setup: {type: none}
evaluation:
  criteria_path: quality/rules.yaml
""".lstrip(),
        encoding="utf-8",
    )
    sdk = ConfigurableSdkAdapter(profile)
    context = tmp_path / "context"
    context.mkdir()

    prepared = build.resolve_and_stage_evaluation_criteria(
        sdk=sdk,
        source=SdkSource(source_type="repo", repo_path=repo),
        explicit_path=None,
        context=context,
    )

    assert prepared.source_type == "sdk_repo"
    assert prepared.source_path == criteria
    assert prepared.sha256 == build.path_tree_sha256(criteria)
    assert prepared.staged_entrypoint == "rules.yaml"
    assert (context / "evaluation_rules" / "rules.yaml").read_text(encoding="utf-8") == criteria.read_text(
        encoding="utf-8"
    )


def test_nvflare_sdk_native_skill_evals_are_converted_to_rules(tmp_path):
    from benchmark.harness.evaluation import load_evaluation_rules
    from benchmark.harness.host import build
    from benchmark.harness.sdks.base import SdkSource

    repo = tmp_path / "NVFlare"
    evals_dir = repo / "dev_tools" / "agent" / "skill_evals" / "nvflare-convert-pytorch"
    evals_dir.mkdir(parents=True)
    evals_dir.joinpath("evals.json").write_text(
        json.dumps(
            {
                "skill_name": "nvflare-convert-pytorch",
                "evals": [
                    {
                        "id": "pytorch-basic",
                        "nvflare": {
                            "mandatory_behavior": [
                                {
                                    "id": "inspect-first",
                                    "description": "runs nvflare agent inspect before editing",
                                }
                            ],
                            "prohibited_behavior": [
                                {
                                    "id": "no-poc-production-submit",
                                    "description": "does not submit to POC or production",
                                }
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    context = tmp_path / "context"
    context.mkdir()
    sdk = SimpleNamespace(
        name="nvflare",
        evaluation_criteria=lambda: SimpleNamespace(repo_relative_path=Path("dev_tools/agent/skill_evals")),
    )

    prepared = build.resolve_and_stage_evaluation_criteria(
        sdk=sdk,
        source=SdkSource(source_type="repo", repo_path=repo),
        explicit_path=None,
        context=context,
    )

    staged = context / "evaluation_rules"
    rules = load_evaluation_rules("nvflare", staged, task="conversion", framework="pytorch")
    signals = rules["signals"]
    assert prepared.source_type == "sdk_repo"
    assert prepared.source_format == "nvflare_skill_evals"
    assert prepared.staged_entrypoint == "."
    assert "mandatory_behavior__inspect-first" in signals
    assert signals["mandatory_behavior__inspect-first"]["native_behavior"]["cases"] == ["pytorch-basic"]
    assert (staged / "native" / "nvflare_skill_evals" / "nvflare-convert-pytorch" / "evals.json").is_file()


def test_sdk_profile_rejects_evaluation_path_escape(tmp_path):
    from benchmark.harness.sdks.config import ConfigurableSdkAdapter

    profile = tmp_path / "sdk.yaml"
    source = (Path(__file__).resolve().parents[1] / "config" / "sdks" / "nvflare-profile.yaml").read_text(
        encoding="utf-8"
    )
    profile.write_text(source + "\nevaluation:\n  criteria_path: ../outside.yaml\n", encoding="utf-8")

    try:
        ConfigurableSdkAdapter(profile)
    except ValueError as exc:
        assert "evaluation.criteria_path must be relative" in str(exc)
    else:
        raise AssertionError("evaluation criteria must not escape the SDK repo")


def test_build_main_routes_prepared_evaluation_to_both_image_metadata(tmp_path, monkeypatch):
    from benchmark.harness.host import build
    from benchmark.harness.sdks.base import SdkSkillsSetup, SdkSource

    rules = tmp_path / "rules.yaml"
    rules.write_text("schema_version: 1\nsdk: example\nsignals: {}\n", encoding="utf-8")
    wheel = tmp_path / "example.whl"
    wheel.write_bytes(b"wheel")
    sdk = SimpleNamespace(
        name="example",
        display_name="Example SDK",
        package_name="example",
        build_env_name="",
        wheel_build=lambda: SimpleNamespace(build_type="uv_wheel"),
        wheel_variant=lambda name: SimpleNamespace(name=name, label=name, build_env_value=""),
        docker_build_args=lambda: {},
    )
    adapter = SimpleNamespace(
        name="agent",
        display_name="Agent",
        availability_probe=[],
        image_targets=lambda: SimpleNamespace(skills="skills", baseline="baseline", report="report"),
        build_args=lambda *, cli_version="": {},
    )
    monkeypatch.setattr(build, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(build, "load_sdk_profile", lambda _profile: sdk)
    monkeypatch.setattr(build, "load_agent_profile", lambda _profile: adapter)
    monkeypatch.setattr(build, "resolve_sdk_source", lambda _sdk: SdkSource(source_type="repo", repo_path=tmp_path))
    monkeypatch.setattr(build, "resolve_sdk_skills_setup", lambda _sdk: SdkSkillsSetup(setup_type="none"))
    monkeypatch.setattr(build, "stage_sdk_skills_setup", lambda *_args: None)
    monkeypatch.setattr(build, "docker_build", lambda **_kwargs: None)

    def fake_prepare_sdk_wheel(*, source, wheel_build, sdk, variant, out_dir, rebuild=False):
        return build.PreparedSdkWheel(wheel=wheel, source_type=source.source_type, source_path=source.repo_path)

    metadata_calls = []
    monkeypatch.setattr(build, "prepare_sdk_wheel", fake_prepare_sdk_wheel)
    monkeypatch.setattr(build, "write_wheel_metadata", lambda **kwargs: metadata_calls.append(kwargs))

    assert build.main(["--evaluation-criteria", str(rules)]) == 0
    assert len(metadata_calls) == 2
    assert {call["variant"].name for call in metadata_calls} == {"skills", "baseline"}
    assert all(call["evaluation"].source_path == rules for call in metadata_calls)


def test_latest_sdk_wheel_skips_stat_failures(tmp_path, monkeypatch):
    from benchmark.harness.host import build

    stale = tmp_path / "nvflare-0.0.1-py3-none-any.whl"
    current = tmp_path / "nvflare-0.0.2-py3-none-any.whl"
    baseline = tmp_path / "nvflare-0.0.3-no_skills-py3-none-any.whl"
    stale.write_text("stale\n", encoding="utf-8")
    current.write_text("current\n", encoding="utf-8")
    baseline.write_text("baseline\n", encoding="utf-8")
    original_stat = type(stale).stat

    def flaky_stat(path):
        if path == stale:
            raise OSError("wheel disappeared")
        return original_stat(path)

    monkeypatch.setattr(type(stale), "stat", flaky_stat)

    assert build.latest_sdk_wheel(tmp_path, ("nvflare-*.whl", "nvflare_nightly-*.whl"), ("*no_skills*.whl",)) == current


def test_nvflare_sdk_adapter_loads_build_contract():
    from benchmark.harness.sdks.registry import load_sdk_adapter, supported_sdk_names

    sdk = load_sdk_adapter("nvflare-profile")
    skills = sdk.wheel_variant("skills")
    baseline = sdk.wheel_variant("baseline")
    build_args = sdk.docker_build_args()
    source = sdk.source(repo_root=Path(__file__).resolve().parents[3], home=Path.home())

    assert sdk.name == "nvflare"
    assert "nvflare-profile" in supported_sdk_names()
    assert sdk.wheel_build().build_type == "uv_wheel"
    assert source.source_type == "repo"
    assert source.repo_markers == ("pyproject.toml", "nvflare/")
    # NVFLARE PR #4837 removed the wheel-bundling toggle: no build env, and both
    # variants stage the SAME wheel — the A/B distinction is purely skills.setup.
    assert sdk.build_env_name == ""
    assert skills.build_env_value == ""
    assert skills.reuse_existing is True
    assert skills.wheel_exclude_globs == ()
    assert baseline.reuse_existing is True
    assert baseline.wheel_globs == skills.wheel_globs == ("nvflare-*.whl", "nvflare_nightly-*.whl")
    assert sdk.evaluation_criteria().repo_relative_path == Path("dev_tools/agent/skill_evals")
    # NVFLARE dropped `nvflare agent skills install`; skills come from the
    # Agent Skills CLI against a configurable git source.
    assert 'npx --yes skills add "${SKILLS_SOURCE_REF}"' in build_args["SKILLS_INSTALL_COMMAND"]
    assert "-a claude-code" in build_args["SKILLS_INSTALL_COMMAND"]
    # Remote-relative dev ref; build-time resolution maps it to the checkout's origin URL.
    assert build_args["SKILLS_SOURCE_REF"] == "origin#milestone8-agent-skills"
    assert build_args["SKILLS_INSTALL_EXPECTED_SOURCE"] == "skills_cli"
    # No SDK list CLI anymore: the harness's generic lister walks the target.
    assert build_args["SKILLS_LIST_COMMAND"] == ""


def test_configurable_sdk_adapter_loads_non_nvflare_contract(tmp_path):
    from benchmark.harness.sdks.config import ConfigurableSdkAdapter

    config_path = tmp_path / "example_sdk.yaml"
    config_path.write_text(
        """
name: example
display_name: Example SDK
package_name: example-sdk
import_name: example_sdk
source:
  type: repo
  path: "{repo_root}"
  markers:
    - pyproject.toml
build:
  type: uv_wheel
  env_name: EXAMPLE_PACKAGE_SKILLS
  variants:
    skills:
      label: skills
      build_env_value: "with"
      wheel_globs:
        - example_sdk-*.whl
    baseline:
      label: baseline
      build_env_value: "without"
      wheel_globs:
        - example_sdk_baseline-*.whl
docker:
  version_command: example-sdk --version
skills:
  setup:
    type: command
    install_command: example-sdk skills install --agent "${BENCHMARK_DOCKER_AGENT}" --target "${BENCHMARK_AGENT_HOME}/skills"
    list_command: example-sdk skills list --agent "${BENCHMARK_DOCKER_AGENT}" --target "${BENCHMARK_AGENT_HOME}/skills"
    install_output: skills_build_install.json
    list_output: skills_list.json
    expected_source: local_sdk_wheel
""".lstrip(),
        encoding="utf-8",
    )

    sdk = ConfigurableSdkAdapter(config_path)
    skills = sdk.wheel_variant("skills")
    baseline = sdk.wheel_variant("baseline")
    build_args = sdk.docker_build_args()
    source = sdk.source(repo_root=tmp_path, home=tmp_path)

    assert sdk.name == "example"
    assert sdk.package_name == "example-sdk"
    assert sdk.import_name == "example_sdk"
    assert sdk.wheel_build().build_type == "uv_wheel"
    assert source.source_type == "repo"
    assert source.repo_path == tmp_path
    assert sdk.build_env_name == "EXAMPLE_PACKAGE_SKILLS"
    assert skills.build_env_value == "with"
    assert baseline.build_env_value == "without"
    assert skills.reuse_existing is True
    assert baseline.reuse_existing is True
    assert build_args["SDK_PACKAGE_NAME"] == "example-sdk"
    assert build_args["SKILLS_SETUP_TYPE"] == "command"
    assert build_args["SKILLS_INSTALL_OUTPUT"] == "skills_build_install.json"


def test_configurable_sdk_adapter_reports_unknown_source_placeholder(tmp_path):
    from benchmark.harness.sdks.config import ConfigurableSdkAdapter

    config_path = tmp_path / "example_sdk.yaml"
    config_path.write_text(
        """
name: example
display_name: Example SDK
package_name: example-sdk
import_name: example_sdk
source:
  type: repo
  path: "{missing_root}"
  markers:
    - pyproject.toml
build:
  type: uv_wheel
  variants:
    skills:
      wheel_globs:
        - example_sdk-*.whl
    baseline:
      wheel_globs:
        - example_sdk_baseline-*.whl
docker:
  version_command: example-sdk --version
skills:
  setup:
    type: none
""".lstrip(),
        encoding="utf-8",
    )

    sdk = ConfigurableSdkAdapter(config_path)

    try:
        sdk.source(repo_root=tmp_path, home=tmp_path)
    except ValueError as exc:
        assert str(config_path) in str(exc)
        assert "unknown placeholder {missing_root} in source.path" in str(exc)
    else:
        raise AssertionError("unknown SDK source placeholders should fail with a clear error")


def test_configurable_sdk_adapter_loads_wheel_source_contract(tmp_path):
    from benchmark.harness.sdks.config import ConfigurableSdkAdapter

    skills_wheel = tmp_path / "example_sdk-1.0.0-py3-none-any.whl"
    baseline_wheel = tmp_path / "example_sdk_baseline-1.0.0-py3-none-any.whl"
    skills_wheel.write_bytes(b"skills wheel")
    baseline_wheel.write_bytes(b"baseline wheel")
    config_path = tmp_path / "example_sdk_wheels.yaml"
    config_path.write_text(
        f"""
name: example
display_name: Example SDK
package_name: example-sdk
import_name: example_sdk
source:
  type: wheels
  wheels:
    skills: {skills_wheel}
    baseline: {baseline_wheel}
build:
  type: provided_wheels
  variants:
    skills:
      label: skills
      build_env_value: "with"
      wheel_globs:
        - example_sdk-*.whl
    baseline:
      label: baseline
      build_env_value: "without"
      wheel_globs:
        - example_sdk_baseline-*.whl
docker:
  version_command: example-sdk --version
skills:
  setup:
    type: command
    install_command: example-sdk skills install --agent "${{BENCHMARK_DOCKER_AGENT}}" --target "${{BENCHMARK_AGENT_HOME}}/skills"
    list_command: example-sdk skills list --agent "${{BENCHMARK_DOCKER_AGENT}}" --target "${{BENCHMARK_AGENT_HOME}}/skills"
""".lstrip(),
        encoding="utf-8",
    )

    sdk = ConfigurableSdkAdapter(config_path)
    source = sdk.source(repo_root=tmp_path, home=tmp_path)

    assert source.source_type == "wheels"
    assert sdk.wheel_build().build_type == "provided_wheels"
    assert source.wheel_paths == {"skills": skills_wheel, "baseline": baseline_wheel}


def _repo_uv_wheel_sdk(tmp_path):
    """A minimal repo-source / uv_wheel SDK profile + a fake repo checkout, for the
    reuse-vs-rebuild wheel-build tests."""
    from benchmark.harness.sdks.config import ConfigurableSdkAdapter

    repo = tmp_path / "repo"
    (repo / "dist").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='example-sdk'\n", encoding="utf-8")
    config_path = tmp_path / "example_repo_sdk.yaml"
    config_path.write_text(
        f"""
name: example
display_name: Example SDK
package_name: example-sdk
import_name: example_sdk
source:
  type: repo
  path: {repo}
  markers:
    - pyproject.toml
build:
  type: uv_wheel
  variants:
    skills:
      label: skills
      build_env_value: "with"
      wheel_globs:
        - example_sdk-*.whl
      wheel_exclude_globs:
        - "*_baseline-*.whl"
    baseline:
      label: baseline
      build_env_value: "without"
      wheel_globs:
        - example_sdk-*.whl
docker:
  version_command: example-sdk --version
skills:
  setup:
    type: command
    install_command: example-sdk skills install --agent "${{BENCHMARK_DOCKER_AGENT}}" --target "${{BENCHMARK_AGENT_HOME}}/skills"
    list_command: example-sdk skills list --agent "${{BENCHMARK_DOCKER_AGENT}}" --target "${{BENCHMARK_AGENT_HOME}}/skills"
""".lstrip(),
        encoding="utf-8",
    )
    return ConfigurableSdkAdapter(config_path), repo


def test_build_sdk_wheel_reuses_existing_dist_wheel_by_default(tmp_path, monkeypatch):
    from benchmark.harness.host import build

    sdk, repo = _repo_uv_wheel_sdk(tmp_path)
    existing = repo / "dist" / "example_sdk-1.0.0-py3-none-any.whl"
    existing.write_bytes(b"prebuilt wheel")

    # uv build must NOT be invoked when a wheel already exists and rebuild is not requested.
    monkeypatch.setattr(build.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(
        build.subprocess, "call", lambda *a, **k: (_ for _ in ()).throw(AssertionError("uv build should be skipped"))
    )

    prepared = build.build_sdk_wheel_from_repo(
        repo=repo, sdk=sdk, variant=sdk.wheel_variant("baseline"), out_dir=tmp_path / "staged"
    )
    assert prepared.wheel.name == existing.name
    assert prepared.wheel.read_bytes() == b"prebuilt wheel"


def test_build_sdk_wheel_honors_variant_reuse_existing_false(tmp_path, monkeypatch):
    from dataclasses import replace

    from benchmark.harness.host import build

    sdk, repo = _repo_uv_wheel_sdk(tmp_path)
    existing = repo / "dist" / "example_sdk-1.0.0-py3-none-any.whl"
    existing.write_bytes(b"stale wheel")
    variant = replace(sdk.wheel_variant("baseline"), reuse_existing=False)
    calls = {"n": 0}

    def fake_uv_build(cmd, **kwargs):
        calls["n"] += 1
        out_dir = Path(cmd[-1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "example_sdk-2.0.0-py3-none-any.whl").write_bytes(b"freshly built")
        return 0

    monkeypatch.setattr(build.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(build.subprocess, "call", fake_uv_build)

    prepared = build.build_sdk_wheel_from_repo(repo=repo, sdk=sdk, variant=variant, out_dir=tmp_path / "staged")

    assert calls["n"] == 1
    assert prepared.wheel.name == "example_sdk-2.0.0-py3-none-any.whl"
    assert prepared.wheel.read_bytes() == b"freshly built"


def test_build_sdk_wheel_builds_when_no_wheel_and_when_rebuild_requested(tmp_path, monkeypatch):
    from benchmark.harness.host import build

    sdk, repo = _repo_uv_wheel_sdk(tmp_path)
    calls = {"n": 0}

    def fake_uv_build(cmd, **kwargs):
        calls["n"] += 1
        # Emulate uv writing the wheel into the repo dist/ (the --out-dir).
        out_dir = Path(cmd[-1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "example_sdk-2.0.0-py3-none-any.whl").write_bytes(b"freshly built")
        return 0

    monkeypatch.setattr(build.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(build.subprocess, "call", fake_uv_build)

    # No wheel present -> builds.
    prepared = build.build_sdk_wheel_from_repo(
        repo=repo, sdk=sdk, variant=sdk.wheel_variant("baseline"), out_dir=tmp_path / "s1"
    )
    assert calls["n"] == 1 and prepared.wheel.read_bytes() == b"freshly built"
    # Wheel now exists -> rebuild=True still forces a build.
    build.build_sdk_wheel_from_repo(
        repo=repo, sdk=sdk, variant=sdk.wheel_variant("baseline"), out_dir=tmp_path / "s2", rebuild=True
    )
    assert calls["n"] == 2


def test_wheel_source_stages_configured_wheel_without_repo_build(tmp_path):
    from benchmark.harness.host import build
    from benchmark.harness.sdks.config import ConfigurableSdkAdapter

    skills_wheel = tmp_path / "example_sdk-1.0.0-py3-none-any.whl"
    baseline_wheel = tmp_path / "example_sdk_baseline-1.0.0-py3-none-any.whl"
    skills_wheel.write_bytes(b"skills wheel")
    baseline_wheel.write_bytes(b"baseline wheel")
    config_path = tmp_path / "example_sdk_wheels.yaml"
    config_path.write_text(
        f"""
name: example
display_name: Example SDK
package_name: example-sdk
import_name: example_sdk
source:
  type: wheels
  wheels:
    skills: {skills_wheel}
    baseline: {baseline_wheel}
build:
  type: provided_wheels
  variants:
    skills:
      label: skills
      build_env_value: "with"
      wheel_globs:
        - example_sdk-*.whl
    baseline:
      label: baseline
      build_env_value: "without"
      wheel_globs:
        - example_sdk_baseline-*.whl
docker:
  version_command: example-sdk --version
skills:
  setup:
    type: command
    install_command: example-sdk skills install --agent "${{BENCHMARK_DOCKER_AGENT}}" --target "${{BENCHMARK_AGENT_HOME}}/skills"
    list_command: example-sdk skills list --agent "${{BENCHMARK_DOCKER_AGENT}}" --target "${{BENCHMARK_AGENT_HOME}}/skills"
""".lstrip(),
        encoding="utf-8",
    )
    sdk = ConfigurableSdkAdapter(config_path)
    source = build.resolve_sdk_source(sdk)

    prepared = build.prepare_sdk_wheel(
        source=source,
        wheel_build=sdk.wheel_build(),
        sdk=sdk,
        variant=sdk.wheel_variant("skills"),
        out_dir=tmp_path / "staged",
    )

    assert prepared.source_type == "wheels"
    assert prepared.source_path == skills_wheel.resolve()
    assert prepared.wheel.name == skills_wheel.name
    assert prepared.wheel.read_bytes() == b"skills wheel"


def test_copy_skills_setup_stages_profile_skills_folder(tmp_path):
    from benchmark.harness.host import build
    from benchmark.harness.sdks.config import ConfigurableSdkAdapter

    skills_dir = tmp_path / "agent-skills"
    skills_dir.mkdir()
    (skills_dir / "README.md").write_text("example skill\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'example'\n", encoding="utf-8")
    config_path = tmp_path / "example_sdk_copy_skills.yaml"
    config_path.write_text(
        f"""
name: example
display_name: Example SDK
package_name: example-sdk
import_name: example_sdk
source:
  type: repo
  path: {tmp_path}
  markers:
    - pyproject.toml
build:
  type: uv_wheel
  variants:
    skills:
      label: skills
      build_env_value: "with"
      wheel_globs:
        - example_sdk-*.whl
    baseline:
      label: baseline
      build_env_value: "without"
      wheel_globs:
        - example_sdk_baseline-*.whl
docker:
  version_command: example-sdk --version
skills:
  setup:
    type: copy
    source_path: {skills_dir}
    expected_source: profile_skills_folder
""".lstrip(),
        encoding="utf-8",
    )
    sdk = ConfigurableSdkAdapter(config_path)
    setup = build.resolve_sdk_skills_setup(sdk)
    context = tmp_path / "context"
    context.mkdir()

    build.stage_sdk_skills_setup(context, setup)

    assert setup.setup_type == "copy"
    assert (context / "sdk_skills" / "README.md").read_text(encoding="utf-8") == "example skill\n"


def test_container_sdk_skills_setup_copy_mode_installs_staged_folder(tmp_path, monkeypatch):
    from benchmark.harness.container import sdk_skills_setup

    staged = tmp_path / "sdk_skills"
    staged.mkdir()
    (staged / "README.md").write_text("example skill\n", encoding="utf-8")
    home = tmp_path / "agent-home"
    monkeypatch.setattr(sdk_skills_setup, "SDK_SKILLS_SOURCE", staged)
    monkeypatch.setenv("BENCHMARK_AGENT_HOME", str(home))
    monkeypatch.setenv("BENCHMARK_DOCKER_AGENT", "codex")
    monkeypatch.setenv("SKILLS_SETUP_TYPE", "copy")
    monkeypatch.setenv("SKILLS_INSTALL_OUTPUT", "install.json")
    monkeypatch.setenv("SKILLS_LIST_OUTPUT", "list.json")
    monkeypatch.delenv("SKILLS_LIST_COMMAND", raising=False)

    assert sdk_skills_setup.main() == 0

    install = json.loads((home / "install.json").read_text(encoding="utf-8"))
    listing = json.loads((home / "list.json").read_text(encoding="utf-8"))
    assert (home / "skills" / "README.md").read_text(encoding="utf-8") == "example skill\n"
    assert install["mechanism"] == "copy"
    assert install["file_count"] == 1
    assert listing["installed"] == ["README.md"]


def test_container_sdk_skills_setup_write_json_uses_atomic_helper(tmp_path, monkeypatch):
    from benchmark.harness.container import sdk_skills_setup

    calls = []

    def fake_write_json_atomic(path, payload):
        calls.append((Path(path), payload))

    monkeypatch.setattr(sdk_skills_setup, "write_json_atomic", fake_write_json_atomic)

    sdk_skills_setup.write_json(tmp_path / "metadata.json", {"status": "ok"})

    assert calls == [(tmp_path / "metadata.json", {"status": "ok"})]


def test_container_sdk_skills_setup_copy_mode_dereferences_symlinks(tmp_path, monkeypatch):
    from benchmark.harness.container import sdk_skills_setup

    staged = tmp_path / "sdk_skills"
    staged.mkdir()
    home = tmp_path / "agent-home"
    calls = []

    def fake_copytree(source, target, **kwargs):
        calls.append((source, target, kwargs))

    monkeypatch.setattr(sdk_skills_setup, "SDK_SKILLS_SOURCE", staged)
    monkeypatch.setattr(sdk_skills_setup.shutil, "copytree", fake_copytree)
    monkeypatch.setenv("BENCHMARK_AGENT_HOME", str(home))

    result = sdk_skills_setup.copy_skills_folder()

    assert result["status"] == "success"
    assert calls == [(staged, home / "skills", {"dirs_exist_ok": True, "symlinks": False})]


def test_container_sdk_skills_setup_visible_files_skips_symlinks(tmp_path):
    from benchmark.harness.container import sdk_skills_setup

    root = tmp_path / "skills"
    root.mkdir()
    root.joinpath("README.md").write_text("example skill\n", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    try:
        root.joinpath("linked.txt").symlink_to(outside)
        root.joinpath("linked_dir").symlink_to(tmp_path, target_is_directory=True)
    except (OSError, NotImplementedError):
        return

    assert sdk_skills_setup.visible_files(root) == ["README.md"]


def test_container_sdk_skills_setup_run_command_uses_non_login_shell(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from benchmark.harness.container import sdk_skills_setup

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"ok": true}', stderr="")

    monkeypatch.setattr(sdk_skills_setup.subprocess, "run", fake_run)

    out = tmp_path / "out.json"
    sdk_skills_setup.run_command("example-sdk skills list", out)

    assert calls[0][0] == ["/bin/bash", "-c", "example-sdk skills list"]
    # Both streams captured so a failure reason is never lost.
    assert calls[0][1]["stdout"] is sdk_skills_setup.subprocess.PIPE
    assert calls[0][1]["stderr"] is sdk_skills_setup.subprocess.PIPE
    # stdout is still persisted to the output file.
    assert out.read_text(encoding="utf-8") == '{"ok": true}'


def test_container_sdk_skills_setup_run_command_reports_failures(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from benchmark.harness.container import sdk_skills_setup

    def fake_run(command, **_kwargs):
        # A command that fails by writing to stdout (e.g. a --format json error
        # payload) and nothing to stderr — the case that showed only "exit 1".
        return SimpleNamespace(returncode=17, stdout='{"error": "boom on stdout"}', stderr="")

    monkeypatch.setattr(sdk_skills_setup.subprocess, "run", fake_run)

    try:
        sdk_skills_setup.run_command("example-sdk skills install", tmp_path / "install.json")
    except SystemExit as exc:
        message = str(exc)
    else:
        raise AssertionError("failed setup command should raise SystemExit")

    assert "skills.setup command failed with exit code 17" in message
    assert "example-sdk skills install" in message
    assert str(tmp_path / "install.json") in message
    # The stdout error payload is surfaced even though stderr was empty.
    assert "boom on stdout" in message


def test_host_common_all_is_curated():
    from benchmark.harness.host import common

    assert "parse_host_cli_options" in common.__all__
    assert "subprocess" not in common.__all__


def test_resolve_host_agent_cli_version_parses_probe_output(monkeypatch):
    import subprocess as sp
    from types import SimpleNamespace

    from benchmark.harness.host import build

    # availability_probe is a METHOD on real adapters, not an attribute.
    adapter = SimpleNamespace(availability_probe=lambda: ["codex", "--version"], name="codex")
    monkeypatch.setattr(
        build.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="codex-cli 0.142.5\n", stderr=""),
    )
    assert build.resolve_host_agent_cli_version(adapter) == "0.142.5"

    def raise_missing(*_a, **_k):
        raise FileNotFoundError("codex not on PATH")

    monkeypatch.setattr(build.subprocess, "run", raise_missing)
    assert build.resolve_host_agent_cli_version(adapter) == ""
    assert sp  # keep import referenced


def test_resolve_host_agent_cli_version_no_probe_returns_empty():
    from types import SimpleNamespace

    from benchmark.harness.host import build

    assert build.resolve_host_agent_cli_version(SimpleNamespace(availability_probe=lambda: [], name="x")) == ""


def test_resolve_host_agent_cli_version_uses_real_adapter_probe(monkeypatch):
    from types import SimpleNamespace

    from benchmark.harness.agents.registry import load_agent_adapter
    from benchmark.harness.host import build

    # Guard the method-vs-attribute contract against the real adapter, which
    # exposes availability_probe as a bound method.
    adapter = load_agent_adapter("codex")
    monkeypatch.setattr(
        build.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="codex-cli 0.142.5\n", stderr="")
    )
    assert build.resolve_host_agent_cli_version(adapter) == "0.142.5"


def test_skills_source_ref_env_override(monkeypatch):
    """BENCHMARK_SKILLS_SOURCE_REF picks the skills repo#branch for one build."""

    from benchmark.harness.sdks.config import ConfigurableSdkAdapter

    monkeypatch.setenv("BENCHMARK_SKILLS_SOURCE_REF", "chester/NVFlare#skills-experiment")
    sdk = ConfigurableSdkAdapter(Path("config/sdks/nvflare-profile.yaml"))
    assert sdk.docker_build_args()["SKILLS_SOURCE_REF"] == "chester/NVFlare#skills-experiment"


def test_skills_setup_verifies_nonempty_install(tmp_path, monkeypatch):
    """A successful-exit install that wrote nothing must fail the image build."""

    import pytest

    from benchmark.harness.container import sdk_skills_setup

    monkeypatch.setenv("BENCHMARK_AGENT_HOME", str(tmp_path))
    with pytest.raises(SystemExit, match="installed nothing"):
        sdk_skills_setup.verify_skills_installed()

    skill_dir = tmp_path / "skills" / "nvflare-convert-pytorch"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# skill\n", encoding="utf-8")
    sdk_skills_setup.verify_skills_installed()  # non-empty: no raise


def test_generic_skills_list_reports_names_not_paths(tmp_path, monkeypatch):
    import json

    from benchmark.harness.container import sdk_skills_setup
    from benchmark.harness.reports._skill_usage import available_skill_names

    monkeypatch.setenv("BENCHMARK_AGENT_HOME", str(tmp_path))
    for skill in ("nvflare-convert-pytorch", "nvflare-orient"):
        d = tmp_path / "skills" / skill
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("# skill\n", encoding="utf-8")
    shared = tmp_path / "skills" / ".nvflare-shared" / "references"
    shared.mkdir(parents=True)
    (shared / "common.md").write_text("ref\n", encoding="utf-8")

    out = tmp_path / "skills_list.json"
    sdk_skills_setup.write_generic_list(out)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["available"] == [".nvflare-shared", "nvflare-convert-pytorch", "nvflare-orient"]
    # The report layer's name filter drops the shared container and keeps skill NAMES.
    assert available_skill_names(payload) == ["nvflare-convert-pytorch", "nvflare-orient"]


def test_resolve_skills_source_ref_from_local_remote(tmp_path):
    """`origin#branch` resolves the repo URL from the SDK checkout's git config.

    Development installs must follow each developer's own fork (their `origin`),
    not a hardcoded owner; ssh remotes rewrite to https because the image build
    clones without ssh keys.
    """

    import subprocess

    import pytest

    from benchmark.harness.host.build import resolve_skills_source_ref

    repo = tmp_path / "sdk"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "remote", "add", "origin", "git@github.com:chesterxgchen/NVFlare.git"],
        check=True,
    )

    resolved = resolve_skills_source_ref("origin#milestone8-agent-skills", repo)
    assert resolved == "https://github.com/chesterxgchen/NVFlare.git#milestone8-agent-skills"

    # Explicit owner/repo and URL forms pass through unchanged.
    assert resolve_skills_source_ref("NVIDIA/NVFlare#main", repo) == "NVIDIA/NVFlare#main"
    url_ref = "https://github.com/NVIDIA/NVFlare.git#main"
    assert resolve_skills_source_ref(url_ref, repo) == url_ref
    # An explicit ssh URL rewrites to https: the image build has no ssh keys.
    assert (
        resolve_skills_source_ref("git@github.com:NVIDIA/NVFlare.git#main", repo)
        == "https://github.com/NVIDIA/NVFlare.git#main"
    )
    assert resolve_skills_source_ref("", repo) == ""

    with pytest.raises(SystemExit, match="not found"):
        resolve_skills_source_ref("upstream#main", repo)
    with pytest.raises(SystemExit, match="no SDK repo checkout"):
        resolve_skills_source_ref("origin#main", None)


def test_variant_wheel_mismatch_fails_without_build_toggle(tmp_path):
    """Different staged bytes across variants must fail when no toggle exists.

    Issue #1: with reuse_existing and stale wheels in dist/, the benchmark
    could silently compare an old SDK baseline against a new skills build.
    """

    from types import SimpleNamespace

    import pytest

    from benchmark.harness.host.build import PreparedSdkWheel, verify_identical_variant_wheels

    sdk = SimpleNamespace(build_env_name="", package_name="nvflare")
    a = tmp_path / "nvflare-new.whl"
    b = tmp_path / "nvflare-stale.whl"
    a.write_bytes(b"new wheel bytes")
    b.write_bytes(b"stale wheel bytes")
    same = PreparedSdkWheel(wheel=a, source_type="repo", source_path=tmp_path)
    stale = PreparedSdkWheel(wheel=b, source_type="repo", source_path=tmp_path)

    verify_identical_variant_wheels(sdk, same, same)  # identical: no raise
    with pytest.raises(SystemExit, match="wheel mismatch"):
        verify_identical_variant_wheels(sdk, same, stale)


def test_stage_shared_variant_wheel_copies_identical_bytes(tmp_path):
    from benchmark.harness.host import build
    from benchmark.harness.host.build import PreparedSdkWheel, stage_shared_variant_wheel

    skills_dir = tmp_path / "dist" / "skills"
    skills_dir.mkdir(parents=True)
    wheel = skills_dir / "nvflare_nightly-2.8.0rc1-py3-none-any.whl"
    wheel.write_bytes(b"wheel-bytes")
    stale = tmp_path / "dist" / "baseline"
    stale.mkdir(parents=True)
    (stale / "nvflare_nightly-old-py3-none-any.whl").write_bytes(b"stale")

    prepared = PreparedSdkWheel(wheel=wheel, source_type="repo", source_path=tmp_path)
    shared = stage_shared_variant_wheel(prepared, stale)

    assert shared.wheel == stale / wheel.name
    assert shared.wheel.read_bytes() == b"wheel-bytes"
    assert not (stale / "nvflare_nightly-old-py3-none-any.whl").exists()
    assert build.file_sha256(shared.wheel) == build.file_sha256(wheel)
