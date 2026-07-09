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

"""Host-side Docker image build orchestration."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..agents.base import AgentAdapter
from ..agents.config import ConfigurableAgentAdapter
from ..agents.registry import DEFAULT_BENCHMARK_AGENT, load_agent_adapter
from ..common import path_tree_sha256
from ..evaluation import validate_evaluation_rules_source
from ..profile_metadata import build_profile_metadata_block
from ..sdks.base import SdkAdapter, SdkSkillsSetup, SdkSource, SdkWheelBuild, SdkWheelVariant
from ..sdks.capture_spec import resolve_capture_spec
from ..sdks.config import ConfigurableSdkAdapter
from ..sdks.registry import DEFAULT_BENCHMARK_SDK, load_sdk_adapter
from .common import SCRIPT_DIR, emit, write_json

# SDK source checkout used to build the SDK wheel + agent skills baked into the
# benchmark image. Set explicitly via --sdk-repo / SDK_REPO (see bin/build.sh);
# when unset, REPO_ROOT is None and build main() fails fast with a clear message.
# Kept SDK-agnostic on purpose: what an SDK checkout looks like (its markers)
# lives in the SDK profile, not here.
_env_repo = os.environ.get("SDK_REPO")
REPO_ROOT = Path(_env_repo).expanduser().resolve() if _env_repo else None
DEFAULT_UV_IMAGE = "ghcr.io/astral-sh/uv:0.11.19"
DEFAULT_NODE_IMAGE = "node:22.16.0-bookworm-slim"


@dataclass(frozen=True)
class PreparedSdkWheel:
    wheel: Path
    source_type: str
    source_path: Path


@dataclass(frozen=True)
class PreparedEvaluationCriteria:
    source_path: Path
    source_type: str
    sha256: str
    staged_entrypoint: str
    source_format: str = "harness_yaml"


def looks_like_profile_path(value: str) -> bool:
    candidate = Path(value).expanduser()
    return candidate.is_absolute() or len(candidate.parts) > 1 or candidate.suffix in {".yaml", ".yml"}


def load_sdk_profile(profile: str) -> SdkAdapter:
    candidate = Path(profile).expanduser()
    if candidate.is_file():
        return ConfigurableSdkAdapter(candidate.resolve())
    if looks_like_profile_path(profile):
        raise SystemExit(f"SDK profile file does not exist: {candidate}")
    try:
        return load_sdk_adapter(profile)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def load_agent_profile(profile: str) -> AgentAdapter:
    candidate = Path(profile).expanduser()
    if candidate.is_file():
        return ConfigurableAgentAdapter(candidate.resolve())
    if looks_like_profile_path(profile):
        raise SystemExit(f"Agent profile file does not exist: {candidate}")
    try:
        return load_agent_adapter(profile)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def canonical_dir(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if not expanded.is_dir():
        raise SystemExit(f"{label} directory does not exist: {expanded}")
    return expanded.resolve()


def assert_sdk_repo_not_in_harness_source(path: Path) -> None:
    try:
        path.relative_to(SCRIPT_DIR)
    except ValueError:
        return
    raise SystemExit(
        "SDK profile source.path must not be inside dev_tools/agent/skills/benchmark. "
        "Only built wheel artifacts are staged into the Docker build context."
    )


def repo_has_markers(path: Path, markers: tuple[str, ...]) -> bool:
    for marker in markers:
        marker_path = path / marker.rstrip("/")
        if marker.endswith("/"):
            if not marker_path.is_dir():
                return False
        elif not marker_path.exists():
            return False
    return True


_CLI_VERSION_RE = re.compile(r"\d+\.\d+\.\d+(?:[-.][0-9A-Za-z.]+)?")


def resolve_host_agent_cli_version(adapter: AgentAdapter) -> str:
    """The agent CLI version installed on the HOST, or "" when undetectable.

    Runs the profile's availability probe (e.g. ``codex --version``) and parses
    a semver. Lets the image track the operator's own CLI so the container
    accepts the same host config (auth/config.toml) instead of failing on
    settings a stale pinned CLI does not understand. Any failure — no CLI on
    PATH, unexpected output — yields "" and the profile default is used."""

    probe_attr = getattr(adapter, "availability_probe", None)
    probe_value = probe_attr() if callable(probe_attr) else probe_attr
    probe = list(probe_value or [])
    if not probe:
        return ""
    try:
        result = subprocess.run(probe, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return ""
    match = _CLI_VERSION_RE.search(f"{result.stdout}\n{result.stderr}")
    return match.group(0) if match else ""


def resolve_sdk_source(sdk: SdkAdapter) -> SdkSource:
    source = sdk.source(repo_root=REPO_ROOT or Path(), home=Path.home())
    if source.source_type == "repo":
        if REPO_ROOT is None:
            raise SystemExit("SDK repo not found: pass --sdk-repo /path/to/<sdk-checkout> or set SDK_REPO.")
        if source.repo_path is None:
            raise SystemExit(f"{sdk.display_name} SDK profile source.type=repo requires source.path")
        repo = canonical_dir(source.repo_path, "SDK profile source.path")
        if not repo_has_markers(repo, source.repo_markers):
            markers = ", ".join(source.repo_markers)
            raise SystemExit(
                f"SDK profile source.path does not look like a {sdk.display_name} checkout: {repo}. "
                f"Expected marker(s): {markers}."
            )
        assert_sdk_repo_not_in_harness_source(repo)
        return SdkSource(source_type="repo", repo_path=repo, repo_markers=source.repo_markers)

    if source.source_type == "wheels":
        raw_wheels = source.wheel_paths or {}
        wheel_paths = {}
        for variant_name in ("skills", "baseline"):
            wheel = raw_wheels.get(variant_name)
            if wheel is None:
                raise SystemExit(f"{sdk.display_name} SDK profile source.wheels.{variant_name} is required")
            expanded = wheel.expanduser()
            if not expanded.is_file():
                raise SystemExit(f"SDK profile source.wheels.{variant_name} file does not exist: {expanded}")
            if expanded.suffix != ".whl":
                raise SystemExit(f"SDK profile source.wheels.{variant_name} must be a .whl file: {expanded}")
            wheel_paths[variant_name] = expanded.resolve()
        return SdkSource(source_type="wheels", wheel_paths=wheel_paths)

    raise SystemExit(f"Unsupported SDK profile source.type={source.source_type!r}")


def latest_sdk_wheel(search_dir: Path, include_globs: tuple[str, ...], exclude_globs: tuple[str, ...]) -> Path | None:
    wheels: dict[Path, None] = {}
    for pattern in include_globs:
        for wheel in search_dir.glob(pattern):
            if wheel.suffix == ".whl":
                wheels[wheel] = None
    matches: list[tuple[Path, float]] = []
    for wheel in wheels:
        if any(fnmatch.fnmatch(wheel.name, pattern) for pattern in exclude_globs):
            continue
        try:
            matches.append((wheel, wheel.stat().st_mtime))
        except OSError:
            continue
    if not matches:
        return None
    return max(matches, key=lambda item: item[1])[0]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def clean_wheels(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for wheel in out_dir.glob("*.whl"):
        wheel.unlink()


def clean_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def copy_directory_contents(src: Path, dst: Path) -> None:
    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {name for name in names if name == "__pycache__" or name.endswith(".pyc")}

    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target, ignore=ignore, symlinks=False)
        else:
            shutil.copy2(child, target)


def nvflare_skill_eval_files(path: Path) -> list[Path]:
    """NVFLARE SDK-native eval criteria layout: skill_evals/<skill>/evals.json."""

    if not path.is_dir():
        return []
    return sorted(candidate for candidate in path.glob("*/evals.json") if candidate.is_file())


def _native_skill_patterns_by_task(index: dict[str, Any]) -> dict[str, list[str]]:
    """Skill-name glob patterns each task declares in the manifest (``tasks.<task>.
    native_skills``). The manifest is the single registry: onboarding a new
    skill family = declaring its pattern on its task entry, not editing code."""

    tasks = index.get("tasks") if isinstance(index.get("tasks"), dict) else {}
    patterns: dict[str, list[str]] = {}
    for task, entry in tasks.items():
        values = entry.get("native_skills") if isinstance(entry, dict) else None
        if isinstance(values, list):
            declared = [str(value) for value in values if str(value or "").strip()]
            if declared:
                patterns[str(task)] = declared
    return patterns


def _evaluation_task_for_nvflare_skill(skill_name: str, patterns_by_task: dict[str, list[str]] | None = None) -> str:
    for task, patterns in (patterns_by_task or {}).items():
        # fnmatchcase: the manifest patterns are a routing contract — matching
        # must not vary with the platform's filename case-folding.
        if any(fnmatch.fnmatchcase(skill_name, pattern) for pattern in patterns):
            return task
    # Legacy name heuristics for manifests that predate `native_skills`.
    if "-convert-" in skill_name:
        return "conversion"
    if "-diagnose-" in skill_name:
        return "diagnosis"
    if "-orient" in skill_name:
        return "orientation"
    return "general"


def _native_behavior_rules(category: str) -> list[dict[str, Any]]:
    if category == "optional_behavior":
        return [
            {"contains": "status=pass", "verdict": "good"},
            {"contains": "status=fail", "verdict": "caution"},
            {"contains_any": ["status=missing", "status=not_applicable"], "verdict": "unknown"},
        ]
    return [
        {"contains": "status=pass", "verdict": "good"},
        {"contains_any": ["status=fail", "status=missing"], "verdict": "bad"},
        {"contains": "status=not_applicable", "verdict": "unknown"},
    ]


def _native_behavior_label(category: str, description: str, behavior_id: str) -> str:
    prefix = {
        "mandatory_behavior": "Mandatory behavior",
        "prohibited_behavior": "Prohibited behavior",
        "optional_behavior": "Optional behavior",
    }.get(category, category.replace("_", " ").title())
    detail = description.strip() or behavior_id
    return f"{prefix}: {detail}"


def _load_nvflare_skill_eval_documents(path: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for evals_path in nvflare_skill_eval_files(path):
        try:
            payload = json.loads(evals_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{evals_path}: invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{evals_path}: evals.json must contain a JSON object")
        skill_name = str(payload.get("skill_name") or evals_path.parent.name)
        evals = payload.get("evals")
        if not isinstance(evals, list):
            raise ValueError(f"{evals_path}: evals must be a list")
        documents.append({"path": evals_path, "skill_name": skill_name, "evals": evals})
    if not documents:
        raise ValueError(f"no NVFLARE evals.json files found under {path}")
    return documents


def _native_nvflare_behavior_signals(
    documents: list[dict[str, Any]], task: str, patterns_by_task: dict[str, list[str]] | None = None
) -> dict[str, Any]:
    signals: dict[str, Any] = {}
    for document in documents:
        skill_name = str(document.get("skill_name") or "")
        if _evaluation_task_for_nvflare_skill(skill_name, patterns_by_task) != task:
            continue
        for eval_case in document.get("evals") or []:
            if not isinstance(eval_case, dict):
                continue
            case_id = str(eval_case.get("id") or "")
            nvflare = eval_case.get("nvflare") if isinstance(eval_case.get("nvflare"), dict) else {}
            for category in ("mandatory_behavior", "prohibited_behavior", "optional_behavior"):
                behaviors = nvflare.get(category) if isinstance(nvflare, dict) else None
                if not isinstance(behaviors, list):
                    continue
                for behavior in behaviors:
                    if not isinstance(behavior, dict) or not behavior.get("id"):
                        continue
                    behavior_id = str(behavior["id"]).strip()
                    if not behavior_id:
                        continue
                    key = f"{category}__{behavior_id}"
                    entry = signals.setdefault(
                        key,
                        {
                            "label": _native_behavior_label(
                                category,
                                str(behavior.get("description") or ""),
                                behavior_id,
                            ),
                            "native_behavior": {
                                "category": category,
                                "id": behavior_id,
                                "skills": [],
                                "cases": [],
                            },
                            "rules": _native_behavior_rules(category),
                        },
                    )
                    metadata = entry.setdefault("native_behavior", {})
                    if skill_name and skill_name not in metadata.setdefault("skills", []):
                        metadata["skills"].append(skill_name)
                    if case_id and case_id not in metadata.setdefault("cases", []):
                        metadata["cases"].append(case_id)
    return signals


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def stage_nvflare_skill_evals_as_evaluation_rules(source: Path, target: Path) -> None:
    documents = _load_nvflare_skill_eval_documents(source)
    packaged_rules = SCRIPT_DIR / "benchmark" / "config" / "evaluation"
    copy_directory_contents(packaged_rules, target)

    native_target = target / "native" / "nvflare_skill_evals"
    native_target.mkdir(parents=True, exist_ok=True)
    for document in documents:
        evals_path = document["path"]
        relative = evals_path.relative_to(source)
        destination = native_target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(evals_path, destination)

    index_path = target / "nvflare" / "index.yaml"
    index = yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}
    tasks = index.get("tasks") if isinstance(index.get("tasks"), dict) else {}
    patterns_by_task = _native_skill_patterns_by_task(index)

    # Every skill whose evals map to a REGISTERED task feeds that task's report
    # signals; say which skills stay signal-less instead of silently dropping
    # them (their raw evals.json copies are still staged).
    skipped = sorted(
        {
            str(document["skill_name"])
            for document in documents
            if _evaluation_task_for_nvflare_skill(str(document["skill_name"]), patterns_by_task) not in tasks
        }
    )
    if skipped:
        emit(f"Evaluation criteria: no registered task derives signals for skills: {', '.join(skipped)}")

    signal_counts: dict[str, int] = {}
    for task in tasks:
        native_signals = _native_nvflare_behavior_signals(documents, str(task), patterns_by_task)
        if not native_signals:
            continue
        signal_counts[str(task)] = len(native_signals)
        native_ref = f"tasks/{task}/native_skill_evals.yaml"
        _write_yaml(
            target / "nvflare" / "tasks" / str(task) / "native_skill_evals.yaml",
            {
                "schema_version": 1,
                "source_format": "nvflare_skill_evals",
                "source_path": "native/nvflare_skill_evals",
                "signals": native_signals,
            },
        )
        entry = tasks.get(task) or {}
        compose = entry.get("compose")
        if not isinstance(compose, list):
            compose = [entry.get("common")] if entry.get("common") else []
            entry.pop("common", None)
        if native_ref not in compose:
            compose.append(native_ref)
        entry["compose"] = compose
        tasks[str(task)] = entry
    if signal_counts:
        index["tasks"] = tasks
        index.setdefault("native_sources", {})["nvflare_skill_evals"] = {
            "path": "native/nvflare_skill_evals",
            "skill_count": len(documents),
            "signal_counts_by_task": signal_counts,
            # Kept for readers of the previous single-task layout.
            "conversion_signal_count": signal_counts.get("conversion", 0),
        }
        _write_yaml(index_path, index)


def resolve_skills_source_ref(ref: str, repo_root: Path | None) -> str:
    """Resolve a ``<remote>#<branch>`` skills ref against the SDK repo checkout.

    A profile can name a git REMOTE of the developer's local SDK checkout
    (e.g. ``origin#milestone8-agent-skills``) instead of hardcoding an owner —
    each developer's build then installs skills from THEIR fork's branch, per
    their git config. Explicit ``owner/repo#branch`` and full-URL refs pass
    through unchanged. ssh remotes are rewritten to https: the image build
    clones without ssh keys.
    """

    if not ref or "#" not in ref:
        return ref
    remote, branch = ref.split("#", 1)
    if not remote or "/" in remote or ":" in remote:
        # Explicit owner/repo or URL ref. Rewrite an ssh URL to https here too —
        # the image build clones without ssh keys, so a pasted ssh remote would
        # only fail later inside the container.
        explicit = re.match(r"^(?:ssh://)?git@([^:/]+)[:/](.+?)(?:\.git)?$", remote)
        if explicit:
            return f"https://{explicit.group(1)}/{explicit.group(2)}.git#{branch}"
        return ref
    if repo_root is None:
        raise SystemExit(
            f"Skills source ref {ref!r} names git remote {remote!r}, but no SDK repo checkout "
            "is configured (--sdk-repo/SDK_REPO)."
        )
    result = subprocess.run(
        ["git", "-C", str(repo_root), "remote", "get-url", remote], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise SystemExit(
            f"Cannot resolve skills source ref {ref!r}: git remote {remote!r} not found in {repo_root}"
            + (f" ({result.stderr.strip()})" if result.stderr.strip() else "")
        )
    url = result.stdout.strip()
    match = re.match(r"^(?:ssh://)?git@([^:/]+)[:/](.+?)(?:\.git)?$", url)
    if match:
        url = f"https://{match.group(1)}/{match.group(2)}.git"
    return f"{url}#{branch}"


def resolve_sdk_skills_setup(sdk: SdkAdapter) -> SdkSkillsSetup:
    setup = sdk.skills_setup(repo_root=REPO_ROOT or Path(), home=Path.home())
    if setup.setup_type == "copy":
        if setup.source_path is None:
            raise SystemExit(f"{sdk.display_name} SDK profile skills.setup.source_path is required")
        source_path = canonical_dir(setup.source_path, "SDK profile skills.setup.source_path")
        return SdkSkillsSetup(
            setup_type=setup.setup_type,
            source_path=source_path,
            install_command=setup.install_command,
            list_command=setup.list_command,
            install_output=setup.install_output,
            list_output=setup.list_output,
            expected_source=setup.expected_source,
        )
    return setup


def stage_sdk_skills_setup(context: Path, setup: SdkSkillsSetup) -> None:
    target = context / "sdk_skills"
    clean_directory(target)
    if setup.setup_type != "copy":
        return
    if setup.source_path is None:
        raise SystemExit("SDK profile skills.setup.source_path is required for copy setup")
    copy_directory_contents(setup.source_path, target)
    emit(f"Using SDK skills folder: {setup.source_path}")


def resolve_and_stage_evaluation_criteria(
    *,
    sdk: SdkAdapter,
    source: SdkSource,
    explicit_path: str | None,
    context: Path,
) -> PreparedEvaluationCriteria:
    if explicit_path:
        candidate = Path(explicit_path).expanduser().resolve()
        source_type = "explicit"
    else:
        relative = sdk.evaluation_criteria().repo_relative_path
        if source.repo_path is None or relative is None:
            raise SystemExit(
                "Evaluation criteria not found: pass --evaluation-criteria /path/to/rules.yaml (or rules directory), "
                "or configure evaluation.criteria_path in the SDK profile for use with --sdk-repo."
            )
        candidate = (source.repo_path / relative).resolve()
        try:
            candidate.relative_to(source.repo_path)
        except ValueError as exc:
            raise SystemExit(f"SDK evaluation criteria path escapes the SDK repo: {relative}") from exc
        source_type = "sdk_repo"
    if not candidate.exists() or not (candidate.is_file() or candidate.is_dir()):
        raise SystemExit(f"Evaluation criteria path does not exist: {candidate}")
    if candidate.is_symlink() or (candidate.is_dir() and any(path.is_symlink() for path in candidate.rglob("*"))):
        raise SystemExit(f"Evaluation criteria must not contain symbolic links: {candidate}")

    target = context / "evaluation_rules"
    clean_directory(target)
    source_format = "harness_yaml"
    if sdk.name == "nvflare" and nvflare_skill_eval_files(candidate):
        staged_entrypoint = "."
        source_format = "nvflare_skill_evals"
        try:
            stage_nvflare_skill_evals_as_evaluation_rules(candidate, target)
            validate_evaluation_rules_source(sdk.name, target)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise SystemExit(f"Invalid evaluation criteria at {candidate}: {exc}") from exc
    else:
        try:
            validate_evaluation_rules_source(sdk.name, candidate)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise SystemExit(f"Invalid evaluation criteria at {candidate}: {exc}") from exc
        if candidate.is_file():
            staged_entrypoint = "rules.yaml"
            shutil.copy2(candidate, target / staged_entrypoint)
        else:
            staged_entrypoint = "."
            copy_directory_contents(candidate, target)
    staged_source = target / staged_entrypoint if candidate.is_file() else target
    prepared = PreparedEvaluationCriteria(
        source_path=candidate,
        source_type=source_type,
        source_format=source_format,
        sha256=path_tree_sha256(staged_source),
        staged_entrypoint=staged_entrypoint,
    )
    emit(f"Using evaluation criteria: {candidate} ({source_type}, sha256={prepared.sha256[:12]})")
    return prepared


def stage_configured_wheel(
    *,
    wheel: Path,
    sdk: SdkAdapter,
    variant: SdkWheelVariant,
    out_dir: Path,
) -> PreparedSdkWheel:
    if not any(fnmatch.fnmatch(wheel.name, pattern) for pattern in variant.wheel_globs):
        raise SystemExit(
            f"Configured {sdk.package_name} {variant.label} wheel {wheel.name!r} does not match "
            f"expected pattern(s): {variant.wheel_globs}."
        )
    if any(fnmatch.fnmatch(wheel.name, pattern) for pattern in variant.wheel_exclude_globs):
        raise SystemExit(
            f"Configured {sdk.package_name} {variant.label} wheel {wheel.name!r} matches excluded "
            f"pattern(s): {variant.wheel_exclude_globs}."
        )
    clean_wheels(out_dir)
    target = out_dir / wheel.name
    shutil.copy2(wheel, target)
    emit(f"Using configured {variant.label} wheel: {wheel}")
    return PreparedSdkWheel(wheel=target, source_type="wheels", source_path=wheel)


def _stage_repo_wheel(wheel: Path, out_dir: Path, repo: Path) -> PreparedSdkWheel:
    """Copy a built SDK wheel into the Docker build context staging dir."""

    clean_wheels(out_dir)
    staged = out_dir / wheel.name
    shutil.copy2(wheel, staged)
    return PreparedSdkWheel(wheel=staged, source_type="repo", source_path=repo)


def build_sdk_wheel_from_repo(
    *,
    repo: Path,
    sdk: SdkAdapter,
    variant: SdkWheelVariant,
    out_dir: Path,
    rebuild: bool = False,
) -> PreparedSdkWheel:
    # Reuse an existing built wheel by DEFAULT; only run ``uv build`` when no wheel exists
    # or a rebuild is explicitly requested (``--rebuild``). Wheels persist in the SDK
    # repo's ``dist/`` (uv's default output) so repeated benchmark builds don't rebuild the
    # SDK when only agent config or Docker layers changed.
    dist_dir = repo / "dist"
    if not rebuild and variant.reuse_existing:
        existing = latest_sdk_wheel(dist_dir, variant.wheel_globs, variant.wheel_exclude_globs)
        if existing is not None:
            emit(
                f"=== Reusing existing {sdk.package_name} {variant.label} wheel {existing.name} "
                f"(pass --rebuild to force a fresh uv build) ==="
            )
            return _stage_repo_wheel(existing, out_dir, repo)
        emit(f"=== No existing {sdk.package_name} {variant.label} wheel under {dist_dir}; building ===")
    elif rebuild:
        emit(f"=== Rebuilding {sdk.package_name} {variant.label} wheel (--rebuild) ===")
    else:
        emit(f"=== Building {sdk.package_name} {variant.label} wheel (profile disables dist/ reuse) ===")

    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit(
            "Host uv is required to build SDK wheels. "
            "To use existing wheels, set source.type: wheels and build.type: provided_wheels in the SDK profile."
        )
    env = {**os.environ}
    if sdk.build_env_name:
        env[sdk.build_env_name] = variant.build_env_value
    temp_build_dir = (
        tempfile.TemporaryDirectory(prefix=f"{sdk.name}-{variant.name}-wheel.") if not variant.reuse_existing else None
    )
    build_out_dir = Path(temp_build_dir.name) if temp_build_dir is not None else dist_dir
    try:
        # Build into the repo's persistent dist/ when reuse is enabled. Variants that
        # disable reuse build into an isolated directory so stale wheels for another
        # variant cannot be selected when filenames overlap.
        status = subprocess.call([uv, "build", "--wheel", "--out-dir", str(build_out_dir)], cwd=repo, env=env)
        if status != 0:
            raise SystemExit(status)

        wheel = latest_sdk_wheel(build_out_dir, variant.wheel_globs, variant.wheel_exclude_globs)
        if wheel is None:
            raise SystemExit(
                f"No {sdk.package_name} {variant.label} wheel found under {build_out_dir}. "
                f"Expected a wheel matching {variant.wheel_globs} excluding {variant.wheel_exclude_globs}."
            )
        return _stage_repo_wheel(wheel, out_dir, repo)
    finally:
        if temp_build_dir is not None:
            temp_build_dir.cleanup()


def stage_shared_variant_wheel(prepared: PreparedSdkWheel, out_dir: Path) -> PreparedSdkWheel:
    """Stage the wheel one variant already prepared into another variant's dir.

    Without a build-time variant toggle both images must stage IDENTICAL wheel
    bytes, and a second ``uv build`` of the same source is never byte-identical
    (archive timestamps differ) — so the second variant copies the first
    variant's staged wheel instead of building its own.
    """

    clean_wheels(out_dir)
    target = out_dir / prepared.wheel.name
    shutil.copy2(prepared.wheel, target)
    return PreparedSdkWheel(wheel=target, source_type=prepared.source_type, source_path=prepared.source_path)


def verify_identical_variant_wheels(
    sdk: SdkAdapter, skills_prepared: PreparedSdkWheel, baseline_prepared: PreparedSdkWheel
) -> None:
    """Without a build-time toggle, both variants MUST stage the same wheel bytes.

    NVFLARE dropped its wheel-bundling hook (PR #4837): the SDK wheel is
    identical in both images and the A/B distinction is purely the skills
    setup. If the two variants stage different bytes anyway — a stale wheel
    lingering in the SDK repo's dist/ is the classic cause — the comparison
    would silently pit an old SDK against a new one. Fail loudly instead.
    """

    skills_sha = file_sha256(skills_prepared.wheel)
    baseline_sha = file_sha256(baseline_prepared.wheel)
    if skills_sha != baseline_sha:
        raise SystemExit(
            f"SDK wheel mismatch between image variants: skills staged {skills_prepared.wheel.name} "
            f"({skills_sha[:12]}) but baseline staged {baseline_prepared.wheel.name} ({baseline_sha[:12]}). "
            "The profile defines no build-time variant toggle, so both must be the same wheel — a stale "
            "wheel in the SDK repo's dist/ is the usual cause. Re-run with --rebuild or clean dist/."
        )
    emit(f"Variant wheel check: skills and baseline stage identical bytes (sha256 {skills_sha[:12]})")


def prepare_sdk_wheel(
    *,
    source: SdkSource,
    wheel_build: SdkWheelBuild,
    sdk: SdkAdapter,
    variant: SdkWheelVariant,
    out_dir: Path,
    rebuild: bool = False,
) -> PreparedSdkWheel:
    if wheel_build.build_type == "uv_wheel":
        if source.repo_path is None:
            raise SystemExit(f"{sdk.display_name} SDK profile build.type=uv_wheel requires source.type=repo")
        return build_sdk_wheel_from_repo(
            repo=source.repo_path, sdk=sdk, variant=variant, out_dir=out_dir, rebuild=rebuild
        )
    if wheel_build.build_type == "provided_wheels":
        wheel_paths = source.wheel_paths or {}
        wheel = wheel_paths.get(variant.name)
        if wheel is None:
            raise SystemExit(f"{sdk.display_name} SDK profile build.type=provided_wheels requires source.wheels")
        return stage_configured_wheel(wheel=wheel, sdk=sdk, variant=variant, out_dir=out_dir)
    raise SystemExit(f"Unsupported SDK profile build.type={wheel_build.build_type!r}")


def write_wheel_metadata(
    *,
    sdk: SdkAdapter,
    variant: SdkWheelVariant,
    wheel_build: SdkWheelBuild,
    prepared: PreparedSdkWheel,
    out_dir: Path,
    evaluation: PreparedEvaluationCriteria,
) -> None:
    # The §4.3 identity block (schema_version, sdk_name, benchmark_profile_id,
    # report_plugin_id, capture_spec_version) is built in one place, so build
    # and the read-time lift never drift.
    payload = {
        "build_env": {"name": sdk.build_env_name, "value": variant.build_env_value} if sdk.build_env_name else None,
        "build_type": wheel_build.build_type,
        "filename": prepared.wheel.name,
        "git_commit": git_commit(prepared.source_path) if prepared.source_type == "repo" else None,
        "import_name": sdk.import_name,
        "package_name": sdk.package_name,
        "sdk": sdk.metadata(),
        "sha256": file_sha256(prepared.wheel),
        "source_path": str(prepared.source_path),
        "source_type": prepared.source_type,
        "variant": variant.name,
        "evaluation_criteria": {
            "source_type": evaluation.source_type,
            "source_format": evaluation.source_format,
            "source_path": str(evaluation.source_path),
            "sha256": evaluation.sha256,
            "entrypoint": evaluation.staged_entrypoint,
        },
        # Declarative capture spec (§4.1): generic in-container Stage-3 code
        # applies this serialized data instead of hardcoding SDK rules.
        "capture_spec": resolve_capture_spec(sdk.name).to_payload(),
        **build_profile_metadata_block(sdk),
    }
    write_json(out_dir / "sdk_wheel_metadata.json", payload)


def copy_harness(src: Path, dst: Path) -> None:
    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {name for name in names if name == "__pycache__" or name.endswith(".pyc")}

    shutil.copytree(src, dst, ignore=ignore, symlinks=False)


def copy_harness_package(context: Path) -> None:
    copy_harness(SCRIPT_DIR / "config", context / "config")
    copy_harness(SCRIPT_DIR / "benchmark", context / "benchmark")
    copy_harness(SCRIPT_DIR / "docker" / "scripts", context / "docker_scripts")


def prepare_build_context() -> Path:
    context = Path(tempfile.mkdtemp(prefix="skills-benchmark-build-context.", dir=os.environ.get("TMPDIR") or None))
    try:
        (context / "dist" / "skills").mkdir(parents=True)
        (context / "dist" / "baseline").mkdir(parents=True)
        (context / "sdk_skills").mkdir(parents=True)
        (context / "evaluation_rules").mkdir(parents=True)
        shutil.copy2(SCRIPT_DIR / "docker" / "Dockerfile", context / "Dockerfile")
        copy_harness_package(context)
        shutil.copy2(SCRIPT_DIR / "docker" / "build_context.dockerignore", context / ".dockerignore")
    except BaseException:
        shutil.rmtree(context, ignore_errors=True)
        raise
    return context


def docker_build(
    *,
    image: str,
    target: str,
    context: Path,
    uv_image: str,
    node_image: str,
    sdk_build_args: dict[str, str],
    agent_build_args: dict[str, str],
    no_cache: bool,
) -> None:
    cache_args = ["--no-cache"] if no_cache else []
    rendered_sdk_build_args = render_docker_build_args(sdk_build_args, allow_value_equals=True)
    rendered_agent_build_args = render_agent_build_args(agent_build_args)
    status = subprocess.call(
        [
            "docker",
            "build",
            *cache_args,
            "--target",
            target,
            "--build-arg",
            f"UV_IMAGE={uv_image}",
            "--build-arg",
            f"NODE_IMAGE={node_image}",
            *rendered_sdk_build_args,
            *rendered_agent_build_args,
            "-t",
            image,
            str(context),
        ]
    )
    if status != 0:
        raise SystemExit(status)


def render_docker_build_args(build_args: dict[str, str], *, allow_value_equals: bool = False) -> list[str]:
    rendered_build_args = []
    for key, value in sorted(build_args.items()):
        if "=" in str(key):
            raise ValueError(f"Docker build arg key must not contain '=': {key}")
        if "=" in str(value) and not allow_value_equals:
            raise ValueError(f"Docker build arg {key} value must not contain '='")
        rendered_build_args.extend(["--build-arg", f"{key}={value}"])
    return rendered_build_args


def render_agent_build_args(agent_build_args: dict[str, str]) -> list[str]:
    return render_docker_build_args(agent_build_args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build agent skills benchmark Docker images.")
    parser.add_argument(
        "--sdk-profile",
        default=DEFAULT_BENCHMARK_SDK,
        help=f"SDK profile name or YAML path. Defaults to {DEFAULT_BENCHMARK_SDK}.",
    )
    parser.add_argument(
        "--agent-profile",
        "--agent",
        dest="agent_profile",
        default=DEFAULT_BENCHMARK_AGENT,
        help=f"Agent profile name or YAML path. Defaults to {DEFAULT_BENCHMARK_AGENT}.",
    )
    parser.add_argument("--skip-skills-image", action="store_true", help="Do not build the skills image.")
    parser.add_argument("--skip-baseline-image", action="store_true", help="Do not build the baseline image.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a fresh uv build of the SDK wheel even if one already exists in the SDK repo's dist/. "
        "Default: reuse the existing wheel, building only when none is found.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Pass --no-cache to docker build.")
    parser.add_argument(
        "--evaluation-criteria",
        help="Harness evaluation YAML file or composed rules directory; overrides SDK-profile repo resolution.",
    )
    parser.add_argument("--uv-image", default=DEFAULT_UV_IMAGE, help="uv image used as the Docker uv source stage.")
    parser.add_argument("--node-image", default=DEFAULT_NODE_IMAGE, help="Node runtime image used as the Docker base.")
    args = parser.parse_args(argv)

    try:
        adapter = load_agent_profile(args.agent_profile)
        sdk = load_sdk_profile(args.sdk_profile)
        wheel_build = sdk.wheel_build()
        skills_variant = sdk.wheel_variant("skills")
        baseline_variant = sdk.wheel_variant("baseline")
        skills_setup = resolve_sdk_skills_setup(sdk)
        targets = adapter.image_targets()
        sdk_build_args = sdk.docker_build_args()
        if sdk_build_args.get("SKILLS_SOURCE_REF"):
            sdk_build_args["SKILLS_SOURCE_REF"] = resolve_skills_source_ref(
                sdk_build_args["SKILLS_SOURCE_REF"], REPO_ROOT
            )
        host_cli_version = resolve_host_agent_cli_version(adapter)
        agent_build_args = adapter.build_args(cli_version=host_cli_version)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    image_name = targets.skills
    baseline_image_name = targets.baseline
    report_image_name = targets.report
    build_skills_image = not args.skip_skills_image
    build_baseline_image = not args.skip_baseline_image

    context = None
    try:
        emit("=== Preparing minimal Docker build context ===")
        context = prepare_build_context()
        emit(f"Agent profile: {args.agent_profile} -> {adapter.display_name} ({adapter.name})")
        if host_cli_version:
            emit(f"Agent CLI version: tracking host {adapter.name} {host_cli_version}")
        else:
            emit(f"Agent CLI version: host {adapter.name} not detected; using profile default")
        emit(f"SDK profile: {args.sdk_profile} -> {sdk.display_name} ({sdk.package_name})")
        emit(f"SDK wheel build: {wheel_build.build_type}")
        if build_skills_image:
            emit(f"SDK skills setup: {skills_setup.setup_type}")
            if sdk_build_args.get("SKILLS_SOURCE_REF"):
                emit(f"Skills source: {sdk_build_args['SKILLS_SOURCE_REF']}")
            stage_sdk_skills_setup(context, skills_setup)
        if build_skills_image or build_baseline_image:
            source = resolve_sdk_source(sdk)
            if source.source_type == "repo":
                emit(f"Using SDK repo: {source.repo_path}")
            else:
                emit("Using SDK wheels from profile.")
            evaluation = resolve_and_stage_evaluation_criteria(
                sdk=sdk,
                source=source,
                explicit_path=args.evaluation_criteria,
                context=context,
            )

            if build_skills_image:
                skills_prepared = prepare_sdk_wheel(
                    source=source,
                    wheel_build=wheel_build,
                    sdk=sdk,
                    variant=skills_variant,
                    out_dir=context / "dist" / "skills",
                    rebuild=args.rebuild,
                )
                emit(f"Using skills wheel: {skills_prepared.wheel.name}")
                write_wheel_metadata(
                    sdk=sdk,
                    variant=skills_variant,
                    wheel_build=wheel_build,
                    prepared=skills_prepared,
                    out_dir=context / "dist" / "skills",
                    evaluation=evaluation,
                )

            if build_baseline_image:
                if build_skills_image and not sdk.build_env_name and wheel_build.build_type == "uv_wheel":
                    # No variant toggle: reuse the exact wheel the skills
                    # variant staged (see stage_shared_variant_wheel) — a
                    # second build would fail the identical-bytes check below.
                    baseline_prepared = stage_shared_variant_wheel(skills_prepared, context / "dist" / "baseline")
                    emit(f"Using baseline wheel: {baseline_prepared.wheel.name} (shared with skills variant)")
                else:
                    baseline_prepared = prepare_sdk_wheel(
                        source=source,
                        wheel_build=wheel_build,
                        sdk=sdk,
                        variant=baseline_variant,
                        out_dir=context / "dist" / "baseline",
                        rebuild=args.rebuild,
                    )
                    emit(f"Using baseline wheel: {baseline_prepared.wheel.name}")
                write_wheel_metadata(
                    sdk=sdk,
                    variant=baseline_variant,
                    wheel_build=wheel_build,
                    prepared=baseline_prepared,
                    out_dir=context / "dist" / "baseline",
                    evaluation=evaluation,
                )

            if build_skills_image and build_baseline_image and not sdk.build_env_name:
                verify_identical_variant_wheels(sdk, skills_prepared, baseline_prepared)

        emit(f"Docker build context: {context}")
        emit(f"UV image: {args.uv_image}")
        emit(f"Node runtime image: {args.node_image}")
        for key, value in sorted(sdk_build_args.items()):
            emit(f"SDK build arg: {key}={value}")
        for key, value in sorted(agent_build_args.items()):
            emit(f"Agent build arg: {key}={value}")
        emit(f"Docker build no-cache: {str(args.no_cache).lower()}")
        if build_skills_image:
            emit(f"=== Building Docker skills image: {image_name} ===")
            docker_build(
                image=image_name,
                target="skills",
                context=context,
                uv_image=args.uv_image,
                node_image=args.node_image,
                sdk_build_args=sdk_build_args,
                agent_build_args=agent_build_args,
                no_cache=args.no_cache,
            )
        if build_baseline_image:
            emit(f"=== Building Docker baseline image: {baseline_image_name} ===")
            docker_build(
                image=baseline_image_name,
                target="baseline",
                context=context,
                uv_image=args.uv_image,
                node_image=args.node_image,
                sdk_build_args=sdk_build_args,
                agent_build_args=agent_build_args,
                no_cache=args.no_cache,
            )

        emit(f"Skills image: {image_name}")
        emit(f"Baseline image: {baseline_image_name}")
        emit(f"Report image: {report_image_name}")
        return 0
    finally:
        if context is not None:
            shutil.rmtree(context, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
