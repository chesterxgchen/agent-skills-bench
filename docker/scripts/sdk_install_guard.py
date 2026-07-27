#!/workspace/venv/bin/python
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

"""Keep project dependency installs from replacing the SDK under evaluation.

The benchmark installs its SDK from a staged local wheel before the agent runs.
Project requirements may still name that SDK for deployment.  Passing those
entries to an index-backed installer can fail for an unpublished development
version or silently install a different published build.  This wrapper removes
only the harness-owned SDK from ``uv pip install`` and ``pip install`` inputs.
Requested SDK extras are expanded from the installed local wheel metadata so
their dependencies are still installed.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from packaging.markers import default_environment
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name
except ModuleNotFoundError:
    # ``uv venv --seed`` guarantees pip in the benchmark environment. Keep the
    # guard usable for SDKs that do not independently depend on ``packaging``.
    from pip._vendor.packaging.markers import default_environment
    from pip._vendor.packaging.requirements import InvalidRequirement, Requirement
    from pip._vendor.packaging.utils import canonicalize_name

RequirementProvider = Callable[[set[str]], list[str]]


def _overlay_manifest() -> dict[str, Any]:
    overlay = Path(os.environ.get("BENCHMARK_SDK_OVERLAY", "/opt/benchmark-sdk-overlay"))
    try:
        value = json.loads((overlay / "overlay_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def protected_package_names(package_name: str) -> set[str]:
    names = {canonicalize_name(package_name)}
    distribution_name = str(_overlay_manifest().get("distribution_name") or "").strip()
    if distribution_name:
        names.add(canonicalize_name(distribution_name))
    return names


def protected_requirement(text: str, package_name: str) -> Requirement | None:
    """Return a parsed requirement when ``text`` names the protected package."""

    candidate = text.strip()
    if not candidate or candidate.startswith(("#", "-")):
        return None
    # Requirement-file inline comments require whitespace before ``#``.  Do
    # not split URL fragments such as ``...whl#sha256=...``.
    candidate = candidate.split(" #", 1)[0].strip()
    try:
        requirement = Requirement(candidate)
    except InvalidRequirement:
        return None
    if canonicalize_name(requirement.name) not in protected_package_names(package_name):
        return None
    return requirement


def _requirement_without_marker(requirement: Requirement) -> str:
    name = requirement.name
    if requirement.extras:
        name += "[" + ",".join(sorted(requirement.extras)) + "]"
    if requirement.url:
        return f"{name} @ {requirement.url}"
    return name + str(requirement.specifier)


def installed_extra_requirements(package_name: str, extras: set[str]) -> list[str]:
    """Expand requested extras from the already-installed local SDK metadata."""

    if not extras:
        return []
    distribution_name = str(_overlay_manifest().get("distribution_name") or "").strip()
    try:
        distribution = importlib.metadata.distribution(distribution_name or package_name)
    except importlib.metadata.PackageNotFoundError:
        # Profiles normally use the public package name while development
        # wheels may use a channel suffix (for example ``-nightly``).
        candidates = []
        import_name = os.environ.get("SDK_IMPORT_NAME", "").strip()
        if import_name:
            for candidate in importlib.metadata.distributions():
                if any(Path(str(file)).parts[:1] == (import_name,) for file in candidate.files or []):
                    candidates.append(candidate)
        if len(candidates) != 1:
            raise
        distribution = candidates[0]
    environment = default_environment()
    expanded: dict[str, str] = {}
    for raw in distribution.requires or []:
        try:
            dependency = Requirement(raw)
        except InvalidRequirement:
            continue
        marker = dependency.marker
        if marker is None or "extra" not in str(marker):
            continue
        applies = False
        for extra in extras:
            marker_environment = dict(environment)
            marker_environment["extra"] = extra.lower()
            if marker.evaluate(marker_environment):
                applies = True
                break
        if not applies:
            continue
        rendered = _requirement_without_marker(dependency)
        expanded.setdefault(canonicalize_name(dependency.name), rendered)
    return list(expanded.values())


def _included_requirement(line: str) -> tuple[str, str] | None:
    """Return (option, path) for a requirement/constraint include line."""

    try:
        fields = shlex.split(line, comments=True)
    except ValueError:
        return None
    if len(fields) == 2 and fields[0] in {"-r", "--requirement", "-c", "--constraint"}:
        return fields[0], fields[1]
    if len(fields) == 1:
        for option in ("--requirement=", "--constraint="):
            if fields[0].startswith(option):
                return option[:-1], fields[0][len(option) :]
        if fields[0].startswith("-r") and len(fields[0]) > 2:
            return "-r", fields[0][2:]
        if fields[0].startswith("-c") and len(fields[0]) > 2:
            return "-c", fields[0][2:]
    return None


def filter_requirement_file(
    path: Path,
    package_name: str,
    *,
    extra_provider: RequirementProvider | None = None,
    cache: dict[Path, Path] | None = None,
    temporary_paths: list[Path] | None = None,
    removals: list[dict[str, Any]] | None = None,
) -> tuple[Path, bool]:
    """Return a temporary sibling with protected SDK entries removed.

    Sibling files preserve the meaning of relative editable paths and direct
    references. Nested ``-r``/``-c`` includes are filtered recursively.
    """

    extra_provider = extra_provider or (lambda extras: installed_extra_requirements(package_name, extras))
    cache = cache if cache is not None else {}
    temporary_paths = temporary_paths if temporary_paths is not None else []
    removals = removals if removals is not None else []
    resolved = path.resolve()
    if resolved in cache:
        return cache[resolved], cache[resolved] != resolved

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    output: list[str] = []
    changed = False
    for line_number, line in enumerate(lines, start=1):
        include = _included_requirement(line)
        if include is not None:
            option, included_text = include
            included_path = Path(included_text)
            if not included_path.is_absolute():
                included_path = path.parent / included_path
            if included_path.is_file():
                filtered_include, include_changed = filter_requirement_file(
                    included_path,
                    package_name,
                    extra_provider=extra_provider,
                    cache=cache,
                    temporary_paths=temporary_paths,
                    removals=removals,
                )
                if include_changed:
                    output.append(f"{option} {shlex.quote(str(filtered_include))}\n")
                    changed = True
                    continue

        requirement = protected_requirement(line, package_name)
        if requirement is None:
            output.append(line)
            continue

        extras = {extra.lower() for extra in requirement.extras}
        replacements = extra_provider(extras)
        output.append(
            f"# benchmark SDK guard: {package_name} is supplied by the staged local wheel"
            f"{' (extras expanded below)' if replacements else ''}\n"
        )
        output.extend(f"{replacement}\n" for replacement in replacements)
        removals.append(
            {
                "source": str(path),
                "line": line_number,
                "package": canonicalize_name(package_name),
                "extras": sorted(extras),
                "specifier": str(requirement.specifier),
                "direct_reference": requirement.url is not None,
                "replacement_packages": [
                    canonicalize_name(Requirement(replacement).name) for replacement in replacements
                ],
            }
        )
        changed = True

    if not changed:
        cache[resolved] = resolved
        return resolved, False

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.benchmark-sdk-",
        suffix=".txt",
        dir=path.parent,
        delete=False,
    )
    with handle:
        handle.writelines(output)
    filtered = Path(handle.name)
    temporary_paths.append(filtered)
    cache[resolved] = filtered
    return filtered, True


def guard_install_arguments(
    arguments: list[str],
    package_name: str,
    *,
    prefix_length: int,
    extra_provider: RequirementProvider | None = None,
) -> tuple[list[str], list[Path], list[dict[str, Any]]]:
    """Rewrite installer arguments without changing the project files."""

    rewritten = list(arguments)
    temporary_paths: list[Path] = []
    removals: list[dict[str, Any]] = []
    cache: dict[Path, Path] = {}
    provider = extra_provider or (lambda extras: installed_extra_requirements(package_name, extras))

    index = prefix_length
    while index < len(rewritten):
        value = rewritten[index]
        if value in {"-r", "--requirement", "-c", "--constraint"} and index + 1 < len(rewritten):
            source = Path(rewritten[index + 1])
            if source.is_file():
                filtered, changed = filter_requirement_file(
                    source,
                    package_name,
                    extra_provider=provider,
                    cache=cache,
                    temporary_paths=temporary_paths,
                    removals=removals,
                )
                if changed:
                    rewritten[index + 1] = str(filtered)
            index += 2
            continue
        matched_option = next(
            (option for option in ("--requirement=", "--constraint=") if value.startswith(option)),
            None,
        )
        if matched_option is not None:
            source = Path(value[len(matched_option) :])
            if source.is_file():
                filtered, changed = filter_requirement_file(
                    source,
                    package_name,
                    extra_provider=provider,
                    cache=cache,
                    temporary_paths=temporary_paths,
                    removals=removals,
                )
                if changed:
                    rewritten[index] = matched_option + str(filtered)
            index += 1
            continue

        requirement = protected_requirement(value, package_name)
        if requirement is None:
            index += 1
            continue
        extras = {extra.lower() for extra in requirement.extras}
        replacements = provider(extras)
        rewritten[index : index + 1] = replacements
        removals.append(
            {
                "source": "command_argument",
                "package": canonicalize_name(package_name),
                "extras": sorted(extras),
                "specifier": str(requirement.specifier),
                "direct_reference": requirement.url is not None,
                "replacement_packages": [
                    canonicalize_name(Requirement(replacement).name) for replacement in replacements
                ],
            }
        )
        index += len(replacements)

    if removals and not any(
        value in {"-r", "--requirement"} or value.startswith("--requirement=") for value in rewritten[prefix_length:]
    ):
        # A direct install of only the protected SDK would otherwise leave the
        # installer without a target. An empty requirements file is a portable
        # no-op for both uv and pip and preserves installer option validation.
        has_replacement = any(removal["replacement_packages"] for removal in removals)
        if not has_replacement:
            handle = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=".benchmark-sdk-empty-",
                suffix=".txt",
                delete=False,
            )
            handle.close()
            empty = Path(handle.name)
            temporary_paths.append(empty)
            rewritten.extend(["-r", str(empty)])

    return rewritten, temporary_paths, removals


def _append_guard_log(
    installer: str,
    package_name: str,
    removals: list[dict[str, Any]],
    exit_code: int,
) -> None:
    result_dir = os.environ.get("RESULT_DIR")
    if not result_dir or not removals:
        return
    path = Path(result_dir) / "sdk_install_guard.jsonl"
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "installer": installer,
        "protected_package": canonicalize_name(package_name),
        "removed_count": len(removals),
        "removals": removals,
        "exit_code": exit_code,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
    except OSError:
        # Evidence capture must never turn a successful dependency install into
        # an agent-visible failure.
        pass


def _real_installer(invoked_as: str) -> tuple[str, list[str], int] | None:
    arguments = sys.argv[1:]
    if invoked_as == "uv":
        if arguments[:2] != ["pip", "install"]:
            return None
        return os.environ.get("BENCHMARK_REAL_UV", "/usr/local/bin/uv"), arguments, 2
    if invoked_as in {"pip", "pip3"}:
        if arguments[:1] != ["install"]:
            return None
        default = f"/workspace/venv/bin/{invoked_as}"
        return os.environ.get("BENCHMARK_REAL_PIP", default), arguments, 1
    if invoked_as in {"python", "python3"}:
        executable = os.environ.get("BENCHMARK_REAL_PYTHON", "/workspace/venv/bin/python")
        if arguments[:3] == ["-m", "pip", "install"]:
            return executable, arguments, 3
        if arguments[:2] == ["-mpip", "install"]:
            return executable, arguments, 2
        return None
    raise SystemExit(f"unsupported SDK install guard invocation: {invoked_as}")


def _real_executable(invoked_as: str) -> str:
    if invoked_as == "uv":
        return os.environ.get("BENCHMARK_REAL_UV", "/usr/local/bin/uv")
    if invoked_as in {"python", "python3"}:
        return os.environ.get("BENCHMARK_REAL_PYTHON", "/workspace/venv/bin/python")
    return os.environ.get("BENCHMARK_REAL_PIP", f"/workspace/venv/bin/{invoked_as}")


def _replace_guard_python_argument(arguments: list[str]) -> list[str]:
    """Do not make uv inspect the Python forwarding shim as an interpreter."""

    guard_bin = os.environ.get("BENCHMARK_SDK_GUARD_BIN")
    if not guard_bin:
        return arguments
    guarded = {str(Path(guard_bin) / "python"), str(Path(guard_bin) / "python3")}
    real_python = os.environ.get("BENCHMARK_REAL_PYTHON", "/workspace/venv/bin/python")
    rewritten = list(arguments)
    for index, value in enumerate(rewritten):
        if value == "--python" and index + 1 < len(rewritten) and rewritten[index + 1] in guarded:
            rewritten[index + 1] = real_python
        elif value.startswith("--python=") and value.removeprefix("--python=") in guarded:
            rewritten[index] = f"--python={real_python}"
    return rewritten


def main() -> int:
    invoked_as = Path(sys.argv[0]).name
    real = _real_installer(invoked_as)
    if real is None:
        executable = _real_executable(invoked_as)
        os.execv(executable, [executable, *sys.argv[1:]])
        raise AssertionError("unreachable")

    executable, arguments, prefix_length = real
    package_name = os.environ.get("SDK_PACKAGE_NAME", "").strip()
    if not package_name:
        os.execv(executable, [executable, *arguments])
        raise AssertionError("unreachable")

    rewritten, temporary_paths, removals = guard_install_arguments(
        arguments,
        package_name,
        prefix_length=prefix_length,
    )
    rewritten = _replace_guard_python_argument(rewritten)
    if removals:
        replacement_count = sum(len(removal["replacement_packages"]) for removal in removals)
        print(
            f"Benchmark SDK guard: using the staged local {package_name} wheel; "
            f"excluded {len(removals)} SDK requirement(s) from index resolution"
            f" and retained {replacement_count} requested extra dependency requirement(s).",
            file=sys.stderr,
            flush=True,
        )
    exit_code = 127
    try:
        completed = subprocess.run([executable, *rewritten])
        exit_code = completed.returncode
        return exit_code
    finally:
        _append_guard_log(invoked_as, package_name, removals, exit_code)
        for path in reversed(temporary_paths):
            try:
                path.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
