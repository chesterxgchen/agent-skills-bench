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

"""NVFLARE interpretation logic — neutral leaf helpers (migration steps 4–5).

This module OWNS the NVFLARE-specific report logic carved out of
``benchmark_insights`` so the NVFLARE report plugin carries it directly
(architecture §6 Stage 4). It must remain a leaf:

- It does NOT import ``benchmark_insights`` (no import cycle). It may import the
  generic ``reports._text`` substrate (a stdlib-only leaf) for shell/command
  parsing.
- It contains only pure, side-effect-free functions.

``benchmark_insights`` re-imports the names its remaining (out-of-scope) callers
still use, so there is one single implementation — no duplicated body.

Owned here: structure scoring (step 4); NVFLARE command/job/simulator
classification (step 5).
"""

from __future__ import annotations

import ast
import io
import json
import re
import tokenize
from datetime import datetime
from pathlib import Path
from typing import Any

from ...common import load_json
from ...evaluation import load_evaluation_rules, overall_thresholds, score_signal, verdict_points
from ...quality_signals import canonical_metric_name, is_plausible_metric_value, metric_names_match
from ...reports._events import (
    _job_rerun_reason,
    _span_total_seconds,
    agent_command_events,
    agent_command_spans,
    agent_failure_category,
    artifact_validation_metric_evidence,
    artifact_validation_metric_is_runtime_evidence,
    as_number,
    bash_permission_denial_count,
    command_error_summary,
    command_failed,
    command_recovery_key,
    command_succeeded,
    commands_for_run,
    dependency_install_events,
    dependency_install_evidence,
    dependency_install_evidence_brief,
    exit_code,
    failure_evidence,
    fmt_seconds_with_unit,
    inline_code_text,
    is_dependency_install_command,
    job_output_has_failure_marker,
    job_output_succeeded,
    missing_python_module_name,
    parse_event_timestamp,
    truncate,
)
from ...reports._runs import (
    _all_python_workspace_text,
    _final_message_without_event_log,
    _workspace_artifact_path,
    _workspace_file_text,
    _workspace_python_sources,
    combined_text,
    manifest_paths,
    read_text,
    run_record,
    run_workspace_delta,
    unique_paths,
)
from ...reports._text import (
    _command_tokens,
    _is_file_inspection_segment,
    _shell_command_parts,
    _shell_command_segments,
    _strip_quoted,
    _strip_execution_prefix_tokens,
    fmt_number,
    markdown_cell,
    strip_ansi,
)

# Product-specific structure contract: the core converted source files an
# NVFLARE job is expected to produce.
REQUIRED_STRUCTURE_FILES = ("client.py", "model.py", "job.py")
FEDERATED_STATISTICS_STRUCTURE_FILES = ("client.py", "job.py")
OPTIONAL_STRUCTURE_FILES = ("prepare_data.py", "download_data.py")


def current_workspace_structure_file_matches(run: dict[str, Any], filename: str) -> list[str]:
    paths = unique_paths(manifest_paths(run, "final_structure_files"))
    return [path for path in paths if Path(path).name == filename and len(Path(path).parts) == 1]


def _generated_source_file_paths(run: dict[str, Any], filenames: tuple[str, ...]) -> list[str]:
    paths = []
    seen: set[str] = set()
    for key in ("final_structure_files", "changed_files", "final_files"):
        for path in manifest_paths(run, key):
            rel_path = Path(path)
            if rel_path.name not in filenames:
                continue
            if _workspace_runtime_or_export_tree_root(path):
                continue
            if path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def _federated_statistics_required_file_folders(run: dict[str, Any]) -> dict[str, set[str]]:
    folders: dict[str, set[str]] = {}
    for path in _generated_source_file_paths(run, FEDERATED_STATISTICS_STRUCTURE_FILES):
        rel_path = Path(path)
        folder = str(rel_path.parent) if len(rel_path.parts) > 1 else ""
        folders.setdefault(folder, set()).add(rel_path.name)
    return folders


def _federated_statistics_generated_job_folders(run: dict[str, Any]) -> list[str]:
    folders = _federated_statistics_required_file_folders(run)
    required = set(FEDERATED_STATISTICS_STRUCTURE_FILES)
    return sorted(folder for folder, names in folders.items() if required.issubset(names))


def _federated_statistics_structure_file_matches(run: dict[str, Any], filename: str) -> list[str]:
    return [path for path in _generated_source_file_paths(run, (filename,)) if Path(path).name == filename]


def _federated_statistics_best_required_file_names(run: dict[str, Any]) -> set[str]:
    required = set(FEDERATED_STATISTICS_STRUCTURE_FILES)
    folders = _federated_statistics_required_file_folders(run)
    if not folders:
        return set()
    return max(
        (names & required for _folder, names in sorted(folders.items())),
        key=lambda names: (len(names), tuple(filename in names for filename in FEDERATED_STATISTICS_STRUCTURE_FILES)),
    )


def structure_required_files(run: dict[str, Any]) -> tuple[str, ...]:
    if _resolved_evaluation_task(run) == "federated-statistics":
        return FEDERATED_STATISTICS_STRUCTURE_FILES
    return REQUIRED_STRUCTURE_FILES


def structure_optional_files(run: dict[str, Any]) -> tuple[str, ...]:
    return OPTIONAL_STRUCTURE_FILES


def structure_required_label(run: dict[str, Any]) -> str:
    if _resolved_evaluation_task(run) == "federated-statistics":
        return "Required federated-statistics job files"
    return "Required converted files"


def structure_accepted_required_folders(run: dict[str, Any]) -> tuple[str, ...]:
    if _resolved_evaluation_task(run) == "federated-statistics":
        return tuple(_federated_statistics_generated_job_folders(run))
    return ()


def structure_file_matches(run: dict[str, Any], filename: str) -> list[str]:
    if _resolved_evaluation_task(run) == "federated-statistics":
        return _federated_statistics_structure_file_matches(run, filename)
    return current_workspace_structure_file_matches(run, filename)


def structure_present_required_files(run: dict[str, Any]) -> tuple[str, ...]:
    required = structure_required_files(run)
    if _resolved_evaluation_task(run) == "federated-statistics":
        present = _federated_statistics_best_required_file_names(run)
        return tuple(filename for filename in required if filename in present)
    return tuple(filename for filename in required if structure_file_matches(run, filename))


def structure_present_optional_files(run: dict[str, Any]) -> tuple[str, ...]:
    return tuple(filename for filename in structure_optional_files(run) if structure_file_matches(run, filename))


def _workspace_runtime_or_export_tree_root(path: str) -> str:
    parts = Path(str(path or "")).parts
    if not parts:
        return ""
    if parts[0] == "fl_workspace":
        return "/".join(parts[:2]) if len(parts) > 1 else parts[0]
    if parts[0] == "fl_job":
        return "/".join(parts[:2]) if len(parts) > 1 else parts[0]
    if "simulate_job" in parts:
        index = parts.index("simulate_job")
        return "/".join(parts[: index + 1])
    return ""


def _workspace_runtime_or_export_tree_roots(run: dict[str, Any]) -> list[str]:
    roots = []
    seen: set[str] = set()
    for key in ("changed_files", "final_structure_files", "final_files"):
        for path in manifest_paths(run, key):
            root = _workspace_runtime_or_export_tree_root(path)
            if root and root not in seen:
                roots.append(root)
                seen.add(root)
    return roots


def _nested_runtime_or_export_source_folders(run: dict[str, Any]) -> list[str]:
    folders = []
    seen: set[str] = set()
    source_names = set(REQUIRED_STRUCTURE_FILES) | set(FEDERATED_STATISTICS_STRUCTURE_FILES) | {
        "fl_data.py",
        "train.py",
        "fl_train.py",
    }
    for key in ("changed_files", "final_structure_files", "final_files"):
        for path in manifest_paths(run, key):
            rel_path = Path(path)
            if rel_path.name not in source_names or len(rel_path.parts) <= 1:
                continue
            if not _workspace_runtime_or_export_tree_root(path):
                continue
            folder = str(rel_path.parent)
            if folder not in seen:
                folders.append(folder)
                seen.add(folder)
    return folders


def _short_path_list(paths: list[str], *, limit: int = 3) -> str:
    if not paths:
        return ""
    rendered = ", ".join(paths[:limit])
    if len(paths) > limit:
        rendered += f", +{len(paths) - limit} more"
    return rendered


def structure_score(run: dict[str, Any]) -> float | None:
    if not run.get("available"):
        return None
    required = structure_required_files(run)
    present = len(structure_present_required_files(run))
    return present / len(required)


# --- NVFLARE command / job / simulator classification (step 5) --------------


def _python_script_name_from_segment(command: str) -> str:
    tokens = _command_tokens(command)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        name = Path(token).name.lower()
        if name in {"timeout", "gtimeout"}:
            index += 1
            while index < len(tokens) and (
                tokens[index].startswith("-") or re.fullmatch(r"\d+(?:\.\d+)?[smhd]?", tokens[index])
            ):
                index += 1
            continue
        if name == "env":
            index += 1
            while index < len(tokens) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[index]):
                index += 1
            continue
        if name in {"python", "python3"}:
            args = tokens[index + 1 :]
            for arg_index, arg in enumerate(args):
                if arg == "-m" and arg_index + 1 < len(args):
                    module = args[arg_index + 1]
                    if module in {"py_compile", "compileall"}:
                        return ""
                    break
            for arg in args:
                if arg.endswith(".py"):
                    return Path(arg).name.lower()
            return ""
        break
    match = re.search(r"\bpython(?:3)?\s+([A-Za-z0-9_./-]+\.py)\b", _strip_quoted(command))
    return Path(match.group(1)).name.lower() if match else ""


def python_script_name(command: str) -> str:
    for segment in _shell_command_segments(command):
        if _is_file_inspection_segment(segment):
            continue
        script_name = _python_script_name_from_segment(segment)
        if script_name:
            return script_name
    return ""


def job_entrypoint_match(command: str) -> str:
    """Return direct, ambiguous, or empty for Python scripts that look like job runners."""
    script_name = python_script_name(command)
    if not script_name:
        return ""
    stem = Path(script_name).stem
    if stem == "job":
        return "direct"
    tokens = re.split(r"[_.-]+", stem)
    helper_tokens = {"check", "validate", "verify", "test", "tests", "setup", "config", "lint", "probe"}
    action_tokens = {"run", "start", "launch", "execute"}
    if tokens[0] == "job":
        return "" if helper_tokens.intersection(tokens[1:]) else "ambiguous"
    if tokens in (["run", "job"], ["start", "job"], ["launch", "job"], ["execute", "job"]):
        return "direct"
    if "job" not in tokens or helper_tokens.intersection(tokens):
        return ""
    if tokens[-1:] == ["job"] or action_tokens.intersection(tokens):
        return "ambiguous"
    return ""


def is_job_entrypoint_command(command: str) -> bool:
    return bool(job_entrypoint_match(command))


def is_simulation_entrypoint_command(command: str) -> bool:
    script_name = python_script_name(command)
    if not script_name:
        return False
    stem = Path(script_name).stem
    if "simulat" not in stem:
        return False
    tokens = set(re.split(r"[_.-]+", stem))
    return bool(tokens & {"run", "start", "launch", "execute"}) or stem in {"simulate", "simulation", "simulator"}


def is_simulation_or_job_command(command: str) -> bool:
    return is_job_entrypoint_command(command) or is_simulation_entrypoint_command(command)


def _wrapper_target_name(segment: str) -> str:
    """Return the script/target name a segment invokes, or '' when it is not a runner.

    Recognizes ``python foo.py``, ``bash/sh foo.sh``, ``./foo.sh`` and ``make <target>`` so
    non-Python wrappers can be classified alongside Python ones. File-inspection segments are
    not runners and return ''.
    """
    if _is_file_inspection_segment(segment):
        return ""
    script_name = _python_script_name_from_segment(segment)
    if script_name:
        return Path(script_name).stem.lower()
    tokens = _command_tokens(segment)
    if not tokens:
        return ""
    head = Path(tokens[0]).name.lower()
    if head == "make":
        for token in tokens[1:]:
            if not token.startswith("-"):
                return token.lower()
        return ""
    if head in {"bash", "sh", "zsh"}:
        for token in tokens[1:]:
            if token.startswith("-"):
                continue
            return Path(token).stem.lower()
        return ""
    if tokens[0].endswith(".sh"):
        return Path(tokens[0]).stem.lower()
    return ""


def _name_is_simulator_wrapper(name: str) -> bool:
    if not name:
        return False
    tokens = set(re.split(r"[_.-]+", name.lower()))
    helper_tokens = {"check", "validate", "verify", "test", "tests", "setup", "config", "lint", "probe"}
    action_tokens = {"run", "start", "launch", "execute"}
    runtime_tokens = {"job", "nvflare", "simulat", "simulate", "simulation", "simulator"}
    if helper_tokens.intersection(tokens):
        return False
    if name in {"simulate", "simulation", "simulator"}:
        return True
    return bool(action_tokens.intersection(tokens)) and bool(runtime_tokens.intersection(tokens))


def is_nvflare_simulator_wrapper_command(command: str) -> bool:
    return any(
        _name_is_simulator_wrapper(_wrapper_target_name(segment)) for segment in _shell_command_segments(command)
    )


def invokes_nvflare_simulator(command: str, output: str) -> bool:
    command_text = "\n".join(
        segment for segment in _shell_command_segments(command) if not _is_file_inspection_segment(segment)
    )
    if re.search(
        r"\b(?:python(?:3)?\s+-m\s+)?nvflare(?:\.cli)?\s+simulator\b",
        strip_ansi(command_text),
        flags=re.IGNORECASE,
    ):
        return True
    if not is_nvflare_simulator_wrapper_command(command):
        return False
    text = output
    return bool(
        re.search(
            r"\b(?:python(?:3)?\s+-m\s+)?nvflare(?:\.cli)?\s+simulator\b",
            strip_ansi(text),
            flags=re.IGNORECASE,
        )
    )


def _simulator_thread_flag(command: str, output: str) -> str:
    match = re.search(r"\bnvflare(?:\.cli)?\s+simulator\b[^\n]*\s-t\s+(\d+)\b", f"{command}\n{output}")
    return f" ... -t {match.group(1)}" if match else ""


def job_runtime_path(span: dict[str, Any] | None) -> str:
    """Describe the NVFLARE runtime path a successful job span used (SDK render)."""
    if not span:
        return ""
    command = str(span.get("command") or "")
    output = str(span.get("output") or "")
    if (
        "PTClientAPILauncherExecutor" in output
        or "_start_external_process" in output
        or invokes_nvflare_simulator(command, output)
    ):
        thread_flag = _simulator_thread_flag(command, output)
        return f"exported job + `nvflare.cli simulator{thread_flag}` with external client processes"
    if "PTInProcessClientAPIExecutor" in output or re.search(r"\bpython(?:3)?\s+job\.py\b", command):
        return "`recipe.execute(SimEnv(...))` with `PTInProcessClientAPIExecutor`"
    return ""


def longest_successful_job_span(run: dict[str, Any]) -> dict[str, Any] | None:
    spans = _successful_job_spans(run)
    if not spans:
        return None
    return max(spans, key=lambda span: as_number(span.get("duration_seconds")) or 0)


def _python_heredoc_body(segment: str) -> str:
    lines = str(segment or "").splitlines()
    if len(lines) < 2:
        return ""
    match = re.search(r"<<-?\s*['\"]?(?P<tag>[A-Za-z_][A-Za-z0-9_]*)['\"]?", lines[0])
    if not match:
        return ""
    tag = match.group("tag")
    body: list[str] = []
    for line in lines[1:]:
        if line.strip() == tag:
            return "\n".join(body)
        body.append(line)
    return "\n".join(body)


def _python_inline_sources_from_segment(segment: str) -> list[str]:
    heredoc_body = _python_heredoc_body(segment)
    token_source = segment.splitlines()[0] if heredoc_body else segment
    tokens = _strip_execution_prefix_tokens(_command_tokens(token_source))
    if not tokens or Path(tokens[0]).name.lower() not in {"python", "python3"}:
        return []
    args = tokens[1:]
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "-c":
            return [args[index + 1]] if index + 1 < len(args) else []
        if arg.startswith("-c") and len(arg) > 2:
            return [arg[2:]]
        if arg == "-" or arg.startswith("<<"):
            return [heredoc_body] if heredoc_body else []
        if arg in {"-m", "--"}:
            return []
        if arg.startswith("-"):
            index += 2 if arg in {"-W", "-X"} and index + 1 < len(args) else 1
            continue
        return []
    return [heredoc_body] if heredoc_body else []


def _python_source_imports_nvflare(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return _python_statement_list_imports_nvflare(tree.body)


def _python_statement_imports_nvflare(node: ast.stmt) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name == "nvflare" or alias.name.startswith("nvflare.") for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        return module == "nvflare" or module.startswith("nvflare.")
    return False


def _python_statement_list_imports_nvflare(statements: list[ast.stmt]) -> bool:
    for node in statements:
        if _python_statement_imports_nvflare(node):
            return True
        if isinstance(node, ast.If) and _python_if_test_is_inline_main_guard(node.test):
            if _python_statement_list_imports_nvflare(node.body):
                return True
    return False


def _python_source_queries_nvflare_runtime_identity(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return _python_statement_list_queries_nvflare_runtime_identity(tree.body)


def _python_statement_list_queries_nvflare_runtime_identity(
    statements: list[ast.stmt],
    nvflare_names: set[str] | None = None,
    nvflare_version_names: set[str] | None = None,
) -> bool:
    active_nvflare_names = set(nvflare_names or ())
    active_version_names = set(nvflare_version_names or ())
    for node in statements:
        _python_update_nvflare_runtime_bindings(node, active_nvflare_names, active_version_names)
        if _python_statement_prints_nvflare_runtime_identity(node, active_nvflare_names, active_version_names):
            return True
        if isinstance(node, ast.If) and _python_if_test_is_inline_main_guard(node.test):
            if _python_statement_list_queries_nvflare_runtime_identity(
                node.body, active_nvflare_names, active_version_names
            ):
                return True
    return False


def _python_update_nvflare_runtime_bindings(
    node: ast.stmt, nvflare_names: set[str], nvflare_version_names: set[str]
) -> None:
    for name in _python_assigned_names(node):
        nvflare_names.discard(name)
        nvflare_version_names.discard(name)
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name == "nvflare":
                nvflare_names.add(alias.asname or "nvflare")
            elif alias.name.startswith("nvflare.") and not alias.asname:
                nvflare_names.add("nvflare")
    elif isinstance(node, ast.ImportFrom) and node.module == "nvflare":
        for alias in node.names:
            if alias.name == "__version__":
                nvflare_version_names.add(alias.asname or alias.name)


def _python_assigned_names(node: ast.stmt) -> set[str]:
    names: set[str] = set()
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        targets = [node.target]
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        targets = [item.optional_vars for item in node.items if item.optional_vars is not None]
    else:
        return names
    for target in targets:
        names.update(_python_target_names(target))
    return names


def _python_target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for element in node.elts:
            names.update(_python_target_names(element))
        return names
    return set()


def _python_statement_prints_nvflare_runtime_identity(
    node: ast.stmt, nvflare_names: set[str], nvflare_version_names: set[str]
) -> bool:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    call = node.value
    if not isinstance(call.func, ast.Name) or call.func.id != "print":
        return False
    return any(
        _python_expr_queries_nvflare_runtime_identity(arg, nvflare_names, nvflare_version_names) for arg in call.args
    )


def _python_expr_queries_nvflare_runtime_identity(
    node: ast.AST, nvflare_names: set[str], nvflare_version_names: set[str]
) -> bool:
    if isinstance(node, ast.Name):
        return node.id in nvflare_version_names
    if isinstance(node, ast.Attribute):
        return node.attr in {"__file__", "__version__"} and _python_expr_is_nvflare_binding(node.value, nvflare_names)
    if _python_expr_is_nvflare_identity_getattr(node, nvflare_names):
        return True
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            continue
        if _python_expr_queries_nvflare_runtime_identity(child, nvflare_names, nvflare_version_names):
            return True
    return False


def _python_expr_is_nvflare_binding(node: ast.AST, nvflare_names: set[str]) -> bool:
    return isinstance(node, ast.Name) and node.id in nvflare_names


def _python_expr_is_nvflare_identity_getattr(node: ast.AST, nvflare_names: set[str]) -> bool:
    if not isinstance(node, ast.Call) or len(node.args) < 2:
        return False
    if not isinstance(node.func, ast.Name) or node.func.id != "getattr":
        return False
    attr = node.args[1]
    return _python_expr_is_nvflare_binding(node.args[0], nvflare_names) and (
        isinstance(attr, ast.Constant) and attr.value in {"__file__", "__version__"}
    )


def _python_if_test_is_inline_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
        return False
    if not isinstance(node.ops[0], ast.Eq):
        return False
    left = node.left
    right = node.comparators[0]
    return (_is_python_name_dunder_main(left) and _is_python_main_literal(right)) or (
        _is_python_name_dunder_main(right) and _is_python_main_literal(left)
    )


def _is_python_name_dunder_main(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "__name__"


def _is_python_main_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == "__main__"


def _command_imports_nvflare(command: str) -> bool:
    for segment in _shell_command_segments(command):
        if _is_file_inspection_segment(segment):
            continue
        if any(_python_source_imports_nvflare(source) for source in _python_inline_sources_from_segment(segment)):
            return True
    return False


def _command_queries_nvflare_runtime_identity(command: str) -> bool:
    for segment in _shell_command_segments(command):
        if _is_file_inspection_segment(segment):
            continue
        for source in _python_inline_sources_from_segment(segment):
            if _python_source_queries_nvflare_runtime_identity(source):
                return True
    return False


def _output_mentions_nvflare_package_path(output: str) -> bool:
    normalized = str(output or "").replace("\\", "/")
    return bool(
        re.search(
            r"(?i)(?:^|[\s\"'`=:(])(?:[A-Za-z]:)?[^\s\"'`]*?(?:site|dist)-packages/nvflare(?:/|$)",
            normalized,
        )
    )


def _output_mentions_version(value: str) -> bool:
    return bool(re.search(r"\b(?:\d+!)?\d+(?:\.\d+)+(?:[A-Za-z0-9_.!+-]*)?\b", str(value or "")))


def _command_output_proves_nvflare_import(command: str, output: str) -> bool:
    if not _command_imports_nvflare(command):
        return False
    if _output_mentions_nvflare_package_path(output):
        return True
    return _command_queries_nvflare_runtime_identity(command) and _output_mentions_version(output)


def _nvflare_package_available_for_missing_submodule(run: dict[str, Any], event: dict[str, Any]) -> bool:
    """True when a missing ``nvflare.*`` module is better read as a bad internal path.

    A failed ``from nvflare.x.y import ...`` in an exploratory snippet can look like a
    missing dependency, but if another captured command imports NVFLARE successfully,
    the package is installed and the failure is an incorrect submodule path.
    """

    event_id = str(event.get("id") or "")
    for candidate in agent_command_events(run):
        if str(candidate.get("id") or "") == event_id:
            continue
        command = str(candidate.get("command") or "")
        output = str(candidate.get("output") or "")
        if command_succeeded(candidate) and _command_output_proves_nvflare_import(command, output):
            return True
        if _output_mentions_nvflare_package_path(output):
            return True
    return False


def _is_incorrect_nvflare_import_path(run: dict[str, Any], event: dict[str, Any], module_name: str) -> bool:
    return module_name.startswith("nvflare.") and _nvflare_package_available_for_missing_submodule(run, event)


def _missing_module_status_phrase(run: dict[str, Any], event: dict[str, Any], module_name: str) -> str:
    if _is_incorrect_nvflare_import_path(run, event, module_name):
        return f"incorrect NVFLARE import path `{module_name}`"
    return f"missing Python dependency `{module_name}`"


def _missing_module_dependency_evidence(run: dict[str, Any], event: dict[str, Any], module_name: str) -> str:
    if _is_incorrect_nvflare_import_path(run, event, module_name):
        return "installed top-level `nvflare` package was available; failure was an incorrect import path"
    return dependency_install_evidence(run)


def _missing_module_recovery_summary(run: dict[str, Any], event: dict[str, Any], module_name: str) -> str:
    if _is_incorrect_nvflare_import_path(run, event, module_name):
        return (
            f"earlier incorrect NVFLARE import path `{module_name}` was recovered "
            "(installed top-level `nvflare` package was available)"
        )
    return (
        f"earlier missing Python dependency `{module_name}` was recovered "
        f"({dependency_install_evidence_brief(run)})"
    )


def command_failure_rows(run: dict[str, Any]) -> list[dict[str, str]]:
    """Realized command-failure diagnostic rows for a run (SDK interpretation).

    Each row: command, exit, recovery, root_cause, dependency. Returns ALL rows
    (the renderer applies any display limit), so the derived view stays complete.
    """
    events = agent_command_events(run)
    failed_events = [event for event in events if command_failed(event)]
    material_events = [event for event in failed_events if is_material_failed_command(event)]
    selected_events = material_events or [
        event
        for event in failed_events
        if "git status" not in str(event.get("command") or "")
        and "rg: command not found" not in str(event.get("output") or "")
    ]
    diagnostics = []
    for event in selected_events:
        command = str(event.get("command") or "")
        output = str(event.get("output") or "")
        if recovered_by_later_success(event, events):
            recovery = "recovered by a later successful similar command"
        elif recovered_by_later_successful_job(event, events):
            recovery = "recovered by a later successful simulator/job command"
        else:
            recovery = "not recovered in this run"
        dependency_evidence = ""
        missing_module = missing_python_module_name(output)
        root_cause = command_error_summary(output)
        if missing_module:
            dependency_evidence = _missing_module_dependency_evidence(run, event, missing_module)
            timing_reason = _missing_module_timing_reason(run, event, missing_module)
            if timing_reason:
                root_cause = f"{root_cause}; {timing_reason}"
        diagnostics.append(
            {
                "command": inline_code_text(command, 180),
                "exit": str(event.get("exit_code")),
                "recovery": recovery,
                "root_cause": root_cause,
                "dependency": dependency_evidence,
            }
        )
    return diagnostics


def _event_payloads_with_index(run: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    payloads = []
    for index, line in enumerate(str(run.get("agent_events_text") or "").splitlines()):
        try:
            payload = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict):
            payloads.append((index, payload))
    return payloads


def _tool_result_index(payloads: list[tuple[int, dict[str, Any]]], tool_id: str) -> int | None:
    for index, payload in payloads:
        for item in _message_content(payload):
            if item.get("type") == "tool_result" and str(item.get("tool_use_id") or "") == tool_id:
                return index
    return None


def _background_dependency_install_windows(run: dict[str, Any]) -> list[tuple[int, int | None, str]]:
    payloads = _event_payloads_with_index(run)
    background_installs: dict[str, tuple[int, str]] = {}
    tasks_by_tool_id: dict[str, str] = {}
    task_completion: dict[str, int] = {}
    for index, payload in payloads:
        for item in _message_content(payload):
            if item.get("type") != "tool_use" or item.get("name") != "Bash":
                continue
            tool_input = item.get("input") if isinstance(item.get("input"), dict) else {}
            command = str(tool_input.get("command") or payload.get("command_text") or "")
            tool_id = str(item.get("id") or "")
            if tool_id and tool_input.get("run_in_background") and is_dependency_install_command(command):
                background_installs[tool_id] = (index, inline_code_text(command, 100))
        event_type = str(payload.get("event_type") or payload.get("type") or "")
        task_id = str(payload.get("task_id") or "")
        tool_id = str(payload.get("tool_use_id") or "")
        if event_type == "system.task_started" and task_id and tool_id:
            tasks_by_tool_id[tool_id] = task_id
            continue
        if event_type in {"system.task_updated", "system.task_notification"} and task_id:
            status = ""
            if isinstance(payload.get("patch"), dict):
                status = str(payload["patch"].get("status") or "")
            status = status or str(payload.get("status") or "")
            if status.lower() == "completed" and task_id not in task_completion:
                task_completion[task_id] = index

    windows = []
    for tool_id, (start_index, command) in background_installs.items():
        task_id = tasks_by_tool_id.get(tool_id)
        windows.append((start_index, task_completion.get(task_id) if task_id else None, command))
    return windows


def _later_successful_module_probe(run: dict[str, Any], failed_index: int | None, module_name: str) -> str:
    if failed_index is None:
        return ""
    for event in agent_command_events(run):
        if int(event.get("index") or 0) <= failed_index or not command_succeeded(event):
            continue
        command = str(event.get("command") or "")
        output = str(event.get("output") or "")
        if re.search(rf"\bimport\s+{re.escape(module_name)}\b", command) and re.search(
            rf"\b{re.escape(module_name)}\s+[0-9]", output
        ):
            return f"later verification imported `{module_name}` successfully"
    return ""


def _missing_module_timing_reason(run: dict[str, Any], event: dict[str, Any], module_name: str) -> str:
    payloads = _event_payloads_with_index(run)
    result_index = _tool_result_index(payloads, str(event.get("id") or ""))
    if result_index is None:
        return ""
    for start_index, completion_index, install_command in _background_dependency_install_windows(run):
        if start_index < result_index and (completion_index is None or result_index < completion_index):
            detail = (
                f"`{module_name}` was probed while background dependency install `{install_command}` "
                "was still running"
            )
            later_probe = _later_successful_module_probe(run, int(event.get("index") or 0), module_name)
            if later_probe:
                detail += f"; {later_probe}"
            return detail
    return ""


# --- NVFLARE recipe / FL-algorithm detection (step 5b) ---
def _workflow_algorithm_name(workflow_path: str) -> str:
    class_name = str(workflow_path or "").rsplit(".", 1)[-1]
    normalized = re.sub(r"[^a-z0-9]+", "", class_name.lower())
    known = {
        "scaffold": "SCAFFOLD",
        "fedavg": "FedAvg",
        "fedopt": "FedOpt",
        "fedprox": "FedProx",
        "cyclic": "Cyclic",
        "fedeval": "FedEval",
        "scatterandgather": "ScatterAndGather",
    }
    if normalized in known:
        return known[normalized]
    if not class_name:
        return "unknown"
    return re.sub(r"(?<!^)(?=[A-Z])", " ", class_name)


def _workflow_training_score(workflow: dict[str, Any]) -> int:
    workflow_path = str(workflow.get("path") or "")
    class_name = workflow_path.rsplit(".", 1)[-1]
    normalized = re.sub(r"[^a-z0-9]+", "", class_name.lower())
    args = workflow.get("args") if isinstance(workflow.get("args"), dict) else {}
    score = 0
    if args.get("num_rounds") is not None:
        score += 100
    if args.get("train_task_name") or args.get("train_task"):
        score += 80
    if normalized in {"scatterandgather", "scaffold", "fedavg", "fedopt", "fedprox", "cyclic"}:
        score += 60
    if "num_rounds" in workflow:
        score += 30
    if normalized in {"initializeglobalweights", "crosssiteeval", "fedeval"} or re.search(
        r"(?:initialize|evaluation|eval)", workflow_path, flags=re.IGNORECASE
    ):
        score -= 40
    return score


def _recipe_from_generated_source(run: dict[str, Any]) -> str:
    recipe_modules = {
        "cyclic": "cyclic-pt",
        "fedavg": "fedavg-pt",
        "fedavg_he": "fedavg-he-pt",
        "fedeval": "fedeval-pt",
        "fedopt": "fedopt-pt",
        "fedprox": "fedprox-pt",
        "scaffold": "scaffold-pt",
    }
    for _, text in _workspace_python_sources(run):
        for match in re.finditer(
            r"from\s+nvflare\.app_opt\.pt\.recipes\.([A-Za-z0-9_]+)\s+import\s+([A-Za-z0-9_]+Recipe[A-Za-z0-9_]*)",
            text,
        ):
            recipe = recipe_modules.get(match.group(1))
            if recipe:
                return recipe
    return ""


def _first_matching_recipe(text: str, patterns: tuple[str, ...]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def _recipe_show_command_evidence(text: str) -> str:
    matches = re.findall(r"\bnvflare\s+recipe\s+show\s+([A-Za-z0-9_.-]+)", text)
    return matches[-1] if matches else ""


def _recipe_evidence(run: dict[str, Any]) -> str:
    final_text = _final_message_without_event_log(str(run.get("agent_last_message") or ""))
    classification_excerpt = str(run_record(run).get("classification_excerpt") or "")
    final_slice = _final_message_without_event_log(classification_excerpt)
    explicit_patterns = (
        r"\bSelected\s+the\s+recipe\b.*?`([A-Za-z0-9_.-]+)`",
        r"\bselected\s+recipe\b.*?`([A-Za-z0-9_.-]+)`",
    )
    generic_patterns = (
        r"\bRecipe:\*{0,2}\s*`?([A-Za-z0-9_.-]+)`?",
        r"`([A-Za-z0-9_.-]+)`\s*(?:→|->)\s*`?[A-Za-z0-9_.]*Recipe`?",
    )
    explicit_recipe = _first_matching_recipe(final_text, explicit_patterns) or _first_matching_recipe(
        final_slice, explicit_patterns
    )
    if explicit_recipe:
        return explicit_recipe

    text = combined_text(run)
    command_recipe = _recipe_show_command_evidence(text)
    source_recipe = _recipe_from_generated_source(run)
    if source_recipe == "fedavg-pt" and command_recipe == "fedprox-pt":
        return command_recipe
    if source_recipe:
        return source_recipe
    generic_recipe = _first_matching_recipe(final_text, generic_patterns) or _first_matching_recipe(
        final_slice, generic_patterns
    )
    if generic_recipe:
        return generic_recipe
    if command_recipe:
        return command_recipe
    return ""


def _server_config_items(run: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    delta = run_workspace_delta(run)
    items = []
    for key in ("runtime_artifacts", "changed_files", "final_structure_files", "final_files"):
        values = delta.get(key)
        if not isinstance(values, list):
            continue
        for item_index, item in enumerate(values):
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or item.get("artifact_path") or "")
            if Path(path).name != "config_fed_server.json":
                continue
            items.append((key, item_index, item))

    def priority(entry: tuple[str, int, dict[str, Any]]) -> tuple[int, int, int, int, str]:
        key, item_index, item = entry
        path = str(item.get("path") or item.get("artifact_path") or "")
        key_priority = 0 if key == "runtime_artifacts" else 1
        server_priority = 0 if re.search(r"(^|/)(server|app_server)(/|$)", path) else 1
        probe_priority = 1 if "probe" in path.lower() else 0
        return key_priority, server_priority, probe_priority, item_index, path

    return [(key, item) for key, _item_index, item in sorted(items, key=priority)]


def fl_algorithm_info(run: dict[str, Any]) -> dict[str, Any]:
    for key, item in _server_config_items(run):
        path = _workspace_artifact_path(run, item)
        if not path or not path.exists():
            continue
        config = load_json(path, {}) or {}
        workflows = config.get("workflows") if isinstance(config, dict) else None
        if not isinstance(workflows, list):
            continue
        candidates = [
            workflow for workflow in workflows if isinstance(workflow, dict) and str(workflow.get("path") or "")
        ]
        if not candidates:
            continue
        workflow = max(enumerate(candidates), key=lambda entry: (_workflow_training_score(entry[1]), -entry[0]))[1]
        workflow_path = str(workflow.get("path") or "")
        args = workflow.get("args") if isinstance(workflow.get("args"), dict) else {}
        recipe = _recipe_evidence(run)
        evidence_parts = [f"{Path(str(item.get('path') or item.get('artifact_path') or '')).name}: {workflow_path}"]
        if recipe:
            evidence_parts.append(f"recipe {recipe}")
        return {
            "algorithm": _workflow_algorithm_name(workflow_path),
            "evidence": "; ".join(evidence_parts),
            "num_rounds": args.get("num_rounds"),
            "recipe": recipe,
            "source": key,
            "workflow_id": workflow.get("id"),
            "workflow_path": workflow_path,
        }
    if job_run_status(run) == "not_started":
        return {
            "algorithm": "not captured",
            "evidence": "no server workflow config captured; job was not started",
        }
    return {"algorithm": "not captured", "evidence": "no server workflow config captured"}


def _server_config_key_metric(run: dict[str, Any]) -> str:
    for _key, item in _server_config_items(run):
        path = _workspace_artifact_path(run, item)
        if not path or not path.exists():
            continue
        config = load_json(path, {}) or {}
        workflows = config.get("workflows") if isinstance(config, dict) else None
        if not isinstance(workflows, list):
            workflows = []
        components = config.get("components") if isinstance(config, dict) else None
        if not isinstance(components, list):
            components = []
        for entry in [*workflows, *components]:
            if not isinstance(entry, dict):
                continue
            args = entry.get("args")
            if not isinstance(args, dict):
                continue
            key_metric = canonical_metric_name(args.get("key_metric"))
            if key_metric:
                return key_metric
    return ""


_BEST_VALIDATION_METRIC_RE = r"\bnew best validation metric at round\s+(\d+):\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)"


def server_selected_best_metric(run: dict[str, Any], expected_metric: Any = None) -> dict[str, Any]:
    """The FL-level scalar the server's model selector actually selected.

    The metric NAME comes from the captured ``config_fed_server.json``
    (``IntimeModelSelector.key_metric``) and the VALUE from the server log's
    ``new best validation metric at round N: V`` line — both structured runtime
    evidence, so this follows whatever metric the job declares without grepping
    metric names out of log lines. Empty when either half is missing, when the
    declared key_metric is a different metric than ``expected_metric``, or when
    the value is implausible for the metric.
    """

    key_metric = _server_config_key_metric(run)
    if not key_metric:
        return {}
    if expected_metric and not metric_names_match(key_metric, expected_metric):
        return {}
    best: tuple[int, float] | None = None
    source_paths: list[str] = []
    for rel_path, text in _runtime_artifact_texts(run, r"(^|/)server/log(?:_fl)?\.txt$", max_bytes=128_000):
        for line in strip_ansi(text).splitlines():
            best_match = re.search(_BEST_VALIDATION_METRIC_RE, line, flags=re.IGNORECASE)
            if not best_match:
                continue
            candidate = (int(best_match.group(1)), float(best_match.group(2)))
            if best is None or candidate[0] >= best[0]:
                best = candidate
                if rel_path not in source_paths:
                    source_paths.append(rel_path)
    if best is None or not is_plausible_metric_value(key_metric, best[1]):
        return {}
    return {"name": key_metric, "value": best[1], "round": best[0], "source_paths": source_paths}


def recovered_runtime_metric_evidence(run: dict[str, Any]) -> str:
    """Recover non-authoritative metric context from NVFLARE server logs."""

    key_metric = _server_config_key_metric(run)
    if not key_metric:
        return ""
    best: tuple[int, float] | None = None
    fallback_values: list[float] = []
    for _rel_path, text in _runtime_artifact_texts(run, r"(^|/)server/log(?:_fl)?\.txt$", max_bytes=128_000):
        for line in strip_ansi(text).splitlines():
            best_match = re.search(_BEST_VALIDATION_METRIC_RE, line, flags=re.IGNORECASE)
            if best_match:
                candidate = (int(best_match.group(1)), float(best_match.group(2)))
                if best is None or candidate[0] >= best[0]:
                    best = candidate
                continue
            value_match = re.search(
                r"\bvalidation metric\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s+from client\b",
                line,
                flags=re.IGNORECASE,
            )
            if value_match:
                fallback_values.append(float(value_match.group(1)))
    if best is not None:
        round_number, value = best
        return (
            f"{key_metric} {fmt_number(value)} "
            f"(NVFLARE IntimeModelSelector best validation metric at round {round_number})"
        )
    if fallback_values:
        value = fallback_values[-1]
        return f"{key_metric} {fmt_number(value)} (NVFLARE IntimeModelSelector validation metric log)"
    return ""


def _recipe_expected_algorithms(recipe: str) -> set[str]:
    recipe_algorithms = {
        "cyclic-pt": {"Cyclic"},
        "fedavg-he-pt": {"FedAvg", "ScatterAndGather"},
        "fedavg-pt": {"FedAvg", "ScatterAndGather"},
        "fedeval-pt": {"FedEval"},
        "fedopt-pt": {"FedOpt"},
        "fedprox-pt": {"FedProx"},
        "scaffold-pt": {"SCAFFOLD"},
    }
    return recipe_algorithms.get(str(recipe or "").lower(), set())


def fl_algorithm_recipe_mismatch(run: dict[str, Any]) -> str:
    info = fl_algorithm_info(run)
    algorithm = str(info.get("algorithm") or "")
    recipe = str(info.get("recipe") or "")
    expected = _recipe_expected_algorithms(recipe)
    if not algorithm or algorithm == "not captured" or not recipe or not expected:
        return ""
    if algorithm in expected:
        return ""
    return (
        f"runtime workflow `{algorithm}` does not match selected recipe `{recipe}` "
        f"(expected one of: {', '.join(sorted(expected))})."
    )


# --- NVFLARE job-run status (step 5a-cont) ---


def is_material_failed_command(event: dict[str, Any]) -> bool:
    command = str(event.get("command") or "")
    output = str(event.get("output") or "")
    if is_simulation_or_job_command(command):
        return True
    if is_dependency_install_command(command):
        return True
    return bool(
        re.search(
            r"Traceback|RuntimeError|ConfigError|ModuleNotFoundError|No module named|Simulator run failed",
            output,
            flags=re.IGNORECASE,
        )
    )


def _direct_job_exit_is_trustworthy(command: str) -> bool:
    """Return True when the aggregate exit code reflects the direct job's own success.

    A direct ``python job.py`` segment's exit code only stands in for the job when nothing
    runs after it that could mask a failure. That holds when the job is the last segment, or
    when every separator after it is ``&&`` (a non-zero overall exit would otherwise short the
    chain). A trailing ``;`` or ``||`` segment can flip the exit code, so success evidence is
    required instead.
    """
    parts = _shell_command_parts(command)
    job_index = None
    for index, (segment, _operator) in enumerate(parts):
        if job_entrypoint_match(segment) == "direct":
            job_index = index
    if job_index is None:
        return False
    return all(operator == "&&" for _segment, operator in parts[job_index : len(parts) - 1])


def job_command_succeeded(event: dict[str, Any]) -> bool:
    command = str(event.get("command") or "")
    output = str(event.get("output") or "")
    if not command_succeeded(event):
        return False
    if job_output_has_failure_marker(output):
        return False
    job_match = job_entrypoint_match(command)
    if job_match == "direct":
        if _direct_job_exit_is_trustworthy(command):
            return True
        return job_output_succeeded(output)
    if job_match == "ambiguous":
        return job_output_succeeded(output)
    if is_simulation_entrypoint_command(command):
        return job_output_succeeded(output)
    if invokes_nvflare_simulator(command, output):
        return job_output_succeeded(output)
    return False


def recovered_by_later_success(event: dict[str, Any], events: list[dict[str, Any]]) -> bool:
    key = command_recovery_key(str(event.get("command") or ""))
    index = int(event.get("index") or 0)
    for candidate in events:
        if int(candidate.get("index") or 0) <= index:
            continue
        candidate_command = str(candidate.get("command") or "")
        if command_recovery_key(candidate_command) != key:
            continue
        if is_simulation_or_job_command(candidate_command):
            if job_command_succeeded(candidate):
                return True
            continue
        if command_succeeded(candidate):
            return True
    return False


def recovered_by_later_successful_job(event: dict[str, Any], events: list[dict[str, Any]]) -> bool:
    index = int(event.get("index") or 0)
    for candidate in events:
        if int(candidate.get("index") or 0) <= index:
            continue
        if job_command_succeeded(candidate):
            return True
    return False


def last_successful_job_event(run: dict[str, Any]) -> dict[str, Any] | None:
    for event in reversed(agent_command_events(run)):
        if job_command_succeeded(event):
            command = str(event.get("command") or "")
            if "--help" not in command and "--export" not in command:
                return event
    return None


def _runtime_started_but_incomplete(run: dict[str, Any]) -> bool:
    """Return true when captured NVFLARE artifacts show a started, unfinished run."""

    if _has_task_result_artifact(run):
        return False
    progress = _server_progress_summary(run)
    if not progress or "no terminal `Finished` marker was captured" not in progress:
        return False
    metrics = _metrics_artifact_summary(run)
    return "`metrics_summary.json` was not captured" in metrics


_BACKGROUND_INTERRUPTED_STATUSES = {"failed", "killed", "stopped", "cancelled", "canceled", "interrupted"}
_BACKGROUND_COMPLETED_STATUSES = {"complete", "completed", "done", "finished", "success", "succeeded"}
_BACKGROUND_TERMINAL_STATUSES = _BACKGROUND_INTERRUPTED_STATUSES | _BACKGROUND_COMPLETED_STATUSES


def _background_simulation_task_state(run: dict[str, Any]) -> dict[str, Any]:
    """Parse background simulation task events from the agent event stream."""

    payloads = _agent_event_payloads(run)
    background_tools: dict[str, dict[str, str]] = {}
    task_by_tool_id: dict[str, str] = {}
    task_statuses: dict[str, list[dict[str, Any]]] = {}
    result_payload = None
    result_timestamp = None
    result_index = None
    saw_schedule_wakeup = False
    for index, payload in enumerate(payloads):
        event_type = str(payload.get("event_type") or payload.get("type") or "")
        timestamp = parse_event_timestamp(payload.get("harness_timestamp") or payload.get("timestamp"))
        for item in _message_content(payload):
            if item.get("type") == "tool_use" and item.get("name") == "Bash":
                tool_input = item.get("input") if isinstance(item.get("input"), dict) else {}
                command = str(tool_input.get("command") or payload.get("command_text") or "")
                tool_id = str(item.get("id") or "")
                if (
                    tool_id
                    and tool_input.get("run_in_background")
                    and (
                        is_simulation_or_job_command(command)
                        or invokes_nvflare_simulator(command, "")
                        or is_nvflare_simulator_wrapper_command(command)
                    )
                ):
                    background_tools[tool_id] = {
                        "command": command,
                        "description": str(tool_input.get("description") or payload.get("description") or ""),
                    }
            elif item.get("type") == "tool_result":
                tool_id = str(item.get("tool_use_id") or "")
                result = payload.get("tool_use_result") if isinstance(payload.get("tool_use_result"), dict) else {}
                background_task_id = str(result.get("backgroundTaskId") or "")
                if not background_task_id:
                    match = re.search(r"\bbackground with ID:\s*([A-Za-z0-9_-]+)", str(item.get("content") or ""))
                    background_task_id = match.group(1) if match else ""
                if tool_id and background_task_id:
                    task_by_tool_id[tool_id] = background_task_id
            elif item.get("type") == "tool_use" and item.get("name") == "ScheduleWakeup":
                saw_schedule_wakeup = True
        if event_type == "result.success" or (
            str(payload.get("type") or "") == "result" and str(payload.get("subtype") or "") == "success"
        ):
            result_payload = payload
            result_timestamp = timestamp
            result_index = index
            continue
        if event_type == "system.task_started":
            task_id = str(payload.get("task_id") or "")
            tool_id = str(payload.get("tool_use_id") or "")
            if task_id and tool_id:
                task_by_tool_id[tool_id] = task_id
            continue
        if event_type in {"system.task_updated", "system.task_notification"}:
            task_id = str(payload.get("task_id") or "")
            if not task_id:
                continue
            status = ""
            if isinstance(payload.get("patch"), dict):
                status = str(payload["patch"].get("status") or "")
            status = status or str(payload.get("status") or "")
            if status:
                task_statuses.setdefault(task_id, []).append(
                    {"index": index, "status": status.lower(), "timestamp": timestamp}
                )
    return {
        "background_tools": background_tools,
        "task_by_tool_id": task_by_tool_id,
        "task_statuses": task_statuses,
        "result_payload": result_payload,
        "result_timestamp": result_timestamp,
        "result_index": result_index,
        "saw_schedule_wakeup": saw_schedule_wakeup,
    }


def _background_simulation_interruption_status(run: dict[str, Any]) -> str:
    """Classify an unfinished background simulation at agent finalization time."""

    state = _background_simulation_task_state(run)
    background_tools = state["background_tools"]
    task_by_tool_id = state["task_by_tool_id"]
    task_statuses = state["task_statuses"]
    result_timestamp = state["result_timestamp"]
    result_index = state["result_index"]
    if not background_tools or result_index is None:
        return ""

    saw_unfinished_background_task = False
    for tool_id in background_tools:
        task_id = task_by_tool_id.get(tool_id)
        statuses = task_statuses.get(task_id, []) if task_id else []
        terminal_status_records = [
            record for record in statuses if record.get("status") in _BACKGROUND_TERMINAL_STATUSES
        ]
        for record in terminal_status_records:
            if record.get("status") not in _BACKGROUND_INTERRUPTED_STATUSES:
                continue
            status_timestamp = record.get("timestamp")
            if result_timestamp and status_timestamp:
                if status_timestamp >= result_timestamp:
                    return "background_task_killed"
            elif int(record.get("index") or -1) > result_index:
                return "background_task_killed"
        if not terminal_status_records:
            saw_unfinished_background_task = True
    return "agent_left_simulation_running" if saw_unfinished_background_task else ""


def job_run_status(run: dict[str, Any]) -> str:
    """Return a concise job execution status."""
    if not run.get("available"):
        return "unknown"
    executed_events = [
        event
        for event in agent_command_events(run)
        if "--help" not in str(event.get("command") or "")
        and "--export" not in str(event.get("command") or "")
        and (
            is_simulation_or_job_command(str(event.get("command") or ""))
            or invokes_nvflare_simulator(str(event.get("command") or ""), str(event.get("output") or ""))
        )
    ]
    attempted_commands = [
        command
        for command in commands_for_run(run)
        if is_simulation_or_job_command(command) and "--help" not in command and "--export" not in command
    ]
    attempted = bool(executed_events or attempted_commands)
    # Successful evidence (a completed job command or a runtime metric artifact) must win over
    # background interruption classification: a background run can finish and capture results
    # without ever emitting a terminal task-status event, which would otherwise be misread as
    # "agent_left_simulation_running".
    has_success_evidence = last_successful_job_event(run) or _has_task_result_artifact(run)
    if not has_success_evidence:
        background_status = _background_simulation_interruption_status(run)
        if background_status:
            return background_status
    if _runtime_started_but_incomplete(run):
        return "started_failed"
    if last_successful_job_event(run):
        return "completed"
    if _has_task_result_artifact(run):
        return "completed"
    if not attempted:
        return "not_started"
    return "started_failed"


def job_run_status_reason(run: dict[str, Any]) -> str:
    """Return a concise human-readable reason string for the job run status."""
    if not run.get("available"):
        return "run artifacts not available"
    status = job_run_status(run)
    hint_counts = run.get("activity", {}).get("hint_counts") or {}
    sim_count = int(hint_counts.get("simulation", 0) or 0)
    py_count = int(hint_counts.get("python_job_py", 0) or 0)

    # Check Bash blocking first — it's the most actionable reason for not_started
    bash_blocked_count = bash_permission_denial_count(run)

    if status == "not_started":
        failure_category = agent_failure_category(run)
        if exit_code(run) not in (None, 0) and failure_category and failure_category != "agent_unknown_failure":
            evidence = failure_evidence(run)
            if evidence:
                return (
                    "simulation not attempted — agent failed before starting job work "
                    f"({failure_category}: {truncate(evidence, 180)})"
                )
            return f"simulation not attempted — agent failed before starting job work ({failure_category})"
        if bash_blocked_count > 0:
            return (
                f"Bash blocked {bash_blocked_count} time(s) — simulation never ran "
                f"(permission errors prevented tool use)"
            )
        activity = run.get("activity") if isinstance(run.get("activity"), dict) else {}
        denials = activity.get("permission_denials") or []
        if denials:
            denial_summary = "; ".join(str(d) for d in denials[:3])
            return f"simulation not attempted — permission denials: {denial_summary}"
        commands = commands_for_run(run)
        if commands:
            return (
                "simulation not attempted — captured commands did not run job.py "
                f"(first command: `{inline_code_text(commands[0], 120)}`)"
            )
        return "simulation not attempted — no captured job.py or simulator command"

    if status in {"background_task_killed", "agent_left_simulation_running"}:
        parts = [
            _background_task_interruption_cause(run),
            _background_task_interruption_summary(run),
            _server_progress_summary(run),
            _metrics_artifact_summary(run),
        ]
        return "; ".join(part for part in parts if part)

    if status == "started_failed":
        if _runtime_started_but_incomplete(run):
            parts = [
                "simulation started but did not complete",
                _server_progress_summary(run),
                _metrics_artifact_summary(run),
            ]
            return "; ".join(part for part in parts if part)
        if bash_blocked_count > 0:
            return (
                f"simulation command ran but Bash was blocked {bash_blocked_count} time(s); "
                f"simulation did not complete successfully"
            )
        # Look for a failed job event
        events = agent_command_events(run)
        for event in reversed(events):
            if command_failed(event) and is_simulation_or_job_command(str(event.get("command") or "")):
                output = str(event.get("output") or "")
                missing_module = missing_python_module_name(output)
                if missing_module:
                    return (
                        f"simulation command ran but {_missing_module_status_phrase(run, event, missing_module)} — "
                        f"{_missing_module_dependency_evidence(run, event, missing_module)}"
                    )
                summary = command_error_summary(output)
                return f"simulation command ran but exited with error — {truncate(summary, 200)}"
        for event in reversed(events):
            if is_simulation_or_job_command(str(event.get("command") or "")):
                output = str(event.get("output") or "")
                missing_module = missing_python_module_name(output)
                if missing_module:
                    return (
                        f"simulation command ran but {_missing_module_status_phrase(run, event, missing_module)} — "
                        f"{_missing_module_dependency_evidence(run, event, missing_module)}"
                    )
                summary = command_error_summary(output)
                return f"simulation command ran but success was not confirmed — {truncate(summary, 200)}"
        return "simulation command ran but no command output was captured"

    if status == "completed":
        event = last_successful_job_event(run)
        output = str(event.get("output") or "") if event else ""
        recovered_issue = completed_job_recovered_issue_summary(run)
        repeated_runs = repeated_job_run_summary(run)
        fedstats_artifact = _captured_federated_statistics_artifact(run)
        artifact_evidence = _task_result_artifact_evidence(run)
        if artifact_evidence and not event:
            artifact_kind = "task result artifact" if fedstats_artifact else "runtime metric artifact"
            return (
                f"job execution inferred from captured {artifact_kind} — "
                f"{artifact_evidence}; command detector did not identify a direct job.py or simulator command"
            )
        if "Finished" in output:
            reason = "simulation completed — FL workflow reached Finished state"
            if repeated_runs:
                reason = f"{reason}; {repeated_runs}"
            return f"{reason}; {recovered_issue}" if recovered_issue else reason
        if sim_count > 0 or py_count > 0:
            reason = f"simulation completed successfully (hint count: simulation={sim_count}, python_job_py={py_count})"
            if repeated_runs:
                reason = f"{reason}; {repeated_runs}"
            return f"{reason}; {recovered_issue}" if recovered_issue else reason
        reason = "simulation completed successfully"
        if repeated_runs:
            reason = f"{reason}; {repeated_runs}"
        return f"{reason}; {recovered_issue}" if recovered_issue else reason

    return "status unknown — no simulation hint counts or events found"


def _runtime_artifacts(run: dict[str, Any]) -> list[dict[str, Any]]:
    delta = run_workspace_delta(run)
    artifacts = delta.get("runtime_artifacts") if isinstance(delta.get("runtime_artifacts"), list) else []
    return [item for item in artifacts if isinstance(item, dict)]


def _artifact_label(item: dict[str, Any]) -> str:
    return str(item.get("path") or item.get("artifact_path") or "")


def _read_runtime_artifact(run: dict[str, Any], pattern: str, *, max_bytes: int = 128_000) -> tuple[str, str]:
    for item in _runtime_artifacts(run):
        label = _artifact_label(item).replace("\\", "/")
        if not re.search(pattern, label):
            continue
        path = _workspace_artifact_path(run, item)
        if path and path.exists():
            return label, read_text(path, max_bytes=max_bytes)
    return "", ""


def _runtime_artifact_present(run: dict[str, Any], pattern: str) -> bool:
    return any(re.search(pattern, _artifact_label(item).replace("\\", "/")) for item in _runtime_artifacts(run))


def _has_runtime_scalar_result_metric(run: dict[str, Any]) -> bool:
    return artifact_validation_metric_is_runtime_evidence(run)


def _captured_federated_statistics_artifact(run: dict[str, Any]) -> str:
    for item in _runtime_artifacts(run):
        label = _artifact_label(item).replace("\\", "/")
        if not re.search(r"(^|/)server/simulate_job/", label):
            continue
        if re.search(r"(?:fedstats|global_stats|statistics|stats)[^/]*\.json$", label, re.IGNORECASE):
            return label
    return ""


def _has_federated_statistics_result_artifact(run: dict[str, Any]) -> bool:
    return bool(_captured_federated_statistics_artifact(run))


def _has_task_result_artifact(run: dict[str, Any]) -> bool:
    return _has_runtime_scalar_result_metric(run) or _has_federated_statistics_result_artifact(run)


def _task_result_artifact_evidence(run: dict[str, Any]) -> str:
    fedstats_artifact = _captured_federated_statistics_artifact(run)
    if fedstats_artifact:
        return f"captured federated-statistics result artifact `{fedstats_artifact}`"
    return artifact_validation_metric_evidence(run)


def _agent_event_payloads(run: dict[str, Any]) -> list[dict[str, Any]]:
    payloads = []
    for line in str(run.get("agent_events_text") or "").splitlines():
        try:
            payload = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _message_content(payload: dict[str, Any]) -> list[dict[str, Any]]:
    message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
    content = message.get("content")
    return [item for item in content if isinstance(item, dict)] if isinstance(content, list) else []


def _background_task_interruption_summary(run: dict[str, Any]) -> str:
    background_tools: dict[str, dict[str, str]] = {}
    task_starts: dict[str, dict[str, str]] = {}
    task_updates: dict[str, list[str]] = {}
    task_notifications: dict[str, list[str]] = {}
    for payload in _agent_event_payloads(run):
        for item in _message_content(payload):
            if item.get("type") != "tool_use" or item.get("name") != "Bash":
                continue
            tool_input = item.get("input") if isinstance(item.get("input"), dict) else {}
            if not tool_input.get("run_in_background"):
                continue
            tool_id = str(item.get("id") or "")
            if not tool_id:
                continue
            background_tools[tool_id] = {
                "command": str(tool_input.get("command") or payload.get("command_text") or ""),
                "description": str(tool_input.get("description") or payload.get("description") or ""),
            }
        event_type = str(payload.get("event_type") or payload.get("type") or "")
        task_id = str(payload.get("task_id") or "")
        if not task_id:
            continue
        if event_type == "system.task_started":
            task_starts[task_id] = {
                "description": str(payload.get("description") or ""),
                "tool_use_id": str(payload.get("tool_use_id") or ""),
            }
            continue
        if event_type == "system.task_updated":
            patch = payload.get("patch") if isinstance(payload.get("patch"), dict) else {}
            status = str(patch.get("status") or "")
            if status:
                task_updates.setdefault(task_id, []).append(status)
            continue
        if event_type == "system.task_notification":
            status = str(payload.get("status") or "")
            if status:
                task_notifications.setdefault(task_id, []).append(status)

    interrupted_statuses = {"failed", "killed", "stopped", "cancelled", "canceled", "interrupted"}
    for task_id, started in task_starts.items():
        tool_id = started.get("tool_use_id") or ""
        if tool_id not in background_tools:
            continue
        updates = [status for status in task_updates.get(task_id, []) if status.lower() in interrupted_statuses]
        notifications = [
            status for status in task_notifications.get(task_id, []) if status.lower() in interrupted_statuses
        ]
        if not updates and not notifications:
            continue
        details = []
        description = started.get("description") or background_tools[tool_id].get("description")
        if description:
            details.append(description)
        if updates:
            details.append("task update " + ", ".join(f"`{status}`" for status in updates))
        if notifications:
            details.append("notification " + ", ".join(f"`{status}`" for status in notifications))
        return f"Background task `{task_id}` was interrupted after launch ({'; '.join(details)})."
    return ""


# --- NVFLARE generated-code quality comparison -----------------------------


def _runtime_python_sources(run: dict[str, Any], *, max_files: int = 8) -> list[tuple[str, str]]:
    delta = run_workspace_delta(run)
    values = delta.get("runtime_artifacts")
    if not isinstance(values, list):
        return []
    items: list[tuple[int, int, str, dict[str, Any]]] = []
    for item_index, item in enumerate(values):
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or item.get("artifact_path") or "")
        if Path(path).suffix != ".py" or "/custom/" not in path:
            continue
        probe_priority = 1 if "probe" in path.lower() else 0
        server_priority = 0 if "/server/simulate_job/" in path else 1
        items.append((probe_priority, server_priority, path, item))

    sources = []
    for _probe_priority, _server_priority, rel_path, item in sorted(items, key=lambda entry: entry[:3]):
        artifact_path = _workspace_artifact_path(run, item)
        if not artifact_path or not artifact_path.exists():
            continue
        sources.append((rel_path, read_text(artifact_path, max_bytes=128_000)))
        if len(sources) >= max_files:
            break
    return sources


def _workspace_text(run: dict[str, Any]) -> str:
    sources = list(_workspace_python_sources(run))
    seen_names = {Path(rel_path).name for rel_path, _text in sources}
    for rel_path, text in _runtime_python_sources(run, max_files=16):
        name = Path(rel_path).name
        if name in seen_names:
            continue
        sources.append((rel_path, text))
        seen_names.add(name)
        if len(sources) >= 16:
            break
    return "\n\n".join(f"# {rel_path}\n{text}" for rel_path, text in sources[:16])


def _detect_training_control_path(text: str) -> str:
    if "nvflare.client.lightning" in text or "flare.patch(trainer)" in text:
        return "Lightning Client API patch (`flare.patch(trainer)`)"
    if "flare.receive" in text and "flare.send" in text and "FLModel(" in text:
        return "manual Client API loop (`receive` / train / `send FLModel`)"
    if "flare.receive" in text or "flare.send" in text:
        return "manual Client API loop"
    return "not captured"


def _detect_partitioning(text: str) -> str:
    if "stratified_partition_frame" in text or (
        "np.array_split(indices, num_sites)" in text and "random_state=seed + site_index" in text
    ):
        return "stratified seeded site partition"
    if re.search(r"\.sample\([^)]*random_state\s*=\s*seed", text) and "iloc[index::num_clients]" in text:
        return "seeded shuffled site partition"
    if "self.site_index :: self.num_clients" in text or "iloc[self.site_index :: self.num_clients]" in text:
        return "deterministic stride partition without shuffle"
    if "site_partition(" in text:
        return "site partition helper"
    return "not captured"


def _detect_class_weighting(text: str) -> str:
    if "positive_class_weight(train_frame" in text:
        return "per-site loss weight from local training partition"
    if "datamodule.pos_weight" in text or re.search(r"self\.pos_weight\s*=\s*neg_count\s*/\s*max\(pos_count", text):
        return "per-site loss weight from local training partition"
    if "neg_count / max(pos_count" in text:
        return "loss weight computed from loaded training data"
    pos_weight_match = re.search(r"['\"]?pos_weight['\"]?\s*:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not pos_weight_match:
        pos_weight_match = re.search(r"--pos-weight['\"]?\s*,\s*['\"]([0-9]+(?:\.[0-9]+)?)['\"]", text)
    if pos_weight_match:
        return f"fixed/global `pos_weight={pos_weight_match.group(1)}` passed to clients"
    if "--pos-weight" in text:
        return "fixed/global `pos_weight` passed to clients"
    return "not captured"


def _detect_metric_reporting(text: str) -> str:
    if "BinaryAUROC" in text or "torchmetrics" in text:
        return "Lightning/torchmetrics validation metrics"
    if "binary_auroc(" in text:
        return "manual AUROC from validation predictions"
    if "metrics=" in text and "FLModel(" in text:
        return "manual client-reported metric dict"
    return "not captured"


_RUN_WORKSPACE_PATH_MARKERS = ("/tmp/nvflare", "simulate_job", "simulator_workspace")
# Token-ish "data"/"dataset(s)" (underscores/slashes/dots count as separators,
# letters do not: `train_data` and `data_dir` match, `database` does not) or a
# data-file extension.
_DATA_PATH_HINT_RE = re.compile(
    r"(?<![a-z0-9])data(?:sets?)?(?![a-z])|\.csv\b|\.parquet\b|\.npz\b|\.jsonl?\b", re.IGNORECASE
)
_DATA_IDENTIFIER_PARTS = {"data", "dataset", "datasets"}
_DATA_PATH_IDENTIFIER_PARTS = {"dir", "directory", "root", "path"}
_IDENTIFIER_PART_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+")
_ADD_FILE_DATA_PATH_KWARGS = {
    "dest_dir",
    "dest_path",
    "dst",
    "dst_dir",
    "file",
    "file_path",
    "path",
    "source",
    "source_path",
    "src",
    "src_path",
}


def _identifier_parts(identifier: str) -> list[str]:
    parts: list[str] = []
    for chunk in re.split(r"[^0-9A-Za-z]+", identifier):
        if not chunk:
            continue
        parts.extend(match.group(0).lower() for match in _IDENTIFIER_PART_RE.finditer(chunk))
    return parts


def _identifier_points_at_data_path(identifier: str) -> bool:
    parts = _identifier_parts(identifier)
    return any(part in _DATA_IDENTIFIER_PARTS for part in parts) and any(
        part in _DATA_PATH_IDENTIFIER_PARTS for part in parts
    )


def _source_identifiers_point_at_data_path(source: str) -> bool:
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        return any(token.type == tokenize.NAME and _identifier_points_at_data_path(token.string) for token in tokens)
    except (IndentationError, SyntaxError, tokenize.TokenError):
        pass
    return any(
        _identifier_points_at_data_path(match.group(0)) for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\b", source)
    )


def _string_token_value(token_value: str) -> str:
    try:
        value = ast.literal_eval(token_value)
    except (SyntaxError, ValueError):
        without_prefix = re.sub(r"^[rRuUbBfF]*", "", token_value)
        for quote in ('"""', "'''", '"', "'"):
            if without_prefix.startswith(quote) and without_prefix.endswith(quote):
                return without_prefix[len(quote) : -len(quote)]
        return token_value
    return value if isinstance(value, str) else ""


def _source_string_literal_spans(text: str) -> list[tuple[str, int]]:
    spans: list[tuple[str, int]] = []
    line_offsets = [0]
    for line in text.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(line))
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        for token in tokens:
            if token.type == tokenize.STRING:
                row, column = token.start
                spans.append((_string_token_value(token.string), line_offsets[row - 1] + column))
        return spans
    except (IndentationError, SyntaxError, tokenize.TokenError):
        spans.clear()

    # Fallback for malformed snippets. Skip comment-only lines and drop inline comments
    # before scanning for simple quoted values.
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            offset += len(line)
            continue
        code = line.split("#", 1)[0]
        spans.extend((match.group(1), offset + match.start()) for match in re.finditer(r"""["']([^"']*)["']""", code))
        offset += len(line)
    return spans


def _source_string_literals(text: str) -> list[str]:
    return [literal for literal, _offset in _source_string_literal_spans(text)]


def _ephemeral_workspace_data_path_marker(text: str) -> str:
    # Only actual string literals count. Source headers and comments routinely mention
    # simulate_job/runtime workspace paths without reading data from them.
    for literal in _source_string_literals(text):
        marker = next((marker for marker in _RUN_WORKSPACE_PATH_MARKERS if marker in literal), "")
        if marker and _DATA_PATH_HINT_RE.search(literal):
            return marker
    return ""


def _hardcoded_absolute_data_path(text: str) -> str:
    for literal, offset in _source_string_literal_spans(text):
        if not re.fullmatch(r"/\S+", literal):
            continue
        if not _DATA_PATH_HINT_RE.search(literal):
            continue
        prefix = text[max(0, offset - 48) : offset]
        # An absolute path is acceptable as a configurable-arg default in simulation.
        if re.search(r"default\s*=\s*(?:str\(\s*)?(?:Path\(\s*)?$", prefix):
            continue
        return literal
    return ""


_MAX_CALL_SOURCE_CHARS = 20_000


def _iter_call_sources(text: str, function_name: str) -> list[str]:
    calls: list[str] = []
    pattern = re.compile(rf"(?<!\w){re.escape(function_name)}\s*\(")
    for match in pattern.finditer(text):
        start = match.end() - 1
        depth = 0
        quote = ""
        triple = False
        escaped = False
        index = start
        # Bounded scan: a real call fits well inside this window; malformed
        # input (an unclosed paren) must not turn the scan quadratic.
        scan_limit = min(len(text), start + _MAX_CALL_SOURCE_CHARS)
        while index < scan_limit:
            char = text[index]
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif triple and text.startswith(quote * 3, index):
                    quote = ""
                    triple = False
                    index += 2
                elif not triple and char == quote:
                    quote = ""
                index += 1
                continue
            if char in {"'", '"'}:
                quote = char
                triple = text.startswith(char * 3, index)
                if triple:
                    index += 3
                else:
                    index += 1
                continue
            if char == "#":
                newline = text.find("\n", index)
                if newline == -1:
                    break
                index = newline + 1
                continue
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    calls.append(text[match.start() : index + 1])
                    break
            index += 1
    return calls


def _path_literal_points_at_data(literal: str) -> bool:
    normalized = literal.replace("\\", "/").lower().strip()
    if not normalized:
        return False
    parts = [part for part in normalized.split("/") if part]
    if any(part in {"data", "dataset", "datasets"} for part in parts):
        return True
    basename = parts[-1] if parts else normalized
    stem = basename.split(".", 1)[0]
    if stem in {"data", "dataset", "datasets"} and re.search(r"\.(?:csv|json|jsonl|parquet|npz)(?:$|[?#])", basename):
        return True
    return bool(re.search(r"\.(?:csv|parquet|npz|jsonl)(?:$|[?#])", normalized))


def _parsed_call_source(call: str) -> ast.Call | None:
    try:
        expression = ast.parse(call, mode="eval").body
    except SyntaxError:
        return None
    return expression if isinstance(expression, ast.Call) else None


def _node_identifiers_point_at_data_path(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and _identifier_points_at_data_path(child.id):
            return True
        if isinstance(child, ast.Attribute) and _identifier_points_at_data_path(child.attr):
            return True
    return False


def _call_arg_points_at_data(call: str, node: ast.AST, *, inspect_literals: bool) -> bool:
    if _node_identifiers_point_at_data_path(node):
        return True
    if inspect_literals:
        source = ast.get_source_segment(call, node) or ""
        return any(_path_literal_points_at_data(literal) for literal in _source_string_literals(source))
    return False


_NON_DATA_SOURCE_LITERAL_RE = re.compile(
    r"(?:requirements[^/'\"]*\.txt|\.py|\.md|\.toml|\.cfg|\.ini|\.yaml|\.yml|readme|license)$",
    re.IGNORECASE,
)


def _add_file_source_is_non_data(expression: ast.Call, call: str) -> bool:
    """True when the copied SOURCE is clearly not a dataset (requirements file,
    module, docs/config) — shipping it to a `data/` dest dir is not dataset
    packaging."""

    if not expression.args:
        return False
    source = ast.get_source_segment(call, expression.args[0]) or ""
    literals = _source_string_literals(source)
    return bool(literals) and all(
        _NON_DATA_SOURCE_LITERAL_RE.search(literal.strip().rstrip("'\"")) for literal in literals
    )


def _add_file_to_clients_copies_data(text: str) -> bool:
    for call in _iter_call_sources(text, "add_file_to_clients"):
        expression = _parsed_call_source(call)
        if expression is not None and _add_file_source_is_non_data(expression, call):
            continue
        if expression is None:
            if _source_identifiers_point_at_data_path(call):
                return True
            if any(_path_literal_points_at_data(literal) for literal in _source_string_literals(call)):
                return True
            continue
        if any(_call_arg_points_at_data(call, arg, inspect_literals=True) for arg in expression.args):
            return True
        for keyword in expression.keywords:
            if keyword.arg and _identifier_points_at_data_path(keyword.arg):
                return True
            inspect_literals = keyword.arg in _ADD_FILE_DATA_PATH_KWARGS if keyword.arg else True
            if _call_arg_points_at_data(call, keyword.value, inspect_literals=inspect_literals):
                return True
    return False


def _detect_data_packaging(text: str) -> str:
    marker = _ephemeral_workspace_data_path_marker(text)
    if marker:
        return f"data path points into ephemeral nvflare run workspace (`{marker}`)"
    if _add_file_to_clients_copies_data(text):
        return "copies dataset into client app; clients read it from the ephemeral run workspace"
    hardcoded = _hardcoded_absolute_data_path(text)
    if hardcoded:
        return f"hardcoded absolute data path in generated client code (`{hardcoded}`)"
    if re.search(
        r"add_argument\(\s*['\"]--data[-_](?:root|dir|path)['\"](?:(?!add_argument\().)*?\bdefault\s*=",
        text,
        re.DOTALL,
    ):
        return "configurable data_root argument with default, overridable per site, pointing at original data"
    if "--data-dir {args.data_dir.resolve()}" in text or "args.data_dir.resolve()" in text:
        return "passes original data path to clients via configurable data-dir argument"
    if "--data-dir" in text or "--data-root" in text:
        return "passes data directory argument"
    return "not captured"


_RECIPE_ENV_MARKERS = (
    (r"\bSimEnv\b|\bSimulatorEnv\b", "simulator env"),
    (r"\bPocEnv\b", "POC env"),
    (r"\bProdEnv\b|\bProductionEnv\b", "production env"),
    (r"\bExecEnv\b", "exec env"),
)

_RECIPE_ENV_CLASS_NAMES = frozenset({"SimEnv", "SimulatorEnv", "PocEnv", "ProdEnv", "ProductionEnv", "ExecEnv"})


def _called_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _call_receives_recipe_env(node: ast.Call) -> bool:
    for value in list(node.args) + [keyword.value for keyword in node.keywords]:
        for child in ast.walk(value):
            if isinstance(child, ast.Name) and child.id in _RECIPE_ENV_CLASS_NAMES:
                return True
            if isinstance(child, ast.Attribute) and child.attr in _RECIPE_ENV_CLASS_NAMES:
                return True
    return False


def _execution_receiver_is_recipe(func: ast.Attribute, recipe_vars: set[str]) -> bool:
    receiver = func.value
    if isinstance(receiver, ast.Name):
        return receiver.id in recipe_vars or "recipe" in receiver.id.lower()
    if isinstance(receiver, ast.Attribute):
        return "recipe" in receiver.attr.lower()
    if isinstance(receiver, ast.Call):
        return _called_name(receiver).endswith("Recipe")
    return False


def _recipe_job_evidence(text: str) -> tuple[str, bool]:
    """(recipe class name, executed) parsed from Python syntax.

    A bare `*Recipe` token, `nvflare.recipe` import, or a comment mention is
    not evidence of a recipe-based job. `executed` is True only when a recipe
    is actually run: an `execute`/`export` call on a recipe (a name bound to a
    recipe constructor, a recipe-named receiver, or a chained `*Recipe(...)`
    call), or an `execute`/`export` call handed an explicit
    SimEnv/PocEnv/ProdEnv environment.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _recipe_job_evidence_unparsed(text)
    recipe_name = ""
    recipe_vars: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _called_name(node).endswith("Recipe"):
            recipe_name = recipe_name or _called_name(node)
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and _called_name(node.value).endswith("Recipe")
        ):
            recipe_vars.update(target.id for target in node.targets if isinstance(target, ast.Name))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in ("execute", "export"):
            continue
        if _execution_receiver_is_recipe(node.func, recipe_vars) or _call_receives_recipe_env(node):
            return recipe_name or "Recipe", True
    return recipe_name, False


def _recipe_job_evidence_unparsed(text: str) -> tuple[str, bool]:
    """Fallback when the concatenated sources do not parse as one module:
    verify individual call expressions instead of trusting bare tokens."""

    recipe_name = ""
    for match in re.finditer(r"\b(\w*Recipe)\s*\(", text):
        name = match.group(1)
        if any(_parsed_call_source(call) is not None for call in _iter_call_sources(text, name)):
            recipe_name = name
            break
    if not recipe_name:
        return "", False
    env_names = "|".join(sorted(_RECIPE_ENV_CLASS_NAMES))
    executed = bool(
        re.search(r"\b\w*[Rr]ecipe\w*\s*\.\s*(?:execute|export)\s*\(", text)
        or re.search(rf"\.\s*(?:execute|export)\s*\(\s*(?:\w+\s*=\s*)?(?:{env_names})\s*\(", text)
    )
    return recipe_name, executed


def _detect_execution_model(text: str) -> str:
    if "launch_external_process=False" in text:
        details = ["in-process Client API executor"]
        if "server_expected_format=ExchangeFormat.PYTORCH" in text:
            details.append("PyTorch exchange format")
        if "TransferType.FULL" in text:
            details.append("full parameter transfer")
        return "; ".join(details)
    if "launch_external_process=True" in text:
        return "external client process runner"
    # Recipe API (newer NVFLARE): the client execution mode is a recipe/env
    # default, so there is no launch_external_process flag to read — the launch
    # style lives in the Recipe + its environment (SimEnv/PocEnv/ProdEnv). A
    # detector that only looked for the legacy flag reported "not captured" for
    # these (correct) recipe-based conversions. Only a recipe that is actually
    # constructed and run counts — an unused import, comment mention, or
    # constructor-only stub is not a client execution model.
    name, executed = _recipe_job_evidence(text)
    if executed:
        env = next((label for pattern, label in _RECIPE_ENV_MARKERS if re.search(pattern, text)), "")
        details = [f"recipe-based job ({name})" + (f" via {env}" if env else "")]
        if "ExchangeFormat.PYTORCH" in text:
            details.append("PyTorch exchange format")
        if "TransferType.FULL" in text:
            details.append("full parameter transfer")
        return "; ".join(details)
    return "not captured"


def _round_metric_items(run: dict[str, Any]) -> list[tuple[int, int, dict[str, Any]]]:
    delta = run_workspace_delta(run)
    items: list[tuple[int, int, dict[str, Any]]] = []
    for key_priority, key in enumerate(("runtime_artifacts", "changed_files", "final_structure_files", "final_files")):
        values = delta.get(key)
        if not isinstance(values, list):
            continue
        for item_index, item in enumerate(values):
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or item.get("artifact_path") or "")
            if Path(path).name != "round_metrics.jsonl":
                continue
            probe_priority = 1 if "probe" in path.lower() else 0
            items.append((key_priority + probe_priority * 10, item_index, item))
    return sorted(items, key=lambda entry: (entry[0], entry[1], str(entry[2].get("path") or "")))


def _expected_metric_name(run: dict[str, Any]) -> str:
    metric = run.get("validation_metric") if isinstance(run.get("validation_metric"), dict) else {}
    name = canonical_metric_name(metric.get("name")) if isinstance(metric, dict) else ""
    if name:
        return name
    record = run_record(run)
    metric = (
        record.get("reported_validation_metric") if isinstance(record.get("reported_validation_metric"), dict) else {}
    )
    return canonical_metric_name(metric.get("name")) if isinstance(metric, dict) else ""


def _round_metric_progression(run: dict[str, Any]) -> str:
    expected = _expected_metric_name(run)
    if not expected:
        return "not captured"
    for _priority, _item_index, item in _round_metric_items(run):
        path = _workspace_artifact_path(run, item)
        if not path or not path.exists():
            continue
        values: list[float] = []
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for line in lines:
            try:
                payload = json.loads(line)
            except Exception:
                continue
            metrics = payload.get("aggregated_metrics") if isinstance(payload, dict) else None
            if not isinstance(metrics, list):
                continue
            for metric in metrics:
                if not isinstance(metric, dict) or not metric_names_match(metric.get("name"), expected):
                    continue
                value = as_number(metric.get("value"))
                if value is not None:
                    values.append(value)
                break
        if values:
            rendered = " -> ".join(f"{value:.4f}" for value in values)
            if len(values) > 1 and max(values) - min(values) < 1e-9:
                rendered += " (flat)"
            return f"{expected} {rendered}"
    return "not captured"


def _native_behavior_profile(run: dict[str, Any]) -> dict[str, str]:
    record = run_record(run)
    if not isinstance(record, dict):
        return {}
    profile = {}
    for category in ("mandatory_behavior", "prohibited_behavior", "optional_behavior"):
        behaviors = record.get(category)
        if not isinstance(behaviors, dict):
            continue
        for behavior_id, entry in behaviors.items():
            if not behavior_id:
                continue
            if not isinstance(entry, dict):
                entry = {}
            status = str(entry.get("status") or "missing")
            evidence = str(entry.get("evidence") or "No behavior evidence supplied.")
            profile[f"{category}__{behavior_id}"] = f"status={status}; {evidence}"
    return profile


def conversion_quality_profile(run: dict[str, Any]) -> dict[str, str]:
    # The deterministic detectors below are conversion-shaped (training path,
    # partitioning, round metrics); for any other evaluation task they would
    # produce misleading evidence, so the profile carries only the native
    # behavior signals and each task criterion renders "not captured" until
    # the code-eval agent judges it (or a task detector lands).
    profile: dict[str, str] = {}
    if _resolved_evaluation_task(run) == "conversion":
        text = _workspace_text(run)
        profile = {
            "training_control": _detect_training_control_path(text),
            "partitioning": _detect_partitioning(text),
            "class_weighting": _detect_class_weighting(text),
            "metric_reporting": _detect_metric_reporting(text),
            "data_packaging": _detect_data_packaging(text),
            "execution_model": _detect_execution_model(text),
            "metric_progression": _round_metric_progression(run),
        }
    profile.update(_native_behavior_profile(run))
    return profile


def _target_framework(run: dict[str, Any] | None) -> str:
    if not isinstance(run, dict):
        return ""
    text = " ".join(str(run.get(key) or "") for key in ("framework", "job_name", "job_slug", "scenario_name")).lower()
    if "lightning" in text:
        return "lightning"
    if "pytorch" in text or "torch" in text:
        return "pytorch"
    return ""


def _evaluation_rules_path(run: dict[str, Any] | None) -> Path | None:
    if not isinstance(run, dict):
        return None
    path = run.get("evaluation_rules_path")
    candidate = Path(path) if isinstance(path, (str, Path)) else None
    return candidate if candidate and candidate.exists() else None


def _declared_evaluation_task(run: dict[str, Any] | None) -> str:
    if not isinstance(run, dict):
        return ""
    for source in (run, run.get("run"), run.get("run_plan_entry")):
        task = source.get("evaluation_task") if isinstance(source, dict) else None
        if isinstance(task, str) and task.strip():
            return task.strip()
    return ""


def _declared_evaluation_selectors(run: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(run, dict):
        return {}
    # Same source order as _declared_evaluation_task: bundle top-level, then
    # the nested record summary, then the plan entry — so direct and nested
    # callers resolve identically.
    for source in (run, run.get("run"), run.get("run_plan_entry")):
        selectors = source.get("evaluation_selectors") if isinstance(source, dict) else None
        if isinstance(selectors, dict) and selectors:
            return {str(key): str(value) for key, value in selectors.items() if value}
    return {}


def _inferred_evaluation_task(run: dict[str, Any] | None) -> str:
    if isinstance(run, dict) and _has_federated_statistics_result_artifact(run):
        return "federated-statistics"
    return ""


def _run_evaluation_rules(run: dict[str, Any] | None = None):
    """Composed rules for the run's DECLARED task (scenario `evaluation_task`).

    Scenario-declared routing fails CLOSED: a typoed task or selector returns a
    routing-error rules object (empty signals + ``routing_error``) instead of
    silently scoring different criteria, and the report renders the failure as
    a bad row. Only the detector-derived framework hint stays lenient — an
    unregistered detected framework falls back to the task's common rules."""

    declared_selectors = _declared_evaluation_selectors(run)
    path = _evaluation_rules_path(run)
    declared_task = _declared_evaluation_task(run)
    task = declared_task or _inferred_evaluation_task(run) or None
    if declared_task or declared_selectors:
        try:
            load_evaluation_rules("nvflare", path, task=task, selectors=declared_selectors, strict_selectors=True)
        except ValueError as exc:
            return {
                "sdk": "nvflare",
                "task": task or "",
                "signals": {},
                "scoring": {},
                "routing_error": f"declared evaluation routing is invalid: {exc}",
            }
    selectors = dict(declared_selectors)
    framework = _target_framework(run)
    if framework:
        selectors.setdefault("framework", framework)
    return load_evaluation_rules("nvflare", path, task=task, selectors=selectors)


def _resolved_evaluation_task(run: dict[str, Any] | None) -> str:
    # Self-contained rules FILES compose no manifest and carry no task key;
    # they inherit the declared task, defaulting to conversion (legacy shape).
    return str(
        _run_evaluation_rules(run).get("task")
        or _declared_evaluation_task(run)
        or _inferred_evaluation_task(run)
        or "conversion"
    )


def conversion_quality_score(signal: str, value: str, run: dict[str, Any] | None = None) -> str:
    # The verdict rules are standard evaluation input (benchmark/config/evaluation/nvflare/,
    # composed as manifest -> task common -> declared/detected overlays) applied
    # through the SDK-neutral scorer, so external tools can score the same signal
    # profile outside the reporting engine and get identical verdicts.
    framework = _target_framework(run)
    rules = _run_evaluation_rules(run)
    return score_signal(rules, signal, value, {"target_framework": framework})


def _background_task_interruption_cause(run: dict[str, Any]) -> str:
    state = _background_simulation_task_state(run)
    background_tools = state["background_tools"]
    task_by_tool_id = state["task_by_tool_id"]
    task_statuses = state["task_statuses"]
    result_payload = state["result_payload"]
    result_timestamp = state["result_timestamp"]
    result_index = state["result_index"]
    saw_schedule_wakeup = bool(state["saw_schedule_wakeup"])
    if not background_tools or result_payload is None or result_index is None:
        return ""

    interrupted_timestamp = None
    saw_interrupted_after_result = False
    saw_unfinished_background_task = False
    for tool_id in background_tools:
        task_id = task_by_tool_id.get(tool_id)
        statuses = task_statuses.get(task_id, []) if task_id else []
        terminal_status_records = [
            record for record in statuses if record.get("status") in _BACKGROUND_TERMINAL_STATUSES
        ]
        for record in terminal_status_records:
            if record.get("status") not in _BACKGROUND_INTERRUPTED_STATUSES:
                continue
            status_timestamp = record.get("timestamp")
            status_after_result = False
            if result_timestamp and status_timestamp:
                status_after_result = status_timestamp >= result_timestamp
            else:
                status_after_result = int(record.get("index") or -1) > result_index
            if not status_after_result:
                continue
            saw_interrupted_after_result = True
            if status_timestamp and (interrupted_timestamp is None or status_timestamp > interrupted_timestamp):
                interrupted_timestamp = status_timestamp
        if not terminal_status_records:
            saw_unfinished_background_task = True

    if not saw_interrupted_after_result and not saw_unfinished_background_task:
        return ""

    stop_reason = str(result_payload.get("stop_reason") or "not captured")
    terminal_reason = str(result_payload.get("terminal_reason") or result_payload.get("subtype") or "not captured")
    parts = [
        "agent run ended while the background simulation was still running",
        f"stop_reason `{stop_reason}`",
        f"terminal_reason `{terminal_reason}`",
    ]
    if result_timestamp and interrupted_timestamp:
        delta = round((interrupted_timestamp - result_timestamp).total_seconds())
        if delta >= 0:
            parts.append(f"task was killed/stopped {delta}s after the agent result")
    if saw_schedule_wakeup:
        parts.append("scheduled wakeup did not keep the non-interactive benchmark run alive")
    return "; ".join(parts) + "."


def _server_progress_summary(run: dict[str, Any]) -> str:
    _label, text = _read_runtime_artifact(run, r"(^|/)server/log_fl\.txt$")
    if not text:
        return ""
    rounds = [int(match.group(1)) for match in re.finditer(r"\bRound\s+(\d+)\s+started\b", text)]
    aggregations = re.findall(r"\baggregating\s+(\d+)\s+update\(s\)\s+at round\s+(\d+)\b", text, re.IGNORECASE)
    finished = bool(re.search(r"\bFinished\b|\bEnd\s+Scaffold\b", text, re.IGNORECASE))
    parts = []
    if rounds:
        parts.append(f"server log reached `Round {max(rounds)} started`")
    if aggregations:
        updates, round_number = aggregations[-1]
        parts.append(f"round {round_number} aggregated {updates} update(s)")
    if not finished:
        parts.append("no terminal `Finished` marker was captured")
    return "; ".join(parts)


def _metrics_artifact_summary(run: dict[str, Any]) -> str:
    has_summary = _runtime_artifact_present(run, r"(^|/)server/simulate_job/metrics/metrics_summary\.json$")
    _label, round_metrics = _read_runtime_artifact(
        run, r"(^|/)server/simulate_job/metrics/round_metrics\.jsonl$", max_bytes=64_000
    )
    if has_summary:
        return "`metrics_summary.json` was captured."
    if round_metrics:
        row_count = sum(1 for line in round_metrics.splitlines() if line.strip())
        return (
            f"`round_metrics.jsonl` was captured with {row_count} non-empty row(s), "
            "but `metrics_summary.json` was not captured."
        )
    return "`metrics_summary.json` was not captured."


def _error_log_summary(run: dict[str, Any]) -> str:
    _label, text = _read_runtime_artifact(run, r"(^|/)server/error_log\.txt$", max_bytes=32_000)
    if text == "":
        if _runtime_artifact_present(run, r"(^|/)server/error_log\.txt$"):
            return "`server/error_log.txt` is empty; no NVFLARE/Python exception was captured there."
        return ""
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return f"`server/error_log.txt` captured: {truncate(first_line, 180)}"


def result_failure_root_cause_block(run: dict[str, Any]) -> str:
    """Explain missing NVFLARE result metrics from captured runtime artifacts."""

    if not run.get("available") or _has_task_result_artifact(run):
        return ""
    # This diagnostic explains a missing TRAINING result scalar
    # (metrics_summary.json). Tasks whose result is a different artifact —
    # federated-statistics writes statistics JSON, never a metrics summary —
    # must not be told their "final metrics summary was not produced".
    if _resolved_evaluation_task(run) != "conversion":
        return ""
    status = job_run_status(run)
    if status not in {
        "started_failed",
        "background_task_killed",
        "agent_left_simulation_running",
    } and not _runtime_started_but_incomplete(run):
        return ""
    rows = [
        ("Interruption cause", _background_task_interruption_cause(run)),
        ("Background task", _background_task_interruption_summary(run)),
        ("Simulation progress", _server_progress_summary(run)),
        ("Metric artifacts", _metrics_artifact_summary(run)),
        ("Error log", _error_log_summary(run)),
    ]
    rows = [(label, value) for label, value in rows if value]
    if not rows:
        return ""
    if status in {"background_task_killed", "agent_left_simulation_running"}:
        explanation = (
            "The captured evidence points to an incomplete NVFLARE simulation: the agent ended while the "
            "simulation was still active, so the server never reached a terminal completion state and the "
            "final metrics summary was not produced."
        )
    else:
        explanation = (
            "The captured evidence points to an incomplete NVFLARE simulation: the simulation started but did "
            "not reach a terminal completion state, so the final metrics summary was not produced."
        )
    lines = [
        "**Root cause of missing FL result**",
        "",
        explanation,
        "",
        "| Evidence | What it shows |",
        "|---|---|",
    ]
    for label, value in rows:
        lines.append(f"| {markdown_cell(label)} | {markdown_cell(value)} |")
    return "\n".join(lines)


def completed_job_recovered_issue_summary(run: dict[str, Any]) -> str:
    parts = []
    blocked_count = bash_permission_denial_count(run)
    if blocked_count:
        parts.append(f"Bash/tool permission was blocked {blocked_count} time(s) before a later job command completed")
    for event in agent_command_events(run):
        if not command_failed(event) or not is_material_failed_command(event):
            continue
        events = agent_command_events(run)
        if not (recovered_by_later_success(event, events) or recovered_by_later_successful_job(event, events)):
            continue
        output = str(event.get("output") or "")
        missing_module = missing_python_module_name(output)
        if missing_module:
            parts.append(_missing_module_recovery_summary(run, event, missing_module))
        else:
            parts.append(f"earlier command failure was recovered ({truncate(command_error_summary(output), 160)})")
        break
    return "; ".join(parts)


def _successful_job_spans(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        span
        for span in agent_command_spans(run)
        if job_command_succeeded(span)
        and "--help" not in str(span.get("command") or "")
        and "--export" not in str(span.get("command") or "")
    ]


def repeated_job_run_summary(run: dict[str, Any]) -> str:
    spans = _successful_job_spans(run)
    if len(spans) <= 1:
        return ""
    total = fmt_seconds_with_unit(_span_total_seconds(spans))
    reason = _job_rerun_reason(spans, run)
    return (
        f"{len(spans)} successful job/simulator executions captured (total job time {total}; likely reason: {reason})"
    )


# --- NVFLARE generated-code-quality assessment (step 5c) ---


def _first_match(pattern: str, text: str, *, flags: int = re.IGNORECASE | re.MULTILINE) -> str:
    match = re.search(pattern, text, flags=flags)
    return match.group(0).strip() if match else ""


def _data_split_signal(run: dict[str, Any]) -> str:
    text = _workspace_file_text(run, "client.py") or _all_python_workspace_text(run)
    if not text:
        return "not captured"
    signals = []
    if re.search(r"\b(?:site_index|site_name|client_id|rank)\b", text, flags=re.IGNORECASE):
        signals.append("site-aware")
    if re.search(
        r"\b(?:array_split|iloc\s*\[.*::|partition(?:_\w*)?|\w*shard\w*|split_indices)\b",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        signals.append("explicit sharding")
    if re.search(r"\bvalid(?:_frame|_loader|ation)?\b", text, flags=re.IGNORECASE):
        signals.append("validation data referenced")
    if re.search(r"\btest(?:_frame|_loader)?\b", text, flags=re.IGNORECASE):
        signals.append("test data referenced")
    if not signals:
        return "no explicit client data split detected"
    return ", ".join(dict.fromkeys(signals))


def _api_pattern_signal(run: dict[str, Any]) -> str:
    text = _generated_python_source_text(run)
    if not text:
        return "not captured"
    if "flare.is_running" in text or re.search(r"\bflare\.(?:receive|send)\s*\(", text):
        return "Client API loop pattern"
    if re.search(r"\bclass\s+\w+\s*\([^)]*ModelLearner", text) or re.search(
        r"\bdef\s+train\s*\([^)]*\bFLModel\b", text
    ):
        return "ModelLearner pattern"
    if "FLModel" in text:
        return "FLModel-based pattern"
    return "no explicit NVFLARE client API pattern detected"


def _generated_python_source_text(run: dict[str, Any]) -> str:
    client_text = _workspace_file_text(run, "client.py")
    if client_text:
        return client_text
    ranked = sorted(_workspace_python_sources(run), key=lambda item: _client_training_source_score(*item), reverse=True)
    if ranked and _client_training_source_score(*ranked[0]) > 0:
        return f"# {ranked[0][0]}\n{ranked[0][1]}"
    return _all_python_workspace_text(run)


def _client_training_source_score(rel_path: str, text: str) -> int:
    score = 0
    name = Path(rel_path).name.lower()
    lowered = text.lower()
    if name == "client.py":
        score += 80
    if "flare.is_running" in text:
        score += 90
    if re.search(r"\bdef\s+train\s*\([^)]*\bFLModel\b", text):
        score += 90
    if "flmodel" in lowered or "params_type" in lowered:
        score += 50
    if "modellearner" in lowered or "learner" in name:
        score += 40
    if "current_round" in text or "total_rounds" in text:
        score += 30
    if "site_name" in text or "client_index" in text:
        score += 20
    if re.search(DATA_LOAD_PATTERN, text) or re.search(DATA_LOADER_PATTERN, text):
        score += 10
    if re.search(LOSS_OPTIMIZER_BUILD_PATTERN, text):
        score += 10
    if re.search(r"\bevaluate\s*\(", text):
        score += 10
    if re.search(r"(^|/)(?:server|app_server)(/|$)", rel_path):
        score -= 20
    return score


def _fl_client_loop_body(source_text: str) -> tuple[str, bool]:
    loop_match = re.search(
        r"\bwhile\s+flare\.is_running\s*\(\)\s*:(?P<body>.*?)(?:\n# [^\n]+\.py\n|\Z)",
        source_text,
        flags=re.DOTALL,
    )
    if loop_match:
        return loop_match.group("body"), True
    train_match = re.search(
        r"(?m)^[ \t]+def\s+train\s*\([^)]*\bFLModel\b[^)]*\)\s*(?:->[^\n:]+)?\s*:\s*(?P<body>.*?)(?=^[ \t]+def\s+|\nclass\s+|\Z)",
        source_text,
        flags=re.DOTALL,
    )
    if train_match:
        return train_match.group("body"), True
    return "", False


LOSS_OPTIMIZER_BUILD_PATTERN = (
    r"\bbuild_loss_and_optimizer\s*\("
    r"|\b(?:criterion|loss_fn|loss_func|loss_function)\s*="
    r"|\boptimizer\s*="
    r"|\btorch\.optim\."
    r"|\boptim\.[A-Za-z_][A-Za-z0-9_]*\s*\("
)

DATA_LOAD_PATTERN = r"\bload_(?:split|data_frames)\s*\(|\bread_csv\s*\(|\bload_dataset\s*\("

DATA_LOADER_PATTERN = r"\bmake_loader\s*\(|\bbuild_data_loaders\s*\(|\bDataLoader\s*\("


def _loss_optimizer_lifecycle_signal(run: dict[str, Any]) -> str:
    source_text = _generated_python_source_text(run)
    if not source_text:
        return "not captured"
    loop_body, loop_found = _fl_client_loop_body(source_text)
    if loop_found and re.search(LOSS_OPTIMIZER_BUILD_PATTERN, loop_body):
        return "loss/optimizer rebuilt inside FL loop"
    if re.search(LOSS_OPTIMIZER_BUILD_PATTERN, source_text):
        return (
            "loss/optimizer built outside FL loop"
            if loop_found
            else "loss/optimizer setup present; FL loop not captured"
        )
    return "no loss/optimizer lifecycle signal detected"


def _data_loader_lifecycle_signal(run: dict[str, Any]) -> str:
    source_text = _generated_python_source_text(run)
    if not source_text:
        return "not captured"
    loop_body, loop_found = _fl_client_loop_body(source_text)
    signals = []
    if not loop_found:
        if re.search(DATA_LOAD_PATTERN, source_text):
            signals.append("data loading present")
        if re.search(DATA_LOADER_PATTERN, source_text):
            signals.append("DataLoader construction present")
        if signals:
            return f"{', '.join(signals)}; FL loop not captured"
        return "no data/DataLoader lifecycle signal detected"
    if loop_found and re.search(DATA_LOAD_PATTERN, loop_body):
        signals.append("data loaded inside FL loop")
    elif re.search(DATA_LOAD_PATTERN, source_text):
        signals.append("data loaded before FL loop")
    if loop_found and re.search(DATA_LOADER_PATTERN, loop_body):
        signals.append("DataLoader built inside FL loop")
    elif re.search(DATA_LOADER_PATTERN, source_text):
        signals.append("DataLoader built before FL loop")
    if signals:
        return ", ".join(signals)
    return "no data/DataLoader lifecycle signal detected"


def _metric_work_signal(run: dict[str, Any]) -> str:
    client_text = _generated_python_source_text(run)
    if not client_text:
        return "not captured"
    body, loop_found = _fl_client_loop_body(client_text)
    if not loop_found:
        body = client_text
    eval_calls = len(re.findall(r"\bevaluate\s*\(", body))
    signals = []
    if eval_calls:
        scope = "in FL loop" if loop_found else "in generated code"
        signals.append(f"{eval_calls} evaluate call(s) {scope}")
    if re.search(r"\btest_(?:frame|loader|metrics)\b", body, flags=re.IGNORECASE):
        signals.append("test evaluation inside FL loop" if loop_found else "test evaluation present")
    if re.search(r"\bglobal_metrics\b", body) and re.search(r"\blocal_metrics\b", body):
        signals.append("global and local metrics reported")
    if re.search(r"\bmetrics\.jsonl\b|append_record\s*\(", body):
        signals.append("per-round metrics sidecar written")
    if signals and not loop_found:
        signals.append("FL loop not captured")
    return ", ".join(signals) if signals else "no per-round metric workload detected"


def _observability_signal(run: dict[str, Any]) -> str:
    source_text = _generated_python_source_text(run)
    mode_dir = run.get("mode_dir")
    logs = ""
    if isinstance(mode_dir, Path):
        logs = "\n".join(
            read_text(path, max_bytes=128_000)
            for path in sorted(mode_dir.glob("workspace_delta/runtime_artifacts/**/log.txt"))
        )
    signals = []
    if re.search(r"round\s+\{?.*epoch|epoch\s+\{?", source_text, flags=re.IGNORECASE):
        signals.append("generated code prints per-epoch progress")
    if re.search(r"\bmetrics?\.(?:jsonl|json|csv|tsv)\b|append_record\s*\(", source_text, flags=re.IGNORECASE):
        signals.append("generated code writes per-round metric sidecar")
    metric_artifacts = _metric_artifact_paths(run)
    if metric_artifacts:
        signals.append(f"captured metric artifact(s): {', '.join(metric_artifacts[:3])}")
    if re.search(r"\bround\s+\d+\s+epoch\b", logs, flags=re.IGNORECASE):
        signals.append("runtime logs show per-epoch progress")
    if re.search(r"\bdevice=", logs):
        signals.append(_first_match(r"\bdevice=[A-Za-z0-9_:-]+", logs))
    if not signals:
        return "limited per-round progress evidence"
    return ", ".join(dict.fromkeys(signals))


def _metric_artifact_paths(run: dict[str, Any]) -> list[str]:
    delta = run_workspace_delta(run)
    paths = []
    seen: set[str] = set()
    for key in ("runtime_artifacts", "changed_files", "final_structure_files"):
        values = delta.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            candidates = [str(item.get(name) or "") for name in ("path", "source_path", "artifact_path")]
            if not any(
                re.search(r"\bmetrics?\b|[_/-]metrics?[_./-]", value, flags=re.IGNORECASE) for value in candidates
            ):
                continue
            display_path = candidates[0] or candidates[1] or candidates[2]
            if not display_path or display_path in seen:
                continue
            seen.add(display_path)
            paths.append(display_path)
    return paths


def _runtime_output_locality_signal(run: dict[str, Any]) -> str:
    delta = run_workspace_delta(run)
    runtime_artifacts = delta.get("runtime_artifacts") if isinstance(delta.get("runtime_artifacts"), list) else []
    changed_paths = manifest_paths(run, "changed_files")
    signals = []
    source_paths = [
        str(item.get("source_path") or "")
        for item in runtime_artifacts
        if isinstance(item, dict) and item.get("source_path")
    ]
    if source_paths:
        if any(path.startswith("/tmp/") for path in source_paths):
            signals.append("runtime artifacts captured separately from temp/runtime paths")
        else:
            signals.append("runtime artifacts captured separately")
    if any(
        re.search(r"(^|/)(?:server|site-[^/]+|simulate_job)(/|$)", path)
        or re.search(r"(^|/)(?:log(?:_fl)?\.txt|metrics_summary\.json|round_metrics\.jsonl)$", path)
        for path in changed_paths
    ):
        signals.append("runtime output appears in workspace changes")
    return ", ".join(dict.fromkeys(signals)) if signals else "no runtime-output locality evidence"


def _dependency_strategy_signal(run: dict[str, Any]) -> str:
    install_events = dependency_install_events(run)
    if not install_events:
        return dependency_install_evidence_brief(run)
    succeeded = [event for event in install_events if command_succeeded(event)]
    failed = [event for event in install_events if command_failed(event)]
    event = (succeeded or failed or install_events)[-1]
    command = str(event.get("command") or "")
    output = str(event.get("output") or "")
    text = f"{command}\n{output}".lower()
    parts = []
    if "-r" in command and "requirements" in command:
        parts.append("requirements-file install")
    elif "pip install" in command:
        parts.append("targeted package install")
    if "download.pytorch.org/whl/cpu" in text or "+cpu" in text:
        parts.append("CPU-only framework wheel")
    if re.search(r"\bnvidia-(?:cuda|cudnn|cublas|cusolver|nccl|cufft|curand)|\btriton\b|cuda-toolkit", text):
        parts.append("accelerator-capable dependency stack")
    if command_succeeded(event):
        parts.append("succeeded")
    elif command_failed(event):
        parts.append("failed")
    if (
        run.get("skills") == "with skills"
        and "requirements-file install" not in parts
        and "CPU-only framework wheel" in parts
    ):
        parts.append("skill requirements install not followed")
    if not parts:
        parts.append(dependency_install_evidence_brief(run))
    return ", ".join(dict.fromkeys(parts))


def _assessment_from_data_split(evidence: str) -> str:
    if evidence == "not captured":
        return "unknown"
    if "site-aware" in evidence and "explicit sharding" in evidence:
        return "good"
    if "site-aware" in evidence or "explicit sharding" in evidence:
        return "caution"
    return "poor"


def _assessment_from_loss_optimizer_lifecycle(evidence: str) -> str:
    if evidence == "not captured":
        return "unknown"
    if "rebuilt inside FL loop" in evidence:
        return "poor"
    if "FL loop not captured" in evidence:
        return "caution"
    if "built outside FL loop" in evidence or "before FL loop" in evidence:
        return "good"
    return "unknown"


def _assessment_from_data_loader_lifecycle(evidence: str) -> str:
    if evidence == "not captured":
        return "unknown"
    if "data loaded inside FL loop" in evidence or "DataLoader built inside FL loop" in evidence:
        return "poor"
    if "FL loop not captured" in evidence:
        return "caution"
    if "data loaded before FL loop" in evidence or "DataLoader built before FL loop" in evidence:
        return "good"
    return "unknown"


def _assessment_from_metric_work(evidence: str) -> str:
    if evidence == "not captured":
        return "unknown"
    if (
        "sidecar written" in evidence
        or "global and local metrics reported" in evidence
        or "test evaluation" in evidence
    ):
        return "good"
    if "evaluate call" in evidence:
        return "caution"
    return "poor"


def _assessment_from_observability(evidence: str) -> str:
    if evidence == "not captured":
        return "unknown"
    if "per-epoch progress" in evidence or "device=" in evidence or "metric" in evidence:
        return "good"
    if "limited" in evidence:
        return "caution"
    return "unknown"


def _runtime_export_location_signal(run: dict[str, Any]) -> str:
    if not run.get("available"):
        return "not captured"
    roots = _workspace_runtime_or_export_tree_roots(run)
    nested_source = _nested_runtime_or_export_source_folders(run)
    signals = []
    if roots:
        signals.append(f"runtime/export outputs in source workspace: {_short_path_list(roots)}")
    if nested_source:
        signals.append(f"export/runtime copies of generated source: {_short_path_list(nested_source)}")
    if run.get("skills") == "with skills" and roots:
        signals.append("skill runtime-output path not followed")
    return ", ".join(signals) if signals else "not captured"


def _assessment_from_runtime_export_location(evidence: str) -> str:
    lowered = evidence.lower()
    if not lowered or lowered == "not captured":
        return "unknown"
    if "skill runtime-output path not followed" in lowered:
        return "poor"
    if "runtime/export outputs in source workspace" in lowered:
        return "caution"
    return "unknown"


def _assessment_from_locality(evidence: str) -> str:
    if evidence == "not captured" or "no runtime-output" in evidence:
        return "unknown"
    if "workspace changes" in evidence:
        return "caution"
    if "separately" in evidence:
        return "good"
    return "unknown"


def _assessment_from_dependency(evidence: str) -> str:
    lowered = evidence.lower()
    if "skill requirements install not followed" in lowered or "failed" in lowered:
        return "poor"
    if "no dependency install" in lowered or "not captured" in lowered:
        return "unknown"
    if "requirements-file install" in lowered and "succeeded" in lowered:
        return "good"
    if "cpu-only framework wheel" in lowered:
        return "caution"
    if "succeeded" in lowered:
        return "good"
    return "unknown"


CODE_QUALITY_ROWS = (
    ("Client data split/use", _data_split_signal, _assessment_from_data_split),
    ("Loss/optimizer lifecycle", _loss_optimizer_lifecycle_signal, _assessment_from_loss_optimizer_lifecycle),
    ("Data/DataLoader lifecycle", _data_loader_lifecycle_signal, _assessment_from_data_loader_lifecycle),
    ("Per-round metric workload", _metric_work_signal, _assessment_from_metric_work),
    ("Runtime observability", _observability_signal, _assessment_from_observability),
    ("Runtime/export output location", _runtime_export_location_signal, _assessment_from_runtime_export_location),
    ("Runtime/output locality", _runtime_output_locality_signal, _assessment_from_locality),
    ("Dependency install strategy", _dependency_strategy_signal, _assessment_from_dependency),
)

CODE_QUALITY_CONTEXT_ROWS = (("API pattern", _api_pattern_signal),)


def _conversion_quality_rows_from_rules(run: dict[str, Any] | None = None) -> tuple[tuple[str, str], ...]:
    """One report row per evaluation criterion in benchmark/config/evaluation/nvflare.yaml.

    The rules file is the single source of truth for the criteria list: a
    criterion added there appears in the Generated Code Quality table even
    before a detector captures evidence for it (rendered as not captured)."""

    rules = _run_evaluation_rules(run)
    signals = rules.get("signals")
    task_label = str(rules.get("task") or "conversion").replace("-", " ").title()
    rows: list[tuple[str, str]] = []
    for key, entry in (signals or {}).items():
        label = entry.get("label") if isinstance(entry, dict) else None
        rows.append((str(key), str(label or f"{task_label}: {key}")))
    return tuple(rows)


CONVERSION_QUALITY_ROWS = _conversion_quality_rows_from_rules()


def _code_quality_points(run: dict[str, Any] | None = None) -> dict[str, float]:
    return verdict_points(_run_evaluation_rules(run))


# Module-level view kept for callers/tests; sourced from the evaluation rules file.
CODE_QUALITY_POINTS = _code_quality_points()


def code_quality_criteria(run: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """The rule-signal criteria (key + human description) an evaluation agent
    judges the generated code against — sourced from the evaluation rules so
    the list stays in sync with the report rows."""

    return [{"key": key, "description": label} for key, label in _conversion_quality_rows_from_rules(run)]


def _agent_code_quality_verdicts(run: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Per-criterion verdicts an evaluation agent produced (see code_eval), or
    ``{}``. The agent reads the captured code and judges each criterion, so it
    is robust to code shapes the deterministic detectors do not recognize."""

    assessment = run.get("code_quality_assessment") if isinstance(run, dict) else None
    verdicts = assessment.get("assessments") if isinstance(assessment, dict) else None
    return verdicts if isinstance(verdicts, dict) else {}


def generated_code_quality_assessments(run: dict[str, Any]) -> list[tuple[str, str, str]]:
    rows = []
    # The static detector rows are conversion/training-shaped (loss/optimizer,
    # per-round metrics, ...): they only render — and only score — for the
    # conversion task, so another task cannot be dragged down by them.
    if _resolved_evaluation_task(run) == "conversion":
        for label, evidence_getter, assessment_getter in CODE_QUALITY_ROWS:
            evidence = evidence_getter(run)
            rows.append((label, assessment_getter(evidence), evidence))
    rules = _run_evaluation_rules(run)
    routing_error = str(rules.get("routing_error") or "")
    if routing_error:
        # Fail closed and visibly: no criteria are scored for a typoed
        # declaration; the single bad row names the routing failure.
        rows.append(("Evaluation routing", "bad", routing_error))
        return rows
    # Prefer an evaluation agent's verdicts when present (it read the code and
    # judged each criterion); fall back to the deterministic detector + rules.
    agent_verdicts = _agent_code_quality_verdicts(run)
    profile = conversion_quality_profile(run)
    for key, label in _conversion_quality_rows_from_rules(run):
        agent = agent_verdicts.get(key)
        if isinstance(agent, dict) and agent.get("verdict"):
            rows.append((label, str(agent.get("verdict")), str(agent.get("evidence") or "")))
            continue
        evidence = profile.get(key, "not captured")
        rows.append((label, conversion_quality_score(key, evidence, run), evidence))
    return rows


def generated_code_quality_overall(run: dict[str, Any]) -> str:
    assessments = generated_code_quality_assessments(run)
    points_by_status = _code_quality_points(run)
    known = [(status, evidence) for _, status, evidence in assessments if status in points_by_status]
    total = len(assessments)
    if not known:
        return "unknown: no generated-code evidence captured"
    points = sum(points_by_status[status] for status, _ in known)
    score_ratio = points / total
    thresholds = overall_thresholds(_run_evaluation_rules(run))
    if score_ratio >= thresholds.get("good", 0.8):
        label = "good"
    elif score_ratio >= thresholds.get("caution", 0.5):
        label = "caution"
    else:
        label = "poor"
    unknown_count = total - len(known)
    unknown_note = f"; {len(known)}/{total} scored, {unknown_count} unknown" if unknown_count else ""
    return f"{label}: {points:.1f}/{total} evidence points{unknown_note}"


def generated_code_quality_score(run: dict[str, Any]) -> float | None:
    assessments = generated_code_quality_assessments(run)
    if not assessments:
        return None
    points_by_status = _code_quality_points(run)
    known = [status for _, status, _ in assessments if status in points_by_status]
    if not known:
        return None
    return sum(points_by_status[status] for status in known) / len(assessments)


# --- NVFLARE runtime-path log parsers (E3): consumed by the report plugin's
# ``explain()`` runtime-path note; never re-read the result root. ---


def _successful_non_install_command_spans(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        span
        for span in agent_command_spans(run.raw)
        if command_succeeded(span) and not is_dependency_install_command(str(span.get("command") or ""))
    ]


def _max_download_tx_elapsed(output: str) -> float | None:
    values = [float(match.group(1)) for match in re.finditer(r"\bdownload tx\b[^\n]*\belapsed=([0-9.]+)s", output)]
    return max(values) if values else None


def _round_durations_from_output(output: str) -> list[tuple[int, float]]:
    starts: dict[int, datetime] = {}
    durations: list[tuple[int, float]] = []
    current_round: int | None = None
    last_timestamp: datetime | None = None
    for line in strip_ansi(output).splitlines():
        timestamp_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", line)
        if timestamp_match:
            last_timestamp = parse_event_timestamp(timestamp_match.group(1))
        round_match = re.search(r"\bRound\s+(\d+)\s+started\b", line)
        if round_match and last_timestamp:
            current_round = int(round_match.group(1))
            starts[current_round] = last_timestamp
            continue
        if re.search(r"\bAggregated\s+(\d+)/\1\s+results\b", line) and current_round is not None and last_timestamp:
            start = starts.get(current_round)
            if start:
                durations.append((current_round, (last_timestamp - start).total_seconds()))
            current_round = None
    return durations


def _runtime_artifact_texts(run: dict[str, Any], pattern: str, *, max_bytes: int = 128_000) -> list[tuple[str, str]]:
    delta = run_workspace_delta(run)
    texts: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in delta.get("runtime_artifacts") or []:
        if not isinstance(item, dict):
            continue
        rel_path = str(item.get("path") or item.get("artifact_path") or "").replace("\\", "/")
        if not rel_path or rel_path in seen or not re.search(pattern, rel_path):
            continue
        path = _workspace_artifact_path(run, item)
        if not path or not path.exists():
            continue
        texts.append((rel_path, read_text(path, max_bytes=max_bytes)))
        seen.add(rel_path)
    return texts


def _log_timestamp(line: str) -> datetime | None:
    match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", line)
    return parse_event_timestamp(match.group(1)) if match else None


def _lightning_site_fit_timings(run: dict[str, Any]) -> list[dict[str, Any]]:
    timings: list[dict[str, Any]] = []
    for rel_path, text in _runtime_artifact_texts(run, r"(^|/)site-[^/]+/log\.txt$"):
        site_match = re.search(r"(^|/)(site-[^/]+)/log\.txt$", rel_path)
        site = site_match.group(2) if site_match else "site"
        current: dict[str, Any] | None = None
        for line in strip_ansi(text).splitlines():
            timestamp = _log_timestamp(line)
            round_match = re.search(r"\bsite-[^|]+\|\s*round=(\d+)\b", line)
            if round_match and timestamp:
                current = {
                    "site": site,
                    "round": int(round_match.group(1)),
                    "start": timestamp,
                    "max_epochs": None,
                }
                continue
            stop_match = re.search(r"`Trainer\.fit`\s+stopped:\s+`max_epochs=(\d+)`\s+reached", line)
            if stop_match and timestamp and current:
                current["stop"] = timestamp
                current["seconds"] = (timestamp - current["start"]).total_seconds()
                current["max_epochs"] = int(stop_match.group(1))
                timings.append(current)
                current = None
    return timings


def _client_config_epoch_values(run: dict[str, Any]) -> list[int]:
    values: list[int] = []
    for _rel_path, text in _runtime_artifact_texts(run, r"(^|/)config_fed_client\.json$"):
        try:
            payload = json.loads(text)
        except Exception:
            continue
        executors = payload.get("executors") if isinstance(payload, dict) else None
        if not isinstance(executors, list):
            continue
        for executor_entry in executors:
            if not isinstance(executor_entry, dict):
                continue
            executor = executor_entry.get("executor")
            args = executor.get("args") if isinstance(executor, dict) and isinstance(executor.get("args"), dict) else {}
            task_args = str(args.get("task_script_args") or "")
            match = re.search(r"(?:^|\s)--epochs\s+(\d+)(?:\s|$)", task_args)
            if match:
                values.append(int(match.group(1)))
    return values


def _reuses_patched_lightning_trainer_across_rounds(run: dict[str, Any]) -> bool:
    text = _generated_python_source_text(run)
    if not text:
        return False
    loop_match = re.search(r"\bwhile\s+flare\.is_running\s*\(\)\s*:", text)
    if not loop_match:
        return False
    pre_loop = text[: loop_match.start()]
    loop_body, has_loop = _fl_client_loop_body(text)
    return bool(
        has_loop
        and re.search(r"\btrainer\s*=\s*pl\.Trainer\s*\(", pre_loop)
        and re.search(r"\bflare\.patch\s*\(\s*trainer\s*\)", pre_loop)
        and re.search(r"\btrainer\.fit\s*\(", loop_body)
    )


def _group_timings_by_round(timings: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for timing in timings:
        round_number = int(timing.get("round") or 0)
        if round_number > 0:
            groups.setdefault(round_number, []).append(timing)
    return groups


def lightning_slow_round_diagnostics(run: dict[str, Any], server_round: int, min_seconds: float = 300) -> str:
    timings = _lightning_site_fit_timings(run)
    if not timings:
        return ""
    client_round = server_round + 1
    round_timings = [timing for timing in timings if timing.get("round") == client_round]
    if not round_timings:
        groups = _group_timings_by_round(timings)
        if not groups:
            return ""
        client_round, round_timings = max(
            groups.items(),
            key=lambda item: max(float(timing.get("seconds") or 0) for timing in item[1]),
        )
    max_fit = max(float(timing.get("seconds") or 0) for timing in round_timings)
    if max_fit < min_seconds:
        return ""

    round_maxes = {
        round_number: max(float(timing.get("seconds") or 0) for timing in group)
        for round_number, group in _group_timings_by_round(timings).items()
    }
    previous_round_parts = [
        f"round {round_number} max {fmt_seconds_with_unit(seconds)}"
        for round_number, seconds in sorted(round_maxes.items())
        if round_number < client_round
    ]
    site_parts = [
        f"{timing.get('site')} {fmt_seconds_with_unit(float(timing.get('seconds') or 0))}"
        for timing in sorted(round_timings, key=lambda item: str(item.get("site") or ""))
    ]
    epochs = sorted({int(timing.get("max_epochs") or 0) for timing in round_timings if timing.get("max_epochs")})
    configured_epochs = sorted(set(_client_config_epoch_values(run)))

    parts = [
        f"site logs isolate the slowdown to server Round {server_round} / client round {client_round}",
    ]
    if previous_round_parts:
        parts.append(f"previous client rounds were shorter ({'; '.join(previous_round_parts)})")
    parts.append(f"client round {client_round} fit timings: {', '.join(site_parts)}")
    parts.append("this points to local Lightning `Trainer.fit`, not NVFLARE transfer/aggregation")
    if epochs:
        parts.append(f"Lightning stopped at max_epochs={','.join(str(epoch) for epoch in epochs)}")
    if configured_epochs:
        parts.append(f"site config passed --epochs {','.join(str(epoch) for epoch in configured_epochs)}")
    if _reuses_patched_lightning_trainer_across_rounds(run):
        parts.append(
            "generated client creates and patches one `pl.Trainer` before `while flare.is_running()` "
            "and reuses it for repeated `trainer.fit(...)` calls"
        )
    return "; ".join(parts) + "."
