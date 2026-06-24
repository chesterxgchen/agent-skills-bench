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

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ...common import load_json
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
    strip_ansi,
)

# Product-specific structure contract: the core converted source files an
# NVFLARE job is expected to produce.
REQUIRED_STRUCTURE_FILES = ("client.py", "model.py", "job.py")
OPTIONAL_STRUCTURE_FILES = ("prepare_data.py", "download_data.py")


def current_workspace_structure_file_matches(run: dict[str, Any], filename: str) -> list[str]:
    paths = unique_paths(manifest_paths(run, "final_structure_files"))
    return [path for path in paths if Path(path).name == filename and len(Path(path).parts) == 1]


def structure_score(run: dict[str, Any]) -> float | None:
    if not run.get("available"):
        return None
    present = sum(1 for filename in REQUIRED_STRUCTURE_FILES if current_workspace_structure_file_matches(run, filename))
    return present / len(REQUIRED_STRUCTURE_FILES)


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
            for arg in tokens[index + 1 :]:
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
        if missing_python_module_name(output):
            dependency_evidence = dependency_install_evidence(run)
        diagnostics.append(
            {
                "command": inline_code_text(command, 180),
                "exit": str(event.get("exit_code")),
                "recovery": recovery,
                "root_cause": command_error_summary(output),
                "dependency": dependency_evidence,
            }
        )
    return diagnostics


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
        for item in values:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or item.get("artifact_path") or "")
            if Path(path).name != "config_fed_server.json":
                continue
            items.append((key, item))

    def priority(entry: tuple[str, dict[str, Any]]) -> tuple[int, int, str]:
        key, item = entry
        path = str(item.get("path") or item.get("artifact_path") or "")
        key_priority = 0 if key == "runtime_artifacts" else 1
        server_priority = 0 if re.search(r"(^|/)(server|app_server)(/|$)", path) else 1
        return key_priority, server_priority, path

    return sorted(items, key=priority)


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
    text = combined_text(run)
    for name in ("SCAFFOLD", "FedAvg", "FedOpt", "FedProx", "Cyclic", "FedEval"):
        if re.search(rf"\b{re.escape(name)}\b", text, flags=re.IGNORECASE):
            recipe = _recipe_evidence(run)
            evidence = "agent final message or command text"
            if recipe:
                evidence += f"; recipe {recipe}"
            return {"algorithm": name, "evidence": evidence, "num_rounds": None, "recipe": recipe}
    return {"algorithm": "not captured", "evidence": "no server workflow config or algorithm mention captured"}


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


def job_run_status(run: dict[str, Any]) -> str:
    """Return one of 'completed', 'started_failed', 'not_started', or 'unknown'."""
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
    if last_successful_job_event(run):
        return "completed"
    if artifact_validation_metric_is_runtime_evidence(run):
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

    if status == "started_failed":
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
                        f"simulation command ran but missing Python dependency `{missing_module}` — "
                        f"{dependency_install_evidence(run)}"
                    )
                summary = command_error_summary(output)
                return f"simulation command ran but exited with error — {truncate(summary, 200)}"
        for event in reversed(events):
            if is_simulation_or_job_command(str(event.get("command") or "")):
                output = str(event.get("output") or "")
                missing_module = missing_python_module_name(output)
                if missing_module:
                    return (
                        f"simulation command ran but missing Python dependency `{missing_module}` — "
                        f"{dependency_install_evidence(run)}"
                    )
                summary = command_error_summary(output)
                return f"simulation command ran but success was not confirmed — {truncate(summary, 200)}"
        return "simulation command ran but no command output was captured"

    if status == "completed":
        event = last_successful_job_event(run)
        output = str(event.get("output") or "") if event else ""
        recovered_issue = completed_job_recovered_issue_summary(run)
        repeated_runs = repeated_job_run_summary(run)
        artifact_evidence = artifact_validation_metric_evidence(run)
        if artifact_evidence and not event:
            return (
                "job execution inferred from captured runtime metric artifact — "
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
            parts.append(
                f"earlier missing Python dependency `{missing_module}` was recovered "
                f"({dependency_install_evidence_brief(run)})"
            )
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
    ("Runtime/output locality", _runtime_output_locality_signal, _assessment_from_locality),
    ("Dependency install strategy", _dependency_strategy_signal, _assessment_from_dependency),
)

CODE_QUALITY_CONTEXT_ROWS = (("API pattern", _api_pattern_signal),)

CODE_QUALITY_POINTS = {"good": 1.0, "caution": 0.5, "poor": 0.0}


def generated_code_quality_assessments(run: dict[str, Any]) -> list[tuple[str, str, str]]:
    rows = []
    for label, evidence_getter, assessment_getter in CODE_QUALITY_ROWS:
        evidence = evidence_getter(run)
        rows.append((label, assessment_getter(evidence), evidence))
    return rows


def generated_code_quality_overall(run: dict[str, Any]) -> str:
    assessments = generated_code_quality_assessments(run)
    known = [(status, evidence) for _, status, evidence in assessments if status in CODE_QUALITY_POINTS]
    total = len(assessments)
    if not known:
        return "unknown: no generated-code evidence captured"
    points = sum(CODE_QUALITY_POINTS[status] for status, _ in known)
    score_ratio = points / total
    if score_ratio >= 0.8:
        label = "good"
    elif score_ratio >= 0.5:
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
    known = [status for _, status, _ in assessments if status in CODE_QUALITY_POINTS]
    if not known:
        return None
    return sum(CODE_QUALITY_POINTS[status] for status in known) / len(assessments)


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
