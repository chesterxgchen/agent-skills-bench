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

"""Generic agent-run analysis substrate (migration step 5-events-pre).

A neutral leaf: agent event/command parsing, dependency-install detection,
failure analysis, and small formatting utils shared by the generic report
engine and the SDK plugins. Stdlib + the other neutral leaves only (no
``benchmark_insights`` import), so SDK plugins can depend on it without a cycle.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ._runs import combined_text
from ._text import fmt_number, strip_ansi


def parse_event_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00").replace(",", "."))
    except ValueError:
        return None


def fmt_seconds(value: Any) -> str:
    number = as_number(value)
    if number is None:
        return "NA"
    if 0 < abs(number) < 1:
        text = f"{number:.3f}".rstrip("0").rstrip(".")
        return text if text not in {"0", "-0"} else ("0.001" if number > 0 else "-0.001")
    return str(round(number))


def fmt_seconds_with_unit(value: Any) -> str:
    formatted = fmt_seconds(value)
    return formatted if formatted == "NA" else f"{formatted}s"


def as_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def truncate(value: Any, limit: int = 180) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def inline_code_text(value: Any, limit: int = 180) -> str:
    raw = str(value or "").strip()
    text = re.sub(
        r"<<\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1\n.*?\n\2(?=\s|$)",
        lambda match: f"<<{match.group(1)}{match.group(2)}{match.group(1)} ... {match.group(2)}",
        raw,
        flags=re.DOTALL,
    )
    if text == raw:
        match = re.search(r"<<\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1", text)
        if match:
            text = f"{text[: match.end()]} ... {match.group(2)}"
    text = re.sub(r"\s+", " ", text).replace("`", "'")
    return truncate(text, limit)


def exit_code(run: dict[str, Any]) -> int | None:
    summary = run.get("run") if isinstance(run.get("run"), dict) else {}
    container_exit = run.get("container_exit") if isinstance(run.get("container_exit"), dict) else {}
    for value in (
        summary.get("final_container_exit_code"),
        summary.get("report_inclusive_exit_code"),
        summary.get("agent_exit_code"),
        container_exit.get("exit_code"),
    ):
        if isinstance(value, bool) or value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def unsupported_model_message(text: str) -> str:
    match = re.search(r"The '[^']+' model is not supported[^.\n]*(?:\.[^\n]*)?", text)
    return match.group(0).strip() if match else ""


def message_content_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    message = payload.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [item for item in content if isinstance(item, dict)]


def tool_result_output(payload: dict[str, Any], item: dict[str, Any]) -> str:
    parts = []
    result = payload.get("tool_use_result")
    if isinstance(result, dict):
        for key in ("stdout", "stderr"):
            value = result.get(key)
            text = str(value or "")
            if text and text not in parts:
                parts.append(text)
    elif result:
        text = str(result)
        if text not in parts:
            parts.append(text)
    for key in ("content", "text"):
        value = item.get(key)
        text = str(value or "")
        if text and text not in parts:
            parts.append(text)
    return strip_ansi("\n".join(parts))


def tool_result_exit(payload: dict[str, Any], item: dict[str, Any], output: str) -> tuple[int | None, str]:
    result = payload.get("tool_use_result")
    is_error = bool(item.get("is_error"))
    interrupted = False
    if isinstance(result, dict):
        is_error = is_error or bool(result.get("is_error"))
        interrupted = bool(result.get("interrupted"))
    exit_match = re.search(r"\bExit code\s+([0-9]+)\b", output, flags=re.IGNORECASE)
    exit_code = int(exit_match.group(1)) if exit_match else None
    if interrupted and exit_code is None:
        exit_code = 124
    if is_error and exit_code is None:
        exit_code = 1
    if exit_code is None and not is_error and not interrupted:
        exit_code = 0
    status = "failed" if (exit_code not in (None, 0) or is_error or interrupted) else "completed"
    return exit_code, status


def _event_payloads(text: str) -> list[dict[str, Any]]:
    payloads = []
    for line in str(text or "").splitlines():
        try:
            payload = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _background_task_status_by_tool_id(payloads: list[dict[str, Any]]) -> dict[str, str]:
    task_by_tool_id: dict[str, str] = {}
    status_by_task_id: dict[str, str] = {}
    for payload in payloads:
        event_type = str(payload.get("event_type") or payload.get("type") or "")
        if event_type == "system.task_started":
            task_id = str(payload.get("task_id") or "")
            tool_id = str(payload.get("tool_use_id") or "")
            if task_id and tool_id:
                task_by_tool_id[tool_id] = task_id
        elif event_type in {"system.task_updated", "system.task_notification"}:
            task_id = str(payload.get("task_id") or "")
            patch = payload.get("patch") if isinstance(payload.get("patch"), dict) else {}
            status = str(payload.get("status") or patch.get("status") or "").lower()
            if task_id and status in {"completed", "failed", "killed", "stopped"}:
                status_by_task_id[task_id] = status

        result = payload.get("tool_use_result")
        background_task_id = str(result.get("backgroundTaskId") or "") if isinstance(result, dict) else ""
        if not background_task_id:
            continue
        for content_item in message_content_items(payload):
            if content_item.get("type") == "tool_result":
                tool_id = str(content_item.get("tool_use_id") or "")
                if tool_id:
                    task_by_tool_id[tool_id] = background_task_id

    return {
        tool_id: status_by_task_id[task_id]
        for tool_id, task_id in task_by_tool_id.items()
        if status_by_task_id.get(task_id)
    }


def _tool_result_backgrounded(payload: dict[str, Any], output: str, background_status: str) -> bool:
    result = payload.get("tool_use_result")
    return bool(
        background_status
        or "Command running in background with ID:" in output
        or (isinstance(result, dict) and result.get("backgroundTaskId"))
    )


def _adjust_background_command_status(
    payload: dict[str, Any],
    output: str,
    exit_code: int | None,
    status: str,
    background_status: str,
) -> tuple[int | None, str, str]:
    if not _tool_result_backgrounded(payload, output, background_status):
        return exit_code, status, output
    if background_status == "completed":
        return 0, "completed", output
    if background_status in {"failed", "killed", "stopped"}:
        note = f"background task {background_status} before command completion"
        output = f"{note}\n{output}".strip()
        return 124, "failed", output
    return None, "running", output


def agent_command_events(run: dict[str, Any]) -> list[dict[str, Any]]:
    events = []
    pending_tool_commands: dict[str, dict[str, Any]] = {}
    payloads = _event_payloads(str(run.get("agent_events_text") or ""))
    background_status_by_tool_id = _background_task_status_by_tool_id(payloads)
    for payload in payloads:
        for content_item in message_content_items(payload):
            if content_item.get("type") == "tool_use" and content_item.get("name") == "Bash":
                tool_input = content_item.get("input") if isinstance(content_item.get("input"), dict) else {}
                command = str(tool_input.get("command") or "")
                tool_id = str(content_item.get("id") or "")
                if command and tool_id:
                    pending_tool_commands[tool_id] = {
                        "command": command,
                        "id": tool_id,
                        "index": len(events),
                    }
            elif content_item.get("type") == "tool_result":
                tool_id = str(content_item.get("tool_use_id") or "")
                pending = pending_tool_commands.pop(tool_id, None)
                if not pending:
                    continue
                output = tool_result_output(payload, content_item)
                exit_code, status = tool_result_exit(payload, content_item, output)
                exit_code, status, output = _adjust_background_command_status(
                    payload,
                    output,
                    exit_code,
                    status,
                    background_status_by_tool_id.get(tool_id, ""),
                )
                events.append(
                    {
                        "command": pending["command"],
                        "exit_code": exit_code,
                        "id": pending["id"],
                        "index": len(events),
                        "output": output,
                        "status": status,
                    }
                )
        item = payload.get("item")
        if not isinstance(item, dict) or item.get("type") != "command_execution":
            continue
        command = str(item.get("command") or "")
        if not command:
            continue
        events.append(
            {
                "command": command,
                "exit_code": item.get("exit_code"),
                "id": item.get("id"),
                "index": len(events),
                "output": strip_ansi(str(item.get("aggregated_output") or "")),
                "status": str(item.get("status") or ""),
            }
        )
    return events


def agent_command_spans(run: dict[str, Any]) -> list[dict[str, Any]]:
    spans = []
    pending: dict[str, dict[str, Any]] = {}
    pending_tool_commands: dict[str, dict[str, Any]] = {}
    payloads = _event_payloads(str(run.get("agent_events_text") or ""))
    background_status_by_tool_id = _background_task_status_by_tool_id(payloads)
    for payload in payloads:
        timestamp = parse_event_timestamp(payload.get("harness_timestamp") or payload.get("timestamp"))
        for content_item in message_content_items(payload):
            if content_item.get("type") == "tool_use" and content_item.get("name") == "Bash":
                tool_input = content_item.get("input") if isinstance(content_item.get("input"), dict) else {}
                command = str(tool_input.get("command") or "")
                tool_id = str(content_item.get("id") or "")
                if command and tool_id:
                    pending_tool_commands[tool_id] = {
                        "command": command,
                        "description": str(tool_input.get("description") or ""),
                        "id": tool_id,
                        "start": timestamp,
                    }
            elif content_item.get("type") == "tool_result":
                tool_id = str(content_item.get("tool_use_id") or "")
                pending_tool = pending_tool_commands.pop(tool_id, None)
                if not pending_tool:
                    continue
                output = tool_result_output(payload, content_item)
                exit_code, status = tool_result_exit(payload, content_item, output)
                exit_code, status, output = _adjust_background_command_status(
                    payload,
                    output,
                    exit_code,
                    status,
                    background_status_by_tool_id.get(tool_id, ""),
                )
                start = pending_tool.get("start")
                duration = (timestamp - start).total_seconds() if timestamp and start else None
                spans.append(
                    {
                        "command": pending_tool["command"],
                        "description": pending_tool.get("description") or "",
                        "duration_seconds": duration,
                        "exit_code": exit_code,
                        "id": pending_tool["id"],
                        "index": len(spans),
                        "output": output,
                        "status": status,
                    }
                )
        item = payload.get("item")
        if not isinstance(item, dict) or item.get("type") != "command_execution":
            continue
        command = str(item.get("command") or "")
        item_id = str(item.get("id") or "")
        if not command or not item_id:
            continue
        event_type = str(payload.get("type") or "")
        if event_type == "item.started":
            pending[item_id] = {"command": command, "start": timestamp}
            continue
        if event_type != "item.completed":
            continue
        start = pending.pop(item_id, {}).get("start")
        duration = (timestamp - start).total_seconds() if timestamp and start else None
        spans.append(
            {
                "command": command,
                "duration_seconds": duration,
                "exit_code": item.get("exit_code"),
                "id": item_id,
                "index": len(spans),
                "output": strip_ansi(str(item.get("aggregated_output") or "")),
                "status": str(item.get("status") or ""),
            }
        )
    return spans


def agent_message_texts(run: dict[str, Any]) -> list[str]:
    messages = []
    for line in str(run.get("agent_events_text") or "").splitlines():
        try:
            payload = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        item = payload.get("item")
        if isinstance(item, dict) and item.get("type") == "agent_message":
            text = strip_ansi(str(item.get("text") or "")).strip()
            if text:
                messages.append(text)
        for content_item in message_content_items(payload):
            if content_item.get("type") not in {"text", "agent_message"}:
                continue
            text = strip_ansi(str(content_item.get("text") or content_item.get("content") or "")).strip()
            if text:
                messages.append(text)
    return messages


def command_failed(event: dict[str, Any]) -> bool:
    exit_value = event.get("exit_code")
    if isinstance(exit_value, bool):
        return False
    if exit_value not in (None, 0):
        return True
    return str(event.get("status") or "") == "failed"


def command_succeeded(event: dict[str, Any]) -> bool:
    return (event.get("exit_code") == 0 and str(event.get("status") or "") == "completed") or job_output_succeeded(
        str(event.get("output") or "")
    )


def command_recovery_key(command: str) -> str:
    command = str(command)
    if re.search(r"\bpip\s+install\b", command):
        requirements = re.search(r"-r\s+([A-Za-z0-9_./-]*requirements[A-Za-z0-9_.-]*\.txt)", command)
        return f"pip install {Path(requirements.group(1)).name}" if requirements else "pip install"
    script = re.search(r"\bpython(?:3)?\s+([A-Za-z0-9_./-]+\.py)\b", command)
    if script:
        role = "export" if "--export" in command else "run"
        return f"python {Path(script.group(1)).name} {role}"
    first_word = re.search(r"(?:^|['\"])([A-Za-z0-9_./-]+)", command)
    return first_word.group(1) if first_word else command[:80]


def job_output_has_failure_status(output: str) -> bool:
    """Return True when an explicit job status line reports a terminal failure state.

    NVFLARE result-location lines are printed for any terminal status, including failures
    (e.g. ``FINISHED:EXECUTION_EXCEPTION``), so a failed status must veto result-path evidence.
    Covers both the ``FINISHED:<state>`` enum forms (job_def.RunStatus) and the legacy bare
    terminal statuses the CLI/flare_api still emit (``FINISHED_EXCEPTION``, ``FAILED``,
    ``ABORTED``, ``ABANDONED``). Success statuses (``FINISHED:COMPLETED``, ``FINISHED_OK``)
    are deliberately excluded.
    """
    return bool(
        re.search(
            r"\b(?:Job\s+)?Status(?:\s+is)?\s*:\s*"
            r"(?:FINISHED:(?!COMPLETED\b)[A-Z_]+|FINISHED_EXCEPTION|FAILED(?:_TO_RUN)?|ABORTED|ABANDONED)\b",
            strip_ansi(output),
            flags=re.IGNORECASE,
        )
    )


def job_output_succeeded(output: str) -> bool:
    text = strip_ansi(output)
    if job_output_has_failure_status(text) or job_output_has_failure_marker(text):
        return False
    return bool(
        re.search(
            r"\bFinished\s+FedAvg\b|"
            r"\bSimulation workspace\s*:\s*|"
            r"\bResult workspace\s*:\s*|"
            r"\bResult can be found in\s*:?\s+\S+|"
            r"\bResult location\s*:\s*\S+|"
            r"\b(?:Job\s+)?Status(?:\s+is)?\s*:\s*(?:FINISHED:COMPLETED|FINISHED_OK|COMPLETED)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def job_output_has_failure_marker(output: str) -> bool:
    return bool(re.search(r"\bConfigError\s*:|\bAbort signal triggered\b", strip_ansi(output), flags=re.IGNORECASE))


def missing_python_module_name(output: str) -> str:
    text = strip_ansi(output)
    match = re.search(r"ModuleNotFoundError:\s+No module named ['\"]([^'\"]+)['\"]", text)
    if match:
        return match.group(1)
    match = re.search(r"No module named ['\"]([^'\"]+)['\"]", text)
    return match.group(1) if match else ""


def command_error_summary(output: str) -> str:
    text = strip_ansi(output)
    patterns = (
        r"TypeError: [^\n]+",
        r"ConfigError: [^\n]+",
        r"RuntimeError: [^\n]+",
        r"ModuleNotFoundError: [^\n]+",
        r"ProtocolError: [^\n]+",
        r"IncompleteRead\([^\n]+",
        r"Connection broken: [^\n]+",
        r"No module named [^\n]+",
        r"sed: can't read [^\n]+",
        r"ERROR - [^\n]+",
        r"Error processing [^\n]+",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return truncate(match.group(0), 320)
    for line in text.splitlines():
        lowered = line.lower()
        if any(token in lowered for token in ("error", "failed", "traceback", "missing", "not found")):
            return truncate(line, 320)
    return truncate(text, 320) if text.strip() else "no command output captured"


def result_permission_denial_count(run: dict[str, Any]) -> int:
    count = 0
    for line in str(run.get("agent_events_text") or "").splitlines():
        try:
            payload = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        denials = payload.get("permission_denials")
        if isinstance(denials, list):
            count = max(count, len(denials))
    return count


def bash_permission_denial_count(run: dict[str, Any]) -> int:
    events_text = str(run.get("agent_events_text") or "")
    needle = "requested permissions to use bash"
    raw_count = events_text.lower().count(needle)
    tool_result_count = 0
    for line in events_text.splitlines():
        try:
            payload = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        text_parts = [str(payload.get("tool_use_result") or "")]
        message = payload.get("message")
        if isinstance(message, dict):
            for item in message.get("content") or []:
                if isinstance(item, dict):
                    text_parts.append(str(item.get("content") or item.get("text") or ""))
        if any(needle in text.lower() for text in text_parts):
            tool_result_count += 1
    if tool_result_count:
        return max(result_permission_denial_count(run), tool_result_count)
    return max(result_permission_denial_count(run), raw_count)


def artifact_validation_metric_evidence(run: dict[str, Any]) -> str:
    metric = run.get("validation_metric") if isinstance(run.get("validation_metric"), dict) else {}
    if metric.get("source") != "metrics_artifact" or not metric.get("reported_values"):
        return ""
    source_path = str(metric.get("source_path") or "")
    if source_path:
        return f"captured validation metric artifact `{truncate(source_path, 180)}`"
    return "captured validation metric artifact"


def artifact_validation_metric_is_runtime_evidence(run: dict[str, Any]) -> bool:
    metric = run.get("validation_metric") if isinstance(run.get("validation_metric"), dict) else {}
    if metric.get("source") != "metrics_artifact" or not metric.get("reported_values"):
        return False
    source_path = str(metric.get("source_path") or "").replace("\\", "/")
    if source_path.endswith("/round_metrics.jsonl") or source_path == "round_metrics.jsonl":
        return False
    source_path_with_root = "/" + source_path.lstrip("/")
    copied_workspace_artifact_keys = ("changed_files", "workspace_added_files", "workspace_modified_files")
    if any(
        f"/workspace_delta/{key}/" in source_path_with_root or source_path_with_root.startswith(f"/{key}/")
        for key in copied_workspace_artifact_keys
    ):
        return False
    return bool(
        "workspace_delta/runtime_artifacts/" in source_path
        or "/runtime_artifacts/" in source_path
        or re.search(r"(^|/)server/simulate_job/metrics/[^/]+$", source_path)
        or re.search(r"(^|/)simulate_job/metrics/(?:metrics_summary\.json|round_metrics\.jsonl)$", source_path)
    )


def failure_evidence(run: dict[str, Any]) -> str:
    text = combined_text(run)
    model_error = unsupported_model_message(text)
    if model_error:
        return model_error
    for source_name in ("agent_last_message", "agent_stderr", "console_text", "agent_events_text"):
        for line in str(run.get(source_name) or "").splitlines():
            lowered = line.lower()
            if any(
                token in lowered
                for token in (
                    "error",
                    "failed",
                    "pull access denied",
                    "not supported",
                    "authentication_failed",
                    "not logged in",
                    "please run /login",
                    "api key",
                )
            ):
                return line.strip()[:500]
    return ""


def agent_failure_category(run: dict[str, Any]) -> str:
    record = run.get("record") if isinstance(run.get("record"), dict) else {}
    exit_summary = record.get("agent_exit_summary") if isinstance(record.get("agent_exit_summary"), dict) else {}
    failure_category = record.get("failure_category") or exit_summary.get("failure_category")
    if failure_category and failure_category != "agent_unknown_failure":
        return str(failure_category)
    text = combined_text(run).lower()
    if any(token in text for token in ("authentication_failed", "not logged in", "please run /login", "api key")):
        return "agent_auth_failure"
    if failure_category:
        return str(failure_category)
    return ""


def run_activity(run: dict[str, Any]) -> dict[str, Any]:
    activity = run.get("activity")
    return activity if isinstance(activity, dict) else {}


def commands_for_run(run: dict[str, Any]) -> list[str]:
    commands = run_activity(run).get("commands")
    return [str(command) for command in commands] if isinstance(commands, list) else []


def is_dependency_install_command(command: str) -> bool:
    lowered = str(command).lower()
    install_pattern = r"\b(?:uv\s+)?pip\s+install\b|\bpython3?\s+-m\s+pip\s+install\b"
    if re.search(r"\b(?:grep|rg|sed|awk)\b", lowered) and re.search(install_pattern, lowered):
        return False
    return bool(re.search(install_pattern, lowered))


def dependency_install_events(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event for event in agent_command_events(run) if is_dependency_install_command(str(event.get("command") or ""))
    ]


def dependency_install_evidence_brief(run: dict[str, Any]) -> str:
    events = dependency_install_events(run)
    if events:
        if any(command_failed(event) for event in events):
            return "dependency install was attempted and failed"
        if any(command_succeeded(event) for event in events):
            return "a dependency install command later succeeded"
        return "dependency install command was captured without success/failure status"
    if any(is_dependency_install_command(command) for command in commands_for_run(run)):
        return "dependency install command was listed but no command result was captured"
    return "no dependency install command was captured"


def dependency_install_evidence(run: dict[str, Any]) -> str:
    events = dependency_install_events(run)
    if events:
        failed = [event for event in events if command_failed(event)]
        if failed:
            event = failed[-1]
            return (
                f"dependency install attempted and failed (`{inline_code_text(str(event.get('command') or ''), 100)}` "
                f"exit {event.get('exit_code')}: {truncate(command_error_summary(str(event.get('output') or '')), 160)})"
            )
        succeeded = [event for event in events if command_succeeded(event)]
        if succeeded:
            event = succeeded[-1]
            return f"dependency install command succeeded (`{inline_code_text(str(event.get('command') or ''), 100)}`)"
        event = events[-1]
        return (
            "dependency install command captured without success/failure status "
            f"(`{inline_code_text(str(event.get('command') or ''), 100)}`)"
        )
    commands = [command for command in commands_for_run(run) if is_dependency_install_command(command)]
    if commands:
        return (
            "dependency install command listed in activity but no command result was captured "
            f"(`{inline_code_text(commands[-1], 100)}`)"
        )
    return "no dependency install command was captured before the failed job run"


def _span_total_seconds(spans: list[dict[str, Any]]) -> float | None:
    durations = [as_number(span.get("duration_seconds")) for span in spans]
    captured = [duration for duration in durations if duration is not None]
    return sum(captured) if captured else None


def _format_command_span(span: dict[str, Any]) -> str:
    seconds = as_number(span.get("duration_seconds")) or 0
    command = truncate(re.sub(r"\s+", " ", str(span.get("command") or "")).strip(), 120)
    exit_code = span.get("exit_code")
    exit_note = f", exit {exit_code}" if exit_code not in (None, "") else ""
    return f"`{command}` ({fmt_number(round(seconds))}s{exit_note})"


def _longest_span(spans: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not spans:
        return None
    return max(spans, key=lambda span: as_number(span.get("duration_seconds")) or 0)


def _rerun_reason_from_agent_messages(run: dict[str, Any]) -> list[str]:
    reasons = []
    trigger = re.compile(
        r"\b(?:re-?run|re-?running|run(?:ning)?\s+(?:the\s+)?(?:simulation|job)\s+again|"
        r"re-?export(?:ing)?|final\s+(?:verification|validation)\s+pass)\b",
        flags=re.IGNORECASE,
    )
    reason_context = re.compile(
        r"\b(?:after|because|before|so|patch|fix|change|configuration|metric|aligned|current\s+source|"
        r"final\s+artifacts|validation|verification|robustness|match)\b",
        flags=re.IGNORECASE,
    )
    progress_only = re.compile(
        r"\b(?:healthy|completed\s+successfully|finished\s+successfully|is\s+running|is\s+underway|in\s+progress)\b",
        flags=re.IGNORECASE,
    )
    for text in agent_message_texts(run):
        if not trigger.search(text):
            continue
        sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
        for sentence in sentences:
            if trigger.search(sentence) and reason_context.search(sentence) and not progress_only.search(sentence):
                reasons.append(sentence)
                break
    return reasons


def _job_rerun_reason(spans: list[dict[str, Any]], run: dict[str, Any]) -> str:
    reasons = []
    for span in spans[1:]:
        description = str(span.get("description") or "").strip()
        if description:
            reasons.append(description)
        command = str(span.get("command") or "")
        if re.search(r"\brm\s+-rf\b", command):
            reasons.append("runtime workspace was cleared before rerun")
    reasons.extend(_rerun_reason_from_agent_messages(run))
    unique_reasons = []
    for reason in reasons:
        if reason and reason not in unique_reasons:
            unique_reasons.append(reason)
    if unique_reasons:
        return "; ".join(unique_reasons[:3])
    return "not captured; inspect commands around the repeated run"
