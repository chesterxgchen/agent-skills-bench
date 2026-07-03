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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..metric_artifacts import metric_is_runtime_result_artifact
from ._runs import combined_text
from ._text import (
    FILE_INSPECTION_COMMANDS,
    _command_tokens,
    _shell_command_parts,
    fmt_number,
    strip_ansi,
)


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
    # The heredoc regex (DOTALL + backreference) backtracks over the whole
    # input; the output is truncated to ~limit chars anyway, so the input is
    # capped first — an adversarial multi-MB command must not go quadratic.
    raw = str(value or "").strip()[:4096]
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
    # No exit code means the command's outcome was never captured (stream cut
    # off, tool_result missing): an unresolved command is not a failure, even
    # when a coarse status field says "failed".
    if exit_value is None:
        return False
    return str(event.get("status") or "") == "failed"


def command_succeeded(event: dict[str, Any]) -> bool:
    return (event.get("exit_code") == 0 and str(event.get("status") or "") == "completed") or job_output_succeeded(
        str(event.get("output") or "")
    )


#: NVFLARE long options that never take a separate value (zero-argument
#: argparse actions in the nvflare CLIs, plus common built-ins). For nvflare
#: modules every other bare ``--flag`` takes a separate value, so it is
#: assumed to consume the next token — otherwise an option value could occupy
#: the subcommand/job-target slots of a module recovery key. That assumption
#: is nvflare-specific: it must not extend to other ``python -m`` CLIs, whose
#: boolean long flags (``--rebuild``, ``--no-cache``, ...) cannot be
#: enumerated here and would otherwise swallow the following positional.
_NVFLARE_BOOLEAN_LONG_OPTIONS = frozenset(
    "--clean --debug --force --help --no-color --prepare --quiet --start --stop "
    "--ui_tool --verbose --version --with_debug --yes".split()
)


#: Value-taking long options of the repo's own ``python -m`` CLIs, keyed by
#: module. Unlike nvflare, these CLIs mix boolean flags with value-taking
#: options, so the value-taking set is enumerated explicitly; a bare long
#: option outside the set is boolean. Without this, option values such as
#: ``--prompt p.txt`` would occupy the recovery key's positional slots.
_MODULE_VALUE_LONG_OPTIONS: dict[str, frozenset[str]] = {
    "benchmark.harness.host.runner": frozenset(
        "--agent --agent-home --job-scale --model --output-dir --prompt --result-root "
        "--results-root --training-code --workflow".split()
    ),
    "benchmark.harness.host.build": frozenset("--agent --agent-profile --node-image --sdk-profile --uv-image".split()),
}


#: Options whose value *is* the job target (the host runner's usage declares
#: the positional PATH "equivalent to --training-code"), so the value fills a
#: positional slot of the recovery key instead of being dropped — otherwise
#: ``pair --training-code jobs/a`` and ``pair --training-code jobs/b`` would
#: collapse onto the same key.
_MODULE_TARGET_LONG_OPTIONS: dict[str, frozenset[str]] = {
    "benchmark.harness.host.runner": frozenset({"--training-code"}),
}


#: Short options known to never take a value (`-h` help, `-q` quiet). An
#: unknown short option keeps the conservative default of consuming the next
#: token: a wrongly-consumed positional only broadens the key — and a broad
#: key is refused as rerun evidence — whereas an unconsumed option value would
#: fill a positional slot and stop a genuine rerun's key from matching.
_BOOLEAN_SHORT_OPTIONS = frozenset({"-h", "-q"})


_SHELL_WRAPPER_RE = re.compile(r"^\s*(?:/bin/)?(?:ba|z)?sh\s+-l?c\s+", re.IGNORECASE)


def _unwrap_shell_command(command: str) -> str:
    """Strip a `sh -c '...'` wrapper (and its quotes) to expose the real command."""

    text = _SHELL_WRAPPER_RE.sub("", str(command or "")).strip()
    if text[:1] in {"'", '"'} and text[-1:] == text[:1]:
        text = text[1:-1]
    return text


@dataclass(frozen=True)
class Invocation:
    """One pipeline stage of one shell segment, parsed once.

    Every classifier reads this model instead of re-tokenizing the command
    string, so quoted text cannot pass for shell syntax and env/wrapper
    prefixes and redirections are handled in exactly one place.
    """

    kind: str  # "pip_install" | "module" | "script" | "inspection" | "guard" | "other"
    executable: str = ""  # basename, lowercased; "" when the stage cannot be tokenized
    module: str = ""  # `python -m` target
    positionals: tuple[str, ...] = ()  # first two operands (basenames); the module key's suffix slots
    args: tuple[str, ...] = ()  # raw tokens after the executable (after the module for module runs)
    operator_after: str = ""  # "&&" | "||" | ";" | "|" | ""


_ENV_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*", re.DOTALL)

#: Wrapper executables whose operands are the real command, with their
#: value-taking options (`env -u NAME`, `timeout -k 5s`, `sudo -u user`);
#: `timeout` additionally takes a DURATION operand before the command, and
#: `env`/`sudo` accept further VAR=val assignments among their arguments.
_WRAPPER_VALUE_OPTIONS = {
    "env": frozenset({"-u", "--unset"}),
    "sudo": frozenset({"-u", "--user", "-g", "--group"}),
    "timeout": frozenset({"-k", "-s", "--kill-after", "--signal"}),
}


def _strip_execution_prefix(tokens: list[str]) -> list[str]:
    """Drop leading VAR=val assignments and env/nohup/time/timeout wrappers:
    `CUDA_VISIBLE_DEVICES=0 python train.py` and `timeout 600 python3 -m ...`
    execute the same job as their unprefixed spellings.
    """

    index = 0
    while index < len(tokens):
        token = tokens[index]
        if _ENV_ASSIGNMENT_RE.fullmatch(token):
            index += 1
            continue
        name = Path(token).name.lower()
        if name in ("nohup", "time"):
            index += 1
            continue
        if name in _WRAPPER_VALUE_OPTIONS:
            value_options = _WRAPPER_VALUE_OPTIONS[name]
            index += 1
            while index < len(tokens):
                if name in ("env", "sudo") and _ENV_ASSIGNMENT_RE.fullmatch(tokens[index]):
                    index += 1
                elif tokens[index] in value_options and index + 1 < len(tokens):
                    index += 2
                elif tokens[index].startswith("-"):
                    index += 1
                else:
                    break
            if name == "timeout":
                index += 1
            continue
        break
    return tokens[index:]


# `2>&1`, `>out.log`, `>` + target, `&`: neither a redirection operator nor
# its target is an operand of the command, so both leave positional
# consideration.
_REDIRECT_TOKEN_RE = re.compile(r"(?:\d+|&)?(?:>>?|<+)(\S*)")


def _drop_redirections(tokens: list[str]) -> list[str]:
    cleaned: list[str] = []
    skip_target = False
    for token in tokens:
        match = _REDIRECT_TOKEN_RE.fullmatch(token)
        if skip_target:
            skip_target = False
        elif match:
            skip_target = not match.group(1)  # a bare operator (`>`; not `>out.log`) consumes its target
        elif token != "&":
            cleaned.append(token)
    return cleaned


# Shell status commands (`grep ... || true`) that neither execute work nor
# produce output of their own.
_HARMLESS_STATUS_COMMANDS = frozenset({"true", "false", ":", "exit"})

# ``exit``'s status operand is inert only when it's a literal code (``1``,
# ``143``) or a plain status expansion: ``$?``, ``$status``, ``${status}``.
# Anything else — ``$(cmd)``, backticks, arithmetic, glob metacharacters —
# can execute or hide extra work and must not count as a pure status guard.
_SAFE_EXIT_STATUS_OPERAND_RE = re.compile(r"\d+|\$\?|\$[A-Za-z_][A-Za-z0-9_]*|\$\{[A-Za-z_][A-Za-z0-9_]*\}")


def _module_positionals(module: str, args: tuple[str, ...]) -> tuple[str, ...]:
    """Positional tokens (subcommand and job target) that narrow a module key.

    Option values are skipped — except job-target options like the host
    runner's ``--training-code``, whose value fills a positional slot — and
    paths reduce to their basename so relative/absolute rerun paths match.
    Long-option semantics are module-aware: nvflare's bare long options all
    take a separate value except a fixed boolean set (otherwise `--workspace
    /tmp/ws` would let `ws` steal the job-target slot), the repo's own host
    CLIs enumerate their value-taking options, and any other module keeps the
    conservative reading that a bare long option is a boolean flag — assuming
    a value would let `--rebuild target` drop the real positional target.
    """

    module_is_nvflare = module.split(".", 1)[0] == "nvflare"
    value_options = _MODULE_VALUE_LONG_OPTIONS.get(module, frozenset())
    target_options = _MODULE_TARGET_LONG_OPTIONS.get(module, frozenset())
    positionals: list[str] = []
    skip_value = False
    value_is_target = False
    for token in args:
        if len(positionals) == 2:
            break
        if skip_value:
            skip_value = False
            if value_is_target:
                value_is_target = False
                positionals.append(Path(token).name)
            continue
        if token.startswith("--"):
            name, inline_sep, inline_value = token.partition("=")
            if inline_sep:
                # A job-target option's inline `--flag=value` value still
                # fills a positional slot.
                if name in target_options and inline_value:
                    positionals.append(Path(inline_value).name)
                continue
            skip_value = (token not in _NVFLARE_BOOLEAN_LONG_OPTIONS) if module_is_nvflare else (token in value_options)
            value_is_target = skip_value and token in target_options
            continue
        if token.startswith("-"):
            # Short alphabetic options (`-w ws`, `-gpu 0`) consume the next
            # token unless known boolean; attached (`-n2`) forms do not.
            skip_value = bool(re.fullmatch(r"-[A-Za-z]+", token)) and token not in _BOOLEAN_SHORT_OPTIONS
            value_is_target = False
            continue
        positionals.append(Path(token).name)
    return tuple(positionals)


def _generic_positionals(args: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    positionals: list[str] = []
    for token in args:
        # A quoted operand can carry arbitrary text; only its first word can
        # serve as a key slot.
        words = Path(token).name.split() if token and not token.startswith("-") else []
        if words:
            positionals.append(words[0])
            if len(positionals) == 2:
                break
    return tuple(positionals)


# Interpreter tooling run via ``python -m``: environment plumbing, not a job run.
_PYTHON_TOOLING_MODULES = frozenset({"pip", "venv", "ensurepip"})

_PYTHON_EXECUTABLE_RE = re.compile(r"python[\d.]*")

#: Python interpreter options that take a separate value (`-W ignore`,
#: `-X faulthandler`); without consuming it, the value would read as the
#: script positional.
_PYTHON_VALUE_OPTIONS = frozenset({"-W", "-X", "--check-hash-based-pycs"})


def _parse_python_stage(executable: str, tokens: list[str], operator_after: str) -> Invocation:
    args = tuple(tokens[1:])
    index = 1
    while index < len(tokens):
        token = tokens[index]
        # Python accepts attached option arguments (`-mnvflare.cli`, `-c'code'`),
        # so match the option prefix, not the whole token.
        if token.startswith("-m") and not token.startswith("--"):
            module = token[2:] or (tokens[index + 1] if index + 1 < len(tokens) else "")
            if not module:
                break
            module_args = tuple(tokens[index + (1 if token[2:] else 2) :])
            kind = "pip_install" if module in ("pip", "pip3") and module_args[:1] == ("install",) else "module"
            positionals = _module_positionals(module, module_args) if kind == "module" else ()
            return Invocation(
                kind, executable, module=module, positionals=positionals, args=module_args,
                operator_after=operator_after,
            )
        if token.startswith("-c") and not token.startswith("--"):
            # Inline `-c` snippets are not job runs and key on the interpreter.
            break
        if token in _PYTHON_VALUE_OPTIONS:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        kind = "script" if token.endswith(".py") else "other"
        return Invocation(
            kind, executable, positionals=_generic_positionals(tokens[index:]), args=args, operator_after=operator_after
        )
    return Invocation("other", executable, args=args, operator_after=operator_after)


def _parse_stage(tokens: list[str], operator_after: str) -> Invocation | None:
    head = Path(tokens[0]).name.lower()
    if head in _HARMLESS_STATUS_COMMANDS:
        # A pure status guard is the status command ALONE (plus an inert exit
        # operand); anything more — redirection, substitution, extra operands
        # — can hide real work.
        guard = len(tokens) == 1 or (
            head == "exit" and len(tokens) == 2 and _SAFE_EXIT_STATUS_OPERAND_RE.fullmatch(tokens[1]) is not None
        )
        return Invocation("guard" if guard else "other", head, args=tuple(tokens[1:]), operator_after=operator_after)
    tokens = _drop_redirections(_strip_execution_prefix(tokens))
    if not tokens:
        # A pure `VAR=val` assignment: nothing executes.
        return None
    executable = Path(tokens[0]).name.lower()
    if _PYTHON_EXECUTABLE_RE.fullmatch(executable):
        return _parse_python_stage(executable, tokens, operator_after)
    args = tuple(tokens[1:])
    positionals = _generic_positionals(args)
    if executable in ("pip", "pip3", "uv") and (
        positionals[:1] == ("install",) if executable != "uv" else positionals[:2] == ("pip", "install")
    ):
        kind = "pip_install"
    elif executable in FILE_INSPECTION_COMMANDS:
        kind = "inspection"
    else:
        kind = "other"
    return Invocation(kind, executable, positionals=positionals, args=args, operator_after=operator_after)


# Operator tokens that survive nested `sh -c` unwrapping inside a segment
# re-split the stage stream exactly like their unquoted counterparts.
_OPERATOR_TOKENS = {"&&", "||", ";", "|"}


def _split_unquoted_pipes(segment: str) -> list[str]:
    """Split a segment on unquoted single ``|`` pipeline operators.

    shlex yields ``|`` as its own token only when whitespace surrounds it, so
    the spaceless form (``cat cfg.json|python3 run.py``) must be split at the
    character level, where quoting is still visible. A quoted or
    backslash-escaped ``|`` is operand text, ``>|`` is the noclobber redirect,
    and ``||`` (only reachable here out of a nested unwrap) is a segment
    join, not a pipe — none of them split.
    """

    pieces: list[str] = []
    buf: list[str] = []
    quote = ""
    index = 0
    length = len(segment)
    while index < length:
        char = segment[index]
        if quote:
            buf.append(char)
            if char == quote:
                quote = ""
            index += 1
            continue
        if char in ("'", '"'):
            quote = char
            buf.append(char)
            index += 1
            continue
        if char == "|" and not (buf and buf[-1] in ("\\", ">", "|")):
            if index + 1 < length and segment[index + 1] == "|":
                buf.append("||")
                index += 2
                continue
            piece = "".join(buf).strip()
            if piece:
                pieces.append(piece)
            buf = []
            index += 1
            continue
        buf.append(char)
        index += 1
    piece = "".join(buf).strip()
    if piece:
        pieces.append(piece)
    return pieces


def parse_shell_command(command: str) -> list[Invocation]:
    """Parse a shell command into per-stage Invocations: unwrap a `sh -c '...'`
    wrapper once, split `&&`/`||`/`;` segments and `|` pipeline stages
    (quote-aware, with or without surrounding spaces), strip env/wrapper
    prefixes and redirections per stage, and classify each stage. A segment
    whose quoting cannot be tokenized contributes an empty-executable marker,
    distinguishing "could not parse" from "parsed as harmless".
    """

    invocations: list[Invocation] = []
    for segment, operator in _shell_command_parts(_unwrap_shell_command(command)):
        pieces = _split_unquoted_pipes(segment) or [segment]
        piece_operators = ["|"] * (len(pieces) - 1) + [operator]
        for piece, piece_operator in zip(pieces, piece_operators):
            tokens = _command_tokens(piece)
            if not tokens:
                invocations.append(Invocation("other", operator_after=piece_operator))
                continue
            stages: list[list[str]] = [[]]
            stage_operators: list[str] = []
            for token in tokens:
                if token in _OPERATOR_TOKENS:
                    stage_operators.append(token)
                    stages.append([])
                else:
                    stages[-1].append(token)
            stage_operators.append(piece_operator)
            for stage, stage_operator in zip(stages, stage_operators):
                invocation = _parse_stage(stage, stage_operator) if stage else None
                if invocation is not None:
                    invocations.append(invocation)
    return invocations


def _command_executes_python(command: str) -> bool:
    """True when some segment's executable (wrappers peeled) is a python interpreter."""

    return any(
        _PYTHON_EXECUTABLE_RE.fullmatch(_segment_executable_name(segment) or "-")
        for segment in _shell_command_segments(command)
        if segment.strip()
    )


def command_recovery_key(command: str) -> str:
    """Key under which a later successful command can count as a rerun.

    Installer runs share the `pip install [requirements file]` key; `python
    <script>.py` runs key on the script; `python -m <module>` runs key on the
    module plus its positional subcommand/job target — not the bare
    interpreter or bare module, otherwise any later successful `python3 ...`
    (e.g. an import probe) or same-module non-rerun (`--help`) would look like
    a rerun of the failed module job. Everything else keys on the real
    executable plus its first positional, so a failed `bash run_job.sh` is not
    cleared by an unrelated `bash cleanup.sh`.
    """

    invocations = parse_shell_command(command)
    for invocation in invocations:
        if invocation.kind != "pip_install":
            continue
        for index, token in enumerate(invocation.args[:-1]):
            value = invocation.args[index + 1]
            if token == "-r" and re.fullmatch(r"[A-Za-z0-9_./-]*requirements[A-Za-z0-9_.-]*\.txt", value):
                return f"pip install {Path(value).name}"
        return "pip install"
    for invocation in invocations:
        if invocation.kind == "script":
            role = "export" if "--export" in invocation.args else "run"
            return f"python {invocation.positionals[0]} {role}"
    for invocation in invocations:
        if invocation.kind == "module":
            return f"python -m {invocation.module} {' '.join(invocation.positionals)}".rstrip()
    if not invocations or any(not invocation.executable for invocation in invocations):
        # String-level fallback for quoting the tokenizer cannot parse: the
        # command stays on the (bare) module key, which the anchor logic
        # refuses as broad-match rerun evidence. `\s*` covers the attached
        # `-mmodule` form Python accepts.
        module = re.search(r"\bpython[\d.]*\s+(?:-[^m]\S*\s+)*-m\s*([A-Za-z0-9_.]+)", _unwrap_shell_command(command))
        if module:
            return f"python -m {module.group(1)}"
    for invocation in invocations:
        # Guards and `cd`/`export`/`set` position the real work rather than perform it.
        if invocation.kind == "guard" or not invocation.executable or invocation.executable in ("cd", "export", "set"):
            continue
        return " ".join((invocation.executable, *invocation.positionals[:1]))
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
    source = metric.get("source")
    if source not in {"metrics_artifact", "runtime_log_artifact"} or not metric.get("reported_values"):
        return ""
    if not metric_is_runtime_result_artifact(metric):
        return ""
    source_path = str(metric.get("source_path") or "")
    if source_path:
        return f"captured validation metric artifact `{truncate(source_path, 180)}`"
    return "captured validation metric artifact"


def artifact_validation_metric_is_runtime_evidence(run: dict[str, Any]) -> bool:
    metric = run.get("validation_metric") if isinstance(run.get("validation_metric"), dict) else {}
    return metric_is_runtime_result_artifact(metric)


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
    """True when some shell stage EXECUTES an installer (executable position).

    ``echo 'pip install torch'`` is not an install (echo is the executable),
    while ``pip install torch | grep Successfully`` is (the first pipeline
    stage executes pip). Covers `pip`/`pip3`/`uv pip` and the `python -m pip`
    forms, including the attached `-mpip` spelling Python accepts.
    """

    return any(invocation.kind == "pip_install" for invocation in parse_shell_command(command))


def event_timeline_from_text(events_text: str) -> list[dict[str, Any]]:
    """Ordered command/message items from a captured agent event stream.

    Understands both stream shapes: Claude (message.content text/tool_use items,
    with outputs from tool_result content items matched by tool_use_id or a
    top-level tool_use_result) and codex (item.completed command_execution /
    agent_message / reasoning items).
    """

    items: list[dict[str, Any]] = []
    pending_commands_by_tool_id: dict[str, dict[str, Any]] = {}
    for line in str(events_text or "").splitlines():
        try:
            payload = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        codex_item = payload.get("item")
        if isinstance(codex_item, dict):
            if codex_item.get("type") == "command_execution" and payload.get("type") == "item.completed":
                items.append(
                    {
                        "kind": "command",
                        "command": str(codex_item.get("command") or ""),
                        "output": str(codex_item.get("aggregated_output") or ""),
                        "exit_code": codex_item.get("exit_code"),
                    }
                )
            elif codex_item.get("type") in ("agent_message", "reasoning"):
                text = str(codex_item.get("text") or "").strip()
                if text:
                    items.append({"kind": "message", "text": text})
            continue
        resolved_tool_result = False
        for entry in message_content_items(payload):
            if entry.get("type") == "text" and str(entry.get("text") or "").strip():
                items.append({"kind": "message", "text": str(entry["text"]).strip()})
            elif entry.get("type") == "tool_use" and isinstance(entry.get("input"), dict):
                command = str(entry["input"].get("command") or "")
                if command:
                    command_item = {"kind": "command", "command": command, "output": "", "exit_code": None}
                    items.append(command_item)
                    tool_id = str(entry.get("id") or "")
                    if tool_id:
                        pending_commands_by_tool_id[tool_id] = command_item
            elif entry.get("type") == "tool_result":
                command_item = pending_commands_by_tool_id.pop(str(entry.get("tool_use_id") or ""), None)
                if command_item is None:
                    continue
                output = tool_result_output(payload, entry)
                command_item["output"] = output
                command_item["exit_code"], _status = tool_result_exit(payload, entry, output)
                resolved_tool_result = True
        result = payload.get("tool_use_result")
        if not resolved_tool_result and isinstance(result, dict) and items and items[-1]["kind"] == "command":
            items[-1]["output"] = "\n".join(
                str(result.get(key) or "") for key in ("stdout", "stderr") if result.get(key)
            )
    return items


# Error-class taxonomy: how to read a failure signature out of command output.
# Data, not narrative — extending coverage means adding a row here.
ERROR_SIGNATURE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"ModuleNotFoundError: No module named ['\"]?([A-Za-z0-9_.]+)['\"]?", "missing_python_module"),
    (r"ImportError: cannot import name ['\"]?([A-Za-z0-9_.]+)['\"]?", "import_error"),
    (r"(?:FileNotFoundError|No such file or directory)[:\s]+['\"]?([\w./-]+)", "missing_file"),
    (r"([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)): ([^\n]{1,120})", "exception"),
)


def error_signature_from_output(output: str) -> dict[str, str] | None:
    for pattern, kind in ERROR_SIGNATURE_PATTERNS:
        match = re.search(pattern, output)
        if match:
            return {"kind": kind, "subject": match.group(1), "display": match.group(0).splitlines()[0]}
    return None


def _command_is_inspection_only(command: str) -> bool:
    """True when every pipeline stage of every segment only reads/inspects files.

    Inspection commands (cat/grep/sed/...) echo file contents, so their output
    can quote an old traceback or an old success marker — they are neither
    failure anchors nor recovery evidence for `terminal_failure_anchor`.
    Status guards like ``|| true`` are ignored (they cannot make an inspection
    pipeline execute anything), while a pipeline stage that executes real work
    (``cat cfg.json | python3 run.py``) disqualifies the whole command.
    """

    invocations = [invocation for invocation in parse_shell_command(command) if invocation.kind != "guard"]
    return bool(invocations) and all(invocation.kind == "inspection" for invocation in invocations)


_SHELL_INTERPRETERS = frozenset({"bash", "sh", "zsh", "dash", "ksh"})

#: Non-Python runtime wrappers that execute a job when invoked: `make
#: simulate` runs whatever its Makefile target wraps, and the SDK console
#: script (`nvflare simulator ...`) is the CLI twin of `python -m
#: nvflare.cli`, which the module branch already accepts.
_JOB_WRAPPER_EXECUTABLES = frozenset({"make", "nvflare"})


def _invocation_is_job_run(invocation: Invocation, *, python_only: bool = False) -> bool:
    if invocation.kind == "script":
        return True
    if invocation.kind == "module":
        return invocation.module.split(".")[0].lower() not in _PYTHON_TOOLING_MODULES
    if python_only:
        return False
    if invocation.executable in _JOB_WRAPPER_EXECUTABLES:
        return True
    if invocation.executable in _SHELL_INTERPRETERS and invocation.positionals:
        return True
    return invocation.executable.endswith((".sh", ".py"))


def _command_is_python_job_run(command: str) -> bool:
    """True when any stage executes a Python script or module (a job run) —
    including behind ``sh -c`` wrappers, env-var prefixes, and
    ``env``/``nohup``/``time``/``timeout``. Interpreter tooling (``-m pip``,
    ``-m venv``, ``-m ensurepip``) and inline ``-c`` snippets are not job
    runs, so a dependency install can still recover a failed import probe.
    """

    return any(_invocation_is_job_run(invocation, python_only=True) for invocation in parse_shell_command(command))


def _command_is_job_run(command: str) -> bool:
    """True when any stage is job-run-shaped: a Python script/module run, a
    script-interpreter execution, or a runtime wrapper (``make simulate``,
    the SDK console script). Only such commands can vouch for a job's
    success — an arbitrary command (``python -c "print(open('old.log')...)"``)
    can echo a stale success marker without running anything.
    """

    return any(_invocation_is_job_run(invocation) for invocation in parse_shell_command(command))


def _invocation_success_implied_by_zero_exit(invocations: list[Invocation], index: int) -> bool:
    """True when the command exiting 0 proves invocation ``index`` ran and exited 0.

    That holds only when every join from the invocation to the end is ``&&``
    (a later ``;`` discards its status; a later ``||`` can mask its failure —
    ``python -m mod || true`` exits 0 even when the module run failed; a
    following ``|`` replaces its status with the pipeline tail's) and the
    invocation is not itself the fallback arm of a ``||`` (which may never
    have run at all).
    """

    if index > 0 and invocations[index - 1].operator_after == "||":
        return False
    return all(invocation.operator_after == "&&" for invocation in invocations[index:-1])


def _is_bare_module_invocation(command: str, module: str, *, require_success_implied_by_zero_exit: bool = True) -> bool:
    """True when one stage runs exactly ``python -m <module>`` and nothing more.

    A bare module recovery key can come from a command whose tokens cannot be
    parsed even though it really carried a job target, and the key equally
    matches same-module non-reruns like ``--help``; parse-level verification
    (nothing after the module) is what lets a bare-key match stand in for a
    rerun. When the match vouches for a SUCCESSFUL rerun via the command's
    overall zero exit (the default), the module stage only counts when that
    zero exit actually entails the stage's success — a status guard like
    ``python -m mod || true`` must not pass. Pass
    ``require_success_implied_by_zero_exit=False`` for the FAILED command
    itself, whose nonzero exit is not the success signal: a status guard there
    must not disqualify later recovery — but the module stage must still be
    the command's ONLY real work (every other stage a pure status guard), or a
    chained ``python -m mod ; bash run_job.sh`` whose failure came from the
    OTHER stage would be clearable by rerunning just the bare module.
    """

    invocations = parse_shell_command(command)
    for index, invocation in enumerate(invocations):
        if invocation.kind != "module" or invocation.module != module or invocation.args:
            continue
        if require_success_implied_by_zero_exit:
            if _invocation_success_implied_by_zero_exit(invocations, index):
                return True
            continue
        if all(other.kind == "guard" for other_index, other in enumerate(invocations) if other_index != index):
            return True
    return False


def terminal_failure_anchor(timeline: list[dict[str, Any]]) -> tuple[int, dict[str, str]] | None:
    """Last failed command with a recognized error signature, if the run never recovered.

    Returns None only when a later command demonstrates recovery of the failing
    operation itself: a successful command sharing the failed command's recovery
    key, a job-run-shaped command whose output carries a known job-success
    marker, or — for a missing-module/import failure — a dependency-install
    command that installed the failure's subject. Unrelated successful commands
    (diagnostics like `ls`, `cat`, or log inspection after the failure — even
    ones that echo install logs or a stale success marker) do not count as
    recovery, so they cannot suppress a real terminal failure. When the failed
    command was itself a script/job run, installing the missing module alone is
    not recovery either: the job must be rerun successfully or a job-success
    marker must appear. A module-run key that carries no subcommand/job target
    matches any same-module invocation, so a bare-key match only counts as
    rerun evidence when both commands verify, token by token, as bare
    invocations of that module; otherwise such a run recovers only via an exact
    rerun of the failed command or a job-success marker.
    """

    anchor_index: int | None = None
    signature: dict[str, str] | None = None
    for index, item in enumerate(timeline):
        if item.get("kind") != "command":
            continue
        # Inspection commands (cat/grep/...) echo file contents: an old
        # traceback they quote is not this run's failure.
        if _command_is_inspection_only(str(item.get("command") or "")):
            continue
        candidate = error_signature_from_output(str(item.get("output") or ""))
        # A None exit is unknown, not failed: it anchors only together with a
        # recognized error signature in the output, never on its own.
        if candidate and item.get("exit_code") not in (0,):
            anchor_index, signature = index, candidate
    if anchor_index is None or signature is None:
        return None
    failed_command = str(timeline[anchor_index].get("command") or "")
    failed_key = command_recovery_key(failed_command)
    missing_module = (
        signature["subject"].split(".")[0] if signature["kind"] in ("missing_python_module", "import_error") else ""
    )
    failed_command_was_job_run = _command_is_job_run(failed_command)
    # A module key without a subcommand/job-target suffix (`python -m
    # nvflare.cli`, e.g. when the failed command's tokens cannot be parsed)
    # would match ANY later successful invocation of the same module —
    # `--help`, another subcommand. A bare-key match is therefore rerun
    # evidence only when BOTH commands verify, at the parse level, as
    # genuinely bare invocations of the module — which still accepts
    # semantically identical reruns (`python` vs `python3`, `sh -c` wrapping)
    # that raw string equality would reject. Otherwise recovery needs an
    # exact rerun or the job-success output check below.
    bare_module_key = re.fullmatch(r"python -m (\S+)", failed_key)
    # The failed command exited nonzero, so its overall exit status is not a
    # success signal: a status guard on the FAILED command (`python -m mod ||
    # exit 1`) must not stop a later clean bare rerun from counting as
    # recovery — the zero-exit implication check applies only to the recovery
    # candidate below.
    failed_is_verified_bare_module_run = bool(bare_module_key) and _is_bare_module_invocation(
        failed_command, bare_module_key.group(1), require_success_implied_by_zero_exit=False
    )
    for item in timeline[anchor_index + 1 :]:
        if item.get("kind") != "command":
            continue
        command = str(item.get("command") or "")
        # Inspection commands can quote an OLD success log; they are not
        # recovery evidence.
        if _command_is_inspection_only(command):
            continue
        output = str(item.get("output") or "")
        # An output-borne success marker only vouches for the job when the
        # command that printed it is itself job-run-shaped: any other command
        # can echo a stale marker (e.g. printing an old log via `python -c`).
        if job_output_succeeded(output) and _command_is_job_run(command):
            return None
        if item.get("exit_code") != 0:
            continue
        # A failed job run only recovers via another job run: broad keys (bare
        # interpreter/shell names) must not let an unrelated successful command
        # (e.g. an import probe after an install) pass for a job rerun. An
        # exact rerun of the failed command is always rerun evidence, even
        # when its key is a bare module key.
        key_match_is_rerun = command_recovery_key(command) == failed_key and (
            not failed_command_was_job_run or _command_is_job_run(command)
        )
        if bare_module_key:
            key_match_is_rerun = (
                key_match_is_rerun
                and failed_is_verified_bare_module_run
                and _is_bare_module_invocation(command, bare_module_key.group(1))
            )
        if command.strip() == failed_command.strip() or key_match_is_rerun:
            return None
        if (
            missing_module
            and not failed_command_was_job_run
            and is_dependency_install_command(command)
            and re.search(
                rf"\bSuccessfully installed\b[^\n]*\b{re.escape(missing_module)}\b", output, flags=re.IGNORECASE
            )
        ):
            return None
    return anchor_index, signature


# Each cue must itself express a PREDICTION of failure or blockage. Bare state
# descriptions ("torch is missing", "not installed") are excluded: they also
# appear in remediation statements ("torch is not installed, so I'll install
# it first"), which are the opposite of a known-doomed execution. A bare
# expectation verb ("expect", "anticipate") also matches predictions of
# success, so those verbs only count with a failure word in the same sentence.
_FAILURE_PREDICTION_CUE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b(?:will|would|going to|known to|expected to|likely to|bound to)\s+(?:fail|stop|break|crash|error)\b",
        r"\b(?:expect|anticipat)\w*\b[^.!?\n]{0,80}\b(?:fail\w*|error\w*|stop|break|crash\w*|blocked?)\b",
        r"\bstop at\b",
    )
)


def predicted_failure_message(events_text: str) -> dict[str, str] | None:
    """Detect a known-doomed execution in the captured event stream.

    Returns the agent message that predicted the terminal failure (it names the
    failure's subject alongside a cue expressing expected failure or blockage)
    yet preceded running the failing command anyway — a lint signal that the
    agent should have either
    resolved the blocker or skipped the command with an explicit blocker.
    """

    timeline = event_timeline_from_text(events_text)
    anchored = terminal_failure_anchor(timeline)
    if anchored is None:
        return None
    anchor_index, signature = anchored
    subject = signature["subject"].split(".")[0].lower()
    for item in timeline[:anchor_index]:
        if item["kind"] != "message":
            continue
        lowered = str(item.get("text") or "").lower()
        if subject in lowered and any(pattern.search(lowered) for pattern in _FAILURE_PREDICTION_CUE_PATTERNS):
            return {"quote": str(item["text"]), "error": signature["display"]}
    return None


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
