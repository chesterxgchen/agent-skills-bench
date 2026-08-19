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

"""Prevent Claude from ending a benchmark turn with live background tasks."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# A settings hook is launched from the agent's task workspace, not the harness
# source root. Make the baked-in /workspace package importable without relying
# on the measured job's Python environment.
HARNESS_ROOT = Path(__file__).resolve().parents[3]
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

from benchmark.harness.background_tasks import active_background_tasks, iter_jsonl_objects  # noqa: E402


def _has_unquoted_background_operator(command: str) -> bool:
    """Return whether a shell command contains a standalone, unquoted ``&``."""

    quote = ""
    escaped = False
    for index, char in enumerate(command):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote != "'":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char != "&":
            continue
        previous = command[index - 1] if index else ""
        following = command[index + 1] if index + 1 < len(command) else ""
        if previous not in {"&", ">", "<"} and following not in {"&", ">"}:
            return True
    return False


def pre_tool_use_decision(hook_input: dict[str, Any]) -> dict[str, Any] | None:
    """Reject Bash calls that create an untrackable second background layer."""

    if str(hook_input.get("tool_name") or "").lower() not in {"bash", "shell"}:
        return None
    tool_input = hook_input.get("tool_input")
    if not isinstance(tool_input, dict) or tool_input.get("run_in_background") is not True:
        return None
    command = str(tool_input.get("command") or "")
    if not _has_unquoted_background_operator(command):
        return None
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                "Do not combine Bash run_in_background=true with a shell-level '&'. "
                "That detaches the real process from Claude's tracked task. Remove the shell '&' "
                "and let run_in_background own the task lifecycle, then wait for its terminal status."
            ),
        }
    }


def _event_source(hook_input: dict[str, Any], environ: dict[str, str]) -> Path | None:
    result_dir = environ.get("RESULT_DIR")
    if result_dir:
        events_path = Path(result_dir) / "agent_events.jsonl"
        if events_path.is_file():
            return events_path
    transcript_path = hook_input.get("transcript_path")
    if isinstance(transcript_path, str) and transcript_path:
        path = Path(transcript_path)
        if path.is_file():
            return path
    return None


def stop_decision(hook_input: dict[str, Any], environ: dict[str, str] | None = None) -> dict[str, str] | None:
    """Return a Claude Stop-hook block decision when background work is live."""

    source = _event_source(hook_input, environ if environ is not None else os.environ)
    if source is None:
        return None
    pending = active_background_tasks(iter_jsonl_objects(source))
    if not pending:
        return None
    labels = []
    for task in pending[:3]:
        task_id = task.get("task_id") or task.get("tool_use_id") or "unknown"
        detail = task.get("description") or task.get("command") or "background command"
        labels.append(f"{task_id} ({detail})")
    more = f" and {len(pending) - len(labels)} more" if len(pending) > len(labels) else ""
    reason = (
        "The benchmark still has active background work: "
        + "; ".join(labels)
        + more
        + ". Do not finish yet. Wait for each task to reach a terminal status, inspect its result, "
        "and explicitly stop any task that is intentionally no longer needed before answering."
    )
    return {"decision": "block", "reason": reason}


def main() -> int:
    try:
        hook_input = json.load(sys.stdin)
    except (OSError, TypeError, ValueError):
        return 0
    if not isinstance(hook_input, dict):
        return 0
    try:
        if hook_input.get("hook_event_name") == "PreToolUse":
            decision = pre_tool_use_decision(hook_input)
        else:
            decision = stop_decision(hook_input)
    except Exception as exc:  # Hooks must fail open; post-processing is the backstop.
        print(f"background-task Stop hook failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 0
    if decision:
        print(json.dumps(decision, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
