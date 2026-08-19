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

"""Agent background-task lifecycle tracking shared by runtime guards."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

TERMINAL_BACKGROUND_TASK_STATUSES = {"completed", "failed", "killed", "stopped"}
_BACKGROUND_TASK_ID_RE = re.compile(r"\bbackground with ID:\s*([A-Za-z0-9_-]+)", re.IGNORECASE)
_TASK_NOTIFICATION_RE = re.compile(
    r"<task-id>\s*([^<]+?)\s*</task-id>.*?<status>\s*([^<]+?)\s*</status>",
    re.IGNORECASE | re.DOTALL,
)
_MAX_DETAIL_LENGTH = 240


def iter_jsonl_objects(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from ``path``, ignoring partial or non-object lines."""

    try:
        stream = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return
    with stream:
        for line in stream:
            try:
                payload = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict):
                yield payload


def _message_content(payload: dict[str, Any]) -> list[dict[str, Any]]:
    message = payload.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [item for item in content if isinstance(item, dict)]


def _short(value: Any) -> str:
    text = str(value or "").strip()
    return text if len(text) <= _MAX_DETAIL_LENGTH else text[: _MAX_DETAIL_LENGTH - 3] + "..."


def _event_type(payload: dict[str, Any]) -> str:
    event_type = str(payload.get("event_type") or "")
    if event_type:
        return event_type
    raw_type = str(payload.get("type") or "")
    subtype = str(payload.get("subtype") or "")
    return f"{raw_type}.{subtype}" if raw_type and subtype else raw_type


def _is_success_result(payload: dict[str, Any]) -> bool:
    return _event_type(payload) == "result.success"


def _background_task_id(payload: dict[str, Any], item: dict[str, Any]) -> str:
    candidates = (payload.get("tool_use_result"), payload.get("toolUseResult"), item.get("tool_use_result"))
    for result in candidates:
        if not isinstance(result, dict):
            continue
        task_id = result.get("backgroundTaskId") or result.get("background_task_id")
        if task_id:
            return str(task_id)
    content = str(item.get("content") or item.get("text") or "")
    match = _BACKGROUND_TASK_ID_RE.search(content)
    return match.group(1) if match else ""


class BackgroundTaskTracker:
    """Track background tools and tasks in one ordered event stream."""

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, dict[str, Any]] = {}
        self._task_by_tool_id: dict[str, str] = {}

    def _tool(self, tool_id: str) -> dict[str, Any]:
        return self._tools.setdefault(
            tool_id,
            {
                "tool_use_id": tool_id,
                "task_id": "",
                "command": "",
                "description": "",
                "status": "running",
                "background": False,
            },
        )

    def _task(self, task_id: str) -> dict[str, Any]:
        return self._tasks.setdefault(
            task_id,
            {
                "task_id": task_id,
                "tool_use_id": "",
                "command": "",
                "description": "",
                "status": "running",
            },
        )

    def _connect(self, tool_id: str, task_id: str) -> None:
        if not tool_id or not task_id:
            return
        tool = self._tool(tool_id)
        task = self._task(task_id)
        tool["background"] = True
        tool["task_id"] = task_id
        self._task_by_tool_id[tool_id] = task_id
        task["tool_use_id"] = tool_id
        for key in ("command", "description"):
            if tool.get(key) and not task.get(key):
                task[key] = tool[key]
        if task.get("status"):
            tool["status"] = task["status"]

    def _set_task_status(self, task_id: str, status: str) -> None:
        status = str(status or "").strip().lower()
        if not task_id or not status:
            return
        task = self._task(task_id)
        task["status"] = status
        tool_id = str(task.get("tool_use_id") or "")
        if tool_id:
            self._tool(tool_id)["status"] = status

    def _process_text_notification(self, text: Any) -> None:
        for match in _TASK_NOTIFICATION_RE.finditer(str(text or "")):
            self._set_task_status(match.group(1).strip(), match.group(2).strip())

    def process(self, payload: dict[str, Any]) -> None:
        for item in _message_content(payload):
            item_type = str(item.get("type") or "")
            if item_type == "tool_use" and str(item.get("name") or "").lower() in {"bash", "shell"}:
                tool_id = str(item.get("id") or "")
                tool_input = item.get("input") if isinstance(item.get("input"), dict) else {}
                if tool_id:
                    tool = self._tool(tool_id)
                    tool["command"] = _short(
                        tool_input.get("command") or tool_input.get("cmd") or tool_input.get("shell_command")
                    )
                    tool["description"] = _short(tool_input.get("description") or payload.get("description"))
                    if tool_input.get("run_in_background") is True:
                        tool["background"] = True
            elif item_type == "tool_result":
                tool_id = str(item.get("tool_use_id") or "")
                task_id = _background_task_id(payload, item)
                if tool_id and task_id:
                    self._connect(tool_id, task_id)
                self._process_text_notification(item.get("content") or item.get("text"))
            elif item_type == "text":
                self._process_text_notification(item.get("text"))

        event_type = _event_type(payload)
        if event_type == "system.task_started":
            task_id = str(payload.get("task_id") or "")
            tool_id = str(payload.get("tool_use_id") or "")
            if task_id:
                task = self._task(task_id)
                description = _short(payload.get("description"))
                if description:
                    task["description"] = description
                if tool_id:
                    self._connect(tool_id, task_id)
        elif event_type in {"system.task_updated", "system.task_notification"}:
            task_id = str(payload.get("task_id") or "")
            patch = payload.get("patch") if isinstance(payload.get("patch"), dict) else {}
            status = str(payload.get("status") or patch.get("status") or "")
            if task_id and status:
                self._set_task_status(task_id, status)
            if task_id:
                task = self._task(task_id)
                description = _short(payload.get("description") or payload.get("summary"))
                if description:
                    task["description"] = description

        self._process_text_notification(payload.get("content"))

    def active_tasks(self) -> list[dict[str, str]]:
        active: list[dict[str, str]] = []
        mapped_task_ids = set(self._task_by_tool_id.values())
        for task_id, task in sorted(self._tasks.items()):
            status = str(task.get("status") or "running").lower()
            if status in TERMINAL_BACKGROUND_TASK_STATUSES:
                continue
            active.append(
                {
                    "task_id": task_id,
                    "tool_use_id": str(task.get("tool_use_id") or ""),
                    "status": status,
                    "command": _short(task.get("command")),
                    "description": _short(task.get("description")),
                }
            )
        for tool_id, tool in sorted(self._tools.items()):
            if not tool.get("background") or str(tool.get("task_id") or "") in mapped_task_ids:
                continue
            status = str(tool.get("status") or "running").lower()
            if status in TERMINAL_BACKGROUND_TASK_STATUSES:
                continue
            active.append(
                {
                    "task_id": "",
                    "tool_use_id": tool_id,
                    "status": status,
                    "command": _short(tool.get("command")),
                    "description": _short(tool.get("description")),
                }
            )
        return active


def active_background_tasks(payloads: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    tracker = BackgroundTaskTracker()
    for payload in payloads:
        tracker.process(payload)
    return tracker.active_tasks()


def background_tasks_pending_at_success(
    payloads: Iterable[dict[str, Any]],
) -> tuple[bool, list[dict[str, str]]]:
    """Return the pending-task snapshot at the last successful result event."""

    tracker = BackgroundTaskTracker()
    saw_success = False
    pending: list[dict[str, str]] = []
    for payload in payloads:
        tracker.process(payload)
        if _is_success_result(payload):
            saw_success = True
            pending = tracker.active_tasks()
    return saw_success, pending
