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

"""Report-only skill usage display helpers."""

from __future__ import annotations

import json
import posixpath
import re
from pathlib import PurePosixPath
from typing import Any, Mapping

_SKILL_PATH_RE = re.compile(r"(/[^\s\"'`]*\.(?:codex|claude)/skills/[^\s\"'`]+)")


def _append_unique(values: list[str], seen: set[str], value: str) -> None:
    if value and value not in seen:
        seen.add(value)
        values.append(value)


def _skill_relative_path(file_path: str) -> str:
    normalized = posixpath.normpath(str(file_path or "").strip())
    for marker in ("/.codex/skills/", "/.claude/skills/"):
        if marker in normalized:
            return normalized.split(marker, 1)[1].lstrip("/")
    return ""


def _valid_skill_name(skill_name: str) -> str:
    skill_name = str(skill_name or "").strip()
    if not skill_name or skill_name == "_shared" or skill_name.startswith("."):
        return ""
    return skill_name


def _skill_name_from_relative_path(rel_path: str) -> str:
    if not rel_path:
        return ""
    top_level = rel_path.split("/", 1)[0]
    return _valid_skill_name(top_level)


def _skill_name_from_path(file_path: str) -> str:
    return _skill_name_from_relative_path(_skill_relative_path(file_path))


def _is_top_level_skill_md(rel_path: str) -> bool:
    parts = PurePosixPath(rel_path).parts
    return len(parts) == 2 and parts[1].lower() == "skill.md"


def _skill_inspection_name_from_path(file_path: str) -> str:
    rel_path = _skill_relative_path(file_path)
    if not _is_top_level_skill_md(rel_path):
        return ""
    return _skill_name_from_relative_path(rel_path)


def _skill_reference_name_from_path(file_path: str) -> str:
    rel_path = _skill_relative_path(file_path)
    if not rel_path or _is_top_level_skill_md(rel_path):
        return ""
    return _skill_name_from_relative_path(rel_path)


# Shared-reference containers observed under the skills root: the visible
# `_shared/` directory and hidden dot-dirs like `.nvflare-shared/`. A dot-dir
# may nest a content-hash segment (`.nvflare-shared/<sha256>/references/...`).
_SHARED_CONTAINER_RE = re.compile(r"^(?:_shared|\.[A-Za-z0-9_-]*shared)$", re.IGNORECASE)
_HEX_SEGMENT_RE = re.compile(r"^[0-9a-f]{32,}$", re.IGNORECASE)


def _shared_ref_from_path(file_path: str) -> str:
    rel_path = _skill_relative_path(file_path)
    parts = PurePosixPath(rel_path).parts if rel_path else ()
    if not parts or not _SHARED_CONTAINER_RE.match(parts[0]):
        return ""
    # Drop content-hash segments so refs display and dedupe by meaningful path.
    cleaned = [parts[0], *(part for part in parts[1:] if not _HEX_SEGMENT_RE.match(part))]
    if len(cleaned) < 2:
        # The bare container (e.g. a directory listing) is not a reference read.
        return ""
    return str(PurePosixPath(*cleaned))


def _skill_paths_from_text(text: str) -> list[str]:
    return [match.group(1) for match in _SKILL_PATH_RE.finditer(str(text or ""))]


def _event_read_tool_paths(event: dict[str, Any]) -> list[str]:
    message = event.get("message")
    if not isinstance(message, dict):
        return []
    paths: list[str] = []
    for item in message.get("content") or []:
        if not isinstance(item, dict) or item.get("name") != "Read":
            continue
        tool_input = item.get("input")
        if not isinstance(tool_input, dict):
            continue
        paths.append(str(tool_input.get("file_path") or ""))
    return paths


def _event_skill_paths(event: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    paths.extend(_skill_paths_from_text(_event_command_text(event)))
    paths.extend(_event_read_tool_paths(event))
    return paths


def _event_command_text(event: dict[str, Any]) -> str:
    item = event.get("item")
    if isinstance(item, dict):
        return str(item.get("command") or "")
    return ""


def explicit_skill_tool_calls(events_text: str) -> list[str]:
    skills: list[str] = []
    seen: set[str] = set()
    for line in events_text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        for item in message.get("content") or []:
            if not isinstance(item, dict) or item.get("name") != "Skill":
                continue
            tool_input = item.get("input")
            if not isinstance(tool_input, dict):
                continue
            skill_name = str(tool_input.get("skill") or "").strip()
            _append_unique(skills, seen, skill_name)
    return skills


def skill_instruction_reads(events_text: str) -> list[str]:
    skills: list[str] = []
    seen: set[str] = set()
    for line in events_text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        for file_path in _event_skill_paths(event):
            _append_unique(skills, seen, _skill_name_from_path(file_path))
    return skills


def skill_inspection_reads(events_text: str) -> list[str]:
    """Return skills whose top-level SKILL.md was read while routing/inspecting."""

    skills: list[str] = []
    seen: set[str] = set()
    for line in events_text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        for file_path in _event_skill_paths(event):
            _append_unique(skills, seen, _skill_inspection_name_from_path(file_path))
    return skills


def skill_reference_reads(events_text: str) -> list[str]:
    """Return skills with deeper non-shared instruction/reference evidence."""

    skills: list[str] = []
    seen: set[str] = set()
    for line in events_text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        for file_path in _event_skill_paths(event):
            _append_unique(skills, seen, _skill_reference_name_from_path(file_path))
    return skills


def shared_skill_reference_reads(events_text: str) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for line in events_text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        for file_path in _event_skill_paths(event):
            _append_unique(refs, seen, _shared_ref_from_path(file_path))
    return refs


def available_skill_names(skills_list: Any) -> list[str]:
    """Return callable skill names exposed to the agent, excluding shared refs."""

    if not isinstance(skills_list, Mapping):
        return []
    data = skills_list.get("data") if isinstance(skills_list.get("data"), Mapping) else skills_list
    candidates = data.get("available") if isinstance(data, Mapping) else None
    if not isinstance(candidates, list):
        candidates = data.get("installed") if isinstance(data, Mapping) else None
    if not isinstance(candidates, list):
        return []

    names: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if isinstance(item, Mapping):
            name = str(item.get("name") or "").strip()
        else:
            name = str(item or "").strip()
        name = _valid_skill_name(name)
        if not name:
            continue
        _append_unique(names, seen, name)
    return names


def skill_availability_display(skills_list: Any, *, skills_enabled: Any = None) -> str:
    names = available_skill_names(skills_list)
    if names:
        return "; ".join(names)
    status = ""
    if isinstance(skills_list, Mapping):
        status = str(skills_list.get("status") or "").strip().lower()
    if skills_enabled is False or status == "skipped":
        return "not enabled"
    return "not recorded" if skills_enabled else "none"


def skill_usage_display(*, events_text: str = "", observed_skill_name: Any = None, skills_enabled: Any = None) -> str:
    explicit = explicit_skill_tool_calls(events_text)
    if explicit:
        return "; ".join(explicit)
    reference_reads = skill_reference_reads(events_text)
    if reference_reads:
        return "; ".join(reference_reads)
    observed = _valid_skill_name(str(observed_skill_name or ""))
    if observed:
        return observed
    return "none recorded" if skills_enabled else "none"


def skill_inspection_display(events_text: str = "") -> str:
    inspected = skill_inspection_reads(events_text)
    return "; ".join(inspected) if inspected else "none"


def shared_skill_usage_display(events_text: str = "") -> str:
    refs = shared_skill_reference_reads(events_text)
    return "; ".join(refs) if refs else "none"
