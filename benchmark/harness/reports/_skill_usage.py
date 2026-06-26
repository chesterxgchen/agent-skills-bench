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
from pathlib import PurePosixPath
from typing import Any


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
            if skill_name and skill_name not in seen:
                seen.add(skill_name)
                skills.append(skill_name)
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
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        for item in message.get("content") or []:
            if not isinstance(item, dict) or item.get("name") != "Read":
                continue
            tool_input = item.get("input")
            if not isinstance(tool_input, dict):
                continue
            file_path = str(tool_input.get("file_path") or "").strip()
            if "/_shared/" not in file_path:
                continue
            ref = "_shared/" + file_path.split("/_shared/", 1)[1].lstrip("/")
            ref = str(PurePosixPath(ref))
            if ref and ref not in seen:
                seen.add(ref)
                refs.append(ref)
    return refs


def skill_usage_display(*, events_text: str = "", observed_skill_name: Any = None, skills_enabled: Any = None) -> str:
    explicit = explicit_skill_tool_calls(events_text)
    if explicit:
        return "; ".join(explicit)
    return "none recorded" if skills_enabled else "none"


def shared_skill_usage_display(events_text: str = "") -> str:
    refs = shared_skill_reference_reads(events_text)
    return "; ".join(refs) if refs else "none"
