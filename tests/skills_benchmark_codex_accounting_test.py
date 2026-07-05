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

"""Codex lifecycle-event accounting: command dedupe, assistant turns, shared refs."""

from __future__ import annotations

import json

from benchmark.harness.events import parse_usage_and_activity_data
from benchmark.harness.reports._skill_usage import shared_skill_reference_reads


def _codex_lifecycle_events() -> str:
    # One command item across started/completed (same id, same command), plus an
    # assistant message item. Codex emits the item payload on EVERY lifecycle event.
    events = [
        {"type": "item.started", "item": {"id": "item_1", "type": "command_execution", "command": "ls -la"}},
        {
            "type": "item.completed",
            "item": {"id": "item_1", "type": "command_execution", "command": "ls -la", "exit_code": 0},
        },
        {"type": "item.started", "item": {"id": "item_2", "type": "command_execution", "command": "cat train.py"}},
        {
            "type": "item.completed",
            "item": {"id": "item_2", "type": "command_execution", "command": "cat train.py", "exit_code": 0},
        },
        {"type": "item.completed", "item": {"id": "item_3", "type": "agent_message", "text": "done"}},
    ]
    return "\n".join(json.dumps(event) for event in events) + "\n"


def test_codex_commands_deduped_by_item_id(tmp_path):
    """Regression: start/completion lifecycle events doubled every Codex command."""

    events_path = tmp_path / "agent_events.jsonl"
    events_path.write_text(_codex_lifecycle_events(), encoding="utf-8")
    _usage, activity = parse_usage_and_activity_data(events_path)
    assert activity["command_count"] == 2
    assert activity["unique_command_count"] == 2
    assert activity["commands"] == ["ls -la", "cat train.py"]


def test_repeated_command_with_distinct_item_ids_still_counts_each(tmp_path):
    events_path = tmp_path / "agent_events.jsonl"
    events = [
        {"type": "item.started", "item": {"id": "item_1", "type": "command_execution", "command": "pytest -q"}},
        {"type": "item.started", "item": {"id": "item_2", "type": "command_execution", "command": "pytest -q"}},
    ]
    events_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    _usage, activity = parse_usage_and_activity_data(events_path)
    assert activity["command_count"] == 2
    assert activity["unique_command_count"] == 1


def test_events_without_item_ids_keep_prior_counting(tmp_path):
    events_path = tmp_path / "agent_events.jsonl"
    events = [
        {"type": "assistant", "message": {"content": []}, "command": "echo one"},
        {"type": "assistant", "message": {"content": []}, "command": "echo one"},
    ]
    events_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    _usage, activity = parse_usage_and_activity_data(events_path)
    assert activity["command_count"] == 2


def test_codex_assistant_turns_fall_back_to_agent_messages(tmp_path):
    """Regression: Codex runs reported 0 assistant turns."""

    from benchmark.harness.reports.evidence import _run_evidence_from_bundle
    from benchmark.harness.reports.insights._spans import _assistant_turns

    events_path = tmp_path / "agent_events.jsonl"
    events_path.write_text(_codex_lifecycle_events(), encoding="utf-8")
    _usage, activity = parse_usage_and_activity_data(events_path)
    run = _run_evidence_from_bundle({"available": True, "label": "Run", "activity": activity})
    assert _assistant_turns(run) == 1
    # Claude-style explicit assistant events still take precedence.
    claude_run = _run_evidence_from_bundle(
        {"available": True, "label": "Run", "activity": {"event_types": {"assistant": 7, "agent_message": 3}}}
    )
    assert _assistant_turns(claude_run) == 7


def test_shared_refs_detected_in_hidden_hashed_shared_dir():
    """Regression: refs under `.nvflare-shared/<sha256>/references/` reported as none read."""

    sha = "629d845fcdc8052fb94a544b91ce76b0a43abcb9b794a4df360317e1d5d23450"
    events = [
        {
            "type": "item.started",
            "item": {
                "id": "item_1",
                "type": "command_execution",
                "command": f"cat /workspace/.codex/skills/.nvflare-shared/{sha}/references/dependency-install.md",
            },
        },
        {
            "type": "item.started",
            "item": {
                "id": "item_2",
                "type": "command_execution",
                "command": "cat /workspace/.claude/skills/_shared/references/common.md",
            },
        },
        # A bare listing of the shared container is not a reference read.
        {
            "type": "item.started",
            "item": {
                "id": "item_3",
                "type": "command_execution",
                "command": "find /workspace/.codex/skills/.nvflare-shared -maxdepth 4 -type f",
            },
        },
    ]
    text = "\n".join(json.dumps(event) for event in events) + "\n"
    refs = shared_skill_reference_reads(text)
    assert refs == [
        ".nvflare-shared/references/dependency-install.md",
        "_shared/references/common.md",
    ]
