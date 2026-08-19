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

import json


def _background_events(*, terminal_before_success: bool) -> list[dict]:
    events = [
        {
            "event_type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "id": "toolu_sim",
                        "input": {
                            "command": "python job.py --num-rounds 3",
                            "description": "Run full simulation",
                            "run_in_background": True,
                        },
                    }
                ]
            },
        },
        {
            "event_type": "system.task_started",
            "task_id": "sim_task",
            "tool_use_id": "toolu_sim",
            "description": "Run full simulation",
        },
    ]
    terminal = {
        "event_type": "system.task_updated",
        "task_id": "sim_task",
        "patch": {"status": "completed" if terminal_before_success else "killed"},
    }
    if terminal_before_success:
        events.append(terminal)
    events.append({"type": "result", "subtype": "success", "event_type": "result.success"})
    if not terminal_before_success:
        events.append(terminal)
    return events


def test_background_task_snapshot_preserves_pending_state_at_agent_success():
    from benchmark.harness.background_tasks import background_tasks_pending_at_success

    saw_success, pending = background_tasks_pending_at_success(_background_events(terminal_before_success=False))

    assert saw_success is True
    assert pending == [
        {
            "task_id": "sim_task",
            "tool_use_id": "toolu_sim",
            "status": "running",
            "command": "python job.py --num-rounds 3",
            "description": "Run full simulation",
        }
    ]


def test_background_task_snapshot_allows_task_completed_before_agent_success():
    from benchmark.harness.background_tasks import background_tasks_pending_at_success

    saw_success, pending = background_tasks_pending_at_success(_background_events(terminal_before_success=True))

    assert saw_success is True
    assert pending == []


def test_background_task_tracker_accepts_automatic_background_result_mapping():
    from benchmark.harness.background_tasks import active_background_tasks

    events = [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "id": "toolu_auto",
                        "input": {"command": "python long_job.py"},
                    }
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_auto",
                        "content": "Command running in background with ID: auto_task",
                    }
                ]
            },
            "tool_use_result": {"backgroundTaskId": "auto_task"},
        },
    ]

    assert active_background_tasks(events) == [
        {
            "task_id": "auto_task",
            "tool_use_id": "toolu_auto",
            "status": "running",
            "command": "python long_job.py",
            "description": "",
        }
    ]


def test_claude_stop_guard_blocks_until_background_task_is_terminal(tmp_path):
    from benchmark.harness.hooks.claude_stop_guard import stop_decision

    result_dir = tmp_path / "results"
    result_dir.mkdir()
    events_path = result_dir / "agent_events.jsonl"
    events = _background_events(terminal_before_success=False)[:2]
    events_path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    decision = stop_decision(
        {"transcript_path": str(tmp_path / "missing.jsonl"), "stop_hook_active": False},
        {"RESULT_DIR": str(result_dir)},
    )

    assert decision is not None
    assert decision["decision"] == "block"
    assert "sim_task" in decision["reason"]
    assert "Wait for each task" in decision["reason"]

    with events_path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "event_type": "system.task_notification",
                    "task_id": "sim_task",
                    "status": "completed",
                }
            )
            + "\n"
        )
    assert stop_decision({}, {"RESULT_DIR": str(result_dir)}) is None


def test_claude_pre_tool_guard_rejects_double_backgrounding():
    from benchmark.harness.hooks.claude_stop_guard import pre_tool_use_decision

    decision = pre_tool_use_decision(
        {
            "tool_name": "Bash",
            "tool_input": {
                "command": "(python job.py > sim.log 2>&1; echo $? > sim.exit) &",
                "run_in_background": True,
            },
        }
    )

    assert decision is not None
    output = decision["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    assert "detaches the real process" in output["permissionDecisionReason"]


def test_claude_pre_tool_guard_allows_one_background_owner_and_shell_redirection():
    from benchmark.harness.hooks.claude_stop_guard import pre_tool_use_decision

    base = {"tool_name": "Bash", "tool_input": {"run_in_background": True}}
    base["tool_input"]["command"] = "python job.py > sim.log 2>&1"
    assert pre_tool_use_decision(base) is None

    base["tool_input"]["command"] = "echo 'R&D' && python job.py"
    assert pre_tool_use_decision(base) is None


def test_record_policy_failure_includes_background_task_lifecycle():
    from benchmark.harness.container.agent_run import record_has_policy_failure

    assert record_has_policy_failure({"background_task_lifecycle": {"status": "fail"}}) is True
    assert record_has_policy_failure({"background_task_lifecycle": {"status": "pass"}}) is False
