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

"""SDK-NEUTRAL result-root fixture for the architecture guard.

Unlike ``_report_fixtures.build_result_root`` (which is deliberately NVFLARE-
flavoured, to lock the NVFLARE golden), this builds a canonical-layout result
root whose CAPTURED DATA carries no FL/SDK vocabulary: neutral file paths
(``src/app/...``), a neutral metric (``score`` / ``best score``), and neutral
command output. Rendered under a non-FL plugin it must contain zero FL terms, so
the architecture guard measures the ENGINE, not fixture data (architecture §6).

It is signal-rich: two available runs with metrics + token/command deltas, so the
always-on sections (exec summary, status, job run, algorithm, outcome, cost) all
render non-vacuously.
"""

from __future__ import annotations

import json
from pathlib import Path

WITHOUT_SKILLS_MODE = "without_skills"
WITH_SKILLS_MODE = "with_skills"
AGENT = "codex"
AGENT_MODEL = "default"

# Neutral per-mode inputs. No FL vocabulary anywhere.
_MODE_INPUTS = {
    WITHOUT_SKILLS_MODE: {
        "label": "No skills baseline",
        "skills_enabled": False,
        "elapsed_seconds": 120,
        "token_count": 9000,
        "command_count": 3,
        "unique_command_count": 3,
        "metric_value": 0.80,
        "commands": [
            (
                "pip install -r requirements.txt",
                "Install deps",
                "2026-06-13T20:00:00Z",
                "2026-06-13T20:00:20Z",
                "Successfully installed numpy-2.1.0",
            ),
            (
                "python train.py",
                "Run the task",
                "2026-06-13T20:00:30Z",
                "2026-06-13T20:02:00Z",
                "Done. best score: 0.80",
            ),
        ],
    },
    WITH_SKILLS_MODE: {
        "label": "With skills",
        "skills_enabled": True,
        "elapsed_seconds": 150,
        "token_count": 11000,
        "command_count": 4,
        "unique_command_count": 4,
        "metric_value": 0.85,
        "commands": [
            (
                "uv pip install -r requirements.txt",
                "Install deps",
                "2026-06-13T20:00:00Z",
                "2026-06-13T20:00:25Z",
                "Successfully installed numpy-2.1.0",
            ),
            (
                "python train.py",
                "Run the task",
                "2026-06-13T20:00:35Z",
                "2026-06-13T20:02:30Z",
                "Done. best score: 0.85",
            ),
        ],
    },
}

_STRUCTURE_ROOT = "src/app"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _agent_events_text(mode: str) -> str:
    lines: list[str] = []
    for index, (command, description, start, end, output) in enumerate(_MODE_INPUTS[mode]["commands"], start=1):
        lines.append(
            json.dumps(
                {
                    "type": "item.started",
                    "harness_timestamp": start,
                    "item": {
                        "command": command,
                        "description": description,
                        "id": f"{mode}-cmd-{index}",
                        "status": "in_progress",
                        "type": "command_execution",
                    },
                }
            )
        )
        lines.append(
            json.dumps(
                {
                    "type": "item.completed",
                    "harness_timestamp": end,
                    "item": {
                        "aggregated_output": output,
                        "command": command,
                        "description": description,
                        "exit_code": 0,
                        "id": f"{mode}-cmd-{index}",
                        "status": "completed",
                        "type": "command_execution",
                    },
                }
            )
        )
    return "\n".join(lines) + "\n"


def _populate_mode_dir(mode_dir: Path, mode: str) -> None:
    inputs = _MODE_INPUTS[mode]
    _write_json(
        mode_dir / "run_summary.json",
        {
            "mode": mode,
            "agent": AGENT,
            "agent_model": AGENT_MODEL,
            "model_source": "scenario",
            "elapsed_seconds": inputs["elapsed_seconds"],
            "token_count": inputs["token_count"],
            "agent_exit_code": 0,
            "final_container_exit_code": 0,
        },
    )
    _write_json(mode_dir / "container_exit_code.json", {"exit_code": 0})
    _write_json(
        mode_dir / "agent_usage.json", {"token_count": inputs["token_count"], "input_tokens": 0, "output_tokens": 0}
    )
    _write_json(
        mode_dir / "agent_activity.json",
        {"command_count": inputs["command_count"], "unique_command_count": inputs["unique_command_count"]},
    )
    _write_json(mode_dir / "runtime_image.json", {"image": "benchmark:base", "digest": "sha256:fixed"})

    changed_files = [
        {"artifact_path": f"changed_files/{_STRUCTURE_ROOT}/main.py", "path": f"{_STRUCTURE_ROOT}/main.py"},
        {"artifact_path": f"changed_files/{_STRUCTURE_ROOT}/model.py", "path": f"{_STRUCTURE_ROOT}/model.py"},
        {"artifact_path": f"changed_files/{_STRUCTURE_ROOT}/train.py", "path": f"{_STRUCTURE_ROOT}/train.py"},
    ]
    workspace_delta = {
        "changed_file_count": len(changed_files),
        "runtime_artifact_count": 0,
        "changed_files": changed_files,
        "final_structure_files": [{"path": f["path"]} for f in changed_files],
        "runtime_artifacts": [],
    }
    _write_json(mode_dir / "workspace_delta_manifest.json", workspace_delta)
    for entry in changed_files:
        target = mode_dir / "workspace_delta" / entry["artifact_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("def main():\n    return train_and_score()\n", encoding="utf-8")

    (mode_dir / "agent_events.jsonl").write_text(_agent_events_text(mode), encoding="utf-8")
    (mode_dir / "agent_last_message.txt").write_text(
        f"- Result: best score `{inputs['metric_value']}`\n", encoding="utf-8"
    )
    (mode_dir / "agent_stderr.txt").write_text("", encoding="utf-8")
    _write_json(
        mode_dir / "benchmark_record.json",
        {
            "mode": mode,
            "agent": AGENT,
            "agent_model": AGENT_MODEL,
            "model_source": "scenario",
            "final_container_exit_code": 0,
            "reported_validation_metric": {
                "name": "score",
                "value": inputs["metric_value"],
                "value_scope": "summary_metric",
                "summary_value_label": "best score",
            },
            "validation_metric_policy": {"expected_primary_metric": "score"},
            "workspace_delta": workspace_delta,
        },
    )


def build_neutral_result_root(root: Path) -> Path:
    """Build a deterministic SDK-neutral result root under ``root``."""

    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, mode in enumerate((WITHOUT_SKILLS_MODE, WITH_SKILLS_MODE), start=1):
        record_dir = (
            root
            / "records"
            / f"agent={AGENT}"
            / f"model={AGENT_MODEL}"
            / "workflow=default"
            / "job=demo"
            / "repeat=01"
            / f"mode={mode}"
        )
        record_dir.mkdir(parents=True, exist_ok=True)
        _populate_mode_dir(record_dir, mode)
        entries.append(
            {
                "run_id": f"run_{index:05d}",
                "mode": mode,
                "agent": AGENT,
                "agent_model": AGENT_MODEL,
                "model_source": "scenario",
                "skills_enabled": _MODE_INPUTS[mode]["skills_enabled"],
                "record_dir": str(record_dir.relative_to(root)),
            }
        )

    _write_json(
        root / "run_plan.json",
        {
            "scenario_name": "demo_scenario",
            "comparison_type": "skills",
            "entries": entries,
            "comparison_groups": [
                {
                    "id": "demo_scenario",
                    "label": "Demo",
                    "members": {WITHOUT_SKILLS_MODE: "run_00001", WITH_SKILLS_MODE: "run_00002"},
                }
            ],
        },
    )
    _write_json(root / "scenario.json", {"name": "demo_scenario", "reproducibility": {"seed": 1234}})
    (root / "console_output.log").write_text(
        "[without_skills] starting run\n[with_skills] starting run\n", encoding="utf-8"
    )
    return root
