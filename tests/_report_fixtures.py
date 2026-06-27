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

"""Deterministic result-root fixture builder for report snapshot tests.

This builds a canonical-layout result root (the same shape the host runner
writes) with two modes (``without_skills`` / ``with_skills``) populated with a
representative-but-FIXED set of evidence artifacts. Every input is constant so
the report outputs are byte-stable across runs and machines.

The builder is intentionally reusable by later migration steps: it is the
single source of a representative result root for snapshot/golden tests.
"""

from __future__ import annotations

import json
from pathlib import Path

# Modes, kept literal so the fixture does not depend on production constants.
WITHOUT_SKILLS_MODE = "without_skills"
WITH_SKILLS_MODE = "with_skills"

# Fixed identity used across the fixture.
AGENT = "codex"
AGENT_MODEL = "default"

# Per-mode fixed inputs. Durations come straight from these values (and from the
# fixed event timestamps below), never from wall-clock time.
_MODE_INPUTS = {
    WITHOUT_SKILLS_MODE: {
        "label": "No skills baseline",
        "skills_enabled": False,
        "elapsed_seconds": 180,
        "token_count": 12000,
        "command_count": 4,
        "unique_command_count": 3,
        "metric_value": 0.7421,
        # (command, description, start, end, output)
        "commands": [
            (
                "python3 -m pip install -r requirements.txt",
                "Install dependencies",
                "2026-06-13T20:00:00Z",
                "2026-06-13T20:00:30Z",
                "Successfully installed torch-2.12.0",
            ),
            (
                "python3 job.py --num-sites 3 --num-rounds 3",
                "Run baseline simulation",
                "2026-06-13T20:00:40Z",
                "2026-06-13T20:03:00Z",
                "Finished FedAvg. aggregated best validation metric: 0.7421",
            ),
        ],
    },
    WITH_SKILLS_MODE: {
        "label": "With skills",
        "skills_enabled": True,
        "skill_name": "nvflare-convert-pytorch",
        "elapsed_seconds": 240,
        "token_count": 15000,
        "command_count": 5,
        "unique_command_count": 4,
        "metric_value": 0.7689,
        "commands": [
            (
                "uv pip install -r requirements-train.txt",
                "Install training dependencies",
                "2026-06-13T20:00:00Z",
                "2026-06-13T20:01:00Z",
                "Successfully installed nvidia-cublas-13.1.1.3 torch-2.12.0",
            ),
            (
                "python3 job.py --num-sites 3 --num-rounds 3",
                "Run 3-site simulation",
                "2026-06-13T20:01:10Z",
                "2026-06-13T20:04:00Z",
                "Finished FedAvg. aggregated best validation metric: 0.7689",
            ),
        ],
    },
}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _command_event_lines(command: str, description: str, start: str, end: str, output: str, item_id: str) -> list[str]:
    """Codex-style command_execution event pair with FIXED timestamps."""

    return [
        json.dumps(
            {
                "type": "item.started",
                "harness_timestamp": start,
                "item": {
                    "command": command,
                    "description": description,
                    "id": item_id,
                    "status": "in_progress",
                    "type": "command_execution",
                },
            }
        ),
        json.dumps(
            {
                "type": "item.completed",
                "harness_timestamp": end,
                "item": {
                    "aggregated_output": output,
                    "command": command,
                    "description": description,
                    "exit_code": 0,
                    "id": item_id,
                    "status": "completed",
                    "type": "command_execution",
                },
            }
        ),
    ]


def _agent_events_text(mode: str) -> str:
    lines: list[str] = []
    skill_name = str(_MODE_INPUTS[mode].get("skill_name") or "")
    if skill_name:
        lines.append(
            json.dumps(
                {
                    "event_type": "assistant",
                    "harness_timestamp": "2026-06-13T19:59:59Z",
                    "message": {
                        "content": [
                            {
                                "input": {"skill": skill_name},
                                "name": "Skill",
                                "type": "tool_use",
                            }
                        ]
                    },
                    "tool_kind": "Skill",
                    "type": "assistant",
                }
            )
        )
    for index, (command, description, start, end, output) in enumerate(_MODE_INPUTS[mode]["commands"], start=1):
        lines.extend(_command_event_lines(command, description, start, end, output, f"{mode}-cmd-{index}"))
    return "\n".join(lines) + "\n"


def _server_workflow_config() -> dict:
    return {
        "workflows": [
            {
                "id": "controller",
                "path": "nvflare.app_common.workflows.fedavg.FedAvg",
                "args": {"num_rounds": 3},
            }
        ]
    }


def _client_source() -> str:
    return (
        "import nvflare.client as flare\n"
        "train_frame = load_split(args.data_dir, 'train')\n"
        "local_train = site_shard(train_frame, site_name)\n"
        "train_loader = DataLoader(local_train)\n"
        "criterion, optimizer, _ = build_loss_and_optimizer(model, local_train, args, device)\n"
        "while flare.is_running():\n"
        "    input_model = flare.receive()\n"
        "    for epoch in range(1, args.local_epochs + 1):\n"
        "        print(f'[{site_name}] round {round_num} epoch {epoch:02d}')\n"
        "    metrics = evaluate(model, valid_loader, criterion, device)\n"
    )


def _job_source() -> str:
    return (
        "from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe\n\n"
        "recipe = FedAvgRecipe(name='ames_fedavg', min_clients=3, num_rounds=3)\n"
    )


def _skills_list(mode: str) -> dict:
    if not _MODE_INPUTS[mode]["skills_enabled"]:
        return {
            "installed": [],
            "reason": "skills intentionally removed for baseline run",
            "status": "skipped",
        }
    return {
        "schema_version": "1",
        "status": "ok",
        "data": {
            "available": [
                {"name": "nvflare-convert-lightning"},
                {"name": "nvflare-convert-pytorch"},
                {"name": "nvflare-diagnose-job"},
                {"name": "nvflare-orient"},
            ],
            "installed": [
                {"name": "_shared"},
                {"name": "nvflare-convert-lightning"},
                {"name": "nvflare-convert-pytorch"},
                {"name": "nvflare-diagnose-job"},
                {"name": "nvflare-orient"},
            ],
        },
    }


def _populate_mode_dir(mode_dir: Path, mode: str) -> None:
    inputs = _MODE_INPUTS[mode]
    skill_name = str(inputs.get("skill_name") or "")

    run_summary = {
        "mode": mode,
        "agent": AGENT,
        "agent_model": AGENT_MODEL,
        "model_source": "scenario",
        "elapsed_seconds": inputs["elapsed_seconds"],
        "token_count": inputs["token_count"],
        "agent_exit_code": 0,
        "final_container_exit_code": 0,
    }
    if skill_name:
        run_summary.update(
            {
                "observed_skill_name": skill_name,
                "skill_discovery": {"selected_skill": skill_name},
                "skill_name": skill_name,
            }
        )
    _write_json(mode_dir / "run_summary.json", run_summary)
    _write_json(mode_dir / "container_exit_code.json", {"exit_code": 0})
    # Fixed usage/activity. The runner replay path overwrites these from
    # agent_events.jsonl using the real adapter parsers; both produce stable
    # values because the events themselves are fixed.
    _write_json(
        mode_dir / "agent_usage.json",
        {"token_count": inputs["token_count"], "input_tokens": 0, "output_tokens": 0},
    )
    _write_json(
        mode_dir / "agent_activity.json",
        {"command_count": inputs["command_count"], "unique_command_count": inputs["unique_command_count"]},
    )
    _write_json(
        mode_dir / "runtime_image.json",
        {"image": "agent-skills-benchmark:codex-baseline", "digest": "sha256:fixed-fixture-digest"},
    )
    _write_json(mode_dir / "skills_list.json", _skills_list(mode))

    structure_root = "nvflare_jobs/ames_fedavg"
    changed_files = [
        {"artifact_path": f"changed_files/{structure_root}/client.py", "path": f"{structure_root}/client.py"},
        {"artifact_path": f"changed_files/{structure_root}/model.py", "path": f"{structure_root}/model.py"},
        {"artifact_path": f"changed_files/{structure_root}/job.py", "path": f"{structure_root}/job.py"},
    ]
    final_structure_files = [
        {"path": f"{structure_root}/client.py"},
        {"path": f"{structure_root}/model.py"},
        {"path": f"{structure_root}/job.py"},
        {"path": "download_data.py"},
    ]
    config_rel = "runtime_workspaces/job/server/simulate_job/app_server/config/config_fed_server.json"
    runtime_artifacts = [
        {
            "artifact_path": f"runtime_artifacts/{config_rel}",
            "path": config_rel,
            "source_path": "/tmp/nvflare/workspaces/job/server/simulate_job/app_server/config/config_fed_server.json",
        }
    ]
    workspace_delta = {
        "changed_file_count": len(changed_files),
        "runtime_artifact_count": len(runtime_artifacts),
        "changed_files": changed_files,
        "final_structure_files": final_structure_files,
        "runtime_artifacts": runtime_artifacts,
    }
    _write_json(mode_dir / "workspace_delta_manifest.json", workspace_delta)

    # Materialize the captured workspace_delta files the report reads on disk.
    for entry in changed_files:
        target = mode_dir / "workspace_delta" / entry["artifact_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if entry["path"].endswith("client.py"):
            target.write_text(_client_source(), encoding="utf-8")
        elif entry["path"].endswith("job.py"):
            target.write_text(_job_source(), encoding="utf-8")
        else:
            target.write_text("# generated model\n", encoding="utf-8")
    config_target = mode_dir / "workspace_delta" / runtime_artifacts[0]["artifact_path"]
    config_target.parent.mkdir(parents=True, exist_ok=True)
    config_target.write_text(json.dumps(_server_workflow_config(), indent=2, sort_keys=True), encoding="utf-8")

    (mode_dir / "agent_events.jsonl").write_text(_agent_events_text(mode), encoding="utf-8")
    (mode_dir / "agent_last_message.txt").write_text(
        "- **Recipe:** `fedavg-pt` -> `FedAvgRecipe`\n"
        f"- aggregated best validation metric: `{inputs['metric_value']}`\n",
        encoding="utf-8",
    )
    (mode_dir / "agent_stderr.txt").write_text("", encoding="utf-8")

    benchmark_record = {
        "mode": mode,
        "agent": AGENT,
        "agent_model": AGENT_MODEL,
        "model_source": "scenario",
        "final_container_exit_code": 0,
        "reported_validation_metric": {
            "name": "AUROC",
            "value": inputs["metric_value"],
            "value_scope": "fl_summary_metric",
            "summary_value_label": "aggregated best validation metric",
        },
        "validation_metric_policy": {"expected_primary_metric": "AUROC"},
        "workspace_delta": workspace_delta,
    }
    if skill_name:
        benchmark_record.update(
            {
                "observed_skill_name": skill_name,
                "skill_discovery": {"selected_skill": skill_name},
                "skill_name": skill_name,
            }
        )
    _write_json(mode_dir / "benchmark_record.json", benchmark_record)


def build_result_root(root: Path) -> Path:
    """Build the deterministic canonical-layout result root under ``root``.

    Returns ``root`` for convenience.
    """

    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, mode in enumerate((WITHOUT_SKILLS_MODE, WITH_SKILLS_MODE), start=1):
        record_dir = (
            root
            / "records"
            / f"agent={AGENT}"
            / f"model={AGENT_MODEL}"
            / "workflow=default"
            / "job=ames"
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
            "scenario_name": "ames_fedavg",
            "comparison_type": "skills",
            "entries": entries,
            "comparison_groups": [
                {
                    "id": "ames_fedavg",
                    "label": "AMES FedAvg",
                    "members": {
                        WITHOUT_SKILLS_MODE: "run_00001",
                        WITH_SKILLS_MODE: "run_00002",
                    },
                }
            ],
        },
    )
    _write_json(
        root / "scenario.json",
        {
            "name": "ames_fedavg",
            "reproducibility": {"seed": 1234},
        },
    )
    (root / "console_output.log").write_text(
        "[without_skills] starting run\n[with_skills] starting run\n",
        encoding="utf-8",
    )
    # §4.3 profile/identity descriptor — what a real NVFLARE run stamps via
    # profile_metadata.build_profile_metadata_block. Captured here so the report
    # resolves the NVFLARE plugin by explicit identity (Inversion 3: absence
    # resolves to the null plugin, not NVFLARE). Kept literal to match the
    # fixture's no-production-imports style.
    _write_json(
        root / "benchmark_profile_metadata.json",
        {
            "schema_version": 1,
            "sdk_name": "nvflare",
            "benchmark_profile_id": "nvflare",
            "report_plugin_id": "nvflare",
            "capture_spec_version": "1",
        },
    )
    return root
