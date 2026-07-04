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

"""Realistic-usage result-root fixture that EXERCISES the deep causal "Why".

The canonical snapshot fixture (``tests/_report_fixtures.py``) is deliberately
minimal: its usage JSON carries no cache/cost breakdown, both runs install the
same way, and each run has exactly one job span. That starves the deep causal
branches in ``benchmark_insights.why_section`` -> ``_why_slower`` /
``_why_more_tokens`` (slowdown-driver table, repeated job runs, dependency
divergence, cache-dominant token narrative, code-quality-vs-runtime), so the
existing golden never demonstrates them.

This builder produces a SECOND, richer result root where the with-skills run
realistically has, relative to the no-skills baseline:

- higher ``cache_read_input_tokens`` (the dominant token driver) plus a real
  cost/cache-creation/output breakdown,
- more ``Skill`` tool calls and more assistant turns,
- more than one successful job execution (repeated runs),
- a heavier, accelerator-capable dependency-install path (different tool +
  packages, taking far longer than the baseline install),
- a loss/optimizer rebuilt *inside* the FL loop (a "poor" code-quality signal)
  coupled with a slower runtime-after-install.

Every input is FIXED (no clocks, no randomness). The mode/record layout matches
``_report_fixtures.build_result_root`` so ``collect_benchmark_runs`` reads it
through the real loaders. Because ``collect_benchmark_runs`` (no replay) reads
``agent_activity.json`` and ``agent_usage.json`` straight from disk, this
fixture writes explicit ``event_types`` / ``tool_counts`` and a full usage
breakdown there -- that is the realistic-capture data the minimal fixture omits.
"""

from __future__ import annotations

import json
from pathlib import Path

WITHOUT_SKILLS_MODE = "without_skills"
WITH_SKILLS_MODE = "with_skills"

AGENT = "claude"
AGENT_MODEL = "default"


# Per-mode fixed inputs. Durations come straight from the fixed event
# timestamps below; tokens/usage/activity come straight from these dicts.
_MODE_INPUTS = {
    WITHOUT_SKILLS_MODE: {
        "label": "No skills baseline",
        "skills_enabled": False,
        "elapsed_seconds": 600,
        "token_count": 90000,
        "command_count": 3,
        "unique_command_count": 3,
        "metric_value": 0.7421,
        "usage": {
            "token_count": 90000,
            "input_tokens": 20000,
            "output_tokens": 8000,
            "cache_read_input_tokens": 40000,
            "cache_creation_input_tokens": 12000,
            "total_cost_usd": 0.42,
        },
        "activity": {
            "command_count": 3,
            "unique_command_count": 3,
            "event_types": {"assistant": 9, "user": 9, "system.thinking_tokens": 1},
            "tool_counts": {"Bash": 3, "Read": 4, "Edit": 2},
        },
        # (command, description, start, end, output)
        "commands": [
            (
                "python3 -m pip install -r requirements.txt",
                "Install dependencies",
                "2026-06-13T20:00:00Z",
                "2026-06-13T20:00:30Z",
                "Collecting torch\nSuccessfully installed torch-2.12.0",
            ),
            (
                "python3 job.py --num-sites 3 --num-rounds 3",
                "Run baseline simulation",
                "2026-06-13T20:00:40Z",
                "2026-06-13T20:10:00Z",
                "Finished FedAvg. aggregated best validation metric: 0.7421",
            ),
        ],
    },
    WITH_SKILLS_MODE: {
        "label": "With skills",
        "skills_enabled": True,
        "elapsed_seconds": 900,
        "token_count": 180000,
        "command_count": 6,
        "unique_command_count": 5,
        "metric_value": 0.7689,
        "usage": {
            "token_count": 180000,
            "input_tokens": 30000,
            "output_tokens": 16000,
            "cache_read_input_tokens": 110000,
            "cache_creation_input_tokens": 24000,
            "total_cost_usd": 0.95,
        },
        "activity": {
            "command_count": 6,
            "unique_command_count": 5,
            "event_types": {"assistant": 21, "user": 21, "system.thinking_tokens": 6},
            "tool_counts": {"Skill": 4, "Bash": 6, "Read": 8, "Edit": 5},
        },
        "commands": [
            (
                "uv pip install -r requirements-train.txt",
                "Install accelerator training dependencies",
                "2026-06-13T20:00:00Z",
                "2026-06-13T20:02:30Z",
                "Downloading nvidia-cublas-cu13 (520.4MB)\n"
                "Downloading nvidia-cudnn-cu13 (410.1MB)\n"
                "Downloading torch (210.0MB)\n"
                "Retrying (Retry(total=4)) after connection broken\n"
                "Successfully installed nvidia-cublas-cu13-13.1.1.3 "
                "nvidia-cudnn-cu13-9.5.0 torch-2.12.0",
            ),
            (
                "python3 job.py --num-sites 3 --num-rounds 3",
                "Run 3-site simulation (first attempt)",
                "2026-06-13T20:02:40Z",
                "2026-06-13T20:08:40Z",
                "Finished FedAvg. aggregated best validation metric: 0.7501",
            ),
            (
                "rm -rf /tmp/nvflare/workspaces/job",
                "Clear workspace before rerun",
                "2026-06-13T20:08:45Z",
                "2026-06-13T20:08:46Z",
                "",
            ),
            (
                "python3 job.py --num-sites 3 --num-rounds 3",
                "Re-run 3-site simulation after tuning",
                "2026-06-13T20:08:50Z",
                "2026-06-13T20:15:00Z",
                "Finished FedAvg. aggregated best validation metric: 0.7689",
            ),
        ],
    },
}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _command_event_lines(command: str, description: str, start: str, end: str, output: str, item_id: str) -> list[str]:
    """Codex/Claude-style command_execution event pair with FIXED timestamps."""

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


def _client_source(mode: str) -> str:
    if mode == WITH_SKILLS_MODE:
        # Loss/optimizer REBUILT inside the FL loop -> "poor" code-quality signal.
        return (
            "import nvflare.client as flare\n"
            "train_frame = load_split(args.data_dir, 'train')\n"
            "local_train = site_shard(train_frame, site_name)\n"
            "train_loader = DataLoader(local_train)\n"
            "while flare.is_running():\n"
            "    input_model = flare.receive()\n"
            "    criterion, optimizer, _ = build_loss_and_optimizer(model, local_train, args, device)\n"
            "    for epoch in range(1, args.local_epochs + 1):\n"
            "        print(f'[{site_name}] round {round_num} epoch {epoch:02d}')\n"
            "    metrics = evaluate(model, valid_loader, criterion, device)\n"
        )
    # Baseline: loss/optimizer built OUTSIDE the loop -> "good" (not "poor").
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
        "from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe\n"
        "from nvflare.recipe import SimEnv\n\n"
        "recipe = FedAvgRecipe(name='ames_fedavg', min_clients=3, num_rounds=3)\n"
        "recipe.execute(SimEnv(num_clients=3))\n"
    )


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
    # Realistic usage breakdown (cache_read dominant for with-skills) and a
    # realistic activity histogram (assistant turns, Skill calls, thinking
    # events). collect_benchmark_runs (no replay) reads these straight off disk.
    _write_json(mode_dir / "agent_usage.json", inputs["usage"])
    _write_json(mode_dir / "agent_activity.json", inputs["activity"])
    _write_json(
        mode_dir / "runtime_image.json",
        {"image": "agent-skills-benchmark:claude-baseline", "digest": "sha256:fixed-deep-why-digest"},
    )

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
    metrics_rel = "runtime_workspaces/job/server/simulate_job/metrics/metrics_summary.json"
    runtime_artifacts = [
        {
            "artifact_path": f"runtime_artifacts/{config_rel}",
            "path": config_rel,
            "source_path": "/tmp/nvflare/workspaces/job/server/simulate_job/app_server/config/config_fed_server.json",
        },
        {
            "artifact_path": f"runtime_artifacts/{metrics_rel}",
            "path": metrics_rel,
            "source_path": "/tmp/nvflare/workspaces/job/server/simulate_job/metrics/metrics_summary.json",
        },
    ]
    workspace_delta = {
        "changed_file_count": len(changed_files),
        "runtime_artifact_count": len(runtime_artifacts),
        "changed_files": changed_files,
        "final_structure_files": final_structure_files,
        "runtime_artifacts": runtime_artifacts,
    }
    _write_json(mode_dir / "workspace_delta_manifest.json", workspace_delta)

    for entry in changed_files:
        target = mode_dir / "workspace_delta" / entry["artifact_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if entry["path"].endswith("client.py"):
            target.write_text(_client_source(mode), encoding="utf-8")
        elif entry["path"].endswith("job.py"):
            target.write_text(_job_source(), encoding="utf-8")
        else:
            target.write_text("# generated model\n", encoding="utf-8")
    config_target = mode_dir / "workspace_delta" / runtime_artifacts[0]["artifact_path"]
    config_target.parent.mkdir(parents=True, exist_ok=True)
    config_target.write_text(json.dumps(_server_workflow_config(), indent=2, sort_keys=True), encoding="utf-8")
    metrics_target = mode_dir / "workspace_delta" / runtime_artifacts[1]["artifact_path"]
    metrics_target.parent.mkdir(parents=True, exist_ok=True)
    metrics_target.write_text(
        json.dumps({"final_aggregated_metrics": [{"name": "AUROC", "value": inputs["metric_value"]}]}),
        encoding="utf-8",
    )

    (mode_dir / "agent_events.jsonl").write_text(_agent_events_text(mode), encoding="utf-8")
    (mode_dir / "agent_last_message.txt").write_text(
        "- **Recipe:** `fedavg-pt` -> `FedAvgRecipe`\n"
        f"- aggregated best validation metric: `{inputs['metric_value']}`\n",
        encoding="utf-8",
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
                "name": "AUROC",
                "value": inputs["metric_value"],
                "value_scope": "fl_summary_metric",
                "summary_value_label": "aggregated best validation metric",
            },
            "validation_metric_policy": {"expected_primary_metric": "AUROC"},
            "workspace_delta": workspace_delta,
        },
    )


def build_result_root(root: Path) -> Path:
    """Build the deterministic deep-Why result root under ``root``.

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
