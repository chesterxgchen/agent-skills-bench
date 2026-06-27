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
import stat
import sys

from benchmark.harness.reports.evidence import RunEvidence, _run_evidence_from_bundle


def _ev(run):
    """Wrap a test's per-run bundle dict as RunEvidence (idempotent)."""
    return run if isinstance(run, RunEvidence) else _run_evidence_from_bundle(run)


def _evruns(runs):
    """Wrap a {mode: bundle} dict as {mode: RunEvidence} (idempotent)."""
    return {mode: _ev(run) for mode, run in runs.items()}


def _nv_ev(run):
    """NVFLARE per-run PluginEvidence (the sidecar render helpers consume).

    Inversion 3 made absence resolve to the null plugin, so a test exercising
    NVFLARE rendering must supply the NVFLARE evidence explicitly rather than
    relying on a default.
    """
    from benchmark.harness.sdks.nvflare.plugin import NvflareReportPlugin

    return NvflareReportPlugin().collect(_ev(run))


def _nv_ctx(runs, modes=None):
    """NVFLARE ReportContext for a {mode: run} dict (explicit-plugin render)."""
    from benchmark.harness.reports.benchmark_insights import _report_context
    from benchmark.harness.sdks.nvflare.plugin import NvflareReportPlugin

    typed = _evruns(runs)
    modes = modes if modes is not None else list(typed)
    return _report_context(typed, modes, NvflareReportPlugin())


def test_skill_usage_keeps_explicit_skills_and_shared_refs_separate():
    from benchmark.harness.reports._skill_usage import (
        shared_skill_usage_display,
        skill_inspection_display,
        skill_usage_display,
    )

    events_text = "\n".join(
        [
            json.dumps(
                {
                    "message": {
                        "content": [
                            {
                                "name": "Skill",
                                "input": {"skill": "nvflare-convert-lightning"},
                            }
                        ]
                    }
                }
            ),
            json.dumps(
                {
                    "message": {
                        "content": [
                            {
                                "name": "Read",
                                "input": {"file_path": "/workspace/.claude/skills/_shared/dependency-install.md"},
                            }
                        ]
                    }
                }
            ),
            json.dumps(
                {
                    "message": {
                        "content": [
                            {
                                "name": "Read",
                                "input": {"file_path": "/workspace/.claude/skills/_shared/dependency-install.md"},
                            },
                            {
                                "name": "Read",
                                "input": {"file_path": "/workspace/.claude/skills/_shared/runtime-output-guidance.md"},
                            },
                        ]
                    }
                }
            ),
        ]
    )

    assert skill_inspection_display(events_text) == "none"
    assert skill_usage_display(events_text=events_text, skills_enabled=True) == "nvflare-convert-lightning"
    assert (
        shared_skill_usage_display(events_text)
        == "_shared/dependency-install.md; _shared/runtime-output-guidance.md"
    )


def test_skill_usage_separates_availability_from_codex_instruction_reads():
    from benchmark.harness.reports._skill_usage import (
        shared_skill_usage_display,
        skill_availability_display,
        skill_inspection_display,
        skill_usage_display,
    )

    available = {
        "status": "ok",
        "data": {
            "available": [
                {"name": "nvflare-convert-lightning"},
                {"name": "nvflare-convert-pytorch"},
                {"name": "nvflare-diagnose-job"},
                {"name": "nvflare-orient"},
            ],
            "installed": [{"name": "_shared"}],
        },
    }
    events_text = "\n".join(
        [
            json.dumps(
                {
                    "item": {
                        "type": "command_execution",
                        "command": (
                            "/bin/bash -lc \"sed -n '1,260p' "
                            "/workspace/.codex/skills/nvflare-convert-lightning/SKILL.md\""
                        ),
                    }
                }
            ),
            json.dumps(
                {
                    "item": {
                        "type": "command_execution",
                        "command": (
                            "/bin/bash -lc \"sed -n '1,220p' "
                            "/workspace/.codex/skills/_shared/dependency-install.md\""
                        ),
                    }
                }
            ),
        ]
    )

    assert (
        skill_availability_display(available, skills_enabled=True)
        == "nvflare-convert-lightning; nvflare-convert-pytorch; nvflare-diagnose-job; nvflare-orient"
    )
    assert skill_usage_display(events_text="", skills_enabled=True) == "none recorded"
    assert skill_inspection_display(events_text) == "nvflare-convert-lightning"
    assert skill_usage_display(events_text=events_text, skills_enabled=True) == "none recorded"
    assert shared_skill_usage_display(events_text) == "_shared/dependency-install.md"


def test_skill_usage_counts_deeper_codex_references_as_applied_usage():
    from benchmark.harness.reports._skill_usage import (
        shared_skill_usage_display,
        skill_inspection_display,
        skill_usage_display,
    )

    events_text = "\n".join(
        [
            json.dumps(
                {
                    "item": {
                        "type": "command_execution",
                        "command": (
                            "/bin/bash -lc \"sed -n '1,260p' "
                            "/workspace/.codex/skills/nvflare-convert-lightning/SKILL.md\""
                        ),
                    }
                }
            ),
            json.dumps(
                {
                    "item": {
                        "type": "command_execution",
                        "command": (
                            "/bin/bash -lc \"sed -n '1,220p' "
                            "/workspace/.codex/skills/nvflare-convert-pytorch/SKILL.md\""
                        ),
                    }
                }
            ),
            json.dumps(
                {
                    "message": {
                        "content": [
                            {
                                "name": "Read",
                                "input": {
                                    "file_path": (
                                        "/workspace/.codex/skills/nvflare-convert-pytorch/references/"
                                        "pytorch-client-api-conversion.md"
                                    )
                                },
                            },
                            {
                                "name": "Read",
                                "input": {"file_path": "/workspace/.codex/skills/_shared/dependency-install.md"},
                            },
                        ]
                    }
                }
            ),
        ]
    )

    assert skill_inspection_display(events_text) == "nvflare-convert-lightning; nvflare-convert-pytorch"
    assert skill_usage_display(events_text=events_text, skills_enabled=True) == "nvflare-convert-pytorch"
    assert shared_skill_usage_display(events_text) == "_shared/dependency-install.md"


def test_skill_usage_ignores_shared_observed_skill_name_fallback():
    from benchmark.harness.reports._skill_usage import skill_usage_display

    assert (
        skill_usage_display(events_text="", observed_skill_name="_shared", skills_enabled=True)
        == "none recorded"
    )


def test_benchmark_insights_explains_docker_image_failures(tmp_path):
    from benchmark.harness.modes import NO_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import (
        collect_benchmark_runs,
        failure_root_cause,
        human_readable_status,
    )

    mode_dir = tmp_path / NO_SKILLS_MODE
    mode_dir.mkdir()
    (mode_dir / "container_exit_code.json").write_text(json.dumps({"exit_code": 1}) + "\n", encoding="utf-8")
    (tmp_path / "console_output.log").write_text(
        "[without_skills] Unable to find image 'agent-skills-benchmark:codex-baseline' locally\n"
        "[without_skills] docker: Error response from daemon: pull access denied for agent-skills-benchmark\n",
        encoding="utf-8",
    )

    run = collect_benchmark_runs(tmp_path)[NO_SKILLS_MODE]

    assert run["available"] is True
    assert "Docker image unavailable" in failure_root_cause(_ev(run))
    assert "container exit 1" in human_readable_status(_ev(run))


def test_benchmark_insights_scopes_shared_console_evidence_by_mode(tmp_path):
    from benchmark.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import collect_benchmark_runs, dependency_reference_notes

    for mode in (NO_SKILLS_MODE, WITH_SKILLS_MODE):
        records_dir = tmp_path / mode / "records"
        records_dir.mkdir(parents=True)
        (records_dir / f"{mode}_record.json").write_text(
            json.dumps(
                {
                    "source_input_delta": {"final_files": [{"path": "requirements-train.txt"}]},
                    "workspace_delta": {
                        "workspace_added_files": (
                            [{"path": "requirements-federated.txt"}] if mode == NO_SKILLS_MODE else []
                        )
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
    (tmp_path / "console_output.log").write_text(
        "[without_skills] python3 -m pip install -r requirements-federated.txt failed\n"
        "[with_skills] completed without dependency install errors\n",
        encoding="utf-8",
    )

    runs = collect_benchmark_runs(tmp_path)

    assert dependency_reference_notes(_ev(runs[NO_SKILLS_MODE])) == [
        "`requirements-federated.txt` provenance: agent-generated file.",
    ]
    assert dependency_reference_notes(_ev(runs[WITH_SKILLS_MODE])) == []


def test_benchmark_insights_caps_agent_events_text(tmp_path, monkeypatch):
    from benchmark.harness.modes import NO_SKILLS_MODE
    from benchmark.harness.reports import _loader

    mode_dir = tmp_path / NO_SKILLS_MODE
    mode_dir.mkdir()
    (mode_dir / "agent_events.jsonl").write_text("0123456789", encoding="utf-8")
    monkeypatch.setattr(_loader, "MAX_AGENT_EVENTS_TEXT_BYTES", 8)

    run = _loader.collect_benchmark_runs(tmp_path)[NO_SKILLS_MODE]

    assert run["agent_events_text"] == "01234567"


def test_reports_resolve_unspecified_agent_model_from_agent_events(tmp_path):
    from benchmark.harness.common import write_json
    from benchmark.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import collect_benchmark_runs
    from benchmark.harness.reports.metrics_report import collect_runs
    from benchmark.harness.scenario_summaries import write_scenario_summaries

    record_dir = tmp_path / "records" / "agent=claude" / "model=unspecified_default" / "mode=without_skills"
    record_dir.mkdir(parents=True)
    events_text = "\n".join(
        [
            json.dumps({"event_type": "system.init", "model": "claude-opus-4-8[1m]"}),
            json.dumps({"type": "assistant", "message": {"model": "claude-opus-4-8"}}),
        ]
    )
    (record_dir / "agent_events.jsonl").write_text(events_text + "\n", encoding="utf-8")
    write_json(
        record_dir / "run_summary.json",
        {
            "agent": "claude",
            "agent_model": "unspecified_default",
            "model_source": "adapter_default",
            "agent_process_passed": True,
            "final_container_exit_code": 0,
        },
    )
    write_json(record_dir / "benchmark_record.json", {"agent_model": "unspecified_default"})
    write_json(record_dir / "container_exit_code.json", {"exit_code": 0})
    write_json(
        tmp_path / "run_plan.json",
        {
            "entries": [
                {
                    "run_id": "run_00001",
                    "mode": NO_SKILLS_MODE,
                    "agent": "claude",
                    "agent_model": "unspecified_default",
                    "model_source": "adapter_default",
                    "record_dir": str(record_dir.relative_to(tmp_path)),
                },
                {
                    "run_id": "run_00002",
                    "mode": WITH_SKILLS_MODE,
                    "agent": "claude",
                    "agent_model": "explicit-model",
                    "model_source": "scenario",
                    "record_dir": "missing",
                },
            ]
        },
    )

    insight_runs = collect_benchmark_runs(tmp_path)
    metrics_rows = {row["mode"]: row for row in collect_runs(tmp_path)}
    scenario_summary = write_scenario_summaries(tmp_path, {"run_00001": 0})

    assert insight_runs[NO_SKILLS_MODE]["agent_model"] == "claude-opus-4-8"
    assert insight_runs[NO_SKILLS_MODE]["model_source"] == "agent_events"
    assert metrics_rows[NO_SKILLS_MODE]["agent_model"] == "claude-opus-4-8"
    assert metrics_rows[NO_SKILLS_MODE]["model_source"] == "agent_events"
    assert scenario_summary["runs"][0]["agent_model"] == "claude-opus-4-8"
    assert scenario_summary["runs"][0]["model_source"] == "agent_events"
    assert insight_runs[WITH_SKILLS_MODE]["agent_model"] == "explicit-model"


def test_benchmark_reports_read_canonical_record_layout(tmp_path):
    from benchmark.harness.common import write_json
    from benchmark.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from benchmark.harness.reports import metrics_report
    from benchmark.harness.reports.benchmark_insights import benchmark_report, collect_benchmark_runs

    entries = []
    record_dirs = {}
    for index, mode in enumerate((NO_SKILLS_MODE, WITH_SKILLS_MODE), start=1):
        record_dir = (
            tmp_path
            / "records"
            / "agent=codex"
            / "model=default"
            / "workflow=default"
            / "job=ames"
            / "repeat=01"
            / f"mode={mode}"
        )
        record_dir.mkdir(parents=True)
        record_dirs[mode] = record_dir
        entries.append(
            {
                "run_id": f"run_{index:05d}",
                "mode": mode,
                "agent": "codex",
                "agent_model": "default",
                "model_source": "scenario",
                "scenario_name": "pair codex ames-lightning",
                "job_name": "ames-lightning",
                "job_slug": "ames_lightning",
                "job_path": "/tmp/jobs/ames-lightning",
                "record_dir": str(record_dir.relative_to(tmp_path)),
            }
        )
        run_summary = {
            "mode": mode,
            "elapsed_seconds": 10 + index,
            "token_count": 100 + index,
            "agent_exit_code": 0,
            "final_container_exit_code": 0,
        }
        if mode == WITH_SKILLS_MODE:
            run_summary["observed_skill_name"] = "nvflare-convert-pytorch"
            (record_dir / "agent_events.jsonl").write_text(
                json.dumps(
                    {
                        "message": {
                            "content": [
                                {
                                    "input": {"skill": "nvflare-convert-pytorch"},
                                    "name": "Skill",
                                    "type": "tool_use",
                                }
                            ]
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        write_json(
            record_dir / "run_summary.json",
            run_summary,
        )
        write_json(record_dir / "container_exit_code.json", {"exit_code": 0})
        write_json(record_dir / "agent_activity.json", {"command_count": index})
        write_json(
            record_dir / "benchmark_record.json",
            {
                "mode": mode,
                "reported_validation_metric": {"name": "AUROC", "value": 0.7 + index / 100},
            },
        )
    write_json(tmp_path / "run_plan.json", {"entries": entries})

    runs = collect_benchmark_runs(tmp_path)
    assert runs[NO_SKILLS_MODE]["available"] is True
    assert runs[NO_SKILLS_MODE]["agent"] == "codex"
    assert runs[NO_SKILLS_MODE]["agent_model"] == "default"
    assert runs[NO_SKILLS_MODE]["job_name"] == "ames-lightning"
    assert runs[WITH_SKILLS_MODE]["record"]["reported_validation_metric"]["name"] == "AUROC"
    insights = benchmark_report(tmp_path, runs)
    assert "### Benchmark Target" in insights
    assert "| ames-lightning | Lightning target | pair codex ames-lightning | /tmp/jobs/ames-lightning |" in insights
    assert "| Run | Job | Framework | Agent/model | Algorithm/workflow |" in insights
    assert "| No skills baseline | ames-lightning | Lightning target | agent=codex, model=default |" in insights
    assert (
        "| With skills | ames-lightning | Lightning target | agent=codex, model=default |"
        in insights
    )
    assert "### Skill Evidence" in insights
    assert "| Run | Skills available | Skills inspected | Skills applied/used | Shared refs read |" in insights
    assert "| No skills baseline | not enabled | none | none | none |" in insights
    assert "| With skills | not recorded | none | nvflare-convert-pytorch | none |" in insights
    assert "## Run Identity" in insights
    assert "| No skills baseline | codex | default | scenario | without_skills |" in insights
    assert "## Cost And Work Comparison" in insights
    assert "| Dependency install seconds |" in insights
    assert "| Run | Elapsed seconds | Tokens | Commands |" not in insights

    metrics_report.write_reports(tmp_path, "Synthetic Metrics")

    assert (tmp_path / "metrics_report.json").is_file()
    metrics_markdown = (tmp_path / "metrics_report.md").read_text(encoding="utf-8")
    assert "Metrics (AUROC)" in metrics_markdown
    assert "| No skills baseline | codex | default |" in metrics_markdown
    metric_rows = metrics_report.collect_runs(tmp_path)
    insight_runs = metrics_report.runs_by_mode_for_insights(tmp_path, metric_rows)
    assert insight_runs[WITH_SKILLS_MODE].mode_dir == record_dirs[WITH_SKILLS_MODE]
    metrics_json = json.loads((tmp_path / "metrics_report.json").read_text(encoding="utf-8"))
    assert abs(metrics_json["comparison"]["validation_metric_AUROC_with_skills_minus_without_skills"] - 0.01) < 1e-12


def test_metrics_report_preserves_run_plan_identity_for_code_quality_scoring(tmp_path):
    from _report_fixtures import build_result_root

    from benchmark.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from benchmark.harness.reports import benchmark_insights, metrics_report

    root = build_result_root(tmp_path / "result_root")
    run_plan_path = root / "run_plan.json"
    run_plan = json.loads(run_plan_path.read_text(encoding="utf-8"))
    run_plan["scenario_name"] = "pair codex ames-lightning"
    for entry in run_plan["entries"]:
        entry["scenario_name"] = "pair codex ames-lightning"
        entry["job_name"] = "ames-lightning"
        entry["job_slug"] = "ames_lightning"
        entry["job_path"] = "/tmp/jobs/ames-lightning"
    run_plan_path.write_text(json.dumps(run_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    insight_runs = _evruns(benchmark_insights.collect_benchmark_runs(root))
    insight_ctx = _nv_ctx(insight_runs, [NO_SKILLS_MODE, WITH_SKILLS_MODE])
    metric_rows = metrics_report.collect_runs(root)
    metric_runs = metrics_report.runs_by_mode_for_insights(root, metric_rows)
    metric_ctx = metrics_report._insights_context(root, metric_runs)

    assert metric_runs[NO_SKILLS_MODE].raw["job_name"] == "ames-lightning"
    assert metric_ctx.code_quality(NO_SKILLS_MODE).score == insight_ctx.code_quality(NO_SKILLS_MODE).score


def test_benchmark_target_infers_plain_pytorch_framework_from_captured_evidence(tmp_path):
    from benchmark.harness.common import write_json
    from benchmark.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import benchmark_report, collect_benchmark_runs

    entries = []
    for index, mode in enumerate((NO_SKILLS_MODE, WITH_SKILLS_MODE), start=1):
        record_dir = tmp_path / "records" / "agent=claude" / "model=default" / "job=ames" / f"mode={mode}"
        record_dir.mkdir(parents=True)
        entries.append(
            {
                "run_id": f"run_{index:05d}",
                "mode": mode,
                "agent": "claude",
                "agent_model": "default",
                "scenario_name": "pair claude ames",
                "job_name": "ames",
                "job_slug": "ames",
                "job_path": "/tmp/jobs/ames",
                "record_dir": str(record_dir.relative_to(tmp_path)),
            }
        )
        write_json(
            record_dir / "run_summary.json",
            {
                "mode": mode,
                "elapsed_seconds": 10,
                "token_count": 100,
                "agent_exit_code": 0,
                "final_container_exit_code": 0,
            },
        )
        write_json(record_dir / "container_exit_code.json", {"exit_code": 0})
        write_json(record_dir / "benchmark_record.json", {"mode": mode})
        (record_dir / "agent_last_message.txt").write_text(
            "Converted the plain-PyTorch AMES training code into an NVFLARE job.\n",
            encoding="utf-8",
        )
        if mode == WITH_SKILLS_MODE:
            (record_dir / "agent_events.jsonl").write_text(
                json.dumps({"message": {"content": [{"name": "Skill", "input": {"skill": "nvflare-convert-pytorch"}}]}})
                + "\n",
                encoding="utf-8",
            )
    write_json(tmp_path / "run_plan.json", {"entries": entries})

    report = benchmark_report(tmp_path, collect_benchmark_runs(tmp_path))

    assert "| ames | PyTorch target | pair claude ames | /tmp/jobs/ames |" in report
    assert "| No skills baseline | ames | PyTorch target | agent=claude, model=default |" in report
    assert "| With skills | not recorded | none | nvflare-convert-pytorch | none |" in report


def test_framework_inference_ignores_skill_usage_names(tmp_path):
    from benchmark.harness.common import write_json
    from benchmark.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import benchmark_report, collect_benchmark_runs

    entries = []
    for index, mode in enumerate((NO_SKILLS_MODE, WITH_SKILLS_MODE), start=1):
        record_dir = tmp_path / "records" / "agent=codex" / "model=default" / "job=ames" / f"mode={mode}"
        record_dir.mkdir(parents=True)
        entries.append(
            {
                "run_id": f"run_{index:05d}",
                "mode": mode,
                "agent": "codex",
                "agent_model": "default",
                "job_name": "ames",
                "job_slug": "ames",
                "job_path": "/tmp/jobs/ames",
                "record_dir": str(record_dir.relative_to(tmp_path)),
            }
        )
        write_json(
            record_dir / "run_summary.json",
            {
                "mode": mode,
                "elapsed_seconds": 10,
                "token_count": 100,
                "agent_exit_code": 0,
                "final_container_exit_code": 0,
            },
        )
        write_json(record_dir / "container_exit_code.json", {"exit_code": 0})
        write_json(record_dir / "benchmark_record.json", {"mode": mode})
        (record_dir / "agent_last_message.txt").write_text(
            "Ran nvflare agent inspect: detected plain PyTorch, not converted.\n",
            encoding="utf-8",
        )
        if mode == WITH_SKILLS_MODE:
            (record_dir / "agent_events.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "item": {
                                    "type": "command_execution",
                                    "command": (
                                        "/bin/bash -lc \"sed -n '1,260p' "
                                        "/workspace/.codex/skills/nvflare-convert-lightning/SKILL.md\""
                                    ),
                                }
                            }
                        ),
                        json.dumps(
                            {
                                "item": {
                                    "type": "command_execution",
                                    "command": (
                                        "/bin/bash -lc \"sed -n '1,260p' "
                                        "/workspace/.codex/skills/nvflare-convert-pytorch/SKILL.md\""
                                    ),
                                }
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
    write_json(tmp_path / "run_plan.json", {"entries": entries})

    report = benchmark_report(tmp_path, collect_benchmark_runs(tmp_path))

    assert "| ames | PyTorch target | NA | /tmp/jobs/ames |" in report
    assert "Lightning target" not in report
    assert (
        "| With skills | not recorded | nvflare-convert-lightning; nvflare-convert-pytorch | none recorded | none |"
        in report
    )


def test_common_dl_metric_aliases_are_recognized_and_unknown_names_kept_verbatim():
    """Characterization (pins the metric-vocabulary contract after the de-hardcode +
    restore churn): the harness keeps a small RECOGNITION vocabulary of common DL metrics
    with their spelling variants -- a run logging ``AUC``/``acc`` is recognized as the
    job's AUROC/accuracy. Which metric a job REQUIRES is never constrained: an unknown
    (job-specific) metric name is normalized structurally and kept as-is, never aliased.
    """

    from benchmark.harness.quality_signals import canonical_metric_name, metric_mentioned, metric_signal

    # Common-DL spelling variants fold to a canonical name (recognition aid).
    assert canonical_metric_name("auc") == "AUROC"
    assert canonical_metric_name("val_auroc") == "AUROC"
    assert canonical_metric_name("acc") == "accuracy"
    assert canonical_metric_name("val_accuracy") == "accuracy"
    assert canonical_metric_name("validation_loss") == "loss"
    assert canonical_metric_name("val_loss") == "loss"
    # A job-specific / unrecognized metric is NOT aliased -- kept verbatim (structural only).
    assert canonical_metric_name("dice") == "dice"
    # Detection: the job declares AUROC, the agent logs the `AUC` variant -> still matched.
    assert metric_mentioned("AUROC", "Final AUC: 0.81")
    signal = metric_signal(None, "AUROC is the primary metric.", "Final AUC: 0.81")
    reported = signal["reported_validation_metric"]
    assert reported["name"] == "AUROC" and reported["value"] == 0.81


def test_fl_algorithm_section_reads_captured_server_workflow_config(tmp_path):
    from benchmark.harness.modes import WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import fl_algorithm_summary
    from benchmark.harness.reports.evidence import SCHEMA_VERSION, ComparisonEvidence
    from benchmark.harness.sdks.nvflare.plugin import NvflareReportPlugin

    mode_dir = tmp_path / WITH_SKILLS_MODE
    config_path = (
        mode_dir
        / "workspace_delta"
        / "runtime_artifacts"
        / "runtime_workspaces"
        / "job"
        / "server"
        / "simulate_job"
        / "app_server"
        / "config"
        / "config_fed_server.json"
    )
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "workflows": [
                    {
                        "id": "controller",
                        "path": "nvflare.app_common.workflows.scaffold.Scaffold",
                        "args": {"num_rounds": 3},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    run = {
        "available": True,
        "label": "With skills",
        "mode_dir": mode_dir,
        "agent_last_message": "- **Recipe:** `scaffold-pt` -> `ScaffoldRecipe`",
        "workspace_delta": {
            "runtime_artifacts": [
                {
                    "artifact_path": (
                        "runtime_artifacts/runtime_workspaces/job/server/simulate_job/app_server/config/"
                        "config_fed_server.json"
                    ),
                    "path": "runtime_workspaces/job/server/simulate_job/app_server/config/config_fed_server.json",
                }
            ]
        },
    }
    runs = {WITH_SKILLS_MODE: run}

    assert (
        fl_algorithm_summary(_evruns(runs), [WITH_SKILLS_MODE], _nv_ctx(runs, [WITH_SKILLS_MODE]))
        == "With skills: SCAFFOLD (3 rounds)"
    )
    # The FL Algorithm/Workflow section is plugin-contributed (E1b): NVFLARE builds it
    # from its own AlgorithmSignal, anchored after the generic Executive Summary block.
    ctx = _nv_ctx(runs, [WITH_SKILLS_MODE])
    cmp = ComparisonEvidence(
        schema_version=SCHEMA_VERSION, runs=_evruns(runs), modes=[WITH_SKILLS_MODE], sdk_metadata={}
    )
    fl_sections = NvflareReportPlugin().sections(cmp, ctx.evidence)
    assert len(fl_sections) == 1
    assert fl_sections[0].anchor == "exec_summary" and fl_sections[0].placement == "after"
    section = f"{fl_sections[0].title}\n\n{fl_sections[0].body}"
    assert "## FL Algorithm / Workflow" in section
    assert "| With skills | SCAFFOLD | scaffold-pt | 3 |" in section
    assert "nvflare.app_common.workflows.scaffold.Scaffold" in section


def test_nvflare_conversion_quality_rows_merge_into_generated_code_quality(tmp_path):
    from benchmark.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import generated_code_quality_section
    from benchmark.harness.reports.evidence import SCHEMA_VERSION, ComparisonEvidence
    from benchmark.harness.sdks.nvflare.plugin import NvflareReportPlugin

    def write_source(mode: str, rel_path: str, source: str) -> dict:
        mode_dir = tmp_path / mode
        source_path = mode_dir / "workspace_delta" / "changed_files" / rel_path
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(source, encoding="utf-8")
        return {
            "artifact_path": f"changed_files/{rel_path}",
            "path": rel_path,
        }

    def write_round_metrics(mode: str, values: list[float]) -> dict:
        metrics_path = (
            tmp_path
            / mode
            / "workspace_delta"
            / "runtime_artifacts"
            / "runtime_workspaces"
            / "ames-smiles-fedavg"
            / "server"
            / "simulate_job"
            / "metrics"
            / "round_metrics.jsonl"
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            "\n".join(
                json.dumps({"round": index, "aggregated_metrics": [{"name": "val_auroc", "value": value}]})
                for index, value in enumerate(values)
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "artifact_path": (
                "runtime_artifacts/runtime_workspaces/ames-smiles-fedavg/server/simulate_job/"
                "metrics/round_metrics.jsonl"
            ),
            "path": "runtime_workspaces/ames-smiles-fedavg/server/simulate_job/metrics/round_metrics.jsonl",
        }

    no_skill_files = [
        write_source(
            NO_SKILLS_MODE,
            "fl_client.py",
            """
import nvflare.client as flare
from nvflare.client import FLModel

def site_partition(frame, index, num_clients, seed):
    shuffled = frame.sample(frac=1.0, random_state=seed)
    return shuffled.iloc[index::num_clients].reset_index(drop=True)

while flare.is_running():
    model = flare.receive()
    pos_weight = positive_class_weight(train_frame["label"].to_numpy(dtype=np.float32))
    auroc = binary_auroc(predictions, labels)
    flare.send(FLModel(params=model.state_dict(), metrics={"AUROC": auroc}))
""",
        ),
        write_source(
            NO_SKILLS_MODE,
            "train.py",
            """
train_args = quote_args(["--data-dir", "data"])
recipe = FedAvgRecipe(
    name="ames",
    launch_external_process=True,
    command=f"{sys.executable} -u",
    key_metric="auroc",
)
recipe.job.add_file_to_clients(str(args.data_dir), dest_dir="data")
""",
        ),
    ]
    with_skill_files = [
        write_source(
            WITH_SKILLS_MODE,
            "client.py",
            """
import nvflare.client.lightning as flare
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC

def partition(frame):
    return frame.iloc[self.site_index :: self.num_clients].reset_index(drop=True)

trainer = pl.Trainer(max_epochs=3)
flare.patch(trainer)
while flare.is_running():
    trainer.fit(model, datamodule=datamodule)
""",
        ),
        write_source(
            WITH_SKILLS_MODE,
            "job.py",
            """
MODEL_ARGS = {"pos_weight": 0.8750402576489533}
train_args = [f"--data-dir {args.data_dir.resolve()}", "--pos-weight", "0.8750402576489533"]
recipe = FedAvgRecipe(
    name="ames",
    launch_external_process=False,
    server_expected_format=ExchangeFormat.PYTORCH,
    params_transfer_type=TransferType.FULL,
    key_metric="val_auroc",
)
""",
        ),
    ]
    no_metrics = write_round_metrics(NO_SKILLS_MODE, [0.7305, 0.7449, 0.7573])
    with_metrics = write_round_metrics(WITH_SKILLS_MODE, [0.486, 0.486, 0.486])
    runs = {
        NO_SKILLS_MODE: _ev(
            {
                "available": True,
                "job_name": "ames-lightning",
                "job_slug": "ames_lightning",
                "label": "No skills baseline",
                "mode": NO_SKILLS_MODE,
                "mode_dir": tmp_path / NO_SKILLS_MODE,
                "validation_metric": {"name": "AUROC", "value": 0.7573},
                "workspace_delta": {
                    "changed_files": no_skill_files,
                    "runtime_artifacts": [no_metrics],
                },
            }
        ),
        WITH_SKILLS_MODE: _ev(
            {
                "available": True,
                "job_name": "ames-lightning",
                "job_slug": "ames_lightning",
                "label": "With skills",
                "mode": WITH_SKILLS_MODE,
                "mode_dir": tmp_path / WITH_SKILLS_MODE,
                "validation_metric": {"name": "AUROC", "value": 0.486},
                "workspace_delta": {
                    "changed_files": with_skill_files,
                    "runtime_artifacts": [with_metrics],
                },
            }
        ),
    }
    modes = [NO_SKILLS_MODE, WITH_SKILLS_MODE]
    cmp = ComparisonEvidence(schema_version=SCHEMA_VERSION, runs=runs, modes=modes, sdk_metadata={})

    plugin = NvflareReportPlugin()
    sections = plugin.sections(cmp, {mode: plugin.collect(run) for mode, run in runs.items()})
    section = generated_code_quality_section(
        runs,
        modes,
        _nv_ctx(runs, modes),
    )

    assert all(section.id != "nvflare_conversion_quality" for section in sections)
    assert "## Generated Code Quality Signals" in section
    assert "## NVFLARE Conversion Quality Comparison" not in section
    assert "Conversion: client training/control path" in section
    assert "caution: manual Client API loop" in section
    assert "good: Lightning Client API patch" in section
    assert "Conversion: site data partitioning" in section
    assert "good: seeded shuffled site partition" in section
    assert "bad: deterministic stride partition without shuffle" in section
    assert "Conversion: loss weighting (`pos_weight`)" in section
    assert "good: per-site loss weight from local training partition" in section
    assert "bad: fixed/global `pos_weight=0.8750402576489533` passed to clients" in section
    assert "Conversion: data packaging/path" in section
    assert "good: packages dataset into client app" in section
    assert "bad: passes absolute workspace data path to clients" in section
    assert "Conversion: client execution/model exchange" in section
    assert "good: external client process runner" in section
    assert "good: in-process Client API executor" in section
    assert "Conversion: round metric progression" in section
    assert "good: AUROC 0.7305 -> 0.7449 -> 0.7573" in section
    assert "bad: AUROC 0.4860 -> 0.4860 -> 0.4860 (flat)" in section


def test_nvflare_conversion_quality_prefers_local_client_pos_weight_over_server_default(tmp_path):
    from benchmark.harness.modes import WITH_SKILLS_MODE
    from benchmark.harness.sdks.nvflare._logic import conversion_quality_profile, conversion_quality_score

    def write_source(rel_path: str, source: str) -> dict:
        source_path = tmp_path / WITH_SKILLS_MODE / "workspace_delta" / "changed_files" / rel_path
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(source, encoding="utf-8")
        return {"artifact_path": f"changed_files/{rel_path}", "path": rel_path}

    changed_files = [
        write_source(
            "train.py",
            """
def stratified_partition_frame(frame, site_index, num_sites, seed):
    selected_parts = []
    for _, group in frame.groupby("label", sort=True):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        selected_parts.append(frame.loc[np.array_split(indices, num_sites)[site_index]])
    return pd.concat(selected_parts).sample(frac=1.0, random_state=seed + site_index)

class AmesDataModule:
    def setup(self, stage=None):
        train_frame = stratified_partition_frame(train_frame, self.site_index, self.num_sites, self.partition_seed)
        pos_count = float(train_frame["label"].sum())
        neg_count = float(len(train_frame) - pos_count)
        self.pos_weight = neg_count / max(pos_count, 1.0)
""",
        ),
        write_source(
            "client.py",
            """
datamodule.setup("fit")
model = LitSmilesCNN(pos_weight=datamodule.pos_weight)
""",
        ),
        write_source(
            "job.py",
            """
MODEL_ARGS = {"pos_weight": 1.0}
""",
        ),
    ]
    run = {
        "available": True,
        "mode": WITH_SKILLS_MODE,
        "mode_dir": tmp_path / WITH_SKILLS_MODE,
        "workspace_delta": {"changed_files": changed_files},
    }

    profile = conversion_quality_profile(run)

    assert profile["partitioning"] == "stratified seeded site partition"
    assert conversion_quality_score("partitioning", profile["partitioning"], run) == "good"
    assert profile["class_weighting"] == "per-site loss weight from local training partition"
    assert conversion_quality_score("class_weighting", profile["class_weighting"], run) == "good"


def test_fl_algorithm_prefers_training_workflow_over_initialization(tmp_path):
    from benchmark.harness.modes import WITH_SKILLS_MODE
    from benchmark.harness.sdks.nvflare._logic import fl_algorithm_info

    mode_dir = tmp_path / WITH_SKILLS_MODE
    config_path = (
        mode_dir
        / "workspace_delta"
        / "runtime_artifacts"
        / "runtime_workspaces"
        / "job"
        / "server"
        / "simulate_job"
        / "app_server"
        / "config"
        / "config_fed_server.json"
    )
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "workflows": [
                    {
                        "id": "init",
                        "path": "nvflare.app_common.workflows.initialize_global_weights.InitializeGlobalWeights",
                        "args": {},
                    },
                    {
                        "id": "train",
                        "path": "nvflare.app_common.workflows.scatter_and_gather.ScatterAndGather",
                        "args": {"num_rounds": 5, "train_task_name": "train"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    run = {
        "available": True,
        "mode_dir": mode_dir,
        "workspace_delta": {
            "runtime_artifacts": [
                {
                    "artifact_path": (
                        "runtime_artifacts/runtime_workspaces/job/server/simulate_job/app_server/config/"
                        "config_fed_server.json"
                    ),
                    "path": "runtime_workspaces/job/server/simulate_job/app_server/config/config_fed_server.json",
                }
            ]
        },
    }

    info = fl_algorithm_info(run)

    assert info["algorithm"] == "ScatterAndGather"
    assert info["num_rounds"] == 5
    assert info["workflow_id"] == "train"


def test_fl_algorithm_prefers_final_runtime_config_over_probe_config(tmp_path):
    from benchmark.harness.modes import WITH_SKILLS_MODE
    from benchmark.harness.sdks.nvflare._logic import fl_algorithm_info

    mode_dir = tmp_path / WITH_SKILLS_MODE
    final_config = (
        mode_dir
        / "workspace_delta"
        / "runtime_artifacts"
        / "runtime_workspaces"
        / "ames-smiles-fedavg"
        / "ames-smiles-fedavg"
        / "server"
        / "simulate_job"
        / "app_server"
        / "config"
        / "config_fed_server.json"
    )
    probe_config = (
        mode_dir
        / "workspace_delta"
        / "runtime_artifacts"
        / "runtime_workspaces"
        / "ames-smiles-fedavg-inproc-probe"
        / "ames-smiles-fedavg-inproc-probe"
        / "server"
        / "simulate_job"
        / "app_server"
        / "config"
        / "config_fed_server.json"
    )
    for path, rounds in ((final_config, 3), (probe_config, 1)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "workflows": [
                        {
                            "id": "controller",
                            "path": "nvflare.app_common.workflows.fedavg.FedAvg",
                            "args": {"num_rounds": rounds},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
    run = {
        "available": True,
        "mode": WITH_SKILLS_MODE,
        "mode_dir": mode_dir,
        "workspace_delta": {
            "runtime_artifacts": [
                {
                    "artifact_path": (
                        "runtime_artifacts/runtime_workspaces/ames-smiles-fedavg-inproc-probe/"
                        "ames-smiles-fedavg-inproc-probe/server/simulate_job/app_server/config/"
                        "config_fed_server.json"
                    ),
                    "path": (
                        "runtime_workspaces/ames-smiles-fedavg-inproc-probe/"
                        "ames-smiles-fedavg-inproc-probe/server/simulate_job/app_server/config/"
                        "config_fed_server.json"
                    ),
                },
                {
                    "artifact_path": (
                        "runtime_artifacts/runtime_workspaces/ames-smiles-fedavg/ames-smiles-fedavg/"
                        "server/simulate_job/app_server/config/config_fed_server.json"
                    ),
                    "path": (
                        "runtime_workspaces/ames-smiles-fedavg/ames-smiles-fedavg/server/simulate_job/"
                        "app_server/config/config_fed_server.json"
                    ),
                },
            ]
        },
    }

    info = fl_algorithm_info(run)

    assert info["algorithm"] == "FedAvg"
    assert info["num_rounds"] == 3


def test_fl_algorithm_recipe_prefers_generated_source_over_recipe_list_catalog(tmp_path):
    from benchmark.harness.modes import WITH_SKILLS_MODE
    from benchmark.harness.sdks.nvflare._logic import fl_algorithm_info

    mode_dir = tmp_path / WITH_SKILLS_MODE
    config_path = (
        mode_dir
        / "workspace_delta"
        / "runtime_artifacts"
        / "runtime_workspaces"
        / "job"
        / "server"
        / "simulate_job"
        / "app_server"
        / "config"
        / "config_fed_server.json"
    )
    job_path = mode_dir / "workspace_delta" / "changed_files" / "job.py"
    config_path.parent.mkdir(parents=True)
    job_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "workflows": [
                    {
                        "id": "controller",
                        "path": "nvflare.app_common.workflows.fedavg.FedAvg",
                        "args": {"num_rounds": 3},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    job_path.write_text(
        "from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe\n\n"
        "recipe = FedAvgRecipe(name='job', min_clients=3, num_rounds=3)\n",
        encoding="utf-8",
    )
    run = {
        "available": True,
        "mode_dir": mode_dir,
        "agent_last_message": ("Recipe: `cyclic-pt`\n" '{"data": [{"name": "cyclic-pt"}, {"name": "fedavg-pt"}]}'),
        "workspace_delta": {
            "changed_files": [{"artifact_path": "changed_files/job.py", "path": "job.py"}],
            "runtime_artifacts": [
                {
                    "artifact_path": (
                        "runtime_artifacts/runtime_workspaces/job/server/simulate_job/app_server/config/"
                        "config_fed_server.json"
                    ),
                    "path": "runtime_workspaces/job/server/simulate_job/app_server/config/config_fed_server.json",
                }
            ],
        },
    }

    info = fl_algorithm_info(run)

    assert info["algorithm"] == "FedAvg"
    assert info["recipe"] == "fedavg-pt"
    assert "recipe fedavg-pt" in info["evidence"]


def test_fl_algorithm_recipe_keeps_explicit_selection_for_shared_recipe_source(tmp_path):
    from benchmark.harness.modes import WITH_SKILLS_MODE
    from benchmark.harness.sdks.nvflare._logic import fl_algorithm_info

    mode_dir = tmp_path / WITH_SKILLS_MODE
    config_path = (
        mode_dir
        / "workspace_delta"
        / "runtime_artifacts"
        / "runtime_workspaces"
        / "job"
        / "server"
        / "simulate_job"
        / "app_server"
        / "config"
        / "config_fed_server.json"
    )
    job_path = mode_dir / "workspace_delta" / "changed_files" / "job.py"
    config_path.parent.mkdir(parents=True)
    job_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "workflows": [
                    {
                        "id": "controller",
                        "path": "nvflare.app_common.workflows.fedavg.FedAvg",
                        "args": {"num_rounds": 3},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    job_path.write_text(
        "from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe\n\n"
        "recipe = FedAvgRecipe(name='job', min_clients=3, num_rounds=3)\n",
        encoding="utf-8",
    )
    run = {
        "available": True,
        "mode_dir": mode_dir,
        "agent_last_message": "Selected recipe: `fedprox-pt`.",
        "workspace_delta": {
            "changed_files": [{"artifact_path": "changed_files/job.py", "path": "job.py"}],
            "runtime_artifacts": [
                {
                    "artifact_path": (
                        "runtime_artifacts/runtime_workspaces/job/server/simulate_job/app_server/config/"
                        "config_fed_server.json"
                    ),
                    "path": "runtime_workspaces/job/server/simulate_job/app_server/config/config_fed_server.json",
                }
            ],
        },
    }

    info = fl_algorithm_info(run)

    assert info["recipe"] == "fedprox-pt"


def test_fl_algorithm_recipe_mismatch_is_quality_issue(tmp_path):
    from benchmark.harness.modes import WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import run_quality_issues, run_status_kind

    mode_dir = tmp_path / WITH_SKILLS_MODE
    config_path = (
        mode_dir
        / "workspace_delta"
        / "runtime_artifacts"
        / "runtime_workspaces"
        / "job"
        / "server"
        / "simulate_job"
        / "app_server"
        / "config"
        / "config_fed_server.json"
    )
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "workflows": [
                    {
                        "id": "controller",
                        "path": "nvflare.app_common.workflows.fedavg.FedAvg",
                        "args": {"num_rounds": 3},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    run = {
        "available": True,
        "mode_dir": mode_dir,
        "agent_last_message": "**Recipe:** `cyclic-pt`",
        "container_exit": {"exit_code": 0},
        "record": {},
        "run": {"final_container_exit_code": 0},
        "validation_metric": {"name": "AUROC", "value": 0.725},
        "workspace_delta": {
            "runtime_artifacts": [
                {
                    "artifact_path": (
                        "runtime_artifacts/runtime_workspaces/job/server/simulate_job/app_server/config/"
                        "config_fed_server.json"
                    ),
                    "path": "runtime_workspaces/job/server/simulate_job/app_server/config/config_fed_server.json",
                }
            ]
        },
    }

    issues = run_quality_issues(_ev(run), _nv_ev(run))

    assert run_status_kind(_ev(run), _nv_ev(run)) == "needs review"
    assert issues == [
        "Failed check `fl_algorithm_recipe_match`: runtime workflow `FedAvg` does not match selected recipe "
        "`cyclic-pt` (expected one of: Cyclic)."
    ]


def test_fl_algorithm_does_not_infer_from_text_when_job_never_started():
    from benchmark.harness.sdks.nvflare._logic import fl_algorithm_info

    run = {
        "available": True,
        "agent_last_message": "I plan to use SCAFFOLD.",
        "activity": {"commands": ["python - <<'PY'\nprint('SCAFFOLD')\nPY"]},
        "agent_events_text": json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "content": "SCAFFOLD planning note",
                        }
                    ]
                },
            }
        ),
        "workspace_delta": {},
    }

    info = fl_algorithm_info(run)

    assert info["algorithm"] == "not captured"
    assert "job was not started" in info["evidence"]


def test_mode_dir_for_benchmark_does_not_guess_ambiguous_canonical_layout(tmp_path):
    from benchmark.harness.modes import NO_SKILLS_MODE
    from benchmark.harness.reports._loader import mode_dir_for_benchmark

    for repeat in ("01", "02"):
        (
            tmp_path
            / "records"
            / "agent=codex"
            / "model=default"
            / "workflow=default"
            / "job=ames"
            / f"repeat={repeat}"
            / f"mode={NO_SKILLS_MODE}"
        ).mkdir(parents=True)

    assert mode_dir_for_benchmark(tmp_path, NO_SKILLS_MODE) == tmp_path / NO_SKILLS_MODE


def test_numeric_comparison_rejects_bool_values():
    from benchmark.harness.reports.metrics_report import numeric_comparison

    rows = [
        {"summary": {"elapsed_seconds": 10, "token_count": 100}},
        {"summary": {"elapsed_seconds": True, "token_count": False}},
    ]

    assert numeric_comparison(rows) == {}


def test_numeric_comparison_uses_mode_names_not_row_order():
    from benchmark.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from benchmark.harness.reports.metrics_report import numeric_comparison

    rows = [
        {
            "mode": WITH_SKILLS_MODE,
            "summary": {"elapsed_seconds": 13, "token_count": 150},
            "validation_metric": {"name": "AUROC", "value": 0.77},
        },
        {
            "mode": NO_SKILLS_MODE,
            "summary": {"elapsed_seconds": 10, "token_count": 100},
            "validation_metric": {"name": "AUROC", "value": 0.72},
        },
    ]

    assert numeric_comparison(rows) == {
        "elapsed_seconds_with_skills_minus_without_skills": 3,
        "token_count_with_skills_minus_without_skills": 50,
        "validation_metric_AUROC_with_skills_minus_without_skills": 0.050000000000000044,
    }


def test_metrics_report_recovers_validation_metric_from_runtime_artifacts(tmp_path):
    from benchmark.harness.common import write_json
    from benchmark.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from benchmark.harness.reports import metrics_report

    entries = []
    for index, (mode, value) in enumerate(((NO_SKILLS_MODE, 0.72), (WITH_SKILLS_MODE, 0.77)), start=1):
        record_dir = (
            tmp_path
            / "records"
            / "agent=codex"
            / "model=default"
            / "workflow=default"
            / "job=ames"
            / f"mode={mode}"
        )
        record_dir.mkdir(parents=True)
        entries.append(
            {
                "run_id": f"run_{index:05d}",
                "mode": mode,
                "agent": "codex",
                "agent_model": "default",
                "model_source": "scenario",
                "record_dir": str(record_dir.relative_to(tmp_path)),
            }
        )
        write_json(
            record_dir / "run_summary.json",
            {
                "mode": mode,
                "elapsed_seconds": 10 + index,
                "token_count": 100 + index,
                "agent_exit_code": 0,
                "final_container_exit_code": 0,
            },
        )
        write_json(record_dir / "container_exit_code.json", {"exit_code": 0})
        write_json(record_dir / "agent_activity.json", {"command_count": index})
        write_json(
            record_dir / "benchmark_record.json",
            {
                "mode": mode,
                "validation_metric_policy": {"expected_primary_metric": "AUROC"},
            },
        )
        delta_dir = record_dir / "workspace_delta"
        artifact_path = delta_dir / "runtime_artifacts" / "metrics_summary.json"
        artifact_path.parent.mkdir(parents=True)
        write_json(artifact_path, {"AUROC": value})
        write_json(
            record_dir / "workspace_delta_manifest.json",
            {
                "delta_dir": str(delta_dir),
                "runtime_artifacts": [{"artifact_path": "runtime_artifacts/metrics_summary.json"}],
            },
        )
    write_json(tmp_path / "run_plan.json", {"entries": entries})

    summary = metrics_report.write_reports(tmp_path, "Synthetic Metrics")

    assert summary["runs"][0]["validation_metric"]["value"] == 0.72
    assert summary["runs"][1]["validation_metric"]["value"] == 0.77
    assert abs(summary["comparison"]["validation_metric_AUROC_with_skills_minus_without_skills"] - 0.05) < 1e-12
    metrics_markdown = (tmp_path / "metrics_report.md").read_text(encoding="utf-8")
    assert "| Metrics (AUROC) | AUROC 0.7200 | AUROC 0.7700 |" in metrics_markdown


def test_replay_recovers_metric_artifact_when_policy_is_missing(tmp_path):
    from benchmark.harness.common import write_json
    from benchmark.harness.modes import WITH_SKILLS_MODE
    from benchmark.harness.reports import metrics_report
    from benchmark.harness.reports.benchmark_insights import collect_benchmark_runs

    record_dir = (
        tmp_path
        / "records"
        / "agent=claude"
        / "model=default"
        / "workflow=default"
        / "job=ames_lightning"
        / f"mode={WITH_SKILLS_MODE}"
    )
    record_dir.mkdir(parents=True)
    write_json(
        tmp_path / "run_plan.json",
        {
            "entries": [
                {
                    "run_id": "run_00002",
                    "mode": WITH_SKILLS_MODE,
                    "agent": "claude",
                    "agent_model": "default",
                    "record_dir": str(record_dir.relative_to(tmp_path)),
                }
            ]
        },
    )
    write_json(record_dir / "run_summary.json", {"mode": WITH_SKILLS_MODE, "final_container_exit_code": 0})
    write_json(record_dir / "container_exit_code.json", {"exit_code": 0})
    write_json(
        record_dir / "benchmark_record.json",
        {
            "mode": WITH_SKILLS_MODE,
            "reported_validation_metric": {
                "name": "AUROC",
                "value": None,
                "reported_values": [],
                "source": "agent_last_message",
            },
            "quality_signals": {
                "job_guidance_primary_validation_metric": {
                    "status": "not_available",
                    "expected_primary_metric": None,
                    "reported_validation_metric": {"name": "AUROC", "value": None, "reported_values": []},
                }
            },
        },
    )
    delta_dir = record_dir / "workspace_delta"
    artifact_path = (
        delta_dir
        / "runtime_artifacts"
        / "runtime_workspaces"
        / "ames_scaffold"
        / "server"
        / "simulate_job"
        / "metrics"
        / "metrics_summary.json"
    )
    artifact_path.parent.mkdir(parents=True)
    write_json(
        artifact_path,
        {
            "final_aggregated_metrics": [
                {"name": "loss", "value": 0.536},
                {"name": "accuracy", "value": 0.712},
                {"name": "auroc", "value": 0.7648461238800156},
            ]
        },
    )
    write_json(
        record_dir / "workspace_delta_manifest.json",
        {
            "delta_dir": str(delta_dir),
            "runtime_artifacts": [
                {
                    "artifact_path": (
                        "runtime_artifacts/runtime_workspaces/ames_scaffold/server/"
                        "simulate_job/metrics/metrics_summary.json"
                    )
                }
            ],
        },
    )

    runs = collect_benchmark_runs(tmp_path)
    metric = runs[WITH_SKILLS_MODE]["validation_metric"]
    assert metric["source"] == "metrics_artifact"
    assert metric["name"] == "AUROC"
    assert metric["value"] == 0.7648461238800156

    summary = metrics_report.write_reports(tmp_path, "Synthetic Metrics")
    with_skills_row = next(row for row in summary["runs"] if row["mode"] == WITH_SKILLS_MODE)
    assert with_skills_row["validation_metric"]["value"] == 0.7648461238800156


def test_replay_metric_artifact_fallback_uses_dynamic_metric_name(tmp_path):
    from benchmark.harness.common import write_json
    from benchmark.harness.modes import WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import collect_benchmark_runs

    record_dir = (
        tmp_path
        / "records"
        / "agent=claude"
        / "model=default"
        / "workflow=default"
        / "job=segmentation"
        / f"mode={WITH_SKILLS_MODE}"
    )
    record_dir.mkdir(parents=True)
    write_json(
        tmp_path / "run_plan.json",
        {
            "entries": [
                {
                    "run_id": "run_00002",
                    "mode": WITH_SKILLS_MODE,
                    "agent": "claude",
                    "agent_model": "default",
                    "record_dir": str(record_dir.relative_to(tmp_path)),
                }
            ]
        },
    )
    write_json(record_dir / "run_summary.json", {"mode": WITH_SKILLS_MODE, "final_container_exit_code": 0})
    write_json(record_dir / "container_exit_code.json", {"exit_code": 0})
    write_json(
        record_dir / "benchmark_record.json",
        {
            "mode": WITH_SKILLS_MODE,
            "reported_validation_metric": {
                "name": "dice_score",
                "value": None,
                "reported_values": [],
                "source": "agent_last_message",
            },
        },
    )
    delta_dir = record_dir / "workspace_delta"
    artifact_path = delta_dir / "runtime_artifacts" / "metrics_summary.json"
    artifact_path.parent.mkdir(parents=True)
    write_json(
        artifact_path,
        {
            "final_aggregated_metrics": [
                {"name": "loss", "value": 1.91},
                {"name": "dice_score", "value": 62.75},
            ]
        },
    )
    write_json(
        record_dir / "workspace_delta_manifest.json",
        {
            "delta_dir": str(delta_dir),
            "runtime_artifacts": [{"artifact_path": "runtime_artifacts/metrics_summary.json"}],
        },
    )

    metric = collect_benchmark_runs(tmp_path)[WITH_SKILLS_MODE]["validation_metric"]

    assert metric["source"] == "metrics_artifact"
    assert metric["name"] == "dice_score"
    assert metric["value"] == 62.75


def test_cost_comparison_separates_dependency_install_time():
    from benchmark.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import chart_value_display, cost_comparison_section

    def event_lines(command: str, start: str, end: str, item_id: str) -> list[str]:
        return [
            json.dumps(
                {
                    "timestamp": start,
                    "type": "item.started",
                    "item": {"command": command, "id": item_id, "type": "command_execution"},
                }
            ),
            json.dumps(
                {
                    "timestamp": end,
                    "type": "item.completed",
                    "item": {
                        "aggregated_output": "ok",
                        "command": command,
                        "exit_code": 0,
                        "id": item_id,
                        "status": "completed",
                        "type": "command_execution",
                    },
                }
            ),
        ]

    runs = {
        NO_SKILLS_MODE: {
            "label": "No skills baseline",
            "run": {"elapsed_seconds": 100, "token_count": 10},
            "activity": {"command_count": 2, "unique_command_count": 2},
            "workspace_delta": {},
            "agent_events_text": "\n".join(
                event_lines("uv pip install -r requirements.txt", "2026-06-13T00:00:00Z", "2026-06-13T00:00:15Z", "a")
                + event_lines("python job.py", "2026-06-13T00:00:20Z", "2026-06-13T00:01:30Z", "b")
            ),
        },
        WITH_SKILLS_MODE: {
            "label": "With skills",
            "run": {"elapsed_seconds": 300, "token_count": 12},
            "activity": {"command_count": 2, "unique_command_count": 2},
            "workspace_delta": {},
            "agent_events_text": "\n".join(
                event_lines(
                    "uv pip install -r requirements-train.txt",
                    "2026-06-13T00:00:00Z",
                    "2026-06-13T00:02:00Z",
                    "a",
                )
                + event_lines("python job.py", "2026-06-13T00:02:10Z", "2026-06-13T00:04:50Z", "b")
            ),
        },
    }

    section = cost_comparison_section(_evruns(runs), [NO_SKILLS_MODE, WITH_SKILLS_MODE])

    assert "`Runtime seconds` is total elapsed time minus captured dependency-install command/background-task time" in section
    assert "`Dependency install seconds` is captured dependency-install command/background-task time" in section
    assert "Command span timing is operation-level evidence, not a strict wall-clock partition" in section
    assert "| Total time seconds | 100 | 300 | 200 |" in section
    assert "| Runtime seconds | 85 | 180 | 95 |" in section
    assert "| Dependency install seconds | 15 | 120 | 105 |" in section
    assert "| Non-install command seconds | 70 | 160 | 90 |" in section
    assert chart_value_display(1009.055, "seconds") == "1009"
    assert chart_value_display(0.071, "seconds") == "0.071"


def test_dependency_install_time_uses_background_task_duration():
    from benchmark.harness.modes import WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import _dependency_install_spans, _dependency_install_total_seconds

    install_command = "uv pip install -r requirements-train.txt 2>&1 | tail -20"
    events = [
        {
            "event_type": "assistant",
            "harness_timestamp": "2026-06-26T00:00:00.000Z",
            "message": {
                "content": [
                    {
                        "id": "toolu_install",
                        "input": {
                            "command": install_command,
                            "description": "Install training requirements into venv",
                            "run_in_background": True,
                        },
                        "name": "Bash",
                        "type": "tool_use",
                    }
                ]
            },
        },
        {
            "event_type": "system.task_started",
            "harness_timestamp": "2026-06-26T00:00:00.000Z",
            "task_id": "install_task",
            "tool_use_id": "toolu_install",
        },
        {
            "event_type": "user",
            "harness_timestamp": "2026-06-26T00:00:00.071Z",
            "message": {
                "content": [
                    {
                        "content": "Command running in background with ID: install_task",
                        "tool_use_id": "toolu_install",
                        "type": "tool_result",
                    }
                ]
            },
            "tool_use_result": {"backgroundTaskId": "install_task"},
        },
        {
            "event_type": "system.task_updated",
            "harness_timestamp": "2026-06-26T00:01:23.000Z",
            "patch": {"status": "completed"},
            "task_id": "install_task",
        },
    ]
    run = _ev(
        {
            "available": True,
            "mode": WITH_SKILLS_MODE,
            "label": "With skills",
            "run": {"elapsed_seconds": 120, "token_count": 12},
            "activity": {"commands": [install_command]},
            "workspace_delta": {},
            "agent_events_text": "\n".join(json.dumps(event) for event in events),
        }
    )

    spans = _dependency_install_spans(run)

    assert len(spans) == 1
    assert spans[0]["duration_source"] == "background_task"
    assert spans[0]["duration_seconds"] == 83
    assert _dependency_install_total_seconds(run) == 83


def test_cost_comparison_treats_missing_dependency_install_spans_as_zero():
    from benchmark.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import cost_comparison_section

    runs = {
        NO_SKILLS_MODE: {
            "label": "No skills baseline",
            "run": {"elapsed_seconds": 100, "token_count": 10},
            "activity": {},
            "workspace_delta": {},
            "agent_events_text": "",
        },
        WITH_SKILLS_MODE: {
            "label": "With skills",
            "run": {"elapsed_seconds": 120, "token_count": 12},
            "activity": {},
            "workspace_delta": {},
            "agent_events_text": "",
        },
    }

    section = cost_comparison_section(_evruns(runs), [NO_SKILLS_MODE, WITH_SKILLS_MODE])

    assert "| Runtime seconds | 100 | 120 | 20 |" in section
    assert "| Dependency install seconds | 0 | 0 | 0 |" in section


def test_cost_comparison_keeps_attempted_install_with_missing_timing_unknown():
    from benchmark.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import _elapsed_time_accounting_note, cost_comparison_section

    runs = {
        NO_SKILLS_MODE: {
            "label": "No skills baseline",
            "run": {"elapsed_seconds": 100, "token_count": 10},
            "activity": {},
            "workspace_delta": {},
            "agent_events_text": "",
        },
        WITH_SKILLS_MODE: {
            "label": "With skills",
            "run": {"elapsed_seconds": 120, "token_count": 12},
            "activity": {"commands": ["uv pip install -r requirements.txt"]},
            "workspace_delta": {},
            "agent_events_text": "",
        },
    }

    section = cost_comparison_section(_evruns(runs), [NO_SKILLS_MODE, WITH_SKILLS_MODE])

    assert "| Runtime seconds | 100 | NA | NA |" in section
    assert "| Dependency install seconds | 0 | NA | NA |" in section
    accounting = _elapsed_time_accounting_note(_ev(runs[WITH_SKILLS_MODE]), _ev(runs[NO_SKILLS_MODE]))
    assert "| With skills | 120s | NA | NA | NA |" in accounting
    assert "NAs" not in accounting


def test_why_section_surfaces_repeated_successful_job_executions():
    from benchmark.harness.reports.benchmark_insights import _why_slower, job_run_status_section
    from benchmark.harness.sdks.nvflare._logic import job_run_status_reason

    def bash_pair(
        tool_id: str,
        command: str,
        description: str,
        start: str,
        end: str,
        output: str = "Finished FedAvg.",
    ) -> list[str]:
        return [
            json.dumps(
                {
                    "harness_timestamp": start,
                    "message": {
                        "content": [
                            {
                                "id": tool_id,
                                "input": {"command": command, "description": description},
                                "name": "Bash",
                                "type": "tool_use",
                            }
                        ]
                    },
                }
            ),
            json.dumps(
                {
                    "harness_timestamp": end,
                    "message": {
                        "content": [
                            {
                                "content": output,
                                "is_error": False,
                                "tool_use_id": tool_id,
                                "type": "tool_result",
                            }
                        ]
                    },
                    "tool_use_result": {"stdout": output, "stderr": ""},
                }
            ),
        ]

    with_run = {
        "available": True,
        "mode": "with_skills",
        "label": "With skills",
        "run": {"elapsed_seconds": 300},
        "agent_events_text": "\n".join(
            bash_pair(
                "run-1",
                "python3 job.py --num-sites 3 --num-rounds 3",
                "Run 3-site simulation",
                "2026-06-13T20:00:00Z",
                "2026-06-13T20:01:20Z",
            )
            + bash_pair(
                "run-2",
                "rm -rf /tmp/nvflare/workspaces/ames && python3 job.py --num-sites 3 --num-rounds 3",
                "Re-run simulation with aligned metric names",
                "2026-06-13T20:02:00Z",
                "2026-06-13T20:04:00Z",
            )
        ),
    }
    base_run = {
        "available": True,
        "mode": "without_skills",
        "label": "No skills baseline",
        "run": {"elapsed_seconds": 100},
        "agent_events_text": "\n".join(
            bash_pair(
                "base-run",
                "python3 job.py --num-sites 3 --num-rounds 3",
                "Run baseline simulation",
                "2026-06-13T20:00:00Z",
                "2026-06-13T20:01:00Z",
            )
        ),
    }

    reason = job_run_status_reason(with_run)
    status_section = job_run_status_section(
        _evruns({"with": with_run}), ["with"], _nv_ctx({"with": with_run}, ["with"])
    )
    why_text = "\n".join(
        _why_slower(
            _ev(with_run),
            _ev(base_run),
            _nv_ctx({"with_skills": with_run, "without_skills": base_run}),
        )
    )

    assert "2 successful job/simulator executions captured" in reason
    assert "total job time 200s" in reason
    assert "Re-run simulation with aligned metric names" in reason
    assert "### Repeated Job/Simulation Executions" not in status_section
    assert "### Repeated Job/Simulation Executions" in why_text
    assert "| With skills | 2 | 200s |" in why_text
    assert "runtime workspace was cleared before rerun" in why_text
    assert (
        "Baseline comparison: No skills baseline had 1 command classified successful job/simulator execution totaling 60s."
        in why_text
    )


def test_why_section_surfaces_skill_guidance_overridden_by_docstring_discovery():
    from benchmark.harness.reports.benchmark_insights import _why_slower

    events_text = "\n".join(
        [
            json.dumps(
                {
                    "message": {
                        "content": [
                            {
                                "id": "skill-1",
                                "input": {"skill": "nvflare-convert-lightning"},
                                "name": "Skill",
                                "type": "tool_use",
                            }
                        ]
                    }
                }
            ),
            json.dumps(
                {
                    "message": {
                        "content": [
                            {
                                "content": (
                                    "Client script requirement: Unlike FedAvgRecipe, the client script must use "
                                    "PTScaffoldHelper and include SCAFFOLD_CTRL_DIFF."
                                ),
                                "is_error": False,
                                "tool_use_id": "read-1",
                                "type": "tool_result",
                            }
                        ]
                    },
                    "tool_use_result": {
                        "stdout": "/workspace/venv/lib/python3.11/site-packages/nvflare/app_opt/pt/recipes/scaffold.py"
                    },
                }
            ),
            json.dumps(
                {
                    "message": {
                        "content": [
                            {
                                "text": (
                                    "The standard Lightning `flare.patch(trainer)` exchange genuinely cannot express "
                                    "SCAFFOLD, so I need to switch implementation strategy."
                                ),
                                "type": "text",
                            }
                        ]
                    }
                }
            ),
        ]
    )
    with_run = {
        "available": True,
        "mode": "with_skills",
        "label": "With skills",
        "run": {"elapsed_seconds": 300},
        "activity": {"tool_counts": {"Skill": 1}, "event_types": {"assistant": 10}},
        "agent_events_text": events_text,
    }
    base_run = {
        "available": True,
        "mode": "without_skills",
        "label": "No skills baseline",
        "run": {"elapsed_seconds": 100},
        "activity": {"tool_counts": {}, "event_types": {"assistant": 3}},
    }

    why_text = "\n".join(_why_slower(_ev(with_run), _ev(base_run)))

    assert "Skill guidance overridden after local API/source inspection" in why_text
    assert "recipes/scaffold.py" in why_text
    assert "local source/docstring evidence" in why_text
    assert "conflicting with the skill guidance" in why_text
    assert "changed implementation strategy" in why_text
    assert "does not assert that the local API/source claim is permanently true" in why_text
    assert "genuinely cannot express SCAFFOLD" in why_text


def test_repeated_job_reason_uses_codex_agent_message_context():
    from benchmark.harness.reports.benchmark_insights import repeated_job_runs_section
    from benchmark.harness.sdks.nvflare._logic import repeated_job_run_summary

    def codex_command_pair(item_id: str, command: str, start: str, end: str) -> list[str]:
        return [
            json.dumps(
                {
                    "type": "item.started",
                    "harness_timestamp": start,
                    "item": {
                        "command": command,
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
                        "aggregated_output": "Finished FedAvg. Simulation workspace: /tmp/nvflare/workspaces/job",
                        "command": command,
                        "exit_code": 0,
                        "id": item_id,
                        "status": "completed",
                        "type": "command_execution",
                    },
                }
            ),
        ]

    run = {
        "available": True,
        "label": "With skills",
        "agent_events_text": "\n".join(
            codex_command_pair("run-1", "/bin/bash -lc 'python job.py'", "2026-06-13T20:00:00Z", "2026-06-13T20:01:00Z")
            + [
                json.dumps(
                    {
                        "type": "item.completed",
                        "harness_timestamp": "2026-06-13T20:01:10Z",
                        "item": {
                            "text": (
                                "I noticed one small robustness improvement in the client send path. "
                                "I am re-exporting and re-running the simulation so the final artifacts "
                                "match the current source exactly."
                            ),
                            "type": "agent_message",
                        },
                    }
                )
            ]
            + codex_command_pair(
                "run-2", "/bin/bash -lc 'python job.py'", "2026-06-13T20:02:00Z", "2026-06-13T20:03:10Z"
            )
        ),
    }

    summary = repeated_job_run_summary(run)
    section = repeated_job_runs_section(_evruns({"with": run}), ["with"], _nv_ctx({"with": run}, ["with"]))

    assert "2 successful job/simulator executions captured" in summary
    assert "re-exporting and re-running the simulation" in summary
    assert "final artifacts match the current source" in section


def test_why_section_surfaces_repeated_dependency_install_reason():
    from benchmark.harness.reports.benchmark_insights import _why_slower

    def command_event(
        item_id: str,
        command: str,
        start: str,
        end: str,
        output: str,
        *,
        exit_code: int = 0,
    ) -> list[str]:
        status = "completed" if exit_code == 0 else "failed"
        return [
            json.dumps(
                {
                    "type": "item.started",
                    "harness_timestamp": start,
                    "item": {
                        "command": command,
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
                        "exit_code": exit_code,
                        "id": item_id,
                        "status": status,
                        "type": "command_execution",
                    },
                }
            ),
        ]

    failed_output = (
        "Downloading nvidia-cublas (517.7MiB)\n"
        "pip._vendor.urllib3.exceptions.ProtocolError: "
        "('Connection broken: IncompleteRead(197972800 bytes read, 23078544 more expected)')\n"
    )
    retry_output = "Successfully installed nvidia-cublas-13.1.1.3 nvidia-cudnn-cu13-9.20.0.48 torch-2.12.0\n"
    with_run = {
        "available": True,
        "label": "With skills",
        "run": {"elapsed_seconds": 1200},
        "agent_events_text": "\n".join(
            command_event(
                "install-1",
                "python -m pip install -r requirements-train.txt",
                "2026-06-13T20:00:00Z",
                "2026-06-13T20:11:42Z",
                failed_output,
                exit_code=2,
            )
            + command_event(
                "install-2",
                "python -m pip install -r requirements-train.txt --retries 10 --timeout 120",
                "2026-06-13T20:12:00Z",
                "2026-06-13T20:16:41Z",
                retry_output,
            )
        ),
    }
    base_run = {
        "available": True,
        "label": "No skills baseline",
        "run": {"elapsed_seconds": 500},
        "agent_events_text": "\n".join(
            command_event(
                "install-base",
                "python -m pip install -r requirements-train.txt",
                "2026-06-13T20:00:00Z",
                "2026-06-13T20:06:31Z",
                "Successfully installed torch-2.12.0\n",
            )
        ),
    }

    explanation = "\n".join(_why_slower(_ev(with_run), _ev(base_run)))

    assert "### Repeated Dependency Install Attempts" in explanation
    assert "| With skills | 2 | 983s |" in explanation
    assert "first failed: ProtocolError" in explanation
    assert "later dependency install succeeded" in explanation
    assert "broken/incomplete download" in explanation
    assert "accelerator package evidence: nvidia-cublas, nvidia-cudnn-cu13" in explanation
    assert "Dependency install path differed" in explanation
    assert "large accelerator/framework wheels can dominate install time" in explanation


def test_repeated_dependency_install_reason_does_not_mark_prior_success_as_recovery():
    from benchmark.harness.reports.benchmark_insights import repeated_dependency_install_section

    def command_event(item_id: str, command: str, exit_code: int, output: str) -> str:
        status = "completed" if exit_code == 0 else "failed"
        return json.dumps(
            {
                "type": "item.completed",
                "harness_timestamp": "2026-06-13T20:00:00Z",
                "item": {
                    "aggregated_output": output,
                    "command": command,
                    "exit_code": exit_code,
                    "id": item_id,
                    "status": status,
                    "type": "command_execution",
                },
            }
        )

    run = {
        "available": True,
        "label": "With skills",
        "agent_events_text": "\n".join(
            [
                command_event("install-1", "python -m pip install -r requirements.txt", 0, "Successfully installed a"),
                command_event("install-2", "python -m pip install -r requirements.txt", 0, "Successfully installed a"),
                command_event(
                    "install-3", "python -m pip install -r requirements.txt", 1, "ERROR: final install failed"
                ),
            ]
        ),
    }

    section = repeated_dependency_install_section(_evruns({"with": run}), ["with"])

    assert "first failed: ERROR: final install failed" in section
    assert "later dependency install succeeded" not in section


def test_structure_tree_falls_back_to_final_workspace_when_changed_python_is_empty():
    from benchmark.harness.modes import WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import structure_trees_section

    report = structure_trees_section(
        _evruns(
            {
                WITH_SKILLS_MODE: {
                    "available": True,
                    "label": "With skills",
                    "workspace_delta": {
                        "changed_files": [
                            {"path": "nvflare_jobs/ames_fedavg/README.md"},
                            {"path": "nvflare_jobs/ames_fedavg/requirements.txt"},
                        ],
                        "final_structure_files": [
                            {"path": "download_data.py"},
                            {"path": "model.py"},
                            {"path": "nvflare_jobs/ames_fedavg/client.py"},
                            {"path": "nvflare_jobs/ames_fedavg/job.py"},
                            {"path": "nvflare_jobs/ames_fedavg/model.py"},
                        ],
                    },
                }
            }
        ),
        [WITH_SKILLS_MODE],
    )

    assert "Final workspace:" in report
    assert "Changed/generated files:" not in report
    assert "none" not in report
    assert "nvflare_jobs" in report
    assert "client.py" in report
    assert "job.py" in report
    assert "README.md" not in report
    assert "requirements.txt" not in report


def test_structure_score_does_not_count_nested_job_source_as_current_structure():
    from benchmark.harness.reports.benchmark_insights import (
        nested_generated_structure_display,
        structure_required_display,
    )
    from benchmark.harness.reports.evidence import _run_evidence_from_bundle
    from benchmark.harness.sdks.nvflare._logic import structure_score
    from benchmark.harness.sdks.nvflare.plugin import NvflareReportPlugin

    run = {
        "available": True,
        "workspace_delta": {
            "final_structure_files": [
                {"path": "model.py"},
                {"path": "nvflare_jobs/ames_fedavg/client.py"},
                {"path": "nvflare_jobs/ames_fedavg/job.py"},
                {"path": "nvflare_jobs/ames_fedavg/model.py"},
            ],
            "changed_files": [
                {"path": "nvflare_jobs/ames_fedavg/client.py"},
                {"path": "nvflare_jobs/ames_fedavg/job.py"},
                {"path": "nvflare_jobs/ames_fedavg/model.py"},
            ],
        },
    }

    score = structure_score(run)
    assert score == 1 / 3
    view = NvflareReportPlugin().collect(_run_evidence_from_bundle(run)).structure_view
    assert view.score == 1 / 3
    assert structure_required_display(_ev(run), view).startswith("1/3 present; missing client.py, job.py")
    assert "nested copies ignored" in structure_required_display(_ev(run), view)
    assert (
        nested_generated_structure_display(_ev(run), view) == "nvflare_jobs/ames_fedavg (client.py, job.py, model.py)"
    )


def test_generated_code_quality_section_reports_evidence_without_gate_language(tmp_path):
    from benchmark.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import generated_code_quality_section

    def command_event(command: str, output: str, exit_code: int = 0) -> str:
        status = "completed" if exit_code == 0 else "failed"
        return json.dumps(
            {
                "item": {
                    "aggregated_output": output,
                    "command": command,
                    "exit_code": exit_code,
                    "id": "item_1",
                    "status": status,
                    "type": "command_execution",
                },
                "type": "item.completed",
            }
        )

    def run_with_client(
        mode: str, client_source: str, install_command: str, install_output: str, rel_path: str = "client.py"
    ) -> dict:
        mode_dir = tmp_path / mode
        client_path = mode_dir / "workspace_delta" / "changed_files" / rel_path
        log_path = mode_dir / "workspace_delta" / "runtime_artifacts" / "site-1" / "log.txt"
        client_path.parent.mkdir(parents=True)
        log_path.parent.mkdir(parents=True)
        client_path.write_text(client_source, encoding="utf-8")
        log_path.write_text("[site-1] round 1 epoch 01 train_loss=0.4 device=cpu\n", encoding="utf-8")
        return {
            "available": True,
            "label": mode,
            "skills": "with skills" if mode == WITH_SKILLS_MODE else "without skills",
            "mode_dir": mode_dir,
            "workspace_delta": {
                "changed_files": [
                    {
                        "artifact_path": f"changed_files/{rel_path}",
                        "path": rel_path,
                    }
                ],
                "runtime_artifacts": [
                    {
                        "artifact_path": "runtime_artifacts/site-1/log.txt",
                        "path": "site-1/log.txt",
                        "source_path": "/tmp/nvflare/workspaces/job/site-1/log.txt",
                    }
                ],
            },
            "agent_events_text": command_event(install_command, install_output),
        }

    repeated_setup_client = """
import nvflare.client as flare
def partition_frame(frame, site_index, num_clients): return frame
while flare.is_running():
    input_model = flare.receive()
    train_frame, valid_frame, test_frame = load_data_frames(args.data_dir)
    train_frame = partition_frame(train_frame, site_index, args.num_clients)
    train_loader = make_loader(train_frame)
    criterion, optimizer, _ = build_loss_and_optimizer(model, train_frame, args, device)
    valid_metrics = evaluate(model, valid_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)
    append_record(results_path, {"metrics": test_metrics})
"""
    lean_client = """
import nvflare.client as flare
train_frame = load_split(args.data_dir, "train")
local_train = site_shard(train_frame, site_name)
train_loader = DataLoader(local_train)
criterion, optimizer, pos_weight_value = build_loss_and_optimizer(model, local_train, args, device)
while flare.is_running():
    input_model = flare.receive()
    for epoch in range(1, args.local_epochs + 1):
        print(f"[{site_name}] round {round_num} epoch {epoch:02d}")
    global_metrics = evaluate(model, valid_loader, criterion, device)
    local_metrics = evaluate(model, valid_loader, criterion, device)
"""
    runs = {
        NO_SKILLS_MODE: run_with_client(
            NO_SKILLS_MODE,
            lean_client,
            "python -m pip install torch --index-url https://download.pytorch.org/whl/cpu",
            "Successfully installed torch-2.12.0+cpu",
            rel_path="run_nvflare_fedavg.py",
        ),
        WITH_SKILLS_MODE: run_with_client(
            WITH_SKILLS_MODE,
            repeated_setup_client,
            "uv pip install -r requirements-train.txt",
            "Successfully installed nvidia-cublas-13.1 nvidia-cudnn-cu13-9.20 triton-3.7 torch-2.12",
        ),
    }

    section = generated_code_quality_section(
        _evruns(runs), [NO_SKILLS_MODE, WITH_SKILLS_MODE], _nv_ctx(runs, [NO_SKILLS_MODE, WITH_SKILLS_MODE])
    )

    assert "These are evidence signals" in section
    assert "They do not change pass/fail quality gates" in section
    assert "Overall code quality signal" in section
    assert "/14 evidence points" in section
    assert "explicit sharding" in section
    assert "API pattern" in section
    assert "context: Client API loop pattern" in section
    assert "Loss/optimizer lifecycle" in section
    assert "Data/DataLoader lifecycle" in section
    assert "poor: loss/optimizer rebuilt inside FL loop" in section
    assert "good: loss/optimizer built outside FL loop" in section
    assert "good: data loaded before FL loop, DataLoader built before FL loop" in section
    assert "poor: data loaded inside FL loop, DataLoader built inside FL loop" in section
    assert "good: 2 evaluate call(s) in FL loop, test evaluation inside FL loop" in section
    assert "runtime logs show per-epoch progress" in section
    assert "runtime artifacts captured separately from temp/runtime paths" in section
    assert "CPU-only framework wheel" in section
    assert "accelerator-capable dependency stack" in section
    assert "skill requirements install not followed" not in section
    assert "CPU-only framework installs are faster, but they should only be treated as comparable" in section


def test_runtime_output_locality_scores_workspace_changes_as_caution():
    from benchmark.harness.sdks.nvflare._logic import _assessment_from_locality, _runtime_output_locality_signal

    run = {
        "workspace_delta": {
            "changed_files": [{"path": "fl_workspace/ames_fedavg/server/simulate_job/app_server/custom/client.py"}],
            "runtime_artifacts": [
                {
                    "path": "server/log.txt",
                    "source_path": "/tmp/nvflare/workspaces/ames/server/log.txt",
                }
            ],
        }
    }

    evidence = _runtime_output_locality_signal(run)

    assert "runtime artifacts captured separately from temp/runtime paths" in evidence
    assert "runtime output appears in workspace changes" in evidence
    assert _assessment_from_locality(evidence) == "caution"


def test_generated_code_quality_does_not_claim_loop_placement_when_loop_missing(tmp_path):
    from benchmark.harness.modes import NO_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import generated_code_quality_section

    mode_dir = tmp_path / NO_SKILLS_MODE
    source_path = mode_dir / "workspace_delta" / "changed_files" / "run_nvflare_fedavg.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        """
from torch.utils.data import DataLoader

train_frame = load_split(args.data_dir, "train")
train_loader = DataLoader(train_frame)
test_frame = load_split(args.data_dir, "test")
test_loader = DataLoader(test_frame)
criterion, optimizer = build_loss_and_optimizer(model, train_frame, args, device)
metric = evaluate(model, test_loader, criterion, device)
""",
        encoding="utf-8",
    )
    run = {
        "available": True,
        "label": NO_SKILLS_MODE,
        "mode_dir": mode_dir,
        "workspace_delta": {
            "changed_files": [
                {
                    "artifact_path": "changed_files/run_nvflare_fedavg.py",
                    "path": "run_nvflare_fedavg.py",
                }
            ]
        },
    }

    section = generated_code_quality_section(
        _evruns({NO_SKILLS_MODE: run}), [NO_SKILLS_MODE], _nv_ctx({NO_SKILLS_MODE: run}, [NO_SKILLS_MODE])
    )

    assert "caution: loss/optimizer setup present; FL loop not captured" in section
    assert "caution: data loading present, DataLoader construction present; FL loop not captured" in section
    assert "good: 1 evaluate call(s) in generated code, test evaluation present, FL loop not captured" in section
    assert "data loaded before FL loop; FL loop not captured" not in section
    assert "evaluate call(s) in FL loop" not in section
    assert "test evaluation inside FL loop" not in section


def test_generated_code_quality_detects_model_learner_api_pattern_without_filename_bias(tmp_path):
    from benchmark.harness.modes import NO_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import generated_code_quality_section

    mode_dir = tmp_path / NO_SKILLS_MODE
    source_path = mode_dir / "workspace_delta" / "changed_files" / "custom_training_component.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        """
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.abstract.model_learner import ModelLearner

class CustomTrainingComponent(ModelLearner):
    def initialize(self):
        train_frame, valid_frame, test_frame = load_data_frames(self.data_dir)
        self._train_loader, self._valid_loader, self._test_loader = build_data_loaders(
            train_frame, valid_frame, test_frame, self.vocab, self._args
        )
        self._criterion, _, _ = build_loss_and_optimizer(self._model, train_frame, self._args, self._device)

    def train(self, model: FLModel) -> FLModel:
        optimizer = torch.optim.AdamW(self._model.parameters())
        train_one_epoch(self._model, self._train_loader, self._criterion, optimizer, self._device)
        valid_metrics = evaluate(self._model, self._valid_loader, self._criterion, self._device)
        return FLModel(metrics=valid_metrics)
""",
        encoding="utf-8",
    )
    run = {
        "available": True,
        "label": NO_SKILLS_MODE,
        "mode_dir": mode_dir,
        "workspace_delta": {
            "changed_files": [
                {
                    "artifact_path": "changed_files/custom_training_component.py",
                    "path": "custom_training_component.py",
                }
            ]
        },
    }

    section = generated_code_quality_section(
        _evruns({NO_SKILLS_MODE: run}), [NO_SKILLS_MODE], _nv_ctx({NO_SKILLS_MODE: run}, [NO_SKILLS_MODE])
    )

    assert "context: ModelLearner pattern" in section
    assert "poor: loss/optimizer rebuilt inside FL loop" in section
    assert "good: data loaded before FL loop, DataLoader built before FL loop" in section


def test_dependency_strategy_scores_with_skills_cpu_shortcut_as_instruction_failure():
    from benchmark.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import generated_code_quality_section

    def run(mode: str, command: str, output: str) -> dict:
        return {
            "available": True,
            "label": mode,
            "skills": "with skills" if mode == WITH_SKILLS_MODE else "without skills",
            "agent_events_text": json.dumps(
                {
                    "item": {
                        "aggregated_output": output,
                        "command": command,
                        "exit_code": 0,
                        "id": "item_1",
                        "status": "completed",
                        "type": "command_execution",
                    },
                    "type": "item.completed",
                }
            ),
            "workspace_delta": {},
        }

    runs = {
        NO_SKILLS_MODE: run(
            NO_SKILLS_MODE,
            "python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu",
            "Successfully installed torch-2.12.0+cpu",
        ),
        WITH_SKILLS_MODE: run(
            WITH_SKILLS_MODE,
            "python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu",
            "Successfully installed torch-2.12.0+cpu",
        ),
    }
    section = generated_code_quality_section(
        _evruns(runs),
        [NO_SKILLS_MODE, WITH_SKILLS_MODE],
        _nv_ctx(runs, [NO_SKILLS_MODE, WITH_SKILLS_MODE]),
    )

    assert "caution: targeted package install, CPU-only framework wheel, succeeded" in section
    assert (
        "poor: targeted package install, CPU-only framework wheel, succeeded, skill requirements install not followed"
        in section
    )


def test_workspace_delta_issue_allows_final_structure_and_runtime_artifacts():
    from benchmark.harness.reports.benchmark_insights import run_quality_issues

    run = {
        "available": True,
        "record": {
            "workspace_delta": {
                "changed_file_count": 0,
                "runtime_artifact_count": 37,
                "final_structure_files": [
                    {"path": "nvflare_jobs/ames_fedavg/client.py"},
                    {"path": "nvflare_jobs/ames_fedavg/job.py"},
                ],
            }
        },
    }

    issues = run_quality_issues(_ev(run))

    assert not any("workspace_delta" in issue for issue in issues)


def test_workspace_delta_issue_allows_manifest_counts_without_file_lists():
    from benchmark.harness.reports.benchmark_insights import run_quality_issues

    run = {
        "available": True,
        "record": {
            "workspace_delta": {
                "changed_file_count": 0,
                "copied_file_count": 37,
                "final_structure_file_count": 5,
                "runtime_artifact_count": 37,
            }
        },
    }

    issues = run_quality_issues(_ev(run))

    assert not any("workspace_delta" in issue for issue in issues)


def test_status_summary_is_human_readable_for_failures():
    from benchmark.harness.modes import NO_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import status_summary

    runs = {
        NO_SKILLS_MODE: {
            "available": True,
            "container_exit": {"exit_code": 1},
            "console_text": "docker: Error response from daemon: pull access denied for agent-skills-benchmark",
            "run": {},
            "status": "missing",
            "validation_metric": {},
        }
    }

    summary = status_summary(_evruns(runs), [NO_SKILLS_MODE])

    assert "No skills baseline: failed" in summary
    assert "container exit 1" in summary
    assert "Docker image unavailable" in summary
    assert "exit=1" not in summary


def test_failure_analysis_extracts_unsupported_model_message():
    from benchmark.harness.reports.benchmark_insights import failure_evidence, failure_root_cause

    run = {
        "available": True,
        "agent_events_text": "The 'gpt-5.3-codex' model is not supported when using Codex with a ChatGPT account.",
        "container_exit": {"exit_code": 1},
        "run": {"agent_exit_code": 1},
        "status": "missing",
        "validation_metric": {},
    }

    assert failure_root_cause(_ev(run)) == (
        "Agent model selection failed: The 'gpt-5.3-codex' model is not supported when using Codex with a ChatGPT account."
    )
    assert failure_evidence(run) == (
        "The 'gpt-5.3-codex' model is not supported when using Codex with a ChatGPT account."
    )


def test_failure_root_cause_prefers_agent_exit_classifier():
    from benchmark.harness.reports.benchmark_insights import failure_root_cause

    run = {
        "available": True,
        "agent_events_text": "unstructured error text",
        "record": {"agent_exit_summary": {"failure_category": "agent_auth_failure"}},
        "run": {"agent_exit_code": 1},
    }

    assert failure_root_cause(_ev(run)) == "Agent failure category: agent_auth_failure"


def test_failure_root_cause_infers_auth_from_agent_last_message():
    from benchmark.harness.modes import WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import (
        failure_analysis_section,
        failure_root_cause,
        job_run_action,
    )
    from benchmark.harness.sdks.nvflare._logic import job_run_status_reason

    run = {
        "available": True,
        "agent_events_text": '{"error":"authentication_failed","message":{"content":[{"text":"Not logged in"}]}}',
        "agent_last_message": "Not logged in - Please run /login",
        "container_exit": {"exit_code": 1},
        "record": {"agent_exit_summary": {"failure_category": "agent_unknown_failure"}},
        "run": {"final_container_exit_code": 1},
    }

    assert failure_root_cause(_ev(run)) == "Agent failure category: agent_auth_failure"
    assert "agent_auth_failure" in job_run_status_reason(run)
    assert "Not logged in" in job_run_status_reason(run)
    assert '{"error"' not in job_run_status_reason(run)
    assert "Authenticate the selected agent" in job_run_action(_ev(run), _nv_ev(run).job_execution)

    section = failure_analysis_section(
        _evruns({WITH_SKILLS_MODE: run}), [WITH_SKILLS_MODE], _nv_ctx({WITH_SKILLS_MODE: run}, [WITH_SKILLS_MODE])
    )
    assert "Job run status: not_started" in section
    assert "agent_auth_failure" in section
    assert "Not logged in" in section


def test_failure_analysis_identifies_agent_generated_requirements_file():
    from benchmark.harness.reports.benchmark_insights import dependency_reference_notes

    run = {
        "agent_last_message": "Install with python3 -m pip install -r requirements-federated.txt.",
        "record": {
            "source_input_delta": {
                "final_files": [
                    {"path": "requirements-train.txt"},
                ]
            },
            "workspace_delta": {
                "workspace_added_files": [
                    {"path": "requirements-federated.txt"},
                ]
            },
        },
    }

    assert dependency_reference_notes(_ev(run)) == [
        "`requirements-federated.txt` provenance: agent-generated file.",
    ]


def test_readme_metric_alignment_uses_aggregated_validation_metric_scalar():
    from benchmark.harness.quality_signals import metric_signal

    signal = metric_signal(
        None,
        "AUROC is the main metric.\n",
        """
Round 2 validation AUROC by site:
- `site-1`: `0.7659574468`
- `site-2`: `0.7554566645`
- `site-3`: `0.7373779931`
- aggregated best validation metric: `0.7529307015`
""",
    )

    metric = signal["reported_validation_metric"]
    assert signal["status"] == "pass"
    assert signal["aligned_with_readme"] is True
    assert signal["metric_value_available"] is True
    assert signal["metric_scalar_available"] is True
    assert metric["name"] == "AUROC"
    assert metric["value"] == 0.7529307015
    assert metric["value_scope"] == "fl_summary_metric"
    assert metric["site_value_count"] == 3
    assert metric["summary_value_label"] == "aggregated best validation metric"


def test_readme_metric_alignment_uses_named_aggregated_metric_scalar():
    from benchmark.harness.quality_signals import metric_signal

    signal = metric_signal(
        None,
        "AUROC is the main metric.\n",
        """
Validation:
- Local training AUROC: 0.7531
- Best aggregated validation AUROC: 0.7623334631865992
- Final site metrics: site-1 valid AUROC 0.767293, site-2 valid AUROC 0.757374
""",
    )

    metric = signal["reported_validation_metric"]
    assert signal["status"] == "pass"
    assert signal["metric_scalar_available"] is True
    assert metric["name"] == "AUROC"
    assert metric["value"] == 0.7623334631865992
    assert metric["value_scope"] == "fl_summary_metric"
    assert metric["summary_value_label"] == "Best aggregated validation AUROC"


def test_metric_alignment_rejects_out_of_range_auroc_from_dependency_version():
    from benchmark.harness.quality_signals import metric_signal

    signal = metric_signal(
        None,
        "Primary validation metric: AUROC.\n",
        """
The job uses NVFLARE 2.8 recipe APIs.
The job config uses AUROC as key_metric.
Requirements:
- nvflare[PT]>=2.8.0

No simulation was run.
""",
    )

    metric = signal["reported_validation_metric"]
    assert signal["status"] == "missing"
    assert signal["metric_value_available"] is False
    assert metric["name"] == "AUROC"
    assert metric["value"] is None
    assert metric["reported_values"] == []


def test_benchmark_report_trusts_captured_metric_value_without_range_assumption():
    # The harness carries no metric vocabulary and makes NO assumption about a metric's
    # range (which metric a job uses, and any valid range, is job/instruction data, not
    # engine knowledge). A captured finite value is therefore trusted as-is, even if it
    # falls outside a range one might expect for a given metric name. The real guard
    # against a version number being mistaken for a metric is at PARSE time (a dotted
    # version token is never extracted) -- see
    # test_metric_alignment_rejects_out_of_range_auroc_from_dependency_version.
    from benchmark.harness.reports._loader import validation_metric_from_record
    from benchmark.harness.reports.benchmark_insights import quality_signal

    record = {
        "reported_validation_metric": {
            "name": "AUROC",
            "reported_value_entries": [{"value": 2.8}],
            "reported_values": [2.8],
            "source": "agent_last_message",
            "value": 2.8,
            "value_scope": "reported_scalar",
        },
        "quality_signals": {
            "job_guidance_primary_validation_metric": {
                "expected_primary_metric": "AUROC",
                "evidence": "Job guidance declares AUROC as the primary metric, and the final response reported AUROC 2.8000.",
                "metric_value_available": True,
                "reported_validation_metric": {
                    "name": "AUROC",
                    "reported_value_entries": [{"value": 2.8}],
                    "reported_values": [2.8],
                    "value": 2.8,
                    "value_scope": "reported_scalar",
                },
                "status": "pass",
            }
        },
    }

    metric = validation_metric_from_record(record)
    signal = quality_signal(record)

    assert metric["name"] == "AUROC"
    assert metric["value"] == 2.8
    assert metric["reported_values"] == [2.8]
    assert signal["status"] == "pass"
    assert signal["metric_value_available"] is True


def test_job_guidance_metric_alignment_uses_non_readme_docs(tmp_path):
    from benchmark.harness.quality_signals import metric_signal
    from benchmark.harness.records import discover_job_guidance

    job = tmp_path / "job"
    docs = job / "docs"
    docs.mkdir(parents=True)
    docs.joinpath("metrics.md").write_text("Target validation metric: accuracy.\n", encoding="utf-8")

    sources, guidance_text = discover_job_guidance(job)
    signal = metric_signal(
        sources,
        guidance_text,
        "Server best validation metric at round 3: 0.8123 accuracy",
    )

    assert signal["expected_primary_metric"] == "accuracy"
    assert signal["aligned_with_job_guidance"] is True
    assert signal["sources"][0]["path"].endswith("metrics.md")
    assert signal["reported_validation_metric"]["name"] == "accuracy"


def test_job_guidance_skips_symlink_guidance_files(tmp_path):
    from benchmark.harness import records

    job = tmp_path / "job"
    job.mkdir()
    job.joinpath("target.md").write_text("Target validation metric: accuracy.\n", encoding="utf-8")
    left = job / "readme-left.md"
    right = job / "readme-right.md"
    try:
        left.symlink_to("target.md")
        right.symlink_to("target.md")
    except (OSError, NotImplementedError):
        return

    sources, guidance_text = records.discover_job_guidance(job)

    assert sources == []
    assert guidance_text == ""


def test_job_guidance_skips_symlinked_docs_directory(tmp_path):
    from benchmark.harness import records

    job = tmp_path / "job"
    outside_docs = tmp_path / "outside_docs"
    job.mkdir()
    outside_docs.mkdir()
    outside_docs.joinpath("README.md").write_text("Target validation metric: accuracy.\n", encoding="utf-8")
    try:
        job.joinpath("docs").symlink_to(outside_docs, target_is_directory=True)
    except (OSError, NotImplementedError):
        return

    sources, guidance_text = records.discover_job_guidance(job)

    assert sources == []
    assert guidance_text == ""


def test_job_guidance_skips_oversized_guidance_files(tmp_path):
    from benchmark.harness.records import MAX_GUIDANCE_FILE_BYTES, discover_job_guidance

    job = tmp_path / "job"
    job.mkdir()
    with job.joinpath("README.md").open("wb") as stream:
        stream.truncate(MAX_GUIDANCE_FILE_BYTES + 1)

    sources, guidance_text = discover_job_guidance(job)

    assert sources == []
    assert guidance_text == ""


def test_job_guidance_stops_collecting_after_guidance_file_cap(tmp_path, monkeypatch):
    from benchmark.harness import records

    job = tmp_path / "job"
    job.mkdir()
    for index in range(20):
        job.joinpath(f"readme-{index:02d}.md").write_text("Target validation metric: accuracy.\n", encoding="utf-8")
    calls = {"count": 0}

    def counted_is_guidance_file(path):
        calls["count"] += 1
        return True

    monkeypatch.setattr(records, "MAX_GUIDANCE_FILES", 3)
    monkeypatch.setattr(records, "is_guidance_file", counted_is_guidance_file)

    sources, _guidance_text = records.discover_job_guidance(job)

    assert len(sources) == 3
    assert calls["count"] == 3


def test_job_guidance_metric_alignment_includes_prompt(tmp_path):
    from benchmark.harness.quality_signals import metric_signal
    from benchmark.harness.records import discover_job_guidance

    job = tmp_path / "job"
    job.mkdir()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Convert this job. Primary validation metric: AUROC.\n", encoding="utf-8")

    sources, guidance_text = discover_job_guidance(job, prompt)
    signal = metric_signal(
        sources,
        guidance_text,
        "Aggregated best validation metric: 0.7529 AUROC",
    )

    assert signal["expected_primary_metric"] == "AUROC"
    assert signal["aligned_with_job_guidance"] is True
    assert signal["sources"][0]["source_type"] == "prompt"


def test_job_guidance_metric_alignment_uses_source_priority(tmp_path):
    from benchmark.harness.quality_signals import metric_signal
    from benchmark.harness.records import discover_job_guidance

    job = tmp_path / "job"
    job.mkdir()
    job.joinpath("README.md").write_text("AUROC is the main metric.\n", encoding="utf-8")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Convert this job. Primary validation metric: accuracy.\n", encoding="utf-8")

    sources, guidance_text = discover_job_guidance(job, prompt)
    signal = metric_signal(
        sources,
        guidance_text,
        "Server best validation metric at round 3: 0.8123 accuracy",
    )

    assert signal["expected_primary_metric"] == "accuracy"
    assert signal["source"] == str(prompt)
    assert signal["matched_source"] == {"path": str(prompt), "source_type": "prompt"}
    assert signal["aligned_with_job_guidance"] is True


def test_job_guidance_metric_alignment_reports_matched_doc_source(tmp_path):
    from benchmark.harness.quality_signals import metric_signal
    from benchmark.harness.records import discover_job_guidance

    job = tmp_path / "job"
    job.mkdir()
    readme = job / "README.md"
    readme.write_text("AUROC is the main metric.\n", encoding="utf-8")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Convert this job with NVFLARE.\n", encoding="utf-8")

    sources, guidance_text = discover_job_guidance(job, prompt)
    signal = metric_signal(
        sources,
        guidance_text,
        "Aggregated best validation metric: 0.7529 AUROC",
    )

    assert signal["expected_primary_metric"] == "AUROC"
    assert signal["source"] == str(readme)
    assert signal["matched_source"] == {"path": str(readme), "source_type": "job_documentation"}
    assert signal["sources"][0]["source_type"] == "prompt"
    assert signal["aligned_with_job_guidance"] is True


def test_metric_mismatch_reports_actual_metric_without_marking_missing():
    from benchmark.harness.modes import NO_SKILLS_MODE
    from benchmark.harness.quality_signals import metric_signal
    from benchmark.harness.reports.benchmark_insights import (
        benchmark_outcome,
        human_readable_status,
        missing_result_metrics_section,
        outcome_metrics_table,
    )

    signal = metric_signal(
        None,
        "AUROC is the main metric.\n",
        "Best validation accuracy: 0.8123",
    )
    run = {
        "available": True,
        "label": "No skills baseline",
        "container_exit": {"exit_code": 0},
        "run": {"final_container_exit_code": 0},
        "record": {"quality_signals": {"job_guidance_primary_validation_metric": signal}},
        "validation_metric": signal["reported_validation_metric"],
    }
    runs = {NO_SKILLS_MODE: run}

    assert signal["mismatch"] is True
    assert signal["reported_validation_metric"]["name"] == "accuracy"
    assert "completed with metric mismatch" in human_readable_status(_ev(run))
    assert benchmark_outcome(_ev(run)).startswith("warn:")
    assert "accuracy 0.8123" in missing_result_metrics_section(_evruns(runs), [NO_SKILLS_MODE])
    assert "no parseable validation metric" not in missing_result_metrics_section(_evruns(runs), [NO_SKILLS_MODE])
    assert "| Metrics (accuracy) | accuracy 0.8123 |" in outcome_metrics_table(_evruns(runs), [NO_SKILLS_MODE])


def test_artifact_metric_satisfies_result_gate_when_final_response_metric_is_incomplete():
    from benchmark.harness.modes import WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import (
        benchmark_outcome,
        failure_analysis_section,
        human_readable_status,
        missing_result_metrics_section,
        quality_signal_table,
        run_quality_issues,
    )

    run = {
        "available": True,
        "label": "With skills",
        "container_exit": {"exit_code": 0},
        "run": {"final_container_exit_code": 0},
        "record": {
            "quality_signals": {
                "job_guidance_primary_validation_metric": {
                    "status": "missing",
                    "expected_primary_metric": "AUROC",
                    "evidence": (
                        "Job guidance declares AUROC as the primary metric, and the final response mentioned "
                        "AUROC but did not report a plausible numeric value."
                    ),
                    "reported_validation_metric": {
                        "name": "AUROC",
                        "value": None,
                        "reported_values": [],
                        "reported_value_entries": [],
                    },
                }
            }
        },
        "validation_metric": {
            "name": "AUROC",
            "source": "metrics_artifact",
            "summary_value_label": "artifact aggregated validation metric final_aggregated_metrics.[2].value",
            "value": 0.7816101804960395,
            "value_scope": "fl_summary_metric",
        },
    }
    runs = {WITH_SKILLS_MODE: run}

    assert run_quality_issues(_ev(run)) == []
    assert human_readable_status(_ev(run)) == "passed"
    assert benchmark_outcome(_ev(run), _nv_ev(run)) == "pass: scalar FL result metric available"
    assert missing_result_metrics_section(_evruns(runs), [WITH_SKILLS_MODE]) == ""
    failure_analysis = failure_analysis_section(_evruns(runs), [WITH_SKILLS_MODE])
    assert "Outcome: passed. AUROC 0.7816" in failure_analysis
    assert "Reporting note: Final response reporting gap" in failure_analysis
    quality_table = quality_signal_table(_evruns(runs), [WITH_SKILLS_MODE], _nv_ctx(runs, [WITH_SKILLS_MODE]))
    assert "artifact metric present; final response gap" in quality_table


def test_why_section_renders_when_with_skills_missing_result_even_if_faster(tmp_path):
    from benchmark.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import (
        benchmark_outcome,
        human_readable_status,
        run_quality_issues,
        why_section,
    )

    with_mode_dir = tmp_path / WITH_SKILLS_MODE
    artifact_root = with_mode_dir / "workspace_delta" / "runtime_artifacts"

    def runtime_artifact(rel_path: str, text: str):
        path = artifact_root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return {
            "artifact_path": f"runtime_artifacts/{rel_path}",
            "path": rel_path,
            "size_bytes": len(text.encode("utf-8")),
        }

    runtime_artifacts = [
        runtime_artifact(
            "runtime_workspaces/ames_scaffold/ames_scaffold/server/log_fl.txt",
            "\n".join(
                [
                    "2026-06-26 06:30:27,539 - Scaffold - INFO - Start Scaffold.",
                    "2026-06-26 06:30:27,539 - Scaffold - INFO - Round 0 started.",
                    "2026-06-26 06:30:51,982 - Scaffold - INFO - aggregating 3 update(s) at round 0",
                    "2026-06-26 06:30:52,001 - Scaffold - INFO - Round 1 started.",
                    "2026-06-26 06:30:52,002 - Scaffold - INFO - Sending task train to ['site-1', 'site-2', 'site-3']",
                ]
            ),
        ),
        runtime_artifact(
            "runtime_workspaces/ames_scaffold/ames_scaffold/server/simulate_job/metrics/round_metrics.jsonl",
            '{"round": 0, "AUROC": 0.72}\n',
        ),
        runtime_artifact("runtime_workspaces/ames_scaffold/ames_scaffold/server/error_log.txt", ""),
    ]
    background_events = [
        {
            "event_type": "assistant",
            "message": {
                "content": [
                    {
                        "id": "toolu_job",
                        "input": {
                            "command": "python3 job.py --num-sites 3 --num-rounds 3",
                            "description": "Run 3-site 3-round SCAFFOLD simulation",
                            "run_in_background": True,
                        },
                        "name": "Bash",
                        "type": "tool_use",
                    }
                ]
            },
        },
        {
            "description": "Run 3-site 3-round SCAFFOLD simulation",
            "event_type": "system.task_started",
            "task_id": "b7449z95m",
            "tool_use_id": "toolu_job",
        },
        {
            "event_type": "assistant",
            "message": {
                "content": [
                    {
                        "id": "toolu_wakeup",
                        "input": {"duration": "30s"},
                        "name": "ScheduleWakeup",
                        "type": "tool_use",
                    }
                ]
            },
        },
        {
            "event_type": "result.success",
            "harness_timestamp": "2026-06-26T06:30:46.246Z",
            "stop_reason": "end_turn",
            "subtype": "success",
            "terminal_reason": "completed",
            "type": "result",
        },
        {
            "event_type": "system.task_updated",
            "harness_timestamp": "2026-06-26T06:30:51.283Z",
            "patch": {"status": "killed"},
            "task_id": "b7449z95m",
        },
        {"event_type": "system.task_notification", "status": "stopped", "task_id": "b7449z95m"},
    ]
    runs = {
        NO_SKILLS_MODE: {
            "available": True,
            "mode": NO_SKILLS_MODE,
            "label": "No skills baseline",
            "container_exit": {"exit_code": 0},
            "run": {"final_container_exit_code": 0, "elapsed_seconds": 848, "token_count": 8_644_807},
            "record": {},
            "validation_metric": {"name": "AUROC", "value": 0.725},
        },
        WITH_SKILLS_MODE: {
            "available": True,
            "mode": WITH_SKILLS_MODE,
            "label": "With skills",
            "container_exit": {"exit_code": 0},
            "run": {"final_container_exit_code": 0, "elapsed_seconds": 630, "token_count": 4_725_568},
            "record": {},
            "activity": {"commands": ["python3 job.py --num-sites 3 --num-rounds 3"]},
            "agent_events_text": "\n".join(json.dumps(event) for event in background_events),
            "mode_dir": with_mode_dir,
            "workspace_delta": {"runtime_artifacts": runtime_artifacts},
        },
    }
    modes = [NO_SKILLS_MODE, WITH_SKILLS_MODE]
    typed = _evruns(runs)
    ctx = _nv_ctx(runs, modes)

    assert run_quality_issues(typed[NO_SKILLS_MODE], ctx.evidence[NO_SKILLS_MODE]) == []
    assert "result_metric_available" in "; ".join(
        run_quality_issues(typed[WITH_SKILLS_MODE], ctx.evidence[WITH_SKILLS_MODE])
    )
    assert human_readable_status(typed[WITH_SKILLS_MODE], ctx.evidence[WITH_SKILLS_MODE]).startswith("needs review")
    assert benchmark_outcome(typed[WITH_SKILLS_MODE], ctx.evidence[WITH_SKILLS_MODE]).startswith("fail:")
    why = why_section(typed, modes, ctx)
    assert "## Why" in why
    assert "**Why With skills needs more work**" in why
    assert "**Primary result failure**" in why
    assert "did not produce a usable FL result" in why
    assert "**Root cause of missing FL result**" in why
    assert "Interruption cause" in why
    assert "agent run ended while the background simulation was still running" in why
    assert "task was killed/stopped 5s after the agent result" in why
    assert "scheduled wakeup did not keep the non-interactive benchmark run alive" in why
    assert "Background task `b7449z95m`" in why
    assert "`killed`" in why
    assert "`stopped`" in why
    assert "`Round 1 started`" in why
    assert "`round_metrics.jsonl` was captured with 1 non-empty row" in why
    assert "`metrics_summary.json` was not captured" in why
    assert "`server/error_log.txt` is empty" in why
    assert "Result quality issue" in why
    assert "Result metric" in why
    assert "Slowdown driver comparison" not in why


def test_incomplete_background_runtime_overrides_successful_launch_status(tmp_path):
    from benchmark.harness.modes import WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import why_section
    from benchmark.harness.sdks.nvflare._logic import job_run_status, job_run_status_reason

    mode_dir = tmp_path / WITH_SKILLS_MODE
    artifact_root = mode_dir / "workspace_delta" / "runtime_artifacts"

    def runtime_artifact(rel_path: str, text: str):
        path = artifact_root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return {
            "artifact_path": f"runtime_artifacts/{rel_path}",
            "path": rel_path,
            "size_bytes": len(text.encode("utf-8")),
        }

    runtime_artifacts = [
        runtime_artifact(
            "runtime_workspaces/ames_fedavg_lightning/server/log_fl.txt",
            "\n".join(
                [
                    "2026-06-26 23:58:05,843 - FedAvg - INFO - Start FedAvg.",
                    "2026-06-26 23:58:05,843 - FedAvg - INFO - Round 0 started.",
                ]
            ),
        ),
        runtime_artifact("runtime_workspaces/ames_fedavg_lightning/server/error_log.txt", ""),
        runtime_artifact(
            "runtime_workspaces/ames_fedavg_lightning/site-1/log_fl.txt",
            "Epoch 0:  40%|####      | 36/91 [00:20<00:31, 1.74it/s]\n",
        ),
    ]
    events = [
        {
            "event_type": "assistant",
            "message": {
                "content": [
                    {
                        "id": "toolu_job",
                        "input": {
                            "command": "python3 job.py > /tmp/nvflare/sim_run.log 2>&1",
                            "description": "Re-run 3-site 3-round FedAvg simulation",
                            "run_in_background": True,
                        },
                        "name": "Bash",
                        "type": "tool_use",
                    }
                ]
            },
        },
        {
            "event_type": "user",
            "message": {
                "content": [
                    {
                        "content": "Command running in background with ID: badr1qlpf.",
                        "is_error": False,
                        "tool_use_id": "toolu_job",
                        "type": "tool_result",
                    }
                ]
            },
            "tool_use_result": {"backgroundTaskId": "badr1qlpf"},
        },
        {
            "event_type": "assistant",
            "message": {
                "content": [
                    {
                        "id": "toolu_wakeup",
                        "input": {"delaySeconds": 270},
                        "name": "ScheduleWakeup",
                        "type": "tool_use",
                    }
                ]
            },
        },
        {
            "event_type": "result.success",
            "harness_timestamp": "2026-06-26T23:58:28.557Z",
            "stop_reason": "end_turn",
            "subtype": "success",
            "terminal_reason": "completed",
            "type": "result",
        },
    ]
    run = {
        "available": True,
        "mode": WITH_SKILLS_MODE,
        "label": "With skills",
        "container_exit": {"exit_code": 0},
        "run": {"final_container_exit_code": 0, "elapsed_seconds": 997, "token_count": 5_508_753},
        "record": {},
        "activity": {
            "command_count": 1,
            "commands": ["python3 job.py > /tmp/nvflare/sim_run.log 2>&1"],
            "hint_counts": {"simulation": 1, "python_job_py": 1},
        },
        "agent_events_text": "\n".join(json.dumps(event) for event in events),
        "mode_dir": mode_dir,
        "workspace_delta": {"runtime_artifacts": runtime_artifacts},
    }

    assert job_run_status(run) == "started_failed"
    assert "no terminal `Finished` marker" in job_run_status_reason(run)

    from benchmark.harness.modes import NO_SKILLS_MODE

    runs = {
        NO_SKILLS_MODE: {
            "available": True,
            "mode": NO_SKILLS_MODE,
            "label": "No skills baseline",
            "container_exit": {"exit_code": 0},
            "run": {"final_container_exit_code": 0, "elapsed_seconds": 1321, "token_count": 4_989_109},
            "record": {},
            "validation_metric": {"name": "AUROC", "value": 0.7469},
        },
        WITH_SKILLS_MODE: run,
    }
    modes = [NO_SKILLS_MODE, WITH_SKILLS_MODE]
    typed = _evruns(runs)
    ctx = _nv_ctx(runs, modes)
    why = why_section(typed, modes, ctx)
    assert "**Root cause of missing FL result**" in why
    assert "agent run ended while the background simulation was still running" in why
    assert "scheduled wakeup did not keep the non-interactive benchmark run alive" in why
    assert "`Round 0 started`" in why
    assert "`metrics_summary.json` was not captured" in why


def test_why_section_renders_when_both_runs_fail_result_gate():
    from benchmark.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import why_section

    def failed_metric_run(mode: str, label: str, evidence: str, metric: dict):
        return {
            "available": True,
            "mode": mode,
            "label": label,
            "container_exit": {"exit_code": 0},
            "run": {"final_container_exit_code": 0, "elapsed_seconds": 100, "token_count": 1000},
            "record": {
                "quality_signals": {
                    "job_guidance_primary_validation_metric": {
                        "status": "fail",
                        "expected_primary_metric": "AUROC",
                        "evidence": evidence,
                        "reported_validation_metric": metric,
                    }
                }
            },
            "validation_metric": metric,
        }

    runs = {
        NO_SKILLS_MODE: failed_metric_run(
            NO_SKILLS_MODE,
            "No skills baseline",
            "AUROC was reported, but no FL-level scalar value was found.",
            {"name": "AUROC", "value": None, "reported_values": [0.72, 0.756]},
        ),
        WITH_SKILLS_MODE: failed_metric_run(
            WITH_SKILLS_MODE,
            "With skills",
            "final response mentioned AUROC but did not report a plausible numeric value.",
            {"name": "AUROC", "value": None, "reported_values": []},
        ),
    }

    why = why_section(_evruns(runs), [NO_SKILLS_MODE, WITH_SKILLS_MODE])

    assert "## Why" in why
    assert "**Why the comparison needs review**" in why
    assert "Both runs need review; neither side is a valid comparison winner" in why
    assert "Failed check `primary_metric_reporting`" in why
    assert "partial: 2 reported values, no single result scalar" in why
    assert "missing scalar: AUROC mentioned without value" in why
    assert "| Result metric |" in why
    assert "Why With skills needs more work" not in why


def test_metric_mismatch_evidence_includes_integer_metric_value():
    from benchmark.harness.quality_signals import format_metric_value

    assert format_metric_value(1) == " 1."
    assert format_metric_value(1.0) == " 1.0000."
    assert format_metric_value(None) == "."


def test_missing_target_metric_section_reports_observed_alternate_metrics():
    from benchmark.harness.modes import NO_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import (
        additional_or_observed_metric_values_display,
        missing_result_metrics_section,
        outcome_details_table,
    )

    run = {
        "available": True,
        "label": "No skills baseline",
        "container_exit": {"exit_code": 0},
        "run": {"final_container_exit_code": 0},
        "record": {
            "quality_signals": {
                "job_guidance_primary_validation_metric": {
                    "status": "missing",
                    "expected_primary_metric": "AUROC",
                    "evidence": "Job guidance declares AUROC as the primary metric, but the final response did not report it.",
                    "reported_validation_metric": {
                        "name": None,
                        "value": None,
                        "reported_values": [],
                        "reported_value_entries": [],
                    },
                }
            }
        },
        "validation_metric": {"name": None, "value": None, "reported_values": [], "reported_value_entries": []},
        "agent_last_message": "Validation accuracy: 0.8123\nValidation loss: 0.421",
    }

    section = missing_result_metrics_section(_evruns({NO_SKILLS_MODE: run}), [NO_SKILLS_MODE])

    assert "accuracy 0.8123" in section
    assert "loss 0.4210" in section
    assert "no parseable validation metric" not in section
    assert additional_or_observed_metric_values_display(_ev(run), "AUROC") == "accuracy 0.8123; loss 0.4210"
    assert "Additional/other validation metric values" in outcome_details_table(
        _evruns({NO_SKILLS_MODE: run}), [NO_SKILLS_MODE]
    )


def test_outcome_details_reports_observed_metrics_from_runtime_artifact(tmp_path):
    from benchmark.harness.modes import WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import additional_or_observed_metric_values_display

    mode_dir = tmp_path / "mode=with_skills"
    artifact = mode_dir / "workspace_delta" / "runtime_artifacts" / "metrics_summary.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(
            {
                "final_aggregated_metrics": [
                    {"name": "train_loss", "value": 0.4508},
                    {"name": "val_loss", "value": 0.5386},
                    {"name": "val_accuracy", "value": 0.7079},
                    {"name": "val_auroc", "value": 0.7757},
                ]
            }
        ),
        encoding="utf-8",
    )
    run = {
        "available": True,
        "mode": WITH_SKILLS_MODE,
        "label": "With skills",
        "mode_dir": mode_dir,
        "workspace_delta": {
            "delta_dir": "/workspace/results/workspace_delta",
            "runtime_artifacts": [{"artifact_path": "runtime_artifacts/metrics_summary.json"}],
        },
        "validation_metric": {
            "name": "AUROC",
            "value": 0.7757,
            "reported_values": [0.7757],
            "reported_value_entries": [{"label": "artifact aggregated validation metric", "value": 0.7757}],
        },
        "agent_last_message": "Final AUROC: 0.7757",
    }

    display = additional_or_observed_metric_values_display(_ev(run), "AUROC")

    assert "AUROC 0.7757" in display
    assert "accuracy 0.7079" in display
    assert "loss 0.5386" in display


def test_failure_analysis_reports_recovered_job_failure_and_metric_gap():
    from benchmark.harness.modes import NO_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import (
        additional_or_observed_metric_values_display,
        failure_analysis_section,
        outcome_details_table,
    )

    failed_output = (
        "TypeError: SmilesCNN.__init__() missing 4 required positional arguments: "
        "'vocab_size', 'embed_dim', 'num_filters', and 'dropout'\n"
        "RuntimeError: Simulator run failed with exit code 2.\n"
    )
    success_output = (
        "Finished FedAvg.\n"
        "site-1: round=0 train_loss=0.6275 valid_auroc=0.7049\n"
        "site-2: round=0 train_loss=0.6259 valid_auroc=0.7342\n"
        "Result workspace: /tmp/agent_benchmark/ames-smoke\n"
    )
    events = [
        {
            "item": {
                "type": "command_execution",
                "id": "item_1",
                "command": "python3 fedavg_job.py --n-clients 2",
                "status": "failed",
                "exit_code": 1,
                "aggregated_output": failed_output,
            }
        },
        {
            "item": {
                "type": "command_execution",
                "id": "item_2",
                "command": "python3 fedavg_job.py --n-clients 2",
                "status": "completed",
                "exit_code": 0,
                "aggregated_output": success_output,
            }
        },
    ]
    run = {
        "available": True,
        "label": "No skills baseline",
        "container_exit": {"exit_code": 0},
        "run": {"final_container_exit_code": 0},
        "record": {
            "quality_signals": {
                "job_guidance_primary_validation_metric": {
                    "status": "missing",
                    "expected_primary_metric": "AUROC",
                    "evidence": "Job guidance declares AUROC as the primary metric, but the final response did not report it.",
                    "reported_validation_metric": {
                        "name": None,
                        "value": None,
                        "reported_values": [],
                        "reported_value_entries": [],
                    },
                }
            }
        },
        "validation_metric": {"name": None, "value": None, "reported_values": [], "reported_value_entries": []},
        "agent_events_text": "\n".join(json.dumps(event) for event in events),
    }

    section = failure_analysis_section(
        _evruns({NO_SKILLS_MODE: run}), [NO_SKILLS_MODE], _nv_ctx({NO_SKILLS_MODE: run}, [NO_SKILLS_MODE])
    )

    assert "Command Evidence" in section
    assert "recovered by a later successful similar command" in section
    assert "SmilesCNN.__init__() missing 4 required positional arguments" in section
    assert "Recovery evidence" in section
    assert "a later job/simulator command exited 0" in section
    assert "valid_auroc=0.7049" in section
    assert "Metric reporting gap" in section
    assert "aggregate `AUROC` scalar" in section
    assert additional_or_observed_metric_values_display(
        _ev(run), "AUROC", _nv_ev(run), _nv_ctx({NO_SKILLS_MODE: run}, [NO_SKILLS_MODE])
    ) == (
        "Final site metrics=NA; log/per-site evidence: site-1: round=0 train_loss=0.6275 valid_auroc=0.7049; "
        "site-2: round=0 train_loss=0.6259 valid_auroc=0.7342"
    )
    details = outcome_details_table(
        _evruns({NO_SKILLS_MODE: run}), [NO_SKILLS_MODE], _nv_ctx({NO_SKILLS_MODE: run}, [NO_SKILLS_MODE])
    )
    assert "Reported validation metric | AUROC NA" in details
    assert "log/per-site evidence" in details


def test_job_run_status_section_reports_bash_blocked_not_started(tmp_path):
    from benchmark.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import (
        bash_blocked_diagnostic,
        benchmark_report,
        job_run_status_section,
    )
    from benchmark.harness.sdks.nvflare._logic import job_run_status

    permission_result = {
        "message": {
            "content": [
                {
                    "content": "Claude requested permissions to use Bash, but you haven't granted it yet.",
                    "type": "tool_result",
                }
            ]
        },
        "type": "user",
    }
    final_result = {
        "final_message": "Usage:\n```bash\ncd nvflare_jobs/ames_fedavg\npython job.py\n```",
        "permission_denials": [{"tool_name": "Bash", "tool_input": {"command": "python job.py"}}],
        "subtype": "success",
        "type": "result",
    }
    run = {
        "available": True,
        "label": "With skills",
        "activity": {
            "commands": ["find /workspace/run/with_skills/workspace -type f"],
            "hint_counts": {"python_job_py": 0, "simulation": 0},
        },
        "agent_events_text": "\n".join(json.dumps(event) for event in (permission_result, final_result)),
        "agent_last_message": final_result["final_message"],
        "container_exit": {"exit_code": 0},
        "record": {},
        "run": {"final_container_exit_code": 0},
    }
    runs = {
        NO_SKILLS_MODE: {"available": False, "label": "No skills baseline"},
        WITH_SKILLS_MODE: run,
    }

    assert job_run_status(run) == "not_started"

    section = job_run_status_section(
        _evruns(runs), [NO_SKILLS_MODE, WITH_SKILLS_MODE], _nv_ctx(runs, [NO_SKILLS_MODE, WITH_SKILLS_MODE])
    )
    # benchmark_report resolves its plugin from the captured report identity; stamp
    # the NVFLARE descriptor so the full-report path uses NVFLARE rendering
    # (Inversion 3: an unidentified result root resolves to the null plugin).
    (tmp_path / "benchmark_profile_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sdk_name": "nvflare",
                "benchmark_profile_id": "nvflare",
                "report_plugin_id": "nvflare",
                "capture_spec_version": "1",
            }
        ),
        encoding="utf-8",
    )
    report = benchmark_report(tmp_path, runs)
    assert "## Job Run Status" not in report
    assert "| No skills baseline | missing | unknown | fail | missing |" in report
    assert "- Job: unknown: run artifacts not available" in report
    assert "- Result gate: fail: run artifacts missing" in report
    assert "not_started: Bash blocked 1 time(s)" in report
    assert "| With skills | not_started | Bash blocked 1 time(s)" in section
    assert "Fix agent Bash/tool permissions and rerun" in section
    diagnostic = bash_blocked_diagnostic(_ev(run))
    assert "--tools" in diagnostic
    assert "--allowedTools" not in diagnostic
    assert "python job.py" not in section


def test_job_run_status_uses_claude_bash_output_to_detect_completed_simulation():
    from benchmark.harness.sdks.nvflare._logic import job_run_status, job_run_status_reason

    tool_id = "toolu_job"
    command_event = {
        "event_type": "assistant",
        "message": {
            "content": [
                {
                    "id": tool_id,
                    "input": {"command": "timeout 300 python job.py --num-rounds 1"},
                    "name": "Bash",
                    "type": "tool_use",
                }
            ]
        },
    }
    result_event = {
        "event_type": "user",
        "message": {
            "content": [
                {
                    "content": "site-1: round=0 train_loss=0.5440 valid_auroc=0.7254\nFinished FedAvg.\nSimulation workspace: /tmp/nvflare/workspaces/ames_fedavg",
                    "is_error": False,
                    "tool_use_id": tool_id,
                    "type": "tool_result",
                }
            ]
        },
        "tool_use_result": {
            "interrupted": False,
            "stderr": "",
            "stdout": "site-1: round=0 train_loss=0.5440 valid_auroc=0.7254\nFinished FedAvg.\nSimulation workspace: /tmp/nvflare/workspaces/ames_fedavg",
        },
    }
    run = {
        "available": True,
        "activity": {"commands": ["timeout 300 python job.py --num-rounds 1"]},
        "agent_events_text": "\n".join(json.dumps(event) for event in (command_event, result_event)),
    }

    assert job_run_status(run) == "completed"
    assert job_run_status_reason(run) == "simulation completed — FL workflow reached Finished state"


def test_agent_command_spans_pair_claude_bash_tool_use_and_result():
    from benchmark.harness.reports.benchmark_insights import agent_command_spans

    tool_id = "toolu_install"
    command_event = {
        "event_type": "assistant",
        "harness_timestamp": "2026-06-13T00:00:00Z",
        "message": {
            "content": [
                {
                    "id": tool_id,
                    "input": {"command": "uv pip install -r requirements.txt"},
                    "name": "Bash",
                    "type": "tool_use",
                }
            ]
        },
    }
    result_event = {
        "event_type": "user",
        "harness_timestamp": "2026-06-13T00:00:25Z",
        "message": {
            "content": [
                {
                    "content": "Successfully installed dependencies",
                    "is_error": False,
                    "tool_use_id": tool_id,
                    "type": "tool_result",
                }
            ]
        },
        "tool_use_result": {
            "interrupted": False,
            "stderr": "",
            "stdout": "Successfully installed dependencies",
        },
    }
    run = {"agent_events_text": "\n".join(json.dumps(event) for event in (command_event, result_event))}

    spans = agent_command_spans(run)

    assert spans == [
        {
            "command": "uv pip install -r requirements.txt",
            "description": "",
            "duration_seconds": 25.0,
            "exit_code": 0,
            "id": tool_id,
            "index": 0,
            "output": "Successfully installed dependencies",
            "status": "completed",
        }
    ]


def test_job_run_status_detects_generated_simulation_entrypoint():
    from benchmark.harness.sdks.nvflare._logic import job_run_status, job_run_status_reason

    event = {
        "item": {
            "aggregated_output": (
                "Finished FedAvg.\n"
                "Simulation workspace: outputs/nvflare_workspace/ames_fedavg\n"
                "Final weighted validation metrics: AUROC 0.7592"
            ),
            "command": "python3 run_nvflare_simulation.py",
            "exit_code": 0,
            "id": "item_1",
            "status": "completed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["python3 run_nvflare_simulation.py"]},
        "agent_events_text": json.dumps(event),
    }

    assert job_run_status(run) == "completed"
    assert job_run_status_reason(run) == "simulation completed — FL workflow reached Finished state"


def test_job_run_status_detects_wrapper_that_invokes_nvflare_simulator():
    from benchmark.harness.sdks.nvflare._logic import job_run_status, job_run_status_reason

    event = {
        "item": {
            "aggregated_output": (
                "Running: /workspace/venv/bin/python3 -m nvflare.cli simulator "
                "/workspace/fl_job/ames_fedavg -w /workspace/fl_workspace -n 3 -t 3\n"
                "Finished FedAvg.\n"
                "Simulation workspace: /workspace/fl_workspace\n"
            ),
            "command": "python3 run_nvflare_fedavg.py",
            "exit_code": 0,
            "id": "item_1",
            "status": "completed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["python3 run_nvflare_fedavg.py"]},
        "agent_events_text": json.dumps(event),
    }

    assert job_run_status(run) == "completed"
    assert job_run_status_reason(run) == "simulation completed — FL workflow reached Finished state"


def test_job_run_status_uses_metric_artifact_to_avoid_not_started_contradiction():
    from benchmark.harness.sdks.nvflare._logic import job_run_status, job_run_status_reason

    run = {
        "available": True,
        "activity": {"commands": ["/bin/bash -lc 'rg --files'"]},
        "validation_metric": {
            "name": "AUROC",
            "reported_values": [0.7652],
            "source": "metrics_artifact",
            "source_path": "/workspace/results/workspace_delta/runtime_artifacts/server/metrics/summary.json",
        },
    }

    assert job_run_status(run) == "completed"
    reason = job_run_status_reason(run)
    assert "job execution inferred from captured runtime metric artifact" in reason
    assert "summary.json" in reason
    assert "command detector did not identify" in reason


def test_job_run_status_does_not_infer_completion_from_changed_file_metric_artifact():
    from benchmark.harness.sdks.nvflare._logic import job_run_status

    run = {
        "available": True,
        "activity": {"commands": ["/bin/bash -lc 'rg --files'"]},
        "validation_metric": {
            "name": "AUROC",
            "reported_values": [0.7652],
            "source": "metrics_artifact",
            "source_path": (
                "/workspace/results/workspace_delta/changed_files/fl_workspace/ames_fedavg/"
                "server/simulate_job/metrics/metrics_summary.json"
            ),
        },
    }

    assert job_run_status(run) == "not_started"


def test_job_run_status_ignores_successful_simulation_helper_scripts():
    from benchmark.harness.sdks.nvflare._logic import job_run_status

    events = [
        {
            "item": {
                "aggregated_output": "simulation config ok",
                "command": "python check_simulation_config.py",
                "exit_code": 0,
                "id": "item_1",
                "status": "completed",
                "type": "command_execution",
            }
        },
        {
            "item": {
                "aggregated_output": "simulator import ok",
                "command": "python validate_simulator_install.py",
                "exit_code": 0,
                "id": "item_2",
                "status": "completed",
                "type": "command_execution",
            }
        },
        {
            "item": {
                "aggregated_output": "nvflare import ok",
                "command": "python check_nvflare_install.py",
                "exit_code": 0,
                "id": "item_3",
                "status": "completed",
                "type": "command_execution",
            }
        },
        {
            "item": {
                "aggregated_output": "job config ok",
                "command": "python validate_job_config.py",
                "exit_code": 0,
                "id": "item_4",
                "status": "completed",
                "type": "command_execution",
            }
        },
        {
            "item": {
                "aggregated_output": "job setup ok",
                "command": "python check_job_setup.py",
                "exit_code": 0,
                "id": "item_5",
                "status": "completed",
                "type": "command_execution",
            }
        },
        {
            "item": {
                "aggregated_output": "job helper ok",
                "command": "python validate_job.py",
                "exit_code": 0,
                "id": "item_6",
                "status": "completed",
                "type": "command_execution",
            }
        },
        {
            "item": {
                "aggregated_output": "job helper ok",
                "command": "python check_job.py",
                "exit_code": 0,
                "id": "item_7",
                "status": "completed",
                "type": "command_execution",
            }
        },
        {
            "item": {
                "aggregated_output": "job tests passed",
                "command": "python run_job_tests.py",
                "exit_code": 0,
                "id": "item_8",
                "status": "completed",
                "type": "command_execution",
            }
        },
    ]
    run = {
        "available": True,
        "activity": {
            "commands": [
                "python check_simulation_config.py",
                "python validate_simulator_install.py",
                "python check_nvflare_install.py",
                "python validate_job_config.py",
                "python check_job_setup.py",
                "python validate_job.py",
                "python check_job.py",
                "python run_job_tests.py",
            ]
        },
        "agent_events_text": "\n".join(json.dumps(event) for event in events),
    }

    assert job_run_status(run) == "not_started"


def test_why_slower_reports_long_running_command_spans(tmp_path):
    from benchmark.harness.reports.benchmark_insights import _why_slower

    def command_events(command, start, end, item_id="item_1", output="ok", exit_code=0):
        status = "completed" if exit_code == 0 else "failed"
        return [
            {
                "timestamp": start,
                "type": "item.started",
                "item": {
                    "command": command,
                    "id": item_id,
                    "status": "in_progress",
                    "type": "command_execution",
                },
            },
            {
                "timestamp": end,
                "type": "item.completed",
                "item": {
                    "aggregated_output": output,
                    "command": command,
                    "exit_code": exit_code,
                    "id": item_id,
                    "status": status,
                    "type": "command_execution",
                },
            },
        ]

    def source_fields(mode: str, source: str) -> dict:
        mode_dir = tmp_path / mode
        source_path = mode_dir / "workspace_delta" / "changed_files" / "client.py"
        source_path.parent.mkdir(parents=True)
        source_path.write_text(source, encoding="utf-8")
        return {
            "mode_dir": mode_dir,
            "workspace_delta": {
                "changed_files": [
                    {
                        "artifact_path": "changed_files/client.py",
                        "path": "client.py",
                    }
                ]
            },
        }

    cuda_install_output = (
        "Downloading nvidia-cublas (517.7MiB)\n"
        "WARNING: Connection timed out while downloading.\n"
        "WARNING: Attempting to resume incomplete download (211.0 MB/426.4 MB, attempt 1)\n"
        "WARNING: Retrying after connection broken by "
        "NameResolutionError(\"Failed to resolve 'files.pythonhosted.org'\")\n"
        "Downloading nvidia-cudnn-cu13 (424.0MiB)\n"
        "Downloading triton (179.7MiB)\n"
        "Installed torch==2.12.0\n"
    )
    in_process_output = (
        "2026-06-13 08:50:40,563 - INFO - Round 1 started.\n"
        "2026-06-13 09:06:12,642 - INFO - [server] download tx T1 done: status=finished elapsed=905.27s\n"
        "2026-06-13 09:06:28,198 - INFO - Aggregated 3/3 results\n"
        "PTInProcessClientAPIExecutor - INFO - Waiting for result from peer\n"
        "Finished FedAvg.\n"
    )
    simulator_output = (
        "Running: /workspace/venv/bin/python3 -m nvflare.cli simulator /workspace/fl_job -w /workspace/fl_workspace -n 3 -t 3\n"
        "2026-06-13 08:01:40,056 - INFO - Round 2 started.\n"
        "2026-06-13 08:01:45,239 - INFO - [server] download tx T2 done: status=finished elapsed=8.53s\n"
        "2026-06-13 08:02:01,678 - INFO - Aggregated 3/3 results\n"
        "PTClientAPILauncherExecutor - INFO - received result\n"
        "Finished FedAvg.\n"
        "Simulation workspace: /workspace/fl_workspace\n"
    )
    with_events = command_events(
        "uv pip install -r requirements-train.txt",
        "2026-06-13T08:00:00Z",
        "2026-06-13T08:20:00Z",
        output=cuda_install_output,
    ) + command_events(
        "python job.py",
        "2026-06-13T08:21:00Z",
        "2026-06-13T08:51:00Z",
        item_id="item_2",
        output=in_process_output,
    )
    base_events = (
        command_events(
            "python3 -m pip install -r requirements-train.txt",
            "2026-06-13T08:00:00Z",
            "2026-06-13T08:03:22Z",
            output="Downloading torch and nvidia-cudnn-cu13\n",
            exit_code=-1,
        )
        + command_events(
            "python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu",
            "2026-06-13T08:03:30Z",
            "2026-06-13T08:03:57Z",
            item_id="item_2",
            output="Successfully installed torch-2.12.0+cpu\n",
        )
        + command_events(
            "python3 run_nvflare_fedavg.py",
            "2026-06-13T08:04:00Z",
            "2026-06-13T08:06:00Z",
            item_id="item_3",
            output=simulator_output,
        )
    )
    with_run = {
        "label": "With skills",
        "mode": "with_skills",
        "run": {"elapsed_seconds": 3200},
        "activity": {},
        "agent_events_text": "\n".join(json.dumps(event) for event in with_events),
        **source_fields(
            "with_skills",
            """
import nvflare.client as flare
train_frame = load_split(args.data_dir, "train")
train_loader = DataLoader(train_frame)
while flare.is_running():
    criterion, optimizer = build_loss_and_optimizer(model, train_frame, args, device)
    test_metrics = evaluate(model, test_loader, criterion, device)
    append_record(results_path, {"metrics": test_metrics})
""",
        ),
    }
    base_run = {
        "label": "No skills baseline",
        "mode": "without_skills",
        "run": {"elapsed_seconds": 300},
        "activity": {},
        "agent_events_text": "\n".join(json.dumps(event) for event in base_events),
        **source_fields(
            "without_skills",
            """
import nvflare.client as flare
train_frame = load_split(args.data_dir, "train")
train_loader = DataLoader(train_frame)
criterion, optimizer = build_loss_and_optimizer(model, train_frame, args, device)
while flare.is_running():
    train_one_epoch(model, train_loader, criterion, optimizer, device)
""",
        ),
    }

    explanation = "\n".join(
        _why_slower(
            _ev(with_run),
            _ev(base_run),
            _nv_ctx({"with_skills": with_run, "without_skills": base_run}),
        )
    )

    assert "Slowdown driver comparison" in explanation
    assert "captured command time contributing to wall-clock slowdown" in explanation
    assert "| Captured command time | 3000s | 349s | +2651s |" in explanation
    assert "Elapsed time accounting" in explanation
    assert "| Run | Total | Dependency install | Runtime after install | Captured non-install commands |" in explanation
    assert "| With skills | 3200s | 1200s | 2000s | 1800s |" in explanation
    assert "| No skills baseline | 300s | 229s | 71s | 120s |" in explanation
    assert "Captured command spans identify slow operations but are not guaranteed to add up exactly" in explanation
    assert "Longest command comparison" in explanation
    assert "| Rank | With skills | No skills baseline |" in explanation
    assert "| 1 | `python job.py` (1800s, exit 0) | `python3 -m pip install -r requirements-train.txt`" in explanation
    assert "| 2 | `uv pip install -r requirements-train.txt`" in explanation
    assert "`python3 run_nvflare_fedavg.py` (120s, exit 0)" in explanation
    assert "uv pip install -r requirements-train.txt" in explanation
    assert "python job.py" in explanation
    assert "Dependency install path differed" in explanation
    assert "requirements-file install" in explanation
    assert "Captured package examples" in explanation
    assert "nvidia-cublas" in explanation
    assert "Installer difference" in explanation
    assert "Network/download evidence" in explanation
    assert "connection timeout" in explanation
    assert "resumed incomplete download" in explanation
    assert "DNS resolution failure" in explanation
    assert "baseline install logs showed no captured network retry/timeout markers" in explanation
    assert "targeted package install" in explanation
    assert "NVFLARE runtime path diverged" in explanation
    assert "| Run | Runtime path | Successful runs | Total captured time | Representative command |" in explanation
    assert (
        "| With skills | `recipe.execute(SimEnv(...))` with `PTInProcessClientAPIExecutor` | 1 command | 1800s |"
        in explanation
    )
    assert (
        "| No skills baseline | exported job + `nvflare.cli simulator ... -t 3` with external client processes | 1 command | 120s |"
        in explanation
    )
    assert "recipe.execute(SimEnv(...))" in explanation
    assert "PTInProcessClientAPIExecutor" in explanation
    assert "nvflare.cli simulator ... -t 3" in explanation
    assert "external client processes" in explanation
    assert "Slow FL round evidence" in explanation
    assert "Transfer/wait evidence" in explanation
    assert "training/validation work, NVFLARE result transfer, synchronization wait" in explanation
    assert "should be investigated separately from generated-code efficiency" in explanation
    assert "Generated-code efficiency issue aligns with slower non-install runtime" in explanation
    assert "loss/optimizer lifecycle" in explanation
    assert "Quality-versus-speed tradeoff: useful validation work also adds per-round workload" in explanation
    assert "may explain part of the long per-round wait" in explanation
    assert "Dependency cost is separate from code efficiency" in explanation


def test_why_section_reports_runtime_regression_when_total_time_is_not_slower():
    from benchmark.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import why_section

    def command_events(command, start, end, item_id, output="ok"):
        return [
            {
                "timestamp": start,
                "type": "item.started",
                "item": {
                    "command": command,
                    "id": item_id,
                    "status": "in_progress",
                    "type": "command_execution",
                },
            },
            {
                "timestamp": end,
                "type": "item.completed",
                "item": {
                    "aggregated_output": output,
                    "command": command,
                    "exit_code": 0,
                    "id": item_id,
                    "status": "completed",
                    "type": "command_execution",
                },
            },
        ]

    with_events = command_events(
        "uv pip install -r requirements-train.txt",
        "2026-06-13T00:00:00Z",
        "2026-06-13T00:00:50Z",
        "with_install",
    ) + command_events(
        "python job.py",
        "2026-06-13T00:01:00Z",
        "2026-06-13T00:07:40Z",
        "with_job",
        output="Finished FedAvg.\n",
    )
    base_events = command_events(
        "python -m pip install -r requirements-train.txt",
        "2026-06-13T00:00:00Z",
        "2026-06-13T00:05:00Z",
        "base_install",
    ) + command_events(
        "python run_job.py",
        "2026-06-13T00:05:05Z",
        "2026-06-13T00:05:45Z",
        "base_job",
        output="Finished FedAvg.\n",
    )
    runs = {
        NO_SKILLS_MODE: {
            "label": "No skills baseline",
            "run": {"elapsed_seconds": 600, "token_count": 1000},
            "activity": {"event_types": {"assistant": 2}},
            "agent_events_text": "\n".join(json.dumps(event) for event in base_events),
        },
        WITH_SKILLS_MODE: {
            "label": "With skills",
            "run": {"elapsed_seconds": 500, "token_count": 1000},
            "activity": {"event_types": {"assistant": 5}},
            "agent_events_text": "\n".join(json.dumps(event) for event in with_events),
        },
    }

    section = why_section(_evruns(runs), [NO_SKILLS_MODE, WITH_SKILLS_MODE])

    assert "## Why" in section
    assert "Why With skills has longer runtime after install" in section
    assert "| With skills | 500s | 50s | 450s | 400s |" in section
    assert "| No skills baseline | 600s | 300s | 300s | 40s |" in section
    assert "Slowdown driver comparison" in section
    assert "| Captured non-install command time | 400s | 40s | +360s |" in section
    assert "450s vs 340s Captured command time" not in section
    assert "captured non-install command time contributing to runtime-after-install regression" in section
    assert "extra wall time came from tools" not in section
    assert "| Assistant turns | 5 | 2 | +3 | extra model round-trips |" in section
    assert "wall-clock overhead;" not in section


def test_why_section_reports_dependency_install_regression_when_runtime_is_not_slower():
    from benchmark.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import why_section

    def command_events(command, start, end, item_id, output="ok"):
        return [
            {
                "timestamp": start,
                "type": "item.started",
                "item": {
                    "command": command,
                    "id": item_id,
                    "status": "in_progress",
                    "type": "command_execution",
                },
            },
            {
                "timestamp": end,
                "type": "item.completed",
                "item": {
                    "aggregated_output": output,
                    "command": command,
                    "exit_code": 0,
                    "id": item_id,
                    "status": "completed",
                    "type": "command_execution",
                },
            },
        ]

    with_events = command_events(
        "uv pip install -r requirements-train.txt",
        "2026-06-13T00:00:00Z",
        "2026-06-13T00:02:30Z",
        "with_install",
        output=(
            "Downloading pytorch_lightning (2.4MiB)\n"
            "Successfully installed nvidia-cublas-cu13 triton pytorch_lightning torchmetrics\n"
        ),
    ) + command_events(
        "python job.py",
        "2026-06-13T00:02:35Z",
        "2026-06-13T00:03:05Z",
        "with_job",
        output="Finished FedAvg.\n",
    )
    base_events = command_events(
        "python -m pip install torch --index-url https://download.pytorch.org/whl/cpu",
        "2026-06-13T00:00:00Z",
        "2026-06-13T00:00:30Z",
        "base_install",
        output="Successfully installed torch-2.12.0+cpu\n",
    ) + command_events(
        "python run_job.py",
        "2026-06-13T00:00:35Z",
        "2026-06-13T00:02:15Z",
        "base_job",
        output="Finished FedAvg.\n",
    )
    runs = {
        NO_SKILLS_MODE: {
            "label": "No skills baseline",
            "run": {"elapsed_seconds": 250, "token_count": 1000},
            "activity": {"event_types": {"assistant": 2}},
            "agent_events_text": "\n".join(json.dumps(event) for event in base_events),
        },
        WITH_SKILLS_MODE: {
            "label": "With skills",
            "run": {"elapsed_seconds": 200, "token_count": 1000},
            "activity": {"event_types": {"assistant": 2}},
            "agent_events_text": "\n".join(json.dumps(event) for event in with_events),
        },
    }

    section = why_section(_evruns(runs), [NO_SKILLS_MODE, WITH_SKILLS_MODE])

    assert "## Why" in section
    assert "Why With skills has longer dependency install" in section
    assert "| With skills | 150s | 50s | 200s |" in section
    assert "| No skills baseline | 30s | 220s | 250s |" in section
    assert "a run can finish faster overall while still spending more time installing dependencies" in section
    assert "Dependency install path differed" in section
    assert "| Run | Install time | Install scope | Stack evidence | Installer | Representative command |" in section
    assert "accelerator-capable dependency stack" in section
    assert "CPU-only framework wheel" in section
    assert "requirements-file install" in section
    assert "targeted package install" in section
    assert "Captured package examples" in section
    assert "pytorch_lightning" in section
    assert "Accelerator dependency evidence" in section
    assert "explicit CPU-only PyTorch wheel index" in section
    assert "Why With skills is still faster overall" in section
    assert "Why With skills is slower" not in section
    assert "Why With skills has longer runtime after install" not in section


def test_runtime_path_note_includes_baseline_fallback_command_time():
    # E3: the runtime-path note is now owned by NvflareReportPlugin.explain() and
    # contributed to the "why_slowdown" anchor. Assert against the plugin fragments
    # (joined back to the engine-rendered list shape) rather than the removed
    # engine-local _runtime_path_slowdown_note.
    from benchmark.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from benchmark.harness.reports.evidence import SCHEMA_VERSION, ComparisonEvidence
    from benchmark.harness.sdks.nvflare.plugin import NvflareReportPlugin

    def command_events(command, start, end, item_id, output="ok"):
        return [
            {
                "timestamp": start,
                "type": "item.started",
                "item": {
                    "command": command,
                    "id": item_id,
                    "status": "in_progress",
                    "type": "command_execution",
                },
            },
            {
                "timestamp": end,
                "type": "item.completed",
                "item": {
                    "aggregated_output": output,
                    "command": command,
                    "exit_code": 0,
                    "id": item_id,
                    "status": "completed",
                    "type": "command_execution",
                },
            },
        ]

    with_run = {
        "label": "With skills",
        "mode": "with_skills",
        "agent_events_text": "\n".join(
            json.dumps(event)
            for event in command_events(
                "python job.py",
                "2026-06-13T08:00:00Z",
                "2026-06-13T08:14:02Z",
                "with_job",
                output="PTInProcessClientAPIExecutor - INFO - result received\nFinished FedAvg.\n",
            )
        ),
    }
    base_run = {
        "label": "No skills baseline",
        "mode": "without_skills",
        "agent_events_text": "\n".join(
            json.dumps(event)
            for event in (
                command_events(
                    "python run_experiment.py",
                    "2026-06-13T08:00:00Z",
                    "2026-06-13T08:03:00Z",
                    "base_cmd",
                )
                + command_events(
                    "nl -ba nvflare_scaffold_job.py | sed -n '1,240p'",
                    "2026-06-13T08:04:00Z",
                    "2026-06-13T08:04:01Z",
                    "inspect_job_source",
                    output='print("Finished FedAvg.")\ncmd = "python -m nvflare.cli simulator job -w workspace"\n',
                )
            )
        ),
    }

    plugin = NvflareReportPlugin()
    modes = [WITH_SKILLS_MODE, NO_SKILLS_MODE]
    runs = {WITH_SKILLS_MODE: _ev(with_run), NO_SKILLS_MODE: _ev(base_run)}
    cmp = ComparisonEvidence(
        schema_version=SCHEMA_VERSION,
        runs=runs,
        modes=modes,
        sdk_metadata={},
    )
    evidence = {mode: plugin.collect(runs[mode]) for mode in modes}
    note = "\n".join(fragment.text for fragment in plugin.explain(cmp, evidence))

    assert "NVFLARE runtime path diverged" in note
    assert "| Run | Runtime path | Successful runs | Total captured time | Representative command |" in note
    assert (
        "| With skills | `recipe.execute(SimEnv(...))` with `PTInProcessClientAPIExecutor` | 1 command | 842s |" in note
    )
    assert "| No skills baseline | no classified successful job/simulator command | 0 commands | NA |" in note
    assert "`python job.py` (842s, exit 0)" in note
    assert "no classified successful job/simulator command" in note
    assert "`python run_experiment.py` (180s, exit 0)" in note
    assert "nl -ba nvflare_scaffold_job.py" not in note


def test_runtime_path_note_explains_slow_lightning_round_against_prior_rounds(tmp_path):
    from benchmark.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from benchmark.harness.reports.evidence import SCHEMA_VERSION, ComparisonEvidence
    from benchmark.harness.sdks.nvflare.plugin import NvflareReportPlugin

    def command_events(command, start, end, item_id, output="ok"):
        return [
            {
                "timestamp": start,
                "type": "item.started",
                "item": {
                    "command": command,
                    "id": item_id,
                    "status": "in_progress",
                    "type": "command_execution",
                },
            },
            {
                "timestamp": end,
                "type": "item.completed",
                "item": {
                    "aggregated_output": output,
                    "command": command,
                    "exit_code": 0,
                    "id": item_id,
                    "status": "completed",
                    "type": "command_execution",
                },
            },
        ]

    mode_dir = tmp_path / WITH_SKILLS_MODE
    client_path = mode_dir / "workspace_delta" / "changed_files" / "client.py"
    client_path.parent.mkdir(parents=True)
    client_path.write_text(
        """
import pytorch_lightning as pl
import nvflare.client.lightning as flare

trainer = pl.Trainer(max_epochs=args.epochs)
flare.patch(trainer)

while flare.is_running():
    trainer.validate(model, datamodule=datamodule, verbose=False)
    trainer.fit(model, datamodule=datamodule)
""",
        encoding="utf-8",
    )
    runtime_artifacts = []
    for site, third_round_end in {
        "site-1": "2026-06-13 08:15:40,000",
        "site-2": "2026-06-13 08:15:41,000",
        "site-3": "2026-06-13 08:15:39,000",
    }.items():
        log_path = mode_dir / "workspace_delta" / "runtime_artifacts" / "runtime_workspaces" / "job" / site / "log.txt"
        log_path.parent.mkdir(parents=True)
        log_path.write_text(
            "\n".join(
                [
                    f"2026-06-13 08:00:00,000 - TaskScriptRunner - INFO - {site} | round=1 task=train",
                    "2026-06-13 08:00:16,000 - rank_zero - INFO - `Trainer.fit` stopped: `max_epochs=1` reached.",
                    f"2026-06-13 08:00:20,000 - TaskScriptRunner - INFO - {site} | round=2 task=train",
                    "2026-06-13 08:00:36,000 - rank_zero - INFO - `Trainer.fit` stopped: `max_epochs=2` reached.",
                    f"2026-06-13 08:00:40,000 - TaskScriptRunner - INFO - {site} | round=3 task=train",
                    f"{third_round_end} - rank_zero - INFO - `Trainer.fit` stopped: `max_epochs=3` reached.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        config_path = (
            mode_dir
            / "workspace_delta"
            / "runtime_artifacts"
            / "runtime_workspaces"
            / "job"
            / site
            / "simulate_job"
            / f"app_{site}"
            / "config"
            / "config_fed_client.json"
        )
        config_path.parent.mkdir(parents=True)
        config_path.write_text(
            json.dumps(
                {
                    "executors": [
                        {
                            "executor": {
                                "args": {
                                    "task_script_args": "--data-path /workspace/input --epochs 1 --device cpu"
                                }
                            }
                        }
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        runtime_artifacts.extend(
            [
                {
                    "artifact_path": f"runtime_artifacts/runtime_workspaces/job/{site}/log.txt",
                    "path": f"runtime_workspaces/job/{site}/log.txt",
                },
                {
                    "artifact_path": (
                        f"runtime_artifacts/runtime_workspaces/job/{site}/simulate_job/app_{site}/config/"
                        "config_fed_client.json"
                    ),
                    "path": f"runtime_workspaces/job/{site}/simulate_job/app_{site}/config/config_fed_client.json",
                },
            ]
        )

    with_output = "\n".join(
        [
            "2026-06-13 08:00:00,000 - INFO - Round 0 started.",
            "2026-06-13 08:00:18,000 - INFO - Aggregated 3/3 results",
            "2026-06-13 08:00:20,000 - INFO - Round 1 started.",
            "2026-06-13 08:00:39,000 - INFO - Aggregated 3/3 results",
            "2026-06-13 08:00:40,000 - INFO - Round 2 started.",
            "2026-06-13 08:15:42,000 - INFO - Aggregated 3/3 results",
            "PTInProcessClientAPIExecutor - INFO - result received",
            "Finished FedAvg.",
        ]
    )
    with_run = {
        "available": True,
        "label": "With skills",
        "mode": WITH_SKILLS_MODE,
        "mode_dir": mode_dir,
        "run": {"elapsed_seconds": 950},
        "agent_events_text": "\n".join(
            json.dumps(event)
            for event in command_events(
                "python job.py",
                "2026-06-13T08:00:00Z",
                "2026-06-13T08:15:45Z",
                "with_job",
                output=with_output,
            )
        ),
        "workspace_delta": {
            "changed_files": [{"artifact_path": "changed_files/client.py", "path": "client.py"}],
            "runtime_artifacts": runtime_artifacts,
        },
    }
    base_run = {
        "available": True,
        "label": "No skills baseline",
        "mode": NO_SKILLS_MODE,
        "run": {"elapsed_seconds": 120},
        "agent_events_text": "\n".join(
            json.dumps(event)
            for event in command_events(
                "python run_job.py",
                "2026-06-13T08:00:00Z",
                "2026-06-13T08:01:00Z",
                "base_job",
                output="2026-06-13 08:00:10,000 - INFO - Round 0 started.\nFinished FedAvg.\n",
            )
        ),
    }

    plugin = NvflareReportPlugin()
    modes = [WITH_SKILLS_MODE, NO_SKILLS_MODE]
    runs = {WITH_SKILLS_MODE: _ev(with_run), NO_SKILLS_MODE: _ev(base_run)}
    cmp = ComparisonEvidence(schema_version=SCHEMA_VERSION, runs=runs, modes=modes, sdk_metadata={})
    evidence = {mode: plugin.collect(runs[mode]) for mode in modes}
    note = "\n".join(fragment.text for fragment in plugin.explain(cmp, evidence))

    assert "Slow FL round evidence" in note
    assert "Slow Lightning client evidence" in note
    assert "server Round 2 / client round 3" in note
    assert "previous client rounds were shorter (round 1 max 16s; round 2 max 16s)" in note
    assert "client round 3 fit timings: site-1 900s, site-2 901s, site-3 899s" in note
    assert "this points to local Lightning `Trainer.fit`, not NVFLARE transfer/aggregation" in note
    assert "Lightning stopped at max_epochs=3" in note
    assert "site config passed --epochs 1" in note
    assert "reuses it for repeated `trainer.fit(...)` calls" in note


def test_longest_command_table_empty_cells_preserve_threshold():
    from benchmark.harness.reports.benchmark_insights import _longest_command_comparison_note

    def command_events(command, start, end, item_id):
        return [
            {
                "timestamp": start,
                "type": "item.started",
                "item": {
                    "command": command,
                    "id": item_id,
                    "status": "in_progress",
                    "type": "command_execution",
                },
            },
            {
                "timestamp": end,
                "type": "item.completed",
                "item": {
                    "aggregated_output": "ok",
                    "command": command,
                    "exit_code": 0,
                    "id": item_id,
                    "status": "completed",
                    "type": "command_execution",
                },
            },
        ]

    with_events = command_events("python long_one.py", "2026-06-13T08:00:00Z", "2026-06-13T08:02:00Z", "w1")
    with_events += command_events("python long_two.py", "2026-06-13T08:03:00Z", "2026-06-13T08:04:00Z", "w2")
    base_events = command_events("python base.py", "2026-06-13T08:00:00Z", "2026-06-13T08:01:00Z", "b1")

    note = _longest_command_comparison_note(
        _ev(
            {
                "label": "With skills",
                "agent_events_text": "\n".join(json.dumps(event) for event in with_events),
            }
        ),
        _ev(
            {
                "label": "No skills baseline",
                "agent_events_text": "\n".join(json.dumps(event) for event in base_events),
            }
        ),
    )

    assert "| 2 | `python long_two.py` (60s, exit 0) | no timed command span >=30s captured |" in note


def test_fewer_turns_note_does_not_invent_command_runtime_cause():
    from benchmark.harness.reports.benchmark_insights import _why_slower

    with_run = {
        "label": "With skills",
        "run": {"elapsed_seconds": 120},
        "activity": {"event_types": {"assistant": 2}},
        "agent_events_text": "",
    }
    base_run = {
        "label": "No skills baseline",
        "run": {"elapsed_seconds": 100},
        "activity": {"event_types": {"assistant": 4}},
        "agent_events_text": "",
    }

    explanation = "\n".join(_why_slower(_ev(with_run), _ev(base_run)))

    assert "Assistant turns" not in explanation
    assert "better explained by captured command/runtime duration" not in explanation


def test_why_slower_does_not_blame_code_quality_when_runtime_excluding_install_is_faster(tmp_path):
    from benchmark.harness.reports.benchmark_insights import _why_slower

    def command_events(command, start, end, item_id, output="ok"):
        return [
            {
                "timestamp": start,
                "type": "item.started",
                "item": {
                    "command": command,
                    "id": item_id,
                    "status": "in_progress",
                    "type": "command_execution",
                },
            },
            {
                "timestamp": end,
                "type": "item.completed",
                "item": {
                    "aggregated_output": output,
                    "command": command,
                    "exit_code": 0,
                    "id": item_id,
                    "status": "completed",
                    "type": "command_execution",
                },
            },
        ]

    def run_with_client(mode: str, source: str, events: list[dict], elapsed_seconds: int) -> dict:
        mode_dir = tmp_path / mode
        source_path = mode_dir / "workspace_delta" / "changed_files" / "client.py"
        source_path.parent.mkdir(parents=True)
        source_path.write_text(source, encoding="utf-8")
        return {
            "label": mode,
            "mode": mode,
            "run": {"elapsed_seconds": elapsed_seconds},
            "activity": {},
            "agent_events_text": "\n".join(json.dumps(event) for event in events),
            "mode_dir": mode_dir,
            "workspace_delta": {
                "changed_files": [
                    {
                        "artifact_path": "changed_files/client.py",
                        "path": "client.py",
                    }
                ]
            },
        }

    with_source = """
import nvflare.client as flare
train_frame = load_split(args.data_dir, "train")
train_loader = DataLoader(train_frame)
while flare.is_running():
    criterion, optimizer = build_loss_and_optimizer(model, train_frame, args, device)
    test_metrics = evaluate(model, test_loader, criterion, device)
"""
    base_source = """
import nvflare.client as flare
train_frame = load_split(args.data_dir, "train")
train_loader = DataLoader(train_frame)
criterion, optimizer = build_loss_and_optimizer(model, train_frame, args, device)
while flare.is_running():
    train_one_epoch(model, train_loader, criterion, optimizer, device)
"""
    with_events = command_events(
        "pip install -r requirements-train.txt",
        "2026-06-13T00:00:00Z",
        "2026-06-13T00:21:58Z",
        "item_1",
        output="Downloading nvidia-cublas\nSuccessfully installed torch\n",
    ) + command_events(
        "python job.py",
        "2026-06-13T00:22:00Z",
        "2026-06-13T00:24:00Z",
        "item_2",
        output="Finished FedAvg.\n",
    )
    base_events = command_events(
        "python -m pip install -r requirements-train.txt",
        "2026-06-13T00:00:00Z",
        "2026-06-13T00:06:30Z",
        "item_1",
        output="Successfully installed torch\n",
    ) + command_events(
        "python run_nvflare_fedavg.py",
        "2026-06-13T00:07:00Z",
        "2026-06-13T00:23:40Z",
        "item_2",
        output="Finished FedAvg.\n",
    )

    with_run = run_with_client("with_skills", with_source, with_events, elapsed_seconds=1855)
    base_run = run_with_client("without_skills", base_source, base_events, elapsed_seconds=1399)
    explanation = "\n".join(
        _why_slower(
            _ev(with_run),
            _ev(base_run),
            _nv_ctx({"with_skills": with_run, "without_skills": base_run}),
        )
    )

    assert "Dependency install path differed" in explanation
    assert "Generated-code efficiency issue is not the measured slowdown driver" in explanation
    assert "runtime excluding dependency install is 537s vs 1009s" in explanation
    assert "Quality evidence did not make non-install runtime slower in this run" in explanation
    assert "Generated-code efficiency issue aligns with slower non-install runtime" not in explanation
    assert "may explain part of the long per-round wait" not in explanation


def test_dependency_install_detection_ignores_process_grep():
    from benchmark.harness.reports.benchmark_insights import is_dependency_install_command

    assert is_dependency_install_command("uv pip install -r requirements-train.txt")
    assert is_dependency_install_command("python3 -m pip install torch")
    assert not is_dependency_install_command("python -m pip show nvflare torch")
    assert not is_dependency_install_command(
        "for p in /proc/[0-9]*; do tr '\\0' ' ' < \"$p/cmdline\" | grep -E 'python3 -m pip|pip install'; done"
    )


def test_job_run_status_requires_success_evidence_for_simulation_script():
    from benchmark.harness.sdks.nvflare._logic import job_run_status

    event = {
        "item": {
            "aggregated_output": "configuration loaded successfully",
            "command": "python run_nvflare_simulation.py",
            "exit_code": 0,
            "id": "item_1",
            "status": "completed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["python run_nvflare_simulation.py"]},
        "agent_events_text": json.dumps(event),
    }

    assert job_run_status(run) == "started_failed"


def test_job_run_status_detects_leading_job_entrypoint():
    from benchmark.harness.sdks.nvflare._logic import job_run_status

    event = {
        "item": {
            "aggregated_output": "Job Status: FINISHED:COMPLETED\nResult can be found in: /tmp/nvflare/fedxgb",
            "command": "python job_vertical.py",
            "exit_code": 0,
            "id": "item_1",
            "status": "completed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["python job_vertical.py"]},
        "agent_events_text": json.dumps(event),
    }

    assert job_run_status(run) == "completed"


def test_job_success_ignores_file_inspection_matching_python_job_text():
    from benchmark.harness.sdks.nvflare._logic import job_command_succeeded, job_run_status

    event = {
        "command": "/bin/bash -lc 'rg \"python job.py|Finished FedAvg\" -n .'",
        "exit_code": 0,
        "output": 'README.md:Run with python job.py\njob.py:print("Finished FedAvg.")',
        "status": "completed",
    }
    run = {
        "available": True,
        "activity": {"commands": [event["command"]]},
        "agent_events_text": json.dumps(
            {"item": {"type": "command_execution", "aggregated_output": event["output"], **event}}
        ),
    }

    assert job_command_succeeded(event) is False
    assert job_run_status(run) == "not_started"


def test_job_success_ignores_wrapped_file_inspection_matching_python_job_text():
    from benchmark.harness.sdks.nvflare._logic import job_command_succeeded, job_run_status

    event = {
        "command": "/bin/bash -lc 'cd /work && rg \"python job.py|Finished FedAvg\" -n .'",
        "exit_code": 0,
        "output": 'README.md:Run with python job.py\njob.py:print("Finished FedAvg.")',
        "status": "completed",
    }
    run = {
        "available": True,
        "activity": {"commands": [event["command"]]},
        "agent_events_text": json.dumps(
            {"item": {"type": "command_execution", "aggregated_output": event["output"], **event}}
        ),
    }

    assert job_command_succeeded(event) is False
    assert job_run_status(run) == "not_started"


def test_job_success_detects_execution_after_wrapped_file_inspection():
    from benchmark.harness.sdks.nvflare._logic import job_command_succeeded, job_run_status

    event = {
        "command": "/bin/bash -lc 'cd /work && rg \"python job.py|Finished FedAvg\" -n . && python job.py'",
        "exit_code": 0,
        "output": 'README.md:Run with python job.py\njob.py:print("Finished FedAvg.")\nFinished FedAvg.',
        "status": "completed",
    }
    run = {
        "available": True,
        "activity": {"commands": [event["command"]]},
        "agent_events_text": json.dumps(
            {"item": {"type": "command_execution", "aggregated_output": event["output"], **event}}
        ),
    }

    assert job_command_succeeded(event) is True
    assert job_run_status(run) == "completed"


def test_job_success_rejects_direct_job_with_config_error_finished_marker():
    from benchmark.harness.sdks.nvflare._logic import job_command_succeeded, job_run_status

    event = {
        "command": "python job.py",
        "exit_code": 0,
        "output": (
            "ConfigError: executors are not specified\n"
            "Abort signal triggered. Finishing FedAvg.\n"
            "Finished FedAvg.\n"
        ),
        "status": "completed",
    }
    run = {
        "available": True,
        "activity": {"commands": [event["command"]]},
        "agent_events_text": json.dumps(
            {"item": {"type": "command_execution", "aggregated_output": event["output"], **event}}
        ),
    }

    assert job_command_succeeded(event) is False
    assert job_run_status(run) == "started_failed"


def test_job_success_ignores_grep_pattern_with_inline_semicolon():
    from benchmark.harness.sdks.nvflare._logic import job_command_succeeded, job_run_status

    # The ';' lives inside the rg search pattern; quote-aware splitting must keep it as a single
    # inspection segment rather than exposing "python job.py" as an executed job.
    event = {
        "command": 'rg "foo; python job.py" -n .',
        "exit_code": 0,
        "output": 'job.py:print("Finished FedAvg.")',
        "status": "completed",
    }
    run = {
        "available": True,
        "activity": {"commands": [event["command"]]},
        "agent_events_text": json.dumps(
            {"item": {"type": "command_execution", "aggregated_output": event["output"], **event}}
        ),
    }

    assert job_command_succeeded(event) is False
    assert job_run_status(run) == "not_started"


def test_job_success_ignores_cd_prefix_before_grep_inspection():
    from benchmark.harness.sdks.nvflare._logic import job_command_succeeded, job_run_status

    # A benign 'cd' prefix must not push python_script_name onto the broad regex and treat the
    # grep pattern as a real python job.py run.
    event = {
        "command": "cd /work && rg 'python job.py|Finished FedAvg' -n .",
        "exit_code": 0,
        "output": 'README.md:Run with python job.py\njob.py:print("Finished FedAvg.")',
        "status": "completed",
    }
    run = {
        "available": True,
        "activity": {"commands": [event["command"]]},
        "agent_events_text": json.dumps(
            {"item": {"type": "command_execution", "aggregated_output": event["output"], **event}}
        ),
    }

    assert job_command_succeeded(event) is False
    assert job_run_status(run) == "not_started"


def test_job_success_requires_evidence_when_inspection_follows_job_with_semicolon():
    from benchmark.harness.sdks.nvflare._logic import job_command_succeeded

    # The job failed but a trailing ';' inspection segment makes the aggregate exit code 0; the
    # direct-job exit code can no longer be trusted, so success evidence is required.
    event = {
        "command": "python job.py ; cat results.txt",
        "exit_code": 0,
        "output": "Traceback (most recent call last):\nRuntimeError: job crashed",
        "status": "completed",
    }

    assert job_command_succeeded(event) is False


def test_job_success_trusts_direct_job_exit_in_and_chain():
    from benchmark.harness.sdks.nvflare._logic import job_command_succeeded

    # An '&&' chain reaching exit 0 implies the job itself succeeded, so the aggregate exit code
    # is trustworthy even without explicit success output.
    event = {
        "command": "python job.py && python validate.py",
        "exit_code": 0,
        "output": "validation ok",
        "status": "completed",
    }

    assert job_command_succeeded(event) is True


def test_simulator_wrapper_detected_after_non_runtime_first_segment():
    from benchmark.harness.sdks.nvflare._logic import job_run_status, job_run_status_reason

    # The runtime wrapper is the second segment; per-segment classification must not stop at the
    # non-runtime first script (prepare_data.py).
    event = {
        "item": {
            "aggregated_output": (
                "Running: python3 -m nvflare.cli simulator /workspace/fl_job -w /workspace/ws -n 2 -t 2\n"
                "Finished FedAvg.\n"
            ),
            "command": "python prepare_data.py && python run_nvflare_fedavg.py",
            "exit_code": 0,
            "id": "item_1",
            "status": "completed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["python prepare_data.py && python run_nvflare_fedavg.py"]},
        "agent_events_text": json.dumps(event),
    }

    assert job_run_status(run) == "completed"
    assert "simulation completed" in job_run_status_reason(run)


def test_simulator_wrapper_detected_for_make_target():
    from benchmark.harness.sdks.nvflare._logic import job_run_status

    event = {
        "item": {
            "aggregated_output": (
                "python3 -m nvflare.cli simulator /workspace/fl_job -w /workspace/ws -n 2 -t 2\n" "Finished FedAvg.\n"
            ),
            "command": "make simulate",
            "exit_code": 0,
            "id": "item_1",
            "status": "completed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["make simulate"]},
        "agent_events_text": json.dumps(event),
    }

    assert job_run_status(run) == "completed"


def test_job_run_status_requires_success_evidence_for_leading_job_entrypoint():
    from benchmark.harness.sdks.nvflare._logic import job_run_status

    event = {
        "item": {
            "aggregated_output": "usage: job_vertical.py [--run_psi] [--run_training]\nerror: select a run mode",
            "command": "python job_vertical.py",
            "exit_code": 0,
            "id": "item_1",
            "status": "completed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["python job_vertical.py"]},
        "agent_events_text": json.dumps(event),
    }

    assert job_run_status(run) == "started_failed"


def test_job_run_status_detects_ambiguous_job_suffix_with_success_evidence():
    from benchmark.harness.sdks.nvflare._logic import job_run_status

    event = {
        "item": {
            "aggregated_output": "Job Status is: FINISHED:COMPLETED\nResult location: /tmp/nvflare/results",
            "command": "python fedavg_job.py",
            "exit_code": 0,
            "id": "item_1",
            "status": "completed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["python fedavg_job.py"]},
        "agent_events_text": json.dumps(event),
    }

    assert job_run_status(run) == "completed"


def test_job_run_status_requires_success_evidence_for_ambiguous_job_suffix_script():
    from benchmark.harness.sdks.nvflare._logic import job_run_status

    event = {
        "item": {
            "aggregated_output": "configuration loaded successfully",
            "command": "python fedavg_job.py",
            "exit_code": 0,
            "id": "item_1",
            "status": "completed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["python fedavg_job.py"]},
        "agent_events_text": json.dumps(event),
    }

    assert job_run_status(run) == "started_failed"


def test_job_run_status_rejects_result_path_with_failed_status():
    from benchmark.harness.sdks.nvflare._logic import job_run_status

    event = {
        "item": {
            "aggregated_output": (
                "Result can be found in: /tmp/nvflare/fedxgb\n" "Job Status is: FINISHED:EXECUTION_EXCEPTION"
            ),
            "command": "python fedavg_job.py",
            "exit_code": 0,
            "id": "item_1",
            "status": "completed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["python fedavg_job.py"]},
        "agent_events_text": json.dumps(event),
    }

    assert job_run_status(run) == "started_failed"


def test_job_output_failure_status_vetoes_result_location():
    from benchmark.harness.reports._events import job_output_succeeded

    assert not job_output_succeeded("Result location: /tmp/r\nJob Status: FINISHED:EXECUTION_EXCEPTION")
    assert not job_output_succeeded("Result can be found in: /tmp/r\nStatus is: FINISHED:ABORTED")
    assert job_output_succeeded("Result location: /tmp/r\nJob Status: FINISHED:COMPLETED")


def test_job_output_failure_status_vetoes_legacy_terminal_statuses():
    from benchmark.harness.reports._events import job_output_succeeded

    assert not job_output_succeeded("Result location: /tmp/r\nJob Status: FAILED")
    assert not job_output_succeeded("Result can be found in: /tmp/r\nJob Status is: FINISHED_EXCEPTION")
    assert not job_output_succeeded("Result location: /tmp/r\nStatus: ABANDONED")
    assert not job_output_succeeded("Result location: /tmp/r\nStatus: ABORTED")
    # Legacy success form must still pass.
    assert job_output_succeeded("Result location: /tmp/r\nJob Status: FINISHED_OK")


def test_job_output_config_error_vetoes_finished_marker():
    from benchmark.harness.reports._events import job_output_succeeded

    assert not job_output_succeeded(
        "ConfigError: executors are not specified\n"
        "Abort signal triggered. Finishing FedAvg.\n"
        "Finished FedAvg.\n"
    )


def test_recovered_by_later_success_requires_simulation_success_evidence():
    from benchmark.harness.reports.benchmark_insights import command_failure_diagnostics

    failed_event = {
        "item": {
            "aggregated_output": "Traceback (most recent call last):\nRuntimeError: simulation failed",
            "command": "python run_nvflare_simulation.py",
            "exit_code": 1,
            "id": "item_1",
            "status": "failed",
            "type": "command_execution",
        }
    }
    incomplete_success_event = {
        "item": {
            "aggregated_output": "configuration loaded successfully",
            "command": "python run_nvflare_simulation.py",
            "exit_code": 0,
            "id": "item_2",
            "status": "completed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["python run_nvflare_simulation.py", "python run_nvflare_simulation.py"]},
        "agent_events_text": "\n".join(json.dumps(event) for event in (failed_event, incomplete_success_event)),
    }

    diagnostics = command_failure_diagnostics(_ev(run), ev=_nv_ev(run))

    assert diagnostics
    assert "not recovered in this run" in diagnostics[0]


def test_failure_analysis_formats_multiline_recovered_command_as_single_line():
    from benchmark.harness.modes import WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import failure_analysis_section

    failed_event = {
        "item": {
            "aggregated_output": "Traceback (most recent call last):\nTypeError: unsupported operand type(s)",
            "command": "python3 -m py_compile client.py job.py && python3 - <<'EOF'\nprint('check')\nEOF",
            "exit_code": 1,
            "id": "item_1",
            "status": "failed",
            "type": "command_execution",
        }
    }
    recovered_event = {
        "item": {
            "aggregated_output": "Finished FedAvg.",
            "command": "python3 job.py",
            "exit_code": 0,
            "id": "item_2",
            "status": "completed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "label": "With skills",
        "agent_events_text": "\n".join(json.dumps(event) for event in (failed_event, recovered_event)),
        "container_exit": {"exit_code": 0},
        "record": {},
        "run": {"final_container_exit_code": 0},
        "validation_metric": {"name": "AUROC", "value": 0.725},
    }

    section = failure_analysis_section(
        _evruns({WITH_SKILLS_MODE: run}), [WITH_SKILLS_MODE], _nv_ctx({WITH_SKILLS_MODE: run}, [WITH_SKILLS_MODE])
    )

    assert "Recovered Command Evidence" in section
    assert "| Command | Exit | Recovery | Root cause | Dependency evidence |" in section
    assert "python3 -m py_compile client.py job.py && python3 - <<'EOF' ... EOF" in section
    assert "print('check')\nEOF" not in section


def test_job_run_status_reason_includes_failed_job_command_error():
    from benchmark.harness.reports.benchmark_insights import job_run_action
    from benchmark.harness.sdks.nvflare._logic import job_run_status, job_run_status_reason

    event = {
        "item": {
            "aggregated_output": "Traceback (most recent call last):\nModuleNotFoundError: No module named 'torch'",
            "command": "python job.py",
            "exit_code": 1,
            "id": "item_1",
            "status": "failed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["python job.py"]},
        "agent_events_text": json.dumps(event),
    }

    assert job_run_status(run) == "started_failed"
    reason = job_run_status_reason(run)
    assert "missing Python dependency `torch`" in reason
    assert "no dependency install command was captured" in reason
    assert "Install the job requirements" in job_run_action(_ev(run), _nv_ev(run).job_execution)


def test_completed_job_run_status_reason_includes_recovered_dependency_failure():
    from benchmark.harness.reports.benchmark_insights import job_run_action
    from benchmark.harness.sdks.nvflare._logic import job_run_status, job_run_status_reason

    failed_probe = {
        "item": {
            "aggregated_output": "Traceback (most recent call last):\nModuleNotFoundError: No module named 'torch'",
            "command": 'python -c "import torch"',
            "exit_code": 1,
            "id": "probe",
            "status": "failed",
            "type": "command_execution",
        }
    }
    install = {
        "item": {
            "aggregated_output": "Successfully installed torch",
            "command": "pip install -r requirements.txt",
            "exit_code": 0,
            "id": "install",
            "status": "completed",
            "type": "command_execution",
        }
    }
    successful_job = {
        "item": {
            "aggregated_output": "site-1: round=0 train_loss=0.5 valid_auroc=0.7\nFinished FedAvg.",
            "command": "python job.py",
            "exit_code": 0,
            "id": "job",
            "status": "completed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ['python -c "import torch"', "pip install -r requirements.txt", "python job.py"]},
        "agent_events_text": "\n".join(json.dumps(event) for event in (failed_probe, install, successful_job)),
    }

    reason = job_run_status_reason(run)

    assert job_run_status(run) == "completed"
    assert "simulation completed" in reason
    assert "earlier missing Python dependency `torch` was recovered" in reason
    assert "a dependency install command later succeeded" in reason
    assert "inspect recovered command failures" in job_run_action(_ev(run), _nv_ev(run).job_execution)


def test_failure_analysis_keeps_recovered_bash_issue_for_passed_run():
    from benchmark.harness.modes import WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import failure_analysis_section

    permission_result = {
        "message": {
            "content": [
                {
                    "content": "Claude requested permissions to use Bash, but you haven't granted it yet.",
                    "type": "tool_result",
                }
            ]
        },
        "type": "user",
    }
    final_result = {
        "final_message": "Completed simulation.",
        "permission_denials": [
            {"tool_name": "Bash", "tool_input": {"command": "rm -rf /tmp/workspace && python job.py"}}
        ],
        "subtype": "success",
        "type": "result",
    }
    run = {
        "available": True,
        "label": "With skills",
        "activity": {"hint_counts": {"python_job_py": 1, "simulation": 1}},
        "agent_events_text": "\n".join(json.dumps(event) for event in (permission_result, final_result)),
        "container_exit": {"exit_code": 0},
        "record": {},
        "run": {"final_container_exit_code": 0},
    }

    section = failure_analysis_section(_evruns({WITH_SKILLS_MODE: run}), [WITH_SKILLS_MODE])

    assert "Outcome: passed" in section
    assert "Recovered Bash/tool issue" in section
    assert "Bash tool was blocked 1 time(s)" in section
    assert "Denied command: `rm -rf /tmp/workspace && python job.py`" in section
    assert "costs extra tool turns, tokens, and elapsed time" in section


def test_failure_analysis_reports_dependency_install_evidence_for_missing_module():
    from benchmark.harness.modes import NO_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import failure_analysis_section

    event = {
        "item": {
            "aggregated_output": "Traceback (most recent call last):\nModuleNotFoundError: No module named 'torch'",
            "command": 'python -c "import torch"',
            "exit_code": 1,
            "id": "probe",
            "status": "failed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "label": "No skills baseline",
        "container_exit": {"exit_code": 1},
        "activity": {"commands": ['python -c "import torch"']},
        "agent_events_text": json.dumps(event),
    }

    section = failure_analysis_section(
        _evruns({NO_SKILLS_MODE: run}), [NO_SKILLS_MODE], _nv_ctx({NO_SKILLS_MODE: run}, [NO_SKILLS_MODE])
    )

    assert "ModuleNotFoundError: No module named 'torch'" in section
    assert "no dependency install command was captured before the failed job run" in section


def test_failure_analysis_explains_missing_module_during_background_dependency_install():
    from benchmark.harness.modes import WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import failure_analysis_section

    install_command = "uv pip install -r requirements-train.txt 2>&1 | tail -20"
    probe_stderr = "Traceback (most recent call last):\nModuleNotFoundError: No module named 'torch'"
    events = [
        {
            "event_type": "assistant",
            "message": {
                "content": [
                    {
                        "id": "toolu_install",
                        "input": {"command": install_command, "run_in_background": True},
                        "name": "Bash",
                        "type": "tool_use",
                    }
                ]
            },
        },
        {"event_type": "system.task_started", "task_id": "install_task", "tool_use_id": "toolu_install"},
        {
            "event_type": "user",
            "message": {
                "content": [
                    {
                        "content": "Command running in background with ID: install_task",
                        "tool_use_id": "toolu_install",
                        "type": "tool_result",
                    }
                ]
            },
            "tool_use_result": {"backgroundTaskId": "install_task"},
        },
        {
            "event_type": "assistant",
            "message": {
                "content": [
                    {
                        "id": "toolu_probe",
                        "input": {"command": 'python3 -c "import torch"'},
                        "name": "Bash",
                        "type": "tool_use",
                    }
                ]
            },
        },
        {
            "event_type": "user",
            "message": {
                "content": [
                    {
                        "content": probe_stderr,
                        "is_error": True,
                        "tool_use_id": "toolu_probe",
                        "type": "tool_result",
                    }
                ]
            },
            "tool_use_result": {"stderr": probe_stderr, "is_error": True},
        },
        {"event_type": "system.task_updated", "patch": {"status": "completed"}, "task_id": "install_task"},
        {
            "event_type": "assistant",
            "message": {
                "content": [
                    {
                        "id": "toolu_verify",
                        "input": {"command": 'python3 -c "import torch; print(\'torch\', torch.__version__)"'},
                        "name": "Bash",
                        "type": "tool_use",
                    }
                ]
            },
        },
        {
            "event_type": "user",
            "message": {
                "content": [
                    {
                        "content": "torch 2.12.1+cu130",
                        "tool_use_id": "toolu_verify",
                        "type": "tool_result",
                    }
                ]
            },
            "tool_use_result": {"stdout": "torch 2.12.1+cu130"},
        },
    ]
    run = {
        "available": True,
        "label": "With skills",
        "container_exit": {"exit_code": 1},
        "activity": {"commands": ['python3 -c "import torch"']},
        "agent_events_text": "\n".join(json.dumps(event) for event in events),
    }

    section = failure_analysis_section(
        _evruns({WITH_SKILLS_MODE: run}), [WITH_SKILLS_MODE], _nv_ctx({WITH_SKILLS_MODE: run}, [WITH_SKILLS_MODE])
    )

    assert "ModuleNotFoundError: No module named 'torch'" in section
    assert "`torch` was probed while background dependency install" in section
    assert "requirements-train.txt" in section
    assert "was still running" in section
    assert "later verification imported `torch` successfully" in section


def test_job_run_status_reason_reports_failed_dependency_install():
    from benchmark.harness.sdks.nvflare._logic import job_run_status_reason

    install_event = {
        "item": {
            "aggregated_output": "ERROR: Could not find a version that satisfies the requirement torch",
            "command": "python -m pip install -r requirements.txt",
            "exit_code": 1,
            "id": "install",
            "status": "failed",
            "type": "command_execution",
        }
    }
    job_event = {
        "item": {
            "aggregated_output": "Traceback (most recent call last):\nModuleNotFoundError: No module named 'torch'",
            "command": "python job.py",
            "exit_code": 1,
            "id": "job",
            "status": "failed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["python -m pip install -r requirements.txt", "python job.py"]},
        "agent_events_text": "\n".join(json.dumps(event) for event in (install_event, job_event)),
    }

    reason = job_run_status_reason(run)

    assert "missing Python dependency `torch`" in reason
    assert "dependency install attempted and failed" in reason
    assert "requirements.txt" in reason


def test_job_run_status_reason_reports_successful_install_with_wrong_runtime():
    from benchmark.harness.reports.benchmark_insights import job_run_action
    from benchmark.harness.sdks.nvflare._logic import job_run_status_reason

    install_event = {
        "item": {
            "aggregated_output": "Successfully installed torch",
            "command": "python -m pip install -r requirements.txt",
            "exit_code": 0,
            "id": "install",
            "status": "completed",
            "type": "command_execution",
        }
    }
    job_event = {
        "item": {
            "aggregated_output": "Traceback (most recent call last):\nModuleNotFoundError: No module named 'torch'",
            "command": "python job.py",
            "exit_code": 1,
            "id": "job",
            "status": "failed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["python -m pip install -r requirements.txt", "python job.py"]},
        "agent_events_text": "\n".join(json.dumps(event) for event in (install_event, job_event)),
    }

    reason = job_run_status_reason(run)

    assert "dependency install command succeeded" in reason
    assert "simulator uses the environment where requirements were installed" in job_run_action(
        _ev(run), _nv_ev(run).job_execution, _nv_ctx({"m": run})
    )


def test_readme_metric_alignment_uses_server_best_validation_metric_scalar():
    from benchmark.harness.quality_signals import metric_signal

    signal = metric_signal(
        None,
        "Primary validation metric: AUROC.\n",
        """
Final round metrics:
- `site-1`: valid AUROC `0.7696`, test AUROC `0.7331`
- `site-2`: valid AUROC `0.7148`, test AUROC `0.7771`
- `site-3`: valid AUROC `0.7708`, test AUROC `0.7352`
- Server best validation metric at round 2: `0.7517306189541327`
""",
    )

    metric = signal["reported_validation_metric"]
    assert signal["status"] == "pass"
    assert signal["aligned_with_readme"] is True
    assert metric["name"] == "AUROC"
    assert metric["value"] == 0.7517306189541327
    assert metric["value_scope"] == "fl_summary_metric"
    assert metric["site_value_count"] == 6
    assert metric["summary_value_label"] == "Server best validation metric at round 2"


def test_readme_metric_alignment_passes_for_site_level_values_without_scalar():
    from benchmark.harness.quality_signals import metric_signal

    signal = metric_signal(
        None,
        "Primary validation metric: AUROC.\n",
        """
Final round metrics:
- `site-1`: valid AUROC `0.7696`
- `site-2`: valid AUROC `0.7148`
- `site-3`: valid AUROC `0.7708`
""",
    )

    metric = signal["reported_validation_metric"]
    assert signal["status"] == "pass"
    assert signal["aligned_with_readme"] is True
    assert signal["metric_value_available"] is True
    assert signal["metric_scalar_available"] is False
    assert signal["mismatch"] is False
    assert metric["name"] == "AUROC"
    assert metric["value"] is None
    assert metric["value_scope"] == "site_values_only"


def test_metrics_chart_names_metric_once_in_panel_title():
    from benchmark.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import embedded_bar_chart, outcome_metrics_table

    def run(label: str, value: float) -> dict:
        return {
            "label": label,
            "available": True,
            "status": "0",
            "run": {"elapsed_seconds": 1, "token_count": 1, "agent_exit_code": 0, "final_container_exit_code": 0},
            "activity": {"command_count": 1},
            "record": {},
            "workspace_delta": {},
            "validation_metric": {"name": "AUROC", "value": value},
        }

    chart = embedded_bar_chart(
        _evruns(
            {
                NO_SKILLS_MODE: run("No skills baseline", 0.7562),
                WITH_SKILLS_MODE: run("With skills", 0.7529),
            }
        )
    )
    table = outcome_metrics_table(
        _evruns(
            {
                NO_SKILLS_MODE: run("No skills baseline", 0.7562),
                WITH_SKILLS_MODE: run("With skills", 0.7529),
            }
        ),
        [NO_SKILLS_MODE, WITH_SKILLS_MODE],
    )

    assert "Metrics (AUROC)" in chart
    assert "Code quality" in chart
    assert "FL scalar result" not in chart
    assert "AUROC 0." not in chart
    assert chart.count("AUROC") == 1
    assert ">0.7529<" in chart
    assert "| Metrics (AUROC) | AUROC 0.7562 | AUROC 0.7529 |" in table
    assert "FL scalar result" not in table


def test_metrics_chart_uses_labeled_aggregated_metric_from_legacy_record():
    from benchmark.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import embedded_bar_chart, outcome_metrics_table

    def run(label: str, metric: dict) -> dict:
        return {
            "label": label,
            "available": True,
            "run": {"final_container_exit_code": 0},
            "activity": {},
            "validation_metric": metric,
        }

    runs = {
        NO_SKILLS_MODE: run("No skills baseline", {"name": "AUROC", "value": None, "reported_value_entries": []}),
        WITH_SKILLS_MODE: run(
            "With skills",
            {
                "name": "AUROC",
                "value": None,
                "reported_value_entries": [
                    {"value": 0.7531},
                    {"label": "Best aggregated validation AUROC", "value": 0.7623334631865992},
                    {"label": "Final site metrics", "value": 0.767293},
                ],
            },
        ),
    }

    # The FL-summary-label SELECTION is now SDK-owned (NVFLARE plugin's assess_metric),
    # surfaced via the per-run sidecar; the generic engine no longer FL-selects. Supply
    # the NVFLARE context explicitly so the FL fallback fires.
    modes = [NO_SKILLS_MODE, WITH_SKILLS_MODE]
    ctx = _nv_ctx(runs, modes)
    chart = embedded_bar_chart(_evruns(runs), ctx)
    table = outcome_metrics_table(_evruns(runs), modes, ctx)

    assert ">0.7623<" in chart
    assert "| Metrics (AUROC) | AUROC NA | AUROC 0.7623 |" in table


def test_metrics_chart_marks_mixed_metric_names_non_comparable():
    from benchmark.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from benchmark.harness.reports.benchmark_insights import embedded_bar_chart, outcome_metrics_table

    def run(label: str, metric_name: str, value: float) -> dict:
        return {
            "label": label,
            "available": True,
            "run": {"final_container_exit_code": 0},
            "activity": {},
            "validation_metric": {"name": metric_name, "value": value},
        }

    runs = {
        NO_SKILLS_MODE: run("No skills baseline", "accuracy", 0.8123),
        WITH_SKILLS_MODE: run("With skills", "AUROC", 0.7529),
    }

    chart = embedded_bar_chart(_evruns(runs))
    table = outcome_metrics_table(_evruns(runs), [NO_SKILLS_MODE, WITH_SKILLS_MODE])

    assert "Metrics (mixed validation metrics)" in chart
    assert "Not comparable" in chart
    assert "No skills baseline: accuracy" in chart
    assert "With skills: AUROC" in chart
    assert "| Metrics (mixed validation metrics) | accuracy 0.8123 | AUROC 0.7529 |" in table


def test_structure_tree_renderer_uses_tree_format():
    from benchmark.harness.reports.benchmark_insights import tree_from_paths

    tree = tree_from_paths(
        [
            "client.py",
            "runtime_job_config/ames_fedavg/ames_fedavg/app/config/config_fed_client.json",
            "runtime_job_config/ames_fedavg/ames_fedavg/app/custom/model.py",
        ]
    )

    assert tree.startswith(".\n")
    assert "|-- client.py" in tree
    assert "`-- runtime_job_config" in tree
    assert "        `-- ames_fedavg" in tree
    assert "- runtime_job_config/ames_fedavg" not in tree


def test_run_summary_uses_agent_keys_without_codex_aliases(tmp_path):
    from benchmark.harness.records import write_json, write_run_summary

    final_record = tmp_path / "record.json"
    summary_path = tmp_path / "run_summary.json"
    write_json(
        final_record,
        {
            "mode": "with_skills",
            "agent_process_passed": True,
            "agent_process_exit_code": 0,
            "codex_process_passed": True,
            "codex_process_exit_code": 0,
            "agent_usage": {"total_tokens": 10},
            "codex_usage": {"total_tokens": 10},
            "process_metrics": {
                "agent_exit_code": 0,
                "codex_exit_code": 0,
                "elapsed_seconds": 1,
                "command_count": 0,
            },
        },
    )

    write_run_summary(final_record, summary_path, print_summary=False)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["agent_process_passed"] is True
    assert summary["agent_process_exit_code"] == 0
    assert summary["agent_exit_code"] == 0
    assert summary["command_count"] == 0
    assert summary["agent_usage"] == {"total_tokens": 10}
    assert "codex_process_passed" not in summary
    assert "codex_process_exit_code" not in summary
    assert "codex_exit_code" not in summary
    assert "codex_usage" not in summary
    assert not any(key.startswith("codex_") for key in summary["all_metrics"])


def test_run_summary_ignores_codex_usage_fallback_and_reports_prompt_hash(tmp_path):
    from benchmark.harness.records import write_json, write_run_summary

    final_record = tmp_path / "record.json"
    summary_path = tmp_path / "run_summary.json"
    write_json(
        final_record,
        {
            "mode": "with_skills",
            "codex_usage": {"total_tokens": 10},
            "process_metrics": {
                "elapsed_seconds": 3,
                "agent_elapsed_seconds": 2,
            },
        },
    )
    write_json(
        tmp_path / "prompt_metadata.json",
        {
            "prompt_sha256": "abc123",
            "template_path": "/workspace/prompts/prompt.txt",
        },
    )

    write_run_summary(final_record, summary_path, print_summary=False)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["agent_usage"] == {}
    assert summary["agent_elapsed_seconds"] == 2
    assert summary["elapsed_seconds"] == 2
    assert summary["prompt_hash"] == "abc123"
    assert summary["prompt_source"] == "/workspace/prompts/prompt.txt"
    assert summary["structure_quality_signal"] == {
        "status": "unavailable",
        "reason": "structure quality was not captured for this run",
    }


def test_make_tree_readable_does_not_follow_symlinked_directories(tmp_path):
    from benchmark.harness.common import make_tree_readable

    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "secret.txt"
    outside_file.write_text("keep private\n", encoding="utf-8")
    outside_file.chmod(0o600)
    root = tmp_path / "result"
    root.mkdir()
    (root / "outside_link").symlink_to(outside, target_is_directory=True)

    make_tree_readable(root)

    assert stat.S_IMODE(outside_file.stat().st_mode) == 0o600


def test_scenario_report_escapes_markdown_table_pipes(tmp_path):
    from benchmark.harness.reports.scenario_report import write_scenario_report

    write_scenario_report(
        tmp_path,
        {
            "scenario_name": "pipe scenario",
            "status": "passed",
            "completed_run_count": 1,
            "expanded_case_count": 1,
            "winner_policy": "quality",
            "runs": [
                {
                    "run_id": "run_pipe",
                    "label": "with|skills",
                    "agent": "claude",
                    "agent_model": "default|model",
                    "model_source": "adapter_default",
                    "mode": "with_skills",
                }
            ],
            "aggregate_results": {
                "by_label": {
                    "with|skills": {
                        "run_count": 1,
                        "quality_pass_count": 1,
                        "agent_elapsed_seconds": {"median": 2.5},
                        "token_count": {"median": 10},
                    }
                },
                "winner": {"label": "with|skills"},
            },
        },
    )

    report = (tmp_path / "reports" / "scenario_report.md").read_text(encoding="utf-8")
    assert "with\\|skills" in report
    assert "## Run Identity" in report
    assert "default\\|model" in report


def test_report_generators_write_two_mode_outputs(tmp_path, monkeypatch):
    from benchmark.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from benchmark.harness.reports import metrics_report
    from benchmark.harness.reports.benchmark_insights import main as insights_main

    for mode, value in ((NO_SKILLS_MODE, 0.7562), (WITH_SKILLS_MODE, 0.7529)):
        mode_dir = tmp_path / mode
        records_dir = mode_dir / "records"
        records_dir.mkdir(parents=True)
        (mode_dir / "container_exit_code.json").write_text(json.dumps({"exit_code": 0}) + "\n", encoding="utf-8")
        (mode_dir / "run_summary.json").write_text(
            json.dumps(
                {
                    "mode": mode,
                    "elapsed_seconds": 10,
                    "token_count": 100,
                    "agent_exit_code": 0,
                    "final_container_exit_code": 0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (records_dir / f"{mode}_record.json").write_text(
            json.dumps(
                {
                    "mode": mode,
                    "reported_validation_metric": {"name": "AUROC", "value": value},
                    "process_metrics": {"elapsed_seconds": 10, "token_count": 100},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (mode_dir / "agent_activity.json").write_text(json.dumps({"command_count": 3}) + "\n", encoding="utf-8")
        (mode_dir / "agent_usage.json").write_text(json.dumps({"total_tokens": 100}) + "\n", encoding="utf-8")

    original_load_json = metrics_report.load_json
    common_file_reads = {
        "run_summary.json": 0,
        "record": 0,
        "agent_activity.json": 0,
        "agent_usage.json": 0,
    }

    def counted_load_json(path, default=None):
        if path.name in common_file_reads:
            common_file_reads[path.name] += 1
        elif path.name.endswith("_record.json") or path.name == "benchmark_record.json":
            common_file_reads["record"] += 1
        return original_load_json(path, default)

    monkeypatch.setattr(metrics_report, "load_json", counted_load_json)
    metrics_report.write_reports(tmp_path, "Synthetic Metrics")
    monkeypatch.setattr(sys, "argv", ["benchmark_insights", str(tmp_path)])
    insights_main()

    assert (tmp_path / "metrics_report.json").is_file()
    metrics_markdown = (tmp_path / "metrics_report.md").read_text(encoding="utf-8")
    insights_markdown = (tmp_path / "benchmark_insights.md").read_text(encoding="utf-8")
    assert "<svg" in metrics_markdown
    assert "<svg" in insights_markdown
    assert "Metrics (AUROC)" in metrics_markdown
    assert "Metrics (AUROC)" in insights_markdown
    assert "Benchmark Metrics Comparison" not in insights_markdown
    assert "with_skills_eval" not in insights_markdown
    assert "Evaluator" not in insights_markdown
    assert common_file_reads == {
        "run_summary.json": 2,
        "record": 2,
        "agent_activity.json": 2,
        "agent_usage.json": 2,
    }
