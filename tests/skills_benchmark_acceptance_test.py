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

"""Scenario-declared evaluation identity and acceptance gates.

Covers the data-not-code onboarding contract: `evaluation_task` /
`evaluation_selectors` route criteria, `result_artifact` and
`acceptance_checks` gate deterministically — all declared in scenario YAML.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def write_prompt_and_job(tmp_path: Path) -> tuple[Path, Path]:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Compute the federated statistics.", encoding="utf-8")
    job = tmp_path / "job"
    job.mkdir(exist_ok=True)
    return prompt, job


def base_scenario(tmp_path: Path) -> dict:
    prompt, job = write_prompt_and_job(tmp_path)
    return {
        "name": "fedstats scaffold",
        "prompt": prompt.name,
        "agents": [{"name": "codex", "models": ["gpt-test"]}],
        "comparison": {"type": "mode_ablation", "modes": ["without_skills", "with_skills"]},
        "workflows": [{"name": "STATS"}],
        "jobs": [{"path": job.name, "scale": "small"}],
    }


# --- compile-time schema -----------------------------------------------------


def test_scenario_evaluation_fields_flow_into_run_plan_entries(tmp_path):
    from benchmark.harness.scenarios import compile_scenario

    checks = tmp_path / "checks.py"
    checks.write_text("print('{}')\n", encoding="utf-8")
    raw = base_scenario(tmp_path)
    raw["evaluation_task"] = "federated-statistics"
    raw["evaluation_selectors"] = {"data-format": "no_header"}
    raw["result_artifact"] = {"glob": "**/statistics.json", "format": "json"}
    raw["acceptance_checks"] = {"script": checks.name, "timeout_seconds": 60}

    compilation = compile_scenario(raw, base_dir=tmp_path)
    entry = compilation.run_plan["entries"][0]
    assert entry["evaluation_task"] == "federated-statistics"
    assert entry["evaluation_selectors"] == {"data-format": "no_header"}
    assert entry["result_artifact"] == {"glob": "**/statistics.json", "format": "json"}
    assert entry["acceptance_checks"] == {"script": str(checks.resolve()), "timeout_seconds": 60}
    job = compilation.scenario["jobs"][0]
    assert job["evaluation_task"] == "federated-statistics"


def test_job_level_evaluation_task_overrides_scenario_default(tmp_path):
    from benchmark.harness.scenarios import compile_scenario

    raw = base_scenario(tmp_path)
    raw["evaluation_task"] = "conversion"
    raw["jobs"][0]["evaluation_task"] = "federated-statistics"
    compilation = compile_scenario(raw, base_dir=tmp_path)
    assert compilation.run_plan["entries"][0]["evaluation_task"] == "federated-statistics"


def test_scenario_without_evaluation_fields_compiles_with_null_defaults(tmp_path):
    from benchmark.harness.scenarios import compile_scenario

    compilation = compile_scenario(base_scenario(tmp_path), base_dir=tmp_path)
    entry = compilation.run_plan["entries"][0]
    assert entry["evaluation_task"] is None
    assert entry["evaluation_selectors"] == {}
    assert entry["result_artifact"] is None
    assert entry["acceptance_checks"] is None


@pytest.mark.parametrize(
    "field,value,fragment",
    [
        ("evaluation_task", "Bad Task!", "must match"),
        ("result_artifact", {"glob": "/etc/passwd"}, "relative workspace pattern"),
        ("result_artifact", {"glob": "x.json", "format": "xml"}, "format must be one of"),
        ("result_artifact", {"glob": "x.json", "mode": "strict"}, "unsupported key"),
        ("acceptance_checks", {"script": "missing.py"}, "existing file"),
    ],
)
def test_invalid_evaluation_declarations_fail_compile(tmp_path, field, value, fragment):
    from benchmark.harness.scenarios import ScenarioValidationError, compile_scenario

    raw = base_scenario(tmp_path)
    raw[field] = value
    with pytest.raises(ScenarioValidationError, match=fragment):
        compile_scenario(raw, base_dir=tmp_path)


# --- result_artifact gate ----------------------------------------------------


def record_dir_with_manifest(tmp_path: Path, files: dict[str, str]) -> Path:
    record_dir = tmp_path / "record"
    record_dir.mkdir(exist_ok=True)
    changed = []
    for path, content in files.items():
        artifact_path = path.replace("/", "__")
        captured = record_dir / "workspace_delta" / artifact_path
        captured.parent.mkdir(parents=True, exist_ok=True)
        captured.write_text(content, encoding="utf-8")
        changed.append({"path": path, "artifact_path": artifact_path})
    (record_dir / "workspace_delta_manifest.json").write_text(json.dumps({"changed_files": changed}), encoding="utf-8")
    return record_dir


def test_result_artifact_gate_passes_on_matching_valid_json(tmp_path):
    from benchmark.harness.acceptance import evaluate_result_artifact

    record_dir = record_dir_with_manifest(
        tmp_path, {"workspace/server/simulate_job/statistics/adults.json": '{"count": {}}'}
    )
    result = evaluate_result_artifact(record_dir, {"glob": "**/simulate_job/**/*.json", "format": "json"})
    assert result["check"]["passed"] is True
    assert result["matches"] == ["workspace/server/simulate_job/statistics/adults.json"]
    assert result["parsed_matches"] == ["workspace/server/simulate_job/statistics/adults.json"]
    assert result["selected_match"] == "workspace/server/simulate_job/statistics/adults.json"


def test_result_artifact_gate_fails_when_missing_or_unparsable(tmp_path):
    from benchmark.harness.acceptance import evaluate_result_artifact

    record_dir = record_dir_with_manifest(tmp_path, {"notes.md": "no stats here"})
    missing = evaluate_result_artifact(record_dir, {"glob": "**/statistics.json", "format": "json"})
    assert missing["check"]["passed"] is False
    assert "no workspace artifact matches" in missing["check"]["evidence"]

    record_dir = record_dir_with_manifest(tmp_path, {"out/statistics.json": "not json"})
    broken = evaluate_result_artifact(record_dir, {"glob": "**/statistics.json", "format": "json"})
    assert broken["check"]["passed"] is False


def test_result_artifact_gate_is_skipped_when_not_declared(tmp_path):
    from benchmark.harness.acceptance import evaluate_result_artifact

    record_dir = record_dir_with_manifest(tmp_path, {})
    assert evaluate_result_artifact(record_dir, None) is None


def test_result_artifact_gate_scans_runtime_artifacts(tmp_path):
    from benchmark.harness.acceptance import evaluate_result_artifact

    # Simulator outputs are captured under runtime_artifacts, not changed_files.
    record_dir = tmp_path / "record"
    captured = record_dir / "workspace_delta" / "runtime" / "stats.json"
    captured.parent.mkdir(parents=True)
    captured.write_text('{"count": {}}', encoding="utf-8")
    (record_dir / "workspace_delta_manifest.json").write_text(
        json.dumps(
            {
                "changed_files": [],
                "runtime_artifacts": [
                    {
                        "path": "workspace/server/simulate_job/statistics/stats.json",
                        "artifact_path": "runtime/stats.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = evaluate_result_artifact(record_dir, {"glob": "**/simulate_job/statistics/*.json", "format": "json"})
    assert result["check"]["passed"] is True


def test_acceptance_script_outside_scenario_dir_fails_compile(tmp_path):
    from benchmark.harness.scenarios import ScenarioValidationError, compile_scenario

    outside = tmp_path.parent / f"{tmp_path.name}-outside.py"
    outside.write_text("print('{}')\n", encoding="utf-8")
    raw = base_scenario(tmp_path)
    raw["acceptance_checks"] = {"script": str(outside)}
    with pytest.raises(ScenarioValidationError, match="inside the scenario directory"):
        compile_scenario(raw, base_dir=tmp_path)


def test_code_eval_prompt_uses_declared_task_framing():
    from benchmark.harness.code_eval import build_eval_prompt

    criteria = [{"key": "privacy_thresholds", "description": "privacy filters intact"}]
    stats_prompt = build_eval_prompt(criteria, "with_skills", task="federated-statistics")
    assert "federated statistics task" in stats_prompt
    assert "converting a training job" not in stats_prompt
    assert "converting a training job" in build_eval_prompt(criteria, "with_skills")


# --- acceptance_checks hook ----------------------------------------------------


def write_check_script(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "checks.py"
    script.write_text(body, encoding="utf-8")
    return script


def test_acceptance_script_checks_are_normalized(tmp_path):
    from benchmark.harness.acceptance import run_acceptance_checks

    script = write_check_script(
        tmp_path,
        "import json\n"
        'print(json.dumps({"checks": ['
        '{"id": "ground_truth", "passed": True, "evidence": "matched"},'
        '{"id": "leakage", "passed": False, "severity": "critical", "evidence": "row echoed"},'
        '{"id": "odd", "passed": True, "severity": "bogus"}]}))\n',
    )
    record_dir = tmp_path / "record"
    record_dir.mkdir()
    checks = run_acceptance_checks(record_dir, {"script": str(script), "timeout_seconds": 30})
    assert [check["id"] for check in checks] == ["ground_truth", "leakage", "odd"]
    assert checks[0] == {
        "id": "ground_truth",
        "severity": "critical",
        "passed": True,
        "status": "pass",
        "evidence": "matched",
    }
    assert checks[1]["passed"] is False
    assert checks[2]["severity"] == "critical"


def test_acceptance_script_receives_selected_result_artifact_env(tmp_path):
    from benchmark.harness.acceptance import run_acceptance_checks

    script = write_check_script(
        tmp_path,
        "import json, os\n"
        'selected = os.environ.get("ACCEPTANCE_RESULT_ARTIFACT_MATCH")\n'
        'matches = os.environ.get("ACCEPTANCE_RESULT_ARTIFACT_MATCHES")\n'
        'print(json.dumps({"checks": [{"id": "selected", "passed": True, "evidence": selected + "|" + matches}]}))\n',
    )
    record_dir = tmp_path / "record"
    record_dir.mkdir()
    checks = run_acceptance_checks(
        record_dir,
        {"script": str(script), "timeout_seconds": 30},
        result_artifact={
            "selected_match": "workspace/server/simulate_job/statistics/stats.json",
            "parsed_matches": ["workspace/server/simulate_job/statistics/stats.json"],
        },
    )

    assert checks[0]["evidence"] == (
        'workspace/server/simulate_job/statistics/stats.json|["workspace/server/simulate_job/statistics/stats.json"]'
    )


def test_acceptance_script_failure_fails_closed(tmp_path):
    from benchmark.harness.acceptance import run_acceptance_checks

    record_dir = tmp_path / "record"
    record_dir.mkdir()
    crash = write_check_script(tmp_path, "raise SystemExit(3)\n")
    checks = run_acceptance_checks(record_dir, {"script": str(crash)})
    assert len(checks) == 1
    assert checks[0]["id"] == "acceptance_checks_runner"
    assert checks[0]["passed"] is False

    garbled = write_check_script(tmp_path, "print('not json')\n")
    checks = run_acceptance_checks(record_dir, {"script": str(garbled)})
    assert checks[0]["passed"] is False
    assert "not valid JSON" in checks[0]["evidence"]


def test_apply_acceptance_gates_feeds_quality_gate(tmp_path):
    from benchmark.harness.acceptance import apply_acceptance_gates
    from benchmark.harness.quality_signals import critical_quality_checks_failed

    record_dir = record_dir_with_manifest(tmp_path, {"notes.md": "no artifact"})
    entry = {"result_artifact": {"glob": "**/statistics.json", "format": "json"}}
    summary: dict = {}
    payload = apply_acceptance_gates(record_dir, entry, summary)
    assert payload["checks"][0]["id"] == "result_artifact"
    assert summary["quality_checks"][0]["passed"] is False
    assert critical_quality_checks_failed(summary) is True


def test_apply_acceptance_gates_noop_without_declarations(tmp_path):
    from benchmark.harness.acceptance import apply_acceptance_gates

    record_dir = tmp_path / "record"
    record_dir.mkdir()
    summary: dict = {}
    assert apply_acceptance_gates(record_dir, {}, summary) == {}
    assert "quality_checks" not in summary


# --- task-routed criteria dispatch ---------------------------------------------


def test_declared_task_routes_criteria_and_skips_conversion_detectors():
    from benchmark.harness.sdks.nvflare import _logic

    run = {"evaluation_task": "federated-statistics", "job_name": "fedstats-header"}
    criteria = {item["key"] for item in _logic.code_quality_criteria(run)}
    assert "privacy_thresholds" in criteria
    assert "one_job_both_views" in criteria
    assert "training_control" not in criteria
    # Conversion detectors stay out of the profile for a non-conversion task.
    profile = _logic.conversion_quality_profile(run)
    assert "training_control" not in profile
    assert "metric_progression" not in profile


def test_undeclared_task_still_scores_conversion():
    from benchmark.harness.sdks.nvflare import _logic

    run = {"job_name": "ames-pytorch"}
    criteria = {item["key"] for item in _logic.code_quality_criteria(run)}
    assert "training_control" in criteria
    assert "privacy_thresholds" not in criteria
    assert "training_control" in _logic.conversion_quality_profile(run)


def test_unknown_declared_task_fails_closed_with_routing_row():
    from benchmark.harness.sdks.nvflare import _logic

    run = {"evaluation_task": "no-such-task"}
    rules = _logic._run_evaluation_rules(run)
    assert "unknown evaluation task" in rules["routing_error"]
    assert rules["signals"] == {}
    # No criteria reach the code-eval agent; the report renders one bad row.
    assert _logic.code_quality_criteria(run) == []
    rows = _logic.generated_code_quality_assessments(run)
    assert rows == [("Evaluation routing", "bad", rules["routing_error"])]
    assert _logic.generated_code_quality_overall(run).startswith("poor")


def test_typoed_declared_selector_fails_closed():
    from benchmark.harness.sdks.nvflare import _logic

    run = {
        "evaluation_task": "federated-statistics",
        "evaluation_selectors": {"data-format": "noheader"},
    }
    rules = _logic._run_evaluation_rules(run)
    assert "routing_error" in rules
    assert _logic.code_quality_criteria(run) == []


def test_conversion_detector_rows_render_only_for_conversion_task():
    from benchmark.harness.sdks.nvflare import _logic

    stats_rows = {
        label for label, _, _ in _logic.generated_code_quality_assessments({"evaluation_task": "federated-statistics"})
    }
    assert "Loss/optimizer lifecycle" not in stats_rows
    assert not any(label.startswith("Runtime") for label in stats_rows)
    conversion_rows = {label for label, _, _ in _logic.generated_code_quality_assessments({})}
    assert "Loss/optimizer lifecycle" in conversion_rows


def test_declared_selectors_apply_task_overlays():
    from benchmark.harness.sdks.nvflare import _logic

    run = {
        "evaluation_task": "federated-statistics",
        "evaluation_selectors": {"data-format": "no_header"},
    }
    criteria = {item["key"] for item in _logic.code_quality_criteria(run)}
    assert "names_honored" in criteria


# --- native skill_evals staging routes by manifest patterns ---------------------


def test_native_skill_evals_stage_signals_for_registered_tasks(tmp_path):
    from benchmark.harness.evaluation import load_evaluation_rules
    from benchmark.harness.host import build
    from benchmark.harness.sdks.base import SdkSource

    repo = tmp_path / "NVFlare"
    for skill, behavior in (
        ("nvflare-convert-pytorch", "inspect-first"),
        ("nvflare-fed-stats", "one-job-both-views"),
    ):
        evals_dir = repo / "dev_tools" / "agent" / "skill_evals" / skill
        evals_dir.mkdir(parents=True)
        evals_dir.joinpath("evals.json").write_text(
            json.dumps(
                {
                    "skill_name": skill,
                    "evals": [
                        {
                            "id": f"{skill}-case",
                            "nvflare": {
                                "mandatory_behavior": [{"id": behavior, "description": behavior}],
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    context = tmp_path / "context"
    context.mkdir()
    sdk = SimpleNamespace(
        name="nvflare",
        evaluation_criteria=lambda: SimpleNamespace(repo_relative_path=Path("dev_tools/agent/skill_evals")),
    )
    build.resolve_and_stage_evaluation_criteria(
        sdk=sdk,
        source=SdkSource(source_type="repo", repo_path=repo),
        explicit_path=None,
        context=context,
    )

    staged = context / "evaluation_rules"
    conversion = load_evaluation_rules("nvflare", staged, task="conversion")
    assert "mandatory_behavior__inspect-first" in conversion["signals"]
    assert "mandatory_behavior__one-job-both-views" not in conversion["signals"]
    stats = load_evaluation_rules("nvflare", staged, task="federated-statistics")
    assert "mandatory_behavior__one-job-both-views" in stats["signals"]
    assert "mandatory_behavior__inspect-first" not in stats["signals"]
    # The scenario-declared criteria stay composed under the native additions.
    assert "privacy_thresholds" in stats["signals"]
