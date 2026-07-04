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

"""Agent-driven generated-code-quality evaluation."""

from __future__ import annotations

from pathlib import Path

_CRITERIA = [
    {"key": "execution_model", "description": "client execution/model exchange"},
    {"key": "class_weighting", "description": "loss weighting from data"},
]


def test_parse_eval_assessments_normalizes_and_filters():
    from benchmark.harness.code_eval import parse_eval_assessments

    raw = (
        'here are my findings [{"key": "execution_model", "verdict": "GOOD", "evidence": "recipe"}, '
        '{"key": "class_weighting", "verdict": "nonsense", "evidence": "x"}, '
        '{"key": "not_a_real_key", "verdict": "good", "evidence": "y"}] done'
    )
    out = parse_eval_assessments(raw, _CRITERIA)
    assert out["execution_model"] == {"verdict": "good", "evidence": "recipe"}  # case-normalized
    assert out["class_weighting"]["verdict"] == "unknown"  # invalid verdict -> unknown
    assert "not_a_real_key" not in out  # unrequested keys dropped


def test_parse_eval_assessments_skips_echoed_criteria_array():
    from benchmark.harness.code_eval import parse_eval_assessments

    # Agents often echo the criteria list (key + description, no verdict)
    # before answering; the echo must not become "unknown" verdicts.
    raw = (
        'The criteria are [{"key": "execution_model", "description": "client execution"}, '
        '{"key": "class_weighting", "description": "loss weighting"}]\n'
        'My verdicts: [{"key": "execution_model", "verdict": "good", "evidence": "recipe"}]'
    )
    out = parse_eval_assessments(raw, _CRITERIA)
    assert out == {"execution_model": {"verdict": "good", "evidence": "recipe"}}


def test_parse_eval_assessments_ignores_entries_without_verdict():
    from benchmark.harness.code_eval import parse_eval_assessments

    raw = (
        '[{"key": "execution_model", "verdict": "bad", "evidence": "manual loop"}, '
        '{"key": "class_weighting", "description": "echoed, no verdict"}]'
    )
    out = parse_eval_assessments(raw, _CRITERIA)
    assert out == {"execution_model": {"verdict": "bad", "evidence": "manual loop"}}
    assert "class_weighting" not in out


def test_build_eval_prompt_lists_keys_and_asks_for_json():
    from benchmark.harness.code_eval import build_eval_prompt

    prompt = build_eval_prompt(_CRITERIA, "with_skills")
    assert "execution_model" in prompt and "class_weighting" in prompt
    assert "JSON array" in prompt and "verdict" in prompt


def test_evaluate_code_quality_persists_agent_verdicts(tmp_path):
    from benchmark.harness.code_eval import ASSESSMENT_FILENAME, evaluate_code_quality
    from benchmark.harness.common import load_json, write_json
    from benchmark.harness.modes import WITH_SKILLS_MODE

    mode_dir = tmp_path / "records" / f"mode={WITH_SKILLS_MODE}"
    mode_dir.mkdir(parents=True)
    (mode_dir / "agent_events.jsonl").write_text("{}\n", encoding="utf-8")
    write_json(
        tmp_path / "run_plan.json",
        {"entries": [{"mode": WITH_SKILLS_MODE, "record_dir": f"records/mode={WITH_SKILLS_MODE}"}]},
    )

    seen = {}

    def fake_invoker(prompt: str, cwd: Path) -> str:
        seen["prompt"] = prompt
        seen["cwd"] = cwd
        return '[{"key":"execution_model","verdict":"good","evidence":"FedAvgRecipe via SimEnv"},{"key":"class_weighting","verdict":"unknown","evidence":""}]'

    path = evaluate_code_quality(tmp_path, WITH_SKILLS_MODE, fake_invoker, _CRITERIA, agent_name="fake")
    assert path is not None and path.name == ASSESSMENT_FILENAME
    data = load_json(path)
    assert data["agent"] == "fake"
    assert data["assessments"]["execution_model"] == {"verdict": "good", "evidence": "FedAvgRecipe via SimEnv"}
    assert "execution_model" in seen["prompt"]
    # The invoker must run from the staged copy of the SELECTED run's record
    # directory, not the staged result root holding every mode's evidence.
    staged_cwd = Path(seen["cwd"])
    assert staged_cwd.name == f"mode={WITH_SKILLS_MODE}" and staged_cwd.parent.name == "records"
    assert (staged_cwd / "agent_events.jsonl").name  # path shape, staged tree is deleted after the run


def test_evaluate_code_quality_skips_when_record_dir_not_captured(tmp_path):
    from benchmark.harness.code_eval import evaluate_code_quality

    def fake_invoker(prompt, cwd):
        raise AssertionError("invoker must not run without a captured record directory")

    # No run_plan.json and no records/: mode_dir resolution falls back to a
    # nonexistent legacy dir, so there is nothing to evaluate.
    assert evaluate_code_quality(tmp_path, "with_skills", fake_invoker, _CRITERIA, agent_name="fake") is None


def test_evaluate_code_quality_no_criteria_is_noop(tmp_path):
    from benchmark.harness.code_eval import evaluate_code_quality

    def fake_invoker(prompt, cwd):
        raise AssertionError("invoker must not run without criteria")

    assert evaluate_code_quality(tmp_path, "with_skills", fake_invoker, [], agent_name="fake") is None


def test_report_prefers_agent_verdict_over_detector():
    from benchmark.harness.sdks.nvflare._logic import generated_code_quality_assessments

    # No captured code, so the detector would say "not captured"; the agent
    # verdict must win and populate the row instead.
    run = {
        "framework": "pytorch",
        "code_quality_assessment": {
            "assessments": {"execution_model": {"verdict": "good", "evidence": "recipe-based (FedAvgRecipe)"}}
        },
    }
    rows = generated_code_quality_assessments(run)
    match = [r for r in rows if r[0] == "Conversion: client execution/model exchange"]
    assert match == [("Conversion: client execution/model exchange", "good", "recipe-based (FedAvgRecipe)")]
