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
        return '[{"key":"execution_model","verdict":"good","evidence":"FedAvgRecipe via SimEnv"},{"key":"class_weighting","verdict":"unknown","evidence":""}]'

    path = evaluate_code_quality(tmp_path, WITH_SKILLS_MODE, fake_invoker, _CRITERIA, agent_name="fake")
    assert path is not None and path.name == ASSESSMENT_FILENAME
    data = load_json(path)
    assert data["agent"] == "fake"
    assert data["assessments"]["execution_model"] == {"verdict": "good", "evidence": "FedAvgRecipe via SimEnv"}
    assert "execution_model" in seen["prompt"]


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
