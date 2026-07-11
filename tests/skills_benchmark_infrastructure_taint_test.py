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

"""Infrastructure-tainted runs must be flagged and never win the latency comparison."""

from __future__ import annotations

from benchmark.harness.infrastructure_taint import (
    IDLE_GAP_TAINT_SECONDS,
    assess_infrastructure_taint,
    run_is_infrastructure_tainted,
)
from benchmark.harness.scenario_summaries import aggregate_results

_RECONNECT_LINE = (
    '{"message":"Reconnecting... 2/5 (stream disconnected before completion: IO error: '
    'peer closed connection without sending TLS close_notify)"}\n'
)
_MODEL_MANAGER_LINE = (
    "2026-07-05T05:01:32Z ERROR codex_models_manager::manager: "
    "failed to refresh available models: timeout waiting for child process to exit\n"
)
_CLAUDE_SOCKET_LINE = (
    "API Error: The socket connection was closed unexpectedly. "
    "For more information, pass `verbose: true` in the second argument to fetch()\n"
)


def test_idle_gap_above_threshold_taints():
    taint = assess_infrastructure_taint(None, {"max_inter_event_gap_seconds": IDLE_GAP_TAINT_SECONDS + 1})
    assert taint["tainted"] and "idle inter-event gap" in taint["reasons"][0]


def test_idle_gap_at_or_below_threshold_is_clean():
    taint = assess_infrastructure_taint(None, {"max_inter_event_gap_seconds": IDLE_GAP_TAINT_SECONDS})
    assert not taint["tainted"]


def test_provider_error_lines_taint(tmp_path):
    (tmp_path / "agent_events.jsonl").write_text(_RECONNECT_LINE + _CLAUDE_SOCKET_LINE, encoding="utf-8")
    (tmp_path / "agent_stderr.txt").write_text(_MODEL_MANAGER_LINE, encoding="utf-8")
    taint = assess_infrastructure_taint(tmp_path, {"max_inter_event_gap_seconds": 12.0})
    assert taint["tainted"]
    text = " | ".join(taint["reasons"])
    assert "provider stream reconnect" in text
    assert "provider socket disconnect" in text
    assert "model-manager failure" in text
    # The reconnect/disconnect/TLS patterns all fire on the SAME line — one reason,
    # plus one independent socket-disconnect reason from the next line.
    assert sum("agent_events.jsonl" in reason for reason in taint["reasons"]) == 2


def test_agent_prose_about_timeouts_does_not_taint(tmp_path):
    (tmp_path / "agent_events.jsonl").write_text(
        '{"message":"If the run exceeds the allowed time, report it as blocked or timed out."}\n',
        encoding="utf-8",
    )
    assert not assess_infrastructure_taint(tmp_path, {})["tainted"]


def _run(label, elapsed, *, tainted=False, quality=True, tokens=100.0):
    return {
        "mode": label,
        "comparison_type": "mode_ablation",
        "run_id": f"run_{label}",
        "quality_gate_passed": quality,
        "agent_elapsed_seconds": elapsed,
        "token_count": tokens,
        "infrastructure_taint": {"tainted": tainted, "reasons": ["x"] if tainted else []},
    }


def test_tainted_run_cannot_win_latency():
    # The tainted run is far faster on paper — provider stall accounting aside —
    # but must not be crowned: the clean, slower run wins.
    results = aggregate_results([_run("with_skills", 100.0, tainted=True), _run("without_skills", 900.0)])
    assert results["winner"]["label"] == "without_skills"
    assert results["by_label"]["with_skills"]["infrastructure_tainted_count"] == 1
    assert results["by_label"]["with_skills"]["clean_agent_elapsed_seconds"]["median"] is None


def test_all_tainted_yields_no_winner_with_explicit_policy():
    results = aggregate_results(
        [_run("with_skills", 100.0, tainted=True), _run("without_skills", 900.0, tainted=True)]
    )
    assert results["winner"] is None
    assert results["winner_policy"] == "no_infrastructure_clean_winner"


def test_quality_failed_runs_keep_prior_policy():
    results = aggregate_results(
        [_run("with_skills", 100.0, quality=False), _run("without_skills", 900.0, quality=False)]
    )
    assert results["winner"] is None
    assert results["winner_policy"] == "no_quality_qualified_winner"


def test_run_is_infrastructure_tainted_reads_summary_field():
    assert run_is_infrastructure_tainted({"infrastructure_taint": {"tainted": True}})
    assert not run_is_infrastructure_tainted({"infrastructure_taint": {"tainted": False}})
    assert not run_is_infrastructure_tainted({})


def test_auto_diagnostic_step_timeout_kills_wedged_investigator(tmp_path):
    """The automatic-diagnostics bound must kill a wedged agent process.

    Review #580: with no idle/wall timeout, a wedged investigator blocked
    scenario completion forever. Manual runs still wait without bound
    (timeout_seconds=None); the automatic path passes a backstop.
    """

    import time

    import pytest

    from benchmark.harness.rca import AgentInvocationError, _checked_agent_run

    start = time.monotonic()
    with pytest.raises(AgentInvocationError, match="exceeded the automatic-diagnostics bound"):
        _checked_agent_run(["sleep", "60"], tmp_path, timeout_seconds=1.0)
    assert time.monotonic() - start < 30


def test_auto_diagnostic_step_timeout_env_override(monkeypatch):
    from benchmark.harness.rca import AUTO_DIAGNOSTIC_STEP_TIMEOUT_SECONDS, auto_diagnostic_step_timeout_seconds

    monkeypatch.delenv("BENCHMARK_AUTO_DIAGNOSTIC_STEP_TIMEOUT_SECONDS", raising=False)
    assert auto_diagnostic_step_timeout_seconds() == AUTO_DIAGNOSTIC_STEP_TIMEOUT_SECONDS
    monkeypatch.setenv("BENCHMARK_AUTO_DIAGNOSTIC_STEP_TIMEOUT_SECONDS", "120")
    assert auto_diagnostic_step_timeout_seconds() == 120.0
    # <=0 disables the bound (waits forever, like manual runs).
    monkeypatch.setenv("BENCHMARK_AUTO_DIAGNOSTIC_STEP_TIMEOUT_SECONDS", "0")
    assert auto_diagnostic_step_timeout_seconds() is None


def test_pass_at_k_and_paired_deltas_over_repeats():
    """Issue #2 item 2: pass@k per label and paired per-task deltas with CIs."""

    from benchmark.harness.scenario_summaries import aggregate_results, pass_at_k_estimates

    runs = []
    # 3 repeats: with_skills passes 2/3 and is consistently ~100s faster.
    for repeat, (with_pass, with_time, base_time) in enumerate(
        [(True, 700.0, 800.0), (True, 710.0, 815.0), (False, 720.0, 818.0)], start=1
    ):
        common = {"comparison_type": "mode_ablation", "agent": "codex", "agent_model": "m",
                  "workflow": "default", "job_slug": "ames", "repeat_index": repeat,
                  "infrastructure_taint": {"tainted": False}}
        runs.append({**common, "mode": "with_skills", "run_id": f"w{repeat}",
                     "quality_gate_passed": with_pass, "agent_elapsed_seconds": with_time, "token_count": 100.0})
        runs.append({**common, "mode": "without_skills", "run_id": f"b{repeat}",
                     "quality_gate_passed": True, "agent_elapsed_seconds": base_time, "token_count": 150.0})

    import pytest

    results = aggregate_results(runs)
    with_agg = results["by_label"]["with_skills"]
    assert with_agg["pass_at_k"]["1"] == pytest.approx(2 / 3)
    assert with_agg["pass_at_k"]["3"] == 1.0  # at least one of 3 passed
    assert results["by_label"]["without_skills"]["pass_at_k"]["1"] == 1.0

    paired = results["paired_deltas"]
    assert paired["pair_count"] == 3
    assert paired["success_delta_mean"] == pytest.approx(-1 / 3)
    elapsed = paired["agent_elapsed_seconds"]
    assert round(elapsed["mean_delta"]) == -101
    lo, hi = elapsed["ci95"]
    assert lo <= elapsed["mean_delta"] <= hi and hi < 0  # consistently faster: CI excludes 0
    # Deterministic bootstrap: same inputs, same CI.
    assert aggregate_results(runs)["paired_deltas"]["agent_elapsed_seconds"]["ci95"] == [lo, hi]


def test_pass_at_k_estimator_matches_closed_form():
    from benchmark.harness.scenario_summaries import pass_at_k_estimates

    # n=4, c=2: pass@1 = 0.5; pass@3 = 1 - C(2,3)/C(4,3) = 1.0; pass@4 = 1.0
    estimates = pass_at_k_estimates(4, 2)
    assert estimates == {"1": 0.5, "3": 1.0, "4": 1.0}
    assert pass_at_k_estimates(1, 0) == {"1": 0.0}
    assert pass_at_k_estimates(0, 0) == {}
