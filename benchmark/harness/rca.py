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

"""Agent-driven root-cause analysis over a captured benchmark run.

The deterministic evidence extractors only SEED the investigation: they name
the terminal failure (e.g. ``ModuleNotFoundError: torch`` from ``python3
job.py``). From there an investigator agent drives the analysis itself — the
harness asks the seed question, the agent answers from the captured evidence
files and proposes the next question ("was a requirements install expected?",
"was the instruction missing, or present but not followed?", "why was it not
followed?"), and the loop continues until the agent declares a conclusion or
the step budget runs out. Every step is recorded verbatim in
``rca/investigation.jsonl``; the final synthesis is written to
``rca/rca_report.md``, which the report engine embeds in the Why section.

The loop is topic- and SDK-independent: it reads only generic captured
artifacts (``run_summary.json``, ``agent_events.jsonl``, ``prompt.txt``,
``workspace_delta/``) and never imports SDK plugins. Built-in topics seed the
common questions — a terminal failure, "why is this run slower", "why did it
use more tokens" — and ``--question`` seeds any other investigation.

Standalone usage (after a benchmark run)::

    python -m benchmark.harness.rca <result_root> [--mode with_skills] \
        [--topic auto|failure|slowdown|tokens] [--question "..."] \
        [--agent claude|codex] [--max-steps 8]

The investigator agent runs with its working directory set to the result
root, so every answer can be grounded by reading the captured artifacts
(``agent_events.jsonl``, ``prompt.txt``, skill references quoted in event
output, workspace deltas, ...).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .common import load_json, write_json
from .reports._events import error_signature_from_output, event_timeline_from_text
from .reports._loader import mode_dir_for_benchmark

MAX_INVESTIGATION_STEPS = 8
AGENT_STEP_TIMEOUT_SECONDS = 300

_ANSWER_SCHEMA_INSTRUCTIONS = """\
Answer strictly as a single JSON object with these fields:
{
  "answer": "<direct answer to the question, grounded in the evidence files>",
  "evidence": [{"file": "<path relative to the result root>", "quote": "<verbatim line(s) supporting the answer>"}],
  "next_question": "<the next question the investigation must answer, or null if the root cause is established>",
  "conclusion": "<null until established; then the root-cause statement>"
}
Rules:
- Only claim what the evidence files show; quote them. If evidence is absent, say so in the answer.
- Drive toward the root cause: symptom -> missing/failed action -> why that action was missing/failed
  (e.g. instruction absent, instruction present but not followed, environment constraint) -> final cause.
- Set "conclusion" only when a further question would not change the analysis.
"""


@dataclass
class InvestigationStep:
    question: str
    answer: str
    evidence: list[dict[str, str]] = field(default_factory=list)
    next_question: str | None = None
    conclusion: str | None = None


AgentInvoker = Callable[[str, Path], str]


def _claude_invoker(prompt: str, cwd: Path) -> str:
    result = subprocess.run(
        ["claude", "-p", prompt],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=AGENT_STEP_TIMEOUT_SECONDS,
    )
    return result.stdout


def _codex_invoker(prompt: str, cwd: Path) -> str:
    result = subprocess.run(
        ["codex", "exec", "--skip-git-repo-check", "-"],
        cwd=cwd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=AGENT_STEP_TIMEOUT_SECONDS,
    )
    return result.stdout


_AGENT_INVOKERS: dict[str, AgentInvoker] = {
    "claude": _claude_invoker,
    "codex": _codex_invoker,
}


def resolve_invoker(agent: str | None) -> tuple[str, AgentInvoker]:
    if agent:
        if agent not in _AGENT_INVOKERS:
            raise SystemExit(f"Unknown RCA agent {agent!r}; choose from {sorted(_AGENT_INVOKERS)}.")
        return agent, _AGENT_INVOKERS[agent]
    for name in ("claude", "codex"):
        if shutil.which(name):
            return name, _AGENT_INVOKERS[name]
    raise SystemExit("No investigator agent CLI found on PATH (looked for: claude, codex).")


def parse_agent_answer(raw: str) -> dict[str, Any]:
    """Extract the JSON answer object from agent output (tolerates prose around it)."""

    candidates = re.findall(r"\{.*\}", raw, flags=re.DOTALL)
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "answer" in payload:
            return payload
    return {"answer": raw.strip(), "evidence": [], "next_question": None, "conclusion": None}


def _other_mode(result_root: Path, mode: str) -> str | None:
    run_plan = load_json(result_root / "run_plan.json", {}) or {}
    entries = run_plan.get("entries") if isinstance(run_plan, dict) else None
    modes = [str(entry.get("mode")) for entry in entries or [] if isinstance(entry, dict) and entry.get("mode")]
    return next((candidate for candidate in modes if candidate != mode), None)


def _mode_summary_number(result_root: Path, mode: str, key: str) -> float | None:
    summary = load_json(mode_dir_for_benchmark(result_root, mode) / "run_summary.json", {}) or {}
    value = summary.get(key) if isinstance(summary, dict) else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _base_seed(result_root: Path, mode: str, topic: str) -> dict[str, Any]:
    mode_dir = mode_dir_for_benchmark(result_root, mode)
    seed: dict[str, Any] = {
        "topic": topic,
        "mode": mode,
        "mode_dir": str(mode_dir),
        "events_file": str((mode_dir / "agent_events.jsonl").relative_to(result_root)),
    }
    base_mode = _other_mode(result_root, mode)
    if base_mode:
        seed["base_mode"] = base_mode
        seed["base_mode_dir"] = str(mode_dir_for_benchmark(result_root, base_mode))
    return seed


def seed_failure_context(result_root: Path, mode: str) -> dict[str, Any] | None:
    """Deterministic seed only: the terminal failure the agent will investigate."""

    mode_dir = mode_dir_for_benchmark(result_root, mode)
    events_path = mode_dir / "agent_events.jsonl"
    if not events_path.is_file():
        return None
    timeline = event_timeline_from_text(events_path.read_text(encoding="utf-8", errors="replace"))
    anchor = None
    signature = None
    for item in timeline:
        if item["kind"] != "command":
            continue
        candidate = error_signature_from_output(str(item.get("output") or ""))
        if candidate and item.get("exit_code") not in (0,):
            anchor, signature = item, candidate
    if anchor is None or signature is None:
        return None
    seed = _base_seed(result_root, mode, "failure")
    seed.update(
        {
            "headline": f"command `{anchor.get('command')}` failed with `{signature['display']}`",
            "seed_question": (
                f"The {mode} run's command `{anchor.get('command')}` failed with `{signature['display']}`. "
                "Based on the captured evidence, what immediate condition caused this failure "
                "(e.g. a required setup action that never ran, a wrong path, a broken environment)?"
            ),
            "failed_command": anchor.get("command"),
            "error": signature["display"],
            "error_kind": signature["kind"],
        }
    )
    return seed


def seed_slowdown_context(result_root: Path, mode: str) -> dict[str, Any] | None:
    """Seed a 'why did this run take longer' investigation from the elapsed delta."""

    seed = _base_seed(result_root, mode, "slowdown")
    base_mode = seed.get("base_mode")
    elapsed = _mode_summary_number(result_root, mode, "elapsed_seconds")
    base_elapsed = _mode_summary_number(result_root, base_mode, "elapsed_seconds") if base_mode else None
    if elapsed is None or base_elapsed is None or elapsed <= base_elapsed:
        return None
    delta = elapsed - base_elapsed
    seed.update(
        {
            "headline": f"{mode} took {elapsed:.0f}s vs {base_elapsed:.0f}s for {base_mode} (+{delta:.0f}s)",
            "seed_question": (
                f"The {mode} run took {elapsed:.0f}s while {base_mode} took {base_elapsed:.0f}s (+{delta:.0f}s). "
                "From the captured evidence, what operations account for the extra time — what did this run "
                f"spend time doing that {base_mode} did not?"
            ),
        }
    )
    return seed


def seed_token_context(result_root: Path, mode: str) -> dict[str, Any] | None:
    """Seed a 'why did this run use more tokens' investigation from the usage delta."""

    seed = _base_seed(result_root, mode, "tokens")
    base_mode = seed.get("base_mode")
    tokens = _mode_summary_number(result_root, mode, "token_count")
    base_tokens = _mode_summary_number(result_root, base_mode, "token_count") if base_mode else None
    if tokens is None or base_tokens is None or tokens <= base_tokens:
        return None
    delta = tokens - base_tokens
    seed.update(
        {
            "headline": f"{mode} used {tokens:.0f} tokens vs {base_tokens:.0f} for {base_mode} (+{delta:.0f})",
            "seed_question": (
                f"The {mode} run used {tokens:.0f} tokens while {base_mode} used {base_tokens:.0f} "
                f"(+{delta:.0f}). From the captured evidence (event stream, assistant turns, tool calls, files "
                "read), what activity consumed the extra tokens — what was this run doing more of, and why?"
            ),
        }
    )
    return seed


def seed_custom_context(result_root: Path, mode: str, question: str) -> dict[str, Any]:
    seed = _base_seed(result_root, mode, "custom")
    seed.update({"headline": question, "seed_question": question})
    return seed


_TOPIC_SEEDERS: dict[str, Callable[[Path, str], dict[str, Any] | None]] = {
    "failure": seed_failure_context,
    "slowdown": seed_slowdown_context,
    "tokens": seed_token_context,
}


def resolve_seed(result_root: Path, mode: str, topic: str, question: str | None) -> dict[str, Any] | None:
    if question:
        return seed_custom_context(result_root, mode, question)
    if topic == "auto":
        for name in ("failure", "slowdown", "tokens"):
            seed = _TOPIC_SEEDERS[name](result_root, mode)
            if seed is not None:
                return seed
        return None
    seeder = _TOPIC_SEEDERS.get(topic)
    return seeder(result_root, mode) if seeder else None


def _step_prompt(seed: dict[str, Any], steps: list[InvestigationStep], question: str, result_root: Path) -> str:
    history = ""
    if steps:
        history_lines = []
        for index, step in enumerate(steps, start=1):
            history_lines.append(f"Q{index}: {step.question}\nA{index}: {step.answer}")
        history = "Findings so far:\n" + "\n".join(history_lines) + "\n\n"
    base_pointer = ""
    if seed.get("base_mode_dir"):
        base_pointer = (
            f" The comparison baseline ({seed['base_mode']}) evidence is under "
            f"`{Path(seed['base_mode_dir']).relative_to(result_root)}/` — contrast against it where relevant."
        )
    return (
        "You are investigating the root cause of a benchmark observation. "
        f"Your working directory is the benchmark result root; the investigated run's captured evidence is under "
        f"`{Path(seed['mode_dir']).relative_to(result_root)}/` "
        f"(agent event stream: `{seed['events_file']}`; the agent's input prompt: prompt.txt; "
        f"generated/changed files: workspace_delta/).{base_pointer}\n\n"
        f"Observation under investigation: {seed['headline']}.\n\n"
        f"{history}"
        f"Current question: {question}\n\n" + _ANSWER_SCHEMA_INSTRUCTIONS
    )


def _synthesis_prompt(seed: dict[str, Any], steps: list[InvestigationStep]) -> str:
    qa = "\n".join(
        f"Q{index}: {step.question}\nA{index}: {step.answer}\nEvidence: "
        + "; ".join(f"{item.get('file')}: {item.get('quote')}" for item in step.evidence)
        + (f"\nConclusion: {step.conclusion}" if step.conclusion else "")
        for index, step in enumerate(steps, start=1)
    )
    return (
        "Write the final root-cause analysis report for this benchmark observation as markdown. "
        "Base it ONLY on the investigation findings below; keep every quoted evidence reference. "
        "Structure: `### Root cause` (one paragraph naming the causal chain), `### Evidence` "
        "(bullet list, each with its file reference), `### Recommendation` (what change prevents this "
        "class of issue). Do not add headers above ###.\n\n"
        f"Observation investigated: {seed['headline']}.\n\n"
        f"Investigation trail:\n{qa}\n"
    )


def run_investigation(
    result_root: Path,
    mode: str,
    invoker: AgentInvoker,
    *,
    topic: str = "auto",
    question: str | None = None,
    max_steps: int = MAX_INVESTIGATION_STEPS,
    agent_name: str = "agent",
) -> Path | None:
    seed = resolve_seed(result_root, mode, topic, question)
    if seed is None:
        print(f"No investigable observation for mode={mode} topic={topic} (no failure/slowdown/token delta found).")
        return None
    steps: list[InvestigationStep] = []
    question = str(seed["seed_question"])
    for _ in range(max_steps):
        raw = invoker(_step_prompt(seed, steps, question, result_root), result_root)
        payload = parse_agent_answer(raw)
        step = InvestigationStep(
            question=question,
            answer=str(payload.get("answer") or "").strip(),
            evidence=[item for item in payload.get("evidence") or [] if isinstance(item, dict)],
            next_question=(str(payload["next_question"]).strip() if payload.get("next_question") else None),
            conclusion=(str(payload["conclusion"]).strip() if payload.get("conclusion") else None),
        )
        steps.append(step)
        if step.conclusion or not step.next_question:
            break
        question = step.next_question
    report_markdown = invoker(_synthesis_prompt(seed, steps), result_root).strip()

    mode_dir = Path(seed["mode_dir"])
    rca_dir = mode_dir / "rca"
    rca_dir.mkdir(parents=True, exist_ok=True)
    topic_slug = str(seed["topic"])
    with (rca_dir / f"investigation_{topic_slug}.jsonl").open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"seed": seed, "agent": agent_name}) + "\n")
        for step in steps:
            stream.write(
                json.dumps(
                    {
                        "question": step.question,
                        "answer": step.answer,
                        "evidence": step.evidence,
                        "next_question": step.next_question,
                        "conclusion": step.conclusion,
                    }
                )
                + "\n"
            )
    report_path = rca_dir / f"rca_report_{topic_slug}.md"
    conclusion = next((step.conclusion for step in reversed(steps) if step.conclusion), "")
    header = (
        f"**{seed['headline']}** — investigated by `{agent_name}` over {len(steps)} question(s); "
        f"trail: `rca/investigation_{topic_slug}.jsonl`."
    )
    body = report_markdown if report_markdown else f"### Root cause\n\n{conclusion or 'not established'}"
    report_path.write_text(f"{header}\n\n{body}\n", encoding="utf-8")
    return report_path


def _regenerate_reports(result_root: Path) -> None:
    from .reports.benchmark_insights import benchmark_report, collect_benchmark_runs

    runs = collect_benchmark_runs(result_root)
    (result_root / "benchmark_insights.md").write_text(benchmark_report(result_root, runs), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--mode", default="with_skills", help="Run mode to investigate (default: with_skills)")
    parser.add_argument(
        "--topic",
        default="auto",
        choices=["auto", "failure", "slowdown", "tokens"],
        help="What to investigate: a terminal failure, the elapsed-time delta, or the token-usage delta "
        "(auto picks the first that applies).",
    )
    parser.add_argument("--question", help="Free-form investigation question (overrides --topic).")
    parser.add_argument("--agent", help="Investigator agent CLI: claude or codex (default: first found on PATH)")
    parser.add_argument("--max-steps", type=int, default=MAX_INVESTIGATION_STEPS)
    parser.add_argument(
        "--no-report-refresh",
        action="store_true",
        help="Skip regenerating benchmark_insights.md after the investigation.",
    )
    args = parser.parse_args(argv)
    agent_name, invoker = resolve_invoker(args.agent)
    report_path = run_investigation(
        args.result_root,
        args.mode,
        invoker,
        topic=args.topic,
        question=args.question,
        max_steps=args.max_steps,
        agent_name=agent_name,
    )
    if report_path is None:
        return 1
    print(report_path)
    if not args.no_report_refresh:
        _regenerate_reports(args.result_root)
        print(args.result_root / "benchmark_insights.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
