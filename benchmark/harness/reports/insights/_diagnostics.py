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

"""E2 split module (mechanical, behavior-preserving)."""

from __future__ import annotations

import re
from typing import Any

from .._context import CommandFailureSignal, JobExecutionSignal
from .._events import (
    agent_failure_category,
    bash_permission_denial_count,
    exit_code,
    failure_evidence,
    inline_code_text,
    unsupported_model_message,
)
from .._runs import combined_text
from .._text import markdown_cell
from ..evidence import RunEvidence
from ._plugin_view import (
    _evidence_or_legacy,
    _execution_atom,
    _execution_run_noun,
    _result_term,
    metric_log_lines,
    permission_denial_commands,
)

__all__ = [
    "bash_blocked_diagnostic",
    "command_failure_diagnostic_items",
    "command_failure_diagnostics",
    "command_failure_diagnostics_table",
    "successful_job_evidence",
    "failure_root_cause",
]


def bash_blocked_diagnostic(run: RunEvidence, *, recovered: bool = False, ctx: Any = None) -> str | None:
    """Return a diagnostic string if Bash was blocked due to permission approval failures."""
    blocked_count = bash_permission_denial_count(run.raw)
    if blocked_count == 0:
        return None
    if recovered:
        denied_commands = permission_denial_commands(run)
        command = f" Denied command: `{inline_code_text(denied_commands[0], 180)}`." if denied_commands else ""
        atom = _execution_atom(ctx)
        job_atom = f"job{'/' + atom if atom else ''}"
        return (
            f"Bash tool was blocked {blocked_count} time(s) earlier in this run, but a later {job_atom} "
            f"command completed.{command} This usually means Claude rejected that specific command shape "
            "rather than Bash being unavailable for the whole run; it is still reported because the recovery "
            "costs extra tool turns, tokens, and elapsed time."
        )
    hint_counts = run.activity.get("hint_counts") or {}
    sim_count = hint_counts.get("simulation", 0)
    py_count = hint_counts.get("python_job_py", 0)
    run_noun = _execution_run_noun(ctx)
    impact = ""
    if sim_count == 0 and py_count == 0:
        impact = f" The {run_noun} was never run as a result."
    elif sim_count == 0:
        impact = f" The {run_noun} step was never run."
    return (
        f"Bash tool was blocked {blocked_count} time(s) with 'requested permissions' errors. "
        f"In Claude Code --print (non-interactive) mode, tools require explicit allow rules even with "
        f"--dangerously-skip-permissions. Check that (1) BENCHMARK_AGENT_HOME/settings.json has "
        f"`Bash(*)` in permissions.allow, (2) the agent launch argv uses the configured `--tools` mode, "
        f"and (3) no deny/ask rules exist at /etc/claude-code/managed-settings.json inside Docker. "
        f"Rebuild the Docker image after any agent config changes.{impact}"
    )


def command_failure_diagnostic_items(
    run: RunEvidence | dict[str, Any], limit: int = 3, ev: Any = None
) -> list[dict[str, str]]:
    rows = (_evidence_or_legacy(ev, run).command_failure or CommandFailureSignal()).rows
    return [dict(row) for row in rows][:limit]


def command_failure_diagnostics(run: RunEvidence | dict[str, Any], limit: int = 3, ev: Any = None) -> list[str]:
    diagnostics = []
    for item in command_failure_diagnostic_items(run, limit=limit, ev=ev):
        dependency_evidence = ""
        if item.get("dependency"):
            dependency_evidence = f" Dependency install evidence: {item['dependency']}."
        diagnostics.append(
            f"Command `{item['command']}` failed with exit {item['exit']}; {item['recovery']}. "
            f"Root cause evidence: {item['root_cause']}.{dependency_evidence}"
        )
    return diagnostics


def command_failure_diagnostics_table(
    run: RunEvidence | dict[str, Any],
    *,
    limit: int = 3,
    recovered_only: bool = False,
    ev: Any = None,
) -> str:
    items = command_failure_diagnostic_items(run, limit=limit, ev=ev)
    if recovered_only:
        items = [item for item in items if "not recovered in this run" not in item["recovery"]]
    if not items:
        return ""
    lines = [
        "| Command | Exit | Recovery | Root cause | Dependency evidence |",
        "|---|---:|---|---|---|",
    ]
    for item in items:
        dependency = item["dependency"] or "none"
        command = markdown_cell(f"`{item['command']}`")
        lines.append(
            f"| {command} | {markdown_cell(item['exit'])} | {markdown_cell(item['recovery'])} | "
            f"{markdown_cell(item['root_cause'])} | {markdown_cell(dependency)} |"
        )
    return "\n".join(lines)


def successful_job_evidence(run: RunEvidence | dict[str, Any], ev: Any = None, ctx: Any = None) -> str:
    event = (_evidence_or_legacy(ev, run).job_execution or JobExecutionSignal()).last_successful_job_event
    if not event:
        return ""
    output = str(event.get("output") or "")
    atom = _execution_atom(ctx)
    parts = [f"a later job{'/' + atom if atom else ''} command exited 0"]
    if "Finished" in output:
        parts.append(f"the {_result_term(ctx)}workflow reached a Finished state")
    workspace = re.search(r"Result workspace:\s*([^\n]+)", output)
    if workspace:
        parts.append(f"result workspace `{workspace.group(1).strip()}`")
    metric_lines = metric_log_lines(output, ctx=ctx)
    if metric_lines:
        parts.append("log metrics: " + "; ".join(metric_lines))
    return "; ".join(parts)


def failure_root_cause(run: RunEvidence) -> str:
    record = run.record if isinstance(run.record, dict) else {}
    failure_category = agent_failure_category(run.raw)
    if failure_category and failure_category != "agent_unknown_failure":
        return f"Agent failure category: {failure_category}"
    text = combined_text(run.raw)
    model_error = unsupported_model_message(text)
    if model_error:
        return f"Agent model selection failed: {model_error}"
    lowered = text.lower()
    if "pull access denied" in lowered or "unable to find image" in lowered:
        return "Docker image unavailable: build the benchmark Docker images before running."
    error = record.get("harness_error") if isinstance(record.get("harness_error"), dict) else {}
    if error.get("message"):
        return f"Harness failure: {error['message']}"
    evidence = failure_evidence(run.raw)
    if evidence:
        return evidence
    if failure_category:
        return f"Agent failure category: {failure_category}"
    code = exit_code(run.raw)
    if code not in (None, 0):
        return f"Agent container failed with exit code {code}."
    return "No failure detected."
