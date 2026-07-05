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

"""Detect infrastructure-tainted runs (provider stalls, reconnects, backend failures).

A tainted run's wall-clock time measures the model provider or the network, not
the agent or the skills, so latency comparisons must not crown a winner from it.
Detection is deliberately narrow — long idle inter-event gaps and explicit
provider/transport error lines — because a false taint hides a real skill
slowdown. Long-running *commands* (e.g. an FL simulation) still emit progress
events, so an idle inter-event gap above the threshold means a stalled model
turn, not agent work.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

# One idle stretch this long between consecutive agent events marks the run's
# latency as provider-dominated. Observed stalls are far above this (1,000s+);
# normal turns and tool calls stay far below it.
IDLE_GAP_TAINT_SECONDS = 300.0

# Explicit provider/transport failure lines observed in agent traces. Each entry
# is (compiled-pattern source, human label). Patterns must stay specific to
# infrastructure failures — generic words like "timeout" would false-positive on
# agent prose and harness prompts.
PROVIDER_ERROR_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\breconnecting\.{0,3}\s*\d+/\d+", "provider stream reconnect"),
    (r"\bstream disconnected before completion\b", "provider stream disconnect"),
    (r"\bpeer closed connection without sending tls close_notify\b", "TLS connection drop"),
    (r"\bfailed to refresh available models\b", "model-manager failure"),
    (r"\boverloaded_error\b", "provider overloaded"),
)

_EVIDENCE_FILES = ("agent_events.jsonl", "agent_stderr.txt")
_MAX_SCAN_BYTES = 8 * 1024 * 1024
_EXCERPT_CHARS = 160


def _pattern_hits(text: str, source_name: str) -> list[str]:
    hits = []
    reported_lines: set[int] = set()
    for pattern, label in PROVIDER_ERROR_PATTERNS:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if not matches:
            continue
        first = matches[0]
        line_start = text.rfind("\n", 0, first.start()) + 1
        # Several patterns often fire on the same error line (a reconnect line
        # also names the disconnect and the TLS drop); report each line once.
        if line_start in reported_lines:
            continue
        reported_lines.add(line_start)
        excerpt = text[line_start : line_start + _EXCERPT_CHARS].splitlines()[0].strip()
        hits.append(f"{label} ({len(matches)}x in {source_name}): {excerpt}")
    return hits


def assess_infrastructure_taint(record_dir: Path | None, activity: Mapping[str, Any] | None) -> dict[str, Any]:
    """Assess one run's captured trace for infrastructure-dominated latency.

    ``activity`` is the run's parsed activity block (carries
    ``max_inter_event_gap_seconds``); ``record_dir`` holds the captured
    ``agent_events.jsonl`` / ``agent_stderr.txt`` scanned for provider errors.
    """

    reasons: list[str] = []
    gap = (activity or {}).get("max_inter_event_gap_seconds")
    gap_value = float(gap) if isinstance(gap, (int, float)) and not isinstance(gap, bool) else None
    if gap_value is not None and gap_value > IDLE_GAP_TAINT_SECONDS:
        reasons.append(
            f"idle inter-event gap of {gap_value:.0f}s exceeds the {IDLE_GAP_TAINT_SECONDS:.0f}s threshold"
        )
    if record_dir is not None:
        for name in _EVIDENCE_FILES:
            path = Path(record_dir) / name
            try:
                text = path.read_text(encoding="utf-8", errors="replace")[:_MAX_SCAN_BYTES]
            except OSError:
                continue
            reasons.extend(_pattern_hits(text, name))
    return {
        "tainted": bool(reasons),
        "reasons": reasons,
        "idle_gap_threshold_seconds": IDLE_GAP_TAINT_SECONDS,
        "max_inter_event_gap_seconds": gap_value,
    }


def run_is_infrastructure_tainted(run: Mapping[str, Any]) -> bool:
    taint = run.get("infrastructure_taint")
    return bool(isinstance(taint, Mapping) and taint.get("tainted"))
