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

"""Lock the DEEP causal "Why" analysis under a realistic-usage fixture.

The canonical snapshot (``skills_benchmark_report_snapshot_test``) renders the
full report from a deliberately minimal fixture that STARVES the deep causal
branches in ``benchmark_insights.why_section``: no cache/cost token breakdown,
identical install paths, one job span per run. As a result those branches never
appear in the existing NVFLARE golden.

This module is ADDITIVE. It builds a *separate*, richer result root
(``tests/_deep_why_fixtures.py``), renders ONLY the ``why_section`` output under
the real NVFLARE report plugin, and:

1. snapshots it to its own golden (``tests/golden/deep_why_realistic.md``) with
   its own ``UPDATE_GOLDENS`` support -- the existing NVFLARE goldens are never
   touched;
2. asserts the deep causal notes are PRESENT (non-vacuous), so the test cannot
   pass with a surface-level render.

Regenerate the golden with::

    UPDATE_GOLDENS=1 PYTHONPATH=. python3 -m pytest \
        tests/skills_benchmark_deep_why_snapshot_test.py -q
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path

import pytest
from _deep_why_fixtures import WITH_SKILLS_MODE, WITHOUT_SKILLS_MODE, build_result_root

GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN_NAME = "deep_why_realistic.md"

# Render order is deterministic: baseline first, then the with-skills run.
_MODES = [WITHOUT_SKILLS_MODE, WITH_SKILLS_MODE]


def _render_why(root: Path) -> str:
    """Build a real NVFLARE ctx and render only the deep ``why_section`` output."""

    from benchmark.harness.reports import benchmark_insights
    from benchmark.harness.sdks.nvflare.plugin import NvflareReportPlugin

    runs = benchmark_insights.collect_benchmark_runs(root)
    ctx = benchmark_insights._report_context(runs, _MODES, NvflareReportPlugin())
    return benchmark_insights.why_section(runs, _MODES, ctx)


def _compare_or_update(name: str, actual: str) -> tuple[bool, str]:
    golden_path = GOLDEN_DIR / name
    if os.environ.get("UPDATE_GOLDENS") == "1":
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(actual, encoding="utf-8")
        return True, ""
    assert golden_path.is_file(), f"Missing golden {golden_path}; run with UPDATE_GOLDENS=1 to create it."
    expected = golden_path.read_text(encoding="utf-8")
    if expected == actual:
        return True, ""
    diff = "\n".join(
        difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            fromfile=f"golden/{name}",
            tofile=f"actual/{name}",
            lineterm="",
        )
    )
    return False, diff


def test_deep_why_matches_golden(tmp_path):
    root = build_result_root(tmp_path / "deep_why_root")
    rendered = _render_why(root)

    ok, diff = _compare_or_update(GOLDEN_NAME, rendered)

    if os.environ.get("UPDATE_GOLDENS") == "1":
        pytest.skip("Golden regenerated (UPDATE_GOLDENS=1).")

    assert ok, f"{GOLDEN_NAME} mismatch:\n{diff}"


def test_deep_why_is_non_vacuous(tmp_path):
    """Guard that the rendered Why is the DEEP analysis, not a surface render.

    Each assertion targets a distinct deep causal branch; if any branch stops
    firing (builder regression or capture-shape drift), this fails loudly rather
    than letting the golden re-baseline to a shallower render.
    """

    root = build_result_root(tmp_path / "deep_why_root")
    rendered = _render_why(root)

    # --- token causal branches (_why_more_tokens) ---
    assert "Prompt cache re-reads are the dominant driver" in rendered
    assert "Skill documentation injected into context" in rendered
    assert "New context written to cache" in rendered
    assert "Effective cost" in rendered

    # --- slowdown branches (_why_slower) ---
    # slowdown-driver table header + representative driver rows.
    assert "| Driver |" in rendered
    assert "Total elapsed" in rendered
    assert "Dependency install" in rendered
    assert "Assistant turns" in rendered
    assert "Skill calls" in rendered

    # repeated successful job runs section (with-run has > 1 job span).
    assert "successful" in rendered and "execution" in rendered.lower()
    assert "Baseline comparison:" in rendered

    # dependency-install divergence note (heavier accelerator stack).
    assert "Dependency install path differed" in rendered

    # code-quality-vs-runtime note (loss/optimizer rebuilt inside the FL loop).
    assert "Generated-code efficiency issue aligns with slower non-install runtime" in rendered

    # The vacuous fallback must NOT be present.
    assert "Cause not resolved from available activity signals." not in rendered
