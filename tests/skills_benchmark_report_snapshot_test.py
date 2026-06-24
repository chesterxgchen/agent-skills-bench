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

"""Migration step 1: lock report output compatibility with golden snapshots.

The current report output IS the contract. This module builds a deterministic
result root (see ``tests/_report_fixtures.py``), runs the REAL report entry
points, normalizes machine-specific noise, and compares each output to a
committed golden under ``tests/golden/``.

Regenerate goldens with::

    UPDATE_GOLDENS=1 PYTHONPATH=. python3 -m pytest \
        tests/skills_benchmark_report_snapshot_test.py -q

Determinism handled here:
- The absolute result-root path is written into several outputs; it is
  normalized to ``<RESULT_ROOT>`` before comparison and before storing goldens.
- ``replay_metadata.json``/scenario-report ``replayed_at`` uses ``datetime.now``;
  it is frozen via monkeypatch to a fixed instant so the value is stable.
- Event timestamps / elapsed seconds are fixed in the fixture, so all derived
  durations are stable.
"""

from __future__ import annotations

import difflib
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from _report_fixtures import build_result_root

GOLDEN_DIR = Path(__file__).parent / "golden"

# Fixed replay instant; matches the format runner.replay_result_root produces.
_FROZEN_REPLAY_AT = datetime(2026, 6, 13, 21, 0, 0, tzinfo=timezone.utc)


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):  # noqa: D102 - drop-in for datetime.now
        if tz is None:
            return _FROZEN_REPLAY_AT.replace(tzinfo=None)
        return _FROZEN_REPLAY_AT.astimezone(tz)


def _normalize(text: str, result_root: Path) -> str:
    """Replace machine-specific noise with stable placeholders."""

    normalized = text.replace(str(result_root.resolve()), "<RESULT_ROOT>")
    normalized = normalized.replace(str(result_root), "<RESULT_ROOT>")
    return normalized


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


def _generate_outputs(root: Path, monkeypatch) -> dict[str, str]:
    """Run the real report entry points; return name -> normalized text."""

    from benchmark.harness.host import runner
    from benchmark.harness.reports import benchmark_insights, metrics_report

    # Freeze datetime.now so replay_metadata/scenario-report timestamps are stable.
    monkeypatch.setattr(runner, "datetime", _FrozenDatetime)

    outputs: dict[str, str] = {}

    # benchmark_insights.md via the real collect + report entry points.
    runs = benchmark_insights.collect_benchmark_runs(root)
    outputs["benchmark_insights.md"] = benchmark_insights.benchmark_report(root, runs)

    # metrics_report.{md,json} via the real writer.
    metrics_report.write_reports(root, "Benchmark Metrics")
    outputs["metrics_report.md"] = (root / "metrics_report.md").read_text(encoding="utf-8")
    outputs["metrics_report.json"] = (root / "metrics_report.json").read_text(encoding="utf-8")

    # scenario report via the runner replay path.
    runner.replay_result_root(root)
    runner.write_benchmark_reports(root)
    outputs["reports/scenario_report.md"] = (root / "reports" / "scenario_report.md").read_text(encoding="utf-8")
    outputs["reports/scenario_report.json"] = (root / "reports" / "scenario_report.json").read_text(encoding="utf-8")

    return {name: _normalize(text, root) for name, text in outputs.items()}


def test_report_outputs_match_goldens(tmp_path, monkeypatch):
    root = build_result_root(tmp_path / "result_root")
    outputs = _generate_outputs(root, monkeypatch)

    failures: list[str] = []
    for name, actual in sorted(outputs.items()):
        ok, diff = _compare_or_update(name, actual)
        if not ok:
            failures.append(f"{name} mismatch:\n{diff}")

    if os.environ.get("UPDATE_GOLDENS") == "1":
        pytest.skip("Goldens regenerated (UPDATE_GOLDENS=1).")

    assert not failures, "\n\n".join(failures)
