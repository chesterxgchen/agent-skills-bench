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

"""Report maturity derived from the detached diagnostics lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

DIAGNOSTICS_STATUS_FILENAME = "diagnostics_status.json"


def diagnostics_report_state(root: Path) -> str:
    path = root / DIAGNOSTICS_STATUS_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return ""
    status = str(payload.get("status") or "") if isinstance(payload, dict) else ""
    if status in {"pending", "running"}:
        return "preliminary"
    if status in {"finalizing", "done"}:
        return "final"
    if status == "failed":
        return "diagnostics_failed"
    return ""


def markdown_report_state_note(root: Path) -> str:
    state = diagnostics_report_state(root)
    if state == "preliminary":
        return (
            "**Report state: PRELIMINARY.** Automatic root-cause and code-quality diagnostics are still "
            "running. Do not treat this report as final until `diagnostics_status.json` reports `done`."
        )
    if state == "final":
        return "**Report state: FINAL.** Automatic diagnostics have been applied."
    if state == "diagnostics_failed":
        return (
            "**Report state: DIAGNOSTICS FAILED.** Deterministic results are available, but the automatic "
            "root-cause/code-quality pass did not complete; inspect `diagnostics_status.json`."
        )
    return ""
