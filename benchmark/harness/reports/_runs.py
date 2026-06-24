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

"""Generic run/workspace evidence-access substrate (migration step 5b/5c-pre).

A neutral leaf (stdlib only) shared by the generic report engine
(``benchmark_insights``) and the SDK plugins (``sdks/nvflare``): read captured
per-run text, resolve workspace-delta artifact paths, and read generated source.
It depends on nothing in the package, so either side can import it without an
import cycle — the prerequisite that lets SDK-specific recipe / code-quality
logic move into the plugin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def read_text(path: Path, *, max_bytes: int | None = None) -> str:
    try:
        if max_bytes is None:
            return path.read_text(encoding="utf-8", errors="replace")
        with path.open("rb") as stream:
            return stream.read(max_bytes).decode("utf-8", errors="replace")
    except Exception:
        return ""


def run_record(run: dict[str, Any]) -> dict[str, Any]:
    record = run.get("record")
    return record if isinstance(record, dict) else {}


def run_workspace_delta(run: dict[str, Any]) -> dict[str, Any]:
    record = run_record(run)
    delta = record.get("workspace_delta") if isinstance(record.get("workspace_delta"), dict) else None
    if isinstance(delta, dict):
        return delta
    delta = run.get("workspace_delta")
    return delta if isinstance(delta, dict) else {}


def manifest_paths(run: dict[str, Any], key: str) -> list[str]:
    delta = run_workspace_delta(run)
    values = delta.get(key)
    paths = []
    if isinstance(values, list):
        for item in values:
            if isinstance(item, dict) and item.get("path"):
                paths.append(str(item["path"]))
    return paths


def unique_paths(paths: list[str]) -> list[str]:
    result = []
    seen = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            result.append(path)
    return result


def _workspace_artifact_path(run: dict[str, Any], item: dict[str, Any]) -> Path | None:
    mode_dir = run.get("mode_dir")
    if not isinstance(mode_dir, Path):
        return None
    artifact_path = item.get("artifact_path") if isinstance(item, dict) else None
    if not artifact_path:
        return None
    return mode_dir / "workspace_delta" / str(artifact_path)


def combined_text(run: dict[str, Any]) -> str:
    return "\n".join(
        str(run.get(key) or "") for key in ("agent_events_text", "agent_stderr", "agent_last_message", "console_text")
    )


def _final_message_without_event_log(text: str) -> str:
    return str(text or "").split("\n{", 1)[0]


def _workspace_file_text(run: dict[str, Any], filename: str, *, max_bytes: int = 256_000) -> str:
    delta = run_workspace_delta(run)
    for key in ("changed_files", "final_structure_files"):
        values = delta.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict) or Path(str(item.get("path") or "")).name != filename:
                continue
            path = _workspace_artifact_path(run, item)
            if path and path.exists():
                return read_text(path, max_bytes=max_bytes)
    return ""


def _workspace_python_sources(run: dict[str, Any]) -> list[tuple[str, str]]:
    delta = run_workspace_delta(run)
    sources = []
    seen: set[str] = set()
    for key in ("changed_files", "final_structure_files"):
        values = delta.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            rel_path = str(item.get("path") or "")
            if not rel_path or rel_path in seen or Path(rel_path).suffix != ".py":
                continue
            path = _workspace_artifact_path(run, item)
            if path and path.exists():
                sources.append((rel_path, read_text(path, max_bytes=128_000)))
                seen.add(rel_path)
    return sources


def _all_python_workspace_text(run: dict[str, Any], *, max_files: int = 8) -> str:
    snippets = []
    for rel_path, text in _workspace_python_sources(run):
        snippets.append(f"# {rel_path}\n{text}")
        if len(snippets) >= max_files:
            return "\n\n".join(snippets)
    return "\n\n".join(snippets)
