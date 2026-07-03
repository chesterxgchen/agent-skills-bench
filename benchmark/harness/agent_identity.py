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

"""Agent identity helpers derived from captured runtime evidence."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

UNSPECIFIED_AGENT_MODEL = "unspecified_default"
OBSERVED_AGENT_MODEL_SOURCE = "agent_events"
OBSERVED_AGENT_LOG_MODEL_SOURCE = "agent_log"
OBSERVED_SESSION_MODEL_SOURCE = "agent_session"
MAX_AGENT_EVENTS_TEXT_BYTES = 20 * 1024 * 1024

_MODEL_SUFFIX = re.compile(r"\[[^\]]+\]\s*$")
_LOG_MODEL_PATTERNS = (
    re.compile(
        r"\b(?:selected|current|effective|resolved|using|used)\s+(?:agent\s+)?model\s*[:=]\s*['\"]?([A-Za-z0-9_.+/-]+)",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:selected|current|effective|resolved)_model\s*[:=]\s*['\"]?([A-Za-z0-9_.+/-]+)", re.IGNORECASE),
)
_LOG_MODEL_KEYS = ("selected_model", "current_model", "effective_model", "resolved_model", "agent_model")


def clean_agent_model_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return _MODEL_SUFFIX.sub("", text).strip()


def observed_agent_model_from_events_text(events_text: str) -> str:
    """Return the model reported by the captured agent event stream, if present.

    Claude streams include a clean nested ``message.model`` on assistant events and
    a decorated ``model`` on ``system.init``. Prefer the assistant message value,
    but keep the system-level value as a fallback for adapters that only expose it.
    """

    fallback = ""
    for line in str(events_text or "").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        message = event.get("message")
        if isinstance(message, dict):
            model = clean_agent_model_name(message.get("model"))
            if model:
                return model
        # Codex session rollouts carry the per-turn model on
        # turn_context.payload.model.  Require the event type so an unrelated
        # payload (for example, tool output describing an application model)
        # cannot be mistaken for agent identity.
        payload = event.get("payload")
        if event.get("type") == "turn_context" and isinstance(payload, dict):
            model = clean_agent_model_name(payload.get("model"))
            if model:
                return model
        if not fallback:
            fallback = clean_agent_model_name(event.get("model"))
    return fallback


def observed_agent_model_from_log_text(log_text: str) -> str:
    """Return an explicitly selected model from agent logs, if one is present.

    This intentionally ignores generic model-catalog fields such as ``slug`` or
    ``upgrade.model``. Those are available-model metadata, not evidence of the
    model used for the measured run.
    """

    for line in str(log_text or "").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            event = None
        if isinstance(event, dict):
            for key in _LOG_MODEL_KEYS:
                model = clean_agent_model_name(event.get(key))
                if model:
                    return model
            model_info = event.get("model_info")
            if isinstance(model_info, dict):
                model = clean_agent_model_name(model_info.get("model") or model_info.get("name"))
                if model:
                    return model
        for pattern in _LOG_MODEL_PATTERNS:
            match = pattern.search(line)
            if match:
                return clean_agent_model_name(match.group(1))
    return ""


def observed_agent_model_from_events_path(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            text = stream.read(MAX_AGENT_EVENTS_TEXT_BYTES).decode("utf-8", errors="replace")
    except Exception:
        return ""
    return observed_agent_model_from_events_text(text)


def is_resolved_agent_model(model: Any) -> bool:
    text = str(model or "").strip()
    return bool(text) and text != UNSPECIFIED_AGENT_MODEL


def preferred_agent_model(*candidates: tuple[Any, Any]) -> tuple[Any, Any]:
    """Pick the first ``(model, source)`` pair with a genuinely resolved model.

    The ``unspecified_default`` sentinel is a placeholder, not a resolution — a
    later candidate carrying a concrete model name outranks an earlier sentinel
    (e.g. a plan-time entry that predates the container's config-file lookup).
    Falls back to the first non-empty model/source independently when no
    candidate is resolved.
    """

    for model, source in candidates:
        if is_resolved_agent_model(model):
            return model, source
    model = next((model for model, _ in candidates if model not in (None, "")), None)
    source = next((source for _, source in candidates if source not in (None, "")), None)
    return model, source


def agent_session_file_snapshot(sessions_dir: Path) -> dict[str, tuple[int, int]]:
    """Return stable-enough file state for correlating rollouts with one invocation."""

    snapshot: dict[str, tuple[int, int]] = {}
    try:
        paths = sessions_dir.rglob("*.jsonl")
        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                continue
            snapshot[str(path)] = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return {}
    return snapshot


def observed_agent_session_evidence_from_path(path: Path) -> dict[str, Any]:
    """Extract a minimal, non-sensitive Codex turn-context identity record."""

    bytes_read = 0
    latest_evidence: dict[str, Any] = {}
    try:
        with path.open("rb") as stream:
            for raw_line in stream:
                bytes_read += len(raw_line)
                if bytes_read > MAX_AGENT_EVENTS_TEXT_BYTES:
                    break
                try:
                    event = json.loads(raw_line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(event, dict) or event.get("type") != "turn_context":
                    continue
                payload = event.get("payload")
                if not isinstance(payload, dict):
                    continue
                model = clean_agent_model_name(payload.get("model"))
                if not model:
                    continue
                latest_evidence = {
                    "event_type": "turn_context",
                    "timestamp": event.get("timestamp"),
                    "turn_id": payload.get("turn_id"),
                    "model": model,
                    "source_line_sha256": hashlib.sha256(raw_line).hexdigest(),
                }
    except OSError:
        return {}
    return latest_evidence


def captured_session_thread_id(events_path: Path) -> str:
    """The run's own session/thread id from the captured event stream.

    Codex emits ``thread.started`` with ``thread_id`` matching the uuid in its
    rollout filename — the strongest correlator between an invocation and its
    rollout (child/subagent sessions write their own files)."""

    try:
        with events_path.open("rb") as stream:
            for raw_line in stream:
                try:
                    event = json.loads(raw_line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(event, dict) and event.get("type") == "thread.started":
                    return str(event.get("thread_id") or "")
    except OSError:
        return ""
    return ""


def observed_agent_session_evidence_from_files(
    sessions_dir: Path,
    *,
    previous_snapshot: dict[str, tuple[int, int]] | None = None,
    max_files: int = 8,
    preferred_session_id: str = "",
) -> tuple[dict[str, Any], Path | None]:
    """Return minimal model evidence from a rollout changed by this invocation.

    Rollouts whose filename carries the run's own thread/session id rank first,
    so a child/subagent session written in the same window cannot win over the
    invocation's rollout; recency is the tiebreaker."""

    candidates: list[tuple[Path, tuple[int, int]]] = []
    try:
        for path in sessions_dir.rglob("*.jsonl"):
            try:
                stat = path.stat()
            except OSError:
                continue
            state = (stat.st_mtime_ns, stat.st_size)
            if previous_snapshot is not None and previous_snapshot.get(str(path)) == state:
                continue
            candidates.append((path, state))
        candidates.sort(
            key=lambda item: (
                1 if preferred_session_id and preferred_session_id in item[0].name else 0,
                item[1][0],
            ),
            reverse=True,
        )
    except OSError:
        return {}, None
    for path, _state in candidates[:max_files]:
        evidence = observed_agent_session_evidence_from_path(path)
        if evidence:
            return evidence, path
    return {}, None


def observed_agent_model_from_session_files(
    sessions_dir: Path,
    max_files: int = 8,
    *,
    previous_snapshot: dict[str, tuple[int, int]] | None = None,
) -> tuple[str, Path | None]:
    """Model evidence from agent-written session rollout files (newest first).

    Codex names its model only in ``CODEX_HOME/sessions`` rollout JSONL
    (``turn_context.payload.model``) — not in its ``exec --json`` event stream —
    so these files are the runtime evidence when no model was configured.
    Returns the model and the rollout file it came from.
    """

    evidence, path = observed_agent_session_evidence_from_files(
        sessions_dir,
        previous_snapshot=previous_snapshot,
        max_files=max_files,
    )
    return clean_agent_model_name(evidence.get("model")), path


def resolve_agent_model(
    configured_model: Any,
    model_source: Any,
    events_text: str,
    log_text: str = "",
) -> tuple[Any, Any]:
    configured = str(configured_model or "").strip()
    if configured and configured != UNSPECIFIED_AGENT_MODEL:
        return configured_model, model_source
    observed = observed_agent_model_from_events_text(events_text)
    if observed:
        return observed, OBSERVED_AGENT_MODEL_SOURCE
    observed = observed_agent_model_from_log_text(log_text)
    if observed:
        return observed, OBSERVED_AGENT_LOG_MODEL_SOURCE
    return configured_model, model_source


def resolve_agent_model_from_events_path(
    configured_model: Any,
    model_source: Any,
    events_path: Path,
    log_path: Path | None = None,
) -> tuple[Any, Any]:
    configured = str(configured_model or "").strip()
    if configured and configured != UNSPECIFIED_AGENT_MODEL:
        return configured_model, model_source
    observed = observed_agent_model_from_events_path(events_path)
    if observed:
        return observed, OBSERVED_AGENT_MODEL_SOURCE
    if log_path is not None:
        try:
            with log_path.open("rb") as stream:
                log_text = stream.read(MAX_AGENT_EVENTS_TEXT_BYTES).decode("utf-8", errors="replace")
        except Exception:
            log_text = ""
        observed = observed_agent_model_from_log_text(log_text)
        if observed:
            return observed, OBSERVED_AGENT_LOG_MODEL_SOURCE
    return configured_model, model_source
