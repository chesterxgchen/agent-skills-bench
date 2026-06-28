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

import json
import re
from pathlib import Path
from typing import Any

UNSPECIFIED_AGENT_MODEL = "unspecified_default"
OBSERVED_AGENT_MODEL_SOURCE = "agent_events"
OBSERVED_AGENT_LOG_MODEL_SOURCE = "agent_log"
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
