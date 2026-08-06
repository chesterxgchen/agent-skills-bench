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

"""Parser registries for YAML-driven benchmark agent adapters."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

MAX_ACTIVITY_COMMANDS = 200
CLAUDE_SHELL_TOOL_NAMES = {"bash"}


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def parse_usage_and_activity_data(events_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from ..events import parse_usage_and_activity_data as runtime_parse_usage_and_activity_data

    return runtime_parse_usage_and_activity_data(events_path)


def event_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def normalize_jsonl_event(raw_line: str) -> dict[str, Any] | None:
    stripped = raw_line.rstrip("\n")
    if not stripped:
        return None
    timestamp = event_timestamp()
    try:
        event = json.loads(stripped)
    except json.JSONDecodeError:
        event = {"type": "harness.unparsed_event", "raw": stripped}
    if isinstance(event, dict):
        event.setdefault("timestamp", timestamp)
        event["harness_timestamp"] = timestamp
        return event
    return {
        "type": "harness.non_object_event",
        "timestamp": timestamp,
        "harness_timestamp": timestamp,
        "value": event,
    }


def normalize_claude_stream_event(raw_line: str) -> dict[str, Any] | None:
    event = normalize_jsonl_event(raw_line)
    if event is None:
        return None
    if event.get("type", "").startswith("harness."):
        event.setdefault("event_type", event.get("type"))
        return event

    event_type = str(event.get("type") or "unknown")
    subtype = event.get("subtype")
    event["event_type"] = f"{event_type}.{subtype}" if subtype else event_type
    if event_type == "result" and isinstance(event.get("result"), str):
        event["final_message"] = event["result"]
    # Neutral events expose one primary tool per raw event. Keep the first
    # shell-tool command as the stable activity signal for report aggregation.
    for tool_use in claude_tool_uses(event):
        event.setdefault("tool_kind", tool_use.get("name"))
        command = claude_tool_command(tool_use)
        if command:
            event.setdefault("command_text", command)
            break
    return event


def claude_message_content(event: dict[str, Any]) -> list[Any]:
    message = event.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    return content if isinstance(content, list) else []


def claude_tool_uses(event: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in claude_message_content(event) if isinstance(item, dict) and item.get("type") == "tool_use"]


def claude_tool_command(tool_use: dict[str, Any]) -> str | None:
    tool_name = str(tool_use.get("name") or "").lower()
    if tool_name not in CLAUDE_SHELL_TOOL_NAMES:
        return None
    tool_input = tool_use.get("input")
    if not isinstance(tool_input, dict):
        return None
    for key in ("command", "cmd", "shell_command"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def numeric_token_field(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def claude_usage_total(usage: dict[str, Any]) -> float:
    # Claude cache write/read token fields are included in the headline total
    # because they contribute to run cost; the neutral cache_tokens field also
    # exposes them separately for report layers that split cache cost.
    return sum(
        numeric_token_field(usage, key)
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
    )


def claude_usage_has_tokens(usage: dict[str, Any]) -> bool:
    return any(
        numeric_token_field(usage, key) > 0
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
    )


CLAUDE_MODEL_USAGE_FIELDS = {
    "input_tokens": "inputTokens",
    "output_tokens": "outputTokens",
    "cache_creation_input_tokens": "cacheCreationInputTokens",
    "cache_read_input_tokens": "cacheReadInputTokens",
}


def claude_cumulative_model_usage(
    event: dict[str, Any], primary_model: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Return the cumulative usage for Claude's primary model.

    Claude ``--continue`` emits one incremental ``usage`` object per result,
    while the last result's camel-case ``modelUsage`` map remains cumulative
    for the whole session.  The map may also contain a small auxiliary model,
    so select the model announced by the latest ``system.init`` event instead
    of summing every entry.
    """

    model_usage = event.get("modelUsage")
    if not isinstance(model_usage, dict) or not model_usage:
        return None, None
    selected_name = None
    if primary_model and isinstance(model_usage.get(primary_model), dict):
        selected_name = primary_model
    elif primary_model:
        normalized_primary = re.sub(r"\[[^]]+\]$", "", primary_model)
        selected_name = next(
            (
                str(name)
                for name, value in model_usage.items()
                if isinstance(value, dict) and re.sub(r"\[[^]]+\]$", "", str(name)) == normalized_primary
            ),
            None,
        )
    if selected_name is None:
        candidates = [str(name) for name, value in model_usage.items() if isinstance(value, dict)]
        if len(candidates) == 1:
            selected_name = candidates[0]
    selected = model_usage.get(selected_name) if selected_name else None
    if not isinstance(selected, dict):
        return None, None
    return (
        {target: numeric_token_field(selected, source) for target, source in CLAUDE_MODEL_USAGE_FIELDS.items()},
        selected_name,
    )


def claude_usage_objects(event: dict[str, Any]) -> list[dict[str, Any]]:
    usage_objects = []
    usage = event.get("usage")
    if isinstance(usage, dict):
        usage_objects.append(usage)
    message = event.get("message")
    if isinstance(message, dict) and isinstance(message.get("usage"), dict):
        usage_objects.append(message["usage"])
    return usage_objects


def claude_request_accounting(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Build request-level accounting from Claude's repeated stream events.

    A single model response can be emitted as several ``assistant`` events
    sharing one ``request_id``.  Count that ID once, and retain tool names from
    every content block belonging to the request.  Claude does not currently
    emit a cache-miss reason, so ``tools_changed`` is deliberately an inferred
    signal: a non-initial request rebuilt cache from zero immediately after a
    request invoked ``ToolSearch`` and changed the available tool schemas.
    """

    request_order: list[str] = []
    request_usage: dict[str, dict[str, Any]] = {}
    request_tools: dict[str, set[str]] = {}
    for event in events:
        if event.get("type") != "assistant":
            continue
        request_id = event.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            continue
        if request_id not in request_tools:
            request_order.append(request_id)
            request_tools[request_id] = set()
        for usage in claude_usage_objects(event):
            request_usage[request_id] = usage
        request_tools[request_id].update(
            str(tool_use.get("name") or "") for tool_use in claude_tool_uses(event) if str(tool_use.get("name") or "")
        )

    tools_changed_misses = 0
    tools_changed_cache_creation_tokens = 0.0
    for index, request_id in enumerate(request_order[1:], start=1):
        usage = request_usage.get(request_id)
        if not isinstance(usage, dict):
            continue
        previous_tools = request_tools.get(request_order[index - 1], set())
        cache_read = numeric_token_field(usage, "cache_read_input_tokens")
        cache_creation = numeric_token_field(usage, "cache_creation_input_tokens")
        if "ToolSearch" in previous_tools and cache_read == 0 and cache_creation > 0:
            tools_changed_misses += 1
            tools_changed_cache_creation_tokens += cache_creation

    return {
        "model_request_count": len(request_order) or None,
        "model_request_count_source": ("unique Claude assistant request_id values" if request_order else None),
        "tools_changed_cache_miss_count": tools_changed_misses if request_order else None,
        "tools_changed_cache_creation_input_tokens": (tools_changed_cache_creation_tokens if request_order else None),
        "tools_changed_cache_miss_detection": (
            "inferred from a non-initial zero-cache-read/cache-creation request immediately after ToolSearch"
            if request_order
            else None
        ),
    }


def iter_json_events(events_path: Path) -> tuple[list[dict[str, Any]], int]:
    events = []
    decode_errors = 0
    if not events_path.exists():
        return events, decode_errors
    with events_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                decode_errors += 1
                continue
            if isinstance(event, dict):
                events.append(event)
    return events, decode_errors


def parse_claude_stream_usage(events_path: Path) -> dict[str, Any]:
    events, decode_errors = iter_json_events(events_path)
    request_accounting = claude_request_accounting(events)
    result_usage: dict[str, Any] | None = None
    result_usage_count = 0
    result_usage_sum = {
        "input_tokens": 0.0,
        "output_tokens": 0.0,
        "cache_creation_input_tokens": 0.0,
        "cache_read_input_tokens": 0.0,
    }
    summed = {
        "input_tokens": 0.0,
        "output_tokens": 0.0,
        "cache_creation_input_tokens": 0.0,
        "cache_read_input_tokens": 0.0,
    }
    usage_objects_seen = 0
    summed_request_ids: set[str] = set()
    total_cost_usd = None
    primary_model = ""
    cumulative_model_usage: dict[str, Any] | None = None
    cumulative_model_name: str | None = None
    provider_api_duration_ms = None
    for event in events:
        if event.get("type") == "system" and event.get("subtype") == "init":
            model = event.get("model")
            if isinstance(model, str) and model:
                primary_model = model
        if event.get("type") == "result" and isinstance(event.get("usage"), dict):
            usage_objects_seen += 1
            result_usage_count += 1
            result_usage = event["usage"]
            for key in result_usage_sum:
                result_usage_sum[key] += numeric_token_field(result_usage, key)
            candidate_usage, candidate_name = claude_cumulative_model_usage(event, primary_model)
            if candidate_usage is not None:
                cumulative_model_usage = candidate_usage
                cumulative_model_name = candidate_name
            if isinstance(event.get("message"), dict) and isinstance(event["message"].get("usage"), dict):
                usage_objects_seen += 1
            if isinstance(event.get("total_cost_usd"), (int, float)):
                total_cost_usd = event.get("total_cost_usd")
            if isinstance(event.get("duration_api_ms"), (int, float)):
                provider_api_duration_ms = event.get("duration_api_ms")
            continue
        if event.get("type") == "result" and isinstance(event.get("total_cost_usd"), (int, float)):
            total_cost_usd = event.get("total_cost_usd")
        request_id = event.get("request_id")
        deduplicate_request_usage = (
            event.get("type") == "assistant" and isinstance(request_id, str) and bool(request_id)
        )
        request_usage_already_summed = deduplicate_request_usage and request_id in summed_request_ids
        for usage in claude_usage_objects(event):
            usage_objects_seen += 1
            if request_usage_already_summed:
                continue
            for key in summed:
                summed[key] += numeric_token_field(usage, key)
        if deduplicate_request_usage:
            summed_request_ids.add(request_id)

    usage_source = ""
    is_cumulative = False
    if cumulative_model_usage is not None and claude_usage_has_tokens(cumulative_model_usage):
        selected = cumulative_model_usage
        usage_source = "final cumulative modelUsage for the primary Claude model"
        is_cumulative = True
    elif result_usage_count > 1 and claude_usage_has_tokens(result_usage_sum):
        selected = result_usage_sum
        usage_source = "sum of incremental Claude result usage objects"
        is_cumulative = True
    elif result_usage is not None and claude_usage_has_tokens(result_usage):
        selected = result_usage
        usage_source = "Claude result usage"
    else:
        selected = summed
        usage_source = "sum of deduplicated Claude message usage objects"
    total_tokens = claude_usage_total(selected)
    parser_warnings = []
    if usage_objects_seen == 0:
        parser_warnings.append("No Claude usage objects were found in the stream-json events.")
        total_tokens = None
    elif result_usage is None:
        parser_warnings.append(
            "No Claude result usage object was found; token fields are summed from message usage objects."
        )
    elif not claude_usage_has_tokens(result_usage) and claude_usage_has_tokens(summed):
        parser_warnings.append(
            "Claude result usage object had no nonzero token fields; token fields are summed from message usage objects."
        )
    elif result_usage_count > 1 and cumulative_model_usage is None:
        parser_warnings.append(
            "Multiple Claude result usage objects were found without primary-model modelUsage; "
            "incremental result usage objects were summed."
        )
    if cumulative_model_usage is not None and result_usage_count > 1:
        mismatched_fields = [
            key
            for key in result_usage_sum
            if numeric_token_field(cumulative_model_usage, key) != numeric_token_field(result_usage_sum, key)
        ]
        if mismatched_fields:
            parser_warnings.append(
                "Claude cumulative modelUsage differs from summed continuation result usage for: "
                + ", ".join(mismatched_fields)
                + "."
            )
    cache_creation_tokens = selected.get("cache_creation_input_tokens")
    cache_read_tokens = selected.get("cache_read_input_tokens")
    cache_tokens = numeric_token_field(selected, "cache_creation_input_tokens") + numeric_token_field(
        selected, "cache_read_input_tokens"
    )
    model_request_count = request_accounting.get("model_request_count")
    tokens_per_model_request = (
        total_tokens / model_request_count
        if total_tokens is not None and isinstance(model_request_count, int) and model_request_count > 0
        else None
    )
    return {
        "total_tokens": total_tokens,
        "input_tokens": selected.get("input_tokens"),
        "output_tokens": selected.get("output_tokens"),
        "cache_tokens": cache_tokens,
        "cost": total_cost_usd,
        "parser_warnings": parser_warnings,
        "cache_creation_input_tokens": selected.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": selected.get("cache_read_input_tokens"),
        "raw_cache_creation_input_tokens": cache_creation_tokens,
        "raw_cache_read_input_tokens": cache_read_tokens,
        "total_cost_usd": total_cost_usd,
        "json_decode_errors": decode_errors,
        "usage_objects_seen": usage_objects_seen,
        "result_usage_objects_seen": result_usage_count,
        "model_usage_primary_model": cumulative_model_name,
        "usage_source": usage_source,
        "is_cumulative": is_cumulative,
        "provider_api_duration_ms": provider_api_duration_ms,
        "provider_api_seconds": (
            provider_api_duration_ms / 1000.0 if isinstance(provider_api_duration_ms, (int, float)) else None
        ),
        "provider_api_duration_source": (
            "final Claude result duration_api_ms (cumulative across continuations)"
            if provider_api_duration_ms is not None
            else None
        ),
        "token_parser": (
            "Claude final cumulative primary-model modelUsage; fallback sums incremental result usage, "
            "then deduplicated message usage"
        ),
        **request_accounting,
        "tokens_per_model_request": tokens_per_model_request,
    }


ACTIVITY_HINTS = {
    "skill_md": ["skill.md"],
    "skill_references": ["/references/", " references/"],
    "skill_metadata": ["evals/evals.json", "/evals/"],
    "benchmark_md": ["benchmark.md"],
    "agent_inspect": ["agent inspect"],
    "agent_skill_setup": ["skills install", "skills list"],
    "py_compile": ["py_compile"],
    "python_job_py": ["python job.py", "python3 job.py"],
    "simulation": ["simulator", "simulate", "--workspace-root"],
    "shell_find": ["find "],
    "shell_search": ["rg ", "grep "],
    "shell_cat_or_sed": ["cat ", "sed ", "nl -ba"],
}


def parse_claude_stream_activity(events_path: Path) -> dict[str, Any]:
    events, decode_errors = iter_json_events(events_path)
    event_types: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    command_prefixes: Counter[str] = Counter()
    hint_counts: Counter[str] = Counter()
    commands: list[str] = []
    unique_commands_seen: set[str] = set()
    first_event_dt = None
    first_event_timestamp = None
    last_event_dt = None
    last_event_timestamp = None
    previous_event_dt = None
    max_inter_event_gap_seconds = None
    for event in events:
        event_type = str(event.get("event_type") or event.get("type") or "unknown")
        event_types[event_type] += 1
        timestamp = event.get("harness_timestamp") or event.get("timestamp")
        event_dt = parse_timestamp(timestamp)
        if event_dt is not None:
            if first_event_dt is None:
                first_event_dt = event_dt
                first_event_timestamp = timestamp
            if previous_event_dt is not None:
                gap = (event_dt - previous_event_dt).total_seconds()
                if gap >= 0 and (max_inter_event_gap_seconds is None or gap > max_inter_event_gap_seconds):
                    max_inter_event_gap_seconds = gap
            previous_event_dt = event_dt
            last_event_dt = event_dt
            last_event_timestamp = timestamp
        command = event.get("command_text")
        tool_kind = event.get("tool_kind")
        if isinstance(tool_kind, str) and tool_kind:
            tool_counts[tool_kind] += 1
        if isinstance(command, str) and command.strip():
            command = command.strip()
            if len(commands) < MAX_ACTIVITY_COMMANDS:
                commands.append(command)
            unique_commands_seen.add(command)
            command_prefixes[command.split()[0]] += 1
            lowered = command.lower()
            for name, needles in ACTIVITY_HINTS.items():
                if any(needle in lowered for needle in needles):
                    hint_counts[name] += 1

    # Augment shell-pattern hints with structured tool call counts so that
    # agents using tool APIs (e.g. Claude Read/Skill/Agent tools) are reflected
    # in the same Activity Insights rows as their shell equivalents.
    hint_counts["shell_cat_or_sed"] += tool_counts.get("Read", 0)
    hint_counts["skill_references"] += tool_counts.get("Skill", 0)
    hint_counts["agent_inspect"] += tool_counts.get("Agent", 0)

    return {
        "event_count": len(events),
        "json_decode_errors": decode_errors,
        "timestamp_field": "harness_timestamp",
        "first_event_timestamp": first_event_timestamp,
        "last_event_timestamp": last_event_timestamp,
        "event_span_seconds": (
            round((last_event_dt - first_event_dt).total_seconds(), 3)
            if first_event_dt is not None and last_event_dt is not None
            else None
        ),
        "max_inter_event_gap_seconds": (
            round(max_inter_event_gap_seconds, 3) if max_inter_event_gap_seconds is not None else None
        ),
        "event_types": dict(event_types.most_common()),
        "tool_counts": dict(tool_counts.most_common()),
        "hint_counts": dict(hint_counts.most_common()),
        "command_count": sum(command_prefixes.values()),
        "unique_command_count": len(unique_commands_seen),
        "command_prefix_counts": dict(command_prefixes.most_common()),
        "commands": commands,
        "commands_truncated": sum(command_prefixes.values()) > len(commands),
        "max_recorded_commands": MAX_ACTIVITY_COMMANDS,
    }


@lru_cache(maxsize=32)
def parse_cached_usage_and_activity(events_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return parse_usage_and_activity_data(Path(events_path))


def cached_usage(events_path: Path) -> dict[str, Any]:
    usage, _activity = parse_cached_usage_and_activity(str(events_path))
    return dict(usage)


def cached_activity(events_path: Path) -> dict[str, Any]:
    _usage, activity = parse_cached_usage_and_activity(str(events_path))
    return dict(activity)


EVENT_PARSERS = {
    "claude_stream_json": normalize_claude_stream_event,
    "codex_jsonl": normalize_jsonl_event,
    "generic_jsonl": normalize_jsonl_event,
}

USAGE_PARSERS = {
    "claude_stream_usage": parse_claude_stream_usage,
    "codex_cumulative_usage": cached_usage,
    "generic_cli_usage": cached_usage,
}

ACTIVITY_PARSERS = {
    "claude_stream_activity": parse_claude_stream_activity,
    "codex_jsonl_activity": cached_activity,
    "generic_jsonl_activity": cached_activity,
}

FINAL_MESSAGE_SOURCE_TYPES = {"file", "structured_event", "stdout_tail", "not_available"}
VALID_FINAL_MESSAGE_PARSER_IDS = {
    "generic_stdout_last_message",
    "generic_structured_event_message",
}


def validate_event_parser(parser_id: str) -> None:
    if parser_id not in EVENT_PARSERS:
        raise ValueError(f"Unknown agent event parser: {parser_id}")


def validate_usage_parser(parser_id: str) -> None:
    if parser_id not in USAGE_PARSERS:
        raise ValueError(f"Unknown agent usage parser: {parser_id}")


def validate_activity_parser(parser_id: str) -> None:
    if parser_id not in ACTIVITY_PARSERS:
        raise ValueError(f"Unknown agent activity parser: {parser_id}")


def validate_final_message_config(source_type: str, parser_id: str | None = None) -> None:
    if source_type not in FINAL_MESSAGE_SOURCE_TYPES:
        raise ValueError(
            f"Unknown final message source_type: {source_type}. "
            f"Valid source types: {', '.join(sorted(FINAL_MESSAGE_SOURCE_TYPES))}"
        )
    if parser_id and parser_id not in VALID_FINAL_MESSAGE_PARSER_IDS:
        raise ValueError(f"Unknown final message parser: {parser_id}")


def normalize_event_with_parser(raw_line: str, parser_id: str) -> dict[str, Any] | None:
    validate_event_parser(parser_id)
    parser = EVENT_PARSERS[parser_id]
    return parser(raw_line)


def parse_usage_from_events(events_path: Path, usage_config: Any) -> dict[str, Any]:
    parser_id = getattr(usage_config, "parser", None) or "generic_cli_usage"
    validate_usage_parser(parser_id)
    parser = USAGE_PARSERS[parser_id]
    usage = parser(events_path)
    usage.setdefault("parser_id", parser_id)
    return usage


def parse_activity_from_events(events_path: Path, activity_config: Any) -> dict[str, Any]:
    parser_id = getattr(activity_config, "parser", None) or "generic_jsonl_activity"
    validate_activity_parser(parser_id)
    parser = ACTIVITY_PARSERS[parser_id]
    activity = parser(events_path)
    activity.setdefault("parser_id", parser_id)
    return activity
