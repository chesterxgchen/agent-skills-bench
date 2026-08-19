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

"""Exit and failure classifiers for benchmark agent adapters."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

CLASSIFICATION_EVIDENCE_LIMIT = 4000


def normalized_signal_name(value: Any) -> str:
    """Return the comparison form used for structured diagnostic names."""

    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def structured_cache_miss_reasons(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[str, str]]:
    """Yield exact ``(json_path, reason)`` pairs from cache-miss diagnostics."""

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (*path, str(key))
            if normalized_signal_name(key) == "cache_miss_reason":
                if isinstance(child, dict):
                    for field_name in ("type", "reason", "code"):
                        reason = child.get(field_name)
                        if isinstance(reason, str) and reason.strip():
                            yield ".".join((*child_path, field_name)), reason.strip()
                            break
                else:
                    reason = child
                    if isinstance(reason, str) and reason.strip():
                        yield ".".join(child_path), reason.strip()
            yield from structured_cache_miss_reasons(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from structured_cache_miss_reasons(child, (*path, str(index)))


def causal_event_summary(
    path: Path,
    line_number: int,
    event: dict[str, Any],
    json_path: str,
    reason: str,
) -> dict[str, Any]:
    source = path.name
    evidence = f"{json_path}={reason}"
    summary = {
        "source": source,
        "line": line_number,
        "reference": f"{source}:{line_number}",
        "json_path": json_path,
        "reason": reason,
        "evidence": evidence,
    }
    for source_key, target_key in (
        ("event_type", "event_type"),
        ("type", "event_type"),
        ("timestamp", "timestamp"),
        ("harness_timestamp", "harness_timestamp"),
        ("request_id", "request_id"),
        ("uuid", "event_id"),
    ):
        value = event.get(source_key)
        if value not in (None, "") and target_key not in summary:
            summary[target_key] = value
    return summary


def terminal_result_failure(event: dict[str, Any]) -> bool | None:
    """Return a terminal result's failure state, or ``None`` for non-results."""

    raw_type = normalized_signal_name(event.get("type"))
    event_type = normalized_signal_name(event.get("event_type"))
    if raw_type != "result" and not event_type.startswith("result_"):
        return None
    is_error = event.get("is_error")
    if isinstance(is_error, bool):
        return is_error
    subtype = normalized_signal_name(event.get("subtype"))
    if subtype:
        return subtype != "success"
    if event_type.startswith("result_"):
        return event_type != "result_success"
    return None


def final_structured_cache_miss(
    paths: Iterable[Path],
    configured_reasons: set[str],
) -> tuple[str, dict[str, Any]] | None:
    """Return a configured cache miss only from the final failing result."""

    terminal_event: tuple[Path, int, dict[str, Any], bool] | None = None
    for path in paths:
        try:
            stream = path.open("r", encoding="utf-8", errors="replace")
        except OSError:
            continue
        with stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                failed = terminal_result_failure(event)
                if failed is not None:
                    terminal_event = (path, line_number, event, failed)
    if terminal_event is None or not terminal_event[3]:
        return None
    path, line_number, event, _failed = terminal_event
    for json_path, reason in structured_cache_miss_reasons(event):
        normalized_reason = normalized_signal_name(reason)
        if normalized_reason in configured_reasons:
            return normalized_reason, causal_event_summary(path, line_number, event, json_path, reason)
    return None


def stderr_excerpt(stderr_path: Path) -> str:
    stderr_text = ""
    try:
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError:
        pass
    return stderr_text


def evidence_excerpt(paths: Iterable[Path]) -> str:
    evidence_parts = []
    remaining = CLASSIFICATION_EVIDENCE_LIMIT
    for path in paths:
        if remaining <= 0:
            break
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        text = text.strip()
        if not text:
            continue
        if len(text) > remaining:
            text = text[:remaining]
        evidence_parts.append(text)
        remaining -= len(text)
    return "\n".join(evidence_parts)[:CLASSIFICATION_EVIDENCE_LIMIT]


def generic_cli_exit(
    exit_code: int,
    stderr_path: Path,
    classifier_id: str = "generic_cli",
    evidence_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    stderr_text = stderr_excerpt(stderr_path)
    summary = {
        "classifier": classifier_id,
        "exit_code": exit_code,
        "passed": exit_code == 0,
        "failure_category": "agent_unknown_failure" if exit_code else None,
        "stderr_excerpt": stderr_text,
    }
    extra_evidence = evidence_excerpt(evidence_paths)
    if extra_evidence:
        summary["classification_excerpt"] = stderr_text
        if stderr_text:
            summary["classification_excerpt"] += "\n"
        summary["classification_excerpt"] += extra_evidence
        summary["classification_excerpt"] = summary["classification_excerpt"][:CLASSIFICATION_EVIDENCE_LIMIT]
    return summary


EXIT_CLASSIFIERS = {"generic_cli", "stderr_patterns"}


def validate_exit_classifier(classifier_id: str) -> None:
    if classifier_id not in EXIT_CLASSIFIERS:
        raise ValueError(f"Unknown agent exit classifier: {classifier_id}")


def as_string_list(value: Any, field_path: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{field_path} must be a list of non-empty strings")
    return [str(item).lower() for item in value]


def as_exit_codes(value: Any, field_path: str) -> set[int]:
    if value is None:
        return set()
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{field_path} must be a list of integer exit codes")
    return {int(item) for item in value}


def validate_stderr_pattern_rules(config: dict[str, Any]) -> None:
    rules = config.get("rules") or []
    if not isinstance(rules, list):
        raise ValueError("exit.rules must be a list")
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"exit.rules[{index}] must be a mapping")
        category = rule.get("category")
        if not isinstance(category, str) or not category:
            raise ValueError(f"exit.rules[{index}].category must be a non-empty string")
        any_patterns = as_string_list(rule.get("any"), f"exit.rules[{index}].any")
        all_patterns = as_string_list(rule.get("all"), f"exit.rules[{index}].all")
        cache_miss_reasons = as_string_list(
            rule.get("structured_cache_miss_reasons"),
            f"exit.rules[{index}].structured_cache_miss_reasons",
        )
        exit_codes = as_exit_codes(rule.get("exit_codes"), f"exit.rules[{index}].exit_codes")
        if not any_patterns and not all_patterns and not cache_miss_reasons and not exit_codes:
            raise ValueError(
                f"exit.rules[{index}] must define at least one of any, all, "
                "structured_cache_miss_reasons, or exit_codes"
            )


def validate_exit_config(config: dict[str, Any]) -> None:
    classifier_id = str(config.get("classifier") or "")
    validate_exit_classifier(classifier_id)
    if classifier_id == "stderr_patterns":
        validate_stderr_pattern_rules(config)


def stderr_rule_matches(rule: dict[str, Any], exit_code: int, evidence_lower: str) -> bool:
    if exit_code == 0:
        return False
    exit_codes = as_exit_codes(rule.get("exit_codes"), "exit.rules[].exit_codes")
    if exit_codes and exit_code not in exit_codes:
        return False
    all_patterns = as_string_list(rule.get("all"), "exit.rules[].all")
    if all_patterns and not all(pattern in evidence_lower for pattern in all_patterns):
        return False
    any_patterns = as_string_list(rule.get("any"), "exit.rules[].any")
    if any_patterns and not any(pattern in evidence_lower for pattern in any_patterns):
        return False
    return bool(exit_codes or all_patterns or any_patterns)


def stderr_pattern_exit(
    exit_code: int,
    stderr_path: Path,
    config: dict[str, Any],
    evidence_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    evidence_paths = tuple(evidence_paths)
    summary = generic_cli_exit(exit_code, stderr_path, classifier_id="stderr_patterns", evidence_paths=evidence_paths)
    structured_rules: dict[str, dict[str, Any]] = {}
    for rule in config.get("rules") or []:
        for reason in as_string_list(
            rule.get("structured_cache_miss_reasons"),
            "exit.rules[].structured_cache_miss_reasons",
        ):
            structured_rules.setdefault(normalized_signal_name(reason), rule)
    structured_match = final_structured_cache_miss(evidence_paths, set(structured_rules)) if exit_code else None
    if structured_match is not None:
        reason, causal_event = structured_match
        summary["failure_category"] = str(structured_rules[reason]["category"])
        summary["causal_event"] = causal_event
        summary["classification_excerpt"] = (
            f"Final causal event {causal_event['reference']}: {causal_event['evidence']}"
        )[:CLASSIFICATION_EVIDENCE_LIMIT]
        return summary
    evidence_lower = str(summary.get("classification_excerpt") or summary.get("stderr_excerpt") or "").lower()
    for rule in config.get("rules") or []:
        if rule.get("structured_cache_miss_reasons") and not any(rule.get(key) for key in ("any", "all", "exit_codes")):
            continue
        if stderr_rule_matches(rule, exit_code, evidence_lower):
            summary["failure_category"] = str(rule["category"])
            break
    return summary


def classify_exit(
    exit_code: int,
    stderr_path: Path,
    config: dict[str, Any],
    evidence_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    validate_exit_config(config)
    classifier_id = str(config.get("classifier") or "")
    if classifier_id == "generic_cli":
        return generic_cli_exit(exit_code, stderr_path, evidence_paths=evidence_paths)
    if classifier_id == "stderr_patterns":
        return stderr_pattern_exit(exit_code, stderr_path, config, evidence_paths=evidence_paths)
    raise ValueError(f"Unknown agent exit classifier: {classifier_id}")
