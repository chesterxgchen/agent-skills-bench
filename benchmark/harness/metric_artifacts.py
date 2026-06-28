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

"""Read observed metric values from captured runtime artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .quality_signals import (
    canonical_metric_name,
    is_numeric_metric_value,
    metric_names_match,
    metric_value_entries,
    reported_metric_payload,
)

METRIC_NAME_KEYS = frozenset({"name", "metric", "metric_name", "key", "tag"})
METRIC_VALUE_KEYS = frozenset({"value", "score", "scalar", "mean"})
ARTIFACT_SUFFIXES = frozenset({".json", ".jsonl"})
LOG_ARTIFACT_NAMES = frozenset({"log.txt", "log_fl.txt"})
METRIC_FILE_MARKERS = frozenset(
    {
        "eval",
        "metric",
        "metrics",
        "result",
        "results",
        "score",
        "scores",
        "summary",
        "validation",
    }
)
COPIED_WORKSPACE_ARTIFACT_KEYS = ("changed_files", "workspace_added_files", "workspace_modified_files")


def _candidate_delta_dirs(manifest: Mapping[str, Any], manifest_path: Path) -> list[Path]:
    candidates: list[Path] = []
    configured = str(manifest.get("delta_dir") or "")
    if configured:
        candidates.append(Path(configured))
    candidates.extend([manifest_path.parent / "workspace_delta", manifest_path.parent])

    result: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = path.as_posix()
        if key in seen:
            continue
        result.append(path)
        seen.add(key)
    return result


def captured_metric_artifact_paths(manifest: Mapping[str, Any], manifest_path: Path) -> list[Path]:
    """Return copied structured artifact paths from a workspace-delta manifest."""

    delta_dirs = _candidate_delta_dirs(manifest, manifest_path)
    paths: list[tuple[int, int, Path]] = []
    seen: set[Path] = set()
    for category_index, key in enumerate(
        ("runtime_artifacts", "changed_files", "workspace_added_files", "workspace_modified_files")
    ):
        items = manifest.get(key)
        if not isinstance(items, list):
            continue
        for item_index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            artifact_text = str(item.get("artifact_path") or "")
            if not artifact_text:
                continue
            candidate = next(
                (delta_dir / artifact_text for delta_dir in delta_dirs if (delta_dir / artifact_text).is_file()),
                None,
            )
            if (
                candidate is not None
                and candidate.suffix.lower() in ARTIFACT_SUFFIXES
                and is_metric_artifact_path(candidate)
                and candidate not in seen
            ):
                paths.append((category_index, item_index, candidate))
                seen.add(candidate)
    return [
        path
        for _category_index, _item_index, path in sorted(
            paths, key=lambda item: (item[0], metric_artifact_rank(item[2]), item[1], item[2].as_posix())
        )
    ]


def captured_metric_log_paths(manifest: Mapping[str, Any], manifest_path: Path) -> list[Path]:
    """Return copied runtime log paths that can contain framework metric output."""

    delta_dirs = _candidate_delta_dirs(manifest, manifest_path)
    paths: list[tuple[int, Path]] = []
    seen: set[Path] = set()
    items = manifest.get("runtime_artifacts")
    if not isinstance(items, list):
        return []
    for item_index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        artifact_text = str(item.get("artifact_path") or "")
        if not artifact_text:
            continue
        name = Path(artifact_text).name.lower()
        if name not in LOG_ARTIFACT_NAMES:
            continue
        candidate = next(
            (delta_dir / artifact_text for delta_dir in delta_dirs if (delta_dir / artifact_text).is_file()),
            None,
        )
        if candidate is None or candidate in seen:
            continue
        paths.append((item_index, candidate))
        seen.add(candidate)
    return [path for _item_index, path in sorted(paths, key=lambda item: (item[0], item[1].as_posix()))]


def is_metric_artifact_path(path: Path) -> bool:
    text = path.as_posix().lower()
    name = path.name.lower()
    if "/config/" in text or name.startswith("config_") or name.endswith("_config.json"):
        return False
    if "/metrics/" in text or "/metric/" in text:
        return True
    return any(marker in name for marker in METRIC_FILE_MARKERS)


def _metric_source_paths(metric: Mapping[str, Any]) -> list[str]:
    source_path = str(metric.get("source_path") or "")
    return [part.strip().replace("\\", "/") for part in source_path.split(";") if part.strip()]


def metric_source_path_is_runtime_artifact(source_path: str) -> bool:
    path = str(source_path or "").replace("\\", "/").strip()
    if not path:
        return False
    path_with_root = "/" + path.lstrip("/")
    if any(
        f"/workspace_delta/{key}/" in path_with_root or path_with_root.startswith(f"/{key}/")
        for key in COPIED_WORKSPACE_ARTIFACT_KEYS
    ):
        return False
    return "workspace_delta/runtime_artifacts/" in path or "/runtime_artifacts/" in path_with_root


def metric_is_runtime_result_artifact(metric: Mapping[str, Any]) -> bool:
    source = metric.get("source")
    if source not in {"metrics_artifact", "runtime_log_artifact"} or not metric.get("reported_values"):
        return False
    source_paths = _metric_source_paths(metric)
    if not source_paths:
        return False
    if source == "metrics_artifact" and any(
        path.endswith("/round_metrics.jsonl") or path == "round_metrics.jsonl" for path in source_paths
    ):
        return False
    return all(metric_source_path_is_runtime_artifact(path) for path in source_paths)


def metric_artifact_rank(path: Path) -> int:
    name = path.name.lower()
    if name == "metrics_summary.json":
        rank = 0
    elif name.endswith("_summary.json") or name == "summary.json":
        rank = 1
    elif name == "round_metrics.jsonl":
        rank = 2
    elif "metric" in name:
        rank = 3
    elif "result" in name or "score" in name:
        rank = 4
    else:
        rank = 5
    text = path.as_posix().lower()
    if "probe" in text:
        rank += 10
    return rank


def load_artifact_payloads(path: Path) -> Iterable[Any]:
    if path.suffix.lower() == ".jsonl":
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return []
        payloads = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                payloads.append(json.loads(line))
            except Exception:
                continue
        return payloads
    try:
        return [json.loads(path.read_text(encoding="utf-8", errors="replace"))]
    except Exception:
        return []


def _json_path(parts: tuple[str, ...]) -> str:
    return ".".join(part for part in parts if part)


def _artifact_label(path_parts: tuple[str, ...], source_path: Path) -> str:
    context = _json_path(path_parts)
    if any(token in context.lower() for token in ("site", "client")):
        return f"artifact site validation metric {context or source_path.name}"
    return f"artifact aggregated validation metric {context or source_path.name}"


def metric_entries_from_payload(
    payload: Any,
    expected_metric: str,
    source_path: Path,
    path_parts: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    expected = canonical_metric_name(expected_metric)
    entries: list[dict[str, Any]] = []
    if isinstance(payload, Mapping):
        name_values = [
            value
            for key, value in payload.items()
            if str(key).lower() in METRIC_NAME_KEYS and metric_names_match(value, expected_metric)
        ]
        if name_values:
            for key, value in payload.items():
                if str(key).lower() in METRIC_VALUE_KEYS and is_numeric_metric_value(value):
                    entries.append({"label": _artifact_label(path_parts + (str(key),), source_path), "value": value})
        for key, value in payload.items():
            next_parts = path_parts + (str(key),)
            if metric_names_match(key, expected_metric) and is_numeric_metric_value(value):
                entries.append({"label": _artifact_label(next_parts, source_path), "value": value})
            elif isinstance(value, str):
                for entry in metric_value_entries(expected, value):
                    labeled = dict(entry)
                    labeled.setdefault("label", _artifact_label(next_parts, source_path))
                    entries.append(labeled)
            else:
                entries.extend(metric_entries_from_payload(value, expected, source_path, next_parts))
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            entries.extend(metric_entries_from_payload(item, expected, source_path, path_parts + (f"[{index}]",)))
    elif isinstance(payload, str):
        entries.extend(metric_value_entries(expected, payload))
    return entries


def _named_numeric_metric_names(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    names = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        name = canonical_metric_name(item.get("name"))
        if name and is_numeric_metric_value(item.get("value")):
            names.append(name)
    return names


def _single_metric_name(items: Any) -> str:
    names = _named_numeric_metric_names(items)
    unique_names = []
    seen = set()
    for name in names:
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique_names.append(name)
    return unique_names[0] if len(unique_names) == 1 else ""


def inferred_metric_name_from_payload(payload: Any) -> str:
    """Infer the primary metric only from explicit structured artifact metadata."""

    if not isinstance(payload, Mapping):
        return ""
    key_metric = payload.get("key_metric")
    if isinstance(key_metric, Mapping):
        name = canonical_metric_name(key_metric.get("name"))
        if name:
            return name
    for key in ("best_metrics", "final_aggregated_metrics"):
        name = _single_metric_name(payload.get(key))
        if name:
            return name
    return ""


def validation_metric_from_artifact(path: Path, expected_metric: str | None) -> dict[str, Any]:
    metric_name = canonical_metric_name(expected_metric)
    payloads = list(load_artifact_payloads(path))
    if not metric_name:
        for payload in payloads:
            metric_name = inferred_metric_name_from_payload(payload)
            if metric_name:
                break
    if not metric_name:
        return {}
    entries: list[dict[str, Any]] = []
    for payload in payloads:
        entries.extend(metric_entries_from_payload(payload, metric_name, path))
    if not entries:
        return {}
    result = reported_metric_payload(metric_name, entries)
    if not result.get("reported_values"):
        return {}
    result["source"] = "metrics_artifact"
    result["source_path"] = str(path)
    return result


def _runtime_workspace_key(path: Path) -> str:
    parts = path.parts
    for index, part in enumerate(parts):
        if re.fullmatch(r"site[-_]\d+", part, flags=re.IGNORECASE):
            return "/".join(parts[:index])
    return "/".join(parts[:-1])


def _runtime_site_label(path: Path) -> str:
    for part in path.parts:
        if re.fullmatch(r"site[-_]\d+", part, flags=re.IGNORECASE):
            return part
    return ""


def _runtime_workspace_is_probe(workspace_key: str) -> bool:
    return bool(re.search(r"\b(?:probe|smoke)\b", workspace_key, flags=re.IGNORECASE))


def _metric_log_values(path: Path, expected_metric: str) -> list[float]:
    values: list[float] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return values
    for line in lines:
        for entry in metric_value_entries(expected_metric, line):
            value = entry.get("value")
            if is_numeric_metric_value(value):
                values.append(float(value))
    return values


def validation_metric_from_runtime_logs(paths: Sequence[Path], expected_metric: str | None) -> dict[str, Any]:
    metric_name = canonical_metric_name(expected_metric)
    if not metric_name:
        return {}
    groups: dict[str, dict[str, Any]] = {}
    for path in paths:
        site = _runtime_site_label(path)
        if not site:
            continue
        values = _metric_log_values(path, metric_name)
        if not values:
            continue
        group = groups.setdefault(
            _runtime_workspace_key(path),
            {"site_values": {}, "source_paths": [], "value_count": 0},
        )
        group["site_values"][site] = values[-1]
        group["source_paths"].append(str(path))
        group["value_count"] += len(values)
    if not groups:
        return {}
    _key, selected = max(
        groups.items(),
        key=lambda item: (
            not _runtime_workspace_is_probe(item[0]),
            len(item[1]["site_values"]),
            item[1]["value_count"],
        ),
    )
    site_values = selected["site_values"]
    if not site_values:
        return {}
    aggregate = sum(site_values.values()) / len(site_values)
    entries = [
        {
            "label": f"artifact site validation metric {site} final log {metric_name}",
            "value": value,
        }
        for site, value in sorted(site_values.items())
    ]
    entries.append(
        {
            "label": f"artifact aggregated validation metric final log mean {metric_name}",
            "value": aggregate,
        }
    )
    result = reported_metric_payload(metric_name, entries)
    if not result.get("reported_values"):
        return {}
    result["source"] = "runtime_log_artifact"
    result["source_path"] = "; ".join(selected["source_paths"])
    return result


def validation_metric_from_workspace_delta_manifest(
    manifest: Mapping[str, Any], manifest_path: Path, expected_metric: str | None = None
) -> dict[str, Any]:
    for path in captured_metric_artifact_paths(manifest, manifest_path):
        metric = validation_metric_from_artifact(path, expected_metric)
        if metric:
            return metric
    metric = validation_metric_from_runtime_logs(captured_metric_log_paths(manifest, manifest_path), expected_metric)
    if metric:
        return metric
    return {}


def observed_metric_payloads_from_workspace_delta_manifest(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    metric_names: Sequence[str],
    *,
    skip_names: Iterable[str] = (),
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    seen = {canonical_metric_name(name) for name in skip_names if canonical_metric_name(name)}
    for name in metric_names:
        canonical_name = canonical_metric_name(name)
        if not canonical_name or canonical_name in seen:
            continue
        for path in captured_metric_artifact_paths(manifest, manifest_path):
            metric = validation_metric_from_artifact(path, canonical_name)
            if metric:
                payloads.append(metric)
                seen.add(canonical_name)
                break
    return payloads
