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

"""Rule-driven evaluation scoring over detected evidence signals.

A neutral leaf: it never imports the report engine or any SDK plugin. The
verdict rules live in standard YAML documents shipped as package data under
``benchmark/config/evaluation/`` (one per SDK), so evaluation and scoring can
run outside the reporting engine and inside an installed wheel/sdist
— the engine and any external tool apply the same rules file to the same
signal profile and get the same verdicts.

Standalone usage over a signals profile (a JSON object of signal -> evidence
string, e.g. the serialized ``conversion_quality_profile`` of a run)::

    python -m benchmark.harness.evaluation --sdk nvflare --profile profile.json

Rule semantics (per signal, first match wins):

- ``contains`` / ``contains_any`` — substring match on the lower-cased evidence
- ``trend`` — direction of the numeric series embedded in the evidence
  (``improving`` | ``not_improving`` | ``single_value``)
- ``when_context`` — extra gate: every key must equal the caller-supplied
  context value (e.g. ``target_framework``)

Empty or ``not captured`` evidence scores ``unknown``; with no matching rule
the document's ``default_verdict`` applies.
"""

from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import yaml

EVALUATION_RULES_PACKAGE = "benchmark"
EVALUATION_RULES_SCHEMA_VERSION = 1

_NUMBER_RE = re.compile(r"\b([0-9]+\.[0-9]+)\b")


def evaluation_sdk_dir(sdk: str):
    """Packaged per-SDK rules directory (package data works in wheels/sdists)."""
    return resources.files(EVALUATION_RULES_PACKAGE) / "config" / "evaluation" / sdk


def _parse_document(text: str, source: str) -> dict[str, Any]:
    document = yaml.safe_load(text) or {}
    if not isinstance(document, dict):
        raise ValueError(f"evaluation rules must be a mapping: {source}")
    return document


def _parse_manifest(text: str, source: str) -> dict[str, Any]:
    document = _parse_document(text, source)
    version = document.get("schema_version")
    if version != EVALUATION_RULES_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported evaluation rules schema_version {version!r} in {source} "
            f"(expected {EVALUATION_RULES_SCHEMA_VERSION})"
        )
    return document


def _merge_document(composed: dict[str, Any], document: Mapping[str, Any]) -> None:
    """Later documents win. Signals replace whole entries (label + rules), so each
    document in the composition chain stays independently readable."""

    signals = document.get("signals")
    if isinstance(signals, Mapping):
        composed.setdefault("signals", {}).update(signals)
    if document.get("default_verdict"):
        composed["default_verdict"] = document["default_verdict"]
    scoring = document.get("scoring")
    if isinstance(scoring, Mapping):
        composed.setdefault("scoring", {}).update(scoring)


@lru_cache(maxsize=64)
def _load_rules_cached(
    sdk: str, task: str | None, selector_items: tuple[tuple[str, str], ...], path_text: str | None
) -> Mapping[str, Any]:
    if path_text is not None:
        # Explicit rules file: a single self-contained document (no composition).
        return _parse_manifest(Path(path_text).read_text(encoding="utf-8"), path_text)
    sdk_dir = evaluation_sdk_dir(sdk)
    manifest_resource = sdk_dir / "index.yaml"
    manifest = _parse_manifest(manifest_resource.read_text(encoding="utf-8"), str(manifest_resource))
    tasks = manifest.get("tasks") if isinstance(manifest.get("tasks"), Mapping) else {}
    task_name = task or str(manifest.get("default_task") or "")
    task_entry = tasks.get(task_name)
    if not isinstance(task_entry, Mapping):
        known = sorted(str(name) for name in tasks)
        raise ValueError(f"unknown evaluation task {task_name!r} for sdk {sdk!r}; known tasks: {known}")
    composed: dict[str, Any] = {
        "schema_version": manifest.get("schema_version"),
        "sdk": manifest.get("sdk") or sdk,
        "task": task_name,
        "signals": {},
        "scoring": {},
    }
    _merge_document(composed, manifest)
    common_ref = task_entry.get("common")
    if common_ref:
        common_resource = sdk_dir / str(common_ref)
        _merge_document(composed, _parse_document(common_resource.read_text(encoding="utf-8"), str(common_resource)))
    # Overlay dimensions are task-defined (framework, algorithm, deployment, ...):
    # applied in manifest declaration order; a selector for an unregistered
    # dimension or value simply applies no overlay.
    selectors = dict(selector_items)
    overlays = task_entry.get("overlays") if isinstance(task_entry.get("overlays"), Mapping) else {}
    applied: dict[str, str] = {}
    for dimension, values in overlays.items():
        if not isinstance(values, Mapping):
            continue
        selected = selectors.get(str(dimension))
        if not selected or selected not in values:
            continue
        overlay_resource = sdk_dir / str(values[selected])
        _merge_document(
            composed, _parse_document(overlay_resource.read_text(encoding="utf-8"), str(overlay_resource))
        )
        applied[str(dimension)] = selected
    if applied:
        composed["selectors"] = applied
    return composed


def load_evaluation_rules(
    sdk: str,
    path: str | Path | None = None,
    *,
    task: str | None = None,
    selectors: Mapping[str, str] | None = None,
    framework: str | None = None,
) -> Mapping[str, Any]:
    """Composed rules for (sdk, task, selectors).

    Composition: manifest defaults -> the task group's common document -> one
    overlay per task-declared dimension whose selector carries a registered
    value (in manifest declaration order). ``task`` falls back to the
    manifest's ``default_task``; a selector with no registered overlay simply
    applies nothing. ``framework=`` is sugar for ``selectors={"framework":
    ...}``. ``path`` bypasses composition and loads one self-contained rules
    document.
    """

    combined = dict(selectors or {})
    if framework:
        combined.setdefault("framework", framework)
    selector_items = tuple(sorted((str(key), str(value)) for key, value in combined.items() if value))
    return _load_rules_cached(sdk, task, selector_items, str(Path(path)) if path else None)


def available_tasks(sdk: str) -> list[str]:
    sdk_dir = evaluation_sdk_dir(sdk)
    manifest_resource = sdk_dir / "index.yaml"
    manifest = _parse_manifest(manifest_resource.read_text(encoding="utf-8"), str(manifest_resource))
    tasks = manifest.get("tasks") if isinstance(manifest.get("tasks"), Mapping) else {}
    return sorted(str(name) for name in tasks)


def _evidence_numbers(evidence: str) -> list[float]:
    return [float(match.group(1)) for match in _NUMBER_RE.finditer(evidence)]


def _trend_matches(kind: str, evidence: str) -> bool:
    values = _evidence_numbers(evidence)
    if kind == "single_value":
        return len(values) == 1
    if len(values) < 2:
        return False
    if kind == "improving":
        return values[-1] > values[0]
    if kind == "not_improving":
        return values[-1] <= values[0]
    return False


def _rule_matches(rule: Mapping[str, Any], evidence_lower: str, evidence: str, context: Mapping[str, Any]) -> bool:
    gate = rule.get("when_context")
    if isinstance(gate, Mapping):
        for key, expected in gate.items():
            if str(context.get(key) or "") != str(expected):
                return False
    if "contains" in rule:
        return str(rule["contains"]).lower() in evidence_lower
    if "contains_any" in rule:
        needles = rule["contains_any"] if isinstance(rule["contains_any"], list) else [rule["contains_any"]]
        return any(str(needle).lower() in evidence_lower for needle in needles)
    if "trend" in rule:
        return _trend_matches(str(rule["trend"]), evidence)
    # A context-only rule matches once its gate passed.
    return isinstance(gate, Mapping)


def score_signal(
    rules: Mapping[str, Any],
    signal: str,
    evidence: str,
    context: Mapping[str, Any] | None = None,
) -> str:
    evidence = str(evidence or "")
    evidence_lower = evidence.lower()
    if not evidence_lower or evidence_lower == "not captured":
        return "unknown"
    signals = rules.get("signals") if isinstance(rules.get("signals"), Mapping) else {}
    entry = signals.get(signal) if isinstance(signals, Mapping) else None
    signal_rules = entry.get("rules") if isinstance(entry, Mapping) else None
    for rule in signal_rules or []:
        if isinstance(rule, Mapping) and _rule_matches(rule, evidence_lower, evidence, context or {}):
            return str(rule.get("verdict") or rules.get("default_verdict") or "caution")
    return str(rules.get("default_verdict") or "caution")


def score_profile(
    rules: Mapping[str, Any],
    profile: Mapping[str, str],
    context: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    return {signal: score_signal(rules, signal, evidence, context) for signal, evidence in profile.items()}


def verdict_points(rules: Mapping[str, Any]) -> dict[str, float]:
    scoring = rules.get("scoring") if isinstance(rules.get("scoring"), Mapping) else {}
    points = scoring.get("points") if isinstance(scoring, Mapping) else None
    if not isinstance(points, Mapping):
        return {"good": 1.0, "caution": 0.5, "poor": 0.0, "bad": 0.0}
    return {str(verdict): float(value) for verdict, value in points.items()}


def overall_thresholds(rules: Mapping[str, Any]) -> dict[str, float]:
    scoring = rules.get("scoring") if isinstance(rules.get("scoring"), Mapping) else {}
    thresholds = scoring.get("overall_thresholds") if isinstance(scoring, Mapping) else None
    if not isinstance(thresholds, Mapping):
        return {"good": 0.8, "caution": 0.5}
    return {str(label): float(value) for label, value in thresholds.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a signals profile against an SDK's evaluation rules.")
    parser.add_argument("--sdk", required=True, help="SDK id, resolves benchmark/config/evaluation/<sdk>/index.yaml")
    parser.add_argument("--task", help="Task group (default: the manifest's default_task)")
    parser.add_argument(
        "--select",
        action="append",
        default=[],
        metavar="DIM=VALUE",
        help="Overlay selector, repeatable (e.g. --select framework=pytorch --select algorithm=fedavg)",
    )
    parser.add_argument("--rules", help="Explicit self-contained rules file path (overrides composition)")
    parser.add_argument("--profile", required=True, help="JSON file of {signal: evidence string}")
    parser.add_argument("--context", help="JSON object of context values (e.g. {\"target_framework\": \"lightning\"})")
    args = parser.parse_args(argv)
    selectors = {}
    for item in args.select:
        dimension, _, value = str(item).partition("=")
        if not dimension or not value:
            raise SystemExit(f"--select expects DIM=VALUE, got {item!r}")
        selectors[dimension] = value
    rules = load_evaluation_rules(args.sdk, args.rules, task=args.task, selectors=selectors)
    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    context = json.loads(args.context) if args.context else {}
    verdicts = score_profile(rules, profile, context)
    result = {"sdk": args.sdk, "task": rules.get("task"), "verdicts": verdicts}
    if rules.get("selectors"):
        result["selectors"] = rules["selectors"]
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
