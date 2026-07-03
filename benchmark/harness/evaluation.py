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


def evaluation_rules_resource(sdk: str):
    """Packaged rules document (package data works in wheels/sdists, not just checkouts)."""
    return resources.files(EVALUATION_RULES_PACKAGE) / "config" / "evaluation" / f"{sdk}.yaml"


def _parse_rules(text: str, source: str) -> Mapping[str, Any]:
    document = yaml.safe_load(text) or {}
    if not isinstance(document, dict):
        raise ValueError(f"evaluation rules must be a mapping: {source}")
    version = document.get("schema_version")
    if version != EVALUATION_RULES_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported evaluation rules schema_version {version!r} in {source} "
            f"(expected {EVALUATION_RULES_SCHEMA_VERSION})"
        )
    return document


@lru_cache(maxsize=8)
def _load_rules_cached(sdk: str, path_text: str | None) -> Mapping[str, Any]:
    if path_text is not None:
        return _parse_rules(Path(path_text).read_text(encoding="utf-8"), path_text)
    resource = evaluation_rules_resource(sdk)
    return _parse_rules(resource.read_text(encoding="utf-8"), str(resource))


def load_evaluation_rules(sdk: str, path: str | Path | None = None) -> Mapping[str, Any]:
    return _load_rules_cached(sdk, str(Path(path)) if path else None)


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
    parser.add_argument("--sdk", required=True, help="SDK id, resolves benchmark/config/evaluation/<sdk>.yaml")
    parser.add_argument("--rules", help="Explicit rules file path (overrides --sdk resolution)")
    parser.add_argument("--profile", required=True, help="JSON file of {signal: evidence string}")
    parser.add_argument("--context", help="JSON object of context values (e.g. {\"target_framework\": \"lightning\"})")
    args = parser.parse_args(argv)
    rules = load_evaluation_rules(args.sdk, args.rules)
    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    context = json.loads(args.context) if args.context else {}
    verdicts = score_profile(rules, profile, context)
    print(json.dumps({"sdk": args.sdk, "verdicts": verdicts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
