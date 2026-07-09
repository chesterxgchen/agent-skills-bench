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

"""Scenario-declared acceptance gates evaluated host-side after a run.

The harness owns two generic MECHANISMS here; scenarios supply the data:

- ``result_artifact`` — a declared workspace glob (plus optional format) that
  proves the run produced a real result. Evaluated against the captured
  workspace delta, so a run whose "result" never landed in the workspace
  cannot pass by narration alone.
- ``acceptance_checks`` — a scenario-local script executed against the record
  directory. It prints ``{"checks": [{"id", "passed", ...}]}`` on stdout;
  failed critical checks feed the existing ``critical_quality_checks_failed``
  quality gate. The script is TRUSTED HOST CODE (same trust as the scenario
  file and bin/run.sh): it runs unsandboxed as the operator's user, and
  compile-time validation confines it to the scenario's directory tree.

Both results persist to ``acceptance_checks.json`` in the record directory and
merge into the record summary's ``quality_checks`` list, which the quality
gate and reports already consume. Neither mechanism knows any SDK, task, or
skill: onboarding a new skill needs criteria/scenario data, not harness code.
"""

from __future__ import annotations

import fnmatch
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from .common import load_json

ACCEPTANCE_CHECKS_FILENAME = "acceptance_checks.json"
MAX_CHECK_EVIDENCE_CHARS = 2_000
MAX_SCRIPT_OUTPUT_BYTES = 1_000_000
_ALLOWED_SEVERITIES = {"critical", "warning", "info"}


def _clip(text: Any) -> str:
    value = str(text or "")
    return value if len(value) <= MAX_CHECK_EVIDENCE_CHARS else value[: MAX_CHECK_EVIDENCE_CHARS - 1] + "…"


def _manifest_artifact_items(record_dir: Path) -> list[dict[str, Any]]:
    manifest = load_json(record_dir / "workspace_delta_manifest.json", {}) or {}
    if not isinstance(manifest, dict):
        return []
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    # runtime_artifacts carries execution outputs (e.g. a simulator workspace's
    # statistics JSON) that never appear as changed source files.
    for key in ("changed_files", "final_structure_files", "runtime_artifacts"):
        values = manifest.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or "")
            if not path or path in seen:
                continue
            seen.add(path)
            items.append(item)
    return items


def evaluate_result_artifact(record_dir: Path, declared: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Evaluate a declared result artifact against the captured workspace delta.

    Returns a quality-check entry (``severity: critical``) plus match metadata,
    or ``None`` when the job declares no result artifact.
    """

    if not isinstance(declared, Mapping) or not declared.get("glob"):
        return None
    pattern = str(declared["glob"])
    artifact_format = str(declared.get("format") or "")
    matches = []
    parsed_matches = []
    parse_failures = []
    for item in _manifest_artifact_items(record_dir):
        path = str(item.get("path") or "")
        # fnmatchcase (platform-independent, no case-folding): "*" crosses "/"
        # so "**/x.json" matches nested paths; the stripped form covers a
        # root-level match, the prefixed form lets a bare filename pattern
        # match at any depth.
        candidates = [pattern, f"*/{pattern}"]
        if pattern.startswith("**/"):
            candidates.append(pattern[3:])
        if not any(fnmatch.fnmatchcase(path, candidate) for candidate in candidates):
            continue
        matches.append(path)
        if artifact_format != "json":
            continue
        artifact_path = item.get("artifact_path")
        captured = record_dir / "workspace_delta" / str(artifact_path) if artifact_path else None
        if captured is None or not captured.is_file():
            parse_failures.append(f"{path}: captured copy not available")
            continue
        try:
            json.loads(captured.read_text(encoding="utf-8", errors="replace"))
        except ValueError as exc:
            parse_failures.append(f"{path}: {exc}")
        else:
            parsed_matches.append(path)
    passed = bool(parsed_matches) if artifact_format == "json" else bool(matches)
    if not matches:
        evidence = f"no workspace artifact matches {pattern!r}"
    elif passed:
        evidence = f"matched {', '.join(matches[:5])}"
        if artifact_format == "json":
            evidence += " (valid JSON)"
    else:
        evidence = "; ".join(parse_failures[:5])
    check = {
        "id": "result_artifact",
        "severity": "critical",
        "passed": passed,
        "status": "pass" if passed else "fail",
        "evidence": _clip(evidence),
    }
    selected_match = None
    if artifact_format == "json" and parsed_matches:
        selected_match = parsed_matches[0]
    elif matches:
        selected_match = matches[0]
    return {
        "check": check,
        "glob": pattern,
        "format": artifact_format or None,
        "matches": matches,
        "parsed_matches": parsed_matches,
        "selected_match": selected_match,
        "parse_failures": parse_failures,
        "description": declared.get("description"),
    }


def _normalized_check(raw: Any, index: int, script: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {
            "id": f"check_{index:02d}",
            "severity": "critical",
            "passed": False,
            "status": "error",
            "evidence": _clip(f"{script}: check entry {index} is not an object: {raw!r}"),
        }
    severity = str(raw.get("severity") or "critical").lower()
    if severity not in _ALLOWED_SEVERITIES:
        severity = "critical"
    passed = raw.get("passed") is True
    return {
        "id": str(raw.get("id") or f"check_{index:02d}"),
        "severity": severity,
        "passed": passed,
        "status": "pass" if passed else "fail",
        "evidence": _clip(raw.get("evidence") or ""),
    }


def _runner_failure(script: str, evidence: str) -> list[dict[str, Any]]:
    # Fail closed: a broken/timed-out checker must not read as "gates passed".
    return [
        {
            "id": "acceptance_checks_runner",
            "severity": "critical",
            "passed": False,
            "status": "error",
            "evidence": _clip(f"{script}: {evidence}"),
        }
    ]


def run_acceptance_checks(
    record_dir: Path,
    declared: Mapping[str, Any] | None,
    *,
    job_path: str = "",
    mode: str = "",
    result_artifact: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run the scenario-local acceptance script and normalize its checks.

    Contract: the script receives the record directory as its only argument
    (plus ``RECORD_DIR``/``JOB_PATH``/``BENCHMARK_MODE`` in the environment)
    and prints ``{"checks": [{"id", "passed", "severity", "evidence"}]}``.
    A non-zero exit, timeout, or unparsable output is a failed critical check.
    """

    if not isinstance(declared, Mapping) or not declared.get("script"):
        return []
    script = str(declared["script"])
    script_path = Path(script)
    if not script_path.is_file():
        return _runner_failure(script, "acceptance check script not found")
    timeout = declared.get("timeout_seconds")
    timeout = timeout if isinstance(timeout, int) and not isinstance(timeout, bool) and timeout > 0 else 600
    command = [sys.executable, str(script_path)] if script_path.suffix == ".py" else [str(script_path)]
    command.append(str(record_dir))
    artifact_match = ""
    artifact_matches = "[]"
    if isinstance(result_artifact, Mapping):
        artifact_match = str(result_artifact.get("selected_match") or "")
        raw_matches = result_artifact.get("parsed_matches")
        if isinstance(raw_matches, list):
            artifact_matches = json.dumps([str(match) for match in raw_matches])
    try:
        result = subprocess.run(
            command,
            cwd=str(record_dir),
            env={
                "PATH": "/usr/bin:/bin:/usr/local/bin",
                "RECORD_DIR": str(record_dir),
                "JOB_PATH": str(job_path or ""),
                "BENCHMARK_MODE": str(mode or ""),
                "ACCEPTANCE_RESULT_ARTIFACT_MATCH": artifact_match,
                "ACCEPTANCE_RESULT_ARTIFACT_MATCHES": artifact_matches,
            },
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return _runner_failure(script, f"timed out after {timeout}s")
    except OSError as exc:
        return _runner_failure(script, f"failed to execute: {exc}")
    stdout = (result.stdout or "")[:MAX_SCRIPT_OUTPUT_BYTES]
    if result.returncode != 0:
        detail = (result.stderr or stdout or "no output").strip()
        return _runner_failure(script, f"exited with status {result.returncode}: {detail}")
    try:
        payload = json.loads(stdout or "{}")
    except ValueError as exc:
        return _runner_failure(script, f"stdout is not valid JSON: {exc}")
    checks = payload.get("checks") if isinstance(payload, dict) else None
    if not isinstance(checks, list) or not checks:
        return _runner_failure(script, 'output must contain a non-empty "checks" list')
    return [_normalized_check(raw, index, script) for index, raw in enumerate(checks)]


def apply_acceptance_gates(record_dir: Path, entry: Mapping[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """Evaluate both gates for a finished run and fold them into ``summary``.

    Persists ``acceptance_checks.json`` beside the record and appends every
    check to ``summary["quality_checks"]`` — the list the quality gate's
    ``critical_quality_checks_failed`` already scans. Returns the persisted
    payload (empty when the job declares neither gate).
    """

    artifact_result = evaluate_result_artifact(record_dir, entry.get("result_artifact"))
    script_checks = run_acceptance_checks(
        record_dir,
        entry.get("acceptance_checks"),
        job_path=str(entry.get("job_path") or ""),
        mode=str(entry.get("mode") or ""),
        result_artifact=artifact_result,
    )
    checks = ([artifact_result["check"]] if artifact_result else []) + script_checks
    if not checks:
        return {}
    payload: dict[str, Any] = {"schema_version": 1, "checks": checks}
    if artifact_result:
        payload["result_artifact"] = {key: value for key, value in artifact_result.items() if key != "check"}
    if entry.get("acceptance_checks"):
        payload["acceptance_checks"] = dict(entry["acceptance_checks"])
    existing = summary.get("quality_checks")
    merged = [check for check in existing if isinstance(check, dict)] if isinstance(existing, list) else []
    known_ids = {str(check.get("id")) for check in checks}
    merged = [check for check in merged if str(check.get("id")) not in known_ids]
    summary["quality_checks"] = merged + checks
    return payload
