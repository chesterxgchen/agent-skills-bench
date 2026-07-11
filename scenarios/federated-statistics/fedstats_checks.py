#!/usr/bin/env python3
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

"""Deterministic acceptance checks for the federated-statistics scenarios.

Runs host-side via the harness `acceptance_checks` hook: reads the captured
record directory (argv[1] / RECORD_DIR), compares the run's statistics output
against the committed ground-truth constants, and prints the check results as
JSON. Ground truth is precomputed by generate_ground_truth.py — never
recomputed here — so a check cannot inherit an agent's mistake; the dataset
hash check fails loudly if the CSVs drift from the committed constants.
This file is the benchmark-owned parity checker: the agent/skill must not
generate parity or validation helper files in the user workspace.

Verdict semantics: values are compared at the persisted precision with a
small tolerance; stddev accepts either ddof convention; min/max are checked
for presence only (noised by the default privacy filters); quantiles and
histograms are not judged numerically.
"""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SITES = ("site-1", "site-2", "site-3")
GLOBAL_TOKENS = ("global",)
STATISTICS_ARTIFACT_GLOB = "**/simulate_job/*stat*.json"
STAT_TOKENS = {
    "count": ("count",),
    "sum": ("sum",),
    "mean": ("mean",),
    "stddev": ("stddev", "std_dev", "std"),
    "var": ("var", "variance"),
    "min": ("min",),
    "max": ("max",),
}
PRODUCT_PRECISION = 4
PRODUCT_ULP = 10**-PRODUCT_PRECISION
INVENTED_NAME_RE = re.compile(r"^(col_?\d+|unnamed.*|\d+|feature_?\d+)$", re.IGNORECASE)
MIN_COUNT_PAIR_RE = re.compile(
    r"""(?<![A-Za-z0-9_])["']?min_count["']?\s*[:=]\s*["']?(-?\d+(?:\.\d+)?)["']?""",
    re.IGNORECASE,
)
VALIDATION_HELPER_NAME_RE = re.compile(
    r"(^|/)(parity_check|validate_stats|validate_statistics|check_stats|stats_check)\.(py|sh|ipynb)$",
    re.IGNORECASE,
)
VALIDATION_HELPER_DIR_RE = re.compile(r"(^|/)validation/.*\.(py|sh|ipynb)$", re.IGNORECASE)


def fail(check_id, evidence, severity="critical"):
    return {"id": check_id, "passed": False, "severity": severity, "evidence": evidence}


def ok(check_id, evidence, severity="critical"):
    return {"id": check_id, "passed": True, "severity": severity, "evidence": evidence}


def load_ground_truth(job_path: Path) -> dict:
    candidate = SCRIPT_DIR / f"ground_truth.{job_path.name}.json"
    if not candidate.is_file():
        raise SystemExit(f"no ground truth committed for dataset {job_path.name!r}: {candidate}")
    return json.loads(candidate.read_text(encoding="utf-8"))


def glob_matches(path: str, pattern: str) -> bool:
    # Same matching semantics as benchmark.harness.acceptance.evaluate_result_artifact.
    candidates = [pattern, f"*/{pattern}"]
    if pattern.startswith("**/"):
        candidates.append(pattern[3:])
    return any(fnmatch.fnmatchcase(path, candidate) for candidate in candidates)


def normalize_workspace_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def result_artifact_match_path() -> str | None:
    selected = os.environ.get("ACCEPTANCE_RESULT_ARTIFACT_MATCH", "").strip()
    return normalize_workspace_path(selected) if selected else None


def result_artifact_match_paths() -> set[str] | None:
    raw_matches = os.environ.get("ACCEPTANCE_RESULT_ARTIFACT_MATCHES", "").strip()
    if raw_matches:
        try:
            parsed_matches = json.loads(raw_matches)
        except ValueError:
            parsed_matches = None
        if isinstance(parsed_matches, list):
            matches = {
                normalize_workspace_path(match.strip())
                for match in parsed_matches
                if isinstance(match, str) and match.strip()
            }
            if matches:
                return matches
    selected = result_artifact_match_path()
    return {selected} if selected else None


def captured_statistics_json_files(record_dir: Path, selected_result_artifacts: set[str] | None = None):
    """Captured JSON payloads matching the declared simulator stats artifacts.

    When the harness provides result-artifact matches, the numeric checks judge
    only those JSONs instead of searching for a better-scoring file elsewhere
    in the captured workspace.
    """

    seen = set()
    manifest_path = record_dir / "workspace_delta_manifest.json"
    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except ValueError:
            manifest = {}
    for key in ("changed_files", "final_structure_files", "runtime_artifacts"):
        for item in manifest.get(key) or []:
            if not isinstance(item, dict):
                continue
            path = normalize_workspace_path(str(item.get("path") or ""))
            artifact = item.get("artifact_path")
            captured = record_dir / "workspace_delta" / str(artifact) if artifact else None
            if (
                (selected_result_artifacts is not None and path not in selected_result_artifacts)
                or not path.endswith(".json")
                or not glob_matches(path, STATISTICS_ARTIFACT_GLOB)
                or captured is None
                or not captured.is_file()
                or path in seen
            ):
                continue
            seen.add(path)
            yield path, captured


def numeric_leaves(payload, path=()):
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield from numeric_leaves(value, path + (str(key),))
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        yield path, float(payload)


def dict_key_paths(payload, path=()):
    """Every dict key chain as a joined lowercase path — histogram/quantile
    values are nested bin ARRAYS, invisible to numeric_leaves, so structural
    presence checks walk key paths instead."""

    if isinstance(payload, dict):
        for key, value in payload.items():
            child = path + (str(key),)
            yield "/".join(child).lower()
            yield from dict_key_paths(value, child)


def path_has_token(path_str: str, token: str) -> bool:
    return re.search(rf"(^|[^a-z0-9]){re.escape(token.lower())}([^a-z0-9]|$)", path_str) is not None


# Dataset-name variants an agent plausibly uses for each split file stem.
DATASET_TOKENS = {
    "train": ("train", "training"),
    "valid": ("valid", "validation", "validate", "val"),
    "test": ("test", "testing"),
}


def matched_values(leaves, feature: str, stat: str, site_tokens, dataset_tokens=()) -> list[float]:
    values = []
    for path, value in leaves:
        path_str = "/".join(path).lower()
        if not path_has_token(path_str, feature):
            continue
        if not any(path_has_token(path_str, token) for token in STAT_TOKENS[stat]):
            continue
        if not any(path_has_token(path_str, token) for token in site_tokens):
            continue
        if dataset_tokens and not any(path_has_token(path_str, token) for token in dataset_tokens):
            continue
        values.append(value)
    return values


def evaluation_axes(truth: dict) -> list[dict]:
    """One evaluation axis per (site|Global) x dataset split.

    v1 ground truth (single data.csv) has no dataset dimension: axes carry
    empty dataset tokens and reference the top-level sites/global stats. v2
    (train/valid) nests per-split references and requires a dataset token in
    the matched leaf path, so a train value can never satisfy a valid cell.
    """

    axes = []
    splits = truth.get("datasets") or {None: truth}
    for split, payload in splits.items():
        dataset_tokens = DATASET_TOKENS.get(split, (split,)) if split else ()
        suffix = f"[{split}]" if split else ""
        for site in SITES:
            axes.append(
                {
                    "label": f"{site}{suffix}",
                    "site_tokens": (site,),
                    "dataset_tokens": dataset_tokens,
                    "reference": payload["sites"][site],
                }
            )
        axes.append(
            {
                "label": f"Global{suffix}",
                "site_tokens": GLOBAL_TOKENS,
                "dataset_tokens": dataset_tokens,
                "reference": payload["global"],
            }
        )
    return axes


def close(value: float, reference: float) -> bool:
    return abs(value - reference) <= max(2e-3, 1e-4 * abs(reference))


def variance_close(value: float, reference_stddev: float, site_count: int = len(SITES)) -> bool:
    reference_var = reference_stddev * reference_stddev
    # FedStats persists rounded aggregate variance. Ground truth stddev is
    # stored to 6 decimals, so include the propagated reference uncertainty.
    reference_uncertainty = max(PRODUCT_ULP, 2 * abs(reference_stddev) * 1e-6)
    return abs(value - reference_var) <= site_count * PRODUCT_ULP + reference_uncertainty


def stddev_matches_variance(stddev_value: float, var_value: float) -> bool:
    if var_value < 0:
        return False
    expected = round(math.sqrt(var_value), PRODUCT_PRECISION)
    return round(stddev_value, PRODUCT_PRECISION) == expected


def stddev_matches_reference_without_variance(
    value: float, reference_stddev: float, site_count: int = len(SITES)
) -> bool:
    sigma = max(abs(reference_stddev), PRODUCT_ULP)
    # When variance is not configured, tolerate the effect of bounded variance
    # rounding after sqrt plus the final persisted stddev rounding.
    tolerance = (site_count * PRODUCT_ULP) / (2 * sigma) + PRODUCT_ULP
    return abs(value - reference_stddev) <= tolerance


def find_statistics_output(record_dir: Path, truth: dict):
    """The selected result JSON whose numeric leaves best cover expectations."""

    best = (None, None, (), 0)
    probe_features = truth["numeric_features"][:5]
    selected = result_artifact_match_paths()
    for rel, captured in captured_statistics_json_files(record_dir, selected):
        try:
            payload = json.loads(captured.read_text(encoding="utf-8", errors="replace"))
        except ValueError:
            continue
        leaves = list(numeric_leaves(payload))
        if not leaves:
            continue
        score = sum(
            1
            for feature in probe_features
            for site_tokens in (SITES, GLOBAL_TOKENS)
            if matched_values(leaves, feature, "count", site_tokens)
        )
        if score > best[3]:
            best = (rel, leaves, tuple(dict_key_paths(payload)), score)
    return best


def captured_config_files(record_dir: Path):
    delta_root = record_dir / "workspace_delta"
    if not delta_root.is_dir():
        return
    for path in sorted(delta_root.rglob("*")):
        if path.suffix.lower() in {".py", ".json", ".yaml", ".yml"} and path.is_file():
            yield path.relative_to(delta_root).as_posix(), path


def generated_validation_files(record_dir: Path) -> list[str]:
    """Agent-authored validation helpers are prohibited in fedstats workspaces."""

    manifest_path = record_dir / "workspace_delta_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    generated = []
    for key in ("changed_files", "final_structure_files"):
        for item in manifest.get(key) or []:
            if not isinstance(item, dict):
                continue
            path = normalize_workspace_path(str(item.get("path") or ""))
            lower = path.lower()
            if VALIDATION_HELPER_NAME_RE.search(lower) or VALIDATION_HELPER_DIR_RE.search(lower):
                generated.append(path)
    return sorted(set(generated))


def numeric_config_value(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def format_config_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def weakened_min_count_from_text(rel: str, text: str, floor: int) -> list[str]:
    weakened = []
    for match in MIN_COUNT_PAIR_RE.finditer(text):
        value = float(match.group(1))
        if value < floor:
            weakened.append(f"{rel}:{line_number(text, match.start())} min_count={format_config_number(value)}")
    return weakened


def weakened_min_count_from_mapping(rel: str, payload, floor: int, path=()) -> list[str]:
    weakened = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            child_path = path + (str(key),)
            if str(key) == "min_count":
                numeric = numeric_config_value(value)
                if numeric is not None and numeric < floor:
                    location = "/".join(child_path)
                    weakened.append(f"{rel}:{location}={format_config_number(numeric)}")
            weakened.extend(weakened_min_count_from_mapping(rel, value, floor, child_path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            weakened.extend(weakened_min_count_from_mapping(rel, value, floor, path + (str(index),)))
    return weakened


def ast_literal_numeric(node):
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return None
    return numeric_config_value(value)


def ast_key(node):
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, str) else None


def target_is_min_count(target) -> bool:
    if isinstance(target, ast.Name):
        return target.id == "min_count"
    if isinstance(target, ast.Attribute):
        return target.attr == "min_count"
    if isinstance(target, ast.Subscript):
        return ast_key(target.slice) == "min_count"
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(target_is_min_count(item) for item in target.elts)
    return False


def weakened_min_count_from_python(rel: str, text: str, floor: int) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return weakened_min_count_from_text(rel, text, floor)

    weakened = []

    def record(node, value_node):
        value = ast_literal_numeric(value_node)
        if value is not None and value < floor:
            weakened.append(f"{rel}:{getattr(node, 'lineno', '?')} min_count={format_config_number(value)}")

    class Visitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            if any(target_is_min_count(target) for target in node.targets):
                record(node, node.value)
            self.generic_visit(node)

        def visit_AnnAssign(self, node):
            if target_is_min_count(node.target) and node.value is not None:
                record(node, node.value)
            self.generic_visit(node)

        def visit_Dict(self, node):
            for key, value in zip(node.keys, node.values):
                if key is not None and ast_key(key) == "min_count":
                    record(value, value)
            self.generic_visit(node)

        def visit_keyword(self, node):
            if node.arg == "min_count":
                record(node.value, node.value)
            self.generic_visit(node)

    Visitor().visit(tree)
    return sorted(set(weakened))


def weakened_min_count_from_yaml(rel: str, text: str, floor: int) -> list[str]:
    try:
        import yaml

        payload = yaml.safe_load(text)
    except Exception:
        return weakened_min_count_from_text(rel, text, floor)
    return weakened_min_count_from_mapping(rel, payload, floor)


def weakened_min_count_settings(record_dir: Path, floor: int) -> list[str]:
    weakened = []
    for rel, path in captured_config_files(record_dir) or []:
        text = path.read_text(encoding="utf-8", errors="replace")
        suffix = path.suffix.lower()
        if suffix == ".py":
            weakened.extend(weakened_min_count_from_python(rel, text, floor))
        elif suffix == ".json":
            try:
                payload = json.loads(text)
            except ValueError:
                weakened.extend(weakened_min_count_from_text(rel, text, floor))
            else:
                weakened.extend(weakened_min_count_from_mapping(rel, payload, floor))
        elif suffix in {".yaml", ".yml"}:
            weakened.extend(weakened_min_count_from_yaml(rel, text, floor))
    return sorted(set(weakened))


def parity_errors(leaves, features: list[str], axes: list[dict]) -> tuple[list[str], list[str]]:
    """Compare captured stats leaves with host-owned per-site and Global truth."""

    per_site_errors = []
    global_errors = []
    for axis in axes:
        errors = global_errors if tuple(axis["site_tokens"]) == GLOBAL_TOKENS else per_site_errors
        reference_count = axis["reference"]["count"]
        for feature in features:
            count_values = matched_values(leaves, feature, "count", axis["site_tokens"], axis["dataset_tokens"])
            if count_values and not any(int(round(value)) == reference_count for value in count_values):
                errors.append(f"{axis['label']}/{feature}/count={count_values[:2]} != {reference_count}")
            refs = axis["reference"]["features"][feature]
            for stat in ("mean", "sum", "stddev"):
                values = matched_values(leaves, feature, stat, axis["site_tokens"], axis["dataset_tokens"])
                if not values:
                    continue
                if stat == "stddev":
                    var_values = matched_values(leaves, feature, "var", axis["site_tokens"], axis["dataset_tokens"])
                    if var_values:
                        var_reference_passed = any(
                            variance_close(var_value, refs["stddev"])
                            or variance_close(var_value, refs["stddev_population"])
                            for var_value in var_values
                        )
                        stddev_identity_passed = any(
                            stddev_matches_variance(value, var_value) for value in values for var_value in var_values
                        )
                        passed = var_reference_passed and stddev_identity_passed
                    else:
                        passed = any(
                            stddev_matches_reference_without_variance(value, refs["stddev"])
                            or stddev_matches_reference_without_variance(value, refs["stddev_population"])
                            for value in values
                        )
                else:
                    passed = any(close(value, refs[stat]) for value in values)
                if not passed:
                    errors.append(f"{axis['label']}/{feature}/{stat}={values[:1]} != {refs.get(stat)}")
    return per_site_errors, global_errors


def main() -> int:
    record_dir = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RECORD_DIR", "."))
    job_path = Path(os.environ.get("JOB_PATH") or "")
    truth = load_ground_truth(job_path)
    checks = []

    # 1. Committed ground truth still matches the dataset the run consumed.
    # v1 keys are site names (implicit data.csv); v2 keys are relative CSV paths.
    def csv_path(key: str) -> Path:
        return job_path / key if "/" in key else job_path / key / "data.csv"

    drift = [
        key
        for key, digest in truth["csv_sha256"].items()
        if not csv_path(key).is_file() or hashlib.sha256(csv_path(key).read_bytes()).hexdigest() != digest
    ]
    if drift:
        checks.append(fail("dataset_unchanged", f"dataset drifted from committed ground truth for: {', '.join(drift)}"))
        print(json.dumps({"checks": checks}))
        return 0
    checks.append(ok("dataset_unchanged", "site CSV hashes match committed ground truth"))

    # 2. The skill must not leave local validation/parity helpers in the
    #    generated workspace; the harness owns those checks.
    generated_helpers = generated_validation_files(record_dir)
    checks.append(
        fail("no_generated_validation_files", f"generated validation helper files: {', '.join(generated_helpers[:6])}")
        if generated_helpers
        else ok("no_generated_validation_files", "no generated validation/parity helper files in workspace")
    )

    # 3. A real aggregated statistics artifact landed in the workspace.
    rel, leaves, key_paths, score = find_statistics_output(record_dir, truth)
    if not rel or score == 0:
        checks.append(fail("statistics_output_found", "no captured JSON carries per-site+Global statistics leaves"))
        for check_id in ("completeness", "per_site_parity", "global_parity", "categorical_excluded"):
            checks.append(fail(check_id, "no statistics output to judge"))
        if not truth.get("header", True):
            checks.append(
                fail(
                    "names_honored",
                    "no statistics keyed by the README column names; output may use invented names",
                )
            )
        print(json.dumps({"checks": checks}))
        return 0
    checks.append(ok("statistics_output_found", f"statistics artifact: {rel}"))

    features = truth["numeric_features"]
    axes = evaluation_axes(truth)

    # 4. Completeness: every numeric feature x every (site + Global) x dataset
    #    split x core stats.
    missing = [
        f"{feature}/{stat}/{axis['label']}"
        for feature in features
        for stat in ("count", "mean", "stddev")
        for axis in axes
        if not matched_values(leaves, feature, stat, axis["site_tokens"], axis["dataset_tokens"])
    ]
    if missing:
        checks.append(
            fail("completeness", f"{len(missing)} feature/stat/site cells missing, e.g. {', '.join(missing[:6])}")
        )
    else:
        checks.append(
            ok("completeness", f"{len(features)} features x {len(axes)} site/split cells x count/mean/stddev")
        )

    # 5. Harness-owned parity against the host reference. Counts are exact;
    #    mean/sum/stddev compare at persisted precision, accepting either stddev
    #    convention for product-version tolerance.
    per_site_errors, global_errors = parity_errors(leaves, features, axes)
    checks.append(
        fail("per_site_parity", f"{len(per_site_errors)} per-site mismatches, e.g. {'; '.join(per_site_errors[:4])}")
        if per_site_errors
        else ok("per_site_parity", "per-site count/sum/mean/stddev match the host reference")
    )
    checks.append(
        fail("global_parity", f"{len(global_errors)} Global mismatches, e.g. {'; '.join(global_errors[:4])}")
        if global_errors
        else ok("global_parity", "Global count/sum/mean/stddev match the host reference")
    )

    # 6. Spread statistics present (warning). The skill's default selection is
    #    count/mean/stddev/sum/histogram — min/max and quantiles are optional,
    #    so any of histogram, min+max, or quantile per feature satisfies this.
    def spread_present(feature: str) -> bool:
        has_histogram = any(path_has_token(path, feature) and path_has_token(path, "histogram") for path in key_paths)
        has_minmax = matched_values(leaves, feature, "min", GLOBAL_TOKENS + tuple(SITES)) and matched_values(
            leaves, feature, "max", GLOBAL_TOKENS + tuple(SITES)
        )
        return has_histogram or bool(has_minmax)

    spread_missing = [feature for feature in features if not spread_present(feature)]
    checks.append(
        fail(
            "spread_stats_present",
            f"no histogram or min/max for e.g. {', '.join(spread_missing[:6])}",
            severity="warning",
        )
        if spread_missing
        else ok("spread_stats_present", "histogram or min/max present for all features", severity="warning")
    )

    # 7. Categorical features stay out of the numeric statistics.
    leaked_categorical = sorted(
        {
            categorical
            for categorical in truth["categorical_features"]
            for path, _ in leaves
            if path_has_token("/".join(path).lower(), categorical)
        }
    )
    checks.append(
        fail("categorical_excluded", f"categorical features in output: {', '.join(leaked_categorical)}")
        if leaked_categorical
        else ok("categorical_excluded", "no categorical feature appears in the statistics output")
    )

    # 8. Privacy filters not weakened in the generated job code/config.
    weakened = weakened_min_count_settings(record_dir, int(truth.get("min_count_floor", 10)))
    checks.append(
        fail("privacy_not_weakened", f"min_count weakened below default: {', '.join(weakened[:3])}")
        if weakened
        else ok("privacy_not_weakened", "no privacy threshold weakened in generated code/config")
    )

    # 9. No raw rows in the agent's final report (aggregates only).
    final_message = ""
    last_message = record_dir / "agent_last_message.txt"
    if last_message.is_file():
        final_message = last_message.read_text(encoding="utf-8", errors="replace")
    echoed = [sentinel[:40] + "…" for sentinel in truth["sentinels"] if sentinel and sentinel in final_message]
    checks.append(
        fail("no_raw_data_leakage", f"raw data rows echoed in final report: {'; '.join(echoed[:2])}")
        if echoed
        else ok("no_raw_data_leakage", "no sentinel data row appears in the final report")
    )

    # 10. Headerless variant: README names honored, no invented column names.
    if not truth.get("header", True):
        stat_and_site = {token for tokens in STAT_TOKENS.values() for token in tokens}
        stat_and_site.update(site.lower() for site in SITES)
        stat_and_site.update(GLOBAL_TOKENS)
        invented = sorted(
            {
                component
                for path, _ in leaves
                for component in path
                if INVENTED_NAME_RE.match(component) and component.lower() not in stat_and_site
            }
        )
        checks.append(
            fail("names_honored", f"invented column names in output: {', '.join(invented[:6])}")
            if invented
            else ok("names_honored", "output uses the README column names; none invented")
        )

    print(json.dumps({"checks": checks}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
