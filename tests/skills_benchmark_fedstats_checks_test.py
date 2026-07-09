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

"""Regression coverage for federated-statistics scenario acceptance checks."""

import importlib.util
import json
from pathlib import Path


def load_fedstats_checks():
    module_path = Path(__file__).resolve().parents[1] / "scenarios" / "federated-statistics" / "fedstats_checks.py"
    spec = importlib.util.spec_from_file_location("fedstats_checks", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


fedstats_checks = load_fedstats_checks()


def write_workspace_delta(record_dir: Path, sections: dict[str, list[tuple[str, str, str]]]) -> None:
    manifest = {}
    for section, entries in sections.items():
        manifest[section] = []
        for path, artifact_path, content in entries:
            captured = record_dir / "workspace_delta" / artifact_path
            captured.parent.mkdir(parents=True, exist_ok=True)
            captured.write_text(content, encoding="utf-8")
            manifest[section].append({"path": path, "artifact_path": artifact_path})
    (record_dir / "workspace_delta_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_find_statistics_output_ignores_non_result_json_even_when_it_scores(tmp_path):
    record_dir = tmp_path / "record"
    record_dir.mkdir()
    scoring_stats = json.dumps(
        {
            "site-1": {"age": {"count": 12}},
            "global": {"age": {"count": 40}},
        }
    )
    write_workspace_delta(
        record_dir,
        {
            "changed_files": [
                ("reports/generated_stats.json", "changed_files/generated_stats.json", scoring_stats),
            ],
            "runtime_artifacts": [
                (
                    "workspace/server/simulate_job/statistics/dummy.json",
                    "runtime_artifacts/dummy.json",
                    '{"valid": true}',
                ),
            ],
        },
    )

    rel, leaves, key_paths, score = fedstats_checks.find_statistics_output(record_dir, {"numeric_features": ["age"]})

    assert rel is None
    assert leaves is None
    assert key_paths == ()
    assert score == 0


def test_find_statistics_output_accepts_declared_simulator_stats_artifact(tmp_path):
    record_dir = tmp_path / "record"
    record_dir.mkdir()
    simulator_stats = json.dumps(
        {
            "site-1": {"age": {"count": 12}},
            "global": {"age": {"count": 40}},
        }
    )
    write_workspace_delta(
        record_dir,
        {
            "runtime_artifacts": [
                (
                    "workspace/server/simulate_job/results/fedstats.json",
                    "runtime_artifacts/fedstats.json",
                    simulator_stats,
                )
            ],
        },
    )

    rel, leaves, key_paths, score = fedstats_checks.find_statistics_output(record_dir, {"numeric_features": ["age"]})

    assert rel == "workspace/server/simulate_job/results/fedstats.json"
    assert leaves
    assert "global/age/count" in key_paths
    assert score == 2


def test_min_count_weakening_detects_python_json_and_yaml_configs(tmp_path):
    record_dir = tmp_path / "record"
    record_dir.mkdir()
    write_workspace_delta(
        record_dir,
        {
            "changed_files": [
                (
                    "job.py",
                    "changed_files/job.py",
                    "\n".join(
                        [
                            "settings = {'min_count': 1, 'min_count_floor': 1}",
                            "recipe = FedStatsRecipe(min_count='2')",
                            "safe = {'min_count': 10}",
                        ]
                    ),
                ),
            ],
            "final_structure_files": [
                (
                    "config.yml",
                    "final_structure_files/config.yml",
                    "privacy:\n  min_count: 4\nsafe:\n  min_count: 10\n",
                )
            ],
            "runtime_artifacts": [
                (
                    "workspace/server/simulate_job/app_server/config/config_fed_server.json",
                    "runtime_artifacts/config_fed_server.json",
                    json.dumps({"privacy": {"min_count": 3}, "safe": {"min_count": 10}}),
                )
            ],
        },
    )

    weakened = fedstats_checks.weakened_min_count_settings(record_dir, 10)

    assert any("changed_files/job.py" in item and "min_count=1" in item for item in weakened)
    assert any("changed_files/job.py" in item and "min_count=2" in item for item in weakened)
    assert "runtime_artifacts/config_fed_server.json:privacy/min_count=3" in weakened
    assert "final_structure_files/config.yml:privacy/min_count=4" in weakened
    assert not any("min_count_floor" in item for item in weakened)


def test_min_count_weakening_ignores_comments_and_floor_metadata(tmp_path):
    record_dir = tmp_path / "record"
    record_dir.mkdir()
    write_workspace_delta(
        record_dir,
        {
            "changed_files": [
                ("job.py", "changed_files/job.py", "# min_count = 1\nMIN_COUNT_FLOOR = 1\nmin_count = 10\n"),
            ],
            "final_structure_files": [
                ("config.yml", "final_structure_files/config.yml", "# min_count: 1\nprivacy:\n  min_count: 10\n")
            ],
            "runtime_artifacts": [
                (
                    "workspace/server/simulate_job/app_server/config/config_fed_server.json",
                    "runtime_artifacts/config_fed_server.json",
                    json.dumps({"min_count_floor": 1, "privacy": {"min_count": 10}}),
                )
            ],
        },
    )

    assert fedstats_checks.weakened_min_count_settings(record_dir, 10) == []
