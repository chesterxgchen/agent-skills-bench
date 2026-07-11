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


def test_find_statistics_output_honors_selected_result_artifact_env(tmp_path, monkeypatch):
    record_dir = tmp_path / "record"
    record_dir.mkdir()
    scoring_stats = json.dumps(
        {
            "site-1": {"age": {"count": 12}},
            "global": {"age": {"count": 40}},
        }
    )
    selected_dummy = "workspace/server/simulate_job/statistics/dummy.json"
    write_workspace_delta(
        record_dir,
        {
            "runtime_artifacts": [
                (selected_dummy, "runtime_artifacts/dummy.json", '{"valid": true}'),
                (
                    "workspace/server/simulate_job/statistics/fedstats.json",
                    "runtime_artifacts/fedstats.json",
                    scoring_stats,
                ),
            ],
        },
    )
    monkeypatch.setenv("ACCEPTANCE_RESULT_ARTIFACT_MATCH", selected_dummy)

    rel, leaves, key_paths, score = fedstats_checks.find_statistics_output(record_dir, {"numeric_features": ["age"]})

    assert rel is None
    assert leaves is None
    assert key_paths == ()
    assert score == 0


def test_find_statistics_output_scores_best_declared_result_artifact_match(tmp_path, monkeypatch):
    record_dir = tmp_path / "record"
    record_dir.mkdir()
    scoring_stats = json.dumps(
        {
            "site-1": {"age": {"count": 12}},
            "global": {"age": {"count": 40}},
        }
    )
    selected_dummy = "workspace/server/simulate_job/statistics/dummy.json"
    valid_stats = "workspace/server/simulate_job/statistics/fedstats.json"
    write_workspace_delta(
        record_dir,
        {
            "runtime_artifacts": [
                (selected_dummy, "runtime_artifacts/dummy.json", '{"valid": true}'),
                (valid_stats, "runtime_artifacts/fedstats.json", scoring_stats),
                (
                    "workspace/server/simulate_job/statistics/unmatched_stats.json",
                    "runtime_artifacts/unmatched_stats.json",
                    scoring_stats,
                ),
            ],
        },
    )
    monkeypatch.setenv("ACCEPTANCE_RESULT_ARTIFACT_MATCH", selected_dummy)
    monkeypatch.setenv("ACCEPTANCE_RESULT_ARTIFACT_MATCHES", json.dumps([selected_dummy, valid_stats]))

    rel, leaves, key_paths, score = fedstats_checks.find_statistics_output(record_dir, {"numeric_features": ["age"]})

    assert rel == valid_stats
    assert leaves
    assert "global/age/count" in key_paths
    assert score == 2


def test_generated_validation_files_detects_agent_authored_checkers(tmp_path):
    record_dir = tmp_path / "record"
    record_dir.mkdir()
    write_workspace_delta(
        record_dir,
        {
            "changed_files": [
                ("validation/parity_check.py", "changed_files/validation/parity_check.py", "print('check')\n"),
                ("job.py", "changed_files/job.py", "print('job')\n"),
                ("tools/split_data.py", "changed_files/tools/split_data.py", "print('prep')\n"),
            ],
            "final_structure_files": [
                ("tools/validate_stats.py", "final_structure_files/tools/validate_stats.py", "print('check')\n"),
            ],
            "runtime_artifacts": [
                (
                    "workspace/server/simulate_job/statistics/image_statistics.json",
                    "runtime_artifacts/image_statistics.json",
                    '{"valid": true}',
                )
            ],
        },
    )

    generated = fedstats_checks.generated_validation_files(record_dir)

    assert generated == ["tools/validate_stats.py", "validation/parity_check.py"]


def test_parity_errors_split_per_site_and_global_mismatches():
    features = ["age"]
    axes = [
        {
            "label": "site-1",
            "site_tokens": ("site-1",),
            "dataset_tokens": (),
            "reference": {
                "count": 2,
                "features": {
                    "age": {
                        "count": 2,
                        "mean": 3.0,
                        "sum": 6.0,
                        "stddev": 1.414214,
                        "stddev_population": 1.0,
                    }
                },
            },
        },
        {
            "label": "Global",
            "site_tokens": ("global",),
            "dataset_tokens": (),
            "reference": {
                "count": 4,
                "features": {
                    "age": {
                        "count": 4,
                        "mean": 4.0,
                        "sum": 16.0,
                        "stddev": 2.581989,
                        "stddev_population": 2.236068,
                    }
                },
            },
        },
    ]
    leaves = [
        (("site-1", "age", "count"), 3.0),
        (("site-1", "age", "mean"), 3.0),
        (("site-1", "age", "sum"), 6.0),
        (("site-1", "age", "stddev"), 1.414214),
        (("global", "age", "count"), 4.0),
        (("global", "age", "mean"), 9.0),
        (("global", "age", "sum"), 16.0),
        (("global", "age", "stddev"), 2.581989),
    ]

    per_site_errors, global_errors = fedstats_checks.parity_errors(leaves, features, axes)

    assert len(per_site_errors) == 1
    assert "site-1/age/count" in per_site_errors[0]
    assert len(global_errors) == 1
    assert "Global/age/mean" in global_errors[0]


def test_parity_errors_accepts_stddev_from_matching_variance_for_small_sigma():
    features = ["tiny"]
    axes = [
        {
            "label": "Global",
            "site_tokens": ("global",),
            "dataset_tokens": (),
            "reference": {
                "count": 30,
                "features": {
                    "tiny": {
                        "count": 30,
                        "mean": 1.0,
                        "sum": 30.0,
                        "stddev": 0.05,
                        "stddev_population": 0.049,
                    }
                },
            },
        }
    ]
    var_value = 0.0028
    leaves = [
        (("global", "tiny", "count"), 30.0),
        (("global", "tiny", "mean"), 1.0),
        (("global", "tiny", "sum"), 30.0),
        (("global", "tiny", "var"), var_value),
        (("global", "tiny", "stddev"), round(var_value**0.5, 4)),
    ]

    per_site_errors, global_errors = fedstats_checks.parity_errors(leaves, features, axes)

    assert per_site_errors == []
    assert global_errors == []


def test_parity_errors_rejects_stddev_inconsistent_with_variance():
    features = ["tiny"]
    axes = [
        {
            "label": "Global",
            "site_tokens": ("global",),
            "dataset_tokens": (),
            "reference": {
                "count": 30,
                "features": {
                    "tiny": {
                        "count": 30,
                        "mean": 1.0,
                        "sum": 30.0,
                        "stddev": 0.05,
                        "stddev_population": 0.049,
                    }
                },
            },
        }
    ]
    leaves = [
        (("global", "tiny", "count"), 30.0),
        (("global", "tiny", "mean"), 1.0),
        (("global", "tiny", "sum"), 30.0),
        (("global", "tiny", "var"), 0.0028),
        (("global", "tiny", "stddev"), 0.05),
    ]

    per_site_errors, global_errors = fedstats_checks.parity_errors(leaves, features, axes)

    assert per_site_errors == []
    assert len(global_errors) == 1
    assert "Global/tiny/stddev" in global_errors[0]


def test_parity_errors_accepts_var_configured_per_site_and_global_pair():
    features = ["tiny"]
    axes = [
        {
            "label": "site-1",
            "site_tokens": ("site-1",),
            "dataset_tokens": (),
            "reference": {
                "count": 12,
                "features": {
                    "tiny": {
                        "count": 12,
                        "mean": 1.0,
                        "sum": 12.0,
                        "stddev": 0.05,
                        "stddev_population": 0.047871,
                    }
                },
            },
        },
        {
            "label": "Global",
            "site_tokens": ("global",),
            "dataset_tokens": (),
            "reference": {
                "count": 36,
                "features": {
                    "tiny": {
                        "count": 36,
                        "mean": 1.0,
                        "sum": 36.0,
                        "stddev": 0.05,
                        "stddev_population": 0.049301,
                    }
                },
            },
        },
    ]
    site_var = 0.0028
    global_var = 0.0027
    leaves = [
        (("site-1", "tiny", "count"), 12.0),
        (("site-1", "tiny", "mean"), 1.0),
        (("site-1", "tiny", "sum"), 12.0),
        (("site-1", "tiny", "var"), site_var),
        (("site-1", "tiny", "stddev"), round(site_var**0.5, 4)),
        (("global", "tiny", "count"), 36.0),
        (("global", "tiny", "mean"), 1.0),
        (("global", "tiny", "sum"), 36.0),
        (("global", "tiny", "var"), global_var),
        (("global", "tiny", "stddev"), round(global_var**0.5, 4)),
    ]

    per_site_errors, global_errors = fedstats_checks.parity_errors(leaves, features, axes)

    assert per_site_errors == []
    assert global_errors == []


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
