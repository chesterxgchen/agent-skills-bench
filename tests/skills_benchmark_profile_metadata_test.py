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

import json
from pathlib import Path

from benchmark.harness import profile_metadata


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_read_report_plugin_id_from_root_descriptor(tmp_path):
    _write_json(
        tmp_path / profile_metadata.ROOT_DESCRIPTOR_FILENAME,
        {"report_plugin_id": "nvflare"},
    )

    assert profile_metadata.read_report_plugin_id(tmp_path) == "nvflare"


def test_read_report_plugin_id_falls_back_to_mode_dir(tmp_path):
    mode_dir = tmp_path / "records" / "mode=with_skills"
    _write_json(
        mode_dir / profile_metadata.MODE_METADATA_FILENAME,
        {"report_plugin_id": "custom-profile"},
    )

    assert profile_metadata.read_report_plugin_id(tmp_path) == "custom-profile"


def test_read_report_plugin_id_prefers_root_over_mode_dir(tmp_path):
    _write_json(
        tmp_path / profile_metadata.ROOT_DESCRIPTOR_FILENAME,
        {"report_plugin_id": "root-id"},
    )
    mode_dir = tmp_path / "records" / "mode=with_skills"
    _write_json(
        mode_dir / profile_metadata.MODE_METADATA_FILENAME,
        {"report_plugin_id": "mode-id"},
    )

    assert profile_metadata.read_report_plugin_id(tmp_path) == "root-id"


def test_read_report_plugin_id_returns_none_when_absent(tmp_path):
    assert profile_metadata.read_report_plugin_id(tmp_path) is None


def test_read_report_plugin_id_returns_none_for_legacy_mode_metadata(tmp_path):
    mode_dir = tmp_path / "records" / "mode=with_skills"
    _write_json(
        mode_dir / profile_metadata.MODE_METADATA_FILENAME,
        {"sdk_name": "nvflare"},  # legacy metadata: no report_plugin_id
    )

    assert profile_metadata.read_report_plugin_id(tmp_path) is None


def test_write_root_descriptor_lifts_only_block_fields(tmp_path):
    mode_metadata = {
        "schema_version": "1",
        "sdk_name": "nvflare",
        "benchmark_profile_id": "nvflare",
        "report_plugin_id": "nvflare",
        "capture_spec_version": "1",
        "filename": "nvflare-1.0-py3-none-any.whl",  # not part of the §4.3 block
        "sha256": "deadbeef",
    }

    assert profile_metadata.write_root_descriptor(tmp_path, mode_metadata) is True

    written = json.loads((tmp_path / profile_metadata.ROOT_DESCRIPTOR_FILENAME).read_text(encoding="utf-8"))
    assert written == {
        "schema_version": "1",
        "sdk_name": "nvflare",
        "benchmark_profile_id": "nvflare",
        "report_plugin_id": "nvflare",
        "capture_spec_version": "1",
    }


def test_write_root_descriptor_noops_when_no_block_fields(tmp_path):
    # Metadata carrying none of the §4.3 block keys (e.g. only build-mechanics
    # fields) leaves the result root untouched.
    assert profile_metadata.write_root_descriptor(tmp_path, {"filename": "x.whl", "sha256": "ab"}) is False
    assert not (tmp_path / profile_metadata.ROOT_DESCRIPTOR_FILENAME).exists()


def test_write_root_descriptor_untrusted_source_carries_no_criteria_and_never_overwrites(tmp_path):
    # Host-side (image-baked) metadata anchors the descriptor, criteria included.
    trusted = {"sdk_name": "nvflare", "evaluation_criteria": {"entrypoint": ".", "sha256": "aa"}}
    assert profile_metadata.write_root_descriptor(tmp_path, trusted) is True
    assert profile_metadata.read_evaluation_criteria(tmp_path)["sha256"] == "aa"

    # Mount-resident metadata is container-writable: its criteria block must
    # not enter the descriptor, and it must not replace the trusted anchor.
    forged = {"sdk_name": "nvflare", "evaluation_criteria": {"entrypoint": ".", "sha256": "ff"}}
    assert profile_metadata.write_root_descriptor(tmp_path, forged, include_criteria=False, overwrite=False) is False
    assert profile_metadata.read_evaluation_criteria(tmp_path)["sha256"] == "aa"


def test_write_root_descriptor_mount_fallback_lifts_identity_without_criteria(tmp_path):
    # Legacy fallback (no host-anchored descriptor yet): the identity block is
    # lifted, but the mount copy's criteria stays out, so the report treats the
    # rules copy as unverifiable instead of blessing a forgeable hash.
    mount_metadata = {
        "sdk_name": "nvflare",
        "report_plugin_id": "nvflare",
        "evaluation_criteria": {"entrypoint": ".", "sha256": "ff"},
    }
    assert (
        profile_metadata.write_root_descriptor(tmp_path, mount_metadata, include_criteria=False, overwrite=False)
        is True
    )
    assert profile_metadata.read_report_plugin_id(tmp_path) == "nvflare"
    assert profile_metadata.read_evaluation_criteria(tmp_path) == {}


def test_clear_root_descriptor_removes_stale_anchor(tmp_path):
    # A reused result root carries the previous run's descriptor; a fresh run
    # must clear it so a failed image anchor degrades to unverifiable instead
    # of inheriting the stale criteria, and the mount fallback's overwrite
    # guard does not mistake the stale file for a current-run anchor.
    _write_json(
        tmp_path / profile_metadata.ROOT_DESCRIPTOR_FILENAME,
        {"sdk_name": "nvflare", "evaluation_criteria": {"entrypoint": ".", "sha256": "aa"}},
    )
    profile_metadata.clear_root_descriptor(tmp_path)
    assert not (tmp_path / profile_metadata.ROOT_DESCRIPTOR_FILENAME).exists()
    assert profile_metadata.read_evaluation_criteria(tmp_path) == {}
    # No descriptor present: clearing is a no-op, not an error.
    profile_metadata.clear_root_descriptor(tmp_path)


def test_write_root_descriptor_for_pre_step2_metadata_yields_no_plugin_id(tmp_path):
    # A tree built before step 2 has sdk_name but no report_plugin_id. The lift
    # still surfaces what it has, and the reader returns None (legacy fallback).
    assert profile_metadata.write_root_descriptor(tmp_path, {"sdk_name": "nvflare"}) is True
    assert profile_metadata.read_report_plugin_id(tmp_path) is None


def test_profile_schema_version_reader_tolerates_str_and_int(tmp_path):
    # Absent -> None (legacy tree, treated as compatible).
    assert profile_metadata.read_profile_schema_version(tmp_path) is None
    assert profile_metadata.profile_schema_supported(tmp_path) is True
    # Legacy string form and current int form both parse to the same major.
    _write_json(tmp_path / profile_metadata.ROOT_DESCRIPTOR_FILENAME, {"schema_version": "1"})
    assert profile_metadata.read_profile_schema_version(tmp_path) == 1
    _write_json(tmp_path / profile_metadata.ROOT_DESCRIPTOR_FILENAME, {"schema_version": 1})
    assert profile_metadata.read_profile_schema_version(tmp_path) == 1
    assert profile_metadata.profile_schema_supported(tmp_path) is True


def test_unknown_major_profile_schema_degrades_to_null_plugin(tmp_path):
    from benchmark.harness.sdks.report_plugin import NullReportPlugin
    from benchmark.harness.sdks.report_registry import resolve_from_result_root

    # A future major with a real report_plugin_id must NOT be trusted to pick a plugin.
    _write_json(
        tmp_path / profile_metadata.ROOT_DESCRIPTOR_FILENAME,
        {"schema_version": 2, "report_plugin_id": "nvflare"},
    )
    assert profile_metadata.profile_schema_supported(tmp_path) is False
    assert isinstance(resolve_from_result_root(tmp_path), NullReportPlugin)
