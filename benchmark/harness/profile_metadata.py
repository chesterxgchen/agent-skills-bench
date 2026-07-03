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

"""Profile/identity metadata helpers (Contract A, §4.2/§4.3).

Capture + read path only: stamp the §4.3 identity block at build time, lift a
small root-level descriptor when a run finalizes, and read the captured
``report_plugin_id`` back. No plugin resolution lives here — that is Stage 4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .common import load_json, write_json

# Root-level descriptor (§4.3). The durable concept is a profile/evidence
# descriptor; ``sdk_wheel_metadata.json`` is the v1 in-mode-dir home and stays
# the fallback. Readers should treat the BLOCK, not the filename, as the contract.
ROOT_DESCRIPTOR_FILENAME = "benchmark_profile_metadata.json"
MODE_METADATA_FILENAME = "sdk_wheel_metadata.json"

# The §4.3 identity/version block keys (single source of truth for both the
# build-time builder below and the read-time lift).
PROFILE_METADATA_KEYS = (
    "schema_version",
    "sdk_name",
    "benchmark_profile_id",
    "report_plugin_id",
    "capture_spec_version",
)

# The evaluation-criteria identity (sha256 of the build-staged rules tree, plus
# provenance) is lifted alongside the §4.3 block. The root descriptor is
# host-written and lives OUTSIDE the container-writable result mount, so it is
# the trust anchor the report verifies the mount-resident rules copy against —
# the same role the descriptor already plays for report_plugin_id.
EVALUATION_CRITERIA_KEY = "evaluation_criteria"

# Single owner of the §4.3 block's format version (Contract A). This is DISTINCT
# from the per-artifact ``schema_version`` fields elsewhere (records / artifacts /
# runner / scenario reports) and from Contract B's ``evidence.SCHEMA_VERSION`` —
# each versions its own format and has its own owner. The report degrades on an
# unknown major (see ``profile_schema_supported``).
PROFILE_METADATA_SCHEMA_VERSION = 1


def build_profile_metadata_block(sdk: Any) -> dict[str, Any]:
    """Build the §4.3 identity block for an SDK profile (single source of truth).

    ``sdk.name`` is the profile's declared name for both built-in adapters and
    custom ``--sdk-profile`` YAMLs (never the file path), so a captured result
    stays resolvable elsewhere. ``benchmark_profile_id`` and ``report_plugin_id``
    are distinct keys (they may diverge once a report_plugin YAML field lands) but
    both default to ``sdk.name`` today.
    """

    return {
        "schema_version": PROFILE_METADATA_SCHEMA_VERSION,
        "sdk_name": sdk.name,
        "benchmark_profile_id": sdk.name,
        "report_plugin_id": sdk.name,
        "capture_spec_version": "1",
    }


def write_root_descriptor(
    result_root: Path,
    metadata: Mapping[str, Any],
    *,
    include_criteria: bool = True,
    overwrite: bool = True,
) -> bool:
    """Lift the §4.3 block to a root-level descriptor (§4.2 step 2).

    No-ops (returns ``False``) when the source metadata carries no block fields
    so legacy trees stay unchanged. Returns ``True`` when a descriptor is written.

    The descriptor is the trust anchor the report verifies the mount-resident
    rules copy against, so what may flow into it depends on where ``metadata``
    came from. Metadata read from HOST-SIDE state (the image baked at build
    time) may carry the evaluation-criteria identity and replace an existing
    descriptor. Metadata read from the container-WRITABLE result mount must
    pass ``include_criteria=False`` (a run that rewrites ``evaluation_rules/``
    can rewrite that copy to bless a tampered rules hash) and
    ``overwrite=False`` (it must not clobber a descriptor already lifted from
    trusted state); it remains only a legacy fallback for the identity block.
    """

    if not overwrite and (result_root / ROOT_DESCRIPTOR_FILENAME).is_file():
        return False
    block = {key: metadata[key] for key in PROFILE_METADATA_KEYS if key in metadata}
    if include_criteria:
        criteria = metadata.get(EVALUATION_CRITERIA_KEY)
        if isinstance(criteria, Mapping) and criteria:
            block[EVALUATION_CRITERIA_KEY] = dict(criteria)
    if not block:
        return False
    write_json(result_root / ROOT_DESCRIPTOR_FILENAME, block)
    return True


def read_evaluation_criteria(result_root: str | Path) -> dict[str, Any]:
    """Return the host-anchored evaluation-criteria identity block, or ``{}``.

    Only the root-level descriptor is consulted: the in-mode-dir metadata sits
    in the container-writable mount and cannot anchor trust.
    """

    descriptor = load_json(Path(result_root) / ROOT_DESCRIPTOR_FILENAME, {}) or {}
    criteria = descriptor.get(EVALUATION_CRITERIA_KEY) if isinstance(descriptor, dict) else None
    return dict(criteria) if isinstance(criteria, dict) else {}


def read_report_plugin_id(result_root: str | Path) -> str | None:
    """Read ``report_plugin_id`` from a finalized result root (§4.2 step 3).

    Resolution order: the root-level descriptor first, then any mode dir's
    ``sdk_wheel_metadata.json``. Returns ``None`` when absent (legacy tree).
    This ONLY reads the id; it performs no plugin resolution.
    """

    root = Path(result_root)
    descriptor = load_json(root / ROOT_DESCRIPTOR_FILENAME, {}) or {}
    if isinstance(descriptor, dict):
        plugin_id = descriptor.get("report_plugin_id")
        if isinstance(plugin_id, str) and plugin_id:
            return plugin_id

    for metadata_path in sorted(root.rglob(MODE_METADATA_FILENAME)):
        metadata = load_json(metadata_path, {}) or {}
        if isinstance(metadata, dict):
            plugin_id = metadata.get("report_plugin_id")
            if isinstance(plugin_id, str) and plugin_id:
                return plugin_id
    return None


def read_profile_metadata_block(result_root: str | Path) -> dict[str, Any]:
    """Return the captured §4.3 identity block (root descriptor, mode-dir fallback).

    Empty dict for a legacy tree with no block.
    """

    root = Path(result_root)
    descriptor = load_json(root / ROOT_DESCRIPTOR_FILENAME, {}) or {}
    if isinstance(descriptor, dict) and any(key in descriptor for key in PROFILE_METADATA_KEYS):
        return {key: descriptor[key] for key in PROFILE_METADATA_KEYS if key in descriptor}
    for metadata_path in sorted(root.rglob(MODE_METADATA_FILENAME)):
        metadata = load_json(metadata_path, {}) or {}
        if isinstance(metadata, dict) and any(key in metadata for key in PROFILE_METADATA_KEYS):
            return {key: metadata[key] for key in PROFILE_METADATA_KEYS if key in metadata}
    return {}


def read_profile_schema_version(result_root: str | Path) -> int | None:
    """Read the §4.3 block's major schema version, or ``None`` when absent (legacy).

    Tolerant of the legacy string form (``"1"``) and the current int form (``1``);
    returns ``None`` for an absent or unparseable value so legacy trees are
    treated as compatible.
    """

    root = Path(result_root)
    descriptor = load_json(root / ROOT_DESCRIPTOR_FILENAME, {}) or {}
    raw = descriptor.get("schema_version") if isinstance(descriptor, dict) else None
    if raw is None:
        for metadata_path in sorted(root.rglob(MODE_METADATA_FILENAME)):
            metadata = load_json(metadata_path, {}) or {}
            if isinstance(metadata, dict) and metadata.get("schema_version") is not None:
                raw = metadata.get("schema_version")
                break
    if raw is None:
        return None
    try:
        return int(str(raw).split(".", 1)[0])
    except (TypeError, ValueError):
        return None


def profile_schema_supported(result_root: str | Path) -> bool:
    """Whether the captured profile block's schema major is understood (§5 degrade).

    Absent version -> supported (legacy tree). A known major -> supported. An
    unknown (future) major -> NOT supported; the caller should degrade rather than
    trust an incompatible identity block.
    """

    major = read_profile_schema_version(result_root)
    return major is None or major <= PROFILE_METADATA_SCHEMA_VERSION
