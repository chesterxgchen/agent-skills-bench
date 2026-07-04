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

"""Declarative evidence-capture spec (architecture §4.1 / §6b).

What Stage-3 capture should persist for an SDK is **data**, not behavior: the
spec is serialized into image metadata at build time and applied by *generic*
in-container capture code (no plugin code runs in the container). This is what
moves the NVFLARE-specific capture rules out of the generic Stage-3 helpers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

logger = logging.getLogger(__name__)

# Single owner of the EvidenceCaptureSpec payload format version. DISTINCT from
# the §4.3 profile block's ``capture_spec_version`` (profile_metadata.py): this
# version lives ON the spec payload itself. The in-container reader degrades on
# an unknown (future) major, mirroring ``profile_schema_supported``.
CAPTURE_SPEC_VERSION = 1


@dataclass(frozen=True)
class EvidenceCaptureSpec:
    # File basenames to flag as "structure files" in the workspace delta.
    structure_file_names: tuple[str, ...] = ()
    # Extra (label, absolute-path) runtime artifact sources to capture.
    runtime_sources: tuple[tuple[str, str], ...] = ()
    # Glob patterns (rglob, relative to the run workspace root) whose matches are
    # turned into runtime artifact sources at capture time. Robustly captures
    # artifacts whose exact path is not known ahead of time (e.g. simulator logs).
    artifact_globs: tuple[str, ...] = ()
    # ABSOLUTE glob patterns whose matched directories become runtime artifact
    # sources at capture time. Discovers runtime output roots OUTSIDE the run
    # workspace whose exact names are unpredictable by design — e.g. the private
    # per-user run directories the NVFLARE skills mandate
    # (``/tmp/nvflare-<uid>/run-<random>/``).
    runtime_source_globs: tuple[str, ...] = ()
    # Payload format version (see CAPTURE_SPEC_VERSION). Legacy payloads without
    # a version are treated as v1.
    version: int = CAPTURE_SPEC_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "structure_file_names": list(self.structure_file_names),
            "runtime_sources": [list(source) for source in self.runtime_sources],
            "artifact_globs": list(self.artifact_globs),
            "runtime_source_globs": list(self.runtime_source_globs),
        }

    @classmethod
    def from_payload(cls, payload: Any) -> "EvidenceCaptureSpec":
        if not isinstance(payload, Mapping):
            return cls()
        names = payload.get("structure_file_names") or []
        sources = payload.get("runtime_sources") or []
        globs = payload.get("artifact_globs") or []
        source_globs = payload.get("runtime_source_globs") or []
        return cls(
            structure_file_names=tuple(str(name) for name in names if isinstance(name, str)),
            runtime_sources=tuple(
                (str(source[0]), str(source[1]))
                for source in sources
                if isinstance(source, (list, tuple)) and len(source) == 2
            ),
            artifact_globs=tuple(str(pattern) for pattern in globs if isinstance(pattern, str)),
            runtime_source_globs=tuple(str(pattern) for pattern in source_globs if isinstance(pattern, str)),
            version=_payload_version(payload),
        )


def _payload_version(payload: Mapping[str, Any]) -> int:
    """Parse the payload ``version`` defensively; absent/unparseable -> v1 (legacy)."""

    raw = payload.get("version")
    if raw is None:
        return CAPTURE_SPEC_VERSION
    try:
        return int(str(raw).split(".", 1)[0])
    except (TypeError, ValueError):
        return CAPTURE_SPEC_VERSION


def resolve_capture_spec(sdk_name: str | None) -> EvidenceCaptureSpec:
    """Resolve the capture spec for an SDK by its declared name.

    Mirrors report-plugin resolution (§4.2): an absent name (a legacy tree
    produced before identity was stamped) falls back to NVFLARE for capture
    stability; a present-but-unknown SDK gets an empty spec (no product-specific
    capture). The NVFLARE spec data lives in the NVFLARE plugin, not here.
    """

    from .nvflare.capture import NVFLARE_CAPTURE_SPEC

    if not sdk_name or sdk_name == "nvflare":
        return NVFLARE_CAPTURE_SPEC
    return EvidenceCaptureSpec()


def capture_spec_from_metadata(metadata: Mapping[str, Any]) -> EvidenceCaptureSpec:
    """Resolve the capture spec a result/image carries (generic, in-container).

    Prefers the serialized ``capture_spec`` block written at build time; falls
    back to resolving by ``sdk_name`` for legacy images that predate it.
    """

    if not isinstance(metadata, Mapping):
        return resolve_capture_spec(None)
    payload = metadata.get("capture_spec")
    if payload is not None:
        if isinstance(payload, Mapping) and _payload_version(payload) > CAPTURE_SPEC_VERSION:
            # A future spec major may carry rules this generic reader cannot apply
            # safely. Degrade to a structure-only minimal spec (mirrors
            # profile_metadata.profile_schema_supported's unknown-major degrade).
            logger.warning(
                "capture_spec payload version %s is newer than supported %s; "
                "degrading to a minimal structure-only spec.",
                _payload_version(payload),
                CAPTURE_SPEC_VERSION,
            )
            return _minimal_spec(payload)
        return EvidenceCaptureSpec.from_payload(payload)
    sdk_name = metadata.get("sdk_name")
    return resolve_capture_spec(sdk_name if isinstance(sdk_name, str) else None)


def _minimal_spec(payload: Mapping[str, Any]) -> EvidenceCaptureSpec:
    """Safe fallback for an unsupported spec version: structure files only.

    Structure-file names are plain basenames with no behavior, so they are safe
    to carry forward; runtime sources and artifact globs (which drive what gets
    read from the filesystem) are dropped under degrade.
    """

    names = payload.get("structure_file_names") or []
    return EvidenceCaptureSpec(
        structure_file_names=tuple(str(name) for name in names if isinstance(name, str)),
    )
