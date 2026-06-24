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

"""Stage 4 — resolve a captured ``report_plugin_id`` to a ``ReportPlugin``.

Resolution rules (architecture §4.2/§6; Inversion 3 — absence resolves to null):
- ``'nvflare'`` -> ``NvflareReportPlugin``.
- ABSENT/``None`` id -> ``NullReportPlugin``. NVFLARE is selected only by explicit
  captured identity; a tree with no profile descriptor is not assumed to be
  NVFLARE. (Real NVFLARE runs stamp ``report_plugin_id='nvflare'`` at build time.)
- PRESENT-but-unknown id -> ``NullReportPlugin`` (warn, never fail).

No live SDK or adapter object is touched at report time — resolution is by
captured id only.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .. import profile_metadata
from .nvflare.plugin import NvflareReportPlugin
from .report_plugin import NullReportPlugin, ReportPlugin

logger = logging.getLogger(__name__)

# Keyed by captured report_plugin_id. Keep thin until a second SDK lands.
_REGISTRY: dict[str, type[ReportPlugin]] = {
    "nvflare": NvflareReportPlugin,
}


def resolve_report_plugin(report_plugin_id: str | None) -> ReportPlugin:
    """Resolve a captured id to a plugin instance (§4.2)."""

    if report_plugin_id is None:
        # Absence resolves to null (Inversion 3): NVFLARE is chosen only by
        # explicit captured identity, never assumed for an unidentified tree.
        return NullReportPlugin()
    plugin_cls = _REGISTRY.get(report_plugin_id)
    if plugin_cls is None:
        logger.warning(
            "Unknown report_plugin_id %r; falling back to the null report plugin.",
            report_plugin_id,
        )
        return NullReportPlugin()
    return plugin_cls()


def resolve_from_result_root(result_root: str | Path) -> ReportPlugin:
    """Resolve the plugin for a finalized result root via captured identity.

    Degrades to the null plugin on an unknown-major profile schema (§5): an
    incompatible identity block is not trusted to pick an SDK plugin.
    """

    if not profile_metadata.profile_schema_supported(result_root):
        logger.warning(
            "Unsupported profile schema (major %r > %r); degrading to the null report plugin.",
            profile_metadata.read_profile_schema_version(result_root),
            profile_metadata.PROFILE_METADATA_SCHEMA_VERSION,
        )
        return NullReportPlugin()
    report_plugin_id = profile_metadata.read_report_plugin_id(result_root)
    return resolve_report_plugin(report_plugin_id)
