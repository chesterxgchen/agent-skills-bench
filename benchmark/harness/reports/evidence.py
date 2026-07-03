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

"""Contract B — the typed evidence spine (architecture §5).

``RunEvidence``/``ComparisonEvidence`` are immutable, captured-only views over
what Stage 3 wrote to disk. Evidence is loaded by the neutral ``reports._loader``
(which this module imports) — Contract B does NOT depend on the report product
(``benchmark_insights``); the dependency runs the other way (Inversion 1). The
typed fields carry the normalized per-run data; ``raw`` keeps the source bundle
accessible during the migration off the per-run dict.

Plugin-derived interpretation is a SEPARATE sidecar (``PluginEvidence``) and is
NEVER stored here: Contract B is captured evidence only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ._loader import collect_benchmark_runs

# Contract B evidence schema version — its own owner, distinct from Contract A's
# ``profile_metadata.PROFILE_METADATA_SCHEMA_VERSION`` and the per-artifact
# ``schema_version`` fields (records / artifacts / runner / scenario reports). Its
# read-time unknown-major degrade lands when ``ComparisonEvidence`` drives the
# render path (B3); today there is exactly one schema.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunEvidence:
    """A single captured run (architecture §5).

    Generic fields map 1:1 from the per-run bundle produced by
    ``reports._loader.collect_benchmark_runs``. ``raw`` keeps that bundle
    accessible during the migration off the per-run dict.
    """

    mode: str
    label: str
    available: bool
    agent: str | None
    agent_model: str | None
    model_source: str | None
    mode_dir: Path | None
    summary: Mapping[str, Any]
    record: Mapping[str, Any]
    usage: Mapping[str, Any]
    activity: Mapping[str, Any]
    workspace_delta: Mapping[str, Any]
    skills_list: Mapping[str, Any]
    runtime_image: Mapping[str, Any]
    container_exit: Mapping[str, Any]
    validation_metric: Mapping[str, Any] | None
    # The verbatim agent input prompt captured per run (prompt.txt +
    # prompt_metadata.json) — generic across SDKs.
    prompt_text: str
    prompt_metadata: Mapping[str, Any]
    # Structured captured-text artifacts (§5): the agent/console text Stage 3 wrote.
    agent_last_message: str
    agent_stderr: str
    agent_events_text: str
    console_text: str
    # The original per-run dict. ``raw`` is the BRIDGE to the dict-based neutral
    # substrate (``_events``/``_runs``/``_logic``): render helpers read the typed
    # fields above for direct data and pass ``raw`` only to those substrate
    # functions (architecture §5 "genuinely-opaque passthrough").
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class ComparisonEvidence:
    """All runs in a comparison, keyed by mode (architecture §5)."""

    schema_version: int
    runs: dict[str, RunEvidence]
    modes: list[str]
    # The captured §4.3 profile/identity block (Contract A), resolved once per root.
    sdk_metadata: Mapping[str, Any]


def _run_evidence_from_bundle(bundle: Mapping[str, Any]) -> RunEvidence:
    mode_dir = bundle.get("mode_dir")
    return RunEvidence(
        mode=str(bundle.get("mode") or ""),
        label=str(bundle.get("label") or ""),
        available=bool(bundle.get("available")),
        agent=bundle.get("agent"),
        agent_model=bundle.get("agent_model"),
        model_source=bundle.get("model_source"),
        mode_dir=mode_dir if isinstance(mode_dir, Path) else (Path(mode_dir) if mode_dir else None),
        summary=bundle.get("run") or {},
        record=bundle.get("record") or {},
        usage=bundle.get("usage") or {},
        activity=bundle.get("activity") or {},
        workspace_delta=bundle.get("workspace_delta") or {},
        skills_list=bundle.get("skills_list") or {},
        runtime_image=bundle.get("runtime_image") or {},
        container_exit=bundle.get("container_exit") or {},
        validation_metric=bundle.get("validation_metric"),
        prompt_text=str(bundle.get("prompt_text") or ""),
        prompt_metadata=bundle.get("prompt_metadata") or {},
        agent_last_message=str(bundle.get("agent_last_message") or ""),
        agent_stderr=str(bundle.get("agent_stderr") or ""),
        agent_events_text=str(bundle.get("agent_events_text") or ""),
        console_text=str(bundle.get("console_text") or ""),
        raw=bundle,
    )


def build_comparison_evidence(result_root: str | Path) -> ComparisonEvidence:
    """Adapt ``collect_benchmark_runs`` output into the typed spine.

    Behavior-preserving adapter only: it does not re-read or re-interpret
    anything beyond what ``collect_benchmark_runs`` already produced. Not wired
    into the rendering path (step 3 introduces + unit-tests it only).
    """

    from .. import profile_metadata

    bundles = collect_benchmark_runs(Path(result_root))
    runs = {mode: _run_evidence_from_bundle(bundle) for mode, bundle in bundles.items()}
    return ComparisonEvidence(
        schema_version=SCHEMA_VERSION,
        runs=runs,
        modes=list(bundles.keys()),
        sdk_metadata=profile_metadata.read_profile_metadata_block(result_root),
    )
