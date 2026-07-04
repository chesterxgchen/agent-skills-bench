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

"""NVFLARE evidence-capture spec (the data Stage-3 capture persists).

Owns the NVFLARE-specific capture rules that used to be hardcoded in the generic
Stage-3 code (``artifacts.STRUCTURE_FILE_NAMES`` and
``agent_run.NVFLARE_RUNTIME_WORKSPACES_ROOT``). Hardcoded here for now — no YAML
config until a second SDK needs different rules (architecture §10, no premature
config).
"""

from __future__ import annotations

from ..capture_spec import EvidenceCaptureSpec

NVFLARE_CAPTURE_SPEC = EvidenceCaptureSpec(
    structure_file_names=("client.py", "model.py", "job.py", "train.py", "prepare_data.py", "download_data.py"),
    runtime_sources=(("runtime_workspaces", "/tmp/nvflare/workspaces"),),
    # FL-specific evidence whose exact in-workspace path varies by job/run:
    # the simulator/server/client console logs and the federated job configs.
    # Resolved (rglob) against the run workspace root at capture time.
    artifact_globs=(
        "**/log.txt",
        "**/*.log",
        "**/config_fed_*.json",
    ),
    # The NVFLARE skills' runtime-output-guidance mandates a private per-user run
    # root (`/tmp/nvflare-<uid>/run-<random>/`) for simulation workspaces, exported
    # jobs, and results — an unpredictable path by design, so it cannot be a fixed
    # runtime_sources entry. Capture each matched run root (its run-manifest.json,
    # workspace metrics, logs, and exported job config) as runtime evidence;
    # without this, skill-compliant runs surface zero runtime artifacts and their
    # FL result metrics cannot be graded.
    runtime_source_globs=("/tmp/nvflare-*/run-*",),
)
