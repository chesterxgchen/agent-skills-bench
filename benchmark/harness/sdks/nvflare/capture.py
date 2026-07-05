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
    # The FL run/export folder location is NOT the harness's to assume: a skill
    # runs the simulation in a private temp root (observed:
    # `/tmp/nvflare-<job>.<rand>/workspace/<job>/server/simulate_job/...`), and a
    # prompt can direct the run/export anywhere else. Any hardcoded absolute glob
    # (`/tmp/nvflare-*`, `.../run-*`) bakes in a path prefix that breaks the moment
    # the location changes. Instead, find the run folder by its OUTPUT STRUCTURE:
    # capture searches the system temp bases (and any designated output dir) for
    # these markers and captures whichever run root actually holds them. The
    # metrics_summary.json marker anchors the FL result scalar
    # (best_metrics[0].value); the config_fed_* markers still locate a run root
    # that produced a job config but no metrics (e.g. an early failure), so its
    # logs are captured for RCA.
    runtime_output_markers=(
        "**/simulate_job/metrics/metrics_summary.json",
        "**/simulate_job/metrics/round_metrics.jsonl",
        # Every simulator run materializes simulate_job/meta.json, even when the
        # job wrote no metrics files — this anchors run-root discovery (and the
        # in-workspace runtime/source split) for metrics-less runs.
        "**/simulate_job/meta.json",
        "**/config_fed_server.json",
        "**/config_fed_client.json",
    ),
)
