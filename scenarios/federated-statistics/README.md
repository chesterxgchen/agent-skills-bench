# Federated-statistics benchmark scenarios

Two scenarios over the local `~/projects/flare_test` site-split patient
datasets, with the canonical fed-stats prompt:

- `fedstats-header.yaml` — `tabular_with_header/` (CSV header row present)
- `fedstats-noheader.yaml` — `tabular_no_header/` (column names only in the
  dataset README; adds the `data-format: no_header` criteria overlay and the
  names-honored acceptance check)

**Local dataset dependency (intentional):** the job paths point at
`~/projects/flare_test/...` on the operator's machine — these scenarios are a
local benchmark fixture, not portable definitions. The committed ground truth
records each site CSV's sha256; `fedstats_checks.py` fails the
`dataset_unchanged` gate loudly if the dataset drifts from the constants.

## Files

- `ground_truth.<dataset>.json` — committed constants (counts, sum/mean/stddev
  per site + Global, both stddev conventions, categorical lists, leakage
  sentinels, CSV hashes). Regenerate only when the dataset intentionally
  changes: `python generate_ground_truth.py <dataset_dir> <out.json>
  [--no-header]`.
- `fedstats_checks.py` — deterministic hard gates run host-side by the
  harness's `acceptance_checks` hook: real aggregated statistics artifact,
  completeness (features x sites x stats), exact counts, value accuracy at
  persisted precision, categorical exclusion, privacy defaults not weakened,
  no raw-row leakage in the final report, README names honored (noheader).
  min/max are presence-only (noised by the default privacy filters);
  quantiles/histograms are not judged numerically.

Acceptance scripts are trusted host code (same trust as the scenario file);
the compiler requires them to live inside this directory.

## What scores what

- Hard pass/fail: `result_artifact` (statistics JSON under the simulator
  workspace) + the checks above, via `critical_quality_checks_failed`.
- Quality shading: the `federated-statistics` task criteria in
  `benchmark/config/evaluation/nvflare/tasks/federated-statistics/`, judged by
  the code-eval agent and rendered as report rows.
