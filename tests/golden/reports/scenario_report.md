# Scenario Report: ames_fedavg

Result root: `<RESULT_ROOT>`
Status: `failed`
Agent invocation: `replayed`
Host OS: `Ubuntu 24.04 LTS`
Runs: 2/2 completed

Replay: `true`
Replayed at: `2026-06-13T21:00:00Z`
This report was regenerated from captured artifacts; no agent or Docker run was executed.

## Run Identity

| Run ID | Label | Agent | Model | Model source | Mode | Host OS | Skills available | Skills inspected | Skills applied/used | Shared refs read |
|---|---|---|---|---|---|---|---|---|---|---|
| run_00001 | without_skills | codex | default | scenario | without_skills | Ubuntu 24.04 LTS | not enabled | none | none | none |
| run_00002 | with_skills | codex | default | scenario | with_skills | Ubuntu 24.04 LTS | nvflare-convert-lightning; nvflare-convert-pytorch; nvflare-diagnose-job; nvflare-orient | none | nvflare-convert-pytorch | none |

## Aggregate Results

| Label | Runs | Quality pass | Median agent seconds | Median tokens |
|---|---:|---:|---:|---:|
| with_skills | 1 | 0 | NA | 15000.0 |
| without_skills | 1 | 0 | NA | 12000.0 |

## Winner Policy

`median_agent_elapsed_seconds_then_tokens_with_quality_gate`

No winner selected because no compared label passed the quality gate with timing data.
