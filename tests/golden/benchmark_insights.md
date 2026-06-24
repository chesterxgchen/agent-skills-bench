# Agent Benchmark Insights

Result root: `<RESULT_ROOT>`

## Executive Summary

| Signal | Value |
|---|---|
| Status | No skills baseline: passed; With skills: passed |
| Agent/model | No skills baseline: agent=codex, model=default; With skills: agent=codex, model=default |
| Job execution | No skills baseline: completed (simulation completed — FL workflow reached Finished state); With skills: completed (simulation completed — FL workflow reached Finished state) |
| FL algorithm/workflow | No skills baseline: FedAvg (3 rounds); With skills: FedAvg (3 rounds) |
| FL result quality gate | No skills baseline: pass: scalar FL result metric available; With skills: pass: scalar FL result metric available |
| Missing/partial result metrics | none |
| Source input protection | No skills baseline: not captured; With skills: not captured |
| Captured generated artifacts | No skills baseline: 3 changed/generated files, 1 runtime artifacts; With skills: 3 changed/generated files, 1 runtime artifacts |

## Status

| Run | Status | Analysis |
|---|---|---|
| No skills baseline | passed | No failure detected. |
| With skills | passed | No failure detected. |

## Run Identity

| Run | Agent | Model | Model source | Mode |
|---|---|---|---|---|
| No skills baseline | codex | default | scenario | without_skills |
| With skills | codex | default | scenario | with_skills |

## Job Run Status

This section tracks whether the generated NVFLARE job or simulator actually ran. Agent/container exit code 0 only means the agent process finished; it does not prove the generated job executed.

| Run | Job run status | Evidence | Action |
|---|---|---|---|
| No skills baseline | completed | simulation completed — FL workflow reached Finished state | Use job logs and reported metrics for quality comparison. |
| With skills | completed | simulation completed — FL workflow reached Finished state | Use job logs and reported metrics for quality comparison. |

## FL Algorithm / Workflow

This section reports the FL workflow captured in generated/runtime NVFLARE server config. It is derived from artifacts such as `config_fed_server.json`; agent final-message text is used only as a fallback.

| Run | Algorithm/workflow | Recipe | Rounds | Evidence |
|---|---|---|---:|---|
| No skills baseline | FedAvg | fedavg-pt | 3 | config_fed_server.json: nvflare.app_common.workflows.fedavg.FedAvg; recipe fedavg-pt |
| With skills | FedAvg | fedavg-pt | 3 | config_fed_server.json: nvflare.app_common.workflows.fedavg.FedAvg; recipe fedavg-pt |

## Failure Analysis

### No skills baseline

- Job run status: completed — simulation completed — FL workflow reached Finished state
- Outcome: passed. AUROC 0.7421 (aggregated best validation metric).
- Dependency reference: `requirements.txt` provenance: not found in captured input or workspace manifests.

### With skills

- Job run status: completed — simulation completed — FL workflow reached Finished state
- Outcome: passed. AUROC 0.7689 (aggregated best validation metric).
- Dependency reference: `requirements-train.txt` provenance: not found in captured input or workspace manifests.

## Metrics

<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="728" viewBox="0 0 1180 728">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="32" y="35" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">Run comparison</text>
<text x="32" y="58" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">Metrics are mode-local. Missing scalar results are shown as NA instead of drawing a numeric bar.</text>
<text x="32.0" y="104.0" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#111827">Total time seconds</text>
<line x1="32.0" y1="326" x2="293.0" y2="326" stroke="#d1d5db" stroke-width="1"/>
<line x1="32.0" y1="181" x2="32.0" y2="326" stroke="#d1d5db" stroke-width="1"/>
<rect x="112.5" y="217.2" width="38.0" height="108.8" fill="#16a34a" rx="3"/>
<text x="131.5" y="210.2" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">180</text>
<text x="131.5" y="345" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">No skills</text>
<rect x="174.5" y="181.0" width="38.0" height="145.0" fill="#2563eb" rx="3"/>
<text x="193.5" y="174.0" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">240</text>
<text x="193.5" y="345" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">With skills</text>
<text x="317.0" y="104.0" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#111827">Runtime seconds</text>
<line x1="317.0" y1="326" x2="578.0" y2="326" stroke="#d1d5db" stroke-width="1"/>
<line x1="317.0" y1="181" x2="317.0" y2="326" stroke="#d1d5db" stroke-width="1"/>
<rect x="397.5" y="205.2" width="38.0" height="120.8" fill="#16a34a" rx="3"/>
<text x="416.5" y="198.2" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">150</text>
<text x="416.5" y="345" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">No skills</text>
<rect x="459.5" y="181.0" width="38.0" height="145.0" fill="#2563eb" rx="3"/>
<text x="478.5" y="174.0" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">180</text>
<text x="478.5" y="345" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">With skills</text>
<text x="602.0" y="104.0" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#111827">Dependency install</text>
<line x1="602.0" y1="326" x2="863.0" y2="326" stroke="#d1d5db" stroke-width="1"/>
<line x1="602.0" y1="181" x2="602.0" y2="326" stroke="#d1d5db" stroke-width="1"/>
<rect x="682.5" y="253.5" width="38.0" height="72.5" fill="#16a34a" rx="3"/>
<text x="701.5" y="246.5" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">30</text>
<text x="701.5" y="345" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">No skills</text>
<rect x="744.5" y="181.0" width="38.0" height="145.0" fill="#2563eb" rx="3"/>
<text x="763.5" y="174.0" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">60</text>
<text x="763.5" y="345" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">With skills</text>
<text x="887.0" y="104.0" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#111827">Total tokens</text>
<line x1="887.0" y1="326" x2="1148.0" y2="326" stroke="#d1d5db" stroke-width="1"/>
<line x1="887.0" y1="181" x2="887.0" y2="326" stroke="#d1d5db" stroke-width="1"/>
<rect x="967.5" y="210.0" width="38.0" height="116.0" fill="#16a34a" rx="3"/>
<text x="986.5" y="203.0" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">12.0k</text>
<text x="986.5" y="345" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">No skills</text>
<rect x="1029.5" y="181.0" width="38.0" height="145.0" fill="#2563eb" rx="3"/>
<text x="1048.5" y="174.0" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">15.0k</text>
<text x="1048.5" y="345" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">With skills</text>
<text x="32.0" y="406.0" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#111827">Commands</text>
<line x1="32.0" y1="628" x2="293.0" y2="628" stroke="#d1d5db" stroke-width="1"/>
<line x1="32.0" y1="483" x2="32.0" y2="628" stroke="#d1d5db" stroke-width="1"/>
<rect x="112.5" y="512.0" width="38.0" height="116.0" fill="#16a34a" rx="3"/>
<text x="131.5" y="505.0" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">4</text>
<text x="131.5" y="647" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">No skills</text>
<rect x="174.5" y="483.0" width="38.0" height="145.0" fill="#2563eb" rx="3"/>
<text x="193.5" y="476.0" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">5</text>
<text x="193.5" y="647" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">With skills</text>
<text x="317.0" y="406.0" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#111827">Structure score</text>
<line x1="317.0" y1="628" x2="578.0" y2="628" stroke="#d1d5db" stroke-width="1"/>
<line x1="317.0" y1="483" x2="317.0" y2="628" stroke="#d1d5db" stroke-width="1"/>
<rect x="397.5" y="624.0" width="38.0" height="4.0" fill="#16a34a" rx="3"/>
<text x="416.5" y="617.0" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">0%</text>
<text x="416.5" y="647" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">No skills</text>
<rect x="459.5" y="624.0" width="38.0" height="4.0" fill="#2563eb" rx="3"/>
<text x="478.5" y="617.0" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">0%</text>
<text x="478.5" y="647" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">With skills</text>
<text x="602.0" y="406.0" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#111827">Code quality</text>
<line x1="602.0" y1="628" x2="863.0" y2="628" stroke="#d1d5db" stroke-width="1"/>
<line x1="602.0" y1="483" x2="602.0" y2="628" stroke="#d1d5db" stroke-width="1"/>
<rect x="682.5" y="483.0" width="38.0" height="145.0" fill="#16a34a" rx="3"/>
<text x="701.5" y="476.0" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">93%</text>
<text x="701.5" y="647" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">No skills</text>
<rect x="744.5" y="483.0" width="38.0" height="145.0" fill="#2563eb" rx="3"/>
<text x="763.5" y="476.0" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">93%</text>
<text x="763.5" y="647" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">With skills</text>
<text x="887.0" y="406.0" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#111827">Metrics (AUROC)</text>
<line x1="887.0" y1="628" x2="1148.0" y2="628" stroke="#d1d5db" stroke-width="1"/>
<line x1="887.0" y1="483" x2="887.0" y2="628" stroke="#d1d5db" stroke-width="1"/>
<rect x="967.5" y="488.1" width="38.0" height="139.9" fill="#16a34a" rx="3"/>
<text x="986.5" y="481.1" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">0.7421</text>
<text x="986.5" y="647" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">No skills</text>
<rect x="1029.5" y="483.0" width="38.0" height="145.0" fill="#2563eb" rx="3"/>
<text x="1048.5" y="476.0" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">0.7689</text>
<text x="1048.5" y="647" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">With skills</text>
<rect x="32" y="690" width="14" height="14" fill="#16a34a" rx="2"/>
<text x="54" y="702" font-family="Arial, sans-serif" font-size="13" fill="#111827">No skills baseline</text>
<rect x="252" y="690" width="14" height="14" fill="#2563eb" rx="2"/>
<text x="274" y="702" font-family="Arial, sans-serif" font-size="13" fill="#111827">With skills</text>
</svg>

| Metric | No skills baseline | With skills |
|---|---|---|
| Metrics (AUROC) | AUROC 0.7421 | AUROC 0.7689 |

## Quality Signals

| Run | Expected metric | Reported result | Status | Evidence |
|---|---|---|---|---|
| No skills baseline | AUROC | AUROC 0.7421 (aggregated best validation metric) | NA | NA |
| With skills | AUROC | AUROC 0.7689 (aggregated best validation metric) | NA | NA |

## Output Changes

| Run | Changed files | Added | Modified | Notable files |
|---|---:|---:|---:|---|
| No skills baseline | 3 | NA | NA | nvflare_jobs/ames_fedavg/client.py; nvflare_jobs/ames_fedavg/model.py; nvflare_jobs/ames_fedavg/job.py |
| With skills | 3 | NA | NA | nvflare_jobs/ames_fedavg/client.py; nvflare_jobs/ames_fedavg/model.py; nvflare_jobs/ames_fedavg/job.py |

## Outcome Details

| Signal | No skills baseline | With skills |
|---|---|---|
| Agent/container outcome | passed | passed |
| FL result quality gate | pass: scalar FL result metric available | pass: scalar FL result metric available |
| Reported validation metric | AUROC 0.7421 | AUROC 0.7689 |
| Additional/other validation metric values | AUROC 0.7421 | AUROC 0.7689 |
| Source input protection | not captured | not captured |
| Copied workspace changes | 3 changed | 3 changed |
| Captured generated artifacts | 3 changed/generated files, 1 runtime artifacts | 3 changed/generated files, 1 runtime artifacts |
| Required structure files | 0/3 present; missing client.py, model.py, job.py; nested copies ignored for current-structure score: nvflare_jobs/ames_fedavg | 0/3 present; missing client.py, model.py, job.py; nested copies ignored for current-structure score: nvflare_jobs/ames_fedavg |
| Optional structure files | download_data.py | download_data.py |

## Structure Correctness

The structure checks look for the core converted source files and captured runtime/export artifacts. They are report signals, not a substitute for running the generated job.

| Structure signal | No skills baseline | With skills |
|---|---|---|
| Required converted files | 0/3 present; missing client.py, model.py, job.py; nested copies ignored for current-structure score: nvflare_jobs/ames_fedavg | 0/3 present; missing client.py, model.py, job.py; nested copies ignored for current-structure score: nvflare_jobs/ames_fedavg |
| Nested generated job source | nvflare_jobs/ames_fedavg (client.py, job.py, model.py) | nvflare_jobs/ames_fedavg (client.py, job.py, model.py) |
| Optional helper files | download_data.py | download_data.py |
| Final workspace Python inventory | none | none |
| Changed/generated Python inventory | client.py, job.py, model.py | client.py, job.py, model.py |
| Runtime artifact config inventory | config_fed_server.json | config_fed_server.json |

### Captured Structure Trees

Trees are rendered from captured artifact manifests in tree-command format.

#### No skills baseline

Final workspace:

```text
.
|-- download_data.py
|-- nvflare_jobs
|   `-- ames_fedavg
|       |-- client.py
|       |-- job.py
|       `-- model.py
`-- runtime_workspaces
    `-- job
        `-- server
            `-- simulate_job
                `-- app_server
                    `-- config
                        `-- config_fed_server.json
```

Changed/generated files:

```text
.
|-- nvflare_jobs
|   `-- ames_fedavg
|       |-- client.py
|       |-- job.py
|       `-- model.py
`-- runtime_workspaces
    `-- job
        `-- server
            `-- simulate_job
                `-- app_server
                    `-- config
                        `-- config_fed_server.json
```

#### With skills

Final workspace:

```text
.
|-- download_data.py
|-- nvflare_jobs
|   `-- ames_fedavg
|       |-- client.py
|       |-- job.py
|       `-- model.py
`-- runtime_workspaces
    `-- job
        `-- server
            `-- simulate_job
                `-- app_server
                    `-- config
                        `-- config_fed_server.json
```

Changed/generated files:

```text
.
|-- nvflare_jobs
|   `-- ames_fedavg
|       |-- client.py
|       |-- job.py
|       `-- model.py
`-- runtime_workspaces
    `-- job
        `-- server
            `-- simulate_job
                `-- app_server
                    `-- config
                        `-- config_fed_server.json
```

## Generated Code Quality Signals

These are evidence signals for interpreting runtime and maintenance quality. They do not change pass/fail quality gates or the winner policy.

| Evidence signal | No skills baseline | With skills |
|---|---|---|
| Overall code quality signal | good: 6.5/7 evidence points | good: 6.5/7 evidence points |
| Client data split/use | good: site-aware, explicit sharding, validation data referenced | good: site-aware, explicit sharding, validation data referenced |
| Loss/optimizer lifecycle | good: loss/optimizer built outside FL loop | good: loss/optimizer built outside FL loop |
| Data/DataLoader lifecycle | good: data loaded before FL loop, DataLoader built before FL loop | good: data loaded before FL loop, DataLoader built before FL loop |
| Per-round metric workload | caution: 1 evaluate call(s) in FL loop | caution: 1 evaluate call(s) in FL loop |
| Runtime observability | good: generated code prints per-epoch progress | good: generated code prints per-epoch progress |
| Runtime/output locality | good: runtime artifacts captured separately from temp/runtime paths | good: runtime artifacts captured separately from temp/runtime paths |
| Dependency install strategy | good: requirements-file install, succeeded | good: requirements-file install, accelerator-capable dependency stack, succeeded |
| API pattern | context: Client API loop pattern | context: Client API loop pattern |

Dependency policy note: accelerator-capable framework installs are valid for accelerator-backed training jobs but can dominate benchmark wall time when uncached. CPU-only framework installs are faster, but they should only be treated as comparable when the benchmark is intentionally CPU-only.

## Activity Insights

| Activity signal | No skills baseline | With skills | Interpretation |
|---|---:|---:|---|
| File reads (`cat`/`sed`/Read tool) | 0 | 0 | Direct file-read behavior; includes shell cat/sed and Read tool calls. |
| `find` commands | 0 | 0 | Filesystem discovery proxy. |
| `rg`/`grep` search commands | 0 | 0 | Search use proxy; covers rg and grep. |
| Simulation references | 0 | 0 | Shows validation effort against generated jobs. |
| Python compile checks | 0 | 0 | Shows syntax validation effort. |
| Skill calls / skill references | 0 | 0 | Only skills-enabled runs should usually show these; includes Skill tool calls. |
| Agent / inspect calls | 0 | 0 | Shows use of agent inspection commands; includes Agent tool calls. |
| Python job.py references | 0 | 0 | Shows repeated exercise of generated job entry points. |

## Event Mix

| Event type | No skills baseline | With skills |
|---|---:|---:|
| `command_execution` | 0 | 0 |
| `agent_message` | 0 | 0 |
| `file_change` | 0 | 0 |
| `todo_list` | 0 | 0 |

## Cost And Work Comparison

Cost numbers are descriptive only. Quality gates decide whether a cost comparison is meaningful.

`Runtime seconds` is total elapsed time minus captured dependency-install command time. `Dependency install seconds` is captured dependency-install command time. `Non-install command seconds` is summed duration of captured non-install shell/tool commands, so it can be lower than runtime when the agent spends time reasoning, waiting, or using non-command tools.
Command span timing is operation-level evidence, not a strict wall-clock partition; it can differ from total elapsed time when agent event timestamps overlap, are truncated, or come from a different clock than the harness timer.

| Signal | No skills baseline | With skills | Delta right-left |
|---|---:|---:|---:|
| Total time seconds | 180 | 240 | 60 |
| Runtime seconds | 150 | 180 | 30 |
| Dependency install seconds | 30 | 60 | 30 |
| Non-install command seconds | 140 | 170 | 30 |
| Total tokens | 12.0k | 15.0k | 3.0k |
| Commands | 4 | 5 | 1 |
| Unique commands | 3 | 4 | 1 |
| Changed/generated files | 3 | 3 | 0 |
| Runtime artifacts | 1 | 1 | 0 |

## Why

**Why With skills is slower and has longer runtime after install** (+60s total / +33%; +30s runtime / +20% vs No skills baseline):

**Slowdown driver comparison**

| Driver | With skills | No skills baseline | Delta | Interpretation |
|---|---:|---:|---:|---|
| Total elapsed | 240s | 180s | +60s | overall wall-clock comparison |
| Dependency install | 60s | 30s | +30s | dependency setup/download time |
| Runtime after install | 180s | 150s | +30s | agent/job runtime after dependency setup |
| Captured command time | 230s | 170s | +60s | captured command time contributing to wall-clock slowdown |


**Elapsed time accounting**

| Run | Total | Dependency install | Runtime after install | Captured non-install commands |
|---|---:|---:|---:|---:|
| With skills | 240s | 60s | 180s | 170s |
| No skills baseline | 180s | 30s | 150s | 140s |

`Runtime after install` is total elapsed time minus captured dependency-install command time. Captured command spans identify slow operations but are not guaranteed to add up exactly to total elapsed time.

**Longest command comparison**

| Rank | With skills | No skills baseline |
|---:|---|---|
| 1 | `python3 job.py --num-sites 3 --num-rounds 3` (170s, exit 0) | `python3 job.py --num-sites 3 --num-rounds 3` (140s, exit 0) |
| 2 | `uv pip install -r requirements-train.txt` (60s, exit 0) | `python3 -m pip install -r requirements.txt` (30s, exit 0) |

**NVFLARE runtime path diverged**

| Run | Runtime path | Successful runs | Total captured time | Representative command |
|---|---|---:|---:|---|
| With skills | `recipe.execute(SimEnv(...))` with `PTInProcessClientAPIExecutor` | 1 command | 170s | `python3 job.py --num-sites 3 --num-rounds 3` (170s, exit 0) |
| No skills baseline | `recipe.execute(SimEnv(...))` with `PTInProcessClientAPIExecutor` | 1 command | 140s | `python3 job.py --num-sites 3 --num-rounds 3` (140s, exit 0) |

- **Dependency cost is separate from code efficiency**: the code-quality table records `good: requirements-file install, accelerator-capable dependency stack, succeeded`. That explains install-time cost. Generated-code lifecycle signals remain quality evidence, but they should only be treated as runtime slowdown evidence when non-install runtime is also slower.

**Why With skills uses more tokens** (+3.0k / +25% vs No skills baseline):

**Token usage comparison**

| Driver | With skills | No skills baseline | Delta | Interpretation |
|---|---:|---:|---:|---|
| Total tokens | 15.0k | 12.0k | +3.0k | overall token comparison |
| Cache-read tokens | NA | NA | NA | cached context re-read across turns |
| Cache-creation tokens | NA | NA | NA | new context written into prompt cache |
| Output tokens | 0 | 0 | 0 | model response text |
| Assistant turns | NA | NA | NA | model round-trips |
| Skill calls | NA | NA | NA | skill documentation/context loading |

- Detailed token subcomponents were not available or did not isolate one dominant cause; use the table above to see which captured token/work drivers changed.


## Interpretation

All available runs passed the benchmark quality gates captured by this report.
Runtime winner by wall-clock seconds: No skills baseline (180s vs 240s, delta 60s).
Token-use winner: No skills baseline (12.0k vs 15.0k, delta 3.0k).
Read cost winners only after checking the quality gates; a cheaper run that does not report the requested FL result is not a successful benchmark winner.

## Artifacts

- `metrics_report.md`
- `metrics_report.html`
- `records/`
