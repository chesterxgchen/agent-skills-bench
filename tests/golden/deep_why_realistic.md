## Root Cause Analysis

**Why With skills is slower and has longer runtime after install** (+300s total / +50%; +180s runtime / +32% vs No skills baseline):

**Time contributors (ranked by attributed time)**

1. **Repeated simulator executions +370s** — With skills ran 2 successful executions (total 730s) vs 1 for No skills baseline; the reruns beyond the first are re-validation work (captured rationale in the repeated-executions table below).
2. **Dependency install +120s** — With skills spent 150s on 1 requirements-file install(s), vs 30s for No skills baseline. The With skills install resolved an accelerator-capable dependency stack (nvidia-cublas-cu13, nvidia-cudnn-cu13).

**Slowdown driver comparison**

| Driver | With skills | No skills baseline | Delta | Interpretation |
|---|---:|---:|---:|---|
| Total elapsed | 900s | 600s | +300s | overall wall-clock comparison |
| Dependency install | 150s | 30s | +120s | dependency setup/download time |
| Runtime after install | 750s | 570s | +180s | agent/job runtime after dependency setup |
| Agent/provider residual | 19s | 10s | +9s | uninstrumented time not attributed to captured dependency or non-install command spans |
| Captured command time | 881s | 590s | +291s | captured command time contributing to wall-clock slowdown |
| Unique model requests | 15 | 8 | +7 | extra provider requests, deduplicated by request ID |
| Extended-reasoning events | 6 | 1 | +5 | extra reasoning activity |
| Skill calls | 4 | 0 | +4 | skill loading/context overhead |

### Repeated Job/Simulation Executions

These are full successful job or simulator executions, excluding export, help, and preflight commands. Repeated runs materially affect elapsed time and usually mean the agent reran after validation, recovery, or configuration changes.

| Run | Successful executions | Total captured job time | Executions | Captured reason/evidence |
|---|---:|---:|---|---|
| With skills | 2 | 730s | 1. `python3 job.py --num-sites 3 --num-rounds 3` (360s, exit 0); 2.<br>`python3 job.py --num-sites 3 --num-rounds 3` (370s, exit 0) | not captured; inspect commands around the repeated run |

Baseline comparison: No skills baseline had 1 command classified successful job/simulator execution totaling 560s.


**Elapsed time accounting**

| Run | Total | Dependency install | Runtime after install | Captured non-install commands | Agent/provider residual |
|---|---:|---:|---:|---:|---:|
| With skills | 900s | 150s | 750s | 731s | 19s |
| No skills baseline | 600s | 30s | 570s | 560s | 10s |

`Runtime after install` is total elapsed time minus captured dependency-install command/background-task time. Captured command spans identify slow operations but are not guaranteed to add up exactly to total elapsed time.
The residual column is uninstrumented time after captured command spans. It can include provider round trips, tool orchestration, background gaps, and other activity, so do not assign it to skill-induced reasoning without separate evidence.

**Longest command comparison**

| Rank | With skills | No skills baseline |
|---:|---|---|
| 1 | `python3 job.py --num-sites 3 --num-rounds 3` (370s, exit 0) | `python3 job.py --num-sites 3 --num-rounds 3` (560s, exit 0) |
| 2 | `python3 job.py --num-sites 3 --num-rounds 3` (360s, exit 0) | `python3 -m pip install -r requirements.txt` (30s, exit 0) |
| 3 | `uv pip install -r requirements-train.txt` (150s, exit 0) | no timed command span >=30s captured |

**Dependency install path differed**

| Run | Install time | Install scope | Stack evidence | Installer | Representative command |
|---|---:|---|---|---|---|
| With skills | 150s | 1 requirements-file install(s) | accelerator-capable dependency stack (nvidia-cublas-cu13, nvidia-cudnn-cu13) | uv pip | `uv pip install -r requirements-train.txt` |
| No skills baseline | 30s | 1 requirements-file install(s) | framework dependency stack | python -m pip | `python3 -m pip install -r requirements.txt` |

- **Why the install is longer**: With skills used 1 requirements-file install command with accelerator-capable dependency stack (nvidia-cublas-cu13, nvidia-cudnn-cu13); No skills baseline used 1 requirements-file install command with framework dependency stack. The with-skills install logs show accelerator-capable framework packages.
- **Captured package examples**: nvidia-cublas-cu13, nvidia-cudnn-cu13, torch.
- **Accelerator dependency evidence**: with-skills install logs included nvidia-cublas-cu13, nvidia-cudnn-cu13; large accelerator/framework wheels can dominate install time.
- **Installer difference**: with-skills used uv pip, while the baseline used python -m pip.
- **Network/download evidence**: with-skills install logs showed broken/incomplete download, download retry; baseline install logs showed no captured network retry/timeout markers.

**NVFLARE runtime path diverged**

| Run | Runtime path | Successful runs | Total captured time | Representative command |
|---|---|---:|---:|---|
| With skills | `recipe.execute(SimEnv(...))` with `PTInProcessClientAPIExecutor` | 2 commands | 730s | `python3 job.py --num-sites 3 --num-rounds 3` (370s, exit 0) |
| No skills baseline | `recipe.execute(SimEnv(...))` with `PTInProcessClientAPIExecutor` | 1 command | 560s | `python3 job.py --num-sites 3 --num-rounds 3` (560s, exit 0) |

- **Generated-code efficiency issue aligns with slower non-install runtime**: the code-quality signal flags With skills as `poor` for loss/optimizer lifecycle (loss/optimizer rebuilt inside FL loop), while the baseline is `good` (loss/optimizer built outside FL loop). Runtime excluding dependency install is 750s vs 570s, so repeated setup inside the per-round training boundary is plausible runtime overhead. This does not prove sole causality, but it is a generated-code issue worth investigating.
- **Dependency cost is separate from code efficiency**: the code-quality table records `good: requirements-file install, accelerator-capable dependency stack, succeeded`. That explains install-time cost. Generated-code lifecycle signals remain quality evidence, but they should only be treated as runtime slowdown evidence when non-install runtime is also slower.

**Why With skills uses more tokens** (+90.0k / +100% vs No skills baseline):

**Token usage comparison**

| Driver | With skills | No skills baseline | Delta | Interpretation |
|---|---:|---:|---:|---|
| Total tokens | 180.0k | 90.0k | +90.0k | overall token comparison |
| Cache-read tokens | 110.0k | 40.0k | +70.0k | cached context re-read across turns |
| Cache-creation tokens | 24.0k | 12.0k | +12.0k | new context written into prompt cache |
| Output tokens | 16.0k | 8.0k | +8.0k | model response text |
| Unique model requests | 15 | 8 | +7 | provider requests deduplicated by request ID |
| Tokens per request | 12.0k | 11.2k | +750 | total tokens divided by unique model requests |
| `tools_changed` cache misses | 1 | 0 | +1 | inferred cache rebuilds immediately after ToolSearch |
| Skill calls | 4 | 0 | +4 | skill documentation/context loading |
| Effective cost | $0.9500 | $0.4200 | +$0.5300 | model/provider reported cost |

- **Prompt cache re-reads are the dominant driver** (110.0k vs 40.0k, +70.0k, 78% of the total token delta): cache-read tokens represent context cached from previous turns being re-read on each new request. The With skills run repeatedly re-read its accumulated context across 15 unique model requests (vs 8 in the No skills baseline run). Aggregate usage does not identify which context segment produced those reads.
- **Skill context was loaded, but its token share is not isolated** (4 Skill call(s) vs 0): Skill documentation is one source of added context, alongside tool schemas, conversation history, tool results, and generated text. The aggregate cache counters cannot attribute the full cache growth—or a precise fraction—to the Skill call(s).
- **Tool-set-change cache misses are reported separately** (1 vs 0; associated cache creation delta 8.0k): this conservative signal requires a non-initial request with zero cache-read and nonzero cache creation immediately after `ToolSearch`. Its cache rebuild is attributed to changed tool schemas, not to a Skill call.
- **New context written to cache** (+12.0k cache-creation tokens): the With skills run wrote more new content into the prompt cache (skill docs, tool schemas, or conversation history not present in the No skills baseline run).
- **Output tokens increased** (16.0k vs 8.0k, +8.0k): the With skills run generated more text, contributing directly to the token delta.
- **Effective cost** ($0.9500 vs $0.4200, +$0.5300 / +126%): despite 100% more total tokens, the cost premium can be smaller when more of the usage is lower-priced cache reads.
