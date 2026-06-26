## Why

**Why With skills is slower and has longer runtime after install** (+300s total / +50%; +180s runtime / +32% vs No skills baseline):

**Slowdown driver comparison**

| Driver | With skills | No skills baseline | Delta | Interpretation |
|---|---:|---:|---:|---|
| Total elapsed | 900s | 600s | +300s | overall wall-clock comparison |
| Dependency install | 150s | 30s | +120s | dependency setup/download time |
| Runtime after install | 750s | 570s | +180s | agent/job runtime after dependency setup |
| Captured command time | 881s | 590s | +291s | captured command time contributing to wall-clock slowdown |
| Assistant turns | 21 | 9 | +12 | extra model round-trips |
| Extended-reasoning events | 6 | 1 | +5 | extra reasoning activity |
| Skill calls | 4 | 0 | +4 | skill loading/context overhead |

### Repeated Job/Simulation Executions

These are full successful job or simulator executions, excluding export, help, and preflight commands. Repeated runs materially affect elapsed time and usually mean the agent reran after validation, recovery, or configuration changes.

| Run | Successful executions | Total captured job time | Executions | Captured reason/evidence |
|---|---:|---:|---|---|
| With skills | 2 | 730s | 1. `python3 job.py --num-sites 3 --num-rounds 3` (360s, exit 0); 2. `python3 job.py --num-sites 3 --num-rounds 3` (370s, exit 0) | not captured; inspect commands around the repeated run |

Baseline comparison: No skills baseline had 1 command classified successful job/simulator execution totaling 560s.


**Elapsed time accounting**

| Run | Total | Dependency install | Runtime after install | Captured non-install commands |
|---|---:|---:|---:|---:|
| With skills | 900s | 150s | 750s | 731s |
| No skills baseline | 600s | 30s | 570s | 560s |

`Runtime after install` is total elapsed time minus captured dependency-install command/background-task time. Captured command spans identify slow operations but are not guaranteed to add up exactly to total elapsed time.

**Longest command comparison**

| Rank | With skills | No skills baseline |
|---:|---|---|
| 1 | `python3 job.py --num-sites 3 --num-rounds 3` (370s, exit 0) | `python3 job.py --num-sites 3 --num-rounds 3` (560s, exit 0) |
| 2 | `python3 job.py --num-sites 3 --num-rounds 3` (360s, exit 0) | `python3 -m pip install -r requirements.txt` (30s, exit 0) |
| 3 | `uv pip install -r requirements-train.txt` (150s, exit 0) | no timed command span >=30s captured |

- **Dependency install path differed**: with-skills spent 150s across 1 install command(s) (1 requirements-file install(s)), while the baseline spent 30s across 1 install command(s) (1 requirements-file install(s)). The longest with-skills install was `uv pip install -r requirements-train.txt` (150s, exit 0); downloaded packages included nvidia-cublas-cu13, nvidia-cudnn-cu13, torch; baseline longest install was `python3 -m pip install -r requirements.txt` (30s, exit 0). Installer form differed: with-skills used uv pip; baseline longest install used python -m pip. Accelerator dependency evidence: with-skills install logs included nvidia-cublas-cu13, nvidia-cudnn-cu13; large accelerator/framework wheels can dominate install time. Network/download evidence: with-skills install logs showed broken/incomplete download, download retry; baseline install logs showed no captured network retry/timeout markers.

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
| Assistant turns | 21 | 9 | +12 | model round-trips |
| Skill calls | 4 | 0 | +4 | skill documentation/context loading |
| Effective cost | $0.9500 | $0.4200 | +$0.5300 | model/provider reported cost |

- **Prompt cache re-reads are the dominant driver** (110.0k vs 40.0k, +70.0k, 78% of the total token delta): cache-read tokens represent context cached from previous turns being re-read on each new turn. The With skills run accumulated a larger cached context window — primarily skill documentation injected via 4 Skill call(s) — and then re-read that context across all 21 turns (vs 9 turns in the No skills baseline run).
- **Skill documentation injected into context** (4 Skill call(s) vs 0): each Skill invocation adds skill documentation to the context window. That content is written into the prompt cache on first use, then re-read as cached context on every subsequent turn — compounding the cache-read cost with each additional turn.
- **New context written to cache** (+12.0k cache-creation tokens): the With skills run wrote more new content into the prompt cache (skill docs, tool schemas, or conversation history not present in the No skills baseline run).
- **Output tokens increased** (16.0k vs 8.0k, +8.0k): the With skills run generated more text, contributing directly to the token delta.
- **Effective cost** ($0.9500 vs $0.4200, +$0.5300 / +126%): despite 100% more total tokens, the cost premium is much smaller because cache-read tokens are priced significantly lower than regular input tokens.
