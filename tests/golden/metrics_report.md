# Benchmark Metrics

Result root: `<RESULT_ROOT>`

Status: No skills baseline: needs review (Failed check `result_metric_scalar`: AUROC was reported, but no FL-level scalar value was found.); With skills: needs review (Failed check `result_metric_scalar`: AUROC was reported, but no FL-level scalar value was found.)

## Runs

| Run | Agent | Model | Host OS | Status | Skills available | Skills inspected | Skills applied/used | Shared refs read | Elapsed seconds | Tokens | Commands | Root cause |
|---|---|---|---|---|---|---|---|---|---:|---:|---:|---|
| No skills baseline | codex | default | Ubuntu 24.04 LTS | needs review (Failed check `result_metric_scalar`: AUROC was reported, but no FL-level scalar value was found.) | not enabled | none | none | none | 180 | 12000 | 4 | Failed check `result_metric_scalar`: AUROC was reported, but no FL-level scalar value was found. |
| With skills | codex | default | Ubuntu 24.04 LTS | needs review (Failed check `result_metric_scalar`: AUROC was reported, but no FL-level scalar value was found.) | nvflare-convert-lightning; nvflare-convert-pytorch; nvflare-diagnose-job; nvflare-orient | none | nvflare-convert-pytorch | none | 240 | 15000 | 5 | Failed check `result_metric_scalar`: AUROC was reported, but no FL-level scalar value was found. |

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
<text x="701.5" y="476.0" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">50%</text>
<text x="701.5" y="647" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">No skills</text>
<rect x="744.5" y="483.0" width="38.0" height="145.0" fill="#2563eb" rx="3"/>
<text x="763.5" y="476.0" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">50%</text>
<text x="763.5" y="647" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">With skills</text>
<text x="887.0" y="406.0" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#111827">Metrics (AUROC)</text>
<line x1="887.0" y1="628" x2="1148.0" y2="628" stroke="#d1d5db" stroke-width="1"/>
<line x1="887.0" y1="483" x2="887.0" y2="628" stroke="#d1d5db" stroke-width="1"/>
<rect x="967.5" y="604" width="38.0" height="20" fill="#e5e7eb" rx="3"/>
<text x="986.5" y="619" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="700" fill="#4b5563">NA</text>
<text x="986.5" y="647" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">No skills</text>
<rect x="1029.5" y="604" width="38.0" height="20" fill="#e5e7eb" rx="3"/>
<text x="1048.5" y="619" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="700" fill="#4b5563">NA</text>
<text x="1048.5" y="647" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">With skills</text>
<rect x="32" y="690" width="14" height="14" fill="#16a34a" rx="2"/>
<text x="54" y="702" font-family="Arial, sans-serif" font-size="13" fill="#111827">No skills baseline</text>
<rect x="252" y="690" width="14" height="14" fill="#2563eb" rx="2"/>
<text x="274" y="702" font-family="Arial, sans-serif" font-size="13" fill="#111827">With skills</text>
</svg>

| Metric | No skills baseline | With skills |
|---|---|---|
| Metrics (AUROC) | AUROC NA | AUROC NA |

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

## Interpretation

Quality comparison is incomplete because these runs failed a benchmark quality gate: No skills baseline, With skills.
For this artifact, the missing/partial signal is `AUROC` reporting, not necessarily a Docker or Python execution crash.
Runtime winner by wall-clock seconds: No skills baseline (180s vs 240s, delta 60s).
Token-use winner: No skills baseline (12.0k vs 15.0k, delta 3.0k).
Read cost winners only after checking the quality gates; a cheaper run that does not report the requested FL result is not a successful benchmark winner.

## Comparison

| Metric | Delta |
|---|---:|
| elapsed_seconds_with_skills_minus_without_skills | 60 |
| token_count_with_skills_minus_without_skills | 3000 |

## Why

**Why the comparison needs review**

At least one run failed the job/result quality gates, so elapsed time, token use, and artifact count should not be treated as benchmark wins until the result issue is resolved.

| Run | Job run status | Result quality issue | Result metric |
|---|---|---|---|
| No skills baseline | completed: simulation completed — FL workflow reached Finished state | Failed check `result_metric_scalar`: AUROC was reported, but no FL-level scalar value was found. | partial: 1 reported values, no FL-level scalar |
| With skills | completed: simulation completed — FL workflow reached Finished state | Failed check `result_metric_scalar`: AUROC was reported, but no FL-level scalar value was found. | partial: 1 reported values, no FL-level scalar |

Both runs need review; neither side is a valid comparison winner until the result metrics are fixed.

**Why With skills is slower and has longer runtime after install** (+60s total / +33%; +30s runtime / +20% vs No skills baseline):

**Slowdown driver comparison**

| Driver | With skills | No skills baseline | Delta | Interpretation |
|---|---:|---:|---:|---|
| Total elapsed | 240s | 180s | +60s | overall wall-clock comparison |
| Dependency install | 60s | 30s | +30s | dependency setup/download time |
| Runtime after install | 180s | 150s | +30s | agent/job runtime after dependency setup |
| Captured command time | 230s | 170s | +60s | captured command time contributing to wall-clock slowdown |


**Elapsed time accounting**

| Run | Total | Dependency install | Runtime after install | Captured non-install commands | Agent/model interaction residual |
|---|---:|---:|---:|---:|---:|
| With skills | 240s | 60s | 180s | 170s | 10s |
| No skills baseline | 180s | 30s | 150s | 140s | 10s |

`Runtime after install` is total elapsed time minus captured dependency-install command/background-task time. Captured command spans identify slow operations but are not guaranteed to add up exactly to total elapsed time. The residual column is the best available indicator that wall time came from agent/model round trips, tool orchestration, background command gaps, or other non-command activity.

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

