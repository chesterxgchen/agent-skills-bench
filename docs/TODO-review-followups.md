# Review follow-ups (deferred)

Remaining items from the architect/principal/security review rounds that were
**not** code-fixed, with the reasoning for deferral. Everything else from the
review is merged to `main`.

---

## 4. RCA container egress is unrestricted (security, residual)

**Status:** Open (mitigated at the CLI layer; egress allowlist still pending).

The RCA container keeps network egress (needed for the model API) and holds
the vendor API key in the CLI env, while the mounted evidence is
attacker-authored. The investigator needs working tools — shell, file
reads/searches, and the ability to write/run small analysis scripts — so it
does NOT run read-only, but it no longer gets the blanket bypass flags either:

- **claude (sandboxed):** explicit
  `--allowedTools Bash,Read,Grep,Glob,Write,Edit` instead of
  `--dangerously-skip-permissions`; `WebFetch`/`WebSearch`/`Task` disallowed,
  `--strict-mcp-config`. Residual: Bash children inherit
  the CLI env (claude has no shell env policy), so an injected shell command
  can still reach `ANTHROPIC_API_KEY` and the network.
- **codex (sandboxed):** codex's own sandbox (read-only/workspace-write) uses
  bubblewrap, which cannot create a user namespace inside the unprivileged
  container (`bwrap: No permissions to create new namespace`) — so any
  `--sandbox` mode there fails to run a single command and the investigation
  dies. The container is the boundary instead, so codex runs
  `--dangerously-bypass-approvals-and-sandbox` (no bwrap) with
  `shell_environment_policy.inherit=core` retained, so spawned commands still
  don't get the API key. Container network egress remains the residual (below).

**Proper fix (if we want to close it):** apply a network policy to the RCA
container that allows egress only to the model API endpoint(s) — e.g. a
`--network` with an allowlist / proxy, or run on an isolated docker network with
egress filtering. Would also apply to the benchmark container for symmetry.

---

## 1. RCA investigator read scope needs OS/container sandboxing (security) — DONE

**Status:** Implemented. `rca.py` now supports `--sandbox {auto,docker,host}`
(default `auto`). In docker mode the investigator runs in the built agent image
with ONLY the staged evidence mounted read-only (`-w /evidence`), vendor API
keys forwarded via `--env`, one container per step (force-removed even on
timeout). `auto` uses docker when the image is built, else falls back to the
host invoker with a warning. Reads are confined to the mounted evidence, so a
prompt-injected read of `~/.ssh`/`/etc`/other projects finds nothing. Original
write-up kept below for context.

---



**Finding (5a):** The RCA investigator (`benchmark/harness/rca.py`) runs the
agent CLI as a host subprocess over attacker-authored captured evidence. A
prompt-injected investigation can read non-home host files and quote them into
the RCA report on disk.

**Why not fixed with flags:** Verified empirically that Claude Code permission
deny rules **cannot** confine reads to the cwd:
- `Read(~/**)` works (home denied), but `Read(/etc/**)`, `Read(**/.ssh/**)`,
  etc. are **ignored** — `/etc/hosts` and `Grep path="/etc"` still succeed.
- `Grep`/`Glob` take no path specifier at all (only `Read` deny rules apply,
  best-effort).
- `Read(//**)` denies everything *including the cwd*, so it can't be used to
  "allow cwd, deny the rest."

A denylist would give false confidence, so none was added.

**Current mitigations already in place (`rca.py`):**
- Network exfil path blocked: `WebFetch`/`WebSearch`/`Task` denied,
  `--strict-mcp-config` (no host MCP servers).
- `Read(~/**)` denied; evidence staged into a symlink-free temp copy (cwd).
- Codex invoker: `--sandbox read-only` + `shell_environment_policy.inherit=core`
  (spawned commands don't see the API keys); per-vendor env allowlist.
- Residual risk: **local host-file disclosure into the report file** (not
  network exfil).

**Proper fix (needs decision):** Run the investigator inside a container with
only the staged evidence dir mounted (network still needed for the model API).
The harness already has the Docker machinery for benchmark runs. Scope:
- image with the agent CLI installed,
- auth mounted read-only,
- staged evidence dir mounted as the only readable tree,
- wire the invoker to `docker run` instead of a host subprocess.

---

## 3. Auto-run RCA on a with-skills regression — DONE

**Status:** Implemented. The live pair run (`run_pair`) now calls
`autorun_rca_investigations` right after report generation: for each mode the
Root Cause Analysis section would flag (failure / slowdown / tokens /
structure regression, via `resolve_seed(..., "auto")`), it runs the RCA
investigation in the container sandbox and regenerates
`benchmark_insights.md` so the explanation embeds automatically. Structure
regressions are gated on the host-persisted `quality_summary.json`
(`persist_quality_summary`). Best-effort: skips cleanly when no investigator
image/CLI is available, never fails the run, and `BENCHMARK_AUTO_RCA=0`
disables it.

---

## 2. NVFLARE quality scoring hard-coded to `task="conversion"` (correctness / feature)

**Finding (5b):** `benchmark/harness/sdks/nvflare/_logic.py`
(`_conversion_evaluation_rules`, ~line 1763) always scores with
`task="conversion"`. The `federated-statistics` criteria subtree exists under
`benchmark/config/evaluation/nvflare/` but is never reached by the report
scorer, so a federated-statistics run is judged by conversion criteria.

**Why not a one-liner:**
- The run bundle carries `job_name`/`job_slug`/`scenario_name` but **no captured
  task** — there's no authoritative signal to pick the task from.
- The `conversion_quality_*` detectors (`training_control`, `partitioning`,
  `class_weighting`, …) are themselves conversion-specific; a
  federated-statistics run needs its own detectors, not just a different
  `task=` argument.

**Proper fix (needs design):**
- Declare a task per job/scenario and capture it into the run (plan-time →
  `run_plan` entry → loader bundle, e.g. `evaluation_task`).
- Thread that task into `load_evaluation_rules(..., task=...)` in the plugin.
- Add federated-statistics detectors / a task-dispatched quality profile so the
  report renders the right signal rows per task.

**RESOLVED (2026-07-08):** scenarios declare `evaluation_task` /
`evaluation_selectors` (plan-time validation in
`scenario_run_plan.resolve_jobs`), the run record and loader bundle carry
them, and `_logic._run_evaluation_rules` scores the declared task (conversion
detectors gate on the resolved task; other tasks are judged by the code-eval
agent against the task's criteria). Deterministic per-task checks are
scenario-local data, not harness detectors: `result_artifact` +
`acceptance_checks` (see `benchmark/harness/acceptance.py` and
`scenarios/federated-statistics/`). Native skill_evals map to tasks via
`native_skills` patterns in the evaluation manifest. Onboarding a new skill
family = criteria YAML + manifest line + scenario YAML (+ optional check
script); no harness code.
