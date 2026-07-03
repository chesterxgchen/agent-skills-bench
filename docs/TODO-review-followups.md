# Review follow-ups (deferred)

Remaining items from the architect/principal/security review rounds that were
**not** code-fixed, with the reasoning for deferral. Everything else from the
review is merged to `main`.

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
