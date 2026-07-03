# Agent Skills Benchmark — Architecture

Status: current architecture reference
Audience: maintainers of this harness
Scope: the harness as an evidence pipeline + two cross-cutting contracts
(profile/identity and evidence), and an incremental plan to make the
SDK-specific reporting pluggable — **without rewriting the report.**

---

## 1. Context & goals

The harness answers one question: **does giving an agent a packaged set of
skills change how it performs a real task, and *why*?**

It runs an agent CLI (Codex/Claude) twice against the same job — once
`with_skills`, once `without_skills` — in identical Docker runtimes, captures
everything each run did, and produces a comparative report.

The **reporting layer is the heart of the tool.** Running two containers is easy;
**explaining the difference between them in evidence-backed prose is the product.**
We call this the **"why" engine**: it turns structured comparative evidence into a
narrative — e.g. *"the with-skills run was slower because it spent N seconds
re-installing accelerator dependencies the without-skills run skipped."*

**Goal of this design:** make the SDK-specific interpretation **pluggable** — so the
SDK under test (today NVFLARE) can be swapped — while keeping the "why" engine generic
and intact. NVFLARE is simply the first SDK; the harness is built to support others.

---

## 2. Principles

The architecture follows from a few load-bearing rules:

1. **The evidence is the center of gravity.** The system orbits a versioned,
   immutable evidence bundle plus a captured profile identity — not the SDK adapter
   and not the plugin interface. The plugin API serves the evidence contract, not
   the reverse.
2. **The report output is the contract.** What the report produces for a result is
   stable, and a result root can be re-reported offline with no container and no agent.
3. **Plugins never reach backward into runtime.** Runtime materializes a declarative
   capture contract; reports only *read* captured evidence. The only thing crossing
   into runtime is serialized data — never plugin code.
4. **Capture is DATA; interpretation is CODE.** What to capture is declarative data
   authored with the profile and applied by generic runtime code; interpreting
   captured evidence is report-time code. This makes principle 3 structural rather
   than a convention.
5. **The SDK is a dimension, not a layer.** It does not "own" reporting; it
   participates through the profile identity, the capture spec, and the interpretation
   plugin — resolved by *captured identity*, never a live adapter.
6. **General interface, SDK-specific plugins.** The plugin interface is
   domain-neutral; an SDK's domain concepts (for FL: client/server/site/FL-level)
   live in the plugin as custom vocabulary, not in the engine. The profile stays
   **thin** (an id + a registry entry) until a second SDK needs more.

---

## 3. Architecture overview

The load-bearing structure is a linear **evidence pipeline** with **two
cross-cutting contracts**. The contracts are the stable boundaries everything
depends on (§4, §5); the stages are mechanics that may change (§6).

```
        ┌──────────────────── CONTRACT A: Profile & Identity (§4) ────────────────┐
        │  benchmark_profile_id · report_plugin_id · declarative EvidenceCaptureSpec│
        │  · report_plugin_id → plugin registry · resolution rules                │
        │  —  declared once, consumed by Build, Capture, Interpretation           │
        └────────────────────────────────────────────────────────────────────────┘
                 │ feeds
                 ▼
  STAGE 1        STAGE 2          STAGE 3              STAGE 4           STAGE 5
  Build     →    Execution   →    Evidence Capture →   Interpretation →  Report Product
  image +        run plan,        generic code         resolve plugin    existing
  embed          container,       applies the          by captured id;   benchmark_insights
  profile        run agent;       capture spec →       interpret         output & layout
  metadata/spec  SDK-ignorant     immutable artifacts  evidence meaning  (INTACT; calls
                                                                         into Interpretation)
                                       │ writes            ▲ reads
                                       ▼                   │
        ┌──────────────── CONTRACT B: Evidence (§5) ─────────────────────────────┐
        │  versioned RunEvidence / ComparisonEvidence · result-root schema ·      │
        │  provenance / replay metadata   —   loaded from RESULT_ROOT             │
        │  —  the stable replay/report boundary (offline `report` lives here)     │
        └────────────────────────────────────────────────────────────────────────┘
```

**The five stages (§6):** Build (image + embed profile metadata/capture spec),
Execution (run plan, container, agent; SDK-ignorant), Evidence Capture (generic
code applies the capture spec → immutable artifacts), Interpretation (resolve the
plugin by captured id, interpret evidence), Report Product (the existing report;
stays intact, calls into Interpretation).

**The two contracts:** *A — Profile & Identity* (§4) is the pluggability seam; *B
— Evidence* (§5) is the replay/report boundary. Hold these two stable; everything
else is mechanics.

Two distinctions worth naming up front. **Generic vs. plugin:** within
Interpretation + Report Product, "generic" code is SDK-agnostic (timing, tokens,
activity, comparison, markdown) and "plugin" code is SDK-specific (structure,
recipe, FL semantics). **Agent adapters** (`agents/`) are orthogonal to all of
this — they handle agent CLI invocation/parsing (Codex vs Claude) and are already
SDK-agnostic.

---

## 4. Contract A — Profile & Identity

Contract A is **identity + capture data + resolution**: `benchmark_profile_id`,
`report_plugin_id`, the serialized **`EvidenceCaptureSpec`**, and the rules that
resolve an id to a plugin. It *names* the report plugin (by id) but does not
contain it — the `ReportPlugin` interface and its FL vocabulary are
interpretation **code** and live in Stage 4 (§6), keeping the "capture is data,
interpretation is code" line clean. The SDK adapter **declares a default**
`report_plugin_id`; it is one *producer* of identity, **not** the owner of
interpretation (principle 5). Reporting resolves by **captured id**, through a
registry, with a null/default fallback — no live SDK or adapter object at report
time.

Keep this thin for now (one SDK): `report_plugin_id` is a captured string plus a
registry. `benchmark_profile_id` is its alias if the profile ever grows.

### 4.1 The capture spec (declarative DATA)

```python
@dataclass(frozen=True)
class EvidenceCaptureSpec:           # serialized into image metadata at build time
    structure_file_names: tuple[str, ...]            # e.g. client.py/model.py/job.py
    runtime_sources: tuple[tuple[str, str], ...]     # (label, path) e.g. /tmp/nvflare/workspaces
    artifact_globs: tuple[str, ...]                  # S1 — e.g. **/*.log, **/config_fed_*.json
    version: int                                     # S2 — degrade on unknown major
```

It is **data**, not a method the plugin runs at runtime: authored with the
profile, serialized into image metadata at build, and applied by *generic* code
at Stage 3 (§6). No plugin code runs in the container. `capture_spec_version`
(below) records which rules produced a tree, so old roots stay readable.
(Implemented in `sdks/capture_spec.py`; NVFLARE's spec data in
`sdks/nvflare/capture.py`, including `artifact_globs` — for robust simulator-log
capture — and `capture_spec_version`.)

### 4.2 Identity resolution

The host **never loads an SDK profile during a run** (`run.sh`/`runner.py` know
only the agent/image; `write_runtime_image` carries no SDK fields). The one place
identity is reliably available at run time is **inside the container**, via
`sdk_wheel_metadata.json` (built into the image, copied into each `mode_dir`). So
identity travels through that artifact, not a host-side stamp.

1. **Build time** (`host/build.py`): write `report_plugin_id` into the metadata
   block alongside `sdk_name`/`variant`. Built-in profile → `sdk.name`; custom
   `--sdk-profile` → the profile's declared `name` (never the file path, so the
   result is portable).
2. **Run time (in container, `agent_run.py`):** read `report_plugin_id` from the
   copied metadata to resolve the capture spec (§6). Optionally lift it to a
   root-level metadata file after runs so the report path need not descend into a
   mode dir.
3. **Report time:** `report` receives only `RESULT_ROOT` (no adapter). It reads
   `report_plugin_id` from the root file (or a mode dir's metadata) and looks the
   plugin up in a **registry keyed by id**. Fallbacks: **absent** id (legacy
   tree) → NVFLARE plugin for output stability; **present-but-unknown** id → null
   plugin (warn, don't fail). The *conceptual* default is the null plugin — the
   NVFLARE fallback is a compatibility affordance for trees with no recorded id, not an NVFLARE-centered
   architecture.

YAML mapping: an optional `report_plugin:` field on the SDK profile (defaulting
to `sdk.name`, then null), resolved through `sdks/report_registry`, keeps
`ConfigurableSdkAdapter` declarative.

### 4.3 The versioned metadata block

Pin a tiny, explicit identity/version block at build time into the image and
result root — the whole resolution contract in one place:

```json
{
  "schema_version": "1",          // evidence/result-root schema (Contract B)
  "sdk_name": "nvflare",
  "benchmark_profile_id": "nvflare",
  "report_plugin_id": "nvflare",
  "capture_spec_version": "1"     // bump when capture rules change
}
```

`schema_version` and `capture_spec_version` are independent: the profile
`schema_version` gates report-plugin resolution (an unsupported major degrades to
the null plugin), while `capture_spec_version` records which capture rules produced
the tree (so old trees self-describe) and a newer capture spec degrades to a minimal
spec when applied. **Durable naming:**
`sdk_wheel_metadata.json` is the practical v1 home (it already exists and is
copied into each `mode_dir`), but the durable concept is a *profile/evidence
descriptor* — promote it to `benchmark_profile_metadata.json` long term. Readers
should treat the **block, not the filename**, as the contract.

### 4.4 Evaluation criteria input

Evaluation policy follows the same capture-before-interpretation rule. An SDK
profile may declare `evaluation.criteria_path`, relative to the checkout
provided by `--sdk-repo`; `--evaluation-criteria` is the explicit override and
the required source when no repo-relative path is available. Build preparation
normalizes the source into a harness rules bundle, validates the composition,
hashes it, and stages it into both images. Sources can be native harness YAML
or SDK-native criteria layouts with an explicit converter; NVFLARE's
`dev_tools/agent/skill_evals/*/evals.json` is converted this way. The build
records the bundle's content hash in the §4.3 identity block, which the host
lifts to the root-level descriptor — outside the container-writable result
mount. Stage 3 captures the resolved bundle under each mode's
`evaluation_rules/`, but Stage 4 scores from that copy **only when its recomputed
hash matches the host-anchored hash**; any mismatch (a run that rewrote its own
grading criteria in the mount) or missing anchor falls back to the packaged
rules. Criteria are thus a verified build-time input, never something a run can
influence in its own favor, and report replay never rereads a mutable SDK
checkout.

---

## 5. Contract B — Evidence

Everything reporting needs arrives as **captured, normalized evidence** on disk —
never a live re-computation. This is why `report` (the regeneration command) can
rebuild every artifact from an existing result root with no Docker and no agent.
The evidence bundle is the single source of truth: both the generic engine and the
plugin read `RunEvidence`, so the plugin API can evolve without callers passing ad-hoc dicts.

```python
@dataclass(frozen=True)
class RunEvidence:
    identity: RunIdentity          # mode, label, agent, model, available
    paths: RunPaths                # mode_dir + artifact accessors (no live reads)
    record: Mapping                # normalized per-run record
    summary: Mapping               # run_summary.json
    usage: Mapping                 # tokens
    activity: Mapping              # classified commands
    workspace_delta: Mapping       # final_structure_files, changed_files
    runtime_image: Mapping
    container_exit: Mapping
    sdk_metadata: Mapping          # the §4.3 block incl. report_plugin_id
    text: RunTextArtifacts         # console, agent_events, last_message, stderr
    validation_metric: Mapping | None

@dataclass(frozen=True)
class ComparisonEvidence:
    schema_version: int            # provenance stamp (current SCHEMA_VERSION; not a runtime gate)
    runs: dict[str, RunEvidence]   # keyed by mode
    modes: list[str]
```

**Contract B is captured evidence only — never interpreted data.** `RunEvidence`
holds what Stage 3 wrote to disk; nothing on it is filled in later by a plugin.
Plugin-derived interpretation lives in a **separate sidecar** (`PluginEvidence`,
§6 Stage 4), so the line stays clean: *evidence is captured; interpretation is
derived.*

**Versioning.** `ComparisonEvidence.schema_version` is a *provenance stamp* (set to
the current `SCHEMA_VERSION`); nothing reads it to change behavior. The version
checks that actually degrade behavior live elsewhere: the profile `schema_version`
gates report-plugin resolution (an unsupported major falls back to the null plugin),
and the capture-spec version degrades a newer capture spec to a minimal structure-only
spec when applied.

**Provenance.** The replay markers (`replay_metadata.json`, `"replayed"`,
`agent_invocation`) are part of this contract and are consumed by
`scenario_summaries.py` and `reports/scenario_report.py`.

**Rule.** New reporting capability is added by enriching the evidence schema +
a generic consumer — never by reaching back into runtime (Stage 2–3) internals.
A plugin that needs SDK-specific artifacts the generic schema does not carry
declares them in its `EvidenceCaptureSpec` (§4.1); Stage 3 persists them; the
plugin reads them read-only at report time. Interpretation never re-runs the SDK
and never mutates the evidence — the regeneration guarantee holds.

---

## 6. The stages

The pipeline (§3) is five stages. The two contracts above are what they read and
write; the stages themselves are mechanics.

- **Stage 1 — Build.** Build the image, install the SDK wheel/skills, and embed
  the profile metadata block (§4.3) + serialized capture spec into image
  metadata.
- **Stage 2 — Execution.** Compile the run plan, launch the container, drive the
  agent, record process/timing/usage/activity. Knows no SDK report semantics.
- **Stage 3 — Evidence Capture.** *Generic* in-container code applies the
  serialized capture spec and writes immutable artifacts (Contract B). Concretely
  it runs in `post_process()` (`agent_run.py`), after
  `persist_container_runtime_metadata()` and before `capture_workspace_delta()`.
  This is the single point where declarative capture data turns into persisted
  artifacts — and the only place the capture spec is consulted.
- **Stage 4 — Interpretation.** Resolve the plugin by captured id (§4.2), call
  `collect()` to derive a per-run **`PluginEvidence`** sidecar, then ask the
  plugin for structure signals, participant/metric semantics, and narrative. The
  generic engine owns the comparison, timing/token/activity rollups, generic
  metric *extraction*, and markdown, and merges plugin output in. Interpretation
  reads `RunEvidence` (Contract B) but never mutates it (§5).
- **Stage 5 — Report Product.** Renders the comparison report from the typed
  evidence and the plugin's derived view: the generic engine formats; the plugin
  contributes vocabulary and whole sections.

### Stage 4 detail: the report plugin (interpretation CODE)

The plugin runs only at report time, read-only over evidence, returning a
derived `PluginEvidence` sidecar — it never touches Contract B. Its methods supply
the SDK's vocabulary, metric assessment, and structure scoring, plus `sections()`
for whole plugin-contributed sections (the composition system below).

```python
@dataclass(frozen=True)
class PluginEvidence:                # derived sidecar, paired with a RunEvidence at report time
    structure: StructureSignal | None
    sdk_activity: SdkActivitySignal | None
    metric: MetricAssessment | None
    extra: Mapping                   # SDK-specific facts parsed from captured artifacts

class ReportPlugin(ABC):
    # --- current milestone: helpers used by the existing report ---
    def collect(self, run: RunEvidence) -> PluginEvidence: ...   # read-only over captured evidence
    def participant_model(self) -> ParticipantModel: ...          # site/client, server/FL-level
    def assess_metric(self, run: RunEvidence,
                      expected: ExpectedMetric) -> MetricAssessment: ...
    def score_structure(self, run: RunEvidence) -> StructureSignal: ...
    def detect_sdk_activity(self, run: RunEvidence) -> SdkActivitySignal: ...
    def explain(self, cmp: ComparisonEvidence,
                plugin: Mapping[str, PluginEvidence]) -> list[NarrativeFragment]: ...

    # --- whole-section composition (E1b; see Report composition below) ---
    def sections(self, cmp: ComparisonEvidence,
                 plugin: Mapping[str, PluginEvidence]) -> list[ReportSection]: ...
```

The interface deliberately carries the **domain vocabulary**
(`participant_model`, `assess_metric`) so the generic engine stays
domain-neutral: it knows "an expected metric and N participant runs," and the
plugin defines what site/client/server/FL-level mean. The FLARE plugin and a
null/default plugin are implemented (the default returns a flat participant model
and empty sidecars). `sections()` whole-section composition is supported: the FL
Algorithm section is contributed by the NVFLARE plugin, not the engine.

### Report composition

> Report composition is **implemented** (E1b): the report is a generic skeleton with
> named anchors into which a plugin slots whole sections via `sections()`, so which
> plugin is active changes *which sections exist*, not just their wording. The shape
> below is the as-built design.

A report is a **generic skeleton with named anchors**, into which the
plugin slots **whole dynamic sections** — so which plugin is active changes
*which sections exist*, not just their wording.

```
Generic skeleton                       Plugin sections (ordered at anchors)
──────────────────                     ─────────────────────────────────────
# Summary / verdict
# Run identity table
# Comparison (timing, tokens, activity)──┐
   <anchor: comparison.inline>           ├─ explain() fragments merged inline
# Metric assessment                      │
   <anchor: after.metrics>  ────────────►├─ ReportSection("FL convergence", …)
# (anchor: sdk.details) ────────────────►├─ ReportSection("Simulator timeline", …)
                                         └─ ReportSection("Recipe structure", …)
# Artifacts / provenance
```

A `ReportSection` is `(id, title, body, anchor, placement, order)`; the engine
merges each section at its named anchor (insert-only) and orders deterministically by
`(anchor, placement, order, id)`. Sections are pure functions of `ComparisonEvidence` + the
derived `PluginEvidence` sidecars — no live SDK, no Docker — so `report`
regeneration still works from `RESULT_ROOT` alone. A result root predating a
capture rule renders the affected section as "evidence not captured" rather than
failing.

## 7. Post-Stage-3 additions (evaluation rules, agentic RCA)

Two components sit OUTSIDE the Stage 1-3 capture pipeline and its contracts:

**Evaluation rules (`benchmark/harness/evaluation.py` + `benchmark/config/evaluation/`).**
A neutral leaf (stdlib + PyYAML only; never imports the engine or SDK plugins)
that scores detected evidence signals against rules composed from a per-SDK
tree: `config/evaluation/<sdk>/index.yaml` (manifest: tasks, overlay
dimensions, scoring defaults) -> the task's `compose` documents (a `shared:`
ref resolves from the SDK-agnostic `config/evaluation/common/` layer) -> one
overlay per task-declared dimension (e.g. `framework`), with whole-signal
replacement. Detection (evidence strings) stays in SDK detectors; judgment
(verdict/points/thresholds) is data. The same rules power the report engine
and the standalone CLI (`python -m benchmark.harness.evaluation`). Composed
documents may declare `schema_version`; a mismatch is an error, never a
silent fallthrough.

**Agent-driven RCA (`benchmark/harness/rca.py`).** An OPTIONAL post-run tool:
a deterministic seed names an observation (terminal failure signature,
elapsed/token delta, or a custom question), then an investigator agent CLI
(read-only tools, allowlisted env, cwd = a staged symlink-free evidence
copy) answers iterative questions from the
captured evidence until it declares a conclusion. The Q/A trail
(`rca/investigation_<topic>.jsonl`) is written incrementally; the synthesized
report lands in `rca/rca_report_<topic>.md`. These artifacts are
AGENT-AUTHORED INTERPRETATION, not Stage-3 capture: the loader carries them in
the raw bundle only (never as typed `RunEvidence` fields), and the Why section
embeds them sanitized and labeled unverified, alongside — never instead of —
the deterministic evidence chain. Prompts sent to the investigator delimit
captured text as untrusted data (see `_UNTRUSTED_DATA_PREAMBLE`).

**Contract B additions.** `RunEvidence` gained `prompt_text`/`prompt_metadata`
— genuinely captured Stage-3 artifacts (`prompt.txt`, `prompt_metadata.json`)
— consistent with the captured-only rule above. Per-agent session-evidence
layout (codex model recovery from `sessions/` rollouts) is declared by the
agent config (`session_evidence_dir` in `config/agents/<agent>.yaml`), not
hardcoded in the generic container runner.
