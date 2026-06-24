# Architecture — visual overview

Companion to [`architecture.md`](architecture.md). Diagrams of the major system
components (the build → run → report pipeline) and the key design components (the
SDK plugin seam, the two contracts, and how a new SDK plugs in without engine changes).

> Rendered with Mermaid — GitHub renders fenced `mermaid` code blocks natively.

---

## 1. Conceptual overview — two contracts around a five-stage pipeline

The architecture's center of gravity is **two contracts** bracketing the pipeline.
**Contract A (Profile & Identity)** is declared once and consumed by every stage; the
**five stages** run Build → Execution → Evidence Capture → Interpretation → Report;
**Contract B (Evidence)** is what Capture *writes* and Interpretation *reads* — the stable
replay/report boundary, so the report can be regenerated offline from a result root.
(This is the Mermaid form of the ASCII diagram in
[`architecture.md`](architecture.md), "Architecture overview".)

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontFamily':'Inter, Segoe UI, Helvetica, Arial, sans-serif','fontSize':'13px','lineColor':'#94a3b8','clusterBkg':'#ffffff','clusterBorder':'#cbd5e1'}}}%%
flowchart TB
  subgraph CA["CONTRACT A - Profile and Identity (declared once)"]
    direction LR
    aid["Identity<br/>sdk_name, benchmark_profile_id,<br/>report_plugin_id"]
    aspec["EvidenceCaptureSpec<br/>structure files, runtime sources,<br/>artifact globs (declarative DATA)"]
    ares["Resolution<br/>report_plugin_id -> plugin registry<br/>(null/default fallback)"]
    aver["Versions<br/>schema_version, capture_spec_version"]
    aid --- aspec
    aspec --- ares
    ares --- aver
  end

  subgraph PIPE["The five-stage pipeline"]
    direction LR
    b["Build"] --> e["Execution"] --> c["Evidence Capture"] --> i["Interpretation"] --> r["Report Product"]
  end

  subgraph CB["CONTRACT B - Evidence (loaded from RESULT_ROOT; offline report lives here)"]
    direction LR
    bre["RunEvidence<br/>(per run, versioned)"]
    bce["ComparisonEvidence<br/>schema_version, runs by mode"]
    bschema["Result-root schema"]
    bprov["Provenance / replay metadata"]
    bre --- bce
    bce --- bschema
    bschema --- bprov
  end

  CA -. "feeds every stage" .-> PIPE
  PIPE -- "writes / reads" --> CB

  %% hide the intra-contract connector lines (indices 0-2 = Contract A, 7-9 = Contract B)
  linkStyle 0,1,2,7,8,9 stroke:transparent,stroke-width:0px
  classDef contract fill:#eef2ff,stroke:#6366f1,color:#1e293b;
  classDef stage fill:#f8fafc,stroke:#64748b,color:#1e293b;
  class aid,aspec,ares,aver,bre,bce,bschema,bprov contract
  class b,e,c,i,r stage
```

Contract A is consumed per stage: **Identity** is baked in at Build and resolves the
plugin at Interpretation; the **EvidenceCaptureSpec** is applied generically at Evidence
Capture. Evidence Capture **writes** Contract B and Interpretation **reads** it.

---

## 2. System components & data flow

The benchmark answers one question: **do an SDK's agent skills make an agent do a real
conversion task better?** You supply the SDK, an agent, and a job; the framework builds
two images (skills off / on), runs the agent in each, captures evidence, and reports the
comparison.

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontFamily':'Inter, Segoe UI, Helvetica, Arial, sans-serif','fontSize':'13px','lineColor':'#94a3b8','clusterBkg':'#ffffff','clusterBorder':'#cbd5e1'}}}%%
flowchart TB
  subgraph IN["Inputs - you provide"]
    sdkrepo["SDK source repo<br/>(e.g. NVFLARE)"]
    sdkprof["SDK profile<br/>config/sdks/*.yaml"]
    agprof["Agent profile<br/>Codex / Claude"]
    job["Job folder + prompt<br/>(the conversion task)"]
  end

  subgraph BUILD["Build - host/build.py"]
    skills["SDK agent skills"]
    wheel["SDK wheel<br/>(reuse / --rebuild)"]
    img0["Baseline image<br/>SDK + agent, skills OFF"]
    img1["Skills image<br/>SDK + agent, skills ON"]
    idmeta["Identity + capture spec<br/>baked into image metadata"]
  end

  subgraph RUN["Run a pair - host/runner.py (isolated containers)"]
    agent["Agent CLI<br/>(Codex / Claude)"]
    model[["Model API"]]
    convert["Converts the job<br/>using the SDK (skills on/off)"]
    capture["Generic capture<br/>(applies the capture spec)"]
    evidence[("Evidence<br/>records / artifacts /<br/>agent events / workspace delta")]
  end

  subgraph REP["Reporting - reports/"]
    plugin["SDK report plugin<br/>(resolved by identity)"]
    engine["Generic report engine"]
    report["Diagnostic 'why'-focused report<br/>(clues to optimize the skills)<br/>benchmark_insights.md / metrics_report.md"]
  end

  sdkrepo --> wheel
  sdkrepo --> skills
  sdkprof --> wheel
  agprof --> agent
  wheel --> img0
  wheel --> img1
  skills --> img1
  img0 --> idmeta
  img1 --> idmeta
  img0 --> agent
  img1 --> agent
  job --> convert
  agent <--> model
  agent --> convert
  convert --> capture --> evidence
  idmeta -. "captured identity" .-> plugin
  evidence --> engine
  plugin --> engine --> report

  classDef in fill:#eef2ff,stroke:#6366f1,color:#1e293b;
  classDef build fill:#dcfce7,stroke:#16a34a,color:#14532d;
  classDef agent fill:#fef3c7,stroke:#d97706,color:#78350f;
  classDef sdk fill:#fae8ff,stroke:#c026d3,color:#701a75;
  classDef data fill:#f1f5f9,stroke:#475569,color:#1e293b;
  classDef out fill:#dbeafe,stroke:#2563eb,color:#1e3a8a;

  class sdkrepo,sdkprof,agprof,job in
  class wheel,img0,img1,idmeta build
  class skills,plugin sdk
  class agent,model,convert agent
  class capture,evidence data
  class engine,report out
```

---

## 3. The SDK plugin seam

The report engine is **generic**. All SDK-specific meaning — vocabulary, metric
assessment, whole sections — comes from a **report plugin** resolved by the *captured
identity*, and is read as a per-run **sidecar**; the engine never calls the SDK by name.

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontFamily':'Inter, Segoe UI, Helvetica, Arial, sans-serif','fontSize':'13px','lineColor':'#94a3b8','clusterBkg':'#ffffff','clusterBorder':'#cbd5e1'}}}%%
flowchart TB
  root[("Result root +<br/>captured evidence")]
  root --> idblk["Contract A identity<br/>report_plugin_id / capture_spec_version"]
  idblk --> resolve{{"resolve_report_plugin"}}
  resolve --> nul["NullReportPlugin<br/>(default - flat / unknown SDK)"]
  resolve --> nv["NvflareReportPlugin<br/>(selected by captured id)"]

  root --> cb["Contract B - typed render input<br/>ComparisonEvidence -> RunEvidence"]

  nul --> coll["plugin.collect(run)"]
  nv --> coll
  coll --> side["PluginEvidence sidecar (per run)<br/>metric / structure / algorithm /<br/>job-execution / code-quality"]

  cb --> ctx["ReportContext"]
  side --> ctx
  ctx --> engine["Generic engine - reports/insights/*<br/>(reads typed evidence + sidecar)"]
  nv -. "sections()" .-> comp["Section composer<br/>generic blocks + ReportSection merge"]
  engine --> comp
  comp --> out["Report markdown"]

  classDef data fill:#f1f5f9,stroke:#475569,color:#1e293b;
  classDef sdk fill:#fae8ff,stroke:#c026d3,color:#701a75;
  classDef eng fill:#e0e7ff,stroke:#4f46e5,color:#312e81;
  classDef out fill:#dbeafe,stroke:#2563eb,color:#1e3a8a;

  class root,cb,side,ctx data
  class idblk,resolve,nul,nv sdk
  class engine,comp eng
  class out out
```

**Plugin hooks** (`sdks/report_plugin.py`): `collect` → `PluginEvidence`;
`participant_model` (e.g. site/server vs worker/coordinator vocabulary); `assess_metric`;
`score_structure`; `detect_sdk_activity`; `explain` (narrative fragments); `sections`
(`ReportSection`, insert-only composition); `section_copy` (bounded vocabulary embedded
inside generic sections); `metric_log_patterns`.

---

## 4. How a new SDK plugs in without changing the engine

The load-bearing rule: both the report engine and the SDK plugins depend only on shared,
SDK-agnostic helpers, and **the engine and an SDK plugin never import each other by name**.
That is what lets a new SDK be added without touching the engine.

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontFamily':'Inter, Segoe UI, Helvetica, Arial, sans-serif','fontSize':'13px','lineColor':'#94a3b8','clusterBkg':'#ffffff','clusterBorder':'#cbd5e1'}}}%%
flowchart TB
  subgraph SDKP["SDK plugins - sdks/* (e.g. nvflare/)"]
    rp["ReportPlugin ABC + ReportSection"]
    nvp["nvflare/plugin.py<br/>builds its sections from its own<br/>AlgorithmSignal + neutral leaves"]
  end

  subgraph ENG["Generic report engine - reports/insights/*"]
    eng["reads typed RunEvidence + PluginEvidence<br/>no SDK imported by name"]
  end

  subgraph LEAF["Neutral leaves - SDK-agnostic"]
    txt["_text / formatters"]
    evx["evidence / _events / _runs"]
    qs["quality_signals"]
    cspec["capture_spec"]
  end

  facade["benchmark_insights facade"] --> ENG
  ENG --> LEAF
  SDKP --> LEAF
  SDKP -. "resolved by identity,<br/>read as sidecar (runtime, not import)" .-> ENG

  classDef sdk fill:#fae8ff,stroke:#c026d3,color:#701a75;
  classDef eng fill:#e0e7ff,stroke:#4f46e5,color:#312e81;
  classDef leaf fill:#f1f5f9,stroke:#475569,color:#1e293b;
  classDef fac fill:#dbeafe,stroke:#2563eb,color:#1e3a8a;

  class rp,nvp sdk
  class eng eng
  class txt,evx,qs,cspec leaf
  class facade fac
```

**Forbidden edges (guarded by tests):** no `sdks/*` → `reports/insights/*` or
`benchmark_insights` import (a plugin never imports the engine), and no
`reports/insights/*` → SDK-by-name import (the engine never names an SDK at module load).

---

## Three core design principles

| Principle | What it means |
|---|---|
| **Typed evidence is the render input** | the engine renders from one typed evidence object (`RunEvidence`), never from raw dicts |
| **Meaning lives in the plugin** | SDK vocabulary, metric assessment, and section content come from the plugin, not the engine |
| **The default is neutral** | with no SDK identity the report uses a neutral (`Null`) plugin; a specific SDK is selected only by the *captured identity* |

Capture is **declarative data** (the capture spec, applied by generic in-container code);
interpretation is **code** (the report plugin). See [`architecture.md`](architecture.md)
for the full narrative and section-level detail.
