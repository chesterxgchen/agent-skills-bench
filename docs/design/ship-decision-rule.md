# Skill Ship/Keep Decision Rule

The benchmark reports stay diagnostic — they explain *why* a run behaved as it
did. Catalog decisions ("does a proposed skill ship? does an existing skill
stay?") need a policy layer on top of those diagnostics. This document is that
policy. It is deliberately thin: the measurements come from the harness; only
the thresholds live here.

## Inputs (all produced by the harness today)

Per scenario cell (agent × model × job × mode), from `scenario_summary.json`:

- `pass_at_k` per mode — quality-gated success rates over `--repeats` runs
  (`pass@1` = per-attempt rate; `pass@n` = any repetition succeeded).
- `paired_deltas` — per-task with-skills-minus-baseline deltas for success,
  `agent_elapsed_seconds`, and `token_count`, with deterministic bootstrap 95%
  CIs. Pairing is agent+model+workflow+job+repeat, so task difficulty cancels.
- `infrastructure_tainted_count` — provider-stall-affected runs, already
  excluded from latency winner selection.
- Quality-gate details per run (result metric present, job executed, source
  input unmodified, critical checks).

## The rule

A skill **ships** (new) or **stays** (existing) when, over the held-out corpus
with at least 3 repeats per cell:

1. **Effectiveness** — `success_delta_mean > 0` and its 95% CI excludes 0, or
   pass@1 improves by ≥ 10 points aggregate with the CI lower bound ≥ 0; and
2. **No harm** — the negative/refusal task slice shows no increase in
   harmful-action rate (wrong-framework conversions attempted, destructive
   host repairs, refusal failures); and
3. **Acceptable cost** — token and latency paired deltas are either favorable
   or within +20% with the effectiveness bar met (a skill may buy quality with
   tokens, but the report must say so explicitly).

A skill is **flagged for rework** (not removed) on one failed bar; **removed**
when it fails the effectiveness bar on two consecutive corpus evaluations.

## Measurement discipline

- **Held-out corpus only.** `dev_tools/agent/skill_evals` suites are authored
  alongside the skills — they are conformance criteria, not impact evidence.
  Impact numbers measured on author-visible tasks stop meaning UX (Goodhart).
- **Infrastructure-tainted runs never count** toward latency bars; if taint
  strips a cell below 3 clean repeats, rerun before deciding.
- **Same wheel both arms.** The build enforces byte-identical SDK wheels
  across images; a decision made on mismatched builds is void.
- **Decisions cite the run roots.** Every ship/keep call records the
  scenario_summary paths it was computed from.
