# Handoff: add a "device selection" conversion-quality criterion (NVFLARE side)

Goal: mark a conversion **bad** when the source training code uses the GPU
when available but the converted NVFLARE job runs CPU-only; **good** when the
conversion preserves GPU-when-available; and **not penalized** when the source
never used the GPU.

This lives entirely in the NVFLARE repo's evals input
(`dev_tools/agent/skill_evals/<skill>/evals.json` + the checker that produces
per-run status). The benchmark harness consumes it as a verified input — it
does **not** judge device correctness itself.

There are two halves.

---

## Half 1 — Declare the behavior (report row + scoring signal)

In the **convert** skill's `evals.json`, add a behavior under a case's
`nvflare` block. The declaration is the **list** form (`id` + `description`).

Skill-to-task routing is declared in the harness's evaluation manifest, not by
naming convention: each task in
`benchmark/config/evaluation/nvflare/index.yaml` lists `native_skills` glob
patterns, and a skill's evals feed the task whose pattern matches its
`skill_name` (`*-convert-*` is conversion's declared pattern; a name-substring
fallback exists only for manifests that predate `native_skills`). A new skill
family = one pattern line on its task entry — do not rely on the naming
convention.

```json
{
  "skill_name": "nvflare-convert-pytorch",
  "evals": [
    {
      "id": "cifar10-convert",
      "nvflare": {
        "mandatory_behavior": [
          {
            "id": "device-selection-respects-availability",
            "description": "Training selects CUDA when torch.cuda.is_available() and falls back to CPU; must not hard-code CPU when the source used GPU-when-available."
          }
        ]
      }
    }
  ]
}
```

The harness build converts this into a scoring signal
`mandatory_behavior__device-selection-respects-availability` and a Generated
Code Quality row labeled from the description — even before any evidence is
captured (it renders as "not captured" until Half 2 emits a status).

Use `mandatory_behavior` so an unmet criterion scores **bad**. Use
`optional_behavior` instead if you only want to reward it (unmet → caution/
unknown, never bad).

---

## Half 2 — Emit a per-run status (the checker)

Something on the NVFLARE evals side must judge each conversion and write a
status into the **run record**, keyed by the same `id`, in the **dict** form
the harness reads:

```json
{
  "mandatory_behavior": {
    "device-selection-respects-availability": {
      "status": "pass",
      "evidence": "generated train.py uses device = 'cuda' if torch.cuda.is_available() else 'cpu'; matches source"
    }
  }
}
```

### Status → verdict contract (mandatory behavior)

| emitted `status`  | report verdict | when to emit it                                             |
| ----------------- | -------------- | ----------------------------------------------------------- |
| `pass`            | good           | conversion selects GPU-when-available (matches the source)  |
| `fail`            | bad            | source used GPU-when-available; conversion is CPU-only       |
| `missing`         | bad            | no device-selection logic found in the conversion at all     |
| `not_applicable`  | unknown        | **source itself never used the GPU** — do not penalize       |

The `not_applicable` branch is what makes "bad" conditional on the *source*
having used GPU-when-available — express that condition here in the checker,
not in the harness.

### Checker sketch (static, PyTorch)

```python
import re

# "use GPU when available, else CPU"
_AVAILABILITY_GATE = re.compile(
    r"torch\.cuda\.is_available\(\)|torch\.accelerator\b", re.IGNORECASE
)
# any GPU intent at all
_GPU_INTENT = re.compile(
    r"\.cuda\(\)|\.to\(\s*[\"']cuda|device\s*=\s*[\"']cuda|torch\.cuda", re.IGNORECASE
)
# hard-coded CPU
_HARD_CPU = re.compile(
    r"device\s*=\s*[\"']cpu[\"']|\.to\(\s*[\"']cpu[\"']\)|\.cpu\(\)", re.IGNORECASE
)


def evaluate_device_selection(source_code: str, converted_code: str) -> dict:
    if not _GPU_INTENT.search(source_code):
        return {
            "status": "not_applicable",
            "evidence": "source training code does not use the GPU; device handling not required",
        }
    if _AVAILABILITY_GATE.search(converted_code):
        return {
            "status": "pass",
            "evidence": "conversion selects CUDA when available and falls back to CPU, matching the source",
        }
    if _GPU_INTENT.search(converted_code):
        # GPU usage preserved but hard-coded: no availability gate means the
        # job crashes on CPU-only boxes — NOT the criterion's "GPU-when-available".
        return {
            "status": "fail",
            "evidence": "conversion hard-codes CUDA without a cuda.is_available() gate (no CPU fallback)",
        }
    if _HARD_CPU.search(converted_code):
        return {
            "status": "fail",
            "evidence": "source uses GPU-when-available but the conversion runs CPU-only (no cuda.is_available gate)",
        }
    return {"status": "missing", "evidence": "no device-selection logic found in the conversion"}
```

### Stronger signal: use runtime evidence too

Static regex is a floor. If the eval actually runs the converted job on a
GPU-capable box, the training logs usually print the device (e.g.
`device=cuda:0` vs `device=cpu`). A conversion that logs `device=cpu` while a
GPU is present is a definitive `fail` — more reliable than reading the code.
Prefer runtime evidence when available and fall back to the static check.

Note the checker needs BOTH the original source and the generated conversion
in scope to distinguish `fail` from `not_applicable`.

---

## Summary for the NVFLARE team

1. Add the `device-selection-respects-availability` behavior to the relevant
   `*-convert-*` skill's `evals.json` (`mandatory_behavior`, id + description).
2. In the evals runner, compute a status (`pass`/`fail`/`missing`/
   `not_applicable`) per the contract above and write it into the run record
   under `mandatory_behavior.<id> = {status, evidence}`.
3. The benchmark harness scores it automatically: `pass` → good, `fail`/
   `missing` → bad, `not_applicable` → unknown.
