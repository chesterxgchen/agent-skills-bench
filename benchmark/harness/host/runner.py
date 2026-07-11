# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-side benchmark orchestration CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

from ..acceptance import ACCEPTANCE_CHECKS_FILENAME, apply_acceptance_gates
from ..agent_identity import preferred_agent_model
from ..agents.parsers import parse_cached_usage_and_activity
from ..agents.registry import DEFAULT_BENCHMARK_AGENT, load_agent_adapter
from ..common import load_json, write_json
from ..host_environment import host_os_display, read_host_environment, write_host_environment
from ..modes import PAIR_RUNS
from ..profile_metadata import MODE_METADATA_FILENAME, clear_root_descriptor, write_root_descriptor
from ..reports import benchmark_insights, metrics_report
from ..scenarios import (
    ScenarioCompilation,
    ScenarioValidationError,
    compile_scenario,
    compile_scenario_file,
    slugify,
    validate_path_budget,
    write_scenario_summaries,
)
from .common import (
    CONTAINER_PROMPT_PATH,
    CaseConfig,
    ImageConfig,
    absolute_path,
    add_agent_auth_mounts,
    add_agent_passthrough_env,
    default_results_root,
    docker_args_for_case,
    docker_env,
    emit,
    host_idle_sleep_prevention_command,
    parse_host_cli_options,
    prepare_result_mount,
    stream_command,
    timestamp_slug,
    write_runtime_image,
)

STALE_RESULT_FILES = (
    "comprehensive_report.json",
    "comprehensive_report.md",
    "metrics_plots.png",
    "metrics_plots.svg",
    "metrics_summary.json",
    "metrics_report.html",
    "metrics_report.json",
    "metrics_report.md",
    "benchmark_insights.md",
    "pair_summary.json",
    "process_eval_ablation_summary.json",
    "report_generator_status.json",
    "skill_benchmark.json",
    "skill_benchmark.md",
    "skill_performance.json",
    "skill_performance.txt",
    "skill_report_status.json",
    "scenario.json",
    "run_plan.json",
    "scenario_summary.json",
)
STALE_RESULT_DIRS = (
    "process_eval_runs",
    "records",
    "reports",
    "with_skills_eval_off",
    "with_skills_eval_on",
)
BENCHMARK_METRICS_TITLE = "Agent Skills Benchmark Metrics"
FAILURE_STDERR_LINE_LIMIT = 4
FAILURE_STDERR_CHAR_LIMIT = 1200


@dataclass(frozen=True)
class InteractiveRuntimeConfig:
    agent: str
    agent_model: str
    model_was_explicit: bool

    @property
    def unattended(self) -> bool:
        # An interactive `docker run -it` debug shell is never unattended, so it
        # must not receive the harness->skill unattended-mode env (unattended_env
        # in the agent config). A skill run from this shell keeps its approval
        # prompts. Benchmark case configs do not define this and default to True.
        return False


@dataclass(frozen=True)
class ScenarioCliOptions:
    scenario_path: Path
    results_root: Path | None = None
    result_root: Path | None = None
    agent_home: Path | None = None
    mount_agent_auth: bool | None = None


@dataclass(frozen=True)
class ReportCliOptions:
    result_root: Path
    diagnostics: bool = False


@dataclass(frozen=True)
class RuntimeAuthOptions:
    agent_home: Path | None = None
    mount_agent_auth: bool | None = None

    @property
    def has_overrides(self) -> bool:
        return self.agent_home is not None or self.mount_agent_auth is not None


def run_one_case(config: CaseConfig, *, logs: Iterable[Path] = (), prefix: str | None = None) -> int:
    prepare_result_mount(config.result_dir, logs=logs, prefix=prefix)
    emit(f"Running mode={config.mode} with runtime image: {config.run_image}", logs=logs, prefix=prefix)
    emit(f"Report image: {config.images.report_image_name}", logs=logs, prefix=prefix)
    emit(f"Job folder: {config.job_input_dir} -> /workspace/input", logs=logs, prefix=prefix)
    emit(f"Prompt file: {config.prompt_path} -> {CONTAINER_PROMPT_PATH}", logs=logs, prefix=prefix)
    write_runtime_image(config)
    with tempfile.TemporaryDirectory(prefix="agent-benchmark-auth-") as auth_staging:
        command = docker_args_for_case(config, logs=logs, prefix=prefix, auth_staging_dir=Path(auth_staging))
        command = host_idle_sleep_prevention_command(command)
        if command[:2] == ["caffeinate", "-i"]:
            emit("Host idle-sleep prevention enabled: caffeinate -i", logs=logs, prefix=prefix)
        status = stream_command(
            command,
            logs=logs,
            prefix=prefix,
            timeout_seconds=config.container_timeout_seconds,
        )
    write_json(config.result_dir / "container_exit_code.json", {"exit_code": status})
    if status != 0:
        emit_case_failure_summary(config, status, logs=logs, prefix=prefix)
    if enforce_result_size_budget(config, logs=logs, prefix=prefix):
        return 1
    return combined_exit_status({config.mode: status})


def truncate_text(text: object, limit: int = 240) -> str:
    rendered = str(text or "").strip().replace("\n", " ")
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 3].rstrip() + "..."


def bounded_stderr_excerpt(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    excerpt = "\n".join(lines[:FAILURE_STDERR_LINE_LIMIT])
    if len(excerpt) > FAILURE_STDERR_CHAR_LIMIT:
        excerpt = excerpt[: FAILURE_STDERR_CHAR_LIMIT - 3].rstrip() + "..."
    return excerpt


def read_agent_stderr_excerpt(result_dir: Path, exit_summary: dict[str, object]) -> str:
    excerpt = str(exit_summary.get("stderr_excerpt") or "")
    if not excerpt:
        stderr_path = result_dir / "agent_stderr.txt"
        try:
            excerpt = stderr_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            excerpt = ""
    return bounded_stderr_excerpt(excerpt)


def emit_case_failure_summary(
    config: CaseConfig,
    status: int,
    *,
    logs: Iterable[Path] = (),
    prefix: str | None = None,
) -> None:
    result_dir = config.result_dir
    run_summary = load_json(result_dir / "run_summary.json", {}) or {}
    exit_summary = load_json(result_dir / "agent_exit_summary.json", {}) or {}
    early_failure = load_json(result_dir / "early_failure.json", {}) or {}
    host_error = load_json(result_dir / "host_case_error.json", {}) or {}
    if not isinstance(run_summary, dict):
        run_summary = {}
    if not isinstance(exit_summary, dict):
        exit_summary = {}
    if not isinstance(early_failure, dict):
        early_failure = {}
    if not isinstance(host_error, dict):
        host_error = {}

    emit(
        f"Run failed: mode={config.mode}; final_status={status}; result_dir={result_dir}",
        logs=logs,
        prefix=prefix,
        stderr=True,
    )
    agent_exit = run_summary.get("agent_process_exit_code")
    final_exit = run_summary.get("final_container_exit_code")
    if agent_exit is not None or final_exit is not None:
        emit(
            f"Failure exit codes: agent_process_exit={agent_exit}; final_container_exit={final_exit}",
            logs=logs,
            prefix=prefix,
            stderr=True,
        )
    failure_category = (
        run_summary.get("failure_root_cause")
        or run_summary.get("failure_category")
        or exit_summary.get("failure_category")
    )
    if failure_category:
        emit(
            f"Failure category: {truncate_text(failure_category)}",
            logs=logs,
            prefix=prefix,
            stderr=True,
        )
    harness_message = early_failure.get("message") or host_error.get("message")
    if harness_message:
        phase = early_failure.get("phase") or host_error.get("error_type") or "host"
        emit(
            f"Harness failure detail: {phase}: {truncate_text(harness_message)}",
            logs=logs,
            prefix=prefix,
            stderr=True,
        )
    stderr_excerpt = read_agent_stderr_excerpt(result_dir, exit_summary)
    if stderr_excerpt:
        emit("Agent stderr excerpt:", logs=logs, prefix=prefix, stderr=True)
        for line in stderr_excerpt.splitlines():
            emit(f"  {line}", logs=logs, prefix=prefix, stderr=True)
    emit(
        f"Failure artifacts: {result_dir / 'agent_stderr.txt'}, {result_dir / 'agent_exit_summary.json'}, "
        f"{result_dir / 'run_summary.json'}",
        logs=logs,
        prefix=prefix,
        stderr=True,
    )


def directory_size_bytes(path: Path) -> int:
    total = 0
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        dirnames[:] = [name for name in dirnames if not (Path(dirpath) / name).is_symlink()]
        for name in filenames:
            item = Path(dirpath) / name
            try:
                if item.is_symlink() or not item.is_file():
                    continue
                total += item.stat().st_size
            except OSError:
                continue
    return total


def enforce_result_size_budget(config: CaseConfig, *, logs: Iterable[Path] = (), prefix: str | None = None) -> bool:
    budget = config.result_size_budget_bytes
    if budget is None:
        return False
    used = directory_size_bytes(config.result_dir)
    failed = used > budget
    write_json(
        config.result_dir / "result_size_budget.json",
        {
            "status": "fail" if failed else "pass",
            "result_size_bytes": used,
            "budget_bytes": budget,
        },
    )
    if failed:
        emit(
            f"Result size budget exceeded: {used} bytes > {budget} bytes.",
            logs=logs,
            prefix=prefix,
            stderr=True,
        )
    return failed


def write_host_error(path: Path, exc: BaseException) -> None:
    write_json(
        path,
        {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        },
    )


def run_case_safely(config: CaseConfig, *, logs: Iterable[Path] = (), prefix: str | None = None) -> int:
    try:
        return run_one_case(config, logs=logs, prefix=prefix)
    except Exception as exc:
        config.result_dir.mkdir(parents=True, exist_ok=True)
        write_host_error(config.result_dir / "host_case_error.json", exc)
        emit(
            f"Case failed before completion: {type(exc).__name__}: {exc}",
            logs=logs,
            prefix=prefix,
            stderr=True,
        )
        return 1


def comparison_result_root(options, *, default_prefix: str | None = None) -> Path:
    if options.result_root is not None:
        return options.result_root
    timestamp = timestamp_slug()
    default_name = f"{default_prefix}_{timestamp}" if default_prefix else timestamp
    if options.results_root is not None:
        return options.results_root / default_name
    return absolute_path(os.environ.get("RESULT_ROOT", str(default_results_root() / default_name)))


def scenario_result_root(options: ScenarioCliOptions, compilation: ScenarioCompilation) -> Path:
    if options.result_root is not None:
        return options.result_root
    slug = compilation.scenario.get("scenario_slug") or slugify(str(compilation.scenario.get("name") or "scenario"))
    default_name = f"{slug}_{timestamp_slug()}"
    if options.results_root is not None:
        return options.results_root / default_name
    return absolute_path(os.environ.get("RESULT_ROOT", str(default_results_root() / default_name)))


def parse_scenario_cli_options(argv: list[str]) -> ScenarioCliOptions:
    scenario_path: Path | None = None
    results_root: Path | None = None
    result_root: Path | None = None
    agent_home: Path | None = None
    mount_agent_auth: bool | None = None
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--results-root" or arg.startswith("--results-root="):
            value, index = _scenario_option_value(argv, index, "--results-root")
            if results_root is not None:
                raise SystemExit("Expected only one --results-root")
            results_root = absolute_path(value)
        elif (
            arg in {"--output-dir", "--result-root"}
            or arg.startswith("--output-dir=")
            or arg.startswith("--result-root=")
        ):
            option = "--output-dir" if arg.startswith("--output-dir") else "--result-root"
            value, index = _scenario_option_value(argv, index, option)
            if result_root is not None:
                raise SystemExit("Expected only one exact output directory")
            result_root = absolute_path(value)
        elif arg == "--agent-home" or arg.startswith("--agent-home="):
            value, index = _scenario_option_value(argv, index, "--agent-home")
            if agent_home is not None:
                raise SystemExit("Expected only one --agent-home")
            agent_home = absolute_path(value)
        elif arg == "--no-agent-auth-mount":
            if mount_agent_auth is False:
                raise SystemExit("Expected only one --no-agent-auth-mount")
            mount_agent_auth = False
            index += 1
        elif arg in {"-h", "--help"}:
            print(
                "Usage: run.sh scenario SCENARIO.yaml "
                "[--results-root PATH|--output-dir PATH] [--agent-home PATH] [--no-agent-auth-mount]"
            )
            raise SystemExit(0)
        elif arg.startswith("-"):
            raise SystemExit(f"Unknown scenario option: {arg}")
        else:
            if scenario_path is not None:
                raise SystemExit("Expected only one scenario YAML file")
            scenario_path = absolute_path(arg)
            index += 1
    if scenario_path is None:
        raise SystemExit("Scenario YAML file is required.")
    if not scenario_path.is_file():
        raise SystemExit(f"Scenario YAML file must exist: {scenario_path}")
    if results_root is not None and result_root is not None:
        raise SystemExit("Use --results-root or --output-dir, not both.")
    return ScenarioCliOptions(
        scenario_path=scenario_path,
        results_root=results_root,
        result_root=result_root,
        agent_home=agent_home,
        mount_agent_auth=mount_agent_auth,
    )


def parse_report_cli_options(argv: list[str]) -> ReportCliOptions:
    result_root: Path | None = None
    diagnostics = False
    index = 0
    while index < len(argv):
        arg = argv[index]
        if (
            arg in {"--result-root", "--output-dir"}
            or arg.startswith("--result-root=")
            or arg.startswith("--output-dir=")
        ):
            option = "--result-root" if arg.startswith("--result-root") else "--output-dir"
            value, index = _scenario_option_value(argv, index, option)
            if result_root is not None:
                raise SystemExit("Expected only one report result root")
            result_root = absolute_path(value)
        elif arg == "--diagnostics":
            diagnostics = True
            index += 1
        elif arg in {"-h", "--help"}:
            print(
                "Usage: run.sh report RESULT_ROOT [--diagnostics]\n\n"
                "  --diagnostics   Backfill MISSING agentic diagnostics (auto-RCA and\n"
                "                  code-quality evaluation) for the existing result root,\n"
                "                  then regenerate reports. Existing diagnostic outputs\n"
                "                  are kept; only killed/skipped ones run."
            )
            raise SystemExit(0)
        elif arg.startswith("-"):
            raise SystemExit(f"Unknown report option: {arg}")
        else:
            if result_root is not None:
                raise SystemExit("Expected only one report result root")
            result_root = absolute_path(arg)
            index += 1
    if result_root is None:
        raise SystemExit("Report result root is required.")
    if not (result_root / "run_plan.json").is_file():
        raise SystemExit(f"Report result root must contain run_plan.json: {result_root}")
    return ReportCliOptions(result_root=result_root, diagnostics=diagnostics)


def _scenario_option_value(argv: list[str], index: int, option: str) -> tuple[str, int]:
    arg = argv[index]
    if arg.startswith(f"{option}="):
        return arg.split("=", 1)[1], index + 1
    if index + 1 >= len(argv):
        raise SystemExit(f"{option} requires a path")
    return argv[index + 1], index + 2


def reject_parallel_comparison_runs(command: str) -> None:
    parallel = os.environ.get("PARALLEL_CASES", "false").strip().lower()
    if parallel not in {"", "0", "false", "no", "off"}:
        raise SystemExit(
            f"PARALLEL_CASES is no longer supported for benchmark comparisons; {command} runs sequentially."
        )


def clean_pair_result_root(result_root: Path) -> None:
    """Remove generated artifacts from older harness layouts before a fresh pair run."""

    for spec in PAIR_RUNS:
        path = result_root / spec.mode
        if path.exists():
            shutil.rmtree(path)
    for name in STALE_RESULT_DIRS:
        path = result_root / name
        if path.exists():
            shutil.rmtree(path)
    for name in STALE_RESULT_FILES:
        path = result_root / name
        if path.exists() and path.is_file():
            path.unlink()


def _read_small_text(path: Path, *, max_bytes: int = 64_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_bytes]
    except OSError:
        return ""


def _pair_input_guidance_text(job_input: Path, prompt_path: Path) -> str:
    parts = [job_input.name, _read_small_text(prompt_path)]
    for pattern in ("README*", "readme*"):
        for path in sorted(job_input.glob(pattern)):
            if path.is_file():
                parts.append(_read_small_text(path))
    return "\n".join(part for part in parts if part)


def _looks_like_site_split_csv_dataset(job_input: Path) -> bool:
    site_csvs = []
    for site_dir in sorted(job_input.glob("site-*")):
        if site_dir.is_dir() and any(path.is_file() for path in site_dir.glob("*.csv")):
            site_csvs.append(site_dir)
    return len(site_csvs) >= 2


def _infer_pair_evaluation_metadata(options) -> dict[str, object]:
    """Infer task routing for ad hoc `pair` runs from prompt + input context."""

    text = _pair_input_guidance_text(options.job_input, options.prompt_path).lower()
    has_fedstats_language = any(
        phrase in text
        for phrase in (
            "federated stats",
            "federated statistics",
            "federated status",
            "fed stats",
            "fedstats",
            "global stats",
            "site stats",
        )
    )
    if not has_fedstats_language or not _looks_like_site_split_csv_dataset(options.job_input):
        return {}

    selectors: dict[str, str] = {}
    if any(
        phrase in text for phrase in ("no header", "no-header", "no_header", "noheader", "headerless", "without header")
    ):
        selectors["data-format"] = "no_header"

    return {
        "evaluation_task": "federated-statistics",
        "evaluation_selectors": selectors,
        "result_artifact": {
            "glob": "**/simulate_job/*stat*.json",
            "format": "json",
            "description": "aggregated federated statistics output from the simulator workspace",
        },
        "quality_gate": {"required_validation_metric_status": ["not_required"]},
        "workflow": "FEDSTATS",
    }


def pair_compilation_from_options(options) -> ScenarioCompilation:
    adapter = agent_adapter_from_options(options)
    agent_model, model_was_explicit = agent_model_from_options(adapter, options)
    agent_entry: dict[str, object] = {"name": adapter.name}
    if model_was_explicit:
        agent_entry["models"] = [agent_model]
    comparison: dict[str, object] = {"type": "mode_ablation", "modes": [spec.mode for spec in PAIR_RUNS]}
    repeats = options.repeats if options.repeats is not None else int(os.environ.get("BENCHMARK_REPEATS") or 1)
    if repeats > 1:
        comparison["repeats"] = repeats
    inferred = _infer_pair_evaluation_metadata(options)
    workflow = options.workflow or os.environ.get("BENCHMARK_WORKFLOW") or str(inferred.get("workflow") or "default")
    raw = {
        "name": f"pair {adapter.name} {options.job_input.name}",
        "prompt": str(options.prompt_path),
        "agents": [agent_entry],
        "comparison": comparison,
        "workflows": [{"name": workflow}],
        "jobs": [
            {
                "name": options.job_input.name,
                "path": str(options.job_input),
                "scale": options.job_scale
                or os.environ.get("BENCHMARK_JOB_SCALE", os.environ.get("JOB_SCALE", "small")),
            }
        ],
    }
    for key in ("evaluation_task", "evaluation_selectors", "result_artifact", "quality_gate"):
        if key in inferred:
            raw[key] = inferred[key]
    return compile_scenario(raw, base_dir=Path.cwd(), allow_external_prompt=True)


def image_config_for_agent(agent: str) -> ImageConfig:
    return image_config_from_adapter(load_agent_adapter(agent))


def image_config_from_adapter(adapter) -> ImageConfig:
    try:
        return ImageConfig.for_adapter(adapter)
    except ValueError as exc:
        raise ScenarioValidationError(str(exc)) from exc


def model_was_explicit_for_entry(entry: dict[str, object]) -> bool:
    return str(entry.get("model_source") or "") != "adapter_default"


def positive_int_resource_value(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def case_config_for_entry(
    entry: dict[str, object],
    result_root: Path,
    runtime_auth_options: RuntimeAuthOptions | None = None,
) -> CaseConfig:
    adapter = load_agent_adapter(str(entry["agent"]))
    resource_policy = entry.get("resource_policy") if isinstance(entry.get("resource_policy"), dict) else {}
    runtime_auth_options = runtime_auth_options or RuntimeAuthOptions()
    host_agent_home = runtime_auth_options.agent_home or absolute_path(str(adapter.host_home_from_env(os.environ)))
    mount_host_agent_auth = (
        runtime_auth_options.mount_agent_auth
        if runtime_auth_options.mount_agent_auth is not None
        else adapter.mount_auth_from_env(os.environ)
    )
    return CaseConfig(
        mode=str(entry["mode"]),
        use_preinstalled_skills=bool(entry["skills_enabled"]),
        job_input_dir=Path(str(entry["job_path"])),
        result_dir=result_root / str(entry["record_dir"]),
        prompt_path=Path(str(entry["prompt_source"])),
        images=image_config_for_agent(adapter.name),
        progress_interval_seconds=os.environ.get("PROGRESS_INTERVAL_SECONDS", "60"),
        agent=adapter.name,
        agent_model=str(entry["agent_model"]),
        model_was_explicit=model_was_explicit_for_entry(entry),
        adapter=adapter,
        host_agent_home=host_agent_home,
        mount_host_agent_auth=mount_host_agent_auth,
        agent_timeout_seconds=positive_int_resource_value(resource_policy.get("agent_timeout_seconds")),
        container_timeout_seconds=positive_int_resource_value(resource_policy.get("container_timeout_seconds")),
        result_size_budget_bytes=positive_int_resource_value(resource_policy.get("result_size_budget_bytes")),
    )


def inspect_docker_image(image: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return result.returncode == 0, result.stderr.strip()


def docker_context_name() -> str:
    try:
        result = subprocess.run(
            ["docker", "context", "show"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        return f"unavailable ({type(exc).__name__}: {exc})"
    if result.returncode == 0:
        return result.stdout.strip() or "unknown"
    detail = result.stderr.strip() or result.stdout.strip()
    return f"unavailable ({detail})" if detail else "unavailable"


def docker_benchmark_image_list() -> list[str]:
    try:
        result = subprocess.run(
            [
                "docker",
                "image",
                "ls",
                "agent-skills-benchmark",
                "--format",
                "{{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.CreatedSince}}\t{{.Size}}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        return [f"unavailable ({type(exc).__name__}: {exc})"]
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        return [f"unavailable ({detail})"] if detail else ["unavailable"]
    return [line for line in result.stdout.splitlines() if line.strip()]


def preflight_docker_images(
    entries: Iterable[dict[str, object]],
    *,
    result_root: Path,
    logs: Iterable[Path] = (),
    runtime_auth_options: RuntimeAuthOptions | None = None,
) -> None:
    images_by_run: dict[str, list[str]] = {}
    agents_by_image: dict[str, set[str]] = {}
    inspect_results: dict[str, dict[str, object]] = {}
    docker_context = docker_context_name()
    local_benchmark_images = docker_benchmark_image_list()
    for entry in entries:
        config = case_config_for_entry(entry, result_root, runtime_auth_options)
        images_by_run.setdefault(config.run_image, []).append(str(entry.get("run_id")))
        agents_by_image.setdefault(config.run_image, set()).add(config.agent)
    for image in sorted(images_by_run):
        available, detail = inspect_docker_image(image)
        inspect_results[image] = {
            "available": available,
            "agents": sorted(agents_by_image.get(image, set())),
            "detail": detail,
            "run_ids": images_by_run[image],
        }
    missing = [image for image, result in inspect_results.items() if not result["available"]]
    payload = {
        "status": "fail" if missing else "pass",
        "docker_context": docker_context,
        "images": inspect_results,
        "local_benchmark_images": local_benchmark_images,
        "missing_images": missing,
    }
    write_json(result_root / "docker_image_preflight.json", payload)
    if not missing:
        return
    message = (
        "Benchmark Docker image(s) are missing locally or could not be inspected: "
        + ", ".join(missing)
        + f". Docker context: {docker_context}. "
        + "Selected benchmark agent(s): "
        + ", ".join(sorted({agent for image in missing for agent in agents_by_image.get(image, set())}) or ["unknown"])
        + ". "
        + "Run ./bin/build.sh from dev_tools/agent/skills/benchmark before running the benchmark, "
        + "and verify the same Docker context is used for build and run."
    )
    details = [f"{image}: {str(inspect_results[image].get('detail') or 'not found')}" for image in missing]
    if details:
        message += " Inspect details: " + "; ".join(details)
    if local_benchmark_images:
        message += " Local agent-skills-benchmark tags: " + "; ".join(local_benchmark_images)
    emit(message, logs=logs, stderr=True)
    raise ScenarioValidationError(message)


def read_image_profile_metadata(image: str) -> dict[str, object]:
    """Extract the build-baked ``sdk_wheel_metadata.json`` from a benchmark image.

    The image copy was staged by the host at build time and container writes
    cannot reach it, so it is the trusted source for the root descriptor's
    identity block and evaluation-criteria anchor — unlike the mode-dir copy,
    which sits in the container-writable result mount. Extraction uses
    ``docker create`` + ``docker cp`` (no container code executes). Returns
    ``{}`` when docker or the file is unavailable.
    """

    agent_home = "/workspace/agent-home"
    try:
        env_result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{json .Config.Env}}", image],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if env_result.returncode == 0:
            try:
                env_entries = json.loads(env_result.stdout or "[]")
            except ValueError:
                env_entries = []
            for env_entry in env_entries if isinstance(env_entries, list) else []:
                name, sep, value = str(env_entry).partition("=")
                if sep and name == "BENCHMARK_AGENT_HOME" and value:
                    agent_home = value
        create_result = subprocess.run(
            ["docker", "create", image],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError:
        return {}
    container_id = create_result.stdout.strip()
    if create_result.returncode != 0 or not container_id:
        return {}
    try:
        with tempfile.TemporaryDirectory(prefix="agent-benchmark-image-metadata-") as staging:
            target = Path(staging) / MODE_METADATA_FILENAME
            copy_result = subprocess.run(
                ["docker", "cp", f"{container_id}:{agent_home}/{MODE_METADATA_FILENAME}", str(target)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if copy_result.returncode != 0:
                return {}
            metadata = load_json(target, {}) or {}
            return metadata if isinstance(metadata, dict) else {}
    except OSError:
        return {}
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def anchor_root_descriptor_from_images(
    entries: Iterable[dict[str, object]],
    result_root: Path,
    runtime_auth_options: RuntimeAuthOptions | None = None,
    *,
    logs: Iterable[Path] = (),
) -> bool:
    """Write the root descriptor from image-baked metadata BEFORE any run.

    All of a plan's images are built from one criteria input, so the first
    image that yields the §4.3 block anchors the whole result root. Anchoring
    before the containers start means a run that rewrites its mount-resident
    metadata or rules copy can no longer influence the trust anchor; without
    an anchor the report refuses to score the mount copy ("unverifiable").
    """

    images: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        config = case_config_for_entry(entry, result_root, runtime_auth_options)
        if config.run_image not in images:
            images.append(config.run_image)
    for image in images:
        if write_root_descriptor(result_root, read_image_profile_metadata(image)):
            emit(f"Root profile descriptor anchored from image metadata: {image}", logs=logs)
            return True
    return False


def copy_file_if_present(source: Path, target: Path) -> bool:
    if not source.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return True


def canonicalize_entry_artifacts(result_root: Path, entry: dict[str, object], status: int | None) -> None:
    record_dir = result_root / str(entry["record_dir"])
    mode = str(entry["mode"])
    record_dir.mkdir(parents=True, exist_ok=True)
    write_json(record_dir / "run_plan_entry.json", entry)
    copy_file_if_present(record_dir / "records" / f"{mode}_agent_record.json", record_dir / "agent_record.json")
    copy_file_if_present(record_dir / "records" / f"{mode}_record.json", record_dir / "benchmark_record.json")
    copy_file_if_present(record_dir / "run_summary.json", record_dir / "record_summary.json")
    summary = load_json(record_dir / "record_summary.json", {}) or {}
    if not isinstance(summary, dict):
        summary = {}
    runtime_image = load_json(record_dir / "runtime_image.json", {}) or {}
    if not isinstance(runtime_image, dict):
        runtime_image = {}
    activity = load_json(record_dir / "agent_activity.json", {}) or {}
    if not isinstance(activity, dict):
        activity = {}
    host_environment = read_host_environment(result_root)
    host_os = host_os_display(host_environment)
    # The plan entry is authoritative for run identity, except the model: the
    # container resolves the real model after plan time (config file / runtime
    # evidence), so a plan-time sentinel must not clobber a resolved name.
    resolved_model, resolved_source = preferred_agent_model(
        (summary.get("agent_model"), summary.get("model_source")),
        (entry.get("agent_model"), entry.get("model_source")),
    )
    summary.update(
        {
            key: entry.get(key)
            for key in (
                "run_id",
                "scenario_name",
                "comparison_type",
                "comparison_group_id",
                "agent",
                "workflow",
                "job_name",
                "job_slug",
                "job_path",
                "job_scale",
                "evaluation_task",
                "evaluation_selectors",
                "mode",
                "skills_enabled",
                "prompt_hash",
                "prompt_source",
            )
        }
    )
    # Scenario-declared gates (result artifact + acceptance checks) evaluate
    # host-side over the captured record, so a run cannot rewrite its own gate.
    acceptance_payload = apply_acceptance_gates(record_dir, entry, summary)
    if acceptance_payload:
        write_json(record_dir / ACCEPTANCE_CHECKS_FILENAME, acceptance_payload)
    else:
        (record_dir / ACCEPTANCE_CHECKS_FILENAME).unlink(missing_ok=True)
    summary["agent_model"] = resolved_model
    if resolved_source not in (None, ""):
        summary["model_source"] = resolved_source
    summary["host_status"] = status
    summary["artifact_paths"] = entry.get("artifact_paths") or {}
    summary.setdefault("runtime_image", runtime_image.get("runtime_image"))
    summary.setdefault("wheel_variant", runtime_image.get("sdk_image_kind"))
    summary.setdefault("command_count", activity.get("command_count"))
    if host_os:
        summary["host_os"] = host_os
    if host_environment:
        summary["host_environment"] = host_environment
    write_json(record_dir / "record_summary.json", summary)
    run_summary = load_json(record_dir / "run_summary.json", {}) or {}
    if isinstance(run_summary, dict):
        if host_os:
            run_summary["host_os"] = host_os
        if host_environment:
            run_summary["host_environment"] = host_environment
        write_json(record_dir / "run_summary.json", run_summary)

    # Legacy fallback for the §4.3 identity block only: the mode-dir copy sits
    # in the container-writable result mount, so it must not carry the
    # evaluation-criteria anchor into the root descriptor and must not replace
    # a descriptor already anchored from host-side image metadata.
    mode_metadata = load_json(record_dir / MODE_METADATA_FILENAME, {}) or {}
    if isinstance(mode_metadata, dict):
        write_root_descriptor(result_root, mode_metadata, include_criteria=False, overwrite=False)


def execute_run_plan(
    compilation: ScenarioCompilation,
    *,
    result_root: Path,
    logs: Iterable[Path] = (),
    runtime_auth_options: RuntimeAuthOptions | None = None,
) -> tuple[dict[str, int], dict[str, object]]:
    path_budget = compilation.scenario.get("path_budget")
    if isinstance(path_budget, int):
        validate_path_budget(
            str(compilation.scenario.get("name") or compilation.run_plan.get("scenario_name")),
            compilation.run_plan.get("entries") if isinstance(compilation.run_plan.get("entries"), list) else [],
            path_budget,
            result_root,
        )
    execution = compilation.write(result_root)
    run_plan = execution.run_plan
    entries = run_plan.get("entries") if isinstance(run_plan.get("entries"), list) else []
    # A reused result root may carry the previous run's descriptor; clear it
    # so this run either anchors from its own images below or stays
    # unverifiable — a stale anchor must not survive into a fresh run.
    clear_root_descriptor(result_root)
    try:
        preflight_docker_images(
            entries,
            result_root=result_root,
            logs=logs,
            runtime_auth_options=runtime_auth_options,
        )
    except ScenarioValidationError as exc:
        write_scenario_summaries(
            result_root,
            {},
            harness_failure={
                "status": "failed",
                "failure_category": "harness_preflight_failure",
                "message": str(exc),
            },
        )
        raise
    anchor_root_descriptor_from_images(entries, result_root, runtime_auth_options, logs=logs)
    statuses: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        prefix = str(entry["run_id"])
        emit(
            "Starting run_id={} agent={} model={} mode={} record_dir={}".format(
                entry["run_id"],
                entry["agent"],
                entry["agent_model"],
                entry["mode"],
                entry["record_dir"],
            ),
            logs=logs,
            prefix=prefix,
        )
        config = case_config_for_entry(entry, result_root, runtime_auth_options)
        status = run_case_safely(config, logs=logs, prefix=prefix)
        statuses[str(entry["run_id"])] = status
        canonicalize_entry_artifacts(result_root, entry, status)
        emit(f"Finished run_id={entry['run_id']} status={status}", logs=logs, prefix=prefix)
        if status != 0 and run_plan.get("fail_fast"):
            emit("Stopping scenario early because fail_fast=true.", logs=logs, stderr=True)
            break
    summary = write_scenario_summaries(result_root, statuses)
    return statuses, summary


def replay_result_root(result_root: Path, *, logs: Iterable[Path] = ()) -> dict[str, object]:
    replay_metadata = {
        "schema_version": "1",
        "replayed": True,
        "agent_invocation": "replayed",
        "replayed_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_result_root": str(result_root.resolve()),
        "note": "Replay regenerates parser/report artifacts from captured records and does not execute Docker.",
    }
    write_json(result_root / "replay_metadata.json", replay_metadata)
    run_plan = load_json(result_root / "run_plan.json", {}) or {}
    entries = run_plan.get("entries") if isinstance(run_plan.get("entries"), list) else []
    statuses: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        record_dir = result_root / str(entry["record_dir"])
        events_path = record_dir / "agent_events.jsonl"
        if events_path.is_file():
            adapter = load_agent_adapter(str(entry["agent"]))
            write_json(record_dir / "agent_usage.json", adapter.parse_usage(events_path))
            write_json(record_dir / "agent_activity.json", adapter.parse_activity(events_path))
            emit(f"Replayed parsers for {entry['run_id']}: {events_path}", logs=logs)
        container_exit = load_json(record_dir / "container_exit_code.json", {}) or {}
        if isinstance(container_exit, dict) and isinstance(container_exit.get("exit_code"), int):
            statuses[str(entry["run_id"])] = int(container_exit["exit_code"])
        canonicalize_entry_artifacts(result_root, entry, statuses.get(str(entry["run_id"])))
    return write_scenario_summaries(result_root, statuses)


def write_report_generator_status(result_root: Path, statuses: dict[str, int]) -> None:
    write_json(
        result_root / "report_generator_status.json",
        {
            "status": "ok" if all(status == 0 for status in statuses.values()) else "failed",
            "exit_codes": statuses,
        },
    )


def write_benchmark_reports(result_root: Path, *, logs: Iterable[Path] = ()) -> dict[str, int]:
    statuses: dict[str, int] = {}
    try:
        try:
            metrics_report.write_reports(result_root, BENCHMARK_METRICS_TITLE)
        except Exception as exc:
            write_host_error(result_root / "metrics_report_error.json", exc)
            emit(f"Metrics report failed: {type(exc).__name__}: {exc}", logs=logs, stderr=True)
            statuses["metrics_report"] = 1
        else:
            with suppress(OSError):
                (result_root / "metrics_report_error.json").unlink()
            statuses["metrics_report"] = 0

        try:
            runs = benchmark_insights.collect_benchmark_runs(result_root)
            # Persist per-mode SDK quality scores so the SDK-agnostic RCA loop
            # can auto-gate a structure regression; best-effort, never blocks
            # the report.
            with suppress(Exception):
                benchmark_insights.persist_quality_summary(result_root, runs)
            (result_root / "benchmark_insights.md").write_text(
                benchmark_insights.benchmark_report(result_root, runs),
                encoding="utf-8",
            )
        except Exception as exc:
            write_host_error(result_root / "benchmark_insights_error.json", exc)
            emit(f"Benchmark insights report failed: {type(exc).__name__}: {exc}", logs=logs, stderr=True)
            statuses["benchmark_insights"] = 1
        else:
            with suppress(OSError):
                (result_root / "benchmark_insights_error.json").unlink()
            statuses["benchmark_insights"] = 0

        write_report_generator_status(result_root, statuses)
        return statuses
    finally:
        parse_cached_usage_and_activity.cache_clear()


def _mode_has_rca_report(result_root: Path, mode: str) -> bool:
    from ..reports._loader import mode_dir_for_benchmark

    rca_dir = mode_dir_for_benchmark(result_root, mode) / "rca"
    return rca_dir.is_dir() and any(rca_dir.glob("rca_report_*.md"))


def _mode_has_code_quality_assessment(result_root: Path, mode: str) -> bool:
    from ..code_eval import ASSESSMENT_FILENAME
    from ..reports._loader import mode_dir_for_benchmark

    return (mode_dir_for_benchmark(result_root, mode) / ASSESSMENT_FILENAME).is_file()


def autorun_rca_investigations(result_root: Path, *, logs: Iterable[Path] = (), only_missing: bool = False) -> None:
    """Drill into every mode the Root Cause Analysis section flags as worse.

    RCA exists to explain a with-skills regression — a failure, a failed
    quality check on a completed run, slowdown, extra tokens, or a structure
    regression. This fires it automatically right after
    the report so the explanation lands in benchmark_insights.md with no manual
    step. It requires the container sandbox: the evidence is attacker-authored,
    so the investigator must never run unsandboxed on the host. Best-effort: it
    never fails the run, skips cleanly when no investigator image is built, and
    BENCHMARK_AUTO_RCA=0 disables it.
    """

    if os.environ.get("BENCHMARK_AUTO_RCA", "1") == "0":
        return
    from ..rca import auto_diagnostic_step_timeout_seconds, resolve_invoker, resolve_seed, run_investigation

    targets = [spec.mode for spec in PAIR_RUNS if resolve_seed(result_root, spec.mode, "auto", None) is not None]
    if only_missing:
        targets = [mode for mode in targets if not _mode_has_rca_report(result_root, mode)]
    if not targets:
        return
    try:
        # sandbox="docker" is a hard requirement here, not a preference: with
        # "auto", resolve_invoker falls back to the host CLI when no image is
        # built, which would run the agent unsandboxed over captured evidence.
        agent_name, invoker = resolve_invoker(
            None, sandbox="docker", step_timeout_seconds=auto_diagnostic_step_timeout_seconds()
        )
    except SystemExit as exc:
        emit(f"Auto-RCA skipped: {exc}", logs=logs, stderr=True)
        return
    emit(f"Running automatic RCA on {', '.join(targets)} (set BENCHMARK_AUTO_RCA=0 to disable)", logs=logs)
    investigated = False
    for mode in targets:
        emit(f"Auto-RCA ({mode}, {agent_name}): investigating (heartbeat every 60s) ...", logs=logs)
        try:
            report_path = run_investigation(result_root, mode, invoker, topic="auto", agent_name=agent_name)
        except Exception as exc:
            emit(f"Auto-RCA failed for {mode}: {type(exc).__name__}: {exc}", logs=logs, stderr=True)
            continue
        if report_path is not None:
            investigated = True
            emit(f"Auto-RCA ({mode}, {agent_name}): {report_path}", logs=logs)
    if investigated:
        # Regenerate so the fresh RCA reports embed in the Root Cause Analysis section.
        write_benchmark_reports(result_root, logs=logs)


def autorun_code_quality_evaluations(
    result_root: Path, *, logs: Iterable[Path] = (), only_missing: bool = False
) -> None:
    """Have an agent judge the generated code against the criteria list.

    Replaces the brittle per-code-shape detectors: an agent reads the captured
    generated code and scores each SDK criterion, so it works whether the agent
    wrote a manual loop or a Recipe, flat files or a nested job folder. Runs in
    the container sandbox (attacker-authored code), regenerates the report so
    the verdicts show in Generated Code Quality, best-effort, skips when no
    image is built, and BENCHMARK_AUTO_CODE_EVAL=0 disables it.
    """

    if os.environ.get("BENCHMARK_AUTO_CODE_EVAL", "1") == "0":
        return
    from ..code_eval import evaluate_code_quality
    from ..rca import auto_diagnostic_step_timeout_seconds, resolve_invoker
    from ..reports.benchmark_insights import _as_run_evidence, collect_benchmark_runs
    from ..sdks.report_registry import resolve_from_result_root

    plugin = resolve_from_result_root(result_root)
    runs = collect_benchmark_runs(result_root)
    targets: list[tuple[str, list[dict[str, str]], str]] = []
    for mode, bundle in runs.items():
        if not bundle.get("available"):
            continue
        if only_missing and _mode_has_code_quality_assessment(result_root, mode):
            continue
        try:
            criteria = plugin.code_quality_criteria(_as_run_evidence(bundle))
        except Exception:
            criteria = []
        if criteria:
            targets.append((mode, criteria, str(bundle.get("evaluation_task") or "")))
    if not targets:
        return
    try:
        agent_name, invoker = resolve_invoker(
            None, sandbox="docker", step_timeout_seconds=auto_diagnostic_step_timeout_seconds()
        )
    except SystemExit as exc:
        emit(f"Auto code-eval skipped: {exc}", logs=logs, stderr=True)
        return
    modes = ", ".join(mode for mode, _, _ in targets)
    emit(f"Running automatic code-quality evaluation on {modes} (set BENCHMARK_AUTO_CODE_EVAL=0 to disable)", logs=logs)
    evaluated = False
    for mode, criteria, task in targets:
        emit(
            f"Auto code-eval ({mode}, {agent_name}): judging {len(criteria)} criteria (heartbeat every 60s) ...",
            logs=logs,
        )
        try:
            path = evaluate_code_quality(result_root, mode, invoker, criteria, agent_name=agent_name, task=task)
        except Exception as exc:
            emit(f"Auto code-eval failed for {mode}: {type(exc).__name__}: {exc}", logs=logs, stderr=True)
            continue
        if path is not None:
            evaluated = True
            emit(f"Auto code-eval ({mode}, {agent_name}): {path}", logs=logs)
    if evaluated:
        write_benchmark_reports(result_root, logs=logs)


DIAGNOSTICS_STATUS_FILENAME = "diagnostics_status.json"
DIAGNOSTICS_CONSOLE_FILENAME = "diagnostics_console.log"


def launch_diagnostics(result_root: Path, *, logs: Iterable[Path] = ()) -> None:
    """Run the agentic diagnostics WITHOUT blocking scenario completion.

    The diagnostics (auto-RCA, code-quality evaluation) can legitimately run
    for a long time, and blocking the scenario on them forced a kill-timer
    trade-off: too short kills healthy work, too long stalls the run. Instead
    the work is detached: the scenario finishes and emits its report paths
    immediately, and the worker process fires the completion event itself —
    it regenerates the reports, appends its console lines, and records the
    outcome in diagnostics_status.json when it exits. Its process exit IS the
    done signal; nothing waits on a clock for it.

    BENCHMARK_DIAGNOSTICS_BACKGROUND=0 keeps the legacy synchronous behavior
    (useful for CI wrappers that must observe the fully-final report).
    """

    if os.environ.get("BENCHMARK_DIAGNOSTICS_BACKGROUND", "1") == "0":
        autorun_rca_investigations(result_root, logs=logs)
        autorun_code_quality_evaluations(result_root, logs=logs)
        return
    console_path = result_root / DIAGNOSTICS_CONSOLE_FILENAME
    status_path = result_root / DIAGNOSTICS_STATUS_FILENAME
    try:
        with console_path.open("ab") as console:
            process = subprocess.Popen(
                [sys.executable, "-m", "benchmark.harness.host.runner", "diagnostics-worker", str(result_root)],
                stdout=console,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=str(Path.cwd()),
            )
    except OSError as exc:
        emit(f"Diagnostics worker failed to launch ({exc}); running inline instead.", logs=logs, stderr=True)
        autorun_rca_investigations(result_root, logs=logs)
        autorun_code_quality_evaluations(result_root, logs=logs)
        return
    write_json(
        status_path,
        {"status": "running", "pid": process.pid, "console": str(console_path)},
    )
    emit(
        f"Automatic diagnostics (RCA + code-eval) continue in the background (pid {process.pid}); "
        f"reports refresh when they finish. Watch: tail -f {console_path}; "
        f"status: {status_path}",
        logs=logs,
    )


def run_diagnostics_worker(argv: list[str]) -> int:
    """Detached diagnostics worker: do the work, then fire the done event.

    The "event" is concrete: regenerated reports on disk, a terminal line in
    the diagnostics console, and diagnostics_status.json flipping to done —
    written LAST, so status=done guarantees the refreshed reports exist.
    """

    if len(argv) != 1:
        raise SystemExit("Usage: diagnostics-worker RESULT_ROOT")
    result_root = absolute_path(argv[0])
    logs = (result_root / "console_output.log",)
    status_path = result_root / DIAGNOSTICS_STATUS_FILENAME
    try:
        autorun_rca_investigations(result_root, logs=logs, only_missing=True)
        autorun_code_quality_evaluations(result_root, logs=logs, only_missing=True)
    except Exception as exc:
        write_json(status_path, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        emit(f"Diagnostics worker failed: {type(exc).__name__}: {exc}", logs=logs, stderr=True)
        return 1
    emit_benchmark_report_paths(result_root, logs=logs)
    write_json(status_path, {"status": "done"})
    emit("Automatic diagnostics finished; reports are final.", logs=logs)
    return 0


def emit_benchmark_report_paths(result_root: Path, *, logs: Iterable[Path] = ()) -> None:
    emit(f"Benchmark insights: {result_root / 'benchmark_insights.md'}", logs=logs)
    emit(f"Metrics report: {result_root / 'metrics_report.md'}", logs=logs)
    emit(f"Metrics HTML: {result_root / 'metrics_report.html'}", logs=logs)


def write_host_report_status(result_root: Path, report_generator_statuses: dict[str, int] | None = None) -> None:
    report_generator_statuses = report_generator_statuses or {}
    scenario_report = result_root / "reports" / "scenario_report.md"
    benchmark_report = result_root / "benchmark_insights.md"
    metrics_report_path = result_root / "metrics_report.md"
    benchmark_reports_expected = bool(report_generator_statuses)
    all_reports_ok = (
        scenario_report.is_file()
        and all(status == 0 for status in report_generator_statuses.values())
        and (not benchmark_reports_expected or (benchmark_report.is_file() and metrics_report_path.is_file()))
    )
    payload = {
        "status": "ok" if all_reports_ok else "missing_report",
        "scenario_report": str(scenario_report),
        "benchmark_insights": str(benchmark_report),
        "metrics_report": str(metrics_report_path),
        "report_generators": report_generator_statuses,
    }
    write_json(
        result_root / "host_report_status.json",
        payload,
    )


def combined_exit_status(case_statuses: dict[str, int], report_statuses: dict[str, int] | None = None) -> int:
    report_statuses = report_statuses or {}
    return 1 if any(status != 0 for status in [*case_statuses.values(), *report_statuses.values()]) else 0


def emit_scenario_validation_error(exc: ScenarioValidationError, *, logs: Iterable[Path] = ()) -> int:
    emit(f"Scenario validation failed: {exc}", logs=logs, stderr=True)
    return 1


def run_pair(argv: list[str]) -> int:
    reject_parallel_comparison_runs("pair")
    options = parse_host_cli_options(argv, "pair")
    result_root = comparison_result_root(options)
    result_root.mkdir(parents=True, exist_ok=True)
    clean_pair_result_root(result_root)
    write_host_environment(result_root)
    console_log = result_root / "console_output.log"
    console_log.write_text("", encoding="utf-8")
    logs = (console_log,)
    try:
        adapter = agent_adapter_from_options(options)
        compilation = pair_compilation_from_options(options)
        images = image_config_from_adapter(adapter)
    except ScenarioValidationError as exc:
        return emit_scenario_validation_error(exc, logs=logs)

    emit(f"Result root: {result_root}", logs=logs)
    emit(f"Console log: {console_log}", logs=logs)
    emit(f"Skills image: {images.image_name}", logs=logs)
    emit(f"Baseline image: {images.baseline_image_name}", logs=logs)
    emit(f"Report image: {images.report_image_name}", logs=logs)
    emit(f"Job folder: {options.job_input}", logs=logs)
    emit(f"Prompt file: {options.prompt_path} -> {CONTAINER_PROMPT_PATH}", logs=logs)

    try:
        run_statuses, scenario_summary = execute_run_plan(
            compilation,
            result_root=result_root,
            logs=logs,
            **execute_auth_kwargs(options),
        )
    except ScenarioValidationError as exc:
        status = emit_scenario_validation_error(exc, logs=logs)
        write_host_report_status(result_root)
        return status

    emit(f"Scenario summary: {result_root / 'scenario_summary.json'}", logs=logs)
    emit(f"Scenario report: {result_root / 'reports' / 'scenario_report.md'}", logs=logs)
    report_statuses = write_benchmark_reports(result_root, logs=logs)
    # RCA first: it is the priority diagnostic and must not be gated on a slow
    # code-evaluation pass (each is best-effort and independently regenerates
    # the report).
    launch_diagnostics(result_root, logs=logs)
    emit_benchmark_report_paths(result_root, logs=logs)
    write_host_report_status(result_root, report_statuses)
    return (
        1
        if any(status != 0 for status in run_statuses.values())
        or scenario_summary.get("status") in {"degraded", "failed"}
        or any(status != 0 for status in report_statuses.values())
        else 0
    )


def run_scenario(argv: list[str]) -> int:
    reject_parallel_comparison_runs("scenario")
    options = parse_scenario_cli_options(argv)
    try:
        compilation = compile_scenario_file(options.scenario_path)
    except ScenarioValidationError as exc:
        return emit_scenario_validation_error(exc)
    result_root = scenario_result_root(options, compilation)
    result_root.mkdir(parents=True, exist_ok=True)
    write_host_environment(result_root)
    console_log = result_root / "console_output.log"
    console_log.write_text("", encoding="utf-8")
    logs = (console_log,)

    emit(f"Result root: {result_root}", logs=logs)
    emit(f"Console log: {console_log}", logs=logs)
    emit(f"Scenario file: {options.scenario_path}", logs=logs)
    emit(f"Run count: {compilation.run_plan.get('run_count')}", logs=logs)

    try:
        statuses, summary = execute_run_plan(
            compilation,
            result_root=result_root,
            logs=logs,
            **execute_auth_kwargs(options),
        )
    except ScenarioValidationError as exc:
        status = emit_scenario_validation_error(exc, logs=logs)
        write_host_report_status(result_root)
        return status
    emit(f"Scenario summary: {result_root / 'scenario_summary.json'}", logs=logs)
    emit(f"Scenario report: {result_root / 'reports' / 'scenario_report.md'}", logs=logs)
    write_host_report_status(result_root)
    return (
        1 if any(status != 0 for status in statuses.values()) or summary.get("status") in {"degraded", "failed"} else 0
    )


def run_report(argv: list[str]) -> int:
    options = parse_report_cli_options(argv)
    console_log = options.result_root / "report_console_output.log"
    console_log.write_text("", encoding="utf-8")
    logs = (console_log,)
    emit(f"Report result root: {options.result_root}", logs=logs)
    summary = replay_result_root(options.result_root, logs=logs)
    emit(f"Scenario summary: {options.result_root / 'scenario_summary.json'}", logs=logs)
    emit(f"Scenario report: {options.result_root / 'reports' / 'scenario_report.md'}", logs=logs)
    report_statuses = write_benchmark_reports(options.result_root, logs=logs)
    if options.diagnostics:
        # Backfill ONLY missing diagnostics (a killed/skipped auto-RCA or
        # code-eval); existing outputs are kept, and each hook regenerates the
        # reports when it produced something new.
        autorun_rca_investigations(options.result_root, logs=logs, only_missing=True)
        autorun_code_quality_evaluations(options.result_root, logs=logs, only_missing=True)
    emit_benchmark_report_paths(options.result_root, logs=logs)
    write_host_report_status(options.result_root, report_statuses)
    # Report regenerates artifacts from an existing result tree; it preserves
    # degraded benchmark status instead of re-asserting pass/fail.
    return (
        0
        if summary.get("status") in {"passed", "degraded"} and all(status == 0 for status in report_statuses.values())
        else 1
    )


def run_interactive(argv: list[str]) -> int:
    options = parse_host_cli_options(argv, "interactive")
    try:
        adapter = agent_adapter_from_options(options)
        agent_model, model_was_explicit = agent_model_from_options(adapter, options)
        images = image_config_from_adapter(adapter)
    except ScenarioValidationError as exc:
        return emit_scenario_validation_error(exc)
    runtime_auth_options = runtime_auth_options_from_host_cli(options)
    host_agent_home = runtime_auth_options.agent_home or absolute_path(str(adapter.host_home_from_env(os.environ)))
    mount_agent_auth = (
        runtime_auth_options.mount_agent_auth
        if runtime_auth_options.mount_agent_auth is not None
        else adapter.mount_auth_from_env(os.environ)
    )
    container_records = os.environ.get("CONTAINER_RECORDS", "/tmp/agent_benchmark/records")
    args = [
        "docker",
        "run",
        "--rm",
        "-it",
        "-v",
        f"{options.job_input}:/workspace/input",
        "-v",
        f"{options.prompt_path}:{CONTAINER_PROMPT_PATH}:ro",
        *docker_env("JOB_INPUT_DIR", "/workspace/input"),
        *docker_env("TRAINING_CODE", "/workspace/input"),
        *docker_env("PROMPT_SOURCE", CONTAINER_PROMPT_PATH),
        *docker_env("RECORDS_DIR", container_records),
    ]
    for name, value in sorted(
        adapter.runtime_env(
            InteractiveRuntimeConfig(
                agent=adapter.name,
                agent_model=agent_model,
                model_was_explicit=model_was_explicit,
            )
        ).items()
    ):
        args.extend(docker_env(name, value))
    add_agent_passthrough_env(args, adapter)
    if mount_agent_auth:
        interactive_config = SimpleNamespace(host_agent_home=host_agent_home)
        add_agent_auth_mounts(args, mounts=adapter.auth_mounts(interactive_config))
    emit(f"Mounting job folder: {options.job_input} -> /workspace/input")
    emit(f"Using prompt file: {options.prompt_path} -> {CONTAINER_PROMPT_PATH}")
    try:
        return subprocess.call([*args, images.image_name, "bash"])
    except OSError as exc:
        emit(f"Failed to start interactive container: {type(exc).__name__}: {exc}", stderr=True)
        return 127


def agent_adapter_from_options(options):
    agent_name = getattr(options, "agent", None) or os.environ.get("BENCHMARK_AGENT", DEFAULT_BENCHMARK_AGENT)
    try:
        return load_agent_adapter(agent_name)
    except ValueError as exc:
        raise ScenarioValidationError(str(exc)) from exc


def runtime_auth_options_from_host_cli(options) -> RuntimeAuthOptions:
    return RuntimeAuthOptions(
        agent_home=getattr(options, "agent_home", None),
        mount_agent_auth=getattr(options, "mount_agent_auth", None),
    )


def execute_auth_kwargs(options) -> dict[str, RuntimeAuthOptions]:
    runtime_auth_options = runtime_auth_options_from_host_cli(options)
    return {"runtime_auth_options": runtime_auth_options} if runtime_auth_options.has_overrides else {}


def agent_model_from_options(adapter, options) -> tuple[str, bool]:
    env = os.environ if not getattr(options, "model", None) else {**os.environ, "BENCHMARK_AGENT_MODEL": options.model}
    try:
        return adapter.model_from_env(env), adapter.model_was_explicit(env)
    except ValueError as exc:
        raise ScenarioValidationError(str(exc)) from exc


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(
            "Usage: python -m benchmark.harness.host.runner {pair,scenario,report,interactive} "
            "--prompt PATH [--training-code PATH] [--results-root PATH] [PATH]"
        )
        raise SystemExit(0 if len(sys.argv) >= 2 else 2)
    command, argv = sys.argv[1], sys.argv[2:]
    if command == "pair":
        status = run_pair(argv)
    elif command == "scenario":
        status = run_scenario(argv)
    elif command == "report":
        status = run_report(argv)
    elif command == "diagnostics-worker":
        status = run_diagnostics_worker(argv)
    elif command == "interactive":
        status = run_interactive(argv)
    else:
        raise SystemExit(f"Unknown command: {command}")
    raise SystemExit(status)


if __name__ == "__main__":
    main()
