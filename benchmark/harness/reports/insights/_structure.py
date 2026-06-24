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

"""E2 split module (mechanical, behavior-preserving)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .._context import ReportContext
from .._runs import manifest_paths, run_record, run_workspace_delta, unique_paths
from .._text import fmt_number, markdown_cell
from ..evidence import RunEvidence
from ._plugin_view import (
    CONFIG_STRUCTURE_SUFFIXES,
    MODE_LABELS,
    TREE_RUNTIME_SUFFIXES,
    _report_context,
    basename_count_display,
    run_source_input_delta,
)

__all__ = [
    "run_identity_table",
    "run_identity_summary",
    "output_changes_table",
    "artifact_summary",
    "workspace_change_display",
    "source_input_protection_display",
    "nested_structure_file_matches",
    "structure_required_display",
    "structure_optional_display",
    "nested_generated_structure_display",
    "structure_inventory_display",
    "tree_from_paths",
    "tree_paths_for_keys",
    "structure_correctness_table",
    "structure_trees_section",
]


def run_identity_table(runs: dict[str, RunEvidence], modes: list[str]) -> str:
    lines = [
        "| Run | Agent | Model | Model source | Mode |",
        "|---|---|---|---|---|",
    ]
    for mode in modes:
        run = runs[mode]
        lines.append(
            f"| {markdown_cell(run.label or mode)} | {markdown_cell(run.agent)} | "
            f"{markdown_cell(run.agent_model)} | {markdown_cell(run.model_source)} | "
            f"{markdown_cell(mode)} |"
        )
    return "\n".join(lines)


def run_identity_summary(runs: dict[str, RunEvidence], modes: list[str]) -> str:
    return "; ".join(
        f"{runs[mode].label or mode}: agent={runs[mode].agent or 'NA'}, " f"model={runs[mode].agent_model or 'NA'}"
        for mode in modes
    )


def output_changes_table(runs: dict[str, RunEvidence], modes: list[str]) -> str:
    lines = [
        "| Run | Changed files | Added | Modified | Notable files |",
        "|---|---:|---:|---:|---|",
    ]
    for mode in modes:
        run = runs[mode]
        record = run.record if isinstance(run.record, dict) else {}
        delta = record.get("workspace_delta") if isinstance(record.get("workspace_delta"), dict) else {}
        if not delta:
            delta = run.workspace_delta if isinstance(run.workspace_delta, dict) else {}
        changed_files = delta.get("changed_files") if isinstance(delta.get("changed_files"), list) else []
        changed_count = delta.get("changed_file_count")
        added = delta.get("workspace_added_file_count")
        modified = delta.get("workspace_modified_file_count")
        names = []
        for item in changed_files[:8]:
            if isinstance(item, dict) and item.get("path"):
                names.append(str(item["path"]))
        suffix = "" if len(changed_files) <= 8 else f"; +{len(changed_files) - 8} more"
        lines.append(
            f"| {markdown_cell(run.label)} | {fmt_number(changed_count)} | {fmt_number(added)} | "
            f"{fmt_number(modified)} | {markdown_cell('; '.join(names) + suffix if names else 'NA')} |"
        )
    return "\n".join(lines)


def artifact_summary(run: RunEvidence) -> str:
    delta = run_workspace_delta(run.raw)
    changed = delta.get("changed_file_count")
    runtime = delta.get("runtime_artifact_count")
    copied = delta.get("copied_file_count")
    parts = []
    if changed is not None:
        parts.append(f"{fmt_number(changed)} changed/generated files")
    if runtime is not None:
        parts.append(f"{fmt_number(runtime)} runtime artifacts")
    if copied is not None:
        parts.append(f"{fmt_number(copied)} copied artifacts")
    return ", ".join(parts) if parts else "not captured"


def workspace_change_display(run: RunEvidence) -> str:
    delta = run_workspace_delta(run.raw)
    changed = delta.get("changed_file_count")
    added = delta.get("workspace_added_file_count")
    modified = delta.get("workspace_modified_file_count")
    deleted = delta.get("workspace_deleted_baseline_file_count")
    if changed is None:
        return "not captured"
    parts = [f"{fmt_number(changed)} changed"]
    if added is not None:
        parts.append(f"{fmt_number(added)} added")
    if modified is not None:
        parts.append(f"{fmt_number(modified)} modified")
    if deleted:
        parts.append(f"{fmt_number(deleted)} deleted")
    return ", ".join(parts)


def source_input_protection_display(run: RunEvidence) -> str:
    record = run_record(run.raw)
    policy = record.get("source_input_immutable_policy")
    if isinstance(policy, dict) and policy.get("status"):
        status = str(policy["status"])
        reason = policy.get("reason")
        return f"{status}: {reason}" if reason else status
    delta = run_source_input_delta(run)
    if not delta:
        return "not captured"
    changed = delta.get("changed_file_count")
    deleted = delta.get("deleted_file_count")
    if changed == 0 and deleted == 0:
        return "pass: immutable input snapshot unchanged"
    return f"fail: input snapshot changed={fmt_number(changed)}, deleted={fmt_number(deleted)}"


def nested_structure_file_matches(run: RunEvidence, filename: str) -> list[str]:
    paths = unique_paths(manifest_paths(run.raw, "final_structure_files") + manifest_paths(run.raw, "changed_files"))
    return [path for path in paths if Path(path).name == filename and len(Path(path).parts) > 1]


def structure_required_display(run: RunEvidence, view: Any) -> str:
    if view.score is None:
        return "not captured"
    required_files = view.required_files
    present = list(view.present_required)
    missing = [filename for filename in required_files if filename not in present]
    text = f"{len(present)}/{len(required_files)} present"
    if missing:
        text += "; missing " + ", ".join(missing)
    nested = {
        filename: nested_structure_file_matches(run, filename)
        for filename in required_files
        if nested_structure_file_matches(run, filename)
    }
    if nested:
        folders = sorted({str(Path(path).parent) for paths in nested.values() for path in paths})
        text += "; nested copies ignored for current-structure score: " + ", ".join(folders[:3])
        if len(folders) > 3:
            text += f", +{len(folders) - 3} more"
    return text


def structure_optional_display(run: RunEvidence, view: Any) -> str:
    if view.score is None:
        return "not captured"
    present = list(view.present_optional)
    return ", ".join(present) if present else "none"


def nested_generated_structure_display(run: RunEvidence, view: Any) -> str:
    folders: dict[str, set[str]] = {}
    for filename in view.required_files:
        for path in nested_structure_file_matches(run, filename):
            folders.setdefault(str(Path(path).parent), set()).add(filename)
    if not folders:
        return "none"
    entries = []
    for folder in sorted(folders)[:4]:
        entries.append(f"{folder} ({', '.join(sorted(folders[folder]))})")
    if len(folders) > 4:
        entries.append(f"+{len(folders) - 4} more")
    return "; ".join(entries)


def structure_inventory_display(run: RunEvidence, key: str, suffixes: tuple[str, ...]) -> str:
    paths = [path for path in manifest_paths(run.raw, key) if Path(path).suffix in suffixes]
    return basename_count_display(paths)


def tree_from_paths(paths: list[str], *, max_paths: int = 80) -> str:
    sorted_paths = sorted(unique_paths(paths))
    truncated = len(sorted_paths) > max_paths
    paths = sorted_paths[:max_paths]
    if not paths:
        return "none"
    tree: dict[str, Any] = {}
    for path in paths:
        node = tree
        for part in Path(path).parts:
            if not part or part == ".":
                continue
            node = node.setdefault(part, {})

    lines = ["."]

    def render(node: dict[str, Any], prefix: str = "") -> None:
        entries = sorted(node)
        for index, name in enumerate(entries):
            connector = "`-- " if index == len(entries) - 1 else "|-- "
            lines.append(f"{prefix}{connector}{name}")
            child = node[name]
            if child:
                extension = "    " if index == len(entries) - 1 else "|   "
                render(child, prefix + extension)

    render(tree)
    if truncated:
        lines.append(f"... {len(sorted_paths) - max_paths} more paths not shown")
    return "\n".join(lines)


def tree_paths_for_keys(
    run: RunEvidence,
    keys: tuple[str, ...],
    *,
    suffixes: tuple[str, ...] | None = None,
) -> list[str]:
    paths = []
    for key in keys:
        for path in manifest_paths(run.raw, key):
            if suffixes is not None and Path(path).suffix not in suffixes:
                continue
            paths.append(path)
    return unique_paths(paths)


def structure_correctness_table(
    runs: dict[str, RunEvidence], modes: list[str], ctx: ReportContext | None = None
) -> str:
    # Structure is read from the per-run StructureView (realized in collect()).
    ctx = ctx or _report_context(runs, modes)
    rows = [
        ("Required converted files", lambda run, view: structure_required_display(run, view)),
        ("Nested generated job source", lambda run, view: nested_generated_structure_display(run, view)),
        ("Optional helper files", lambda run, view: structure_optional_display(run, view)),
        (
            "Final workspace Python inventory",
            lambda run, view: structure_inventory_display(run, "final_files", (".py",)),
        ),
        (
            "Changed/generated Python inventory",
            lambda run, view: structure_inventory_display(run, "changed_files", (".py",)),
        ),
        (
            "Runtime artifact config inventory",
            lambda run, view: structure_inventory_display(run, "runtime_artifacts", CONFIG_STRUCTURE_SUFFIXES),
        ),
    ]
    lines = [
        "| Structure signal | " + " | ".join(MODE_LABELS.get(mode, mode) for mode in modes) + " |",
        "|---|" + "|".join("---" for _ in modes) + "|",
    ]
    for label, getter in rows:
        lines.append(
            f"| {markdown_cell(label)} | "
            + " | ".join(markdown_cell(getter(runs[mode], ctx.structure_view(mode))) for mode in modes)
            + " |"
        )
    return "\n".join(lines)


def structure_trees_section(runs: dict[str, RunEvidence], modes: list[str]) -> str:
    lines = [
        "### Captured Structure Trees",
        "",
        "Trees are rendered from captured artifact manifests in tree-command format.",
    ]
    for mode in modes:
        run = runs[mode]
        lines.append("")
        lines.append(f"#### {run.label or mode}")
        lines.append("")
        final_paths = tree_paths_for_keys(run, ("final_files",)) or tree_paths_for_keys(
            run,
            ("final_structure_files", "runtime_artifacts"),
            suffixes=TREE_RUNTIME_SUFFIXES,
        )
        changed_paths = tree_paths_for_keys(run, ("changed_files", "runtime_artifacts"))
        lines.append("Final workspace:")
        lines.append("")
        lines.append("```text")
        lines.append(tree_from_paths(final_paths))
        lines.append("```")
        lines.append("")
        lines.append("Changed/generated files:")
        lines.append("")
        lines.append("```text")
        lines.append(tree_from_paths(changed_paths))
        lines.append("```")
    return "\n".join(lines)
