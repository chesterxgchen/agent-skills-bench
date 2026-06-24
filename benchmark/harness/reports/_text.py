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

"""Generic command/text-parsing substrate (migration step 5-pre).

A neutral leaf used by both the generic report engine (``benchmark_insights``)
and the SDK plugins (``sdks/nvflare``). It depends only on the standard library,
so it can be imported from either side without an import cycle. This is the
prerequisite that lets SDK-specific command detection move into the plugin
without dragging generic shell-parsing helpers along.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

FILE_INSPECTION_COMMANDS = {"cat", "sed", "nl", "head", "tail", "grep", "rg", "find", "ls"}
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_PATTERN.sub("", text)


def markdown_cell(value: Any) -> str:
    text = "NA" if value is None or value == "" else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def fmt_number(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.4f}"


def _command_count_display(count: int) -> str:
    return f"{count} command" if count == 1 else f"{count} commands"


def _classification_command(command: str) -> str:
    text = str(command).strip()
    try:
        tokens = shlex.split(text)
    except ValueError:
        return text
    if len(tokens) >= 3 and Path(tokens[0]).name in {"bash", "sh"}:
        for index, token in enumerate(tokens[1:], start=1):
            if token.startswith("-") and "c" in token and index + 1 < len(tokens):
                return tokens[index + 1]
    return text


def _command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(_classification_command(command))
    except ValueError:
        return []


def _first_command_name(command: str) -> str:
    tokens = _command_tokens(command)
    return Path(tokens[0]).name.lower() if tokens else ""


def _shell_command_parts(command: str) -> list[tuple[str, str]]:
    """Split a command into (segment, operator_after) pairs, honoring shell quoting.

    Only unquoted ``&&``, ``||`` and ``;`` separate segments; operators inside quoted
    strings (e.g. an ``rg`` search pattern) stay within the segment. ``operator_after`` is
    the operator that joins the segment to the next one ("" for the final segment).
    """
    text = _classification_command(command)
    parts: list[tuple[str, str]] = []
    buf: list[str] = []
    quote = ""
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if quote:
            buf.append(char)
            if char == quote:
                quote = ""
            index += 1
            continue
        if char in ("'", '"'):
            quote = char
            buf.append(char)
            index += 1
            continue
        if char == ";":
            segment = "".join(buf).strip()
            if segment:
                parts.append((segment, ";"))
            buf = []
            index += 1
            continue
        if char in ("&", "|") and index + 1 < length and text[index + 1] == char:
            segment = "".join(buf).strip()
            if segment:
                parts.append((segment, char * 2))
            buf = []
            index += 2
            continue
        buf.append(char)
        index += 1
    segment = "".join(buf).strip()
    if segment:
        parts.append((segment, ""))
    return parts


def _shell_command_segments(command: str) -> list[str]:
    return [segment for segment, _operator in _shell_command_parts(command)]


def _is_file_inspection_segment(segment: str) -> bool:
    if _first_command_name(segment) not in FILE_INSPECTION_COMMANDS:
        return False
    return bool(
        re.search(
            r"\b(?:cat|sed|nl|head|tail|grep|rg|find|ls)\b[^\n]*(?:\.py|job|simulat)",
            segment,
            flags=re.IGNORECASE,
        )
    )


def _strip_quoted(text: str) -> str:
    """Remove single/double quoted spans so quoted text is not mistaken for shell syntax."""
    out: list[str] = []
    quote = ""
    for ch in text:
        if quote:
            if ch == quote:
                quote = ""
            continue
        if ch in ("'", '"'):
            quote = ch
            continue
        out.append(ch)
    return "".join(out)
