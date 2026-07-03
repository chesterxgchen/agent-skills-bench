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

"""Common helpers used by harness command modules."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def path_tree_sha256(path: Path) -> str:
    """Order-independent content hash of a file or directory tree.

    Symlinks are skipped (never followed): the tree may be an agent-writable
    result-mount copy, so a planted link must not pull host bytes into the
    digest or let the hash follow a link the reader won't. Directory symlinks
    are pruned so a link cycle can't recurse unbounded.
    """

    digest = hashlib.sha256()
    if path.is_symlink():
        return digest.hexdigest()
    if path.is_file():
        candidates = [(path.name, path)]
    else:
        candidates = []
        for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
            directory = Path(dirpath)
            dirnames[:] = [name for name in dirnames if not (directory / name).is_symlink()]
            for name in filenames:
                candidate = directory / name
                if candidate.is_symlink() or not candidate.is_file():
                    continue
                candidates.append((candidate.relative_to(path).as_posix(), candidate))
        candidates.sort()
    for relative, candidate in candidates:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def epoch_seconds() -> int:
    return int(time.time())


def load_json(path: str | Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def load_text(path: str | Path, default: str = "") -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return default


def as_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_json(path: str | Path, value: Any) -> None:
    write_json_atomic(path, value)


def write_json_atomic(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    tmp_path = None
    try:
        fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        tmp_path = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            fd = None
            stream.write(json.dumps(value, indent=2, sort_keys=True))
        os.replace(tmp_path, target)
        tmp_path = None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


def make_tree_readable(path: str | Path) -> None:
    """Best-effort fix for Docker-created result files on bind mounts."""
    root = Path(path)
    paths = [root]
    if root.is_dir() and not root.is_symlink():
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            directory = Path(dirpath)
            dirnames[:] = [name for name in dirnames if not (directory / name).is_symlink()]
            paths.extend(directory / name for name in dirnames)
            paths.extend(directory / name for name in filenames if not (directory / name).is_symlink())
    for item in paths:
        try:
            if item.is_symlink():
                continue
            mode = item.stat().st_mode
            if item.is_dir():
                mode |= (
                    stat.S_IRUSR
                    | stat.S_IWUSR
                    | stat.S_IXUSR
                    | stat.S_IRGRP
                    | stat.S_IXGRP
                    | stat.S_IROTH
                    | stat.S_IXOTH
                )
            else:
                mode |= stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
            item.chmod(mode)
        except OSError:
            continue


def bool_from_text(value: str) -> bool:
    return str(value).lower() == "true"


def flatten_numbers(obj: Any, prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "all_metrics":
                continue
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_numbers(value, child_prefix))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            flattened.update(flatten_numbers(value, child_prefix))
    elif isinstance(obj, bool):
        flattened[prefix] = 1 if obj else 0
    elif isinstance(obj, (int, float)):
        flattened[prefix] = obj
    return flattened
