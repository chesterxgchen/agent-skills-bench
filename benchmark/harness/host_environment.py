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

"""Host environment capture for benchmark reproducibility."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any, Mapping

from .common import load_json, write_json


def _linux_os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"')
    except OSError:
        return {}
    return values


def detect_host_environment() -> dict[str, Any]:
    system = platform.system() or "unknown"
    family = system.lower() or "unknown"
    distribution: dict[str, str] = {}
    display = system
    if system == "Darwin":
        family = "macos"
        mac_version = platform.mac_ver()[0]
        display = f"macOS {mac_version}" if mac_version else "macOS"
    elif system == "Linux":
        distribution = _linux_os_release()
        distro_id = (distribution.get("ID") or "").lower()
        family = distro_id or "linux"
        display = distribution.get("PRETTY_NAME") or distribution.get("NAME") or "Linux"
    elif system == "Windows":
        family = "windows"
        release = platform.release()
        display = f"Windows {release}" if release else "Windows"
    return {
        "schema_version": "1",
        "host_os": {
            "display": display,
            "family": family,
            "system": system,
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": sys.platform,
            "distribution": distribution,
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
    }


def read_host_environment(result_root: str | Path) -> dict[str, Any]:
    payload = load_json(Path(result_root) / "host_environment.json", {}) or {}
    return payload if isinstance(payload, dict) else {}


def write_host_environment(result_root: str | Path) -> dict[str, Any]:
    payload = detect_host_environment()
    write_json(Path(result_root) / "host_environment.json", payload)
    return payload


def host_os_display(host_environment: Mapping[str, Any] | None) -> str:
    if not isinstance(host_environment, Mapping):
        return ""
    host_os = host_environment.get("host_os")
    if isinstance(host_os, Mapping):
        display = str(host_os.get("display") or "").strip()
        if display:
            return display
        system = str(host_os.get("system") or "").strip()
        release = str(host_os.get("release") or "").strip()
        return f"{system} {release}".strip()
    return ""
