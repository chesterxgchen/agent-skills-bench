# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extract the harness-owned SDK import package into a protected overlay."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from configparser import ConfigParser
from email.parser import BytesParser
from pathlib import Path, PurePosixPath


def install_overlay(
    wheel: Path,
    destination: Path,
    import_name: str,
    scripts_source: Path | None = None,
    scripts_destination: Path | None = None,
) -> dict[str, object]:
    """Copy only the configured SDK import package from ``wheel``.

    Job dependencies remain writable in the benchmark virtual environment. The
    overlay is placed first on ``PYTHONPATH`` and made root-owned by the Docker
    build, so a later ``pip install`` cannot replace the SDK implementation that
    the benchmark is evaluating.
    """

    if not import_name or "/" in import_name or "\\" in import_name or import_name in {".", ".."}:
        raise ValueError(f"invalid SDK import name: {import_name!r}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    package_prefix = f"{import_name}/"
    module_name = f"{import_name}.py"
    extracted: list[str] = []
    console_scripts: list[str] = []
    distribution_name: str | None = None
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            name = info.filename
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe wheel member: {name!r}")
            if not (name.startswith(package_prefix) or name == module_name):
                continue
            target = destination.joinpath(*path.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted.append(name)
        entry_points = [info for info in archive.infolist() if info.filename.endswith(".dist-info/entry_points.txt")]
        if entry_points:
            parser = ConfigParser()
            parser.read_string(archive.read(entry_points[0]).decode("utf-8"))
            console_scripts = sorted(parser["console_scripts"]) if parser.has_section("console_scripts") else []
        metadata_files = [info for info in archive.infolist() if info.filename.endswith(".dist-info/METADATA")]
        if len(metadata_files) == 1:
            distribution_name = BytesParser().parsebytes(archive.read(metadata_files[0])).get("Name")

    if not extracted:
        raise ValueError(f"{wheel.name} does not contain import package {import_name!r}")
    if (scripts_source is None) != (scripts_destination is None):
        raise ValueError("scripts source and destination must be provided together")
    if scripts_source is not None and scripts_destination is not None:
        scripts_destination.mkdir(parents=True, exist_ok=True)
        for name in console_scripts:
            if Path(name).name != name:
                raise ValueError(f"unsafe console script name: {name!r}")
            source = scripts_source / name
            if not source.is_file():
                raise ValueError(f"installed console script is missing: {source}")
            shutil.copy2(source, scripts_destination / name)

    manifest = {
        "schema_version": 1,
        "wheel": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "import_name": import_name,
        "distribution_name": distribution_name,
        "file_count": len(extracted),
        "files": sorted(extracted),
        "console_scripts": console_scripts,
    }
    (destination / "overlay_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    if len(sys.argv) not in {4, 6}:
        print(
            "usage: install_sdk_overlay.py <wheel> <destination> <import-name> "
            "[<scripts-source> <scripts-destination>]",
            file=sys.stderr,
        )
        return 2
    scripts_source = Path(sys.argv[4]) if len(sys.argv) == 6 else None
    scripts_destination = Path(sys.argv[5]) if len(sys.argv) == 6 else None
    install_overlay(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], scripts_source, scripts_destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
