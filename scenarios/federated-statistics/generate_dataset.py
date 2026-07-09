#!/usr/bin/env python3
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

"""Generate the v2 federated-statistics benchmark datasets (seeded, reproducible).

Bootstraps rows per site from the ORIGINAL 1000-row patient-encounter extract
(preserving each site's distribution shape and every column's dtype/format),
jitters continuous columns, and writes train/valid splits:

    <out_root>/tabular_with_header_v2/site-N/{train,valid}.csv   (header row)
    <out_root>/tabular_no_header_v2/site-N/{train,valid}.csv     (no header)
    <out_root>/tabular_no_header_v2/README.md                    (numbered columns)

Deterministic for a given --seed: rerunning yields byte-identical CSVs, so the
committed ground truth (see generate_ground_truth.py) stays regenerable.

Usage:
    generate_dataset.py <original_with_header_dir> <out_root>
        [--seed 20260709] [--total 20000] [--site-fractions 0.45,0.35,0.20]
        [--valid-fraction 0.2]
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

SITES = ("site-1", "site-2", "site-3")


def load_original_site(site_dir: Path) -> tuple[list[str], list[list[str]]]:
    with (site_dir / "data.csv").open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.reader(stream) if row]
    return [name.strip() for name in rows[0]], rows[1:]


def column_decimals(values: list[str]) -> int | None:
    """Decimal places used by a numeric column (None => not numeric)."""

    decimals = 0
    for value in values:
        if not re.fullmatch(r"-?\d+(\.\d+)?", value):
            return None
        if "." in value:
            decimals = max(decimals, len(value.split(".", 1)[1]))
    return decimals


def jitter(value: float, scale: float, decimals: int, rng: random.Random, floor: float) -> str:
    jittered = max(floor, value + rng.gauss(0.0, scale))
    if decimals == 0:
        return str(int(round(jittered)))
    return f"{jittered:.{decimals}f}"


def synthesize_site(
    columns: list[str], source_rows: list[list[str]], target: int, rng: random.Random
) -> list[list[str]]:
    numeric_meta: dict[int, tuple[int, float, float]] = {}
    for index in range(len(columns)):
        values = [row[index] for row in source_rows]
        decimals = column_decimals(values)
        if decimals is None:
            continue
        floats = [float(value) for value in values]
        mean = sum(floats) / len(floats)
        std = (sum((value - mean) ** 2 for value in floats) / (len(floats) - 1)) ** 0.5
        # 5% of the column's spread keeps the site's distribution shape while
        # making bootstrapped rows distinct; integer columns are not jittered
        # (counts like num_prior_admissions must stay honest integers).
        scale = 0.0 if decimals == 0 else std * 0.05
        numeric_meta[index] = (decimals, scale, min(floats))
    rows = []
    for _ in range(target):
        base = rng.choice(source_rows)
        row = list(base)
        for index, (decimals, scale, floor) in numeric_meta.items():
            if scale > 0.0:
                row[index] = jitter(float(base[index]), scale, decimals, rng, floor)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[list[str]], header: list[str] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        if header is not None:
            writer.writerow(header)
        writer.writerows(rows)


def write_noheader_readme(path: Path, columns: list[str]) -> None:
    lines = [
        "# Patient Encounter Extract (v2)",
        "",
        "Each site keeps its own data: `site-1/`, `site-2/`, and `site-3/`, each",
        "split into `train.csv` and `valid.csv`. The CSV files have no header",
        "row. The columns are, in order:",
        "",
    ]
    lines += [f"{index}. {name}" for index, name in enumerate(columns, start=1)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_dir", type=Path, help="original tabular_with_header directory")
    parser.add_argument("out_root", type=Path, help="directory to create the *_v2 dataset dirs in")
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--total", type=int, default=20000)
    parser.add_argument("--site-fractions", default="0.45,0.35,0.20")
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    args = parser.parse_args()

    fractions = [float(part) for part in args.site_fractions.split(",")]
    if len(fractions) != len(SITES) or abs(sum(fractions) - 1.0) > 1e-6:
        raise SystemExit(f"--site-fractions must be {len(SITES)} values summing to 1.0")
    rng = random.Random(args.seed)

    header_root = args.out_root.expanduser() / "tabular_with_header_v2"
    noheader_root = args.out_root.expanduser() / "tabular_no_header_v2"
    columns: list[str] = []
    for site, fraction in zip(SITES, fractions):
        columns, source_rows = load_original_site(args.original_dir.expanduser() / site)
        target = int(round(args.total * fraction))
        rows = synthesize_site(columns, source_rows, target, rng)
        valid_count = int(round(target * args.valid_fraction))
        splits = {"valid": rows[:valid_count], "train": rows[valid_count:]}
        for split, split_rows in splits.items():
            write_csv(header_root / site / f"{split}.csv", split_rows, columns)
            write_csv(noheader_root / site / f"{split}.csv", split_rows, None)
        print(f"{site}: train={len(splits['train'])} valid={len(splits['valid'])}")
    write_noheader_readme(noheader_root / "README.md", columns)
    print(f"wrote {header_root} and {noheader_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
