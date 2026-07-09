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

"""Precompute the federated-statistics ground truth for a site-split dataset.

Run once per dataset and commit the JSON: the acceptance checks compare a
run's statistics output against these constants instead of recomputing them,
so a check can never inherit an agent's mistake. Each site CSV's sha256 is
recorded; the checks fail loudly when the dataset no longer matches.

Usage:
    generate_ground_truth.py <dataset_dir> <output.json> [--no-header]

The dataset directory holds site-1/ site-2/ site-3/ each with one data.csv.
With --no-header, column names come from the numbered list in the dataset's
README.md (the headerless benchmark variant).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

SITES = ("site-1", "site-2", "site-3")
SENTINELS_PER_SITE = 2


def readme_columns(readme: Path) -> list[str]:
    """Column names from the README's numbered list — fail loudly on any entry
    this parser cannot represent, and on numbering gaps: a silently skipped
    line would shift every later column name against the CSV fields."""

    numbers: list[int] = []
    columns: list[str] = []
    for line in readme.read_text(encoding="utf-8").splitlines():
        numbered = re.match(r"^\s*(\d+)\.\s*(.*)$", line)
        if not numbered:
            continue
        name = numbered.group(2).strip()
        if not re.fullmatch(r"\S+", name):
            raise SystemExit(
                f"Unparseable column entry in {readme}: {line.strip()!r} (expected one single-token name)"
            )
        numbers.append(int(numbered.group(1)))
        columns.append(name)
    if not columns:
        raise SystemExit(f"No numbered column list found in {readme}")
    if numbers != list(range(1, len(columns) + 1)):
        raise SystemExit(f"Column list in {readme} is not numbered contiguously 1..{len(columns)}: {numbers}")
    return columns


def load_site(site_dir: Path, *, header: bool, columns: list[str] | None) -> tuple[list[str], list[dict[str, str]]]:
    csv_path = site_dir / "data.csv"
    with csv_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        rows = [row for row in reader if row]
    if header:
        columns = [name.strip() for name in rows[0]]
        rows = rows[1:]
    assert columns, "columns required for headerless data"
    for index, row in enumerate(rows, start=2 if header else 1):
        if len(row) != len(columns):
            raise SystemExit(
                f"{csv_path} line {index}: {len(row)} fields but {len(columns)} columns declared — "
                "refusing to zip-truncate silently"
            )
    records = [dict(zip(columns, row)) for row in rows]
    return columns, records


def is_numeric_column(records: list[dict[str, str]], column: str) -> bool:
    for record in records:
        value = record.get(column, "")
        try:
            float(value)
        except ValueError:
            return False
    return True


def feature_stats(values: list[float]) -> dict[str, float]:
    count = len(values)
    if count < 2:
        raise SystemExit(f"feature statistics need at least 2 values (got {count}); sample stddev is undefined")
    total = sum(values)
    mean = total / count
    sq = sum((value - mean) ** 2 for value in values)
    return {
        "count": count,
        "sum": round(total, 6),
        "mean": round(mean, 6),
        # Both conventions: pandas-style sample stddev (ddof=1) and the
        # population form — federated aggregation implementations differ.
        "stddev": round(math.sqrt(sq / (count - 1)), 6),
        "stddev_population": round(math.sqrt(sq / count), 6),
        "min": min(values),
        "max": max(values),
    }


def sentinel_rows(site_dir: Path, *, header: bool) -> list[str]:
    lines = [line for line in (site_dir / "data.csv").read_text(encoding="utf-8").splitlines() if line.strip()]
    if header:
        lines = lines[1:]
    # Distinctive full rows from fixed offsets — deterministic, mid-file.
    picks = [lines[len(lines) // 3], lines[(2 * len(lines)) // 3]][:SENTINELS_PER_SITE]
    return picks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--no-header", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset_dir.expanduser().resolve()
    header = not args.no_header
    columns = None if header else readme_columns(dataset / "README.md")

    per_site_records: dict[str, list[dict[str, str]]] = {}
    for site in SITES:
        site_columns, records = load_site(dataset / site, header=header, columns=columns)
        if columns is not None and site_columns != columns:
            raise SystemExit(
                f"{site} declares a different schema than {SITES[0]}: {site_columns} != {columns} — "
                "cross-site schema drift would corrupt the ground truth"
            )
        columns = site_columns
        per_site_records[site] = records

    all_records = [record for records in per_site_records.values() for record in records]
    numeric = [column for column in columns if is_numeric_column(all_records, column)]
    categorical = [column for column in columns if column not in numeric]

    def stats_for(records: list[dict[str, str]]) -> dict[str, dict[str, float]]:
        return {
            feature: feature_stats([float(record[feature]) for record in records]) for feature in numeric
        }

    payload = {
        "schema_version": 1,
        "dataset": dataset.name,
        "header": header,
        "columns": columns,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "csv_sha256": {
            site: hashlib.sha256((dataset / site / "data.csv").read_bytes()).hexdigest() for site in SITES
        },
        "sites": {
            site: {"count": len(records), "features": stats_for(records)}
            for site, records in per_site_records.items()
        },
        "global": {"count": len(all_records), "features": stats_for(all_records)},
        "sentinels": [row for site in SITES for row in sentinel_rows(dataset / site, header=header)],
        "min_count_floor": 10,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"{dataset.name}: {payload['global']['count']} rows, "
        f"{len(numeric)} numeric / {len(categorical)} categorical features -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
