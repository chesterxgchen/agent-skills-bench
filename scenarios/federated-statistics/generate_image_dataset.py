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

"""Generate the image federated-statistics benchmark dataset (seeded).

Three sites, no train/valid split, each with enough 8-bit grayscale PNGs to
make pixel-intensity statistics meaningful. Site counts are deliberately
distinct (anti-site-mixup) and each site draws from a different intensity
profile (per-site histograms differ visibly):

    <out_dir>/site-1/  450 images, darker    (mean ~100)
    <out_dir>/site-2/  350 images, brighter  (mean ~140)
    <out_dir>/site-3/  200 images, high-variance (mean ~80)

PNGs are written with a pure-stdlib encoder (no Pillow), so rerunning with
the same seed is byte-identical and the dataset stays regenerable. A summary
of the exact per-site image counts and pixel-intensity moments is printed —
the input for a future ground-truth file once image statistics land in the
fed-stats skill.

Usage:
    generate_image_dataset.py <out_dir> [--seed 20260709] [--size 64]
"""

from __future__ import annotations

import argparse
import random
import struct
import sys
import zlib
from pathlib import Path

# (count, intensity mean, per-image mean jitter, in-image stddev)
SITE_PROFILES = {
    "site-1": (450, 100.0, 12.0, 30.0),
    "site-2": (350, 140.0, 10.0, 25.0),
    "site-3": (200, 80.0, 18.0, 40.0),
}


def png_bytes(pixels: list[list[int]]) -> bytes:
    """Encode 8-bit grayscale rows as a PNG (pure stdlib, deterministic)."""

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    height, width = len(pixels), len(pixels[0])
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)  # 8-bit grayscale
    raw = b"".join(b"\x00" + bytes(row) for row in pixels)  # filter 0 per scanline
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def synthesize_image(size: int, mean: float, stddev: float, rng: random.Random) -> list[list[int]]:
    return [
        [max(0, min(255, int(round(rng.gauss(mean, stddev))))) for _ in range(size)]
        for _ in range(size)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = args.out_dir.expanduser()
    total_pixels = 0
    for site, (count, site_mean, mean_jitter, stddev) in SITE_PROFILES.items():
        site_dir = out / site
        site_dir.mkdir(parents=True, exist_ok=True)
        pixel_sum = 0.0
        pixel_sq = 0.0
        for index in range(count):
            image_mean = rng.gauss(site_mean, mean_jitter)
            pixels = synthesize_image(args.size, image_mean, stddev, rng)
            (site_dir / f"img_{index:05d}.png").write_bytes(png_bytes(pixels))
            flat = [value for row in pixels for value in row]
            pixel_sum += sum(flat)
            pixel_sq += sum(value * value for value in flat)
        n = count * args.size * args.size
        total_pixels += n
        mean = pixel_sum / n
        var = pixel_sq / n - mean * mean
        print(f"{site}: {count} images, pixel mean {mean:.3f}, stddev {var ** 0.5:.3f}")

    (out / "README.md").write_text(
        "# Image Extract for Federated Statistics\n\n"
        "Each site keeps its own imaging data: `site-1/`, `site-2/`, and\n"
        "`site-3/`, each a flat folder of 8-bit grayscale PNG files (64x64).\n"
        "There is no train/valid split. Per-site image counts:\n\n"
        + "".join(f"- {site}: {profile[0]} images\n" for site, profile in SITE_PROFILES.items())
        + "\nIntended statistics: per-site and Global image counts and\n"
        "pixel-intensity histograms (0-255).\n",
        encoding="utf-8",
    )
    # The job ships its own dependency requirements (see the harness prewarm):
    # reading PNGs needs an image library; numpy for pixel math.
    (out / "requirements.txt").write_text("numpy\npillow\n", encoding="utf-8")
    print(f"wrote {sum(profile[0] for profile in SITE_PROFILES.values())} images ({total_pixels} pixels) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
