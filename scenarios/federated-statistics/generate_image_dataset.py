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
make pixel-intensity statistics meaningful. Images are SYNTHETIC
chest-radiograph-like phantoms (no real patient data): a torso ellipse on an
air-dark background, two darker lung fields, a bright mediastinum/spine band,
rib banding across the lungs, and radial vignetting — so intensity histograms
get realistic bimodal air/tissue structure instead of a single noise bump.

Site counts are deliberately distinct (anti-site-mixup) and each site has its
own exposure/noise profile (per-site histograms differ visibly):

    <out_dir>/site-1/  450 images, under-exposed
    <out_dir>/site-2/  350 images, over-exposed, low noise
    <out_dir>/site-3/  200 images, dark and noisy

PNGs are written with a pure-stdlib encoder (no Pillow), so rerunning with
the same seed is byte-identical and the dataset stays regenerable. A summary
of the exact per-site image counts and pixel-intensity moments is printed —
the input for a future ground-truth file once image statistics land in the
fed-stats skill.

Usage:
    generate_image_dataset.py <out_dir> [--seed 20260709] [--size 128]
"""

from __future__ import annotations

import argparse
import math
import random
import struct
import sys
import zlib
from pathlib import Path

# (count, exposure offset, per-image exposure jitter, pixel noise stddev)
SITE_PROFILES = {
    "site-1": (450, -12.0, 8.0, 7.0),
    "site-2": (350, 20.0, 6.0, 4.0),
    "site-3": (200, -26.0, 12.0, 11.0),
}


def png_bytes(pixels: list[list[int]]) -> bytes:
    """Encode 8-bit grayscale rows as a PNG (pure stdlib, deterministic)."""

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    height, width = len(pixels), len(pixels[0])
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)  # 8-bit grayscale
    raw = b"".join(b"\x00" + bytes(row) for row in pixels)  # filter 0 per scanline
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def synthesize_image(size: int, exposure: float, noise: float, rng: random.Random) -> list[list[int]]:
    """One chest-radiograph-like phantom with per-patient geometry variation."""

    shift_x = rng.uniform(-0.06, 0.06)
    shift_y = rng.uniform(-0.05, 0.05)
    torso_rx = rng.uniform(0.78, 0.92)
    torso_ry = rng.uniform(0.88, 1.00)
    lung_rx = torso_rx * rng.uniform(0.33, 0.40)
    lung_ry = torso_ry * rng.uniform(0.50, 0.62)
    lung_dx = torso_rx * rng.uniform(0.42, 0.50)
    spine_width = rng.uniform(0.14, 0.20)
    rib_freq = rng.uniform(9.0, 12.0)
    rib_phase = rng.uniform(0.0, 2.0 * math.pi)

    rows = []
    for j in range(size):
        y = (2.0 * j / (size - 1)) - 1.0 - shift_y
        row = []
        for i in range(size):
            x = (2.0 * i / (size - 1)) - 1.0 - shift_x
            torso = (x / torso_rx) ** 2 + (y / torso_ry) ** 2
            if torso >= 1.0:
                value = 16.0  # air: near-black background
            else:
                # soft tissue, brightening toward the central mediastinum/spine
                value = 118.0 + 72.0 * math.exp(-((x / spine_width) ** 2))
                for side in (-1.0, 1.0):
                    lung = ((x - side * lung_dx) / lung_rx) ** 2 + ((y + 0.05) / lung_ry) ** 2
                    if lung < 1.0:
                        field = math.sqrt(1.0 - lung)
                        value -= 78.0 * field  # radiolucent lung field
                        # ribs: brighter bands crossing the lung field
                        value += 20.0 * field * max(0.0, math.sin(rib_freq * y + rib_phase)) ** 2
                value *= 1.0 - 0.18 * (x * x + y * y)  # vignette
            value += exposure + rng.gauss(0.0, noise)
            row.append(max(0, min(255, int(round(value)))))
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = args.out_dir.expanduser()
    total_pixels = 0
    for site, (count, site_exposure, exposure_jitter, noise) in SITE_PROFILES.items():
        site_dir = out / site
        site_dir.mkdir(parents=True, exist_ok=True)
        for stale in site_dir.glob("*.png"):
            stale.unlink()
        pixel_sum = 0.0
        pixel_sq = 0.0
        for index in range(count):
            exposure = rng.gauss(site_exposure, exposure_jitter)
            pixels = synthesize_image(args.size, exposure, noise, rng)
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
        "# Chest Imaging Extract for Federated Statistics\n\n"
        "Each site keeps its own imaging data: `site-1/`, `site-2/`, and\n"
        f"`site-3/`, each a flat folder of 8-bit grayscale PNG files ({args.size}x{args.size}) —\n"
        "synthetic chest-radiograph-like phantoms (no real patient data), with\n"
        "site-specific exposure and noise characteristics.\n"
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
