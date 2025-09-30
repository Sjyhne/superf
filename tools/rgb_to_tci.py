#!/usr/bin/env python3
"""
Convert an RGB PNG (assumed linear-ish reflectance) to a TCI-like visualization.

Steps:
- Read image as float in [0,1] if 8-bit, otherwise normalize by dtype max
- Optional white balance by per-channel percentiles
- Global percentile stretch (e.g., 2-98%)
- sRGB gamma encode
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def read_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    else:
        arr = arr.astype(np.float32)
        # assume already in 0..1
    return np.clip(arr, 0.0, 1.0)


def percentile_stretch(
    arr: np.ndarray,
    p_low: float = 2.0,
    p_high: float = 98.0,
    per_channel: bool = True,
) -> np.ndarray:
    out = arr.copy()
    if per_channel:
        for c in range(3):
            low = np.percentile(out[..., c], p_low)
            high = np.percentile(out[..., c], p_high)
            if high <= low:
                continue
            out[..., c] = (out[..., c] - low) / (high - low)
    else:
        low = np.percentile(out, p_low)
        high = np.percentile(out, p_high)
        if high > low:
            out = (out - low) / (high - low)
    return np.clip(out, 0.0, 1.0)


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    a = 0.055
    threshold = 0.0031308
    out = np.where(
        linear <= threshold,
        linear * 12.92,
        (1 + a) * np.power(np.clip(linear, 0.0, 1.0), 1 / 2.4) - a,
    )
    return np.clip(out, 0.0, 1.0)


def to_uint8(arr01: np.ndarray) -> np.ndarray:
    return (np.clip(arr01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def convert_to_tci(
    src: Path,
    dst: Path,
    p_low: float = 1.0,
    p_high: float = 99.0,
    per_channel: bool = True,
) -> None:
    arr = read_image(src)
    stretched = percentile_stretch(arr, p_low, p_high, per_channel=per_channel)
    encoded = srgb_encode(stretched)
    out = Image.fromarray(to_uint8(encoded), mode="RGB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RGB PNG to TCI-like visualization")
    parser.add_argument("src", type=Path, help="Input RGB image (PNG/JPG)")
    parser.add_argument("dst", type=Path, help="Output TCI PNG path")
    parser.add_argument("--p-low", type=float, default=1.0, help="Low percentile")
    parser.add_argument("--p-high", type=float, default=99.0, help="High percentile")
    parser.add_argument(
        "--global",
        dest="per_channel",
        action="store_false",
        help="Use global stretch instead of per-channel",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_to_tci(args.src, args.dst, args.p_low, args.p_high, per_channel=args.per_channel)


if __name__ == "__main__":
    main()


