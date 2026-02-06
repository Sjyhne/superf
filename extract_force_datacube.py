#!/usr/bin/env python3
"""
Extract RGB images from a FORCE datacube tile into a folder of images,
so that optimize_against_folder.py can be used for super-resolution.

FORCE datacube layout (see https://force-eo.readthedocs.io):
  - Datacube root contains tile directories: X0001_Y0002, X0001_Y0003, ...
  - Each tile dir contains GeoTIFFs: YYYYMMDD_LEVEL2_SEN2A_BOA.tif (and QAI, etc.)
  - BOA = Bottom-of-Atmosphere reflectance, scale 10000, nodata -9999
  - First three bands are typically Blue, Green, Red (Landsat 8/9, Sentinel-2)

Usage:
  python extract_force_datacube.py --datacube /path/to/level2 --tile X0134_Y0097 --output_dir ./force_extract
  python optimize_against_folder.py --input_folder ./force_extract --output_folder ./sr_results --df 2
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import tifffile

# FORCE BOA scale and nodata
FORCE_BOA_SCALE = 10000.0
FORCE_NODATA = -9999


def parse_force_filename(path: Path) -> dict | None:
    """Parse FORCE 29-digit convention: YYYYMMDD_LEVEL2_SEN2A_BOA.tif"""
    name = path.stem
    # Optional: allow LEVEL3 and other products
    m = re.match(r"(\d{8})_LEVEL\d_([A-Z0-9]+)_([A-Z0-9]+)$", name, re.IGNORECASE)
    if not m:
        return None
    return {"date": m.group(1), "sensor": m.group(2), "product": m.group(3)}


def find_boa_files(tile_dir: Path, product: str = "BOA", date_from: str | None = None,
                   date_to: str | None = None, sensors: list[str] | None = None,
                   max_files: int | None = None) -> list[Path]:
    """List BOA (or other product) GeoTIFFs in a tile directory, optionally filtered."""
    if not tile_dir.is_dir():
        return []
    pattern = f"*_*_{product}.tif"
    files = sorted(tile_dir.glob(pattern))
    out = []
    for f in files:
        info = parse_force_filename(f)
        if not info:
            continue
        if date_from and info["date"] < date_from:
            continue
        if date_to and info["date"] > date_to:
            continue
        if sensors and info["sensor"] not in sensors:
            continue
        out.append(f)
        if max_files and len(out) >= max_files:
            break
    return out


def read_force_boa_rgb(path: Path, band_indices: tuple[int, int, int] = (0, 1, 2),
                       nodata: int = FORCE_NODATA, scale: float = FORCE_BOA_SCALE) -> np.ndarray:
    """
    Read a FORCE BOA GeoTIFF and return an HWC uint8 RGB array.
    band_indices: 0-based indices for R, G, B (FORCE L2: 0=Blue, 1=Green, 2=Red → default is BGR order; we output RGB).
    """
    data = tifffile.imread(path)
    # FORCE: signed 16-bit, band order typically (bands, H, W) for multi-band
    if data.ndim == 2:
        data = np.asarray(data, dtype=np.float32)
        data = np.clip(data, 0, None)
        data[data == nodata] = np.nan
        rgb = np.stack([data, data, data], axis=-1)
    else:
        data = np.asarray(data, dtype=np.float32)
        if data.shape[0] in (3, 4, 6, 10):
            # (C, H, W)
            r = np.clip(data[band_indices[0]], 0, None)
            g = np.clip(data[band_indices[1]], 0, None)
            b = np.clip(data[band_indices[2]], 0, None)
            r[r == nodata] = np.nan
            g[g == nodata] = np.nan
            b[b == nodata] = np.nan
            rgb = np.stack([r, g, b], axis=-1)
        else:
            # (H, W, C)
            rgb = np.clip(data[:, :, list(band_indices)], 0, None)
            rgb[rgb == nodata] = np.nan
    # Scale 0–10000 → 0–1, then to 0–255; nan → 0
    rgb = rgb / scale
    rgb = np.nan_to_num(rgb, nan=0.0)
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Extract RGB images from a FORCE datacube tile for use with optimize_against_folder.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--datacube", type=str, required=True,
                        help="Path to FORCE datacube root (e.g. .../level2 or .../level3)")
    parser.add_argument("--tile", type=str, required=True,
                        help="Tile ID (e.g. X0134_Y0097)")
    parser.add_argument("--output_dir", type=str, default="force_extract",
                        help="Output directory for extracted PNGs (default: force_extract)")
    parser.add_argument("--product", type=str, default="BOA",
                        help="Product type to extract (default: BOA). Use BOA or TOA for reflectance.")
    parser.add_argument("--band_indices", type=int, nargs=3, default=[0, 1, 2],
                        metavar=("R", "G", "B"),
                        help="0-based band indices for R, G, B (default: 0 1 2 = Blue Green Red)")
    parser.add_argument("--date_from", type=str, default=None,
                        help="Filter: only include dates >= YYYYMMDD")
    parser.add_argument("--date_to", type=str, default=None,
                        help="Filter: only include dates <= YYYYMMDD")
    parser.add_argument("--sensors", type=str, nargs="*", default=None,
                        help="Filter: only these sensor IDs (e.g. SEN2A LND08)")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of images to extract (default: all)")
    parser.add_argument("--prefix", type=str, default="",
                        help="Filename prefix for output PNGs (default: date_sensor_)")
    args = parser.parse_args()

    datacube_root = Path(args.datacube)
    tile_dir = datacube_root / args.tile
    if not tile_dir.is_dir():
        raise SystemExit(f"Tile directory not found: {tile_dir}")

    files = find_boa_files(
        tile_dir,
        product=args.product,
        date_from=args.date_from,
        date_to=args.date_to,
        sensors=args.sensors,
        max_files=args.max_files,
    )
    if not files:
        raise SystemExit(f"No {args.product} files found in {tile_dir}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    band_indices = tuple(args.band_indices)

    manifest = []
    for i, path in enumerate(files):
        info = parse_force_filename(path)
        if not info:
            continue
        try:
            rgb = read_force_boa_rgb(path, band_indices=band_indices)
        except Exception as e:
            print(f"Skip {path.name}: {e}")
            continue
        # Output filename: optional prefix + date_sensor + index so order is stable for optimize_against_folder
        name = f"{args.prefix}{info['date']}_{info['sensor']}_{i:04d}.png"
        out_path = out_dir / name
        cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        manifest.append({"source": path.name, "out": name, "date": info["date"], "sensor": info["sensor"]})
        print(f"Wrote {out_path.name} ({rgb.shape[1]}x{rgb.shape[0]})")

    # Write a small manifest so user knows what was extracted
    manifest_path = out_dir / "extract_manifest.txt"
    with open(manifest_path, "w") as f:
        f.write(f"# FORCE datacube extract: {datacube_root} tile {args.tile}\n")
        f.write(f"# Product: {args.product}  Band indices (R,G,B): {band_indices}\n")
        for m in manifest:
            f.write(f"{m['out']}\t{m['source']}\t{m['date']}\t{m['sensor']}\n")
    print(f"\nExtracted {len(manifest)} images to {out_dir}")
    print(f"Run super-resolution with:")
    print(f"  python optimize_against_folder.py --input_folder {out_dir} --output_folder <sr_output> --df 2")


if __name__ == "__main__":
    main()
