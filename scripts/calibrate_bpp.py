"""Measure real JPEG XL bytes-per-pixel on a small stratified image sample.

Since no images are downloaded yet, the storage budget must be estimated
from metadata (width x height). This script downloads a stratified sample
(per unit x megapixel bucket), encodes it with cjxl at the target quality
(default q=95), and measures bytes-per-pixel at native resolution and at
the two policy caps (8,388,608 px main / 2,097,152 px secondary, downscaled
with OpenCV Lanczos like the existing conversion pipeline).

It also measures bytes-per-pixel of the small pretraining derivatives
(256px/512px longest edge, JPEG by default). The production pipeline
downscales those with ImageMagick Lanczos2Sharp; OpenCV Lanczos4 is used
here as a close stand-in since the file-size difference is negligible.

Optionally probes whether the IDS delivery service honours a `max=` size
parameter - if it does, tier B/C images can be downloaded pre-downscaled,
saving enormous bandwidth.

Output: reports/calibration.json (consumed by estimate_budget.py).

Usage:
    python scripts/calibrate_bpp.py [--db PATH] [--max-images 200] [--quality 95]
                                    [--cjxl cjxl] [--probe-ids] [--keep-files]
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx

from _common import (
    DEFAULT_DERIVATIVE_QUALITY,
    DEFAULT_JXL_QUALITY,
    DERIVATIVE_EDGES,
    MAIN_MAX_PIXELS,
    SECONDARY_MAX_PIXELS,
    add_common_arguments,
    ensure_dir,
    median,
    mp_bucket,
    open_db,
)

try:
    cv2: Any = importlib.import_module("cv2")
except Exception:
    cv2 = None

USER_AGENT = "smithsonian-subsampling"


def fmt_bytes(size: int | None) -> str:
    if size is None:
        return "unknown"
    units = ("B", "KB", "MB", "GB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def fmt_mp(pixels: int | None) -> str:
    return "unknown MP" if not pixels else f"{pixels / 1_000_000:.1f} MP"


def sample_resources(conn, per_stratum: int, max_images: int) -> list[dict]:
    """Uniform random sampling via rowid probes - avoids full-table scans."""
    max_rowid = conn.execute("SELECT max(rowid) FROM media_assets").fetchone()[0] or 0
    if not max_rowid:
        return []
    samples: list[dict] = []
    seen: set[int] = set()
    quotas: dict[tuple[str, str], int] = {}
    attempts = 0
    while len(samples) < max_images and attempts < max_images * 50:
        attempts += 1
        row = conn.execute(
            """
            SELECT rowid, media_key AS resource_key, url,
                   resource_width AS width, resource_height AS height, unit_code
            FROM media_assets
            WHERE rowid >= ? AND downloadable = 1 AND media_type = 'Images'
            ORDER BY rowid LIMIT 1
            """,
            (random.randint(1, max_rowid),),
        ).fetchone()
        if row is None or row["rowid"] in seen:
            continue
        seen.add(row["rowid"])
        bucket = mp_bucket(row["width"] * row["height"]) if row["width"] and row["height"] else "unknown"
        key = (row["unit_code"], bucket)
        if quotas.get(key, 0) >= per_stratum:
            continue
        quotas[key] = quotas.get(key, 0) + 1
        samples.append({**dict(row), "bucket": bucket})
        if len(samples) % 25 == 0 or len(samples) == max_images:
            print(f"  sampled {len(samples)}/{max_images} resources after {attempts:,} row probes")
    return samples


def download(client: httpx.Client, url: str, target: Path) -> Path | None:
    try:
        started = time.monotonic()
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with target.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
        elapsed = time.monotonic() - started
        print(f"    downloaded {fmt_bytes(target.stat().st_size)} in {elapsed:.1f}s")
        return target
    except httpx.HTTPError as error:
        print(f"  ! download failed: {url} ({error})")
        return None


def encode_jxl(cjxl: str, source: Path, target: Path, quality: int) -> int | None:
    command = [cjxl, str(source), str(target), "-q", str(quality), "--lossless_jpeg=0", "--num_threads", "4"]
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not target.exists():
        print(f"  ! cjxl failed for {source.name}: {result.stderr.strip()[:200]}")
        return None
    elapsed = time.monotonic() - started
    print(f"    encoded {target.name} -> {fmt_bytes(target.stat().st_size)} in {elapsed:.1f}s")
    return target.stat().st_size


def downscale(image, max_pixels: int):
    assert cv2 is not None
    height, width = image.shape[:2]
    pixels = width * height
    if pixels <= max_pixels:
        return image
    scale = (max_pixels / pixels) ** 0.5
    return cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_LANCZOS4)


def downscale_edge(image, edge: int):
    """Aspect-preserving resize so the longest edge equals `edge` (no upscaling)."""
    assert cv2 is not None
    height, width = image.shape[:2]
    scale = edge / max(width, height)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_LANCZOS4)


def probe_ids(client: httpx.Client, samples: list[dict], work_dir: Path, limit: int = 5) -> list[dict]:
    results = []
    candidates = [s for s in samples if "ids.si.edu" in s["url"]][:limit]
    for sample in candidates:
        joiner = "&" if "?" in sample["url"] else "?"
        url = f"{sample['url']}{joiner}max=3000"
        target = work_dir / f"ids_probe_{sample['resource_key'][:12]}.bin"
        path = download(client, url, target)
        entry = {"url": url, "native_width": sample["width"], "native_height": sample["height"]}
        if path:
            entry["bytes"] = path.stat().st_size
            if cv2 is not None:
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is not None:
                    entry["returned_height"], entry["returned_width"] = image.shape[:2]
                    if sample["width"] and sample["height"]:
                        entry["honored"] = max(image.shape[:2]) <= 3000 < max(sample["width"], sample["height"])
        results.append(entry)
    return results


def stats(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 4),
        "median": round(median(values), 4),
        "p90": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))], 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--per-stratum", type=int, default=3)
    parser.add_argument("--quality", type=int, default=DEFAULT_JXL_QUALITY)
    parser.add_argument("--derivative-edges", default=",".join(str(edge) for edge in DERIVATIVE_EDGES),
                        help="Comma-separated longest-edge sizes for derivative bpp measurement ('' = skip).")
    parser.add_argument("--derivative-quality", type=int, default=DEFAULT_DERIVATIVE_QUALITY,
                        help="JPEG quality used for derivative measurement.")
    parser.add_argument("--cjxl", default="cjxl", help="Path to the cjxl binary.")
    parser.add_argument("--work-dir", type=Path, default=None, help="Scratch dir (default: temp).")
    parser.add_argument("--keep-files", action="store_true")
    parser.add_argument("--probe-ids", action="store_true", help="Test whether ids.si.edu honours max= resizing.")
    args = parser.parse_args()

    if shutil.which(args.cjxl) is None:
        raise SystemExit(f"cjxl not found ({args.cjxl!r}); install libjxl or pass --cjxl")
    if cv2 is None:
        print("warning: OpenCV unavailable; downscaled cap measurements will be skipped")

    print(f"opening database: {args.db}")
    conn = open_db(args.db)
    reports = ensure_dir(args.reports_dir)
    work_dir = ensure_dir(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="smithsonian_bpp_"))

    print(f"sampling up to {args.max_images} images ({args.per_stratum} per unit/bucket stratum)")
    samples = sample_resources(conn, args.per_stratum, args.max_images)
    print(f"sampled {len(samples)} resources across units/buckets; downloading to {work_dir}")

    native_bpp: list[float] = []
    main_bpp: list[float] = []
    secondary_bpp: list[float] = []
    derivative_edges = [int(part) for part in str(args.derivative_edges).split(",") if part.strip()]
    derivative_bpp: dict[int, list[float]] = {edge: [] for edge in derivative_edges}
    per_unit: dict[str, list[float]] = defaultdict(list)
    per_unit_px: dict[str, list[int]] = defaultdict(list)

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=120.0, follow_redirects=True) as client:
        for index, sample in enumerate(samples, 1):
            pixels = sample["width"] * sample["height"] if sample["width"] and sample["height"] else None
            print(
                f"[{index}/{len(samples)}] {sample['unit_code']} {sample['bucket']} "
                f"{fmt_mp(pixels)} {sample['resource_key'][:12]}"
            )
            source = download(client, sample["url"], work_dir / f"src_{sample['resource_key'][:12]}.bin")
            if source is None:
                continue
            jxl_size = encode_jxl(args.cjxl, source, source.with_suffix(".jxl"), args.quality)
            image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED) if cv2 is not None else None
            if image is not None:
                pixels = image.shape[0] * image.shape[1]
                per_unit_px[sample["unit_code"]].append(pixels)
            elif sample["width"] and sample["height"]:
                pixels = sample["width"] * sample["height"]
            else:
                pixels = 0
            if jxl_size and pixels:
                bpp = jxl_size / pixels
                native_bpp.append(bpp)
                per_unit[sample["unit_code"]].append(bpp)

            if image is not None:
                assert cv2 is not None
                for cap, sink in ((MAIN_MAX_PIXELS, main_bpp), (SECONDARY_MAX_PIXELS, secondary_bpp)):
                    if pixels <= cap:
                        continue
                    print(f"    downscaling to cap {fmt_mp(cap)}")
                    scaled = downscale(image, cap)
                    png = source.with_name(f"{source.stem}_{cap}.png")
                    cv2.imwrite(str(png), scaled)
                    scaled_size = encode_jxl(args.cjxl, png, png.with_suffix(".jxl"), args.quality)
                    if scaled_size:
                        sink.append(scaled_size / (scaled.shape[0] * scaled.shape[1]))
                    if not args.keep_files:
                        png.unlink(missing_ok=True)
                        png.with_suffix(".jxl").unlink(missing_ok=True)
                for edge in derivative_edges:
                    if max(image.shape[:2]) <= edge:
                        continue
                    scaled = downscale_edge(image, edge)
                    jpg = source.with_name(f"{source.stem}_deriv{edge}.jpg")
                    if cv2.imwrite(str(jpg), scaled, [int(cv2.IMWRITE_JPEG_QUALITY), args.derivative_quality]):
                        derivative_bpp[edge].append(jpg.stat().st_size / (scaled.shape[0] * scaled.shape[1]))
                    if not args.keep_files:
                        jpg.unlink(missing_ok=True)
            if not args.keep_files:
                source.unlink(missing_ok=True)
                source.with_suffix(".jxl").unlink(missing_ok=True)
                print("    cleaned temporary files")
            if index % 10 == 0 or index == len(samples):
                print(f"  {index}/{len(samples)} processed")

        if args.probe_ids:
            print("probing IDS max= resizing")
        ids_results = probe_ids(client, samples, work_dir) if args.probe_ids else []

    calibration = {
        "quality": args.quality,
        "native_bpp": stats(native_bpp),
        "main_cap_bpp": stats(main_bpp),
        "secondary_cap_bpp": stats(secondary_bpp),
        "derivative_quality": args.derivative_quality,
        "derivative_bpp": {str(edge): stats(values) for edge, values in derivative_bpp.items()},
        "per_unit_native_median": {unit: round(median(values), 4) for unit, values in sorted(per_unit.items())},
        "per_unit_measured_mp_median": {
            unit: round(median(values) / 1e6, 3) for unit, values in sorted(per_unit_px.items())
        },
        "ids_probe": ids_results,
    }
    path = reports / "calibration.json"
    path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in calibration.items() if not k.startswith("per_unit")}, indent=2))
    print(f"output: {path}")
    if not args.keep_files and args.work_dir is None:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
