"""Export the final download/conversion manifest for a chosen artist cutoff.

Writes a JSONL manifest with one line per media resource:
  {"resource_key", "record_id", "unit_code", "url", "tier",
   "target_max_pixels", "jxl_quality", "est_bytes", "width", "height",
   "artist_rank"}

Tiers: "full" (top artist, native resolution), "main" (<= 8,388,608 px),
"secondary" (<= 4,194,304 px), "drop" (only with --median-cap; excluded
from the manifest unless --include-drops). Pick the cutoff with
estimate_budget.py first.

Usage:
    python scripts/export_manifest.py --top-n 2500 [--median-cap] [--cc0-only]
"""

from __future__ import annotations

import argparse
import json

from _common import (
    DEFAULT_JXL_QUALITY,
    MAIN_MAX_PIXELS,
    SECONDARY_MAX_PIXELS,
    GroupedCursor,
    add_common_arguments,
    ensure_dir,
    load_bpp,
    load_rank_map,
    open_db,
)
from estimate_budget import UNRANKED, iter_record_images, record_rank, unit_medians


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--top-n", type=int, required=True, help="Artist rank cutoff for tier A (from budget_curve.csv).")
    parser.add_argument("--median-cap", action="store_true")
    parser.add_argument("--include-drops", action="store_true", help="Keep tier D rows in the manifest for auditing.")
    parser.add_argument("--cc0-only", action="store_true")
    parser.add_argument("--media-type", default="Images")
    parser.add_argument("--quality", type=int, default=DEFAULT_JXL_QUALITY)
    args = parser.parse_args()

    conn = open_db(args.db)
    reports = ensure_dir(args.reports_dir)
    rank_map = load_rank_map(reports / "artist_rankings.csv")
    native_bpp, downscaled_bpp = load_bpp(reports / "calibration.json")
    median_px, median_images = unit_medians(conn, args.media_type)
    facets = GroupedCursor(
        conn.execute("SELECT record_id, content FROM record_freetext WHERE category = 'name' ORDER BY record_id")
    )

    manifest_path = reports / "manifest.jsonl"
    totals = {"full": 0, "main": 0, "secondary": 0, "drop": 0}
    est_bytes_total = 0.0

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record_id, unit_code, images in iter_record_images(conn, args.media_type, args.cc0_only):
            rank = record_rank(facets, record_id, rank_map)
            is_tier_a = rank < args.top_n
            image_cap = median_images.get(unit_code, 1)
            for index, (_, _, _, width, height, resource_key, url) in enumerate(images):
                px = width * height if width and height else median_px.get(unit_code, 4_000_000)
                cap = MAIN_MAX_PIXELS if index == 0 else SECONDARY_MAX_PIXELS
                if is_tier_a:
                    tier, target, est = "full", None, px * native_bpp
                elif args.median_cap and index >= image_cap:
                    tier, target, est = "drop", None, 0.0
                elif px <= cap:
                    tier = "main" if index == 0 else "secondary"
                    target, est = cap, px * native_bpp
                else:
                    tier = "main" if index == 0 else "secondary"
                    target, est = cap, cap * downscaled_bpp
                totals[tier] += 1
                if tier == "drop" and not args.include_drops:
                    continue
                est_bytes_total += est
                handle.write(
                    json.dumps(
                        {
                            "resource_key": resource_key,
                            "record_id": record_id,
                            "unit_code": unit_code,
                            "url": url,
                            "tier": tier,
                            "target_max_pixels": target,
                            "jxl_quality": args.quality,
                            "est_bytes": round(est),
                            "width": width,
                            "height": height,
                            "artist_rank": rank if rank != UNRANKED else None,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")

    kept = totals["full"] + totals["main"] + totals["secondary"]
    print(f"tier full:      {totals['full']:>12,}")
    print(f"tier main:      {totals['main']:>12,}")
    print(f"tier secondary: {totals['secondary']:>12,}")
    print(f"tier drop:      {totals['drop']:>12,}")
    print(f"kept images:    {kept:>12,}")
    print(f"estimated size: {est_bytes_total / 1e12:>12.2f} TB")
    print(f"output: {manifest_path}")


if __name__ == "__main__":
    main()
