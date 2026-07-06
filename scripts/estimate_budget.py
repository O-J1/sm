"""Project storage cost of the tiered subsampling policy and sweep the
artist cutoff so the collection fits the disk budget.

Policy per record (user decision):
  - Tier A: records by a top-N ranked artist keep every image at native
    resolution (JXL q=97).
  - Tier B: the record's main image (first media item) capped at 8,388,608 px.
  - Tier C: additional views capped at 4,194,304 px.
  - Tier D (optional, --median-cap): views beyond the per-unit median
    images-per-record are dropped entirely (for non-tier-A records).

Sizes are estimated as pixels x bytes-per-pixel, using measured values from
reports/calibration.json when present (run calibrate_bpp.py), otherwise
conservative fallbacks. A single streaming pass accumulates bytes per artist
rank, so the whole cutoff curve comes from one database scan.

Usage:
    python scripts/estimate_budget.py [--db PATH] [--budget-tb 8]
                                      [--top-ns 0,250,500,1000,2500,5000,10000,all]
                                      [--median-cap] [--cc0-only]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from _common import (
    MAIN_MAX_PIXELS,
    SECONDARY_MAX_PIXELS,
    GroupedCursor,
    add_common_arguments,
    histogram_median,
    load_bpp,
    load_rank_map,
    median,
    name_key,
    open_db,
    write_csv,
)

UNRANKED = 1 << 30


def unit_medians(conn, media_type: str) -> tuple[dict[str, int], dict[str, int]]:
    """Per-unit median pixels (dims fallback) and median images-per-record."""
    pixels: dict[str, list[int]] = defaultdict(list)
    for unit, px in conn.execute(
        """
        SELECT mr.unit_code, mr.width * mr.height
        FROM media_resources mr JOIN media_items mi ON mi.media_key = mr.media_key
        WHERE mr.preferred_download = 1 AND mi.media_type = ? AND mr.width > 0 AND mr.height > 0
        """,
        (media_type,),
    ):
        pixels[unit].append(px)
    median_px = {unit: int(median(values)) for unit, values in pixels.items()}

    hist: dict[str, dict[int, int]] = defaultdict(dict)
    for unit, cnt, records in conn.execute(
        """
        SELECT unit_code, cnt, COUNT(*)
        FROM (SELECT mi.unit_code, mi.record_id, COUNT(*) AS cnt
              FROM media_items mi WHERE mi.media_type = ?
              GROUP BY mi.unit_code, mi.record_id)
        GROUP BY unit_code, cnt
        """,
        (media_type,),
    ):
        hist[unit][cnt] = records
    median_images = {unit: max(1, int(histogram_median(h))) for unit, h in hist.items()}
    return median_px, median_images


def iter_record_images(conn, media_type: str, cc0_only: bool):
    """Yield (record_id, unit_code, [rows]) grouped by record, plus a facet lookup."""
    rights_clause = "AND mi.usage_access = 'CC0'" if cc0_only else ""
    cursor = conn.execute(
        f"""
        SELECT mr.record_id, mr.unit_code, mi.position, mr.width, mr.height, mr.resource_key, mr.url
        FROM media_resources mr JOIN media_items mi ON mi.media_key = mr.media_key
        WHERE mr.preferred_download = 1 AND mi.media_type = ? {rights_clause}
        ORDER BY mr.record_id, mi.position
        """,
        (media_type,),
    )
    group_key = None
    group: list = []
    for row in cursor:
        if row[0] != group_key and group_key is not None:
            yield group_key, group[0][1], group
            group = []
        group_key = row[0]
        group.append(row)
    if group_key is not None:
        yield group_key, group[0][1], group


def record_rank(facets: GroupedCursor, record_id: str, rank_map: dict[str, int]) -> int:
    best = UNRANKED
    for _, value in facets.rows_for(record_id):
        rank = rank_map.get(name_key(value), UNRANKED)
        if rank < best:
            best = rank
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--rankings-csv", type=Path, default=None)
    parser.add_argument("--budget-tb", type=float, default=8.0)
    parser.add_argument("--top-ns", default="0,250,500,1000,2500,5000,10000,25000,all")
    parser.add_argument("--median-cap", action="store_true", help="Also drop views beyond the per-unit median (tier D).")
    parser.add_argument("--cc0-only", action="store_true")
    parser.add_argument("--media-type", default="Images")
    args = parser.parse_args()

    conn = open_db(args.db)
    reports = args.reports_dir
    rankings_csv = args.rankings_csv or (reports / "artist_rankings.csv")
    rank_map = load_rank_map(rankings_csv)
    native_bpp, downscaled_bpp = load_bpp(reports / "calibration.json")
    print(f"bpp model: native={native_bpp} downscaled={downscaled_bpp} "
          f"({'calibrated' if (reports / 'calibration.json').exists() else 'FALLBACK - run calibrate_bpp.py'})")

    median_px, median_images = unit_medians(conn, args.media_type)
    facets = GroupedCursor(
        conn.execute("SELECT record_id, value FROM record_facets WHERE facet_type = 'name' ORDER BY record_id")
    )

    # Per-rank accumulators: [images, native_bytes, capped_bytes, capped_bytes_median_cap, dropped_images]
    per_rank: dict[int, list[float]] = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0])
    total_images = 0

    for record_id, unit_code, images in iter_record_images(conn, args.media_type, args.cc0_only):
        rank = record_rank(facets, record_id, rank_map)
        bucket = per_rank[rank]
        image_cap = median_images.get(unit_code, 1)
        for index, (_, _, _, width, height, _, _) in enumerate(images):
            px = width * height if width and height else median_px.get(unit_code, 4_000_000)
            cap = MAIN_MAX_PIXELS if index == 0 else SECONDARY_MAX_PIXELS
            native_bytes = px * native_bpp
            capped_bytes = native_bytes if px <= cap else cap * downscaled_bpp
            bucket[0] += 1
            bucket[1] += native_bytes
            bucket[2] += capped_bytes
            if index < image_cap:
                bucket[3] += capped_bytes
            else:
                bucket[4] += 1
            total_images += 1

    ranks = sorted(per_rank)
    ranked_artists = len(rank_map)
    sweep = []
    for token in args.top_ns.split(","):
        token = token.strip().lower()
        sweep.append(ranked_artists if token == "all" else int(token))

    capped_index = 3 if args.median_cap else 2
    rows = []
    print(f"\ntotal images: {total_images:,}  ranked artists: {ranked_artists:,}  "
          f"scenario: {'median-cap (tier D)' if args.median_cap else 'no tier D'}")
    print(f"{'top_n':>8} {'total_TB':>10} {'full_res':>12} {'downscaled':>12} {'dropped':>10} {'fits_' + str(args.budget_tb) + 'TB':>10}")
    for top_n in sweep:
        total_bytes = 0.0
        full_res = 0
        dropped = 0
        for rank in ranks:
            images, native_bytes, capped_bytes, capped_median, dropped_count = per_rank[rank]
            if rank < top_n:
                total_bytes += native_bytes
                full_res += images
            else:
                total_bytes += (capped_median if args.median_cap else capped_bytes)
                dropped += dropped_count if args.median_cap else 0
        total_tb = total_bytes / 1e12
        downscaled = total_images - full_res - dropped
        fits = "yes" if total_tb <= args.budget_tb else "no"
        print(f"{top_n:>8,} {total_tb:>10.2f} {full_res:>12,} {downscaled:>12,} {dropped:>10,} {fits:>10}")
        rows.append([top_n, args.median_cap, round(total_tb, 3), full_res, downscaled, dropped, fits])

    path = write_csv(
        reports / "budget_curve.csv",
        ["top_n", "median_cap", "total_tb", "full_res_images", "downscaled_images", "dropped_images", "fits_budget"],
        rows,
    )
    print(f"\noutput: {path}")


if __name__ == "__main__":
    main()
