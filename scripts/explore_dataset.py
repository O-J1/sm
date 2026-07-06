"""Explore the Smithsonian metadata database.

Writes CSV reports (per-unit overview, media types, rights, megapixel
distribution, images-per-record histogram) to the reports directory and
prints a console summary. Read-only; works on partial databases.

Usage:
    python scripts/explore_dataset.py [--db PATH] [--reports-dir PATH]
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from _common import (
    MP_BUCKETS,
    add_common_arguments,
    ensure_dir,
    histogram_median,
    median,
    mp_bucket,
    open_db,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    args = parser.parse_args()
    conn = open_db(args.db)
    reports = ensure_dir(args.reports_dir)

    record_counts = dict(conn.execute("SELECT unit_code, COUNT(*) FROM records GROUP BY 1"))
    media_counts = dict(conn.execute("SELECT unit_code, COUNT(*) FROM media_items GROUP BY 1"))

    resource_stats: dict[str, tuple[int, int]] = {}
    for row in conn.execute(
        """
        SELECT unit_code,
               COUNT(*) AS preferred,
               SUM(CASE WHEN width > 0 AND height > 0 THEN 1 ELSE 0 END) AS with_dims
        FROM media_resources
        WHERE preferred_download = 1
        GROUP BY unit_code
        """
    ):
        resource_stats[row["unit_code"]] = (row["preferred"], row["with_dims"] or 0)

    # Megapixel distribution and per-unit pixel medians (streamed).
    bucket_counts: dict[str, Counter] = defaultdict(Counter)
    unit_pixels: dict[str, list[int]] = defaultdict(list)
    cursor = conn.execute(
        """
        SELECT unit_code, width * height AS px
        FROM media_resources
        WHERE preferred_download = 1 AND width > 0 AND height > 0
        """
    )
    for unit, px in cursor:
        bucket_counts[unit][mp_bucket(px)] += 1
        unit_pixels[unit].append(px)

    # Images-per-record histogram.
    images_hist: dict[str, dict[int, int]] = defaultdict(dict)
    for unit, cnt, records in conn.execute(
        """
        SELECT unit_code, cnt, COUNT(*) AS records
        FROM (SELECT unit_code, record_id, COUNT(*) AS cnt FROM media_items GROUP BY unit_code, record_id)
        GROUP BY unit_code, cnt
        """
    ):
        images_hist[unit][cnt] = records

    media_types = conn.execute(
        "SELECT media_type, COUNT(*) FROM media_items GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall()
    rights = conn.execute(
        "SELECT usage_access, COUNT(*) FROM media_items GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall()

    units = sorted(set(record_counts) | set(media_counts) | set(resource_stats))
    unit_rows = []
    for unit in units:
        preferred, with_dims = resource_stats.get(unit, (0, 0))
        pixels = unit_pixels.get(unit, [])
        unit_rows.append(
            [
                unit,
                record_counts.get(unit, 0),
                media_counts.get(unit, 0),
                preferred,
                with_dims,
                preferred - with_dims,
                round(median(pixels) / 1e6, 2) if pixels else "",
                round(sum(pixels) / 1e6) if pixels else 0,
                histogram_median(images_hist.get(unit, {})),
            ]
        )
    write_csv(
        reports / "units.csv",
        [
            "unit_code",
            "records",
            "media_items",
            "preferred_resources",
            "with_dimensions",
            "missing_dimensions",
            "median_megapixels",
            "total_native_megapixels",
            "median_images_per_record",
        ],
        unit_rows,
    )

    write_csv(
        reports / "mp_distribution.csv",
        ["unit_code", *MP_BUCKETS],
        [[unit, *[bucket_counts[unit].get(bucket, 0) for bucket in MP_BUCKETS]] for unit in sorted(bucket_counts)],
    )
    write_csv(
        reports / "images_per_record.csv",
        ["unit_code", "images_per_record", "records"],
        [
            [unit, cnt, images_hist[unit][cnt]]
            for unit in sorted(images_hist)
            for cnt in sorted(images_hist[unit])
        ],
    )
    write_csv(reports / "media_types.csv", ["media_type", "count"], [list(row) for row in media_types])
    write_csv(reports / "rights.csv", ["usage_access", "count"], [list(row) for row in rights])

    total_records = sum(record_counts.values())
    total_media = sum(media_counts.values())
    total_preferred = sum(stats[0] for stats in resource_stats.values())
    total_with_dims = sum(stats[1] for stats in resource_stats.values())
    all_pixels_mp = sum(sum(pixels) for pixels in unit_pixels.values()) / 1e6
    global_hist: Counter = Counter()
    for hist in images_hist.values():
        global_hist.update(hist)

    print(f"records:                 {total_records:>12,}")
    print(f"media items:             {total_media:>12,}")
    print(f"preferred resources:     {total_preferred:>12,}")
    if total_preferred:
        print(f"with dimensions:         {total_with_dims:>12,} ({total_with_dims / total_preferred:.1%})")
    print(f"total native megapixels: {all_pixels_mp:>12,.0f}")
    print(f"median images/record:    {histogram_median(dict(global_hist)):>12.0f}")
    print(f"reports written to:      {reports}")


if __name__ == "__main__":
    main()
