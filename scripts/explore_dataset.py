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

    # Single streamed scan of media_assets for everything except images/record.
    media_counts: Counter = Counter()
    dims_counts: Counter = Counter()
    bucket_counts: dict[str, Counter] = defaultdict(Counter)
    unit_pixels: dict[str, list[int]] = defaultdict(list)
    type_counts: Counter = Counter()
    rights_counts: Counter = Counter()
    cursor = conn.execute(
        """
        SELECT unit_code, media_type, usage_access, downloadable,
               resource_width, resource_height
        FROM media_assets
        """
    )
    for unit, media_type, usage_access, downloadable, width, height in cursor:
        media_counts[unit] += 1
        type_counts[media_type] += 1
        rights_counts[usage_access] += 1
        if downloadable and width and height and width > 0 and height > 0:
            px = width * height
            dims_counts[unit] += 1
            bucket_counts[unit][mp_bucket(px)] += 1
            unit_pixels[unit].append(px)

    # Second pass: images-per-record histogram (grouped, spills to temp on disk).
    images_hist: dict[str, dict[int, int]] = defaultdict(dict)
    for unit, cnt, records in conn.execute(
        """
        SELECT unit_code, cnt, COUNT(*) AS records
        FROM (SELECT unit_code, record_id, COUNT(*) AS cnt FROM media_assets GROUP BY unit_code, record_id)
        GROUP BY unit_code, cnt
        """
    ):
        images_hist[unit][cnt] = records

    units = sorted(set(record_counts) | set(media_counts))
    unit_rows = []
    for unit in units:
        total_media = media_counts.get(unit, 0)
        with_dims = dims_counts.get(unit, 0)
        pixels = unit_pixels.get(unit, [])
        unit_rows.append(
            [
                unit,
                record_counts.get(unit, 0),
                total_media,
                with_dims,
                total_media - with_dims,
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
            "media_assets",
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
    write_csv(reports / "media_types.csv", ["media_type", "count"], [list(item) for item in type_counts.most_common()])
    write_csv(reports / "rights.csv", ["usage_access", "count"], [list(item) for item in rights_counts.most_common()])

    total_records = sum(record_counts.values())
    total_media = sum(media_counts.values())
    total_with_dims = sum(dims_counts.values())
    all_pixels_mp = sum(sum(pixels) for pixels in unit_pixels.values()) / 1e6
    global_hist: Counter = Counter()
    for hist in images_hist.values():
        global_hist.update(hist)

    print(f"records:                 {total_records:>12,}")
    print(f"media assets:            {total_media:>12,}")
    if total_media:
        print(f"with dimensions:         {total_with_dims:>12,} ({total_with_dims / total_media:.1%})")
    print(f"total native megapixels: {all_pixels_mp:>12,.0f}")
    print(f"median images/record:    {histogram_median(dict(global_hist)):>12.0f}")
    print(f"reports written to:      {reports}")


if __name__ == "__main__":
    main()
