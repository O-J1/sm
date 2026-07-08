"""Export the final download/conversion manifest for a chosen artist cutoff.

Writes a JSONL manifest with one line per output job (a source image can
produce several outputs):
  {"resource_key", "record_id", "unit_code", "url", "output_kind",
   "tier", "target_max_pixels", "target_edge", "resize_algorithm",
   "output_format", "quality", "output_path", "est_bytes", "width",
   "height", "artist_rank", "position", "drop_reason"}

Output kinds:
  highres - JXL q=95, Lanczos; native for tier 'full', else capped
            (8,388,608 px main / NMNHBOTANY 4,194,304 px main /
             2,097,152 px secondary).
  256/512 - Lanczos2Sharp longest-edge pretraining resizes generated from
            the ORIGINAL source file, only when native pixels >= 4x the
            highres cap.
Dropped images (below 1MP, or NMNHBOTANY views beyond the first 10) are
excluded unless --include-drops. Pick the cutoff with estimate_budget.py.

Output layout encoded in output_path (relative to the images root):
  images/highres/<UNIT>/<shard>/<resource_key>.jxl
  images/256/<UNIT>/<shard>/<resource_key>.jpg
  images/512/<UNIT>/<shard>/<resource_key>.jpg

Usage:
    python scripts/export_manifest.py --top-n 2500 [--cc0-only]
                                      [policy knobs, see --help]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from _common import (
    GroupedCursor,
    add_common_arguments,
    add_policy_arguments,
    derivative_pixels,
    ensure_dir,
    load_bpp,
    load_derivative_bpp,
    load_rank_map,
    open_db,
    policy_from_args,
    warn_calibration_quality,
)
from estimate_budget import (
    UNRANKED,
    iter_record_images,
    load_measured_px,
    record_rank,
    unit_median_pixels,
)


def output_path(kind: str, unit_code: str, resource_key: str, extension: str) -> str:
    return f"images/{kind}/{unit_code}/{resource_key[:2]}/{resource_key}.{extension}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    add_policy_arguments(parser)
    parser.add_argument("--top-n", type=int, required=True, help="Artist rank cutoff for tier 'full' (from budget_curve.csv).")
    parser.add_argument("--include-drops", action="store_true", help="Keep dropped rows in the manifest for auditing.")
    parser.add_argument("--cc0-only", action="store_true")
    parser.add_argument("--media-type", default="Images")
    args = parser.parse_args()

    policy = policy_from_args(args)
    conn = open_db(args.db)
    reports = ensure_dir(args.reports_dir)
    rank_map = load_rank_map(reports / "artist_rankings.csv")
    calibration_path = reports / "calibration.json"
    native_bpp, downscaled_bpp = load_bpp(calibration_path)
    derivative_bpp = load_derivative_bpp(calibration_path, policy.derivative_edges)
    warn_calibration_quality(calibration_path, policy.quality)
    median_px = unit_median_pixels(conn, args.media_type)
    for unit, px in load_measured_px(calibration_path).items():
        median_px.setdefault(unit, px)
    facets = GroupedCursor(
        conn.execute(
            "SELECT record_id, content, label FROM record_freetext WHERE category = 'name' ORDER BY record_id"
        )
    )

    manifest_path = reports / "manifest.jsonl"
    totals: Counter = Counter()
    est_bytes_total = 0.0

    def write_row(handle, **row) -> None:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record_id, unit_code, images in iter_record_images(conn, args.media_type, args.cc0_only):
            rank, is_artist = record_rank(facets, record_id, rank_map)
            is_top_artist = rank < args.top_n
            for index, (_, _, _, width, height, resource_key, url) in enumerate(images):
                px = width * height if width and height else median_px.get(unit_code, 4_000_000)
                decision = policy.classify(
                    unit_code=unit_code, index=index, pixels=px, is_top_artist=is_top_artist, is_artist=is_artist
                )
                common = {
                    "resource_key": resource_key,
                    "record_id": record_id,
                    "unit_code": unit_code,
                    "url": url,
                    "width": width,
                    "height": height,
                    "artist_rank": rank if rank != UNRANKED else None,
                    "position": index,
                }
                if decision.tier == "drop":
                    totals[f"drop:{decision.drop_reason}"] += 1
                    if args.include_drops:
                        write_row(
                            handle,
                            **common,
                            output_kind="highres",
                            tier="drop",
                            drop_reason=decision.drop_reason,
                        )
                    continue

                cap = decision.target_max_pixels
                if cap is None or px <= cap:
                    est = px * native_bpp
                else:
                    est = cap * downscaled_bpp
                totals[decision.tier] += 1
                est_bytes_total += est
                write_row(
                    handle,
                    **common,
                    output_kind="highres",
                    tier=decision.tier,
                    target_max_pixels=cap,
                    resize_algorithm="lanczos4",
                    output_format="jxl",
                    quality=policy.quality,
                    output_path=output_path("highres", unit_code, resource_key, "jxl"),
                    est_bytes=round(est),
                )

                for edge in decision.derivative_edges:
                    est = derivative_pixels(width, height, edge) * derivative_bpp[edge]
                    totals[f"derivative:{edge}"] += 1
                    est_bytes_total += est
                    write_row(
                        handle,
                        **common,
                        output_kind=str(edge),
                        tier=decision.tier,
                        target_edge=edge,
                        resize_algorithm="lanczos2sharp",
                        output_format=policy.derivative_format,
                        quality=policy.derivative_quality,
                        output_path=output_path(str(edge), unit_code, resource_key, policy.derivative_format),
                        est_bytes=round(est),
                    )

    kept = totals["full"] + totals["main"] + totals["secondary"]
    dropped = sum(count for key, count in totals.items() if key.startswith("drop:"))
    print(f"tier full:        {totals['full']:>12,}")
    print(f"tier main:        {totals['main']:>12,}")
    print(f"tier secondary:   {totals['secondary']:>12,}")
    for key in sorted(key for key in totals if key.startswith("drop:")):
        print(f"{key + ':':<18}{totals[key]:>12,}")
    for edge in policy.derivative_edges:
        print(f"derivative {edge}:   {totals[f'derivative:{edge}']:>12,}")
    print(f"kept images:      {kept:>12,}")
    print(f"dropped images:   {dropped:>12,}")
    print(f"estimated size:   {est_bytes_total / 1e12:>12.2f} TB")
    print(f"output: {manifest_path}")


if __name__ == "__main__":
    main()
