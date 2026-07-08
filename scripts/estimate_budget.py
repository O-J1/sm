"""Project storage cost of the tiered subsampling policy and sweep the
artist cutoff so the collection fits the disk budget.

Policy per record (user decision, see _common.SubsamplePolicy):
  - full:      records by a top-N ranked artist keep every image at native
               resolution (JXL q=95).
  - main:      the record's main image (first media item) capped at
               8,388,608 px (NMNHBOTANY: 4,194,304 px).
  - secondary: additional views capped at 2,097,152 px.
  - drop:      images below 1MP, and NMNHBOTANY views beyond the first 10
               per specimen record.
  - derivatives: kept non-full images reduced by >= 4x also get 256px and
               512px longest-edge Lanczos2Sharp resizes from the original
               file (pretraining set; counted in the budget).

Sizes are estimated as pixels x bytes-per-pixel, using measured values from
reports/calibration.json when present (run calibrate_bpp.py --quality 95),
otherwise conservative fallbacks. A single streaming pass accumulates bytes
per artist rank, so the whole cutoff curve comes from one database scan.

Usage:
    python scripts/estimate_budget.py [--db PATH] [--budget-tb 8]
                                      [--top-ns 0,250,500,1000,2500,5000,10000,all]
                                      [--cc0-only] [policy knobs, see --help]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from _common import (
    GroupedCursor,
    add_common_arguments,
    add_policy_arguments,
    derivative_pixels,
    is_artist_role,
    load_bpp,
    load_derivative_bpp,
    load_rank_map,
    median,
    name_key,
    open_db,
    policy_from_args,
    warn_calibration_quality,
    write_csv,
)

UNRANKED = 1 << 30


def load_measured_px(calibration_path: Path) -> dict[str, int]:
    """Per-unit median pixels measured from downloaded samples (calibrate_bpp.py),
    the dimensions fallback for units whose metadata lacks width/height."""
    if not calibration_path.exists():
        return {}
    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    return {
        unit: int(mp * 1e6)
        for unit, mp in (data.get("per_unit_measured_mp_median") or {}).items()
        if mp
    }


def unit_median_pixels(conn, media_type: str) -> dict[str, int]:
    """Per-unit median pixels from metadata dimensions, the fallback for
    images whose metadata lacks width/height."""
    pixels: dict[str, list[int]] = defaultdict(list)
    for unit, px in conn.execute(
        """
        SELECT unit_code, resource_width * resource_height
        FROM media_assets
        WHERE downloadable = 1 AND media_type = ? AND resource_width > 0 AND resource_height > 0
        """,
        (media_type,),
    ):
        pixels[unit].append(px)
    return {unit: int(median(values)) for unit, values in pixels.items()}


def iter_record_images(conn, media_type: str, cc0_only: bool):
    """Yield (record_id, unit_code, [rows]) grouped by record, plus a facet lookup."""
    rights_clause = "AND usage_access = 'CC0'" if cc0_only else ""
    cursor = conn.execute(
        f"""
        SELECT record_id, unit_code, rowid, resource_width, resource_height, media_key, url
        FROM media_assets
        WHERE downloadable = 1 AND media_type = ? {rights_clause}
        ORDER BY record_id, rowid
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


def record_rank(facets: GroupedCursor, record_id: str, rank_map: dict[str, int]) -> tuple[int, bool]:
    """Best artist rank for the record plus whether any of its names carries
    an artist-type role label (drives the --artist-cap-mp raise)."""
    best = UNRANKED
    has_artist = False
    for _, value, label in facets.rows_for(record_id):
        rank = rank_map.get(name_key(value), UNRANKED)
        if rank < best:
            best = rank
        if not has_artist and is_artist_role(label):
            has_artist = True
    return best, has_artist or best != UNRANKED


@dataclass
class RankBucket:
    """Per-artist-rank accumulator holding both possible outcomes for its
    images: the tier-A ('full') outcome and the non-tier-A policy outcome."""

    images: int = 0
    full_bytes: float = 0.0
    full_kept: int = 0
    policy_bytes: float = 0.0
    policy_native: int = 0
    policy_main_down: int = 0
    policy_secondary_down: int = 0
    drop_low: int = 0
    drop_cap: int = 0
    derivative_bytes: float = 0.0
    derivative_counts: dict[int, int] = field(default_factory=dict)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    add_policy_arguments(parser)
    parser.add_argument("--rankings-csv", type=Path, default=None)
    parser.add_argument("--budget-tb", type=float, default=8.0)
    parser.add_argument("--top-ns", default="0,250,500,1000,2500,5000,10000,25000,all")
    parser.add_argument("--cc0-only", action="store_true")
    parser.add_argument("--media-type", default="Images")
    args = parser.parse_args()

    policy = policy_from_args(args)
    conn = open_db(args.db)
    reports = args.reports_dir
    rankings_csv = args.rankings_csv or (reports / "artist_rankings.csv")
    rank_map = load_rank_map(rankings_csv)
    calibration_path = reports / "calibration.json"
    native_bpp, downscaled_bpp = load_bpp(calibration_path)
    derivative_bpp = load_derivative_bpp(calibration_path, policy.derivative_edges)
    warn_calibration_quality(calibration_path, policy.quality)
    print(f"bpp model: native={native_bpp} downscaled={downscaled_bpp} derivative={derivative_bpp} "
          f"({'calibrated' if calibration_path.exists() else 'FALLBACK - run calibrate_bpp.py'})")

    median_px = unit_median_pixels(conn, args.media_type)
    for unit, px in load_measured_px(calibration_path).items():
        median_px.setdefault(unit, px)
    facets = GroupedCursor(
        conn.execute(
            "SELECT record_id, content, label FROM record_freetext WHERE category = 'name' ORDER BY record_id"
        )
    )

    buckets: dict[int, RankBucket] = defaultdict(RankBucket)
    total_images = 0

    for record_id, unit_code, images in iter_record_images(conn, args.media_type, args.cc0_only):
        rank, is_artist = record_rank(facets, record_id, rank_map)
        bucket = buckets[rank]
        for index, (_, _, _, width, height, _, _) in enumerate(images):
            px = width * height if width and height else median_px.get(unit_code, 4_000_000)
            bucket.images += 1
            total_images += 1

            # Outcome if this record's artist makes the top-N cut.
            full = policy.classify(unit_code=unit_code, index=index, pixels=px, is_top_artist=True)
            if full.tier != "drop":
                bucket.full_bytes += px * native_bpp
                bucket.full_kept += 1

            # Outcome if it does not.
            decision = policy.classify(
                unit_code=unit_code, index=index, pixels=px, is_top_artist=False, is_artist=is_artist
            )
            if decision.tier == "drop":
                if decision.drop_reason == "low_quality":
                    bucket.drop_low += 1
                else:
                    bucket.drop_cap += 1
                continue
            cap = decision.target_max_pixels or px
            if px <= cap:
                bucket.policy_bytes += px * native_bpp
                bucket.policy_native += 1
            else:
                bucket.policy_bytes += cap * downscaled_bpp
                if decision.tier == "main":
                    bucket.policy_main_down += 1
                else:
                    bucket.policy_secondary_down += 1
            for edge in decision.derivative_edges:
                bucket.derivative_bytes += derivative_pixels(width, height, edge) * derivative_bpp[edge]
                bucket.derivative_counts[edge] = bucket.derivative_counts.get(edge, 0) + 1

    ranks = sorted(buckets)
    ranked_artists = len(rank_map)
    sweep = []
    for token in args.top_ns.split(","):
        token = token.strip().lower()
        sweep.append(ranked_artists if token == "all" else int(token))

    edge_labels = [f"deriv_{edge}" for edge in policy.derivative_edges]
    rows = []
    print(f"\ntotal images: {total_images:,}  ranked artists: {ranked_artists:,}  quality: q{policy.quality}")
    header = (f"{'top_n':>8} {'total_TB':>9} {'hires_TB':>9} {'deriv_TB':>9} {'full_res':>10} "
              f"{'native':>10} {'downscaled':>11} {'drop_low':>9} {'drop_cap':>9} "
              f"{'fits_' + str(args.budget_tb) + 'TB':>9}")
    print(header)
    for top_n in sweep:
        highres_bytes = 0.0
        derivative_bytes = 0.0
        full_res = native_kept = main_down = secondary_down = 0
        drop_low = drop_cap = 0
        derivative_counts = {edge: 0 for edge in policy.derivative_edges}
        for rank in ranks:
            bucket = buckets[rank]
            drop_low += bucket.drop_low
            if rank < top_n:
                highres_bytes += bucket.full_bytes
                full_res += bucket.full_kept
            else:
                highres_bytes += bucket.policy_bytes
                derivative_bytes += bucket.derivative_bytes
                native_kept += bucket.policy_native
                main_down += bucket.policy_main_down
                secondary_down += bucket.policy_secondary_down
                drop_cap += bucket.drop_cap
                for edge, count in bucket.derivative_counts.items():
                    derivative_counts[edge] = derivative_counts.get(edge, 0) + count
        total_tb = (highres_bytes + derivative_bytes) / 1e12
        downscaled = main_down + secondary_down
        fits = "yes" if total_tb <= args.budget_tb else "no"
        print(f"{top_n:>8,} {total_tb:>9.2f} {highres_bytes / 1e12:>9.2f} {derivative_bytes / 1e12:>9.3f} "
              f"{full_res:>10,} {native_kept:>10,} {downscaled:>11,} {drop_low:>9,} {drop_cap:>9,} {fits:>9}")
        rows.append(
            [
                top_n,
                policy.quality,
                round(total_tb, 3),
                round(highres_bytes / 1e12, 3),
                round(derivative_bytes / 1e12, 4),
                full_res,
                native_kept,
                main_down,
                secondary_down,
                drop_low,
                drop_cap,
                *[derivative_counts[edge] for edge in policy.derivative_edges],
                fits,
            ]
        )

    path = write_csv(
        reports / "budget_curve.csv",
        [
            "top_n",
            "quality",
            "total_tb",
            "highres_tb",
            "derivative_tb",
            "full_res_images",
            "native_kept_images",
            "main_downscaled_images",
            "secondary_downscaled_images",
            "low_quality_dropped",
            "unit_cap_dropped",
            *edge_labels,
            "fits_budget",
        ],
        rows,
    )
    print(f"\noutput: {path}")


if __name__ == "__main__":
    main()
