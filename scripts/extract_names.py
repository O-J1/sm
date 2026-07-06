"""Extract distinct creator names with roles and usage counts.

Reads the freetext `name` entries (record_freetext, category='name'), which
carry the role label directly (Artist / Maker / Sitter / Manufacturer, ...),
and counts records and images per name. Roles matter: NPG portrait *sitters*
and NMNH *collectors* appear among the names and must not later be ranked
as artists.

Streams two record_id-ordered cursors with a merge-join, so it scales to
the full database. Output: reports/names.csv sorted by image count.

Usage:
    python scripts/extract_names.py [--db PATH] [--reports-dir PATH] [--min-records N]
"""

from __future__ import annotations

import argparse
from collections import Counter

from _common import (
    GroupedCursor,
    add_common_arguments,
    display_form,
    ensure_dir,
    name_key,
    open_db,
    write_csv,
)


class NameStats:
    __slots__ = ("records", "images", "roles", "values", "units")

    def __init__(self) -> None:
        self.records = 0
        self.images = 0
        self.roles: Counter = Counter()
        self.values: Counter = Counter()
        self.units: Counter = Counter()


def _format_counter(counter: Counter, limit: int = 8) -> str:
    return ";".join(f"{value}:{count}" for value, count in counter.most_common(limit))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--min-records", type=int, default=1, help="Only output names on at least N records.")
    args = parser.parse_args()
    conn = open_db(args.db)
    reports = ensure_dir(args.reports_dir)

    names_cursor = conn.execute(
        """
        SELECT record_id, unit_code, content, label
        FROM record_freetext
        WHERE category = 'name'
        ORDER BY record_id, position
        """
    )
    media_counts = GroupedCursor(
        conn.execute(
            "SELECT record_id, COUNT(*) FROM media_assets GROUP BY record_id ORDER BY record_id"
        )
    )

    stats: dict[str, NameStats] = {}
    current_record = None
    record_images = 0
    seen_keys_for_record: set[str] = set()

    for record_id, unit_code, value, label in names_cursor:
        if record_id != current_record:
            current_record = record_id
            media_rows = media_counts.rows_for(record_id)
            record_images = media_rows[0][1] if media_rows else 0
            seen_keys_for_record = set()

        key = name_key(value)
        if not key or key in seen_keys_for_record:
            continue
        seen_keys_for_record.add(key)

        entry = stats.get(key)
        if entry is None:
            entry = stats[key] = NameStats()
        entry.records += 1
        entry.images += record_images
        entry.values[value] += 1
        entry.units[unit_code] += 1
        entry.roles[label or "(unlabeled)"] += 1

    rows = []
    for key, entry in stats.items():
        if entry.records < args.min_records:
            continue
        raw_value = entry.values.most_common(1)[0][0]
        top_role = entry.roles.most_common(1)[0][0]
        rows.append(
            [
                display_form(raw_value),
                raw_value,
                key,
                entry.records,
                entry.images,
                top_role,
                _format_counter(entry.roles),
                _format_counter(entry.units),
            ]
        )
    rows.sort(key=lambda row: (-row[4], -row[3], row[0]))

    path = write_csv(
        reports / "names.csv",
        ["name", "raw_value", "name_key", "record_count", "image_count", "top_role", "roles", "units"],
        rows,
    )
    print(f"distinct names: {len(stats):,} (written: {len(rows):,})")
    print(f"output: {path}")


if __name__ == "__main__":
    main()
