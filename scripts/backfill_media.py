"""Backfill the empty media_assets table from the raw JSONL metadata.

The metadata-only scrape never projected media into SQLite, but every
record's online_media block (URLs, dimensions when available, rights) is in
data/metadata/<UNIT>/records.jsonl. This script streams those files and
inserts one media_assets row per media item, matching what the scraper
would have written. Downstream analysis scripts then work unchanged.

Safe by design: per-unit atomic transactions (a unit is either fully
backfilled or absent), resumable (already-populated units are skipped),
and --dry-run parses without writing.

Usage:
    python scripts/backfill_media.py --db "$db" [--metadata-dir DIR]
                                     [--units SAAM,NPG] [--dry-run] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import time
from pathlib import Path

try:
    import orjson as _json

    def _loads(line: str | bytes):
        return _json.loads(line)
except ImportError:
    import json as _json

    def _loads(line: str | bytes):
        return _json.loads(line)

from _common import DEFAULT_DB

INSERT_SQL = """
INSERT OR REPLACE INTO media_assets (
    media_key, record_id, unit_code, record_hash, kind, media_type, url,
    thumbnail, caption, preferred_citation, usage_access, usage_text,
    usage_codes_json, usage_flag, guid, media_id, ids_id, alt_text,
    extended_description, resource_label, resource_width, resource_height,
    resource_dimensions, parent_media_url, screen_url, thumbnail_url,
    downloadable, status, local_path, size_bytes, attempts, error, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', 0, 0, '', ?)
"""

# Preference order for the download URL within a media item's resources.
LABEL_PRIORITY = {
    "jpeg": ("High-resolution JPEG", "High-resolution TIFF"),
    "tiff": ("High-resolution TIFF", "High-resolution JPEG"),
}


def _string(value) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def iter_media(content: dict):
    """Yield media dicts from the record content (descriptiveNonRepeating or top level)."""
    for container in (content.get("descriptiveNonRepeating"), content):
        if not isinstance(container, dict):
            continue
        online_media = container.get("online_media") or container.get("onlineMedia")
        if not isinstance(online_media, dict):
            continue
        media_list = online_media.get("media")
        if isinstance(media_list, list):
            for media in media_list:
                if isinstance(media, dict):
                    yield media
            return  # do not double-yield from the top-level fallback


def media_row(record: dict, media: dict, position: int, prefer: str, now: int) -> tuple:
    record_id = _string(record.get("id"))
    unit_code = _string(record.get("unitCode"))
    content_url = _string(media.get("content"))
    thumbnail = _string(media.get("thumbnail"))
    usage = media.get("usage") if isinstance(media.get("usage"), dict) else {}
    codes = usage.get("codes") if isinstance(usage.get("codes"), list) else []

    # Pick the best downloadable resource; fall back to the content URL.
    url = content_url
    label = ""
    width = height = None
    dimensions = ""
    screen_url = ""
    thumbnail_url = ""
    resources = media.get("resources")
    if isinstance(resources, list):
        by_label = { _string(r.get("label")): r for r in resources if isinstance(r, dict) and r.get("url") }
        for wanted in LABEL_PRIORITY[prefer]:
            if wanted in by_label:
                chosen = by_label[wanted]
                url = _string(chosen.get("url"))
                label = wanted
                width = chosen.get("width") if isinstance(chosen.get("width"), int) else None
                height = chosen.get("height") if isinstance(chosen.get("height"), int) else None
                dimensions = _string(chosen.get("dimensions"))
                break
        screen_url = _string((by_label.get("Screen Image") or {}).get("url"))
        thumbnail_url = _string((by_label.get("Thumbnail Image") or {}).get("url"))

    media_key = hashlib.sha1(
        "|".join([record_id, _string(media.get("guid")), _string(media.get("id")),
                  _string(media.get("idsId")), url, str(position)]).encode("utf-8")
    ).hexdigest()

    return (
        media_key,
        record_id,
        unit_code,
        _string(record.get("hash")),
        "media",
        _string(media.get("type")),
        url,
        thumbnail,
        _string(media.get("caption")),
        _string(media.get("preferred_citation") or media.get("preferredCitation")),
        _string(usage.get("access")),
        _string(usage.get("text")),
        _json_dumps([_string(code) for code in codes if _string(code)]),
        _string(media.get("usage_flag") or media.get("usageFlag")),
        _string(media.get("guid")),
        _string(media.get("id")),
        _string(media.get("idsId") or media.get("idsID")),
        _string(media.get("altTextAccessibility") or media.get("altText")),
        _string(media.get("extDescrAccessibility") or media.get("extendedDescription")),
        label,
        width,
        height,
        dimensions,
        content_url,
        screen_url,
        thumbnail_url,
        1 if url else 0,
        "metadata",
        now,
    )


def _json_dumps(value) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def backfill_unit(conn: sqlite3.Connection, jsonl: Path, unit: str, prefer: str, dry_run: bool) -> tuple[int, int]:
    now = int(time.time())
    records = 0
    inserted = 0
    batch: list[tuple] = []
    with jsonl.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = _loads(line)
            except ValueError:
                continue
            records += 1
            content = record.get("content")
            if not isinstance(content, dict):
                continue
            for position, media in enumerate(iter_media(content)):
                batch.append(media_row(record, media, position, prefer, now))
            if len(batch) >= 5000:
                if not dry_run:
                    conn.executemany(INSERT_SQL, batch)
                inserted += len(batch)
                batch.clear()
    if batch:
        if not dry_run:
            conn.executemany(INSERT_SQL, batch)
        inserted += len(batch)
    return records, inserted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--metadata-dir", type=Path, default=None, help="Default: <db dir>/metadata")
    parser.add_argument("--units", default="", help="Comma-separated unit codes; default: all found.")
    parser.add_argument("--prefer", choices=("jpeg", "tiff"), default="jpeg",
                        help="Preferred high-res resource for the download URL (default jpeg: far smaller transfers).")
    parser.add_argument("--force", action="store_true", help="Re-backfill units that already have rows.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"database not found: {args.db}")
    metadata_dir = args.metadata_dir or (args.db.parent / "metadata")
    if not metadata_dir.is_dir():
        raise SystemExit(f"metadata dir not found: {metadata_dir}")

    wanted = {unit.strip() for unit in args.units.split(",") if unit.strip()}
    unit_files = sorted(
        (path.parent.name, path)
        for path in metadata_dir.glob("*/records.jsonl")
        if not wanted or path.parent.name in wanted
    )
    if not unit_files:
        raise SystemExit(f"no records.jsonl files found under {metadata_dir}")

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA journal_mode = WAL")

    total_media = 0
    started = time.time()
    for unit, jsonl in unit_files:
        existing = conn.execute(
            "SELECT EXISTS(SELECT 1 FROM media_assets WHERE unit_code = ?)", (unit,)
        ).fetchone()[0]
        if existing and not args.force:
            print(f"{unit:<14} skipped (already backfilled)")
            continue
        unit_start = time.time()
        try:
            if existing:
                conn.execute("DELETE FROM media_assets WHERE unit_code = ?", (unit,))
            records, inserted = backfill_unit(conn, jsonl, unit, args.prefer, args.dry_run)
            if args.dry_run:
                conn.rollback()
            else:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        total_media += inserted
        rate = records / max(time.time() - unit_start, 0.001)
        print(f"{unit:<14} records: {records:>10,}  media rows: {inserted:>10,}  ({rate:,.0f} rec/s)")

    action = "would insert" if args.dry_run else "inserted"
    print(f"\n{action} {total_media:,} media rows in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
