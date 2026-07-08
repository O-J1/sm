"""Export failed-media reports suitable for sharing with the museum.

Collects every failure class persisted by the pipeline:

- downloads that exhausted retries (media_resources / legacy media_assets,
  status='failed', with the last error and attempt count)
- conversions that failed to open, resize or encode (media_conversions,
  status='failed')
- miscellaneous recorded failures (failures table)

Supports both the current schema (media_items/media_resources) and the
legacy flattened schema (media_assets).
"""

from __future__ import annotations

import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

REPORT_HEADER = [
    "stage",
    "unit_code",
    "record_id",
    "record_title",
    "record_url",
    "media_guid",
    "ids_id",
    "media_url",
    "attempts",
    "error",
    "failed_at",
]

_NEW_DOWNLOAD_SQL = """
SELECT 'download' AS stage, mr.unit_code, mr.record_id, r.title, r.url AS record_url,
       mi.guid, mi.ids_id, mr.url AS media_url, mr.attempts, mr.error, mr.updated_at
FROM media_resources mr
LEFT JOIN media_items mi ON mi.media_key = mr.media_key
LEFT JOIN records r ON r.id = mr.record_id
WHERE mr.status = 'failed'
"""

_NEW_CONVERSION_SQL = """
SELECT 'conversion (' || mc.target_format || ')' AS stage, mr.unit_code, mr.record_id,
       r.title, r.url AS record_url, mi.guid, mi.ids_id, mr.url AS media_url,
       mc.attempts, mc.error, mc.updated_at
FROM media_conversions mc
JOIN media_resources mr ON mr.resource_key = mc.resource_key
LEFT JOIN media_items mi ON mi.media_key = mr.media_key
LEFT JOIN records r ON r.id = mr.record_id
WHERE mc.status = 'failed'
"""

_LEGACY_DOWNLOAD_SQL = """
SELECT 'download' AS stage, ma.unit_code, ma.record_id, r.title, r.url AS record_url,
       ma.guid, ma.ids_id, ma.url AS media_url, ma.attempts, ma.error, ma.updated_at
FROM media_assets ma
LEFT JOIN records r ON r.id = ma.record_id
WHERE ma.status = 'failed'
"""

_LEGACY_CONVERSION_SQL = """
SELECT 'conversion (' || mc.target_format || ')' AS stage, ma.unit_code, ma.record_id,
       r.title, r.url AS record_url, ma.guid, ma.ids_id, ma.url AS media_url,
       mc.attempts, mc.error, mc.updated_at
FROM media_conversions mc
JOIN media_assets ma ON ma.media_key = mc.resource_key
LEFT JOIN records r ON r.id = ma.record_id
WHERE mc.status = 'failed'
"""

_FAILURES_SQL = """
SELECT 'other (' || source || ')' AS stage, '' AS unit_code, identifier AS record_id,
       '' AS title, '' AS record_url, '' AS guid, '' AS ids_id, '' AS media_url,
       NULL AS attempts, error, created_at AS updated_at
FROM failures
"""


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def _iso(timestamp) -> str:
    if not timestamp:
        return ""
    try:
        return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).isoformat()
    except (ValueError, OSError, OverflowError):
        return str(timestamp)


def write_failure_report(database_path: Path, output_path: Path) -> dict[str, int]:
    """Write a CSV of all failed media to output_path; returns counts per stage."""
    conn = sqlite3.connect(f"file:{Path(database_path).as_posix()}?mode=ro", uri=True)
    try:
        tables = _table_names(conn)
        queries: list[str] = []
        if "media_resources" in tables:
            queries.append(_NEW_DOWNLOAD_SQL)
            if "media_conversions" in tables:
                queries.append(_NEW_CONVERSION_SQL)
        elif "media_assets" in tables:
            queries.append(_LEGACY_DOWNLOAD_SQL)
            if "media_conversions" in tables:
                queries.append(_LEGACY_CONVERSION_SQL)
        if "failures" in tables:
            queries.append(_FAILURES_SQL)
        if not queries:
            raise ValueError(f"no recognizable media/failure tables in {database_path}")

        counts: dict[str, int] = {}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(REPORT_HEADER)
            for query in queries:
                try:
                    rows = conn.execute(query)
                except sqlite3.OperationalError as error:
                    counts[f"skipped query ({error})"] = 0
                    continue
                for row in rows:
                    stage = row[0]
                    counts[stage] = counts.get(stage, 0) + 1
                    writer.writerow([*row[:-1], _iso(row[-1])])
        return counts
    finally:
        conn.close()
