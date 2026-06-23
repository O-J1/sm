from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import FreetextEntry, MediaAsset, SmithsonianRecord


@dataclass(frozen=True)
class PageJob:
    partition_key: str
    query: str
    start: int
    rows: int
    sort: str
    record_type: str
    row_group: str


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = asyncio.Lock()

    async def close(self) -> None:
        async with self._lock:
            self._connection.close()

    async def initialize(self) -> None:
        async with self._lock:
            self._connection.executescript(SCHEMA)
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute("UPDATE pages SET status = 'pending' WHERE status = 'running'")
            self._connection.execute("UPDATE media_assets SET status = 'pending' WHERE status = 'running'")
            self._suppress_jpeg_downloads_with_tiff()
            self._connection.execute("UPDATE media_conversions SET status = 'pending' WHERE status = 'running'")
            self._connection.commit()

    async def upsert_partition(self, partition_key: str, query: str, row_count: int | None) -> None:
        async with self._lock:
            self._connection.execute(
                """
                INSERT INTO partitions (partition_key, query, row_count, status, created_at, updated_at)
                VALUES (?, ?, ?, 'pending', unixepoch(), unixepoch())
                ON CONFLICT(partition_key) DO UPDATE SET
                    query = excluded.query,
                    row_count = COALESCE(excluded.row_count, partitions.row_count),
                    updated_at = unixepoch()
                """,
                (partition_key, query, row_count),
            )
            self._connection.commit()

    def _suppress_jpeg_downloads_with_tiff(self) -> None:
        self._connection.execute(
            """
            UPDATE media_assets
            SET downloadable = 0,
                status = 'metadata',
                error = 'suppressed because TIFF is available',
                updated_at = unixepoch()
            WHERE kind = 'highres_jpeg'
                AND status != 'complete'
                AND EXISTS (
                    SELECT 1
                    FROM media_assets AS tiff
                    WHERE tiff.kind = 'highres_tiff'
                        AND tiff.record_id = media_assets.record_id
                        AND (
                            (tiff.media_id != '' AND tiff.media_id = media_assets.media_id)
                            OR (tiff.ids_id != '' AND tiff.ids_id = media_assets.ids_id)
                            OR (tiff.guid != '' AND tiff.guid = media_assets.guid)
                            OR (tiff.parent_media_url != '' AND tiff.parent_media_url = media_assets.parent_media_url)
                        )
                )
            """
        )

    async def page_status(self, partition_key: str, start: int) -> str | None:
        async with self._lock:
            row = self._connection.execute(
                "SELECT status FROM pages WHERE partition_key = ? AND start = ?",
                (partition_key, start),
            ).fetchone()
            return str(row["status"]) if row else None

    async def mark_page(self, job: PageJob, status: str, *, error: str = "", row_count: int | None = None) -> None:
        async with self._lock:
            self._connection.execute(
                """
                INSERT INTO pages (partition_key, start, rows, sort, record_type, row_group, status, error, row_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, unixepoch())
                ON CONFLICT(partition_key, start) DO UPDATE SET
                    rows = excluded.rows,
                    sort = excluded.sort,
                    record_type = excluded.record_type,
                    row_group = excluded.row_group,
                    status = excluded.status,
                    error = excluded.error,
                    row_count = COALESCE(excluded.row_count, pages.row_count),
                    attempts = pages.attempts + CASE WHEN excluded.status = 'failed' THEN 1 ELSE 0 END,
                    updated_at = unixepoch()
                """,
                (
                    job.partition_key,
                    job.start,
                    job.rows,
                    job.sort,
                    job.record_type,
                    job.row_group,
                    status,
                    error,
                    row_count,
                ),
            )
            self._connection.commit()

    async def upsert_record(self, record: SmithsonianRecord, raw_path: Path) -> bool:
        async with self._lock:
            existing = self._connection.execute(
                "SELECT doc_signature FROM records WHERE id = ?",
                (record.id,),
            ).fetchone()
            changed = existing is None or existing["doc_signature"] != record.doc_signature
            self._connection.execute(
                """
                INSERT INTO records (
                    id, title, unit_code, linked_id, type, url, hash, doc_signature,
                    timestamp, last_time_updated, status, public_search, version, raw_path, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, unixepoch())
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    unit_code = excluded.unit_code,
                    linked_id = excluded.linked_id,
                    type = excluded.type,
                    url = excluded.url,
                    hash = excluded.hash,
                    doc_signature = excluded.doc_signature,
                    timestamp = excluded.timestamp,
                    last_time_updated = excluded.last_time_updated,
                    status = excluded.status,
                    public_search = excluded.public_search,
                    version = excluded.version,
                    raw_path = excluded.raw_path,
                    updated_at = unixepoch()
                """,
                (
                    record.id,
                    record.title,
                    record.unit_code,
                    record.linked_id,
                    record.type,
                    record.url,
                    record.hash,
                    record.doc_signature,
                    record.timestamp,
                    record.last_time_updated,
                    record.status,
                    record.public_search,
                    record.version,
                    str(raw_path),
                ),
            )
            self._connection.commit()
            return changed

    async def enqueue_media(self, media: MediaAsset) -> None:
        async with self._lock:
            self._connection.execute(
                """
                INSERT INTO media_assets (
                    media_key, record_id, unit_code, record_hash, kind, media_type, url,
                    thumbnail, caption, preferred_citation, usage_access, usage_text,
                    usage_codes_json, usage_flag, guid, media_id, ids_id, alt_text,
                    extended_description, resource_label, resource_width, resource_height,
                    resource_dimensions, parent_media_url, screen_url, thumbnail_url,
                    downloadable, status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, unixepoch())
                ON CONFLICT(media_key) DO UPDATE SET
                    record_id = excluded.record_id,
                    unit_code = excluded.unit_code,
                    record_hash = excluded.record_hash,
                    kind = excluded.kind,
                    media_type = excluded.media_type,
                    url = excluded.url,
                    thumbnail = excluded.thumbnail,
                    caption = excluded.caption,
                    preferred_citation = excluded.preferred_citation,
                    usage_access = excluded.usage_access,
                    usage_text = excluded.usage_text,
                    usage_codes_json = excluded.usage_codes_json,
                    usage_flag = excluded.usage_flag,
                    guid = excluded.guid,
                    media_id = excluded.media_id,
                    ids_id = excluded.ids_id,
                    alt_text = excluded.alt_text,
                    extended_description = excluded.extended_description,
                    resource_label = excluded.resource_label,
                    resource_width = excluded.resource_width,
                    resource_height = excluded.resource_height,
                    resource_dimensions = excluded.resource_dimensions,
                    parent_media_url = excluded.parent_media_url,
                    screen_url = excluded.screen_url,
                    thumbnail_url = excluded.thumbnail_url,
                    downloadable = excluded.downloadable,
                    updated_at = unixepoch()
                """,
                (
                    media.key,
                    media.record_id,
                    media.unit_code,
                    media.record_hash,
                    media.kind,
                    media.media_type,
                    media.url,
                    media.thumbnail,
                    media.caption,
                    media.preferred_citation,
                    media.usage_access,
                    media.usage_text,
                    json.dumps(media.usage_codes),
                    media.usage_flag,
                    media.guid,
                    media.media_id,
                    media.ids_id,
                    media.alt_text,
                    media.extended_description,
                    media.resource_label,
                    media.resource_width,
                    media.resource_height,
                    media.resource_dimensions,
                    media.parent_media_url,
                    media.screen_url,
                    media.thumbnail_url,
                    int(media.downloadable),
                    "pending" if media.downloadable else "metadata",
                ),
            )
            if media.kind in {"highres_tiff", "highres_jpeg"}:
                self._suppress_jpeg_downloads_with_tiff()
            self._connection.commit()

    async def replace_freetext_entries(self, record_id: str, entries: list[FreetextEntry]) -> None:
        async with self._lock:
            self._connection.execute("DELETE FROM record_freetext WHERE record_id = ?", (record_id,))
            self._connection.executemany(
                """
                INSERT INTO record_freetext (record_id, unit_code, category, label, content, position, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, unixepoch())
                """,
                [
                    (entry.record_id, entry.unit_code, entry.category, entry.label, entry.content, entry.position)
                    for entry in entries
                ],
            )
            self._connection.commit()

    async def next_media_batch(self, limit: int, *, max_attempts: int) -> list[sqlite3.Row]:
        async with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM media_assets
                                WHERE downloadable = 1
                                    AND (status = 'pending'
                                        OR (status = 'failed' AND attempts < ?))
                ORDER BY updated_at ASC
                LIMIT ?
                """,
                (max_attempts, limit),
            ).fetchall()
            for row in rows:
                self._connection.execute(
                    "UPDATE media_assets SET status = 'running', updated_at = unixepoch() WHERE media_key = ?",
                    (row["media_key"],),
                )
            self._connection.commit()
            return rows

    async def mark_media_complete(self, media_key: str, path: Path, size: int) -> None:
        async with self._lock:
            self._connection.execute(
                """
                UPDATE media_assets
                SET status = 'complete', local_path = ?, size_bytes = ?, error = '', updated_at = unixepoch()
                WHERE media_key = ?
                """,
                (str(path), size, media_key),
            )
            self._connection.commit()

    async def mark_media_failed(self, media_key: str, error: str) -> None:
        async with self._lock:
            self._connection.execute(
                """
                UPDATE media_assets
                SET status = 'failed', attempts = attempts + 1, error = ?, updated_at = unixepoch()
                WHERE media_key = ?
                """,
                (error[:1000], media_key),
            )
            self._connection.commit()

    async def enqueue_media_conversion(
        self,
        media_key: str,
        source_path: Path,
        *,
        target_format: str = "jxl",
    ) -> None:
        async with self._lock:
            self._connection.execute(
                """
                INSERT INTO media_conversions (
                    media_key, target_format, source_path, output_path, status,
                    attempts, error, size_bytes, created_at, updated_at
                ) VALUES (?, ?, ?, '', 'pending', 0, '', 0, unixepoch(), unixepoch())
                ON CONFLICT(media_key, target_format) DO UPDATE SET
                    source_path = CASE
                        WHEN media_conversions.status = 'complete' THEN media_conversions.source_path
                        ELSE excluded.source_path
                    END,
                    status = CASE
                        WHEN media_conversions.status = 'complete' THEN media_conversions.status
                        ELSE 'pending'
                    END,
                    error = CASE
                        WHEN media_conversions.status = 'complete' THEN media_conversions.error
                        ELSE ''
                    END,
                    updated_at = CASE
                        WHEN media_conversions.status = 'complete' THEN media_conversions.updated_at
                        ELSE unixepoch()
                    END
                """,
                (media_key, target_format, str(source_path)),
            )
            self._connection.commit()

    async def enqueue_pending_tiff_conversions(self, *, target_format: str = "jxl") -> int:
        async with self._lock:
            cursor = self._connection.execute(
                """
                INSERT INTO media_conversions (
                    media_key, target_format, source_path, output_path, status,
                    attempts, error, size_bytes, created_at, updated_at
                )
                SELECT media_key, ?, local_path, '', 'pending', 0, '', 0, unixepoch(), unixepoch()
                FROM media_assets
                WHERE status = 'complete'
                    AND kind = 'highres_tiff'
                    AND local_path != ''
                    AND (lower(local_path) LIKE '%.tif' OR lower(local_path) LIKE '%.tiff')
                ON CONFLICT(media_key, target_format) DO UPDATE SET
                    source_path = CASE
                        WHEN media_conversions.status = 'complete' THEN media_conversions.source_path
                        ELSE excluded.source_path
                    END,
                    updated_at = CASE
                        WHEN media_conversions.status = 'complete' THEN media_conversions.updated_at
                        ELSE unixepoch()
                    END
                """,
                (target_format,),
            )
            self._connection.commit()
            return int(cursor.rowcount)

    async def next_conversion_batch(
        self,
        limit: int,
        *,
        max_attempts: int,
        exclude: set[tuple[str, str]] | None = None,
    ) -> list[sqlite3.Row]:
        async with self._lock:
            excluded_keys = exclude or set()
            exclusion_clause = ""
            exclusion_params: list[str] = []
            if excluded_keys:
                exclusion_clause = " AND " + " AND ".join(
                    "NOT (media_key = ? AND target_format = ?)" for _ in excluded_keys
                )
                for media_key, target_format in sorted(excluded_keys):
                    exclusion_params.extend([media_key, target_format])
            rows = self._connection.execute(
                f"""
                SELECT * FROM media_conversions
                WHERE (status = 'pending'
                    OR (status = 'failed' AND attempts < ?))
                    {exclusion_clause}
                ORDER BY updated_at ASC
                LIMIT ?
                """,
                (max_attempts, *exclusion_params, limit),
            ).fetchall()
            for row in rows:
                self._connection.execute(
                    """
                    UPDATE media_conversions
                    SET status = 'running', updated_at = unixepoch()
                    WHERE media_key = ? AND target_format = ?
                    """,
                    (row["media_key"], row["target_format"]),
                )
            self._connection.commit()
            return rows

    async def conversion_backlog_count(self, *, max_attempts: int) -> int:
        async with self._lock:
            row = self._connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM media_conversions
                WHERE status IN ('pending', 'running')
                    OR (status = 'failed' AND attempts < ?)
                """,
                (max_attempts,),
            ).fetchone()
            return int(row["count"])

    async def mark_conversion_complete(
        self,
        media_key: str,
        output_path: Path,
        size: int,
        *,
        target_format: str = "jxl",
    ) -> None:
        async with self._lock:
            self._connection.execute(
                """
                UPDATE media_conversions
                SET status = 'complete', output_path = ?, size_bytes = ?, error = '', updated_at = unixepoch()
                WHERE media_key = ? AND target_format = ?
                """,
                (str(output_path), size, media_key, target_format),
            )
            self._connection.commit()

    async def mark_conversion_failed(self, media_key: str, error: str, *, target_format: str = "jxl") -> None:
        async with self._lock:
            self._connection.execute(
                """
                UPDATE media_conversions
                SET status = 'failed', attempts = attempts + 1, error = ?, updated_at = unixepoch()
                WHERE media_key = ? AND target_format = ?
                """,
                (error[:1000], media_key, target_format),
            )
            self._connection.commit()

    async def record_failure(self, *, source: str, identifier: str, payload: Any, error: str) -> None:
        async with self._lock:
            self._connection.execute(
                """
                INSERT INTO failures (source, identifier, payload_json, error, created_at)
                VALUES (?, ?, ?, ?, unixepoch())
                """,
                (source, identifier, json.dumps(payload, ensure_ascii=False), error[:1000]),
            )
            self._connection.commit()

    async def status_counts(self) -> dict[str, int]:
        async with self._lock:
            pairs = []
            for table in ("partitions", "pages", "media_assets", "media_conversions"):
                rows = self._connection.execute(
                    f"SELECT status, COUNT(*) AS count FROM {table} GROUP BY status"
                ).fetchall()
                pairs.extend((f"{table}.{row['status']}", int(row["count"])) for row in rows)
            record_count = self._connection.execute("SELECT COUNT(*) AS count FROM records").fetchone()
            failure_count = self._connection.execute("SELECT COUNT(*) AS count FROM failures").fetchone()
            freetext_count = self._connection.execute("SELECT COUNT(*) AS count FROM record_freetext").fetchone()
            pairs.append(("records.total", int(record_count["count"])))
            pairs.append(("record_freetext.total", int(freetext_count["count"])))
            pairs.append(("failures.total", int(failure_count["count"])))
            return dict(pairs)


SCHEMA = """
CREATE TABLE IF NOT EXISTS partitions (
    partition_key TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    row_count INTEGER,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    partition_key TEXT NOT NULL,
    start INTEGER NOT NULL,
    rows INTEGER NOT NULL,
    sort TEXT NOT NULL,
    record_type TEXT NOT NULL,
    row_group TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    row_count INTEGER,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (partition_key, start)
);

CREATE TABLE IF NOT EXISTS records (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    unit_code TEXT NOT NULL,
    linked_id TEXT NOT NULL,
    type TEXT NOT NULL,
    url TEXT NOT NULL,
    hash TEXT NOT NULL,
    doc_signature TEXT NOT NULL,
    timestamp INTEGER,
    last_time_updated INTEGER,
    status INTEGER,
    public_search INTEGER,
    version TEXT NOT NULL,
    raw_path TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_records_unit_code ON records(unit_code);
CREATE INDEX IF NOT EXISTS idx_records_doc_signature ON records(doc_signature);
CREATE INDEX IF NOT EXISTS idx_records_last_time_updated ON records(last_time_updated);

CREATE TABLE IF NOT EXISTS record_freetext (
    record_id TEXT NOT NULL,
    unit_code TEXT NOT NULL,
    category TEXT NOT NULL,
    label TEXT NOT NULL,
    content TEXT NOT NULL,
    position INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (record_id, position)
);

CREATE INDEX IF NOT EXISTS idx_record_freetext_record_id ON record_freetext(record_id);
CREATE INDEX IF NOT EXISTS idx_record_freetext_category ON record_freetext(category);
CREATE INDEX IF NOT EXISTS idx_record_freetext_label ON record_freetext(label);

CREATE TABLE IF NOT EXISTS media_assets (
    media_key TEXT PRIMARY KEY,
    record_id TEXT NOT NULL,
    unit_code TEXT NOT NULL,
    record_hash TEXT NOT NULL,
    kind TEXT NOT NULL,
    media_type TEXT NOT NULL,
    url TEXT NOT NULL,
    thumbnail TEXT NOT NULL,
    caption TEXT NOT NULL,
    preferred_citation TEXT NOT NULL,
    usage_access TEXT NOT NULL,
    usage_text TEXT NOT NULL,
    usage_codes_json TEXT NOT NULL,
    usage_flag TEXT NOT NULL,
    guid TEXT NOT NULL,
    media_id TEXT NOT NULL,
    ids_id TEXT NOT NULL,
    alt_text TEXT NOT NULL,
    extended_description TEXT NOT NULL,
    resource_label TEXT NOT NULL,
    resource_width INTEGER,
    resource_height INTEGER,
    resource_dimensions TEXT NOT NULL,
    parent_media_url TEXT NOT NULL,
    screen_url TEXT NOT NULL,
    thumbnail_url TEXT NOT NULL,
    downloadable INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL,
    local_path TEXT NOT NULL DEFAULT '',
    size_bytes INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_media_status ON media_assets(status);
CREATE INDEX IF NOT EXISTS idx_media_record_id ON media_assets(record_id);

CREATE TABLE IF NOT EXISTS media_conversions (
    media_key TEXT NOT NULL,
    target_format TEXT NOT NULL DEFAULT 'jxl',
    source_path TEXT NOT NULL,
    output_path TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    size_bytes INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (media_key, target_format)
);

CREATE INDEX IF NOT EXISTS idx_media_conversions_status ON media_conversions(status);
CREATE INDEX IF NOT EXISTS idx_media_conversions_media_key ON media_conversions(media_key);

CREATE TABLE IF NOT EXISTS failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    identifier TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    error TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
"""