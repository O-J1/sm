from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import SmithsonianRecord
from .normalization import NormalizedRecordProjection


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
            self._migrate_legacy_media_conversions()
            self._connection.executescript(SCHEMA)
            self._ensure_conversion_policy_columns()
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute("UPDATE pages SET status = 'pending' WHERE status = 'running'")
            self._connection.execute("UPDATE media_resources SET status = 'pending' WHERE status = 'running'")
            self._connection.execute("UPDATE media_conversions SET status = 'pending' WHERE status = 'running'")
            self._connection.commit()

    def _migrate_legacy_media_conversions(self) -> None:
        """Migrate pre-warehouse databases whose media_conversions table was
        keyed by media_key. The new schema keys conversion work by
        resource_key, so the legacy table (and its indexes) must move out of
        the way before the schema script creates the new table."""
        table = self._connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'media_conversions'"
        ).fetchone()
        if table is None:
            return
        columns = {row["name"] for row in self._connection.execute("PRAGMA table_info(media_conversions)")}
        if "resource_key" in columns:
            return
        self._connection.execute("DROP INDEX IF EXISTS idx_media_conversions_status")
        self._connection.execute("DROP INDEX IF EXISTS idx_media_conversions_media_key")
        count = self._connection.execute("SELECT COUNT(*) AS count FROM media_conversions").fetchone()
        if int(count["count"]) == 0:
            self._connection.execute("DROP TABLE media_conversions")
        else:
            self._connection.execute("DROP TABLE IF EXISTS media_conversions_legacy")
            self._connection.execute("ALTER TABLE media_conversions RENAME TO media_conversions_legacy")
        self._connection.commit()

    def _ensure_conversion_policy_columns(self) -> None:
        """Migrate pre-policy databases: add per-output policy columns."""
        existing = {row["name"] for row in self._connection.execute("PRAGMA table_info(media_conversions)")}
        for column, ddl in (
            ("output_kind", "output_kind TEXT NOT NULL DEFAULT 'highres'"),
            ("target_max_pixels", "target_max_pixels INTEGER"),
            ("target_edge", "target_edge INTEGER"),
            ("quality", "quality INTEGER"),
            ("resize_algorithm", "resize_algorithm TEXT NOT NULL DEFAULT ''"),
            ("output_format", "output_format TEXT NOT NULL DEFAULT ''"),
        ):
            if column not in existing:
                self._connection.execute(f"ALTER TABLE media_conversions ADD COLUMN {ddl}")

    async def upsert_record_projection(
        self,
        record: SmithsonianRecord,
        raw_path: Path,
        projection: NormalizedRecordProjection,
    ) -> bool:
        async with self._lock:
            existing = self._connection.execute(
                "SELECT doc_signature FROM records WHERE id = ?",
                (record.id,),
            ).fetchone()
            raw_changed = existing is None or existing["doc_signature"] != record.doc_signature
            projection_needs_replace = raw_changed or self._projection_needs_replace(record, projection)
            self._upsert_record_row(record, raw_path)
            if projection_needs_replace:
                self._replace_record_projection(record, raw_path, projection)
            self._connection.commit()
            return raw_changed

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

    def _upsert_record_row(self, record: SmithsonianRecord, raw_path: Path) -> None:
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

    def _projection_needs_replace(self, record: SmithsonianRecord, projection: NormalizedRecordProjection) -> bool:
        raw_version = self._connection.execute(
            "SELECT 1 FROM record_raw_versions WHERE record_id = ? AND doc_signature = ?",
            (record.id, record.doc_signature),
        ).fetchone()
        if raw_version is None:
            return True

        expected_counts = (
            ("record_text_entries", len(projection.text_entries)),
            ("record_identifiers", len(projection.identifiers)),
            ("record_dates", len(projection.dates)),
            ("record_rights", len(projection.rights)),
            ("record_facets", len(projection.facets)),
            ("record_relationships", len(projection.relationships)),
            ("media_items", len(projection.media_items)),
            ("record_media", len(projection.media_items)),
            ("media_resources", len(projection.media_resources)),
            ("media_usage_codes", len(projection.media_usage_codes)),
        )
        for table, expected in expected_counts:
            row = self._connection.execute(
                f"SELECT COUNT(*) AS count FROM {table} WHERE record_id = ?",
                (record.id,),
            ).fetchone()
            if int(row["count"]) != expected:
                return True
        return False

    def _replace_record_projection(
        self,
        record: SmithsonianRecord,
        raw_path: Path,
        projection: NormalizedRecordProjection,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO record_raw_versions (record_id, doc_signature, unit_code, raw_json, raw_path, scraped_at)
            VALUES (?, ?, ?, ?, ?, unixepoch())
            ON CONFLICT(record_id, doc_signature) DO UPDATE SET
                unit_code = excluded.unit_code,
                raw_json = excluded.raw_json,
                raw_path = excluded.raw_path,
                scraped_at = unixepoch()
            """,
            (
                record.id,
                record.doc_signature,
                record.unit_code,
                json.dumps(record.raw, ensure_ascii=False, separators=(",", ":")),
                str(raw_path),
            ),
        )
        for table in (
            "record_text_entries",
            "record_identifiers",
            "record_dates",
            "record_rights",
            "record_facets",
            "record_relationships",
            "media_usage_codes",
            "media_resources",
            "record_media",
            "media_items",
        ):
            self._connection.execute(f"DELETE FROM {table} WHERE record_id = ?", (record.id,))
        self._connection.executemany(
            """
            INSERT INTO record_text_entries (
                record_id, unit_code, category, normalized_category, label, normalized_label,
                content, content_hash, position, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, unixepoch())
            """,
            [
                (
                    entry.record_id,
                    entry.unit_code,
                    entry.category,
                    entry.normalized_category,
                    entry.label,
                    entry.normalized_label,
                    entry.content,
                    entry.content_hash,
                    entry.position,
                )
                for entry in projection.text_entries
            ],
        )
        self._connection.executemany(
            """
            INSERT INTO record_identifiers (
                record_id, unit_code, identifier_type, identifier_value, source_category, source_label, position
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    identifier.record_id,
                    identifier.unit_code,
                    identifier.identifier_type,
                    identifier.identifier_value,
                    identifier.source_category,
                    identifier.source_label,
                    identifier.position,
                )
                for identifier in projection.identifiers
            ],
        )
        self._connection.executemany(
            """
            INSERT INTO record_dates (
                record_id, unit_code, date_text, start_year, end_year, precision, source_category, source_label, position
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    date.record_id,
                    date.unit_code,
                    date.date_text,
                    date.start_year,
                    date.end_year,
                    date.precision,
                    date.source_category,
                    date.source_label,
                    date.position,
                )
                for date in projection.dates
            ],
        )
        self._connection.executemany(
            """
            INSERT INTO record_rights (record_id, unit_code, rights_text, normalized_rights, source, source_label, position)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    rights.record_id,
                    rights.unit_code,
                    rights.rights_text,
                    rights.normalized_rights,
                    rights.source,
                    rights.source_label,
                    rights.position,
                )
                for rights in projection.rights
            ],
        )
        self._connection.executemany(
            """
            INSERT INTO record_facets (record_id, unit_code, facet_type, value, normalized_value, source_path, position)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    facet.record_id,
                    facet.unit_code,
                    facet.facet_type,
                    facet.value,
                    facet.normalized_value,
                    facet.source_path,
                    facet.position,
                )
                for facet in projection.facets
            ],
        )
        self._connection.executemany(
            """
            INSERT INTO media_items (
                media_key, record_id, unit_code, media_type, guid, media_id, ids_id, caption,
                preferred_citation, usage_access, usage_text, usage_flag, alt_text,
                extended_description, parent_url, position
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item.media_key,
                    item.record_id,
                    item.unit_code,
                    item.media_type,
                    item.guid,
                    item.media_id,
                    item.ids_id,
                    item.caption,
                    item.preferred_citation,
                    item.usage_access,
                    item.usage_text,
                    item.usage_flag,
                    item.alt_text,
                    item.extended_description,
                    item.parent_url,
                    item.position,
                )
                for item in projection.media_items
            ],
        )
        self._connection.executemany(
            "INSERT INTO record_media (record_id, media_key, position) VALUES (?, ?, ?)",
            [(item.record_id, item.media_key, item.position) for item in projection.media_items],
        )
        self._connection.executemany(
            """
            INSERT INTO media_resources (
                resource_key, media_key, record_id, unit_code, role, url, label, width,
                height, dimensions, downloadable, preferred_download, status, local_path,
                size_bytes, attempts, error, position, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', 0, 0, '', ?, unixepoch())
            """,
            [
                (
                    resource.resource_key,
                    resource.media_key,
                    resource.record_id,
                    resource.unit_code,
                    resource.role,
                    resource.url,
                    resource.label,
                    resource.width,
                    resource.height,
                    resource.dimensions,
                    int(resource.downloadable),
                    int(resource.preferred_download),
                    "pending" if resource.downloadable and resource.preferred_download else "metadata",
                    resource.position,
                )
                for resource in projection.media_resources
            ],
        )
        self._connection.executemany(
            "INSERT INTO media_usage_codes (media_key, record_id, unit_code, code, position) VALUES (?, ?, ?, ?, ?)",
            [
                (code.media_key, code.record_id, code.unit_code, code.code, code.position)
                for code in projection.media_usage_codes
            ],
        )
        self._connection.executemany(
            """
            INSERT INTO record_relationships (record_id, unit_code, target, relation_type, label, source, position)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    relationship.record_id,
                    relationship.unit_code,
                    relationship.target,
                    relationship.relation_type,
                    relationship.label,
                    relationship.source,
                    relationship.position,
                )
                for relationship in projection.relationships
            ],
        )

    async def next_media_batch(self, limit: int, *, max_attempts: int) -> list[sqlite3.Row]:
        async with self._lock:
            rows = self._connection.execute(
                """
                SELECT
                    media_resources.resource_key,
                    media_resources.media_key,
                    media_resources.record_id,
                    media_resources.unit_code,
                    records.hash AS record_hash,
                    media_resources.role AS kind,
                    media_items.media_type,
                    media_resources.url,
                    COALESCE((
                        SELECT thumbnail.url
                        FROM media_resources AS thumbnail
                        WHERE thumbnail.media_key = media_resources.media_key
                            AND thumbnail.role = 'thumbnail'
                        ORDER BY thumbnail.position
                        LIMIT 1
                    ), '') AS thumbnail,
                    media_items.caption,
                    media_items.preferred_citation,
                    media_items.usage_access,
                    media_items.usage_text,
                    media_items.usage_flag,
                    media_items.guid,
                    media_items.media_id,
                    media_items.ids_id,
                    media_items.alt_text,
                    media_items.extended_description,
                    media_resources.label AS resource_label,
                    media_resources.width AS resource_width,
                    media_resources.height AS resource_height,
                    media_resources.dimensions AS resource_dimensions,
                    media_items.parent_url AS parent_media_url,
                    COALESCE((
                        SELECT screen.url
                        FROM media_resources AS screen
                        WHERE screen.media_key = media_resources.media_key
                            AND screen.role = 'screen'
                        ORDER BY screen.position
                        LIMIT 1
                    ), '') AS screen_url,
                    COALESCE((
                        SELECT thumbnail.url
                        FROM media_resources AS thumbnail
                        WHERE thumbnail.media_key = media_resources.media_key
                            AND thumbnail.role = 'thumbnail'
                        ORDER BY thumbnail.position
                        LIMIT 1
                    ), '') AS thumbnail_url,
                    media_resources.downloadable,
                    media_resources.status,
                    media_resources.local_path,
                    media_resources.size_bytes,
                    media_resources.attempts,
                    media_resources.error,
                    media_resources.updated_at
                FROM media_resources
                JOIN media_items ON media_items.media_key = media_resources.media_key
                JOIN records ON records.id = media_resources.record_id
                WHERE media_resources.downloadable = 1
                    AND media_resources.preferred_download = 1
                    AND (media_resources.status = 'pending'
                        OR (media_resources.status = 'failed' AND media_resources.attempts < ?))
                    AND NOT EXISTS (
                        SELECT 1 FROM media_policy
                        WHERE media_policy.url = media_resources.url
                            AND media_policy.output_kind = 'highres'
                            AND media_policy.tier = 'drop'
                    )
                ORDER BY media_resources.updated_at ASC
                LIMIT ?
                """,
                (max_attempts, limit),
            ).fetchall()
            for row in rows:
                self._connection.execute(
                    "UPDATE media_resources SET status = 'running', updated_at = unixepoch() WHERE resource_key = ?",
                    (row["resource_key"],),
                )
            self._connection.commit()
            return rows

    async def mark_media_complete(self, resource_key: str, path: Path, size: int) -> None:
        async with self._lock:
            self._connection.execute(
                """
                UPDATE media_resources
                SET status = 'complete', local_path = ?, size_bytes = ?, error = '', updated_at = unixepoch()
                WHERE resource_key = ?
                """,
                (str(path), size, resource_key),
            )
            self._connection.commit()

    async def mark_media_failed(self, resource_key: str, error: str) -> None:
        async with self._lock:
            self._connection.execute(
                """
                UPDATE media_resources
                SET status = 'failed', attempts = attempts + 1, error = ?, updated_at = unixepoch()
                WHERE resource_key = ?
                """,
                (error[:1000], resource_key),
            )
            self._connection.commit()

    async def apply_manifest(self, manifest_path: Path) -> int:
        """Load a manifest.jsonl (from scripts/export_manifest.py) into the
        media_policy table. Rows are keyed by (url, output_kind); the URL is
        the join key between the analysis manifest and media_resources."""
        async with self._lock:
            count = 0
            batch: list[tuple] = []

            def flush() -> None:
                if batch:
                    self._connection.executemany(
                        """
                        INSERT OR REPLACE INTO media_policy (
                            url, output_kind, tier, drop_reason, target_max_pixels,
                            target_edge, resize_algorithm, output_format, quality, output_path
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        batch,
                    )
                    batch.clear()

            with manifest_path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    url = str(row.get("url") or "")
                    if not url:
                        continue
                    batch.append(
                        (
                            url,
                            str(row.get("output_kind") or "highres"),
                            str(row.get("tier") or ""),
                            str(row.get("drop_reason") or ""),
                            row.get("target_max_pixels"),
                            row.get("target_edge"),
                            str(row.get("resize_algorithm") or ""),
                            str(row.get("output_format") or ""),
                            row.get("quality"),
                            str(row.get("output_path") or ""),
                        )
                    )
                    count += 1
                    if len(batch) >= 5000:
                        flush()
            flush()
            self._connection.commit()
            return count

    async def has_media_policy(self, url: str) -> bool:
        async with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM media_policy WHERE url = ? LIMIT 1",
                (url,),
            ).fetchone()
            return row is not None

    async def enqueue_media_conversion(
        self,
        resource_key: str,
        source_path: Path,
        *,
        target_format: str = "jxl",
    ) -> None:
        async with self._lock:
            url_row = self._connection.execute(
                "SELECT url FROM media_resources WHERE resource_key = ?",
                (resource_key,),
            ).fetchone()
            url = str(url_row["url"]) if url_row else ""
            policies = (
                self._connection.execute(
                    "SELECT * FROM media_policy WHERE url = ?",
                    (url,),
                ).fetchall()
                if url
                else []
            )
            highres = next((p for p in policies if str(p["output_kind"]) == "highres"), None)
            if highres is not None and str(highres["tier"]) == "drop":
                self._connection.commit()
                return
            self._connection.execute(
                """
                INSERT INTO media_conversions (
                    resource_key, target_format, source_path, output_path, status,
                    attempts, error, size_bytes, created_at, updated_at
                ) VALUES (?, ?, ?, '', 'pending', 0, '', 0, unixepoch(), unixepoch())
                ON CONFLICT(resource_key, target_format) DO UPDATE SET
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
                (resource_key, target_format, str(source_path)),
            )
            if highres is not None:
                self._connection.execute(
                    """
                    UPDATE media_conversions
                    SET output_kind = 'highres', target_max_pixels = ?, quality = ?,
                        resize_algorithm = ?, output_format = ?, output_path = ?
                    WHERE resource_key = ? AND target_format = ? AND status != 'complete'
                    """,
                    (
                        highres["target_max_pixels"],
                        highres["quality"],
                        str(highres["resize_algorithm"] or ""),
                        str(highres["output_format"] or ""),
                        str(highres["output_path"] or ""),
                        resource_key,
                        target_format,
                    ),
                )
            for policy in policies:
                kind = str(policy["output_kind"])
                if kind == "highres" or not policy["target_edge"]:
                    continue
                # Derivative jobs reuse the (resource_key, target_format) primary
                # key with a synthetic format like 'jpg256'; the real output
                # location comes from output_path.
                derivative_format = f"{policy['output_format'] or 'jpg'}{policy['target_edge']}"
                self._connection.execute(
                    """
                    INSERT INTO media_conversions (
                        resource_key, target_format, source_path, output_path, status,
                        attempts, error, size_bytes, created_at, updated_at,
                        output_kind, target_edge, quality, resize_algorithm, output_format
                    ) VALUES (?, ?, ?, ?, 'pending', 0, '', 0, unixepoch(), unixepoch(), ?, ?, ?, ?, ?)
                    ON CONFLICT(resource_key, target_format) DO UPDATE SET
                        source_path = CASE
                            WHEN media_conversions.status = 'complete' THEN media_conversions.source_path
                            ELSE excluded.source_path
                        END,
                        status = CASE
                            WHEN media_conversions.status = 'complete' THEN media_conversions.status
                            ELSE 'pending'
                        END,
                        updated_at = CASE
                            WHEN media_conversions.status = 'complete' THEN media_conversions.updated_at
                            ELSE unixepoch()
                        END
                    """,
                    (
                        resource_key,
                        derivative_format,
                        str(source_path),
                        str(policy["output_path"] or ""),
                        kind,
                        policy["target_edge"],
                        policy["quality"],
                        str(policy["resize_algorithm"] or ""),
                        str(policy["output_format"] or ""),
                    ),
                )
            self._connection.commit()

    async def enqueue_pending_tiff_conversions(self, *, target_format: str = "jxl") -> int:
        async with self._lock:
            cursor = self._connection.execute(
                """
                INSERT INTO media_conversions (
                    resource_key, target_format, source_path, output_path, status,
                    attempts, error, size_bytes, created_at, updated_at
                )
                SELECT resource_key, ?, local_path, '', 'pending', 0, '', 0, unixepoch(), unixepoch()
                FROM media_resources
                WHERE status = 'complete'
                    AND role = 'highres_tiff'
                    AND local_path != ''
                    AND (lower(local_path) LIKE '%.tif' OR lower(local_path) LIKE '%.tiff')
                ON CONFLICT(resource_key, target_format) DO UPDATE SET
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
                    "NOT (resource_key = ? AND target_format = ?)" for _ in excluded_keys
                )
                for resource_key, target_format in sorted(excluded_keys):
                    exclusion_params.extend([resource_key, target_format])
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
                    WHERE resource_key = ? AND target_format = ?
                    """,
                    (row["resource_key"], row["target_format"]),
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

    async def incomplete_conversions_for_source(self, source_path: str) -> int:
        """Outputs still owed from this source file - the source must not be
        deleted until this reaches zero (derivatives read the original)."""
        async with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM media_conversions WHERE source_path = ? AND status != 'complete'",
                (source_path,),
            ).fetchone()
            return int(row["count"])

    async def mark_conversion_complete(
        self,
        resource_key: str,
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
                WHERE resource_key = ? AND target_format = ?
                """,
                (str(output_path), size, resource_key, target_format),
            )
            self._connection.commit()

    async def mark_conversion_failed(self, resource_key: str, error: str, *, target_format: str = "jxl") -> None:
        async with self._lock:
            self._connection.execute(
                """
                UPDATE media_conversions
                SET status = 'failed', attempts = attempts + 1, error = ?, updated_at = unixepoch()
                WHERE resource_key = ? AND target_format = ?
                """,
                (error[:1000], resource_key, target_format),
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
            for table in ("partitions", "pages", "media_resources", "media_conversions"):
                rows = self._connection.execute(
                    f"SELECT status, COUNT(*) AS count FROM {table} GROUP BY status"
                ).fetchall()
                pairs.extend((f"{table}.{row['status']}", int(row["count"])) for row in rows)
            record_count = self._connection.execute("SELECT COUNT(*) AS count FROM records").fetchone()
            failure_count = self._connection.execute("SELECT COUNT(*) AS count FROM failures").fetchone()
            text_count = self._connection.execute("SELECT COUNT(*) AS count FROM record_text_entries").fetchone()
            pairs.append(("records.total", int(record_count["count"])))
            pairs.append(("record_text_entries.total", int(text_count["count"])))
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

CREATE TABLE IF NOT EXISTS record_raw_versions (
    record_id TEXT NOT NULL,
    doc_signature TEXT NOT NULL,
    unit_code TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    raw_path TEXT NOT NULL,
    scraped_at INTEGER NOT NULL,
    PRIMARY KEY (record_id, doc_signature),
    FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_record_raw_versions_record_id ON record_raw_versions(record_id);
CREATE INDEX IF NOT EXISTS idx_record_raw_versions_unit_code ON record_raw_versions(unit_code);

CREATE TABLE IF NOT EXISTS record_text_entries (
    record_id TEXT NOT NULL,
    unit_code TEXT NOT NULL,
    category TEXT NOT NULL,
    normalized_category TEXT NOT NULL,
    label TEXT NOT NULL,
    normalized_label TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    position INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (record_id, position),
    FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_record_text_entries_category ON record_text_entries(normalized_category);
CREATE INDEX IF NOT EXISTS idx_record_text_entries_label ON record_text_entries(normalized_label);
CREATE INDEX IF NOT EXISTS idx_record_text_entries_hash ON record_text_entries(content_hash);

CREATE TABLE IF NOT EXISTS record_identifiers (
    record_id TEXT NOT NULL,
    unit_code TEXT NOT NULL,
    identifier_type TEXT NOT NULL,
    identifier_value TEXT NOT NULL,
    source_category TEXT NOT NULL,
    source_label TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (record_id, position),
    FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_record_identifiers_value ON record_identifiers(identifier_value);
CREATE INDEX IF NOT EXISTS idx_record_identifiers_type ON record_identifiers(identifier_type);

CREATE TABLE IF NOT EXISTS record_dates (
    record_id TEXT NOT NULL,
    unit_code TEXT NOT NULL,
    date_text TEXT NOT NULL,
    start_year INTEGER,
    end_year INTEGER,
    precision TEXT NOT NULL,
    source_category TEXT NOT NULL,
    source_label TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (record_id, position),
    FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_record_dates_start_year ON record_dates(start_year);
CREATE INDEX IF NOT EXISTS idx_record_dates_end_year ON record_dates(end_year);

CREATE TABLE IF NOT EXISTS record_rights (
    record_id TEXT NOT NULL,
    unit_code TEXT NOT NULL,
    rights_text TEXT NOT NULL,
    normalized_rights TEXT NOT NULL,
    source TEXT NOT NULL,
    source_label TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (record_id, position),
    FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_record_rights_normalized ON record_rights(normalized_rights);

CREATE TABLE IF NOT EXISTS record_facets (
    record_id TEXT NOT NULL,
    unit_code TEXT NOT NULL,
    facet_type TEXT NOT NULL,
    value TEXT NOT NULL,
    normalized_value TEXT NOT NULL,
    source_path TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (record_id, position),
    FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_record_facets_type ON record_facets(facet_type);
CREATE INDEX IF NOT EXISTS idx_record_facets_value ON record_facets(normalized_value);

CREATE TABLE IF NOT EXISTS media_items (
    media_key TEXT PRIMARY KEY,
    record_id TEXT NOT NULL,
    unit_code TEXT NOT NULL,
    media_type TEXT NOT NULL,
    guid TEXT NOT NULL,
    media_id TEXT NOT NULL,
    ids_id TEXT NOT NULL,
    caption TEXT NOT NULL,
    preferred_citation TEXT NOT NULL,
    usage_access TEXT NOT NULL,
    usage_text TEXT NOT NULL,
    usage_flag TEXT NOT NULL,
    alt_text TEXT NOT NULL,
    extended_description TEXT NOT NULL,
    parent_url TEXT NOT NULL,
    position INTEGER NOT NULL,
    FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_media_items_record_id ON media_items(record_id);
CREATE INDEX IF NOT EXISTS idx_media_items_ids_id ON media_items(ids_id);
CREATE INDEX IF NOT EXISTS idx_media_items_guid ON media_items(guid);

CREATE TABLE IF NOT EXISTS record_media (
    record_id TEXT NOT NULL,
    media_key TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (record_id, media_key),
    FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE,
    FOREIGN KEY (media_key) REFERENCES media_items(media_key) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS media_resources (
    resource_key TEXT PRIMARY KEY,
    media_key TEXT NOT NULL,
    record_id TEXT NOT NULL,
    unit_code TEXT NOT NULL,
    role TEXT NOT NULL,
    url TEXT NOT NULL,
    label TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    dimensions TEXT NOT NULL,
    downloadable INTEGER NOT NULL DEFAULT 0 CHECK (downloadable IN (0, 1)),
    preferred_download INTEGER NOT NULL DEFAULT 0 CHECK (preferred_download IN (0, 1)),
    status TEXT NOT NULL DEFAULT 'metadata',
    local_path TEXT NOT NULL DEFAULT '',
    size_bytes INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    position INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE,
    FOREIGN KEY (media_key) REFERENCES media_items(media_key) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_media_resources_media_key ON media_resources(media_key);
CREATE INDEX IF NOT EXISTS idx_media_resources_role ON media_resources(role);
CREATE INDEX IF NOT EXISTS idx_media_resources_preferred ON media_resources(preferred_download);
CREATE INDEX IF NOT EXISTS idx_media_resources_status ON media_resources(status);

CREATE TABLE IF NOT EXISTS media_usage_codes (
    media_key TEXT NOT NULL,
    record_id TEXT NOT NULL,
    unit_code TEXT NOT NULL,
    code TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (media_key, position),
    FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE,
    FOREIGN KEY (media_key) REFERENCES media_items(media_key) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_media_usage_codes_code ON media_usage_codes(code);

CREATE TABLE IF NOT EXISTS record_relationships (
    record_id TEXT NOT NULL,
    unit_code TEXT NOT NULL,
    target TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    label TEXT NOT NULL,
    source TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (record_id, position),
    FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_record_relationships_target ON record_relationships(target);

CREATE TABLE IF NOT EXISTS media_conversions (
    resource_key TEXT NOT NULL,
    target_format TEXT NOT NULL DEFAULT 'jxl',
    source_path TEXT NOT NULL,
    output_path TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    size_bytes INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    output_kind TEXT NOT NULL DEFAULT 'highres',
    target_max_pixels INTEGER,
    target_edge INTEGER,
    quality INTEGER,
    resize_algorithm TEXT NOT NULL DEFAULT '',
    output_format TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (resource_key, target_format),
    FOREIGN KEY (resource_key) REFERENCES media_resources(resource_key) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_media_conversions_status ON media_conversions(status);
CREATE INDEX IF NOT EXISTS idx_media_conversions_resource_key ON media_conversions(resource_key);
CREATE INDEX IF NOT EXISTS idx_media_conversions_source_path ON media_conversions(source_path);

CREATE TABLE IF NOT EXISTS media_policy (
    url TEXT NOT NULL,
    output_kind TEXT NOT NULL,
    tier TEXT NOT NULL DEFAULT '',
    drop_reason TEXT NOT NULL DEFAULT '',
    target_max_pixels INTEGER,
    target_edge INTEGER,
    resize_algorithm TEXT NOT NULL DEFAULT '',
    output_format TEXT NOT NULL DEFAULT '',
    quality INTEGER,
    output_path TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (url, output_kind)
);

CREATE TABLE IF NOT EXISTS failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    identifier TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    error TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
"""