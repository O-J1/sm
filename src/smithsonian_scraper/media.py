from __future__ import annotations

import asyncio
import os
import sqlite3
from pathlib import Path

import httpx

from .storage import RecordStore


class MediaDownloader:
    def __init__(self, *, store: RecordStore, timeout: float, retry_limit: int, max_connections: int) -> None:
        self._store = store
        self._retry_limit = retry_limit
        limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections)
        self._client = httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True)

    async def close(self) -> None:
        await self._client.aclose()

    async def download_row(self, row: sqlite3.Row) -> tuple[Path, int]:
        url = str(row["url"])
        existing_path = Path(str(row["local_path"])) if row["local_path"] else None
        if existing_path and existing_path.exists():
            return existing_path, existing_path.stat().st_size

        media = _media_from_row(row)
        part = self._store.media_part_path(media)
        part.parent.mkdir(parents=True, exist_ok=True)

        generic_target = self._store.media_path(media)
        if generic_target.exists() and generic_target.suffix != ".bin":
            return generic_target, generic_target.stat().st_size

        for attempt in range(self._retry_limit + 1):
            try:
                target = await self._download_to_part(url, media, part)
                return target, target.stat().st_size
            except httpx.HTTPError:
                if attempt >= self._retry_limit:
                    raise
                await asyncio.sleep(min(2.0**attempt, 60.0))

        raise RuntimeError("media download exhausted retries")

    async def _download_to_part(self, url: str, media, part: Path) -> Path:
        headers = {}
        mode = "wb"
        existing_size = part.stat().st_size if part.exists() else 0
        if existing_size:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"

        async with self._client.stream("GET", url, headers=headers) as response:
            if response.status_code == 200 and existing_size:
                existing_size = 0
                mode = "wb"
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            content_disposition = response.headers.get("Content-Disposition", "")
            chunks = response.aiter_bytes()
            try:
                first_chunk = await anext(chunks)
            except StopAsyncIteration:
                first_chunk = b""
            target = self._store.media_path(media, content_type, content_disposition, first_chunk)
            if target.exists():
                if part.exists():
                    part.unlink()
                return target
            with part.open(mode) as handle:
                if first_chunk:
                    handle.write(first_chunk)
                async for chunk in chunks:
                    handle.write(chunk)
        os.replace(part, target)
        return target


def _media_from_row(row: sqlite3.Row):
    from .models import MediaDownload

    return MediaDownload(
        resource_key=str(row["resource_key"]),
        record_id=str(row["record_id"]),
        unit_code=str(row["unit_code"]),
        record_hash=str(row["record_hash"]),
        kind=str(row["kind"]),
        media_type=str(row["media_type"]),
        url=str(row["url"]),
        thumbnail=str(row["thumbnail"]),
        caption=str(row["caption"]),
        preferred_citation=str(row["preferred_citation"]),
        usage_access=str(row["usage_access"]),
        usage_text=str(row["usage_text"]),
        usage_flag=str(row["usage_flag"]),
        guid=str(row["guid"]),
        media_id=str(row["media_id"]),
        ids_id=str(row["ids_id"]),
        alt_text=str(row["alt_text"]),
        extended_description=str(row["extended_description"]),
        resource_label=str(row["resource_label"]),
        resource_width=row["resource_width"],
        resource_height=row["resource_height"],
        resource_dimensions=str(row["resource_dimensions"]),
        parent_media_url=str(row["parent_media_url"]),
        screen_url=str(row["screen_url"]),
        thumbnail_url=str(row["thumbnail_url"]),
        downloadable=bool(row["downloadable"]),
    )