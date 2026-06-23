from __future__ import annotations

import asyncio
from typing import Any

from .api import SmithsonianAPIError, SmithsonianClient, search_rows
from .config import ScraperConfig
from .parser import RecordParseError, extract_freetext_entries, extract_media_assets, parse_record
from .state import PageJob, StateStore
from .storage import RecordStore


class CrawlScheduler:
    def __init__(
        self,
        *,
        config: ScraperConfig,
        client: SmithsonianClient,
        state: StateStore,
        store: RecordStore,
    ) -> None:
        self._config = config
        self._client = client
        self._state = state
        self._store = store
        self._records_processed = 0

    async def probe(self) -> dict[str, Any]:
        stats_payload = await self._client.stats()
        terms_payload = await self._client.terms("unit_code")
        search_payload = await self._client.search(
            q=self._config.query,
            start=0,
            rows=min(self._config.rows_per_page, 10),
            sort=self._config.sort,
            record_type=self._config.record_type,
            row_group=self._config.row_group,
        )
        rows, row_count = search_rows(search_payload)
        return {
            "stats": stats_payload.get("response", stats_payload),
            "unit_terms": terms_payload.get("response", terms_payload),
            "sample_row_count": row_count,
            "sample_rows": len(rows),
            "sample_ids": [row.get("id") for row in rows[:5]],
        }

    async def scrape(self) -> None:
        units = await self._discover_units()
        for unit in units:
            if self._limit_reached():
                break
            await self._initialize_partition(unit)

    async def _discover_units(self) -> list[str]:
        if self._config.include_units:
            units = list(self._config.include_units)
        else:
            payload = await self._client.terms("unit_code")
            units = _extract_terms(payload)
        if self._config.exclude_units:
            excluded = set(self._config.exclude_units)
            units = [unit for unit in units if unit not in excluded]
        return sorted(set(unit for unit in units if unit))

    async def _initialize_partition(self, unit_code: str) -> None:
        remaining = self._remaining_limit()
        if remaining == 0:
            return

        query = f"{self._config.query} AND unit_code:{unit_code}"
        first_payload = await self._client.search(
            q=query,
            start=0,
            rows=self._config.rows_per_page,
            sort=self._config.sort,
            record_type=self._config.record_type,
            row_group=self._config.row_group,
        )
        first_rows, row_count = search_rows(first_payload)
        effective_row_count = min(row_count, remaining) if remaining is not None else row_count
        await self._state.upsert_partition(unit_code, query, row_count)

        page_queue: asyncio.Queue[PageJob | None] = asyncio.Queue()
        workers = [asyncio.create_task(self._metadata_worker(page_queue)) for _ in range(self._config.metadata_workers)]

        first_job = PageJob(
            unit_code,
            query,
            0,
            self._config.rows_per_page,
            self._config.sort,
            self._config.record_type,
            self._config.row_group,
        )
        if await self._state.page_status(unit_code, 0) != "complete":
            await self._process_page_rows(first_job, first_rows, row_count)

        for start in range(self._config.rows_per_page, effective_row_count, self._config.rows_per_page):
            if self._limit_reached():
                break
            if await self._state.page_status(unit_code, start) == "complete":
                continue
            await page_queue.put(
                PageJob(
                    partition_key=unit_code,
                    query=query,
                    start=start,
                    rows=self._config.rows_per_page,
                    sort=self._config.sort,
                    record_type=self._config.record_type,
                    row_group=self._config.row_group,
                )
            )

        for _ in workers:
            await page_queue.put(None)
        await asyncio.gather(*workers)

    async def _metadata_worker(self, page_queue: asyncio.Queue[PageJob | None]) -> None:
        while True:
            job = await page_queue.get()
            if job is None:
                page_queue.task_done()
                return
            if self._limit_reached():
                page_queue.task_done()
                continue
            try:
                payload = await self._client.search(
                    q=job.query,
                    start=job.start,
                    rows=job.rows,
                    sort=job.sort,
                    record_type=job.record_type,
                    row_group=job.row_group,
                )
                rows, row_count = search_rows(payload)
                await self._process_page_rows(job, rows, row_count)
            except Exception as exc:
                await self._state.mark_page(job, "failed", error=str(exc))
            finally:
                page_queue.task_done()

    async def _process_page_rows(self, job: PageJob, rows: list[dict[str, Any]], row_count: int) -> int:
        remaining = self._remaining_limit()
        if remaining == 0:
            await self._state.mark_page(job, "partial", error="run record limit reached", row_count=row_count)
            return 0

        rows_to_process = rows[:remaining] if remaining is not None else rows
        for raw in rows_to_process:
            self._records_processed += 1
            try:
                record = parse_record(raw)
                raw_path = self._store.record_path(record)
                changed = await self._state.upsert_record(record, raw_path)
                if changed:
                    self._store.append_record(record)
                    await self._state.replace_freetext_entries(record.id, extract_freetext_entries(record))
                if self._config.download_media and changed:
                    for media in extract_media_assets(record):
                        await self._state.enqueue_media(media)
            except RecordParseError as exc:
                await self._state.record_failure(source="record", identifier=str(raw.get("id", "")), payload=raw, error=str(exc))
        status = "complete" if len(rows_to_process) == len(rows) else "partial"
        error = "" if status == "complete" else "run record limit reached"
        await self._state.mark_page(job, status, error=error, row_count=row_count)
        return len(rows_to_process)

    def _remaining_limit(self) -> int | None:
        if self._config.record_limit is None:
            return None
        return max(self._config.record_limit - self._records_processed, 0)

    def _limit_reached(self) -> bool:
        return self._remaining_limit() == 0


def _extract_terms(payload: dict[str, Any]) -> list[str]:
    response = payload.get("response", payload)
    candidates: Any = response
    if isinstance(response, dict):
        candidates = response.get("terms") or response.get("rows") or response.get("data") or response.get("term")
    if not isinstance(candidates, list):
        raise SmithsonianAPIError("Could not find unit terms in /terms/unit_code response")

    terms: list[str] = []
    for item in candidates:
        if isinstance(item, str):
            terms.append(item)
        elif isinstance(item, dict):
            value = item.get("term") or item.get("key") or item.get("value") or item.get("name")
            if value:
                terms.append(str(value))
    return terms